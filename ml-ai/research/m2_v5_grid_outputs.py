"""Stage D descriptive probability, CQR, exploratory weighting and local freeze."""
from __future__ import annotations

import platform
import subprocess
from concurrent.futures import as_completed
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from m2_v5_cv import CATEGORIES, arm_folds, describe_ids, matrix, split_off
from m2_v5_grid import (A1C, BUNDLE, MANIFEST, NEW, OLD, PROB_FIELDS, PROTOCOL, REPORT, ROOT, SEED, SEEDS,
                        SPLITS, bundle_predict, descriptor, freeze_point,
                        point_matrix, point_job, subgroup_masks, weights)
from medpredict_ml.common import read_json, sha256, write_json

PROB_CONFIG = {"max_leaf_nodes": 15, "learning_rate": 0.05, "l2_regularization": 10, "max_iter": 200,
               "early_stopping": False}
Q_CONFIG = {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 20, "loss": "quantile"}
QUANTILES = (0.1, 0.5, 0.9)
PROB_NAMES = ("hist_gradient_boosting", "logistic_same_inputs", "logistic_age_only", "constant")
RANGES = {"lt110": (-np.inf, 110), "110_to_lt130": (110, 130),
          "130_to_lt150": (130, 150), "ge150": (150, np.inf)}


def normalize(w):
    return w / w.mean()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probability_threshold(y, p, w):
    fpr, tpr, thresholds = roc_curve(y, p, sample_weight=w, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= .20 + 1e-12)
    best_tpr = tpr[eligible].max()
    tied = eligible[np.abs(tpr[eligible] - best_tpr) <= 1e-12]
    return float(min(1.0, thresholds[tied].max()))


def fit_probability(core, cal):
    y, yc = core.target.ge(140).to_numpy(), cal.target.ge(140).to_numpy()
    w, wc = normalize(weights(core, "wide_weight")), normalize(weights(cal, "wide_weight"))
    x, xc = matrix(core, cal, fields=PROB_FIELDS)
    center = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=w))
    scale[scale < 1e-8] = 1
    candidates = {
        "hist_gradient_boosting": [HistGradientBoostingClassifier(**PROB_CONFIG, random_state=seed)
                                   .fit(x, y, sample_weight=w) for seed in SEEDS],
        "logistic_same_inputs": [LogisticRegression(C=1, max_iter=1000)
                                  .fit((x - center) / scale, y, sample_weight=w)],
        "logistic_age_only": [LogisticRegression(C=1, max_iter=1000)
                               .fit(((x - center) / scale)[:, :1], y, sample_weight=w)],
    }
    models = {}
    for name, predictors in candidates.items():
        use = xc if name == PROB_NAMES[0] else (xc - center) / scale
        if name == "logistic_age_only":
            use = use[:, :1]
        models[name] = {
            "predictors": predictors,
            "platt": [LogisticRegression(C=1, max_iter=1000)
                      .fit(logit(model.predict_proba(use)[:, 1]), yc, sample_weight=wc) for model in predictors]}
    bundle = {"fields": PROB_FIELDS, "center": center, "scale": scale, "models": models,
              "constant": float(np.average(y, weights=w))}
    scores = probability_predict(bundle, cal)
    bundle["thresholds"] = {key: probability_threshold(yc, p, wc) for key, p in scores.items()}
    return bundle


def probability_predict(bundle, rows):
    x = point_matrix(rows, bundle["fields"])
    result = {}
    for name, model in bundle["models"].items():
        values = x if name == PROB_NAMES[0] else (x - bundle["center"]) / bundle["scale"]
        if name == "logistic_age_only":
            values = values[:, :1]
        result[name] = np.mean([cal.predict_proba(logit(predictor.predict_proba(values)[:, 1]))[:, 1]
                                for predictor, cal in zip(model["predictors"], model["platt"], strict=True)], axis=0)
    result["constant"] = np.full(len(rows), bundle["constant"])
    return result


def fit_quantiles(core):
    x = point_matrix(core, PROB_FIELDS)
    w = normalize(weights(core, "wide_weight"))
    y = core.target.to_numpy()
    return {"fields": PROB_FIELDS, "quantiles": [
        [GradientBoostingRegressor(**Q_CONFIG, alpha=q, random_state=seed).fit(x, y, sample_weight=w)
         for seed in SEEDS] for q in QUANTILES]}


def quantile_bounds(model, rows, point):
    x = point_matrix(rows, model["fields"])
    q = np.sort(np.array([np.mean([estimator.predict(x) for estimator in seed_models], axis=0)
                           for seed_models in model["quantiles"]]), axis=0)
    return np.minimum(q[0], point), np.maximum(q[-1], point)


def conformal_radius(lower, upper, target):
    score = np.maximum.reduce([lower - target, target - upper, np.zeros(len(target))])
    rank = int(np.ceil((len(target) + 1) * 0.9))
    if rank > len(target):
        raise ValueError("CQR calibration cohort cannot produce a finite interval")
    return float(np.partition(score, rank - 1)[rank - 1])


def cqr_predict(bundle, rows, point):
    lower, upper = quantile_bounds(bundle, rows, point)
    return lower - bundle["radius"], upper + bundle["radius"]


def output_job(task):
    number, fold, chosen, config = task
    core, early, cqr_cal, prob_cal, valid = (fold[key] for key in
                                             ("fit", "early_stop", "cqr", "probability", "validation"))
    point = freeze_point(chosen, core, early, config)
    probability = fit_probability(core, prob_cal)
    quantiles = fit_quantiles(core)
    point_cal = bundle_predict(point, cqr_cal)
    lo, hi = quantile_bounds(quantiles, cqr_cal, point_cal)
    quantiles["radius"] = conformal_radius(lo, hi, cqr_cal.target.to_numpy())
    point_valid = bundle_predict(point, valid)
    lower, upper = cqr_predict(quantiles, valid, point_valid)
    return number, point_valid, lower, upper, probability_predict(probability, valid), probability["thresholds"], quantiles["radius"]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def probability_metrics(rows, score, pred):
    y = rows.target.ge(140).to_numpy()
    w = weights(rows, "wide_weight")
    tp, fn, tn, fp = (w[mask].sum() for mask in (pred & y, ~pred & y, ~pred & ~y, pred & ~y))
    bins = np.minimum((score * 10).astype(int), 9)
    curve = []
    for number in range(10):
        inside = bins == number
        curve.append({"low_inclusive": number / 10, "high": (number + 1) / 10, "n": int(inside.sum()),
                      "weighted_mean_probability": float(np.average(score[inside], weights=w[inside])) if inside.any() else None,
                      "weighted_event_rate": float(np.average(y[inside], weights=w[inside])) if inside.any() else None})
    return {"n": len(y), "event_n": int(y.sum()), "weighted_prevalence": float(np.average(y, weights=w)),
            "roc_auc": float(roc_auc_score(y, score, sample_weight=w)) if np.unique(y).size == 2 else None,
            "pr_auc_average_precision": float(average_precision_score(y, score, sample_weight=w)) if y.any() else None,
            "brier": float(brier_score_loss(y, score, sample_weight=w)),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn), "calibration_bins": curve,
            "undiagnosed_sensitivity": {
                key: {"n": int(mask.sum()),
                      "sensitivity": float(np.average(pred[mask], weights=weights(rows.loc[mask], "wide_weight")))
                      if mask.any() else None} for key, mask in subgroup_masks(rows).items()}}


def interval_metrics(rows, lower, upper, column):
    y = rows.target.to_numpy()
    hit, width = (y >= lower) & (y <= upper), upper - lower
    assert np.isfinite(width).all() and (width >= 0).all()
    ranges = {name: rows.target.ge(lo) & rows.target.lt(hi) for name, (lo, hi) in RANGES.items()}
    groups = {"overall": pd.Series(True, index=rows.index), **ranges, **subgroup_masks(rows)}
    out = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        if n:
            w = weights(rows.loc[mask], column)
            out[name] = {"n": n, "coverage": float(hit[mask].mean()), "width": float(width[mask].mean()),
                         "weighted_coverage": float(np.average(hit[mask], weights=w)),
                         "weighted_width": float(np.average(width[mask], weights=w))}
        else:
            out[name] = {"n": 0, "coverage": None, "width": None, "weighted_coverage": None, "weighted_width": None}
    return out


def cv_outputs(pool, frame, folds, chosen, config, selected, baseline):
    futures = [pool.submit(output_job, (i, fold, chosen, config)) for i, fold in enumerate(folds)]
    parts, calibration = [], {}
    for future in as_completed(futures):
        index, point, lower, upper, p, thresholds, radius = future.result()
        ids = folds[index]["validation"].index
        if not np.allclose(point, selected.loc[ids], rtol=0, atol=1e-6):
            raise ValueError("CQR point anchors do not reproduce the selected OOF prediction")
        parts.append(pd.DataFrame({"lo": lower, "hi": upper, **p,
                                   **{f"{key}_decision": p[key] >= cutoff for key, cutoff in thresholds.items()}}, index=ids))
        calibration[str(index + 1)] = {"thresholds": thresholds, "cqr_radius": radius,
                                      "probability_n": len(folds[index]["probability"]),
                                      "cqr_n": len(folds[index]["cqr"])}
        print(f"Probability and CQR: fold {index + 1} complete", flush=True)
    predictions = pd.concat(parts, verify_integrity=True).sort_index()
    rows = frame.loc[predictions.index]
    paired = rows.loc[baseline.index]
    subset = predictions.loc[paired.index]
    offset = read_json(ROOT / "artifacts/metadata.json")["predictionInterval"]
    return {
        "fields": PROB_FIELDS, "fold_calibration": calibration,
        "probability": {name: probability_metrics(rows, predictions[name].to_numpy(),
                                                   predictions[f"{name}_decision"].to_numpy()) for name in PROB_NAMES},
        "interval": {
            "cqr_complete_case": interval_metrics(rows, predictions.lo.to_numpy(), predictions.hi.to_numpy(), "wide_weight"),
            "cqr_paired": interval_metrics(paired, subset.lo.to_numpy(), subset.hi.to_numpy(), "fasting_weight"),
            "v2_symmetric_paired": interval_metrics(paired, baseline.to_numpy() + offset["lowerOffset"],
                                                     baseline.to_numpy() + offset["upperOffset"], "fasting_weight")}}


def exploratory_job(task):
    label, payload = task
    _, fold, result = point_job(payload)
    return label, fold, result


def bias_by_range(rows, predictions):
    result = {}
    for name, (lo, hi) in {"overall": (-np.inf, np.inf), **RANGES}.items():
        mask = rows.target.ge(lo) & rows.target.lt(hi)
        selected = rows.loc[mask]
        err = predictions.loc[selected.index].to_numpy() - selected.target.to_numpy()[:, None]
        w = weights(selected, "wide_weight")
        bias, mae = (np.average(x, axis=0, weights=w) for x in (err, np.abs(err)))
        result[name] = {"n": len(selected),
                        "signed_bias": {"mean": float(bias.mean()), "sd": float(bias.std(ddof=1))},
                        "mae": {"mean": float(mae.mean()), "sd": float(mae.std(ddof=1))}}
    return result


def distribution_shift(rows):
    result = {}
    for field in ("totalCholesterol", "hba1c", "target"):
        per_cycle = {}
        for cycle in (OLD, NEW):
            subset = rows.loc[rows.cycle.eq(cycle)]
            w = weights(subset, "wide_weight")
            values = subset[field].to_numpy()
            center = np.average(values, weights=w)
            per_cycle[cycle] = {"mean": float(center), "sd": float(np.sqrt(np.average((values - center) ** 2, weights=w)))}
        a, b = per_cycle[OLD], per_cycle[NEW]
        result[field] = {**per_cycle, "standardized_difference_new_minus_old":
                         float((b["mean"] - a["mean"]) / np.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2))}
    return result


def diagnose_c3(pool, frame, roles, config):
    arms = {"old_equal": ("hba1c_old_full", "equal_mass"),
            "both_equal": ("hba1c_both", "equal_mass"),
            "both_raw": ("hba1c_both", "raw_survey"),
            "old_unweighted": ("hba1c_old_full", "unweighted"),
            "both_unweighted": ("hba1c_both", "unweighted")}
    spec = {"family": "mlp", "fields": A1C, "weight": 1, "alpha": None}
    futures = []
    all_ids = {}
    for name, (arm, weighting) in arms.items():
        for i, fold in enumerate(arm_folds(frame, roles, arm)):
            core, early, validation = [frame.loc[fold[key]] for key in ("fit", "early_stop", "validation")]
            all_ids[name, i] = validation.index
            futures.append(pool.submit(exploratory_job,
                                       (name, (spec, core, early, validation, config, i, "wide_weight", weighting))))
    parts = {}
    for future in as_completed(futures):
        name, i, predictions = future.result()
        parts.setdefault(name, []).append(pd.DataFrame(np.asarray(predictions).T, index=all_ids[name, i]))
        print(f"Exploratory C3: {name}, fold {i + 1} complete", flush=True)
    all_metrics = {}
    for name, series in parts.items():
        predictions = pd.concat(series, verify_integrity=True).sort_index()
        rows = frame.loc[predictions.index]
        all_metrics[name] = {label: bias_by_range(group, predictions)
                             for label, group in (("combined", rows), *rows.groupby("cycle"))}
    # For one cycle the raw weights and equal-mass weights become identical after mean normalization.
    all_metrics["old_raw"] = all_metrics["old_equal"]
    comparisons = {}
    for method in ("equal", "raw", "unweighted"):
        comparisons[method] = {}
        a, b = (all_metrics[f"{scope}_{method}"] for scope in ("old", "both"))
        for group in ("combined", OLD, NEW):
            comparisons[method][group] = {
                name: {key: b[group][name][key]["mean"] - a[group][name][key]["mean"]
                       for key in ("mae", "signed_bias")} for name in a[group]}
    ids = np.concatenate([fold["validation"] for fold in roles])
    rows = frame.loc[ids]
    return {"exploratory_only": True, "training_weight_strategies": arms, "metrics": all_metrics,
            "delta_both_minus_old": comparisons, "weighted_distribution_shift": distribution_shift(rows),
            "n": len(rows), "roster_sha256": hash_ids(ids)}


def final_sets(frame, splits):
    ids_all = frame.index.to_numpy()
    from m2_v5_cv import SEED as split_seed
    from sklearn.model_selection import train_test_split
    remaining, cqr = split_off(frame, ids_all, .10, split_seed)
    cv, probability = train_test_split(remaining, test_size=len(cqr), stratify=frame.loc[remaining, "stratum"],
                                       random_state=split_seed)
    for label, ids in (("cv_pool", cv), ("final_cqr", cqr), ("final_probability", probability)):
        if hash_ids(ids) != splits[label]["sha256"]:
            raise ValueError(f"Final {label} membership differs from the C freeze")
    assert len(set(cv) | set(cqr) | set(probability)) == sum(map(len, (cv, cqr, probability)))
    return np.sort(cv), np.sort(cqr), np.sort(probability)


def freeze_bundles(frame, splits, config, chosen, ridge, base_spec):
    cv, cqrcal, pcal = final_sets(frame, splits)
    cv_frame = frame.loc[cv]
    common = cv_frame.loc[cv_frame.pad680.notna()]
    core_ids, stop_ids = split_off(frame, common.index.to_numpy(), .15, SEED)
    point_core, early = frame.loc[core_ids], frame.loc[stop_ids]
    references = {
        "point": chosen,
        "ridge_reference": ridge,
        "point_base_5f": {**base_spec, "form": chosen["form"]},
    }
    selected = {}
    for name, spec in references.items():
        fit = point_core if spec["family"] == "mlp" else common
        selected[name] = freeze_point(spec, fit, early, config)
    probability = fit_probability(cv_frame, frame.loc[pcal])
    cqr = fit_quantiles(cv_frame)
    cal = frame.loc[cqrcal]
    if "pad680" in chosen["fields"]:
        cal = cal.loc[cal.pad680.notna()]
    lo, hi = quantile_bounds(cqr, cal, bundle_predict(selected["point"], cal))
    cqr["radius"] = conformal_radius(lo, hi, cal.target.to_numpy())
    bundles = {**selected, "probability": probability, "cqr": cqr}

    paths = {name: BUNDLE / f"{name}.joblib" for name in bundles}
    relatives = [str(path.relative_to(ROOT.parent)) for path in paths.values()]
    checks = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT.parent,
                            input="\n".join(relatives) + "\n", capture_output=True, text=True, check=True)
    if set(checks.stdout.splitlines()) != set(relatives):
        raise ValueError("Local frozen model files must all be ignored by Git")
    BUNDLE.mkdir(parents=True)
    artifacts = {}
    for name, path in paths.items():
        joblib.dump(bundles[name], path)
        artifacts[name] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    sample = cal.iloc[:8]
    assert len(sample) == 8
    original = bundle_predict(selected["point"], sample)
    for name in selected:
        assert np.array_equal(bundle_predict(selected[name], sample), bundle_predict(joblib.load(paths[name]), sample))
    for key, values in probability_predict(probability, sample).items():
        assert np.array_equal(values, probability_predict(joblib.load(paths["probability"]), sample)[key])
    for before, after in zip(cqr_predict(cqr, sample, original),
                             cqr_predict(joblib.load(paths["cqr"]), sample, original), strict=True):
        assert np.array_equal(before, after)

    return {"artifacts": artifacts, "point_common_fit_rows": describe_ids(frame, common.index),
            "point_mlp_fit_rows": describe_ids(frame, core_ids), "point_mlp_early_stop_rows": describe_ids(frame, stop_ids),
            "probability_and_quantile_fit_rows": describe_ids(frame, cv),
            "probability_calibration": describe_ids(frame, pcal),
            "cqr_calibration": describe_ids(frame, cal.index),
            "probability_thresholds": probability["thresholds"], "cqr_radius": cqr["radius"],
            "serialization_replay": {"rows": len(sample), "exact_prediction_agreement": True}}


def complete_stage_d(pool, frame, roles, folds, config, report, chosen, ridge, base_spec, selected, baseline):
    report["probability_and_cqr_cv"] = cv_outputs(pool, frame, folds, chosen, config, selected, baseline)
    write_json(REPORT, report)
    report["exploratory_c3"] = diagnose_c3(pool, frame, roles, config)
    write_json(REPORT, report)
    print("Exploratory diagnosis complete; freezing local artifacts", flush=True)
    frozen = freeze_bundles(frame, read_json(SPLITS)["splits"], config, chosen, ridge, base_spec)
    report["final_freeze"] = {key: value for key, value in frozen.items() if key != "artifacts"}
    write_json(REPORT, report)
    src = ROOT / "research"
    manifest = {
        "protocol_sha256": sha256(PROTOCOL), "split_manifest_sha256": sha256(SPLITS),
        "grid_report_sha256": sha256(REPORT),
        "scripts": {name: sha256(src / name) for name in ("m2_v5_grid.py", "m2_v5_grid_outputs.py",
                                                        "m2_v5_cv.py", "m2_v5_audit.py")},
        "mlp_source_sha256": sha256(ROOT / "src/medpredict_ml/model.py"),
        "baseline_onnx_sha256": sha256(ROOT / "artifacts/model.onnx"),
        "baseline_metadata_sha256": sha256(ROOT / "artifacts/metadata.json"),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pyreadstat", "torch",
                                                      "catboost", "tabm", "rtdl-num-embeddings", "joblib")},
        "point": {**descriptor(chosen), "form": chosen["form"]},
        "ridge_reference": {**descriptor(ridge), "form": ridge["form"]},
        "point_base_5f": {**base_spec, "form": chosen["form"]},
        "point_selection_deployment_candidate": report["selection_trace"]["deployment_candidate"],
        "deployment_blocking_risk_groups": report["deployment_blocking_risk_groups"],
        "feature_contract": {"age": "years", "sex": {"male": 0, "female": 1}, "bmi": "kg/m^2",
                             "totalCholesterol": "mmol/L", "hba1c": "percent", "pad680": "minutes/day",
                             "categorical_encoding": "fixed one-hot categories after numeric fields",
                             "categories": {key: values for key, values in CATEGORIES.items()
                                            if key != "medication_legacy"}},
        "point_config": config,
        "probability_config": {"fields": PROB_FIELDS, "classifier": PROB_CONFIG, "logistic_C": 1,
                               "platt_C": 1, "event_sbp_gte": 140, "calibration_min_specificity": .8},
        "cqr_config": {"fields": PROB_FIELDS, "config": Q_CONFIG, "quantiles": QUANTILES, "level": .9},
        "final": frozen,
        "sealed_cycle_manifest_reference": {"stage_b_sha256": sha256(ROOT / "reports/m2_v5_audit.json"),
                                            "cycle": "2015-2016", "records_opened": False},
    }
    write_json(MANIFEST, manifest)
