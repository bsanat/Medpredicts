"""M2-v5 stage D: frozen-fold point grid, descriptive outputs, and local model freeze."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tabm import TabM
from torch.nn.functional import smooth_l1_loss

from m2_v5_audit import NEW, OLD, load_development
from m2_v5_cv import (A1C, ARMS, CATEGORIES, ROOT, SEED, SEEDS, SETTINGS, SPLITS, arm_folds, counts,
                      development, hash_ids, matrix, measure, split_plan, weights, worker_init)
from medpredict_ml.common import FEATURES, read_json, sha256, write_json
from medpredict_ml.model import RawInputModel, fit_preprocessor, set_seed, train_network

REPORT = ROOT / "reports/m2_v5_grid.json"
MANIFEST = ROOT / "reports/m2_v5_frozen.json"
BUNDLE = ROOT / "data/processed/m2-v5/frozen"
PROTOCOL = ROOT / "protocols/m2-v5.md"
POINT_FIELDS = ("medication", "bpq020", "smoking", "pad680")
PROB_FIELDS = [*A1C, "medication"]
FAMILY_ORDER = {"ridge": 0, "mlp": 1, "catboost": 2, "tabm": 3}
FORMS = ("seed_2026", "ensemble")
BOOTSTRAP_SEED = 3031


def configuration():
    settings = read_json(SETTINGS)
    archive = read_json(ROOT / "artifacts/selection.json")
    return {"mlp": archive["mlp_selected_config"],
            "settings": {key: settings[key] for key in ("max_epochs", "patience", "batch_size")},
            "tabm": {"n_blocks": 2, "d_block": 64, "k": 8, "dropout": 0.1, "epochs": 80,
                     "lr": 0.003, "weight_decay": 0.01, "batch_size": 128, "smooth_l1_beta_mmhg": 8},
            "catboost": {"iterations": 400, "depth": 4, "learning_rate": 0.035, "l2_leaf_reg": 5,
                         "loss_function": "MAE", "task_type": "CPU"},
            "ridge_alphas": [0.1, 1, 10, 100], "seeds": SEEDS, "high_weights": [1, 2]}


def grid_configs():
    feature_sets = [tuple(A1C), *[(*A1C, field) for field in POINT_FIELDS], (*A1C, *POINT_FIELDS)]
    yield {"family": "mean", "fields": list(A1C), "weight": 1, "alpha": None}
    for fields, multiplier in itertools.product(feature_sets, (1, 2)):
        for family, alpha in [*(("ridge", alpha) for alpha in (0.1, 1, 10, 100)),
                              ("mlp", None), ("tabm", None), ("catboost", None)]:
            yield {"family": family, "fields": list(fields), "weight": multiplier, "alpha": alpha}


def descriptor(item):
    return {key: item[key] for key in ("family", "fields", "weight", "alpha")}


def identify(spec):
    fields = "fasting_v2" if spec["fields"] == FEATURES else "+".join(spec["fields"][5:]) or "base"
    return f"{spec['family']}:{fields}:{spec['weight']}:{spec['alpha']}"


def fit_tabm(x, y, weight, prep, seed, conf):
    set_seed(seed)
    dims = {key: conf[key] for key in ("n_blocks", "d_block", "k", "dropout")}
    model = TabM.make(n_num_features=x.shape[1], d_out=1, **dims)
    shuffle = torch.Generator().manual_seed(seed)
    x = torch.as_tensor((x - prep["x_mean"]) / prep["x_scale"], dtype=torch.float32)
    yt = torch.as_tensor((y - prep["y_mean"]) / prep["y_scale"], dtype=torch.float32)
    wt = torch.as_tensor(weight / weight.mean(), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
    for _ in range(conf["epochs"]):
        for batch in torch.randperm(len(x), generator=shuffle).split(conf["batch_size"]):
            scores = model(x[batch]).reshape(len(batch), -1)
            target = yt[batch, None].expand_as(scores)
            loss = smooth_l1_loss(scores, target, beta=conf["smooth_l1_beta_mmhg"] / prep["y_scale"], reduction="none")
            optimizer.zero_grad()
            (loss.mean(dim=1) * wt[batch]).mean().backward()
            optimizer.step()
    return model.eval()


def standardize(x, prep):
    return (x - prep["x_mean"]) / prep["x_scale"]


def fit_point(spec, train, early, column, seed, config, *, weighting="equal_mass"):
    x, xe = matrix(train, early, fields=spec["fields"])
    y, ye = train.target.to_numpy(), early.target.to_numpy()
    if weighting == "equal_mass":
        w, we = weights(train, column), weights(early, column)
    elif weighting == "raw_survey":
        w, we = train[column].to_numpy(), early[column].to_numpy()
    elif weighting == "unweighted":
        w, we = np.ones(len(train)), np.ones(len(early))
    else:
        raise ValueError(f"Unknown fit weighting {weighting}")
    w = w * np.where(y >= 150, spec["weight"], 1)
    w = w / w.mean()
    family = spec["family"]
    prep = fit_preprocessor(x, y, w)
    if family == "mean":
        model = float(np.average(y, weights=w))
    elif family == "ridge":
        model = Ridge(alpha=spec["alpha"]).fit(standardize(x, prep), y, sample_weight=w)
    elif family == "mlp":
        net, _, _ = train_network(x, y, w, xe, ye, we, prep, config["mlp"], config["settings"], seed)
        model = RawInputModel(net, prep).eval()
    elif family == "tabm":
        model = fit_tabm(x, y, w, prep, seed, config["tabm"])
    elif family == "catboost":
        model = CatBoostRegressor(**config["catboost"], random_seed=seed, thread_count=1,
                                  allow_writing_files=False, verbose=False).fit(x, y, sample_weight=w)
    else:
        raise ValueError(f"Unknown family {family}")
    return {"family": family, "preprocessing": prep, "model": model}


def point_matrix(rows, fields):
    # Categories and order are fixed by the protocol. The fitted encoder does not infer categories.
    numeric = rows[[field for field in fields if field not in CATEGORIES]].to_numpy(float)
    dummy = [rows[field].eq(label).to_numpy(float) for field in fields if field in CATEGORIES
             for label in CATEGORIES[field]]
    return np.column_stack([numeric, *dummy]) if dummy else numeric


def predict_point(model, x):
    kind, prep, fitted = model["family"], model["preprocessing"], model["model"]
    if kind == "mean":
        return np.full(len(x), fitted)
    if kind == "ridge":
        return fitted.predict(standardize(x, prep))
    if kind == "catboost":
        return fitted.predict(x)
    with torch.inference_mode():
        if kind == "mlp":
            return fitted(torch.as_tensor(x, dtype=torch.float32)).numpy().ravel()
        y = fitted(torch.as_tensor(standardize(x, prep), dtype=torch.float32)).numpy().mean(axis=1).ravel()
    return y * prep["y_scale"] + prep["y_mean"]


def bundle_predict(bundle, rows):
    x = point_matrix(rows, bundle["spec"]["fields"])
    return np.mean([predict_point(model, x) for model in bundle["models"]], axis=0)


def choose_seeds(spec, form=None):
    if spec["family"] in ("mean", "ridge"):
        return (SEED,)
    return (SEED,) if form == "seed_2026" else SEEDS


def point_job(task):
    spec, fit, early, test, config, fold, component, weighting = task
    seeds = choose_seeds(spec)
    x = point_matrix(test, spec["fields"])
    predictions = [predict_point(fit_point(spec, fit, early, component, seed, config, weighting=weighting), x)
                   for seed in seeds]
    return identify(spec), fold, predictions


def common_folds(frame, roles):
    data = frame.loc[frame.pad680.notna()]
    folds = [{role: data.loc[np.intersect1d(ids, data.index)] for role, ids in fold.items()} for fold in roles]
    for fold in folds:
        for name, rows in fold.items():
            if len(rows) < 2 or not rows.index.is_unique:
                raise ValueError(f"D {name}: empty or repeated rows")
    return folds


def load_inputs():
    frame, _ = development()
    roles, splits = split_plan(frame)
    frozen = read_json(SPLITS)
    # Protocol now includes the new D note; all C membership hashes must match exactly.
    if splits != frozen["splits"]:
        raise ValueError("Recomputed folds differ from the C freeze")
    cv = read_json(ROOT / "reports/m2_v5_cv.json")
    if cv["split_manifest_sha256"] != sha256(SPLITS):
        raise ValueError("CV measurements do not match the frozen splits")
    admitted = [name for name, summary in cv["stages"]["C4"].items() if summary["admitted_to_d"]]
    if admitted != list(POINT_FIELDS):
        raise ValueError("The admitted feature registry changed")
    return frame, roles, splits


def gather(pool, tasks, cohorts):
    futures = [pool.submit(point_job, task) for task in tasks]
    rows = {}
    for index, future in enumerate(as_completed(futures), start=1):
        name, fold, values = future.result()
        cols = [str(seed) for seed in SEEDS[:len(values)]]
        rows.setdefault(name, []).append(pd.DataFrame(dict(zip(cols, values)), index=cohorts[name, fold].index))
        if index % 30 == 0 or index == len(futures):
            print(f"Point grid: {index}/{len(futures)} fold fits (all seeds per fit)", flush=True)
    return {name: pd.concat(parts, verify_integrity=True).sort_index() for name, parts in rows.items()}


def evaluate_grid(pool, frame, roles, folds, config):
    specs = list(grid_configs())
    tasks, cohorts = [], {}
    v2 = {"family": "mlp", "fields": FEATURES, "weight": 1, "alpha": None}
    old_folds = arm_folds(frame, roles, "v2_recipe")
    for index, (fold, historical) in enumerate(zip(folds, old_folds, strict=True)):
        paired = fold["validation"].loc[fold["validation"].paired]
        for spec in specs:
            cohorts[identify(spec), index] = fold["validation"]
            tasks.append((spec, fold["fit"], fold["early_stop"], fold["validation"], config,
                          index, "wide_weight", "equal_mass"))
        cohorts[identify(v2), index] = paired
        tasks.append((v2, frame.loc[historical["fit"]], frame.loc[historical["early_stop"]], paired,
                      config, index, "fasting_weight", "equal_mass"))
    predictions = gather(pool, tasks, cohorts)
    baseline = predictions.pop(identify(v2))[str(SEED)]
    paired = frame.loc[baseline.index]
    baseline_metrics = measure(paired, baseline.to_numpy(), "fasting_weight")
    results = []
    for spec in specs:
        arm = predictions[identify(spec)].loc[paired.index]
        runs = {seed: measure(paired, arm[seed].to_numpy(), "fasting_weight") for seed in arm}
        sd = {key: float(np.std([row[key] for row in runs.values()], ddof=1)) if len(runs) > 1 else None
              for key in ("overall_mae", "high_mae")}
        forms = {"deterministic": arm[str(SEED)]} if len(runs) == 1 else {
            "seed_2026": arm[str(SEED)], "ensemble": arm.mean(axis=1)}
        for form, values in forms.items():
            metric = measure(paired, values.to_numpy(), "fasting_weight")
            delta = metric["overall_mae"] - baseline_metrics["overall_mae"]
            results.append({**spec, "form": form, "metrics": metric,
                            "delta_overall_vs_fold_v2": delta,
                            "high_reduction_percent": 100 * (1 - metric["high_mae"] / baseline_metrics["high_mae"]),
                            "admissible": delta <= 0.5, "sd_over_seeds": sd, "seed_metrics": runs})
    return predictions, baseline, baseline_metrics, results


def select(results):
    available = [row for row in results if row["admissible"]]
    if not available:
        chosen = min(results, key=lambda r: (r["metrics"]["overall_mae"], r["metrics"]["high_mae"], identify(r), r["form"]))
        return chosen, {"deployment_candidate": False, "reason": "No configuration passes the overall MAE limit"}
    stages = {"admissible_count": len(available)}

    def keep(name, key, tolerance=0):
        best = min(key(item) for item in available)
        available[:] = [item for item in available if key(item) <= best + tolerance]
        stages[name] = {"best": best, "remaining_n": len(available)}

    keep("within_0_01_high", lambda row: row["metrics"]["high_mae"], 0.01)
    keep("within_0_01_overall", lambda row: row["metrics"]["overall_mae"], 0.01)
    keep("fewest_fields", lambda row: len(row["fields"]))
    if len(available) > 1 and any(row["family"] == "mean" for row in available):
        raise ValueError("Tied constant model has no position in the registered family tie-break")
    keep("family_order", lambda row: FAMILY_ORDER.get(row["family"], -1))
    keep("fixed_seed_before_ensemble", lambda row: int(row["form"] == "ensemble"))
    keep("smaller_multiplier", lambda row: row["weight"])
    keep("smaller_alpha", lambda row: row["alpha"] or 0)
    winner = min(available, key=lambda row: tuple(sorted(row["fields"])))
    stages["lexicographic_fields"] = winner["fields"]
    return winner, {"deployment_candidate": True, **stages}


def form_predictions(predictions, spec, form):
    rows = predictions[identify(spec)]
    return rows.mean(axis=1) if form == "ensemble" else rows[str(SEED)]


def reference_ridge(results, chosen):
    same = [row for row in results if row["family"] == "ridge" and row["fields"] == chosen["fields"]]
    return min(same, key=lambda r: (not r["admissible"], r["metrics"]["high_mae"],
                                    r["metrics"]["overall_mae"], r["weight"], r["alpha"]))


def subgroup_masks(rows):
    return {f"{field}_no_ge{threshold}": rows[field].eq("no") & rows.target.ge(threshold)
            for field, threshold in itertools.product(("bpq020", "medication"), (150, 140))}


def subgroup_point(rows, predictions):
    result = {}
    for label, mask in subgroup_masks(rows).items():
        subset = rows.loc[mask]
        w = weights(subset, "fasting_weight")
        result[label] = {"n": len(subset), "models": {}}
        for name, series in predictions.items():
            errors = series.loc[subset.index].to_numpy() - subset.target.to_numpy()
            result[label]["models"][name] = {
                "mae": float(np.average(np.abs(errors), weights=w)),
                "signed_bias": float(np.average(errors, weights=w))}
    return result


def psu_intervals(rows, left, right, survey):
    design = survey.loc[rows.index]
    positions = design.assign(position=np.arange(len(rows)))
    units = [list(part.groupby("psu").position.apply(np.asarray))
             for _, part in positions.groupby(["cycle", "stratum"])]
    raw = rows.fasting_weight.to_numpy()
    cycle = rows.cycle.to_numpy()
    high = rows.target.ge(150).to_numpy()
    diff = np.abs(left.to_numpy() - rows.target.to_numpy()) - np.abs(right.to_numpy() - rows.target.to_numpy())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {"overall_mae": [], "high_mae": []}
    for _ in range(1000):
        draw = np.concatenate([np.concatenate([stratum[i] for i in rng.integers(0, len(stratum), len(stratum))])
                               for stratum in units])
        for label, idx in (("overall_mae", draw), ("high_mae", draw[high[draw]])):
            if all(np.any(cycle[idx] == c) for c in (OLD, NEW)):
                samples[label].append(float(np.mean([np.average(diff[idx][cycle[idx] == c],
                                                                  weights=raw[idx][cycle[idx] == c])
                                                     for c in (OLD, NEW)])))
    return {name: {"valid_replicates": len(draw), "ci95": np.quantile(draw, [.025, .975]).tolist()
                  if len(draw) >= 950 else None} for name, draw in samples.items()}


def survey_design(frame):
    manifest = read_json(ROOT / "reports/m2_v5_audit.json")["source_manifest"]
    rows = []
    for cycle in (OLD, NEW):
        raw, _ = load_development(cycle, manifest)
        local = raw.set_index("SEQN").loc[:, ["SDMVSTRA", "SDMVPSU"]]
        local.columns = ["stratum", "psu"]
        rows.append(local.assign(cycle=cycle))
    return pd.concat(rows, verify_integrity=True).loc[frame.index]


def freeze_point(solution, frame, early, config):
    spec = descriptor(solution)
    seeds = choose_seeds(spec, solution["form"])
    return {"spec": spec, "form": solution["form"], "seeds": seeds,
            "models": [fit_point(spec, frame, early, "wide_weight", seed, config) for seed in seeds]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    if args.workers < 1 or REPORT.exists() or MANIFEST.exists() or BUNDLE.exists():
        parser.error("positive workers and fresh D output paths are required")
    frame, roles, splits = load_inputs()
    folds, config = common_folds(frame, roles), configuration()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init) as pool:
        predictions, baseline, baseline_metrics, results = evaluate_grid(pool, frame, roles, folds, config)
        chosen, trace = select(results)
        ridge = reference_ridge(results, chosen)
        base_spec = {**descriptor(chosen), "fields": A1C}
        base = form_predictions(predictions, base_spec, chosen["form"])
        selected = form_predictions(predictions, chosen, chosen["form"])
        paired = frame.loc[baseline.index]
        subgroups = subgroup_point(paired, {"fold_v2": baseline, "base_5f": base, "selected": selected})
        blockers = []
        for name, group in subgroups.items():
            upgraded, existing = (group["models"][key] for key in ("selected", "base_5f"))
            if ("bpq020" in chosen["fields"] and
                (upgraded["mae"] > existing["mae"] or abs(upgraded["signed_bias"]) > abs(existing["signed_bias"]))):
                blockers.append(name)
        ridge_oof = form_predictions(predictions, ridge, ridge["form"]).loc[paired.index]
        report = {"protocol_sha256": sha256(PROTOCOL), "split_manifest_sha256": sha256(SPLITS),
                  "config": config, "scope": {"sealed_cycle_read": False, "stage_e_started": False},
                  "cohort": {"cv_original": splits["cv_pool"]["n"],
                             "cv_complete_case": counts(frame.loc[selected.index]),
                             "cv_complete_case_paired": counts(paired),
                             "excluded_missing_pad680": splits["cv_pool"]["n"] - len(selected),
                             "paired_ids_sha256": hash_ids(paired.index),
                             "complete_case_ids_sha256": hash_ids(selected.index)},
                  "fold_v2": baseline_metrics, "point_grid": results, "selected": chosen,
                  "selection_trace": trace, "ridge_reference": ridge,
                  "paired_minus_ridge_ci95": psu_intervals(paired, selected.loc[paired.index], ridge_oof,
                                                          survey_design(frame)),
                  "undiagnosed_point": subgroups, "deployment_blocking_risk_groups": blockers}
        write_json(REPORT, report)
        print(f"Point model selected: {identify(chosen)} {chosen['form']}", flush=True)
        from m2_v5_grid_outputs import complete_stage_d
        complete_stage_d(pool, frame, roles, folds, config, report, chosen, ridge, base_spec, selected, baseline)
    print(f"Frozen study manifest: {sha256(MANIFEST)}", flush=True)


if __name__ == "__main__":
    main()
