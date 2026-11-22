"""M2-v5 stage C point-estimation experiments. Reads development tables only."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from m2_v5_audit import NEW, OLD, ROOT, SEED, load_development
from medpredict_ml.common import FEATURES, read_json, sha256, write_json
from medpredict_ml.data import COLUMNS, prepare_frame
from medpredict_ml.metrics import regression_metrics
from medpredict_ml.model import RawInputModel, fit_preprocessor, train_network

REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "m2_v5_cv.json"
SPLITS = REPORTS / "m2_v5_cv_splits.json"
PROTOCOL = ROOT / "protocols/m2-v5.md"
SETTINGS = ROOT / "configs/experiment.json"
MODEL_SOURCE = ROOT / "src/medpredict_ml/model.py"
SEEDS = tuple(range(2026, 2031))
A1C = [*FEATURES[:-1], "hba1c"]
CATEGORIES = {"medication": ["yes", "no", "unknown"], "medication_legacy": ["yes", "no", "unknown"],
              "bpq020": ["yes", "no", "unknown"], "smoking": ["never", "former", "current", "unknown"]}
HISTORICAL_CV = {"overall_mae": 10.739729004886678, "high_mae": 34.470781792200306}
MAX_BASELINE_DRIFT = {"overall_mae": 0.75, "high_mae": 5.0}


def medication_status(frame, cycle):
    old_route = cycle == OLD
    direct = frame["BPQ050A" if old_route else "BPQ150"]
    reached = frame.BPQ020.eq(1)
    no = frame.BPQ020.eq(2)
    if old_route:
        no |= reached & frame.BPQ040A.eq(2)
        reached &= frame.BPQ040A.eq(1)
    if (direct.notna() & ~reached).any():
        raise ValueError(f"{cycle}: medication response outside documented route")
    return pd.Series(np.select([reached & direct.eq(1), no | (reached & direct.eq(2))],
                               ["yes", "no"], default="unknown"), index=frame.index)


def positive(values):
    return np.isfinite(values) & (values > 0)


def old_eligibility(frame, bounds):
    eligible = pd.Series(True, index=frame.index)
    for key, (minimum, maximum) in bounds.items():
        eligible &= frame[key].between(minimum, maximum)
    return eligible


def counts(frame):
    return {"n": len(frame), "high_n": int(frame.target.ge(150).sum())}


def medication_summary(frame):
    return {name: {label: counts(frame.loc[frame[name].eq(label)]) for label in CATEGORIES[name]}
            for name in ("medication", "medication_legacy")}


def reconcile(historical, paired, rules):
    old_n = len(historical)
    remaining = historical
    exclusions = {}
    for name, mask in rules:
        excluded = remaining.loc[~mask.loc[remaining.index]]
        exclusions[name] = counts(excluded)
        remaining = remaining.loc[mask.loc[remaining.index]]
    if not remaining.index.equals(paired.index):
        raise ValueError("Archived cohort and sequential protocol exclusions do not reconcile")
    return {"historical_n": old_n, "paired_n": len(paired), "exclusions": exclusions}


def development():
    audit = read_json(REPORTS / "m2_v5_audit.json")
    if audit["stop_findings"] or audit["sealed_records_parsed"]:
        raise ValueError("Stage B is not approved for CV")
    settings = read_json(SETTINGS)
    metadata = read_json(ROOT / "artifacts/metadata.json")
    bounds = {f["key"]: (f["min"], f["max"]) for f in metadata["fields"] if f["type"] == "number"}
    frames, checks = [], {}
    for cycle in (OLD, NEW):
        frame, _ = load_development(cycle, audit["source_manifest"])
        wide, fasting, drug = ("WTMECPRP", "WTSAFPRP", "BPQ050A") if cycle == OLD else ("WTPH2YR", "WTSAF2YR", "BPQ150")
        frame = frame.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "BMXBMI": "bmi",
                                      "LBDTCSI": "totalCholesterol", "LBDGLUSI": "glucose", "LBXGH": "hba1c"})
        frame["medication"] = medication_status(frame, cycle)
        frame["medication_legacy"] = frame[drug].map({1: "yes", 2: "no"}).fillna("unknown")
        frame["bpq020"] = frame.BPQ020.map({1: "yes", 2: "no"}).fillna("unknown")
        smoke = frame.SMQ040.where(frame.SMQ020.eq(1)).map({1: "current", 2: "current", 3: "former"})
        frame["smoking"] = smoke.mask(frame.SMQ020.eq(2), "never").fillna("unknown")
        frame["pad680"] = frame.PAD680.where(frame.PAD680.between(0, 1439))
        # Reuse the historical target and fasting cohort implementation on the audited joins.
        mapping = {"age": "RIDAGEYR", "sex": "RIAGENDR", "bmi": "BMXBMI", "totalCholesterol": "LBDTCSI",
                   "glucose": "LBDGLUSI", fasting: "WTSAFPRP", drug: "BPQ050A"}
        if cycle == OLD:
            mapping = {key: value for key, value in mapping.items() if key != value}
        legacy = frame.rename(columns=mapping)
        tables = {name: legacy[columns] for name, columns in COLUMNS.items()}
        historical, _ = prepare_frame(tables, settings)
        systolic = frame[[f"BPXOSY{i}" for i in (1, 2, 3)]].to_numpy(float)
        diastolic = frame[[f"BPXODI{i}" for i in (1, 2, 3)]].to_numpy(float)
        valid = np.isfinite(systolic) & (systolic > 0) & (systolic <= 300)
        valid &= np.isnan(diastolic) | (np.isfinite(diastolic) & (diastolic > 0) & (systolic > diastolic))
        targets = pd.DataFrame(np.where(valid, systolic, np.nan)).mean(axis=1)
        target = targets.where(valid.sum(axis=1) >= settings["minimum_sbp_readings"])
        indexed = frame.assign(target=target).set_index("SEQN").sort_index()
        assert np.allclose(indexed.loc[historical.SEQN, "target"], historical.target, rtol=0, atol=1e-12)
        in_domain = old_eligibility(indexed, bounds)
        historical_frame = indexed.loc[historical.SEQN]
        if cycle == NEW:
            historical_frame = historical_frame.loc[in_domain.loc[historical_frame.index]]
        wide_ok, hba1c_ok = positive(indexed[wide]), positive(indexed.hba1c)
        eligible = (indexed.age.between(settings["age_min"], settings["age_max"]) & indexed.sex.isin([1, 2])
                    & np.isfinite(indexed[["SDMVSTRA", "SDMVPSU"]]).all(axis=1) & indexed.target.notna()
                    & positive(indexed.bmi) & positive(indexed.totalCholesterol))
        expanded = indexed.loc[eligible & wide_ok & hba1c_ok].copy()
        expanded["paired"] = (positive(expanded[fasting]) & positive(expanded.glucose)
                              & in_domain.loc[expanded.index])
        paired = expanded.loc[expanded.paired]
        checks[cycle] = {"cohort_reconciliation": reconcile(
            historical_frame, paired, [("positive_expanded_weight", wide_ok), ("positive_hba1c", hba1c_ok),
                                       ("v2_runtime_bounds", in_domain)]),
            "expanded_medication": medication_summary(expanded), "paired_medication": medication_summary(paired)}
        lost = historical_frame.loc[~hba1c_ok.loc[historical_frame.index]]
        checks[cycle]["invalid_hba1c_in_old_cohort"] = {
            "missing": int(lost.hba1c.isna().sum()), "nonpositive_or_nonfinite": int(lost.hba1c.notna().sum())}
        excluded_by_bounds = historical_frame.loc[~in_domain.loc[historical_frame.index]]
        checks[cycle]["extra_bounds_exclusions"] = {
            key: int((~excluded_by_bounds[key].between(low, high)).sum()) for key, (low, high) in bounds.items()}
        expected = audit["development"][cycle]
        assert len(expanded) == expected["expanded"]["n"] and len(paired) == expected["paired_v2"]["n"]
        assert counts(expanded)["high_n"] == expected["expanded"]["sbp_bins"]["ge150"]
        assert counts(paired)["high_n"] == expected["paired_v2"]["sbp_bins"]["ge150"]
        assert len(historical_frame) == (3318 if cycle == OLD else 2680)
        expanded["sex"] = expanded.sex.map({1: 0.0, 2: 1.0})
        expanded["wide_weight"] = expanded[wide]
        expanded["fasting_weight"] = expanded[fasting]
        expanded["cycle"] = cycle
        frames.append(expanded[[*FEATURES, "hba1c", "target", "cycle", "paired", "wide_weight", "fasting_weight",
                                *CATEGORIES, "pad680"]])
    result = pd.concat(frames, verify_integrity=True).sort_index()
    result["stratum"] = result.cycle + ":" + np.digitize(result.target, [110, 140, 150]).astype(str)
    return result, checks


def split_off(frame, ids, fraction, seed):
    left, held = train_test_split(ids, test_size=round(len(ids) * fraction), random_state=seed,
                                  stratify=frame.loc[ids, "stratum"])
    return np.sort(left), np.sort(held)


def hash_ids(ids):
    canonical = "".join(f"{int(value)}\n" for value in np.sort(ids))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def describe_ids(frame, ids):
    rows = frame.loc[ids]
    return {**counts(rows), "sha256": hash_ids(ids),
            "by_cycle_and_sbp_bin": rows.stratum.value_counts().sort_index().to_dict()}


def split_plan(frame):
    ids = frame.index.to_numpy()
    rest, final_cqr = split_off(frame, ids, 0.10, SEED)
    remaining, final_prob = train_test_split(rest, test_size=len(final_cqr), random_state=SEED,
                                             stratify=frame.loc[rest, "stratum"])
    remaining, final_prob = np.sort(remaining), np.sort(final_prob)
    remaining_set, cqr_set, prob_set = map(set, (remaining, final_cqr, final_prob))
    assert remaining_set.isdisjoint(cqr_set | prob_set) and cqr_set.isdisjoint(prob_set)
    assert len(remaining_set | cqr_set | prob_set) == len(frame)
    roles = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, validation in splitter.split(remaining, frame.loc[remaining, "stratum"]):
        outer = remaining[train]
        available, cqr = split_off(frame, outer, 0.10, SEED)
        rest, prob = train_test_split(available, test_size=len(cqr), random_state=SEED,
                                       stratify=frame.loc[available, "stratum"])
        fit, early = split_off(frame, np.sort(rest), 0.15, SEED)
        parts = {"fit": fit, "early_stop": early, "cqr": cqr, "probability": np.sort(prob),
                 "validation": remaining[validation]}
        union = set().union(*(set(part) for part in parts.values()))
        assert len(union) == sum(map(len, parts.values())) and union == remaining_set
        roles.append(parts)
    validation_ids = np.concatenate([fold["validation"] for fold in roles])
    assert np.array_equal(np.sort(validation_ids), remaining)
    return roles, {
        "final_cqr": describe_ids(frame, final_cqr), "final_probability": describe_ids(frame, final_prob),
        "cv_pool": describe_ids(frame, remaining),
        "folds": [{name: describe_ids(frame, ids) for name, ids in fold.items()} for fold in roles],
    }


def cohort_mask(frame, name):
    if name == "old_fasting":
        return frame.cycle.eq(OLD) & frame.paired
    if name == "old_full":
        return frame.cycle.eq(OLD)
    if name == "observed_pad680":
        return frame.pad680.notna()
    if name == "full":
        return pd.Series(True, index=frame.index)
    raise ValueError(f"Unknown cohort {name}")


# Each arm fixes features, the fit cohort, and the evaluation cohort.
# C2 uses predictions from old_full on its shared fasting evaluation rows.
ARMS = {
    "v2_recipe": (FEATURES, "old_fasting", "old_fasting"),
    "hba1c_fasting": (A1C, "old_fasting", "old_fasting"),
    "hba1c_old_full": (A1C, "old_full", "full"),
    "hba1c_both": (A1C, "full", "full"),
    "medication": ([*A1C, "medication"], "full", "full"),
    "medication_legacy": ([*A1C, "medication_legacy"], "full", "full"),
    "bpq020": ([*A1C, "bpq020"], "full", "full"),
    "smoking": ([*A1C, "smoking"], "full", "full"),
    "pad680_base": (A1C, "observed_pad680", "observed_pad680"),
    "pad680": ([*A1C, "pad680"], "observed_pad680", "observed_pad680"),
}


def arm_folds(frame, folds, arm):
    _, train, validation = ARMS[arm]
    fit_mask, eval_mask = cohort_mask(frame, train), cohort_mask(frame, validation)
    return [{part: ids[(eval_mask if part == "validation" else fit_mask).loc[ids].to_numpy()]
             for part, ids in fold.items() if part in ("fit", "early_stop", "validation")} for fold in folds]


def weights(frame, column, high_only=False):
    rows = frame.loc[frame.target.ge(150)] if high_only else frame
    # Normalize each cycle within the scored population, including the high-SBP subset.
    raw = rows[column]
    assert np.isfinite(raw).all() and raw.gt(0).all()
    return (raw / raw.groupby(rows.cycle).transform("sum")).to_numpy()


def matrix(core, *evaluation, fields):
    numeric = [name for name in fields if name not in CATEGORIES]
    categorical = [name for name in fields if name in CATEGORIES]
    rows = (core, *evaluation)
    result = [data[numeric].to_numpy(float) for data in rows]
    if categorical:
        encoder = OneHotEncoder(categories=[CATEGORIES[name] for name in categorical], sparse_output=False)
        encoder.fit(core[categorical])
        result = [np.column_stack((x, encoder.transform(data[categorical]))) for x, data in zip(result, rows)]
    return result


def worker_init():
    torch.set_num_threads(1)
    threadpool_limits(limits=1)


def run_fold(job):
    arm, fold, seed, arrays, settings, config = job
    x, y, w, xe, ye, we, xv = arrays
    prep = fit_preprocessor(x, y, w)
    net, history, _ = train_network(x, y, w, xe, ye, we, prep, config, settings, seed)
    with torch.inference_mode():
        prediction = RawInputModel(net, prep)(torch.as_tensor(xv, dtype=torch.float32)).numpy().ravel()
    return arm, fold, seed, prediction, min(history, key=lambda item: item["validation_weighted_mae"])["epoch"]


def run_arm(pool, frame, folds, arm, settings, config):
    fields, train, _ = ARMS[arm]
    jobs, parts, epochs = [], [], []
    predictions = {}
    for fold_number, fold in enumerate(arm_folds(frame, folds, arm), start=1):
        core, early, valid = (frame.loc[fold[name]] for name in ("fit", "early_stop", "validation"))
        if min(len(core), len(early), len(valid)) == 0:
            raise ValueError(f"{arm} fold {fold_number}: empty partition")
        column = "fasting_weight" if train == "old_fasting" else "wide_weight"
        w, we = weights(core, column), weights(early, column)
        x, xe, xv = matrix(core, early, valid, fields=fields)
        y, ye = core.target.to_numpy(), early.target.to_numpy()
        for seed in SEEDS:
            jobs.append((arm, fold_number, seed, (x, y, w, xe, ye, we, xv), settings, config))
        prep = fit_preprocessor(x, y, w)
        x_mean, x_scale = np.asarray(prep["x_mean"]), np.asarray(prep["x_scale"])
        ridge = Ridge(alpha=10).fit((x - x_mean) / x_scale, y, sample_weight=w / w.mean())
        parts.append(pd.DataFrame({"ridge": ridge.predict((xv - x_mean) / x_scale),
                                   "mean": np.average(y, weights=w)}, index=valid.index))
    for _, fold, seed, prediction, epoch in pool.map(run_fold, jobs):
        predictions.setdefault(fold, {})[seed] = prediction
        epochs.append({"fold": fold, "seed": seed, "best_epoch": epoch})
    for fold, part in enumerate(parts, start=1):
        for seed in SEEDS:
            part[str(seed)] = predictions[fold][seed]
    print(f"{arm}: {len(jobs)} training runs complete", flush=True)
    return pd.concat(parts, verify_integrity=True).sort_index(), epochs


def measure(rows, prediction, column):
    high = rows.target.ge(150)
    y = rows.target.to_numpy()
    info = regression_metrics(y, prediction, weights(rows, column))
    result = {"overall_mae": info["weighted_mae"], "overall_bias": info["weighted_mean_signed_error"],
              "overall_rmse": info["weighted_rmse"], "overall_r2": info["weighted_r2"]}
    if high.sum() < 2:
        return {**result, "high_mae": None, "high_bias": None}
    tail = regression_metrics(y[high], prediction[high], weights(rows, column, high_only=True))
    return {**result, "high_mae": tail["weighted_mae"], "high_bias": tail["weighted_mean_signed_error"]}


def aggregate(rows):
    return {key: {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
            for key in rows[0] if (values := [row[key] for row in rows]) and all(value is not None for value in values)}


def paired_report(frame, base, candidate, ids, column):
    rows = frame.loc[ids]
    base, candidate = base.loc[ids], candidate.loc[ids]
    baseline_runs, candidate_runs, delta_runs = [], [], []
    for seed in SEEDS:
        a, b = [measure(rows, item[str(seed)].to_numpy(), column) for item in (base, candidate)]
        baseline_runs.append(a)
        candidate_runs.append(b)
        delta_runs.append({key: b[key] - a[key] if a[key] is not None else None for key in ("overall_mae", "high_mae")})
    a, b = aggregate(baseline_runs), aggregate(candidate_runs)
    high_reduction = (100 * (1 - b["high_mae"]["mean"] / a["high_mae"]["mean"])) if "high_mae" in a else None
    return {
        **counts(rows), "roster_sha256": hash_ids(ids), "weight": column,
        "base_mlp": a, "candidate_mlp": b, "delta_paired": aggregate(delta_runs), "high_reduction_percent": high_reduction,
        "ridge": {name: measure(rows, item.ridge.to_numpy(), column) for name, item in (("base", base), ("candidate", candidate))},
        "mean_baseline": {name: measure(rows, item["mean"].to_numpy(), column) for name, item in (("base", base), ("candidate", candidate))},
        "runs": [{"seed": seed, "base": a, "candidate": b, "delta": d}
                 for seed, a, b, d in zip(SEEDS, baseline_runs, candidate_runs, delta_runs)],
    }


def bpq_attribution(frame, base, candidate):
    rows = frame.loc[base.index]
    improvement = (base[list(map(str, SEEDS))].sub(rows.target, axis=0).abs()
                   - candidate.loc[rows.index, list(map(str, SEEDS))].sub(rows.target, axis=0).abs()).mean(axis=1)
    result = {}
    for name, mask in (("overall", pd.Series(True, index=rows.index)), ("high", rows.target.ge(150))):
        selected = rows.loc[mask]
        weight = weights(selected, "wide_weight")
        total = float(np.average(improvement.loc[selected.index], weights=weight))
        groups = {}
        for label in CATEGORIES["bpq020"]:
            group = selected.bpq020.eq(label).to_numpy()
            contribution = float(np.sum(weight[group] * improvement.loc[selected.index].to_numpy()[group]) / weight.sum())
            groups[label] = {"n": int(group.sum()), "contribution_to_mae_reduction": contribution,
                             "share_percent": 100 * contribution / total if total else None}
        assert np.isclose(sum(group["contribution_to_mae_reduction"] for group in groups.values()), total)
        result[name] = {"total_reduction": total, "groups": groups}
    return result


def freeze(frame, folds, splits, checks):
    audit_path = REPORTS / "m2_v5_audit.json"
    return {
        "protocol_sha256": sha256(PROTOCOL), "stage_b_sha256": sha256(audit_path),
        "source_manifest_sha256": hashlib.sha256(json.dumps(
            [entry for entry in read_json(audit_path)["source_manifest"] if not entry["sealed_until_stage_e"]],
            sort_keys=True).encode()).hexdigest(),
        "config_sha256": sha256(SETTINGS), "model_source_sha256": sha256(MODEL_SOURCE), "script_sha256": sha256(Path(__file__)),
        "seed": SEED, "model_seeds": SEEDS, "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "pyreadstat")},
        "canonical_id_hash": "numeric SEQN sorted ascending, decimal ASCII, one LF-terminated ID per line",
        "preflight": checks, "splits": splits,
        "arms": {name: {"features": fields, "fit_cohort": train, "score_cohort": validation,
                        "folds": [{role: describe_ids(frame, ids) for role, ids in fold.items()}
                                  for fold in arm_folds(frame, folds, name)]}
                 for name, (fields, train, validation) in ARMS.items()},
        "baseline_reproduction_reference": HISTORICAL_CV, "maximum_absolute_baseline_drift": MAX_BASELINE_DRIFT,
        "sealed_cycle_read": False,
    }


def execute(pool, frame, folds, manifest_hash):
    settings = read_json(SETTINGS)
    config = read_json(ROOT / "artifacts/selection.json")["mlp_selected_config"]
    preds, epochs = {}, {}
    result = {"split_manifest_sha256": manifest_hash, "point_recipe": config,
              "settings": {key: settings[key] for key in ("max_epochs", "patience", "batch_size")},
              "ridge_alpha": 10, "training_high_sbp_multiplier": 1, "stages": {},
              "sealed_cycle_read": False, "probability_or_cqr_trained": False, "grid_d_executed": False,
              "epochs": epochs}

    def run(name):
        preds[name], epochs[name] = run_arm(pool, frame, folds, name, settings, config)
        return preds[name]

    original = run("v2_recipe")
    sanity = paired_report(frame, original, original, original.index, "fasting_weight")
    current = {key: sanity["base_mlp"][key]["mean"] for key in HISTORICAL_CV}
    sanity.update(historical=HISTORICAL_CV, maximum_absolute_drift=MAX_BASELINE_DRIFT, observed=current,
                  passed=all(abs(current[key] - value) <= MAX_BASELINE_DRIFT[key] for key, value in HISTORICAL_CV.items()))
    result["v2_cv_reproduction"] = sanity
    write_json(OUTPUT, result)
    if not sanity["passed"]:
        raise SystemExit("v2 CV reproduction outside frozen tolerance; stopped before C1")
    print(f"v2 CV gate passed: {current}", flush=True)

    hba1c_fasting = run("hba1c_fasting")
    result["stages"]["C1"] = paired_report(frame, original, hba1c_fasting, original.index, "fasting_weight")
    write_json(OUTPUT, result)
    old_full = run("hba1c_old_full")
    result["stages"]["C2"] = paired_report(frame, hba1c_fasting, old_full, original.index, "fasting_weight")
    write_json(OUTPUT, result)
    both = run("hba1c_both")
    c3 = paired_report(frame, old_full, both, both.index, "wide_weight")
    c3["by_cycle"] = {cycle: paired_report(frame, old_full, both, both.index[frame.loc[both.index, "cycle"].eq(cycle)],
                                            "wide_weight") for cycle in (OLD, NEW)}
    result["stages"]["C3"] = c3
    write_json(OUTPUT, result)

    c4 = {}
    for name in ("medication", "bpq020", "smoking", "pad680", "medication_legacy"):
        base = run("pad680_base") if name == "pad680" else both
        comparison = run(name)
        assert base.index.equals(comparison.index)
        entry = paired_report(frame, base, comparison, base.index, "wide_weight")
        delta = entry["delta_paired"]
        entry["admitted_to_d"] = (delta["overall_mae"]["mean"] <= 0.50 and delta["high_mae"]["mean"] < 0
                                  and name != "medication_legacy")
        entry["sensitivity_only"] = name == "medication_legacy"
        if name == "bpq020":
            entry["attribution"] = bpq_attribution(frame, base, comparison)
        c4[name] = entry
        result["stages"]["C4"] = c4
        write_json(OUTPUT, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    frame, checks = development()
    folds, split_summary = split_plan(frame)
    manifest = freeze(frame, folds, split_summary, checks)
    if args.action == "freeze":
        if SPLITS.exists():
            parser.error("split manifest already exists")
        write_json(SPLITS, manifest)
        print(f"Frozen split SHA-256: {sha256(SPLITS)}", flush=True)
        return
    if json.loads(json.dumps(manifest)) != read_json(SPLITS):
        raise ValueError("Current input, configuration or split differs from the frozen manifest")
    if OUTPUT.exists():
        parser.error("CV output already exists; review it before rerunning")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init) as pool:
        execute(pool, frame, folds, sha256(SPLITS))


if __name__ == "__main__":
    main()
