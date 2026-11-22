"""Freeze the pre-approved secondary point model without reading the sealed NHANES cycle."""
from __future__ import annotations

import hashlib
import subprocess

import joblib
import numpy as np
import pandas as pd

from m2_v5_cv import A1C, ARMS, FEATURES, arm_folds, describe_ids, measure, split_off
from m2_v5_grid import (BUNDLE, MANIFEST, PROTOCOL, REPORT, ROOT, SEED, SPLITS, bundle_predict,
                        common_folds, configuration, descriptor, freeze_point, load_inputs, select,
                        subgroup_point, worker_init)
from m2_v5_grid_outputs import (cqr_predict, conformal_radius, final_sets, probability_predict, quantile_bounds)
from medpredict_ml.common import read_json, sha256, write_json
from medpredict_ml.external import onnx_predict


def probe_rows():
    golden = read_json(ROOT / "artifacts/golden.json")
    rows = pd.DataFrame([case["input"] for case in golden["cases"]])
    rows["sex"] = rows.sex.map({"male": 0., "female": 1.})
    n = len(rows)
    rows["hba1c"] = np.resize([5.1, 6.3, 7.2, 5.8], n)
    rows["medication"] = np.resize(["yes", "no", "unknown"], n)
    rows["bpq020"] = np.resize(["yes", "no", "unknown"], n)
    rows["smoking"] = np.resize(["never", "current", "former", "unknown"], n)
    rows["pad680"] = np.resize([300., 700., 50., 0.], n)
    return rows, golden


def digest_predictions(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def load_artifact(item):
    relative = item["path"]
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(BUNDLE.resolve()) or not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"Frozen artifact hash mismatch: {relative}")
    return joblib.load(path)


def load_models(manifest):
    approved = manifest["pre_e"]
    items = {**manifest["final"]["artifacts"], **approved["additional_artifacts"]}
    return {name: load_artifact(item) for name, item in items.items()}


def probe_predictions(models):
    rows, golden = probe_rows()
    onnx = onnx_predict(ROOT / "artifacts/model.onnx", rows[FEATURES].to_numpy())
    expected = np.array([case["expected"] for case in golden["cases"]])
    np.testing.assert_allclose(onnx, expected, rtol=0, atol=golden["absoluteTolerance"])
    point = {name: bundle_predict(models[name], rows) for name in
             ("point", "ridge_reference", "point_base_5f", "safe_point", "safe_base_5f")}
    probability = probability_predict(models["probability"], rows)
    cqr = models["cqr"]
    intervals = {
        "main": cqr_predict(cqr, rows, point["point"]),
        "safe": cqr_predict({**cqr, "radius": models["safe_cqr_radius"]}, rows, point["safe_point"]),
    }
    return {"synthetic_n": len(rows), "v2_onnx_sha256": digest_predictions(onnx),
            "v2_python_golden_max_abs_error": float(np.max(np.abs(onnx - expected))),
            "point_sha256": {name: digest_predictions(pred) for name, pred in point.items()},
            "probability_sha256": {name: digest_predictions(pred) for name, pred in probability.items()},
            "interval_sha256": {name: digest_predictions(np.column_stack(pred)) for name, pred in intervals.items()}}


def replay_selection(frame, roles, config, chosen, safe):
    main_base = {**descriptor(chosen), "fields": A1C, "form": chosen["form"]}
    safe_base = {**descriptor(safe), "fields": A1C, "form": safe["form"]}
    historical = {**descriptor(safe), "fields": FEATURES, "family": "mlp", "weight": 1, "form": "seed_2026"}
    specs = {"main": chosen, "main_base_5f": main_base, "safe": safe, "safe_base_5f": safe_base, "fold_v2": historical}
    pieces = []
    for common, original in zip(common_folds(frame, roles), arm_folds(frame, roles, "v2_recipe"), strict=True):
        validation = common["validation"].loc[common["validation"].paired]
        predictions = {}
        for key, spec in specs.items():
            core = frame.loc[original["fit"]] if key == "fold_v2" else common["fit"]
            early = frame.loc[original["early_stop"]] if key == "fold_v2" else common["early_stop"]
            if key == "fold_v2":
                from m2_v5_grid import fit_point
                bundle = {"spec": spec, "models": [fit_point(spec, core, early, "fasting_weight", SEED, config)]}
            else:
                bundle = freeze_point(spec, core, early, config)
            predictions[key] = bundle_predict(bundle, validation)
        pieces.append(pd.DataFrame(predictions, index=validation.index))
    pred = pd.concat(pieces, verify_integrity=True).sort_index()
    rows = frame.loc[pred.index]
    metrics = {name: measure(rows, pred[name].to_numpy(), "fasting_weight") for name in pred}
    for name, expected in (("main", chosen["metrics"]), ("safe", safe["metrics"]),
                           ("fold_v2", read_json(REPORT)["fold_v2"])):
        np.testing.assert_allclose([metrics[name][k] for k in ("overall_mae", "high_mae")],
                                   [expected[k] for k in ("overall_mae", "high_mae")], rtol=0, atol=1e-8)
    return metrics, subgroup_point(rows, {name: pred[name] for name in pred})


def main():
    worker_init()
    manifest = read_json(MANIFEST)
    if "pre_e" in manifest or manifest["grid_report_sha256"] != sha256(REPORT) or manifest["split_manifest_sha256"] != sha256(SPLITS):
        raise ValueError("Expected the verified stage D freeze without a pre-E supplement")
    grid = read_json(REPORT)
    safe, trace = select([row for row in grid["point_grid"] if "bpq020" not in row["fields"]])
    if not trace["deployment_candidate"]:
        raise ValueError("No eligible no-BPQ020 secondary candidate; review before running E")
    frame, roles, splits = load_inputs()
    config = configuration()
    metrics, subgroups = replay_selection(frame, roles, config, grid["selected"], safe)
    cv, qcal, _ = final_sets(frame, splits)
    common = frame.loc[cv]
    common = common.loc[common.pad680.notna()]
    core_ids, early_ids = split_off(frame, common.index.to_numpy(), .15, SEED)
    core, early = frame.loc[core_ids], frame.loc[early_ids]
    for label, ids in (("point_mlp_fit_rows", core_ids), ("point_mlp_early_stop_rows", early_ids)):
        if describe_ids(frame, ids)["sha256"] != manifest["final"][label]["sha256"]:
            raise ValueError(f"Added model {label} differs from stage D frozen roster")
    safe_base = {**descriptor(safe), "fields": A1C, "form": safe["form"]}
    models = {name: load_artifact(item) for name, item in manifest["final"]["artifacts"].items()}
    models["safe_point"] = freeze_point(safe, core, early, config)
    models["safe_base_5f"] = freeze_point(safe_base, core, early, config)
    # The six-field alternative needs no PAD680 to calibrate its independently anchored interval.
    cal = frame.loc[qcal]
    lo, hi = quantile_bounds(models["cqr"], cal, bundle_predict(models["safe_point"], cal))
    models["safe_cqr_radius"] = conformal_radius(lo, hi, cal.target.to_numpy())

    additions = {}
    for name in ("safe_point", "safe_base_5f", "safe_cqr_radius"):
        path = BUNDLE / f"{name}.joblib"
        relative = str(path.relative_to(ROOT.parent))
        subprocess.run(["git", "check-ignore", "--quiet", relative], cwd=ROOT.parent, check=True)
        if path.exists():
            raise ValueError(f"Refusing to replace existing {name}")
        joblib.dump(models[name], path)
        additions[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
    before = probe_predictions(models)
    for name, artifact in additions.items():
        models[name] = load_artifact(artifact)
    if before != probe_predictions(models):
        raise ValueError("Reloaded pre-E candidate does not reproduce the synthetic probes")
    manifest["pre_e"] = {
        "protocol_sha256": sha256(PROTOCOL), "freeze_script_sha256": sha256(ROOT / "research/m2_v5_pre_e.py"),
        "additional_artifacts": additions,
        "safe_point": {**descriptor(safe), "form": safe["form"], "selection_trace": trace,
                       "delta_overall_vs_fold_v2": safe["delta_overall_vs_fold_v2"],
                       "high_reduction_percent": safe["high_reduction_percent"], "seed_sd": safe["sd_over_seeds"]},
        "cv_point_metrics": metrics, "cv_undiagnosed_risk": subgroups,
        "safe_cqr": {"radius": models["safe_cqr_radius"], "calibration": describe_ids(frame, cal.index)},
        "probability": {"registered_candidate": "hist_gradient_boosting", "interface_candidate": "logistic_same_inputs",
                        "interface_comparator": "logistic_age_only", "thresholds": models["probability"]["thresholds"]},
        "probe_replay": before, "secondary_comparison_multiplicity": "nominal 95% intervals, no multiplicity adjustment",
        "sealed_cycle_read": False,
    }
    write_json(MANIFEST, manifest)
    print(f"Pre-E manifest frozen: {sha256(MANIFEST)}", flush=True)


if __name__ == "__main__":
    main()
