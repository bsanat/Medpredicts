"""Protocol addendum v3.1: exact baseline Shapley explanations and a Mahalanobis out-of-distribution flag.

Needs `--workspace` pointing at the exact v2 reproduction that also holds the external v3 raw files.
Only aggregated statistics, synthetic golden cases and figures are written to the source repository.
"""
from __future__ import annotations

import itertools
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common
from .common import FEATURES, read_json, sha256, write_json
from .external import load_external_tables, onnx_predict
from .data import prepare_frame

N = len(FEATURES)
# All 32 coalitions as boolean masks; row k says which features take the input value (others take the reference).
MASKS = np.array(list(itertools.product([False, True], repeat=N)), dtype=bool)
SEX_INDEX = FEATURES.index("sex")


def shapley_values(predict, x: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact baseline Shapley values for each row of x; returns (phi[n,5], f(reference))."""
    x = np.atleast_2d(np.asarray(x, np.float32))
    reference = np.asarray(reference, np.float32)
    rows = np.where(MASKS[None, :, :], x[:, None, :], reference[None, None, :]).reshape(-1, N)
    values = np.asarray(predict(rows), float).reshape(len(x), len(MASKS))
    size = MASKS.sum(axis=1)
    index = {tuple(mask): k for k, mask in enumerate(MASKS)}
    phi = np.zeros((len(x), N))
    for i in range(N):
        for k, mask in enumerate(MASKS):
            if mask[i]:
                continue
            with_i = mask.copy()
            with_i[i] = True
            s = size[k]
            weight = math.factorial(s) * math.factorial(N - s - 1) / math.factorial(N)
            phi[:, i] += weight * (values[:, index[tuple(with_i)]] - values[:, k])
    return phi, float(values[0, 0])


def mahalanobis_squared(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = np.atleast_2d(np.asarray(x, float)) - mean
    return np.einsum("ij,jk,ik->i", delta, precision, delta)


def explain_ood() -> None:
    out = common.SOURCE_ROOT / "reports/explain_v3_1"
    model_path = common.SOURCE_ROOT / "artifacts/model.onnx"
    frozen = read_json(common.SOURCE_ROOT / "configs/external_v3.json")["frozen_model"]["onnx_sha256"]
    if sha256(model_path) != frozen:
        raise ValueError("Published ONNX differs from the frozen model")
    predict = lambda rows: onnx_predict(model_path, rows)
    v2 = pd.read_csv(common.ROOT / "data/processed/dataset.csv")
    train, validation = v2[v2.split == "train"], v2[v2.split == "validation"]
    xt = train[FEATURES].to_numpy(float)

    reference = np.average(xt, axis=0, weights=train.weight.to_numpy())
    mean = xt.mean(axis=0)
    covariance = np.cov(xt, rowvar=False)
    precision = np.linalg.inv(covariance)
    d2_validation = mahalanobis_squared(validation[FEATURES].to_numpy(float), mean, precision)
    threshold = float(np.quantile(d2_validation, .99))

    # Global, descriptive: mean |phi| on validation.
    phi_validation, reference_prediction = shapley_values(predict, validation[FEATURES].to_numpy(np.float32), reference)
    additivity = np.abs(phi_validation.sum(axis=1) - (predict(validation[FEATURES].to_numpy(np.float32)) - reference_prediction))
    importance = {f: float(np.mean(np.abs(phi_validation[:, i]))) for i, f in enumerate(FEATURES)}
    signed_by_feature = {f: {"mean_phi": float(np.mean(phi_validation[:, i])),
                             "corr_with_feature": float(np.corrcoef(validation[f], phi_validation[:, i])[0, 1])}
                         for i, f in enumerate(FEATURES)}

    # Descriptive OOD analysis on the already-opened external cohort (no threshold tuning).
    external, _ = prepare_frame(load_external_tables(), read_json(common.SOURCE_ROOT / "configs/experiment.json"))
    metadata = read_json(common.SOURCE_ROOT / "artifacts/metadata.json")
    bounds = {f["key"]: (f["min"], f["max"]) for f in metadata["fields"] if f["type"] == "number"}
    accepted = np.logical_and.reduce([external[k].between(lo, hi).to_numpy() for k, (lo, hi) in bounds.items()])
    external = external[accepted].reset_index(drop=True)
    xe = external[FEATURES].to_numpy(np.float32)
    prediction = predict(xe)
    flagged = mahalanobis_squared(xe, mean, precision) > threshold
    offsets = read_json(common.SOURCE_ROOT / "reports/external_v3/conformal.json")["methods"]["split_symmetric_absolute_residual"]
    inside = (external.target >= prediction + offsets["lower_offset"]) & (external.target <= prediction + offsets["upper_offset"])
    groups = {}
    for name, mask in [("flagged", flagged), ("not_flagged", ~flagged)]:
        y, p, w = external.target.to_numpy()[mask], prediction[mask], external.weight.to_numpy()[mask]
        groups[name] = {"n": int(mask.sum()), "weighted_mae": float(np.average(np.abs(p - y), weights=w)) if mask.any() else None,
                        "weighted_bias": float(np.average(p - y, weights=w)) if mask.any() else None,
                        "interval_coverage": float(inside[mask].mean()) if mask.any() else None,
                        "share_true_sbp_ge_150": float((y >= 150).mean()) if mask.any() else None}
    flagged_validation = d2_validation > threshold

    explanation_report = {"method": "exact_baseline_shapley", "reference": dict(zip(FEATURES, reference.tolist())),
        "reference_prediction": reference_prediction, "validation_n": len(validation),
        "max_additivity_error_mmhg": float(additivity.max()), "mean_abs_phi_validation": importance,
        "signed_summary_validation": signed_by_feature,
        "interpretation": "Model behaviour relative to the weighted average training participant; not causal effects."}
    ood_report = {"method": "mahalanobis_squared", "mean": mean.tolist(), "precision": precision.tolist(),
        "threshold": threshold, "threshold_rule": "99th percentile of d2 on v2 validation",
        "validation_flagged_rate": float(flagged_validation.mean()),
        "external": {"n": int(len(external)), "flagged_rate": float(flagged.mean()), "groups": groups},
        "expectation_registered": "flagged: higher error and lower coverage"}
    write_json(out / "explanation.json", explanation_report)
    write_json(out / "ood.json", ood_report)
    make_plots(out, importance, d2_validation, threshold)

    metadata["explanation"] = {"method": "exact_baseline_shapley", "reference": dict(zip(FEATURES, reference.tolist())),
        "referenceDescription": "Weighted means of the NHANES 2017–2020 training split (sex = share of women)",
        "protocol": "protocols/m2-explain-ood-v3_1.md"}
    metadata["outOfDistribution"] = {"method": "mahalanobis_squared", "mean": mean.tolist(), "precision": precision.tolist(),
        "threshold": threshold, "thresholdRule": "99th percentile on validation",
        "externalFlaggedRate": float(flagged.mean()), "protocol": "protocols/m2-explain-ood-v3_1.md"}
    write_json(common.SOURCE_ROOT / "artifacts/metadata.json", metadata)

    golden = read_json(common.SOURCE_ROOT / "artifacts/golden.json")
    cases = [case["input"] for case in golden["cases"]] + [
        {"age": 22, "sex": "female", "bmi": 45.0, "totalCholesterol": 9.5, "glucose": 15.0},
        {"age": 79, "sex": "male", "bmi": 17.0, "totalCholesterol": 2.5, "glucose": 4.0}]
    numeric = np.array([[c[f] if f != "sex" else {"male": 0, "female": 1}[c[f]] for f in FEATURES] for c in cases], np.float32)
    phi, _ = shapley_values(predict, numeric, reference)
    distance = mahalanobis_squared(numeric, mean, precision)
    write_json(common.SOURCE_ROOT / "artifacts/explanation_golden.json", {
        "absoluteTolerance": 0.01, "referencePrediction": reference_prediction,
        "source": "Synthetic inputs, not observations; expected values from Python ONNX Runtime",
        "cases": [{"input": c, "contributions": dict(zip(FEATURES, row.tolist())), "mahalanobisSquared": float(d),
                   "outOfDistribution": bool(d > threshold)} for c, row, d in zip(cases, phi, distance)]})
    print(f"Reference prediction {reference_prediction:.2f}; additivity max {additivity.max():.2e}; "
          f"OOD threshold {threshold:.2f}; external flagged {flagged.mean():.3f}; groups {groups}", flush=True)


def make_plots(out, importance: dict, d2: np.ndarray, threshold: float) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    order = sorted(importance, key=importance.get)
    plt.figure(figsize=(6, 3.6))
    plt.barh(order, [importance[k] for k in order], color="#17447a")
    plt.xlabel("Mean |Shapley contribution| on validation (mmHg)")
    plt.title("Model behaviour, not causal effects")
    plt.tight_layout(); plt.savefig(figures / "global-importance.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 3.6))
    plt.hist(d2, bins=50, color="#17447a")
    plt.axvline(threshold, color="#c0392b", linestyle="--", label=f"99th percentile = {threshold:.1f}")
    plt.xlabel("Squared Mahalanobis distance (validation)"); plt.ylabel("Participants"); plt.legend()
    plt.tight_layout(); plt.savefig(figures / "ood-distance-validation.png", dpi=160); plt.close()
