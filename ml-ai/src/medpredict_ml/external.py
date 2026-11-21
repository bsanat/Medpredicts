"""Protocol M2-external-v3: frozen-model temporal validation on NHANES 2021-2023 and conformal intervals.

`--workspace` must point at an exact v2 reproduction (download..verify) so the v2 internal test split and
the deterministic baselines are available locally. Raw and row-level data stay in that workspace; only
aggregated reports are written to the source repository under reports/external_v3/.
"""
from __future__ import annotations

import datetime as dt
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import pyreadstat

from . import common
from .common import FEATURES, read_json, sha256, write_json
from .conformal import asymmetric_offsets, coverage, symmetric_offsets
from .data import prepare_frame, valid_transport, validate_table
from .metrics import regression_metrics

# External table/column names mapped onto the v2 names so the registered v2 cohort rules apply unchanged.
TABLE_TO_V2 = {"DEMO_L": "P_DEMO", "BMX_L": "P_BMX", "TCHOL_L": "P_TCHOL", "GLU_L": "P_GLU",
               "BPXO_L": "P_BPXO", "BPQ_L": "P_BPQ"}
COLUMN_TO_V2 = {"WTSAF2YR": "WTSAFPRP", "BPQ150": "BPQ050A"}
FAMILIES = ["mean", "ridge", "gradient_boosting", "mlp"]


def settings() -> dict:
    return read_json(common.SOURCE_ROOT / "configs/external_v3.json")


def report_dir() -> Path:
    return common.SOURCE_ROOT / "reports/external_v3"


def raw_dir() -> Path:
    return common.ROOT / "data/external_v3/raw"


def download_external() -> None:
    cfg = settings()
    destination = raw_dir()
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name in cfg["external_tables"]:
        path = destination / f"{name}.xpt"
        url = cfg["external_base_url"] + path.name
        if not path.exists() or not valid_transport(path):
            temporary = path.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "MedPredict-University-Research/0.1"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    size = 0
                    while chunk := response.read(128 * 1024):
                        size += len(chunk)
                        if size > 30_000_000:
                            raise ValueError("Unexpectedly large official data file")
                        output.write(chunk)
                if not valid_transport(temporary):
                    raise ValueError(f"Official download for {name} is not a SAS transport file")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        files.append({"name": name, "url": url, "codebook": cfg["external_base_url"] + name + ".htm",
                      "bytes": path.stat().st_size, "sha256": sha256(path)})
        print(f"Verified official file: {name} ({path.stat().st_size} bytes)", flush=True)
    write_json(report_dir() / "data_manifest.json", {"dataset": cfg["external_dataset"],
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "files": files,
        "terms": "https://www.cdc.gov/nchs/policy/data-user-agreement.html"})


def load_external_tables() -> dict[str, pd.DataFrame]:
    cfg = settings()
    manifest = read_json(report_dir() / "data_manifest.json")
    tables = {}
    for item in manifest["files"]:
        path = raw_dir() / f"{item['name']}.xpt"
        if sha256(path) != item["sha256"]:
            raise ValueError("External raw file differs from its manifest")
        # Some 2021-2023 variable labels are cp1252; values are numeric.
        frame, _ = pyreadstat.read_xport(path, encoding="latin1", usecols=cfg["external_tables"][item["name"]])
        validate_table(frame, cfg["external_tables"][item["name"]], item["name"])
        tables[TABLE_TO_V2[item["name"]]] = frame.rename(columns=COLUMN_TO_V2)
    weights = tables["P_GLU"].WTSAFPRP
    if ((weights > 0) & (weights < 1)).any():
        raise ValueError("Unexpected positive fasting weight below 1; check XPT zero decoding")
    return tables


def onnx_predict(path: Path, x: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    output = session.run(["predictedValue"], {"features": x.astype(np.float32)})[0].reshape(-1)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite ONNX output")
    return output.astype(float)


def check_reproduction(cfg: dict) -> None:
    published = read_json(common.SOURCE_ROOT / "reports/test_evaluation.json")["metrics"]
    reproduced = read_json(common.ROOT / "reports/test_evaluation.json")["metrics"]
    for family in FAMILIES:
        for key in ["mae", "weighted_mae", "rmse"]:
            if abs(published[family][key] - reproduced[family][key]) > cfg["baseline_reproduction_tolerance"]:
                raise ValueError(f"Workspace does not reproduce v2 {family} {key}; baselines unusable")


def psu_bootstrap(frame: pd.DataFrame, predictions: dict, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    frame = frame.reset_index(drop=True)
    groups = [[g.index.to_numpy() for _, g in stratum.groupby("psu")] for _, stratum in frame.groupby("stratum")]
    y, weights = frame.target.to_numpy(), frame.weight.to_numpy()
    samples = {family: [] for family in predictions}
    for _ in range(replicates):
        index = np.concatenate([np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))]) for units in groups])
        for family, values in predictions.items():
            samples[family].append(float(np.average(np.abs(values[index] - y[index]), weights=weights[index])))
    samples = {k: np.array(v) for k, v in samples.items()}
    ci = lambda values: np.quantile(values, [.025, .975]).tolist()
    return {"weighted_mae_95": {f: ci(v) for f, v in samples.items()},
            "mlp_minus_mean_95": ci(samples["mlp"] - samples["mean"]),
            "mlp_minus_ridge_95": ci(samples["mlp"] - samples["ridge"]),
            "method": f"{replicates} stratified masked-PSU resamples within the external complete-case cohort; approximate"}


def subgroup_labels(frame: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    return {"sex": frame.sex.map({0.: "male", 1.: "female"}),
            "age": pd.cut(frame.age, cfg["age_group_edges"], right=False).astype(str),
            "target_range_research_bins": pd.cut(frame.target, cfg["target_range_edges"], right=False).astype(str),
            "medication_descriptive": frame.medication}


def evaluate_external() -> None:
    cfg = settings()
    out = report_dir()
    lock = out / "opened.json"
    if lock.exists():
        raise RuntimeError("External targets were already opened under protocol v3; do not rerun or retune")
    model_path = common.SOURCE_ROOT / "artifacts/model.onnx"
    if sha256(model_path) != cfg["frozen_model"]["onnx_sha256"]:
        raise ValueError("Published ONNX differs from the frozen model registered in protocol v3")
    check_reproduction(cfg)
    metadata = read_json(common.SOURCE_ROOT / "artifacts/metadata.json")
    tables = load_external_tables()  # schema/manifest checks only; targets are derived after the lock below
    write_json(lock, {"opened_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha256(common.SOURCE_ROOT / "protocols/m2-external-v3.md"),
        "config_sha256": sha256(common.SOURCE_ROOT / "configs/external_v3.json"),
        "model_sha256": cfg["frozen_model"]["onnx_sha256"],
        "rule": "Written before deriving external targets; reruns are refused"})

    v2_settings = read_json(common.SOURCE_ROOT / "configs/experiment.json")
    external, audit = prepare_frame(tables, v2_settings)
    v2 = pd.read_csv(common.ROOT / "data/processed/dataset.csv")
    overlap = int(np.isin(external.SEQN, v2.SEQN).sum())
    if overlap:
        raise ValueError("External cohort overlaps v2 participants")
    bounds = {f["key"]: (f["min"], f["max"]) for f in metadata["fields"] if f["type"] == "number"}
    in_domain = np.ones(len(external), bool)
    outside_by_feature = {}
    for key, (low, high) in bounds.items():
        outside = ~external[key].between(low, high)
        outside_by_feature[key] = int(outside.sum())
        in_domain &= ~outside.to_numpy()
    external["api_accepted"] = in_domain

    train = v2[v2.split == "train"]
    shift = {}
    for feature in FEATURES:
        a, b = train[feature].to_numpy(float), external[feature].to_numpy(float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        shift[feature] = {"train_mean": float(a.mean()), "external_mean": float(b.mean()),
                          "standardized_mean_difference": float((b.mean() - a.mean()) / pooled)}
    shift["target"] = {"train_mean": float(train.target.mean()), "external_mean": float(external.target.mean()),
                       "standardized_mean_difference": float((external.target.mean() - train.target.mean()) /
                           np.sqrt((train.target.var(ddof=1) + external.target.var(ddof=1)) / 2))}

    from .training import predictor  # ROOT is the reproduction workspace
    baselines = {family: predictor(family) for family in ["mean", "ridge", "gradient_boosting"]}

    def predict_all(frame: pd.DataFrame) -> dict:
        x = frame[FEATURES].to_numpy(np.float32)
        result = {family: np.asarray(fn(x), float) for family, fn in baselines.items()}
        result["mlp"] = onnx_predict(model_path, x)
        return result

    populations = {"api_accepted": external[external.api_accepted].reset_index(drop=True),
                   "all_eligible": external.reset_index(drop=True)}
    h1 = {}
    for name, frame in populations.items():
        predictions = predict_all(frame)
        y, w = frame.target.to_numpy(), frame.weight.to_numpy()
        entry = {"n": len(frame), "metrics": {f: regression_metrics(y, p, w) for f, p in predictions.items()}}
        entry["bootstrap"] = psu_bootstrap(frame, predictions, cfg["bootstrap_replicates"], cfg["bootstrap_seed"])
        entry["subgroups_mlp"] = {}
        for group, labels in subgroup_labels(frame, cfg).items():
            entry["subgroups_mlp"][group] = {}
            for label in sorted(labels.unique()):
                mask = (labels == label).to_numpy()
                if mask.sum() >= 2:
                    metrics = regression_metrics(y[mask], predictions["mlp"][mask], w[mask])
                    metrics["small_group_caution"] = bool(mask.sum() < cfg["conformal"]["minimum_group_size_for_interpretation"])
                    entry["subgroups_mlp"][group][label] = metrics
        h1[name] = entry
        if name == "api_accepted":
            primary_frame, primary_predictions = frame, predictions

    rule = cfg["transfer_rule"]
    threshold = rule["internal_weighted_mae"] * (1 + rule["max_relative_degradation"])
    primary = h1["api_accepted"]
    decisions = {
        "transfer_threshold_weighted_mae": threshold,
        "external_weighted_mae": primary["metrics"]["mlp"]["weighted_mae"],
        "transfer_acceptable": bool(primary["metrics"]["mlp"]["weighted_mae"] <= threshold
                                    and primary["bootstrap"]["mlp_minus_mean_95"][1] < 0),
        "mlp_better_than_mean": bool(primary["bootstrap"]["mlp_minus_mean_95"][1] < 0),
        "nn_superiority_over_ridge": bool(primary["bootstrap"]["mlp_minus_ridge_95"][1] < 0),
    }

    # H2: calibrate on the v2 internal test split with the published ONNX, evaluate on the external cohort.
    calibration = v2[v2.split == "test"].reset_index(drop=True)
    calibration_prediction = onnx_predict(model_path, calibration[FEATURES].to_numpy(np.float32))
    alpha = cfg["conformal"]["alpha"]
    methods = {"split_symmetric_absolute_residual": symmetric_offsets(calibration.target, calibration_prediction, alpha),
               "split_asymmetric_signed_residual": asymmetric_offsets(calibration.target, calibration_prediction, alpha)}
    h2 = {"alpha": alpha, "calibration_n": len(calibration), "methods": {}}
    for method, (low, high) in methods.items():
        h2["methods"][method] = {"lower_offset": low, "upper_offset": high,
            "calibration": coverage(calibration.target, calibration_prediction + low, calibration_prediction + high, calibration.weight),
            "external": {}}
        for name, frame in populations.items():
            prediction = primary_predictions["mlp"] if name == "api_accepted" else onnx_predict(model_path, frame[FEATURES].to_numpy(np.float32))
            y = frame.target.to_numpy()
            item = coverage(y, prediction + low, prediction + high, frame.weight)
            item["by_group"] = {}
            for group, labels in subgroup_labels(frame, cfg).items():
                item["by_group"][group] = {}
                for label in sorted(labels.unique()):
                    mask = (labels == label).to_numpy()
                    group_cov = coverage(y[mask], prediction[mask] + low, prediction[mask] + high, frame.weight.to_numpy()[mask])
                    group_cov["small_group_caution"] = bool(mask.sum() < cfg["conformal"]["minimum_group_size_for_interpretation"])
                    item["by_group"][group][label] = group_cov
            h2["methods"][method]["external"][name] = item
    primary_method = cfg["conformal"]["primary_method"]
    primary_coverage = h2["methods"][primary_method]["external"]["api_accepted"]["coverage"]
    decisions["interval_external_coverage"] = primary_coverage
    decisions["interval_display_allowed"] = bool(primary_coverage >= 0.85)

    audit.update({"external_overlap_with_v2": overlap, "outside_training_bounds_by_feature": outside_by_feature,
                  "api_accepted": int(in_domain.sum()), "all_eligible": len(external), "distribution_shift": shift})
    for key in ["complete_case_descriptive"]:
        audit.pop(key, None)
    write_json(out / "data_audit.json", audit)
    write_json(out / "external_evaluation.json", {"protocol": "protocols/m2-external-v3.md",
        "frozen_model": cfg["frozen_model"], "h1": h1, "decisions": decisions})
    write_json(out / "conformal.json", h2)
    make_plots(primary_frame, primary_predictions["mlp"], h2, cfg)
    print(f"External cohort: {len(external)} eligible, {int(in_domain.sum())} API-accepted; "
          f"MLP weighted MAE {decisions['external_weighted_mae']:.3f}; transfer_acceptable={decisions['transfer_acceptable']}; "
          f"interval coverage {primary_coverage:.3f}; display_allowed={decisions['interval_display_allowed']}", flush=True)


def make_plots(frame: pd.DataFrame, prediction: np.ndarray, h2: dict, cfg: dict) -> None:
    destination = report_dir() / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    y = frame.target.to_numpy()
    low, high = (h2["methods"][cfg["conformal"]["primary_method"]][k] for k in ["lower_offset", "upper_offset"])

    plt.figure(figsize=(6, 5))
    inside = (y >= prediction + low) & (y <= prediction + high)
    plt.scatter(y[inside], prediction[inside], s=10, alpha=.35, label="inside 90% interval")
    plt.scatter(y[~inside], prediction[~inside], s=10, alpha=.6, color="#c0392b", label="outside")
    lims = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Observed SBP, NHANES 2021–2023 (mmHg)")
    plt.ylabel("Estimated SBP (mmHg)")
    plt.legend()
    plt.title(f"External validation, n={len(y)}")
    plt.tight_layout(); plt.savefig(destination / "external-actual-vs-predicted.png", dpi=160); plt.close()

    groups = h2["methods"][cfg["conformal"]["primary_method"]]["external"]["api_accepted"]["by_group"]["target_range_research_bins"]
    labels = list(groups)
    values = [groups[k]["coverage"] for k in labels]
    errors = np.array([[groups[k]["coverage"] - groups[k]["coverage_95_wilson"][0],
                        groups[k]["coverage_95_wilson"][1] - groups[k]["coverage"]] for k in labels]).T
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, yerr=errors, capsize=4, color="#17447a")
    plt.axhline(1 - cfg["conformal"]["alpha"], color="black", linestyle="--", lw=1, label="nominal 90%")
    plt.ylabel("Coverage of 90% interval")
    plt.xlabel("Observed SBP range (mmHg, descriptive)")
    plt.ylim(0, 1.05); plt.legend()
    plt.tight_layout(); plt.savefig(destination / "external-coverage-by-sbp.png", dpi=160); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(prediction - y, bins=40, color="#17447a")
    plt.xlabel("Estimate minus observed SBP (mmHg)"); plt.ylabel("Participants")
    plt.tight_layout(); plt.savefig(destination / "external-residuals.png", dpi=160); plt.close()


def package_interval() -> None:
    """Write the protocol-approved interval calibration into the published bundle metadata (model bytes unchanged)."""
    cfg = settings()
    evaluation = read_json(report_dir() / "external_evaluation.json")
    decisions = evaluation["decisions"]
    if not decisions["interval_display_allowed"]:
        raise ValueError("Protocol v3 display rule not met; the interval must not be packaged")
    method = cfg["conformal"]["primary_method"]
    calibration = read_json(report_dir() / "conformal.json")
    chosen = calibration["methods"][method]
    external = chosen["external"]["api_accepted"]
    path = common.SOURCE_ROOT / "artifacts/metadata.json"
    metadata = read_json(path)
    if metadata["sha256"] != cfg["frozen_model"]["onnx_sha256"]:
        raise ValueError("Metadata does not describe the frozen model")
    internal = read_json(common.SOURCE_ROOT / "reports/test_evaluation.json")["metrics"]["mlp"]["weighted_mae"]
    metadata["predictionInterval"] = {
        "method": method, "level": 1 - calibration["alpha"],
        "lowerOffset": chosen["lower_offset"], "upperOffset": chosen["upper_offset"],
        "calibrationDataset": "NHANES 2017–March 2020 internal test split", "calibrationN": calibration["calibration_n"],
        "externalDataset": cfg["external_dataset"], "externalN": external["n"],
        "externalCoverage": external["coverage"], "externalCoverage95": external["coverage_95_wilson"],
        "protocol": "protocols/m2-external-v3.md",
        "note": "Marginal coverage averaged over a cohort; not a personal guarantee. Coverage is poor when the true SBP is >=150 mmHg."}
    metadata["externalValidation"] = {"dataset": cfg["external_dataset"], "n": evaluation["h1"]["api_accepted"]["n"],
        "weightedMae": decisions["external_weighted_mae"], "transferAcceptable": decisions["transfer_acceptable"],
        "nnSuperiorityOverRidge": decisions["nn_superiority_over_ridge"]}
    metadata["disclaimer"] = (
        f"Исследовательская оценка текущего САД, не диагноз. Взвешенная MAE: {internal:.1f} мм рт. ст. на внутренней "
        f"выборке NHANES 2017–2020 и {decisions['external_weighted_mae']:.1f} на независимой NHANES 2021–2023. "
        f"Интервал {round((1 - calibration['alpha']) * 100)}% накрыл фактическое значение у "
        f"{external['coverage'] * 100:.0f}% участников внешней проверки, но при фактическом САД ≥150 мм рт. ст. "
        "оценка систематически занижена и интервал часто не содержит истинное значение.")
    write_json(path, metadata)
    print(f"Packaged {method}: [{chosen['lower_offset']:+.2f}, {chosen['upper_offset']:+.2f}] mmHg", flush=True)
