"""Compare fixed regressors on v2 development data. Write aggregate metrics only."""
from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyreadstat
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tabm import TabM
from threadpoolctl import threadpool_limits

from medpredict_ml.common import FEATURES, read_json, sha256, write_json
from medpredict_ml.data import validate_table
from medpredict_ml.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/m2-v4-pilot.md"
SEED = 2026


def development_data(workspace: Path) -> tuple[pd.DataFrame, dict]:
    paths = {
        "dataset": workspace / "data/processed/dataset.csv",
        "body_measures": workspace / "data/raw/P_BMX.xpt",
        "v2_onnx": ROOT / "artifacts/model.onnx",
    }
    expected = {
        "dataset": read_json(ROOT / "reports/protocol.json")["dataset_sha256"],
        "body_measures": next(item["sha256"] for item in read_json(ROOT / "reports/data_manifest.json")["files"]
                              if item["name"] == "P_BMX"),
        "v2_onnx": read_json(ROOT / "artifacts/metadata.json")["sha256"],
    }
    if {name: sha256(path) for name, path in paths.items()} != expected:
        raise ValueError("Inputs differ from the archived v2 records")
    original = pd.read_csv(paths["dataset"], usecols=["SEQN", *FEATURES, "target", "weight", "split"])
    development = original.loc[original.split.isin(["train", "validation"])]
    columns = ["SEQN", "BMXWAIST"]
    waist, _ = pyreadstat.read_xport(paths["body_measures"], usecols=columns)
    validate_table(waist, columns, "P_BMX waist")
    return development.merge(waist, on="SEQN", how="left", validate="one_to_one"), expected


def tabm_predict(x, y, weight, xv):
    torch.manual_seed(SEED)
    scaler = StandardScaler().fit(x, sample_weight=weight)
    center = np.average(y, weights=weight)
    scale = np.sqrt(np.average((y - center) ** 2, weights=weight))
    train_x, train_y, train_w, valid_x = (torch.as_tensor(values, dtype=torch.float32) for values in
        (scaler.transform(x), (y - center) / scale, weight, scaler.transform(xv)))
    model = TabM.make(n_num_features=x.shape[1], d_out=1, k=8, n_blocks=2, d_block=64, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    for _ in range(80):
        for batch in torch.randperm(len(train_x)).split(128):
            prediction = model(train_x[batch]).squeeze(-1)
            target = train_y[batch, None].expand_as(prediction)
            loss = F.smooth_l1_loss(prediction, target, beta=8 / scale, reduction="none")
            optimizer.zero_grad()
            (loss.mean(dim=1) * train_w[batch]).mean().backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return model(valid_x).mean(dim=1).flatten().numpy() * scale + center


def evaluate(frame: pd.DataFrame, session: ort.InferenceSession, waist_comparison: bool) -> dict:
    train = frame[frame.split == "train"]
    valid = frame[frame.split == "validation"]
    y, weight = train.target.to_numpy(), train.weight.to_numpy()
    yv, survey = valid.target.to_numpy(), valid.weight.to_numpy()
    groups = {"overall": np.ones(len(valid), dtype=bool), "sbp_ge_150": yv >= 150, "sbp_lt_110": yv < 110}
    x, xv = train[FEATURES].to_numpy(), valid[FEATURES].to_numpy()
    baseline = session.run(None, {session.get_inputs()[0].name: xv.astype(np.float32)})[0].reshape(-1)
    results = {}

    def record(name, prediction, seconds=0.0):
        row = {"training_seconds": round(seconds, 3)}
        for group, mask in groups.items():
            metrics = regression_metrics(yv[mask], prediction[mask], survey[mask])
            row[group] = {key: metrics[key] for key in ("n", "weighted_mae", "weighted_mean_signed_error")}
        row["tail_mae_reduction_percent_vs_v2"] = 100 * (
            1 - row["sbp_ge_150"]["weighted_mae"] / results.get("v2_onnx", row)["sbp_ge_150"]["weighted_mae"])
        row["overall_mae_delta_vs_v2"] = row["overall"]["weighted_mae"] - results.get(
            "v2_onnx", row)["overall"]["weighted_mae"]
        results[name] = row
        print(f"{name}: MAE {row['overall']['weighted_mae']:.2f}, ≥150 {row['sbp_ge_150']['weighted_mae']:.2f}", flush=True)

    record("v2_onnx", baseline)
    if not waist_comparison:
        archived = read_json(ROOT / "artifacts/selection.json")["validation"]["mlp"]["weighted_mae"]
        if abs(results["v2_onnx"]["overall"]["weighted_mae"] - archived) > 0.001:
            raise ValueError("Published ONNX did not reproduce the archived validation MAE")
        record("v2_plus_5_control", baseline + 5)
        record("weighted_train_mean", np.full(len(valid), np.average(y, weights=weight)))

    feature_sets = [FEATURES, [*FEATURES, "BMXWAIST"]] if waist_comparison else [FEATURES]
    for features in feature_sets:
        suffix = f"{len(features)}f"
        x, xv = train[features].to_numpy(), valid[features].to_numpy()
        started = time.perf_counter()
        scaler = StandardScaler().fit(x, sample_weight=weight)
        ridge = Ridge(alpha=10).fit(scaler.transform(x), y, sample_weight=weight / weight.mean())
        record(f"ridge_{suffix}", ridge.predict(scaler.transform(xv)), time.perf_counter() - started)
        for multiplier in ([1] if waist_comparison else [1, 2, 4]):
            weights = weight * np.where(y >= 150, multiplier, 1)
            started = time.perf_counter()
            catboost = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.035, l2_leaf_reg=5,
                                         loss_function="MAE", random_seed=SEED, thread_count=1,
                                         allow_writing_files=False, verbose=False)
            catboost.fit(x, y, sample_weight=weights / weights.mean())
            record(f"catboost_{suffix}_{multiplier}x", catboost.predict(xv), time.perf_counter() - started)
        if not waist_comparison:
            for multiplier in (1, 4):
                weights = weight * np.where(y >= 150, multiplier, 1)
                started = time.perf_counter()
                prediction = tabm_predict(x, y, weights / weights.mean(), xv)
                record(f"tabm_{suffix}_{multiplier}x", prediction, time.perf_counter() - started)
    v2_tail = results["v2_onnx"]["sbp_ge_150"]["weighted_mae"]
    for name, row in results.items():
        if name.startswith(("ridge_", "catboost_", "tabm_")):
            row["passes_pilot_screen"] = (
                row["sbp_ge_150"]["weighted_mae"] <= 0.75 * v2_tail and row["overall_mae_delta_vs_v2"] <= 0.5)
    return {"train_n": len(train), "validation_n": len(valid), "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "artifacts/model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    frame, hashes = development_data(args.workspace)
    matched = frame.loc[np.isfinite(frame.BMXWAIST) & (frame.BMXWAIST > 0)]
    with threadpool_limits(limits=1):
        report = {"protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                  "input_sha256": hashes, "seed": SEED, "test_used": False,
                  "packages": {name: version(name) for name in ("catboost", "tabm", "rtdl-num-embeddings", "numpy",
                               "pandas", "pyreadstat", "scikit-learn", "torch", "onnxruntime")},
                  "five_features": evaluate(frame, session, False),
                  "matched_waist": evaluate(matched, session, True)}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
