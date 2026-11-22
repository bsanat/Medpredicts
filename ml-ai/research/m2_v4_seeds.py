"""Compare plain MLP and TabM at paired seeds with identical high-SBP weighting."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tabm import TabM
from threadpoolctl import threadpool_limits
from torch.nn.functional import smooth_l1_loss

from medpredict_ml.common import FEATURES, read_json, sha256, write_json
from medpredict_ml.metrics import regression_metrics
from medpredict_ml.model import MLP, fit_preprocessor, set_seed
from m2_v4_pilot import ROOT, development_data

SEEDS = (2026, 2027, 2028, 2029, 2030)
PROTOCOL = ROOT / "protocols/m2-v4-seeds.md"
METRICS = ("overall_mae", "sbp_ge_150_mae")


def train_predict(make_model, seed, x, y, weights, validation, scale):
    set_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    for _ in range(80):
        for batch in torch.randperm(len(x), generator=shuffle).split(128):
            predictions = model(x[batch]).reshape(len(batch), -1)
            targets = y[batch, None].expand_as(predictions)
            losses = smooth_l1_loss(predictions, targets, beta=8 / scale, reduction="none")
            optimizer.zero_grad()
            (losses.mean(dim=1) * weights[batch]).mean().backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return model(validation).reshape(len(validation), -1).mean(dim=1).numpy()


def pilot(workspace):
    frame, hashes = development_data(workspace)
    train = frame[frame.split == "train"]
    valid = frame[frame.split == "validation"]
    x, y, xv = train[FEATURES].to_numpy(), train.target.to_numpy(), valid[FEATURES].to_numpy()
    yv, survey = valid.target.to_numpy(), valid.weight.to_numpy()
    high = yv >= 150
    weights = train.weight.to_numpy() * np.where(y >= 150, 4, 1)
    weights /= weights.mean()
    fitted = fit_preprocessor(x, y, weights)
    x_mean, x_scale = np.asarray(fitted["x_mean"]), np.asarray(fitted["x_scale"])
    center, scale = fitted["y_mean"], fitted["y_scale"]
    train_x, train_y, train_w, valid_x = (
        torch.as_tensor(values, dtype=torch.float32)
        for values in ((x - x_mean) / x_scale, (y - center) / scale, weights, (xv - x_mean) / x_scale)
    )

    def mae(prediction):
        return dict(zip(METRICS, (
            regression_metrics(yv, prediction, survey)["weighted_mae"],
            regression_metrics(yv[high], prediction[high], survey[high])["weighted_mae"],
        )))

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "artifacts/model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    baseline = mae(session.run(None, {session.get_inputs()[0].name: xv.astype(np.float32)})[0].reshape(-1))
    archived = read_json(ROOT / "artifacts/selection.json")["validation"]["mlp"]["weighted_mae"]
    if abs(baseline["overall_mae"] - archived) > 0.001:
        raise ValueError("Published ONNX did not reproduce the archived validation MAE")

    def compare(scores):
        reduction = 100 * (1 - scores["sbp_ge_150_mae"] / baseline["sbp_ge_150_mae"])
        delta = scores["overall_mae"] - baseline["overall_mae"]
        return {**scores, "sbp_ge_150_reduction_percent": reduction, "overall_mae_delta": delta,
                "passes_tail": reduction >= 25, "passes_overall": delta <= 0.5,
                "passes_both": reduction >= 25 and delta <= 0.5}

    models = {}
    for name, make_model in {
        "mlp": lambda: MLP([64, 64], dropout=0.1),
        "tabm": lambda: TabM.make(n_num_features=len(FEATURES), d_out=1, k=8, n_blocks=2,
                                  d_block=64, dropout=0.1),
    }.items():
        runs = []
        for seed in SEEDS:
            prediction = train_predict(make_model, seed, train_x, train_y, train_w, valid_x, scale)
            row = compare(mae(prediction * scale + center))
            runs.append({"seed": seed, **row})
            print(f"{name} {seed}: {row['overall_mae']:.3f} / {row['sbp_ge_150_mae']:.3f}", flush=True)
        models[name] = {
            "runs": runs,
            "mean": compare({key: float(np.mean([run[key] for run in runs])) for key in METRICS}),
            "sample_sd": {key: float(np.std([run[key] for run in runs], ddof=1)) for key in METRICS},
            "n_seeds_passing_both": sum(run["passes_both"] for run in runs),
        }
    return {
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "data_loader_sha256": sha256(Path(__file__).with_name("m2_v4_pilot.py")),
        "input_sha256": hashes, "test_used": False, "seeds": SEEDS,
        "packages": {name: version(name) for name in ("catboost", "tabm", "rtdl-num-embeddings", "numpy",
                      "pandas", "pyreadstat", "scikit-learn", "torch", "onnxruntime")},
        "train_n": len(train), "validation_n": len(valid), "validation_sbp_ge_150_n": int(high.sum()),
        "v2_onnx": baseline,
        "thresholds": {"max_tail_mae": 0.75 * baseline["sbp_ge_150_mae"], "max_overall_mae": baseline["overall_mae"] + 0.5},
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    with threadpool_limits(limits=1):
        report = pilot(args.workspace)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
