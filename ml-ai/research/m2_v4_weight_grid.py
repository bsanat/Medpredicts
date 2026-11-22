"""Select a TabM high-SBP weight on v2 train folds, then check old validation."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabm import TabM
from threadpoolctl import threadpool_limits

from m2_v4_pilot import ROOT, development_data
from m2_v4_seeds import SEEDS, train_predict
from medpredict_ml.common import FEATURES, read_json, sha256, write_json
from medpredict_ml.metrics import regression_metrics
from medpredict_ml.model import RawInputModel, fit_preprocessor, train_network

PROTOCOL = ROOT / "protocols/m2-v4-weight-grid.md"
MULTIPLIERS = (2, 4, 6, 8)
CV_SEEDS = SEEDS[:3]


def arrays(frame):
    return frame[FEATURES].to_numpy(), frame.target.to_numpy(), frame.weight.to_numpy()


def summarize(y, w, predictions):
    high, low = y >= 150, y < 110
    result = []
    for prediction in predictions:
        overall = regression_metrics(y, prediction, w)
        tail = regression_metrics(y[high], prediction[high], w[high])
        result.append({
            "overall_mae": overall["weighted_mae"], "high_mae": tail["weighted_mae"],
            "high_bias": tail["weighted_mean_signed_error"],
            "low_bias": float(np.average(prediction[low] - y[low], weights=w[low])),
        })
    return result


def compare(row, ref):
    delta = row["overall_mae"] - ref["overall_mae"]
    reduction = 100 * (1 - row["high_mae"] / ref["high_mae"])
    return {**row, "overall_delta": delta, "high_reduction_percent": reduction,
            "passes_overall": delta <= 0.5, "passes_both": delta <= 0.5 and reduction >= 25}


def average(rows, ref):
    return compare({key: float(np.mean([row[key] for row in rows])) for key in ref}, ref)


def tabm_predictions(x, y, w, xv, multiplier, seeds):
    weights = w * np.where(y >= 150, multiplier, 1)
    weights /= weights.mean()
    prep = fit_preprocessor(x, y, weights)
    x_mean, x_scale = np.asarray(prep["x_mean"]), np.asarray(prep["x_scale"])
    center, scale = prep["y_mean"], prep["y_scale"]
    train_x, train_y, train_w, query_x = (
        torch.as_tensor(values, dtype=torch.float32)
        for values in ((x - x_mean) / x_scale, (y - center) / scale, weights, (xv - x_mean) / x_scale)
    )

    def make_model():
        return TabM.make(n_num_features=len(FEATURES), d_out=1, k=8, n_blocks=2, d_block=64, dropout=0.1)

    return np.asarray([
        train_predict(make_model, seed, train_x, train_y, train_w, query_x, scale) * scale + center
        for seed in seeds
    ])


def select(train):
    x, y, w = arrays(train)
    settings = read_json(ROOT / "configs/experiment.json")
    archived = read_json(ROOT / "artifacts/selection.json")
    strata = np.digitize(y, [110, 150])
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    coverage = np.zeros(len(y), dtype=int)
    baseline = np.full(len(y), np.nan)
    candidates = {multiplier: np.full((len(CV_SEEDS), len(y)), np.nan) for multiplier in MULTIPLIERS}
    folds = []
    for fold_id, (fit, score) in enumerate(outer.split(x, strata), start=1):
        core, stop = train_test_split(fit, test_size=0.15, random_state=2026 + fold_id, stratify=strata[fit])
        assert not set(core).intersection(score) and not set(stop).intersection(score)
        coverage[score] += 1
        prep = fit_preprocessor(x[core], y[core], w[core])
        net, history, _ = train_network(
            x[core], y[core], w[core], x[stop], y[stop], w[stop], prep,
            archived["mlp_selected_config"], settings, archived["mlp_fixed_representative_seed"])
        with torch.inference_mode():
            baseline[score] = RawInputModel(net, prep)(torch.as_tensor(x[score], dtype=torch.float32)).numpy().ravel()
        folds.append({"fold": fold_id, "fit_n": len(core), "early_stop_n": len(stop), "score_n": len(score),
                      "score_high_n": int((y[score] >= 150).sum()),
                      "v2_best_epoch": min(history, key=lambda entry: entry["validation_weighted_mae"])["epoch"]})
        for multiplier in MULTIPLIERS:
            candidates[multiplier][:, score] = tabm_predictions(x[core], y[core], w[core], x[score],
                                                                multiplier, CV_SEEDS)
            print(f"CV fold {fold_id}: {multiplier}× complete", flush=True)
    assert (coverage == 1).all()
    ref, = summarize(y, w, [baseline])
    results = {}
    for multiplier, prediction in candidates.items():
        rows = [dict(seed=seed, **compare(row, ref)) for seed, row in zip(CV_SEEDS, summarize(y, w, prediction))]
        results[multiplier] = {"seeds": rows, "mean": average(rows, ref)}
    eligible = [multiplier for multiplier in MULTIPLIERS if results[multiplier]["mean"]["passes_overall"]]
    chosen = min(eligible, key=lambda multiplier: (results[multiplier]["mean"]["high_mae"], multiplier), default=None)
    return {"n": len(y), "high_n": int((y >= 150).sum()), "folds": folds, "v2_refit": ref,
            "candidates": results, "selected_multiplier": chosen}


def check_validation(train, valid, chosen):
    x, y, w = arrays(train)
    xv, yv, wv = arrays(valid)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "artifacts/model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    baseline = session.run(None, {session.get_inputs()[0].name: xv.astype(np.float32)})[0].ravel()
    ref, = summarize(yv, wv, [baseline])
    archived = read_json(ROOT / "artifacts/selection.json")["validation"]["mlp"]["weighted_mae"]
    assert abs(ref["overall_mae"] - archived) < 0.001
    candidates = {}
    for multiplier in dict.fromkeys((chosen, 4)):
        predictions = tabm_predictions(x, y, w, xv, multiplier, SEEDS)
        rows = [dict(seed=seed, **compare(row, ref)) for seed, row in zip(SEEDS, summarize(yv, wv, predictions))]
        candidates[multiplier] = {"seeds": rows, "mean": average(rows, ref),
                                  "n_passing_both": sum(row["passes_both"] for row in rows)}
        print(f"Validation: {multiplier}× complete", flush=True)
    return {"n": len(yv), "high_n": int((yv >= 150).sum()), "v2_onnx": ref, "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    frame, hashes = development_data(args.workspace)
    train = frame[frame.split == "train"]
    with threadpool_limits(limits=1):
        cv = select(train)
        report = {"protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
                  "training_script_sha256": sha256(Path(__file__).with_name("m2_v4_seeds.py")),
                  "input_sha256": hashes, "test_used": False, "cross_validation": cv,
                  "packages": {name: version(name) for name in ("tabm", "numpy", "scikit-learn", "torch", "onnxruntime")}}
        write_json(args.output, report)
        chosen = cv["selected_multiplier"]
        print(f"Selected from CV: {chosen}", flush=True)
        if chosen is not None:
            report["old_validation"] = check_validation(train, frame[frame.split == "validation"], chosen)
            write_json(args.output, report)


if __name__ == "__main__":
    main()
