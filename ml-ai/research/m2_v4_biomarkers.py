"""Compare incremental HbA1c and creatinine features on a shared M2 cohort."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyreadstat
import torch
from tabm import TabM
from threadpoolctl import threadpool_limits

from m2_v4_pilot import ROOT, development_data
from m2_v4_seeds import METRICS, SEEDS, train_predict
from medpredict_ml.common import FEATURES, sha256, write_json
from medpredict_ml.data import BASE, valid_transport, validate_table
from medpredict_ml.metrics import regression_metrics
from medpredict_ml.model import fit_preprocessor

PROTOCOL = ROOT / "protocols/m2-v4-biomarkers.md"
BIOMARKERS = {"P_GHB": "LBXGH", "P_BIOPRO": "LBDSCRSI"}
FEATURE_SETS = {"tabm_5f": FEATURES, "tabm_hba1c": [*FEATURES, "LBXGH"],
                "tabm_creatinine": [*FEATURES, "LBDSCRSI"]}


def compare(workspace: Path) -> dict:
    frame, hashes = development_data(workspace)
    source_files = {}
    for table, variable in BIOMARKERS.items():
        path = workspace / "data/raw" / f"{table}.xpt"
        if not valid_transport(path):
            raise ValueError(f"{table} is not a SAS transport file")
        columns = ["SEQN", variable]
        lab, _ = pyreadstat.read_xport(path, usecols=columns)
        validate_table(lab, columns, table)
        frame = frame.merge(lab, on="SEQN", how="left", validate="one_to_one", sort=False)
        source_files[table] = {"url": BASE + path.name, "sha256": sha256(path), "column": variable}
    values = frame[list(BIOMARKERS.values())]
    matched = frame.loc[(np.isfinite(values) & (values > 0)).all(axis=1)]
    train = matched[matched.split == "train"]
    valid = matched[matched.split == "validation"]
    y, yv, survey = train.target.to_numpy(), valid.target.to_numpy(), valid.weight.to_numpy()
    high = yv >= 150
    weights = train.weight.to_numpy() * np.where(y >= 150, 4, 1)
    weights /= weights.mean()

    def mae(prediction):
        return {
            "overall_mae": regression_metrics(yv, prediction, survey)["weighted_mae"],
            "sbp_ge_150_mae": regression_metrics(yv[high], prediction[high], survey[high])["weighted_mae"],
        }

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "artifacts/model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    base_input = valid[FEATURES].to_numpy(dtype=np.float32)
    baseline = mae(session.run(None, {session.get_inputs()[0].name: base_input})[0].reshape(-1))
    max_overall, max_tail = baseline["overall_mae"] + 0.5, baseline["sbp_ge_150_mae"] * 0.75

    def score(scores, reference):
        pass_overall = scores["overall_mae"] <= max_overall
        pass_tail = scores["sbp_ge_150_mae"] <= max_tail
        return {
            **scores, "overall_delta_vs_v2": scores["overall_mae"] - baseline["overall_mae"],
            "tail_reduction_percent_vs_v2": 100 * (1 - scores["sbp_ge_150_mae"] / baseline["sbp_ge_150_mae"]),
            "delta_vs_tabm_5f": {key: scores[key] - reference[key] for key in METRICS},
            "passes_overall": pass_overall, "passes_tail": pass_tail, "passes_both": pass_overall and pass_tail,
        }

    models = {}
    for name, features in FEATURE_SETS.items():
        x, xv = train[features].to_numpy(), valid[features].to_numpy()
        fitted = fit_preprocessor(x, y, weights)
        x_mean, x_scale = np.asarray(fitted["x_mean"]), np.asarray(fitted["x_scale"])
        center, scale = fitted["y_mean"], fitted["y_scale"]
        train_x, train_y, train_w, valid_x = (
            torch.as_tensor(values, dtype=torch.float32)
            for values in ((x - x_mean) / x_scale, (y - center) / scale, weights, (xv - x_mean) / x_scale)
        )

        def make_model():
            return TabM.make(n_num_features=len(features), d_out=1, k=8, n_blocks=2, d_block=64, dropout=0.1)

        runs = []
        for index, seed in enumerate(SEEDS):
            prediction = train_predict(make_model, seed, train_x, train_y, train_w, valid_x, scale)
            scores = mae(prediction * scale + center)
            reference = models["tabm_5f"]["runs"][index] if models else scores
            runs.append({"seed": seed, **score(scores, reference)})
            print(f"{name} {seed}: {scores['overall_mae']:.3f} / {scores['sbp_ge_150_mae']:.3f}", flush=True)
        average = {key: float(np.mean([run[key] for run in runs])) for key in METRICS}
        models[name] = {
            "features": features, "runs": runs,
            "mean": score(average, models["tabm_5f"]["mean"] if models else average),
            "sample_sd": {key: float(np.std([run[key] for run in runs], ddof=1)) for key in METRICS},
            "pass_counts": {key: sum(run[key] for run in runs) for key in ("passes_overall", "passes_tail", "passes_both")},
        }
    return {
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "training_script_sha256": sha256(Path(__file__).with_name("m2_v4_seeds.py")),
        "data_loader_sha256": sha256(Path(__file__).with_name("m2_v4_pilot.py")),
        "input_sha256": hashes, "laboratory_sources": source_files, "seeds": SEEDS, "test_used": False,
        "packages": {name: version(name) for name in ("tabm", "rtdl-num-embeddings", "numpy", "pandas",
                                                     "pyreadstat", "scikit-learn", "torch", "onnxruntime")},
        "original_counts": frame.split.value_counts().to_dict(),
        "train_n": len(train), "validation_n": len(valid), "validation_sbp_ge_150_n": int(high.sum()),
        "v2_onnx": baseline, "thresholds": {"max_overall_mae": max_overall, "max_tail_mae": max_tail},
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    with threadpool_limits(limits=1):
        report = compare(args.workspace)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
