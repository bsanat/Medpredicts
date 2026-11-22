"""Evaluate frozen M2-v5 outputs. Dry-run and sealed mode share one evaluator."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from m2_v5_audit import NEW, OLD, SEALED, TABLES, load_development, raw_path
from m2_v5_cv import FEATURES, medication_status, old_eligibility, positive
from m2_v5_grid import MANIFEST, PROTOCOL, REPORT, ROOT, SPLITS, bundle_predict, subgroup_masks, worker_init
from m2_v5_grid_outputs import cqr_predict, final_sets, interval_metrics, probability_metrics, probability_predict
from m2_v5_pre_e import load_models, probe_predictions
from medpredict_ml.blood_pressure import bp_columns, measurement_target, read_xpt
from medpredict_ml.common import read_json, sha256, write_json
from medpredict_ml.external import onnx_predict
from medpredict_ml.metrics import regression_metrics
from medpredict_ml.model import MLP, RawInputModel

B_REPORT = ROOT / "reports/m2_v5_audit.json"
LOG = ROOT / "reports/m2_v5_e_opened.json"
DRY_REPORT = ROOT / "reports/m2_v5_e_dry_run.json"
V2_ONNX = ROOT / "artifacts/model.onnx"
V2_METADATA = ROOT / "artifacts/metadata.json"
REPLICATES = 1000
BOOTSTRAP_SEED = 3031
POINT_ORDER = ("main", "safe", "main_base_5f", "safe_base_5f", "ridge_reference", "v2")
PHYSICAL_BINS = (0, 110, 130, 150, np.inf)
PROB_REGISTERED, PROB_UI, PROB_AGE = "hist_gradient_boosting", "logistic_same_inputs", "logistic_age_only"


def load_sealed(audit):
    files = {entry["table"]: entry for entry in audit["source_manifest"] if entry["cycle"] == SEALED}
    prescriptions = [
        ["RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"], ["BMXBMI"], ["LBDTCSI"],
        ["LBXGH"], bp_columns(auscultatory=True), ["BPQ020", "BPQ040A", "BPQ050A"],
        ["SMQ020", "SMQ040"], ["PAD680"], ["LBDGLUSI", "WTSAF2YR"],
    ]
    names = TABLES[SEALED]
    if set(files) != set(names):
        raise ValueError("Sealed source list differs from stage B")
    paths = [raw_path(SEALED, name) for name in names]
    if any(sha256(path) != files[name]["sha256"] for path, name in zip(paths, names, strict=True)):
        raise ValueError("Sealed source hashes differ from stage B")
    tables = [read_xpt(path, ["SEQN", *columns]) for path, columns in zip(paths, prescriptions, strict=True)]
    frame = tables[0]
    for cycle in (OLD, NEW):
        table = TABLES[cycle][0]
        source = next(entry for entry in audit["source_manifest"] if entry["cycle"] == cycle and entry["table"] == table)
        path = raw_path(cycle, table)
        if sha256(path) != source["sha256"] or frame.SEQN.isin(read_xpt(path, ["SEQN"]).SEQN).any():
            raise ValueError(f"Cannot verify disjoint demographics with {cycle}")
    for table in tables[1:]:
        if not table.SEQN.isin(frame.SEQN).all():
            raise ValueError("Participant missing from sealed demographics")
        frame = frame.merge(table, on="SEQN", how="left", validate="one_to_one")
    return frame


def prepare(raw, cycle, metadata):
    frame = raw.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "sex", "BMXBMI": "bmi", "LBDTCSI": "totalCholesterol",
                                "LBDGLUSI": "glucose", "LBXGH": "hba1c", "WTSAF2YR": "fasting_weight",
                                "WTMEC2YR" if cycle == SEALED else "WTPH2YR": "wide_weight",
                                "SDMVSTRA": "stratum", "SDMVPSU": "psu"}).copy()
    frame["target"], _ = measurement_target(frame, auscultatory=cycle == SEALED)
    frame["cycle"] = cycle
    # m2_v5_cv encodes the pre-pandemic route for OLD, the same route used by BPQ_I.
    frame["medication"] = medication_status(frame, OLD if cycle == SEALED else NEW)
    frame["bpq020"] = frame.BPQ020.map({1: "yes", 2: "no"}).fillna("unknown")
    smoke = frame.SMQ040.where(frame.SMQ020.eq(1)).map({1: "current", 2: "current", 3: "former"})
    frame["smoking"] = smoke.mask(frame.SMQ020.eq(2), "never").fillna("unknown")
    frame["pad680"] = frame.PAD680.where(frame.PAD680.between(0, 1439))
    flow = {}

    def retain(rows, mask, name):
        selected = rows.loc[mask]
        flow[name] = {"excluded": len(rows) - len(selected), "remaining": len(selected)}
        return selected

    n_source = len(frame)
    frame = retain(frame, frame.age.between(18, 79) & frame.age.mod(1).eq(0), "age")
    frame = retain(frame, frame.sex.isin([1, 2]), "sex")
    frame = retain(frame, frame.target.notna(), "two_valid_readings")
    frame = retain(frame, np.isfinite(frame[["stratum", "psu"]]).all(axis=1), "design")
    for name in ("wide_weight", "bmi", "totalCholesterol", "hba1c"):
        frame = retain(frame, positive(frame[name]), name)
    frame["sex"] = frame.sex.map({1: 0., 2: 1.})
    bounds = {field["key"]: (field["min"], field["max"]) for field in metadata["fields"] if field["type"] == "number"}
    frame["paired"] = positive(frame.fasting_weight) & positive(frame.glucose) & old_eligibility(frame, bounds)
    paired_weight = positive(frame.fasting_weight)
    fasting_glucose = positive(frame.glucose)
    paired_flow = {"missing_or_nonpositive_fasting_weight": int((~paired_weight).sum()),
                   "missing_or_nonpositive_fasting_glucose": int((paired_weight & ~fasting_glucose).sum()),
                   "outside_v2_bounds": int((paired_weight & fasting_glucose & ~frame.paired).sum())}
    return frame.set_index("SEQN", verify_integrity=True).sort_index(), {
        "source_rows_in_scope": n_source, "expanded_attrition": flow, "paired_exclusions": paired_flow}


def load_verified():
    frozen, audit = read_json(MANIFEST), read_json(B_REPORT)
    pre = frozen["pre_e"]
    files = [(PROTOCOL, pre["protocol_sha256"]), (SPLITS, frozen["split_manifest_sha256"]),
             (REPORT, frozen["grid_report_sha256"]), (V2_ONNX, frozen["baseline_onnx_sha256"]),
             (V2_METADATA, frozen["baseline_metadata_sha256"]), (B_REPORT, frozen["sealed_cycle_manifest_reference"]["stage_b_sha256"]),
             (ROOT / "research/m2_v5_pre_e.py", pre["freeze_script_sha256"]),
             (ROOT / "src/medpredict_ml/model.py", frozen["mlp_source_sha256"])]
    files.extend((ROOT / "research" / name, expected) for name, expected in frozen["scripts"].items())
    for path, expected in files:
        if sha256(path) != expected:
            raise ValueError(f"Frozen file hash mismatch: {path.name}")
    models = load_models(frozen)
    replay = probe_predictions(models)
    if replay != pre["probe_replay"]:
        raise ValueError("Frozen synthetic predictions differ from the manifest")
    return frozen, audit, models, replay


def python_parity(checkpoint, rows, onnx):
    expected = read_json(ROOT / "artifacts/selection.json")["artifact_sha256"]["mlp.pt"]
    if sha256(checkpoint) != expected:
        raise ValueError("PyTorch reference checksum differs from archived v2")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network = MLP(state["candidate"]["hidden"], state["candidate"]["dropout"])
    network.load_state_dict(state["state_dict"])
    reference = RawInputModel(network, state["preprocessing"]).eval()
    with torch.inference_mode():
        scores = reference(torch.as_tensor(rows[FEATURES].to_numpy(), dtype=torch.float32)).numpy().ravel()
    np.testing.assert_allclose(scores, onnx, atol=.01, rtol=0)
    return {"n": len(rows), "checkpoint_sha256": expected,
            "max_absolute_difference": float(np.max(np.abs(scores - onnx))), "absolute_tolerance": .01}


def predictions(frame, models):
    pairs = frame.loc[frame.paired]
    if pairs.empty:
        raise ValueError("No participants can be evaluated using v2")
    values = pd.DataFrame(index=frame.index)
    values["v2"] = np.nan
    values.loc[pairs.index, "v2"] = onnx_predict(V2_ONNX, pairs[FEATURES].to_numpy())
    artifacts = {"main": "point", "safe": "safe_point", "main_base_5f": "point_base_5f",
                 "safe_base_5f": "safe_base_5f", "ridge_reference": "ridge_reference"}
    for label, filename in artifacts.items():
        bundle = models[filename]
        complete = frame[bundle["spec"]["fields"]].notna().all(axis=1)
        values.loc[complete, label] = bundle_predict(bundle, frame.loc[complete])
    scores = probability_predict(models["probability"], frame)
    decisions = {name: estimate >= models["probability"]["thresholds"][name] for name, estimate in scores.items()}
    metadata = read_json(V2_METADATA)["predictionInterval"]
    lower, upper = "lowerOffset", "upperOffset"
    intervals = {"v2": (values.v2.to_numpy() + metadata[lower], values.v2.to_numpy() + metadata[upper])}
    for label in ("main", "safe"):
        radius = models["cqr"]["radius"] if label == "main" else models["safe_cqr_radius"]
        intervals[label] = cqr_predict({**models["cqr"], "radius": radius}, frame, values[label].to_numpy())
    return values, scores, decisions, intervals


def point_groups(frame):
    bins = pd.cut(frame.target, PHYSICAL_BINS, right=False, labels=["lt110", "110_to_lt130", "130_to_lt150", "ge150"])
    return {
        "overall": pd.Series(True, index=frame.index),
        **{f"sbp_{key}": bins.eq(key) for key in bins.cat.categories},
        **{f"age_{lo}_to_{hi-1}": frame.age.ge(lo) & frame.age.lt(hi) for lo, hi in ((18, 40), (40, 60), (60, 80))},
        **{f"sex_{key}": frame.sex.eq(code) for key, code in (("male", 0), ("female", 1))},
        **{f"{field}_{label}": frame[field].eq(label) for field in ("medication", "bpq020")
           for label in ("yes", "no", "unknown")},
        **subgroup_masks(frame)}


def point_stats(y, pred, w):
    n = len(y)
    if not n:
        return {"n": 0, "weighted_mae": None, "weighted_mean_signed_error": None}
    if n == 1:
        error = float(pred[0] - y[0])
        return {"n": 1, "mae": abs(error), "weighted_mae": abs(error),
                "rmse": abs(error), "weighted_rmse": abs(error),
                "mean_signed_error": error, "weighted_mean_signed_error": error}
    return regression_metrics(y, pred, w)


def point_summary(frame, values, names, column):
    result = {}
    for group, mask in point_groups(frame).items():
        sel = frame.loc[mask]
        y, w = sel.target.to_numpy(), sel[column].to_numpy()
        result[group] = {"n": len(sel), "descriptive_only": len(sel) < 30,
                         "models": {name: point_stats(y, values.loc[sel.index, name].to_numpy(), w) for name in names}}
    return result


def bootstrapped(frame, values, probability, intervals):
    y = frame.target.to_numpy()
    wide, fasting = (frame[column].to_numpy() for column in ("wide_weight", "fasting_weight"))
    units = [list(block.groupby("psu").position.apply(np.asarray))
             for _, block in frame.assign(position=np.arange(len(frame))).groupby("stratum")]
    paired = frame.paired.to_numpy()
    point = {name: values[name].to_numpy() for name in POINT_ORDER}
    errors = {name: np.abs(score - y) for name, score in point.items()}
    signed = {name: score - y for name, score in point.items()}
    event = y >= 140
    high = y >= 150
    subgroup = {name: mask.to_numpy() for name, mask in subgroup_masks(frame).items()}
    covered = {name: (y >= lo) & (y <= hi) for name, (lo, hi) in intervals.items()}
    draws = {}

    def record(key, value):
        draws.setdefault(key, [])
        if value is not None and np.isfinite(value):
            draws[key].append(float(value))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(REPLICATES):
        resampled = np.concatenate([np.concatenate([group[i] for i in rng.integers(0, len(group), len(group))])
                                    for group in units])
        event_indices = event[resampled]
        roc = {}
        for name in (PROB_REGISTERED, PROB_UI, PROB_AGE):
            roc[name] = roc_auc_score(event_indices, probability[name][resampled], sample_weight=wide[resampled]) \
                if np.unique(event_indices).size > 1 else None
            record(f"probability_{name}_auc", roc[name])
        for label, a, b in (("registered_vs_logistic", PROB_REGISTERED, PROB_UI), ("interface_vs_age", PROB_UI, PROB_AGE)):
            record(f"probability_{label}_delta_auc", roc[a] - roc[b] if roc[a] is not None else None)
        for candidate, base in (("main", "main_base_5f"), ("safe", "safe_base_5f")):
            full = resampled[np.isfinite(point[candidate][resampled])]
            both = full[paired[full]]
            for label, idx in (("overall", both), ("ge150", both[high[both]])):
                w = fasting[idx]
                a, b = [np.average(errors[name][idx], weights=w) if len(idx) else None for name in ("v2", candidate)]
                record(f"{candidate}_{label}_mae", b)
                record(f"{candidate}_{label}_v2_mae", a)
                record(f"{candidate}_{label}_delta_mae", b - a if a is not None else None)
                record(f"{candidate}_{label}_reduction", 1 - b / a if a else None)
                shared = idx[np.isfinite(point["ridge_reference"][idx])]
                record(f"{candidate}_{label}_delta_ridge", np.average(
                    (errors[candidate] - errors["ridge_reference"])[shared], weights=fasting[shared]) if len(shared) else None)
            for name, group_mask in subgroup.items():
                idx = both[group_mask[both]]
                for measure, arrays in (("mae", errors), ("bias", signed)):
                    record(f"{candidate}_{name}_delta_{measure}_base",
                           np.average((arrays[candidate] - arrays[base])[idx], weights=fasting[idx]) if len(idx) else None)
            for label, mask in {"overall": np.ones(len(y), dtype=bool), "ge150": high, **subgroup}.items():
                fidx, pidx = full[mask[full]], both[mask[both]]
                for cohort, idx, weight in (("full", fidx, wide), ("paired", pidx, fasting)):
                    for weighted in (False, True):
                        w = weight[idx] if weighted else None
                        cvg = np.average(covered[candidate][idx], weights=w) if len(idx) else None
                        key = f"{candidate}_{cohort}_{label}_{'weighted_' if weighted else ''}coverage"
                        record(key, cvg)
                        if cohort == "paired":
                            v2 = np.average(covered["v2"][idx], weights=w) if len(idx) else None
                            record(f"{key}_v2", v2)
                            record(f"{key}_delta_v2", cvg - v2 if len(idx) else None)
    return {name: {"valid_replicates": len(values), "ci95": np.quantile(values, [.025, .975]).tolist()
                   if len(values) >= .95 * REPLICATES else None} for name, values in draws.items()}


def gate(candidate, paired, cqr_full, probability, frozen):
    overall, tail = (paired[key]["models"] for key in ("overall", "sbp_ge150"))
    a, b = overall[candidate]["weighted_mae"], overall["v2"]["weighted_mae"]
    c, d = tail[candidate]["weighted_mae"], tail["v2"]["weighted_mae"]
    p, q = (probability[name]["roc_auc"] for name in (PROB_REGISTERED, PROB_UI))
    blockers = []
    spec = frozen["point"] if candidate == "main" else frozen["pre_e"]["safe_point"]
    for label in (key for key in paired if "_no_ge" in key):
        measures = paired[label]["models"]
        current, base = [measures[key] for key in (candidate, f"{candidate}_base_5f")]
        if "bpq020" in spec["fields"] and current["n"] and (
                current["weighted_mae"] > base["weighted_mae"] or
                abs(current["weighted_mean_signed_error"]) > abs(base["weighted_mean_signed_error"])):
            blockers.append(label)
    tests = {"general_mae": bool(a is not None and b is not None and a <= b + .5),
             "high_mae": bool(c is not None and d is not None and tail["v2"]["n"] >= 30 and d > 0 and c <= .75 * d),
             "registered_probability_auc": bool(p is not None and q is not None and p > q),
             "interval_high_coverage": bool(cqr_full["ge150"]["coverage"] is not None and cqr_full["ge150"]["coverage"] > .26),
             "interval_overall_coverage": bool(cqr_full["overall"]["coverage"] is not None and cqr_full["overall"]["coverage"] >= .85)}
    return {"criteria": tests, "blocking_groups": blockers,
            "passes_all_and_unblocked": all(tests.values()) and not blockers,
            "secondary_multiplicity": candidate == "safe"}


def evaluate(frame, values, scores, flags, intervals, frozen):
    probability = {key: probability_metrics(frame, score, flags[key]) for key, score in scores.items()}
    result = []
    for candidate in ("main", "safe"):
        full = frame.loc[values[candidate].notna()]
        both = full.loc[full.paired]
        names = [candidate, f"{candidate}_base_5f"] + (["ridge_reference"] if candidate == "main" else [])
        point_full = point_summary(full, values, names, "wide_weight")
        point_paired = point_summary(both, values, [*names, "v2"], "fasting_weight")

        def interval_on(rows, key, column):
            mask = frame.index.isin(rows.index)
            return interval_metrics(rows, *(bound[mask] for bound in intervals[key]), column)

        cqr_full = interval_on(full, candidate, "wide_weight")
        result.append({"candidate": candidate, "expanded_n": len(full), "paired_n": len(both),
                       "excluded_for_point_fields": len(frame) - len(full), "point_expanded": point_full,
                       "point_paired": point_paired, "cqr_expanded": cqr_full,
                       "cqr_paired": interval_on(both, candidate, "fasting_weight"),
                       "v2_interval_paired": interval_on(both, "v2", "fasting_weight"),
                       "gates": gate(candidate, point_paired, cqr_full, probability, frozen)})
    return {"candidate_order": ["main", "safe"], "candidates": result, "probability": probability,
            "probability_thresholds": frozen["pre_e"]["probability"]["thresholds"],
            "probability_auc_deltas": {
                "registered_candidate_minus_same_fields_logistic": auc_difference(probability, PROB_REGISTERED, PROB_UI),
                "interface_logistic_minus_age_logistic": auc_difference(probability, PROB_UI, PROB_AGE)},
            "paired_psu_bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": REPLICATES,
                                     "metrics": bootstrapped(frame, values, scores, intervals)}}


def auc_difference(metrics, a, b):
    x, y = (metrics[name]["roc_auc"] for name in (a, b))
    return x - y if x is not None and y is not None else None


def format_number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def table(headings, rows):
    return ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows], ""]


def measurement_comparison(reference, current):
    comparison = []
    for dev, external in zip(reference["metrics"]["candidates"], current["candidates"], strict=True):
        field = "point_paired"
        for group, summary in external[field].items():
            if not group.startswith("sbp_"):
                continue
            for model in (external["candidate"], "v2"):
                a, b = [row["models"][model] for row in (dev[field][group], summary)]
                old, new = a["weighted_mean_signed_error"], b["weighted_mean_signed_error"]
                comparison.append({"candidate": external["candidate"], "group": group, "model": model,
                                   "development_n": a["n"], "external_n": b["n"],
                                   "development_bias": old, "external_bias": new,
                                   "external_minus_development": new - old if old is not None and new is not None else None})
    return comparison


def render(report):
    m = report["metrics"]
    section = ["# M2-v5. " + ("Пробный прогон E" if report["non_confirmatory"] else "Внешняя проверка E"),
               "", f"Режим: **{report['mode']}**. Цикл: **{report['cycle']}**.", "",
               ("**Non-confirmatory.** Использованы части development, на которых калибровались выходы. "
                "Значения не участвуют в выборе и не подтверждают пороги E."
                if report["non_confirmatory"] else
                "Оценка замороженного манифеста на закрытом цикле. Вторичный кандидат отчитывается после основного, "
                "его интервалы номинальные и не учитывают множественность."), "",
               f"SHA-256 манифеста: `{report['manifest_sha256']}`.", "",
               "## Поток участников", ""]
    flow = report["cohort_flow"]
    section += table(["Шаг", "Исключено", "Осталось"],
                     [(label, item["excluded"], item["remaining"]) for label, item in flow["expanded_attrition"].items()])
    section += table(["Причина потери парности с v2", "n"], flow["paired_exclusions"].items())
    section += ["## Точечные модели и группы", ""]
    for result in m["candidates"]:
        candidate = result["candidate"]
        name = "Основной Ridge" if candidate == "main" else "Вторичная MLP без BPQ020 (множественное сравнение)"
        excluded = result['excluded_for_point_fields']
        loss = excluded / (result['expanded_n'] + excluded)
        section += [f"### {name}", "", f"Расширенная когорта: {result['expanded_n']}. "
                    f"Общая с v2: {result['paired_n']}. Исключено из-за полей точечной модели: "
                    f"{excluded} ({loss:.1%}).", ""]
        for label, field in (("Общая с v2", "point_paired"), ("Расширенная", "point_expanded")):
            section += [f"#### {label}", ""]
            section += table(["Группа", "Модель", "n", "MAE взв. / невзв.",
                              "RMSE взв. / невзв.", "R² взв. / невзв.", "Bias взв. / невзв."],
                             [(group + (" (опис.)" if item["descriptive_only"] else ""), model, row["n"],
                               *[" / ".join(format_number(row.get(metric)) for metric in keys)
                                 for keys in (("weighted_mae", "mae"), ("weighted_rmse", "rmse"),
                                              ("weighted_r2", "r2"), ("weighted_mean_signed_error", "mean_signed_error"))])
                              for group, item in result[field].items() for model, row in item["models"].items()])
        paired = result["point_paired"]
        changes = []
        for label in ("overall", "sbp_ge150"):
            model, baseline = (paired[label]["models"][key]["weighted_mae"] for key in (candidate, "v2"))
            delta = model - baseline if model is not None and baseline is not None else None
            relative = -100 * delta / baseline if delta is not None and baseline else None
            changes.append((label, paired[label]["n"], format_number(delta), format_number(relative)))
        section += table(["Группа", "n", "ΔMAE к v2", "Снижение MAE, %"], changes)
        section += ["#### Пороги и блокирующие группы", ""]
        section += table(["Критерий", "Выполнен"], [(key, "да" if value else "нет")
                                                    for key, value in result["gates"]["criteria"].items()])
        section += ["Блокирующие срезы — " + (", ".join(result["gates"]["blocking_groups"]) or "нет") + ".", ""]
    section += ["## Вероятность САД ≥140", ""]
    section += table(["Модель", "n / событий", "Частота взв.", "Порог", "ROC-AUC", "PR-AUC", "Brier",
                      "Чувствительность", "Специфичность", "PPV", "NPV"],
                     [(name, f"{row['n']} / {row['event_n']}", format_number(row["weighted_prevalence"], 4),
                       format_number(m["probability_thresholds"][name], 5),
                       *[format_number(row.get(key), 4) for key in
                         ("roc_auc", "pr_auc_average_precision", "brier", "sensitivity", "specificity", "ppv", "npv")])
                      for name, row in m["probability"].items()])
    section += table(["AUC-сравнение", "Парная Δ"], [(name, format_number(value, 4))
                                                      for name, value in m["probability_auc_deltas"].items()])
    section += table(["Модель", "Группа риска", "n", "Чувствительность"],
                     [(name, key, item["n"], format_number(item["sensitivity"], 4))
                      for name, row in m["probability"].items() for key, item in row["undiagnosed_sensitivity"].items()])
    section += table(["Модель", "Бин", "n", "Средняя вероятность", "Доля события"],
                     [(name, f"{item['low_inclusive']:.1f}–{item['high']:.1f}", item["n"],
                       format_number(item["weighted_mean_probability"], 4), format_number(item["weighted_event_rate"], 4))
                      for name, row in m["probability"].items() for item in row["calibration_bins"]])
    section += ["## CQR и симметричный интервал", ""]
    for row in m["candidates"]:
        section += [f"### {row['candidate']}", ""]
        section += table(["Охват и метод", "Группа", "n", "Покрытие взв. / невзв.", "Ширина взв. / невзв."],
                         [(field, group, item["n"],
                           f"{format_number(item['weighted_coverage'])} / {format_number(item['coverage'])}",
                           f"{format_number(item['weighted_width'])} / {format_number(item['width'])}")
                          for field in ("cqr_expanded", "cqr_paired", "v2_interval_paired") for group, item in row[field].items()])
    section += ["## Парный PSU-бутстреп", "",
                f"Повторов: {REPLICATES}. Seed: {BOOTSTRAP_SEED}. Подгруппы выделены после ресэмплинга. "
                "Разницу с замороженным Ridge считаем на общих строках; основной кандидат совпадает с этим Ridge.", ""]
    section += table(["Метрика", "Валидных повторов", "Номинальный 95% ДИ"],
                     [(name, item["valid_replicates"],
                       "…".join(format_number(value, 4) for value in item["ci95"]) if item["ci95"] is not None else "—")
                      for name, item in m["paired_psu_bootstrap"]["metrics"].items()])
    section += ["## Метод измерения САД", "",
                "В 2015–2016 САД измеряли аускультативно, в development — осциллометрически. "
                "Разницу метода нельзя отделить от различий когорт. Development-ориентир получен на калибровочных частях." , "",
                "[Парное сравнение методов NCHS](https://stacks.cdc.gov/view/cdc/104185)", ""]
    if "measurement_comparison" in report:
        section += table(["Кандидат", "Диапазон", "Модель", "n 2021–2023 / 2015–2016",
                          "Bias 2021–2023", "Bias 2015–2016", "Δ bias"],
                         [(row["candidate"], row["group"], row["model"],
                           f"{row['development_n']} / {row['external_n']}",
                           *[format_number(row[key]) for key in
                             ("development_bias", "external_bias", "external_minus_development")])
                          for row in report["measurement_comparison"]])
    section += ["## Проверки артефактов", "", f"Синтетические эталоны: {report['probe_replay']['synthetic_n']}. "
                "Хэши предсказаний совпали с заморозкой.",
                f"Максимальное отклонение ONNX от архивных Python-эталонов: "
                f"{format_number(report['probe_replay']['v2_python_golden_max_abs_error'], 8)} мм рт. ст.", ""]
    if row := report.get("v2_onnx_python_pairs"):
        section += [f"На всех {row['n']} парных development-строках сравнение с замороженным PyTorch: "
                    f"максимальное отклонение {format_number(row['max_absolute_difference'], 8)} мм рт. ст.", ""]
    section += ["Исходные строки и построчные предсказания не сохранены.", ""]
    return "\n".join(section)


def fetch_rows(mode, audit):
    if mode == "sealed":
        return load_sealed(audit), SEALED
    from m2_v5_cv import development
    dev, _ = development()
    _, cqr, prob = final_sets(dev, read_json(SPLITS)["splits"])
    raw, _ = load_development(NEW, audit["source_manifest"])
    return raw.loc[raw.SEQN.isin(np.concatenate([cqr, prob]))], NEW


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(mode, args):
    sealed = mode == "sealed"
    manifest_sha = sha256(MANIFEST)
    output = ROOT / "reports/m2_v5_e_final.json" if sealed else DRY_REPORT
    docs = ROOT / "docs" / ("m2-v5-final.md" if sealed else "m2-v5-e-dry-run.md")
    if output.exists() or docs.exists():
        raise ValueError("Refusing to overwrite an existing evaluation result")
    frozen, audit, models, replay = load_verified()
    attempt = None
    if sealed:
        reference = read_json(DRY_REPORT)
        if not reference["non_confirmatory"] or reference["cycle"] != NEW or reference["manifest_sha256"] != manifest_sha:
            raise ValueError("Development comparison must use the same frozen models")
        history = read_json(LOG) if LOG.exists() else {"manifest_sha256": manifest_sha, "attempts": []}
        if history["manifest_sha256"] != manifest_sha or any(
                item["status"] != "failed_before_metrics" for item in history["attempts"]):
            raise ValueError("Sealed evaluation is not eligible for a retry")
        attempt = {"started_at_utc": timestamp(), "runner_sha256": sha256(Path(__file__)), "status": "loading"}
        history["attempts"].append(attempt)
        write_json(LOG, history)
    try:
        raw, cycle = fetch_rows(mode, audit)
        frame, flow = prepare(raw, cycle, read_json(V2_METADATA))
        values, scores, flags, intervals = predictions(frame, models)
        report = {"mode": "sealed" if sealed else "development_calibration_dry_run",
                  "cycle": cycle, "non_confirmatory": not sealed, "sealed_records_read": sealed,
                  "manifest_sha256": manifest_sha, "runner_sha256": sha256(Path(__file__)),
                  "measurement_source_sha256": sha256(ROOT / "src/medpredict_ml/blood_pressure.py"),
                  "compared_models": POINT_ORDER, "cohort_flow": flow, "probe_replay": replay}
        if not sealed:
            pairs = frame.loc[frame.paired]
            report["v2_onnx_python_pairs"] = python_parity(args.v2_checkpoint, pairs, values.loc[pairs.index, "v2"])
        if attempt is not None:
            attempt["status"] = "computing_metrics"
            write_json(LOG, history)
        report["metrics"] = evaluate(frame, values, scores, flags, intervals, frozen)
        if sealed:
            report["measurement_comparison"] = measurement_comparison(reference, report["metrics"])
        write_json(output, report)
        docs.write_text(render(report), encoding="utf-8")
        if attempt is not None:
            attempt.update(status="completed", result_sha256=sha256(output), finished_at_utc=timestamp())
            write_json(LOG, history)
    except Exception:
        if attempt is not None and attempt["status"] == "loading":
            attempt.update(status="failed_before_metrics", finished_at_utc=timestamp())
            write_json(LOG, history)
        raise
    print(f"Evaluation written to {output.name} and {docs.name} ({report['mode']}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "sealed"))
    parser.add_argument("--v2-checkpoint", type=Path)
    parser.add_argument("--confirm-e", help="Explicit sealed-run confirmation containing the active frozen manifest SHA-256")
    args = parser.parse_args()
    worker_init()
    if args.mode == "dry-run":
        if args.v2_checkpoint is None or args.confirm_e is not None:
            parser.error("dry-run requires the archived v2 PyTorch checkpoint and accepts no sealed confirmation")
        run(args.mode, args)
        return
    if args.confirm_e != sha256(MANIFEST) or args.v2_checkpoint is not None:
        parser.error("sealed evaluation requires explicit confirmation of the frozen manifest SHA-256")
    from fcntl import LOCK_EX, LOCK_NB, flock
    from m2_v5_grid import BUNDLE
    with (BUNDLE.parent / ".evaluation.lock").open("a") as lock:
        flock(lock, LOCK_EX | LOCK_NB)
        run(args.mode, args)


if __name__ == "__main__":
    main()
