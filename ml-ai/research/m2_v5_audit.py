"""Stage B NHANES audit. The sealed cycle is never passed to a table reader."""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from medpredict_ml.common import ROOT, read_json, sha256, write_json
from medpredict_ml.data import valid_transport, validate_table

SEED = 2026  # Registered; the audit itself has no randomized operations.
PROTOCOL = ROOT / "protocols/m2-v5.md"
RAW = ROOT / "data/raw/m2-v5"
OUTPUT = ROOT / "reports/m2_v5_audit.json"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
OLD, NEW, SEALED = "2017-2020", "2021-2023", "2015-2016"
YEARS = {OLD: 2017, NEW: 2021, SEALED: 2015}
TABLES = {
    OLD: ("P_DEMO", "P_BMX", "P_TCHOL", "P_GHB", "P_BPXO", "P_BPQ", "P_SMQ", "P_PAQ", "P_GLU"),
    NEW: ("DEMO_L", "BMX_L", "TCHOL_L", "GHB_L", "BPXO_L", "BPQ_L", "SMQ_L", "PAQ_L", "GLU_L"),
    SEALED: ("DEMO_I", "BMX_I", "TCHOL_I", "GHB_I", "BPX_I", "BPQ_I", "SMQ_I", "PAQ_I", "GLU_I"),
}
REQUIRED = {
    "bmx": ["BMXBMI"], "tchol": ["LBDTCSI"], "bpx": [
        f"BPXO{measurement}{index}" for measurement in ("SY", "DI") for index in (1, 2, 3)
    ], "smq": ["SMQ020", "SMQ040"], "paq": ["PAD680"],
}
RANGES = ["lt110", "110_to_lt140", "140_to_lt150", "ge150"]
ANSWER_CODES = {"yes": 1, "no": 2, "refused": 7, "dont_know": 9}


class AuditStop(ValueError):
    """A finding that requires review before the next stage."""


def raw_path(cycle: str, table: str) -> Path:
    directory = RAW / ("sealed-2015-2016" if cycle == SEALED else f"development/{cycle}")
    return directory / f"{table}.xpt"


def download_file(task: tuple[str, str]) -> dict:
    cycle, table = task
    path = raw_path(cycle, table)
    url = f"{BASE}/{YEARS[cycle]}/DataFiles/{table}.xpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": "MedPredict-University-Research/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as output:
                size = 0
                while block := source.read(128 * 1024):
                    size += len(block)
                    if size > 30_000_000:
                        raise AuditStop(f"{table}: XPT exceeds expected size")
                    output.write(block)
            if not valid_transport(partial):
                raise AuditStop(f"{table}: downloaded object is not XPT")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if not valid_transport(path):
        raise AuditStop(f"{table}: invalid cached XPT header")
    return {"cycle": cycle, "table": table, "url": url, "codebook": url.removesuffix(".xpt") + ".htm",
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "downloaded_at_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
            "xpt_header_verified": True, "records_parsed": False, "sealed_until_stage_e": cycle == SEALED}


def download_manifest() -> list[dict]:
    tasks = [(cycle, table) for cycle, tables in TABLES.items() for table in tables]
    paths = [str(p.relative_to(ROOT.parent)) for task in tasks for p in
             (raw_path(*task), raw_path(*task).with_suffix(".part"))]
    checked = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT.parent,
                             input="\n".join(paths) + "\n", text=True, capture_output=True, check=True)
    if set(checked.stdout.splitlines()) != set(paths):
        raise AuditStop("XPT targets and partial files must all be ignored by Git")
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(download_file, tasks))


def load_development(cycle: str, manifest: list[dict]) -> tuple[pd.DataFrame, dict]:
    if cycle not in (OLD, NEW):
        raise AuditStop("Only development cycles may be parsed")
    wide_weight, fasting_weight, medication = (
        ("WTMECPRP", "WTSAFPRP", "BPQ050A") if cycle == OLD else ("WTPH2YR", "WTSAF2YR", "BPQ150")
    )
    columns = {
        "demo": ["RIDAGEYR", "RIAGENDR", "SDMVSTRA", "SDMVPSU"] + ([wide_weight] if cycle == OLD else []),
        **REQUIRED,
        "ghb": ["LBXGH"] + ([wide_weight] if cycle == NEW else []),
        "bpq": ["BPQ020", medication] + (["BPQ040A"] if cycle == OLD else []),
        "glu": ["LBDGLUSI", fasting_weight],
    }
    names = ("demo", "bmx", "tchol", "ghb", "bpx", "bpq", "smq", "paq", "glu")
    entries = {item["table"]: item for item in manifest if item["cycle"] == cycle}
    checks = {}
    for name, table in zip(names, TABLES[cycle], strict=True):
        path = raw_path(cycle, table)
        if sha256(path) != entries[table]["sha256"]:
            raise AuditStop(f"{table}: XPT changed after verification")
        required = ["SEQN", *columns[name]]
        try:
            records, _ = pyreadstat.read_xport(path, encoding="latin1", usecols=required)
            validate_table(records, required, table)
        except (RuntimeError, ValueError, pyreadstat.ReadstatError) as exc:
            raise AuditStop(f"{table}: schema or unique SEQN check failed: {exc}") from exc
        entries[table]["records_parsed"] = True
        if name == "demo":
            frame = records
        outside = int((~records.SEQN.isin(frame.SEQN)).sum())
        checks[table] = {"rows": len(records), "unique_seqn": True, "outside_demographics": outside}
        if outside:
            raise AuditStop(f"{table}: unmatched participants outside demographics")
        if name != "demo":
            frame = frame.merge(records, how="left", on="SEQN", validate="one_to_one", sort=False)
    return frame, checks


def weight_stats(values: pd.Series) -> dict:
    missing = ~np.isfinite(values)
    zero = values == 0
    positive = np.isfinite(values) & (values > 0)
    if ((values < 0) | (positive & (values < 1))).any():
        raise AuditStop(f"{values.name}: negative or unexpectedly small positive survey weight")
    return {"n": len(values), "positive": int(positive.sum()), "zero": int(zero.sum()),
            "missing": int(missing.sum()), "zero_fraction": float(zero.mean()),
            "missing_fraction": float(missing.mean()), "sum": float(values[positive].sum())}


def design(frame: pd.DataFrame) -> dict:
    sizes = frame.groupby("SDMVSTRA").SDMVPSU.nunique()
    return {"strata": len(sizes), "nested_psu": int(sizes.sum()),
            "singleton_strata": int((sizes == 1).sum())}


def cohort(frame: pd.DataFrame, weight: str) -> dict:
    groups = pd.cut(frame.sbp, [-np.inf, 110, 140, 150, np.inf], labels=RANGES, right=False)
    return {"n": len(frame), "sbp_bins": {key: int((groups == key).sum()) for key in RANGES},
            "weight": weight, "weights": weight_stats(frame[weight]), "design": design(frame)}


def questionnaire(frame: pd.DataFrame, cycle: str) -> tuple[dict, list[str]]:
    med = "BPQ050A" if cycle == OLD else "BPQ150"
    skipped = frame.BPQ020.isin([2, 7, 9])
    medicines_skipped = skipped | frame.BPQ040A.isin([2, 7, 9]) if cycle == OLD else skipped
    dependencies = {med: (medicines_skipped, frame.BPQ040A == 1 if cycle == OLD else frame.BPQ020 == 1),
                    "BPQ020": (pd.Series(False, index=frame.index), frame.RIDAGEYR >= 16)}
    if cycle == OLD:
        dependencies["BPQ040A"] = (skipped, frame.BPQ020 == 1)
    anomalies = []
    raw = {}
    for name in ("BPQ020", "BPQ040A", "BPQ050A", "BPQ150"):
        if name not in frame:
            raw[name] = {"present_in_cycle": False}
            continue
        values = frame[name]
        skip, eligible = dependencies[name]
        not_applicable = skip | (frame.RIDAGEYR < 16)
        counts = {label: int(values.eq(code).sum()) for label, code in ANSWER_CODES.items()}
        mismatch = int((not_applicable & values.notna()).sum())
        invalid = int((values.notna() & ~values.isin(ANSWER_CODES.values())).sum())
        if mismatch or invalid:
            anomalies.append(f"{cycle} {name}: {mismatch} responses outside documented route, {invalid} invalid codes")
        raw[name] = {"present_in_cycle": True, "n": len(frame), **counts,
                     "not_applicable": int((values.isna() & not_applicable).sum()),
                     "missing_on_expected_route": int((values.isna() & eligible & ~not_applicable).sum()),
                     "missing_route_undetermined": int((values.isna() & ~eligible & ~not_applicable).sum()),
                     "route_mismatch": mismatch, "invalid": invalid}
    smoking_values = frame.SMQ040
    if (frame.SMQ020.notna() & ~frame.SMQ020.isin(ANSWER_CODES.values())).any():
        anomalies.append(f"{cycle} SMQ020: invalid codes")
    if (smoking_values.notna() & ~smoking_values.isin([1, 2, 3, 7, 9])).any():
        anomalies.append(f"{cycle} SMQ040: invalid codes")
    if (frame.SMQ020.isin([2, 7, 9]) & smoking_values.notna()).any():
        anomalies.append(f"{cycle} SMQ040: responses outside documented route")
    pad = frame.PAD680
    pad_valid = pad.between(0, 1439)
    pad_invalid = int((pad.notna() & ~pad_valid & ~pad.isin([7777, 9999])).sum())
    if pad_invalid:
        anomalies.append(f"{cycle} PAD680: {pad_invalid} values outside valid range and documented codes")
    return {"bpq": raw, "smq020_codes": codes(frame.SMQ020, [1, 2, 7, 9]),
            "smq040_codes": codes(smoking_values, [1, 2, 3, 7, 9]),
            "pad680": {"n": len(frame), "valid": int(pad_valid.sum()),
                       "valid_fraction": float(pad_valid.mean()), "missing": int(pad.isna().sum()),
                       "code_7777": int(pad.eq(7777).sum()), "code_9999": int(pad.eq(9999).sum()),
                       "code_7777_fraction": float(pad.eq(7777).mean()), "code_9999_fraction": float(pad.eq(9999).mean()),
                       "other_invalid": pad_invalid}}, anomalies


def codes(values: pd.Series, allowed: list[int]) -> dict:
    return {**{str(code): int(values.eq(code).sum()) for code in allowed}, "missing": int(values.isna().sum())}


def c4(frame: pd.DataFrame, cycle: str) -> dict:
    medications = frame["BPQ050A" if cycle == OLD else "BPQ150"].map({1: "yes", 2: "no"}).fillna("unknown")
    told = frame.BPQ020.map({1: "yes", 2: "no"}).fillna("unknown")
    smoking = frame.SMQ040.where(frame.SMQ020 == 1).map({1: "current", 2: "current", 3: "former"})
    smoking = smoking.mask(frame.SMQ020 == 2, "never").fillna("unknown")
    sedentary = frame.PAD680.between(0, 1439)
    measures = {"medication": medications, "bpq020": told, "smoking": smoking,
                "pad680": pd.Series(np.where(sedentary, "valid", "unknown"), index=frame.index)}
    report = {}
    for name, labels in measures.items():
        known = labels != "unknown"
        # Unknown is an explicit registered category for questionnaire items; PAD680 is complete-case.
        included = known if name == "pad680" else pd.Series(True, index=frame.index)
        counts = labels.value_counts()
        report[name] = {"n": len(frame), "known_n": int(known.sum()), "valid_fraction": float(known.mean()),
                        "known_ge150": int((known & (frame.sbp >= 150)).sum()),
                        "categories": counts.to_dict(), "category_fractions": (counts / len(frame)).to_dict(),
                        "complete_case_n": int(included.sum()),
                        "complete_case_ge150": int((included & (frame.sbp >= 150)).sum())}
    return report


def audit_cycle(frame: pd.DataFrame, cycle: str, bounds: list[dict]) -> tuple[dict, list[str]]:
    flow = []

    def retain(rows: pd.DataFrame, mask: pd.Series, rule: str) -> pd.DataFrame:
        kept = rows.loc[mask]
        flow.append({"rule": rule, "n_before": len(rows), "excluded": len(rows) - len(kept), "remaining": len(kept)})
        return kept

    wide_weight, fasting_weight = ("WTMECPRP", "WTSAFPRP") if cycle == OLD else ("WTPH2YR", "WTSAF2YR")
    weights = {name: weight_stats(frame[name]) for name in (wide_weight, fasting_weight)}
    systolic = frame[[f"BPXOSY{i}" for i in (1, 2, 3)]].to_numpy(float)
    diastolic = frame[[f"BPXODI{i}" for i in (1, 2, 3)]].to_numpy(float)
    valid = np.isfinite(systolic) & (systolic > 0) & (systolic <= 300)
    valid &= np.isnan(diastolic) | (np.isfinite(diastolic) & (diastolic > 0) & (systolic > diastolic))
    target = pd.DataFrame(np.where(valid, systolic, np.nan)).mean(axis=1)
    frame = frame.assign(sbp=target.to_numpy(), valid_readings=valid.sum(axis=1))
    expanded = retain(frame, frame.RIDAGEYR.between(18, 79), "age_18_to_79")
    expanded = retain(expanded, expanded.RIAGENDR.isin([1, 2]), "documented_sex")
    expanded = retain(expanded, expanded.valid_readings >= 2, "two_valid_sbp_readings")
    expanded = retain(expanded, np.isfinite(expanded[["SDMVSTRA", "SDMVPSU"]]).all(axis=1), "observed_design")
    expanded = retain(expanded, np.isfinite(expanded[wide_weight]) & (expanded[wide_weight] > 0), "positive_expanded_weight")
    for field in ("BMXBMI", "LBDTCSI", "LBXGH"):
        expanded = retain(expanded, np.isfinite(expanded[field]) & (expanded[field] > 0), f"observed_positive_{field}")
    expanded_flow, flow = flow, []
    paired = retain(expanded, np.isfinite(expanded[fasting_weight]) & (expanded[fasting_weight] > 0), "positive_fasting_weight")
    paired = retain(paired, np.isfinite(paired.LBDGLUSI) & (paired.LBDGLUSI > 0), "observed_positive_fasting_glucose")
    columns = {"age": "RIDAGEYR", "bmi": "BMXBMI", "totalCholesterol": "LBDTCSI", "glucose": "LBDGLUSI"}
    for field in bounds:
        if field["key"] in columns:
            column = columns[field["key"]]
            mask = paired[column].between(field["min"], field["max"])
            if field.get("integer", False):
                mask &= paired[column] % 1 == 0
            paired = retain(paired, mask, f"v2_bounds_{field['key']}")
    if expanded.empty or paired.empty:
        raise AuditStop(f"{cycle}: an audited cohort is empty")
    bpq, anomalies = questionnaire(frame, cycle)
    return {"demographics_n": len(frame), "raw_weights": weights, "raw_questionnaire": bpq,
            "expanded": {**cohort(expanded, wide_weight), "selection_flow": expanded_flow,
                         "c4": c4(expanded, cycle), "questionnaire": questionnaire(expanded, cycle)[0]},
            "paired_v2": {**cohort(paired, fasting_weight), "selection_flow": flow,
                          "c4": c4(paired, cycle), "questionnaire": questionnaire(paired, cycle)[0]}}, anomalies


def audit(manifest: list[dict]) -> dict:
    cycles, frames, findings = {}, {}, []
    bounds = read_json(ROOT / "artifacts/metadata.json")["fields"]
    for cycle in (OLD, NEW):
        frame, checks = load_development(cycle, manifest)
        frames[cycle] = frame
        cycles[cycle], anomalies = audit_cycle(frame, cycle, bounds)
        cycles[cycle]["joins"] = checks
        findings.extend(anomalies)
        for population in ("expanded", "paired_v2"):
            for field, item in cycles[cycle][population]["c4"].items():
                if item["complete_case_ge150"] < 30:
                    findings.append(f"{cycle} {population} {field}: fewer than 30 participants with SBP >=150")
    old, new = (frames[cycle] for cycle in (OLD, NEW))
    seqn_overlap = len(set(old.SEQN) & set(new.SEQN))
    if seqn_overlap:
        findings.append(f"Participant identifiers overlap between development cycles: {seqn_overlap}")
    strata_overlap = len(set(old.SDMVSTRA.dropna()) & set(new.SDMVSTRA.dropna()))
    psu_overlap = len(set(old.SDMVPSU.dropna()) & set(new.SDMVPSU.dropna()))
    raw_pairs = [set(frame[["SDMVSTRA", "SDMVPSU"]].dropna().itertuples(index=False, name=None)) for frame in (old, new)]
    prefixed = [{(cycle, stratum, psu) for stratum, psu in pairs}
                for cycle, pairs in zip((OLD, NEW), raw_pairs, strict=True)]
    assert not prefixed[0] & prefixed[1]
    return {
        "development": cycles, "stop_findings": findings,
        "cross_cycle": {"seqn_overlap": seqn_overlap, "raw_stratum_label_overlap": strata_overlap,
                        "raw_psu_label_overlap": psu_overlap, "raw_nested_psu_overlap": len(raw_pairs[0] & raw_pairs[1]),
                        "merged_stratum_key": ["cycle", "SDMVSTRA"],
                        "merged_psu_key": ["cycle", "SDMVSTRA", "SDMVPSU"], "prefixed_nested_psu_overlap": 0},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = download_manifest()
    report = {"protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__)), "registered_seed": SEED,
              "python": platform.python_version(), "packages": {name: version(name) for name in ("numpy", "pandas", "pyreadstat")},
              "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_manifest": manifest, "sealed_cycle": SEALED, "sealed_records_parsed": False}
    try:
        report.update(audit(manifest))
    except AuditStop as exc:
        report["stop_findings"] = [str(exc)]
    sealed = [entry for entry in manifest if entry["cycle"] == SEALED]
    assert len(sealed) == 9 and all(not entry["records_parsed"] for entry in sealed)
    write_json(args.output, report)
    if report["stop_findings"]:
        raise SystemExit("\n".join(report["stop_findings"]))
    print(f"Audited {len(manifest)} files; {len(sealed)} sealed files verified without parsing rows.")


if __name__ == "__main__":
    main()
