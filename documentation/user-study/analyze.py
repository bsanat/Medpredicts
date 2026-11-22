"""Aggregate the preregistered MedPredict UX study CSV without exporting responses."""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist, mean, median


STEPS = tuple(f"t{i}" for i in range(1, 7))
SUS = tuple(f"sus{i}" for i in range(1, 11))
QUESTIONS = tuple(f"q{i}" for i in range(1, 8))
OPEN = ("q2", "q5")
RATINGS = ("correct", "partial", "incorrect")
STATUSES = ("unaided", "assisted", "not_completed", "technical", "missing")
FIELDS = (
    "participant_id", "round", "language",
    *(f"{step}_{suffix}" for step in STEPS for suffix in ("seconds", "errors", "status")),
    *SUS,
    *(field for q in QUESTIONS for field in (
        (f"{q}_text", f"{q}_rater1", f"{q}_rater2", f"{q}_final") if q in OPEN else (q,)
    )),
)
Z95 = NormalDist().inv_cdf(.975)


def validate_row(row):
    if not re.fullmatch(r"P(?:0[1-9]|[1-9][0-9])", row["participant_id"]):
        raise ValueError("participant_id: expected code P01–P99")
    if row["round"] not in ("1", "2") or row["language"] not in ("ru", "kk"):
        raise ValueError("round / language: expected 1 or 2 and ru or kk")

    for key in SUS:
        if row[key] and row[key] not in ("1", "2", "3", "4", "5"):
            raise ValueError(f"{key}: expected 1–5 or blank")
    for step in STEPS:
        fields = (f"{step}_seconds", f"{step}_errors", f"{step}_status")
        seconds, errors, status = map(row.get, fields)
        if seconds and (not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", seconds)
                        or not math.isfinite(float(seconds.replace(",", ".")))):
            raise ValueError(f"{fields[0]}: expected a nonnegative number or blank")
        if errors and not re.fullmatch(r"[0-9]+", errors):
            raise ValueError(f"{fields[1]}: expected a nonnegative integer or blank")
        if status and status not in STATUSES:
            raise ValueError(f"{fields[2]}: unknown step status")
    for q in QUESTIONS:
        if q not in OPEN:
            if row[q] and row[q] not in ("yes", "no", "unsure"):
                raise ValueError(f"{q}: expected yes, no, unsure or blank")
            continue
        text, r1, r2, final = (row[f"{q}_{field}"] for field in ("text", "rater1", "rater2", "final"))
        if any(grade and grade not in RATINGS for grade in (r1, r2, final)):
            raise ValueError(f"{q}: expected correct, partial, incorrect or blank")
        if not text and any((r1, r2, final)):
            raise ValueError(f"{q}: ratings require an answer")
        if final and (not (r1 and r2) or (r1 == r2 and final != r1)):
            raise ValueError(f"{q}: final must follow two independent ratings")


def read_rows(path):
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        header = reader.fieldnames or []
        if len(header) != len(FIELDS) or set(header) != set(FIELDS):
            raise ValueError("CSV header must match responses-template.csv")
        for raw in reader:
            if None in raw or None in raw.values():
                raise ValueError(f"CSV row {reader.line_num}: unexpected number of cells")
            row = {key: value.strip() for key, value in raw.items()}
            if not any(row.values()):
                continue
            try:
                validate_row(row)
                if row["participant_id"] in seen:
                    raise ValueError("participant_id must be unique across rounds")
            except ValueError as exc:
                raise ValueError(f"CSV row {reader.line_num}: {exc}") from None
            seen.add(row["participant_id"])
            rows.append(row)
    return rows


def wilson(correct, total):
    if not total:
        return None
    proportion = correct / total
    denominator = 1 + Z95**2 / total
    center = (proportion + Z95**2 / (2 * total)) / denominator
    half = Z95 * math.sqrt((proportion * (1 - proportion) + Z95**2 / (4 * total)) / total) / denominator
    return [100 * max(0, center - half), 100 * min(1, center + half)]


def gate(round_number, ready, meets):
    if round_number == 2:
        return "descriptive"
    if not ready:
        return "provisional"
    return "met" if meets else "not_met"


def sus_summary(rows, round_number, planned_size):
    scores = []
    for row in rows:
        values = [int(row[key]) for key in SUS if row[key]]
        if len(values) == len(SUS):
            scores.append(2.5 * sum(value - 1 if i % 2 == 0 else 5 - value for i, value in enumerate(values)))
    average = mean(scores) if scores else None
    missing = len(rows) - len(scores)
    meets = average >= 68 if average is not None else None
    return {"complete_n": len(scores), "incomplete_n": missing, "mean": average,
            "median": median(scores) if scores else None, "threshold": 68,
            "estimate_meets_threshold": meets, "status": gate(round_number, planned_size and missing == 0, meets)}


def rater_agreement(ratings):
    pairs = [(a, b) for a, b in ratings if a and b]
    n = len(pairs)
    if not n:
        return {"paired_n": 0, "disagreements_n": 0, "agreement_pct": None, "cohen_kappa": None}
    matches = sum(a == b for a, b in pairs)
    expected_num = sum(sum(a == grade for a, _ in pairs) * sum(b == grade for _, b in pairs) for grade in RATINGS)
    kappa = (matches * n - expected_num) / (n**2 - expected_num) if expected_num != n**2 else None
    return {"paired_n": n, "disagreements_n": n - matches, "agreement_pct": 100 * matches / n, "cohen_kappa": kappa}


def question_summary(rows, q, round_number, planned_size):
    ratings, unresolved = [], 0
    if q in OPEN:
        answers = []
        for row in rows:
            r1, r2, final = (row[f"{q}_{key}"] for key in ("rater1", "rater2", "final"))
            ratings.append((r1, r2))
            grade = final or (r1 if r1 == r2 else "")
            unresolved += bool(row[f"{q}_text"] and not grade)
            answers.append(grade)
        correct = answers.count("correct")
        categories = RATINGS
    else:
        answers = [row[q] for row in rows]
        correct = answers.count("no")
        categories = ("yes", "no", "unsure")
    total = len(rows)
    missing = answers.count("") - unresolved
    pct = 100 * correct / total if total else None
    meets = correct * 5 >= 4 * total if total else None
    summary = {"n": total, "correct_n": correct, "correct_pct": pct, "wilson_95_pct": wilson(correct, total),
               "response_counts": {**{key: answers.count(key) for key in categories}, "missing": missing},
               "threshold_pct": 80, "estimate_meets_threshold": meets,
               "status": gate(round_number, planned_size and missing == 0 and unresolved == 0, meets)}
    if q in OPEN:
        summary["rater_agreement"] = {**rater_agreement(ratings), "unresolved_n": unresolved}
    return summary


def step_summary(rows, step):
    statuses = [(row[f"{step}_status"] or "missing", row) for row in rows]
    attempted = [row for status, row in statuses if status in ("unaided", "assisted", "not_completed")]
    times = [float(row[f"{step}_seconds"].replace(",", ".")) for row in attempted if row[f"{step}_seconds"]]
    errors = [int(row[f"{step}_errors"]) for row in attempted if row[f"{step}_errors"]]
    return {"statuses": {key: sum(status == key for status, _ in statuses) for key in STATUSES},
            "attempted_n": len(attempted), "timed_n": len(times), "median_seconds": median(times) if times else None,
            "errors_recorded_n": len(errors), "errors_total": sum(errors) if errors else None,
            "participants_with_errors_n": sum(value > 0 for value in errors)}


def round_summary(rows, round_number):
    minimum, maximum = (8, 12) if round_number == 1 else (2, 3)
    planned_size = minimum <= len(rows) <= maximum
    sus = sus_summary(rows, round_number, planned_size)
    questions = {q: question_summary(rows, q, round_number, planned_size) for q in QUESTIONS}
    gates = {sus["status"], *(result["status"] for result in questions.values())}
    overall = next(value for value in ("not_met", "provisional", "descriptive", "met") if value in gates)
    return {"n": len(rows), "language_counts": {language: sum(row["language"] == language for row in rows)
                                                for language in ("ru", "kk")},
            "planned_n_range": [minimum, maximum], "sample_size_within_plan": planned_size,
            "threshold_status": overall, "sus": sus, "questions": questions,
            "steps": {step: step_summary(rows, step) for step in STEPS}}


def analyze(rows):
    return {f"round_{round_number}": round_summary([row for row in rows if row["round"] == str(round_number)], round_number)
            for round_number in (1, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(read_rows(args.csv_file))
    except (OSError, UnicodeError, csv.Error):
        print("Cannot read UTF-8 study CSV. Check the path and CSV format.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
