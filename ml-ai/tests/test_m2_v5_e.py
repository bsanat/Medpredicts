from pathlib import Path

import pytest


@pytest.mark.parametrize("candidate,group,mae,bias", [
    ("main", "bpq020_no_ge140", 21, -19),
    ("safe", "medication_no_ge150", 19, -21),
])
def test_risk_blocker_applies_to_every_candidate(monkeypatch, candidate, group, mae, bias):
    pytest.importorskip("catboost")
    pytest.importorskip("tabm")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "research"))
    from m2_v5_e import PROB_REGISTERED, PROB_UI, gate

    def point(mae, bias=0):
        return {"n": 50, "weighted_mae": mae, "weighted_mean_signed_error": bias}

    paired = {
        "overall": {"models": {candidate: point(10), "v2": point(10)}},
        "sbp_ge150": {"models": {candidate: point(25), "v2": point(40)}},
        group: {"models": {candidate: point(mae, bias), f"{candidate}_base_5f": point(20, -20)}},
    }
    coverage = {label: {"coverage": .9} for label in ("ge150", "overall")}
    probability = {PROB_REGISTERED: {"roc_auc": .8}, PROB_UI: {"roc_auc": .7}}
    result = gate(candidate, paired, coverage, probability)
    assert all(result["criteria"].values())
    assert result["blocking_groups"] == [group]
    assert not result["passes_all_and_unblocked"]

    paired[group]["models"][candidate] = point(20, -20)
    assert gate(candidate, paired, coverage, probability)["passes_all_and_unblocked"]
