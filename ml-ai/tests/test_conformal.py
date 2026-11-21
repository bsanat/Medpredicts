import math

import numpy as np
import pytest

from medpredict_ml.conformal import asymmetric_offsets, coverage, symmetric_offsets, wilson_interval


def test_symmetric_offset_uses_finite_sample_order_statistic():
    y = np.arange(1., 20.)          # absolute residuals 1..19 against zero predictions
    low, high = symmetric_offsets(y, np.zeros_like(y), alpha=.1)
    # n=19, k=ceil(20*0.9)=18 -> 18th smallest residual
    assert (low, high) == (-18., 18.)


def test_too_few_calibration_points_give_infinite_interval():
    low, high = symmetric_offsets(np.array([1., 2., 3.]), np.zeros(3), alpha=.1)
    assert math.isinf(high) and math.isinf(low)


def test_asymmetric_offsets_follow_residual_skew():
    rng = np.random.default_rng(0)
    residual = rng.exponential(10., 2000) - 5.   # long upper tail, like underestimated high SBP
    low, high = asymmetric_offsets(residual, np.zeros_like(residual), alpha=.1)
    assert high > -low > 0


def test_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    covered = []
    for _ in range(200):
        calibration = rng.normal(0, 5, 300)
        test = rng.normal(0, 5, 300)
        low, high = symmetric_offsets(calibration, np.zeros_like(calibration), alpha=.1)
        covered.append(coverage(test, np.full(300, low), np.full(300, high))["coverage"])
    assert np.mean(covered) == pytest.approx(.9, abs=.01)


def test_coverage_report_and_wilson_interval():
    report = coverage(np.array([0., 5., 10.]), np.array([-1., -1., -1.]), np.array([6., 6., 6.]), np.array([1., 1., 2.]))
    assert report["covered"] == 2 and report["coverage"] == pytest.approx(2 / 3)
    assert report["weighted_coverage"] == pytest.approx(.5)
    assert report["above_upper"] == pytest.approx(1 / 3)
    lower, upper = wilson_interval(90, 100)
    assert lower < .9 < upper and 0 <= lower and upper <= 1


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        symmetric_offsets(np.ones(10), np.zeros(10), alpha=1.5)
