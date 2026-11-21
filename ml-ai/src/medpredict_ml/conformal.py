"""Split-conformal prediction intervals for a frozen regression model.

Calibration residuals come from data never used to fit or select the model. The finite-sample
order statistic k = ceil((n+1)(1-alpha)) gives marginal coverage >= 1-alpha under exchangeability
between calibration and new cases. Coverage under distribution shift must be measured, not assumed.
"""
from __future__ import annotations

import math

import numpy as np


def _order_statistic(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, float).reshape(-1))
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a non-empty finite array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    if k > len(values):
        return math.inf
    return float(values[k - 1])


def symmetric_offsets(y: np.ndarray, prediction: np.ndarray, alpha: float) -> tuple[float, float]:
    """Absolute-residual score: interval [prediction - q, prediction + q]."""
    q = _order_statistic(np.abs(np.asarray(y, float) - np.asarray(prediction, float)), alpha)
    return -q, q


def asymmetric_offsets(y: np.ndarray, prediction: np.ndarray, alpha: float) -> tuple[float, float]:
    """Signed residual r = y - prediction, alpha/2 per tail: [prediction + lower, prediction + upper]."""
    residual = np.asarray(y, float) - np.asarray(prediction, float)
    upper = _order_statistic(residual, alpha / 2)
    lower = -_order_statistic(-residual, alpha / 2)
    return lower, upper


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> list[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray | None = None) -> dict:
    y, lower, upper = [np.asarray(v, float).reshape(-1) for v in (y, lower, upper)]
    inside = (y >= lower) & (y <= upper)
    result = {"n": int(len(y)), "covered": int(inside.sum()), "coverage": float(inside.mean()),
              "coverage_95_wilson": wilson_interval(int(inside.sum()), len(y)),
              "mean_width": float(np.mean(upper - lower)),
              "below_lower": float(np.mean(y < lower)), "above_upper": float(np.mean(y > upper))}
    if weight is not None:
        weight = np.asarray(weight, float).reshape(-1)
        result["weighted_coverage"] = float(np.average(inside, weights=weight))
    return result
