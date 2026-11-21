import numpy as np
import pytest

from medpredict_ml.explain import mahalanobis_squared, shapley_values


def test_shapley_of_affine_model_is_weight_times_difference():
    weights = np.array([1., 2., 3., 4., 5.])
    predict = lambda rows: rows @ weights + 100
    x = np.array([[50., 1., 30., 6., 7.]])
    reference = np.array([40., .5, 25., 5., 6.])
    phi, base = shapley_values(predict, x, reference)
    np.testing.assert_allclose(phi[0], weights * (x[0] - reference), atol=1e-4)
    assert base == pytest.approx(reference @ weights + 100, abs=1e-3)


def test_shapley_is_additive_and_symmetric_for_interactions():
    predict = lambda rows: rows[:, 0] * rows[:, 2] + np.maximum(rows[:, 4] - 5, 0) ** 2
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 9, size=(20, 5))
    reference = np.full(5, 4.)
    phi, base = shapley_values(predict, x, reference)
    np.testing.assert_allclose(phi.sum(axis=1), predict(x.astype(np.float32)) - base, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(phi[:, [1, 3]], 0, atol=1e-6)          # dummy features get zero
    product = x[:, 0] * x[:, 2] - 16
    np.testing.assert_allclose(phi[:, 0] + phi[:, 2], product, rtol=1e-4, atol=1e-3)


def test_mahalanobis_matches_euclidean_for_identity_precision():
    x = np.array([[1., 2., 3., 4., 5.]])
    assert mahalanobis_squared(x, np.zeros(5), np.eye(5))[0] == pytest.approx(55.)
