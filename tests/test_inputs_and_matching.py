"""Input checks, unit invariance of the elastic mean, and matching with ties."""

import numpy as np
import pytest

from shape_preserving_effects import (
    elastic_mean,
    fate_drfos,
    local_effect,
    logistic_propensity,
    onset_time,
    pointwise_mean,
    populational_effect,
    srvf_kernel_ate,
)
from shape_preserving_effects.datasets import drop_rebound_trial
from shape_preserving_effects.estimators import _matches

t = np.linspace(0.0, 1.0, 41)
TRIAL = drop_rebound_trial(n=60, t=t, seed=3)


@pytest.mark.parametrize("clip", [-0.1, 0.5, 0.6])
def test_invalid_clip_is_rejected(clip: float) -> None:
    Y, A, X = TRIAL.Y, TRIAL.A, TRIAL.X
    for call in (
        lambda: logistic_propensity(X, A, clip=clip),
        lambda: populational_effect(t, Y, A, X, clip=clip),
        lambda: fate_drfos(t, Y, A, X, clip=clip),
    ):
        with pytest.raises(ValueError, match="clip"):
            call()


def test_propensity_of_zero_without_clipping_is_rejected() -> None:
    e = np.full(len(TRIAL.A), 0.5)
    e[0] = 0.0
    with pytest.raises(ValueError, match="infinite weights"):
        populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, clip=0.0, propensity=e)


def test_given_propensity_is_checked() -> None:
    e = np.full(len(TRIAL.A), 0.5)
    with pytest.raises(ValueError, match="one entry per unit"):
        populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, propensity=e[:-1])
    e[3] = 1.2
    with pytest.raises(ValueError, match="probabilities"):
        populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, propensity=e)


def test_competitors_check_their_inputs() -> None:
    with pytest.raises(ValueError, match="0 / 1"):
        fate_drfos(t, TRIAL.Y, 2 * TRIAL.A, TRIAL.X)
    with pytest.raises(ValueError, match="0 / 1"):
        srvf_kernel_ate(t, TRIAL.Y, 2 * TRIAL.A, TRIAL.X)
    with pytest.raises(ValueError, match="points"):
        fate_drfos(np.linspace(0.0, 1.0, 81), TRIAL.Y, TRIAL.A, TRIAL.X)
    Y = TRIAL.Y.copy()
    Y[2, 7] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fate_drfos(t, Y, TRIAL.A, TRIAL.X)


@pytest.mark.parametrize(("kwargs", "name"), [({"k": 0}, "k"), ({"max_pairs": 0}, "max_pairs")])
def test_local_effect_names_the_invalid_argument(kwargs: dict, name: str) -> None:
    with pytest.raises(ValueError, match=name):
        local_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, **kwargs)


def test_pointwise_mean_rejects_negative_weights() -> None:
    curves = np.exp(-0.5 * ((t[None, :] - np.array([[0.4], [0.5], [0.6]])) / 0.08) ** 2)
    with pytest.raises(ValueError, match="non-negative"):
        pointwise_mean(curves, weights=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="positive sum"):
        pointwise_mean(curves, weights=[0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# Unit of the curves
# ----------------------------------------------------------------------


def _bumps(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres, heights = rng.uniform(0.3, 0.6, n), rng.uniform(0.8, 1.2, n)
    return heights[:, None] * np.exp(-0.5 * ((t[None, :] - centres[:, None]) / 0.08) ** 2)


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1e-3, 1e3])
def test_elastic_mean_does_not_depend_on_the_unit_of_the_curves(scale: float) -> None:
    curves = _bumps(12, seed=0)
    reference = elastic_mean(curves, t, n_iter=5).mean
    rescaled = elastic_mean(scale * curves, t, n_iter=5).mean / scale
    np.testing.assert_allclose(rescaled, reference, rtol=0, atol=1e-10)


def test_onset_time_is_undefined_for_a_curve_that_never_drops() -> None:
    assert np.isnan(onset_time(np.exp(-0.5 * ((t - 0.4) / 0.1) ** 2), t))


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def test_matches_keep_every_unit_tied_at_the_kth_distance() -> None:
    treated = np.array([[0.0], [1.0]])
    controls = np.array([[0.0], [0.0], [0.0], [1.0], [5.0]])
    rows_a, rows_b, weights = _matches(treated, controls, k=2, scale=np.ones(1))
    # unit 0: three controls tied at distance 0, sharing weight k = 2
    np.testing.assert_array_equal(rows_b[rows_a == 0], [0, 1, 2])
    np.testing.assert_allclose(weights[rows_a == 0], [2 / 3] * 3)
    # unit 1: control 3 at distance 0, then three controls tied at distance 1
    np.testing.assert_array_equal(rows_b[rows_a == 1], [3, 0, 1, 2])
    np.testing.assert_allclose(weights[rows_a == 1], [0.5] * 4)


def test_local_effect_does_not_depend_on_the_row_order_with_tied_covariates() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, len(TRIAL.A)).astype(float)  # binary covariate: many ties
    Y, A = TRIAL.Y, TRIAL.A
    order = rng.permutation(len(A))
    a = local_effect(t, Y, A, X, k=2, n_iter=2).effect
    b = local_effect(t, Y[order], A[order], X[order], k=2, n_iter=2).effect
    np.testing.assert_allclose(a, b, atol=1e-8)


def test_local_effect_without_ties_matches_k_neighbours_with_equal_weights() -> None:
    res = local_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, k=2, n_iter=2).mean_result
    assert len(res.weights) == 2 * len(TRIAL.A)
    np.testing.assert_array_equal(res.weights, np.full(len(res.weights), res.weights[0]))


def test_matching_scales_the_covariates_on_the_whole_sample() -> None:
    # Treated unit 0 is close to control 0 on both covariates. Control 1 has
    # the same second covariate as treated unit 0 but a far first covariate.
    # The second covariate barely varies among the controls and varies a lot
    # in the whole sample (treated unit 1): scaled by its standard deviation
    # among the controls it would dominate the distance and select control 1;
    # scaled on the whole sample, control 0 is the nearest neighbour.
    X = np.array([[1.0, 0.1], [5.0, 100.0], [0.0, 0.0], [10.0, 0.1]])
    A = np.array([1, 1, 0, 0])
    Y = np.array([k + np.sin(np.pi * t) for k in (1.0, 2.0, 10.0, 20.0)])
    att = local_effect(t, Y, A, X, k=1, estimand="ATT", n_iter=1)
    np.testing.assert_allclose(att.differences[0], Y[0] - Y[2])


# ----------------------------------------------------------------------
# DR-FoS: cross-fitted nuisance models
# ----------------------------------------------------------------------


def test_drfos_recovers_an_exactly_linear_model() -> None:
    # Outcomes linear in X at every time point, without noise: the
    # cross-fitted outcome regressions are exact, and so is the estimate.
    rng = np.random.default_rng(2)
    n = 80
    X = rng.uniform(size=n)
    A = rng.binomial(1, 0.3 + 0.4 * X)
    Y = (1.0 + X)[:, None] * np.sin(np.pi * t)[None, :] + A[:, None] * (0.5 * t)[None, :]
    effect, arm1, arm0 = fate_drfos(t, Y, A, X, rng=0, return_arms=True)
    np.testing.assert_allclose(effect, 0.5 * t, atol=1e-12)
    np.testing.assert_allclose(arm1 - arm0, effect)


def test_drfos_needs_at_least_two_folds() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        fate_drfos(t, TRIAL.Y, TRIAL.A, TRIAL.X, n_folds=1)


def test_drfos_rejects_folds_too_small_for_the_outcome_model() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(len(TRIAL.A), 40))  # 60 units, 41 coefficients per arm
    with pytest.raises(ValueError, match="too few units of the arm"):
        fate_drfos(t, TRIAL.Y, TRIAL.A, X)
