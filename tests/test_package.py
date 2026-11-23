"""Unit tests of the shape_preserving_effects package.

Install the package first (``pip install -e .`` from the repository root),
then run ``pytest`` from the repository root. The whole file runs in about
a minute.
"""

import numpy as np
import pytest

from shape_preserving_effects import (
    depth,
    drop_rebound_dgp,
    drop_rebound_trial,
    drop_speed,
    elastic_mean,
    fate_drfos,
    from_srsf,
    karcher_mean_function,
    l2_error,
    local_effect,
    logistic_propensity,
    main_dgp,
    mean_curve,
    onset_time,
    pointwise_mean,
    population_estimands,
    populational_effect,
    register,
    srvf_kernel_ate,
    to_srsf,
    warp_curve,
    warp_srsf,
)
from shape_preserving_effects.mean import _sphere_mean
from shape_preserving_effects.srsf import psi_to_warp, warp_to_psi

t = np.linspace(0.0, 1.0, 81)


def bump(c: float, w: float = 0.08) -> np.ndarray:
    return np.exp(-0.5 * ((t - c) / w) ** 2)


# ----------------------------------------------------------------------
# SRSF and registration
# ----------------------------------------------------------------------


def test_srsf_round_trip() -> None:
    f = np.sin(2 * np.pi * t) + 0.5 * t
    f_rec = from_srsf(to_srsf(f, t), t, f0=f[0])
    assert np.max(np.abs(f - f_rec)) < 2e-2


def test_registration_recovers_a_known_warp() -> None:
    f1 = bump(0.45) + 0.4 * bump(0.75)
    gam0 = t + 0.12 * np.sin(np.pi * t)
    f2 = warp_curve(f1, t, gam0)
    q1, q2 = to_srsf(f1, t), to_srsf(f2, t)
    gam_hat, d_aligned = register(q2, q1, t)
    d_unaligned = np.sqrt(np.trapezoid((q1 - q2) ** 2, t))
    assert d_aligned < 0.15 * d_unaligned
    assert np.max(np.abs(gam_hat - gam0)) < 0.03


# ----------------------------------------------------------------------
# Elastic mean
# ----------------------------------------------------------------------

F_SHIFTED = np.array([bump(c) for c in np.linspace(0.30, 0.60, 9)])


def test_elastic_mean_keeps_the_shape_pointwise_mean_flattens_it() -> None:
    em = elastic_mean(F_SHIFTED, t).mean
    pm = pointwise_mean(F_SHIFTED)
    assert em.max() > 0.90
    assert pm.max() < 0.75
    assert abs(t[np.argmax(em)] - 0.45) < 0.05  # typical timing


def test_weights_steer_the_timing() -> None:
    centres = np.linspace(0.30, 0.60, 9)
    w = np.where(centres < 0.45, 1.0, 0.0) + 1e-9
    emw = elastic_mean(F_SHIFTED, t, weights=w).mean
    assert t[np.argmax(emw)] < 0.42


def test_mean_is_the_weighted_average_of_the_registered_curves() -> None:
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 1.0, len(F_SHIFTED))
    res = elastic_mean(F_SHIFTED, t, weights=w)
    assert np.allclose(res.weights.sum(), 1.0)
    assert np.allclose(res.mean, res.weights @ res.registered)
    # every registered curve is the curve re-timed by gamma_i o gamma_bar^{-1}
    ginv = np.interp(t, res.mean_warp, t)
    i = 3
    g_i = warp_curve(F_SHIFTED[i], t, np.interp(ginv, t, res.warps[i]))
    assert np.allclose(res.registered[i], g_i)


def test_mean_keeps_the_mean_end_values() -> None:
    rng = np.random.default_rng(6)
    F = np.array(
        [
            0.3 * k + bump(c) + 0.8 * t * k
            for k, c in zip(rng.normal(size=9), np.linspace(0.30, 0.60, 9))
        ]
    )
    w = rng.uniform(0.1, 1.0, 9)
    for weights in (None, w):
        res = elastic_mean(F, t, weights=weights)
        ww = np.ones(9) / 9 if weights is None else w / w.sum()
        assert abs(res.mean[0] - ww @ F[:, 0]) < 1e-12
        assert abs(res.mean[-1] - ww @ F[:, -1]) < 1e-12


def test_template_model_keeps_the_mean_amplitude() -> None:
    # Z = c + s * g o gamma: the mean is E[c] + E[s] g o gamma_bar^{-1}, so its
    # amplitude is E[s] (the Karcher mean function gives (E[sqrt s])^2).
    s = np.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.6])
    c = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.05])
    F = np.array([ci + si * bump(ce) for ci, si, ce in zip(c, s, np.linspace(0.35, 0.60, 6))])
    em = elastic_mean(F, t, n_iter=5).mean
    amp = em.max() - em[0]
    assert abs(amp - s.mean() * (1 - bump(0.5)[0])) < 0.03
    assert abs(amp - np.mean(np.sqrt(s)) ** 2) > 0.1


def test_karcher_mean_function_shrinks_the_amplitude() -> None:
    # same samples as above: the Karcher mean function has amplitude
    # (E[sqrt s])^2 and starts at the mean initial value
    s = np.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.6])
    c = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.05])
    F = np.array([ci + si * bump(ce) for ci, si, ce in zip(c, s, np.linspace(0.35, 0.60, 6))])
    res = elastic_mean(F, t, n_iter=5)
    km = res.karcher_mean_function
    assert abs(km[0] - F[:, 0].mean()) < 1e-12
    amp = km.max() - km[0]
    assert abs(amp - np.mean(np.sqrt(s)) ** 2 * (1 - bump(0.5)[0])) < 0.04
    assert np.allclose(km, karcher_mean_function(F, t, n_iter=5))
    assert np.allclose(mean_curve(res, "elastic"), res.mean)


def test_estimators_accept_both_means() -> None:
    tr = drop_rebound_trial(n=60, t=t, seed=4)
    pe = populational_effect(t, tr.Y, tr.A, tr.X, n_iter=1)
    pk = populational_effect(t, tr.Y, tr.A, tr.X, n_iter=1, mean="karcher")
    assert np.allclose(pe.effect, pe.arm1.mean - pe.arm0.mean)
    assert np.allclose(pk.effect, pk.arm1.karcher_mean_function - pk.arm0.karcher_mean_function)
    lk = local_effect(t, tr.Y, tr.A, tr.X, k=1, n_iter=1, mean="karcher")
    assert np.allclose(lk.effect, lk.mean_result.karcher_mean_function)
    with pytest.raises(ValueError, match="mean"):
        populational_effect(t, tr.Y, tr.A, tr.X, mean="other")


def test_steps_keep_their_height() -> None:
    steps = np.array([1.0 / (1 + np.exp(-(t - c) / 0.01)) for c in (0.3, 0.45, 0.6, 0.7)])
    em = elastic_mean(steps, t, n_iter=5).mean
    assert abs(em[-1] - 1.0) < 1e-9 and abs(em.max() - 1.0) < 1e-6


def test_warp_action_conserves_total_variation() -> None:
    f = 1.0 / (1 + np.exp(-(t - 0.5) / 0.01))  # a 1-2 point jump
    q = to_srsf(f, t)
    assert np.allclose(warp_srsf(q, t, t, f=f), q)  # identity warp
    for gam in (t**2, t**0.5, t + 0.2 * np.sin(np.pi * t) / np.pi):
        qw = warp_srsf(q, t, gam, f=f)
        assert abs(np.trapezoid(qw**2, t) - np.trapezoid(q**2, t)) < 1e-3
        assert np.allclose(qw, to_srsf(warp_curve(f, t, gam), t))


def test_phase_centering_leaves_an_identity_mean_warp() -> None:
    res = elastic_mean(F_SHIFTED, t)
    warps = np.array([register(res.q_mean, to_srsf(f, t), t)[0] for f in F_SHIFTED])
    psis = np.array([warp_to_psi(g, t) for g in warps])
    gam_bar = psi_to_warp(_sphere_mean(psis, np.ones(len(warps)), t), t)
    assert np.max(np.abs(gam_bar - t)) < 0.04


# ----------------------------------------------------------------------
# Shape metrics
# ----------------------------------------------------------------------


def test_shape_metrics_on_a_rebound_and_a_plateau() -> None:
    tau = -bump(0.35)
    tau_pl = -1.0 / (1 + np.exp(-(t - 0.35) / 0.03))
    half_cross = 0.35 - 0.08 * np.sqrt(2 * np.log(2))
    assert depth(tau) < -0.9 and depth(tau_pl) < -0.9
    assert abs(onset_time(tau, t) - half_cross) < 0.02
    assert abs(onset_time(tau_pl, t) - 0.35) < 0.02
    assert abs(drop_speed(tau, t) - np.exp(-0.5) / 0.08) < 0.4
    assert abs(drop_speed(tau_pl, t) - 1 / (4 * 0.03)) < 1.0


# ----------------------------------------------------------------------
# Propensity score
# ----------------------------------------------------------------------


def test_logistic_propensity() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, 4000)
    e_true = 1 / (1 + np.exp(-(-1.1 + 2.2 * X)))
    A = rng.binomial(1, e_true)
    e_hat = logistic_propensity(X, A)
    assert np.mean(np.abs(e_hat - e_true)) < 0.03


# ----------------------------------------------------------------------
# Simulation model, estimands and estimators
# ----------------------------------------------------------------------

E = population_estimands(main_dgp, t, n_pop=200, seed=1, n_iter=2)
TRIAL = drop_rebound_trial(n=200, t=t, seed=3)


def test_rebound_share() -> None:
    rng = np.random.default_rng(11)
    Y1, _ = drop_rebound_dgp(1.0, p_rebound=0.25).potential(t, rng.uniform(0, 1, 400), rng)
    assert abs(np.mean(np.abs(Y1[:, -1]) < 0.45) - 0.25) < 0.08
    rng = np.random.default_rng(12)
    Y1, _ = main_dgp.potential(t, rng.uniform(0, 1, 200), rng)
    assert np.mean(np.abs(Y1[:, -1]) < 0.45) > 0.97


def test_tau_pop_keeps_the_typical_shape_and_the_fate_flattens_it() -> None:
    assert abs(E["func_tau_pop"]["depth"] - E["ref"]["depth"]) < 0.20
    assert E["func_fate"]["depth"] > 0.65 * E["func_tau_pop"]["depth"]
    assert l2_error(E["tau_ind"], E["tau_loc"], t) < 0.15


def test_populational_estimator_removes_confounding() -> None:
    pop = populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, n_iter=2)
    m1 = elastic_mean(TRIAL.Y[TRIAL.A == 1], t, n_iter=2).mean
    m0 = elastic_mean(TRIAL.Y[TRIAL.A == 0], t, n_iter=2).mean
    err_ipw = l2_error(pop.effect, E["tau_pop"], t)
    assert err_ipw < 0.15
    assert err_ipw < l2_error(m1 - m0, E["tau_pop"], t)


def test_local_estimator() -> None:
    loc = local_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, k=3, n_iter=2, max_pairs=250, rng=1)
    assert l2_error(loc.effect, E["tau_pop"], t) < 0.20


def test_att_weights_and_given_propensity() -> None:
    """ATT: treated weight 1, controls e / (1 - e); scores given by the user
    are used as they are (after clipping)."""
    e = logistic_propensity(TRIAL.X, TRIAL.A)
    pop = populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, n_iter=2, estimand="ATT", propensity=e)
    A = TRIAL.A == 1
    assert np.allclose(pop.weights[A], 1.0)
    assert np.allclose(pop.weights[~A], e[~A] / (1 - e[~A]))
    m1 = elastic_mean(TRIAL.Y[A], t, n_iter=2).mean
    m0 = elastic_mean(TRIAL.Y[~A], t, weights=e[~A] / (1 - e[~A]), n_iter=2).mean
    assert np.allclose(pop.effect, m1 - m0)
    ate = populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, n_iter=2, propensity=e)
    ref = populational_effect(t, TRIAL.Y, TRIAL.A, TRIAL.X, n_iter=2)
    assert np.allclose(ate.effect, ref.effect)


def test_zero_weights_have_no_effect() -> None:
    F = np.array([bump(0.40), bump(0.50, 0.10), 1.5 * bump(0.55)])
    G = np.vstack([F, -bump(0.2, 0.05)])
    a = elastic_mean(F, t, weights=[1.0, 2.0, 1.0], n_iter=2).mean
    b = elastic_mean(G, t, weights=[1.0, 2.0, 1.0, 0.0], n_iter=2).mean
    assert np.allclose(a, b)


def test_invalid_inputs_raise_clear_errors() -> None:
    F = np.array([bump(0.4), bump(0.6)])
    Y, A, X = TRIAL.Y, TRIAL.A, TRIAL.X
    with pytest.raises(ValueError, match="uniform"):
        elastic_mean(F, t**2)
    with pytest.raises(ValueError, match="strictly increasing"):
        elastic_mean(F, t[::-1])
    t_nan = t.copy()
    t_nan[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        elastic_mean(F, t_nan)
    with pytest.raises(ValueError, match="at least 3"):
        elastic_mean(F[:, :2], t[:2])
    with pytest.raises(ValueError, match="points"):
        elastic_mean(F[:, :-1], t)
    with pytest.raises(ValueError, match="2-d"):
        elastic_mean(F[0], t)
    G = F.copy()
    G[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        elastic_mean(G, t)
    with pytest.raises(ValueError, match="0 / 1"):
        populational_effect(t, Y, A + 1, X)
    with pytest.raises(ValueError, match="no control unit"):
        populational_effect(t, Y, np.ones(len(A)), X)
    with pytest.raises(ValueError, match="no treated unit"):
        populational_effect(t, Y, np.zeros(len(A)), X)
    with pytest.raises(ValueError, match="one row per curve"):
        populational_effect(t, Y, A, X[:-1])
    with pytest.raises(ValueError, match="estimand"):
        populational_effect(t, Y, A, X, estimand="ATX")
    with pytest.raises(ValueError, match="exceeds"):
        local_effect(t, Y, A, X, k=10**6)
    # a grid in days, not starting at 0, gives the same mean as [0, 1]
    days = 30.0 + 700.0 * t
    assert np.allclose(
        elastic_mean(F, days, n_iter=2).mean, elastic_mean(F, t, n_iter=2).mean, atol=1e-10
    )


def test_drfos_targets_the_fate() -> None:
    effect = fate_drfos(t, TRIAL.Y, TRIAL.A, TRIAL.X, rng=0)
    assert l2_error(effect, E["fate"], t) < l2_error(effect, E["tau_pop"], t)


def test_effects_do_not_change_when_a_constant_is_added_to_every_curve() -> None:
    Y, A, X = TRIAL.Y[:80], TRIAL.A[:80], TRIAL.X[:80]
    for estimator in (
        lambda Y: fate_drfos(t, Y, A, X, rng=0),
        lambda Y: populational_effect(t, Y, A, X, n_iter=2).effect,
    ):
        assert np.allclose(estimator(Y + 5.0), estimator(Y), atol=1e-8)


def test_srvf_kernel_ate_returns_the_two_arm_curves() -> None:
    effect, arm1, arm0 = srvf_kernel_ate(
        t, TRIAL.Y, TRIAL.A, TRIAL.X, n_iter=2, rng=0, return_arms=True
    )
    assert effect.shape == arm1.shape == arm0.shape == t.shape
    assert np.allclose(effect, arm1 - arm0)
