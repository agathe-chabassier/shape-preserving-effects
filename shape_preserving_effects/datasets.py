"""Simulated observational studies with functional outcomes.

The simulation model of the paper ("The simulation model" in the Simulations
section). Time since treatment is rescaled to [0, 1]. A single confounder
X ~ Unif[0, 1] drives the treatment, the control trajectory, and the depth
and timing of the treated response:

    A | X ~ Bernoulli(e(X)),   e(X) = expit(-1.1 + 2.2 X),
    c = 0.40 + alpha * (0.34 (X - 1/2) + 0.08 U_1)       (nadir time)
    h = 0.90 + 0.55 X + 0.06 U_2                         (depth)
    Y^(1)(t) = -h phi_{c, 0.10}(t) + eps(t)                       if B = 1
             = -h expit((t - c) / 0.03) (1 + 0.05 (t - c)_+) + eps(t)  if B = 0
    Y^(0)(t) = -(0.05 + 0.12 X) t^{3/2} + eps(t)

where phi_{c, w} is a Gaussian bump, U_1, U_2 ~ N(0, 1), B ~
Bernoulli(p_rebound) selects a rebounding (B = 1) or plateauing (B = 0)
treated response, and eps is a smooth mean-zero noise curve of scale
``noise_sd`` (three lowest Fourier frequencies, each of the six coefficients
drawn from N(0, noise_sd^2 / 3); eps(0) = eps(1) = 0).

Both potential outcomes of a unit are generated from the same latent
variables (U_1, U_2, B); only the noise differs. The individual effect
Y^(1) - Y^(0) is therefore well defined, which makes the individual
estimand tau_ind computable in simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from shape_preserving_effects.estimators import _expit
from shape_preserving_effects.mean import (
    MEANS,
    MeanName,
    elastic_mean,
    mean_curve,
    pointwise_mean,
)
from shape_preserving_effects.metrics import METRICS, shape_metrics

# ----------------------------------------------------------------------
# Parameters of the simulation model of the paper
# ----------------------------------------------------------------------

# Propensity score e(X) = expit(PROPENSITY_INTERCEPT + PROPENSITY_SLOPE X).
PROPENSITY_INTERCEPT = -1.1
PROPENSITY_SLOPE = 2.2
# Nadir time c = NADIR_BASE + alpha (PHASE_X (X - 1/2) + PHASE_SD U_1).
NADIR_BASE = 0.40  # nadir time at X = 1/2, U_1 = 0
PHASE_X = 0.34  # covariate-driven part of the nadir time
PHASE_SD = 0.08  # idiosyncratic part of the nadir time
PHASE_MAIN = 1.0  # phase-variation scale alpha of the five scenarios
# Depth h = DEPTH_BASE + DEPTH_X X + DEPTH_SD U_2.
DEPTH_BASE = 0.90
DEPTH_X = 0.55
DEPTH_SD = 0.06
DIP_WIDTH = 0.10  # width of the Gaussian dip of rebounding units
PLATEAU_SCALE = 0.03  # time scale of the plateauing units' decline
PLATEAU_SLOPE = 0.05  # slow drift of plateauing units after the drop
# Control trajectory Y^(0)(t) = -(CONTROL_BASE + CONTROL_X X) t^CONTROL_POWER.
CONTROL_BASE = 0.05
CONTROL_X = 0.12
CONTROL_POWER = 1.5
NOISE_SD = 0.03  # noise level of every simulation of the paper

# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def _bump(
    t: np.ndarray,
    centre: float | np.ndarray,
    height: float | np.ndarray,
    width: float,
) -> np.ndarray:
    """Gaussian bump(s) on the grid ``t``.

    Args:
        t: Grid of length m.
        centre: Centre of the bump, or the (n,) centres of n bumps.
        height: Height of the bump, or the (n,) heights of n bumps.
        width: Standard deviation of the Gaussian.

    Returns:
        The (m,) bump, or the (n, m) array of bumps, one per row.
    """
    t = np.asarray(t)
    centre = np.atleast_1d(np.asarray(centre, dtype=float))[:, None]
    height = np.atleast_1d(np.asarray(height, dtype=float))[:, None]
    out = height * np.exp(-0.5 * ((t[None, :] - centre) / width) ** 2)
    return out[0] if out.shape[0] == 1 else out


def _smooth_noise(n: int, t: np.ndarray, rng: np.random.Generator, sd: float) -> np.ndarray:
    """Smooth mean-zero noise curves: random Fourier terms of frequencies 1 to 3.

    eps(t) = sum_f a_f sin(2 pi f t) + b_f (cos(2 pi f t) - 1), with
    a_f, b_f ~ N(0, sd^2 / 3), so that eps(0) = eps(1) = 0.

    Args:
        n: Number of curves.
        t: Grid of length m, in [0, 1].
        rng: Random generator.
        sd: Scale of the noise.

    Returns:
        The (n, m) array of noise curves.
    """
    freqs = np.arange(1, 4)
    coefs = rng.normal(0.0, sd / np.sqrt(len(freqs)), size=(n, len(freqs), 2))
    out = np.zeros((n, t.size))
    for j, f in enumerate(freqs):
        out += coefs[:, j, 0][:, None] * np.sin(2 * np.pi * f * t)[None, :]
        out += coefs[:, j, 1][:, None] * (np.cos(2 * np.pi * f * t) - 1.0)[None, :]
    return out


# ----------------------------------------------------------------------
# Data-generating process and simulated study
# ----------------------------------------------------------------------


@dataclass
class DGP:
    """A data-generating process.

    Attributes:
        name: Short description.
        potential: ``potential(t, X, rng) -> (Y1, Y0)``: both potential
            outcomes of every unit, as two (n, m) arrays.
        propensity: ``propensity(X) -> P(A = 1 | X)``.
    """

    name: str
    potential: Callable[
        [np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
    ]
    propensity: Callable[[np.ndarray], np.ndarray]


@dataclass
class Trial:
    """One simulated observational study.

    Attributes:
        t: (m,) time grid.
        X: (n,) confounder.
        A: (n,) treatment indicator coded 0 / 1.
        Y: (n, m) observed curves.
        Y1: (n, m) potential outcomes under treatment (unobservable in
            practice, kept for plots and diagnostics).
        Y0: (n, m) potential outcomes under control (same remark).
    """

    t: np.ndarray
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Y1: np.ndarray | None = None
    Y0: np.ndarray | None = None


def sample_trial(dgp: DGP, n: int, t: np.ndarray, seed: int = 0) -> Trial:
    """Draw one observational study of ``n`` units from ``dgp``.

    Args:
        dgp: Data-generating process.
        n: Number of units.
        t: Time grid of length m.
        seed: Seed of the draw.

    Returns:
        The simulated study.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=n)
    Y1, Y0 = dgp.potential(t, X, rng)
    A = rng.binomial(1, dgp.propensity(X)).astype(int)
    Y = np.where(A[:, None] == 1, Y1, Y0)
    return Trial(t=t, X=X, A=A, Y=Y, Y1=Y1, Y0=Y0)


# ----------------------------------------------------------------------
# The simulation model of the paper
# ----------------------------------------------------------------------


def drop_rebound_dgp(
    phase: float = PHASE_MAIN, p_rebound: float = 1.0, noise_sd: float = NOISE_SD
) -> DGP:
    """The drop-and-rebound simulation model.

    Args:
        phase: Phase-variation scale alpha (multiplies both timing terms of
            the nadir time). Scenarios 1 to 5 of the paper use ``phase=1``.
        p_rebound: Share of treated responses that rebound to baseline; the
            others plateau. Scenarios 1 to 5 use 1, 0.75, 0.5, 0.25 and 0.
        noise_sd: Scale sigma_eps of the smooth noise (0.03 in the paper):
            each of the six Fourier coefficients is drawn from
            N(0, noise_sd^2 / 3).

    Returns:
        The data-generating process.
    """

    def potential(
        t: np.ndarray, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Both potential outcomes (Y1, Y0) of every unit, as two (n, m) arrays."""
        X = np.atleast_1d(np.asarray(X, float))
        n = X.size
        u_phase = rng.standard_normal(n)
        u_depth = rng.standard_normal(n)
        rebound = rng.uniform(size=n) < p_rebound
        c = NADIR_BASE + phase * (PHASE_X * (X - 0.5) + PHASE_SD * u_phase)
        h = DEPTH_BASE + DEPTH_X * X + DEPTH_SD * u_depth
        rebound_curves = -_bump(t, c, h, DIP_WIDTH)
        z = (t[None, :] - c[:, None]) / PLATEAU_SCALE
        plateau_curves = (
            -h[:, None]
            * _expit(z)
            * (1 + PLATEAU_SLOPE * np.clip(t[None, :] - c[:, None], 0, None))
        )
        Y1 = np.where(rebound[:, None], rebound_curves, plateau_curves) + _smooth_noise(
            n, t, rng, noise_sd
        )
        Y0 = -(CONTROL_BASE + CONTROL_X * X)[:, None] * (
            t[None, :] ** CONTROL_POWER
        ) + _smooth_noise(n, t, rng, noise_sd)
        return Y1, Y0

    return DGP(
        name=f"drop_rebound(phase={phase:g}, p_rebound={p_rebound:g}, noise_sd={noise_sd:g})",
        potential=potential,
        propensity=lambda X: _expit(PROPENSITY_INTERCEPT + PROPENSITY_SLOPE * np.asarray(X)),
    )


# Scenario 1 of the paper (every treated response rebounds).
main_dgp = drop_rebound_dgp(PHASE_MAIN, p_rebound=1.0)


def drop_rebound_trial(
    n: int, t: np.ndarray, seed: int = 0, phase: float = PHASE_MAIN, p_rebound: float = 1.0
) -> Trial:
    """Draw one study of ``n`` units from :func:`drop_rebound_dgp`.

    Args:
        n: Number of units.
        t: Time grid of length m, in [0, 1].
        seed: Seed of the draw.
        phase: Phase-variation scale alpha.
        p_rebound: Share of treated responses that rebound.

    Returns:
        The simulated study.
    """
    return sample_trial(drop_rebound_dgp(phase, p_rebound), n, t, seed=seed)


# ----------------------------------------------------------------------
# Causal estimands of a data-generating process
# ----------------------------------------------------------------------


class Estimands(TypedDict):
    """Output of :func:`population_estimands`.

    The curves are (m,) arrays; the dicts of shape metrics are keyed by
    :data:`~shape_preserving_effects.metrics.METRICS`.
    """

    tau_pop: np.ndarray  # populational effect mu(Y^(1)) - mu(Y^(0))
    tau_ind: np.ndarray  # individual effect, elastic mean of Y^(1)_i - Y^(0)_i
    tau_loc: np.ndarray  # local (ATE) effect
    fate: np.ndarray  # E[Y^(1)(t)] - E[Y^(0)(t)]
    mu1: np.ndarray  # elastic mean of Y^(1)
    mu0: np.ndarray  # elastic mean of Y^(0)
    ref: dict[str, float]  # median over units of the metrics of Y^(1)_i - Y^(0)_i
    func_tau_pop: dict[str, float]  # shape metrics of tau_pop
    func_tau_ind: dict[str, float]  # shape metrics of tau_ind
    func_fate: dict[str, float]  # shape metrics of fate


def population_estimands(
    dgp: DGP,
    t: np.ndarray,
    n_pop: int = 800,
    seed: int = 20240,
    n_iter: int = 3,
    mean: MeanName = "elastic",
) -> Estimands:
    """Causal estimands of ``dgp``, computed on ``n_pop`` potential-outcome draws.

    The elastic means are computed with the same code as the estimators.
    ``mean="karcher"`` replaces every phase-centered elastic mean by the
    Karcher mean function built from the same warps.

    Args:
        dgp: Data-generating process.
        t: Time grid of length m.
        n_pop: Number of units whose potential outcomes are drawn.
        seed: Seed of the draws.
        n_iter: Step-1 alternations of every elastic mean.
        mean: ``"elastic"`` or ``"karcher"``.

    Returns:
        A dict (:class:`Estimands`) with the (m,) curves

        * ``tau_pop``: populational effect mu(Y^(1)) - mu(Y^(0));
        * ``tau_ind``: individual effect, the elastic mean of
          Y^(1)_i - Y^(0)_i;
        * ``tau_loc``: local (ATE) effect, the elastic mean of Y^(1) - Y^(0)
          where the two curves are drawn independently given the same X;
        * ``fate``: E[Y^(1)(t)] - E[Y^(0)(t)];
        * ``mu1``, ``mu0``: elastic means of the two potential outcomes;

        and the dicts of shape metrics (keyed by
        :data:`~shape_preserving_effects.metrics.METRICS`)

        * ``ref``: reference values, the median over units of the metrics of
          Y^(1)_i - Y^(0)_i;
        * ``func_tau_pop``, ``func_tau_ind``, ``func_fate``: metrics of
          tau_pop, tau_ind and fate.

    Raises:
        ValueError: If ``mean`` is not ``"elastic"`` or ``"karcher"``.
    """
    if mean not in MEANS:
        raise ValueError(f"mean must be one of {MEANS}, got {mean!r}")
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=n_pop)
    Y1, Y0 = dgp.potential(t, X, rng)

    mu1 = mean_curve(elastic_mean(Y1, t, n_iter=n_iter), mean)
    mu0 = mean_curve(elastic_mean(Y0, t, n_iter=n_iter), mean)
    tau_pop = mu1 - mu0
    fate = pointwise_mean(Y1) - pointwise_mean(Y0)

    individual_effects = Y1 - Y0
    tau_ind = mean_curve(elastic_mean(individual_effects, t, n_iter=n_iter), mean)

    # local estimand: treated curves redrawn with new latents, same covariates
    treated_redrawn, _ = dgp.potential(t, X, rng)
    tau_loc = mean_curve(elastic_mean(treated_redrawn - Y0, t, n_iter=n_iter), mean)

    vals = {f: np.empty(n_pop) for f in METRICS}
    for i in range(n_pop):
        fi = shape_metrics(individual_effects[i], t)
        for f in METRICS:
            vals[f][i] = fi[f]
    ref = {f: float(np.median(vals[f])) for f in METRICS}

    return {
        "tau_pop": tau_pop,
        "tau_ind": tau_ind,
        "tau_loc": tau_loc,
        "fate": fate,
        "mu1": mu1,
        "mu0": mu0,
        "ref": ref,
        "func_tau_pop": shape_metrics(tau_pop, t),
        "func_tau_ind": shape_metrics(tau_ind, t),
        "func_fate": shape_metrics(fate, t),
    }
