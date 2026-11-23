"""Introduction figure: the FATE flattens a shape that every unit has.

Drop-and-rebound curves as in Scenario 1, but with nadir times spread
uniformly over [0.12, 0.86] and a narrower dip (width 0.065), so that the
pointwise mean of the individual effects becomes a long shallow plateau
while every unit rebounds. The control trend is shared by both arms.

    (a) potential outcomes of 30 units, control (dark grey) and treated
        (light grey);
    (b) individual effects Y^(1) - Y^(0) of the same units, the FATE (red)
        and tau_pop (blue), computed on 800 draws by ``population_estimands``.

The estimands are stored in ``<results>/intro/estimands.npz`` the first time
(about two minutes) and reused afterwards.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from common import ESTIMAND_KEYS, N_ITER, N_POP, SCENARIOS, T
from shape_preserving_effects import DGP, population_estimands
from shape_preserving_effects.datasets import NOISE_SD, _bump, _smooth_noise

NADIR_LO, NADIR_HI = 0.12, 0.86  # nadir times ~ Unif[NADIR_LO, NADIR_HI]
INTRO_WIDTH = 0.065  # dip width (Scenario 1: 0.10)
SEED = SCENARIOS[1]["seed_est"]
N_SHOW = 30  # units drawn in each panel


def _intro_potential(t, X, rng):
    X = np.atleast_1d(np.asarray(X, float))
    n = X.size
    c = NADIR_LO + (NADIR_HI - NADIR_LO) * rng.uniform(0.0, 1.0, size=n)
    h = 0.90 + 0.55 * X + 0.06 * rng.standard_normal(n)
    trend = -(0.05 + 0.12 * X)[:, None] * (t[None, :] ** 1.5)
    Y1 = trend - _bump(t, c, h, INTRO_WIDTH) + _smooth_noise(n, t, rng, NOISE_SD)
    Y0 = trend + _smooth_noise(n, t, rng, NOISE_SD)
    return Y1, Y0


intro_dgp = DGP(
    name="intro", potential=_intro_potential, propensity=lambda X: 0.5 + 0.0 * np.asarray(X)
)


def intro_estimands(results):
    path = os.path.join(results, "intro", "estimands.npz")
    if not os.path.exists(path):
        E = population_estimands(intro_dgp, T, n_pop=N_POP, seed=SEED, n_iter=N_ITER)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{k: E[k] for k in ESTIMAND_KEYS})
    z = np.load(path)
    return {k: z[k] for k in ESTIMAND_KEYS}


def fig_intro_fate(results, path):
    est = intro_estimands(results)
    rng = np.random.default_rng(SEED)  # same draws as the estimands
    X = rng.uniform(0.0, 1.0, size=N_POP)
    Y1, Y0 = intro_dgp.potential(T, X, rng)
    D = Y1 - Y0
    show = np.random.default_rng(1).choice(N_POP, N_SHOW, replace=False)

    with plt.rc_context(P.DIDACTIC_RC):
        P.fit_fonts(10.0, 1.0)
        fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.0))
        ylim = (-1.75, 0.35)
        c_ctrl, c_trt = "0.52", "0.72"

        # (a) potential outcomes, labelled next to the curves
        a = ax[0]
        for i in show:
            a.plot(T, Y0[i], color=c_ctrl, lw=0.5, alpha=0.7, zorder=1)
            a.plot(T, Y1[i], color=c_trt, lw=0.5, alpha=0.7, zorder=1)
        k = int(np.argmin(np.abs(T - 0.80)))
        i_ctrl = show[int(np.argmin(Y0[show, k]))]
        a.annotate(
            r"control $Y^{(0)}$",
            xy=(T[k], Y0[i_ctrl, k]),
            xytext=(0.99, 0.27),
            ha="right",
            va="center",
            color=c_ctrl,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=c_ctrl, lw=0.6, alpha=0.8),
        )
        k = int(np.argmin(np.abs(T - 0.72)))
        i_trt = show[int(np.argmax(np.argmin(Y1[show], axis=1)))]  # latest nadir
        a.annotate(
            r"treated $Y^{(1)}$",
            xy=(T[k], Y1[i_trt, k]),
            xytext=(0.99, -1.45),
            ha="right",
            va="center",
            color=c_trt,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=c_trt, lw=0.6, alpha=0.8),
        )
        a.set_title("(a) Potential outcomes: one drop, many timings", pad=6)
        P.bare(a, xlabel=P.XLAB, ylabel="Outcome", xlim=(T[0], T[-1]), ylim=ylim)

        # (b) individual effects and the two causal targets
        a = ax[1]
        for i in show:
            a.plot(T, D[i], color=c_trt, lw=0.5, alpha=0.45, zorder=1)
        k = int(np.argmin(np.abs(T - 0.10)))
        i_eff = show[int(np.argmin(D[show, k]))]  # earliest drop
        a.annotate(
            r"individual effects $Y^{(1)}-Y^{(0)}$",
            xy=(T[k], D[i_eff, k]),
            xytext=(0.01, -1.62),
            ha="left",
            va="center",
            color="0.55",
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, alpha=0.8),
        )
        a.plot(T, est["fate"], color=P.RED, lw=2.0, ls=P.DASH_EST, zorder=4)
        a.plot(T, est["tau_pop"], color=P.BLUE, lw=2.0, ls=P.DASH_EST, zorder=5)
        i_pop = int(np.argmin(est["tau_pop"]))
        a.annotate(
            r"$\tau_{\mathrm{pop}}$ (elastic, ours)",
            xy=(T[i_pop] + 0.01, est["tau_pop"][i_pop] - 0.02),
            xytext=(0.99, -1.45),
            ha="right",
            va="center",
            color=P.BLUE,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=P.BLUE, lw=0.7, alpha=0.8),
        )
        i_f = int(np.argmin(np.abs(T - 0.45)))
        a.annotate(
            r"$\tau_{\mathrm{FATE}}$ (pointwise)",
            xy=(T[i_f] + 0.02, est["fate"][i_f] - 0.02),
            xytext=(0.99, -1.0),
            ha="right",
            va="center",
            color=P.RED,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=P.RED, lw=0.7, alpha=0.8),
        )
        a.set_title("(b) Two causal targets", pad=6)
        P.bare(a, xlabel=P.XLAB, ylabel="Effect", xlim=(T[0], T[-1]), ylim=ylim)

        fig.tight_layout(w_pad=2.0)
        fig.savefig(path)
        plt.close(fig)
