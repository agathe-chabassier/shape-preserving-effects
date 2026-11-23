"""Figure: the three steps of the sample phase-centered elastic mean.

Seven drop-and-rebound curves (one shape, different timings and depths) go
through ``elastic_mean``, and the intermediate objects are drawn:

    (a) the sample f_1, ..., f_n;
    (b) Step 1: the registered curves f_i o gamma*_{q_i -> q~}, where q~ is
        the reference representative of the mean shape, and the curve f~
        whose SRSF is q~ (started at the average initial value);
    (c) Step 2: the optimal warps gamma*_{q_i -> q~}, the identity and their
        Frechet mean (computed from the SRSFs of the warps on the sphere);
    (d) Step 3: the registered curves of (b) re-timed by the inverse
        mean warp, f_i o gamma*_{q_i -> q~} o gamma_bar^{-1}, and their
        average, the phase-centered elastic mean. Since re-timing commutes
        with averaging, it is also the average of the registered curves
        (dashed) re-timed by the inverse mean warp (arrow).

Every panel shows functions of time (curves or warps); the SRSFs are only
used in the computations.

The representative q~ is deliberately re-timed to an early position, so that
Step 2 has a visible effect. The final mean does not depend on that choice
(up to the discretisation of the registration); the script prints the
largest difference between the mean obtained from this representative and
the one returned by ``elastic_mean``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from shape_preserving_effects import elastic_mean, register, to_srsf, warp_curve
from shape_preserving_effects.mean import _sphere_mean, phase_centre
from shape_preserving_effects.srsf import from_srsf, psi_to_warp, warp_to_psi

T = np.linspace(0.0, 1.0, 201)
# One drop-and-rebound shape at seven timings, with mild depth and width
# variation. The small upward drift avoids perfectly flat stretches, on which
# the registration is not unique.
DRIFT = 0.05
CENTRES = np.array([0.30, 0.36, 0.43, 0.50, 0.57, 0.64, 0.70])
DEPTHS = np.array([1.00, 0.85, 1.15, 0.95, 1.10, 0.80, 1.05])
WIDTHS = np.array([0.075, 0.085, 0.080, 0.090, 0.078, 0.088, 0.082])
N = len(CENTRES)


def sample_curves():
    F = np.empty((N, T.size))
    for i in range(N):
        F[i] = DRIFT * T - DEPTHS[i] * np.exp(-((T - CENTRES[i]) ** 2) / (2 * WIDTHS[i] ** 2))
    return F


def mean_steps(F, gamma0):
    """The three steps of the mean, with the Step-1 representative re-timed by
    ``gamma0``. Returns every object drawn in the figure."""
    res = elastic_mean(F, T)
    f0 = float(F[:, 0].mean())
    # the reference representative q~ = (reference, gamma0), drawn through
    # the curve f~ whose SRSF is q~
    rep_curve = warp_curve(from_srsf(res.reference, T, f0), T, gamma0)
    q_tilde = to_srsf(rep_curve, T)
    Q = to_srsf(F, T)
    warps = np.array([register(q_tilde, Q[i], T)[0] for i in range(N)])
    registered = np.array([warp_curve(F[i], T, warps[i]) for i in range(N)])
    psis = np.array([warp_to_psi(g, T) for g in warps])
    mean_warp = psi_to_warp(_sphere_mean(psis, np.ones(N), T), T)
    centred = phase_centre(F, T, warps, mean_warp)  # f_i o gamma_i o gamma_bar^{-1}
    mean_curve = centred.mean(axis=0)
    return res, rep_curve, registered, warps, mean_warp, centred, mean_curve


def fig_mean_steps(path):
    F = sample_curves()
    gamma0 = T**0.65
    res, rep, registered, warps, mean_warp, centred, mean_curve = mean_steps(F, gamma0)
    gap = float(np.max(np.abs(mean_curve - res.mean)))
    print(f"    max |mean from the early representative - elastic_mean| = {gap:.1e}")

    greys = [str(g) for g in np.linspace(0.20, 0.66, N)]  # early to late
    ylim = (-1.42, 0.22)

    with plt.rc_context(P.DIDACTIC_RC):
        P.fit_fonts(11.0, 1.0)

        def header(a, title, sub):
            a.set_title(title, pad=34)
            a.text(
                0.5,
                1.06,
                sub,
                transform=a.transAxes,
                ha="center",
                va="bottom",
                fontsize=P.FS_TITLE,
                fontstyle="italic",
                color="0.3",
            )

        fig, ax = plt.subplots(1, 4, figsize=(11.0, 2.9))

        a = ax[0]
        for i in range(N):
            a.plot(T, F[i], color=greys[i], lw=1.0, zorder=2)
        header(a, r"(a) Sample $f_1,\dots,f_n$", "one shape, seven timings")
        P.bare(a, xlabel="$t$", xlim=(0, 1), ylim=ylim)

        a = ax[1]
        for i in range(N):
            a.plot(T, registered[i], color=greys[i], lw=1.0, zorder=2)
        a.plot(T, rep, color=P.BLUE, lw=2.0, ls=P.DASH_EST, zorder=4)
        i_rep = int(np.argmin(rep))
        a.annotate(
            r"$\tilde f$",
            xy=(T[i_rep] + 0.025, rep[i_rep] + 0.04),
            xytext=(0.66, -1.22),
            color=P.BLUE,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=P.BLUE, lw=0.7, alpha=0.8),
        )
        a.text(
            0.50, -0.72, r"$f_i\circ\gamma^\star_{q_i\to\tilde q}$", color="0.35", fontsize=P.FS_TXT
        )
        header(a, "(b) Step 1: mean shape", "curves registered to $\\tilde q$")
        P.bare(a, xlabel="$t$", xlim=(0, 1), ylim=ylim)

        a = ax[2]
        a.plot(T, T, color="0.55", lw=0.9, ls=(0, (6, 2, 1, 2)), zorder=1)
        for i in range(N):
            a.plot(T, warps[i], color=greys[i], lw=1.0, zorder=2)
        a.plot(T, mean_warp, color=P.RED, lw=2.0, zorder=4)
        a.text(0.80, 0.66, r"$\gamma_{\mathrm{id}}$", color="0.45", fontsize=P.FS_TXT)
        j = int(np.argmin(np.abs(T - 0.36)))
        a.annotate(
            r"$\bar\gamma_n$",
            xy=(T[j], mean_warp[j]),
            xytext=(0.10, 0.80),
            color=P.RED,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=P.RED, lw=0.7, alpha=0.8),
        )
        k = int(np.argmin(np.abs(T - 0.62)))
        a.annotate(
            r"$\gamma^\star_{q_i\to\tilde q}$",
            xy=(T[k], warps[0][k]),
            xytext=(0.58, 0.12),
            color="0.35",
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
        )
        header(a, "(c) Step 2: mean warp", "optimal warps, mean warp")
        P.bare(a, xlabel="$t$", ylabel=r"$\gamma(t)$", xlim=(0, 1), ylim=(0, 1), zero=False)

        a = ax[3]
        for i in range(N):  # registered curves, re-timed
            a.plot(T, centred[i], color=greys[i], lw=0.9, zorder=2)
        reg_avg = registered.mean(axis=0)  # average of the registered curves of (b)
        a.plot(T, reg_avg, color="0.25", lw=1.6, ls=P.DASH_EST, zorder=3)
        a.plot(T, mean_curve, color=P.BLUE, lw=2.2, zorder=5)
        i_mu = int(np.argmin(mean_curve))
        i_reg = int(np.argmin(reg_avg))
        y_arrow = float(F.min()) - 0.09
        a.annotate(
            "",
            xy=(T[i_mu], y_arrow),
            xytext=(T[i_reg], y_arrow),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9),
        )
        a.text(
            T[i_mu] + 0.03,
            y_arrow,
            r"$\circ\,\bar\gamma_n^{-1}$",
            color="0.3",
            fontsize=P.FS_TXT,
            ha="left",
            va="center",
        )
        a.annotate(
            r"$\hat\mu_n$",
            xy=(T[i_mu] + 0.02, mean_curve[i_mu] + 0.03),
            xytext=(0.97, -0.72),
            ha="right",
            va="center",
            color=P.BLUE,
            fontsize=P.FS_TXT,
            arrowprops=dict(arrowstyle="-", color=P.BLUE, lw=0.7, alpha=0.8),
        )
        header(a, r"(d) Step 3: result $\hat\mu_n$", "re-timed, then averaged")
        P.bare(a, xlabel="$t$", xlim=(0, 1), ylim=ylim)

        fig.tight_layout(w_pad=2.2)
        fig.savefig(path)
        plt.close(fig)
