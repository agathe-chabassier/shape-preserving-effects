"""Figures and numbers of the main simulation study.

Reads ``<results>/main/`` (written by ``main_study.py``) and writes to the
figures directory:

    fig_est_grid.pdf        causal estimands across the five scenarios
    fig_mc_grid_main.pdf    Monte-Carlo distribution of the estimators, Scenarios 1 and 3
    fig_mc_grid.pdf         the same for the five scenarios
    fig_boxes_grid.pdf      absolute errors of the estimators, five scenarios
    fig_metricschema.pdf    definition of the three shape metrics
    fig_phase.pdf           phase-variation sweep
    numbers_main.json       summary numbers, including those quoted in the text
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from common import N_POP, SCENARIOS, T, load_estimands
from shape_preserving_effects import METRICS, drop_rebound_dgp, pointwise_mean
from shape_preserving_effects.metrics import _left_crossing
from main_study import (
    ALPHAS,
    alpha_estimand_path,
    scenario_csv,
    scenario_curves_dir,
    scenario_estimand_path,
    sweep_csv,
)

ARM_LABEL = {
    k: rf"$p_{{\mathrm{{rebound}}}}={int(round(100 * sc['p_rebound']))}\%$"
    for k, sc in SCENARIOS.items()
}


def scen_label(k, sep="\n"):
    """'Scenario k' and its rebound share, on two lines or on one."""
    if sep == "\n":
        return f"Scenario {k}{sep}{ARM_LABEL[k]}"
    return f"Scenario {k} ({ARM_LABEL[k]})"


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_scores(path):
    """Per-estimator arrays of the L^2 error and of the absolute metric errors."""
    d = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["estimator"]]["l2"].append(float(r["l2"]))
            for fn in METRICS:
                d[r["estimator"]][fn].append(abs(float(r["e_" + fn])))
    return {e: {k: np.array(v) for k, v in d[e].items()} for e in d}


def load_curves(directory):
    """(n_reps, m) arrays of estimated curves, keyed like the stored files."""
    reps = sorted(int(f[3:-4]) for f in os.listdir(directory) if f.startswith("rep"))
    out = {}
    for r in reps:
        z = np.load(os.path.join(directory, f"rep{r}.npz"))
        for k in z.files:
            out.setdefault(k, []).append(z[k])
    return {k: np.array(v) for k, v in out.items()}


def load_sweep(path):
    """{alpha: {estimator: array of L^2 errors}}."""
    d = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            d[float(r["alpha"])][r["estimator"]].append(float(r["l2"]))
    return d


def potentials(k):
    """The potential-outcome draws behind the estimands of Scenario k."""
    rng = np.random.default_rng(SCENARIOS[k]["seed_est"])
    X = rng.uniform(0, 1, N_POP)
    return drop_rebound_dgp(1.0, SCENARIOS[k]["p_rebound"]).potential(T, X, rng)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def estimand_row(
    ax, E, Y1, Y0, top, bot, ylabel0, show_loc=False, legend=None, title_pad=6, ylim=P.YLIM
):
    """One row of an estimand grid: (a) potential outcomes, (b) effect
    estimands over the individual differences, (c) arm-level estimands."""
    D = Y1 - Y0
    pw1, pw0 = pointwise_mean(Y1), pointwise_mean(Y0)
    sub = range(0, len(Y1), 11)
    leg_a = legend or dict(
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        loc="upper right",
        handlelength=1.6,
        borderaxespad=0.2,
    )
    leg_b = legend or dict(
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        loc="upper left",
        handlelength=1.8,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=1.0,
    )
    leg_c = legend or dict(
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        loc="upper right",
        handlelength=2.1,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.9,
    )
    xlab = P.XLAB if bot else None

    a = ax[0]
    for i in sub:
        a.plot(T, Y0[i], color=P.ORANGE, lw=0.35, alpha=0.30, zorder=1)
        a.plot(T, Y1[i], color=P.GREEN, lw=0.35, alpha=0.30, zorder=1)
    if top:
        a.plot([], [], color=P.GREEN, lw=1.4, label=r"$Y_i^{(1)}$")
        a.plot([], [], color=P.ORANGE, lw=1.4, label=r"$Y_i^{(0)}$")
        a.legend(**leg_a, **({"ncol": 2} if legend else {}))
        a.set_title("(a) Potential outcomes", pad=title_pad)
    P.bare(a, xlabel=xlab, ylabel=ylabel0 + "\nOutcome", ylim=ylim)

    a = ax[1]
    for i in sub:
        a.plot(T, D[i], color=P.GREY, lw=0.35, alpha=0.35, zorder=1)
    a.plot(
        T,
        E["tau_pop"],
        color=P.BLUE,
        lw=2.0,
        ls=P.DASH_EST,
        label=r"$\tau_{\mathrm{pop}}$ (ours)",
        zorder=5,
    )
    a.plot(
        T,
        E["tau_ind"],
        color=P.BROWN,
        lw=1.7,
        ls=P.DASH_EST,
        label=r"$\tau_{\mathrm{ind}}$ (ours)",
        zorder=4,
    )
    if show_loc:
        a.plot(
            T,
            E["tau_loc"],
            color=P.PURPLE,
            lw=1.7,
            ls=P.DASH_EST,
            label=r"$\tau_{\mathrm{loc}}$ (ours)",
            zorder=4,
        )
    a.plot(
        T, E["fate"], color=P.RED, lw=1.7, ls=P.DASH_EST, label=r"$\tau_{\mathrm{FATE}}$", zorder=4
    )
    if top:
        a.plot([], [], color=P.GREY, lw=1.0, label=r"$Y_i^{(1)}-Y_i^{(0)}$")
        a.legend(**leg_b, **({"ncol": 3} if legend else {}))
        a.set_title("(b) Causal estimands", pad=title_pad)
    P.bare(a, xlabel=xlab, ylabel="Effect", ylim=ylim)

    a = ax[2]
    for i in sub:
        a.plot(T, Y0[i], color=P.ORANGE, lw=0.3, alpha=0.20, zorder=1)
        a.plot(T, Y1[i], color=P.GREEN, lw=0.3, alpha=0.20, zorder=1)
    a.plot(T, E["mu1"], color=P.BLUE, lw=1.9, ls=P.DASH_T, label=r"$\mu(Y^{(1)})$", zorder=5)
    a.plot(T, E["mu0"], color=P.BLUE, lw=1.9, ls=P.DASH_C, label=r"$\mu(Y^{(0)})$", zorder=5)
    a.plot(T, pw1, color=P.RED, lw=1.6, ls=P.DASH_T, label=r"$\mathrm{E}[Y^{(1)}]$", zorder=4)
    a.plot(T, pw0, color=P.RED, lw=1.6, ls=P.DASH_C, label=r"$\mathrm{E}[Y^{(0)}]$", zorder=4)
    if top:
        a.legend(**leg_c, **({"ncol": 2} if legend else {}))
        a.set_title("(c) Arm-level causal estimands", pad=title_pad)
    P.bare(a, xlabel=xlab, ylabel="Outcome", ylim=ylim)


def fig_est_grid(est, path):
    """Causal estimands, one row per scenario."""
    keys = list(SCENARIOS)
    nr = len(keys)
    fig, ax = plt.subplots(nr, 3, figsize=(11.4, 2.55 * nr))
    for r, k in enumerate(keys):
        Y1, Y0 = potentials(k)
        estimand_row(ax[r], est[k], Y1, Y0, top=(r == 0), bot=(r == nr - 1), ylabel0=scen_label(k))
    fig.tight_layout(w_pad=1.8, h_pad=1.2)
    fig.savefig(path)
    plt.close(fig)


def fig_mc_grid(est, curves, keys, path, ylim=(-1.55, 0.55)):
    """Pointwise median and interquartile band of the four estimators, one
    panel per scenario; with an odd number of panels the legend fills the
    empty cell."""
    ncol = 2
    nrow = -(-len(keys) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(9.4, 2.8 * nrow))
    ax = np.atleast_2d(ax)
    for j, k in enumerate(keys):
        a = ax[j // ncol, j % ncol]
        for e in P.ORDER:
            col = P.EST_STYLE[e][1]
            lo, med, hi = np.percentile(curves[k][e], [25, 50, 75], axis=0)
            a.fill_between(T, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
            a.plot(T, med, color=col, lw=1.6, zorder=3)
        a.plot(T, est[k]["tau_pop"], color=P.BLUE, lw=1.3, ls=P.DASH_EST, zorder=4)
        a.plot(T, est[k]["fate"], color=P.RED, lw=1.3, ls=P.DASH_EST, zorder=4)
        a.set_title(scen_label(k, sep=" "), pad=5)
        last_in_col = j + ncol >= len(keys)
        P.bare(a, xlabel=P.XLAB if last_in_col else None, ylabel="Effect", ylim=ylim)
    Patch = plt.matplotlib.patches.Patch
    handles, labels = [], []
    for e in P.ORDER:
        col = P.EST_STYLE[e][1]
        handles.append(P.combo(plt.Line2D([], [], color=col, lw=2.0), Patch(color=col, alpha=0.25)))
        labels.append(P.TITLES[e])
    handles.append(plt.Line2D([], [], color=P.BLUE, lw=1.5, ls=P.DASH_EST))
    labels.append(r"$\tau_{\mathrm{pop}}$ (ours; target, blue, dashed)")
    handles.append(plt.Line2D([], [], color=P.RED, lw=1.5, ls=P.DASH_EST))
    labels.append(r"$\tau_{\mathrm{FATE}}$ (red, dashed)")
    n_spare = nrow * ncol - len(keys)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    if n_spare:
        for j in range(len(keys), nrow * ncol):
            ax[j // ncol, j % ncol].axis("off")
        pos = ax[-1, -1].get_position()
        P.titled_legend(
            fig,
            "Estimators (median line over IQR band)",
            handles,
            labels,
            x=pos.x0 + 0.5 * pos.width,
            y=pos.y0 + 0.88 * pos.height,
        )
    else:
        P.titled_legend(
            fig, "Estimators (median line over IQR band)", handles, labels, x=0.50, y=-0.05, ncol=2
        )
    fig.savefig(path)
    plt.close(fig)


def _boxes(ax, scores, field):
    bp = ax.boxplot(
        [scores[e][field] for e in P.ORDER],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.1),
        whiskerprops=dict(color="0.35", lw=0.8),
        capprops=dict(color="0.35", lw=0.8),
    )
    for patch, e in zip(bp["boxes"], P.ORDER):
        patch.set_facecolor(P.EST_STYLE[e][1])
        patch.set_alpha(0.50)
        patch.set_edgecolor(P.EST_STYLE[e][1])
    ax.set_xticks(range(1, len(P.ORDER) + 1))
    ax.set_xticklabels(
        [P.SHORT[e] for e in P.ORDER],
        fontsize=plt.rcParams["xtick.labelsize"] - 1,
        rotation=25,
        ha="right",
    )
    P.quant(ax)


def fig_boxes_grid(scores, path):
    """Boxplots of the errors, one row per scenario, one column per metric."""
    fields = ["l2", "depth", "onset_time", "drop_speed"]
    titles = [
        r"$L^2$ error to $\tau_{\mathrm{pop}}$",
        "Absolute depth error",
        "Absolute onset error",
        "Absolute drop-speed error",
    ]
    with_zero = [False, True, True, True]
    keys = list(SCENARIOS)
    nr = len(keys)
    fig, ax = plt.subplots(nr, 4, figsize=(10.2, 2.35 * nr))
    for r, k in enumerate(keys):
        for j, (f, ti, z) in enumerate(zip(fields, titles, with_zero)):
            a = ax[r, j]
            if z:
                a.axhline(0, color="0.55", lw=0.8, ls=":", zorder=0)
            _boxes(a, scores[k], f)
            a.set_ylim(bottom=0)
            if r == 0:
                a.set_title(ti)
            if j == 0:
                a.set_ylabel(scen_label(k))
            if r < nr - 1:
                a.set_xticklabels([])
    fig.tight_layout(w_pad=1.2, h_pad=1.1)
    fig.savefig(path)
    plt.close(fig)


def _annotate_metrics(ax, tt, tau, title):
    """Draw a curve with its depth, onset time and drop speed."""
    i_n = int(np.argmin(tau))
    t_n, y_n = tt[i_n], tau[i_n]
    lvl = 0.5 * y_n
    t_on = _left_crossing(tau, tt, i_n, lvl)  # last down-crossing of half-depth
    grad = np.gradient(tau, tt)  # steepest descent
    i_s = int(np.argmin(grad))
    t_s, y_s, slope = tt[i_s], tau[i_s], grad[i_s]

    ax.plot(tt, tau, color=P.GREEN, lw=2.0, zorder=3)
    ax.scatter([t_n], [y_n], s=14, color=P.GREEN, zorder=4)

    arr = dict(arrowstyle="<->", lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0)
    ax.annotate("", (0.08, 0.0), (0.08, y_n), arrowprops=dict(color=P.BR_DEPTH, **arr))
    ax.plot([0.08, t_n], [y_n, y_n], color=P.BR_DEPTH, lw=0.7, ls=":", zorder=2)
    ax.text(
        0.045,
        y_n / 2,
        "depth",
        color=P.BR_DEPTH,
        rotation=90,
        ha="center",
        va="center",
        fontsize=P.FS_TXT,
    )
    ax.annotate("", (0.0, -1.32), (t_on, -1.32), arrowprops=dict(color=P.BR_ONSET, **arr))
    ax.plot([t_on, t_on], [lvl, -1.32], color=P.BR_ONSET, lw=0.7, ls=":", zorder=2)
    ax.scatter([t_on], [lvl], s=12, color=P.BR_ONSET, zorder=4)
    ax.text(
        max(t_on / 2, 0.115),
        -1.45,
        "onset time\n(half-depth crossing)",
        color=P.BR_ONSET,
        ha="center",
        va="top",
        fontsize=P.FS_TXT,
    )
    dt = 0.085
    ax.plot(
        [t_s - dt, t_s + dt],
        [y_s - slope * dt, y_s + slope * dt],
        color=P.BR_SPEED,
        lw=1.8,
        zorder=5,
    )
    ax.scatter([t_s], [y_s], s=12, color=P.BR_SPEED, zorder=6)
    ax.annotate(
        "drop speed\n(steepest slope)",
        (t_s + dt + 0.02, y_s + 0.06),
        ha="left",
        va="bottom",
        color=P.BR_SPEED,
        fontsize=P.FS_TXT,
    )
    ax.set_title(title, pad=6)


def fig_metricschema(path):
    """The three shape metrics on a rebounding and a plateauing curve."""
    tt = np.linspace(0.0, 1.0, 400)
    tau_reb = -np.exp(-0.5 * ((tt - 0.40) / 0.12) ** 2)
    tau_pl = -1.0 / (1 + np.exp(-(tt - 0.40) / 0.045)) * (1 + 0.05 * np.clip(tt - 0.40, 0, None))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
    _annotate_metrics(ax[0], tt, tau_reb, "(a) Rebound")
    _annotate_metrics(ax[1], tt, tau_pl, "(b) Plateau")
    for j, a in enumerate(ax):
        a.text(0.905, 0.16, r"$Y_i^{(1)}$", color=P.GREEN, ha="center", fontsize=P.FS_TXT)
        P.bare(a, ylabel="Outcome value" if j == 0 else None, ylim=(-1.85, 0.42))
        a.set_xlim(-0.02, 1.0)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def estimand_gap(E):
    return float(np.sqrt(np.trapezoid((E["fate"] - E["tau_pop"]) ** 2, T)))


def fig_phase(sweep_est, sweep, path):
    """(a) FATE / tau_pop gap and (b) estimators' L^2 errors, as alpha grows."""
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
    a = ax[0]
    a.plot(
        ALPHAS, [estimand_gap(E) for E in sweep_est], "-", color="0.2", lw=1.6, marker="s", ms=3.5
    )
    a.set_title(
        r"(a) Causal-estimand gap"
        "\n"
        r"$\|\tau_{\mathrm{FATE}}-\tau_{\mathrm{pop}}\|_{L^2}$"
    )
    a.set_xlabel(r"Phase-variation scale $\alpha$")
    a.set_ylabel(r"$L^2$ gap")
    a.set_ylim(bottom=0)
    P.quant(a)
    P.arrows(a)

    a = ax[1]
    for e in P.ORDER:
        lbl, col = P.EST_STYLE[e]
        med = [np.median(sweep[al][e]) for al in ALPHAS]
        lo = [np.percentile(sweep[al][e], 25) for al in ALPHAS]
        hi = [np.percentile(sweep[al][e], 75) for al in ALPHAS]
        a.plot(ALPHAS, med, color=col, lw=1.6, marker="o", ms=3, label=lbl)
        a.fill_between(ALPHAS, lo, hi, color=col, alpha=0.13, lw=0)
    a.set_title(r"(b) Estimators' $L^2$ error to $\tau_{\mathrm{pop}}$")
    a.set_xlabel(r"Phase-variation scale $\alpha$")
    a.set_ylabel(r"$L^2$ error")
    a.set_ylim(bottom=0)
    P.quant(a)
    P.arrows(a)
    top = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])  # both panels are L^2 quantities
    ax[0].set_ylim(0, top)
    ax[1].set_ylim(0, top)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------


def numbers(est, scores, sweep_est, sweep):
    def l2(a, b):
        return float(np.sqrt(np.trapezoid((a - b) ** 2, T)))

    num = {}
    for k in SCENARIOS:
        E, m = est[k], scores[k]
        num[f"scenario{k}"] = {
            "func_tau_pop": {f: float(v) for f, v in E["func_tau_pop"].items()},
            "func_tau_ind": {f: float(v) for f, v in E["func_tau_ind"].items()},
            "func_fate": {f: float(v) for f, v in E["func_fate"].items()},
            "ref": {f: float(v) for f, v in E["ref"].items()},
            "l2_fate_pop": l2(E["fate"], E["tau_pop"]),
            "l2_ind_pop": l2(E["tau_ind"], E["tau_pop"]),
            "l2_ind_loc": l2(E["tau_ind"], E["tau_loc"]),
            "norm_tau_pop": float(np.sqrt(np.trapezoid(E["tau_pop"] ** 2, T))),
            "n_reps": int(len(m[P.ORDER[0]]["l2"])),
            "mc": {
                e: {
                    "l2_med": float(np.median(m[e]["l2"])),
                    "l2_q25": float(np.percentile(m[e]["l2"], 25)),
                    "l2_q75": float(np.percentile(m[e]["l2"], 75)),
                    "depth_med": float(np.median(m[e]["depth"])),
                    "onset_med": float(np.nanmedian(m[e]["onset_time"])),
                    "speed_med": float(np.median(m[e]["drop_speed"])),
                }
                for e in P.ORDER
            },
        }
    num["phase_sweep"] = {
        "alpha": ALPHAS,
        "gap": [estimand_gap(E) for E in sweep_est],
        "fate_depth": [float(E["func_fate"]["depth"]) for E in sweep_est],
        "pop_depth": [float(E["func_tau_pop"]["depth"]) for E in sweep_est],
        "est_l2_med": {e: [float(np.median(sweep[a][e])) for a in ALPHAS] for e in P.ORDER},
    }
    return num


def make_all(results, figures):
    os.makedirs(figures, exist_ok=True)
    est = {k: load_estimands(scenario_estimand_path(results, k), METRICS) for k in SCENARIOS}
    scores = {k: load_scores(scenario_csv(results, k)) for k in SCENARIOS}
    curves = {k: load_curves(scenario_curves_dir(results, k)) for k in SCENARIOS}
    sweep_est = [load_estimands(alpha_estimand_path(results, a), METRICS) for a in ALPHAS]
    sweep = load_sweep(sweep_csv(results))

    P.use_fonts(16)
    fig_est_grid(est, os.path.join(figures, "fig_est_grid.pdf"))
    P.use_fonts(14)
    fig_mc_grid(est, curves, list(SCENARIOS), os.path.join(figures, "fig_mc_grid.pdf"))
    P.use_fonts(14)
    fig_mc_grid(est, curves, [1, 3], os.path.join(figures, "fig_mc_grid_main.pdf"))
    P.use_fonts(14)
    fig_boxes_grid(scores, os.path.join(figures, "fig_boxes_grid.pdf"))
    P.use_fonts(13)
    fig_metricschema(os.path.join(figures, "fig_metricschema.pdf"))
    P.use_fonts(12)
    fig_phase(sweep_est, sweep, os.path.join(figures, "fig_phase.pdf"))
    with open(os.path.join(figures, "numbers_main.json"), "w") as f:
        json.dump(numbers(est, scores, sweep_est, sweep), f, indent=1)
