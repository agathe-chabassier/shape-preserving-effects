"""Figures, table and numbers of the sensitivity analyses.

Reads ``<results>/sensitivity/`` (written by ``sensitivity.py``) and writes:

    fig_sens_n.pdf              sample size
    fig_sens_mis.pdf            propensity misspecification (omitted interactions)
    fig_sens_dim.pdf            covariate dimension (dense design)
    tab_sensitivity.tex         median [IQR] L^2 errors of every configuration
    numbers_sensitivity.json    summary numbers, including those quoted in the text

Panel (a) of the misspecification and dimension figures reports the error of
the logistic propensity score of the populational estimator, averaged over
independent draws.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from common import T, load_estimands
from shape_preserving_effects import METRICS, l2_error, logistic_propensity
from sensitivity import INT_SLOPE, _expit, _phi, config_csv, estimand_path, interaction_index

ORDER6 = P.ORDER + ["elastic_unw", "elastic_orc"]

NS = [20, 40, 60, 100, 250, 500, 1000]
N_CONFIG = {20: "n20", 40: "n40", 60: "n60", 100: "n100", 250: "base", 500: "n500", 1000: "n1000"}
LAMS = [0.0, 0.5, 1.0]
LAM_CONFIG = {0.0: "int0", 0.5: "int50", 1.0: "int100"}
DS = [2, 10, 25, 50, 100]
D_CONFIG = {2: "g2", 10: "g10", 25: "g25", 50: "g50", 100: "g100"}


def load_config(results, config, field="l2"):
    """Per-estimator arrays of one column (absolute values for metric errors).

    Replications in which an estimator could not be computed (stored as NaN,
    see ``sensitivity.nan_if_not_computable``) are left out, so an array can be
    shorter than the number of replications, or empty.
    """
    d = defaultdict(list)
    with open(config_csv(results, config)) as f:
        for r in csv.DictReader(f):
            v = float(r[field])
            if field.startswith("e_"):
                v = abs(v)
            values = d[r["estimator"]]  # the estimator is kept even without a value
            if np.isfinite(v):
                values.append(v)
    return {e: np.array(v) for e, v in d.items()}


# ----------------------------------------------------------------------
# Propensity-model diagnostics (panel (a) of two figures)
# ----------------------------------------------------------------------


def misspecification_rmse(lam, n=250, n_rep=40, seed0=98000):
    """Mean RMSE of the fitted propensity in the omitted-interaction design."""
    rmses = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed0 + r)
        Z = rng.standard_normal((n, 4))
        e = _expit(INT_SLOPE * interaction_index(Z, lam))
        A = rng.binomial(1, e).astype(float)
        e_hat = logistic_propensity(Z, A)
        rmses.append(np.sqrt(np.mean((e_hat - e) ** 2)))
    return float(np.mean(rmses))


def dimension_rmse(d, n=250, n_rep=30, seed0=99000):
    """Mean RMSE of the fitted propensity in the dense design of dimension d."""
    rmses = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed0 + 7 * d + r)
        X = rng.standard_normal((n, d))
        U = _phi(X.sum(axis=1) / np.sqrt(d))
        e = _expit(2.2 * (U - 0.5))
        A = rng.binomial(1, e).astype(bool)
        with np.errstate(over="ignore"):
            e_hat = logistic_propensity(X, A.astype(float))
        rmses.append(np.sqrt(np.mean((e_hat - e) ** 2)))
    return float(np.mean(rmses))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def target_resolution(results, n0=3200):
    """Monte-Carlo L^2 resolution of tau_pop of Scenario 1 on n0 draws
    (estimand_resolution.py)."""
    path = os.path.join(results, "resolution", "resolution.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: run simulations/estimand_resolution.py first")
    with open(path) as f:
        return float(json.load(f)[f"scenario1_n{n0}"]["l2_resolution"])


def fig_sens_n(results, path):
    L2 = {n: load_config(results, N_CONFIG[n], "l2") for n in NS}
    DE = {n: load_config(results, N_CONFIG[n], "e_depth") for n in NS}
    labels = ["20", "40", "", "100", "250", "500", "1000"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    a = ax[0]
    P.med_band_plot(a, NS, L2, P.ORDER, xticks=NS, xticklabels=labels, logx=True)
    res = target_resolution(results)
    a.axhline(res, color="0.55", lw=0.8, ls=":", zorder=0)
    a.text(
        NS[0],
        res - 0.004,
        "target resolution",
        color="0.45",
        ha="left",
        va="top",
        fontsize=P.FS_TXT - 1,
    )
    a.set_title(r"(a) Estimators' $L^2$ error to $\tau_{\mathrm{pop}}$")
    a.set_xlabel(r"Sample size $n$ (log scale)")
    a.set_ylabel(r"$L^2$ error")
    a.set_ylim(bottom=0)
    a = ax[1]
    P.med_band_plot(a, NS, DE, P.ORDER, xticks=NS, xticklabels=labels, logx=True)
    a.set_title("(b) Estimators' absolute depth error")
    a.set_xlabel(r"Sample size $n$ (log scale)")
    a.set_ylabel("Absolute error")
    a.set_ylim(bottom=0)
    P.bottom_legend(fig, ax[0], 2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def fig_sens_mis(results, path):
    L2 = {lam: load_config(results, LAM_CONFIG[lam], "l2") for lam in LAMS}
    DE = {lam: load_config(results, LAM_CONFIG[lam], "e_depth") for lam in LAMS}
    labels = ["0\n(no interaction)", "0.5", "1\n(pure interaction)"]
    xlab = r"Interaction share $\lambda$ of the treatment model"
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.0))
    a = ax[0]
    rmse = {lam: misspecification_rmse(lam) for lam in LAMS}
    a.plot(LAMS, [rmse[lam] for lam in LAMS], color="0.3", lw=1.6, marker="s", ms=3.5)
    a.set_title("(a) Propensity-model error\n(RMSE of the fitted propensity)")
    a.set_xlabel(xlab)
    a.set_ylabel(r"RMSE$(\hat e, e)$")
    a.set_xticks(LAMS)
    a.set_xticklabels(labels)
    a.set_ylim(bottom=0)
    P.quant(a)
    P.arrows(a)
    a = ax[1]
    P.med_band_plot(a, LAMS, L2, ORDER6, xticks=LAMS, xticklabels=labels)
    a.set_title(r"(b) Estimators' $L^2$ error to $\tau_{\mathrm{pop}}$")
    a.set_xlabel(xlab)
    a.set_ylabel(r"$L^2$ error")
    a.set_ylim(bottom=0)
    a = ax[2]
    P.med_band_plot(a, LAMS, DE, ORDER6, xticks=LAMS, xticklabels=labels)
    a.set_title("(c) Estimators' absolute depth error")
    a.set_xlabel(xlab)
    a.set_ylabel("Absolute error")
    a.set_ylim(bottom=0)
    P.bottom_legend(fig, ax[1], 4)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(path)
    plt.close(fig)
    return rmse


def fig_sens_dim(results, path):
    L2 = {d: load_config(results, D_CONFIG[d], "l2") for d in DS}
    DE = {d: load_config(results, D_CONFIG[d], "e_depth") for d in DS}
    xlab = r"Covariate dimension $d$ (log scale)"
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.0))
    a = ax[0]
    rmse = {d: dimension_rmse(d) for d in DS}
    a.plot(DS, [rmse[d] for d in DS], "-", color="0.3", lw=1.6, marker="s", ms=3.5)
    a.set_title("(a) Propensity-model error\n(RMSE of the fitted propensity)")
    a.set_xlabel(xlab)
    a.set_ylabel(r"RMSE$(\hat e, e)$")
    a.set_xscale("log")
    a.set_xticks(DS)
    a.set_xticklabels([str(d) for d in DS])
    a.minorticks_off()
    a.set_ylim(bottom=0)
    P.quant(a)
    P.arrows(a)
    a = ax[1]
    P.med_band_plot(a, DS, L2, ORDER6, xticks=DS, logx=True)
    a.set_title(r"(b) Estimators' $L^2$ error to $\tau_{\mathrm{pop}}$")
    a.set_xlabel(xlab)
    a.set_ylabel(r"$L^2$ error")
    a.set_ylim(bottom=0)
    a = ax[2]
    P.med_band_plot(a, DS, DE, ORDER6, xticks=DS, logx=True)
    a.set_title("(c) Estimators' absolute depth error")
    a.set_xlabel(xlab)
    a.set_ylabel("Absolute error")
    a.set_ylim(bottom=0)
    P.bottom_legend(fig, ax[1], 3)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(path)
    plt.close(fig)
    return rmse


# ----------------------------------------------------------------------
# Table and numbers
# ----------------------------------------------------------------------

TABLE_ROWS = [
    ("baseline ($n=250$)", "base"),
    (r"\quad $n=20$", "n20"),
    (r"\quad $n=40$", "n40"),
    (r"\quad $n=60$", "n60"),
    (r"\quad $n=100$", "n100"),
    (r"\quad $n=500$", "n500"),
    (r"\quad $n=1000$", "n1000"),
    (r"\quad $\lambda=0$ (well-spec.)", "int0"),
    (r"\quad $\lambda=0.5$", "int50"),
    (r"\quad $\lambda=1$", "int100"),
    (r"\quad $d=2$", "g2"),
    (r"\quad $d=10$", "g10"),
    (r"\quad $d=25$", "g25"),
    (r"\quad $d=50$", "g50"),
    (r"\quad $d=100$", "g100"),
]
TABLE_GROUPS = {
    "n20": "sample size",
    "int0": "propensity misspecification (omitted interaction)",
    "g2": "covariate dimension (dense design)",
}


def med_iqr(a, d=3):
    return (
        f"${np.median(a):.{d}f}$ [${np.percentile(a, 25):.{d}f}$, ${np.percentile(a, 75):.{d}f}$]"
    )


def make_table(results, path):
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"design & elastic pop.\ (ours) & elastic local (ours) & "
        r"SRVF kernel ATE & DR-FoS \\",
        r"\midrule",
    ]
    for label, config in TABLE_ROWS:
        if config in TABLE_GROUPS:
            lines.append(
                r"\addlinespace[2pt]\multicolumn{5}{@{}l}{\emph{"
                + TABLE_GROUPS[config]
                + r"}}\\[1pt]"
            )
        l2 = load_config(results, config, "l2")
        cells = [med_iqr(l2[e]) if len(l2.get(e, ())) else "--" for e in P.ORDER]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def numbers(results, rmse_mis, rmse_dim):
    num = {}
    for _, config in TABLE_ROWS:
        l2 = load_config(results, config, "l2")
        de = load_config(results, config, "e_depth")
        num[config] = {
            e: {
                "l2_med": float(np.median(l2[e])),
                "l2_q25": float(np.percentile(l2[e], 25)),
                "l2_q75": float(np.percentile(l2[e], 75)),
                "depth_med": float(np.median(de[e])),
                "n_reps": int(len(l2[e])),
            }
            if len(l2[e])
            else {"n_reps": 0}  # estimator not computable in this design
            for e in l2
        }
    E = load_estimands(estimand_path(results), METRICS)
    num["estimand"] = {
        "depth": float(np.min(E["tau_pop"])),
        "gap_fate_pop": float(l2_error(E["fate"], E["tau_pop"], T)),
    }
    num["mis_propensity_rmse"] = {str(k): v for k, v in rmse_mis.items()}
    num["dim_propensity_rmse"] = {str(d): v for d, v in rmse_dim.items()}
    return num


def make_all(results, figures):
    os.makedirs(figures, exist_ok=True)
    P.use_fonts(12)
    fig_sens_n(results, os.path.join(figures, "fig_sens_n.pdf"))
    P.use_fonts(12)
    rmse_mis = fig_sens_mis(results, os.path.join(figures, "fig_sens_mis.pdf"))
    P.use_fonts(12)
    rmse_dim = fig_sens_dim(results, os.path.join(figures, "fig_sens_dim.pdf"))
    make_table(results, os.path.join(figures, "tab_sensitivity.tex"))
    with open(os.path.join(figures, "numbers_sensitivity.json"), "w") as f:
        json.dump(numbers(results, rmse_mis, rmse_dim), f, indent=1)
