"""Figures and numbers of the noise-level and no-effect simulations.

Reads ``<results>/noise/`` (written by ``noise_and_null.py``) and writes:

    fig_bias_grid.pdf       causal estimands at noise levels 0.03, 0.10, 0.30
    fig_bias_null_grid.pdf  causal estimands under no effect
    numbers_noise.json      total variations and distances, including those
                            quoted in the text
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from common import ESTIMAND_KEYS, N_ITER, N_POP, T
from shape_preserving_effects import l2_error
from figures_main import estimand_row
from noise_and_null import row_path, rows

ROW_LABEL = {
    "sd00": r"$\sigma_\varepsilon=0$ (no noise)",
    "sd03": r"$\sigma_\varepsilon=0.03$ (paper)",
    "sd10": r"$\sigma_\varepsilon=0.10$",
    "sd30": r"$\sigma_\varepsilon=0.30$",
    "null_flat": "no effect,\nflat arms",
    "null_shape": "no effect,\nshaped arms",
}
ROW_YLIM = {"null_flat": (-0.32, 0.22)}  # enlarged vertical scale


def load_row(results, name):
    z = np.load(row_path(results, name))
    return {k: z[k] for k in ESTIMAND_KEYS}


def draws(dgp, seed):
    """The potential-outcome draws behind a row (same stream as its estimands)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=N_POP)
    return dgp.potential(T, X, rng)


def fig_grid(results, names, path):
    P.use_fonts(16)
    legend = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
        labelspacing=0.3,
        borderaxespad=0.15,
    )
    design = {n: (d, s) for n, d, s in rows()}
    nr = len(names)
    fig, ax = plt.subplots(nr, 3, figsize=(11.4, 2.55 * nr))
    ax = np.atleast_2d(ax)
    for r, name in enumerate(names):
        Y1, Y0 = draws(*design[name])
        estimand_row(
            ax[r],
            load_row(results, name),
            Y1,
            Y0,
            top=(r == 0),
            bot=(r == nr - 1),
            ylabel0=ROW_LABEL[name],
            show_loc=True,
            legend=legend,
            title_pad=60,
            ylim=ROW_YLIM.get(name, P.YLIM),
        )
    fig.tight_layout(w_pad=1.8, h_pad=1.2)
    fig.savefig(path)
    plt.close(fig)


def total_variation(f):
    return float(np.sum(np.abs(np.diff(f))))


def numbers(results):
    out = {"n_pop": N_POP, "n_iter": N_ITER, "rows": {}}
    E0 = load_row(results, "sd00")
    for name, _, _ in rows():
        E = load_row(results, name)
        d = {}
        for k in ESTIMAND_KEYS:
            d[k] = {
                "tv": total_variation(E[k]),
                "range": float(np.ptp(E[k])),
                "depth": float(E[k].min()),
                "l2_to_zero": l2_error(E[k], 0 * E[k], T),
            }
            if name.startswith("sd"):
                d[k]["l2_to_noisefree"] = l2_error(E[k], E0[k], T)
        d["gap_fate_pop"] = l2_error(E["fate"], E["tau_pop"], T)
        out["rows"][name] = d
    return out


def make_all(results, figures):
    os.makedirs(figures, exist_ok=True)
    fig_grid(results, ["sd03", "sd10", "sd30"], os.path.join(figures, "fig_bias_grid.pdf"))
    fig_grid(results, ["null_flat", "null_shape"], os.path.join(figures, "fig_bias_null_grid.pdf"))
    with open(os.path.join(figures, "numbers_noise.json"), "w") as f:
        json.dump(numbers(results), f, indent=1)
