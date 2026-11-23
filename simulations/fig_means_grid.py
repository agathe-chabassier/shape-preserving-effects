"""Appendix figure: functional means across data-generating processes.

Seven samples of 30 curves (one row each) are averaged by six operators (one
column each): the cross-sectional mean, the phase-centered elastic mean of
the paper and four other registration-based means (structural, DBA,
soft-DTW, Wasserstein); see ``functional_means.py``.

The samples and the means are stored in ``<results>/means_grid/means.npz``
the first time (about one minute) and reused afterwards.

Usage
-----
    python3 simulations/fig_means_grid.py [--results DIR] [--figures DIR]
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import plotting as P
from common import DEFAULT_FIGURES, DEFAULT_RESULTS
from functional_means import MEANS, SAMPLES, T, all_means, draw_samples

TEAL = "#2A7F86"  # registration-based means other than the elastic mean
COLOURS = {"cross_sectional": P.RED, "elastic": P.BLUE}


def means_grid_results(results):
    """Samples and means, computed once and stored in ``results``."""
    path = os.path.join(results, "means_grid", "means.npz")
    if not os.path.exists(path):
        samples = draw_samples()
        out = {}
        for key, F in samples.items():
            out[f"{key}__sample"] = F
            for m, curve in all_means(F, T).items():
                out[f"{key}__{m}"] = curve
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **out)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def fig_means_grid(results, path):
    R = means_grid_results(results)
    n_rows, n_cols = len(SAMPLES), len(MEANS)
    width = 11.4
    with plt.rc_context(P.DIDACTIC_RC):
        P.fit_fonts(width, 1.0)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(width, 1.42 * n_rows))
        for r, (key, label, _) in enumerate(SAMPLES):
            F = R[f"{key}__sample"]
            curves = [R[f"{key}__{m}"] for m, _ in MEANS]
            lo = min(F.min(), min(c.min() for c in curves))
            hi = max(F.max(), max(c.max() for c in curves))
            pad = 0.08 * (hi - lo)
            ylim = (lo - pad, hi + pad)
            for c, (m, title) in enumerate(MEANS):
                a = ax[r, c]
                a.plot(T, F.T, color=P.GREY, lw=0.4, alpha=0.7, zorder=1)
                a.plot(T, R[f"{key}__{m}"], color=COLOURS.get(m, TEAL), lw=1.6, zorder=3)
                P.bare(
                    a,
                    xlabel="Time" if r == n_rows - 1 else None,
                    ylabel=label if c == 0 else None,
                    ylim=ylim,
                )
                if r == 0:
                    a.set_title(title, pad=6)
        fig.tight_layout(w_pad=1.0, h_pad=0.9)
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--figures", default=DEFAULT_FIGURES)
    args = parser.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    fig_means_grid(args.results, os.path.join(args.figures, "fig_means_grid.pdf"))
