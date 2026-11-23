"""Figure of the comparison of the two means (Appendix J.3 of the paper).

Writes ``fig_compare_means.pdf``: the phase-centered elastic mean and the
Karcher mean function, from the same run of ``elastic_mean`` (same Frechet
mean shape and mean warp), on three samples of the functional-means figure
(``fig_means_grid.py``).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plotting as P


#: Samples of the functional-means figure shown in the comparison.
COMPARE_ROWS = ("sine", "chirp", "damped")


def fig_compare_means(results, path):
    """The two means from the same run of ``elastic_mean`` (same mean shape
    and mean warp), on three samples of the functional-means figure."""
    from shape_preserving_effects import elastic_mean
    from fig_means_grid import means_grid_results
    from functional_means import ELASTIC_ITER, SAMPLES
    from functional_means import T as TM

    R = means_grid_results(results)
    labels = {key: label.replace("\n", " ") for key, label, _ in SAMPLES}
    with plt.rc_context(P.DIDACTIC_RC):
        P.fit_fonts(11.0, 1.0)
        fig, ax = plt.subplots(1, len(COMPARE_ROWS), figsize=(11.0, 2.9))
        for j, key in enumerate(COMPARE_ROWS):
            F = R[f"{key}__sample"]
            res = elastic_mean(F, TM, n_iter=ELASTIC_ITER)
            a = ax[j]
            a.plot(TM, F.T, color=P.GREY, lw=0.4, alpha=0.7, zorder=1)
            a.plot(
                TM, res.karcher_mean_function, color="0.15", lw=1.6, ls=(0, (1.2, 1.2)), zorder=4
            )
            a.plot(TM, res.mean, color=P.BLUE, lw=1.8, zorder=5)
            a.plot(
                [TM[-1]],
                [float(F[:, -1].mean())],
                marker="_",
                color="0.1",
                ms=9,
                mew=2,
                clip_on=False,
                zorder=6,
            )
            lo, hi = F.min(), F.max()
            pad = 0.08 * (hi - lo)
            a.set_title(labels[key], pad=6)
            P.bare(a, xlabel="Time", ylim=(lo - pad, hi + pad))
        hs = [
            Line2D([], [], color=P.BLUE, lw=1.8),
            Line2D([], [], color="0.15", lw=1.6, ls=(0, (1.2, 1.2))),
            Line2D([], [], color="0.1", marker="_", ls="none", ms=9, mew=2),
        ]
        fig.legend(
            hs,
            [
                "phase-centered elastic mean",
                "Karcher mean function",
                "average final value of the sample",
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=P.FS_TXT,
        )
        fig.tight_layout(w_pad=1.5, rect=(0, 0.1, 1, 1))
        fig.savefig(path)
        plt.close(fig)


def make_all(results, figures):
    fig_compare_means(results, os.path.join(figures, "fig_compare_means.pdf"))
