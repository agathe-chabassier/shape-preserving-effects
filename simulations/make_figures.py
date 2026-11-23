"""Build every simulation figure, table and number of the paper from the results.

Usage
-----
    python3 simulations/make_figures.py                       # stored results
    python3 simulations/make_figures.py --results my_results  # your own run

Takes about half a minute: the scripts only read the stored results. (With a new
``--results`` folder, the introduction figure and the functional-means
figure first compute their own results, about two minutes and one minute.)
Figures are written to ``simulations/figures/`` unless ``--figures`` is
given.
"""

from __future__ import annotations

import argparse
import os
import time

from common import DEFAULT_FIGURES, DEFAULT_RESULTS
import figures_compare_means
import figures_main
import figures_noise
import figures_sensitivity
from estimand_gap import print_table
from fig_intro import fig_intro_fate
from fig_means_grid import fig_means_grid
from fig_mean_steps import fig_mean_steps

# File of each figure / table and its number in the paper.
CONTENTS = [
    ("fig_intro_fate.pdf", "Figure 1"),
    ("fig_mean_steps.pdf", "Figure 3"),
    ("fig_est_grid.pdf", "Figure 4"),
    ("fig_mc_grid_main.pdf", "Figure 5"),
    ("fig_phase.pdf", "Figure 6"),
    ("fig_bias_grid.pdf", "Figure 7"),
    ("fig_compare_means.pdf", "Figure 17"),
    ("fig_metricschema.pdf", "Figure 18"),
    ("fig_mc_grid.pdf", "Figure 19"),
    ("fig_boxes_grid.pdf", "Figure 20"),
    ("fig_means_grid.pdf", "Figure 21"),
    ("fig_bias_null_grid.pdf", "Figure 22"),
    ("fig_sens_n.pdf", "Figure 23"),
    ("fig_sens_mis.pdf", "Figure 24"),
    ("fig_sens_dim.pdf", "Figure 25"),
    ("tab_sensitivity.tex", "Table 2"),
]


def main(results, figures):
    os.makedirs(figures, exist_ok=True)
    steps = [
        (
            "introduction figure",
            lambda: fig_intro_fate(results, os.path.join(figures, "fig_intro_fate.pdf")),
        ),
        ("elastic-mean steps", lambda: fig_mean_steps(os.path.join(figures, "fig_mean_steps.pdf"))),
        ("main study", lambda: figures_main.make_all(results, figures)),
        ("noise and no effect", lambda: figures_noise.make_all(results, figures)),
        ("sensitivity analyses", lambda: figures_sensitivity.make_all(results, figures)),
        (
            "functional means",
            lambda: fig_means_grid(results, os.path.join(figures, "fig_means_grid.pdf")),
        ),
        (
            "elastic mean and Karcher mean function",
            lambda: figures_compare_means.make_all(results, figures),
        ),
    ]
    for name, step in steps:
        t0 = time.time()
        print(f"{name} ...", flush=True)
        step()
        print(f"    done ({time.time() - t0:.0f} s)", flush=True)
    print_table(results)
    print(f"\nWritten to {figures}:")
    for fname, number in CONTENTS:
        print(f"  {number:10s} {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--figures", default=DEFAULT_FIGURES)
    args = parser.parse_args()
    main(args.results, args.figures)
