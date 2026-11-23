"""Settings and helpers shared by the simulation scripts.

Every number below is the value used in the paper. The scripts import the
package ``shape_preserving_effects``: install it first (``pip install -e .``
from the repository root).
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RESULTS = os.path.join(HERE, "results")
DEFAULT_FIGURES = os.path.join(HERE, "figures")

# Time grid: [0, 1] with m = 81 points.
M = 81
T = np.linspace(0.0, 1.0, M)

N_ITER = 2  # Step-1 alternations of every elastic mean (estimands and estimators)
N_POP = 800  # potential-outcome draws used to compute the causal estimands
K_MATCH = 3  # neighbours of the local (matching) estimator
MAX_PAIRS = 300  # cap on the number of matched differences of the local estimator

# The five scenarios, numbered by decreasing share of rebounding treated
# responses (phase-variation scale alpha = 1 throughout).
#   seed_est : seed of the potential-outcome draws of the causal estimands
#   seed0    : replication r of the Monte-Carlo study uses the seed seed0 + r
SCENARIOS = {
    1: dict(p_rebound=1.00, seed_est=701, seed0=1000),
    2: dict(p_rebound=0.75, seed_est=706, seed0=7000),
    3: dict(p_rebound=0.50, seed_est=705, seed0=6000),
    4: dict(p_rebound=0.25, seed_est=704, seed0=4000),
    5: dict(p_rebound=0.00, seed_est=707, seed0=8000),
}

ESTIMAND_KEYS = ("tau_pop", "tau_ind", "tau_loc", "fate", "mu1", "mu0")


def results_parser(description: str) -> argparse.ArgumentParser:
    """Command-line parser with the options shared by the simulation scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help="results directory (default: simulations/results)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="number of parallel worker processes (default: 1)"
    )
    return parser


def run_jobs(func, jobs, n_jobs=1):
    """Yield ``func(job)`` for every job, in completion order.

    With ``n_jobs > 1`` the jobs run in a pool of worker processes. Every
    job carries its own seed, so the results do not depend on ``n_jobs``.
    """
    if n_jobs <= 1:
        for job in jobs:
            yield func(job)
        return
    with Pool(n_jobs) as pool:
        yield from pool.imap_unordered(func, jobs)


def completed_reps(path: str, estimators, group=None) -> set:
    """Replications of a results csv in which every estimator has a row.

    With ``group=(column, value)``, only rows whose ``column`` equals
    ``value`` are considered.
    """
    if not os.path.exists(path):
        return set()
    found = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if group is not None and row[group[0]] != str(group[1]):
                continue
            found.setdefault(int(row["rep"]), set()).add(row["estimator"])
    return {r for r, est in found.items() if set(estimators) <= est}


def append_rows(path: str, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a csv file, writing the header if the file is new."""
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def save_estimands(path: str, E: dict, metrics) -> None:
    """Save the output of ``population_estimands`` to an ``.npz`` file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        **{k: E[k] for k in ESTIMAND_KEYS},
        ref=np.array([E["ref"][f] for f in metrics]),
        **{
            k: np.array([E[k][f] for f in metrics])
            for k in ("func_tau_pop", "func_tau_ind", "func_fate")
        },
    )


def load_estimands(path: str, metrics) -> dict:
    """Inverse of :func:`save_estimands`."""
    z = np.load(path)
    out = {k: z[k] for k in ESTIMAND_KEYS}
    out["ref"] = dict(zip(metrics, z["ref"]))
    for k in ("func_tau_pop", "func_tau_ind", "func_fate"):
        if k in z.files:
            out[k] = dict(zip(metrics, z[k]))
    return out
