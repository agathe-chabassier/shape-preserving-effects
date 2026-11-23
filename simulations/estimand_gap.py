"""Gap between the FATE and tau_pop over a grid of (alpha, p_rebound).

Appendix "The causal-estimand gap over the (alpha, p_rebound) grid": the
L^2 distance between tau_FATE and tau_pop for alpha in {0, 0.5, 1} and the
five rebound shares of Scenarios 1 to 5, each on 800 potential-outcome draws
(the same draws, seed 910, in every cell).

Results are written to ``<results>/estimand_gap/``; the script is resumable
and prints the table of the paper at the end.

Usage
-----
    python3 simulations/estimand_gap.py [--jobs N]
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import N_ITER, N_POP, SCENARIOS, T, results_parser, run_jobs
from shape_preserving_effects import drop_rebound_dgp, elastic_mean, pointwise_mean

ALPHAS = (0.0, 0.5, 1.0)
SEED = 910


def cell_path(results, alpha, p_rebound):
    return os.path.join(
        results, "estimand_gap", f"alpha{int(100 * alpha)}_p{int(100 * p_rebound)}.npz"
    )


def gap(alpha, p_rebound):
    """|| tau_FATE - tau_pop ||_{L^2} on N_POP potential-outcome draws."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0.0, 1.0, N_POP)
    Y1, Y0 = drop_rebound_dgp(alpha, p_rebound).potential(T, X, rng)
    mu1 = elastic_mean(Y1, T, n_iter=N_ITER).mean
    mu0 = elastic_mean(Y0, T, n_iter=N_ITER).mean
    fate = pointwise_mean(Y1) - pointwise_mean(Y0)
    return float(np.sqrt(np.trapezoid((fate - (mu1 - mu0)) ** 2, T)))


def _compute(job):
    alpha, p_rebound, results = job
    t0 = time.time()
    return alpha, p_rebound, gap(alpha, p_rebound), time.time() - t0


def print_table(results):
    print("\n|| tau_FATE - tau_pop ||_L2" + "".join(f"   alpha={a:<4g}" for a in ALPHAS))
    for k, sc in SCENARIOS.items():
        p = sc["p_rebound"]
        cells = [float(np.load(cell_path(results, a, p))["gap"]) for a in ALPHAS]
        print(
            f"Scenario {k} (p_rebound = {100 * p:3.0f} %)" + "".join(f"   {c:10.2f}" for c in cells)
        )


def run_all(results, n_jobs):
    jobs = [
        (a, sc["p_rebound"], results)
        for a in ALPHAS
        for sc in SCENARIOS.values()
        if not os.path.exists(cell_path(results, a, sc["p_rebound"]))
    ]
    for alpha, p, g, dt in run_jobs(_compute, jobs, n_jobs):
        path = cell_path(results, alpha, p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, gap=g, alpha=alpha, p_rebound=p)
        print(
            f"  done: alpha = {alpha:.2f}, p_rebound = {p:.2f}: gap {g:.3f} ({dt:.0f} s)",
            flush=True,
        )
    print_table(results)


if __name__ == "__main__":
    args = results_parser(__doc__.splitlines()[0]).parse_args()
    run_all(args.results, args.jobs)
