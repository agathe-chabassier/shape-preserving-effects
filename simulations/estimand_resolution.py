"""Monte-Carlo resolution of the causal estimand tau_pop.

The causal estimands are computed on a finite number n_0 of potential-outcome
draws (800 in the main study, 3200 in the sensitivity analyses), so they
carry a Monte-Carlo error, to which the stopped alternating algorithm of
Step 1 may add. This script measures both together: in each of the five
scenarios, tau_pop is recomputed on R = 8 independent sets of n_0 = 800
draws (seeds 30000 + 100 k + r in Scenario k), and in Scenario 1 also on
R = 4 independent sets of n_0 = 3200 draws (seeds 31000 + r), with the
elastic-mean settings of the paper (two alternations). The resolution is
the root-mean-square L^2 distance of these replicates to their average,

    res = sqrt( R / (R - 1) * mean_r || tau^(r) - mean_s tau^(s) ||^2_{L^2} ).

Results are written to ``<results>/resolution/``; the script is resumable
and prints the resolutions at the end. They are quoted in the appendix
"Simulation details and reproducibility", and the resolution for 3200 draws
is the dotted line of the sample-size figure.

Usage
-----
    python3 simulations/estimand_resolution.py [--jobs N]
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from common import N_ITER, N_POP, SCENARIOS, T, results_parser, run_jobs
from shape_preserving_effects import drop_rebound_dgp, elastic_mean

# (scenario, n0) -> replicates and first seed
DESIGNS = {(k, N_POP): dict(reps=8, seed0=30000 + 100 * k) for k in SCENARIOS}
DESIGNS[(1, 3200)] = dict(reps=4, seed0=31000)


def design_name(k, n0):
    return f"scenario{k}_n{n0}"


def rep_path(results, k, n0, r):
    return os.path.join(results, "resolution", f"{design_name(k, n0)}_rep{r}.npz")


def summary_path(results):
    return os.path.join(results, "resolution", "resolution.json")


def tau_pop(k, n0, seed):
    """tau_pop of Scenario k on n0 potential-outcome draws."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, n0)
    Y1, Y0 = drop_rebound_dgp(1.0, SCENARIOS[k]["p_rebound"]).potential(T, X, rng)
    return elastic_mean(Y1, T, n_iter=N_ITER).mean - elastic_mean(Y0, T, n_iter=N_ITER).mean


def _compute(job):
    k, n0, r, seed, results = job
    t0 = time.time()
    return k, n0, r, tau_pop(k, n0, seed), time.time() - t0


def _resolution(taus):
    R = len(taus)
    dev = taus - taus.mean(axis=0)
    return float(np.sqrt(R / (R - 1) * np.trapezoid(dev**2, T, axis=1).mean()))


def summarise(results):
    out = {}
    for (k, n0), d in DESIGNS.items():
        taus = np.array([np.load(rep_path(results, k, n0, r))["tau_pop"] for r in range(d["reps"])])
        out[design_name(k, n0)] = {"reps": len(taus), "l2_resolution": _resolution(taus)}
    with open(summary_path(results), "w") as f:
        json.dump(out, f, indent=1)
    for name, v in out.items():
        print(
            f"{name:16s}: L2 resolution of tau_pop {v['l2_resolution']:.4f}"
            f"  ({v['reps']} replicates)"
        )
    return out


def run_all(results, n_jobs):
    jobs = [
        (k, n0, r, d["seed0"] + r, results)
        for (k, n0), d in DESIGNS.items()
        for r in range(d["reps"])
        if not os.path.exists(rep_path(results, k, n0, r))
    ]
    for k, n0, r, tau, dt in run_jobs(_compute, jobs, n_jobs):
        path = rep_path(results, k, n0, r)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, tau_pop=tau)
        print(f"  done: Scenario {k}, n0 = {n0}, replicate {r} ({dt:.0f} s)", flush=True)
    summarise(results)


if __name__ == "__main__":
    args = results_parser(__doc__.splitlines()[0]).parse_args()
    run_all(args.results, args.jobs)
