"""Causal estimands under growing noise and under no effect (template bias).

Section "When noise dominates the signal: the template bias" and Appendix
"The noise level and the causal estimands".

* Noise level: Scenario 1 with the same 800 units (seed of the Scenario-1
  estimands) at noise levels sigma_eps in {0, 0.03, 0.10, 0.30}. The
  noise-free row is the reference of the reported distances and is not drawn.
* No effect, flat arms (``null_flat``): both potential outcomes follow the
  control law, with independent noise.
* No effect, shaped arms (``null_shape``): both potential outcomes follow the
  treated law of Scenario 1, drawn independently given X.

Results are written to ``<results>/noise/<row>.npz``; the script is
resumable.

Usage
-----
    python3 simulations/noise_and_null.py [--jobs N]
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import ESTIMAND_KEYS, N_ITER, N_POP, SCENARIOS, T, results_parser, run_jobs
from shape_preserving_effects import DGP, drop_rebound_dgp, population_estimands

NOISE_LEVELS = [0.0, 0.03, 0.10, 0.30]
SEED_NULL = {"null_flat": 711, "null_shape": 712}


def null_dgp(kind):
    """No-effect designs built from Scenario 1 (see the module docstring)."""
    base = drop_rebound_dgp(1.0, 1.0)

    def potential(t, X, rng):
        if kind == "null_flat":
            _, Y0 = base.potential(t, X, rng)
            _, Y1 = base.potential(t, X, rng)
        else:
            Y1, _ = base.potential(t, X, rng)
            Y0, _ = base.potential(t, X, rng)
        return Y1, Y0

    return DGP(name=kind, potential=potential, propensity=base.propensity)


def noise_row_name(sd):
    return f"sd{int(round(100 * sd)):02d}"


def rows():
    """(name, dgp, seed) of every computed row."""
    out = [
        (noise_row_name(sd), drop_rebound_dgp(1.0, 1.0, noise_sd=sd), SCENARIOS[1]["seed_est"])
        for sd in NOISE_LEVELS
    ]
    out += [(k, null_dgp(k), SEED_NULL[k]) for k in ("null_flat", "null_shape")]
    return out


def row_path(results, name):
    return os.path.join(results, "noise", f"{name}.npz")


def _compute(job):
    name, results = job
    dgp, seed = {n: (d, s) for n, d, s in rows()}[name]
    t0 = time.time()
    E = population_estimands(dgp, T, n_pop=N_POP, seed=seed, n_iter=N_ITER)
    return name, E, time.time() - t0


def run_all(results, n_jobs):
    jobs = [(name, results) for name, _, _ in rows() if not os.path.exists(row_path(results, name))]
    if not jobs:
        print("all rows computed")
        return
    for name, E, dt in run_jobs(_compute, jobs, n_jobs):
        os.makedirs(os.path.dirname(row_path(results, name)), exist_ok=True)
        np.savez(row_path(results, name), **{k: E[k] for k in ESTIMAND_KEYS})
        print(f"  done: {name} ({dt:.0f} s)", flush=True)


if __name__ == "__main__":
    args = results_parser(__doc__.splitlines()[0]).parse_args()
    run_all(args.results, args.jobs)
