"""Recompute a sample of the stored results from scratch and compare.

Recomputes, with the code of this repository:

* the causal estimands of Scenario 1 (800 draws);
* replication 0 of Scenarios 1 and 3 (four estimators, effect curves and
  scores);
* replication 0 of the phase sweep at alpha = 0.5;
* replication 0 of one configuration of each sensitivity analysis: sample
  size n = 250, interaction share lambda = 1 in the misspecified propensity
  model, and covariate dimension d = 50;
* one cell of the estimand-gap table (alpha = 0.5, Scenario 4);
* the six means of one sample of the functional-means figure (step
  functions);

and reports the largest absolute difference with the stored results. On a
standard laptop the whole check takes 5 to 10 minutes; differences are
expected at the level of floating-point round-off (below 1e-8).

Usage
-----
    python3 simulations/check_reproducibility.py [--results DIR]
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

import estimand_gap
import fig_means_grid
import functional_means as FM
import main_study as MS
import sensitivity as S
from common import DEFAULT_RESULTS, N_ITER, N_POP, SCENARIOS, T, load_estimands
from shape_preserving_effects import METRICS, population_estimands

TOL = 1e-8
SCORE_COLUMNS = ["l2"] + ["e_" + f for f in METRICS]


def stored_rows(path, rep, group=None):
    with open(path) as f:
        return {
            r["estimator"]: r
            for r in csv.DictReader(f)
            if int(r["rep"]) == rep and (group is None or r[group[0]] == str(group[1]))
        }


def max_row_diff(new_rows, old_rows):
    return max(
        abs(float(old_rows[r["estimator"]][c]) - r[c]) for r in new_rows for c in SCORE_COLUMNS
    )


def check_estimands(results):
    E = population_estimands(
        MS.scenario_dgp(1), T, n_pop=N_POP, seed=SCENARIOS[1]["seed_est"], n_iter=N_ITER
    )
    old = load_estimands(MS.scenario_estimand_path(results, 1), METRICS)
    diff = max(
        np.max(np.abs(E[k] - old[k]))
        for k in ("tau_pop", "tau_ind", "tau_loc", "fate", "mu1", "mu0")
    )
    return max(diff, max(abs(E["ref"][f] - old["ref"][f]) for f in METRICS))


def check_scenario(results, k, rep=0):
    E = load_estimands(MS.scenario_estimand_path(results, k), METRICS)
    rows, curves = MS.run_study(MS.scenario_dgp(k), SCENARIOS[k]["seed0"] + rep, E)
    old = np.load(f"{MS.scenario_curves_dir(results, k)}/rep{rep}.npz")
    diff_curves = max(np.max(np.abs(curves[key] - old[key])) for key in old.files)
    old_rows = stored_rows(MS.scenario_csv(results, k), rep)
    rows = [{"rep": rep, **r} for r in rows]
    return max(diff_curves, max_row_diff(rows, old_rows))


def check_sweep(results, i=2, rep=0):
    alpha = MS.ALPHAS[i]
    E = load_estimands(MS.alpha_estimand_path(results, alpha), METRICS)
    rows, _ = MS.run_study(MS.drop_rebound_dgp(alpha, 1.0), MS.SWEEP_SEED0 + 100 * i + rep, E)
    return max_row_diff(rows, stored_rows(MS.sweep_csv(results), rep, ("alpha", alpha)))


def check_sensitivity(results, config, rep=0):
    rows = S.run_rep(config, rep, results)
    return max_row_diff(rows, stored_rows(S.config_csv(results, config), rep))


def check_gap(results, alpha=0.5, p_rebound=0.25):
    old = float(np.load(estimand_gap.cell_path(results, alpha, p_rebound))["gap"])
    return abs(estimand_gap.gap(alpha, p_rebound) - old)


def check_means_grid(results, key="step"):
    old = fig_means_grid.means_grid_results(results)
    k = [s[0] for s in FM.SAMPLES].index(key)
    F = FM.SAMPLES[k][2](np.random.default_rng([FM.SEED, k]), FM.N_CURVES, FM.T)
    diff = float(np.max(np.abs(F - old[f"{key}__sample"])))
    for m, curve in FM.all_means(F, FM.T).items():
        diff = max(diff, float(np.max(np.abs(curve - old[f"{key}__{m}"]))))
    return diff


def main(results):
    checks = [
        ("causal estimands, Scenario 1", lambda: check_estimands(results)),
        ("Monte-Carlo, Scenario 1, replication 0", lambda: check_scenario(results, 1)),
        ("Monte-Carlo, Scenario 3, replication 0", lambda: check_scenario(results, 3)),
        ("phase sweep, alpha = 0.5, replication 0", lambda: check_sweep(results)),
        ("sensitivity, n = 250, replication 0", lambda: check_sensitivity(results, "base")),
        (
            "sensitivity, interaction share lambda = 1, replication 0",
            lambda: check_sensitivity(results, "int100"),
        ),
        ("sensitivity, d = 50, replication 0", lambda: check_sensitivity(results, "g50")),
        ("estimand gap, alpha = 0.5, Scenario 4", lambda: check_gap(results)),
        ("functional means, step functions", lambda: check_means_grid(results)),
    ]
    failures = 0
    for name, check in checks:
        t0 = time.time()
        diff = check()
        ok = diff < TOL
        failures += not ok
        print(
            f"[{'ok' if ok else 'FAIL':4s}] {name:56s} max |difference| = {diff:.1e}"
            f"  ({time.time() - t0:.0f} s)",
            flush=True,
        )
    print("\nall checks passed" if not failures else f"\n{failures} check(s) failed")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    raise SystemExit(1 if main(args.results) else 0)
