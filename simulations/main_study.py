"""Main simulation study of the paper (Simulations section and Appendix).

Three parts:

1. causal estimands of the five scenarios and of the five phase-variation
   levels, each computed on N_POP = 800 potential-outcome draws;
2. Monte-Carlo study of the five scenarios: 50 replications of n = 250
   units, four estimators (elastic populational, elastic local, DR-FoS,
   SRVF kernel ATE);
3. phase-variation sweep in Scenario 1: alpha in {0, 0.25, 0.5, 0.75, 1},
   10 replications of n = 250 units per level.

Results are written to ``<results>/main/``. Every replication has its own
seed, and its results are saved as soon as it finishes, so the script can be
interrupted and restarted: it only runs what is missing.

Usage
-----
    python3 simulations/main_study.py               # run everything missing
    python3 simulations/main_study.py --jobs 4      # same, on 4 processes
    python3 simulations/main_study.py --status      # progress report
"""

from __future__ import annotations

import os
import time

import numpy as np

from common import (
    MAX_PAIRS,
    K_MATCH,
    N_ITER,
    N_POP,
    SCENARIOS,
    T,
    append_rows,
    completed_reps,
    load_estimands,
    results_parser,
    run_jobs,
    save_estimands,
)
from shape_preserving_effects import (
    METRICS,
    drop_rebound_dgp,
    fate_drfos,
    l2_error,
    local_effect,
    population_estimands,
    populational_effect,
    sample_trial,
    shape_metrics,
    srvf_kernel_ate,
)

N_STUDY = 250  # units per simulated study
N_REPS = 50  # Monte-Carlo replications per scenario

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SWEEP_REPS = 10
SWEEP_SEED_EST = 900  # estimands of level i: seed 900 + i
SWEEP_SEED0 = 5000  # replication r of level i: seed 5000 + 100 i + r

ESTIMATORS = ("elastic_pop", "elastic_loc", "drfos", "srvf")
COLUMNS = ["rep", "estimator", "l2"] + ["e_" + f for f in METRICS]
SWEEP_COLUMNS = ["alpha"] + COLUMNS


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def scenario_estimand_path(results, k):
    return os.path.join(results, "main", "estimands", f"scenario{k}.npz")


def alpha_estimand_path(results, alpha):
    return os.path.join(results, "main", "estimands", f"alpha{alpha:.2f}.npz")


def scenario_csv(results, k):
    return os.path.join(results, "main", f"scenario{k}.csv")


def scenario_curves_dir(results, k):
    return os.path.join(results, "main", "curves", f"scenario{k}")


def sweep_csv(results):
    return os.path.join(results, "main", "phase_sweep.csv")


def scenario_dgp(k):
    return drop_rebound_dgp(1.0, SCENARIOS[k]["p_rebound"])


# ----------------------------------------------------------------------
# One simulated study
# ----------------------------------------------------------------------


def score(tau_hat, E):
    """L^2 error to tau_pop and signed errors of the three shape metrics."""
    f = shape_metrics(tau_hat, T)
    row = {"l2": l2_error(tau_hat, E["tau_pop"], T)}
    for k in METRICS:
        row["e_" + k] = f[k] - E["ref"][k]
    return row


def run_study(dgp, seed, E):
    """Draw one study of N_STUDY units and fit the four estimators.

    Returns the score rows and the estimated effect curves.
    """
    tr = sample_trial(dgp, N_STUDY, T, seed=seed)
    pop = populational_effect(T, tr.Y, tr.A, tr.X, n_iter=N_ITER)
    curves = {"elastic_pop": pop.effect}
    curves["elastic_loc"] = local_effect(
        T, tr.Y, tr.A, tr.X, k=K_MATCH, n_iter=N_ITER, max_pairs=MAX_PAIRS, rng=seed
    ).effect
    curves["drfos"] = fate_drfos(T, tr.Y, tr.A, tr.X, rng=seed)
    curves["srvf"] = srvf_kernel_ate(T, tr.Y, tr.A, tr.X, n_iter=N_ITER, rng=seed)
    rows = [{"estimator": e, **score(curves[e], E)} for e in ESTIMATORS]
    return rows, curves


# ----------------------------------------------------------------------
# Jobs (run in worker processes)
# ----------------------------------------------------------------------


def run_job(job):
    """Run one job and return ``(job, payload, seconds)``."""
    t0 = time.time()
    kind, results, a, b = job
    if kind == "estimand_scenario":
        k = a
        payload = population_estimands(
            scenario_dgp(k), T, n_pop=N_POP, seed=SCENARIOS[k]["seed_est"], n_iter=N_ITER
        )
    elif kind == "estimand_alpha":
        i = a
        payload = population_estimands(
            drop_rebound_dgp(ALPHAS[i], 1.0), T, n_pop=N_POP, seed=SWEEP_SEED_EST + i, n_iter=N_ITER
        )
    elif kind == "scenario":
        k, rep = a, b
        E = load_estimands(scenario_estimand_path(results, k), METRICS)
        payload = run_study(scenario_dgp(k), SCENARIOS[k]["seed0"] + rep, E)
    elif kind == "sweep":
        i, rep = a, b
        E = load_estimands(alpha_estimand_path(results, ALPHAS[i]), METRICS)
        payload = run_study(drop_rebound_dgp(ALPHAS[i], 1.0), SWEEP_SEED0 + 100 * i + rep, E)
    else:
        raise ValueError(kind)
    return job, payload, time.time() - t0


def save_job(job, payload):
    """Write the result of one job (called in the main process only)."""
    kind, results, a, b = job
    if kind == "estimand_scenario":
        save_estimands(scenario_estimand_path(results, a), payload, METRICS)
    elif kind == "estimand_alpha":
        save_estimands(alpha_estimand_path(results, ALPHAS[a]), payload, METRICS)
    elif kind == "scenario":
        rows, curves = payload
        d = scenario_curves_dir(results, a)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"rep{b}.npz"), **curves)
        append_rows(scenario_csv(results, a), [{"rep": b, **r} for r in rows], COLUMNS)
    elif kind == "sweep":
        rows, _ = payload
        append_rows(
            sweep_csv(results), [{"alpha": ALPHAS[a], "rep": b, **r} for r in rows], SWEEP_COLUMNS
        )


def job_label(job):
    kind, _, a, b = job
    if kind == "estimand_scenario":
        return f"causal estimands, Scenario {a}"
    if kind == "estimand_alpha":
        return f"causal estimands, alpha = {ALPHAS[a]:.2f}"
    if kind == "scenario":
        return f"Scenario {a}, replication {b}"
    return f"phase sweep, alpha = {ALPHAS[a]:.2f}, replication {b}"


def missing_estimands(results):
    jobs = [
        ("estimand_scenario", results, k, None)
        for k in SCENARIOS
        if not os.path.exists(scenario_estimand_path(results, k))
    ]
    jobs += [
        ("estimand_alpha", results, i, None)
        for i in range(len(ALPHAS))
        if not os.path.exists(alpha_estimand_path(results, ALPHAS[i]))
    ]
    return jobs


def missing_replications(results):
    jobs = []
    for k in SCENARIOS:
        done = completed_reps(scenario_csv(results, k), ESTIMATORS)
        jobs += [("scenario", results, k, r) for r in range(N_REPS) if r not in done]
    for i, alpha in enumerate(ALPHAS):
        done = completed_reps(sweep_csv(results), ESTIMATORS, group=("alpha", alpha))
        jobs += [("sweep", results, i, r) for r in range(SWEEP_REPS) if r not in done]
    return jobs


def run_all(jobs, n_jobs):
    if not jobs:
        return
    print(f"{len(jobs)} jobs to run on {n_jobs} process(es)", flush=True)
    for job, payload, dt in run_jobs(run_job, jobs, n_jobs):
        save_job(job, payload)
        print(f"  done: {job_label(job)} ({dt:.0f} s)", flush=True)


def status(results):
    n_est = len(SCENARIOS) + len(ALPHAS) - len(missing_estimands(results))
    print(f"causal estimands: {n_est}/{len(SCENARIOS) + len(ALPHAS)}")
    for k in SCENARIOS:
        done = completed_reps(scenario_csv(results, k), ESTIMATORS)
        print(f"Scenario {k}: {len(done)}/{N_REPS} replications")
    for alpha in ALPHAS:
        done = completed_reps(sweep_csv(results), ESTIMATORS, group=("alpha", alpha))
        print(f"phase sweep, alpha = {alpha:.2f}: {len(done)}/{SWEEP_REPS} replications")


if __name__ == "__main__":
    parser = results_parser(__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    args = parser.parse_args()
    if args.status:
        status(args.results)
    else:
        run_all(missing_estimands(args.results), args.jobs)
        run_all(missing_replications(args.results), args.jobs)
        status(args.results)
