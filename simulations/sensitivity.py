"""Sensitivity analyses of the paper (Appendix, "Sensitivity analyses").

Each analysis perturbs Scenario 1 (alpha = 1, p_rebound = 1, n = 250) along
one axis; the outcome law, hence the causal estimand, is unchanged:

* sample size: n in {20, 40, 60, 100, 250, 500, 1000}
  (configurations ``n20`` ... ``n1000``; n = 250 is ``base``);
* propensity misspecification by omitted interactions: interaction share
  lambda in {0, 0.5, 1} (``int0``, ``int50``, ``int100``);
* covariate dimension, dense design: d in {2, 10, 25, 50, 100}
  (``g2`` ... ``g100``).

The estimand is computed once on 3200 potential-outcome draws. Two reference
estimators are added where relevant: an unweighted elastic estimator
(``elastic_unw``) and an IPW elastic estimator with the true propensities
(``elastic_orc``).

Results are written to ``<results>/sensitivity/``, one csv per
configuration; the script is resumable.

Usage
-----
    python3 simulations/sensitivity.py               # run everything missing
    python3 simulations/sensitivity.py --jobs 4
    python3 simulations/sensitivity.py --status
"""

from __future__ import annotations

import os
import time
from math import erf

import numpy as np

from common import (
    K_MATCH,
    MAX_PAIRS,
    N_ITER,
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
    elastic_mean,
    fate_drfos,
    l2_error,
    local_effect,
    population_estimands,
    populational_effect,
    sample_trial,
    shape_metrics,
    srvf_kernel_ate,
)

N_POP_SENS = 3200  # potential-outcome draws of the estimand
SEED_EST = 60001
N_REF_DRAWS = 2_000_000  # reference draws of the interaction-design quantiles
SEED_REF_DRAWS = 123456

COLUMNS = ["config", "rep", "estimator", "l2"] + ["e_" + f for f in METRICS]

_BASE = drop_rebound_dgp(1.0, 1.0)


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _phi(z):
    """Standard normal distribution function."""
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def sens_dir(results):
    return os.path.join(results, "sensitivity")


def estimand_path(results):
    return os.path.join(sens_dir(results), "estimands.npz")


def quantiles_path(results):
    return os.path.join(sens_dir(results), "interaction_quantiles.npz")


def config_csv(results, config):
    return os.path.join(sens_dir(results), f"{config}.csv")


# ----------------------------------------------------------------------
# Omitted-interaction design
# ----------------------------------------------------------------------
# Four covariates Z ~ N(0, I_4) and a severity index mixing a linear and an
# interaction part,
#     eta = [(1 - lam) (Z w) / |w| + lam (Z1 Z2 + Z3 Z4) / sqrt(2)] / s_lam,
#     s_lam = sqrt((1 - lam)^2 + lam^2),
# with logit e = 2 eta. The outcomes are those of Scenario 1 driven by
# U = F_lam(eta) ~ Unif[0, 1], where F_lam is the distribution function of eta
# (interpolated from 4095 quantiles of 2 * 10^6 reference draws), so that the
# outcome law does not depend on lam. Every estimator fits a linear logistic
# model in Z, which is misspecified as soon as lam > 0.

W_INT = np.array([1.0, 0.5, 0.25, 0.1])
S_INT = float(np.sqrt((W_INT**2).sum()))
INT_SLOPE = 2.0
LAMBDAS = (0.0, 0.5, 1.0)
_QUANTILES = {}


def interaction_index(Z, lam):
    s_lam = float(np.sqrt((1 - lam) ** 2 + lam**2))
    return (
        (1 - lam) * (Z @ W_INT) / S_INT + lam * (Z[:, 0] * Z[:, 1] + Z[:, 2] * Z[:, 3]) / np.sqrt(2)
    ) / s_lam


def build_interaction_quantiles(path):
    """Quantiles of the index eta for each lambda, from fixed reference draws."""
    p = np.arange(1, 4096) / 4096
    Z = np.random.default_rng(SEED_REF_DRAWS).standard_normal((N_REF_DRAWS, 4))
    q = {f"q_lam{int(100 * lam)}": np.quantile(interaction_index(Z, lam), p) for lam in LAMBDAS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, p=p, **q)


def _quantiles(results, lam):
    if results not in _QUANTILES:
        path = quantiles_path(results)
        if not os.path.exists(path):
            build_interaction_quantiles(path)
        _QUANTILES[results] = dict(np.load(path))
    z = _QUANTILES[results]
    return z[f"q_lam{int(100 * lam)}"], z["p"]


def sample_interaction(lam, n, seed, results):
    """One study of the omitted-interaction design; also returns the true e."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 4))
    eta = interaction_index(Z, lam)
    q, p = _quantiles(results, lam)
    U = np.clip(np.interp(eta, q, p), 1e-4, 1 - 1e-4)
    Y1, Y0 = _BASE.potential(T, U, rng)
    e = _expit(INT_SLOPE * eta)
    A = rng.binomial(1, e).astype(int)
    Y = np.where(A[:, None] == 1, Y1, Y0)
    return Z, A, Y, e


# ----------------------------------------------------------------------
# Dense covariate-dimension design
# ----------------------------------------------------------------------
# X ~ N(0, I_d), eta = (X_1 + ... + X_d) / sqrt(d), U = Phi(eta) ~ Unif[0, 1]
# drives the outcomes of Scenario 1, and logit e = 2.2 (U - 1/2) at every d.


def sample_dense(d, n, seed):
    """One study of the dense design in dimension d; also returns the true e."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    U = _phi(X.sum(axis=1) / np.sqrt(d))
    Y1, Y0 = _BASE.potential(T, U, rng)
    e = _expit(2.2 * (U - 0.5))
    A = rng.binomial(1, e).astype(int)
    Y = np.where(A[:, None] == 1, Y1, Y0)
    return X, A, Y, e


# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------
# name -> design, sample size, first seed, replications, estimators

EST4 = ("elastic_pop", "elastic_loc", "drfos", "srvf")
EST5 = EST4 + ("elastic_unw",)
EST6 = EST5 + ("elastic_orc",)

CONFIGS = {
    "base": dict(design=("scenario1",), n=250, seed0=20000, reps=50, estimators=EST4),
    "n20": dict(design=("scenario1",), n=20, seed0=25000, reps=50, estimators=EST4),
    "n40": dict(design=("scenario1",), n=40, seed0=26000, reps=50, estimators=EST4),
    "n60": dict(design=("scenario1",), n=60, seed0=21000, reps=50, estimators=EST4),
    "n100": dict(design=("scenario1",), n=100, seed0=22000, reps=50, estimators=EST4),
    "n500": dict(design=("scenario1",), n=500, seed0=23000, reps=50, estimators=EST4),
    "n1000": dict(design=("scenario1",), n=1000, seed0=24000, reps=30, estimators=EST4),
    "int0": dict(design=("interaction", 0.0), n=250, seed0=48000, reps=50, estimators=EST6),
    "int50": dict(design=("interaction", 0.5), n=250, seed0=49000, reps=50, estimators=EST6),
    "int100": dict(design=("interaction", 1.0), n=250, seed0=49500, reps=50, estimators=EST6),
    "g2": dict(design=("dense", 2), n=250, seed0=55000, reps=50, estimators=EST6),
    "g10": dict(design=("dense", 10), n=250, seed0=55500, reps=50, estimators=EST6),
    "g25": dict(design=("dense", 25), n=250, seed0=56000, reps=50, estimators=EST6),
    "g50": dict(design=("dense", 50), n=250, seed0=56500, reps=50, estimators=EST6),
    "g100": dict(design=("dense", 100), n=250, seed0=57000, reps=50, estimators=EST6),
}


# ----------------------------------------------------------------------
# Estimators specific to the sensitivity analyses
# ----------------------------------------------------------------------


def elastic_oracle(t, Y, A, e, clip=0.02):
    """IPW elastic estimate of tau_pop with the true propensities ``e``."""
    e = np.clip(np.asarray(e, float), clip, 1 - clip)
    m1 = elastic_mean(Y[A == 1], t, weights=1.0 / e[A == 1], n_iter=N_ITER).mean
    m0 = elastic_mean(Y[A == 0], t, weights=1.0 / (1.0 - e[A == 0]), n_iter=N_ITER).mean
    return m1 - m0


def elastic_unweighted(t, Y, A):
    """Difference of the unweighted elastic means of the two arms."""
    m1 = elastic_mean(Y[A == 1], t, n_iter=N_ITER).mean
    m0 = elastic_mean(Y[A == 0], t, n_iter=N_ITER).mean
    return m1 - m0


def nan_if_not_computable(fit):
    """``fit()``, or a curve of NaN when the estimator cannot be computed on the sample.

    Two estimators have a minimal sample size per arm, and stop with an error
    below it:

    * the local estimator needs K_MATCH = 3 units in each arm, which fails in
      one replication at n = 20 (two control units);
    * DR-FoS cross-fits a per-arm linear outcome regression with d + 1
      coefficients, and needs more units of each arm than coefficients in
      the training part of every fold. This fails at d = 100 (about 100
      units of an arm for 101 coefficients) and in the same replication at
      n = 20.

    The scores of these replications are stored as NaN and left out of the
    summaries.
    """
    try:
        return fit()
    except ValueError as error:
        if "exceeds the number of" in str(error) or "fold" in str(error):
            return np.full(T.size, np.nan)
        raise


# ----------------------------------------------------------------------
# One replication
# ----------------------------------------------------------------------


def draw_study(config, rep, results):
    cfg = CONFIGS[config]
    seed = cfg["seed0"] + rep
    kind = cfg["design"][0]
    if kind == "scenario1":
        tr = sample_trial(_BASE, cfg["n"], T, seed=seed)
        return tr.X, tr.A, tr.Y, None, seed
    if kind == "interaction":
        return (*sample_interaction(cfg["design"][1], cfg["n"], seed, results), seed)
    return (*sample_dense(cfg["design"][1], cfg["n"], seed), seed)


def run_rep(config, rep, results):
    """Fit the estimators of ``config`` on replication ``rep``; return score rows."""
    E = load_estimands(estimand_path(results), METRICS)
    X, A, Y, e_true, seed = draw_study(config, rep, results)
    out = {}
    for est in CONFIGS[config]["estimators"]:
        if est == "elastic_pop":
            out[est] = populational_effect(T, Y, A, X, n_iter=N_ITER).effect
        elif est == "elastic_loc":
            out[est] = nan_if_not_computable(
                lambda: (
                    local_effect(
                        T, Y, A, X, k=K_MATCH, n_iter=N_ITER, max_pairs=MAX_PAIRS, rng=seed
                    ).effect
                )
            )
        elif est == "drfos":
            out[est] = nan_if_not_computable(lambda: fate_drfos(T, Y, A, X, rng=seed))
        elif est == "srvf":
            out[est] = srvf_kernel_ate(T, Y, A, X, n_iter=N_ITER, rng=seed)
        elif est == "elastic_unw":
            out[est] = elastic_unweighted(T, Y, A)
        elif est == "elastic_orc":
            out[est] = elastic_oracle(T, Y, A, e_true)
    rows = []
    for est, tau in out.items():
        row = {"config": config, "rep": rep, "estimator": est}
        if np.all(np.isfinite(tau)):
            f = shape_metrics(tau, T)
            row["l2"] = l2_error(tau, E["tau_pop"], T)
            for k in METRICS:
                row["e_" + k] = f[k] - E["ref"][k]
        else:  # estimator not computable on this sample (see nan_if_not_computable)
            row["l2"] = np.nan
            for k in METRICS:
                row["e_" + k] = np.nan
        rows.append(row)
    return rows


def _worker(job):
    config, rep, results = job
    t0 = time.time()
    return config, rep, run_rep(config, rep, results), time.time() - t0


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def build_estimand(results):
    path = estimand_path(results)
    if os.path.exists(path):
        return
    print(f"computing the estimand on {N_POP_SENS} draws ...", flush=True)
    E = population_estimands(_BASE, T, n_pop=N_POP_SENS, seed=SEED_EST, n_iter=N_ITER)
    save_estimands(path, E, METRICS)


def missing(results):
    jobs = []
    for config, cfg in CONFIGS.items():
        done = completed_reps(config_csv(results, config), cfg["estimators"])
        jobs += [(config, r, results) for r in range(cfg["reps"]) if r not in done]
    return jobs


def run_all(results, n_jobs):
    build_estimand(results)
    if not os.path.exists(quantiles_path(results)):
        build_interaction_quantiles(quantiles_path(results))
    jobs = missing(results)
    if not jobs:
        return
    print(f"{len(jobs)} replications to run on {n_jobs} process(es)", flush=True)
    for config, rep, rows, dt in run_jobs(_worker, jobs, n_jobs):
        append_rows(config_csv(results, config), rows, COLUMNS)
        print(f"  done: {config} rep {rep} ({dt:.0f} s)", flush=True)


def status(results):
    print("estimand:", "done" if os.path.exists(estimand_path(results)) else "missing")
    for config, cfg in CONFIGS.items():
        done = completed_reps(config_csv(results, config), cfg["estimators"])
        print(f"{config:7s} {len(done):3d}/{cfg['reps']} replications")


if __name__ == "__main__":
    parser = results_parser(__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    args = parser.parse_args()
    if args.status:
        status(args.results)
    else:
        run_all(args.results, args.jobs)
        status(args.results)
