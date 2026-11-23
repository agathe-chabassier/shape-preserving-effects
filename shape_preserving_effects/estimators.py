"""Estimators of the populational and local elastic treatment effects.

* :func:`populational_effect` estimates tau_pop = mu(Y^(1)) - mu(Y^(0)) by
  inverse-probability weighting: in each arm, the propensity weights enter
  both Frechet objectives of the phase-centered elastic mean (mean shape and
  mean warp) and the final weighted average of the phase-centered registered
  curves, whose weights are normalised to sum to one (Hajek form). The
  weights target the whole population (ATE, the default), the treated (ATT)
  or the controls (ATC).
* :func:`local_effect` estimates the local effect tau_loc by nearest-neighbour
  matching: elastic mean of the pairwise differences between each unit and
  its matched units in the opposite arm.

The default propensity score is a logistic regression fitted by
Newton-Raphson, so that the package depends on numpy only; scores estimated
by any other model can be passed instead (``propensity=``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from shape_preserving_effects.mean import (
    MEANS,
    ElasticMeanResult,
    MeanName,
    elastic_mean,
    mean_curve,
)
from shape_preserving_effects.validation import (
    Seed,
    as_design_matrix,
    check_clip,
    check_curves,
    check_grid,
)

ESTIMANDS = ("ATE", "ATT", "ATC")
Estimand = Literal["ATE", "ATT", "ATC"]


def _check_inputs(
    t: np.ndarray, Y: np.ndarray, A: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check and convert the inputs shared by the estimators.

    Args:
        t: (m,) uniform, strictly increasing time grid (see
            :func:`~shape_preserving_effects.validation.check_grid`).
        Y: (n, m) array of observed outcome curves, one per row, on the grid
            ``t``.
        A: (n,) treatment indicator coded 0 / 1. Both arms must contain at
            least one unit.
        X: (n,) or (n, d) array of confounders, one row per unit. Every
            value must be finite.

    Returns:
        ``(t, Y, A, X)``: ``t`` an (m,) float array, ``Y`` an (n, m) float
        array, ``A`` an (n,) bool array, ``X`` an (n, d) float array (a 1-D
        ``X`` is returned as a single column).

    Raises:
        ValueError: If one of the conditions above does not hold; the
            message names it.
    """
    t = check_grid(t)
    Y = check_curves(Y, t)
    n = Y.shape[0]
    A = np.asarray(A)
    if A.shape != (n,):
        raise ValueError(f"A must have one entry per curve ({n}), got shape {A.shape}")
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError("A must be a binary treatment indicator coded 0 / 1")
    A = A.astype(bool)
    if A.all():
        raise ValueError("A contains no control unit (no entry equal to 0)")
    if not A.any():
        raise ValueError("A contains no treated unit (no entry equal to 1)")
    X = as_design_matrix(X)
    if X.ndim != 2:
        raise ValueError(f"X must be an (n,) or (n, d) array, got {X.ndim} dimensions")
    if X.shape[0] != n:
        raise ValueError(f"X must have one row per curve ({n}), got {X.shape[0]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    return t, Y, A, X


# ----------------------------------------------------------------------
# Propensity score
# ----------------------------------------------------------------------


def _expit(z: np.ndarray) -> np.ndarray:
    """Logistic function 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def _with_intercept(X: np.ndarray) -> np.ndarray:
    """Design matrix [1, X] of a logistic or linear model; ``X`` is (n, d)."""
    return np.column_stack([np.ones(len(X)), X])


def _logistic_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """P(A = 1 | X) of a logistic model with intercept and coefficients ``beta``.

    Args:
        X: (n, d) array of covariates.
        beta: (d + 1,) coefficients, intercept first (from
            :func:`_logistic_fit`).

    Returns:
        The (n,) array of probabilities.
    """
    return _expit(_with_intercept(X) @ beta)


def _logistic_fit(
    X: np.ndarray, y: np.ndarray, n_newton: int = 25, ridge: float = 1e-6
) -> np.ndarray:
    """Logistic regression with intercept, fitted by Newton-Raphson.

    Args:
        X: (n, d) array of covariates.
        y: (n,) array of binary responses coded 0 / 1 (float).
        n_newton: Maximum number of Newton steps.
        ridge: Ridge penalty on the coefficients, which keeps the Hessian
            invertible.

    Returns:
        The (d + 1,) coefficients, intercept first.
    """
    design = _with_intercept(X)
    beta = np.zeros(design.shape[1])
    for _ in range(n_newton):
        p = _expit(design @ beta)
        bernoulli_variance = np.clip(p * (1 - p), 1e-6, None)
        # H = Z^T W Z + ridge * I and g = Z^T (y - p) - ridge * beta
        hessian = design.T @ (bernoulli_variance[:, None] * design) + ridge * np.eye(
            design.shape[1]
        )
        gradient = design.T @ (y - p) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def logistic_propensity(X: np.ndarray, A: np.ndarray, clip: float = 0.02) -> np.ndarray:
    """Logistic estimate of e(X) = P(A = 1 | X), clipped to [clip, 1 - clip].

    Args:
        X: (n,) or (n, d) array of covariates.
        A: (n,) binary treatment indicator coded 0 / 1.
        clip: Clipping level of the estimated propensities, in [0, 0.5).

    Returns:
        The (n,) array of clipped propensity scores.

    Raises:
        ValueError: If ``clip`` is not in [0, 0.5).
    """
    clip = check_clip(clip)
    X = as_design_matrix(X)
    beta = _logistic_fit(X, np.asarray(A, dtype=float))
    return np.clip(_logistic_predict(X, beta), clip, 1 - clip)


# ----------------------------------------------------------------------
# Populational effect (inverse-probability weighting)
# ----------------------------------------------------------------------


@dataclass
class PopulationalEffect:
    """Output of :func:`populational_effect`.

    Attributes:
        effect: (m,) array. Estimated populational effect curve.
        arm1: Weighted elastic mean of the treated arm.
        arm0: Weighted elastic mean of the control arm
            (``arm1.mean - arm0.mean == effect`` with the default
            ``mean="elastic"``; ``arm1.karcher_mean_function -
            arm0.karcher_mean_function`` with ``mean="karcher"``).
        e_hat: (n,) array. Propensity scores used (estimated, or given with
            ``propensity=``), after clipping.
        weights: (n,) array. Inverse-propensity weights of the units before
            normalisation within each arm.
    """

    effect: np.ndarray
    arm1: ElasticMeanResult
    arm0: ElasticMeanResult
    e_hat: np.ndarray
    weights: np.ndarray


def populational_effect(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    clip: float = 0.02,
    n_iter: int = 3,
    mean: MeanName = "elastic",
    estimand: Estimand = "ATE",
    propensity: np.ndarray | None = None,
) -> PopulationalEffect:
    """IPW estimator of the populational elastic effect tau_pop.

    The propensity score e(X) is a logistic regression fitted on the whole
    sample (:func:`logistic_propensity`), unless scores are given with
    ``propensity``. The weights depend on the target population:

    =========  =======================  ========================
    estimand   treated units            control units
    =========  =======================  ========================
    ``ATE``    1 / e(X)                 1 / (1 - e(X))
    ``ATT``    1                        e(X) / (1 - e(X))
    ``ATC``    (1 - e(X)) / e(X)        1
    =========  =======================  ========================

    Each arm's weighted phase-centered elastic mean is computed with
    :func:`elastic_mean`: the weights enter the two Frechet objectives and
    the final average of the registered curves, normalised to sum to one
    within the arm (Hajek form). The estimate is the difference of the two
    arm curves.

    Args:
        t: (m,) common uniform grid of the curves.
        Y: (n, m) array of observed outcome curves, one per row.
        A: (n,) treatment indicator (1 treated, 0 control).
        X: (n,) or (n, d) array of confounders (used for the default
            logistic propensity score).
        clip: Propensity scores, estimated or given, are clipped to
            [clip, 1 - clip]; in [0, 0.5).
        n_iter: Step-1 alternations of the elastic mean (see
            :func:`elastic_mean`).
        mean: Curve built from the two Frechet steps in each arm: the
            phase-centered elastic mean (``"elastic"``, the default and the
            estimand of the paper) or the Karcher mean function
            (``"karcher"``, the alternative compared in the paper's
            appendix).
        estimand: Population targeted by the weights (see the table above).
        propensity: (n,) array of propensity scores P(A = 1 | X) estimated
            by another model (for instance gradient boosting, ideally
            cross-fitted). ``None`` fits the logistic model.

    Returns:
        The estimated effect curve, with the two arm-level means, the
        propensity scores and the weights.

    Raises:
        ValueError: If an input is not valid: unknown ``mean`` or
            ``estimand``, ``clip`` outside [0, 0.5), inputs rejected by
            :func:`_check_inputs`, or given propensities of the wrong shape,
            outside [0, 1], or equal to 0 or 1 after clipping (infinite
            weights).
    """
    if mean not in MEANS:
        raise ValueError(f"mean must be one of {MEANS}, got {mean!r}")
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}, got {estimand!r}")
    clip = check_clip(clip)
    t, Y, A, X = _check_inputs(t, Y, A, X)
    if propensity is None:
        e_hat = logistic_propensity(X, A, clip=clip)
    else:
        e_hat = np.asarray(propensity, dtype=float)
        if e_hat.shape != A.shape:
            raise ValueError(
                f"propensity must have one entry per unit ({len(A)}), got shape {e_hat.shape}"
            )
        if not np.all((e_hat >= 0) & (e_hat <= 1)):
            raise ValueError("propensity must contain probabilities in [0, 1]")
        e_hat = np.clip(e_hat, clip, 1 - clip)
        if np.any((e_hat == 0) | (e_hat == 1)):
            raise ValueError("propensity scores of 0 or 1 give infinite weights; set clip > 0")
    if estimand == "ATE":
        w = np.where(A, 1.0 / e_hat, 1.0 / (1.0 - e_hat))
    elif estimand == "ATT":
        w = np.where(A, 1.0, e_hat / (1.0 - e_hat))
    else:
        w = np.where(A, (1.0 - e_hat) / e_hat, 1.0)
    arm1 = elastic_mean(Y[A], t, weights=w[A], n_iter=n_iter)
    arm0 = elastic_mean(Y[~A], t, weights=w[~A], n_iter=n_iter)
    return PopulationalEffect(
        effect=mean_curve(arm1, mean) - mean_curve(arm0, mean),
        arm1=arm1,
        arm0=arm0,
        e_hat=e_hat,
        weights=w,
    )


# ----------------------------------------------------------------------
# Local effect (matching)
# ----------------------------------------------------------------------


@dataclass
class LocalEffect:
    """Output of :func:`local_effect`.

    Attributes:
        effect: (m,) array. Estimated local effect curve.
        differences: (N, m) array. The matched pairwise difference curves
            whose elastic mean is ``effect``.
        mean_result: Full output of the elastic mean of ``differences``; its
            ``weights`` are the normalised pair weights (equal unless there
            are ties).
    """

    effect: np.ndarray
    differences: np.ndarray
    mean_result: ElasticMeanResult


def _matches(
    a: np.ndarray, b: np.ndarray, k: int, scale: np.ndarray, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matches of the rows of ``a`` among the rows of ``b``.

    Distances are Euclidean after dividing each covariate by ``scale``. Every
    row of ``a`` is matched to all rows of ``b`` at a distance not greater
    than its k-th smallest distance: exactly k rows without ties, more when
    several rows are tied at the k-th distance (Abadie and Imbens, 2006).
    Rows of ``a`` are processed in chunks, so the memory is
    O(chunk * len(b)).

    Args:
        a: (n_a, d) covariates of the units to match.
        b: (n_b, d) covariates of the candidate matches.
        k: Number of neighbours, between 1 and n_b.
        scale: (d,) positive scale of each covariate.
        chunk: Number of rows of ``a`` per distance block.

    Returns:
        ``(rows_a, rows_b, weights)``: one entry per match, unit by unit,
        nearest first. ``weights`` is k / (number of matches of the unit), so
        it is 1 for every match when there are no ties, and every unit of
        ``a`` has total weight k.
    """
    a, b = a / scale, b / scale
    rows_a, rows_b, weights = [], [], []
    for s in range(0, len(a), chunk):
        dist = np.linalg.norm(a[s : s + chunk, None, :] - b[None, :, :], axis=2)
        order = np.argsort(dist, axis=1, kind="stable")
        sorted_dist = np.take_along_axis(dist, order, axis=1)
        is_match = sorted_dist <= sorted_dist[:, k - 1 : k]
        row, rank = np.nonzero(is_match)
        n_matches = is_match.sum(axis=1)
        rows_a.append(s + row)
        rows_b.append(order[row, rank])
        weights.append(k / n_matches[row])
    return np.concatenate(rows_a), np.concatenate(rows_b), np.concatenate(weights)


def local_effect(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    k: int = 5,
    estimand: Estimand = "ATE",
    n_iter: int = 3,
    max_pairs: int | None = None,
    rng: Seed = None,
    mean: MeanName = "elastic",
) -> LocalEffect:
    """Matching estimator of the local elastic effect tau_loc.

    Each treated unit j is matched to its ``k`` nearest control units l (and
    each control unit to its ``k`` nearest treated units, for the ATE), and
    the estimate is the elastic mean of the pairwise differences Y_j - Y_l,
    every difference with the same weight (equally weighted average of the
    phase-centered registered differences). Units tied at the k-th distance
    are all kept, and the unit's k matches are shared among them. The
    covariates are scaled by their standard deviation in the whole sample.
    The differences are taken pair by pair, never with a pointwise average of
    the neighbours. For the ATE, the ATT and ATC objectives are combined with
    weights n1/n and n0/n, which gives every pairwise difference the same
    weight.

    Args:
        t: (m,) common uniform grid of the curves.
        Y: (n, m) array of observed outcome curves, one per row.
        A: (n,) treatment indicator (1 treated, 0 control).
        X: (n,) or (n, d) array of confounders used for matching.
        k: Number of neighbours in the opposite arm, at least 1.
        estimand: Population over which the local effect is averaged.
        n_iter: Step-1 alternations of the elastic mean (see
            :func:`elastic_mean`).
        max_pairs: If the number of pairwise differences exceeds
            ``max_pairs``, a uniform random subsample of that size is used.
            The subsample objective is an unbiased estimate of the matched
            objective; this bounds the computational cost on large samples
            (the cost grows linearly with the number of differences, about 2
            to 4 registrations per difference). Recommended when there are
            more than a few thousand differences.
        rng: Seed of the subsampling (used only when ``max_pairs`` is
            active). Pass an integer for reproducible results.
        mean: Phase-centered elastic mean (``"elastic"``, default) or Karcher
            mean function (``"karcher"``) of the differences (see
            :func:`populational_effect`).

    Returns:
        The estimated effect curve, with the matched differences and the
        full output of their elastic mean.

    Raises:
        ValueError: If an input is not valid: unknown ``mean`` or
            ``estimand``, ``k`` or ``max_pairs`` below 1, ``k`` larger than
            the arm searched, or inputs rejected by :func:`_check_inputs`.
    """
    if mean not in MEANS:
        raise ValueError(f"mean must be one of {MEANS}, got {mean!r}")
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}, got {estimand!r}")
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError(f"max_pairs must be a positive integer or None, got {max_pairs}")
    t, Y, A, X = _check_inputs(t, Y, A, X)
    Y1, Y0 = Y[A], Y[~A]
    X1, X0 = X[A], X[~A]
    if estimand in ("ATT", "ATE") and k > len(Y0):
        raise ValueError(f"k = {k} exceeds the number of control units ({len(Y0)})")
    if estimand in ("ATC", "ATE") and k > len(Y1):
        raise ValueError(f"k = {k} exceeds the number of treated units ({len(Y1)})")
    # One scaling of the covariates for both matching directions.
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0

    # Row r of each block is one pair; treated-to-control pairs first, unit
    # by unit, nearest neighbour first.
    blocks, block_weights = [], []
    if estimand in ("ATT", "ATE"):
        treated, control, weights = _matches(X1, X0, k, scale)
        blocks.append(Y1[treated] - Y0[control])
        block_weights.append(weights)
    if estimand in ("ATC", "ATE"):
        control, treated, weights = _matches(X0, X1, k, scale)
        blocks.append(Y1[treated] - Y0[control])
        block_weights.append(weights)
    diffs = np.concatenate(blocks)
    pair_weights = np.concatenate(block_weights)

    # A uniform subsample keeps each pair with the same probability, so the
    # pair weights stay valid on it.
    if max_pairs is not None and len(diffs) > max_pairs:
        rng = np.random.default_rng(rng)
        kept = rng.choice(len(diffs), size=max_pairs, replace=False)
        diffs, pair_weights = diffs[kept], pair_weights[kept]

    res = elastic_mean(diffs, t, weights=pair_weights, n_iter=n_iter)
    return LocalEffect(effect=mean_curve(res, mean), differences=diffs, mean_result=res)
