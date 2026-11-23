"""The two competing estimators of the paper's simulation study.

* :func:`fate_drfos`: the doubly robust estimator of the functional average
  treatment effect (FATE) of Testa et al. (DR-FoS), i.e. the pointwise AIPW
  formula

      mean_i [ m1(X_i, t) - m0(X_i, t)
               + A_i (Y_i(t) - m1(X_i, t)) / e(X_i)
               - (1 - A_i) (Y_i(t) - m0(X_i, t)) / (1 - e(X_i)) ],

  with a logistic propensity score and per-arm pointwise linear outcome
  regressions, both cross-fitted on the same 5 folds.

* :func:`srvf_kernel_ate`: the "SRVF operator kernel ATE" of Raykov et al.
  (2025), their best-performing estimator, following the authors' released
  implementation: smoothing and per-arm SRVF registration of the outcome
  curves, operator-valued kernel ridge regression of the registered curves on
  (treatment, covariates), and a G-computation contrast.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from shape_preserving_effects.estimators import (
    _check_inputs,
    _logistic_fit,
    _logistic_predict,
    _with_intercept,
)
from shape_preserving_effects.mean import elastic_mean
from shape_preserving_effects.validation import Seed, check_clip

# The tuple (effect, arm1, arm0) returned when return_arms=True.
EffectAndArms = tuple[np.ndarray, np.ndarray, np.ndarray]


# ----------------------------------------------------------------------
# DR-FoS (Testa et al.)
# ----------------------------------------------------------------------


def _random_folds(n: int, n_folds: int, rng: Seed) -> np.ndarray:
    """Fold index (0 to n_folds - 1) of each of ``n`` units, at random."""
    return np.random.default_rng(rng).permutation(n) % n_folds


def _crossfit_logistic_propensity(
    X: np.ndarray, A: np.ndarray, folds: np.ndarray, clip: float = 0.02
) -> np.ndarray:
    """Cross-fitted logistic estimate of e(X), clipped to [clip, 1 - clip].

    The propensity of the units of fold k is predicted by a logistic
    regression fitted on the other folds. Part of the DR-FoS construction.

    Args:
        X: (n, d) array of covariates.
        A: (n,) treatment indicator coded 0 / 1.
        folds: (n,) fold index of each unit.
        clip: Clipping level of the propensities, in [0, 0.5).

    Returns:
        The (n,) array of clipped cross-fitted propensity scores.

    Raises:
        ValueError: If ``clip`` is not in [0, 0.5), or if the training part
            of a fold contains one arm only.
    """
    clip = check_clip(clip)
    A = np.asarray(A, dtype=float)
    e_hat = np.empty(len(A))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if np.unique(A[tr]).size < 2:
            raise ValueError(
                f"the training part of fold {k} contains one arm only; use fewer folds"
            )
        beta = _logistic_fit(X[tr], A[tr])
        e_hat[te] = _logistic_predict(X[te], beta)
    return np.clip(e_hat, clip, 1 - clip)


def _crossfit_pointwise_ols(
    X: np.ndarray, Y: np.ndarray, in_arm: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Cross-fitted per-time-point OLS of Y on [1, X] within one arm.

    The prediction for the units of fold k comes from the units of the arm
    in the other folds.

    Args:
        X: (n, d) array of covariates.
        Y: (n, m) array of outcome curves.
        in_arm: (n,) bool array, True for the units of the arm.
        folds: (n,) fold index of each unit.

    Returns:
        The (n, m) array of predicted curves, for all units.

    Raises:
        ValueError: If the units of the arm outside a fold are too few to
            fit the d + 1 coefficients.
    """
    predictions = np.empty(Y.shape)
    for k in np.unique(folds):
        train, test = in_arm & (folds != k), folds == k
        if train.sum() <= X.shape[1]:
            raise ValueError(
                f"fold {k}: too few units of the arm ({train.sum()}) to fit the outcome "
                f"model ({X.shape[1] + 1} coefficients)"
            )
        beta, *_ = np.linalg.lstsq(_with_intercept(X[train]), Y[train], rcond=None)
        predictions[test] = _with_intercept(X[test]) @ beta
    return predictions


@overload
def fate_drfos(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    n_folds: int = ...,
    clip: float = ...,
    rng: Seed = ...,
    return_arms: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def fate_drfos(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    n_folds: int = ...,
    clip: float = ...,
    rng: Seed = ...,
    *,
    return_arms: Literal[True],
) -> EffectAndArms: ...
def fate_drfos(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    clip: float = 0.02,
    rng: Seed = None,
    return_arms: bool = False,
) -> np.ndarray | EffectAndArms:
    """Doubly robust (pointwise AIPW) estimate of the FATE.

    The two nuisance models, a logistic propensity score and per-arm
    pointwise linear outcome regressions, are cross-fitted on the same
    folds, as specified by Testa et al.: the units of a fold are predicted by
    models fitted on the other folds.

    Args:
        t: (m,) common grid (used to check the curves; the pointwise formula
            does not need it).
        Y: (n, m) array of observed outcome curves.
        A: (n,) treatment indicator coded 0 / 1.
        X: (n,) or (n, d) array of confounders.
        n_folds: Number of cross-fitting folds of the two nuisance models.
        clip: Propensity scores are clipped to [clip, 1 - clip].
        rng: Seed of the fold assignment.
        return_arms: If True, also return the AIPW estimates of E[Y^(1)(t)]
            and E[Y^(0)(t)].

    Returns:
        The (m,) effect curve, or the tuple ``(effect, arm1, arm0)`` when
        ``return_arms`` is True.

    Raises:
        ValueError: If an input is not valid (see
            :func:`shape_preserving_effects.estimators._check_inputs`),
            ``clip`` is not in [0, 0.5), ``n_folds`` is below 2, or the
            training part of a fold is too small: one arm only, or fewer
            units of an arm than coefficients of its outcome model.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    t, Y, treated, covariates = _check_inputs(t, Y, A, X)
    A = treated.astype(float)
    folds = _random_folds(len(A), n_folds, rng)
    e_hat = _crossfit_logistic_propensity(covariates, A, folds, clip=clip)
    m1 = _crossfit_pointwise_ols(covariates, Y, treated, folds)
    m0 = _crossfit_pointwise_ols(covariates, Y, ~treated, folds)
    arm1 = (m1 + (A / e_hat)[:, None] * (Y - m1)).mean(axis=0)
    arm0 = (m0 + ((1 - A) / (1 - e_hat))[:, None] * (Y - m0)).mean(axis=0)
    if return_arms:
        return arm1 - arm0, arm1, arm0
    return arm1 - arm0


# ----------------------------------------------------------------------
# SRVF operator kernel ATE (Raykov et al.)
# ----------------------------------------------------------------------


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``a`` and of ``b``.

    Uses |a|^2 + |b|^2 - 2 a.b, so the memory is len(a) * len(b), not
    len(a) * len(b) * d. Clipped at 0 against round-off.

    Args:
        a: (n_a, d) array.
        b: (n_b, d) array.

    Returns:
        The (n_a, n_b) array of squared distances.
    """
    sq = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T
    return np.maximum(sq, 0.0)


def _median_bandwidth(sq_dists: np.ndarray) -> float:
    """Median pairwise Euclidean distance, from the (n, n) matrix of squared distances."""
    iu = np.triu_indices(len(sq_dists), k=1)
    return float(np.median(np.sqrt(sq_dists[iu]))) if iu[0].size else 1.0


def _moving_average(Y: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average of each row of ``Y`` with edge padding.

    Args:
        Y: (n, m) array of curves, one per row.
        window: Odd number of points of the centred window.

    Returns:
        The (n, m) array of smoothed curves.
    """
    n, m = Y.shape
    pad = (window - 1) // 2
    out = np.empty_like(Y)
    kern = np.ones(window) / window
    for i in range(n):
        padded = np.pad(Y[i], pad, mode="edge")
        out[i] = np.convolve(padded, kern, mode="valid")[:m]
    return out


def _register_by_group(curves: np.ndarray, A: np.ndarray, t: np.ndarray, n_iter: int) -> np.ndarray:
    """Register the curves of each treatment group to their own Frechet mean shape.

    The warps are centred (the phase-centered registered curves), as in
    ``fdasrsf.fdawarp.srsf_align(center=True)``. A group of one curve is left
    as it is.

    Args:
        curves: (n, m) array of curves, one per row.
        A: (n,) treatment indicator coded 0 / 1 (float).
        t: Grid of length m.
        n_iter: Alternations of the per-group elastic mean.

    Returns:
        The (n, m) array of registered curves.
    """
    registered = curves.copy()
    for arm in (0.0, 1.0):
        idx = np.flatnonzero(A == arm)
        if len(idx) > 1:
            registered[idx] = elastic_mean(curves[idx], t, weights=None, n_iter=n_iter).registered
    return registered


def _ovk_fit_predict(
    K_in: np.ndarray, Y_in: np.ndarray, K_cross: np.ndarray, lam: float, K_y: np.ndarray
) -> np.ndarray:
    """Operator-valued kernel ridge regression: fit, then predict.

    Solves ``(kron(K_in, K_y) + n_tr lam I) alpha = vec(Y_in)`` exactly
    through the eigendecompositions of the two kernels, and returns the
    predictions ``kron(k_j, K_y) alpha`` for the rows k_j of ``K_cross``.

    Args:
        K_in: (n_tr, n_tr) input kernel between the training units.
        Y_in: (n_tr, m) array of the training curves.
        K_cross: (n_pred, n_tr) input kernel between the prediction points
            and the training units.
        lam: Ridge parameter lambda.
        K_y: (m, m) output kernel.

    Returns:
        The (n_pred, m) array of predicted curves.
    """
    n_tr = K_in.shape[0]
    lam_x, eigvecs_x = np.linalg.eigh(K_in)
    lam_y, eigvecs_y = np.linalg.eigh(K_y)
    denom = np.outer(lam_x, lam_y) + n_tr * lam
    dual_coefs = eigvecs_x @ ((eigvecs_x.T @ Y_in @ eigvecs_y) / denom) @ eigvecs_y.T
    return (K_cross @ dual_coefs) @ K_y  # (n_pred, m)


@overload
def srvf_kernel_ate(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    lambdas: tuple[float, ...] = ...,
    n_iter: int = ...,
    rng: Seed = ...,
    return_arms: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def srvf_kernel_ate(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    lambdas: tuple[float, ...] = ...,
    n_iter: int = ...,
    rng: Seed = ...,
    *,
    return_arms: Literal[True],
) -> EffectAndArms: ...
def srvf_kernel_ate(
    t: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    lambdas: tuple[float, ...] = (1e-3, 1.0, 1e3),
    n_iter: int = 2,
    rng: Seed = None,
    return_arms: bool = False,
) -> np.ndarray | EffectAndArms:
    """SRVF operator kernel ATE of Raykov et al. (2025).

    Follows the code released by the authors
    (``KernelRidgeRegressionCausalEstimator`` with ``treatment='discrete'``,
    ``use_operator_valued_kernel=True`` and ``apply_srfv_Y=True``, and the
    choice of the ridge parameter of their experiment scripts):

    1. smooth every curve by a 5-point moving average;
    2. register each treatment group separately to its own Frechet mean
       shape (centred warps);
    3. operator-valued kernel ridge regression of the registered curves on
       (treatment, covariates), with the kernels of the released code: one
       on the treatment, a Gaussian kernel on the covariates (median
       pairwise-distance bandwidth) and, for the outcome, the covariance of
       the registered curves. The linear system is solved exactly through
       the eigendecompositions of the kernels;
    4. the effect is the average over units of the predictions at A = 1
       minus those at A = 0.

    Steps 2 and 3 are run twice. The first run, on a random 80 % of the
    units, chooses ``lambda`` in ``lambdas`` by the error of the predictions
    for the other 20 %, at their observed treatment. The second run, on all
    units, gives the estimate.

    One point differs from the released code: it stacks the outcomes time
    by time while its linear system is ordered unit by unit; here both are
    ordered unit by unit, as in Algorithm 1 of their paper.

    Args:
        t: (m,) common uniform grid.
        Y: (n, m) array of observed outcome curves.
        A: (n,) treatment indicator coded 0 / 1.
        X: (n,) or (n, d) array of covariates.
        lambdas: Candidate ridge parameters.
        n_iter: Alternations of the per-group elastic mean.
        rng: Seed of the hold-out split.
        return_arms: If True, also return the averaged predictions at A = 1
            and A = 0.

    Returns:
        The (m,) effect curve, or the tuple ``(effect, arm1, arm0)`` when
        ``return_arms`` is True.

    Raises:
        ValueError: If an input is not valid (see
            :func:`shape_preserving_effects.estimators._check_inputs`).
    """
    t, Y, treated, covariates = _check_inputs(t, Y, A, X)
    A = treated.astype(float)
    n = Y.shape[0]
    rng = np.random.default_rng(rng)

    # 1. smoothing
    smooth_curves = _moving_average(Y)

    # 2. input kernels
    sq_distances = _pairwise_sq_dists(covariates, covariates)
    bandwidth = _median_bandwidth(sq_distances)
    covariate_kernel = np.exp(-sq_distances / (2 * bandwidth**2))
    treatment_kernel = np.outer(A, A)  # kernel on the treatment, as in the released code
    input_kernel = covariate_kernel * treatment_kernel

    # 3. ridge parameter chosen on an 80 / 20 split, as in the released
    # experiments: registration and output kernel from the training part,
    # prediction error on the held-out units.
    perm = rng.permutation(n)
    n_te = max(1, int(0.2 * n))
    te, tr = perm[:n_te], perm[n_te:]
    aligned_tr = _register_by_group(smooth_curves[tr], A[tr], t, n_iter)
    output_kernel_tr = np.cov(aligned_tr.T)
    test_train_kernel = covariate_kernel[np.ix_(te, tr)] * np.outer(A[te], A[tr])
    best_lam, best_err = lambdas[0], np.inf
    for lam in lambdas:
        pred = _ovk_fit_predict(
            input_kernel[np.ix_(tr, tr)], aligned_tr, test_train_kernel, lam, output_kernel_tr
        )
        err = float(np.mean((pred - Y[te]) ** 2))
        if err < best_err:
            best_err, best_lam = err, lam

    # 4. refit on all units; contrast of the predictions at A = 1 and A = 0
    aligned_curves = _register_by_group(smooth_curves, A, t, n_iter)
    output_kernel = np.cov(aligned_curves.T)  # (m, m)
    kernel_treated = covariate_kernel * A[None, :]  # row j: k((x_j, 1), (x_i, A_i))
    kernel_control = covariate_kernel * (0.0 * A)[None, :]  # row j: k((x_j, 0), (x_i, A_i))
    pred1 = _ovk_fit_predict(input_kernel, aligned_curves, kernel_treated, best_lam, output_kernel)
    pred0 = _ovk_fit_predict(input_kernel, aligned_curves, kernel_control, best_lam, output_kernel)
    arm1 = pred1.mean(axis=0)
    arm0 = pred0.mean(axis=0)
    if return_arms:
        return arm1 - arm0, arm1, arm0
    return arm1 - arm0
