"""Phase-centered elastic mean (and Karcher mean function) of a sample of curves.

Vocabulary (the same as in the paper):

* Frechet mean shape: the Step-1 minimiser of the average squared elastic
  distance, an orbit of SRSFs.
* reference representative: the representative q~ of the Frechet mean shape
  to which the curves are registered (attribute ``reference`` in the code).
* mean warp: the Step-2 Frechet mean gamma_bar, on the Hilbert sphere of warp
  SRSFs, of the warps gamma_i registering the curves to q~.
* phase-centered registered curves: f_i o gamma_i o gamma_bar^{-1}.
* phase-centered elastic mean (elastic mean, for short): the Step-3 weighted
  average of the phase-centered registered curves,

      mu(t) = sum_i w_i (f_i o gamma_i o gamma_bar^{-1})(t),   sum_i w_i = 1.

  This is the "mean of the aligned functions" (aligned mean) of Srivastava
  et al. (2011) and Tucker et al. (2013), their estimator of the signal, and
  an elastic version of the "structural mean" of the registration literature
  (Kneip and Gasser, 1992). Without weights it is the average of the aligned
  functions ``fn`` of
  ``fdasrsf.time_warping.fdawarp.srsf_align(..., center=True)``. It is the
  mean used by every estimator of the package.
* Karcher mean function: the curve whose SRSF is q~ re-timed by
  gamma_bar^{-1}, started at the weighted mean initial value (mu_f of Tucker
  et al., 2013; ``fmean`` of ``fdasrsf``; ``fisher_rao_karcher_mean`` of
  ``scikit-fda``). It is computed from the same warps and returned as
  ``ElasticMeanResult.karcher_mean_function``; the paper compares the two in
  an appendix. It averages the SRSFs, not the slopes q|q| of the curves:
  in general it shrinks the amplitudes and does not end at the mean final
  value.

The inverse-propensity weights of the populational estimator enter the two
Frechet objectives (Steps 1 and 2) and the Step-3 average, which ``fdasrsf``
does not support, hence this implementation.
``tests/test_fdasrsf_crosscheck.py`` compares the two on unweighted data.

Step 1 is the weighted alternating register / average algorithm of
Srivastava et al. (2011), started from the weighted medoid of the sample;
the aligned transforms (q_i, gamma_i) are computed as the SRSFs of the
re-timed curves f_i o gamma_i (see ``srsf.warp_srsf``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from shape_preserving_effects.registration import register
from shape_preserving_effects.srsf import (
    from_srsf,
    invert_warp,
    psi_to_warp,
    to_srsf,
    warp_curve,
    warp_srsf,
    warp_to_psi,
)
from shape_preserving_effects.validation import check_curves, check_grid, normalised_weights

# The phase-centered elastic mean (Steps 1 to 3) and the Karcher mean
# function (built from Steps 1 and 2 only), see the module docstring;
# ``mean=`` argument of the estimators.
MEANS = ("elastic", "karcher")
MeanName = Literal["elastic", "karcher"]

# Step 1 stops when the squared L2 change of the reference representative
# falls below this fraction of its squared L2 norm. Relative, so that the
# result does not depend on the unit of the curves.
STEP1_RELATIVE_TOL = 1e-10


@dataclass
class ElasticMeanResult:
    """Output of :func:`elastic_mean`.

    Attributes:
        mean: (m,) array. The phase-centered elastic mean curve,
            ``weights @ registered``.
        registered: (n, m) array. The phase-centered registered curves
            f_i o gamma_i o gamma_bar^{-1}, whose weighted average is
            ``mean``.
        reference: (m,) array. The reference representative q~: an SRSF on
            the orbit of the Step-1 Frechet mean shape.
        warps: (n, m) array. Optimal warps gamma_i of the sample curves to
            ``reference``.
        mean_warp: (m,) array. Step-2 Frechet mean gamma_bar of ``warps``.
        weights: (n,) array. The normalised weights (summing to one).
        q_mean: (m,) array. SRSF of ``mean``.
        karcher_mean_function: (m,) array. The Karcher mean function computed
            from the same reference representative and warps: the curve
            whose SRSF is ``reference`` re-timed by ``mean_warp^{-1}``,
            started at the weighted mean initial value (see the module
            docstring).
    """

    mean: np.ndarray
    registered: np.ndarray
    reference: np.ndarray
    warps: np.ndarray
    mean_warp: np.ndarray
    weights: np.ndarray
    q_mean: np.ndarray
    karcher_mean_function: np.ndarray


def _sphere_mean(
    psis: np.ndarray, weights: np.ndarray, t: np.ndarray, n_iter: int = 20, tol: float = 1e-9
) -> np.ndarray:
    """Weighted Frechet mean of warp SRSFs on the Hilbert sphere.

    The iteration averages the log maps at the current point and takes one
    exponential-map step (the weighted version of
    ``fdasrsf.utility_functions.SqrtMean``).

    Args:
        psis: (n, m) array of non-negative functions of L^2 norm
            sqrt(t_end - t_0), one warp SRSF per row.
        weights: (n,) array of non-negative weights (normalised here).
        t: Grid of length m.
        n_iter: Maximum number of iterations.
        tol: The iteration stops when the geodesic step angle is below
            ``tol``.

    Returns:
        The (m,) mean, a non-negative function of norm sqrt(t_end - t_0).
    """
    w = weights / weights.sum()
    radius = np.sqrt(t[-1] - t[0])

    def _normalise(v: np.ndarray) -> np.ndarray:
        nrm = np.sqrt(np.trapezoid(v * v, t))
        return radius * v / nrm

    mu = _normalise(np.einsum("i,ij->j", w, psis))
    for _ in range(n_iter):
        cos_theta = np.trapezoid(psis * mu[None, :], t, axis=1) / radius**2
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # log map of each psi at mu (zero when psi coincides with mu)
        scale = np.where(theta > 1e-12, theta / np.maximum(np.sin(theta), 1e-12), 0.0)
        logs = scale[:, None] * (psis - cos_theta[:, None] * mu[None, :])
        v = np.einsum("i,ij->j", w, logs)
        vnorm = np.sqrt(np.trapezoid(v * v, t)) / radius  # geodesic step angle
        if vnorm < tol:
            break
        mu = np.cos(vnorm) * mu + np.sin(vnorm) * (v / vnorm)
        mu = _normalise(np.abs(mu))  # project back onto non-negative functions
    return mu


def phase_centre(
    F: np.ndarray, t: np.ndarray, warps: np.ndarray, mean_warp: np.ndarray
) -> np.ndarray:
    """Registered curves re-timed by the inverse mean warp.

    The composed warp gamma_i o gamma_bar^{-1} is formed first, so that each
    curve is interpolated only once.

    Args:
        F: (n, m) array of curves on the grid ``t``, one per row.
        t: Grid of length m.
        warps: (n, m) array of the warps gamma_i registering the curves.
        mean_warp: (m,) array, the mean warp gamma_bar.

    Returns:
        The (n, m) array of f_i o gamma_i o gamma_bar^{-1}.
    """
    ginv = invert_warp(mean_warp, t)
    out = np.empty_like(np.asarray(F, dtype=float))
    for i in range(len(F)):
        out[i] = warp_curve(F[i], t, warp_curve(warps[i], t, ginv))
    return out


def elastic_mean(
    F: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None, n_iter: int = 3
) -> ElasticMeanResult:
    """Weighted phase-centered elastic mean of the curves ``F``.

    Args:
        F: (n, m) array of curves on the common grid ``t``, one per row.
        t: (m,) uniform, strictly increasing grid (any interval and unit).
        weights: (n,) array of non-negative weights (normalised internally).
            ``None`` gives equal weights. Curves with zero weight do not
            affect the result.
        n_iter: Maximum number of register / average alternations in Step 1.
            A final registration to the resulting reference representative
            is always performed. The simulations of the paper use
            ``n_iter=2``.

    Returns:
        The result, whose ``.mean`` is the phase-centered elastic mean; the
        Karcher mean function computed from the same warps is
        ``.karcher_mean_function``.

    Raises:
        ValueError: If the grid, the curves or the weights are not valid
            (see :mod:`shape_preserving_effects.validation`).
    """
    t = check_grid(t)
    F = check_curves(F, t)
    n = F.shape[0]
    w = normalised_weights(weights, n)

    srsfs = to_srsf(F, t)

    # Step 1: mean shape. Start from the weighted medoid of the SRSFs
    # (unaligned L^2 distances) among the curves of positive weight, then
    # alternate registration and averaging. The aligned transforms
    # (q_i, gamma_i) are the SRSFs of the re-timed curves f_i o gamma_i
    # (srsf.warp_srsf with the curve given). The medoid score is computed in
    # O(n m): sum_j w_j ||q_i - q_j||^2
    #   = ||q_i||^2 + sum_j w_j ||q_j||^2 - 2 <q_i, sum_j w_j q_j>.
    sq_norms = np.trapezoid(srsfs * srsfs, t, axis=1)
    score = sq_norms + w @ sq_norms - 2 * np.trapezoid(srsfs * (w @ srsfs)[None, :], t, axis=1)
    score[w <= 0] = np.inf
    ref = srsfs[np.argmin(score)].copy()

    gammas = np.tile(t, (n, 1))
    for _ in range(n_iter):
        aligned = np.empty_like(srsfs)
        for i in range(n):
            gammas[i], _ = register(ref, srsfs[i], t)
            aligned[i] = warp_srsf(srsfs[i], t, gammas[i], f=F[i])
        new_ref = np.einsum("i,ij->j", w, aligned)
        change = np.trapezoid((new_ref - ref) ** 2, t)
        if change < STEP1_RELATIVE_TOL * np.trapezoid(ref**2, t):
            ref = new_ref
            break
        ref = new_ref

    for i in range(n):
        gammas[i], _ = register(ref, srsfs[i], t)

    # Step 2: mean of the warps on the sphere.
    psis = np.array([warp_to_psi(g, t) for g in gammas])
    mu_psi = _sphere_mean(psis, w, t)
    mean_warp = psi_to_warp(mu_psi, t)

    # Step 3: re-time every registered curve by the inverse mean warp and
    # average the curves with the same weights.
    registered = phase_centre(F, t, gammas, mean_warp)
    mu = np.einsum("i,ij->j", w, registered)

    # Karcher mean function, from the same reference representative and
    # warps: the curve of q~ (started at 0) re-timed by gamma_bar^{-1}, i.e.
    # the curve whose SRSF is (q~, gamma_bar^{-1}), shifted to the weighted
    # mean initial value.
    karcher = warp_curve(from_srsf(ref, t), t, invert_warp(mean_warp, t)) + float(w @ F[:, 0])

    return ElasticMeanResult(
        mean=mu,
        registered=registered,
        reference=ref,
        warps=gammas,
        mean_warp=mean_warp,
        weights=w,
        q_mean=to_srsf(mu, t),
        karcher_mean_function=karcher,
    )


def mean_curve(res: ElasticMeanResult, mean: MeanName = "elastic") -> np.ndarray:
    """The elastic mean or the Karcher mean function of an elastic-mean result.

    Args:
        res: Output of :func:`elastic_mean`.
        mean: ``"elastic"`` for the phase-centered elastic mean, ``"karcher"``
            for the Karcher mean function.

    Returns:
        The chosen (m,) curve.

    Raises:
        ValueError: If ``mean`` is not one of :data:`MEANS`.
    """
    if mean == "elastic":
        return res.mean
    if mean == "karcher":
        return res.karcher_mean_function
    raise ValueError(f"mean must be one of {MEANS}, got {mean!r}")


def karcher_mean_function(
    F: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None, n_iter: int = 3
) -> np.ndarray:
    """Weighted Karcher mean function of the curves ``F``.

    See the module docstring for its definition.

    Args:
        F: (n, m) array of curves on the common grid ``t``, one per row.
        t: (m,) uniform, strictly increasing grid.
        weights: (n,) array of non-negative weights, or ``None`` for equal
            weights.
        n_iter: Maximum number of Step-1 alternations (see
            :func:`elastic_mean`).

    Returns:
        The (m,) Karcher mean function.
    """
    return elastic_mean(F, t, weights=weights, n_iter=n_iter).karcher_mean_function


def pointwise_mean(F: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted cross-sectional (pointwise) mean of the curves ``F``.

    Args:
        F: (n, m) array of curves, one per row.
        weights: (n,) array of non-negative weights (normalised internally),
            or ``None`` for equal weights.

    Returns:
        The (m,) mean curve.

    Raises:
        ValueError: If the weights are not valid (wrong shape, not finite,
            negative, or all zero).
    """
    F = np.asarray(F, dtype=float)
    w = normalised_weights(weights, F.shape[0])
    return np.einsum("i,ij->j", w, F)
