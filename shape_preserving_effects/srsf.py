"""Square-root slope function (SRSF) transform and warp actions.

All functions work on curves sampled on a common uniform grid ``t`` of
length ``m``. A curve is a 1-D array of length ``m``; a batch of curves is a
2-D array of shape ``(n, m)``.

Conventions:

* SRSF:          q(t) = sign(f'(t)) * sqrt(|f'(t)|)
* inverse SRSF:  f(t) = f(t_0) + int_{t_0}^t q(s) |q(s)| ds
* warp action:   (q, gamma) = (q o gamma) * sqrt(gamma'); when the curve f
  is known it is computed as the SRSF of the re-timed curve f o gamma (see
  :func:`warp_srsf`)
* a warp is an increasing function with gamma(t_0) = t_0 and
  gamma(t_end) = t_end, stored as its values on the grid ``t``.
"""

from __future__ import annotations

import numpy as np


def _cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of ``y`` along its last axis, starting at 0.

    Args:
        y: Values on the grid ``t``, of shape (m,) or (n, m).
        t: Grid of length m.

    Returns:
        The integral from ``t[0]`` to every grid point, with the shape of ``y``.
    """
    increments = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(t)
    return np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum(increments, axis=-1)], axis=-1)


def to_srsf(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SRSF q = sign(f') sqrt(|f'|) of one curve or of a batch of curves.

    Args:
        f: One curve (m,), or curves (n, m) one per row, on the grid ``t``.
        t: Grid of length m.

    Returns:
        The SRSF, with the shape of ``f``. ``f'`` is computed by central
        differences (``np.gradient``).
    """
    df = np.gradient(f, t, axis=-1)
    return np.sign(df) * np.sqrt(np.abs(df))


def from_srsf(q: np.ndarray, t: np.ndarray, f0: float = 0.0) -> np.ndarray:
    """Inverse SRSF: integrate q|q| (trapezoid rule) starting from ``f0``.

    The trapezoid rule is not the exact inverse of the central differences of
    :func:`to_srsf`: ``from_srsf(to_srsf(f, t), t, f[0])`` differs from ``f``
    by the discretisation error (0.004 on a Gaussian bump of width 0.1 on 81
    points).

    Args:
        q: One SRSF (m,), or SRSFs (n, m) one per row, on the grid ``t``.
        t: Grid of length m.
        f0: Value of the curve(s) at ``t[0]``.

    Returns:
        The curve(s), with the shape of ``q``.
    """
    return f0 + _cumulative_trapezoid(q * np.abs(q), t)


def warp_curve(f: np.ndarray, t: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Composition f o gamma of a function sampled on ``t`` with a warp.

    Works for any function sampled on the grid: a curve (re-timed curve
    f o gamma), an SRSF (q o gamma) or another warp (composed warp).

    Args:
        f: Values of the function on the grid ``t``, of shape (m,).
        t: Grid of length m.
        gamma: Warp on the grid ``t``, of shape (m,).

    Returns:
        The (m,) array of f(gamma(t_j)), by linear interpolation.
    """
    return np.interp(gamma, t, f)


def warp_srsf(
    q: np.ndarray, t: np.ndarray, gamma: np.ndarray, f: np.ndarray | None = None
) -> np.ndarray:
    """Action of the warp ``gamma`` on an SRSF: (q o gamma) * sqrt(gamma').

    If the curve ``f`` of ``q`` is given (``q = to_srsf(f, t)``), the action
    is computed as the SRSF of the re-timed curve, since (q_f, gamma) =
    q_{f o gamma}: ``to_srsf(f o gamma)``. It returns ``q`` exactly for the
    identity warp, and keeps the total variation int |q|^2 up to the
    discretisation error of ``np.gradient`` (central differences), also on
    features one or two grid points wide. This is the form used by
    :func:`elastic_mean`.

    Without ``f``, the pointwise formula (q o gamma) * sqrt(gamma') is used.
    It evaluates q at the warped grid points only, so where gamma compresses
    time it can miss part of a feature that falls between them (a sharp
    step, a narrow bump); pass ``f`` whenever the curve is available.

    Args:
        q: SRSF on the grid ``t``, of shape (m,).
        t: Grid of length m.
        gamma: Warp on the grid ``t``, of shape (m,).
        f: Curve of ``q`` on the grid ``t``, of shape (m,), if available.

    Returns:
        The warped SRSF, of shape (m,).
    """
    if f is not None:
        return to_srsf(warp_curve(f, t, gamma), t)
    return warp_curve(q, t, gamma) * warp_to_psi(gamma, t)


def invert_warp(gamma: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse warp on the same grid, i.e. gamma^{-1} with gamma(gamma^{-1}(s)) = s.

    Args:
        gamma: Warp on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        The inverse warp on ``t``, of shape (m,).
    """
    gam = np.maximum.accumulate(gamma)  # guard against round-off non-monotonicity
    gam[0], gam[-1] = t[0], t[-1]
    return np.interp(t, gam, t)


def normalise_warp(gamma: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rescale a non-decreasing sequence into a warp of [t_0, t_end].

    Args:
        gamma: Non-decreasing values on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        The warp, of shape (m,), going from ``t[0]`` to ``t[-1]``; the
        identity warp if ``gamma`` is constant.
    """
    gam = np.maximum.accumulate(gamma)
    gam = gam - gam[0]
    span = gam[-1]
    if span <= 0:
        return t.copy()
    return t[0] + (t[-1] - t[0]) * gam / span


def warp_to_psi(gamma: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SRSF of a warp, psi = sqrt(gamma'), a point of the Hilbert sphere.

    Args:
        gamma: Warp on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        psi on the grid ``t``, of shape (m,). ``gamma'`` is computed by
        central differences and clipped at 0.
    """
    dgamma = np.clip(np.gradient(gamma, t), 0.0, None)
    return np.sqrt(dgamma)


def psi_to_warp(psi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of :func:`warp_to_psi`: gamma(t) = int_{t_0}^t psi^2, normalised.

    Args:
        psi: SRSF of a warp on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        The warp on ``t``, of shape (m,).
    """
    return normalise_warp(_cumulative_trapezoid(psi * psi, t), t)
