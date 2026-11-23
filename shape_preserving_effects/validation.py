"""Input checks shared by the modules of the package.

Each check tests one condition and raises a ``ValueError`` whose message
names that condition, so that a user can tell which requirement the input
broke.
"""

from __future__ import annotations

import numpy as np

# Seed of a random step: an integer, a numpy Generator, or None (not
# reproducible). Passed as it is to ``numpy.random.default_rng``.
Seed = int | np.random.Generator | None


def check_grid(t: np.ndarray) -> np.ndarray:
    """Return ``t`` as a float array after checking that it is a usable grid.

    The registration works on grid indices, so the grid must be uniform. Any
    interval and unit are allowed (for instance days since the index date).

    Args:
        t: Time grid.

    Returns:
        ``t`` as a 1-D float array.

    Raises:
        ValueError: If ``t`` is not 1-D, has fewer than 3 points, is not
            finite, is not strictly increasing, or is not equally spaced.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"t must be a 1-d grid of time points, got {t.ndim} dimension(s)")
    if t.size < 3:
        raise ValueError(f"t must contain at least 3 time points, got {t.size}")
    not_finite = np.flatnonzero(~np.isfinite(t))
    if not_finite.size:
        raise ValueError(f"t must be finite; t[{not_finite[0]}] = {t[not_finite[0]]}")
    steps = np.diff(t)
    not_increasing = np.flatnonzero(steps <= 0)
    if not_increasing.size:
        i = not_increasing[0]
        raise ValueError(
            f"t must be strictly increasing; t[{i}] = {t[i]} >= t[{i + 1}] = {t[i + 1]}"
        )
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            "t must be a uniform grid (equally spaced time points); "
            "interpolate the curves on a uniform grid first"
        )
    return t


def check_curves(curves: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return ``curves`` as an (n, m) float array after checking it against ``t``.

    Args:
        curves: Curves on the grid ``t``, one per row.
        t: Time grid of length m.

    Returns:
        ``curves`` as an (n, m) float array.

    Raises:
        ValueError: If ``curves`` is not 2-D, is empty, does not have one
            column per grid point, or contains a value that is not finite.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(
            f"curves must be a 2-d array, one curve per row; got {curves.ndim} dimension(s)"
        )
    if curves.shape[0] < 1:
        raise ValueError("curves must contain at least one curve")
    if curves.shape[1] != t.size:
        raise ValueError(
            f"each curve must have {t.size} points, one per grid point of t; "
            f"got {curves.shape[1]} (curves of shape {curves.shape})"
        )
    if not np.all(np.isfinite(curves)):
        raise ValueError(
            "curves contain NaN or infinite values; fill or interpolate the missing values first"
        )
    return curves


def normalised_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    """Weights of ``n`` units, checked and normalised to sum to one.

    Args:
        weights: Non-negative weights, one per unit, or ``None`` for equal
            weights.
        n: Number of units.

    Returns:
        The (n,) array of normalised weights.

    Raises:
        ValueError: If the shape is wrong, a weight is not finite or is
            negative, or all weights are zero.
    """
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have one entry per curve ({n}), got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    return w / w.sum()


def check_clip(clip: float) -> float:
    """Return ``clip`` after checking that it is a valid propensity clipping level.

    Args:
        clip: Propensity scores are clipped to [clip, 1 - clip].

    Returns:
        ``clip`` as a float.

    Raises:
        ValueError: If ``clip`` is not in [0, 0.5). At 0.5 or above the
            interval is empty or a single point and every unit gets the same
            weight, which silently removes the confounding adjustment.
    """
    clip = float(clip)
    if not 0.0 <= clip < 0.5:
        raise ValueError(f"clip must be in [0, 0.5), got {clip}")
    return clip


def as_design_matrix(X: np.ndarray) -> np.ndarray:
    """Covariates as an (n, d) float array; a 1-D input becomes one column."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X
