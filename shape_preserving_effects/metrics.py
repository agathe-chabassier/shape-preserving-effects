"""Shape metrics used to score estimated treatment-effect curves.

The treated responses of the simulations either rebound to baseline or
plateau after the drop. The three metrics below are defined, and stable,
for both shapes:

* ``depth``: the minimum of the curve (nadir level of a rebound, settled
  level of a plateau);
* ``onset_time``: the time at which the curve crosses half of its depth on
  the way down, by linear interpolation of the crossing that precedes the
  minimum;
* ``drop_speed``: the steepest downward slope, max_t -tau'(t), with tau' by
  central finite differences. Pointwise averaging of drops that occur at
  different times turns a steep drop into a shallow ramp, which this metric
  measures directly.

Curves are compared with the L^2 distance (:func:`l2_error`).
"""

from __future__ import annotations

import numpy as np

# The three shape metrics, in reporting order.
METRICS = ("depth", "onset_time", "drop_speed")


def _cross(t0: float, t1: float, y0: float, y1: float, level: float) -> float:
    """Time at which the segment (t0, y0)-(t1, y1) reaches ``level``.

    Args:
        t0: Time of the first end of the segment.
        t1: Time of the second end.
        y0: Value at ``t0``.
        y1: Value at ``t1``.
        level: Level to reach, between ``y0`` and ``y1``.

    Returns:
        The crossing time, by linear interpolation; the midpoint if the
        segment is flat.
    """
    if y1 == y0:
        return 0.5 * (t0 + t1)
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0)


def _left_crossing(y: np.ndarray, t: np.ndarray, idx: int, level: float) -> float:
    """Last down-crossing of ``level`` at or before index ``idx``.

    Args:
        y: Curve on the grid ``t``, of shape (m,).
        t: Grid of length m.
        idx: Grid index from which the search goes backwards.
        level: Level crossed.

    Returns:
        The crossing time; ``t[0]`` if the curve is below ``level`` from the
        start.
    """
    inside = y <= level
    for i in range(idx, 0, -1):
        if inside[i] and not inside[i - 1]:
            return float(_cross(t[i - 1], t[i], y[i - 1], y[i], level))
    return float(t[0])


def depth(tau: np.ndarray) -> float:
    """Minimum of the effect curve (signed).

    Args:
        tau: Curve, of shape (m,).

    Returns:
        Its minimum.
    """
    return float(np.min(tau))


def onset_time(tau: np.ndarray, t: np.ndarray) -> float:
    """Time of the half-depth crossing on the way down to the minimum.

    The half depth is only defined for a drop, so the result is ``nan`` when
    the minimum of the curve is not negative.

    Args:
        tau: Curve on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        The onset time, or ``nan`` for a curve that never drops below zero.
    """
    idx = int(np.argmin(tau))
    if tau[idx] >= 0:
        return float("nan")
    return _left_crossing(tau, t, idx, 0.5 * tau[idx])


def drop_speed(tau: np.ndarray, t: np.ndarray) -> float:
    """Steepest downward slope of the curve, max_t -tau'(t).

    Args:
        tau: Curve on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        The largest value of -tau', with tau' by central differences.
    """
    return float(np.max(-np.gradient(tau, t)))


def shape_metrics(tau: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """The three shape metrics of a curve.

    Args:
        tau: Curve on the grid ``t``, of shape (m,).
        t: Grid of length m.

    Returns:
        A dict keyed by :data:`METRICS`.
    """
    return {
        "depth": depth(tau),
        "onset_time": onset_time(tau, t),
        "drop_speed": drop_speed(tau, t),
    }


def l2_error(tau_hat: np.ndarray, tau_ref: np.ndarray, t: np.ndarray) -> float:
    """L^2 distance between two curves on the grid ``t`` (trapezoid rule).

    Args:
        tau_hat: First curve, of shape (m,).
        tau_ref: Second curve, of shape (m,).
        t: Grid of length m.

    Returns:
        The distance.
    """
    return float(np.sqrt(np.trapezoid((tau_hat - tau_ref) ** 2, t)))
