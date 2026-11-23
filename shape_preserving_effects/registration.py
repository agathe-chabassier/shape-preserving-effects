"""Elastic registration of two SRSFs by dynamic programming.

``register(q1, q2, t)`` returns the warp ``gamma`` that minimises
``|| q1 - (q2, gamma) ||_{L^2}`` over piecewise-linear warps whose local
slopes lie in [1/4, 4]. This is the bi-Lipschitz warping family used in the
paper. The search runs over monotone paths on the grid (t_i, t_j), built
from the local steps listed in ``_STEPS``; costs are vectorised over the
grid, so one registration on ~100 grid points takes a few milliseconds.

The dynamic programme has four stages, one function each: the cost of every
local step from every node (:func:`_step_costs`), the forward pass
(:func:`_cheapest_paths`), the backtracking (:func:`_backtrack`) and the
interpolation of the path into a warp (:func:`register`).
"""

from __future__ import annotations

import numpy as np

from shape_preserving_effects.validation import check_grid

# Local steps (di, dj) of the dynamic programme, in grid units.
# Their slopes dj / di range from 1/4 to 4.
_STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 4), (4, 3))


def _sample_at_offset(q: np.ndarray, offset: float) -> np.ndarray:
    """Values of ``q`` at the grid index ``j + offset``, for every index j.

    Args:
        q: Function on the grid, of shape (m,).
        offset: Non-negative shift in grid units; an integer for ``q1``, a
            fraction for the warped ``q2``.

    Returns:
        The (m,) array whose entry j is ``q`` at index ``j + offset``, by
        linear interpolation between the two neighbouring grid points; NaN
        where ``j + offset`` is past the end of the grid.
    """
    m = q.size
    base = int(np.floor(offset))
    frac = offset - base
    sampled = np.full(m, np.nan)
    n_inside = m - base - (1 if frac > 0 else 0)
    if n_inside <= 0:
        return sampled
    if frac > 0:
        # Explicit linear interpolation, not np.interp: np.interp rounds
        # differently in the last bits, and that is enough to change which
        # of two equally cheap paths the dynamic programme keeps.
        sampled[:n_inside] = (1 - frac) * q[base : base + n_inside] + frac * q[
            base + 1 : base + 1 + n_inside
        ]
    else:
        sampled[:n_inside] = q[base : base + n_inside]
    return sampled


def _step_costs(q1: np.ndarray, q2: np.ndarray, h: float) -> list[np.ndarray]:
    """Segment costs of every local step from every node.

    Args:
        q1: Target SRSF on the grid, of shape (m,).
        q2: SRSF to align, of shape (m,).
        h: Grid spacing.

    Returns:
        One (m, m) array per step of ``_STEPS``. For step k = (di, dj),
        ``costs[k][i, j]`` is the trapezoid approximation of the integral of
        (q1 - (q2, gamma))^2 over [t_i, t_{i+di}] when gamma is the linear
        segment from node (i, j) to node (i + di, j + dj): on that segment
        (q2, gamma) is q2 at the fractional index j + r * dj / di, times
        sqrt(dj / di), for r = 0, ..., di. Entries whose segment leaves the
        grid are NaN.
    """
    m = q1.size
    costs = []
    for di, dj in _STEPS:
        slope = dj / di
        sqrt_slope = np.sqrt(slope)
        trapezoid_weights = np.full(di + 1, h)
        trapezoid_weights[0] = trapezoid_weights[-1] = 0.5 * h
        segment_cost = np.zeros((m, m))
        for r in range(di + 1):
            q1_on_segment = _sample_at_offset(q1, r)  # a vector over i
            warped_q2 = _sample_at_offset(q2, r * slope) * sqrt_slope  # a vector over j
            segment_cost += (
                trapezoid_weights[r] * (q1_on_segment[:, None] - warped_q2[None, :]) ** 2
            )
        costs.append(segment_cost)
    return costs


def _shift_right(values: np.ndarray, by: int) -> np.ndarray:
    """Move a row ``by`` places to the right, filling the left with +inf.

    Used for "node (i, j) is reached from node (i - di, j - dj)": the cost of
    arriving at column j is read at column j - dj.

    Args:
        values: Row of costs, of shape (m,).
        by: Number of places, at least 0. A step longer than the grid
            (``by >= m``) leaves every position at +inf.

    Returns:
        The (m,) array with ``shifted[j] = values[j - by]`` for j >= by and
        +inf elsewhere.
    """
    shifted = np.full(values.size, np.inf)
    if by < values.size:
        shifted[by:] = values[: values.size - by]
    return shifted


def _cheapest_paths(costs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass of the dynamic programme.

    Args:
        costs: Segment costs of every step, from :func:`_step_costs`.

    Returns:
        ``(cumulative_cost, best_step)``, two (m, m) arrays.
        ``cumulative_cost[i, j]`` is the cost of the cheapest path from node
        (0, 0) to node (i, j), +inf if no path reaches it;
        ``best_step[i, j]`` is the index in ``_STEPS`` of the last step of
        that path, -1 if there is none. Among equally cheap steps the first
        one in ``_STEPS`` is kept.
    """
    m = costs[0].shape[0]
    cumulative_cost = np.full((m, m), np.inf)
    best_step = np.full((m, m), -1, dtype=np.int8)
    cumulative_cost[0, 0] = 0.0
    for i in range(1, m):
        cheapest = np.full(m, np.inf)
        cheapest_step = np.full(m, -1, dtype=np.int8)
        for k, (di, dj) in enumerate(_STEPS):
            if di > i:
                continue
            # Cost of reaching row i by step k from every node of row i - di.
            # A segment that leaves the grid has a NaN cost, and NaN never
            # compares as smaller, so it never wins.
            candidate = _shift_right(cumulative_cost[i - di] + costs[k][i - di], dj)
            better = candidate < cheapest
            cheapest[better] = candidate[better]
            cheapest_step[better] = k
        cumulative_cost[i] = cheapest
        best_step[i] = cheapest_step
    return cumulative_cost, best_step


def _backtrack(best_step: np.ndarray) -> tuple[list[int], list[int]]:
    """Nodes of the cheapest path, from (0, 0) to (m - 1, m - 1).

    Args:
        best_step: Last step of the cheapest path to every node, from
            :func:`_cheapest_paths`.

    Returns:
        ``(path_i, path_j)``: the grid indices of the nodes of the path, in
        increasing order.
    """
    m = best_step.shape[0]
    i, j = m - 1, m - 1
    path_i, path_j = [i], [j]
    while i > 0:
        di, dj = _STEPS[best_step[i, j]]
        i, j = i - di, j - dj
        path_i.append(i)
        path_j.append(j)
    path_i.reverse()
    path_j.reverse()
    return path_i, path_j


def register(q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimal warp aligning ``q2`` to ``q1``.

    Args:
        q1: Target SRSF on the grid ``t``, of shape (m,).
        q2: SRSF to align, of shape (m,).
        t: Uniform grid of length m (at least 3 points).

    Returns:
        ``(gamma, dist)``: the optimal warp on ``t``, of shape (m,), and the
        aligned distance ``|| q1 - (q2, gamma) ||_{L^2}``.

    Raises:
        ValueError: If ``t`` is not a uniform grid, or if ``q1`` or ``q2``
            does not have one finite value per grid point.
    """
    t = check_grid(t)
    m = t.size
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    for name, q in (("q1", q1), ("q2", q2)):
        if q.shape != (m,):
            raise ValueError(
                f"{name} must have one value per grid point ({m}), got shape {q.shape}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} contains NaN or infinite values")
    costs = _step_costs(q1, q2, float(t[1] - t[0]))
    cumulative_cost, best_step = _cheapest_paths(costs)
    path_i, path_j = _backtrack(best_step)
    gamma = np.interp(t, t[path_i], t[path_j])
    dist = float(np.sqrt(max(cumulative_cost[m - 1, m - 1], 0.0)))
    return gamma, dist
