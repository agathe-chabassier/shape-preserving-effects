"""Properties of the dynamic-programming registration ``register``.

These hold for any set of local steps, so they keep protecting the
registration when ``_STEPS`` changes. Integer-valued SRSFs are included
because they make paths of exactly equal cost common.
"""

import numpy as np
import pytest

from shape_preserving_effects.registration import _sample_at_offset, _shift_right, register

# The smallest grid check_grid accepts, a tiny, a coarse and a typical size.
GRID_SIZES = (3, 6, 21, 101)


def _unaligned_distance(q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid((q1 - q2) ** 2, t)))


@pytest.mark.parametrize("m", GRID_SIZES)
def test_identity_when_curves_are_equal(m: int) -> None:
    t = np.linspace(0.0, 1.0, m)
    q = np.sin(2 * np.pi * t)
    gamma, dist = register(q, q, t)
    np.testing.assert_array_equal(gamma, t)
    assert dist == 0.0


@pytest.mark.parametrize("m", (3, 4, 6, 7, 10, 20, 40, 101))
@pytest.mark.parametrize("integer_valued", [False, True])
def test_warp_is_valid_and_never_worse_than_no_warp(m: int, integer_valued: bool) -> None:
    rng = np.random.default_rng(1000 + m)
    t = np.linspace(0.0, 2.0, m)
    for _ in range(50):
        if integer_valued:
            q1, q2 = rng.integers(0, 3, (2, m)).astype(float)
        else:
            q1, q2 = rng.normal(size=(2, m))
        gamma, dist = register(q1, q2, t)
        assert gamma[0] == t[0] and gamma[-1] == t[-1]
        slopes = np.diff(gamma) / np.diff(t)
        assert np.all(slopes >= 0.25 - 1e-12) and np.all(slopes <= 4.0 + 1e-12)
        # the identity warp is one of the paths searched (all (1, 1) steps)
        assert dist <= _unaligned_distance(q1, q2, t) + 1e-12


def test_rejects_a_non_uniform_grid() -> None:
    t = np.linspace(0.0, 1.0, 21) ** 2
    with pytest.raises(ValueError, match="uniform"):
        register(np.ones(21), np.ones(21), t)


def test_rejects_srsfs_that_do_not_match_the_grid() -> None:
    t = np.linspace(0.0, 1.0, 21)
    with pytest.raises(ValueError, match="q2"):
        register(np.ones(21), np.ones(20), t)
    q1 = np.ones(21)
    q1[4] = np.nan
    with pytest.raises(ValueError, match="q1"):
        register(q1, np.ones(21), t)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, [0.0, 1.0, 4.0, 9.0]),
        (1, [1.0, 4.0, 9.0, np.nan]),
        (3, [9.0, np.nan, np.nan, np.nan]),
        (0.5, [0.5, 2.5, 6.5, np.nan]),
        (2.5, [6.5, np.nan, np.nan, np.nan]),
        (4, [np.nan, np.nan, np.nan, np.nan]),
    ],
)
def test_sample_at_offset(offset: float, expected: list[float]) -> None:
    q = np.array([0.0, 1.0, 4.0, 9.0])
    np.testing.assert_array_equal(_sample_at_offset(q, offset), expected)


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        (0, [1.0, 2.0, 3.0]),
        (1, [np.inf, 1.0, 2.0]),
        (3, [np.inf, np.inf, np.inf]),
        (4, [np.inf, np.inf, np.inf]),
    ],
)
def test_shift_right(by: int, expected: list[float]) -> None:
    np.testing.assert_array_equal(_shift_right(np.array([1.0, 2.0, 3.0]), by), expected)
