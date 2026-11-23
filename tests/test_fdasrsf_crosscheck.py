"""Comparison with ``fdasrsf``, the reference implementation of elastic FDA.

On unweighted data, checks that

1. pairwise registration agrees with ``optimum_reparam`` (aligned distances
   and optimal warps);
2. the mean warp (Frechet mean of warps) agrees with ``SqrtMean``;
3. the phase-centered elastic mean agrees with the average of the aligned
   functions ``fn`` of ``fdawarp.srsf_align(center=True)``.

Exact equality is not expected: our dynamic programme restricts the local
slopes of the warps to the 11 values of ``registration._STEPS``, between 1/4
and 4, while ``fdasrsf`` searches a finer set of slopes, and the two packages
use different discretisations. A smooth warp is therefore followed less
closely by our piecewise-linear path: on the pair of ``test_registration``
the aligned distance left by our warp is about 3 times that of ``fdasrsf``
(0.10 against 0.03, both about 5 % or less of the unaligned distance, 2.0),
while the two warps differ by less than 0.01. The thresholds below were
calibrated on ``fdasrsf`` 2.7.4.

The tests are skipped when ``fdasrsf`` is not installed
(``pip install fdasrsf``).
"""

import numpy as np
import pytest

from shape_preserving_effects import (
    elastic_mean,
    from_srsf,
    l2_error,
    register,
    to_srsf,
)
from shape_preserving_effects.mean import _sphere_mean
from shape_preserving_effects.srsf import psi_to_warp, warp_to_psi

try:
    import fdasrsf as fs
    from fdasrsf import utility_functions as uf
except ImportError:  # not installed, or installed but broken
    pytest.skip("fdasrsf is not installed", allow_module_level=True)

t = np.linspace(0.0, 1.0, 81)
# Fine grid on which the aligned distances are evaluated.
T_FINE = np.linspace(0.0, 1.0, 641)


def bump(c: float, w: float = 0.08, s: np.ndarray = t) -> np.ndarray:
    return np.exp(-0.5 * ((s - c) / w) ** 2)


def _two_bumps(s: np.ndarray) -> np.ndarray:
    return bump(0.42, s=s) + 0.5 * bump(0.72, s=s)


def _retimed_two_bumps(s: np.ndarray) -> np.ndarray:
    """The curve ``_two_bumps`` re-timed by the warp s + 0.10 sin(pi s)."""
    return _two_bumps(s + 0.10 * np.sin(np.pi * s))


def _aligned_distance(gamma: np.ndarray) -> float:
    """|| q1 - (q2, gamma) || for a warp given on ``t``, evaluated on ``T_FINE``.

    Both curves are known in closed form, so the aligned curve f2 o gamma is
    evaluated exactly on the fine grid, with gamma interpolated linearly. The
    two packages are thus compared with the same warp action.
    """
    gamma_fine = np.interp(T_FINE, t, gamma)
    q1 = to_srsf(_two_bumps(T_FINE), T_FINE)
    q2_aligned = to_srsf(_retimed_two_bumps(gamma_fine), T_FINE)
    return float(np.sqrt(np.trapezoid((q1 - q2_aligned) ** 2, T_FINE)))


def test_registration() -> None:
    q1, q2 = to_srsf(_two_bumps(t), t), to_srsf(_retimed_two_bumps(t), t)
    gam_ours, _ = register(q1, q2, t)
    gam_fs = uf.optimum_reparam(q1, t, q2)
    d_ours, d_fs, d_unaligned = (_aligned_distance(g) for g in (gam_ours, gam_fs, t))
    # both warps remove almost all of the unaligned distance ...
    assert d_ours < 0.06 * d_unaligned
    assert d_fs < 0.06 * d_unaligned
    # ... ours within the factor that the 11 local slopes allow (measured: 3.2)
    assert d_ours < 4 * d_fs
    assert np.max(np.abs(gam_ours - gam_fs)) < 0.03


def test_karcher_mean_of_warps() -> None:
    gams = np.array([t + a * np.sin(np.pi * t) for a in (-0.12, -0.05, 0.04, 0.11)])
    psis = np.array([warp_to_psi(g, t) for g in gams])
    ours = psi_to_warp(_sphere_mean(psis, np.ones(len(gams)), t), t)
    mu_fs, *_ = uf.SqrtMean(gams.T)
    if hasattr(uf, "psi_to_warp"):
        theirs = uf.psi_to_warp(mu_fs, t)
    else:  # older fdasrsf: integrate psi^2
        theirs = np.concatenate(
            [[0.0], np.cumsum(0.5 * (mu_fs[1:] ** 2 + mu_fs[:-1] ** 2) * np.diff(t))]
        )
        theirs = theirs / theirs[-1]
    assert np.max(np.abs(ours - theirs)) < 0.02


def test_phase_centered_elastic_mean() -> None:
    # heights vary by a factor 10, so that the average of the aligned curves
    # (the phase-centered elastic mean) and the curve of the averaged SRSF
    # (the Karcher mean function; fdasrsf's ``mqn``) are clearly apart.
    rng = np.random.default_rng(1)
    centres = 0.45 + 0.08 * rng.standard_normal(24)
    heights = rng.uniform(0.3, 3.0, 24)
    F = np.array([h * (bump(c) - 0.6 * bump(c + 0.25)) for c, h in zip(centres, heights)])
    ours = elastic_mean(F, t, n_iter=3).mean
    obj = fs.fdawarp(F.T, t)
    obj.srsf_align(parallel=False, center=True)
    # fdasrsf returns the aligned functions fn = f_i o gamma_i (centred warps);
    # the phase-centered elastic mean is their average.
    theirs = np.asarray(obj.fn).mean(axis=1)
    curve_of_mqn = from_srsf(obj.mqn, t, f0=float(F[:, 0].mean()))
    assert l2_error(ours, theirs, t) < 0.015
    assert l2_error(ours, curve_of_mqn, t) > 0.03
    # and our Karcher mean function is fdasrsf's curve of the mean SRSF
    karcher = elastic_mean(F, t, n_iter=3).karcher_mean_function
    assert l2_error(karcher, curve_of_mqn, t) < 0.015
