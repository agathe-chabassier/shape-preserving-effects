"""Sample curves and averaging operators of the figure "Functional means
across data-generating processes" (appendix of the paper).

Seven samples of curves that share a common shape but differ in timing (and,
in some rows, in amplitude), and six ways of averaging them:

* ``cross_sectional``: the pointwise mean;
* ``elastic``: the phase-centered elastic mean of the paper
  (:func:`shape_preserving_effects.elastic_mean`);
* ``structural``: landmark registration of the curve maxima, then the
  pointwise mean (Kneip and Gasser, 1992);
* ``dba``: DTW barycenter averaging (Petitjean et al., 2011), computed by
  the majorize-minimize iterations of Schultz and Jain (2018);
* ``soft_dtw``: soft-DTW barycenter (Cuturi and Blondel, 2017), started from
  the DBA barycenter;
* ``wasserstein``: the exact one-dimensional Wasserstein-2 barycenter of
  the curves seen as densities (quantile averaging).

Everything is written with numpy only. DBA and soft-DTW follow the
implementations of the ``tslearn`` package (same initialisation, cost,
stopping rules and smoothing parameter); the soft-DTW objective is minimised
by L-BFGS.
"""

from __future__ import annotations

import numpy as np

from shape_preserving_effects import elastic_mean

N_CURVES = 30
N_POINTS = 100
T = np.linspace(0.0, 1.0, N_POINTS)
SEED = 42


# ----------------------------------------------------------------------
# Samples of curves
# ----------------------------------------------------------------------


def _gauss(t, centre, sd):
    return np.exp(-0.5 * ((t - centre) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def phase_variation(rng, n, t):
    """Gaussian bump (sd 0.05), centre ~ U[0.3, 0.7], amplitude ~ U[0.9, 1.1]."""
    centres = rng.uniform(0.3, 0.7, n)
    centres = centres - centres.mean() + 0.5
    amps = rng.uniform(0.9, 1.1, n)
    return np.array([a * _gauss(t, c, 0.05) for c, a in zip(centres, amps)])


def bimodal(rng, n, t):
    """Two bumps (sd 0.04) with independent centres ~ U[0.2, 0.4], U[0.6, 0.8]."""
    c1 = rng.uniform(0.2, 0.4, n)
    c1 = c1 - c1.mean() + 0.3
    c2 = rng.uniform(0.6, 0.8, n)
    c2 = c2 - c2.mean() + 0.7
    return np.array([_gauss(t, a, 0.04) + _gauss(t, b, 0.04) for a, b in zip(c1, c2)])


def linear(rng, n, t):
    """Lines s (t - h) + 2 with slope s ~ U[1, 5] and shift h ~ U[-0.2, 0.2]."""
    slopes = rng.uniform(1.0, 5.0, n)
    shifts = rng.uniform(-0.2, 0.2, n)
    shifts = shifts - shifts.mean()
    return np.array([s * (t - h) + 2.0 for s, h in zip(slopes, shifts)])


def sinusoid(rng, n, t):
    """sin(4 pi t + phi), phi ~ U[0, 2 pi]."""
    phis = rng.uniform(0.0, 2 * np.pi, n)
    return np.array([np.sin(4 * np.pi * t + p) for p in phis])


def step(rng, n, t):
    """Unit step at a time ~ U[0.3, 0.7], plus N(0, 0.01^2) noise."""
    taus = rng.uniform(0.3, 0.7, n)
    taus = taus - taus.mean() + 0.5
    return np.array([(t >= tau).astype(float) + rng.normal(0.0, 0.01, t.size) for tau in taus])


def chirp(rng, n, t):
    """sin(k t^2 + phi), k ~ U[15, 25], phi ~ U[0, pi]."""
    ks = rng.uniform(15.0, 25.0, n)
    phis = rng.uniform(0.0, np.pi, n)
    return np.array([np.sin(k * t**2 + p) for k, p in zip(ks, phis)])


def damped(rng, n, t):
    """exp(-d t) sin(6 pi t + phi), d ~ U[2, 5], phi ~ U[0, 2 pi]."""
    ds = rng.uniform(2.0, 5.0, n)
    phis = rng.uniform(0.0, 2 * np.pi, n)
    return np.array([np.exp(-d * t) * np.sin(6 * np.pi * t + p) for d, p in zip(ds, phis)])


#: (key, row label, generator), in the order of the rows of the figure.
SAMPLES = [
    ("phase", "Phase\nvariation", phase_variation),
    ("bimodal", "Two bumps", bimodal),
    ("linear", "Shifted lines", linear),
    ("sine", "Sinusoid", sinusoid),
    ("step", "Step", step),
    ("chirp", "Chirp", chirp),
    ("damped", "Damped\noscillation", damped),
]


def draw_samples(seed=SEED, n=N_CURVES, t=T):
    """One sample per data-generating process, each from its own seed."""
    return {
        key: gen(np.random.default_rng([seed, k]), n, t) for k, (key, _, gen) in enumerate(SAMPLES)
    }


# ----------------------------------------------------------------------
# Structural (landmark) mean
# ----------------------------------------------------------------------


def _main_peak(f):
    """Index of the most prominent local maximum of ``f`` (global maximum if
    ``f`` has no interior local maximum)."""
    m = f.size
    best, best_prom = int(np.argmax(f)), -np.inf
    for i in range(1, m - 1):
        if not (f[i] > f[i - 1] and f[i] >= f[i + 1]):
            continue
        # prominence: height above the higher of the two minima separating
        # the peak from a higher point (or from the end of the curve)
        left = f[:i][::-1]
        hi = np.nonzero(left > f[i])[0]
        left_min = left[: hi[0]].min() if hi.size else left.min()
        right = f[i + 1 :]
        hi = np.nonzero(right > f[i])[0]
        right_min = right[: hi[0]].min() if hi.size else right.min()
        prom = f[i] - max(left_min, right_min)
        if prom > best_prom:
            best, best_prom = i, prom
    return best


def structural_mean(F, t):
    """Landmark registration on the main maximum of each curve.

    Each curve is warped by the piecewise-linear map sending the median
    landmark time to its own landmark time, and the warped curves are
    averaged pointwise.
    """
    eps = 1e-3 * (t[-1] - t[0])
    peaks = t[[_main_peak(f) for f in F]]
    target = float(np.clip(np.median(peaks), t[0] + eps, t[-1] - eps))
    aligned = np.empty_like(F)
    for i, f in enumerate(F):
        p = float(np.clip(peaks[i], t[0] + eps, t[-1] - eps))
        gamma = np.interp(t, [t[0], target, t[-1]], [t[0], p, t[-1]])
        aligned[i] = np.interp(gamma, t, f)
    return aligned.mean(axis=0)


# ----------------------------------------------------------------------
# Dynamic time warping: DBA and soft-DTW barycenters
# ----------------------------------------------------------------------


def _diagonals(m):
    """Cells (i, j), 1 <= i, j <= m, grouped by anti-diagonal i + j."""
    out = []
    for k in range(2, 2 * m + 1):
        i = np.arange(max(1, k - m), min(m, k - 1) + 1)
        out.append((i, k - i))
    return out


def _sq_cost(z, X):
    """Squared distances D[k, i, j] = (z_i - X[k, j])^2."""
    return (z[None, :, None] - X[:, None, :]) ** 2


def _dtw_paths(z, X):
    """DTW cost and optimal path between ``z`` and every row of ``X``."""
    n, m = X.shape
    D = _sq_cost(z, X)
    R = np.full((n, m + 1, m + 1), np.inf)
    R[:, 0, 0] = 0.0
    for i, j in _diagonals(m):
        prev = np.minimum(np.minimum(R[:, i - 1, j - 1], R[:, i - 1, j]), R[:, i, j - 1])
        R[:, i, j] = D[:, i - 1, j - 1] + prev
    A = R[:, 1:, 1:]
    paths = []
    for k in range(n):
        i, j = m - 1, m - 1
        path = [(i, j)]
        while (i, j) != (0, 0):
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                c = (A[k, i - 1, j - 1], A[k, i - 1, j], A[k, i, j - 1])
                a = int(np.argmin(c))
                i, j = (i - 1, j - 1) if a == 0 else ((i - 1, j) if a == 1 else (i, j - 1))
            path.append((i, j))
        paths.append(np.array(path[::-1]))
    return A[:, -1, -1], paths


def dba_mean(F, max_iter=50, tol=1e-5):
    """DTW barycenter averaging, started from the pointwise mean.

    At each iteration every curve is aligned to the current barycenter by
    DTW, and each point of the barycenter becomes the average of the curve
    values matched to it. Stops when the mean squared DTW cost changes by
    less than ``tol``.
    """
    n, m = F.shape
    z = F.mean(axis=0)
    cost_prev = np.inf
    for _ in range(max_iter):
        costs, paths = _dtw_paths(z, F)
        cost = float(costs.mean())
        num, cnt = np.zeros(m), np.zeros(m)
        for k, p in enumerate(paths):
            np.add.at(num, p[:, 0], F[k, p[:, 1]])
            np.add.at(cnt, p[:, 0], 1.0)
        z = num / cnt
        if abs(cost_prev - cost) < tol or cost > cost_prev:
            break
        cost_prev = cost
    return z


def _soft_dtw_value_grad(z, X, gamma):
    """Sum over the rows of ``X`` of soft-DTW(z, X_k), and its gradient in z."""
    n, m = X.shape
    D = _sq_cost(z, X)
    R = np.full((n, m + 2, m + 2), np.inf)
    R[:, 0, 0] = 0.0
    diags = _diagonals(m)
    for i, j in diags:
        a = np.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]]) / -gamma
        amax = a.max(axis=0)
        R[:, i, j] = D[:, i - 1, j - 1] - gamma * (amax + np.log(np.exp(a - amax).sum(axis=0)))
    value = R[:, m, m].sum()

    # backward recursion of Cuturi and Blondel (2017, Algorithm 2)
    Dp = np.zeros((n, m + 2, m + 2))
    Dp[:, 1 : m + 1, 1 : m + 1] = D
    R[:, :, m + 1] = -np.inf
    R[:, m + 1, :] = -np.inf
    R[:, m + 1, m + 1] = R[:, m, m]
    E = np.zeros((n, m + 2, m + 2))
    E[:, m + 1, m + 1] = 1.0
    for i, j in reversed(diags):
        r = R[:, i, j]
        wa = np.exp((R[:, i + 1, j] - r - Dp[:, i + 1, j]) / gamma)
        wb = np.exp((R[:, i, j + 1] - r - Dp[:, i, j + 1]) / gamma)
        wc = np.exp((R[:, i + 1, j + 1] - r - Dp[:, i + 1, j + 1]) / gamma)
        E[:, i, j] = E[:, i + 1, j] * wa + E[:, i, j + 1] * wb + E[:, i + 1, j + 1] * wc
    E = E[:, 1 : m + 1, 1 : m + 1]
    grad = 2.0 * (z * E.sum(axis=(0, 2)) - np.einsum("kij,kj->i", E, X))
    return value, grad


def _lbfgs(fun, x0, max_iter=200, tol=1e-5, memory=10):
    """Minimise ``fun`` (returning value and gradient) by L-BFGS with a
    backtracking (Armijo) line search."""
    x = x0.copy()
    f, g = fun(x)
    S, Y = [], []
    for _ in range(max_iter):
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Y)):
            a = s @ q / (y @ s)
            alphas.append(a)
            q -= a * y
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            q += (a - y @ q / (y @ s)) * s
        d = -q
        if g @ d >= 0:  # not a descent direction: restart
            S, Y, d = [], [], -g
        step = 1.0
        while True:
            x_new = x + step * d
            f_new, g_new = fun(x_new)
            if f_new <= f + 1e-4 * step * (g @ d) or step < 1e-10:
                break
            step *= 0.5
        s, y = x_new - x, g_new - g
        if s @ y > 1e-12:
            S.append(s)
            Y.append(y)
            if len(S) > memory:
                S.pop(0)
                Y.pop(0)
        converged = (f - f_new) <= tol * max(abs(f_new), abs(f), 1.0)
        x, f, g = x_new, f_new, g_new
        if converged:
            break
    return x


def soft_dtw_mean(F, gamma=0.1, max_iter=200, tol=1e-5):
    """Soft-DTW barycenter with smoothing ``gamma``, started from DBA."""
    z0 = dba_mean(F, max_iter=15, tol=1e-4)
    return _lbfgs(lambda z: _soft_dtw_value_grad(z, F, gamma), z0, max_iter=max_iter, tol=tol)


# ----------------------------------------------------------------------
# Wasserstein barycenter
# ----------------------------------------------------------------------


def wasserstein_mean(F, t, n_quantiles=4000):
    """Exact Wasserstein-2 barycenter of the curves seen as densities on t.

    The curves are shifted by one common constant to be non-negative,
    normalised to unit mass, and their quantile functions are averaged; the
    density of the result is rescaled by the average mass and shifted back.
    """
    shift = max(0.0, -float(F.min())) + 1e-9
    P = F + shift
    mass = P.sum(axis=1, keepdims=True)
    P = P / mass
    u = np.linspace(0.0, 1.0, n_quantiles)
    Qs = np.empty((len(F), n_quantiles))
    for k, p in enumerate(P):
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        cdf_u, idx = np.unique(cdf, return_index=True)
        Qs[k] = np.interp(u, cdf_u, t[idx])
    cdf_bar = np.interp(t, Qs.mean(axis=0), u, left=0.0, right=1.0)
    dens = np.clip(np.gradient(cdf_bar, t), 0.0, None)
    dens /= dens.sum()
    return dens * float(mass.mean()) - shift


# ----------------------------------------------------------------------
# All means
# ----------------------------------------------------------------------

#: (key, column title), in the order of the columns of the figure.
MEANS = [
    ("cross_sectional", "Cross-sectional"),
    ("elastic", "Elastic (ours)"),
    ("structural", "Structural"),
    ("dba", "DBA"),
    ("soft_dtw", "Soft-DTW"),
    ("wasserstein", "Wasserstein"),
]


ELASTIC_ITER = 10  # Step-1 alternations of the elastic mean (converged here)


def all_means(F, t, n_iter=ELASTIC_ITER):
    """The six means of the sample ``F`` (dict keyed as :data:`MEANS`).

    The elastic mean runs up to ``n_iter`` register / average alternations,
    enough for its Step-1 mean shape to converge on these samples (the
    simulations of the paper use two, for speed).
    """
    return {
        "cross_sectional": F.mean(axis=0),
        "elastic": elastic_mean(F, t, n_iter=n_iter).mean,
        "structural": structural_mean(F, t),
        "dba": dba_mean(F),
        "soft_dtw": soft_dtw_mean(F),
        "wasserstein": wasserstein_mean(F, t),
    }
