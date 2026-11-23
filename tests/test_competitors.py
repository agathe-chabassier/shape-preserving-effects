"""The two competitors against direct, slower implementations of their formulas."""

import numpy as np

from shape_preserving_effects import fate_drfos, srvf_kernel_ate
from shape_preserving_effects.competitors import (
    _median_bandwidth,
    _moving_average,
    _ovk_fit_predict,
    _pairwise_sq_dists,
    _random_folds,
    _register_by_group,
)
from shape_preserving_effects.datasets import drop_rebound_trial
from shape_preserving_effects.estimators import _logistic_fit

t = np.linspace(0.0, 1.0, 41)
TRIAL = drop_rebound_trial(n=60, t=t, seed=3)


def test_ovk_fit_predict_solves_the_kronecker_system() -> None:
    rng = np.random.default_rng(0)
    n_tr, n_pred, m, lam = 7, 4, 5, 0.3
    B, C = rng.normal(size=(n_tr, n_tr)), rng.normal(size=(m, m))
    K_in, K_y = B @ B.T, C @ C.T  # two positive definite kernels
    K_cross = rng.normal(size=(n_pred, n_tr))
    Y_in = rng.normal(size=(n_tr, m)) + 3.0
    # sample-major vectorisation: vec(Y)[i * m + j] = Y[i, j]
    system = np.kron(K_in, K_y) + n_tr * lam * np.eye(n_tr * m)
    alpha = np.linalg.solve(system, Y_in.ravel())
    expected = (np.kron(K_cross, K_y) @ alpha).reshape(n_pred, m)
    np.testing.assert_allclose(_ovk_fit_predict(K_in, Y_in, K_cross, lam, K_y), expected)


def _direct_ovk(
    K_in: np.ndarray, Y_in: np.ndarray, K_cross: np.ndarray, lam: float, K_y: np.ndarray
) -> np.ndarray:
    """The ridge regression of the released code, solved as one linear system."""
    n_tr, m = Y_in.shape
    system = np.kron(K_in, K_y) + n_tr * lam * np.eye(n_tr * m)
    alpha = np.linalg.solve(system, Y_in.ravel())
    return (np.kron(K_cross, K_y) @ alpha).reshape(K_cross.shape[0], m)


def test_srvf_kernel_ate_follows_the_released_code() -> None:
    Y, A, X = TRIAL.Y, TRIAL.A.astype(float), TRIAL.X[:, None]
    lambdas, seed, n = (1e-3, 1.0, 1e3), 0, len(TRIAL.A)
    effect, arm1, arm0 = srvf_kernel_ate(t, Y, A, X, lambdas=lambdas, rng=seed, return_arms=True)
    # the same estimate written directly
    smooth = _moving_average(Y)
    sq = (X - X.T) ** 2
    bandwidth = np.median(np.sqrt(sq[np.triu_indices(n, k=1)]))
    covariate_kernel = np.exp(-sq / (2 * bandwidth**2))
    K = covariate_kernel * np.outer(A, A)  # input kernel, as in the released code
    # ridge parameter: fit on 80 %, registration included, prediction error on the other 20 %
    perm = np.random.default_rng(seed).permutation(n)
    te, tr = perm[: int(0.2 * n)], perm[int(0.2 * n) :]
    aligned_tr = _register_by_group(smooth[tr], A[tr], t, n_iter=2)
    errors = []
    for lam in lambdas:
        pred = _direct_ovk(
            K[np.ix_(tr, tr)], aligned_tr, K[np.ix_(te, tr)], lam, np.cov(aligned_tr.T)
        )
        errors.append(np.mean((pred - Y[te]) ** 2))
    lam = lambdas[int(np.argmin(errors))]
    # refit on all units, average of the predictions with every unit treated
    aligned = _register_by_group(smooth, A, t, n_iter=2)
    pred1 = _direct_ovk(K, aligned, covariate_kernel * A[None, :], lam, np.cov(aligned.T))
    np.testing.assert_allclose(arm1, pred1.mean(axis=0), atol=1e-8)
    pred0 = _direct_ovk(K, aligned, covariate_kernel * (0.0 * A)[None, :], lam, np.cov(aligned.T))
    np.testing.assert_allclose(arm0, pred0.mean(axis=0), atol=1e-8)
    np.testing.assert_allclose(effect, arm1 - arm0)


def test_pairwise_distances_and_bandwidth() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    direct = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(_pairwise_sq_dists(a, b), direct, atol=1e-12)
    d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(axis=2))
    expected = np.median(d[np.triu_indices(6, k=1)])
    np.testing.assert_allclose(_median_bandwidth(_pairwise_sq_dists(a, a)), expected)


def test_drfos_is_the_cross_fitted_aipw_formula() -> None:
    Y, A, X = TRIAL.Y, TRIAL.A.astype(float), TRIAL.X[:, None]
    n, clip = len(A), 0.02
    folds = _random_folds(n, 5, 0)
    design = np.column_stack([np.ones(n), X])
    e_hat, m1, m0 = np.empty(n), np.empty(Y.shape), np.empty(Y.shape)
    for k in range(5):
        train, test = folds != k, folds == k
        beta = _logistic_fit(X[train], A[train])
        e_hat[test] = 1.0 / (1.0 + np.exp(-design[test] @ beta))
        for a, m_a in ((1.0, m1), (0.0, m0)):
            in_arm = train & (A == a)
            coefs = np.linalg.lstsq(design[in_arm], Y[in_arm], rcond=None)[0]
            m_a[test] = design[test] @ coefs
    e_hat = np.clip(e_hat, clip, 1 - clip)
    aipw1 = m1 + (A / e_hat)[:, None] * (Y - m1)
    aipw0 = m0 + ((1 - A) / (1 - e_hat))[:, None] * (Y - m0)
    expected = (aipw1 - aipw0).mean(axis=0)
    np.testing.assert_allclose(fate_drfos(t, Y, A, X, rng=0), expected, atol=1e-12)
