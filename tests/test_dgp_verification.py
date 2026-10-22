"""DGP verification: generated data has the properties the paper's design specifies."""

import numpy as np
import pytest

from src.dgps import (generate_beta, generate_gaussian_design, generate_sparse_design, generate_toeplitz_design, sparsity_k,
                      toeplitz_cholesky, toeplitz_covariance)


@pytest.mark.parametrize("p, expected", [
    (128, {"linear": 52, "sublinear": 14, "fractional": 16}),
    (256, {"linear": 103, "sublinear": 23, "fractional": 26}),
    (512, {"linear": 205, "sublinear": 39, "fractional": 44}),
])
def test_sparsity_k_values(p, expected):
    """k follows the three sparsity formulas of Section VII (alpha = 0.40, delta = 0.75)."""
    for regime, k in expected.items():
        assert sparsity_k(p, regime, alpha=0.40, delta=0.75) == k


def test_beta_has_k_nonzeros_of_size_beta_min():
    """beta* has exactly k nonzeros, all equal to +/- beta_min, on the returned support."""
    rng = np.random.default_rng(0)
    beta, support = generate_beta(p=512, k=44, beta_min=0.5, rng=rng)
    assert np.count_nonzero(beta) == 44
    assert np.array_equal(np.flatnonzero(beta), support)
    assert set(np.abs(beta[support])) == {0.5}


def test_gaussian_design_moments():
    """Identity design: entries have mean 0, variance 1, and columns are uncorrelated."""
    X = generate_gaussian_design(2000, 200, np.random.default_rng(2))
    assert X.shape == (2000, 200)
    assert abs(X.mean()) < 0.01
    assert abs(X.var() - 1.0) < 0.01
    gram = X.T @ X / 2000
    off_diag = gram[~np.eye(200, dtype=bool)]
    assert np.max(np.abs(off_diag)) < 0.15


def test_toeplitz_design_has_target_covariance():
    """Toeplitz design: rows have covariance Sigma_ij = mu^|i-j| (eq. 46)."""
    mu, p = 0.10, 50
    Sigma = toeplitz_covariance(p, mu)
    L = toeplitz_cholesky(p, mu)
    assert np.allclose(L @ L.T, Sigma)
    X = generate_toeplitz_design(200_000, p, mu, np.random.default_rng(5))
    sample_cov = X.T @ X / X.shape[0]
    assert np.max(np.abs(sample_cov - Sigma)) < 0.02
    assert abs(np.mean(np.diag(sample_cov, 1)) - mu) < 0.005        # lag-1 correlation ~ mu
    assert abs(np.mean(np.diag(sample_cov, 2)) - mu**2) < 0.005     # lag-2 correlation ~ mu^2


def test_sparse_design_moments():
    """Sparse random design: entries 0 w.p. 1 - q, otherwise +/- 1/sqrt(q); mean 0, variance 1."""
    q = 0.1
    X = generate_sparse_design(2000, 200, q, np.random.default_rng(7))
    assert abs(np.mean(X == 0) - (1 - q)) < 0.005
    assert set(np.unique(np.round(X * np.sqrt(q), 12))) == {-1.0, 0.0, 1.0}
    assert abs(X.mean()) < 0.01
    assert abs(X.var() - 1.0) < 0.02
