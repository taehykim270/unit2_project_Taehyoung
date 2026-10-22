"""
Data-generating functions for reproducing Wainwright (2009), Section VII.

Model:  y = X beta* + w,   w ~ N(0, sigma^2 I_n)

All randomness goes through a numpy Generator passed in by the caller,
so results are reproducible given the seed set in simulation.py.
"""

import math
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Problem-size helpers: sparsity k=k(p) and sample size n=n(theta, p, k)
# ---------------------------------------------------------------------------

def sparsity_k(p, regime, alpha, delta=None):
    """Number of nonzeros k as a function of p (paper, Section VII).

    regime: "linear"     k = ceil(alpha * p)
            "sublinear"  k = ceil(alpha * p / log(alpha * p))
            "fractional" k = ceil(alpha * p ** delta)
    """
    if regime == "linear":
        x = alpha * p
    elif regime == "sublinear":
        x = alpha * p / math.log(alpha * p)
    elif regime == "fractional":
        if delta is None:
            raise ValueError("fractional regime requires delta")
        x = alpha * p ** delta
    else:
        raise ValueError(f"unknown sparsity regime: {regime}")
    # subtract a tiny tolerance so float error (e.g. 100.00000000001) doesn't bump k by 1
    return int(math.ceil(x - 1e-9))


def sample_size(theta, p, k):
    """n = ceil(2 * theta * k * log(p - k)), natural log."""
    return int(math.ceil(2.0 * theta * k * math.log(p - k)))


def rescaled_sample_size(n, p, k):
    """theta(n, p, k) = n / (2 k log(p - k)); inverse of sample_size (up to rounding)."""
    return n / (2.0 * k * math.log(p - k))


# ---------------------------------------------------------------------------
# Setting 1: standard Gaussian ensemble (Sigma = I_p) - used for Fig. 1 and Fig. 2
# ---------------------------------------------------------------------------

def generate_gaussian_design(n, p, rng):
    """X in R^{n x p} with i.i.d. N(0, 1) entries."""
    return rng.standard_normal((n, p))


# ---------------------------------------------------------------------------
# Setting 2: Toeplitz family - used for Fig. 3
# ---------------------------------------------------------------------------

def toeplitz_covariance(p, mu):
    """Sigma_ij = mu^{|i - j|}   (eq. 46)."""
    idx = np.arange(p)
    return mu ** np.abs(idx[:, None] - idx[None, :])


@lru_cache(maxsize=None)
def toeplitz_cholesky(p, mu):
    """Lower-triangular L with L L^T = Sigma; computed once per (p, mu) and reused."""
    return np.linalg.cholesky(toeplitz_covariance(p, mu))


def generate_toeplitz_design(n, p, mu, rng):
    """X in R^{n x p} with i.i.d. rows N(0, Sigma), Sigma Toeplitz: X = Z L^T, Z standard Gaussian."""
    Z = rng.standard_normal((n, p))
    return Z @ toeplitz_cholesky(p, mu).T


# ---------------------------------------------------------------------------
# Setting 3 (extension): sparse random design - used for Fig. 5
# ---------------------------------------------------------------------------

def generate_sparse_design(n, p, q, rng):
    """X in R^{n x p} with i.i.d. entries +1/sqrt(q), -1/sqrt(q) w.p. q/2 each, and 0 otherwise.

    Entries have mean 0 and variance 1, so Sigma = I as in the identity design, but X is
    non-Gaussian and mostly zeros (cf. the paper's follow-up on sparse measurement matrices, ref. [36]).
    """
    u = rng.random((n, p))
    return (np.where(u < q / 2, 1.0, 0.0) - np.where((u >= q / 2) & (u < q), 1.0, 0.0)) / np.sqrt(q)


# ---------------------------------------------------------------------------
# True coefficient vector and noise
# ---------------------------------------------------------------------------

def generate_beta(p, k, beta_min, rng):
    """beta* with a uniformly random support S of size k and entries +/- beta_min.

    Returns (beta, support) where support is the sorted index array S.
    """
    support = np.sort(rng.choice(p, size=k, replace=False))
    signs = rng.choice(np.array([-1.0, 1.0]), size=k)
    beta = np.zeros(p)
    beta[support] = signs * beta_min
    return beta, support


def _generate_noise(n, sigma, rng):
    """w ~ N(0, sigma^2 I_n)."""
    return sigma * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# One full replicate
# ---------------------------------------------------------------------------

def design_covariance(p, design, toeplitz_mu=None):
    """Population covariance Sigma of the rows of X for a given design."""
    if design in ("identity", "sparse"):    # the sparse design also has Sigma = I
        return np.eye(p)
    if design.startswith("toeplitz"):
        if toeplitz_mu is None:
            raise ValueError("toeplitz design requires toeplitz_mu")
        return toeplitz_covariance(p, toeplitz_mu)
    raise ValueError(f"unknown design: {design}")


def generate_replicate(n, p, k, sigma, beta_min, rng, design="identity", toeplitz_mu=None, sparse_q=None):
    """Draw one data set (X, y, beta*, w) for a single simulation replicate.

    A new X, support, sign pattern and noise vector are drawn every call,
    matching the paper.
    Draw order is fixed (X, beta*, w) so a given rng state always gives the same data.
    """
    if design == "identity":
        X = generate_gaussian_design(n, p, rng)
    elif design.startswith("toeplitz"):   # "toeplitz", "toeplitz_strong": differ only in mu
        if toeplitz_mu is None:
            raise ValueError("toeplitz design requires toeplitz_mu")
        X = generate_toeplitz_design(n, p, toeplitz_mu, rng)
    elif design == "sparse":
        if sparse_q is None:
            raise ValueError("sparse design requires sparse_q")
        X = generate_sparse_design(n, p, sparse_q, rng)
    else:
        raise ValueError(f"unknown design: {design}")

    beta, support = generate_beta(p, k, beta_min, rng)
    w = _generate_noise(n, sigma, rng)
    y = X @ beta + w

    return {"X": X, "y": y, "beta": beta, "support": support, "w": w}
