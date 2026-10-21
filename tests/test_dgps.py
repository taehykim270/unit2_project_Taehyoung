"""DGP verification: generated data has the properties described in the paper."""

import math

import numpy as np
import pytest

from src.dgps import (generate_beta, generate_gaussian_design, generate_noise, generate_replicate,
                      rescaled_sample_size, sample_size, sparsity_k)


@pytest.mark.parametrize("p, expected", [
    (128, {"linear": 52, "sublinear": 14, "fractional": 16}),
    (256, {"linear": 103, "sublinear": 23, "fractional": 26}),
    (512, {"linear": 205, "sublinear": 39, "fractional": 44}),
])
def test_sparsity_k_values(p, expected):
    for regime, k in expected.items():
        assert sparsity_k(p, regime, alpha=0.40, delta=0.75) == k


def test_sample_size_matches_theta():
    p, k = 512, 44
    n = sample_size(1.0, p, k)
    assert n == math.ceil(2 * 44 * math.log(512 - 44))
    # rounding n up can only increase theta, by less than one sample's worth
    theta = rescaled_sample_size(n, p, k)
    assert 1.0 <= theta < 1.0 + 1.0 / (2 * k * math.log(p - k))


def test_beta_has_k_nonzeros_of_size_beta_min():
    rng = np.random.default_rng(0)
    beta, support = generate_beta(p=512, k=44, beta_min=0.5, rng=rng)
    assert np.count_nonzero(beta) == 44
    assert np.array_equal(np.flatnonzero(beta), support)
    assert set(np.abs(beta[support])) == {0.5}


def test_beta_signs_are_balanced():
    rng = np.random.default_rng(1)
    signs = np.concatenate([np.sign(generate_beta(512, 44, 0.5, rng)[0]) for _ in range(200)])
    signs = signs[signs != 0]
    assert abs(np.mean(signs > 0) - 0.5) < 0.03


def test_gaussian_design_moments():
    X = generate_gaussian_design(2000, 200, np.random.default_rng(2))
    assert X.shape == (2000, 200)
    assert abs(X.mean()) < 0.01
    assert abs(X.var() - 1.0) < 0.01
    # columns are uncorrelated: off-diagonal of X^T X / n near zero
    gram = X.T @ X / 2000
    off_diag = gram[~np.eye(200, dtype=bool)]
    assert np.max(np.abs(off_diag)) < 0.15


def test_noise_sd():
    w = generate_noise(100_000, sigma=0.5, rng=np.random.default_rng(3))
    assert abs(w.std() - 0.5) < 0.01


def test_replicate_satisfies_model():
    d = generate_replicate(n=300, p=128, k=16, sigma=0.5, beta_min=0.5, rng=np.random.default_rng(4))
    assert d["X"].shape == (300, 128) and d["y"].shape == (300,)
    assert np.allclose(d["y"], d["X"] @ d["beta"] + d["w"])
