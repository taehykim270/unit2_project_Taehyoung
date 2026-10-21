"""Method correctness: the three support-recovery checks agree, and the PDW behaves correctly in easy cases."""

import math

import numpy as np
import pytest

from src.dgps import generate_replicate, sample_size
from src.methods import (CHECKS, FullLasso, PDWClosedForm, PDWRestrictedLasso, SupportRecoveryCheck,
                         lasso_fit, regularization_lambda)

PDW_CHECKS = [PDWRestrictedLasso, PDWClosedForm]


def test_lambda_formula():
    n, p, k, sigma = 542, 512, 44, 0.5
    expected = math.sqrt(2 * 0.25 * math.log(44) * math.log(468) / 542)
    assert regularization_lambda(n, p, k, sigma) == pytest.approx(expected)


def test_registry_has_all_checks_with_unique_names():
    assert set(CHECKS) == {"pdw_restricted_lasso", "pdw_closed_form", "full_lasso"}
    for name, cls in CHECKS.items():
        assert issubclass(cls, SupportRecoveryCheck) and cls.name == name


@pytest.mark.parametrize("theta", [0.5, 0.9, 1.0, 1.2, 2.0])
def test_all_checks_agree(theta):
    """PDW (both versions) success <=> full Lasso has the correct signed support (Lemma 2)."""
    p, k = 128, 16
    n = sample_size(theta, p, k)
    lam = regularization_lambda(n, p, k, 0.5)
    checks = [PDWRestrictedLasso(), PDWClosedForm(), FullLasso()]
    rng = np.random.default_rng(10)
    for _ in range(10):
        d = generate_replicate(n, p, k, 0.5, 0.5, rng)
        verdicts = [c.check(d, lam)["success"] for c in checks]
        assert len(set(verdicts)) == 1


def test_restricted_solutions_agree_and_match_lasso_on_success():
    """When the PDW succeeds, both restricted solutions equal the full Lasso solution on S (eq. 51)."""
    p, k = 128, 16
    n = sample_size(2.4, p, k)
    lam = regularization_lambda(n, p, k, 0.5)
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(10):
        d = generate_replicate(n, p, k, 0.5, 0.5, rng)
        if not PDWClosedForm().check(d, lam)["success"]:
            continue
        X_S, beta_S = d["X"][:, d["support"]], d["beta"][d["support"]]
        closed, _ = PDWClosedForm().restricted_solution(X_S, d["y"], beta_S, lam)
        numeric, converged = PDWRestrictedLasso().restricted_solution(X_S, d["y"], beta_S, lam)
        full = lasso_fit(d["X"], d["y"], lam)[d["support"]]
        assert converged
        assert np.allclose(numeric, closed, atol=1e-6)
        assert np.allclose(full, closed, atol=1e-6)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_step3_matches_literal_eq10(check_cls):
    """Residual form of Step 3 equals eq. (10), which uses the n x n projection onto X_S's complement.

    Eq. (10) is written with z_S = sign(beta*_S); for the restricted-Lasso PDW that holds
    exactly when Step 4 passes, so those replicates are compared.
    """
    p, k = 128, 16
    check = check_cls()
    rng = np.random.default_rng(15)
    compared = 0
    for theta in (0.6, 1.0, 1.5):
        n = sample_size(theta, p, k)
        lam = regularization_lambda(n, p, k, 0.5)
        for _ in range(5):
            d = generate_replicate(n, p, k, 0.5, 0.5, rng)
            out = check.check(d, lam)
            if not out["sign_consistent"]:
                continue
            S = d["support"]
            X_S, X_Sc = d["X"][:, S], np.delete(d["X"], S, axis=1)
            z_S = np.sign(d["beta"][S])
            gram_inv = np.linalg.inv(X_S.T @ X_S)
            proj_perp = np.eye(n) - X_S @ gram_inv @ X_S.T
            Z = X_Sc.T @ (X_S @ gram_inv @ z_S + proj_perp @ (d["w"] / (lam * n)))
            assert np.max(np.abs(Z)) == pytest.approx(out["max_abs_z"], rel=1e-6)
            compared += 1
    assert compared > 0


def test_solver_nonconvergence_is_reported():
    """A ConvergenceWarning is caught and recorded instead of being printed."""
    p, k = 128, 16
    n = sample_size(1.0, p, k)
    lam = regularization_lambda(n, p, k, 0.5)
    d = generate_replicate(n, p, k, 0.5, 0.5, np.random.default_rng(16))
    assert PDWRestrictedLasso().check(d, lam)["solver_converged"] is True
    assert PDWClosedForm().check(d, lam)["solver_converged"] is True
    starved = PDWRestrictedLasso(solver_tol=1e-16, max_iter=1)   # cannot converge in one sweep
    assert starved.check(d, lam)["solver_converged"] is False


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_pdw_succeeds_with_lots_of_data(check_cls):
    """Sanity check: far above threshold with tiny noise the Lasso always recovers the support."""
    p, k, n = 128, 16, 2000
    lam = regularization_lambda(n, p, k, 0.5)
    rng = np.random.default_rng(12)
    for _ in range(5):
        d = generate_replicate(n, p, k, sigma=0.05, beta_min=0.5, rng=rng)
        assert check_cls().check(d, lam)["success"]


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_pdw_fails_when_n_not_above_k(check_cls):
    p, k = 128, 16
    d = generate_replicate(n=k, p=p, k=k, sigma=0.5, beta_min=0.5, rng=np.random.default_rng(13))
    out = check_cls().check(d, lam=0.5)
    assert out["success"] is False
    assert np.isnan(out["max_abs_z"])


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_success_requires_both_steps(check_cls):
    p, k = 128, 16
    check = check_cls()
    rng = np.random.default_rng(14)
    for theta in (0.6, 1.0, 1.5):
        n = sample_size(theta, p, k)
        lam = regularization_lambda(n, p, k, 0.5)
        for _ in range(10):
            d = generate_replicate(n, p, k, 0.5, 0.5, rng)
            out = check.check(d, lam)
            assert out["success"] == (out["dual_feasible"] and out["sign_consistent"])
            assert out["dual_feasible"] == (out["max_abs_z"] < 1)
            assert out["sign_consistent"] == (out["sign_margin"] > check.sign_tol)
            # failure counts are zero exactly when the corresponding step passes
            assert out["dual_feasible"] == (out["n_violations"] == 0)
            assert out["sign_consistent"] == (out["n_sign_errors"] == 0)
            assert 0 <= out["n_violations"] <= p - k and 0 <= out["n_sign_errors"] <= k
