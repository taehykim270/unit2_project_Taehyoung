"""Method correctness: the PDW checks give the actual Lasso's answer, and recover the truth in easy cases."""

import numpy as np
import pytest

from src.dgps import generate_replicate, sample_size
from src.methods import FullLasso, PDWClosedForm, PDWRestrictedLasso, lasso_fit, regularization_lambda

PDW_CHECKS = [PDWRestrictedLasso, PDWClosedForm]


@pytest.mark.parametrize("theta", [0.5, 0.9, 1.0, 1.2, 2.0])
def test_all_checks_agree(theta):
    """Both PDW versions give the same verdict as solving the full Lasso (Lemma 2)."""
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
    """Our residual form of Step 3 equals eq. (10) as written, with the n x n projection matrix.

    Eq. (10) uses z_S = sign(beta*_S); for the restricted-Lasso PDW that holds exactly when
    Step 4 passes, so those replicates are compared.
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


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_pdw_succeeds_with_lots_of_data(check_cls):
    """Sanity case with a known answer: far above threshold, with tiny noise, recovery always succeeds."""
    p, k, n = 128, 16, 2000
    lam = regularization_lambda(n, p, k, 0.5)
    rng = np.random.default_rng(12)
    for _ in range(5):
        d = generate_replicate(n, p, k, sigma=0.05, beta_min=0.5, rng=rng)
        assert check_cls().check(d, lam)["success"]
