"""Output validation: performance measures and theoretical constants are in valid ranges."""

import numpy as np
import pytest

from helpers import make_fake_raw
from src.dgps import toeplitz_covariance
from src.metrics import (monte_carlo_se, rho_lower, rho_upper, summarize_conditions, summarize_thresholds,
                         threshold_constants, wilson_interval)


@pytest.mark.parametrize("successes", [0, 1, 57, 100, 199, 200])
def test_wilson_interval_valid(successes):
    """The Wilson CI lies in [0, 1] and always contains p_hat, including at 0 and 1."""
    low, high = wilson_interval(successes, 200)
    p_hat = successes / 200
    assert 0.0 <= low <= p_hat <= high <= 1.0


def test_monte_carlo_se():
    """Monte Carlo SE follows the Lec05 formula; for 0/1 outcomes it equals sqrt(p (1 - p) / (R - 1))."""
    assert monte_carlo_se(np.zeros(200)) == 0.0
    assert monte_carlo_se(np.ones(200)) == 0.0
    half = np.r_[np.ones(100), np.zeros(100)]
    assert monte_carlo_se(half) == pytest.approx(np.sqrt(0.25 / 199))
    x = np.arange(10.0)
    assert monte_carlo_se(x) == pytest.approx(x.std(ddof=1) / np.sqrt(10))


def test_summary_ranges_and_thresholds():
    """Every summary column is in its valid range, and the threshold lies inside the theta grid."""
    raw = make_fake_raw({0.5: 0.0, 1.0: 0.5, 1.5: 1.0})
    summary = summarize_conditions(raw)
    assert len(summary) == 3
    for col in ["success_rate", "ci_low", "ci_high", "dual_feasible_rate", "sign_consistent_rate"]:
        assert summary[col].between(0, 1).all()
    assert (summary["ci_low"] <= summary["success_rate"]).all()
    assert (summary["success_rate"] <= summary["ci_high"]).all()
    assert (summary["crosscheck_agreement"] == 1.0).all()

    # failure severity: defined only where Step 3 fails, and always >= 1 there
    sev = summary.dropna(subset=["mean_violations"])
    assert (sev["fail_reps_violations"] > 0).all()
    assert (sev["mean_violations"] >= 1).all()
    assert (sev["mean_violations_lo"] <= sev["mean_violations"]).all()
    assert summary.loc[summary["theta"] == 1.5, "mean_violations"].isna().all()   # no failures there
    assert summary["mean_sign_errors"].isna().all()                              # Step 4 never fails

    assert (summary["n_unconverged"] == 0).all()
    assert (summary["mcse"] >= 0).all()

    thresholds = summarize_thresholds(summary)
    assert 0.5 < thresholds["theta_50"].iloc[0] < 1.5


def test_threshold_constants_identity_are_one():
    """For Sigma = I: gamma = C_min = C_max = 1 and theta_l = theta_u = 1 (Section IV-B)."""
    c = threshold_constants(np.eye(50), np.arange(0, 50, 5))
    for key in ["gamma", "c_min", "c_max", "rho_l", "rho_u", "theta_l", "theta_u"]:
        assert c[key] == pytest.approx(1.0)


def test_threshold_constants_toeplitz():
    """Toeplitz: 0 <= theta_l <= theta_u (eq. 30), and gamma stays within its worst-case bound."""
    mu, p = 0.1, 128
    Sigma = toeplitz_covariance(p, mu)
    rng = np.random.default_rng(3)
    for _ in range(5):
        c = threshold_constants(Sigma, np.sort(rng.choice(p, 30, replace=False)))
        assert 0 <= c["theta_l"] <= c["theta_u"]
        assert 1 - 2 * mu / (1 + mu**2) - 1e-9 <= c["gamma"] <= 1
    assert rho_upper(np.diag([1.0, 2.0])) == 2.0
    assert rho_lower(np.array([[1.0, 0.5], [0.5, 1.0]])) == pytest.approx(0.5)
