"""Output validation: performance measures are in valid ranges and behave as expected."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (compare_with_paper, monte_carlo_se, summarize_collapse, summarize_conditions,
                         summarize_thresholds, threshold_crossing, two_proportion_z, wilson_interval)


def test_monte_carlo_se():
    """Lec05 formula; for 0/1 outcomes it equals sqrt(p (1 - p) / (R - 1))."""
    assert monte_carlo_se(np.zeros(200)) == 0.0
    assert monte_carlo_se(np.ones(200)) == 0.0
    half = np.r_[np.ones(100), np.zeros(100)]
    assert monte_carlo_se(half) == pytest.approx(np.sqrt(0.25 / 199))
    x = np.arange(10.0)
    assert monte_carlo_se(x) == pytest.approx(x.std(ddof=1) / np.sqrt(10))


def test_two_proportion_z():
    assert two_proportion_z(0, 200, 0, 200) == 0.0
    assert two_proportion_z(100, 200, 100, 200) == 0.0
    assert two_proportion_z(60, 200, 140, 200) > 7      # 0.3 vs 0.7 with R = 200: clearly different


@pytest.mark.parametrize("successes", [0, 1, 57, 100, 199, 200])
def test_wilson_interval_valid(successes):
    low, high = wilson_interval(successes, 200)
    p_hat = successes / 200
    assert 0.0 <= low <= p_hat <= high <= 1.0


def test_threshold_crossing_linear_curve():
    x = np.linspace(0, 2, 21)
    y = np.clip(x / 2, 0, 1)            # reaches 0.5 exactly at x = 1
    assert threshold_crossing(x, y, 0.5) == pytest.approx(1.0)
    assert np.isnan(threshold_crossing(x, y * 0.4, 0.5))   # never reaches 0.5


def make_fake_raw(rates, n_reps=100, seed=0, p=128):
    """Raw-results table with a known success probability per theta."""
    rng = np.random.default_rng(seed)
    rows = []
    for theta, rate in rates.items():
        succ = rng.random(n_reps) < rate
        for rep in range(n_reps):
            rows.append({"check": "pdw_restricted_lasso", "solver_converged": True,
                         "design": "identity", "regime": "fractional", "p": p, "k": 16,
                         "theta": theta, "n": int(100 * theta), "theta_actual": theta, "lam": 0.1,
                         "rep": rep, "success": succ[rep], "dual_feasible": succ[rep],
                         "sign_consistent": True, "max_abs_z": 1.5 - theta / 2, "sign_margin": 0.1,
                         "n_violations": 0 if succ[rep] else 1 + rep % 3, "n_sign_errors": 0,
                         "success_sklearn": succ[rep] if rep < 5 else np.nan})
    return pd.DataFrame(rows)


def test_summary_ranges_and_thresholds():
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


def test_checks_are_never_pooled():
    """Rows from two different checks stay separate conditions."""
    a = make_fake_raw({1.0: 0.2}, seed=1)
    b = make_fake_raw({1.0: 0.8}, seed=2).assign(check="pdw_closed_form")
    summary = summarize_conditions(pd.concat([a, b]))
    assert len(summary) == 2 and set(summary["check"]) == {"pdw_restricted_lasso", "pdw_closed_form"}


def test_collapse_detects_overlap_and_separation():
    rates = {0.5: 0.1, 1.0: 0.5, 1.5: 0.9}
    same = pd.concat([make_fake_raw(rates, n_reps=200, seed=s, p=p) for s, p in [(1, 128), (2, 256)]])
    shifted = pd.concat([make_fake_raw(rates, n_reps=200, seed=1, p=128),
                         make_fake_raw({0.5: 0.6, 1.0: 0.95, 1.5: 0.1}, n_reps=200, seed=2, p=256)])
    z_same = summarize_collapse(summarize_conditions(same))["z_max"]
    z_shift = summarize_collapse(summarize_conditions(shifted))["z_max"]
    assert (z_same < 3).all()
    assert (z_shift > 5).all()


def test_compare_with_paper_interpolates_our_curve():
    summary = summarize_conditions(make_fake_raw({0.8: 0.0, 1.2: 1.0}, n_reps=50))
    paper = pd.DataFrame([{"figure": "1b", "design": "identity", "regime": "fractional",
                           "p": 128, "theta": 1.0, "success_rate_paper": 0.6}])
    comp = compare_with_paper(summary, paper)
    assert comp["success_rate_ours"].iloc[0] == pytest.approx(0.5)     # halfway between 0 and 1
    assert comp["difference"].iloc[0] == pytest.approx(-0.1)
    assert comp["ci_low_ours"].iloc[0] <= 0.5 <= comp["ci_high_ours"].iloc[0]
