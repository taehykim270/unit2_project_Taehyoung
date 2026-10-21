"""Shared helpers for the test suite (not tests themselves)."""

import numpy as np
import pandas as pd

from src.simulation import CONFIG


def small_config(**overrides):
    """A copy of CONFIG shrunk to a few seconds of work: 5 reps, p = 128, three theta values."""
    config = dict(CONFIG, n_reps=5, crosscheck_reps=0, p_values=[128], thetas=[0.5, 1.0, 2.0])
    config.update(overrides)
    return config


def make_fake_raw(rates, n_reps=100, seed=0, p=128):
    """Raw-results table with a known success probability per theta (no simulation needed)."""
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
