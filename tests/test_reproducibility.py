"""Reproducibility (same seed -> same results) and end-to-end sanity conditions."""

import pandas as pd
import pytest

from helpers import small_config
from src.dgps import generate_replicate
from src.simulation import CONFIG, build_conditions, make_check, replicate_rng, run_condition, run_simulation


def test_same_seed_gives_identical_results():
    """Rerunning with the same master seed gives an identical raw-results table."""
    a = run_simulation(small_config())
    b = run_simulation(small_config())
    pd.testing.assert_frame_equal(a, b)


def test_adding_conditions_does_not_change_existing_results():
    """Adding designs, regimes, p and theta values (Figs. 2-3) leaves the Fig. 1 rows unchanged."""
    small = run_simulation(small_config(designs=["identity"], regimes=["fractional"]))
    big = run_simulation(small_config(p_values=[128, 256], thetas=[0.3, 0.5, 1.0, 2.0]))
    merged = small.merge(big, on=["design", "regime", "p", "theta", "rep"], suffixes=("", "_big"))
    assert len(merged) == len(small)
    assert (merged["max_abs_z"].fillna(-1) == merged["max_abs_z_big"].fillna(-1)).all()


def test_rows_are_self_describing():
    """Each row's seed key regenerates exactly that row's data and result, for every design."""
    df = run_simulation(small_config())
    assert df["seed_key"].is_unique
    assert df["solver_converged"].all()

    for design in CONFIG["designs"]:
        row = df[df["design"] == design].iloc[7]
        _, _, p, theta_code, rep = (int(v) for v in row["seed_key"].split("-"))
        assert (p, theta_code, rep) == (row["p"], round(1000 * row["theta"]), row["rep"])
        rng = replicate_rng(CONFIG["master_seed"], row["design"], row["regime"], row["p"], row["theta"], row["rep"])
        data = generate_replicate(row["n"], row["p"], row["k"], CONFIG["sigma"], CONFIG["beta_min"], rng,
                                  design=row["design"], toeplitz_mu=CONFIG["toeplitz_mu"].get(row["design"]), sparse_q=CONFIG["sparse_q"])
        rerun = make_check(CONFIG).check(data, row["lam"])
        assert rerun["max_abs_z"] == row["max_abs_z"]


@pytest.mark.parametrize("design, theta, low, high", [
    ("identity", 0.3, 0.0, 0.0), ("toeplitz", 0.3, 0.0, 0.0), ("toeplitz_strong", 0.3, 0.0, 0.0), ("sparse", 0.3, 0.0, 0.0),
    ("identity", 2.4, 0.8, 1.0), ("toeplitz", 2.4, 0.8, 1.0),
    # no "far above" case for toeplitz_strong: with mu = 0.5 the threshold lies beyond theta = 2.4
])
def test_sanity_conditions(design, theta, low, high):
    """Far below the threshold success never happens; far above it (paper's designs) it is nearly certain."""
    config = small_config(n_reps=50, designs=[design], regimes=["fractional"])
    cond = next(c for c in build_conditions(dict(config, thetas=[theta])))
    rate = pd.DataFrame(run_condition(cond, config))["success"].mean()
    assert low <= rate <= high
