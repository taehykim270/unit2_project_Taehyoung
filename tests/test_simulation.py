"""Reproducibility and end-to-end sanity checks for the simulation driver."""

import pandas as pd
import pytest

import json

from src.methods import CHECKS
from src.simulation import (CONFIG, build_conditions, make_check, replicate_rng, run_condition,
                            run_simulation, save_results)
from src.dgps import generate_replicate


def small_config(**overrides):
    config = dict(CONFIG, n_reps=5, crosscheck_reps=0, p_values=[128], thetas=[0.5, 1.0, 2.0])
    config.update(overrides)
    return config


def test_same_seed_gives_identical_results():
    a = run_simulation(small_config())
    b = run_simulation(small_config())
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_results():
    a = run_simulation(small_config())
    b = run_simulation(small_config(master_seed=CONFIG["master_seed"] + 1))
    assert not a["max_abs_z"].equals(b["max_abs_z"])


def test_adding_conditions_does_not_change_existing_results():
    small = run_simulation(small_config())
    big = run_simulation(small_config(p_values=[128, 256], thetas=[0.3, 0.5, 1.0, 2.0]))
    merged = small.merge(big, on=["design", "regime", "p", "theta", "rep"], suffixes=("", "_big"))
    assert len(merged) == len(small)
    assert (merged["max_abs_z"].fillna(-1) == merged["max_abs_z_big"].fillna(-1)).all()


def test_rows_are_self_describing():
    """Each row carries its seed key (and the key regenerates that row's data) and solver status."""
    df = run_simulation(small_config())
    assert df["seed_key"].is_unique
    assert df["solver_converged"].all()

    row = df.iloc[7]
    design, regime, p, theta_code, rep = (int(v) for v in row["seed_key"].split("-"))
    assert (p, theta_code, rep) == (row["p"], round(1000 * row["theta"]), row["rep"])
    rng = replicate_rng(CONFIG["master_seed"], row["design"], row["regime"], row["p"], row["theta"], row["rep"])
    data = generate_replicate(row["n"], row["p"], row["k"], CONFIG["sigma"], CONFIG["beta_min"], rng)
    rerun = make_check(CONFIG).check(data, row["lam"])
    assert rerun["max_abs_z"] == row["max_abs_z"]


def test_saved_metadata_has_provenance(tmp_path):
    config = small_config(output_dir=str(tmp_path))
    save_results(run_simulation(config), config, elapsed=1.0)
    meta = json.loads((tmp_path / "simulation_config.json").read_text())
    for key in ["config", "timestamp_utc", "python_version", "packages", "rng", "git"]:
        assert key in meta
    assert meta["rng"]["bit_generator"] == "PCG64"
    assert {"numpy", "scipy", "scikit-learn", "pandas"} <= set(meta["packages"])


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_every_check_builds_from_config(name):
    """Each check has its own parameter entry in CONFIG, so switching checks is a one-line change."""
    check = make_check(CONFIG, name)
    assert check.name == name


def test_build_conditions_is_fully_crossed():
    config = small_config(p_values=[128, 256, 512])
    assert len(build_conditions(config)) == 3 * 3


@pytest.mark.parametrize("theta, low, high", [(0.3, 0.0, 0.0), (2.4, 0.8, 1.0)])
def test_sanity_conditions(theta, low, high):
    """Far below threshold success is impossible; far above it is (nearly) certain."""
    config = small_config(n_reps=50)
    cond = next(c for c in build_conditions(dict(config, thetas=[theta])))
    rate = pd.DataFrame(run_condition(cond, config))["success"].mean()
    assert low <= rate <= high
