"""
Main simulation orchestration for reproducing Wainwright (2009), Section VII.

For every condition (design, sparsity regime, p, theta) in CONFIG, draw n_reps
independent data sets, run the support-recovery check named in CONFIG (a
primal-dual witness by default), and save one row per replicate to results/raw/.
Summaries and figures are produced by separate steps.

Run from the project root:
    python -m src.simulation                  # full run, settings from CONFIG
    python -m src.simulation --n-reps 10      # quick test run
    python -m src.simulation --check pdw_closed_form --output-dir results/raw_closed
"""

import argparse
import copy
import itertools
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dgps import generate_replicate, rescaled_sample_size, sample_size, sparsity_k
from src.methods import CHECKS, FullLasso, regularization_lambda

# ---------------------------------------------------------------------------
# Configuration: every simulation parameter lives here
# ---------------------------------------------------------------------------

CONFIG = {
    "master_seed": 607,
    "n_reps": 200,                       # replicates per condition (paper: 200)
    "sigma": 0.5,                        # noise SD
    "beta_min": 0.5,                     # |beta*_i| on the support
    "alpha": 0.40,                       # sparsity constant
    "delta": 0.75,                       # exponent for fractional power sparsity
    "designs": ["identity"],             # add "toeplitz" for Fig. 3
    "regimes": ["fractional"],           # add "linear", "sublinear" for Fig. 2
    "p_values": [128, 256, 512],
    "thetas": [round(0.1 * i, 1) for i in range(1, 25)],   # 0.1, 0.2, ..., 2.4
    "check": "pdw_restricted_lasso",     # support-recovery check (methods.CHECKS); paper's PDW order
    "check_params": {                    # constructor arguments for each check
        # zero_tol: |coefficient| below this counts as 0; solver_tol: sklearn tolerance
        "pdw_restricted_lasso": {"zero_tol": 1e-8, "solver_tol": 1e-10},
        "pdw_closed_form": {},           # exact linear solve: no tolerances (sign_tol = 0)
        "full_lasso": {"zero_tol": 1e-8, "solver_tol": 1e-10},
    },
    "crosscheck_reps": 0,                # first m reps per condition also solved by FullLasso
                                         # (0: PDW is exact by Lemmas 2-3; the full-Lasso check lives in tests/)
    "output_dir": "results/raw",
}

# Integer codes used to build per-condition seeds (keep fixed once results exist)
DESIGN_CODES = {"identity": 0, "toeplitz": 1}
REGIME_CODES = {"linear": 0, "sublinear": 1, "fractional": 2}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

SEED_RULE = ("numpy.random.default_rng(SeedSequence(master_seed, spawn_key=(DESIGN_CODES[design], "
             "REGIME_CODES[regime], p, round(1000 * theta), rep))); one PCG64 stream per replicate, "
             "drawing X, then beta* (support, signs), then w")


def seed_key(design, regime, p, theta, rep):
    """spawn_key identifying one replicate's random stream."""
    return (DESIGN_CODES[design], REGIME_CODES[regime], int(p), int(round(theta * 1000)), int(rep))


def replicate_rng(master_seed, design, regime, p, theta, rep):
    """Independent Generator for one replicate.

    The seed depends only on the condition and replicate index (not on the order or
    number of conditions), so adding new conditions later never changes old results.
    """
    key = seed_key(design, regime, p, theta, rep)
    return np.random.default_rng(np.random.SeedSequence(master_seed, spawn_key=key))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def build_conditions(config):
    """Fully crossed list of condition dicts with k, n and lambda_n filled in."""
    conditions = []
    for design, regime, p, theta in itertools.product(
            config["designs"], config["regimes"], config["p_values"], config["thetas"]):
        k = sparsity_k(p, regime, config["alpha"], config["delta"])
        n = sample_size(theta, p, k)
        conditions.append({
            "design": design, "regime": regime, "p": p, "k": k, "theta": theta, "n": n,
            "theta_actual": rescaled_sample_size(n, p, k),   # after rounding n up
            "lam": regularization_lambda(n, p, k, config["sigma"]),
        })
    return conditions


def make_check(config, name=None):
    """Instantiate a support-recovery check (default: the one named in config["check"])."""
    name = config["check"] if name is None else name
    return CHECKS[name](**config["check_params"][name])


def run_condition(cond, config):
    """Run all replicates of one condition; return a list of per-replicate records."""
    check = make_check(config)
    crosscheck = make_check(config, FullLasso.name)
    records = []
    for rep in range(config["n_reps"]):
        rng = replicate_rng(config["master_seed"], cond["design"], cond["regime"],
                            cond["p"], cond["theta"], rep)
        data = generate_replicate(cond["n"], cond["p"], cond["k"], config["sigma"],
                                  config["beta_min"], rng, design=cond["design"])
        result = check.check(data, cond["lam"])

        # Cross-check against a full numerical Lasso fit on the first few reps
        success_sklearn = np.nan
        if rep < config["crosscheck_reps"]:
            success_sklearn = crosscheck.check(data, cond["lam"])["success"]

        key = seed_key(cond["design"], cond["regime"], cond["p"], cond["theta"], rep)
        records.append({**cond, "check": check.name, "rep": rep, "seed_key": "-".join(map(str, key)),
                        **result, "success_sklearn": success_sklearn})
    return records


def run_simulation(config):
    """Run every condition in config and return the raw results as a DataFrame."""
    conditions = build_conditions(config)
    logger.info("Running %d conditions x %d reps = %d replicates",
                len(conditions), config["n_reps"], len(conditions) * config["n_reps"])

    records = []
    for cond in tqdm(conditions, desc="conditions", unit="cond"):
        records.extend(run_condition(cond, config))
    return pd.DataFrame(records)


def git_info():
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}   # e.g. no commits yet


def provenance():
    """Environment and randomness metadata needed to reproduce a run."""
    packages = {name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")}
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "packages": packages,
            "rng": {"bit_generator": np.random.default_rng().bit_generator.__class__.__name__,
                    "seed_rule": SEED_RULE,
                    "design_codes": DESIGN_CODES, "regime_codes": REGIME_CODES},
            "git": git_info()}


def save_results(df, config, elapsed):
    """Write raw per-replicate results plus the exact config and provenance of the run."""
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "simulation_raw.csv"
    df.to_csv(raw_path, index=False)

    meta = {"config": config, "n_rows": len(df), "elapsed_seconds": round(elapsed, 1),
            **provenance()}
    with open(out_dir / "simulation_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %d rows to %s", len(df), raw_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Lasso support-recovery simulation.")
    parser.add_argument("--n-reps", type=int, help="override replicates per condition")
    parser.add_argument("--check", choices=sorted(CHECKS), help="override the support-recovery check")
    parser.add_argument("--crosscheck-reps", type=int, help="override sklearn cross-check reps")
    parser.add_argument("--output-dir", help="override output directory")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    config = copy.deepcopy(CONFIG)   # deep: never mutate the nested dicts of the global CONFIG
    if args.n_reps is not None:
        config["n_reps"] = args.n_reps
    if args.check is not None:
        config["check"] = args.check
    if args.crosscheck_reps is not None:
        config["crosscheck_reps"] = args.crosscheck_reps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    start = time.time()
    df = run_simulation(config)
    elapsed = time.time() - start
    save_results(df, config, elapsed)

    # Quick log of PDW vs sklearn agreement
    checked = df.dropna(subset=["success_sklearn"])
    if len(checked):
        agree = (checked["success"] == checked["success_sklearn"].astype(bool)).mean()
        logger.info("%s vs full Lasso agreement: %.4f on %d replicates",
                    config["check"], agree, len(checked))
    n_unconverged = int(df["solver_converged"].eq(False).sum())
    logger.info("Solver did not converge on %d of %d replicates", n_unconverged, len(df))
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()