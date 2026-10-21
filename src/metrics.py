"""
Performance measures computed from the raw per-replicate simulation output.

Reads  results/raw/simulation_raw.csv
       docs/paper_values.csv          success rates read off the paper's figures
Writes results/summary.csv            one row per condition (success rate, MCSE, Wilson CI, diagnostics)
       results/thresholds.csv         one row per curve (empirical theta_10 / theta_50 / theta_90)
       results/collapse.csv           one row per theta: do the curves for different p overlap?
       results/paper_comparison.csv   our success rates next to the paper's

Run from the project root:
    python -m src.metrics
"""

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION_COLS = ["check", "design", "regime", "p", "k", "theta", "n", "theta_actual", "lam"]
CURVE_COLS = ["check", "design", "regime", "p", "k"]
CURVE_SET_COLS = ["check", "design", "regime"]   # curves for different p that should collapse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic measures
# ---------------------------------------------------------------------------

def monte_carlo_se(values):
    """Monte Carlo standard error of a mean over replicates (Lec05):
    sqrt( sum_i (phi_i - phi_bar)^2 / (R (R - 1)) ). For a 0/1 outcome this is
    sqrt(p_hat (1 - p_hat) / (R - 1))."""
    values = np.asarray(values, dtype=float)
    R = len(values)
    if R < 2:
        return np.nan
    return float(np.sqrt(np.sum((values - values.mean()) ** 2) / (R * (R - 1))))


def wilson_interval(successes, n_reps, z=1.96):
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    successes = np.asarray(successes, dtype=float)
    p_hat = successes / n_reps
    denom = 1.0 + z**2 / n_reps
    center = (p_hat + z**2 / (2 * n_reps)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n_reps + z**2 / (4 * n_reps**2)) / denom
    # exact endpoints at p_hat = 0 or 1 (avoids float round-off like 0.9999999999999999)
    low = np.where(successes == 0, 0.0, center - half)
    high = np.where(successes == n_reps, 1.0, center + half)
    return low, high


def threshold_crossing(x, y, level):
    """Smallest x at which the curve y(x) first reaches `level`, by linear interpolation.

    Returns NaN if the curve never reaches the level (or starts above it).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if y[0] >= level:
        return np.nan
    above = np.nonzero(y >= level)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def two_proportion_z(s1, n1, s2, n2):
    """|p1 - p2| / pooled SE, the standard two-proportion z statistic (0 if both are all 0 or all 1)."""
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else float(abs(p1 - p2) / se)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_conditions(raw):
    """One row per condition: success rate with MCSE and 95% Wilson CI, plus PDW diagnostics."""
    grouped = raw.groupby(CONDITION_COLS, sort=True)
    summary = grouped.agg(
        n_reps=("success", "size"),
        successes=("success", "sum"),
        mcse=("success", monte_carlo_se),
        dual_feasible_rate=("dual_feasible", "mean"),
        sign_consistent_rate=("sign_consistent", "mean"),
        max_abs_z_q10=("max_abs_z", lambda s: s.quantile(0.10)),
        max_abs_z_median=("max_abs_z", "median"),
        max_abs_z_q90=("max_abs_z", lambda s: s.quantile(0.90)),
        n_unconverged=("solver_converged", lambda s: int(s.eq(False).sum())),
        crosscheck_n=("success_sklearn", "count"),
    ).reset_index()

    summary["success_rate"] = summary["successes"] / summary["n_reps"]
    summary["ci_low"], summary["ci_high"] = wilson_interval(summary["successes"], summary["n_reps"])

    # PDW vs sklearn agreement on the cross-checked replicates
    checked = raw.dropna(subset=["success_sklearn"])
    if len(checked):
        agree = (checked["success"] == checked["success_sklearn"].astype(bool))
        agree_rate = agree.groupby([checked[c] for c in CONDITION_COLS]).mean().rename("crosscheck_agreement")
        summary = summary.merge(agree_rate.reset_index(), on=CONDITION_COLS, how="left")

    summary = summary.merge(summarize_failure_severity(raw), on=CONDITION_COLS, how="left")
    return summary


def summarize_failure_severity(raw, z=1.96):
    """How badly each PDW step fails, among the replicates where that step fails.

    violations : # wrong variables entering (|z_j| >= 1), given Step 3 fails
    sign_errors: # true variables zero / wrong sign,      given Step 4 fails
    Mean with a normal-approximation 95% CI (mean +/- z * Monte Carlo SE).
    Replicates with n <= k (counts undefined) are excluded.
    """
    parts = []
    for count_col, step_col, label in [("n_violations", "dual_feasible", "violations"),
                                       ("n_sign_errors", "sign_consistent", "sign_errors")]:
        failed = raw[raw[step_col].eq(False) & raw[count_col].notna()]
        stats = failed.groupby(CONDITION_COLS)[count_col].agg(["size", "mean", monte_carlo_se])
        half = z * stats["monte_carlo_se"]
        parts.append(pd.DataFrame({
            f"fail_reps_{label}": stats["size"],
            f"mean_{label}": stats["mean"],
            f"mean_{label}_lo": stats["mean"] - half,
            f"mean_{label}_hi": stats["mean"] + half,
        }))
    return pd.concat(parts, axis=1).reset_index()


def summarize_thresholds(summary, x_col="theta_actual"):
    """One row per curve: theta where the success rate first reaches 0.1, 0.5 and 0.9."""
    rows = []
    for keys, curve in summary.groupby(CURVE_COLS, sort=True):
        row = dict(zip(CURVE_COLS, keys))
        for level in (0.1, 0.5, 0.9):
            row[f"theta_{int(level * 100)}"] = threshold_crossing(curve[x_col], curve["success_rate"], level)
        row["transition_width"] = row["theta_90"] - row["theta_10"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_collapse(summary):
    """Curve collapse: at each theta, the largest gap between the curves for different p.

    For every pair of p values, the gap |P_hat_1 - P_hat_2| is scaled by the pooled
    two-proportion standard error; z_max < 2 means the curves overlap within Monte Carlo error.
    """
    rows = []
    for keys, at_theta in summary.groupby(CURVE_SET_COLS + ["theta"], sort=True):
        at_theta = at_theta.set_index("p")
        best = {"z_max": 0.0, "abs_diff": 0.0, "pair": ""}
        for p1, p2 in itertools.combinations(sorted(at_theta.index), 2):
            a, b = at_theta.loc[p1], at_theta.loc[p2]
            z = two_proportion_z(a["successes"], a["n_reps"], b["successes"], b["n_reps"])
            if z >= best["z_max"]:
                best = {"z_max": z, "abs_diff": abs(a["success_rate"] - b["success_rate"]),
                        "pair": f"{p1}-{p2}"}
        rows.append({**dict(zip(CURVE_SET_COLS + ["theta"], keys)), **best})
    return pd.DataFrame(rows)


def compare_with_paper(summary, paper, x_col="theta_actual"):
    """Our success rate (and Wilson CI) at the paper's theta values, by linear interpolation
    along our curve, next to the value read off the paper's figure."""
    rows = []
    for _, ref in paper.iterrows():
        curve = summary[(summary["design"] == ref["design"]) & (summary["regime"] == ref["regime"])
                        & (summary["p"] == ref["p"])].sort_values(x_col)
        if curve.empty:
            continue
        for check, c in curve.groupby("check"):
            x = c[x_col].to_numpy()
            ours = {col: float(np.interp(ref["theta"], x, c[col])) for col in ("success_rate", "ci_low", "ci_high")}
            rows.append({"figure": ref["figure"], "check": check, "design": ref["design"],
                         "regime": ref["regime"], "p": ref["p"], "theta": ref["theta"],
                         "success_rate_paper": ref["success_rate_paper"],
                         "success_rate_ours": ours["success_rate"],
                         "ci_low_ours": ours["ci_low"], "ci_high_ours": ours["ci_high"],
                         "difference": ours["success_rate"] - ref["success_rate_paper"]})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Summarize raw simulation results.")
    parser.add_argument("--raw", default="results/raw/simulation_raw.csv")
    parser.add_argument("--paper", default="docs/paper_values.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    summary = summarize_conditions(raw)
    thresholds = summarize_thresholds(summary)
    collapse = summarize_collapse(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    thresholds.to_csv(out_dir / "thresholds.csv", index=False)
    collapse.to_csv(out_dir / "collapse.csv", index=False)

    logger.info("Summarized %d replicates into %d conditions", len(raw), len(summary))
    logger.info("Solver did not converge on %d replicates", int(summary["n_unconverged"].sum()))
    logger.info("Empirical thresholds:\n%s", thresholds.round(3).to_string(index=False))
    for keys, c in collapse.groupby(CURVE_SET_COLS):
        logger.info("Curve collapse %s: z_max > 2 at %d of %d theta values (largest z = %.2f)",
                    "/".join(keys), int((c["z_max"] > 2).sum()), len(c), c["z_max"].max())

    if Path(args.paper).exists():
        paper = pd.read_csv(args.paper, comment="#")
        comparison = compare_with_paper(summary, paper)
        comparison.to_csv(out_dir / "paper_comparison.csv", index=False)
        logger.info("Comparison with the paper:\n%s", comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
