"""
Figures built from results/summary.csv (run `python -m src.metrics` first).
Each figure is saved as PDF (vector, for reports) and PNG (for the README).

    fig1_success_vs_n_and_theta   publication figure: Fig. 1(a) and 1(b) of the paper
    fig1_pdw_diagnostics          diagnostic figure: why the Lasso fails below threshold

Only rows produced by one support-recovery check are plotted (default: CONFIG["check"]).

Run from the project root:
    python -m src.figures
    python -m src.figures --check pdw_closed_form
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt
import pandas as pd

from src.methods import CHECKS
from src.simulation import CONFIG

FORMATS = ("pdf", "png")

# Fixed color + marker per problem size p (color follows p, never its rank in a plot)
P_STYLE = {
    128: {"color": "#2a78d6", "marker": "o"},
    256: {"color": "#eb6834", "marker": "s"},
    512: {"color": "#1baf7a", "marker": "^"},
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
REFERENCE = "#52514e"

# Failure-severity points need enough failing replicates for a stable mean
MIN_FAIL_REPS = 10

logger = logging.getLogger(__name__)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    })


def save_figure(fig, out_stem):
    """Save one figure in every format in FORMATS (out_stem has no extension)."""
    for ext in FORMATS:
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved %s", path)
    plt.close(fig)


def select_check(summary, check):
    """Rows produced by one support-recovery check, so different checks are never mixed."""
    selected = summary[summary["check"] == check]
    if selected.empty:
        raise ValueError(f"no rows for check '{check}' in the summary "
                         f"(available: {sorted(summary['check'].unique())})")
    return selected


def plot_curves(ax, curves, x_col, y_col, low_col=None, high_col=None):
    """One line per p, with an optional shaded uncertainty band."""
    for p, curve in curves.groupby("p"):
        style = P_STYLE[p]
        curve = curve.sort_values(x_col)
        k = int(curve["k"].iloc[0])
        if low_col is not None:
            ax.fill_between(curve[x_col], curve[low_col], curve[high_col],
                            color=style["color"], alpha=0.15, linewidth=0)
        ax.plot(curve[x_col], curve[y_col], color=style["color"], marker=style["marker"],
                markersize=4, linewidth=1.5, label=f"p = {p} (k = {k})")


def fig1_success(summary, out_path):
    """Fig. 1: success probability vs raw n (a) and vs rescaled theta (b)."""
    curves = summary[(summary["design"] == "identity") & (summary["regime"] == "fractional")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ax = axes[0]
    plot_curves(ax, curves, "n", "success_rate", "ci_low", "ci_high")
    ax.set_xlabel("Number of samples n")
    ax.set_ylabel("Probability of exact signed-support recovery")
    ax.set_title("(a) Against raw sample size n")

    ax = axes[1]
    plot_curves(ax, curves, "theta_actual", "success_rate", "ci_low", "ci_high")
    ax.axvline(1.0, color=REFERENCE, linestyle="--", linewidth=1)
    ax.text(1.02, 0.03, "theory: θ* = 1", color=REFERENCE, fontsize=9)
    ax.set_xlabel("Rescaled sample size θ = n / [2k log(p − k)]")
    ax.set_title("(b) Against rescaled sample size θ")
    ax.legend(loc="lower right")

    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(left=0)

    fig.suptitle(r"Lasso support recovery, standard Gaussian design, $k = \lceil 0.40\, p^{0.75} \rceil$  "
                 "(200 reps per point; bands = 95% Wilson CI)", fontsize=10, color=INK)
    fig.tight_layout()
    save_figure(fig, out_path)


def fig1_diagnostics(summary, out_path):
    """Diagnostic: max |z_j| (dual feasibility), pass rates of PDW Steps 3 and 4,
    and how many wrong variables enter when Step 3 fails."""
    curves = summary[(summary["design"] == "identity") & (summary["regime"] == "fractional")]
    curves = curves[curves["max_abs_z_median"].notna()]   # drop n <= k conditions

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.ravel()

    ax = axes[0]
    plot_curves(ax, curves, "theta_actual", "max_abs_z_median", "max_abs_z_q10", "max_abs_z_q90")
    ax.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1)
    ax.text(curves["theta_actual"].max(), 1.03, "feasibility bound |z| < 1",
            color=REFERENCE, fontsize=9, ha="right", va="bottom")
    ax.set_ylim(0.5, 3.0)   # theta <= 0.2 values (up to ~15) are cut off to show the region near 1
    ax.set_ylabel(r"$\max_{j \notin S} |z_j|$")
    ax.set_title("(a) Dual-feasibility statistic\n(median, 10–90% band; y-axis cut at 3)")
    ax.legend(loc="upper right")

    ax = axes[1]
    plot_curves(ax, curves, "theta_actual", "dual_feasible_rate")
    ax.set_ylabel("Pass rate")
    ax.set_title("(b) PDW Step 3: strict dual feasibility")

    ax = axes[2]
    plot_curves(ax, curves, "theta_actual", "sign_consistent_rate")
    ax.set_ylabel("Pass rate")
    ax.set_title("(c) PDW Step 4: sign consistency")

    ax = axes[3]
    severity = curves[curves["fail_reps_violations"] >= MIN_FAIL_REPS]
    plot_curves(ax, severity, "theta_actual", "mean_violations", "mean_violations_lo", "mean_violations_hi")
    ax.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1)
    # counts are >= 1 and span ~1 to ~50: a log axis shows both ends honestly
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.9, None)
    ax.set_ylabel(r"Mean #$\{j \notin S: |z_j| \geq 1\}$ (log scale)")
    ax.set_title(f"(d) Wrong variables entering, given Step 3 fails\n"
                 f"(mean, 95% CI; points with ≥ {MIN_FAIL_REPS} failing reps)")

    for ax in axes:
        ax.set_xlabel("Rescaled sample size θ")
        ax.axvline(1.0, color=REFERENCE, linestyle=":", linewidth=1)
        ax.set_xlim(axes[0].get_xlim())   # same theta range in every panel
    for ax in axes[1:3]:
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Diagnostics: which PDW step fails (standard Gaussian design, fractional power sparsity)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save_figure(fig, out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Make figures from the summary table.")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--check", choices=sorted(CHECKS), default=CONFIG["check"],
                        help="which support-recovery check's results to plot")
    args = parser.parse_args()

    set_style()
    summary = select_check(pd.read_csv(args.summary), args.check)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_success(summary, out_dir / "fig1_success_vs_n_and_theta")
    fig1_diagnostics(summary, out_dir / "fig1_pdw_diagnostics")


if __name__ == "__main__":
    main()
