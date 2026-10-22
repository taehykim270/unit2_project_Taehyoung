"""
Figures built from results/summary.csv (run `python -m src.metrics` first).
Each figure is saved as PDF (vector, for reports) and PNG (for the README).

    fig1_success_vs_n_and_theta   publication figure: Fig. 1(a) and 1(b) of the paper
    fig1_pdw_diagnostics          diagnostic figure: which primal-dual witness step fails, and how badly
    fig2_identity_by_regime       publication figure: Fig. 2 of the paper (identity design, one panel per regime)
    fig3_toeplitz_by_regime       publication figure: Fig. 3 of the paper (Toeplitz design), with the
                                  theoretical thresholds from results/theory_bounds.csv
    fig4_toeplitz_strong_by_regime  extension (not in the paper): Toeplitz design with mu = 0.5
    fig5_sparse_by_regime           extension (not in the paper): sparse random design, q = 0.1

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
from matplotlib.lines import Line2D

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

# Panel order and labels for Figs. 2-3
REGIME_ORDER = ["linear", "sublinear", "fractional"]
DESIGN_LABELS = {"identity": r"Identity ($\Sigma = I$)",
                 "toeplitz": rf"Toeplitz ($\mu = {CONFIG['toeplitz_mu']['toeplitz']}$)",
                 "toeplitz_strong": rf"Toeplitz ($\mu = {CONFIG['toeplitz_mu']['toeplitz_strong']}$, extension)",
                 "sparse": rf"sparse random ($q = {CONFIG['sparse_q']}$, extension)"}
REGIME_LABELS = {"linear": r"Linear: $k = \lceil \alpha p \rceil$",
                 "sublinear": r"Sublinear: $k = \lceil \alpha p / \log(\alpha p) \rceil$",
                 "fractional": r"Fractional power: $k = \lceil \alpha p^{\delta} \rceil$"}

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
    """Save one figure in every format in FORMATS (out_stem has no extension).

    PDFs are written without a creation date, so re-running gives byte-identical files.
    """
    for ext in FORMATS:
        path = out_stem.with_suffix(f".{ext}")
        metadata = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        logger.info("Saved %s", path)
    plt.close(fig)


def select_check(summary, check):
    """Rows produced by one support-recovery check, so different checks are never mixed."""
    selected = summary[summary["check"] == check]
    if selected.empty:
        raise ValueError(f"no rows for check '{check}' in the summary "
                         f"(available: {sorted(summary['check'].unique())})")
    return selected


def _plot_curves(ax, curves, x_col, y_col, low_col=None, high_col=None, show_k=True,
                 linestyle="-", labelled=True, markersize=4):
    """One line per p, with an optional shaded uncertainty band."""
    for p, curve in curves.groupby("p"):
        style = P_STYLE[p]
        curve = curve.sort_values(x_col)
        label = f"p = {p} (k = {int(curve['k'].iloc[0])})" if show_k else f"p = {p}"
        if low_col is not None:
            ax.fill_between(curve[x_col], curve[low_col], curve[high_col],
                            color=style["color"], alpha=0.15, linewidth=0)
        ax.plot(curve[x_col], curve[y_col], color=style["color"], marker=style["marker"],
                markersize=markersize, linewidth=1.5, linestyle=linestyle, label=label if labelled else None)


def fig1_success(summary, out_path):
    """Fig. 1: success probability vs raw n (a) and vs rescaled theta (b)."""
    curves = summary[(summary["design"] == "identity") & (summary["regime"] == "fractional")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ax = axes[0]
    _plot_curves(ax, curves, "n", "success_rate", "ci_low", "ci_high")
    ax.set_xlabel("Number of samples n")
    ax.set_ylabel("Probability of exact signed-support recovery")
    ax.set_title("(a) Against raw sample size n")

    ax = axes[1]
    _plot_curves(ax, curves, "theta_actual", "success_rate", "ci_low", "ci_high")
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
    """Diagnostic for Fig. 1: (a) which primal-dual witness step fails, (b) how badly it fails."""
    curves = summary[(summary["design"] == "identity") & (summary["regime"] == "fractional")]
    curves = curves[curves["max_abs_z_median"].notna()]   # drop n <= k conditions

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    _plot_curves(ax, curves, "theta_actual", "dual_feasible_rate", show_k=False)
    _plot_curves(ax, curves, "theta_actual", "sign_consistent_rate", linestyle="--", labelled=False)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction of replicates passing the step")
    ax.set_title("(a) Pass rate of each primal-dual witness step")

    ax = axes[1]
    severity = curves[curves["fail_reps_violations"] >= MIN_FAIL_REPS]
    _plot_curves(ax, severity, "theta_actual", "mean_violations", "mean_violations_lo", "mean_violations_hi",
                 labelled=False)
    ax.axhline(1.0, color=MUTED, linestyle="-", linewidth=0.8)   # one wrong variable (not dashed: dashed = Step 4)
    # counts are >= 1 and span ~1 to ~50: a log axis shows both ends honestly
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.9, None)
    ax.set_ylabel("Mean number of wrong variables (log scale)")
    ax.set_title("(b) Wrong variables selected, among replicates failing Step 3")

    for ax in axes:
        ax.set_xlabel("Rescaled sample size θ")
        ax.axvline(1.0, color=REFERENCE, linestyle=":", linewidth=1)
        ax.set_xlim(axes[0].get_xlim())   # same theta range in both panels

    # two legend rows: colours = p, line styles = PDW step
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    step_handles = [Line2D([], [], color=REFERENCE, linestyle="-", linewidth=1.5),
                    Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1.5)]
    step_labels = ["Step 3: no wrong variable enters (dual feasibility)",
                   "Step 4: true variables kept with correct signs (sign consistency)"]
    fig.legend(step_handles, step_labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    axes[1].text(0.98, 0.97, f"points with ≥ {MIN_FAIL_REPS} failing reps\nbands = 95% CI",
                 transform=axes[1].transAxes, color=MUTED, fontsize=8, ha="right", va="top")
    fig.suptitle("Diagnostics for Fig. 1 (not in the paper): why and how badly the Lasso misses the support\n"
                 "identity design, fractional power sparsity; success requires passing both steps",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_figure(fig, out_path)


def fig_success_by_regime(summary, bounds, design, out_path):
    """Figs. 2-3: success probability vs theta for one design, one panel per sparsity regime.

    identity: one dashed line at theta* = 1 (there theta_l = theta_u = 1).
    toeplitz: dashed = eq. (29) thresholds theta_l, theta_u (mean over random supports);
              dotted = rho_l, rho_u of Sigma_{S^c|S} alone, which the paper's Fig. 3 lines match.
    Lines are averaged over p; they barely depend on p.
    """
    regimes = [r for r in REGIME_ORDER if r in set(summary["regime"])]
    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 4.2), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, regime in zip(axes, regimes):
        curves = summary[(summary["design"] == design) & (summary["regime"] == regime)]
        _plot_curves(ax, curves, "theta_actual", "success_rate", "ci_low", "ci_high", show_k=False,
                     markersize=2.8)
        b = bounds[(bounds["design"] == design) & (bounds["regime"] == regime)]
        if design in ("identity", "sparse"):   # Sigma = I: theta* = 1
            ax.axvline(1.0, color=REFERENCE, linestyle="--", linewidth=1)
        else:
            # each line is labelled just above the plot area, where no data is drawn;
            # lines beyond the simulated theta range are noted at the right edge instead
            x_max = summary["theta_actual"].max() + 0.1
            for col, style, label, ha in [("theta_l_mean", "--", r"$\theta_\ell$", "center"),
                                          ("theta_u_mean", "--", r"$\theta_u$", "center"),
                                          ("rho_l_mean", ":", r"$\rho_\ell$", "right"),
                                          ("rho_u_mean", ":", r"$\rho_u$", "left")]:
                x = b[col].mean()
                if x > x_max:
                    ax.text(0.98, 0.78, f"{label} ≈ {x:.0f} →\n(off scale)", transform=ax.transAxes,
                            color=REFERENCE, fontsize=9, ha="right", va="center")
                    continue
                ax.axvline(x, color=REFERENCE, linestyle=style, linewidth=1)
                ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), color=REFERENCE,
                        fontsize=9, ha=ha, va="bottom")
            ax.set_xlim(0, x_max)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(left=0)
        ax.set_title(REGIME_LABELS[regime], pad=16)
        ax.set_xlabel("Rescaled sample size θ = n / [2k log(p − k)]")
    axes[0].set_ylabel("Prob. of exact signed-support recovery")

    handles, labels = axes[0].get_legend_handles_labels()
    if design in ("identity", "sparse"):   # Sigma = I: theta* = 1
        handles.append(Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1))
        labels.append(r"theory: $\theta^* = 1$")
    else:
        handles += [Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1),
                    Line2D([], [], color=REFERENCE, linestyle=":", linewidth=1)]
        rho_note = ", matching the paper's lines" if design == "toeplitz" else ""
        labels += [r"$\theta_\ell, \theta_u$: theoretical bounds, eq. (29)",
                   r"$\rho_\ell, \rho_u$: numerators of eq. (29)" + rho_note]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(rf"Lasso support recovery, {DESIGN_LABELS[design]} design ($\alpha = {CONFIG['alpha']}$, "
                 rf"$\delta = {CONFIG['delta']}$; 200 reps per point; bands = 95% Wilson CI)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Make figures from the summary table.")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--bounds", default="results/theory_bounds.csv")
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
    bounds = pd.read_csv(args.bounds)
    fig_success_by_regime(summary, bounds, "identity", out_dir / "fig2_identity_by_regime")
    fig_success_by_regime(summary, bounds, "toeplitz", out_dir / "fig3_toeplitz_by_regime")
    if "toeplitz_strong" in set(summary["design"]):
        fig_success_by_regime(summary, bounds, "toeplitz_strong", out_dir / "fig4_toeplitz_strong_by_regime")
    if "sparse" in set(summary["design"]):
        fig_success_by_regime(summary, bounds, "sparse", out_dir / "fig5_sparse_by_regime")


if __name__ == "__main__":
    main()
