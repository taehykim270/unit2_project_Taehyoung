"""Minor tests: guards for specific pipeline behaviours that the core tests do not cover."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from helpers import make_fake_raw
from src.dgps import generate_replicate, sample_size
from src.figures import FORMATS, fig1_diagnostics, fig1_success, fig_success_by_regime, save_figure, select_check, set_style
from src.methods import PDWClosedForm, PDWRestrictedLasso, regularization_lambda
from src.metrics import summarize_conditions, summarize_theory_bounds
from src.simulation import CONFIG, run_simulation

PDW_CHECKS = [PDWRestrictedLasso, PDWClosedForm]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def test_solver_nonconvergence_is_reported():
    """A solver ConvergenceWarning is caught and recorded as solver_converged = False."""
    p, k = 128, 16
    n = sample_size(1.0, p, k)
    lam = regularization_lambda(n, p, k, 0.5)
    d = generate_replicate(n, p, k, 0.5, 0.5, np.random.default_rng(16))
    assert PDWRestrictedLasso().check(d, lam)["solver_converged"] is True
    assert PDWClosedForm().check(d, lam)["solver_converged"] is True
    starved = PDWRestrictedLasso(solver_tol=1e-16, max_iter=1)   # cannot converge in one sweep
    assert starved.check(d, lam)["solver_converged"] is False


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_pdw_fails_when_n_not_above_k(check_cls):
    """With n <= k the PDW reports failure, with undefined (NaN) diagnostics (ADEMP 2.5)."""
    p, k = 128, 16
    d = generate_replicate(n=k, p=p, k=k, sigma=0.5, beta_min=0.5, rng=np.random.default_rng(13))
    out = check_cls().check(d, lam=0.5)
    assert out["success"] is False
    assert np.isnan(out["max_abs_z"])


@pytest.mark.parametrize("check_cls", PDW_CHECKS)
def test_success_requires_both_steps(check_cls):
    """The returned fields are consistent: success = Step 3 and Step 4, and counts match the steps."""
    p, k = 128, 16
    check = check_cls()
    rng = np.random.default_rng(14)
    for theta in (0.6, 1.0, 1.5):
        n = sample_size(theta, p, k)
        lam = regularization_lambda(n, p, k, 0.5)
        for _ in range(10):
            d = generate_replicate(n, p, k, 0.5, 0.5, rng)
            out = check.check(d, lam)
            assert out["success"] == (out["dual_feasible"] and out["sign_consistent"])
            assert out["dual_feasible"] == (out["max_abs_z"] < 1)
            assert out["sign_consistent"] == (out["sign_margin"] > check.sign_tol)
            assert out["dual_feasible"] == (out["n_violations"] == 0)
            assert out["sign_consistent"] == (out["n_sign_errors"] == 0)
            assert 0 <= out["n_violations"] <= p - k and 0 <= out["n_sign_errors"] <= k


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_checks_are_never_pooled():
    """Rows from two different checks stay separate conditions in the summary."""
    a = make_fake_raw({1.0: 0.2}, seed=1)
    b = make_fake_raw({1.0: 0.8}, seed=2).assign(check="pdw_closed_form")
    summary = summarize_conditions(pd.concat([a, b]))
    assert len(summary) == 2 and set(summary["check"]) == {"pdw_restricted_lasso", "pdw_closed_form"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_saved_figures_are_byte_identical_across_runs(tmp_path):
    """No timestamps in the outputs: saving the same figure twice gives identical files."""
    contents = []
    for run in ("a", "b"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        save_figure(fig, tmp_path / run)
        contents.append({ext: (tmp_path / f"{run}.{ext}").read_bytes() for ext in FORMATS})
    assert contents[0] == contents[1]


def test_figures_are_written_in_every_format(tmp_path):
    """All four figures render from a small simulation and are saved as PDF and PNG."""
    config = dict(CONFIG, n_reps=5, p_values=[128, 256], thetas=[0.5, 1.0, 1.5, 2.0])
    summary = select_check(summarize_conditions(run_simulation(config)), CONFIG["check"])
    bounds = summarize_theory_bounds(summary[["design", "regime", "p", "k"]].drop_duplicates(),
                                     dict(CONFIG, theory_draws=2))
    set_style()
    fig1_success(summary, tmp_path / "fig1")
    fig1_diagnostics(summary, tmp_path / "diag")
    fig_success_by_regime(summary, bounds, "identity", tmp_path / "fig2")
    fig_success_by_regime(summary, bounds, "toeplitz", tmp_path / "fig3")
    for stem in ("fig1", "diag", "fig2", "fig3"):
        for ext in FORMATS:
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 0
