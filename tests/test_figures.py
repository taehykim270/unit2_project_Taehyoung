"""Figure step: only one check is plotted, and every figure is written in every format."""

import pandas as pd
import pytest

from src.figures import FORMATS, fig1_diagnostics, fig1_success, fig23_success_grid, select_check, set_style
from src.metrics import summarize_conditions, summarize_theory_bounds
from src.simulation import CONFIG, run_simulation


def test_select_check_filters_and_rejects_unknown():
    summary = pd.DataFrame({"check": ["pdw_restricted_lasso", "pdw_closed_form"], "x": [1, 2]})
    assert select_check(summary, "pdw_closed_form")["x"].tolist() == [2]
    with pytest.raises(ValueError):
        select_check(summary, "full_lasso")


def test_saved_figures_are_byte_identical_across_runs(tmp_path):
    """No timestamps in the outputs: saving the same figure twice gives identical files."""
    import matplotlib.pyplot as plt
    from src.figures import save_figure

    contents = []
    for run in ("a", "b"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        save_figure(fig, tmp_path / run)
        contents.append({ext: (tmp_path / f"{run}.{ext}").read_bytes() for ext in FORMATS})
    assert contents[0] == contents[1]


def test_figures_are_written_in_every_format(tmp_path):
    config = dict(CONFIG, n_reps=5, p_values=[128, 256], thetas=[0.5, 1.0, 1.5, 2.0])
    summary = select_check(summarize_conditions(run_simulation(config)), CONFIG["check"])
    bounds = summarize_theory_bounds(summary[["design", "regime", "p", "k"]].drop_duplicates(),
                                     dict(CONFIG, theory_draws=2))
    set_style()
    fig1_success(summary, tmp_path / "fig1")
    fig1_diagnostics(summary, tmp_path / "diag")
    fig23_success_grid(summary, bounds, tmp_path / "grid")
    for stem in ("fig1", "diag", "grid"):
        for ext in FORMATS:
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 0
