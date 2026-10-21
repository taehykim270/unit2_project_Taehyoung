"""Figure step: only one check is plotted, and every figure is written in every format."""

import pandas as pd
import pytest

from src.figures import FORMATS, fig1_diagnostics, fig1_success, select_check, set_style
from src.metrics import summarize_conditions
from src.simulation import CONFIG, run_simulation


def test_select_check_filters_and_rejects_unknown():
    summary = pd.DataFrame({"check": ["pdw_restricted_lasso", "pdw_closed_form"], "x": [1, 2]})
    assert select_check(summary, "pdw_closed_form")["x"].tolist() == [2]
    with pytest.raises(ValueError):
        select_check(summary, "full_lasso")


def test_figures_are_written_in_every_format(tmp_path):
    config = dict(CONFIG, n_reps=5, p_values=[128, 256], thetas=[0.5, 1.0, 1.5, 2.0])
    summary = select_check(summarize_conditions(run_simulation(config)), CONFIG["check"])
    set_style()
    fig1_success(summary, tmp_path / "fig1")
    fig1_diagnostics(summary, tmp_path / "diag")
    for stem in ("fig1", "diag"):
        for ext in FORMATS:
            assert (tmp_path / f"{stem}.{ext}").stat().st_size > 0
