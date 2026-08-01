import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import operational_space_impedance_common as common  # noqa: E402
import plot_impedance_feedback_linearization as feedback  # noqa: E402
import plot_operational_space_impedance_paper_figure as paper_plot  # noqa: E402


def test_operational_canonical_figure_layout(monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _: None)
    run = common.load_run_npz(paper_plot.DEFAULT_TRAJECTORY_INPUT)
    problem = common.build_problem(run.trajectory_config)
    comparison = feedback.load_results(paper_plot.DEFAULT_COMPARISON_INPUT)

    fig = paper_plot.build_operational_space_paper_figure(run, problem, comparison)

    np.testing.assert_allclose(fig.get_size_inches(), common.cm_to_inches((16.5, 12.0)))
    assert len(fig.axes) == 6
    assert fig.axes[0].get_ylabel() == r"Position $\mathrm{[m]}$"
    assert fig.axes[4].get_yscale() == "log"
    assert fig.axes[5].get_yscale() == "log"
    plt.close(fig)


def test_feedback_error_axis_uses_saved_error_norms(monkeypatch):
    monkeypatch.setattr(feedback.shutil, "which", lambda _: None)
    results = feedback.load_results(paper_plot.DEFAULT_COMPARISON_INPUT)
    fig, ax = plt.subplots()
    feedback.plot_feedback_error_axis(
        ax,
        results,
        quantity="position",
        units="mm",
        show_legend=False,
    )

    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), results["full"]["position_error_mm"]
    )
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(), results["partial"]["position_error_mm"]
    )
    plt.close(fig)
