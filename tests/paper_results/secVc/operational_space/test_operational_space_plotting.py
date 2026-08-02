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

    fig = paper_plot.build_operational_space_paper_figure(run, problem)

    np.testing.assert_allclose(fig.get_size_inches(), common.cm_to_inches((16.5, 8.5)))
    assert len(fig.axes) == 4
    assert fig.axes[0].get_ylabel() == r"Position $\mathrm{[m]}$"
    assert fig.axes[2].get_ylabel() == r"Orientation $\mathrm{[rad]}$"
    assert all(ax.get_yscale() == "linear" for ax in fig.axes)
    assert all(
        text.get_text() not in tuple("ABCDEF")
        for ax in fig.axes
        for text in ax.texts
    )
    desired_lines = [fig.axes[0].lines[index] for index in (1, 3, 5)]
    assert all(line.get_linewidth() == 2.0 for line in desired_lines)
    assert all(line.get_zorder() == 5 for line in desired_lines)
    plt.close(fig)


def test_feedback_error_axis_uses_saved_error_norms(monkeypatch):
    monkeypatch.setattr(feedback.shutil, "which", lambda _: None)
    results = feedback.load_results(feedback.DEFAULT_DATA_INPUT)
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
