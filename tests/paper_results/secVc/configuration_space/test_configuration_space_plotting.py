import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "configuration_space_comparison"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import configuration_space_comparison_simulation as simulation  # noqa: E402
import plot_configuration_space_comparison as plotter  # noqa: E402

CONTROLLERS = ("PD (model-free)", "Computed Torque")
COLORS = {"PD (model-free)": "#7F7F7F", "Computed Torque": "#E24A33"}


def _result(rmse: tuple[float, float, float]) -> dict:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    q_des = np.column_stack((t, 2.0 * t, 0.1 * t))
    q = 0.9 * q_des
    error = q_des - q
    return {
        "t": t,
        "q": q,
        "qd": np.zeros_like(q),
        "u": np.zeros_like(q),
        "metrics": {
            "rmse": np.asarray(rmse),
            "max_error": np.max(np.abs(error), axis=0),
            "ise": np.sum(error**2, axis=0),
            "tracking_error": error,
            "q_des": q_des,
        },
    }


def comparison_run() -> simulation.ComparisonRun:
    combined = {name: _result((3.0, 2.0, 0.03)) for name in CONTROLLERS}
    regulation = {
        CONTROLLERS[0]: _result((1.0, 2.0, 0.01)),
        CONTROLLERS[1]: _result((3.0, 4.0, 0.02)),
    }
    tracking = {
        CONTROLLERS[0]: _result((5.0, 6.0, 0.03)),
        CONTROLLERS[1]: _result((7.0, 8.0, 0.04)),
    }
    group = simulation.PlotGroup(
        key="combined",
        scenario_key="combined",
        title="Combined",
        filename_prefix="combined",
        controller_names=CONTROLLERS,
        colors=COLORS,
        event_times=(1.0,),
        phase_boundary_time=1.5,
        phase_boundary_label="Trajectory tracking starts",
    )
    summary = simulation.RmseSummary(
        key="phase-rmse",
        filename="phase_rmse.pdf",
        panels=(
            simulation.RmsePanel(
                "Setpoint Regulation: RMSE", "regulation", CONTROLLERS
            ),
            simulation.RmsePanel("Trajectory Tracking: RMSE", "tracking", CONTROLLERS),
        ),
    )
    return simulation.ComparisonRun(
        example_key="combined",
        title="Combined",
        strain_indices=(0, 1, 2),
        strain_names=(r"$\kappa_y$", r"$\kappa_z$", r"$\sigma_x$"),
        scenarios=(
            simulation.ScenarioInfo("combined", "Combined"),
            simulation.ScenarioInfo("regulation", "Regulation"),
            simulation.ScenarioInfo("tracking", "Tracking"),
        ),
        results={
            "combined": combined,
            "regulation": regulation,
            "tracking": tracking,
        },
        plot_groups=(group,),
        rmse_summaries=(summary,),
        render_targets=(),
    )


def test_canonical_figure_has_expected_layout(monkeypatch):
    monkeypatch.setattr(plotter.shutil, "which", lambda _: None)
    fig = plotter.build_configuration_space_paper_figure(comparison_run())

    np.testing.assert_allclose(
        fig.get_size_inches(), plotter.cm_to_inches((16.5, 9.5))
    )
    assert len(fig.axes) == 3
    assert fig.axes[2].get_xlabel() == r"Time $\mathrm{[s]}$"
    assert "Setpoint regulation" in [text.get_text() for text in fig.axes[0].texts]
    assert "Trajectory tracking" in [text.get_text() for text in fig.axes[0].texts]
    assert all(
        text.get_text() not in tuple("ABCDEF")
        for ax in fig.axes
        for text in ax.texts
    )
    desired_line = fig.axes[0].lines[0]
    assert desired_line.get_label() == "Desired"
    assert desired_line.get_linewidth() == 2.0
    assert desired_line.get_zorder() == 5
    plt.close(fig)


def test_rmse_uses_separate_saved_phase_results(monkeypatch):
    monkeypatch.setattr(plotter.shutil, "which", lambda _: None)
    run = comparison_run()
    summary = run.rmse_summaries[0]

    np.testing.assert_allclose(
        plotter.rmse_values(run, summary.panels[0], 0), [1.0, 3.0]
    )
    np.testing.assert_allclose(
        plotter.rmse_values(run, summary.panels[1], 0), [5.0, 7.0]
    )

    fig = plotter.build_rmse_summary_figure(run, summary)
    assert len(fig.axes) == 6
    np.testing.assert_allclose(
        [patch.get_width() for patch in fig.axes[0].patches], [1.0, 3.0]
    )
    np.testing.assert_allclose(
        [patch.get_width() for patch in fig.axes[1].patches], [5.0, 7.0]
    )
    assert fig.axes[0].get_title() == "Setpoint Regulation"
    assert fig.axes[1].get_title() == "Trajectory Tracking"
    assert all(r"\kappa_y" in fig.axes[index].get_xlabel() for index in (0, 1))
    assert all(r"\kappa_z" in fig.axes[index].get_xlabel() for index in (2, 3))
    assert all(r"\sigma_x" in fig.axes[index].get_xlabel() for index in (4, 5))
    assert all(
        text.get_text() not in tuple("ABCDEF")
        for ax in fig.axes
        for text in ax.texts
    )
    plt.close(fig)


def test_canonical_save_requires_force(tmp_path, monkeypatch):
    monkeypatch.setattr(plotter.shutil, "which", lambda _: None)
    output = tmp_path / "configuration.pdf"
    plotter.save_configuration_space_paper_figure(
        comparison_run(), output, show=False, force=False
    )
    assert output.stat().st_size > 0
    with pytest.raises(FileExistsError, match="--force"):
        plotter.save_configuration_space_paper_figure(
            comparison_run(), output, show=False, force=False
        )
