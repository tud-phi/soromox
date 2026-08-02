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
from trajectory_primitives import TaskSpaceTrajectoryConfig  # noqa: E402


def _synthetic_operational_run() -> common.PCSImpedanceRun:
    config = TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="figure-eight",
        orientation_primitive="surface-following",
        surface="sphere",
    )
    t = np.linspace(0.0, 0.2, 3)
    x_des = np.column_stack(
        (
            0.1 * t,
            -0.2 * t,
            0.3 * t,
            0.4 + 0.1 * t,
            -0.2 + 0.05 * t,
            0.6 - 0.1 * t,
        )
    )
    return common.PCSImpedanceRun(
        trajectory_config=config,
        t1=float(t[-1]),
        solver_dt=1e-4,
        save_dt=0.1,
        t_traj=t,
        q_traj=np.zeros((t.size, 12)),
        qd_traj=np.zeros((t.size, 12)),
        u_traj=np.zeros((t.size, 12)),
        x_traj_full=x_des + 0.01,
        x_des_traj_full=x_des,
    )


def _write_feedback_results(path: Path) -> None:
    t = np.linspace(0.0, 0.2, 3)
    arrays: dict[str, np.ndarray] = {
        "controller_modes": np.asarray(feedback.CONTROLLER_MODES),
    }
    for mode_index, mode in enumerate(feedback.CONTROLLER_MODES, start=1):
        scale = float(mode_index)
        arrays.update(
            {
                f"{mode}_t": t,
                f"{mode}_position_mm": scale * np.column_stack((t, 2 * t, 3 * t)),
                f"{mode}_position_desired_mm": np.column_stack((t, t, t)),
                f"{mode}_position_error_mm": scale * np.asarray((0.3, 0.2, 0.1)),
                f"{mode}_orientation_deg": scale * np.column_stack((3 * t, 2 * t, t)),
                f"{mode}_orientation_desired_deg": np.column_stack((t, t, t)),
                f"{mode}_orientation_error_deg": scale * np.asarray((0.6, 0.4, 0.2)),
            }
        )
    np.savez(path, **arrays)


def test_operational_canonical_figure_layout(monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _: None)
    run = _synthetic_operational_run()
    problem = common.build_problem(run.trajectory_config)

    fig = paper_plot.build_operational_space_paper_figure(run, problem)

    np.testing.assert_allclose(fig.get_size_inches(), common.cm_to_inches((16.5, 8.5)))
    assert len(fig.axes) == 4
    assert fig.axes[0].get_ylabel() == r"Position $\mathrm{[m]}$"
    assert fig.axes[2].get_ylabel() == r"Orientation $\mathrm{[rad]}$"
    assert all(ax.get_yscale() == "linear" for ax in fig.axes)
    assert all(
        text.get_text() not in tuple("ABCDEF") for ax in fig.axes for text in ax.texts
    )
    desired_lines = [fig.axes[0].lines[index] for index in (1, 3, 5)]
    assert all(line.get_linewidth() == 2.0 for line in desired_lines)
    assert all(line.get_zorder() == 5 for line in desired_lines)
    plt.close(fig)


def test_feedback_error_axis_uses_saved_error_norms(monkeypatch, tmp_path):
    monkeypatch.setattr(feedback.shutil, "which", lambda _: None)
    input_path = tmp_path / "feedback_results.npz"
    _write_feedback_results(input_path)
    results = feedback.load_results(input_path)
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
