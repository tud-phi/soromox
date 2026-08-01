import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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

ENTRY_POINTS = (
    "setpoint_regulation_comparison.py",
    "trajectory_tracking_comparison.py",
    "regulation_tracking_comparison.py",
    "plot_configuration_space_comparison.py",
    "plot_configuration_space_paper_figure.py",
    "render_configuration_space_comparison.py",
)


def comparison_run() -> simulation.ComparisonRun:
    scenario = simulation.ScenarioInfo("setpoint", "Setpoint")
    result = {
        "t": np.array([0.0, 0.1]),
        "q": np.ones((2, 3)),
        "qd": np.zeros((2, 3)),
        "u": 2.0 * np.ones((2, 3)),
        "metrics": {
            "rmse": np.ones((3,)),
            "max_error": 2.0 * np.ones((3,)),
            "ise": 3.0 * np.ones((3,)),
            "tracking_error": np.ones((2, 3)),
            "q_des": np.zeros((2, 3)),
        },
    }
    return simulation.ComparisonRun(
        example_key="setpoint_regulation",
        title="Synthetic comparison",
        strain_indices=(0, 1, 2),
        strain_names=("a", "b", "c"),
        scenarios=(scenario,),
        results={"setpoint": {"PID": result}},
        plot_groups=(),
        rmse_summaries=(),
        render_targets=(),
    )


def test_comparison_npz_round_trip_and_overwrite_protection(tmp_path):
    output = tmp_path / "nested" / "comparison.npz"
    expected = comparison_run()

    simulation.save_comparison_npz(output, expected)
    actual = simulation.load_comparison_npz(output)

    assert actual.example_key == expected.example_key
    assert actual.scenarios == expected.scenarios
    assert tuple(actual.results["setpoint"]) == ("PID",)
    np.testing.assert_allclose(actual.results["setpoint"]["PID"]["q"], 1.0)
    with pytest.raises(FileExistsError, match="--force"):
        simulation.save_comparison_npz(output, expected)


def test_default_paths_are_case_relative():
    case_dir = MODULE_DIR.parent

    assert (
        case_dir / "data" / "setpoint_regulation_comparison.npz"
    ) == simulation.DEFAULT_SETPOINT_OUTPUT
    assert (
        case_dir / "data" / "trajectory_tracking_comparison.npz"
    ) == simulation.DEFAULT_TRAJECTORY_OUTPUT
    assert (
        case_dir / "data" / "regulation_tracking_comparison.npz"
    ) == simulation.DEFAULT_REGULATION_TRACKING_OUTPUT
    assert case_dir / "outputs" == simulation.OUTPUTS_DIR


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_cli_starts_from_unrelated_working_directory(tmp_path, entry_point):
    subprocess.run(
        [sys.executable, str(MODULE_DIR / entry_point), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
