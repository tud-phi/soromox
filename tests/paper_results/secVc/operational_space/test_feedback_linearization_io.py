import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import compare_impedance_feedback_linearization as generator  # noqa: E402
import plot_impedance_feedback_linearization as plotter  # noqa: E402


def synthetic_results() -> dict[str, dict[str, np.ndarray]]:
    t = np.array([0.0, 0.1, 0.2])
    result = {
        "t": t,
        "position_mm": np.zeros((3, 3)),
        "position_desired_mm": np.ones((3, 3)),
        "position_error_mm": np.ones((3,)),
        "orientation_deg": np.zeros((3, 3)),
        "orientation_desired_deg": np.ones((3, 3)),
        "orientation_error_deg": np.ones((3,)),
    }
    return {
        mode: {key: value.copy() for key, value in result.items()}
        for mode in generator.CONTROLLER_MODES
    }


def test_feedback_comparison_round_trip_and_plot_smoke(tmp_path):
    data_path = tmp_path / "data" / "comparison.npz"
    output_dir = tmp_path / "outputs"
    expected = synthetic_results()

    generator.save_results(
        data_path,
        expected,
        {"full": 1.0, "partial": 2.0},
    )
    actual = plotter.load_results(data_path)
    for mode in generator.CONTROLLER_MODES:
        for key in plotter.RESULT_KEYS:
            np.testing.assert_allclose(actual[mode][key], expected[mode][key])

    plotter.configure_plot_style()
    paths = plotter.output_paths(output_dir, "position")
    plotter.ensure_writable(paths, force=False)
    plotter.plot_tracking(
        actual,
        quantity="position",
        units="mm",
        component_labels=("x", "y", "z"),
        paths=paths,
    )
    assert all(path.stat().st_size > 0 for path in paths)

    with pytest.raises(FileExistsError, match="--force"):
        generator.save_results(
            data_path,
            expected,
            {"full": 1.0, "partial": 2.0},
        )


def test_feedback_default_paths_are_case_relative():
    case_dir = MODULE_DIR.parent

    assert (
        case_dir / "data" / "impedance_feedback_linearization_comparison.npz"
    ) == generator.DEFAULT_DATA_OUTPUT
    assert plotter.DEFAULT_DATA_INPUT == generator.DEFAULT_DATA_OUTPUT
    assert case_dir / "outputs" == plotter.DEFAULT_OUTPUT_DIR
