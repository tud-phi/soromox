import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secIVb_parallel_rollouts_gpu"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import plot_parallel_rollout_benchmark as plotter  # noqa: E402


def test_shared_style_uses_truetype_fonts_for_vector_outputs():
    plotter.configure_matplotlib()

    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


@pytest.mark.parametrize(
    ("num_systems", "expected"),
    [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2))],
)
def test_subplot_grid_is_limited_to_two_columns(num_systems, expected):
    assert plotter._subplot_grid(num_systems) == expected


def test_four_system_figures_stay_within_paper_width(monkeypatch):
    rows = [
        {
            "system": system,
            "size_label": "num_segments",
            "size_value": 1,
            "segment_count": 1,
            "gauss_points": None,
            "batch_size": 1,
            "per_env_speed_ratio": 1.0,
            "total_speed_ratio": 2.0,
        }
        for system in plotter.MODEL_ORDER
    ]
    figures = []
    real_subplots = plotter.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plotter.plt, "subplots", capture_subplots)

    plotter._plot(rows, None, False, False, False)
    plotter._plot_total_throughput(rows, None, False, False, False)

    np.testing.assert_allclose(figures[0].get_size_inches() * 2.54, [18.0, 24.0])
    np.testing.assert_allclose(figures[1].get_size_inches() * 2.54, [18.0, 16.0])


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (["gpu"], "Parallel-Rollout Throughput (GPU)"),
        (["cpu"], "Parallel-Rollout Throughput (CPU)"),
        (["gpu", "cpu"], "Parallel-Rollout Throughput (CPU and GPU)"),
        ([None], "Parallel-Rollout Throughput"),
    ],
)
def test_device_title_reflects_input_artifacts(devices, expected):
    rows = [{"device": device} for device in devices]

    assert plotter._device_title(rows) == expected
