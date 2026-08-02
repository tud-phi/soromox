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

import plot_benchmark_gpu  # noqa: E402


@pytest.mark.parametrize(
    ("num_systems", "expected"),
    [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2))],
)
def test_subplot_grid_is_limited_to_two_columns(num_systems, expected):
    assert plot_benchmark_gpu._subplot_grid(num_systems) == expected


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
        for system in plot_benchmark_gpu.MODEL_ORDER
    ]
    figures = []
    real_subplots = plot_benchmark_gpu.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plot_benchmark_gpu.plt, "subplots", capture_subplots)

    plot_benchmark_gpu._plot(rows, None, False, False, False)
    plot_benchmark_gpu._plot_total_throughput(rows, None, False, False, False)

    np.testing.assert_allclose(figures[0].get_size_inches() * 2.54, [18.0, 24.0])
    np.testing.assert_allclose(figures[1].get_size_inches() * 2.54, [18.0, 16.0])
