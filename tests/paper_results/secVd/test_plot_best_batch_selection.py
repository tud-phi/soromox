"""Regression test for the Section Vd plot best-batch coupling (issue #128).

``build_figure`` used to derive one best-batch index from the collocated loss
history and pass it to both the collocated and the synergistic panel. The
committed datasets happen to share a minimum, which hid the coupling; once the
two are generated independently the synergistic panel can show a suboptimal
batch. The fixtures below deliberately place the two minima on different
batches.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
MODULE_DIR = SECTION_DIR / "code"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import plot_control_gain_optimization as plot_module  # noqa: E402
from plot_control_gain_optimization import best_batch_from_loss  # noqa: E402

NUM_ITERS = 5
NUM_BATCHES = 4
NUM_STEPS = 7
COLLOCATED_BEST_BATCH = 1
SYNERGISTIC_BEST_BATCH = 3


def _loss_history(best_batch: int) -> np.ndarray:
    """Build a ``(iterations, batches)`` loss history minimized at ``best_batch``."""
    history = np.ones((NUM_ITERS, NUM_BATCHES))
    history[-1, best_batch] = 0.1
    return history


def _fake_results(best_batch: int, num_channels: int, target_key: str) -> dict:
    ts = np.linspace(0.0, 5.0, NUM_STEPS)
    raw_initial = np.zeros((NUM_BATCHES, NUM_STEPS, num_channels))
    raw_best = np.stack(
        [np.full((NUM_STEPS, num_channels), float(b)) for b in range(NUM_BATCHES)]
    )
    return {
        "history_loss": _loss_history(best_batch),
        "t_ts": ts[None, :],
        target_key: np.zeros((NUM_STEPS, num_channels)),
        "raw_initial": raw_initial,
        "raw_best": raw_best,
    }


def test_best_batch_from_loss_finds_the_minimizing_batch():
    assert best_batch_from_loss(_loss_history(2)) == 2


def test_each_panel_receives_its_own_best_batch(monkeypatch):
    data_c = _fake_results(COLLOCATED_BEST_BATCH, 6, "q_des_ts")
    data_c["q_ts_init"] = data_c.pop("raw_initial")
    data_c["q_ts_best"] = data_c.pop("raw_best")

    data_s = _fake_results(SYNERGISTIC_BEST_BATCH, 3, "x_des_ts")
    data_s["x_ts_init"] = data_s.pop("raw_initial")
    data_s["x_ts_best"] = data_s.pop("raw_best")

    monkeypatch.setattr(
        plot_module,
        "load_results",
        lambda directory: data_c if directory.name == "collocated" else data_s,
    )

    received: dict[str, int] = {}
    monkeypatch.setattr(
        plot_module,
        "plot_angular_strains",
        lambda ax, data, best_batch, *_: received.update(collocated=best_batch),
    )
    monkeypatch.setattr(
        plot_module,
        "plot_end_effector_position",
        lambda ax, data, best_batch, *_: received.update(synergistic=best_batch),
    )

    figure = plot_module.build_figure(Path("unused"))
    matplotlib.pyplot.close(figure)

    assert received["collocated"] == COLLOCATED_BEST_BATCH
    assert received["synergistic"] == SYNERGISTIC_BEST_BATCH
    assert received["collocated"] != received["synergistic"], (
        "The fixture is only meaningful when the two minima differ"
    )
