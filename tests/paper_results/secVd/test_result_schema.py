"""Tests for the one supported Section Vd archive schema."""

import sys
from pathlib import Path

import numpy as np
import pytest

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_results import (  # noqa: E402
    load_results,
    save_optimization_outputs,
    validate_results,
)


def _write_results(tmp_path: Path, method: str = "synergistic") -> Path:
    iterations, timesteps, dofs, actuators = 5, 7, 6, 3
    q = np.zeros((timesteps, dofs))
    pose = np.zeros((timesteps, 6))
    kwargs = {}
    if method == "collocated":
        kwargs["q_des_ts"] = q
    return save_optimization_outputs(
        tmp_path,
        method=method,
        history_loss=np.arange(iterations, dtype=float),
        history_time=np.ones(iterations),
        opt_vars_history=[
            {"opt_ctr_params": {key: np.ones(3) * i for key in ("Kp", "Ki", "Kd")}}
            for i in range(iterations)
        ],
        history_qd_ts=np.zeros((iterations, timesteps, dofs)),
        history_u_ts=np.zeros((iterations, timesteps, actuators)),
        t_ts=np.linspace(0, 1, timesteps),
        q_ts_init=q,
        q_ts_best=q,
        qd_ts_init=q,
        qd_ts_best=q,
        u_ts_init=np.zeros((timesteps, actuators)),
        u_ts_best=np.zeros((timesteps, actuators)),
        x_ts_init=pose,
        x_ts_best=pose,
        x_des_ts=pose,
        best_iteration=0,
        is_placeholder=True,
        **kwargs,
    )


def test_saved_arrays_have_explicit_single_batch_axes(tmp_path):
    path = _write_results(tmp_path)
    data = load_results(path, expected_method="synergistic")
    assert data["history_loss"].shape == (5, 1)
    assert data["history_Kp"].shape == (5, 1, 3)
    assert data["history_qd_ts"].shape == (5, 1, 7, 6)
    assert data["history_u_ts"].shape == (5, 1, 7, 3)
    assert data["t_ts"].shape == (1, 7)
    assert data["q_ts_best"].shape == (1, 7, 6)
    assert data["x_ts_best"].shape == (1, 7, 6)
    assert data["best_iteration"].shape == (1,)
    assert bool(data["is_placeholder"])
    assert int(data["completed_iterations"]) == 5


def test_old_archive_is_rejected_clearly(tmp_path):
    path = tmp_path / "optimization_results.npz"
    np.savez(path, history_loss=np.zeros((5, 1)))
    with pytest.raises(ValueError, match="legacy archives are not supported"):
        load_results(path)


def test_nonfinite_and_bad_best_iteration_are_rejected(tmp_path):
    data = load_results(_write_results(tmp_path))
    data["history_loss"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_results(data)
    data["history_loss"][0, 0] = 0.0
    data["best_iteration"][0] = 5
    with pytest.raises(ValueError, match="outside"):
        validate_results(data)


def test_synergistic_orientation_and_position_channels_survive(tmp_path):
    path = _write_results(tmp_path)
    with np.load(path) as archive:
        pose = archive["x_ts_best"].copy()
    pose[0, :, :3] = [0.1, 0.2, 0.3]
    pose[0, :, 3:6] = [1.0, 2.0, 3.0]
    assert np.all(pose[0, 0, :3] == [0.1, 0.2, 0.3])
    assert np.all(pose[0, 0, 3:6] == [1.0, 2.0, 3.0])
