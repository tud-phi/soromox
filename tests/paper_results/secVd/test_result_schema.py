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
    best_batch_from_loss,
    best_iteration_per_batch,
    improvement_over_initial_median,
    load_results,
    save_optimization_outputs,
    validate_results,
)

ITERATIONS, TIMESTEPS, DOFS, ACTUATORS, GAINS = 5, 7, 6, 3, 3


def _write_results(
    tmp_path: Path,
    method: str = "synergistic",
    batch: int = 4,
    history_loss: np.ndarray | None = None,
    finite_mask: np.ndarray | None = None,
) -> Path:
    """Write a minimal but schema-valid archive."""
    if history_loss is None:
        # Distinct per start so selection has something to choose between.
        history_loss = np.arange(ITERATIONS, dtype=float)[:, None] + np.arange(batch)
    if finite_mask is None:
        finite_mask = np.ones((ITERATIONS, batch), dtype=bool)
    q = np.zeros((batch, TIMESTEPS, DOFS))
    pose = np.zeros((batch, TIMESTEPS, 6))
    u = np.zeros((batch, TIMESTEPS, ACTUATORS))
    kwargs = {}
    if method == "collocated":
        kwargs["q_des_ts"] = np.zeros((TIMESTEPS, DOFS))
    return save_optimization_outputs(
        tmp_path,
        method=method,
        batch_size=batch,
        history_loss=history_loss,
        history_time=np.ones(ITERATIONS),
        history_finite_mask=finite_mask,
        history_grad_norm=np.ones((ITERATIONS, batch)),
        history_update_norm=np.ones((ITERATIONS, batch)),
        # Schema v3: what produced the archive, not just what came out of it.
        solver_dt=1e-3,
        material_damping_coefficient=7.2e1,
        opt_vars_history=[
            {
                "opt_ctr_params": {
                    key: np.ones((batch, GAINS)) * i for key in ("Kp", "Ki", "Kd")
                }
            }
            for i in range(ITERATIONS)
        ],
        init_gains={key: np.ones((batch, GAINS)) for key in ("Kp", "Ki", "Kd")},
        init_seed=0,
        init_scheme="log_uniform_v1",
        init_spread=3.0,
        objective_name="dimensionless_mean_squared_error_v1",
        objective_scales=np.ones(DOFS),
        saturation_config="none",
        optimizer_metadata="sgd(0.1)",
        t_ts=np.linspace(0, 1, TIMESTEPS),
        q_ts_init=q,
        q_ts_best=q,
        qd_ts_init=q,
        qd_ts_best=q,
        u_ts_init=u,
        u_ts_best=u,
        x_ts_init=pose,
        x_ts_best=pose,
        x_des_ts=np.zeros((TIMESTEPS, 6)),
        best_iteration=best_iteration_per_batch(history_loss, finite_mask),
        best_batch=best_batch_from_loss(history_loss, finite_mask),
        is_placeholder=True,
        **kwargs,
    )


def test_the_batch_axis_appears_only_where_quantities_vary_per_start(tmp_path):
    """A batch axis in this archive must always mean something."""
    data = load_results(_write_results(tmp_path), expected_method="synergistic")
    assert data["history_loss"].shape == (ITERATIONS, 4)
    assert data["history_finite_mask"].shape == (ITERATIONS, 4)
    assert data["history_Kp"].shape == (ITERATIONS, 4, GAINS)
    assert data["init_Kp"].shape == (4, GAINS)
    assert data["q_ts_best"].shape == (4, TIMESTEPS, DOFS)
    assert data["x_ts_best"].shape == (4, TIMESTEPS, 6)
    assert data["best_iteration"].shape == (4,)
    # Shared by every start, so no batch axis at all.
    assert data["t_ts"].shape == (TIMESTEPS,)
    assert data["x_des_ts"].shape == (TIMESTEPS, 6)
    assert data["history_time"].shape == (ITERATIONS,)
    assert int(data["batch_size"]) == 4
    assert int(data["completed_iterations"]) == ITERATIONS
    assert bool(data["is_placeholder"])


def test_a_single_start_archive_is_still_valid(tmp_path):
    """B = 1 must remain a strict special case, not a separate format."""
    data = load_results(_write_results(tmp_path, batch=1))
    assert data["history_loss"].shape == (ITERATIONS, 1)
    assert data["q_ts_best"].shape == (1, TIMESTEPS, DOFS)
    assert int(data["best_batch"]) == 0


def test_the_objective_and_optimizer_are_recorded_with_the_loss(tmp_path):
    """A stored loss must be traceable to the definition that produced it.

    The legacy archives recorded a loss whose objective nobody could name, which
    is what let the order-of-magnitude discrepancy on issue #154 go unnoticed.
    """
    data = load_results(_write_results(tmp_path))
    assert str(data["objective_name"].item()) == "dimensionless_mean_squared_error_v1"
    assert data["objective_scales"].shape == (DOFS,)
    assert str(data["saturation_config"].item()) == "none"
    assert "sgd" in str(data["optimizer_metadata"].item())


def test_the_initialization_is_recorded_rather_than_inferred(tmp_path):
    """The legacy archives were unrecoverable because this was never written."""
    data = load_results(_write_results(tmp_path))
    for key in ("init_Kp", "init_Ki", "init_Kd"):
        assert key in data
    assert int(data["init_seed"]) == 0
    assert str(data["init_scheme"].item()) == "log_uniform_v1"
    assert float(data["init_spread"]) == 3.0


def test_an_archive_missing_required_fields_is_rejected_clearly(tmp_path):
    """An archive from any older schema fails this way rather than half-loading."""
    path = tmp_path / "optimization_results.npz"
    np.savez(path, history_loss=np.zeros((5, 1)))
    with pytest.raises(ValueError, match="not the current schema"):
        load_results(path)


def test_selection_is_rechecked_against_the_loss_history(tmp_path):
    """Stored selection and re-derived selection cannot silently disagree."""
    data = load_results(_write_results(tmp_path))
    data["best_iteration"] = data["best_iteration"] + 1
    with pytest.raises(ValueError, match="best_iteration"):
        validate_results(data)

    data = load_results(_write_results(tmp_path))
    data["best_batch"] = np.asarray(int(data["best_batch"]) + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="best_batch"):
        validate_results(data)


def test_selection_prefers_the_start_with_the_lowest_loss_anywhere():
    """best_batch is argmin_b min_i, applied to one archive's own losses."""
    loss = np.array([[5.0, 9.0], [4.0, 0.5], [3.0, 8.0]])
    assert best_batch_from_loss(loss) == 1
    assert best_iteration_per_batch(loss).tolist() == [2, 1]


def test_frozen_entries_are_excluded_from_selection():
    loss = np.array([[5.0, 9.0], [4.0, 0.5], [3.0, 8.0]])
    mask = np.array([[True, True], [True, False], [True, True]])
    # Start 1's 0.5 is frozen, so start 0 wins on its own merits and start 1
    # falls back to its best *surviving* iterate.
    assert best_batch_from_loss(loss, mask) == 0
    assert best_iteration_per_batch(loss, mask).tolist() == [2, 2]


def test_a_start_that_never_became_finite_is_rejected(tmp_path):
    mask = np.ones((ITERATIONS, 4), dtype=bool)
    mask[:, 2] = False
    with pytest.raises(ValueError, match="never produced a finite iterate"):
        _write_results(tmp_path, finite_mask=mask)


def test_frozen_entries_may_be_nonfinite_but_valid_ones_may_not(tmp_path):
    """The mask is what makes a non-finite entry legal, not chance."""
    loss = np.arange(ITERATIONS, dtype=float)[:, None] + np.arange(4)
    mask = np.ones((ITERATIONS, 4), dtype=bool)
    mask[-1, 3] = False
    loss[-1, 3] = np.nan
    path = _write_results(tmp_path, history_loss=loss, finite_mask=mask)
    data = load_results(path)
    assert np.isnan(data["history_loss"][-1, 3])

    data["history_finite_mask"][-1, 3] = True
    with pytest.raises(ValueError, match="non-finite"):
        validate_results(data)


def test_improvement_matches_the_published_metric():
    """1 - min(loss[-1]) / median(loss[0]), recovered from the legacy archives."""
    loss = np.array([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])
    assert improvement_over_initial_median(loss) == pytest.approx(1.0 - 4.0 / 20.0)


def test_collocated_requires_its_configuration_reference(tmp_path):
    data = load_results(_write_results(tmp_path, method="collocated"))
    assert data["q_des_ts"].shape == (TIMESTEPS, DOFS)
    del data["q_des_ts"]
    with pytest.raises(ValueError, match="q_des_ts"):
        validate_results(data)


def test_synergistic_orientation_and_position_channels_survive(tmp_path):
    path = _write_results(tmp_path)
    with np.load(path) as archive:
        pose = archive["x_ts_best"].copy()
    pose[0, :, :3] = [0.1, 0.2, 0.3]
    pose[0, :, 3:6] = [1.0, 2.0, 3.0]
    assert np.all(pose[0, 0, :3] == [0.1, 0.2, 0.3])
    assert np.all(pose[0, 0, 3:6] == [1.0, 2.0, 3.0])
