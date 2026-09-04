"""Tests for the canonical Section Vd objective and its archive rescoring.

The follow-up on issue #154 found the legacy stored losses to be an order of
magnitude away from what the committed objective scored the same trajectories
at, and nothing in the workflow would have caught it. These tests pin the
definition and make the mismatch detectable.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_objective import (  # noqa: E402
    OBJECTIVE_NAME,
    configuration_space_scales,
    operational_space_scales,
    recompute_archive_losses,
    steady_state_report,
    time_averaged_squared_error,
)
from secvd_results import (  # noqa: E402
    best_batch_from_loss,
    best_iteration_per_batch,
    load_results,
    save_optimization_outputs,
)

SEGMENT_LENGTH = 0.1


def test_strain_scales_reproduce_the_recovered_legacy_weighting():
    """The legacy W was length non-dimensionalization, not a tuning constant.

    ``W = diag([1, 1, 1, 100, 100, 100])`` recovered from the collocated archive
    equals ``(1 / L**2) * diag(L**2, L**2, L**2, 1, 1, 1)`` for ``L = 0.1 m``, so
    the scales here must square to it once the arbitrary global factor is
    divided out.
    """
    scales = np.asarray(configuration_space_scales(jnp.array([SEGMENT_LENGTH])))
    implied_weighting = scales**2 / SEGMENT_LENGTH**2
    np.testing.assert_allclose(
        implied_weighting, [1.0, 1.0, 1.0, 100.0, 100.0, 100.0], rtol=1e-12
    )


def test_strain_scales_follow_each_segment_length():
    scales = np.asarray(configuration_space_scales(jnp.array([0.1, 0.2])))
    np.testing.assert_allclose(scales, [0.1, 0.1, 0.1, 1, 1, 1, 0.2, 0.2, 0.2, 1, 1, 1])


def test_task_scales_divide_position_by_the_characteristic_length():
    selector = jnp.array([False, False, False, True, True, True])
    np.testing.assert_allclose(
        np.asarray(operational_space_scales(selector, SEGMENT_LENGTH)), [10.0] * 3
    )
    # Orientation is already dimensionless, so it keeps a unit scale.
    full = operational_space_scales(jnp.ones(6, dtype=bool), SEGMENT_LENGTH)
    np.testing.assert_allclose(np.asarray(full), [1, 1, 1, 10, 10, 10])


def test_the_loss_is_a_time_average_not_an_integral():
    """Time-averaging keeps runs of different horizons comparable."""
    scales = jnp.ones(2)
    short_t = jnp.linspace(0.0, 1.0, 51)
    long_t = jnp.linspace(0.0, 5.0, 251)
    error_short = jnp.ones((51, 2)) * 0.1
    error_long = jnp.ones((251, 2)) * 0.1
    assert float(
        time_averaged_squared_error(error_short, scales, short_t)
    ) == pytest.approx(
        float(time_averaged_squared_error(error_long, scales, long_t)), rel=1e-12
    )


def test_the_loss_has_no_early_late_weighting():
    """Front-loaded and back-loaded error of equal size must score equally.

    The legacy runs disagreed on this (1:2 collocated against 1:5 synergistic);
    the canonical objective takes no position and the split is reported instead.
    """
    t = jnp.linspace(0.0, 1.0, 101)
    early = jnp.concatenate([jnp.ones((50, 1)), jnp.zeros((51, 1))])
    late = jnp.concatenate([jnp.zeros((50, 1)), jnp.ones((51, 1))])
    scales = jnp.ones(1)
    front = float(time_averaged_squared_error(early, scales, t))
    back = float(time_averaged_squared_error(late, scales, t))
    assert front == pytest.approx(back, rel=2e-2)


def test_a_dimensionless_loss_is_invariant_to_the_unit_of_length():
    """Expressing the same physical error in different units must not move it."""
    t = jnp.linspace(0.0, 1.0, 51)
    selector = jnp.ones(6, dtype=bool)
    error_m = jnp.tile(jnp.array([0.01, 0.0, 0.0, 0.005, 0.0, 0.0]), (51, 1))
    in_metres = time_averaged_squared_error(
        error_m, operational_space_scales(selector, 0.1), t
    )
    # The same rollout described in centimetres, with a length scale to match.
    error_cm = error_m.at[:, 3:].multiply(100.0)
    in_centimetres = time_averaged_squared_error(
        error_cm, operational_space_scales(selector, 10.0), t
    )
    assert float(in_metres) == pytest.approx(float(in_centimetres), rel=1e-12)


def test_steady_state_is_reported_rather_than_optimized():
    t = np.linspace(0.0, 5.0, 501)
    # Large transient error that decays to nothing.
    error = np.zeros((501, 1))
    error[:100, 0] = 1.0
    report = steady_state_report(error, np.ones(1), t)
    assert report["rms_steady_state"] < 1e-12
    assert report["rms_overall"] > 0.4
    assert report["steady_state_starts_at"] == pytest.approx(4.5, abs=1e-6)


def _write_archive(tmp_path: Path, method: str, scales: np.ndarray, errors: np.ndarray):
    """Write an archive whose stored losses genuinely match its trajectories."""
    batch, timesteps = errors.shape[0], errors.shape[1]
    t = np.linspace(0.0, 5.0, timesteps)
    dofs = scales.shape[0]

    def score(e):
        return np.trapezoid(np.sum((e * scales) ** 2, axis=-1), t, axis=-1) / (
            t[-1] - t[0]
        )

    losses = score(errors)
    history_loss = np.stack([losses, losses], axis=0)
    mask = np.ones_like(history_loss, dtype=bool)
    reference = np.zeros((timesteps, dofs))
    trajectory = -errors  # reference - trajectory == errors
    pose = np.zeros((batch, timesteps, 6))
    pose[:, :, 6 - dofs :] = trajectory
    kwargs = {}
    if method == "collocated":
        kwargs["q_des_ts"] = reference
    return save_optimization_outputs(
        tmp_path,
        method=method,
        batch_size=batch,
        history_loss=history_loss,
        history_time=np.ones(2),
        history_finite_mask=mask,
        history_grad_norm=np.ones_like(history_loss),
        history_update_norm=np.ones_like(history_loss),
        opt_vars_history=[
            {"opt_ctr_params": {k: np.ones((batch, 3)) for k in ("Kp", "Ki", "Kd")}}
            for _ in range(2)
        ],
        init_gains={k: np.ones((batch, 3)) for k in ("Kp", "Ki", "Kd")},
        init_seed=0,
        init_scheme="log_uniform_v1",
        init_spread=3.0,
        objective_name=OBJECTIVE_NAME,
        objective_scales=scales,
        saturation_config="none",
        optimizer_metadata="sgd(0.1)",
        # Schema v3 records the two settings that previously changed a result
        # without leaving a trace in the archive. Nothing here depends on their
        # values -- these losses are scored from the stored trajectories -- but
        # they are required, so a synthetic archive has to carry them too.
        solver_dt=1e-3,
        material_damping_coefficient=7.2e1,
        t_ts=t,
        q_ts_init=trajectory if method == "collocated" else np.zeros_like(trajectory),
        q_ts_best=trajectory if method == "collocated" else np.zeros_like(trajectory),
        qd_ts_init=np.zeros_like(trajectory),
        qd_ts_best=np.zeros_like(trajectory),
        u_ts_init=np.zeros((batch, timesteps, 3)),
        u_ts_best=np.zeros((batch, timesteps, 3)),
        x_ts_init=pose,
        x_ts_best=pose,
        x_des_ts=np.zeros((timesteps, 6)),
        best_iteration=best_iteration_per_batch(history_loss, mask),
        best_batch=best_batch_from_loss(history_loss, mask),
        is_placeholder=True,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["collocated", "synergistic"])
def test_stored_losses_are_reproducible_from_their_own_trajectories(tmp_path, method):
    """The check that would have caught the legacy discrepancy."""
    dofs = 6 if method == "collocated" else 3
    scales = (
        np.asarray(configuration_space_scales(jnp.array([SEGMENT_LENGTH])))
        if method == "collocated"
        else np.asarray(
            operational_space_scales(
                jnp.array([False, False, False, True, True, True]), SEGMENT_LENGTH
            )
        )
    )
    rng = np.random.default_rng(0)
    errors = rng.normal(scale=0.01, size=(3, 41, dofs))
    data = load_results(_write_archive(tmp_path, method, scales, errors))
    result = recompute_archive_losses(data)
    np.testing.assert_allclose(
        result["recomputed_init"], result["stored_init"], rtol=1e-10
    )
    np.testing.assert_allclose(
        result["recomputed_best"], result["stored_best"], rtol=1e-10
    )


def test_rescoring_detects_a_loss_that_does_not_match_its_trajectory(tmp_path):
    """A trajectory paired with the wrong loss must not pass silently."""
    scales = np.asarray(configuration_space_scales(jnp.array([SEGMENT_LENGTH])))
    rng = np.random.default_rng(1)
    errors = rng.normal(scale=0.01, size=(2, 41, 6))
    data = load_results(_write_archive(tmp_path, "collocated", scales, errors))
    # The failure mode from issue #154: the stored loss is off by a global factor.
    data["history_loss"] = data["history_loss"] * 10.0
    result = recompute_archive_losses(data)
    assert not np.allclose(result["recomputed_init"], result["stored_init"], rtol=1e-3)


def test_rescoring_refuses_an_archive_from_a_different_objective(tmp_path):
    """Rescoring a foreign objective would compare two different quantities."""
    scales = np.asarray(configuration_space_scales(jnp.array([SEGMENT_LENGTH])))
    errors = np.zeros((2, 41, 6))
    data = load_results(_write_archive(tmp_path, "collocated", scales, errors))
    data["objective_name"] = np.asarray("some_other_objective_v1")
    with pytest.raises(ValueError, match="different quantities"):
        recompute_archive_losses(data)
