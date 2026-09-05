import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from examples.control.configuration_space import track_mckibben_umarm as example
from examples.control.configuration_space.track_mckibben_umarm import (
    DEFAULT_DURATION,
    DEFAULT_MAX_DELTA_PRESSURE,
    DEFAULT_NOMINAL_PRESSURE,
    REFERENCE_AMPLITUDES,
    antagonistic_pressure_map,
    build_robot,
    create_controller,
    create_reference_trajectory,
    simulate,
    tracking_metrics,
)
from soromox.control import ReferenceTrajectory
from soromox.systems import McKibbenActuatedUMArm, SystemState


@pytest.fixture(scope="module")
def robot():
    return build_robot()


@pytest.fixture(scope="module")
def default_result(robot):
    return simulate(robot)


def test_reference_is_large_dynamic_and_starts_and_ends_at_rest():
    reference = create_reference_trajectory()
    assert reference.x_des_fn is not None
    assert reference.xd_des_fn is not None
    assert reference.xdd_des_fn is not None

    q0 = reference.x_des_fn(jnp.array(0.0))
    q1 = reference.x_des_fn(jnp.array(DEFAULT_DURATION))
    qd0 = reference.xd_des_fn(jnp.array(0.0))
    qd1 = reference.xd_des_fn(jnp.array(DEFAULT_DURATION))
    q_des = reference.x_des_ts
    assert q_des is not None

    assert_allclose(q0, jnp.zeros_like(q0), atol=1.0e-12)
    assert_allclose(q1, jnp.zeros_like(q1), atol=1.0e-12)
    assert_allclose(qd0, jnp.zeros_like(qd0), atol=1.0e-12)
    assert_allclose(qd1, jnp.zeros_like(qd1), atol=1.0e-12)
    assert float(jnp.max(jnp.abs(q_des))) > 0.20
    assert float(jnp.max(jnp.abs(reference.xd_des_ts))) > 0.5
    assert float(jnp.max(jnp.abs(reference.xdd_des_ts))) > 2.0
    assert jnp.all(jnp.max(jnp.abs(q_des), axis=0) <= REFERENCE_AMPLITUDES)


def test_antagonistic_pressure_map_has_expected_channel_order(robot):
    pressure_map = antagonistic_pressure_map(robot)
    expected_group_map = jnp.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])

    assert pressure_map.shape == (24, 12)
    for group_index in range(6):
        row_slice = slice(4 * group_index, 4 * group_index + 4)
        column_slice = slice(2 * group_index, 2 * group_index + 2)
        assert_allclose(
            pressure_map[row_slice, column_slice],
            expected_group_map,
        )

    assert_allclose(jnp.sum(pressure_map, axis=0), jnp.zeros((12,)))
    assert_allclose(jnp.sum(jnp.abs(pressure_map), axis=0), 2.0 * jnp.ones((12,)))


def test_example_mounts_umarm_from_high_base_toward_world_negative_z(robot):
    q = jnp.zeros((robot.num_internal_dofs,))
    s = jnp.array([0.0, robot.length])
    poses = robot.forward_kinematics_abscissa_batched(q, s)
    base_position = poses[0, :3, 3]
    tip_position = poses[-1, :3, 3]

    assert_allclose(base_position, jnp.array([0.0, 0.0, 1.2]), atol=1.0e-12)
    assert float(tip_position[2]) < float(base_position[2])
    assert float(base_position[2] - tip_position[2]) > 1.0


def test_reduced_model_includes_map_and_nominal_pressure_force(robot):
    reference = create_reference_trajectory()
    _, control_model, nominal_pressures = create_controller(robot, reference)
    q = jnp.linspace(-0.18, 0.18, robot.num_internal_dofs)
    physical_actuation = robot.actuation_matrix(q)

    assert isinstance(control_model, McKibbenActuatedUMArm)
    assert control_model.num_actuators == robot.num_internal_dofs
    assert_allclose(
        control_model.actuation_matrix(q),
        physical_actuation @ control_model.pressure_map,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert_allclose(
        control_model.elastic_force(q) + physical_actuation @ nominal_pressures,
        robot.elastic_force(q),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_reduced_model_forward_dynamics_matches_physical_pressure_expansion(robot):
    reference = create_reference_trajectory()
    _, control_model, nominal_pressures = create_controller(robot, reference)
    q = jnp.linspace(-0.14, 0.16, robot.num_internal_dofs)
    qd = jnp.linspace(0.08, -0.06, robot.num_internal_dofs)
    delta_pressure = jnp.linspace(-2.0e4, 2.0e4, control_model.num_actuators)
    physical_pressures = nominal_pressures + control_model.pressure_map @ delta_pressure
    y = jnp.concatenate([q, qd])

    reduced_yd = control_model.forward_dynamics(jnp.array(0.0), y, (delta_pressure,))
    physical_yd = robot.forward_dynamics(jnp.array(0.0), y, (physical_pressures,))

    assert_allclose(reduced_yd, physical_yd, rtol=1.0e-11, atol=1.0e-11)


def test_feedforward_term_balances_physical_inverse_dynamics(robot):
    q_des = jnp.linspace(-0.12, 0.12, robot.num_internal_dofs)
    zeros = jnp.zeros_like(q_des)
    reference = ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: q_des,
        xd_des_fn=lambda t: zeros,
        xdd_des_fn=lambda t: zeros,
    )
    controller, control_model, nominal_pressures = create_controller(robot, reference)
    state = SystemState(t=jnp.array(0.5), y=jnp.concatenate([q_des, zeros]))

    differential_pressure, control_state_dot = controller.tracker.model_based_term(
        state
    )
    physical_pressures = (
        nominal_pressures + control_model.pressure_map @ differential_pressure
    )
    actual_force = robot.actuation_matrix(q_des) @ physical_pressures
    _, coriolis_force, gravity_force = robot.dynamics_terms(q_des, zeros)
    expected_force = coriolis_force + gravity_force + robot.elastic_force(q_des)

    assert control_state_dot is None
    assert_allclose(actual_force, expected_force, rtol=1.0e-9, atol=1.0e-9)


def test_controller_clips_deltas_without_breaking_antagonistic_balance(robot):
    q_des = jnp.full((robot.num_internal_dofs,), 0.25)
    zeros = jnp.zeros_like(q_des)
    reference = ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: q_des,
        xd_des_fn=lambda t: zeros,
        xdd_des_fn=lambda t: zeros,
    )
    controller, _, nominal_pressures = create_controller(robot, reference)
    state = SystemState(
        t=jnp.array(0.5),
        y=jnp.concatenate([-q_des, zeros]),
        u=nominal_pressures,
    )

    physical_delta, control_state_dot = controller(state)
    physical_pressures = (nominal_pressures + physical_delta).reshape((6, 4))

    assert control_state_dot is None
    assert float(jnp.min(physical_pressures)) >= 0.0
    assert float(jnp.max(physical_pressures)) <= (
        DEFAULT_NOMINAL_PRESSURE + DEFAULT_MAX_DELTA_PRESSURE
    )
    expected_pair_sum = 2.0 * DEFAULT_NOMINAL_PRESSURE * jnp.ones((6,))
    assert_allclose(
        physical_pressures[:, 1] + physical_pressures[:, 3],
        expected_pair_sum,
    )
    assert_allclose(
        physical_pressures[:, 0] + physical_pressures[:, 2],
        expected_pair_sum,
    )


def test_default_dynamic_rollout_tracks_without_pressure_saturation(default_result):
    metrics = tracking_metrics(default_result)
    pressure_groups = default_result.pressures.reshape((-1, 6, 4))

    assert bool(jnp.all(jnp.isfinite(default_result.q)))
    assert bool(jnp.all(jnp.isfinite(default_result.qd)))
    assert bool(jnp.all(jnp.isfinite(default_result.pressures)))
    assert float(metrics["overall_rmse"]) < 1.0e-8
    assert float(jnp.max(metrics["per_joint_rmse"])) < 1.0e-8
    assert float(metrics["maximum_absolute_error"]) < 1.0e-7
    assert float(jnp.max(jnp.abs(default_result.q))) < 0.30
    assert float(jnp.min(default_result.pressures)) > 0.0
    assert float(jnp.max(default_result.pressures)) < 110.0e3

    expected_pair_sum = 2.0 * DEFAULT_NOMINAL_PRESSURE
    assert_allclose(
        pressure_groups[:, :, 1] + pressure_groups[:, :, 3],
        expected_pair_sum,
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    assert_allclose(
        pressure_groups[:, :, 0] + pressure_groups[:, :, 2],
        expected_pair_sum,
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_render_motion_passes_dynamic_pressures_to_umarm_viser(
    monkeypatch,
    robot,
    default_result,
):
    captured = {}

    class FakeRenderer:
        def __init__(self, rendered_robot, **kwargs):
            captured["robot"] = rendered_robot
            captured["init_kwargs"] = kwargs

        def render_sequence(self, **kwargs):
            captured["sequence_kwargs"] = kwargs

    monkeypatch.setattr(example, "UMArmViserRenderer", FakeRenderer)

    example.render_motion(robot, default_result)

    assert captured["robot"] is robot
    assert captured["init_kwargs"]["actuator_color_mode"] == "pressure"
    sequence_kwargs = captured["sequence_kwargs"]
    assert_allclose(sequence_kwargs["ts"], default_result.t)
    assert_allclose(sequence_kwargs["q_ts"], default_result.q)
    assert_allclose(sequence_kwargs["pressures"], default_result.pressures)
    assert sequence_kwargs["actuator_color_mode"] == "pressure"
    assert sequence_kwargs["plot_configurations"] is True


def test_render_motion_records_viser_at_video_frame_rate(
    monkeypatch,
    tmp_path,
    robot,
    default_result,
):
    captured = {}

    class FakeRenderer:
        def __init__(self, rendered_robot, **kwargs):
            captured["robot"] = rendered_robot
            captured["init_kwargs"] = kwargs

        def render_sequence(self, **kwargs):
            captured["sequence_kwargs"] = kwargs

    monkeypatch.setattr(example, "UMArmViserRenderer", FakeRenderer)
    record_path = tmp_path / "umarm.mp4"

    example.render_motion(robot, default_result, record_path=record_path)

    assert captured["init_kwargs"]["width"] == 1280
    assert captured["init_kwargs"]["height"] == 720
    sequence_kwargs = captured["sequence_kwargs"]
    assert sequence_kwargs["record_path"] == str(record_path)
    assert sequence_kwargs["record_client_timeout"] == 120.0
    assert sequence_kwargs["ts"].shape[0] == 101
    assert_allclose(sequence_kwargs["ts"][0], default_result.t[0])
    assert_allclose(sequence_kwargs["ts"][-1], default_result.t[-1])
    assert_allclose(sequence_kwargs["q_ts"], default_result.q[::3])
    assert_allclose(sequence_kwargs["pressures"], default_result.pressures[::3])
