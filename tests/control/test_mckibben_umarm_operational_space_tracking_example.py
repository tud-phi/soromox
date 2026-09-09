import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from examples.control.operational_space import track_mckibben_umarm as example
from examples.control.operational_space.track_mckibben_umarm import (
    CIRCLE_PERIOD,
    DEFAULT_INITIAL_CONFIGURATION,
    DEFAULT_MAX_DELTA_PRESSURE,
    DEFAULT_NOMINAL_PRESSURE,
    POSITION_TASK_SELECTOR,
    antagonistic_pressure_map,
    build_robot,
    create_circle_position_reference,
    create_controller,
    simulate,
    tracking_metrics,
)
from soromox.systems import McKibbenActuatedUMArm


@pytest.fixture(scope="module")
def robot():
    return build_robot()


@pytest.fixture(scope="module")
def default_result(robot):
    return simulate(robot)


def test_reference_is_position_only_two_second_circle():
    initial_position = jnp.array([0.4, 0.0, 0.37])
    center = jnp.array([0.0, 0.0, 0.37])
    normal = jnp.array([0.0, 0.0, 1.0])
    position_des_fn = create_circle_position_reference(
        initial_position,
        center,
        normal,
    )
    sample_times = jnp.linspace(0.0, CIRCLE_PERIOD, 101)
    positions = jnp.stack([position_des_fn(t) for t in sample_times])

    assert CIRCLE_PERIOD == 2.0
    assert_allclose(position_des_fn(jnp.array(0.0)), initial_position, atol=1.0e-12)
    assert_allclose(
        position_des_fn(jnp.array(CIRCLE_PERIOD)),
        initial_position,
        atol=1.0e-12,
    )
    assert_allclose((positions - center) @ normal, 0.0, atol=1.0e-12)
    assert_allclose(
        jnp.linalg.norm(positions - center, axis=1),
        0.4,
        atol=1.0e-12,
    )


def test_operational_space_selects_position_without_orientation(robot):
    _, _, operational_space, circle_reference, _ = create_controller(robot)
    pose_at_start = circle_reference.trajectory.x_des_fn(jnp.array(0.0))
    pose_after_quarter_lap = circle_reference.trajectory.x_des_fn(jnp.array(0.5))

    assert operational_space.n_operational_space == 3
    assert_allclose(operational_space.task_selector, POSITION_TASK_SELECTOR)
    assert_allclose(pose_after_quarter_lap[:3], pose_at_start[:3], atol=1.0e-12)
    assert not bool(jnp.allclose(pose_after_quarter_lap[3:], pose_at_start[3:]))


def test_default_circle_is_base_parallel_and_nearly_robot_length(robot):
    _, _, _, circle_reference, _ = create_controller(robot)
    base_transform = robot.base_transform
    base_axis = base_transform[:3, 0]
    center_from_base = circle_reference.center - base_transform[:3, 3]

    assert_allclose(
        jnp.abs(circle_reference.normal @ base_axis),
        1.0,
        atol=1.0e-12,
    )
    assert_allclose(jnp.cross(center_from_base, base_axis), 0.0, atol=1.0e-12)
    diameter_ratio = 2.0 * float(circle_reference.radius) / float(robot.length)
    assert 0.75 < diameter_ratio < 1.0


def test_antagonistic_coordinates_make_actuation_square_and_invertible(robot):
    _, control_model, _, _, nominal_pressures = create_controller(robot)
    pressure_map = antagonistic_pressure_map(robot)
    q = DEFAULT_INITIAL_CONFIGURATION
    virtual_actuation = control_model.actuation_matrix(q)

    assert isinstance(control_model, McKibbenActuatedUMArm)
    assert pressure_map.shape == (24, 12)
    assert virtual_actuation.shape == (12, 12)
    assert int(jnp.linalg.matrix_rank(virtual_actuation)) == 12
    assert_allclose(
        virtual_actuation,
        robot.actuation_matrix(q) @ pressure_map,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert_allclose(nominal_pressures, DEFAULT_NOMINAL_PRESSURE)


def test_reduced_dynamics_match_expanded_physical_pressures(robot):
    _, control_model, _, _, nominal_pressures = create_controller(robot)
    q = jnp.linspace(-0.12, 0.16, robot.num_internal_dofs)
    qd = jnp.linspace(0.07, -0.05, robot.num_internal_dofs)
    delta_pressure = jnp.linspace(-2.0e4, 2.0e4, control_model.num_actuators)
    physical_pressures = nominal_pressures + control_model.pressure_map @ delta_pressure
    y = jnp.concatenate([q, qd])

    reduced_yd = control_model.forward_dynamics(jnp.array(0.0), y, (delta_pressure,))
    physical_yd = robot.forward_dynamics(jnp.array(0.0), y, (physical_pressures,))

    assert_allclose(reduced_yd, physical_yd, rtol=1.0e-11, atol=1.0e-11)


def test_default_rollout_tracks_circle_without_pressure_saturation(default_result):
    metrics = tracking_metrics(default_result)
    pressure_groups = default_result.pressures.reshape((-1, 6, 4))

    assert bool(jnp.all(jnp.isfinite(default_result.q)))
    assert bool(jnp.all(jnp.isfinite(default_result.qd)))
    assert bool(jnp.all(jnp.isfinite(default_result.position)))
    assert float(metrics["position_rmse"]) < 1.0e-4
    assert float(metrics["maximum_position_error"]) < 2.0e-4
    assert float(jnp.min(default_result.pressures)) > 0.0
    assert float(jnp.max(default_result.pressures)) < (
        DEFAULT_NOMINAL_PRESSURE + DEFAULT_MAX_DELTA_PRESSURE
    )
    assert_allclose(
        pressure_groups[:, :, 1] + pressure_groups[:, :, 3],
        2.0 * DEFAULT_NOMINAL_PRESSURE,
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    assert_allclose(
        pressure_groups[:, :, 0] + pressure_groups[:, :, 2],
        2.0 * DEFAULT_NOMINAL_PRESSURE,
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_render_motion_passes_pressures_to_umarm_viser(
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
    assert_allclose(captured["sequence_kwargs"]["q_ts"], default_result.q)
    assert_allclose(
        captured["sequence_kwargs"]["pressures"],
        default_result.pressures,
    )
    assert captured["sequence_kwargs"]["static_spheres_positions"].shape[0] > 40
    assert_allclose(
        captured["sequence_kwargs"]["dynamic_spheres_positions"][0],
        default_result.position_des,
    )
    assert captured["sequence_kwargs"]["actuator_color_mode"] == "pressure"
