import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_allclose

from soromox.actuation import IdentityActuator
from soromox.control import PIDControllerState
from soromox.control.configuration_space import (
    FeedforwardCompensationTracker,
    PotentialCompensationRegulator,
)
from soromox.systems import SystemState

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIGURATION_SPACE_DIR = (
    REPOSITORY_ROOT
    / "paper_results"
    / "secVc_model_based_control"
    / "configuration_space_comparison"
    / "code"
)

if str(CONFIGURATION_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGURATION_SPACE_DIR))

import configuration_space_comparison_simulation  # noqa: E402


def assert_identity_actuation(robot) -> None:
    assert robot.num_actuators == robot.num_dofs
    assert len(robot.actuators) == 1
    assert isinstance(robot.actuators[0], IdentityActuator)
    assert_allclose(
        robot.actuation_matrix(jnp.zeros((robot.num_dofs,))),
        jnp.eye(robot.num_dofs),
    )


def test_configuration_space_comparison_uses_identity_actuation():
    robot, num_dofs, _ = configuration_space_comparison_simulation.create_robot()

    assert num_dofs == robot.num_dofs
    assert_identity_actuation(robot)


def test_regulation_tracking_reference_is_smooth_at_phase_boundary():
    reference, setpoint_times = (
        configuration_space_comparison_simulation.create_regulation_tracking_trajectory(
            num_dofs=6,
            t0=0.0,
            tracking_start=15.0,
            t1=30.0,
        )
    )

    assert_allclose(setpoint_times, jnp.array([0.0, 3.0, 6.0, 9.0, 12.0]))
    assert_allclose(
        reference.x_des_fn(jnp.array(15.0)),
        jnp.zeros((6,)),
        atol=1e-12,
    )
    assert_allclose(
        reference.xd_des_fn(jnp.array(15.0)),
        jnp.zeros((6,)),
        atol=1e-12,
    )
    assert_allclose(
        reference.xdd_des_fn(jnp.array(15.0)),
        jnp.zeros((6,)),
        atol=1e-12,
    )
    assert_allclose(
        reference.x_des_fn(jnp.array(4.0))[1:4],
        jnp.array([14.0, 7.0, 0.10]),
        atol=1e-12,
    )
    assert_allclose(
        reference.x_des_fn(jnp.array(7.0))[1:4],
        jnp.array([-14.0, -7.0, -0.05]),
        atol=1e-12,
    )
    assert_allclose(
        reference.x_des_fn(jnp.array(17.0))[1:4],
        jnp.array(
            [
                12.0 * jnp.sin(6.0),
                6.0 * jnp.sin(4.0 + jnp.pi / 4),
                0.06 * jnp.sin(5.0 + jnp.pi / 2),
            ]
        ),
        atol=1e-12,
    )


def test_regulation_tracking_robot_is_horizontal_and_gravity_loaded():
    robot, num_dofs, _ = (
        configuration_space_comparison_simulation.create_regulation_tracking_robot()
    )
    gravity_force = robot.gravitational_force(jnp.zeros((num_dofs,)))

    assert_allclose(
        robot.base_pose,
        jnp.array(
            configuration_space_comparison_simulation.HORIZONTAL_BASE_POSE,
        ),
    )
    assert abs(gravity_force[1]) > 1e-3
    assert_allclose(gravity_force[2], 0.0, atol=1e-12)


def test_model_free_pd_uses_same_proportional_and_derivative_gains():
    robot, _, _ = (
        configuration_space_comparison_simulation.create_regulation_tracking_robot()
    )
    pid_control = configuration_space_comparison_simulation.create_pid_control(robot)
    pd_control = configuration_space_comparison_simulation.remove_integral_action(
        pid_control
    )

    assert_allclose(pd_control.Kp, pid_control.Kp)
    assert_allclose(pd_control.Kd, pid_control.Kd)
    assert_allclose(pd_control.Ki, jnp.zeros_like(pid_control.Ki))


def test_pid_uses_unit_aware_rotational_and_linear_saturation_scales():
    robot, num_dofs, _ = configuration_space_comparison_simulation.create_robot()
    pid_control = configuration_space_comparison_simulation.create_pid_control(robot)
    expected_error_scales = jnp.array(
        [
            configuration_space_comparison_simulation.DEFAULT_ROTATIONAL_INTEGRAL_ERROR_SCALE,
            configuration_space_comparison_simulation.DEFAULT_ROTATIONAL_INTEGRAL_ERROR_SCALE,
            configuration_space_comparison_simulation.DEFAULT_ROTATIONAL_INTEGRAL_ERROR_SCALE,
            configuration_space_comparison_simulation.DEFAULT_LINEAR_INTEGRAL_ERROR_SCALE,
            configuration_space_comparison_simulation.DEFAULT_LINEAR_INTEGRAL_ERROR_SCALE,
            configuration_space_comparison_simulation.DEFAULT_LINEAR_INTEGRAL_ERROR_SCALE,
        ]
    )

    assert num_dofs == 6
    assert_allclose(1.0 / pid_control.gamma, expected_error_scales)

    large_error = 100.0 * expected_error_scales
    saturated_error = pid_control._apply_saturation(large_error)
    assert_allclose(saturated_error, expected_error_scales, rtol=1e-10)


def test_computed_torque_pid_gains_are_normalized_with_straight_inertia():
    robot, num_dofs, _ = configuration_space_comparison_simulation.create_robot()
    baseline = configuration_space_comparison_simulation.create_pid_control(
        robot,
    )
    normalized = configuration_space_comparison_simulation.normalize_pid_control_for_computed_torque(
        robot,
        baseline,
    )
    inertia_straight = robot.inertia_matrix(jnp.zeros((num_dofs,)))

    assert normalized._saturation_fn_name == baseline._saturation_fn_name
    assert_allclose(normalized.gamma, baseline.gamma)

    expected_acceleration_gains = (
        configuration_space_comparison_simulation.DEFAULT_FEEDBACK_NATURAL_FREQUENCY**2,
        configuration_space_comparison_simulation.DEFAULT_FEEDBACK_INTEGRAL_FREQUENCY
        * configuration_space_comparison_simulation.DEFAULT_FEEDBACK_NATURAL_FREQUENCY
        ** 2,
        2.0
        * configuration_space_comparison_simulation.DEFAULT_FEEDBACK_DAMPING_RATIO
        * configuration_space_comparison_simulation.DEFAULT_FEEDBACK_NATURAL_FREQUENCY,
    )
    for baseline_gain, normalized_gain, expected_acceleration_gain in (
        (baseline.Kp, normalized.Kp, expected_acceleration_gains[0]),
        (baseline.Ki, normalized.Ki, expected_acceleration_gains[1]),
        (baseline.Kd, normalized.Kd, expected_acceleration_gains[2]),
    ):
        assert normalized_gain.shape == (num_dofs, num_dofs)
        assert_allclose(
            inertia_straight @ normalized_gain,
            baseline_gain,
            rtol=1e-10,
            atol=1e-10,
        )
        assert_allclose(
            normalized_gain,
            expected_acceleration_gain * jnp.eye(num_dofs),
            rtol=1e-10,
            atol=1e-10,
        )


def test_feedforward_tracker_matches_potential_regulator_at_setpoint():
    robot, num_dofs, _ = configuration_space_comparison_simulation.create_robot()
    reference, _ = (
        configuration_space_comparison_simulation.create_regulation_tracking_trajectory(
            num_dofs=num_dofs,
            t0=0.0,
            tracking_start=15.0,
            t1=30.0,
        )
    )
    pid_control = configuration_space_comparison_simulation.create_pid_control(
        robot,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regulator = PotentialCompensationRegulator(robot, reference, pid_control)
        tracker = FeedforwardCompensationTracker(robot, reference, pid_control)
    state = SystemState(
        t=jnp.array(4.0),
        y=jnp.zeros((2 * num_dofs,)),
        u=jnp.zeros((robot.num_actuators,)),
        control_state=PIDControllerState.zero(num_dofs),
    )

    regulator_input, _ = regulator(state)
    tracker_input, _ = tracker(state)

    assert_allclose(regulator_input, tracker_input, rtol=1e-10, atol=1e-10)
