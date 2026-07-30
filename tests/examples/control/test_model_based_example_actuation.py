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

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION_SPACE_DIR = (
    REPOSITORY_ROOT / "examples" / "control" / "configuration_space"
)
OPERATIONAL_SPACE_DIR = REPOSITORY_ROOT / "examples" / "control" / "operational_space"

for module_dir in (CONFIGURATION_SPACE_DIR, OPERATIONAL_SPACE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import configuration_space_comparison_simulation  # noqa: E402
import pcs_impedance_example  # noqa: E402
from trajectory_primitives import TaskSpaceTrajectoryConfig  # noqa: E402


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
        reference.x_des_fn(jnp.array(17.0))[1],
        10.0 * jnp.sin(6.0),
        atol=1e-12,
    )


def test_computed_torque_pid_gains_are_normalized_with_straight_inertia():
    robot, num_dofs, num_segments = (
        configuration_space_comparison_simulation.create_robot()
    )
    baseline = configuration_space_comparison_simulation.create_pid_control(
        num_dofs,
        num_segments,
    )
    normalized = configuration_space_comparison_simulation.normalize_pid_control_for_computed_torque(
        robot,
        baseline,
    )
    inertia_straight = robot.inertia_matrix(jnp.zeros((num_dofs,)))

    for baseline_gain, normalized_gain in (
        (baseline.Kp, normalized.Kp),
        (baseline.Ki, normalized.Ki),
        (baseline.Kd, normalized.Kd),
    ):
        assert normalized_gain.shape == (num_dofs, num_dofs)
        assert_allclose(
            inertia_straight @ normalized_gain,
            jnp.diag(baseline_gain),
            rtol=1e-10,
            atol=1e-10,
        )


def test_feedforward_tracker_matches_potential_regulator_at_setpoint():
    robot, num_dofs, num_segments = (
        configuration_space_comparison_simulation.create_robot()
    )
    reference, _ = (
        configuration_space_comparison_simulation.create_regulation_tracking_trajectory(
            num_dofs=num_dofs,
            t0=0.0,
            tracking_start=15.0,
            t1=30.0,
        )
    )
    pid_control = configuration_space_comparison_simulation.create_pid_control(
        num_dofs,
        num_segments,
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


def test_operational_space_impedance_uses_identity_actuation():
    problem = pcs_impedance_example.build_problem(TaskSpaceTrajectoryConfig())

    assert problem.num_dofs == problem.robot.num_dofs
    assert_identity_actuation(problem.robot)
