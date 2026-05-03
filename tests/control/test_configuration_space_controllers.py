import jax.numpy as jnp
import pytest

from soromox.control import PIDControl, PIDControllerState
from soromox.control.configuration_space import (
    ComputedTorqueTracker,
    FeedforwardCompensationTracker,
    GravityCancellationRegulator,
    MixedStateFeedbackTracker,
    PotentialCancellationRegulator,
    PotentialCompensationRegulator,
)
from soromox.control.configuration_space import (
    PIDController as ConfigurationSpacePIDController,
)
from soromox.systems.system_state import SystemState


def test_computed_torque_model_based_term_uses_inverse_dynamics_and_actuation_inverse(
    fake_dynamics_robot_factory, make_reference_factory
):
    robot = fake_dynamics_robot_factory(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    reference = make_reference_factory(
        x_des=[0.0, 0.0], xd_des=[0.0, 0.0], xdd_des=[0.3, -0.4]
    )
    controller = ComputedTorqueTracker(robot, reference, PIDControl(0.0, 0.0, 0.0))
    state = SystemState(t=jnp.array(0.0), y=jnp.array([0.1, -0.2, 0.5, -0.25]))

    u_model, control_state_dot = controller.model_based_term(state)

    q, qd = jnp.split(state.y, 2)
    tau_model = (
        robot.inertia_matrix(q) @ reference.xdd_des_fn(state.t)
        + robot.coriolis_matrix(q, qd) @ qd
        + robot.gravitational_force(q)
        + robot.elastic_force(q)
        + robot.damping_matrix(q) @ qd
    )
    expected = jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model
    assert jnp.allclose(u_model, expected)
    assert control_state_dot is None


def test_computed_torque_feedback_term_maps_pid_acceleration_through_dynamics(
    fake_dynamics_robot_factory, make_reference_factory
):
    robot = fake_dynamics_robot_factory(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    reference = make_reference_factory(x_des=[0.5, -0.1], xd_des=[0.2, 0.3])
    pid = PIDControl(
        Kp=jnp.array([2.0, 3.0]),
        Ki=jnp.array([0.5, 0.25]),
        Kd=jnp.array([1.0, 0.5]),
    )
    controller = ComputedTorqueTracker(robot, reference, pid)
    control_state = PIDControllerState(integral_error=jnp.array([0.1, -0.2]))
    state = SystemState(
        t=jnp.array(0.0),
        y=jnp.array([0.1, -0.2, -0.3, 0.4]),
        control_state=control_state,
    )

    u_feedback, control_state_dot = controller.error_based_feedback_term(state)

    q, qd = jnp.split(state.y, 2)
    e = reference.x_des_fn(state.t) - q
    ed = reference.xd_des_fn(state.t) - qd
    qdd_feedback, expected_integral_dot = pid(e, ed, control_state.integral_error)
    expected = jnp.linalg.inv(robot.actuation_matrix(q)) @ (
        robot.inertia_matrix(q) @ qdd_feedback
    )
    assert jnp.allclose(u_feedback, expected)
    assert jnp.allclose(control_state_dot.integral_error, expected_integral_dot)


def test_computed_torque_supports_overactuated_pseudoinverse_and_rejects_underactuation(
    fake_dynamics_robot_factory, make_reference_factory
):
    overactuated_robot = fake_dynamics_robot_factory(
        jnp.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    )
    reference = make_reference_factory([0.0, 0.0], [0.0, 0.0], [0.1, 0.2])

    with pytest.warns(UserWarning, match="overactuated"):
        controller = ComputedTorqueTracker(
            overactuated_robot, reference, PIDControl(0.0, 0.0, 0.0)
        )

    state = SystemState(t=jnp.array(0.0), y=jnp.array([0.0, 0.0, 0.0, 0.0]))
    u_model, _ = controller.model_based_term(state)
    q, qd = jnp.split(state.y, 2)
    tau_model = (
        overactuated_robot.inertia_matrix(q) @ reference.xdd_des_fn(state.t)
        + overactuated_robot.coriolis_matrix(q, qd) @ qd
        + overactuated_robot.gravitational_force(q)
        + overactuated_robot.elastic_force(q)
        + overactuated_robot.damping_matrix(q) @ qd
    )
    assert jnp.allclose(
        u_model, jnp.linalg.pinv(overactuated_robot.actuation_matrix(q)) @ tau_model
    )

    with pytest.raises(ValueError, match="underactuated"):
        ComputedTorqueTracker(
            fake_dynamics_robot_factory(jnp.array([[1.0], [0.0]])),
            reference,
            PIDControl(0.0, 0.0, 0.0),
        )


def test_configuration_space_pid_maps_generalized_pid_force_through_actuation_transpose(
    fake_dynamics_robot_factory, make_reference_factory
):
    robot = fake_dynamics_robot_factory(jnp.array([[1.0, 0.5], [0.0, 2.0]]))
    reference = make_reference_factory(x_des=[0.5, -0.1], xd_des=[0.2, 0.3])
    pid = PIDControl(
        Kp=jnp.array([2.0, 3.0]),
        Ki=jnp.array([0.5, 0.25]),
        Kd=jnp.array([1.0, 0.5]),
    )
    controller = ConfigurationSpacePIDController(robot, reference, pid)
    control_state = PIDControllerState(integral_error=jnp.array([0.1, -0.2]))
    state = SystemState(
        t=jnp.array(0.0),
        y=jnp.array([0.1, -0.2, -0.3, 0.4]),
        control_state=control_state,
    )

    u_feedback, control_state_dot = controller.error_based_feedback_term(state)

    q, qd = jnp.split(state.y, 2)
    e = reference.x_des_fn(state.t) - q
    ed = reference.xd_des_fn(state.t) - qd
    tau, expected_integral_dot = pid(e, ed, control_state.integral_error)
    expected = robot.actuation_matrix(q).T @ tau
    assert jnp.allclose(u_feedback, expected)
    assert jnp.allclose(control_state_dot.integral_error, expected_integral_dot)


def test_configuration_space_pid_warns_for_singular_and_underactuated_systems(
    fake_dynamics_robot_factory, make_reference_factory
):
    reference = make_reference_factory([0.0, 0.0], [0.0, 0.0])
    pid = PIDControl(0.0, 0.0, 0.0)

    with pytest.warns(UserWarning, match="singular"):
        ConfigurationSpacePIDController(
            fake_dynamics_robot_factory(jnp.array([[1.0, 0.0], [0.0, 0.0]])),
            reference,
            pid,
        )

    with pytest.warns(UserWarning, match="underactuated"):
        ConfigurationSpacePIDController(
            fake_dynamics_robot_factory(jnp.array([[1.0], [0.0]])),
            reference,
            pid,
        )


def expected_full_inverse_dynamics(robot, reference, state):
    q, _ = jnp.split(state.y, 2)
    q_des = reference.x_des_fn(state.t)
    qd_des = reference.xd_des_fn(state.t)
    qdd_des = reference.xdd_des_fn(state.t)
    tau_model = (
        robot.inertia_matrix(q_des) @ qdd_des
        + robot.coriolis_matrix(q_des, qd_des) @ qd_des
        + robot.gravitational_force(q_des)
        + robot.elastic_force(q_des)
        + robot.damping_matrix(q_des) @ qd_des
    )
    return jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model


def expected_mixed_state_dynamics(robot, reference, state):
    q, qd = jnp.split(state.y, 2)
    q_des = reference.x_des_fn(state.t)
    qd_des = reference.xd_des_fn(state.t)
    qdd_des = reference.xdd_des_fn(state.t)
    tau_model = (
        robot.inertia_matrix(q) @ qdd_des
        + robot.coriolis_matrix(q, qd) @ qd_des
        + robot.gravitational_force(q)
        + robot.elastic_force(q_des)
        + robot.damping_matrix(q) @ qd_des
    )
    return jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model


def expected_gravity_cancellation(robot, reference, state):
    q, _ = jnp.split(state.y, 2)
    q_des = reference.x_des_fn(state.t)
    tau_model = robot.gravitational_force(q) + robot.elastic_force(q_des)
    return jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model


def expected_potential_cancellation(robot, reference, state):
    q, _ = jnp.split(state.y, 2)
    tau_model = robot.gravitational_force(q) + robot.elastic_force(q)
    return jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model


def expected_potential_compensation(robot, reference, state):
    q, _ = jnp.split(state.y, 2)
    q_des = reference.x_des_fn(state.t)
    tau_model = robot.gravitational_force(q_des) + robot.elastic_force(q_des)
    return jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_model


@pytest.mark.parametrize(
    ("controller_cls", "expected_fn", "warning_match"),
    [
        (
            FeedforwardCompensationTracker,
            expected_full_inverse_dynamics,
            "configuration-independent",
        ),
        (
            MixedStateFeedbackTracker,
            expected_mixed_state_dynamics,
            "Christoffel symbols",
        ),
        (
            GravityCancellationRegulator,
            expected_gravity_cancellation,
            "configuration-independent",
        ),
        (
            PotentialCancellationRegulator,
            expected_potential_cancellation,
            "configuration-independent",
        ),
        (
            PotentialCompensationRegulator,
            expected_potential_compensation,
            "configuration-independent",
        ),
    ],
)
def test_configuration_space_model_based_controllers_compute_expected_terms(
    fake_dynamics_robot_factory,
    make_reference_factory,
    controller_cls,
    expected_fn,
    warning_match,
):
    robot = fake_dynamics_robot_factory(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    reference = make_reference_factory(
        x_des=[0.5, -0.1], xd_des=[0.2, 0.3], xdd_des=[0.1, -0.2]
    )
    state = SystemState(t=jnp.array(0.0), y=jnp.array([0.1, -0.2, -0.3, 0.4]))

    with pytest.warns(UserWarning) as caught:
        controller = controller_cls(robot, reference, PIDControl(0.0, 0.0, 0.0))
    assert any(warning_match in str(warning.message) for warning in caught)

    u_model, control_state_dot = controller.model_based_term(state)

    assert jnp.allclose(u_model, expected_fn(robot, reference, state))
    assert control_state_dot is None
