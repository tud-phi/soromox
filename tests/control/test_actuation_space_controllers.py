import jax.numpy as jnp
import pytest

from soromox.control import PIDControl, PIDControllerState
from soromox.control.actuation_space import (
    FeedforwardCompensationTracker,
)
from soromox.control.actuation_space import (
    PIDController as ActuationSpacePIDController,
)
from soromox.systems.system_state import SystemState


class FakeActuationSpaceDynamics:
    n_actuated = 2
    n_unactuated = 1

    def __init__(self, robot, converted_reference=None):
        self.robot = robot
        self.converted_reference = converted_reference
        self.dynamics_terms_calls = 0

    def convert_reference_trajectory(self, reference_trajectory):
        return self.converted_reference

    def actuated_unactuated_coordinates(self, q):
        return jnp.array([q[0] + 1.0, q[1] - 1.0, q[2]])

    def jacobian(self, q):
        return jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def dynamics_terms(self, q, qd):
        del q, qd
        self.dynamics_terms_calls += 1
        return (
            jnp.diag(jnp.array([2.0, 3.0, 4.0])),
            jnp.array([0.1, -0.2, 0.3]),
            jnp.array([0.4, 0.5, -0.6]),
        )

    def elastic_force(self, q):
        del q
        return jnp.array([0.2, -0.1, 0.3])

    def damping_matrix(self, q):
        del q
        return jnp.diag(jnp.array([0.5, 0.25, 0.1]))


def test_actuation_space_pid_uses_actuated_coordinates_and_updates_gains(
    fake_robot_factory, make_reference_factory
):
    reference = make_reference_factory(x_des=[2.5, -0.5, 0.0], xd_des=[0.4, 0.1, 0.0])
    pid = PIDControl(
        Kp=jnp.array([2.0, 3.0]),
        Ki=jnp.array([0.5, 0.25]),
        Kd=jnp.array([1.0, 0.5]),
    )
    controller = ActuationSpacePIDController(
        FakeActuationSpaceDynamics(fake_robot_factory(jnp.eye(3, 2))),
        reference,
        pid,
        reference_in_configuration_space=False,
    )
    control_state = PIDControllerState(integral_error=jnp.array([0.1, -0.2]))
    state = SystemState(
        t=jnp.array(0.0),
        y=jnp.array([1.0, 0.25, 9.0, 0.2, -0.3, 0.7]),
        control_state=control_state,
    )

    tau, control_state_dot = controller.error_based_feedback_term(state)

    q, qd = jnp.split(state.y, 2)
    y_act = controller.actuation_space_dynamics.actuated_unactuated_coordinates(q)
    yd_act = controller.actuation_space_dynamics.jacobian(q) @ qd
    e_a = reference.x_des_fn(state.t)[:2] - y_act[:2]
    ed_a = reference.xd_des_fn(state.t)[:2] - yd_act[:2]
    expected_tau, expected_integral_dot = pid(e_a, ed_a, control_state.integral_error)
    assert jnp.allclose(tau, expected_tau)
    assert jnp.allclose(control_state_dot.integral_error, expected_integral_dot)

    updated = controller.update_gains({"Kp": jnp.array([9.0, 8.0])})
    assert jnp.allclose(updated.pid_control.Kp, jnp.array([9.0, 8.0]))
    assert jnp.allclose(controller.pid_control.Kp, jnp.array([2.0, 3.0]))


def test_actuation_space_pid_converts_configuration_space_reference(
    fake_robot_factory, make_reference_factory
):
    original_reference = make_reference_factory([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    converted_reference = make_reference_factory([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    asd = FakeActuationSpaceDynamics(
        fake_robot_factory(jnp.eye(3, 2)),
        converted_reference=converted_reference,
    )

    controller = ActuationSpacePIDController(
        asd,
        original_reference,
        PIDControl(0.0, 0.0, 0.0),
        reference_in_configuration_space=True,
    )

    assert controller.reference_trajectory_config_space is original_reference
    assert controller.reference_trajectory is converted_reference


def test_actuation_space_feedforward_uses_contracted_dynamics_terms(
    fake_robot_factory, make_reference_factory
):
    reference = make_reference_factory(
        x_des=[0.2, -0.1, 0.3],
        xd_des=[0.1, 0.2, -0.1],
        xdd_des=[0.3, -0.2, 0.1],
    )
    converted_reference = make_reference_factory(
        x_des=[1.2, -1.1, 0.3],
        xd_des=[0.1, 0.4, -0.1],
        xdd_des=[0.3, -0.4, 0.1],
    )
    asd = FakeActuationSpaceDynamics(
        fake_robot_factory(jnp.eye(3, 2)),
        converted_reference=converted_reference,
    )

    with pytest.warns(UserWarning, match="underactuated"):
        controller = FeedforwardCompensationTracker(
            asd,
            reference,
            PIDControl(0.0, 0.0, 0.0),
            reference_in_configuration_space=True,
        )

    state = SystemState(t=jnp.array(0.0), y=jnp.zeros((6,)))
    tau_model, control_state_dot = controller.model_based_term(state)

    M_y, Cyd, G_y = (
        jnp.diag(jnp.array([2.0, 3.0, 4.0])),
        jnp.array([0.1, -0.2, 0.3]),
        jnp.array([0.4, 0.5, -0.6]),
    )
    yd_des = converted_reference.xd_des_fn(state.t)
    ydd_des = converted_reference.xdd_des_fn(state.t)
    tau_el = asd.elastic_force(reference.x_des_fn(state.t))
    D_y = asd.damping_matrix(reference.x_des_fn(state.t))
    expected = (M_y @ ydd_des + Cyd + G_y + tau_el + D_y @ yd_des)[:2]

    assert jnp.allclose(tau_model, expected)
    assert asd.dynamics_terms_calls == 1
    assert control_state_dot is None
