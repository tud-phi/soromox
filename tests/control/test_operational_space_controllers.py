import jax.numpy as jnp
import pytest

from soromox.control.operational_space import ImpedanceControlTracker
from soromox.systems.system_state import SystemState


class FakeOperationalSpaceDynamics:
    n_operational_space = 2

    def __init__(self, robot):
        self.robot = robot
        self.B_task = jnp.eye(2)

    def jacobian_and_time_derivative(self, q, qd):
        return jnp.eye(2), jnp.zeros((2, 2))

    def dynamically_consistent_pseudoinverse(self, q):
        return jnp.eye(2)

    def inertia_matrix(self, q):
        return self.robot.inertia_matrix(q)

    def operational_space_poses(self, q):
        return q

    def compute_task_pose_error(self, x_full, x_des_full):
        return x_des_full - x_full

    def mu_x(self, q, qd):
        return jnp.array([[0.0, 1.0], [-1.0, 0.0]])


@pytest.mark.parametrize("feedback_linearization", ["full", "partial"])
def test_impedance_control_tracker_computes_operational_space_control(
    fake_dynamics_robot_factory,
    make_reference_factory,
    feedback_linearization,
):
    robot = fake_dynamics_robot_factory(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    osd = FakeOperationalSpaceDynamics(robot)
    reference = make_reference_factory(
        x_des=[1.0, -0.5],
        xd_des=[0.2, 0.3],
        xdd_des=[0.4, -0.2],
    )

    with pytest.warns(UserWarning, match="configuration-independent"):
        controller = ImpedanceControlTracker(
            osd,
            reference,
            K_x=jnp.array([2.0, 3.0]),
            D_x=jnp.array([[0.5, 0.0], [0.0, 0.25]]),
            feedback_linearization=feedback_linearization,
        )

    state = SystemState(t=jnp.array(0.0), y=jnp.array([0.2, -0.1, 0.5, -0.25]))
    u_control, control_state_dot = controller(state)

    q, qd = jnp.split(state.y, 2)
    J, _ = osd.jacobian_and_time_derivative(q, qd)
    J_bar = osd.dynamically_consistent_pseudoinverse(q)
    tau_task_forces = robot.elastic_force(q) + robot.damping_matrix(q) @ qd
    tau_cancel_task = J.T @ (J_bar.T @ tau_task_forces)
    if feedback_linearization == "full":
        qd_coriolis = qd
    else:
        null_space_projector = jnp.eye(robot.num_internal_dofs) - J_bar @ J
        qd_coriolis = null_space_projector @ qd
    tau_cancel_coriolis = J.T @ (osd.mu_x(q, qd) @ qd_coriolis)
    xdd_des = osd.B_task.T @ reference.xdd_des_fn(state.t)
    tau_feedforward = J.T @ (osd.inertia_matrix(q) @ xdd_des)
    e_x = osd.compute_task_pose_error(q, reference.x_des_fn(state.t))
    ed_x = reference.xd_des_fn(state.t) - J @ qd
    tau_pd = J.T @ (
        controller._apply_gain(controller.K_x, e_x)
        + controller._apply_gain(controller.D_x, ed_x)
    )
    tau_expected = (
        tau_cancel_task
        + robot.gravitational_force(q)
        + tau_cancel_coriolis
        + tau_feedforward
        + tau_pd
    )
    expected = jnp.linalg.inv(robot.actuation_matrix(q)) @ tau_expected

    assert jnp.allclose(u_control, expected)
    assert control_state_dot is None


def test_impedance_control_tracker_rejects_unknown_feedback_linearization(
    fake_dynamics_robot_factory, make_reference_factory
):
    with pytest.raises(ValueError, match="either 'full' or 'partial'"):
        ImpedanceControlTracker(
            FakeOperationalSpaceDynamics(
                fake_dynamics_robot_factory(jnp.eye(2)),
            ),
            make_reference_factory([0.0, 0.0], [0.0, 0.0]),
            K_x=1.0,
            D_x=1.0,
            feedback_linearization="invalid",
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "square actuation"),
        (jnp.array([[1.0, 0.0], [0.0, 0.0]]), "invertible"),
    ],
)
def test_impedance_control_tracker_rejects_unsupported_actuation(
    fake_dynamics_robot_factory, make_reference_factory, matrix, message
):
    with pytest.raises(ValueError, match=message):
        ImpedanceControlTracker(
            FakeOperationalSpaceDynamics(fake_dynamics_robot_factory(matrix)),
            make_reference_factory([0.0, 0.0], [0.0, 0.0]),
            K_x=1.0,
            D_x=1.0,
        )
