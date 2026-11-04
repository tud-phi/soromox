import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.operational_space import (
    ImpedanceControlTracker,
    SynergisticController,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.systems import LinkSpec, PlanarPCS
from soromox.systems.system_state import SystemState
from soromox.utils.geometry import poses


class FakeOperationalSpaceDynamics:
    n_operational_space = 2

    def __init__(self, robot):
        self.robot = robot
        self.B_task = jnp.eye(2)

    def jacobian_and_time_derivative(self, q, qd):
        return jnp.array([[1.0, 0.5], [0.0, 2.0]]), jnp.array([[0.1, 0.0], [0.2, -0.1]])

    def operational_space_poses(self, q):
        return q

    def compute_task_pose_error(self, x_full, x_des_full):
        return x_des_full - x_full


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
    J, Jd = osd.jacobian_and_time_derivative(q, qd)
    M_inv = jnp.linalg.inv(robot.inertia_matrix(q))
    Lambda = jnp.linalg.inv(J @ M_inv @ J.T)
    J_bar = M_inv @ J.T @ Lambda
    mu_x = Lambda @ (J @ M_inv @ robot.coriolis_matrix(q, qd) - Jd)
    tau_task_forces = robot.elastic_force(q) + robot.damping_matrix(q) @ qd
    tau_cancel_task = J.T @ (J_bar.T @ tau_task_forces)
    if feedback_linearization == "full":
        qd_coriolis = qd
    else:
        null_space_projector = jnp.eye(robot.num_internal_dofs) - J_bar @ J
        qd_coriolis = null_space_projector @ qd
    tau_cancel_coriolis = J.T @ (mu_x @ qd_coriolis)
    xdd_des = osd.B_task.T @ reference.xdd_des_fn(state.t)
    tau_feedforward = J.T @ (Lambda @ xdd_des)
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


def test_synergistic_controller_matches_explicit_projection():
    # A tendon-driven PlanarPCS with one task coordinate per tendon.
    link = LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1.0e5,
        shear_modulus=4.0e4,
        material_damping_coefficient=1.0e2,
        reference_strain=[0.0, 1.0, 0.0],
    )
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.005, 0.0], [0.0, -0.005, 0.0]]),
        start_segment_index=0,
        end_segment_index=(1, 1),
    )
    robot = PlanarPCS.from_links(
        [link] * 2,
        gravity=jnp.array([0.0, -9.81]),
        base_pose=poses.planar_mounting_pose("hanging"),
        actuators=ThreadlikeActuator.tendons(routing),
    )
    osd = OperationalSpaceDynamics(
        robot=robot, s_ps=jnp.array([0.2]), task_selector=jnp.array([False, True, True])
    )
    q_des = 0.2 * jnp.ones((robot.num_internal_dofs,))
    reference = ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: osd.operational_space_poses((1.0 + t) * q_des),
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )
    o = osd.n_operational_space
    pid = PIDControl(
        Kp=2.0 * jnp.ones((o,)), Ki=0.5 * jnp.ones((o,)), Kd=jnp.ones((o,))
    )
    controller = SynergisticController(osd, reference, pid)
    y = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (robot.state_size,))
    state = SystemState(
        t=jnp.array(0.3),
        y=robot.project_state(y),
        control_state=PIDControllerState(integral_error=0.1 * jnp.arange(1.0, o + 1.0)),
    )

    u, _ = controller(state)

    # u = (J M^-1 A)^-1 J M^-1 J^T f with explicit inverses
    q, _, _ = robot.split_state(state.y)
    J = osd.jacobian(q)
    f, _ = controller.error_based_feedback_term(state)
    M_inv = jnp.linalg.inv(robot.inertia_matrix(q))
    A = robot.actuation_matrix(q)
    expected = jnp.linalg.inv(J @ M_inv @ A) @ J @ M_inv @ J.T @ f
    assert_allclose(u, expected, rtol=1e-9, atol=1e-9 * jnp.max(jnp.abs(expected)))
