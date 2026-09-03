from pathlib import Path

import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, Euler, Tsit5
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, pendulum_params

from soromox.simulation import SemiImplicitEuler
from soromox.systems import (
    PCS,
    PCSStructure,
    Pendulum,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    SystemState,
)


def _pendulum() -> Pendulum:
    return Pendulum(
        pendulum_params(
            length=jnp.array([0.5]),
            center_of_mass_length=jnp.array([0.25]),
            mass=jnp.array([1.0]),
            moment_inertia=jnp.array([0.1]),
            gravity=jnp.array([0.0, -9.81]),
            joint_stiffness=jnp.array([[2.0]]),
            joint_damping=jnp.array([[0.1]]),
        )
    )


def _floating_spatial_pcs() -> PCS:
    return PCS(
        params=pcs_params(
            length=jnp.array([0.1]),
            radius=jnp.array([0.01]),
            density=jnp.array([1000.0]),
            young_modulus=jnp.array([1.0e5]),
            shear_modulus=jnp.array([4.0e4]),
            damping_matrix=jnp.eye(6),
            gravity=jnp.array([0.0, 0.0, -9.81]),
        ),
        structure=PCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


def _floating_hysteretic_planar_hsa() -> PlanarHSA:
    params_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robot_parameters"
        / "planar_hsa"
        / "fpu_control.npz"
    )
    params = PlanarHSAParams.from_npz(params_path)
    num_strains = 3 * params.length.shape[0]
    params = params.replace(
        hysteresis_basis=jnp.ones((num_strains, 1)),
        hysteresis_alpha=0.5 * jnp.ones((num_strains,)),
        hysteresis_A=jnp.ones((1,)),
        hysteresis_n=jnp.ones((1,)),
        hysteresis_beta=0.5 * jnp.ones((1,)),
        hysteresis_gamma=0.5 * jnp.ones((1,)),
    )
    return PlanarHSA(
        params=params,
        structure=PlanarHSAStructure(consider_hysteresis=True),
        floating_base=True,
    )


def test_semi_implicit_euler_is_a_deterministic_diffrax_solver():
    solver = SemiImplicitEuler(_pendulum())

    assert isinstance(solver, AbstractSolver)
    assert not isinstance(solver, Euler)


def test_semi_implicit_euler_applies_kick_before_drift():
    robot = _pendulum()
    q0 = jnp.array([0.2])
    qd0 = jnp.array([0.3])
    u = jnp.zeros((robot.num_actuators,))
    tau_ext = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])
    dt = 1e-4

    derivative = robot.forward_dynamics(jnp.asarray(0.0), y0, (u, tau_ext))
    _, qdd0 = jnp.split(derivative, 2)
    qd1 = qd0 + dt * qdd0
    expected = jnp.concatenate([q0 + dt * qd1, qd1])

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        u=u,
        tau_ext=tau_ext,
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )

    assert jnp.allclose(trajectory.y[-1], expected)


def test_semi_implicit_euler_rollout_is_jittable_and_vmappable():
    robot = _pendulum()
    solver = SemiImplicitEuler(robot)
    q0s = jnp.array([[0.1], [0.2], [-0.15]])
    qd0s = jnp.array([[0.0], [0.1], [-0.05]])

    def rollout(q0, qd0):
        trajectory = robot.rollout_to(
            initial_state=SystemState(t=0.0, y=jnp.concatenate([q0, qd0])),
            t1=0.002,
            solver_dt=1e-4,
            save_ts=jnp.array([0.0, 0.001, 0.002]),
            solver=solver,
        )
        return trajectory.y

    trajectories = jax.jit(jax.vmap(rollout))(q0s, qd0s)

    assert trajectories.shape == (3, 3, 2)
    assert jnp.all(jnp.isfinite(trajectories))


def test_semi_implicit_euler_preserves_absent_optional_rollout_states():
    robot = _pendulum()

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=jnp.array([0.1, 0.0])),
        t1=1e-4,
        solver_dt=1e-4,
        save_ts=jnp.array([0.0, 1e-4]),
        solver=SemiImplicitEuler(robot),
    )

    assert trajectory.control_state is None
    assert trajectory.environment_state is None


def test_semi_implicit_euler_supports_spatial_floating_base_state():
    robot = _floating_spatial_pcs()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.array([0.1, -0.2, 0.3, 0.9, 0.2, -0.1, 0.4]),
    )
    qd0 = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02]),
    )
    initial_y = robot.pack_state(q0, qd0)
    dt = 1e-5

    solver_y0 = robot.project_state(initial_y)
    solver_q0, solver_qd0, _ = robot.split_state(solver_y0)
    dynamics_y0 = robot.project_state(solver_y0)
    derivative = robot.forward_dynamics(jnp.asarray(0.0), dynamics_y0)
    _, qdd0, _ = robot.split_state(derivative)
    qd1 = solver_qd0 + dt * qdd0
    expected_q1 = robot.retract_configuration(solver_q0, dt * qd1)

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=initial_y),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )
    q1, actual_qd1, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_coordinates != robot.num_velocities
    assert_allclose(q1, expected_q1, atol=1e-18)
    assert_allclose(actual_qd1, qd1, atol=1e-15)
    assert auxiliary1.shape == (0,)
    assert_allclose(jnp.linalg.norm(q1[:4]), 1.0)


def test_semi_implicit_euler_advances_system_auxiliary_state():
    robot = _floating_hysteretic_planar_hsa()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.zeros((robot.num_base_coordinates,)),
    )
    qd0 = robot.pack_velocity(
        1e-3 * jnp.ones((robot.num_internal_dofs,)),
        base_velocity=jnp.zeros((robot.num_base_velocities,)),
    )
    auxiliary0 = 0.1 * jnp.ones((robot.num_auxiliary_states,))
    y0 = robot.pack_state(q0, qd0, auxiliary0)
    dt = 1e-7

    derivative = robot.forward_dynamics(jnp.asarray(0.0), y0)
    _, qdd0, auxiliary_derivative0 = robot.split_state(derivative)
    expected_qd1 = qd0 + dt * qdd0
    expected_q1 = robot.retract_configuration(q0, dt * expected_qd1)
    expected_auxiliary1 = auxiliary0 + dt * auxiliary_derivative0

    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=y0),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=SemiImplicitEuler(robot),
    )
    q1, qd1, auxiliary1 = robot.split_state(trajectory.y[-1])

    assert robot.num_auxiliary_states == 1
    assert_allclose(q1, expected_q1)
    assert_allclose(qd1, expected_qd1)
    assert_allclose(auxiliary1, expected_auxiliary1)


def test_tsit5_supports_spatial_floating_base_state():
    robot = _floating_spatial_pcs()
    q0 = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,)),
        base_pose=jnp.array([0.1, -0.2, 0.3, 0.9, 0.2, -0.1, 0.4]),
    )
    qd0 = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02]),
    )

    dt = 1e-7
    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=robot.pack_state(q0, qd0)),
        t1=dt,
        solver_dt=dt,
        save_ts=jnp.array([0.0, dt]),
        solver=Tsit5(),
    )
    q, qd, auxiliary = robot.split_state(trajectory.y)

    assert trajectory.y.shape[-1] == robot.state_size
    assert q.shape[-1] == robot.num_coordinates
    assert qd.shape[-1] == robot.num_velocities
    assert auxiliary.shape[-1] == robot.num_auxiliary_states
    assert jnp.all(jnp.isfinite(trajectory.y))
    assert_allclose(jnp.linalg.norm(q[..., :4], axis=-1), 1.0)
