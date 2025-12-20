import equinox as eqx
from jax import numpy as jnp

from soromox.systems import pendulum
from soromox.systems.system_state import SystemState


def _pendulum_params():
    # Gravity set to zero so the zero configuration is an equilibrium for zero actuation.
    return {
        "L": jnp.array([0.5, 0.3]),
        "Lc": jnp.array([0.25, 0.15]),
        "m": jnp.array([1.0, 0.5]),
        "I": jnp.array([0.1, 0.05]),
        "g": jnp.array([0.0, 0.0]),
    }


class PIDController(eqx.Module):
    kp: float
    ki: float
    kd: float
    target: jnp.ndarray

    def __call__(self, state: SystemState):
        q_len = self.target.shape[0]
        q, qd = jnp.split(state.y, [q_len], axis=0)
        integ = state.control_state

        error = self.target - q
        d_error = -qd  # derivative of (target - q) when target is constant

        u = self.kp * error + self.ki * integ + self.kd * d_error
        control_state_dot = error
        return u, control_state_dot


def test_rollout_to_keeps_equilibrium():
    robot = pendulum.Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))

    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.01,
    )

    assert trajectory.control_state is None
    assert jnp.allclose(trajectory.y, 0.0, atol=1e-6)


def test_rollout_closed_loop_to_tracks_target():
    robot = pendulum.Pendulum(_pendulum_params())

    q0 = jnp.array([0.2, -0.1])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),  # integral term
    )

    controller = PIDController(kp=10.0, ki=3.0, kd=2.0, target=jnp.zeros_like(q0))

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=0.4,
        solver_dt=1e-3,
        save_dt=0.02,
    )

    q_final = trajectory.y[-1, : q0.shape[0]]
    assert jnp.linalg.norm(q_final) < 5e-2
    assert trajectory.control_state is not None


def test_rollout_discrete_closed_loop_to_tracks_target():
    robot = pendulum.Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),
    )

    controller = PIDController(kp=8.0, ki=2.5, kd=1.5, target=jnp.zeros_like(q0))

    trajectory = robot.rollout_discrete_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=0.6,
        solver_dt=1e-3,
        control_dt=0.02,
        save_dt=0.02,
    )

    q_final = trajectory.y[-1, : q0.shape[0]]
    assert jnp.linalg.norm(q_final) < 7e-2
    assert trajectory.control_state is not None
