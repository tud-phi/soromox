import equinox as eqx
import jax
from jax import numpy as jnp
from system_param_builders import pendulum_params

from soromox.systems import Pendulum, SystemState


def _pendulum_params():
    # Gravity set to zero so the zero configuration is an equilibrium for zero actuation.
    return pendulum_params(
        length=jnp.array([0.5, 0.3]),
        center_of_mass_length=jnp.array([0.25, 0.15]),
        mass=jnp.array([1.0, 0.5]),
        moment_inertia=jnp.array([0.1, 0.05]),
        gravity=jnp.array([0.0, 0.0]),
    )


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


class ZeroController(eqx.Module):
    num_actuators: int

    def __call__(self, state: SystemState):
        return jnp.zeros((self.num_actuators,)), None


class ConstantTorqueEnvironment(eqx.Module):
    tau: jnp.ndarray

    def __call__(self, state: SystemState):
        return self.tau, None


class ClockEnvironment(eqx.Module):
    def __call__(self, state: SystemState):
        q_len = state.y.shape[0] // 2
        tau_environment = jnp.zeros((q_len,))
        environment_state_dot = {
            "clock": jnp.ones_like(state.environment_state["clock"])
        }
        return tau_environment, environment_state_dot


class _ConstantStateDotEnvironment(eqx.Module):
    """Returns a fixed scalar environment_state_dot regardless of input state.

    Used to test that rollouts raise when environment_state_dot is returned but
    no initial environment_state was provided.
    """

    def __call__(self, state: SystemState):
        q_len = state.y.shape[0] // 2
        return jnp.zeros((q_len,)), jnp.array(0.0)


def test_rollout_to_keeps_equilibrium():
    robot = Pendulum(_pendulum_params())

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


def test_rollout_to_environment_model_matches_tau_ext():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    tau_ext = jnp.array([0.2, -0.1])
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))

    trajectory_with_tau = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),
        tau_ext=tau_ext,
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.02,
    )
    trajectory_with_environment = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),
        environment_model=ConstantTorqueEnvironment(tau=tau_ext),
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.02,
    )

    assert trajectory_with_environment.environment_state is None
    assert jnp.allclose(trajectory_with_environment.y, trajectory_with_tau.y, atol=1e-6)


def test_rollout_to_tracks_environment_state():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_clock = jnp.array(2.0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        environment_state={"clock": initial_clock},
    )

    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=jnp.zeros_like(q0),
        environment_model=ClockEnvironment(),
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.02,
    )

    assert trajectory.environment_state is not None
    assert jnp.allclose(
        trajectory.environment_state["clock"],
        initial_clock + trajectory.t,
        atol=1e-5,
    )


def test_rollout_closed_loop_to_tracks_target():
    robot = Pendulum(_pendulum_params())

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


def test_rollout_closed_loop_to_environment_model_matches_tau_ext():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    tau_ext = jnp.array([0.15, -0.05])
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
    )
    controller = ZeroController(num_actuators=q0.shape[0])

    trajectory_with_tau = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        tau_ext=tau_ext,
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.02,
    )
    trajectory_with_environment = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        environment_model=ConstantTorqueEnvironment(tau=tau_ext),
        t1=0.1,
        solver_dt=1e-3,
        save_dt=0.02,
    )

    assert trajectory_with_environment.environment_state is None
    assert jnp.allclose(trajectory_with_environment.y, trajectory_with_tau.y, atol=1e-6)


def test_rollout_discrete_closed_loop_to_tracks_target():
    robot = Pendulum(_pendulum_params())

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
        duration=0.6,
        solver_dt=1e-3,
        control_dt=0.02,
        save_dt=0.02,
    )

    q_final = trajectory.y[-1, : q0.shape[0]]
    assert jnp.linalg.norm(q_final) < 7e-2
    assert trajectory.control_state is not None


def test_rollout_discrete_closed_loop_to_tracks_environment_state():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_clock = jnp.array(-1.0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        environment_state={"clock": initial_clock},
    )
    controller = ZeroController(num_actuators=q0.shape[0])

    trajectory = robot.rollout_discrete_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        environment_model=ClockEnvironment(),
        duration=0.1,
        solver_dt=1e-3,
        control_dt=0.02,
        save_dt=0.01,
    )

    assert trajectory.environment_state is not None
    assert jnp.allclose(
        trajectory.environment_state["clock"],
        initial_clock + trajectory.t,
        atol=1e-5,
    )


def test_rollout_discrete_closed_loop_to_is_jittable():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),
    )
    controller = PIDController(kp=8.0, ki=2.5, kd=1.5, target=jnp.zeros_like(q0))

    def loss_fn(y0: jnp.ndarray) -> jnp.ndarray:
        init = eqx.tree_at(lambda s: s.y, initial_state, y0)
        traj = robot.rollout_discrete_closed_loop_to(
            initial_state=init,
            controller=controller,
            duration=0.2,
            solver_dt=1e-3,
            control_dt=0.02,
            save_dt=0.01,
        )
        q_final = traj.y[-1, : q0.shape[0]]
        return jnp.sum(q_final**2)

    y0 = initial_state.y
    loss = jax.jit(loss_fn)(y0)
    assert jnp.isfinite(loss)


def test_rollout_discrete_closed_loop_to_is_vmappable():
    robot = Pendulum(_pendulum_params())
    target = jnp.zeros((2,))
    controller = PIDController(kp=8.0, ki=2.5, kd=1.5, target=target)

    def run(q0: jnp.ndarray) -> jnp.ndarray:
        qd0 = jnp.zeros_like(q0)
        init = SystemState(
            t=0.0,
            y=jnp.concatenate([q0, qd0]),
            u=jnp.zeros_like(q0),
            control_state=jnp.zeros_like(q0),
        )
        traj = robot.rollout_discrete_closed_loop_to(
            initial_state=init,
            controller=controller,
            duration=0.2,
            solver_dt=1e-3,
            control_dt=0.02,
            save_dt=0.01,
        )
        return traj.y[-1]

    q0s = jnp.array([[0.3, -0.2], [0.1, 0.05], [-0.25, 0.15]])
    ys_final = jax.jit(jax.vmap(run))(q0s)
    assert ys_final.shape == (q0s.shape[0], 2 * q0s.shape[1])
    assert jnp.all(jnp.isfinite(ys_final))


def test_rollout_discrete_closed_loop_to_reverse_mode_grad_works():
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.3, -0.2])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),
    )

    def loss_fn(kp: jnp.ndarray) -> jnp.ndarray:
        controller = PIDController(kp=kp, ki=2.5, kd=1.5, target=jnp.zeros_like(q0))
        traj = robot.rollout_discrete_closed_loop_to(
            initial_state=initial_state,
            controller=controller,
            duration=0.2,
            solver_dt=1e-3,
            control_dt=0.02,
            save_dt=0.01,
            max_steps=1024,
        )
        q_final = traj.y[-1, : q0.shape[0]]
        return jnp.sum(q_final**2)

    dloss_dkp = jax.grad(loss_fn)(jnp.array(8.0))
    assert jnp.isfinite(dloss_dkp)


def test_rollout_to_environment_model_raises_without_initial_environment_state():
    """Regression test: error if environment model returns state derivative but no initial state."""
    import pytest

    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))

    with pytest.raises(ValueError, match="environment_state"):
        robot.rollout_to(
            initial_state=initial_state,
            u=jnp.zeros_like(q0),
            environment_model=_ConstantStateDotEnvironment(),
            t1=0.05,
            solver_dt=1e-3,
            save_dt=0.05,
        )


def test_rollout_closed_loop_to_environment_model_raises_without_initial_environment_state():
    """Regression test: error if environment model returns state derivative but no initial state."""
    import pytest

    robot = Pendulum(_pendulum_params())

    q0 = jnp.zeros((2,))
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
    )
    controller = ZeroController(num_actuators=q0.shape[0])

    with pytest.raises(ValueError, match="environment_state"):
        robot.rollout_closed_loop_to(
            initial_state=initial_state,
            controller=controller,
            environment_model=_ConstantStateDotEnvironment(),
            t1=0.05,
            solver_dt=1e-3,
            save_dt=0.05,
        )


def test_rollout_discrete_control_and_save_steps():
    """Test that control steps and save steps are correct in discrete rollout."""
    robot = Pendulum(_pendulum_params())

    q0 = jnp.array([0.1, -0.05])
    qd0 = jnp.zeros_like(q0)
    initial_state = SystemState(
        t=0.0,
        y=jnp.concatenate([q0, qd0]),
        u=jnp.zeros_like(q0),
        control_state=jnp.zeros_like(q0),
    )

    class CountingController(eqx.Module):
        kp: float
        target: jnp.ndarray

        def __call__(self, state: SystemState):
            q_len = self.target.shape[0]
            q, qd = jnp.split(state.y, [q_len], axis=0)

            error = self.target - q
            u = self.kp * error
            control_state_dot = error
            return u, control_state_dot

    controller = CountingController(kp=5.0, target=jnp.zeros_like(q0))

    # Test with explicit parameters
    duration = 0.3  # 300 ms
    control_dt = 0.05  # 50 ms control interval
    save_dt = 0.01  # 10 ms save interval

    # Expected values:
    # - control_steps = duration / control_dt = 0.3 / 0.05 = 6
    # - saves_per_control = control_dt / save_dt = 0.05 / 0.01 = 5
    # - total_saves = saves_per_control * control_steps + 1 = 5 * 6 + 1 = 31
    # Note: Implementation creates (saves_per_control + 1) samples per interval
    # and drops first sample from all but the first interval
    expected_control_steps = int(round(duration / control_dt))
    saves_per_control = int(round(control_dt / save_dt))
    expected_total_saves = saves_per_control * expected_control_steps + 1

    trajectory = robot.rollout_discrete_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        duration=duration,
        solver_dt=1e-3,
        control_dt=control_dt,
        save_dt=save_dt,
    )

    # Verify save steps
    assert trajectory.t.shape[0] == expected_total_saves, (
        f"Expected {expected_total_saves} save steps, got {trajectory.t.shape[0]}"
    )
    assert trajectory.y.shape[0] == expected_total_saves
    assert trajectory.u.shape[0] == expected_total_saves

    # Verify time steps are correct
    # Use half of save_dt as tolerance to ensure we capture all expected time points
    # while accounting for floating-point precision
    expected_times = jnp.arange(0, duration + save_dt / 2, save_dt)
    assert jnp.allclose(trajectory.t, expected_times, atol=1e-10), (
        f"Time steps don't match expected values.\n"
        f"Expected: {expected_times}\n"
        f"Got: {trajectory.t}"
    )

    # Verify that control changes happen at the right intervals
    # The control input should be constant within each control interval
    # First interval: indices 0 to saves_per_control (inclusive)
    # Subsequent intervals: each has saves_per_control entries
    control_boundaries = [0]  # Start of each control interval
    control_boundaries.append(saves_per_control + 1)  # End of first interval
    for _ in range(1, expected_control_steps):
        control_boundaries.append(control_boundaries[-1] + saves_per_control)

    for i in range(expected_control_steps):
        start_idx = control_boundaries[i]
        end_idx = control_boundaries[i + 1]
        # Within each control interval, all u values should be the same
        control_segment = trajectory.u[start_idx:end_idx]
        for j in range(1, control_segment.shape[0]):
            assert jnp.allclose(control_segment[j], control_segment[0], atol=1e-6), (
                f"Control should be constant within interval {i} "
                f"(indices {start_idx}:{end_idx}), "
                f"but differs at index {start_idx + j}: "
                f"{control_segment[0]} vs {control_segment[j]}"
            )
