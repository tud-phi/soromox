"""Parallel SoRoMoX tracking environment for Stable-Baselines3.

This module exposes a JAX-vectorized VecEnv for tendon-actuated SoRoMoX arm
tracking tasks. Each vectorized environment receives an independent target-ball
trajectory at reset time, while all environments are stepped in one batched JAX
call for efficient reinforcement-learning training and evaluation.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
from diffrax import Tsit5
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import PCS, LinkSpec
from soromox.systems.system_state import SystemState

jax.config.update("jax_enable_x64", True)


class EnvState(NamedTuple):
    """Container for one SoRoMoX tracking environment state.

    Attributes:
        arm_state: Continuous SoRoMoX system state.
        current_force: Current nonnegative tendon force vector.
        action_step_counter: Number of RL action steps already applied.
        ball_pos: Current target-ball position in meters.
        ball_vel: Current target-ball velocity in meters per second.
        ball_acc: Current target-ball acceleration in meters per second squared.
        current_step: Current low-level simulator step.
        ball_traj_pos: Full action-rate target-ball position trajectory.
        ball_traj_vel: Full action-rate target-ball velocity trajectory.
        ball_traj_acc: Full action-rate target-ball acceleration trajectory.
    """

    arm_state: SystemState
    current_force: jnp.ndarray
    action_step_counter: jnp.int32
    ball_pos: jnp.ndarray
    ball_vel: jnp.ndarray
    ball_acc: jnp.ndarray
    current_step: jnp.int32
    ball_traj_pos: jnp.ndarray
    ball_traj_vel: jnp.ndarray
    ball_traj_acc: jnp.ndarray


def make_default_y0(num_segments: int) -> jnp.ndarray:
    """Create the zero initial generalized state for a SoRoMoX arm.

    Args:
        num_segments: Number of constant-strain arm segments.

    Returns:
        Concatenated generalized coordinates and velocities with shape
        ``(12 * num_segments,)``.
    """

    q0 = jnp.repeat(
        jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        num_segments,
        axis=0,
    ).flatten()
    qd0 = jnp.zeros_like(q0)
    return jnp.concatenate([q0, qd0])


def build_arm(
    *,
    num_segments: int,
    length: float,
    radius: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> PCS:
    """Build the tendon-actuated SoRoMoX arm model used by the environment.

    Args:
        num_segments: Number of constant-strain segments.
        length: Length of each segment in meters.
        radius: Arm radius in meters.
        density: Material density in kg/m^3.
        youngs_modulus: Young's modulus in Pa.
        poisson_ratio: Poisson ratio.

    Returns:
        A configured ``PCS`` model with threadlike tendon actuation.
    """

    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    segment_lengths = jnp.ones((num_segments,)) * length
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]),
                num_segments,
                axis=0,
            )
            * segment_lengths[:, None]
        ).flatten()
    )

    body_params = PCS.params_from_links(
        [
            LinkSpec.circular(
                length=float(segment_lengths[index]),
                radius=radius,
                density=density,
                young_modulus=youngs_modulus,
                shear_modulus=shear_modulus,
                damping=damping_matrix[
                    6 * index : 6 * (index + 1),
                    6 * index : 6 * (index + 1),
                ],
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for index in range(num_segments)
        ],
        gravity=jnp.array([0.0, 0.0, 0.0]),
    )
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.array(
            [[0.0, 0.0, 0.02], [0.0, 0.0, -0.02], [0.0, 0.02, 0.0], [0.0, -0.02, 0.0]]
        ),
        start_segment_index=0,
        end_segment_index=(0, 0, 0, 0),
    )
    return PCS(
        params=body_params,
        base_pose=jnp.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )


def build_jax_env_fns(
    *,
    arm: PCS,
    y0: jnp.ndarray,
    dt: float,
    steps_per_action: int,
    tendon_force_scale: float,
    tendon_dim: int,
    action_dim: int,
    game_time: float,
    success_threshold: float,
    success_reward: float,
    ball_center: tuple[float, float, float],
    ball_radius: float,
    ball_surface_vmax: float,
) -> tuple[Any, Any, Any, Any]:
    """Build JIT-compiled reset and step functions for batched simulation.

    Args:
        arm: SoRoMoX arm model.
        y0: Initial generalized arm state.
        dt: Low-level simulator time step in seconds.
        steps_per_action: Number of low-level steps per RL action.
        tendon_force_scale: Action-to-force increment scale.
        tendon_dim: Number of tendons.
        action_dim: Number of RL action dimensions.
        game_time: Episode duration in seconds.
        success_threshold: Distance threshold for tracking success in meters.
        success_reward: Reward bonus when within the success threshold.
        ball_center: Center of the target hemisphere in meters.
        ball_radius: Target hemisphere radius in meters.
        ball_surface_vmax: Maximum ball surface speed in meters per second.

    Returns:
        ``reset_single``, ``reset_batch``, ``step_single``, and ``step_batch``.
    """

    total_time_steps = int(game_time / dt)
    dt_action = steps_per_action * dt
    num_action_steps = (total_time_steps + steps_per_action - 1) // steps_per_action
    horizon = num_action_steps + 1
    dense_points = max(2000, horizon * 30)

    center = jnp.asarray(ball_center, dtype=jnp.float32)
    radius = jnp.asarray(ball_radius, dtype=jnp.float32)
    vmax = jnp.asarray(ball_surface_vmax, dtype=jnp.float32)
    success_threshold_jnp = jnp.asarray(success_threshold, dtype=jnp.float32)
    success_reward_jnp = jnp.asarray(success_reward, dtype=jnp.float32)

    fk_end_effector = jax.jit(partial(arm.forward_kinematics, s=jnp.sum(arm.L)))
    jacobian_ee = jax.jit(partial(arm.jacobian_inertialframe, s=jnp.sum(arm.L)))
    solver = Tsit5()

    def reward_fn(
        ee_pos: jnp.ndarray,
        ball_pos: jnp.ndarray,
        ee_vel: jnp.ndarray,
        ball_vel: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute shaped tracking reward and success flag for one env.

        Args:
            ee_pos: End-effector position with shape ``(3,)``.
            ball_pos: Target-ball position with shape ``(3,)``.
            ee_vel: End-effector linear velocity with shape ``(3,)``.
            ball_vel: Target-ball velocity with shape ``(3,)``.

        Returns:
            Scalar reward and scalar boolean success flag.
        """

        error = ee_pos - ball_pos
        distance = jnp.linalg.norm(error)
        direction = error / (distance + 1e-8)
        relative_velocity = ee_vel - ball_vel
        distance_rate = jnp.dot(relative_velocity, direction)

        position_term = distance / 0.2
        velocity_term = jnp.maximum(distance_rate, 0.0) / 0.3
        reward = -(position_term + velocity_term)
        success = distance < success_threshold_jnp
        reward = reward + jnp.where(success, success_reward_jnp, 0.0)
        return reward, success

    def terminate_fn(env_state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate episode termination and truncation for one env.

        Args:
            env_state: Current environment state.

        Returns:
            ``terminated`` and ``truncated`` scalar booleans.
        """

        truncated = env_state.current_step >= total_time_steps
        terminated = jnp.array(False)
        return terminated, truncated

    def get_ee_state(env_state: EnvState) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute end-effector position and linear velocity for one env.

        Args:
            env_state: Current environment state.

        Returns:
            End-effector position and linear velocity, each with shape ``(3,)``.
        """

        q, qd = jnp.split(env_state.arm_state.y, 2, axis=0)
        transform_ee = fk_end_effector(q)
        ee_pos = transform_ee[:3, 3]
        jacobian = jacobian_ee(q)
        ee_velocity_6d = jnp.einsum("ij,j->i", jacobian, qd)
        return ee_pos, ee_velocity_6d[3:]

    def get_obs_fn(
        ee_pos: jnp.ndarray,
        ee_vel: jnp.ndarray,
        current_force: jnp.ndarray,
        ball_pos: jnp.ndarray,
        ball_vel: jnp.ndarray,
    ) -> jnp.ndarray:
        """Construct the observation vector for one env.

        Args:
            ee_pos: End-effector position with shape ``(3,)``.
            ee_vel: End-effector velocity with shape ``(3,)``.
            current_force: Current tendon forces with shape ``(4,)``.
            ball_pos: Target-ball position with shape ``(3,)``.
            ball_vel: Target-ball velocity with shape ``(3,)``.

        Returns:
            Observation vector with shape ``(16,)``.
        """

        return jnp.concatenate([ee_pos, ee_vel, current_force, ball_pos, ball_vel])

    def make_ball_traj(
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample one random target-ball trajectory on a hemisphere.

        Args:
            key: JAX PRNG key for the trajectory.

        Returns:
            Position, velocity, and acceleration arrays with shape
            ``(horizon, 3)``.
        """

        k1, k2, k3, k4, k5, k6 = rnd.split(key, 6)
        n_turns = rnd.uniform(k1, (), minval=0.5, maxval=3.0)
        theta0 = rnd.uniform(k2, (), minval=0.15, maxval=0.5 * jnp.pi - 0.25)
        theta_amp = rnd.uniform(k3, (), minval=0.0, maxval=0.35)
        theta_freq = rnd.uniform(k4, (), minval=0.5, maxval=3.0)
        phi_phase = rnd.uniform(k5, (), minval=0.0, maxval=2 * jnp.pi)
        theta_phase = rnd.uniform(k6, (), minval=0.0, maxval=2 * jnp.pi)

        s = jnp.linspace(0.0, 1.0, dense_points, dtype=jnp.float32)
        phi = 2.0 * jnp.pi * n_turns * s + phi_phase
        theta = theta0 + theta_amp * jnp.sin(
            2.0 * jnp.pi * theta_freq * s + theta_phase
        )
        theta = jnp.clip(theta, 1e-3, 0.5 * jnp.pi - 1e-3)

        pos_dense = jnp.stack(
            [
                radius * jnp.sin(theta) * jnp.cos(phi),
                radius * jnp.sin(theta) * jnp.sin(phi),
                center[2] + radius * jnp.cos(theta),
            ],
            axis=1,
        )

        arc_increments = jnp.linalg.norm(pos_dense[1:] - pos_dense[:-1], axis=1)
        arc = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float32), jnp.cumsum(arc_increments)],
            axis=0,
        )
        target_arc = jnp.minimum(
            jnp.arange(horizon, dtype=jnp.float32) * vmax * dt_action,
            arc[-1],
        )

        pos = jnp.stack(
            [
                jnp.interp(target_arc, arc, pos_dense[:, 0]),
                jnp.interp(target_arc, arc, pos_dense[:, 1]),
                jnp.interp(target_arc, arc, pos_dense[:, 2]),
            ],
            axis=1,
        )

        radial = pos - center[None, :]
        pos = center[None, :] + radius * radial / (
            jnp.linalg.norm(radial, axis=1, keepdims=True) + 1e-8
        )

        vel = (pos[1:] - pos[:-1]) / dt_action
        vel = jnp.concatenate([vel, vel[-1:]], axis=0)
        normal = (pos - center[None, :]) / (radius + 1e-8)
        vel = vel - jnp.sum(vel * normal, axis=1, keepdims=True) * normal
        speed = jnp.linalg.norm(vel, axis=1, keepdims=True) + 1e-8
        vel = vel * jnp.minimum(1.0, vmax / speed)

        acc = (vel[1:] - vel[:-1]) / dt_action
        acc = jnp.concatenate([acc, acc[-1:]], axis=0)
        acc = acc - jnp.sum(acc * normal, axis=1, keepdims=True) * normal
        return pos.astype(jnp.float32), vel.astype(jnp.float32), acc.astype(jnp.float32)

    def reset_single(key: jnp.ndarray) -> tuple[EnvState, jnp.ndarray]:
        """Reset one environment from a PRNG key.

        Args:
            key: JAX PRNG key for one environment.

        Returns:
            Initial environment state and observation.
        """

        key_traj, _ = rnd.split(key, 2)
        ball_traj_pos, ball_traj_vel, ball_traj_acc = make_ball_traj(key_traj)
        env_state = EnvState(
            arm_state=SystemState(t=0.0, y=y0),
            current_force=jnp.zeros((tendon_dim,), dtype=jnp.float32),
            action_step_counter=jnp.int32(0),
            ball_pos=ball_traj_pos[0],
            ball_vel=ball_traj_vel[0],
            ball_acc=ball_traj_acc[0],
            current_step=jnp.int32(0),
            ball_traj_pos=ball_traj_pos,
            ball_traj_vel=ball_traj_vel,
            ball_traj_acc=ball_traj_acc,
        )
        ee_pos, ee_vel = get_ee_state(env_state)
        obs = get_obs_fn(
            ee_pos,
            ee_vel,
            jnp.zeros((tendon_dim,), dtype=jnp.float32),
            env_state.ball_pos,
            env_state.ball_vel,
        )
        return env_state, obs

    def reset_batch(keys: jnp.ndarray) -> tuple[EnvState, jnp.ndarray]:
        """Reset a batch of environments.

        Args:
            keys: JAX PRNG keys with shape ``(num_envs, 2)``.

        Returns:
            Batched environment states and observations.
        """

        return jax.vmap(reset_single)(keys)

    @eqx.filter_jit
    def step_single(
        env_state: EnvState,
        action: jnp.ndarray,
    ) -> tuple[
        EnvState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Step one environment by one RL action.

        Args:
            env_state: Current environment state.
            action: RL action with shape ``(4,)`` in ``[-1, 1]``.

        Returns:
            New state, observation, reward, terminated flag, truncated flag,
            end-effector position, and invalid-observation flag.
        """

        delta_force = action[:action_dim] * tendon_force_scale
        tendon_force = jnp.clip(env_state.current_force + delta_force, 0.0, 25.0)

        action_step_counter = env_state.action_step_counter + jnp.int32(1)
        t1 = action_step_counter * steps_per_action * dt
        trajectory = arm.rollout_to(
            initial_state=SystemState(t=0.0, y=env_state.arm_state.y),
            u=tendon_force,
            t1=steps_per_action * dt,
            solver_dt=dt,
            save_dt=dt,
            solver=solver,
            max_steps=None,
        )
        new_arm_state = SystemState(t=t1, y=trajectory.y[:-1][-1])

        idx = jnp.minimum(action_step_counter, jnp.int32(num_action_steps))
        new_env_state = EnvState(
            arm_state=new_arm_state,
            current_force=tendon_force,
            action_step_counter=action_step_counter,
            ball_pos=env_state.ball_traj_pos[idx],
            ball_vel=env_state.ball_traj_vel[idx],
            ball_acc=env_state.ball_traj_acc[idx],
            current_step=env_state.current_step + jnp.int32(steps_per_action),
            ball_traj_pos=env_state.ball_traj_pos,
            ball_traj_vel=env_state.ball_traj_vel,
            ball_traj_acc=env_state.ball_traj_acc,
        )

        ee_pos, ee_vel = get_ee_state(new_env_state)
        obs = get_obs_fn(
            ee_pos,
            ee_vel,
            tendon_force,
            new_env_state.ball_pos,
            new_env_state.ball_vel,
        )
        reward, _ = reward_fn(
            ee_pos, new_env_state.ball_pos, ee_vel, new_env_state.ball_vel
        )
        terminated, truncated = terminate_fn(new_env_state)

        invalid_obs = jnp.isnan(obs).any() | jnp.isinf(obs).any()
        obs = jnp.where(invalid_obs, jnp.zeros_like(obs), obs)
        reward = jnp.where(invalid_obs, jnp.array(-100.0), reward)
        truncated = jnp.logical_or(truncated, invalid_obs)
        return new_env_state, obs, reward, terminated, truncated, ee_pos, invalid_obs

    step_batch = eqx.filter_jit(jax.vmap(step_single, in_axes=(0, 0)))
    return reset_single, reset_batch, step_single, step_batch


class ParallelSoromoxEnv(VecEnv):
    """Stable-Baselines3 VecEnv for parallel SoRoMoX tracking.

    The environment uses one JAX batched rollout for all vectorized environments.
    Each reset samples independent target-ball trajectories by splitting the
    global PRNG key.
    """

    def __init__(self, num_envs: int = 64, **kwargs: Any) -> None:
        """Initialize the vectorized SoRoMoX environment.

        Args:
            num_envs: Number of parallel environments.
            **kwargs: Optional physical, timing, reward, and target parameters.
        """

        self.obs_dim = 16
        self.action_dim = 4
        self.tendon_dim = 4
        self.num_envs = int(num_envs)

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        super().__init__(self.num_envs, observation_space, action_space)

        self.num_segments = int(kwargs.get("num_segments", 1))
        self.length = float(kwargs.get("length", 0.25))
        self.radius = float(kwargs.get("radius", 0.025))
        self.density = float(kwargs.get("density", 1070.0))
        self.youngs_modulus = float(kwargs.get("youngs_modulus", 2.5e5))
        self.poisson_ratio = float(kwargs.get("poisson_ratio", 0.37))
        self.dt = float(kwargs.get("dt", 1e-4))
        self.game_time = float(kwargs.get("game_time", 10.0))
        self.control_fps = float(kwargs.get("control_fps", 15.0))
        self.steps_per_action = int(np.ceil(1.0 / self.control_fps / self.dt))
        self.tendon_force_scale = float(kwargs.get("tendon_force_scale", 0.5))
        self.success_threshold = float(kwargs.get("success_threshold", 0.01))
        self.success_reward = float(kwargs.get("success_reward", 1.5))
        self.ball_center = tuple(kwargs.get("ball_center", (0.0, 0.0, 0.15)))
        self.ball_radius = float(kwargs.get("ball_radius", 0.10))
        self.ball_surface_vmax = float(kwargs.get("ball_surface_vmax", 0.015))

        self.arm = build_arm(
            num_segments=self.num_segments,
            length=self.length,
            radius=self.radius,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
        )
        self.y0 = make_default_y0(self.num_segments)
        (
            self._reset_single,
            self._reset_batch,
            self._step_single,
            self._step_batch,
        ) = build_jax_env_fns(
            arm=self.arm,
            y0=self.y0,
            dt=self.dt,
            steps_per_action=self.steps_per_action,
            tendon_force_scale=self.tendon_force_scale,
            tendon_dim=self.tendon_dim,
            action_dim=self.action_dim,
            game_time=self.game_time,
            success_threshold=self.success_threshold,
            success_reward=self.success_reward,
            ball_center=self.ball_center,
            ball_radius=self.ball_radius,
            ball_surface_vmax=self.ball_surface_vmax,
        )

        self._env_states: EnvState | None = None
        self._last_actions: np.ndarray | None = None
        self._rng_key = rnd.PRNGKey(int(kwargs.get("seed", 0)))

    def reset(self) -> np.ndarray:
        """Reset all vectorized environments.

        Returns:
            Batch of observations with shape ``(num_envs, 16)``.
        """

        keys = rnd.split(self._rng_key, self.num_envs + 1)
        keys_env, self._rng_key = keys[:-1], keys[-1]
        self._env_states, obs = self._reset_batch(keys_env)
        return np.asarray(jax.device_get(obs), dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for the next synchronous vectorized step.

        Args:
            actions: Batched action array with shape ``(num_envs, 4)``.
        """

        self._last_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[dict]]:
        """Apply the stored actions and return SB3 VecEnv step outputs.

        Returns:
            Observations, rewards, done flags, and per-env info dictionaries.
        """

        if self._env_states is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self._last_actions is None:
            raise RuntimeError("step_async must be called before step_wait.")

        (
            env_states,
            obs,
            rewards,
            terminateds,
            truncateds,
            ee_pos,
            invalid_obs,
        ) = self._step_batch(self._env_states, jnp.asarray(self._last_actions))
        self._env_states = env_states

        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        rewards_np = np.asarray(jax.device_get(rewards), dtype=np.float32)
        terminateds_np = np.asarray(jax.device_get(terminateds), dtype=bool)
        truncateds_np = np.asarray(jax.device_get(truncateds), dtype=bool)
        ee_pos_np = np.asarray(jax.device_get(ee_pos), dtype=np.float32)
        invalid_np = np.asarray(jax.device_get(invalid_obs), dtype=bool)
        dones_np = np.logical_or(terminateds_np, truncateds_np)

        infos = [{} for _ in range(self.num_envs)]
        for env_idx in range(self.num_envs):
            infos[env_idx]["ee_pos"] = ee_pos_np[env_idx].copy()
            infos[env_idx]["invalid_obs"] = bool(invalid_np[env_idx])
            if dones_np[env_idx]:
                infos[env_idx]["terminal_observation"] = obs_np[env_idx].copy()
                if truncateds_np[env_idx]:
                    infos[env_idx]["TimeLimit.truncated"] = True
                self._auto_reset_env(env_idx, obs_np)

        return obs_np, rewards_np, dones_np, infos

    def close(self) -> None:
        """Release environment resources.

        The current implementation does not hold external resources, so this is
        a no-op required by the VecEnv interface.
        """

    def get_attr(self, name: str, indices: Sequence[int] | None = None) -> list:
        """Return a repeated environment attribute for SB3 compatibility.

        Args:
            name: Attribute name to read.
            indices: Optional vectorized environment indices.

        Returns:
            List of attribute values, one for each requested index.
        """

        indices = range(self.num_envs) if indices is None else indices
        return [getattr(self, name) for _ in indices]

    def set_attr(
        self,
        name: str,
        values: Any,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Set an environment attribute for SB3 compatibility.

        Args:
            name: Attribute name to set.
            values: Scalar value or list of values.
            indices: Optional vectorized environment indices.
        """

        indices = list(range(self.num_envs) if indices is None else indices)
        values = (
            values if isinstance(values, (list, tuple)) else [values] * len(indices)
        )
        for value in values:
            setattr(self, name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | None = None,
        **method_kwargs: Any,
    ) -> list:
        """Call an environment method for SB3 compatibility.

        Args:
            method_name: Method name to call.
            *method_args: Positional method arguments.
            indices: Optional vectorized environment indices.
            **method_kwargs: Keyword method arguments.

        Returns:
            List of method return values.
        """

        indices = range(self.num_envs) if indices is None else indices
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(
        self,
        wrapper_class: Any,
        indices: Sequence[int] | None = None,
    ) -> list:
        """Report wrapper status for SB3 compatibility.

        Args:
            wrapper_class: Wrapper class queried by SB3.
            indices: Optional vectorized environment indices.

        Returns:
            ``False`` for each requested index because this env is not wrapped
            internally.
        """

        indices = range(self.num_envs) if indices is None else indices
        return [False for _ in indices]

    def _auto_reset_env(self, env_idx: int, obs_np: np.ndarray) -> None:
        """Reset one finished vectorized environment in-place.

        Args:
            env_idx: Vectorized environment index to reset.
            obs_np: Host observation batch to update with the reset observation.
        """

        subkey, self._rng_key = rnd.split(self._rng_key)
        env_state_single, obs_single = self._reset_single(subkey)

        def update_leaf(old_leaf: jnp.ndarray, new_leaf: jnp.ndarray) -> jnp.ndarray:
            """Replace one leaf entry in a batched PyTree.

            Args:
                old_leaf: Current batched leaf.
                new_leaf: Replacement single-env leaf.

            Returns:
                Updated batched leaf as a JAX array.
            """

            old_host = np.array(old_leaf)
            old_host[env_idx] = np.array(new_leaf)
            return jnp.asarray(old_host)

        self._env_states = jax.tree_util.tree_map(
            update_leaf,
            self._env_states,
            env_state_single,
        )
        obs_np[env_idx] = np.asarray(jax.device_get(obs_single), dtype=np.float32)
