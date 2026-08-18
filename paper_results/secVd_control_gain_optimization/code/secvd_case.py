"""Shared robot, setpoint, and pose construction for Section Vd."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, vmap

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import PCS, PCSParams, SystemState
from soromox.utils.geometry.rotations import rotation_matrix_to_rotation_vector

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SecVdCase:
    robot: PCS
    routing: ThreadlikeRouting
    total_length: float
    q0: Array
    qd0: Array
    q_des: Array
    x_des: Array
    solver_dt: float
    save_ts: Array


def end_effector_pose(robot: PCS, q: Array, total_length: float) -> Array:
    """Return ``[rotation-vector xyz, Cartesian position xyz]`` at the tip."""
    transform = robot.forward_kinematics(q, total_length)
    return jnp.concatenate(
        [rotation_matrix_to_rotation_vector(transform[:3, :3]), transform[:3, 3]]
    )


def end_effector_pose_trajectory(robot: PCS, q_ts: Array, total_length: float) -> Array:
    return vmap(lambda q: end_effector_pose(robot, q, total_length))(q_ts)


def build_sec_vd_robot() -> tuple[PCS, ThreadlikeRouting, float]:
    """Construct the robot without running the steady-state setpoint rollout."""
    num_segments = 1
    radius = 2e-2
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1.0, 1.0, 1.0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=radius * jnp.ones((num_segments,)),
        density=1070.0 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )
    angles = jnp.deg2rad(jnp.array([0.0, 120.0, 240.0]))
    routing = ThreadlikeRouting.linear(
        intercept=0.8
        * radius
        * jnp.stack(
            [jnp.zeros_like(angles), jnp.sin(angles), jnp.cos(angles)], axis=-1
        ),
        start_segment_index=0,
        end_segment_index=(0, 0, 0),
    )
    robot = PCS(params=params, actuators=ThreadlikeActuator.tendons(routing))
    return robot, routing, float(jnp.sum(segment_lengths))


def build_sec_vd_case() -> SecVdCase:
    robot, routing, total_length = build_sec_vd_robot()
    q0 = jnp.zeros(robot.num_active_strains)
    qd0 = jnp.zeros_like(q0)
    steady = robot.rollout_to(
        initial_state=SystemState(t=jnp.array(0.0), y=jnp.concatenate([q0, qd0])),
        u=jnp.array([0.5, 0.2, 0.1]),
        t1=10.0,
        solver_dt=1e-4,
        save_dt=0.1,
    )
    q_des, qd_des = jnp.split(steady.y[-1], 2)
    speed = float(jnp.linalg.norm(qd_des))
    if speed > 1e-3:
        raise RuntimeError(
            f"Section Vd setpoint did not settle (velocity norm {speed:.3e})"
        )
    save_ts = robot._compute_save_times(0.0, 5.0, solver_dt=1e-4, save_dt=1e-4)
    return SecVdCase(
        robot=robot,
        routing=routing,
        total_length=total_length,
        q0=q0,
        qd0=qd0,
        q_des=q_des,
        x_des=end_effector_pose(robot, q_des, total_length),
        solver_dt=1e-4,
        save_ts=save_ts,
    )
