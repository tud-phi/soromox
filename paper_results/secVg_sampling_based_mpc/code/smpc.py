"""iCEM/MPPI push-planner primitives: parameters, push cost, K schedule, colored noise, smoothing."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.signal import savgol_filter

from soromox import DynamicalSystem


class SMPCParams(eqx.Module):
    """Planner configuration: robot, control sampling distribution and push-contact geometry."""

    robot: DynamicalSystem
    control_std: Array
    K: int | tuple[int, ...] | list[int]
    object_footprint: float
    elite_frac: float = eqx.field(static=True, default=0.1)
    smoothing: bool = eqx.field(static=True, default=True)
    contact_segment_index: int = -1
    contact_z: float | None = None
    table_z: float | None = None
    push_goal_xy: Array = eqx.field(default_factory=lambda: jnp.zeros(2))
    push_goal_yaw: Array | None = None


PUSH_W_ROBOT_OBJECT = 3000.0
PUSH_W_OBJECT_GOAL_X = 15000.0
PUSH_W_OBJECT_GOAL_Y = 15000.0
PUSH_W_SIDE_ALIGN = 200.0
PUSH_R_SCALAR = 7.5
PUSH_R_RATE = 50.0
PUSH_R_SCALAR_BASE = 12.5
PUSH_R_RATE_BASE = 1000.0
PUSH_ALIGN_EPS = 1e-6
PUSH_W_TIP_TABLE_CLEARANCE = 100000.0
PUSH_TIP_TABLE_CLEARANCE_MARGIN = 0.02
PUSH_CONTACT_PENETRATION_MARGIN = 0.01
PUSH_W_ORIENTATION = 500.0


def push_cost(params: SMPCParams, X, U):
    """Cost of pushing the object to push_goal_xy, and to push_goal_yaw if set.

    Args:
        params: SMPC parameters; contact_z must be set.
        X: States, shape (horizon + 1, state_dim), [q, qd, body_x, body_vx, body_y, body_vy, body_yaw, base_pose].
        U: Controls, shape (horizon, num_controls), tendon tensions then base controls.

    Returns:
        Scalar cost.
    """
    ndof = params.robot.num_internal_dofs
    q_t = X[:, :ndof]
    body_x_t = X[:, 2 * ndof]
    body_y_t = X[:, 2 * ndof + 2]
    if params.robot.floating_base:
        base_pose_t = X[:, -params.robot.num_base_coordinates :]
        q_fk_t = jnp.concatenate([base_pose_t, q_t], axis=-1)
    else:
        q_fk_t = q_t
    contact_segment_index = params.contact_segment_index
    contact_pose_t = jax.vmap(
        lambda q: params.robot.forward_kinematics_tips(q)[contact_segment_index]
    )(q_fk_t)
    contact_z_t = contact_pose_t[:, 2, 3]

    p_R_xy = jnp.stack([contact_pose_t[:, 0, 3], contact_pose_t[:, 1, 3]], axis=-1)
    p_O_xy = jnp.stack([body_x_t, body_y_t], axis=-1)
    goal = params.push_goal_xy

    r_to_o = p_R_xy - p_O_xy
    o_to_g = goal - p_O_xy
    dist_robot_object_xy = jnp.linalg.norm(r_to_o, axis=-1)
    dist_object_goal = jnp.linalg.norm(o_to_g, axis=-1)

    push_dir_xy = o_to_g / (dist_object_goal[:, None] + PUSH_ALIGN_EPS)
    contact_radius = params.robot.radius_params[contact_segment_index, 1]
    face_offset = (
        0.5 * params.object_footprint + contact_radius - PUSH_CONTACT_PENETRATION_MARGIN
    )
    p_face_xy = p_O_xy - push_dir_xy * face_offset
    dist_robot_face_xy = jnp.linalg.norm(p_R_xy - p_face_xy, axis=-1)

    dz = contact_z_t - params.contact_z
    dist_robot_object = jnp.sqrt(dist_robot_face_xy**2 + dz**2)

    robot_object_cost = PUSH_W_ROBOT_OBJECT * jnp.mean(dist_robot_object)
    object_goal_cost = PUSH_W_OBJECT_GOAL_X * jnp.mean(
        jnp.abs(o_to_g[:, 0])
    ) + PUSH_W_OBJECT_GOAL_Y * jnp.mean(jnp.abs(o_to_g[:, 1]))

    cos_angle = jnp.sum(r_to_o * o_to_g, axis=-1) / (
        dist_robot_object_xy * dist_object_goal + PUSH_ALIGN_EPS
    )
    side_align_cost = PUSH_W_SIDE_ALIGN * jnp.mean(cos_angle + 1.0)

    n_tendon = params.robot.num_actuators
    U_tendon = U[:, :n_tendon]
    U_base = U[:, n_tendon:]
    effort_cost = PUSH_R_SCALAR * jnp.mean(
        jnp.linalg.norm(U_tendon, axis=-1)
    ) + PUSH_R_SCALAR_BASE * jnp.mean(jnp.linalg.norm(U_base, axis=-1))
    rate_cost = PUSH_R_RATE * jnp.mean(
        jnp.linalg.norm(jnp.diff(U_tendon, axis=0), axis=-1)
    ) + PUSH_R_RATE_BASE * jnp.mean(jnp.linalg.norm(jnp.diff(U_base, axis=0), axis=-1))

    if params.table_z is not None:
        tip_pose_t = jax.vmap(lambda q: params.robot.forward_kinematics_tips(q)[-1])(
            q_fk_t
        )
        tip_clearance_t = tip_pose_t[:, 2, 3] - params.table_z
        tip_table_cost = PUSH_W_TIP_TABLE_CLEARANCE * jnp.mean(
            jnp.maximum(PUSH_TIP_TABLE_CLEARANCE_MARGIN - tip_clearance_t, 0.0) ** 2
        )
    else:
        tip_table_cost = 0.0

    if params.push_goal_yaw is not None:
        yaw_err = X[:, 2 * ndof + 4] - params.push_goal_yaw
        yaw_err_wrapped = jnp.arctan2(jnp.sin(yaw_err), jnp.cos(yaw_err))
        orientation_cost = PUSH_W_ORIENTATION * jnp.mean(jnp.abs(yaw_err_wrapped))
    else:
        orientation_cost = 0.0

    return (
        robot_object_cost
        + object_goal_cost
        + side_align_cost
        + effort_cost
        + rate_cost
        + tip_table_cost
        + orientation_cost
    )


def k_schedule(params: SMPCParams, it):
    """Sample count for CEM iteration `it`; a K schedule shorter than the iterations repeats its last entry.

    Args:
        params: SMPC parameters.
        it: Iteration index, 0-based.

    Returns:
        Sample count.
    """
    K = params.K
    if not isinstance(K, (tuple, list)):
        return K
    return K[min(it, len(K) - 1)]


def sample_colored_noise(
    key: jax.Array, beta: float, shape: tuple[int, int, int], target_std: Array
) -> jax.Array:
    """Control noise with a 1/f^beta spectrum along the horizon (Timmer & Koenig 1995).

    Args:
        key: JAX PRNG key.
        beta: Colored-noise exponent; 0 is white noise.
        shape: (K, horizon, num_controls).
        target_std: Per-step std, broadcastable to (horizon, num_controls).

    Returns:
        Noise of shape `shape`.
    """
    K, T, A = shape
    if beta == 0.0 or T <= 1:
        return target_std * jax.random.normal(key, shape)

    freqs = jnp.fft.rfftfreq(T)
    freqs = freqs.at[0].set(freqs[1])
    scale = freqs ** (-beta / 2.0)

    key_r, key_i = jax.random.split(key)
    real_part = jax.random.normal(key_r, (K, freqs.shape[0], A)) * scale[None, :, None]
    imag_part = jax.random.normal(key_i, (K, freqs.shape[0], A)) * scale[None, :, None]
    if T % 2 == 0:
        imag_part = imag_part.at[:, -1, :].set(0.0)
        real_part = real_part.at[:, -1, :].multiply(jnp.sqrt(2))
    imag_part = imag_part.at[:, 0, :].set(0.0)
    real_part = real_part.at[:, 0, :].multiply(jnp.sqrt(2))

    raw = jnp.fft.irfft(real_part + 1j * imag_part, n=T, axis=1)
    ensemble_std = raw.std(axis=(0, 2), keepdims=True)
    return (raw / ensemble_std) * target_std


def smooth_control(U, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing along the horizon axis.

    Args:
        U: Control sequences, shape (horizon, ...).
        window_length: Window in timesteps, clamped to the horizon.
        polyorder: Polynomial order.

    Returns:
        Smoothed U, same shape.
    """
    N = U.shape[0]
    window_length = min(window_length, N if N % 2 == 1 else N - 1)
    if window_length <= polyorder:
        return U
    U_smooth = savgol_filter(np.asarray(U), window_length, polyorder, axis=0)
    return jnp.asarray(U_smooth)
