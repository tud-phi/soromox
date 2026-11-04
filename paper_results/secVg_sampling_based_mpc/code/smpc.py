"""Sampling-based MPC primitives shared by the iCEM/MPPI push planner:
the push cost function, the CEM parameter container, and the CEM
scaffolding (warm-starting, exploration-std init, per-iteration K
schedule, control smoothing) that both planner modes build on.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.signal import savgol_filter
from soromox import DynamicalSystem


class SMPCParams(eqx.Module):
    """Configuration shared by every CEM/MPPI replan: control distribution
    and the push-contact geometry the cost is evaluated against.
    """

    robot: DynamicalSystem
    control_std: Array
    K: int | tuple[int, ...] | list[int]
    elite_frac: float = eqx.field(static=True, default=0.1)
    smoothing: bool = eqx.field(static=True, default=True)
    contact_segment_index: int = -1
    face_offset: float | None = None
    contact_z: float | None = None
    table_z: float | None = None
    push_goal_xy: Array = eqx.field(default_factory=lambda: jnp.zeros(2))


PUSH_W_ROBOT_OBJECT = 3000.0
PUSH_W_OBJECT_GOAL_X = 3000.0
PUSH_W_OBJECT_GOAL_Y = 3000.0
PUSH_W_SIDE_ALIGN = 200.0
PUSH_R_SCALAR = 1.0
PUSH_R_RATE = 15.0
PUSH_R_SCALAR_BASE = 20.0
PUSH_R_RATE_BASE = 70.0
PUSH_ALIGN_EPS = 1e-6
PUSH_W_TIP_TABLE_CLEARANCE = 100000.0
PUSH_TIP_TABLE_CLEARANCE_MARGIN = 0.02
PUSH_CONTACT_RADIUS = 0.0057
PUSH_CONTACT_PENETRATION_MARGIN = 0.01


def compute_face_offset(object_footprint: float) -> float:
    """Distance from the object's center to its push face, minus the rod-tip's contact radius.

    Args:
        object_footprint: Box side length, or cylinder diameter/height [m].

    Returns:
        SMPCParams.face_offset for an object of this size.
    """
    return 0.5 * object_footprint + PUSH_CONTACT_RADIUS - PUSH_CONTACT_PENETRATION_MARGIN


def push_cost(params: SMPCParams, X, U):
    """Push cost toward params.push_goal_xy for the 2-DOF (x, y) object dynamics.

    Args:
        params: SMPC parameters, including push_goal_xy (static (2,) or per-step
            (horizon + 1, 2)) and the contact_segment_index/face_offset/contact_z
            push-contact geometry.
        X: State trajectory, shape (horizon + 1, state_dim); state layout matches
            run_experiment.py's read_bridged_state order: [q(ndof), qd(ndof),
            body_x, body_vx, body_y, body_vy, base_pose].
        U: Control trajectory, shape (horizon, num_controls).

    Returns:
        Scalar total cost.
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
    p_face_xy = p_O_xy - push_dir_xy * params.face_offset
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

    return (
        robot_object_cost
        + object_goal_cost
        + side_align_cost
        + effort_cost
        + rate_cost
        + tip_table_cost
    )


def k_schedule(params: SMPCParams, it):
    """Resolve this iteration's sample count from SMPCParams.K.

    Args:
        params: CEM parameters; K is either a fixed sample count or a
            per-iteration schedule.
        it: Iteration index, 0-based.

    Returns:
        Sample count for iteration `it`.
    """
    K = params.K
    if not isinstance(K, (tuple, list)):
        return K
    return K[min(it, len(K) - 1)]


def sample_colored_noise(key: jax.Array, beta: float, shape: tuple[int, int, int], target_std: float) -> jax.Array:
    """Sample tendon-tension noise with a 1/f^beta power spectrum along the horizon axis (Timmer & Koenig 1995).

    Args:
        key: JAX PRNG key.
        beta: Colored-noise exponent.
        shape: (K, horizon, num_actuators).
        target_std: Desired per-step standard deviation after normalization.

    Returns:
        Array of shape ``shape`` with per-step std equal to ``target_std``.
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

    Raw sampling-based MPC noise is i.i.d. per timestep, so a weighted-average
    or elite U can come out jagged even when the underlying weighting is
    meaningful. `window_length` must be odd and <= U.shape[0]; it's clamped
    here so this stays a no-op-safe drop-in across horizons of different
    lengths.
    """
    N = U.shape[0]
    window_length = min(window_length, N if N % 2 == 1 else N - 1)
    if window_length <= polyorder:
        return U
    U_smooth = savgol_filter(np.asarray(U), window_length, polyorder, axis=0)
    return jnp.asarray(U_smooth)
