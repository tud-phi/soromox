"""Receding-horizon MPC push experiment: a tendon-actuated soft tentacle pushes
a rigid object across a table under closed-loop CEM/MPPI replanning against
the real Newton dynamics.

Uses the ``sap`` coupling backend, and supports a single-shot (plan-once,
execute-open-loop) mode via ``--single-shot``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import newton
import numpy as np
import warp as wp

from four_tendon_soft_arm import FourTendonSoftArm
from newton_cosserat.gvs import GVSMapping, add_gvs, gvs_control_view, gvs_state_views
from newton_cosserat.gvs.sap import make_sap_entries
from newton_cosserat.solvers import SolverCoupledSAP
from newton_cosserat.solvers.integrators import LinearlyImplicitEuler
from newton_cosserat.solvers.sap import SAPConfig
from smpc import (
    SMPCParams,
    compute_face_offset,
    k_schedule,
    push_cost,
    sample_colored_noise,
    smooth_control,
)
from soromox.systems import GVS as GVSRobot

jax.config.update("jax_enable_x64", True)

OBJECT_SHAPE = "box"
OBJECT_FOOTPRINT = 0.05
OBJECT_MASS = 0.05
TABLE_THICKNESS = 0.2
TABLE_HALF_EXTENT = 1.0
FACE_OFFSET = compute_face_offset(OBJECT_FOOTPRINT)

TENTACLE_Z_OFFSET = 0.37
OBJECT_X_OFFSET = 0.05
OBJECT_Y_OFFSET = 0.00
TABLE_TOP_Z = 0.0

PUSH_PATH = [0.05, -0.1, 0.15, -0.1, 0.15, 0.0, 0.05, 0.0]
NUM_PUSH_GOALS = len(PUSH_PATH) // 2

BASE_ORIENTATION = np.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
NUM_BASE_CONTROLS = 3
BASE_VELOCITY_MAX = 0.2

CONTACT_SEGMENT_INDEX = 2
NUM_COLLISION_BODIES = (1, 3, 12, 12)

DISABLED_CONTROL_IDX = ()


def build_model(
    robot: GVSRobot,
    device: str,
    *,
    world_count: int = 1,
    object_footprint: float = OBJECT_FOOTPRINT,
    object_xy: tuple[float, float] = (OBJECT_X_OFFSET, OBJECT_Y_OFFSET),
    table_thickness: float = TABLE_THICKNESS,
    base_height: float = TENTACLE_Z_OFFSET,
    initial_internal_q: np.ndarray | None = None,
    initial_base_xyz: tuple[float, float, float] | None = None,
    num_collision_bodies: int | tuple[int, ...] = NUM_COLLISION_BODIES,
    tentacle_color: tuple[float, float, float] = (0.08, 0.46, 0.76),
    object_color: tuple[float, float, float] = (0.9, 0.35, 0.2),
    table_color: tuple[float, float, float] = (0.9, 0.35, 0.2),
    extra_geometry_fn: Callable[[newton.ModelBuilder], None] | None = None,
) -> tuple[newton.Model, GVSMapping, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the static table and dynamic object, replicated across world_count
    independent Newton worlds in one Model.

    Args:
        robot: SoRoMoX GVS model represented by the Newton bodies.
        device: Warp device used to finalize the model.
        world_count: Number of independent worlds batched into one Model.
        object_footprint: Characteristic in-plane size used to scale the object [m].
        object_xy: Initial object-center (x, y) position in the world frame [m].
        table_thickness: Table slab thickness [m].
        base_height: World z-height of the tentacle's base [m], used when
            initial_base_xyz is None.
        initial_internal_q: Internal generalized coordinates baked into the
            rest pose, shape (robot.num_internal_dofs,). Defaults to zero
            strain.
        initial_base_xyz: World-frame base (x, y, z) baked in the same way.
            Defaults to (0, 0, base_height).
        extra_geometry_fn: Optional hook called with the builder just before
            finalize, for render-only geometry that no solver entry owns and
            whose body poses the caller writes itself.

    Returns:
        Tuple (model, mapping, object_bodies, object_shapes, table_bodies,
        table_shapes): the finalized model, its GVS mapping, and four
        length-``world_count`` arrays of global body/shape indices.
    """

    object_position = (object_xy[0], object_xy[1], 0.5 * object_footprint)

    template = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))

    if initial_internal_q is None:
        initial_internal_q = np.zeros(robot.num_internal_dofs)
    if initial_base_xyz is None:
        initial_base_xyz = (0.0, 0.0, base_height)
    initial_q = np.asarray(
        robot.pack_configuration(
            np.asarray(initial_internal_q),
            base_pose=np.concatenate([BASE_ORIENTATION, np.asarray(initial_base_xyz)]),
        )
    )
    template_mapping = add_gvs(
        template,
        robot,
        num_collision_bodies=num_collision_bodies,
        has_shape_collision=True,
        cfg=newton.ModelBuilder.ShapeConfig(collision_group=2, mu=1.8),
        initial_q=initial_q,
        color=tentacle_color,
    )
    rod_shapes = template_mapping.collision_shape_indices[0]
    for i, shape_a in enumerate(rod_shapes):
        for shape_b in rod_shapes[i + 1 :]:
            template.add_shape_collision_filter_pair(int(shape_a), int(shape_b))

    object_body = template.add_link(
        xform=wp.transform(wp.vec3(*object_position), wp.quat_identity()),
        is_kinematic=False,
        label="rigid_object",
    )
    if OBJECT_SHAPE == "box":
        object_density = OBJECT_MASS / object_footprint**3
    elif OBJECT_SHAPE == "cylinder":
        object_density = OBJECT_MASS / (math.pi * (0.5 * object_footprint) ** 2 * object_footprint)
    else:
        raise ValueError(f"OBJECT_SHAPE must be 'box' or 'cylinder', got {OBJECT_SHAPE!r}")
    object_cfg = newton.ModelBuilder.ShapeConfig(
        density=object_density,
        mu=0.25,
        restitution=0.0,
        margin=0.0,
        gap=0.0,
        collision_group=-1,
    )
    if OBJECT_SHAPE == "box":
        object_shape = template.add_shape_box(
            object_body,
            hx=0.5 * object_footprint,
            hy=0.5 * object_footprint,
            hz=0.5 * object_footprint,
            cfg=object_cfg,
            color=object_color,
            label="rigid_object_shape",
        )
    else:
        object_shape = template.add_shape_cylinder(
            object_body,
            radius=0.5 * object_footprint,
            half_height=0.5 * object_footprint,
            cfg=object_cfg,
            color=object_color,
            label="rigid_object_shape",
        )
    object_inertia = np.array(template.body_inertia[object_body]).reshape(3, 3)
    object_inertia[0, 0] *= 1.0e4
    object_inertia[1, 1] *= 1.0e4
    object_inertia[2, 2] *= 1.0e4
    template.body_inertia[object_body] = wp.mat33(object_inertia)
    template.body_inv_inertia[object_body] = wp.inverse(template.body_inertia[object_body])
    table_body = template.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * table_thickness), wp.quat_identity()),
        mass=0.0,
        inertia=wp.mat33(0.0),
        lock_inertia=True,
        is_kinematic=True,
        label="table_surface",
    )
    table_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.25,
        restitution=0.0,
        margin=0.0,
        gap=0.0,
        collision_group=3,
    )
    table_shape = template.add_shape_box(
        table_body,
        hx=TABLE_HALF_EXTENT,
        hy=TABLE_HALF_EXTENT,
        hz=0.5 * table_thickness,
        cfg=table_cfg,
        color=table_color,
        label="table_shape",
    )

    bodies_per_world = template.body_count
    shapes_per_world = template.shape_count

    if world_count == 1:
        builder = template
        mapping = template_mapping
    else:
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.replicate(template, world_count=world_count)
        mapping = None

    if extra_geometry_fn is not None:
        extra_geometry_fn(builder)

    model = builder.finalize(device=device)

    if mapping is None:
        mapping = GVSMapping.from_model(model, robot=robot, initial_q=initial_q)

    world_offsets = np.arange(world_count)
    object_bodies = world_offsets * bodies_per_world + object_body
    table_bodies = world_offsets * bodies_per_world + table_body
    object_shapes = world_offsets * shapes_per_world + object_shape
    table_shapes = world_offsets * shapes_per_world + table_shape

    return model, mapping, object_bodies, object_shapes, table_bodies, table_shapes


def read_bridged_state(robot: GVSRobot, state: newton.State, object_bodies: np.ndarray) -> np.ndarray:
    """Assemble every world's bridged state row from Newton's native buffers.

    Args:
        robot: SoRoMoX GVS model used to size the coordinate/velocity views.
        state: Newton state to read from.
        object_bodies: Global body index of the pushed object, one per world.

    Returns:
        Array of shape (world_count, state_dim): [q_internal, qd_internal,
        object_x, object_vx, object_y, object_vy, base_pose]. The trailing base_pose
        columns (scalar-last quaternion + xyz) let cost functions
        reconstruct the floating-base configuration for forward kinematics.
    """
    world_count = len(object_bodies)
    q_view, qd_view = gvs_state_views(
        state,
        world_count=world_count,
        coordinate_count=robot.num_coordinates,
        velocity_count=robot.num_velocities,
    )
    q_numpy = q_view.numpy()
    base_pose = q_numpy[:, : robot.num_base_coordinates].copy()
    q_internal = q_numpy[:, robot.num_base_coordinates :].copy()
    qd_internal = qd_view.numpy()[:, robot.num_base_velocities :].copy()

    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    object_x = body_q[object_bodies, 0]
    object_vx = body_qd[object_bodies, 3]
    object_y = body_q[object_bodies, 1]
    object_vy = body_qd[object_bodies, 4]
    object_state = np.stack([object_x, object_vx, object_y, object_vy], axis=1)

    return np.concatenate([q_internal, qd_internal, object_state, base_pose], axis=1)


def write_tendon_control(tendon_control: wp.array2d, u_k: np.ndarray) -> None:
    """Apply every world's tendon-tension vector, held constant (zero-order hold) until the next call.

    Args:
        tendon_control: Control view to write into.
        u_k: Tendon tensions, shape (world_count, num_actuators).
    """
    tendon_control.assign(np.asarray(u_k, dtype=np.float64))


def write_base_control(
    gvs_solver,
    base_orientation: np.ndarray,
    base_position: np.ndarray,
    base_velocity: np.ndarray,
    base_acceleration: np.ndarray | None = None,
) -> None:
    """Push every world's commanded base position/velocity/acceleration into the GVS solver's kinematic-base buffers.

    Args:
        gvs_solver: GVS solver instance owning the kinematic-base buffers.
        base_orientation: Fixed scalar-last quaternion [qx, qy, qz, qw], shape (4,).
        base_position: World-frame base positions, shape (world_count, 3).
        base_velocity: World-frame base linear velocities, shape (world_count, 3).
        base_acceleration: World-frame base linear accelerations, shape (world_count, 3), or ``None`` for zero.
    """
    world_count = base_position.shape[0]
    base_pose = np.zeros((world_count, 7), dtype=np.float64)
    base_pose[:, :4] = base_orientation
    base_pose[:, 4:7] = base_position
    base_twist = np.zeros((world_count, 6), dtype=np.float64)
    base_twist[:, 3:6] = base_velocity
    base_accel_spatial = np.zeros((world_count, 6), dtype=np.float64)
    if base_acceleration is not None:
        base_accel_spatial[:, 3:6] = base_acceleration
    gvs_solver.set_kinematic_base_targets(
        wp.array(base_pose, dtype=wp.float64, device=gvs_solver.device),
        wp.array(base_twist, dtype=wp.float64, device=gvs_solver.device),
        wp.array(base_accel_spatial, dtype=wp.float64, device=gvs_solver.device),
    )


def smooth_motion_blend(step: int, start: int, end: int, dt: float) -> tuple[float, float, float]:
    """Quintic (minimum-jerk) point-to-point blend fraction and its first two time derivatives.

    Args:
        step: Current step index within the phase (0-based substep count).
        start: First step index of the phase.
        end: Exclusive final step index of the phase (``substeps_per_control``).
        dt: Duration of one step [s] (the fine physics substep).

    Returns:
        (blend, blend_rate, blend_rate2): dimensionless blend fraction in
        [0, 1], and its first two time derivatives, [1/s] and [1/s^2].
    """

    duration_steps = end - start
    alpha = np.clip((step - start + 1) / duration_steps, 0.0, 1.0)
    blend = alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)
    blend_rate = 30.0 * alpha * alpha * (1.0 - alpha) * (1.0 - alpha) / (duration_steps * dt)
    blend_rate2 = 60.0 * alpha * (1.0 - alpha) * (1.0 - 2.0 * alpha) / ((duration_steps * dt) ** 2)
    return blend, blend_rate, blend_rate2


def quintic_velocity_target_step(
    v_start: np.ndarray,
    v_target: np.ndarray,
    substep: int,
    substeps_per_control: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One fine substep's smoothly-blended base velocity and acceleration, quintic-blending from ``v_start`` to ``v_target``.

    Args:
        v_start: Base velocity at the start of this control step, shape (world_count, 3).
        v_target: This control step's target velocity, same shape.
        substep: Current substep index within this control step, 0-based.
        substeps_per_control: Total substeps in this control step.
        dt: Fine physics substep [s].

    Returns:
        (velocity, acceleration): this substep's base velocity and
        acceleration, same shape as v_start/v_target.
    """

    blend, blend_rate, _blend_rate2 = smooth_motion_blend(substep, 0, substeps_per_control, dt)
    motion = v_target - v_start
    velocity = v_start + blend * motion
    acceleration = blend_rate * motion
    return velocity, acceleration


def step_physics_substep(
    gvs_solver,
    base_position: np.ndarray,
    base_velocity: np.ndarray,
    base_accel: np.ndarray,
    state_0: newton.State,
    state_1: newton.State,
    control,
    solver,
    dt: float,
) -> tuple[newton.State, newton.State]:
    """Advance physics by one fine substep: write the base control, step the solver, and swap the state double-buffer.

    Args:
        gvs_solver: GVS solver instance owning the kinematic-base buffers.
        base_position: This substep's base position, shape (world_count, 3).
        base_velocity: This substep's base velocity, shape (world_count, 3).
        base_accel: This substep's base acceleration, shape (world_count, 3).
        state_0: Current state, read from.
        state_1: Scratch state, written into.
        control: Control buffer for this step.
        solver: Coupled SAP solver to step.
        dt: Fine physics substep [s].

    Returns:
        (state_0, state_1) with identities swapped after stepping.
    """
    write_base_control(gvs_solver, BASE_ORIENTATION, base_position, base_velocity, base_accel)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts=None, dt=dt)
    return state_1, state_0


def make_solver(
    model: newton.Model,
    robot: GVSRobot,
    *,
    sap_config: SAPConfig | None = None,
    contact_capacity: int = 1024,
) -> SolverCoupledSAP:
    """Construct the GVS/rigid coupled SAP solver for the tentacle/table/object scene, across every replicated world.

    Args:
        model: Finalized Newton model spanning all worlds.
        robot: SoRoMoX GVS model.
        sap_config: Contact config.
        contact_capacity: Max collision points per world.

    Returns:
        The constructed coupled solver.
    """

    return SolverCoupledSAP(
        model,
        make_sap_entries(
            model, robots=[robot] * model.world_count, base_motion="kinematic", integrator=LinearlyImplicitEuler()
        ),
        config=SolverCoupledSAP.Config(contact=sap_config or SAPConfig(), contact_capacity=contact_capacity),
    )


def build_scene(
    robot,
    device: str,
    *,
    world_count: int,
    sap_config: SAPConfig | None = None,
    contact_capacity: int = 1024,
    object_xy: tuple[float, float] = (OBJECT_X_OFFSET, OBJECT_Y_OFFSET),
    initial_internal_q: np.ndarray | None = None,
    initial_base_xyz: tuple[float, float, float] | None = None,
    num_collision_bodies: int | tuple[int, ...] = NUM_COLLISION_BODIES,
    tentacle_color: tuple[float, float, float] = (0.2, 0.55, 0.9),
    object_color: tuple[float, float, float] = (0.9, 0.35, 0.2),
    table_color: tuple[float, float, float] = (0.9, 0.35, 0.2),
    extra_geometry_fn: Callable[[newton.ModelBuilder], None] | None = None,
):
    """Construct one fresh copy of the model/solver/collision scene, batching world_count independent rollouts.

    Args:
        robot: SoRoMoX GVS model.
        device: Warp device used to finalize the model.
        world_count: Number of independent worlds batched into one Model.
        sap_config: Contact config.
        contact_capacity: Max collision points per world.
        object_xy: Initial object-center (x, y) position in the world frame [m].
        initial_internal_q: Internal generalized coordinates baked into the rest pose.
        initial_base_xyz: World-frame base (x, y, z) baked in the same way.
        num_collision_bodies: Rod collision-body counts per segment.
        tentacle_color: Tentacle render color.
        object_color: Object render color.
        table_color: Table render color.
        extra_geometry_fn: Optional render-only geometry hook, forwarded to
            build_model.

    Returns:
        (model, mapping, object_bodies, table_bodies, solver, control, tendon_control).
    """

    model, mapping, object_bodies, _, table_bodies, _ = build_model(
        robot,
        device,
        world_count=world_count,
        object_xy=object_xy,
        initial_internal_q=initial_internal_q,
        initial_base_xyz=initial_base_xyz,
        num_collision_bodies=num_collision_bodies,
        tentacle_color=tentacle_color,
        object_color=object_color,
        table_color=table_color,
        extra_geometry_fn=extra_geometry_fn,
    )
    solver = make_solver(
        model,
        robot,
        sap_config=sap_config,
        contact_capacity=contact_capacity,
    )
    control = model.control()
    tendon_control = gvs_control_view(
        control,
        world_count=model.world_count,
        actuator_count=robot.num_actuators,
    )
    if tendon_control is None:
        raise RuntimeError("The tendon-actuated GVS model did not allocate control state")

    return model, mapping, object_bodies, table_bodies, solver, control, tendon_control


@dataclass
class ExecutionWorld:
    """One persistent world_count=1 scene, stepped forward by real physics one control step at a time."""

    robot: object
    object_bodies: np.ndarray
    solver: object
    gvs_solver: object
    control: object
    tendon_control: object
    state_0: object
    state_1: object
    q_view: object
    qd_view: object
    ndof: int
    num_base_coordinates: int
    num_base_velocities: int
    base_position: np.ndarray
    base_velocity: np.ndarray


def build_execution_world(
    robot,
    device: str,
    *,
    sap_config: SAPConfig | None = None,
    contact_capacity: int = 1024,
) -> ExecutionWorld:
    """Build ONE persistent world_count=1 scene, via `build_scene`.

    Args:
        robot: SoRoMoX GVS model.
        device: Warp device used to finalize the model.
        sap_config: Contact config.
        contact_capacity: Max collision points per world.

    Returns:
        The built `ExecutionWorld`.
    """

    (
        model,
        _,
        object_bodies,
        _,
        solver,
        control,
        tendon_control,
    ) = build_scene(
        robot,
        device,
        world_count=1,
        sap_config=sap_config,
        contact_capacity=contact_capacity,
    )

    gvs_solver = solver.solver("gvs")

    state_0, state_1 = model.state(), model.state()
    q_view, qd_view = gvs_state_views(
        state_0, world_count=1, coordinate_count=robot.num_coordinates, velocity_count=robot.num_velocities
    )

    return ExecutionWorld(
        robot=robot,
        object_bodies=object_bodies,
        solver=solver,
        gvs_solver=gvs_solver,
        control=control,
        tendon_control=tendon_control,
        state_0=state_0,
        state_1=state_1,
        q_view=q_view,
        qd_view=qd_view,
        ndof=robot.num_internal_dofs,
        num_base_coordinates=robot.num_base_coordinates,
        num_base_velocities=robot.num_base_velocities,
        base_position=np.array([[0.0, 0.0, TENTACLE_Z_OFFSET]], dtype=np.float64),
        base_velocity=np.zeros((1, 3), dtype=np.float64),
    )


def step_execution(world: ExecutionWorld, U_step: np.ndarray, dt: float, substeps_per_control: int) -> np.ndarray:
    """Advance `world` by `substeps_per_control` fine steps of real physics under one control step.

    Args:
        world: Execution world to advance in place, any world_count.
        U_step: Control action per world, shape (world_count, num_actuators + 3):
            tendon tensions followed by base target velocity [base_vx, base_vy, base_vz].
        dt: Fine physics substep [s].
        substeps_per_control: Number of fine substeps to take.

    Returns:
        Resulting bridged state, shape (world_count, state_dim).
    """

    tendon_count = world.tendon_control.shape[1]
    U_step = np.asarray(U_step, dtype=np.float64)
    write_tendon_control(world.tendon_control, U_step[:, :tendon_count])
    v_start = world.base_velocity.copy()
    v_target = np.clip(U_step[:, tendon_count : tendon_count + 3], -BASE_VELOCITY_MAX, BASE_VELOCITY_MAX)

    for substep in range(substeps_per_control):
        world.base_velocity, base_accel = quintic_velocity_target_step(
            v_start, v_target, substep, substeps_per_control, dt
        )
        world.base_position = world.base_position + world.base_velocity * dt
        world.state_0, world.state_1 = step_physics_substep(
            world.gvs_solver,
            world.base_position,
            world.base_velocity,
            base_accel,
            world.state_0,
            world.state_1,
            world.control,
            world.solver,
            dt,
        )

    return read_bridged_state(world.robot, world.state_0, world.object_bodies)


def real_dynamics_rollout_from_state(
    robot,
    device: str,
    dt: float,
    substeps_per_control: int,
    current_state_row: np.ndarray,
    U_samples: np.ndarray,
    *,
    sap_config: SAPConfig | None = None,
    contact_capacity: int = 1024,
    initial_base_velocity: np.ndarray | None = None,
    initial_object_body_q: np.ndarray | None = None,
    initial_object_body_qd: np.ndarray | None = None,
) -> np.ndarray:
    """Roll out a K-sample action batch from a fresh scene constructed directly at ``current_state_row``.

    Args:
        robot: SoRoMoX GVS model.
        device: Warp device used to finalize the model.
        dt: Fine physics substep [s].
        substeps_per_control: Number of fine substeps per control step.
        current_state_row: Bridged state row to seed the scene at, shape (state_dim,).
        U_samples: Action samples, shape (world_count, horizon, num_controls).
        sap_config: Contact config.
        contact_capacity: Max collision points per world.
        initial_base_velocity: Real base linear velocity at `current_state_row`'s time, shape (3,), or `None` for zero.
        initial_object_body_q: Real object body_q at `current_state_row`'s time (world xyz + scalar-last
            quaternion), shape (7,), or `None` to fall back on build_model's nominal upright rest pose.
            `current_state_row`'s own object slice only carries planar (x, y); without this, every
            rollout reseed silently drops the object's real z-height and orientation.
        initial_object_body_qd: Real object body_qd (angular velocity then linear velocity) at
            `current_state_row`'s time, shape (6,), or `None` for zero. `current_state_row` only carries
            planar (vx, vy); without this, every rollout reseed silently drops the object's real angular
            velocity and vz, which measurably biases the very next contact solve under this scene's
            zero-margin/zero-gap contact config.

    Returns:
        X: Rollout state trajectory, shape (world_count, horizon + 1, state_dim).
    """

    world_count = U_samples.shape[0]
    ndof = robot.num_internal_dofs
    num_base_coordinates = robot.num_base_coordinates
    num_base_velocities = robot.num_base_velocities

    current_state_row = np.asarray(current_state_row)
    q_internal = current_state_row[:ndof]
    qd_internal = current_state_row[ndof : 2 * ndof]
    object_x, object_vx, object_y, object_vy = current_state_row[2 * ndof : 2 * ndof + 4]
    base_pose = current_state_row[2 * ndof + 4 : 2 * ndof + 4 + num_base_coordinates]
    base_xyz = tuple(float(v) for v in base_pose[4:7])

    (
        model,
        _,
        object_bodies,
        _,
        solver,
        control,
        tendon_control,
    ) = build_scene(
        robot,
        device,
        world_count=world_count,
        sap_config=sap_config,
        contact_capacity=contact_capacity,
        object_xy=(float(object_x), float(object_y)),
        initial_internal_q=q_internal,
        initial_base_xyz=base_xyz,
    )

    gvs_solver = solver.solver("gvs")

    state_0, state_1 = model.state(), model.state()

    q_view, qd_view = gvs_state_views(
        state_0,
        world_count=world_count,
        coordinate_count=robot.num_coordinates,
        velocity_count=robot.num_velocities,
    )
    qd_full = qd_view.numpy()
    qd_full[:, num_base_velocities:] = np.tile(qd_internal[None, :], (world_count, 1))
    qd_view.assign(qd_full)

    body_qd_np = state_0.body_qd.numpy()
    if initial_object_body_qd is None:
        body_qd_np[object_bodies, 3] = object_vx
        body_qd_np[object_bodies, 4] = object_vy
    else:
        body_qd_np[object_bodies, :] = np.tile(np.asarray(initial_object_body_qd), (world_count, 1))
    state_0.body_qd.assign(body_qd_np)
    if initial_object_body_q is not None:
        body_q_np = state_0.body_q.numpy()
        body_q_np[object_bodies, :] = np.tile(np.asarray(initial_object_body_q), (world_count, 1))
        state_0.body_q.assign(body_q_np)

    base_position = np.tile(np.asarray(base_xyz), (world_count, 1))
    base_velocity = np.zeros(3) if initial_base_velocity is None else np.asarray(initial_base_velocity)
    base_velocity = np.tile(base_velocity, (world_count, 1))
    write_base_control(gvs_solver, BASE_ORIENTATION, base_position, base_velocity)

    world = ExecutionWorld(
        robot=robot,
        object_bodies=object_bodies,
        solver=solver,
        gvs_solver=gvs_solver,
        control=control,
        tendon_control=tendon_control,
        state_0=state_0,
        state_1=state_1,
        q_view=q_view,
        qd_view=qd_view,
        ndof=ndof,
        num_base_coordinates=num_base_coordinates,
        num_base_velocities=num_base_velocities,
        base_position=base_position,
        base_velocity=base_velocity,
    )

    rows = [read_bridged_state(robot, world.state_0, object_bodies)]
    num_control_steps = U_samples.shape[1]
    for step in range(num_control_steps):
        rows.append(step_execution(world, U_samples[:, step, :], dt, substeps_per_control))

    return np.stack(rows, axis=0).transpose(1, 0, 2)


@eqx.filter_jit
def _evaluate_cost_batch(params: SMPCParams, X_samples: jax.Array, U_samples: jax.Array) -> jax.Array:
    """JIT-compiled, vmap'd cost evaluation, shared across every CEM iteration/replan.

    Args:
        params: CEM parameters for the push cost.
        X_samples: Candidate state trajectories, shape (K, horizon + 1, state_dim).
        U_samples: Candidate control trajectories, shape (K, horizon, num_controls).

    Returns:
        Per-candidate cost, shape (K,).
    """
    return jax.vmap(push_cost, in_axes=(None, 0, 0))(params, X_samples, U_samples)


def plan_one_step(
    robot,
    device,
    dt,
    substeps_per_control,
    current_state_row,
    U_prev: np.ndarray,
    elite_U_prev: np.ndarray | None,
    key,
    params,
    *,
    num_iters: int,
    noise_beta: float = 1.0,
    fraction_elites_reused: float = 0.3,
    planner: str = "icem",
    temp: float | None = None,
    sap_config: SAPConfig | None = None,
    contact_capacity: int = 1024,
    initial_base_velocity: np.ndarray | None = None,
    initial_object_body_q: np.ndarray | None = None,
    initial_object_body_qd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, jax.Array, dict]:
    """One receding-horizon CEM replan starting from ``current_state_row``, following martius-lab/iCEM's MpcICem.

    Args:
        robot: SoRoMoX GVS model.
        device: Warp device used to finalize the model.
        dt: Fine physics substep [s].
        substeps_per_control: Number of fine substeps per control step.
        current_state_row: Bridged state row to seed the planning batch at, shape (state_dim,).
        U_prev: Previous replan's mean action sequence, warm-started from.
        elite_U_prev: Previous replan's elite action-sequences, or `None` for the first replan.
        key: JAX PRNG key.
        params: SMPCParams for the cost function and control distribution.
        num_iters: Number of CEM refit iterations this replan. Ignored for ``planner="mppi"``
            (classic MPPI is single-shot; forced to 1 internally).
        noise_beta: Colored-noise exponent for tendon-tension sampling, shared by both planners;
            0.0 = white (i.i.d. Gaussian) noise.
        fraction_elites_reused: Fraction of elites reused across/within replans. Ignored for
            ``planner="mppi"`` (forced to 0.0 internally; classic MPPI has no cross-step elite carry-over).
        planner: ``"icem"`` (elite argsort + mean/std refit) or ``"mppi"`` (single softmax-weighted
            average over the whole batch, octimal's ``mppi.solver.mppi_solve`` update rule; std stays
            fixed). Sampling/rollout/cost machinery is shared between both.
        temp: MPPI-only: softmax temperature, in cost units. Required when ``planner="mppi"``.
        sap_config: Contact config.
        contact_capacity: Max collision points per world.
        initial_base_velocity: Real base linear velocity at `current_state_row`'s time, shape (3,), or `None` for zero.
        initial_object_body_q: Real object body_q at `current_state_row`'s time, shape (7,); see
            `real_dynamics_rollout_from_state` for why this matters.
        initial_object_body_qd: Real object body_qd at `current_state_row`'s time, shape (6,); see
            `real_dynamics_rollout_from_state` for why this matters.

    Returns:
        (u_k, U, std, key, diagnostics): u_k is the action to execute next, U/std are the
        final-iteration mean/std, and diagnostics holds X_samples/cost/elite_idx/elite_U.
    """

    if planner not in ("icem", "mppi"):
        raise ValueError(f"planner must be 'icem' or 'mppi', got {planner!r}")
    if planner == "mppi":
        if temp is None:
            raise ValueError("planner='mppi' requires temp")
        num_iters = 1
        fraction_elites_reused = 0.0

    num_actuators = robot.num_actuators

    U_prev_jax = jnp.asarray(U_prev)
    U0 = jnp.concatenate([U_prev_jax[1:], U_prev_jax[-1:]])
    N = U0.shape[0]

    mean = U0
    std = jnp.broadcast_to(params.control_std, (N, params.control_std.shape[0]))
    mppi_mean = None

    num_elites_prev = 0 if elite_U_prev is None else elite_U_prev.shape[0]
    num_carry = round(fraction_elites_reused * num_elites_prev)

    X_samples = None
    cost_np = None
    elite_idx = None
    elite_U = None
    U_samples_jax = None
    prev_elite_U_jax = None
    prev_elite_X_jax = None
    iter_cost: list[np.ndarray] = []
    iter_elite_idx: list[np.ndarray] = []
    iter_U_samples: list[np.ndarray] = []
    iter_X_samples: list[np.ndarray] = []
    iter_time: list[float] = []

    for it in range(num_iters):
        iter_start = time.perf_counter()
        K_it = k_schedule(params, it)

        if it == 0:
            carry_count = num_carry
        elif prev_elite_U_jax is not None:
            carry_count = round(fraction_elites_reused * prev_elite_U_jax.shape[0])
        else:
            carry_count = 0
        carry_count = min(carry_count, max(K_it - 1, 0))
        fresh_count = K_it - carry_count

        key, subkey = jax.random.split(key)
        noise = sample_colored_noise(subkey, noise_beta, (fresh_count, N, mean.shape[1]), std)
        raw_U = np.asarray(mean[None, :, :] + noise)
        U_samples = np.concatenate(
            [np.abs(raw_U[..., :num_actuators]), raw_U[..., num_actuators:]], axis=-1
        )
        U_samples[..., DISABLED_CONTROL_IDX] = 0.0

        if it == num_iters - 1:
            U_samples[0] = np.asarray(mean)

        if it == 0 and carry_count > 0:
            carried = np.asarray(elite_U_prev[:carry_count])
            key, subkey2 = jax.random.split(key)
            tail_noise = sample_colored_noise(subkey2, noise_beta, (carry_count, N, mean.shape[1]), std)
            tail_raw = np.asarray(mean[None, :, :] + tail_noise)
            tail_U = np.concatenate(
                [np.abs(tail_raw[..., :num_actuators]), tail_raw[..., num_actuators:]], axis=-1
            )
            tail_U[..., DISABLED_CONTROL_IDX] = 0.0
            shifted = np.concatenate([carried[:, 1:, :], tail_U[:, -1:, :]], axis=1)
            U_samples = np.concatenate([U_samples, shifted], axis=0)

        X_samples = real_dynamics_rollout_from_state(
            robot,
            device,
            dt,
            substeps_per_control,
            current_state_row,
            U_samples,
            sap_config=sap_config,
            contact_capacity=contact_capacity,
            initial_base_velocity=initial_base_velocity,
            initial_object_body_q=initial_object_body_q,
            initial_object_body_qd=initial_object_body_qd,
        )

        X_samples_jax = jnp.asarray(X_samples)
        U_samples_jax = jnp.asarray(U_samples)

        if it > 0 and carry_count > 0:
            X_samples_jax = jnp.concatenate([X_samples_jax, prev_elite_X_jax[:carry_count]], axis=0)
            U_samples_jax = jnp.concatenate([U_samples_jax, prev_elite_U_jax[:carry_count]], axis=0)
            X_samples = np.asarray(X_samples_jax)
            U_samples = np.asarray(U_samples_jax)

        safe_per_sample = np.all(np.isfinite(X_samples.reshape(X_samples.shape[0], -1)), axis=1)
        num_safe = int(safe_per_sample.sum())
        if num_safe == 0:
            raise RuntimeError(f"plan_one_step iteration {it}: all samples in the batch produced non-finite states")

        cost = _evaluate_cost_batch(params, X_samples_jax, U_samples_jax)
        cost_np = np.where(safe_per_sample, np.asarray(cost), np.inf)

        num_elites = min(max(1, int(K_it * params.elite_frac)), num_safe)
        elite_idx = np.argsort(cost_np)[:num_elites]
        elite_U = U_samples_jax[elite_idx]
        elite_X = X_samples_jax[elite_idx]

        iter_cost.append(np.asarray(cost_np))
        iter_elite_idx.append(np.asarray(elite_idx))
        iter_U_samples.append(np.asarray(U_samples))
        iter_X_samples.append(np.asarray(X_samples))

        if planner == "icem":
            mean = jnp.mean(elite_U, axis=0)
            std = jnp.maximum(jnp.std(elite_U, axis=0), 1e-3)
        else:
            weights = np.exp(-(cost_np - cost_np[safe_per_sample].min()) / temp)
            weights = np.where(safe_per_sample, weights, 0.0)
            weights /= weights.sum()
            mppi_mean = jnp.sum(jnp.asarray(weights)[:, None, None] * U_samples_jax, axis=0)
            mean = mppi_mean

        prev_elite_U_jax = elite_U
        prev_elite_X_jax = elite_X

        iter_time.append(time.perf_counter() - iter_start)

    U = mppi_mean if planner == "mppi" else U_samples_jax[elite_idx[0]]
    if params.smoothing:
        U = smooth_control(np.asarray(U))

    U_np = np.asarray(U)
    std_np = np.asarray(std)
    u_k = np.asarray(U_np[0])

    diagnostics = {
        "X_samples": np.asarray(X_samples),
        "cost": np.asarray(cost_np),
        "elite_idx": np.asarray(elite_idx),
        "elite_U": np.asarray(elite_U),
        "iter_cost": iter_cost,
        "iter_elite_idx": iter_elite_idx,
        "iter_U_samples": iter_U_samples,
        "iter_X_samples": iter_X_samples,
        "iter_time": np.asarray(iter_time),
    }

    return u_k, U_np, std_np, key, diagnostics


PUSH_REACH_TOLERANCE_FRACTION = 0.10


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest distance from a point to a line segment, in the (x, y) plane.

    Args:
        px, py: Query point.
        ax, ay: Segment start.
        bx, by: Segment end.

    Returns:
        Euclidean distance from (px, py) to the closest point on segment [(ax, ay), (bx, by)].
    """
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    closest_x, closest_y = ax + t * dx, ay + t * dy
    return float(np.hypot(px - closest_x, py - closest_y))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed example configuration.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--single-shot", action="store_true", default=False, help="Plan once, execute open-loop.")
    parser.add_argument("--dt", type=float, default=None, help="Fine physics substep [s]; default 5e-4.")
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument("--num-mpc-steps", type=int, default=10)
    parser.add_argument(
        "--stop-after-one-square", action="store_true", default=False,
        help="Stop the run as soon as all NUM_PUSH_GOALS corners have been reached once (one full lap), "
        "instead of always running the full --num-mpc-steps. Saves compute when --num-mpc-steps is set "
        "as a generous upper bound rather than a tight target. Off by default. No effect with "
        "--push-goal (single-goal mode never completes a 'lap').",
    )
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument(
        "--k-samples", type=int, nargs="+", default=[200], help="Single value, or a per-iteration schedule."
    )
    parser.add_argument("--init-std", type=float, default=2.0)
    parser.add_argument("--base-velocity-std", type=float, default=0.1)
    parser.add_argument("--noise-beta", type=float, default=1.0, help="Colored-noise exponent; 0 = white noise.")
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--num-iters", type=int, default=1)
    planner_group = parser.add_mutually_exclusive_group()
    planner_group.add_argument("--icem", action="store_true", default=False, help="Use iCEM (default).")
    planner_group.add_argument(
        "--mmpi", action="store_true", default=False,
        help="Use classic single-iteration MPPI (softmax-weighted average) instead of iCEM.",
    )
    parser.add_argument(
        "--mppi-temp", type=float, default=150.0, help="MPPI-only: softmax temperature, in cost units."
    )
    parser.add_argument("--sap-stiffness", type=float, default=1.0e5)
    parser.add_argument("--sap-relaxation-time", type=float, default=0.005)
    parser.add_argument("--contact-capacity", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction-elites-reused", type=float, default=0.3)
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false", default=True)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--render-rollout-count", type=int, default=10)
    parser.add_argument("--object-shape", choices=("box", "cylinder"), default="box")
    parser.add_argument(
        "--push-goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
        help="Override PUSH_PATH with a single (x, y) goal -- NUM_PUSH_GOALS becomes 1, so the reach-check "
        "loops back onto the same goal forever instead of cycling through the default 4-corner square.",
    )
    parser.add_argument("--object-mass", type=float, default=None, help="Override OBJECT_MASS [kg]; default 0.05.")
    args = parser.parse_args()
    if not args.mmpi:
        args.icem = True
    if args.dt is None:
        args.dt = 5.0e-4
    if args.single_shot and args.planning_horizon != args.num_mpc_steps:
        args.planning_horizon = args.num_mpc_steps
    return args


def main() -> None:
    """Run closed-loop receding-horizon MPC against the real Newton dynamics."""

    args = parse_args()
    global OBJECT_SHAPE, PUSH_PATH, NUM_PUSH_GOALS, OBJECT_MASS
    OBJECT_SHAPE = args.object_shape
    if args.push_goal is not None:
        PUSH_PATH = list(args.push_goal)
        NUM_PUSH_GOALS = len(PUSH_PATH) // 2
    if args.object_mass is not None:
        OBJECT_MASS = args.object_mass
    jax_platform = "gpu" if args.device.startswith("cuda") else "cpu"
    jax_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    jax.config.update("jax_default_device", jax.devices(jax_platform)[jax_index])

    substeps_per_control = max(1, round(args.control_dt / args.dt))

    sap_config = SAPConfig(stiffness=args.sap_stiffness, relaxation_time=args.sap_relaxation_time)

    robot_wrapper = FourTendonSoftArm(gravity=[0.0, 0.0, -9.81])
    robot = robot_wrapper.model
    ndof = robot.num_internal_dofs
    num_actuators = robot.num_actuators
    num_controls = num_actuators + NUM_BASE_CONTROLS

    probe_model, _, probe_object_bodies, *_ = build_model(robot, args.device)
    probe_object_pose = probe_model.state().body_q.numpy()[probe_object_bodies[0]]
    initial_object_x = float(probe_object_pose[0])
    initial_object_y = float(probe_object_pose[1])
    initial_object_z = float(probe_object_pose[2])

    control_std = jnp.concatenate(
        [
            jnp.full((num_actuators,), args.init_std),
            jnp.full((NUM_BASE_CONTROLS,), args.base_velocity_std),
        ]
    )

    control_std = control_std.at[jnp.array(DISABLED_CONTROL_IDX, dtype=jnp.int32)].set(0.0)

    params = SMPCParams(
        robot=robot,
        control_std=control_std,
        K=args.k_samples,
        elite_frac=args.elite_frac,
        contact_segment_index=CONTACT_SEGMENT_INDEX,
        face_offset=FACE_OFFSET,
        contact_z=initial_object_z,
        table_z=TABLE_TOP_Z,
        smoothing=args.smoothing,
        push_goal_xy=jnp.asarray((PUSH_PATH[0], PUSH_PATH[1])),
    )

    planner = "mppi" if args.mmpi else "icem"

    world = build_execution_world(
        robot,
        args.device,
        sap_config=sap_config,
        contact_capacity=args.contact_capacity,
    )
    current_state_row = read_bridged_state(robot, world.state_0, world.object_bodies)[0]
    path_idx = 0
    legs_completed = 0
    object_pose_log = [world.state_0.body_q.numpy()[world.object_bodies, :7][0]]

    prev_object_x, prev_object_y = initial_object_x, initial_object_y
    leg_start_x, leg_start_y = initial_object_x, initial_object_y
    reach_tolerance = PUSH_REACH_TOLERANCE_FRACTION * float(
        np.hypot(float(params.push_goal_xy[0]) - leg_start_x, float(params.push_goal_xy[1]) - leg_start_y)
    )

    U_prev = np.zeros((args.planning_horizon, num_controls))
    elite_U_prev = None
    key = jax.random.PRNGKey(args.seed)

    X_rows = [current_state_row]
    U_executed = []
    diagnostics_costs = []
    diagnostics_elite_idx = []
    diagnostics_iter_costs = []
    diagnostics_iter_elite_idx = []
    diagnostics_iter_U_samples = []
    diagnostics_iter_X_samples = []
    diagnostics_iter_time = []
    rollout_X_log = []
    goal_xy_log = []
    reach_tolerance_log = []

    for step in range(args.num_mpc_steps):
        goal_xy_log.append((float(params.push_goal_xy[0]), float(params.push_goal_xy[1])))
        reach_tolerance_log.append(reach_tolerance)

        if not args.single_shot or step == 0:
            u_k, U_prev, _std, key, diagnostics = plan_one_step(
                robot,
                args.device,
                args.dt,
                substeps_per_control,
                current_state_row,
                U_prev,
                elite_U_prev,
                key,
                params,
                num_iters=args.num_iters,
                noise_beta=args.noise_beta,
                fraction_elites_reused=args.fraction_elites_reused,
                planner=planner,
                temp=args.mppi_temp if args.mmpi else None,
                sap_config=sap_config,
                contact_capacity=args.contact_capacity,
                initial_base_velocity=world.base_velocity[0],
                initial_object_body_q=world.state_0.body_q.numpy()[world.object_bodies[0]],
                initial_object_body_qd=world.state_0.body_qd.numpy()[world.object_bodies[0]],
            )
            elite_U_prev = diagnostics["elite_U"]
        else:
            u_k = U_prev[step]
            diagnostics = {}

        current_state_row = step_execution(world, u_k[None, :], args.dt, substeps_per_control)[0]
        object_pose_log.append(world.state_0.body_q.numpy()[world.object_bodies, :7][0])

        X_rows.append(current_state_row)
        U_executed.append(np.asarray(u_k))
        if "cost" in diagnostics:
            diagnostics_costs.append(np.asarray(diagnostics["cost"]))
        if "elite_idx" in diagnostics:
            diagnostics_elite_idx.append(np.asarray(diagnostics["elite_idx"]))
        if "iter_cost" in diagnostics:
            diagnostics_iter_costs.append(diagnostics["iter_cost"])
        if "iter_elite_idx" in diagnostics:
            diagnostics_iter_elite_idx.append(diagnostics["iter_elite_idx"])
        if "iter_U_samples" in diagnostics:
            diagnostics_iter_U_samples.append(diagnostics["iter_U_samples"])
        if "iter_X_samples" in diagnostics:
            diagnostics_iter_X_samples.append(diagnostics["iter_X_samples"])
        if "iter_time" in diagnostics:
            diagnostics_iter_time.append(diagnostics["iter_time"])
        if "cost" in diagnostics and "X_samples" in diagnostics:
            order = np.argsort(np.asarray(diagnostics["cost"]))
            top_n_idx = order[: args.render_rollout_count]
            rollout_X_log.append(np.asarray(diagnostics["X_samples"])[top_n_idx])

        completed = step + 1
        object_x = current_state_row[2 * ndof]
        object_y = current_state_row[2 * ndof + 2]
        dist_to_goal = point_to_segment_distance(
            float(params.push_goal_xy[0]), float(params.push_goal_xy[1]), prev_object_x, prev_object_y, object_x, object_y
        )
        reached = dist_to_goal < reach_tolerance
        prev_object_x, prev_object_y = object_x, object_y
        if reached:
            leg_start_x, leg_start_y = float(params.push_goal_xy[0]), float(params.push_goal_xy[1])
            legs_completed += 1
            path_idx = (path_idx + 1) % NUM_PUSH_GOALS
            params = dataclasses.replace(
                params, push_goal_xy=jnp.asarray((PUSH_PATH[2 * path_idx], PUSH_PATH[2 * path_idx + 1]))
            )
            reach_tolerance = PUSH_REACH_TOLERANCE_FRACTION * float(
                np.hypot(float(params.push_goal_xy[0]) - leg_start_x, float(params.push_goal_xy[1]) - leg_start_y)
            )
            U_prev = np.zeros((args.planning_horizon, num_controls))
            elite_U_prev = None
            if args.stop_after_one_square and legs_completed >= NUM_PUSH_GOALS:
                break

        if args.save_path is not None:
            partial_X = np.stack(X_rows, axis=0)
            partial_U = np.stack(U_executed, axis=0)
            partial_object_pose = np.stack(object_pose_log, axis=0)
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.save_path,
                X_final=partial_X,
                U_best=partial_U,
                object_pose=partial_object_pose,
                completed_steps=completed,
                goal_xy=np.asarray(goal_xy_log),
                reach_tolerance=np.asarray(reach_tolerance_log),
            )

    X_final = np.stack(X_rows, axis=0)
    U_best = np.stack(U_executed, axis=0)
    object_pose = np.stack(object_pose_log, axis=0)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "X_final": X_final,
            "U_best": U_best,
            "object_pose": object_pose,
            "goal_xy": np.asarray(goal_xy_log),
            "reach_tolerance": np.asarray(reach_tolerance_log),
        }
        if diagnostics_costs:
            try:
                save_kwargs["diagnostics_cost"] = np.stack(diagnostics_costs, axis=0)
            except ValueError:
                save_kwargs["diagnostics_cost"] = np.array(diagnostics_costs, dtype=object)
        if diagnostics_elite_idx:
            try:
                save_kwargs["diagnostics_elite_idx"] = np.stack(diagnostics_elite_idx, axis=0)
            except ValueError:
                save_kwargs["diagnostics_elite_idx"] = np.array(diagnostics_elite_idx, dtype=object)
        if diagnostics_iter_costs:
            save_kwargs["diagnostics_iter_cost"] = np.array(diagnostics_iter_costs, dtype=object)
        if diagnostics_iter_elite_idx:
            save_kwargs["diagnostics_iter_elite_idx"] = np.array(diagnostics_iter_elite_idx, dtype=object)
        if diagnostics_iter_U_samples:
            save_kwargs["diagnostics_iter_U_samples"] = np.array(diagnostics_iter_U_samples, dtype=object)
        if diagnostics_iter_X_samples:
            save_kwargs["diagnostics_iter_X_samples"] = np.array(diagnostics_iter_X_samples, dtype=object)
        if diagnostics_iter_time:
            save_kwargs["diagnostics_iter_time"] = np.array(diagnostics_iter_time, dtype=object)
        if rollout_X_log:
            save_kwargs["rollout_X_samples"] = np.stack(rollout_X_log, axis=0)
        np.savez(args.save_path, allow_pickle=True, **save_kwargs)


if __name__ == "__main__":
    main()
