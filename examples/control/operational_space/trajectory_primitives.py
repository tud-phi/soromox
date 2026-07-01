"""Reusable trajectory primitives for operational-space control examples.

This module intentionally lives under ``examples/``. It is shared by example
scripts, but it is not part of the public ``soromox`` package API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from soromox.control import ReferenceTrajectory
from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    quaternion_to_rotation_matrix,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_to_6d,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_rotation_matrix,
)

TRACKING_MODES = ("position", "pose")
POSITION_PRIMITIVES = (
    "point-to-point",
    "line",
    "planar-circle",
    "figure-eight",
    "helix",
)
ORIENTATION_PRIMITIVES = ("fixed", "sinusoid", "surface-following")
SURFACES = ("plane", "cylinder", "sphere")


@dataclass(frozen=True)
class TaskSpaceTrajectoryConfig:
    """Configuration for example task-space benchmark trajectories."""

    tracking: str = "position"
    position_primitive: str = "figure-eight"
    orientation_primitive: str = "fixed"
    surface: str = "plane"
    period: float = 4.0
    ramp_time: float = 0.0
    position_amplitude_x: float = 0.08
    position_amplitude_y: float = 0.04
    position_amplitude_z: float = 0.0
    line_amplitude: float = 0.08
    circle_radius: float = 0.06
    helix_radius: float = 0.06
    helix_pitch: float = 0.06
    orientation_amplitude: float = 1.0
    surface_radius: float = 0.08

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracking", _normalize_name(self.tracking))
        object.__setattr__(
            self, "position_primitive", _normalize_name(self.position_primitive)
        )
        object.__setattr__(
            self, "orientation_primitive", _normalize_name(self.orientation_primitive)
        )
        object.__setattr__(self, "surface", _normalize_name(self.surface))

        _validate_choice("tracking", self.tracking, TRACKING_MODES)
        _validate_choice(
            "position_primitive", self.position_primitive, POSITION_PRIMITIVES
        )
        _validate_choice(
            "orientation_primitive",
            self.orientation_primitive,
            ORIENTATION_PRIMITIVES,
        )
        _validate_choice("surface", self.surface, SURFACES)

        if self.period <= 0.0:
            raise ValueError("period must be positive.")
        if self.ramp_time < 0.0:
            raise ValueError("ramp_time must be nonnegative.")
        if self.surface_radius <= 0.0:
            raise ValueError("surface_radius must be positive.")


@dataclass(frozen=True)
class SurfaceGeometry:
    """Surface geometry used by surface-following pose references."""

    surface: str
    radius: float
    plane_point: Array
    plane_normal: Array
    preferred_y: Array
    cylinder_origin: Array
    cylinder_axis: Array
    sphere_center: Array


def make_end_effector_task_selector(tracking: str, n_velocity_dim: int) -> Array:
    """Return a one-point task selector for position-only or full-pose tracking."""
    tracking = _normalize_name(tracking)
    _validate_choice("tracking", tracking, TRACKING_MODES)

    if n_velocity_dim == 6:
        if tracking == "position":
            return jnp.array([False, False, False, True, True, True])
        return jnp.ones((6,), dtype=bool)

    if n_velocity_dim == 3:
        if tracking == "position":
            return jnp.array([False, True, True])
        return jnp.ones((3,), dtype=bool)

    raise ValueError(
        "Only 3D velocity spaces with dimension 6 and planar velocity spaces "
        f"with dimension 3 are supported, got {n_velocity_dim}."
    )


def make_operational_space_reference_trajectory(
    osd,
    q0: Array,
    ts: Array,
    config: TaskSpaceTrajectoryConfig,
) -> ReferenceTrajectory:
    """Build a full-pose reference trajectory for an end-effector task."""
    if osd.n_points != 1:
        raise ValueError(
            "This example helper currently supports a single operational-space "
            f"point, got {osd.n_points}."
        )
    if osd.is_planar and config.position_primitive == "helix":
        raise ValueError("The helix position primitive is only defined for 3D poses.")
    if (
        osd.is_planar
        and config.tracking == "pose"
        and config.orientation_primitive == "surface-following"
    ):
        raise ValueError("Surface-following orientation is only defined for 3D poses.")

    x0_full = osd.operational_space_poses(q0)
    x0_pose = x0_full.reshape(osd.n_points, osd.n_pose_dim)[0]

    n_orientation_dim = osd.n_orientation_dim
    orientation0 = x0_pose[:n_orientation_dim]
    position0 = x0_pose[n_orientation_dim:]

    if osd.is_planar:
        x_des_fn = _make_planar_pose_fn(position0, orientation0[0], config)
    else:
        R0 = _orientation_to_rotation_matrix(orientation0, osd.rotation_representation)
        x_des_fn = _make_spatial_pose_fn(
            position0, R0, osd.rotation_representation, config
        )

    return ReferenceTrajectory(
        ts=ts,
        x_des_fn=x_des_fn,
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )


def make_surface_geometry_metadata(
    osd,
    q0: Array,
    config: TaskSpaceTrajectoryConfig,
) -> SurfaceGeometry:
    """Return surface metadata matching the surface-following reference path."""
    if osd.n_points != 1:
        raise ValueError(
            "This example helper currently supports a single operational-space "
            f"point, got {osd.n_points}."
        )
    if osd.is_planar:
        raise ValueError("Surface geometry is only defined for 3D poses.")

    x0_full = osd.operational_space_poses(q0)
    x0_pose = x0_full.reshape(osd.n_points, osd.n_pose_dim)[0]
    orientation0 = x0_pose[: osd.n_orientation_dim]
    position0 = x0_pose[osd.n_orientation_dim :]
    R0 = _orientation_to_rotation_matrix(orientation0, osd.rotation_representation)
    return _make_surface_geometry_metadata(position0, R0, config)


def surface_normal_at(position: Array, surface_geometry: SurfaceGeometry) -> Array:
    """Return the outward normal of an example surface at ``position``."""
    return _surface_normal(
        position,
        surface_geometry.surface,
        surface_geometry.plane_normal,
        surface_geometry.cylinder_origin,
        surface_geometry.cylinder_axis,
        surface_geometry.sphere_center,
    )


def _make_planar_pose_fn(
    position0: Array, theta0: Array, config: TaskSpaceTrajectoryConfig
) -> Callable[[Array], Array]:
    def x_des_fn(t: Array) -> Array:
        position = position0 + _position_offset(t, 2, config)
        theta = theta0
        if config.tracking == "pose" and config.orientation_primitive == "sinusoid":
            theta = theta + _ramp(
                t, config.ramp_time
            ) * config.orientation_amplitude * jnp.sin(_phase(t, config.period))
        return jnp.concatenate([jnp.array([theta]), position])

    return x_des_fn


def _make_spatial_pose_fn(
    position0: Array,
    R0: Array,
    rotation_representation: RotationRepresentation,
    config: TaskSpaceTrajectoryConfig,
) -> Callable[[Array], Array]:
    surface_geometry = _make_surface_geometry_metadata(position0, R0, config)

    def x_des_fn(t: Array) -> Array:
        if (
            config.tracking == "pose"
            and config.orientation_primitive == "surface-following"
        ):
            position = _surface_position(t, surface_geometry, config)
        else:
            position = position0 + _position_offset(t, 3, config)

        if config.tracking == "position" or config.orientation_primitive == "fixed":
            R_des = R0
        elif config.orientation_primitive == "sinusoid":
            angle = (
                _ramp(t, config.ramp_time)
                * config.orientation_amplitude
                * jnp.sin(_phase(t, config.period))
            )
            R_delta = rotation_vector_to_rotation_matrix(jnp.array([0.0, 0.0, angle]))
            R_des = R_delta @ R0
        elif config.orientation_primitive == "surface-following":
            normal = _surface_normal(
                position,
                surface_geometry.surface,
                surface_geometry.plane_normal,
                surface_geometry.cylinder_origin,
                surface_geometry.cylinder_axis,
                surface_geometry.sphere_center,
            )
            R_des = _frame_from_x_axis(normal, surface_geometry.preferred_y)
        else:
            raise ValueError(
                f"Unsupported orientation primitive: {config.orientation_primitive}"
            )

        orientation = _rotation_matrix_to_orientation(R_des, rotation_representation)
        return jnp.concatenate([orientation, position])

    return x_des_fn


def _surface_position(
    t: Array,
    surface_geometry: SurfaceGeometry,
    config: TaskSpaceTrajectoryConfig,
) -> Array:
    chart_offset = _surface_chart_offset(t, config)
    u, v = chart_offset[0], chart_offset[1]
    normal = _normalize(surface_geometry.plane_normal)
    tangent_y, tangent_z = _surface_tangent_axes(
        normal,
        surface_geometry.preferred_y,
    )

    if surface_geometry.surface == "plane":
        return surface_geometry.plane_point + u * tangent_y + v * tangent_z

    if surface_geometry.surface == "cylinder":
        axis = _normalize(surface_geometry.cylinder_axis)
        radial0 = normal - jnp.dot(normal, axis) * axis
        radial0 = _normalize(radial0)
        circumferential = _normalize(jnp.cross(axis, radial0))
        theta = v / surface_geometry.radius
        radial = jnp.cos(theta) * radial0 + jnp.sin(theta) * circumferential
        return (
            surface_geometry.cylinder_origin
            + u * axis
            + surface_geometry.radius * radial
        )

    if surface_geometry.surface == "sphere":
        tangent = u * tangent_y + v * tangent_z
        angle_sq = jnp.dot(tangent, tangent) / surface_geometry.radius**2
        angle = jnp.sqrt(jnp.maximum(angle_sq, 1e-24))
        sinc = jnp.sin(angle) / angle
        radial = jnp.cos(angle) * normal + sinc * tangent / surface_geometry.radius
        return surface_geometry.sphere_center + surface_geometry.radius * radial

    raise ValueError(f"Unsupported surface: {surface_geometry.surface}")


def _surface_chart_offset(t: Array, config: TaskSpaceTrajectoryConfig) -> Array:
    if config.position_primitive == "helix":
        phase = _phase(t, config.period)
        turns = t / config.period
        ramp = _ramp(t, config.ramp_time)
        return ramp * jnp.array(
            [
                config.helix_pitch * turns,
                config.helix_radius * phase,
            ]
        )
    return _position_offset(t, 2, config)


def _surface_tangent_axes(normal: Array, preferred_y: Array) -> tuple[Array, Array]:
    tangent_y = preferred_y - jnp.dot(preferred_y, normal) * normal
    tangent_y = jnp.where(
        jnp.linalg.norm(tangent_y) > 1e-8,
        tangent_y,
        _fallback_axis(normal),
    )
    tangent_y = _normalize(tangent_y)
    tangent_z = _normalize(jnp.cross(normal, tangent_y))
    return tangent_y, tangent_z


def _make_surface_geometry_metadata(
    position0: Array,
    R0: Array,
    config: TaskSpaceTrajectoryConfig,
) -> SurfaceGeometry:
    normal0 = _normalize(R0[:, 0])
    preferred_y = _normalize(R0[:, 1])
    cylinder_axis = _cylinder_axis_from_normal(normal0)
    center = position0 - config.surface_radius * normal0
    return SurfaceGeometry(
        surface=config.surface,
        radius=config.surface_radius,
        plane_point=position0,
        plane_normal=normal0,
        preferred_y=preferred_y,
        cylinder_origin=center,
        cylinder_axis=cylinder_axis,
        sphere_center=center,
    )


def _position_offset(
    t: Array, position_dim: int, config: TaskSpaceTrajectoryConfig
) -> Array:
    primitive = config.position_primitive
    phase = _phase(t, config.period)
    ramp = _ramp(t, config.ramp_time)

    if primitive == "point-to-point":
        progress = _minimum_jerk(jnp.clip(t / config.period, 0.0, 1.0))
        if position_dim == 3:
            return progress * jnp.array(
                [
                    config.position_amplitude_x,
                    config.position_amplitude_y,
                    config.position_amplitude_z,
                ]
            )
        return progress * jnp.array(
            [config.position_amplitude_x, config.position_amplitude_y]
        )

    if primitive == "line":
        if position_dim == 3:
            return ramp * jnp.array([config.line_amplitude * jnp.sin(phase), 0.0, 0.0])
        return ramp * jnp.array([config.line_amplitude * jnp.sin(phase), 0.0])

    if primitive == "planar-circle":
        if position_dim == 3:
            return ramp * jnp.array(
                [
                    config.circle_radius * (jnp.cos(phase) - 1.0),
                    config.circle_radius * jnp.sin(phase),
                    0.0,
                ]
            )
        return ramp * jnp.array(
            [
                config.circle_radius * (jnp.cos(phase) - 1.0),
                config.circle_radius * jnp.sin(phase),
            ]
        )

    if primitive == "figure-eight":
        if position_dim == 3:
            return ramp * jnp.array(
                [
                    config.position_amplitude_x * jnp.sin(phase),
                    config.position_amplitude_y * jnp.sin(2.0 * phase),
                    0.0,
                ]
            )
        return ramp * jnp.array(
            [
                config.position_amplitude_x * jnp.sin(phase),
                config.position_amplitude_y * jnp.sin(2.0 * phase),
            ]
        )

    if primitive == "helix":
        if position_dim != 3:
            raise ValueError(
                "The helix position primitive is only defined for 3D poses."
            )
        turns = t / config.period
        return ramp * jnp.array(
            [
                config.helix_radius * (jnp.cos(phase) - 1.0),
                config.helix_radius * jnp.sin(phase),
                config.helix_pitch * turns,
            ]
        )

    raise ValueError(f"Unsupported position primitive: {primitive}")


def _surface_normal(
    position: Array,
    surface: str,
    plane_normal: Array,
    cylinder_origin: Array,
    cylinder_axis: Array,
    sphere_center: Array,
) -> Array:
    if surface == "plane":
        return _normalize(plane_normal)

    if surface == "cylinder":
        center_to_point = position - cylinder_origin
        radial = (
            center_to_point - jnp.dot(center_to_point, cylinder_axis) * cylinder_axis
        )
        return _normalize(radial)

    if surface == "sphere":
        return _normalize(position - sphere_center)

    raise ValueError(f"Unsupported surface: {surface}")


def _frame_from_x_axis(x_axis: Array, preferred_y: Array) -> Array:
    x_axis = _normalize(x_axis)
    y_candidate = preferred_y - jnp.dot(preferred_y, x_axis) * x_axis
    y_fallback = _fallback_axis(x_axis)
    y_candidate_norm = jnp.linalg.norm(y_candidate)
    y_axis = jnp.where(y_candidate_norm > 1e-8, y_candidate, y_fallback)
    y_axis = _normalize(y_axis)
    z_axis = _normalize(jnp.cross(x_axis, y_axis))
    y_axis = _normalize(jnp.cross(z_axis, x_axis))
    return jnp.column_stack([x_axis, y_axis, z_axis])


def _fallback_axis(x_axis: Array) -> Array:
    world_z = jnp.array([0.0, 0.0, 1.0])
    world_y = jnp.array([0.0, 1.0, 0.0])
    candidate = jnp.where(jnp.abs(jnp.dot(x_axis, world_z)) > 0.95, world_y, world_z)
    projected = candidate - jnp.dot(candidate, x_axis) * x_axis
    return _normalize(projected)


def _cylinder_axis_from_normal(normal: Array) -> Array:
    world_z = jnp.array([0.0, 0.0, 1.0])
    world_y = jnp.array([0.0, 1.0, 0.0])
    axis = jnp.cross(normal, world_z)
    fallback = jnp.cross(normal, world_y)
    axis = jnp.where(jnp.linalg.norm(axis) > 1e-8, axis, fallback)
    return _normalize(axis)


def _orientation_to_rotation_matrix(
    orientation: Array, rotation_representation: RotationRepresentation
) -> Array:
    if rotation_representation == RotationRepresentation.ROTATION_VECTOR:
        return rotation_vector_to_rotation_matrix(orientation)
    if rotation_representation == RotationRepresentation.QUATERNION:
        return quaternion_to_rotation_matrix(orientation)
    if rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
        return rotation_6d_to_rotation_matrix(orientation)
    raise ValueError(f"Unsupported rotation representation: {rotation_representation}")


def _rotation_matrix_to_orientation(
    R: Array, rotation_representation: RotationRepresentation
) -> Array:
    if rotation_representation == RotationRepresentation.ROTATION_VECTOR:
        return rotation_matrix_to_rotation_vector(R)
    if rotation_representation == RotationRepresentation.QUATERNION:
        return rotation_matrix_to_quaternion(R)
    if rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
        return rotation_matrix_to_6d(R)
    raise ValueError(f"Unsupported rotation representation: {rotation_representation}")


def _phase(t: Array, period: float) -> Array:
    return 2.0 * jnp.pi * t / period


def _ramp(t: Array, ramp_time: float) -> Array:
    if ramp_time == 0.0:
        return jnp.asarray(1.0, dtype=jnp.result_type(t, 1.0))
    return _minimum_jerk(jnp.clip(t / ramp_time, 0.0, 1.0))


def _minimum_jerk(s: Array) -> Array:
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _normalize(vec: Array, eps: float = 1e-12) -> Array:
    norm = jnp.linalg.norm(vec)
    return vec / jnp.maximum(norm, eps)


def _normalize_name(value: str) -> str:
    return value.replace("_", "-").lower()


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")


__all__ = [
    "ORIENTATION_PRIMITIVES",
    "POSITION_PRIMITIVES",
    "SURFACES",
    "TRACKING_MODES",
    "SurfaceGeometry",
    "TaskSpaceTrajectoryConfig",
    "make_end_effector_task_selector",
    "make_operational_space_reference_trajectory",
    "make_surface_geometry_metadata",
    "surface_normal_at",
]
