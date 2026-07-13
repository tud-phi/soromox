"""Renderer-facing actuator visual primitives.

An actuator visual layer is a renderer-side grouping of actuator paths that
share one semantic role and visual treatment, such as all active tendons or all
McKibben muscles. It is not a dynamics object: the layer only describes the
geometry and optional display data needed to draw those actuators.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array

ActuatorVisualKind = Literal[
    "tendon", "push_rod", "muscle", "pneumatic", "cable", "generic"
]


@dataclass(frozen=True)
class ActuatorVisualLayer:
    """Visual geometry for one semantic actuator group.

    A layer groups actuators that can be drawn with the same renderer policy.
    Examples are an ``active_tendons`` layer containing sampled tendon paths or
    a ``mckibben_muscles`` layer containing two-point muscle segments.

    ``points`` stores one polyline per actuator with shape
    ``(num_actuators_in_layer, num_points, dim)``. Renderers may draw two-point
    lines as straight actuator segments or sampled polylines as routed actuators.
    """

    name: str
    kind: ActuatorVisualKind
    points: Array
    radius: float | Array | None = None
    line_width: float | None = None
    colors: Array | np.ndarray | list | tuple | None = None
    scalar_fields: Mapping[str, Array] = field(default_factory=dict)

    @property
    def paths(self) -> Array:
        """Semantic alias for the stored actuator polyline points."""
        return self.points

    @property
    def radii(self) -> float | Array | None:
        """Semantic alias for per-path physical radii."""
        return self.radius

    @property
    def scalar_values(self) -> Mapping[str, Array]:
        """Semantic alias for renderer scalar fields."""
        return self.scalar_fields

    def with_points(self, points: Array) -> ActuatorVisualLayer:
        """Return a copy with replaced geometry points."""
        return replace(self, points=points)


@dataclass(frozen=True)
class BatchedActuatorVisualLayer:
    """Batched actuator visual geometry.

    ``points`` has shape ``(num_robots, num_actuators, num_points, dim)``.
    """

    name: str
    kind: ActuatorVisualKind
    points: Array
    radius: float | Array | None = None
    line_width: float | None = None
    colors: Array | np.ndarray | list | tuple | None = None
    scalar_fields: Mapping[str, Array] = field(default_factory=dict)

    def with_points(self, points: Array) -> BatchedActuatorVisualLayer:
        """Return a copy with replaced geometry points."""
        return replace(self, points=points)


@dataclass(frozen=True)
class TrajectoryActuatorVisualLayer:
    """Trajectory actuator visual geometry.

    ``points`` has shape ``(num_robots, num_steps, num_actuators, num_points, dim)``.
    """

    name: str
    kind: ActuatorVisualKind
    points: Array
    radius: float | Array | None = None
    line_width: float | None = None
    colors: Array | np.ndarray | list | tuple | None = None
    scalar_fields: Mapping[str, Array] = field(default_factory=dict)

    def with_points(self, points: Array) -> TrajectoryActuatorVisualLayer:
        """Return a copy with replaced geometry points."""
        return replace(self, points=points)


def validate_actuator_visual_layer(layer: ActuatorVisualLayer) -> None:
    """Validate the common actuator visual layer contract."""
    points = jnp.asarray(layer.points)
    if points.ndim != 3:
        raise ValueError(
            f"Actuator layer {layer.name!r} points must have shape "
            f"(num_actuators, num_points, dim), got {points.shape}."
        )
    if points.shape[-1] not in (2, 3):
        raise ValueError(
            f"Actuator layer {layer.name!r} point dimension must be 2 or 3, "
            f"got {points.shape[-1]}."
        )
    if points.shape[-2] < 2 and points.shape[-3] > 0:
        raise ValueError(
            f"Actuator layer {layer.name!r} needs at least two points per actuator, "
            f"got {points.shape[-2]}."
        )


def validate_batched_actuator_visual_layer(
    layer: BatchedActuatorVisualLayer,
    *,
    num_robots: int | None = None,
) -> None:
    """Validate the batched actuator visual layer contract."""
    points = jnp.asarray(layer.points)
    if points.ndim != 4:
        raise ValueError(
            f"Batched actuator layer {layer.name!r} points must have shape "
            f"(num_robots, num_actuators, num_points, dim), got {points.shape}."
        )
    if num_robots is not None and int(points.shape[0]) != int(num_robots):
        raise ValueError(
            f"Batched actuator layer {layer.name!r} first dimension must match "
            f"num_robots={num_robots}, got {points.shape[0]}."
        )
    if points.shape[-1] not in (2, 3):
        raise ValueError(
            f"Batched actuator layer {layer.name!r} point dimension must be 2 or 3, "
            f"got {points.shape[-1]}."
        )
    if points.shape[-2] < 2 and points.shape[-3] > 0:
        raise ValueError(
            f"Batched actuator layer {layer.name!r} needs at least two points per "
            f"actuator, got {points.shape[-2]}."
        )


def validate_trajectory_actuator_visual_layer(
    layer: TrajectoryActuatorVisualLayer,
    *,
    num_robots: int | None = None,
    num_steps: int | None = None,
) -> None:
    """Validate the trajectory actuator visual layer contract."""
    points = jnp.asarray(layer.points)
    if points.ndim != 5:
        raise ValueError(
            f"Trajectory actuator layer {layer.name!r} points must have shape "
            "(num_robots, num_steps, num_actuators, num_points, dim), "
            f"got {points.shape}."
        )
    if num_robots is not None and int(points.shape[0]) != int(num_robots):
        raise ValueError(
            f"Trajectory actuator layer {layer.name!r} first dimension must match "
            f"num_robots={num_robots}, got {points.shape[0]}."
        )
    if num_steps is not None and int(points.shape[1]) != int(num_steps):
        raise ValueError(
            f"Trajectory actuator layer {layer.name!r} second dimension must match "
            f"num_steps={num_steps}, got {points.shape[1]}."
        )
    if points.shape[-1] not in (2, 3):
        raise ValueError(
            f"Trajectory actuator layer {layer.name!r} point dimension must be 2 or 3, "
            f"got {points.shape[-1]}."
        )
    if points.shape[-2] < 2 and points.shape[-3] > 0:
        raise ValueError(
            f"Trajectory actuator layer {layer.name!r} needs at least two points per "
            f"actuator, got {points.shape[-2]}."
        )


def normalize_actuator_layers(
    layers: tuple[ActuatorVisualLayer, ...] | list[ActuatorVisualLayer] | None,
) -> tuple[ActuatorVisualLayer, ...]:
    """Return a validated tuple of actuator visual layers."""
    if layers is None:
        return ()
    normalized = tuple(layers)
    for layer in normalized:
        validate_actuator_visual_layer(layer)
    return normalized


def normalize_batched_actuator_layers(
    layers: tuple[BatchedActuatorVisualLayer, ...]
    | list[BatchedActuatorVisualLayer]
    | None,
    *,
    num_robots: int | None = None,
) -> tuple[BatchedActuatorVisualLayer, ...]:
    """Return a validated tuple of batched actuator visual layers."""
    if layers is None:
        return ()
    normalized = tuple(layers)
    for layer in normalized:
        validate_batched_actuator_visual_layer(layer, num_robots=num_robots)
    return normalized


def normalize_trajectory_actuator_layers(
    layers: tuple[TrajectoryActuatorVisualLayer, ...]
    | list[TrajectoryActuatorVisualLayer]
    | None,
    *,
    num_robots: int | None = None,
    num_steps: int | None = None,
) -> tuple[TrajectoryActuatorVisualLayer, ...]:
    """Return a validated tuple of trajectory actuator visual layers."""
    if layers is None:
        return ()
    normalized = tuple(layers)
    for layer in normalized:
        validate_trajectory_actuator_visual_layer(
            layer, num_robots=num_robots, num_steps=num_steps
        )
    return normalized


def _as_rgba(color: np.ndarray) -> np.ndarray:
    arr = np.asarray(color, dtype=np.float64)
    if arr.shape[-1] == 3:
        alpha = np.ones((*arr.shape[:-1], 1), dtype=np.float64)
        arr = np.concatenate([arr, alpha], axis=-1)
    if arr.shape[-1] != 4:
        raise ValueError(f"Colors must have 3 or 4 channels, got shape {arr.shape}.")
    return np.clip(arr, 0.0, 1.0)


def _colormap_values(values: np.ndarray, colormap: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((*values.shape, 4), dtype=np.float64)
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    denom = max(v_max - v_min, 1e-12)
    scaled = np.clip((values - v_min) / denom, 0.0, 1.0)
    try:
        import matplotlib

        cmap = matplotlib.colormaps[colormap]
        rgba = cmap(scaled)
        return np.asarray(rgba, dtype=np.float64)
    except Exception:
        low = np.array([0.12, 0.38, 0.72, 1.0], dtype=np.float64)
        high = np.array([0.90, 0.22, 0.12, 1.0], dtype=np.float64)
        return low + (high - low) * scaled[..., None]


def resolve_actuator_rgba(
    layer: ActuatorVisualLayer
    | BatchedActuatorVisualLayer
    | TrajectoryActuatorVisualLayer,
    *,
    default_color: tuple[float, float, float],
    scalar_field: str | None = None,
    scalar_colormap: str = "viridis",
) -> np.ndarray:
    """Resolve per-actuator RGBA colors for a visual layer.

    The returned array follows the leading actuator axes of ``layer.points``
    excluding the polyline-point and coordinate axes.
    """
    points = np.asarray(layer.points)
    target_shape = points.shape[:-2]
    if layer.colors is not None:
        colors = _as_rgba(np.asarray(layer.colors, dtype=np.float64))
        if colors.shape[:-1] == target_shape:
            return colors
        if colors.shape[:-1] == target_shape[-1:]:
            reps = target_shape[:-1] + (1, 1)
            return np.tile(colors, reps)
        try:
            return np.broadcast_to(colors, (*target_shape, 4)).copy()
        except ValueError as exc:
            raise ValueError(
                f"Colors for actuator layer {layer.name!r} must broadcast to "
                f"{(*target_shape, 4)}, got {colors.shape}."
            ) from exc

    if layer.scalar_fields:
        chosen_field = scalar_field
        if chosen_field is None or chosen_field not in layer.scalar_fields:
            chosen_field = next(iter(layer.scalar_fields))
        values = np.asarray(layer.scalar_fields[chosen_field], dtype=np.float64)
        try:
            values = np.broadcast_to(values, target_shape)
        except ValueError as exc:
            raise ValueError(
                f"Scalar field {chosen_field!r} for actuator layer {layer.name!r} "
                f"must broadcast to {target_shape}, got {values.shape}."
            ) from exc
        return _colormap_values(values, scalar_colormap)

    base = _as_rgba(np.asarray(default_color, dtype=np.float64))
    return np.broadcast_to(base, (*target_shape, 4)).copy()
