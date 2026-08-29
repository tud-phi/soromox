"""Base class for robot visualization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.rendering.actuation import (
    actuator_visual_layers,
    vectorized_actuator_visual_layers_trajectory,
)
from soromox.rendering.actuators import (
    ActuatorVisualLayer,
    BatchedActuatorVisualLayer,
    TrajectoryActuatorVisualLayer,
    normalize_actuator_layers,
    normalize_batched_actuator_layers,
    normalize_trajectory_actuator_layers,
)
from soromox.rendering.color_config import (
    DEFAULT_ROBOT_PALETTE,
    DEFAULT_SEGMENT_PALETTE,
    ColorLegend,
    RendererColorConfig,
    ResolvedBackboneColors,
    normalize_color_array,
    normalize_palette,
)
from soromox.systems.soft_robot import SoftRobot


class BaseSoftRobotRenderer(ABC):
    """Abstract base class for soft robot visualization backends.

    Provides cached forward kinematics and a common interface for rendering
    soft robots across different backends (Matplotlib, Open3D, OpenCV).

    Subclasses must implement:
        - render_frame: Render single configuration to RGB image array
        - render_sequence: Render animated sequence
        - show: Display a single frame interactively

    Subclasses may override:
        - is_3d: Default uses robot.is_planar
        - _extract_positions: Extract xyz/xy from FK poses

    Attributes:
        robot: SoftRobot system object with forward_kinematics method
        width: Image width in pixels
        height: Image height in pixels
        num_points: Number of points for discretizing backbone curve
        background_color: RGB tuple for background color (0-1 range)
        color_config: Shared color configuration for renderers
        L_max: Total length of the robot backbone
        base_pose: Robot base pose coordinates. Planar robots use
            ``[theta, x, y]``; spatial robots use scalar-first quaternion pose
            ``[qw, qx, qy, qz, x, y, z]``.
        base_transform: Homogeneous base transform. Shape ``(3, 3)`` for
            planar robots and ``(4, 4)`` for spatial robots.
        show_ground_plane: Whether supported backends render a ground reference
        ground_plane_size: Optional ground-plane side length in meters
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_config: RendererColorConfig | None = None,
        show_ground_plane: bool = False,
        ground_plane_size: float | None = None,
    ):
        """Initialize the renderer with a robot and visualization parameters.

        Args:
            robot: Robot system with forward_kinematics method and length property
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for backbone curve discretization
            background_color: RGB background color tuple (values 0-1)
            color_config: Shared renderer color configuration
            show_ground_plane: Whether supported backends render a ground reference
            ground_plane_size: Optional ground-plane side length in meters
        """
        self.robot: SoftRobot = robot
        self.width = width
        self.height = height
        self.num_points = num_points
        self.background_color = background_color
        self.color_config = color_config or RendererColorConfig()
        self.show_ground_plane = bool(show_ground_plane)
        self.ground_plane_size = (
            None if ground_plane_size is None else float(ground_plane_size)
        )
        if self.ground_plane_size is not None and self.ground_plane_size <= 0.0:
            raise ValueError("ground_plane_size must be positive when provided")
        self._is_planar = bool(robot.is_planar)
        floating_base = bool(getattr(robot, "floating_base", False))
        initial_base_pose = (
            robot._identity_base_pose
            if floating_base
            else getattr(robot, "fixed_base_pose", getattr(robot, "base_pose", None))
        )
        assert initial_base_pose is not None
        self.base_pose = jnp.asarray(initial_base_pose)
        self.base_transform = jnp.asarray(robot.base_transform)

        self._color_cache: dict[tuple[int, int, int], ResolvedBackboneColors] = {}
        self._segment_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Total robot length
        self.L_max = float(jnp.asarray(robot.length))

        # Explicit hooks let specialized and third-party robots override the
        # generic renderer-owned SoftRobot adapter.
        self._has_custom_actuator_visual_layers = callable(
            getattr(robot, "actuator_visual_layers", None)
        )
        self._has_batched_actuator_visual_layers = hasattr(
            robot, "actuator_visual_layers_batched"
        )
        self._has_trajectory_actuator_visual_layers = hasattr(
            robot, "actuator_visual_layers_trajectory"
        )
        self._uses_generic_actuator_visual_adapter = (
            isinstance(robot, SoftRobot) and not self._has_custom_actuator_visual_layers
        )
        self._has_actuator_visual_layers = (
            self._has_custom_actuator_visual_layers
            or self._uses_generic_actuator_visual_adapter
        )
        self._has_actuator_visuals = (
            self._has_actuator_visual_layers
            or self._has_batched_actuator_visual_layers
            or self._has_trajectory_actuator_visual_layers
        )

    def _resolve_ground_plane_size(self, *minimum_sizes: float) -> float:
        """Return the configured size or a robot-scaled backend default."""
        if self.ground_plane_size is not None:
            return self.ground_plane_size
        return max(1.35 * self.L_max, 0.1, *minimum_sizes)

    def _base_position(self, dim: int | None = None) -> np.ndarray:
        """Return the configured base translation in renderer coordinates.

        Args:
            dim: Optional target dimension. ``2`` returns ``[x, y]``; ``3``
                returns ``[x, y, z]`` with planar bases padded by ``z = 0``.
                If omitted, the renderer dimensionality is used.

        Returns:
            Base position as a NumPy array with shape ``(dim,)``.
        """
        target_dim = 3 if (dim is None and self.is_3d) else 2 if dim is None else dim
        transform = np.asarray(self.base_transform, dtype=np.float64)
        if transform.shape == (3, 3):
            pos = np.array([transform[0, 2], transform[1, 2]], dtype=np.float64)
        elif transform.shape == (4, 4):
            pos = np.asarray(transform[:3, 3], dtype=np.float64)
        else:
            raise ValueError(
                "base_transform must have shape (3, 3) or (4, 4), "
                f"got {transform.shape}."
            )
        return self._match_vector_dim(pos, target_dim, name="base position")

    def _base_tangent_axis(self, dim: int | None = None) -> np.ndarray:
        """Return the configured base-frame +x direction.

        The soft-robot convention is that zero base rotation aligns the
        undeformed backbone with the positive base-frame x-axis. This helper
        returns that axis after applying ``base_transform``.

        Args:
            dim: Optional target dimension. ``2`` returns the planar xy axis;
                ``3`` returns xyz, padding planar axes by ``z = 0``.

        Returns:
            Unit vector as a NumPy array with shape ``(dim,)``.
        """
        target_dim = 3 if (dim is None and self.is_3d) else 2 if dim is None else dim
        transform = np.asarray(self.base_transform, dtype=np.float64)
        if transform.shape == (3, 3):
            axis = np.asarray(transform[:2, 0], dtype=np.float64)
        elif transform.shape == (4, 4):
            axis = np.asarray(transform[:3, 0], dtype=np.float64)
        else:
            raise ValueError(
                "base_transform must have shape (3, 3) or (4, 4), "
                f"got {transform.shape}."
            )
        axis = self._match_vector_dim(axis, target_dim, name="base tangent axis")
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            fallback = np.zeros(target_dim, dtype=np.float64)
            fallback[0] = 1.0
            return fallback
        return axis / norm

    @staticmethod
    def _match_vector_dim(
        value: np.ndarray,
        target_dim: int,
        *,
        name: str,
    ) -> np.ndarray:
        """Return ``value`` with dimension ``target_dim`` by optional z-padding."""
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape[0] == target_dim:
            return vector
        if vector.shape[0] + 1 == target_dim:
            return np.concatenate([vector, np.zeros(1, dtype=np.float64)])
        if target_dim == 2 and vector.shape[0] == 3:
            return vector[:2]
        raise ValueError(
            f"{name} must have dimension {target_dim}, got {vector.shape}."
        )

    def _normalize_base_offsets(
        self,
        base_offsets: Array | np.ndarray,
        *,
        num_robots: int,
        target_dim: int,
        allow_single: bool = False,
        allow_extra_rows: bool = False,
        name: str = "base_offsets",
    ) -> jax.Array:
        """Validate and dimension-match per-robot positional base offsets.

        Args:
            base_offsets: Offset array. Expected shape is ``(N, dim)`` for
                batched calls. If ``allow_single`` is true, ``(dim,)`` and
                ``(1, dim)`` are accepted for a single robot.
            num_robots: Required number of rows.
            target_dim: Desired spatial dimension, usually the curve dimension.
            allow_single: Whether to accept a one-dimensional single offset.
            allow_extra_rows: Whether to accept more rows than ``num_robots``
                and use the leading ``num_robots`` rows. Intended for
                constructor-level layout configurations reused across smaller
                batches.
            name: Name used in error messages.

        Returns:
            JAX array with shape ``(num_robots, target_dim)``. 2D offsets are
            padded with ``z = 0`` when ``target_dim == 3``.
        """
        offsets = jnp.asarray(base_offsets)
        if offsets.ndim == 1 and allow_single:
            offsets = offsets.reshape(1, -1)
        if offsets.ndim != 2:
            raise ValueError(f"{name} must have shape (N, dim), got {offsets.shape}.")
        if allow_extra_rows and offsets.shape[0] > num_robots:
            offsets = offsets[:num_robots]
        if offsets.shape[0] != num_robots:
            raise ValueError(
                f"{name} first dimension ({offsets.shape[0]}) must match "
                f"number of robots ({num_robots})."
            )
        if offsets.shape[1] == target_dim:
            return offsets
        if offsets.shape[1] + 1 == target_dim:
            pad = jnp.zeros((offsets.shape[0], 1), dtype=offsets.dtype)
            return jnp.concatenate([offsets, pad], axis=1)
        raise ValueError(
            f"{name} must have shape ({num_robots}, {target_dim}) or "
            f"({num_robots}, {target_dim - 1}), got {offsets.shape}."
        )

    def _single_base_offset(
        self,
        base_offsets: Array | np.ndarray | None,
        *,
        target_dim: int,
        allow_extra_rows: bool = False,
    ) -> jax.Array | None:
        """Return a single normalized base offset or ``None``."""
        if base_offsets is None:
            return None
        return self._normalize_base_offsets(
            base_offsets,
            num_robots=1,
            target_dim=target_dim,
            allow_single=True,
            allow_extra_rows=allow_extra_rows,
        )[0]

    @property
    def is_planar(self) -> bool:
        """Return True for planar (SE(2)) robots."""
        return self._is_planar

    @property
    def is_3d(self) -> bool:
        """Return True for 3D rendering, False for planar rendering."""
        return not self.is_planar

    def compute_backbone_curve(self, q: Array) -> Array:
        """Compute backbone points from configuration.

        Args:
            q: Robot configuration array

        Returns:
            Array of shape (num_points, 3) for 3D or (num_points, 2) for 2D
        """
        poses = self.compute_backbone_poses(q)
        curve = self._extract_positions(poses)  # (num_points, dim)

        return curve

    def compute_backbone_poses(self, q: Array) -> Array:
        """Compute full FK poses at the configured backbone sample points."""
        if bool(getattr(self.robot, "floating_base", False)):
            base_pose, _ = self.robot.split_configuration(q)
            assert base_pose is not None
            self.base_pose = jnp.asarray(base_pose)
            self.base_transform = jnp.asarray(
                self.robot.base_transform_from_configuration(q)
            )
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self.robot.forward_kinematics_abscissa_batched(q, s_ps)

    def compute_actuator_visual_layers(
        self,
        q: Array,
        *,
        actuator_inputs: Array | None = None,
    ) -> tuple[ActuatorVisualLayer, ...]:
        """Compute renderer-facing actuator visual layers for one robot."""
        if not self._has_actuator_visual_layers:
            return ()
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        if self._uses_generic_actuator_visual_adapter:
            layers = actuator_visual_layers(
                self.robot, q, s_ps, actuator_inputs=actuator_inputs
            )
        else:
            layers = self.robot.actuator_visual_layers(  # type: ignore[attr-defined]
                q, s_ps, actuator_inputs=actuator_inputs
            )
        return normalize_actuator_layers(layers)

    def _offset_batched_actuator_layers(
        self,
        layers: tuple[BatchedActuatorVisualLayer, ...],
        base_offsets: Array | np.ndarray,
    ) -> tuple[BatchedActuatorVisualLayer, ...]:
        """Apply per-robot base offsets to batched actuator layer points."""
        offset_layers: list[BatchedActuatorVisualLayer] = []
        for layer in layers:
            points = jnp.asarray(layer.points)
            offsets = self._normalize_base_offsets(
                base_offsets,
                num_robots=int(points.shape[0]),
                target_dim=int(points.shape[-1]),
            )
            offset_layers.append(layer.with_points(points + offsets[:, None, None, :]))
        return tuple(offset_layers)

    def _offset_trajectory_actuator_layers(
        self,
        layers: tuple[TrajectoryActuatorVisualLayer, ...],
        base_offsets: Array | np.ndarray,
    ) -> tuple[TrajectoryActuatorVisualLayer, ...]:
        """Apply per-robot base offsets to trajectory actuator layer points."""
        offset_layers: list[TrajectoryActuatorVisualLayer] = []
        for layer in layers:
            points = jnp.asarray(layer.points)
            offsets = self._normalize_base_offsets(
                base_offsets,
                num_robots=int(points.shape[0]),
                target_dim=int(points.shape[-1]),
            )
            offset_layers.append(
                layer.with_points(points + offsets[:, None, None, None, :])
            )
        return tuple(offset_layers)

    @staticmethod
    def _stack_optional_arrays(
        values: list[Array | np.ndarray | list | tuple | None],
        *,
        name: str,
        layer_name: str,
    ) -> Array | None:
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                f"{name} for actuator layer {layer_name!r} must be provided for "
                "all robots/timesteps or none."
            )
        return jnp.stack([jnp.asarray(value) for value in values])

    @staticmethod
    def _stack_radius(values: list[float | Array | None]) -> float | Array | None:
        if all(value is None for value in values):
            return None
        first = values[0]
        if first is None:
            return None
        if np.asarray(first).ndim == 0:
            return first
        return jnp.stack([jnp.asarray(value) for value in values])

    @staticmethod
    def _stack_scalar_fields(
        layers: list[ActuatorVisualLayer | BatchedActuatorVisualLayer],
    ) -> dict[str, Array]:
        first_keys = tuple(layers[0].scalar_fields.keys())
        stacked: dict[str, Array] = {}
        for layer in layers[1:]:
            if tuple(layer.scalar_fields.keys()) != first_keys:
                raise ValueError(
                    f"Actuator layer {layers[0].name!r} scalar field keys must "
                    "match across robots/timesteps."
                )
        for key in first_keys:
            stacked[key] = jnp.stack(
                [jnp.asarray(layer.scalar_fields[key]) for layer in layers]
            )
        return stacked

    @staticmethod
    def _validate_matching_layers(
        reference: tuple[ActuatorVisualLayer, ...],
        candidate: tuple[ActuatorVisualLayer, ...],
    ) -> None:
        if len(candidate) != len(reference):
            raise ValueError(
                "All robots must expose the same number of actuator visual layers."
            )
        for ref_layer, layer in zip(reference, candidate):
            if (layer.name, layer.kind) != (ref_layer.name, ref_layer.kind):
                raise ValueError(
                    "Actuator visual layer names and kinds must match across "
                    "robots/timesteps."
                )
            if jnp.asarray(layer.points).shape != jnp.asarray(ref_layer.points).shape:
                raise ValueError(
                    f"Actuator visual layer {layer.name!r} shape changed from "
                    f"{jnp.asarray(ref_layer.points).shape} to "
                    f"{jnp.asarray(layer.points).shape}."
                )

    @staticmethod
    def _actuator_inputs_for_batch(
        actuator_inputs: Array | np.ndarray | None,
        *,
        num_robots: int,
    ) -> list[Array | None]:
        if actuator_inputs is None:
            return [None] * num_robots
        inputs = jnp.asarray(actuator_inputs)
        if inputs.ndim >= 2 and inputs.shape[0] == num_robots:
            return [inputs[i] for i in range(num_robots)]
        if num_robots == 1:
            return [inputs]
        raise ValueError(
            "Batched actuator_inputs must have leading dimension matching the "
            f"number of robots ({num_robots}), got {inputs.shape}."
        )

    @staticmethod
    def _actuator_inputs_for_timestep(
        actuator_inputs: Array | np.ndarray | None,
        *,
        num_robots: int,
        num_steps: int,
        frame_idx: int,
    ) -> Array | np.ndarray | None:
        if actuator_inputs is None:
            return None
        inputs = jnp.asarray(actuator_inputs)
        if inputs.ndim >= 3 and inputs.shape[:2] == (num_robots, num_steps):
            return inputs[:, frame_idx]
        if inputs.ndim >= 2 and inputs.shape[0] == num_steps:
            return inputs[frame_idx]
        return inputs

    def compute_actuator_visual_layers_batched(
        self,
        q_batch: Array,
        base_offsets: Array,
        *,
        actuator_inputs: Array | np.ndarray | None = None,
    ) -> tuple[BatchedActuatorVisualLayer, ...]:
        """Compute actuator visual layers for multiple robots with base offsets."""
        if not self._has_actuator_visuals:
            return ()

        q_batch = jnp.asarray(q_batch)
        if q_batch.ndim != 2:
            raise ValueError(
                f"q_batch must have shape (N, DOF), got {q_batch.shape} with ndim={q_batch.ndim}"
            )
        num_robots = int(q_batch.shape[0])

        if self._has_batched_actuator_visual_layers:
            s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
            raw_layers = self.robot.actuator_visual_layers_batched(  # type: ignore[attr-defined]
                q_batch,
                s_ps,
                actuator_inputs=actuator_inputs,
            )
            layers = normalize_batched_actuator_layers(
                raw_layers, num_robots=num_robots
            )
            return self._offset_batched_actuator_layers(layers, base_offsets)

        if not self._has_actuator_visual_layers:
            return ()

        inputs_by_robot = self._actuator_inputs_for_batch(
            actuator_inputs, num_robots=num_robots
        )
        layers_by_robot = [
            self.compute_actuator_visual_layers(
                q_batch[i], actuator_inputs=inputs_by_robot[i]
            )
            for i in range(num_robots)
        ]
        if not layers_by_robot or not layers_by_robot[0]:
            return ()
        reference = layers_by_robot[0]
        for layers in layers_by_robot[1:]:
            self._validate_matching_layers(reference, layers)

        batched_layers: list[BatchedActuatorVisualLayer] = []
        for layer_idx, ref_layer in enumerate(reference):
            robot_layers = [layers[layer_idx] for layers in layers_by_robot]
            points = jnp.stack([jnp.asarray(layer.points) for layer in robot_layers])
            colors = self._stack_optional_arrays(
                [layer.colors for layer in robot_layers],
                name="colors",
                layer_name=ref_layer.name,
            )
            scalar_fields = self._stack_scalar_fields(robot_layers)
            batched_layers.append(
                BatchedActuatorVisualLayer(
                    name=ref_layer.name,
                    kind=ref_layer.kind,
                    points=points,
                    radius=self._stack_radius([layer.radius for layer in robot_layers]),
                    line_width=ref_layer.line_width,
                    colors=colors,
                    scalar_fields=scalar_fields,
                )
            )
        return self._offset_batched_actuator_layers(tuple(batched_layers), base_offsets)

    def compute_actuator_visual_layers_trajectory(
        self,
        q_ts: Array,
        base_offsets: Array,
        *,
        actuator_inputs: Array | np.ndarray | None = None,
    ) -> tuple[TrajectoryActuatorVisualLayer, ...]:
        """Compute actuator visual layers for ``(N, T, DOF)`` trajectories."""
        q_ts = jnp.asarray(q_ts)
        if q_ts.ndim != 3:
            raise ValueError(
                f"q_ts must have shape (N, T, DOF), got {q_ts.shape} with ndim={q_ts.ndim}"
            )
        num_robots, num_steps, _ = q_ts.shape

        if self._has_trajectory_actuator_visual_layers:
            s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
            raw_layers = self.robot.actuator_visual_layers_trajectory(  # type: ignore[attr-defined]
                q_ts,
                s_ps,
                actuator_inputs=actuator_inputs,
            )
            layers = normalize_trajectory_actuator_layers(
                raw_layers,
                num_robots=int(num_robots),
                num_steps=int(num_steps),
            )
            return self._offset_trajectory_actuator_layers(layers, base_offsets)

        # Generic SoftRobot geometry is renderer-owned and JAX-vectorizable.
        # Specialized robot hooks take precedence in the branch above.
        if self._uses_generic_actuator_visual_adapter:
            s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
            raw_layers = vectorized_actuator_visual_layers_trajectory(
                self.robot,
                q_ts,
                s_ps,
                actuator_inputs=actuator_inputs,
            )
            layers = normalize_trajectory_actuator_layers(
                raw_layers,
                num_robots=int(num_robots),
                num_steps=int(num_steps),
            )
            return self._offset_trajectory_actuator_layers(layers, base_offsets)

        if not (
            self._has_batched_actuator_visual_layers or self._has_actuator_visual_layers
        ):
            return ()

        layers_by_time = []
        for frame_idx in range(int(num_steps)):
            frame_inputs = self._actuator_inputs_for_timestep(
                actuator_inputs,
                num_robots=int(num_robots),
                num_steps=int(num_steps),
                frame_idx=frame_idx,
            )
            layers_by_time.append(
                self.compute_actuator_visual_layers_batched(
                    q_ts[:, frame_idx, :],
                    base_offsets,
                    actuator_inputs=frame_inputs,
                )
            )
        if not layers_by_time or not layers_by_time[0]:
            return ()

        trajectory_layers: list[TrajectoryActuatorVisualLayer] = []
        for layer_idx, ref_layer in enumerate(layers_by_time[0]):
            time_layers = [layers[layer_idx] for layers in layers_by_time]
            points = jnp.stack([layer.points for layer in time_layers], axis=1)
            colors = self._stack_optional_arrays(
                [layer.colors for layer in time_layers],
                name="colors",
                layer_name=ref_layer.name,
            )
            if colors is not None:
                colors = jnp.moveaxis(colors, 0, 1)
            scalar_fields = {
                key: jnp.stack(
                    [jnp.asarray(layer.scalar_fields[key]) for layer in time_layers],
                    axis=1,
                )
                for key in ref_layer.scalar_fields
            }
            trajectory_layers.append(
                TrajectoryActuatorVisualLayer(
                    name=ref_layer.name,
                    kind=ref_layer.kind,
                    points=points,
                    radius=ref_layer.radius,
                    line_width=ref_layer.line_width,
                    colors=colors,
                    scalar_fields=scalar_fields,
                )
            )
        return tuple(trajectory_layers)

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xyz or xy positions from FK poses.

        Args:
            poses: Forward kinematics output (backend-specific format)

        Returns:
            Position array of shape (N, 3) for 3D or (N, 2) for 2D
        """
        if self.is_3d:
            return self._extract_positions_3d(poses)
        return self._extract_positions_2d(poses)

    def _extract_positions_2d(self, poses: Array) -> Array:
        """Extract 2D positions from SE(2) or SE(3) poses.

        Args:
            poses: Forward kinematics output - either:
                - (N, 3) for SE(2) poses [theta, x, y]
                - (N, 4, 4) for SE(3) transformation matrices
                - (N, 2) for planar xy positions

        Returns:
            Position array of shape (N, 2) with xy coordinates.
        """
        poses = jnp.asarray(poses)

        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            return poses[:, :2, 3]
        if poses.ndim == 2 and poses.shape[-1] == 3:
            return poses[:, 1:3]
        if poses.ndim == 2 and poses.shape[-1] == 2:
            return poses

        raise ValueError(
            "Unsupported pose format for planar extraction: expected "
            "SE(2) (N, 3), SE(3) (N, 4, 4), or planar (N, 2), "
            f"got shape {poses.shape}"
        )

    def _extract_positions_3d(self, poses: Array) -> Array:
        """Extract 3D positions from SE(2) or SE(3) poses.

        This helper method allows 3D renderers to handle both SE(2) (planar) and
        SE(3) (3D) robots by converting SE(2) poses to 3D coordinates.

        Args:
            poses: Forward kinematics output - either:
                - (N, 3) for SE(2) poses [theta, x, y]
                - (N, 4, 4) for SE(3) transformation matrices

        Returns:
            Position array of shape (N, 3) with xyz coordinates.
            For SE(2) poses, z is set to 0.
        """
        poses = jnp.asarray(poses)

        # Check if poses are SE(3) matrices (4x4) or SE(2) vectors (3-element)
        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            # SE(3): extract translation from 4x4 matrices
            return poses[:, :3, 3]
        elif poses.ndim == 3 and poses.shape[-2:] == (3, 3):
            # Homogeneous SE(2): extract x, y and pad with z=0
            xy = poses[:, :2, 2]
            z = jnp.zeros((poses.shape[0], 1), dtype=poses.dtype)
            return jnp.concatenate([xy, z], axis=1)
        elif poses.ndim == 2 and poses.shape[-1] == 3:
            # SE(2): extract x, y from [theta, x, y] and pad with z=0
            xy = poses[:, 1:3]  # Extract x, y (skip theta at index 0)
            z = jnp.zeros((poses.shape[0], 1), dtype=poses.dtype)
            return jnp.concatenate([xy, z], axis=1)
        else:
            raise ValueError(
                f"Unsupported pose format: expected SE(2) (N, 3) or SE(3) (N, 4, 4), "
                f"got shape {poses.shape}"
            )

    def _extract_material_frames_3d(self, poses: Array) -> Array:
        """Extract 3D material-frame rotations from SE(2) or SE(3) poses."""
        poses = jnp.asarray(poses)
        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            return poses[:, :3, :3]
        if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
            rotations = poses[:, :2, :2]
            frames = jnp.zeros((poses.shape[0], 3, 3), dtype=poses.dtype)
            frames = frames.at[:, :2, :2].set(rotations)
            return frames.at[:, 2, 2].set(1.0)
        if poses.ndim == 2 and poses.shape[-1] == 3:
            theta = poses[:, 0]
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)
            zeros = jnp.zeros_like(theta)
            ones = jnp.ones_like(theta)
            return jnp.stack(
                [
                    jnp.stack([cos_theta, -sin_theta, zeros], axis=1),
                    jnp.stack([sin_theta, cos_theta, zeros], axis=1),
                    jnp.stack([zeros, zeros, ones], axis=1),
                ],
                axis=1,
            )
        raise ValueError(
            "Unsupported pose format for material-frame extraction: expected "
            "SE(2) (N, 3) or (N, 3, 3), or SE(3) (N, 4, 4), "
            f"got shape {poses.shape}"
        )

    @abstractmethod
    def render_frame(self, q: Array) -> np.ndarray:
        """Render single configuration to RGB image array.

        Args:
            q: Robot configuration array

        Returns:
            RGB image as numpy array of shape (height, width, 3), dtype uint8
        """
        pass

    @abstractmethod
    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        playback_speed: float = 1.0,
        record_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Render animated sequence to video file.

        Default implementation uses animate_cv2. Subclasses can override
        for native playback (e.g., Open3D interactive viewer).

        Args:
            ts: Time stamps array of shape (T,)
            q_ts: Configurations array of shape (T, DOF)
            playback_speed: Speed up factor for the video (playback)
            record_path: Path to save video file (required for default impl)
        """
        pass

    @abstractmethod
    def show(self, q: Array, **kwargs) -> None:
        """Display single frame interactively (blocking).

        Default implementation uses matplotlib. Subclasses can override
        for native viewers (e.g., Open3D window).

        Args:
            q: Robot configuration array
        """
        pass

    # =========================================================================
    # Multi-robot (batched) rendering helpers
    # =========================================================================

    def _compute_grid_offsets(
        self, num_robots: int, grid_spacing: tuple[float, float]
    ) -> Array:
        """Compute grid layout offsets for N robots.

        Robots are arranged in a grid pattern centered around the origin.
        Grid dimensions are computed automatically: cols = ceil(sqrt(N)).

        Args:
            num_robots: Number of robots to arrange
            grid_spacing: (x, y) spacing between robot bases in meters

        Returns:
            Array of shape (N, 2) for 2D or (N, 3) for 3D with z=0
        """
        cols = int(np.ceil(np.sqrt(num_robots)))
        rows = int(np.ceil(num_robots / cols))
        offsets = []
        for i in range(num_robots):
            row, col = divmod(i, cols)
            # Center the grid around origin
            x = (col - (cols - 1) / 2) * grid_spacing[0]
            y = (row - (rows - 1) / 2) * grid_spacing[1]
            if self.is_3d:
                offsets.append([x, y, 0.0])
            else:
                offsets.append([x, y])
        return jnp.array(offsets)

    @staticmethod
    def _split_counts_by_lengths(num_points: int, lengths: Array) -> list[int]:
        """Split num_points across segments proportionally to lengths."""
        L = np.asarray(lengths, dtype=np.float64).reshape(-1)
        S = int(L.size)
        if S == 0:
            return [num_points]
        if num_points < S:
            base = [1] * S
            base[-1] += num_points - S
            return base

        Ltot = float(L.sum())
        if Ltot <= 0:
            base = [num_points // S] * S
            rem = num_points - sum(base)
            for i in range(rem):
                base[i] += 1
            return base

        raw = [num_points * (float(li) / Ltot) for li in L]
        base = [max(1, int(round(x))) for x in raw]
        diff = num_points - sum(base)
        if diff != 0:
            fracs = np.array([x - np.floor(x) for x in raw])
            order = np.argsort(fracs)[::-1] if diff > 0 else np.argsort(fracs)
            idxs = order.tolist()
            k = abs(diff)
            j = 0
            while k > 0 and idxs:
                i = idxs[j % len(idxs)]
                if diff > 0:
                    base[i] += 1
                else:
                    if base[i] > 1:
                        base[i] -= 1
                    else:
                        j += 1
                        continue
                k -= 1
                j += 1
        return base

    def _segment_bounds(self, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute segment start/end indices for num_points."""
        if num_points in self._segment_cache:
            return self._segment_cache[num_points]
        lengths = np.asarray(self.robot.segment_length, dtype=np.float64).reshape(-1)
        counts = self._split_counts_by_lengths(num_points, lengths)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        self._segment_cache[num_points] = (starts, ends)
        return starts, ends

    @staticmethod
    def _segment_colors_from_points(
        point_colors: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        """Aggregate per-point colors into per-segment colors."""
        seg_colors = []
        for c0, c1 in zip(starts, ends):
            if c1 <= c0 or c0 >= point_colors.shape[0]:
                seg_colors.append(point_colors[min(c0, point_colors.shape[0] - 1)])
            else:
                seg_colors.append(np.mean(point_colors[c0:c1], axis=0))
        return np.asarray(seg_colors, dtype=np.float64)

    @staticmethod
    def _expand_segment_colors(
        segment_colors: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """Expand per-segment colors into per-point colors."""
        out = np.zeros((num_points, 4), dtype=np.float64)
        for idx, (c0, c1) in enumerate(zip(starts, ends)):
            if c1 <= c0 or c0 >= num_points:
                continue
            out[c0:c1] = segment_colors[idx]
        return out

    def resolve_backbone_colors(
        self,
        num_robots: int,
        *,
        color_config: RendererColorConfig | None = None,
        cache: bool = True,
    ) -> ResolvedBackboneColors:
        """Resolve backbone colors using the shared config hierarchy."""
        cfg = color_config or self.color_config
        cache_key = (id(cfg), int(num_robots), int(self.num_points))
        if cache and cache_key in self._color_cache:
            return self._color_cache[cache_key]

        starts, ends = self._segment_bounds(self.num_points)
        num_segments = int(starts.shape[0])
        backbone = cfg.backbone

        segment_palette = backbone.segment_palette
        if (
            segment_palette == DEFAULT_SEGMENT_PALETTE
            and backbone.segment_colors is None
            and backbone.robot_segment_colors is None
            and backbone.point_colors is None
            and backbone.point_palette is None
            and backbone.robot_point_colors is None
            and (backbone.robot_colors is not None or num_robots > 1)
        ):
            segment_palette = None

        robot_colors_rgba, robot_has_alpha = normalize_palette(
            backbone.robot_colors,
            num_robots,
            default_palette=backbone.robot_palette or DEFAULT_ROBOT_PALETTE,
            name="robot_colors",
        )

        segment_colors_rgba, segment_has_alpha = normalize_color_array(
            backbone.segment_colors,
            (num_segments,),
            name="segment_colors",
        )
        if segment_colors_rgba is None and segment_palette is not None:
            segment_colors_rgba, segment_has_alpha = normalize_palette(
                segment_palette,
                num_segments,
                default_palette=DEFAULT_SEGMENT_PALETTE,
                name="segment_palette",
            )

        point_colors_rgba, point_has_alpha = normalize_color_array(
            backbone.point_colors,
            (self.num_points,),
            name="point_colors",
        )
        if point_colors_rgba is None and backbone.point_palette is not None:
            point_colors_rgba, point_has_alpha = normalize_palette(
                backbone.point_palette,
                self.num_points,
                default_palette=DEFAULT_SEGMENT_PALETTE,
                name="point_palette",
            )

        robot_segment_rgba, robot_segment_has_alpha = normalize_color_array(
            backbone.robot_segment_colors,
            (num_robots, num_segments),
            name="robot_segment_colors",
        )
        robot_point_rgba, robot_point_has_alpha = normalize_color_array(
            backbone.robot_point_colors,
            (num_robots, self.num_points),
            name="robot_point_colors",
        )

        robot_alpha = robot_colors_rgba[:, 3] if robot_has_alpha else None

        if robot_point_rgba is not None:
            per_robot_point = robot_point_rgba
            if not robot_point_has_alpha and robot_alpha is not None:
                per_robot_point = per_robot_point.copy()
                per_robot_point[..., 3] = robot_alpha[:, None]
            legend_level = "point"
        elif point_colors_rgba is not None:
            per_robot_point = np.tile(point_colors_rgba[None, ...], (num_robots, 1, 1))
            if not point_has_alpha and robot_alpha is not None:
                per_robot_point = per_robot_point.copy()
                per_robot_point[..., 3] = robot_alpha[:, None]
            legend_level = "point"
        elif robot_segment_rgba is not None:
            per_robot_point = np.zeros(
                (num_robots, self.num_points, 4), dtype=np.float64
            )
            for r in range(num_robots):
                per_robot_point[r] = self._expand_segment_colors(
                    robot_segment_rgba[r], starts, ends, self.num_points
                )
            if not robot_segment_has_alpha and robot_alpha is not None:
                per_robot_point[..., 3] = robot_alpha[:, None]
            legend_level = "segment"
        elif segment_colors_rgba is not None:
            per_robot_point = np.zeros(
                (num_robots, self.num_points, 4), dtype=np.float64
            )
            expanded = self._expand_segment_colors(
                segment_colors_rgba, starts, ends, self.num_points
            )
            per_robot_point[:] = expanded[None, ...]
            if not segment_has_alpha and robot_alpha is not None:
                per_robot_point[..., 3] = robot_alpha[:, None]
            legend_level = "segment"
        else:
            per_robot_point = np.tile(
                robot_colors_rgba[:, None, :], (1, self.num_points, 1)
            )
            legend_level = "robot"

        if robot_segment_rgba is not None:
            per_robot_segment = robot_segment_rgba
            if not robot_segment_has_alpha and robot_alpha is not None:
                per_robot_segment = per_robot_segment.copy()
                per_robot_segment[..., 3] = robot_alpha[:, None]
        else:
            per_robot_segment = np.zeros(
                (num_robots, num_segments, 4), dtype=np.float64
            )
            for r in range(num_robots):
                per_robot_segment[r] = self._segment_colors_from_points(
                    per_robot_point[r], starts, ends
                )

        legend_segment = per_robot_segment[0]
        legend_point = per_robot_point[0]

        resolved = ResolvedBackboneColors(
            per_robot_point_rgba=per_robot_point,
            per_robot_segment_rgba=per_robot_segment,
            robot_colors_rgba=robot_colors_rgba,
            segment_colors_rgba=legend_segment,
            point_colors_rgba=legend_point,
            legend_level=legend_level,
        )

        if cache:
            self._color_cache[cache_key] = resolved
        return resolved

    def get_color_legend(
        self,
        *,
        num_robots: int = 1,
        color_config: RendererColorConfig | None = None,
    ) -> ColorLegend:
        """Return a lightweight color legend for the current configuration."""
        resolved = self.resolve_backbone_colors(
            num_robots, color_config=color_config, cache=False
        )

        if resolved.legend_level == "point":
            colors = resolved.point_colors_rgba
            labels = [f"point_{i}" for i in range(colors.shape[0])]
        elif resolved.legend_level == "segment":
            colors = resolved.segment_colors_rgba
            labels = [f"segment_{i}" for i in range(colors.shape[0])]
        else:
            colors = resolved.robot_colors_rgba
            labels = [f"robot_{i}" for i in range(colors.shape[0])]

        return ColorLegend(
            level=resolved.legend_level, colors_rgba=colors, labels=labels
        )

    def clear_color_cache(self) -> None:
        """Clear cached color resolutions."""
        self._color_cache.clear()

    def compute_backbone_curves_batched(
        self, q_batch: Array, base_offsets: Array
    ) -> Array:
        """Compute backbone curves for multiple robots with base offsets.

        Vectorizes :meth:`compute_backbone_curve` over configurations. Each
        curve uses the robot's public abscissa-batched kinematics method.

        Args:
            q_batch: Configurations of shape (N, DOF) - batch-first
            base_offsets: Base position offsets of shape (N, 2) or (N, 3)

        Returns:
            Array of shape (N, num_points, dim) - batch-first
        """
        q_batch = jnp.asarray(q_batch)
        if q_batch.ndim != 2:
            raise ValueError(
                f"q_batch must have shape (N, DOF), got {q_batch.shape} with ndim={q_batch.ndim}"
            )

        # vmap over the configurations
        curves = jax.vmap(self.compute_backbone_curve)(q_batch)  # (N, num_points, dim)
        offsets = self._normalize_base_offsets(
            base_offsets,
            num_robots=int(q_batch.shape[0]),
            target_dim=int(curves.shape[-1]),
        )

        # Add base offsets: (N, num_points, dim) + (N, 1, dim)
        return curves + offsets[:, None, :]

    def compute_backbone_curves_and_frames_batched(
        self, q_batch: Array, base_offsets: Array
    ) -> tuple[Array, Array]:
        """Compute 3D backbone positions and material frames for multiple robots.

        The frame columns are the local material-frame axes in world coordinates.
        Positional base offsets translate the curves without rotating the frames.
        """
        q_batch = jnp.asarray(q_batch)
        if q_batch.ndim != 2:
            raise ValueError(
                f"q_batch must have shape (N, DOF), got {q_batch.shape} with ndim={q_batch.ndim}"
            )

        poses = jax.vmap(self.compute_backbone_poses)(q_batch)
        curves = jax.vmap(self._extract_positions_3d)(poses)
        frames = jax.vmap(self._extract_material_frames_3d)(poses)
        offsets = self._normalize_base_offsets(
            base_offsets,
            num_robots=int(q_batch.shape[0]),
            target_dim=3,
        )
        return curves + offsets[:, None, :], frames
