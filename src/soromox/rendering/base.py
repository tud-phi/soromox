"""Base class for robot visualization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_config: RendererColorConfig | None = None,
    ):
        """Initialize the renderer with a robot and visualization parameters.

        Args:
            robot: Robot system with forward_kinematics method and length property
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for backbone curve discretization
            background_color: RGB background color tuple (values 0-1)
            color_config: Shared renderer color configuration
        """
        self.robot: SoftRobot = robot
        self.width = width
        self.height = height
        self.num_points = num_points
        self.background_color = background_color
        self.color_config = color_config or RendererColorConfig()
        self._is_planar = bool(robot.is_planar)

        self._color_cache: dict[tuple[int, int, int], ResolvedBackboneColors] = {}
        self._segment_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Total robot length
        self.L_max = float(jnp.asarray(robot.length))

        # Cache tendon FK if available
        self._has_tendons = hasattr(robot, "forward_kinematics_tendons")
        if self._has_tendons:
            self._batched_fk_tendons = jax.vmap(
                robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1
            )

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
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        if hasattr(self.robot, "forward_kinematics_batched"):
            poses = self.robot.forward_kinematics_batched(q, s_ps)
        else:
            # Fallback to vmap over single-point FK
            poses = jax.vmap(lambda s: self.robot.forward_kinematics(q, s))(s_ps)

        # extract positions
        curve = self._extract_positions(poses)  # (num_points, dim)

        return curve

    def compute_tendon_curves(self, q: Array) -> Array | None:
        """Compute tendon paths if the robot supports tendon kinematics."""
        if not self._has_tendons:
            return None
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self._batched_fk_tendons(q, s_ps)

    def compute_tendon_curves_batched(
        self, q_batch: Array, base_offsets: Array
    ) -> Array | None:
        """Compute tendon paths for multiple robots with base offsets."""
        if not self._has_tendons:
            return None

        q_batch = jnp.asarray(q_batch)
        base_offsets = jnp.asarray(base_offsets)

        if q_batch.ndim != 2:
            raise ValueError(
                f"q_batch must have shape (N, DOF), got {q_batch.shape} with ndim={q_batch.ndim}"
            )
        if base_offsets.ndim != 2:
            raise ValueError(
                f"base_offsets must have shape (N, dim), got {base_offsets.shape}"
            )
        if base_offsets.shape[0] != q_batch.shape[0]:
            raise ValueError(
                f"base_offsets first dimension ({base_offsets.shape[0]}) must "
                f"match batch size ({q_batch.shape[0]})"
            )

        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        tendon_curves = jax.vmap(lambda q: self._batched_fk_tendons(q, s_ps))(q_batch)

        if base_offsets.shape[1] == tendon_curves.shape[-1]:
            offsets = base_offsets
        elif base_offsets.shape[1] + 1 == tendon_curves.shape[-1]:
            pad = jnp.zeros((base_offsets.shape[0], 1), dtype=tendon_curves.dtype)
            offsets = jnp.concatenate([base_offsets, pad], axis=1)
        else:
            raise ValueError(
                f"base_offsets must have shape (N, {tendon_curves.shape[-1]}) or "
                f"(N, {tendon_curves.shape[-1] - 1}), got {base_offsets.shape}"
            )

        return tendon_curves + offsets[:, None, None, :]

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

        Uses forward_kinematics_batched if available (optimized), otherwise
        falls back to jax.vmap over compute_backbone_curve.

        Args:
            q_batch: Configurations of shape (N, DOF) - batch-first
            base_offsets: Base position offsets of shape (N, 2) or (N, 3)

        Returns:
            Array of shape (N, num_points, dim) - batch-first
        """
        q_batch = jnp.asarray(q_batch)
        base_offsets = jnp.asarray(base_offsets)

        if q_batch.ndim != 2:
            raise ValueError(
                f"q_batch must have shape (N, DOF), got {q_batch.shape} with ndim={q_batch.ndim}"
            )
        if base_offsets.ndim != 2:
            raise ValueError(
                f"base_offsets must have shape (N, dim), got {base_offsets.shape}"
            )
        if base_offsets.shape[0] != q_batch.shape[0]:
            raise ValueError(
                f"base_offsets first dimension ({base_offsets.shape[0]}) must "
                f"match batch size ({q_batch.shape[0]})"
            )

        # vmap over the configurations
        curves = jax.vmap(self.compute_backbone_curve)(q_batch)  # (N, num_points, dim)

        # Match base offset dimensionality to curve dimensionality (2D vs 3D)
        if base_offsets.shape[1] == curves.shape[-1]:
            offsets = base_offsets
        elif base_offsets.shape[1] + 1 == curves.shape[-1]:
            pad = jnp.zeros((base_offsets.shape[0], 1), dtype=curves.dtype)
            offsets = jnp.concatenate([base_offsets, pad], axis=1)
        else:
            raise ValueError(
                f"base_offsets must have shape (N, {curves.shape[-1]}) or "
                f"(N, {curves.shape[-1] - 1}), got {base_offsets.shape}"
            )

        # Add base offsets: (N, num_points, dim) + (N, 1, dim)
        return curves + offsets[:, None, :]
