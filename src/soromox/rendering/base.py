"""Base class for robot visualization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.systems.soft_robot import SoftRobot


class BaseSoftRobotRenderer(ABC):
    """Abstract base class for soft robot visualization backends.

    Provides cached forward kinematics and a common interface for rendering
    soft robots across different backends (Matplotlib, Open3D, OpenCV).

    Subclasses must implement:
        - is_3d: Property indicating 2D (planar) or 3D robot
        - _extract_positions: Extract xyz/xy from FK poses
        - render_frame: Render single configuration to RGB image array

    Attributes:
        robot: SoftRobot system object with forward_kinematics method
        width: Image width in pixels
        height: Image height in pixels
        num_points: Number of points for discretizing backbone curve
        background_color: RGB tuple for background color (0-1 range)
        L_max: Total length of the robot backbone
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Initialize the renderer with a robot and visualization parameters.

        Args:
            robot: Robot system with forward_kinematics method and length property
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for backbone curve discretization
            background_color: RGB background color tuple (values 0-1)
        """
        self.robot: SoftRobot = robot
        self.width = width
        self.height = height
        self.num_points = num_points
        self.background_color = background_color

        # Total robot length
        self.L_max = float(jnp.asarray(robot.length))

    @property
    @abstractmethod
    def is_3d(self) -> bool:
        """Return True for 3D robots, False for 2D/planar robots."""
        pass

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

    @abstractmethod
    def _extract_positions(self, poses: Array) -> Array:
        """Extract xyz or xy positions from FK poses.

        Args:
            poses: Forward kinematics output (backend-specific format)

        Returns:
            Position array of shape (N, 3) for 3D or (N, 2) for 2D
        """
        pass

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
        **kwargs,
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
