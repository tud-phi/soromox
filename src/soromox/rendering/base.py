"""Base class for robot visualization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class BaseContinuumSoftRobotRenderer(ABC):
    """Abstract base class for continuum soft robot visualization backends.

    Provides cached forward kinematics and a common interface for rendering
    continuum soft robots across different backends (Matplotlib, Open3D, OpenCV).

    Subclasses must implement:
        - is_3d: Property indicating 2D (planar) or 3D robot
        - _extract_positions: Extract xyz/xy from FK poses
        - render_frame: Render single configuration to RGB image array

    Attributes:
        robot: The robot system object with forward_kinematics method
        width: Image width in pixels
        height: Image height in pixels
        num_points: Number of points for discretizing backbone curve
        background_color: RGB tuple for background color (0-1 range)
        L_max: Total length of the robot backbone
    """

    def __init__(
        self,
        robot,
        width: int = 800,
        height: int = 600,
        num_points: int = 50,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Initialize the renderer with a robot and visualization parameters.

        Args:
            robot: Robot system with forward_kinematics method and L attribute
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for backbone curve discretization
            background_color: RGB background color tuple (values 0-1)
        """
        self.robot = robot
        self.width = width
        self.height = height
        self.num_points = num_points
        self.background_color = background_color

        # Cache forward kinematics (batched over arc-length)
        self._batched_fk = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))

        # Handle different length attribute names (L for PCS, V_L for GVS)
        if hasattr(robot, "L"):
            self.L_max = float(jnp.sum(robot.L))
        elif hasattr(robot, "V_L"):
            self.L_max = float(jnp.sum(robot.V_L))
        else:
            raise AttributeError("Robot must have 'L' or 'V_L' attribute for lengths")

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
        poses = self._batched_fk(q, s_ps)
        return self._extract_positions(poses)

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

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        fps: int = 25,
        output_path: Optional[str] = None,
    ) -> None:
        """Render animated sequence to video file.

        Default implementation uses animate_cv2. Subclasses can override
        for native playback (e.g., Open3D interactive viewer).

        Args:
            ts: Time stamps array of shape (T,)
            q_ts: Configurations array of shape (T, DOF)
            fps: Frames per second for output video
            output_path: Path to save video file (required for default impl)
        """
        from soromox.rendering.animation import animate_cv2

        if output_path is None:
            raise ValueError("output_path is required for render_sequence")

        animate_cv2(
            rendering_fn=self.render_frame,
            t_ts=np.array(ts),
            q_ts=np.array(q_ts),
            filepath=output_path,
            width=self.width,
            height=self.height,
        )

    def show(self, q: Array) -> None:
        """Display single frame interactively (blocking).

        Default implementation uses matplotlib. Subclasses can override
        for native viewers (e.g., Open3D window).

        Args:
            q: Robot configuration array
        """
        import matplotlib.pyplot as plt

        img = self.render_frame(q)
        plt.figure(figsize=(self.width / 100, self.height / 100))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
