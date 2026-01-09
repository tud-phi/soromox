"""Camera configuration for robot visualization.

Provides a unified camera configuration dataclass that can be used
across different rendering backends (Matplotlib, Open3D, Viser).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    """Configuration for camera positioning and view parameters.

    This config provides unified camera settings that work across different
    rendering backends. Settings are interpreted appropriately by each backend.

    The camera can be positioned either:
    1. Automatically based on scene content (position=None, look_at=None)
    2. Explicitly by providing position and look_at coordinates

    For automatic positioning, the camera is placed to view the entire scene
    using the distance_factor and position_offset parameters.

    Attributes:
        fov: Field of view in degrees (used by 3D renderers)
        position: Explicit camera position as (x, y, z) or None for auto
        look_at: Point the camera looks at as (x, y, z) or None for auto (scene center)
        up: Camera up vector as (x, y, z), default is Z-up convention
        distance_factor: Multiplier for scene extent to compute camera distance
            (only used when position is None)
        position_offset: Direction vector for camera placement relative to center,
            as (x_factor, y_factor, z_factor) multiplied by computed distance
            (only used when position is None)

    Example:
        >>> # Default camera with Z-up, auto-positioned
        >>> cam = CameraConfig()

        >>> # Custom FOV with explicit position
        >>> cam = CameraConfig(
        ...     fov=60.0,
        ...     position=(0.5, -0.5, 0.3),
        ...     look_at=(0.0, 0.0, 0.1),
        ... )

        >>> # Auto-positioned but closer to scene
        >>> cam = CameraConfig(distance_factor=5.0)
    """

    fov: float = 75.0
    position: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)  # Z-up convention
    distance_factor: float = 10.0
    position_offset: tuple[float, float, float] = (0.8, -0.8, 0.5)

    def compute_auto_position(
        self,
        center: np.ndarray,
        max_extent: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute camera position and look_at from scene bounds.

        This method computes automatic camera placement based on the scene's
        center and extent, using the distance_factor and position_offset.

        Args:
            center: Scene center point as (3,) array
            max_extent: Maximum extent (size) of the scene

        Returns:
            Tuple of (camera_position, look_at_point) as numpy arrays
        """
        if self.position is not None:
            camera_pos = np.array(self.position)
        else:
            distance = max_extent * self.distance_factor
            offset = np.array(self.position_offset) * distance
            camera_pos = center + offset

        if self.look_at is not None:
            look_at = np.array(self.look_at)
        else:
            look_at = center.copy()

        return camera_pos, look_at
