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
    using the distance_factor and position_offset parameters. Renderers can
    provide the robot base transform so automatic offsets are interpreted in
    the base frame instead of fixed world axes.

    Attributes:
        fov: Field of view in degrees (used by 3D renderers)
        position: Explicit camera position as (x, y, z) or None for auto
        look_at: Point the camera looks at as (x, y, z) or None for auto (scene center)
        up: Camera up vector as (x, y, z), default is world Z-up convention
        distance_factor: Multiplier for scene extent to compute camera distance
            (only used when position is None)
        position_offset: Direction vector for camera placement relative to center,
            as (x_factor, y_factor, z_factor) multiplied by computed distance.
            This is interpreted in the base frame when renderers provide one
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
        reference_transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute camera position and look_at from scene bounds.

        This method computes automatic camera placement based on the scene's
        center and extent, using the distance_factor and position_offset.

        Args:
            center: Scene center point as (3,) array
            max_extent: Maximum extent (size) of the scene
            reference_transform: Optional base transform. When supplied,
                automatic offsets are rotated by the base orientation. Shape
                may be planar homogeneous ``(3, 3)`` or spatial homogeneous
                ``(4, 4)``.

        Returns:
            Tuple of (camera_position, look_at_point) as numpy arrays
        """
        rotation = self._reference_rotation(reference_transform)

        if self.position is not None:
            camera_pos = np.array(self.position)
        else:
            distance = max_extent * self.distance_factor
            offset = np.array(self.position_offset) * distance
            if rotation is not None:
                offset = rotation @ offset
            camera_pos = center + offset

        if self.look_at is not None:
            look_at = np.array(self.look_at)
        else:
            look_at = center.copy()

        return camera_pos, look_at

    def compute_up(self) -> np.ndarray:
        """Compute the camera up vector.

        Returns:
            Up vector as a NumPy array with shape ``(3,)``.
        """
        up = np.asarray(self.up, dtype=np.float64)
        norm = np.linalg.norm(up)
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return up / norm

    @staticmethod
    def _reference_rotation(
        reference_transform: np.ndarray | None,
    ) -> np.ndarray | None:
        """Return a 3D rotation matrix from a planar or spatial base transform."""
        if reference_transform is None:
            return None

        transform = np.asarray(reference_transform, dtype=np.float64)
        if transform.shape == (4, 4):
            return transform[:3, :3]
        if transform.shape == (3, 3):
            rotation = np.eye(3, dtype=np.float64)
            rotation[:2, :2] = transform[:2, :2]
            return rotation
        raise ValueError(
            "reference_transform must have shape (3, 3) or (4, 4), "
            f"got {transform.shape}."
        )
