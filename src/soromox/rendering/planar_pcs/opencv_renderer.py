"""OpenCV-based renderer for Planar PCS robots.

This is a specialized renderer for PlanarPCS robots with segment-aware
backbone thickness based on robot diameter.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import jax.numpy as jnp
from jax import Array
import numpy as np

from soromox.rendering.opencv_base import BaseOpenCVRenderer
from soromox.systems.planar_pcs import PlanarPCS


class OpenCVPlanarPCSRenderer(BaseOpenCVRenderer):
    """OpenCV visualization for Planar PCS robots.

    Renders the backbone with per-segment thickness based on robot radius.

    Example:
        >>> renderer = OpenCVPlanarPCSRenderer(robot)
        >>> img = renderer.render_frame(q)
        >>> renderer.show(q)
    """

    def __init__(
        self,
        robot: PlanarPCS,
        width: int = 700,
        height: int = 700,
        num_points: int = 50,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        base_color: Tuple[int, int, int] = (0, 255, 0),
        backbone_color: Tuple[int, int, int] = (0, 0, 0),
        backbone_thickness: Optional[int] = None,
        length_scale: float = 2.0,
        origin_uv: Optional[Tuple[int, int]] = None,
    ):
        """Initialize OpenCV renderer for Planar PCS.

        Args:
            robot: PlanarPCS robot instance
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for curve discretization
            background_color: RGB background color (0-1 range)
            base_color: BGR color for base circle
            backbone_color: BGR color for backbone
            backbone_thickness: Line thickness for backbone (None = auto from radius)
            length_scale: Scale factor for robot in image (robot occupies height/length_scale)
            origin_uv: Pixel coordinates of robot base (None = center of image)
        """
        super().__init__(robot, width, height, num_points, background_color)

        self.base_color = base_color
        self.backbone_color = backbone_color
        self.backbone_thickness = backbone_thickness
        self.length_scale = length_scale
        self.origin_uv = origin_uv

    @property
    def is_3d(self) -> bool:
        """Planar PCS is 2D."""
        return False

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xy from SE(2) poses [theta, x, y]."""
        return poses[:, 1:3]

    def render_frame(self, q: Array) -> np.ndarray:
        """Render single configuration to BGR image array.

        Args:
            q: Robot configuration array of shape (DOF,) for a single robot.

        Returns:
            img (np.ndarray): BGR image of shape (height, width, 3), dtype uint8.
        """
        robot = self.robot
        h, w = self.height, self.width

        # Pixel per meter
        ppm = h / (self.length_scale * jnp.sum(robot.L))

        # Compute backbone thicknesses per segment
        if self.backbone_thickness is None:
            backbone_thicknesses = jnp.clip(
                (ppm * robot.r).astype(jnp.int32), min=1, max=None
            )
        else:
            backbone_thicknesses = (
                jnp.ones((robot.num_segments,), dtype=jnp.int32) * self.backbone_thickness
            )

        # Robot origin in pixel coordinates
        if self.origin_uv is None:
            origin_uv = np.array([w // 2, h // 2], dtype=np.int32)
        else:
            origin_uv = np.array(self.origin_uv, dtype=np.int32)

        # Arc-length points
        s_ps = jnp.linspace(0, self.L_max, self.num_points)

        # Get poses using cached batched FK
        chi_ps = self._batched_fk(q, s_ps)  # (num_points, 3)

        # Initialize background
        bg_uint8 = tuple(int(c * 255) for c in self.background_color[::-1])  # RGB to BGR
        img = np.full((h, w, 3), bg_uint8, dtype=np.uint8)

        # Draw base circle
        base_radius = int(1.5 * robot.r.mean().item() * ppm)
        cv2.circle(img, tuple(origin_uv), base_radius, self.base_color, -1)

        # Transform poses to pixel coordinates (N, 2)
        curve = np.array((chi_ps[:, 1:3] * ppm), dtype=np.int32)
        curve[:, 1] = -curve[:, 1]  # Invert y for image coordinates

        # Draw each segment with its own thickness
        L_cum = jnp.concatenate((jnp.array([0]), jnp.cumsum(robot.L)))
        for segment_idx in range(robot.num_segments):
            curve_selector = (s_ps >= L_cum[segment_idx]) & (
                s_ps < L_cum[segment_idx + 1]
            )
            segment_curve = origin_uv + curve[curve_selector]
            if len(segment_curve) > 1:
                cv2.polylines(
                    img,
                    [segment_curve],
                    isClosed=False,
                    color=self.backbone_color,
                    thickness=int(backbone_thicknesses[segment_idx].item()),
                )

        return img

    def show(self, q: Array) -> None:
        """Display single frame in OpenCV window.

        Args:
            q: Robot configuration array of shape (DOF,).

        Returns:
            None
        """
        img = self.render_frame(q)
        win = "Planar PCS"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)
