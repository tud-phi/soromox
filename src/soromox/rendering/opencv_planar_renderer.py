"""OpenCV-based renderer for planar soft robots.

Renders the backbone curve for any planar robot that exposes SE(2)
forward kinematics in the form [theta, x, y].
"""

from __future__ import annotations

import cv2
import numpy as np
from jax import Array

from soromox.rendering.actuators import resolve_actuator_rgba
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import RendererColorConfig
from soromox.rendering.opencv_base import BaseOpenCVRenderer
from soromox.rendering.renderer_config import RendererConfig
from soromox.systems.components import CrossSectionGeometry
from soromox.systems.soft_robot import SoftRobot


class OpenCVPlanarRenderer(BaseOpenCVRenderer):
    """OpenCV visualization for planar soft robots.

    Renders the backbone curve with optional segment-aware thickness when
    per-segment radii are available.

    Example:
        ```python
        renderer = OpenCVPlanarRenderer(robot)
        image = renderer.render_frame(q)
        renderer.show(q)
        ```
    """

    def __init__(
        self,
        robot: SoftRobot,
        config: RendererConfig | None = None,
        length_scale: float = 2.0,
        origin_uv: tuple[int, int] | None = None,
    ):
        """Initialize OpenCV renderer for planar robots.

        Args:
            config: Shared scene, camera, color, geometry and output defaults.
            robot: Planar robot instance
            length_scale: Scale factor for robot in image (robot occupies height/length_scale)
            origin_uv: Pixel coordinates of world origin (None = center of image)
        """
        super().__init__(robot, config=config)
        base_color = tuple(
            int(x * 255) for x in self.config.colors.base_plate_color[::-1]
        )
        backbone_color = (0, 0, 0)
        backbone_thickness = self.config.geometry.line_width
        actuator_color = tuple(
            int(x * 255) for x in self.config.colors.actuators.default_color[::-1]
        )
        actuator_thickness = self.config.geometry.actuator_line_width
        base_radius_scale = self.config.geometry.base_plate_radius_scale

        self.base_color = base_color
        self.backbone_color = backbone_color
        self.backbone_thickness = backbone_thickness
        self.actuator_color = actuator_color
        self.actuator_thickness = int(actuator_thickness)
        self.base_radius_scale = float(base_radius_scale)
        self.length_scale = length_scale
        self.origin_uv = origin_uv

    @property
    def is_3d(self) -> bool:
        """Planar robots are 2D."""
        return False

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xy from SE(2) poses [theta, x, y]."""
        if poses.ndim == 2 and poses.shape[1] == 3:
            return poses[:, 1:3]
        if poses.ndim == 2 and poses.shape[1] == 2:
            return poses
        raise ValueError(
            f"Expected planar poses with shape (N, 3) or (N, 2), got {poses.shape}"
        )

    def _cross_section_span(self, q: Array, s: float) -> float:
        geom_tag, geom_params = self.robot.cross_section_geometry(q, s)
        tag = int(np.asarray(geom_tag).item())
        params = np.asarray(geom_params, dtype=float).reshape(-1)
        if params.size == 0:
            return 0.0
        if tag == CrossSectionGeometry.CIRCULAR:
            return float(params[0])
        if tag == CrossSectionGeometry.RECTANGULAR:
            return float(max(params[0], params[1])) if params.size >= 2 else 0.0
        if tag == CrossSectionGeometry.ELLIPTICAL:
            return float(max(params[0], params[1])) if params.size >= 2 else 0.0
        return 0.0

    def _auto_backbone_thickness(
        self, ppm: float, lengths: np.ndarray | None, q: Array
    ) -> tuple[np.ndarray | None, int]:
        if self.backbone_thickness is not None:
            return None, max(1, int(self.backbone_thickness))

        if lengths is None:
            span = self._cross_section_span(q, 0.0)
            thickness = max(1, int(float(span) * ppm)) if span > 0 else 2
            return None, thickness

        L_cum = np.concatenate(([0.0], np.cumsum(lengths)))
        mids = (L_cum[:-1] + L_cum[1:]) * 0.5
        spans = np.array([self._cross_section_span(q, s) for s in mids], dtype=float)
        thicknesses = np.maximum(1, (spans * ppm).astype(np.int32))
        return thicknesses, 0

    def _base_radius_px(self, ppm: float, q: Array) -> int:
        base_radius_m = self.base_radius_scale * self._cross_section_span(q, 0.0)
        if base_radius_m <= 0.0:
            base_radius_m = 0.02 * self.L_max
        return max(2, int(base_radius_m * ppm))

    def _world_to_pixel(
        self,
        points: np.ndarray,
        *,
        origin_uv: np.ndarray,
        ppm: float,
    ) -> np.ndarray:
        """Map planar world xy coordinates to OpenCV pixel coordinates."""
        pts = np.asarray(points, dtype=float)
        px = (pts * ppm).astype(np.int32)
        px[..., 1] = -px[..., 1]
        return origin_uv + px

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
    ) -> np.ndarray:
        """Render single configuration to BGR image array.

        Args:
            q: Robot configuration array of shape (DOF,) for a single robot.
            base_offsets: Optional positional offset with shape ``(2,)`` or
                ``(3,)``. The z component is ignored for this planar renderer.
            color_config: Complete robot-color override.
            camera_config: Camera override; the planar pixel projection is unchanged.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.

        Returns:
            img (np.ndarray): BGR image of shape (height, width, 3), dtype uint8.
        """
        self._warn_simple_appearance(getattr(self, "_rendering_mode", "static"))
        h, w = self.height, self.width

        # Pixel per meter
        ppm = h / (self.length_scale * self.L_max)

        # World origin in pixel coordinates
        if self.origin_uv is None:
            origin_uv = np.array([w // 2, h // 2], dtype=np.int32)
        else:
            origin_uv = np.array(self.origin_uv, dtype=np.int32)

        # Initialize background
        bg_uint8 = tuple(
            int(c * 255) for c in self.background_color[::-1]
        )  # RGB to BGR
        img = np.full((h, w, 3), bg_uint8, dtype=np.uint8)

        # Compute backbone curve in pixel coordinates (N, 2)
        curve = np.asarray(self.compute_backbone_curve(q), dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(
                f"Expected planar backbone curve of shape (N, 2), got {curve.shape}"
            )
        offset = self._single_base_offset(base_offsets, target_dim=2)
        if offset is not None:
            curve = curve + np.asarray(offset)
        curve_uv = self._world_to_pixel(curve, origin_uv=origin_uv, ppm=ppm)

        cfg = color_config or self.color_config
        colors = self.resolve_backbone_colors(1, color_config=cfg).per_robot_point_rgba[
            0
        ]
        lengths = self.robot.segment_length
        thicknesses, uniform_thickness = self._auto_backbone_thickness(ppm, lengths, q)

        img = self._draw_ground(img, curve, origin_uv, ppm)
        starts, ends = self._segment_bounds(self.num_points)
        for i in range(len(curve_uv) - 1):
            width = uniform_thickness
            if thicknesses is not None:
                segment = min(
                    int(np.searchsorted(ends, i, side="right")), len(thicknesses) - 1
                )
                width = thicknesses[segment]
            cv2.line(
                img,
                tuple(curve_uv[i]),
                tuple(curve_uv[i + 1]),
                tuple(int(c * 255) for c in colors[i, :3][::-1]),
                max(1, int(width)),
            )

        if render_actuators and self._has_actuator_visuals:
            actuator_offset = (
                np.zeros((1, 2), dtype=float)
                if offset is None
                else np.asarray(offset, dtype=float).reshape(1, -1)
            )
            actuator_layers = self.compute_actuator_visual_layers_batched(
                np.asarray(q).reshape(1, -1),
                actuator_offset,
                actuator_inputs=actuator_inputs,
            )
            for layer in actuator_layers:
                actuator_colors = resolve_actuator_rgba(
                    layer,
                    default_color=cfg.actuators.color_for_kind(layer.kind),
                    scalar_colormap=cfg.actuators.scalar_colormap,
                    override_color=cfg.robot_override,
                )
                points = np.asarray(layer.points, dtype=float)
                for actuator_idx in range(points.shape[1]):
                    path_uv = self._world_to_pixel(
                        points[0, actuator_idx, :, :2],
                        origin_uv=origin_uv,
                        ppm=ppm,
                    )
                    if path_uv.shape[0] > 1:
                        cv2.polylines(
                            img,
                            [path_uv],
                            isClosed=False,
                            color=tuple(
                                int(c * 255)
                                for c in actuator_colors[0, actuator_idx, :3][::-1]
                            ),
                            thickness=max(1, self.actuator_thickness),
                        )

        # Draw base marker at the transformed base position after the backbone
        # so it remains visible.
        base_radius = self._base_radius_px(ppm, q)
        cv2.circle(
            img,
            tuple(curve_uv[0]),
            base_radius,
            tuple(int(c * 255) for c in cfg.base_plate_color[::-1]),
            -1,
        )

        return img

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
    ) -> None:
        """Display single frame in OpenCV window.

        Args:
            q: Robot configuration array of shape (DOF,).
            base_offsets: Optional positional offset with shape ``(2,)`` or
                ``(3,)``.
            color_config: Complete robot-color override.
            camera_config: Camera override; the planar pixel projection is unchanged.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
        """
        self._warn_simple_appearance(getattr(self, "_rendering_mode", "static"))
        img = self.render_frame(
            q,
            base_offsets=base_offsets,
            color_config=color_config,
            camera_config=camera_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
        )
        win = "Planar Renderer"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)
