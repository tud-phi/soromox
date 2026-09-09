"""OpenCV-based renderer for Planar HSA robots.

This is a specialized renderer for PlanarHSA robots with custom drawing
for virtual backbone, rods, and platforms.
"""

from __future__ import annotations

import cv2
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap

from soromox.rendering.config import RendererConfig
from soromox.rendering.config.camera import CameraConfig
from soromox.rendering.config.colors import RendererColorConfig
from soromox.rendering.opencv_base import BaseOpenCVRenderer
from soromox.systems import PlanarHSA


class OpenCVPlanarHSARenderer(BaseOpenCVRenderer):
    """OpenCV visualization for Planar HSA robots.

    Renders the virtual backbone, rods, and platforms with custom colors.
    This is a specialized renderer that uses HSA-specific forward kinematics.

    Example:
        ```python
        renderer = OpenCVPlanarHSARenderer(robot)
        image = renderer.render_frame(q)
        renderer.show(q)
        ```
    """

    def __init__(
        self,
        robot: PlanarHSA,
        config: RendererConfig | None = None,
        rod_thickness: int = 10,
    ):
        """Initialize OpenCV renderer for Planar HSA.

        Args:
            config: Shared robot color, geometry and output defaults. Scene
                appearance is ignored; the canvas is white.
            robot: PlanarHSA robot instance
            rod_thickness: Line thickness for rods
        """
        super().__init__(robot, config=config)
        base_color = tuple(
            int(x * 255) for x in self.config.colors.base_plate_color[::-1]
        )
        backbone_color = tuple(
            int(x * 255)
            for x in self.resolve_backbone_colors(1).per_robot_point_rgba[0, 0, :3][
                ::-1
            ]
        )
        rod_color = tuple(
            int(x * 255)
            for x in (
                self.color_config.robot_override
                or self.color_config.actuators.color_for_kind("rod")
            )[:3][::-1]
        )
        platform_color = tuple(
            int(x * 255)
            for x in (
                self.color_config.robot_override
                or self.color_config.actuators.color_for_kind("platform")
            )[:3][::-1]
        )
        backbone_thickness = self.config.geometry.line_width

        self.base_color = base_color
        self.backbone_color = backbone_color
        self.rod_color = rod_color
        self.platform_color = platform_color
        self.backbone_thickness = max(1, round(backbone_thickness or 4))
        self.rod_thickness = rod_thickness

        # Cache HSA-specific FK functions
        self._batched_fk_backbone = vmap(
            robot.forward_kinematics,
            in_axes=(None, 0),
            out_axes=-1,
        )
        self._batched_fk_rod = vmap(
            robot.forward_kinematics_rod,
            in_axes=(None, 0, None),
            out_axes=-1,
        )
        self._batched_fk_platform = vmap(
            robot.forward_kinematics_platform,
            in_axes=(None, 0),
            out_axes=0,
        )

    @property
    def is_3d(self) -> bool:
        """Planar HSA is 2D."""
        return False

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xy from SE(2) poses [theta, x, y]."""
        return poses[:, 1:3]

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> np.ndarray:
        """Render single configuration to BGR image array.

        Args:
            color_config: Complete sRGB robot color override.
            camera_config: Camera override; the planar pixel projection is unchanged.
            q: Robot configuration array
            base_offsets: Optional positional offset with shape ``(2,)`` or
                ``(3,)``. The z component is ignored for this planar renderer.

        Returns:
            BGR image as numpy array of shape (height, width, 3), dtype uint8
        """
        self._warn_simple_appearance(getattr(self, "_rendering_mode", "static"))
        cfg = color_config or self.color_config
        base_color = tuple(int(c * 255) for c in cfg.base_plate_color[:3][::-1])
        backbone_color = tuple(
            int(c * 255)
            for c in self.resolve_backbone_colors(
                1, color_config=cfg
            ).per_robot_point_rgba[0, 0, :3][::-1]
        )
        rod_color = tuple(
            int(c * 255)
            for c in (cfg.robot_override or cfg.actuators.color_for_kind("rod"))[:3][
                ::-1
            ]
        )
        platform_color = tuple(
            int(c * 255)
            for c in (cfg.robot_override or cfg.actuators.color_for_kind("platform"))[
                :3
            ][::-1]
        )
        robot = self.robot
        h, w = self.height, self.width

        # Pixel per meter
        ppm = h / (
            2.0 * jnp.sum(robot.proximal_cap_length + robot.L + robot.distal_cap_length)
        )

        # Arc-length points
        s_ps = jnp.linspace(0, robot.length, self.num_points)

        # Get poses
        chiv_ps = self._batched_fk_backbone(q, s_ps)  # virtual backbone
        chiL_ps = self._batched_fk_rod(q, s_ps, 0)  # left rod
        chiR_ps = self._batched_fk_rod(q, s_ps, 1)  # right rod
        chip_ps = self._batched_fk_platform(q, jnp.arange(0, robot.num_segments))

        offset = self._single_base_offset(base_offsets, target_dim=2)
        if offset is not None:
            offset_jax = jnp.asarray(offset, dtype=chiv_ps.dtype)
            chiv_ps = chiv_ps.at[1:3, :].add(offset_jax[:, None])
            chiL_ps = chiL_ps.at[1:3, :].add(offset_jax[:, None])
            chiR_ps = chiR_ps.at[1:3, :].add(offset_jax[:, None])
            chip_ps = chip_ps.at[:, 1:3].add(offset_jax[None, :])

        # Initialize white background
        img = self._blank_frame()

        # World origin in pixel coordinates
        uv_robot_origin = np.array([w // 2, int(h * 0.9)], dtype=np.int32)
        uv_robot_origin_jax = jnp.array(uv_robot_origin)

        @jit
        def chi2u(chi: Array) -> Array:
            """Map SE(2) pose to pixel coordinates."""
            uv_off = jnp.array((chi[1:] * ppm), dtype=jnp.int32)
            uv_off = uv_off.at[1].set(-uv_off[1])  # invert y
            return uv_robot_origin_jax + uv_off

        batched_chi2u = vmap(chi2u, in_axes=-1, out_axes=0)

        base_xy = self._base_position(dim=2)
        if offset is not None:
            base_xy = base_xy + np.asarray(offset)
        base_uv = np.asarray(chi2u(jnp.array([0.0, base_xy[0], base_xy[1]])))

        cv2.circle(img, tuple(base_uv), 5, base_color, -1)

        # Add proximal and distal cap points to backbone
        chiv_ps = jnp.concatenate(
            [
                (
                    chiv_ps[:, 0] - jnp.array([0.0, 0.0, robot.proximal_cap_length[0]])
                ).reshape(3, 1),
                chiv_ps,
                (
                    chiv_ps[:, -1]
                    + jnp.array(
                        [
                            chiv_ps[0, -1],
                            -jnp.sin(chiv_ps[0, -1]) * robot.distal_cap_length[-1],
                            jnp.cos(chiv_ps[0, -1]) * robot.distal_cap_length[-1],
                        ]
                    )
                ).reshape(3, 1),
            ],
            axis=1,
        )
        curve_backbone = np.array(batched_chi2u(chiv_ps))
        cv2.polylines(
            img,
            [curve_backbone],
            isClosed=False,
            color=backbone_color,
            thickness=self.backbone_thickness,
        )

        # Add cap points to left rod
        chiL_ps = jnp.concatenate(
            [
                (
                    chiL_ps[:, 0] - jnp.array([0.0, 0.0, robot.proximal_cap_length[0]])
                ).reshape(3, 1),
                chiL_ps,
                (
                    chiL_ps[:, -1]
                    + jnp.array(
                        [
                            chiL_ps[0, -1],
                            -jnp.sin(chiL_ps[0, -1]) * robot.distal_cap_length[-1],
                            jnp.cos(chiL_ps[0, -1]) * robot.distal_cap_length[-1],
                        ]
                    )
                ).reshape(3, 1),
            ],
            axis=1,
        )
        curve_rod_left = np.array(batched_chi2u(chiL_ps))
        cv2.polylines(
            img,
            [curve_rod_left],
            isClosed=False,
            color=rod_color,
            thickness=self.rod_thickness,
        )

        # Add cap points to right rod
        chiR_ps = jnp.concatenate(
            [
                (
                    chiR_ps[:, 0] - jnp.array([0.0, 0.0, robot.proximal_cap_length[0]])
                ).reshape(3, 1),
                chiR_ps,
                (
                    chiR_ps[:, -1]
                    + jnp.array(
                        [
                            chiR_ps[0, -1],
                            -jnp.sin(chiR_ps[0, -1]) * robot.distal_cap_length[-1],
                            jnp.cos(chiR_ps[0, -1]) * robot.distal_cap_length[-1],
                        ]
                    )
                ).reshape(3, 1),
            ],
            axis=1,
        )
        curve_rod_right = np.array(batched_chi2u(chiR_ps))
        cv2.polylines(
            img,
            [curve_rod_right],
            isClosed=False,
            color=rod_color,
            thickness=self.rod_thickness,
        )

        # Draw platforms
        for i in range(chip_ps.shape[0]):
            platform_R = jnp.array(
                [
                    [1, 0, 0],
                    [0, jnp.cos(chip_ps[i, 0]), -jnp.sin(chip_ps[i, 0])],
                    [0, jnp.sin(chip_ps[i, 0]), jnp.cos(chip_ps[i, 0])],
                ]
            )
            platform_llc = chip_ps[i, :] + platform_R @ jnp.array(
                [
                    0,
                    -robot.platform_dimension[i, 0] / 2,
                    -robot.platform_dimension[i, 1] / 2,
                ]
            )
            platform_ulc = chip_ps[i, :] + platform_R @ jnp.array(
                [
                    0,
                    -robot.platform_dimension[i, 0] / 2,
                    +robot.platform_dimension[i, 1] / 2,
                ]
            )
            platform_urc = chip_ps[i, :] + platform_R @ jnp.array(
                [
                    0,
                    +robot.platform_dimension[i, 0] / 2,
                    +robot.platform_dimension[i, 1] / 2,
                ]
            )
            platform_lrc = chip_ps[i, :] + platform_R @ jnp.array(
                [
                    0,
                    +robot.platform_dimension[i, 0] / 2,
                    -robot.platform_dimension[i, 1] / 2,
                ]
            )
            platform_curve = jnp.stack(
                [platform_llc, platform_ulc, platform_urc, platform_lrc, platform_llc],
                axis=1,
            )
            cv2.fillPoly(
                img,
                [np.array(batched_chi2u(platform_curve))],
                color=platform_color,
            )

        return img

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Display single frame in OpenCV window.

        Args:
            q: Robot configuration.
            color_config: Complete robot color override.
            camera_config: Camera override, approximated by the planar projection.
            base_offsets: Optional positional offset with shape ``(2,)`` or
                ``(3,)``.
        """
        self._warn_simple_appearance(getattr(self, "_rendering_mode", "static"))
        img = self.render_frame(
            q,
            base_offsets=base_offsets,
            color_config=color_config,
            camera_config=camera_config,
        )
        win = "Planar HSA"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)
