"""OpenCV-based renderer for Planar HSA robots.

This is a specialized renderer for PlanarHSA robots with custom drawing
for virtual backbone, rods, and platforms.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Tuple

import cv2
import jax.numpy as jnp
from jax import Array, jit, vmap
import numpy as np

from soromox.rendering.base import BaseContinuumSoftRobotRenderer
from soromox.systems.planar_hsa import PlanarHSA


class OpenCVPlanarHSARenderer(BaseContinuumSoftRobotRenderer):
    """OpenCV visualization for Planar HSA robots.

    Renders the virtual backbone, rods, and platforms with custom colors.
    This is a specialized renderer that uses HSA-specific forward kinematics.

    Example:
        >>> renderer = OpenCVPlanarHSARenderer(robot)
        >>> img = renderer.render_frame(q)
        >>> renderer.show(q)
    """

    def __init__(
        self,
        robot: PlanarHSA,
        width: int = 700,
        height: int = 700,
        num_points: int = 50,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        base_color: Tuple[int, int, int] = (0, 0, 0),
        backbone_color: Tuple[int, int, int] = (255, 0, 0),
        rod_color: Tuple[int, int, int] = (0, 255, 0),
        platform_color: Tuple[int, int, int] = (0, 0, 255),
        backbone_thickness: int = 5,
        rod_thickness: int = 10,
    ):
        """Initialize OpenCV renderer for Planar HSA.

        Args:
            robot: PlanarHSA robot instance
            width: Image width in pixels
            height: Image height in pixels
            num_points: Number of points for curve discretization
            background_color: RGB background color (0-1 range)
            base_color: BGR color for base rectangle
            backbone_color: BGR color for virtual backbone
            rod_color: BGR color for rods
            platform_color: BGR color for platforms
            backbone_thickness: Line thickness for backbone
            rod_thickness: Line thickness for rods
        """
        super().__init__(robot, width, height, num_points, background_color)

        self.base_color = base_color
        self.backbone_color = backbone_color
        self.rod_color = rod_color
        self.platform_color = platform_color
        self.backbone_thickness = backbone_thickness
        self.rod_thickness = rod_thickness

        # Cache HSA-specific FK functions
        self._batched_fk_backbone = vmap(
            robot.forward_kinematics_virtual_backbone_fn, in_axes=(None, 0), out_axes=-1
        )
        self._batched_fk_rod = vmap(
            robot.forward_kinematics_rod_fn, in_axes=(None, 0, None), out_axes=-1
        )
        self._batched_fk_platform = vmap(
            robot.forward_kinematics_platform_fn, in_axes=(None, 0), out_axes=0
        )

    @property
    def is_3d(self) -> bool:
        """Planar HSA is 2D."""
        return False

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xy from SE(2) poses [theta, x, y]."""
        return poses[:, 1:3]

    def render_frame(self, q: Array) -> np.ndarray:
        """Render single configuration to BGR image array.

        Args:
            q: Robot configuration array

        Returns:
            BGR image as numpy array of shape (height, width, 3), dtype uint8
        """
        robot = self.robot
        h, w = self.height, self.width

        # Pixel per meter
        ppm = h / (2.0 * jnp.sum(robot.lpc + robot.L + robot.ldc))

        # Arc-length points
        s_ps = jnp.linspace(0, robot.Lmax, self.num_points)

        # Get poses
        chiv_ps = self._batched_fk_backbone(q, s_ps)  # virtual backbone
        chiL_ps = self._batched_fk_rod(q, s_ps, 0)  # left rod
        chiR_ps = self._batched_fk_rod(q, s_ps, 1)  # right rod
        chip_ps = self._batched_fk_platform(q, jnp.arange(0, robot.num_segments))

        # Initialize white background
        bg_uint8 = tuple(int(c * 255) for c in self.background_color[::-1])  # RGB to BGR
        img = np.full((h, w, 3), bg_uint8, dtype=np.uint8)

        # Robot origin in pixel coordinates
        uv_robot_origin = np.array([w // 2, int(h * 0.9)], dtype=np.int32)
        uv_robot_origin_jax = jnp.array(uv_robot_origin)

        @jit
        def chi2u(chi: Array) -> Array:
            """Map SE(2) pose to pixel coordinates."""
            uv_off = jnp.array((chi[1:] * ppm), dtype=jnp.int32)
            uv_off = uv_off.at[1].set(-uv_off[1])  # invert y
            return uv_robot_origin_jax + uv_off

        batched_chi2u = vmap(chi2u, in_axes=-1, out_axes=0)

        # Draw base
        cv2.rectangle(
            img, (0, uv_robot_origin[1]), (w, h), color=self.base_color, thickness=-1
        )

        # Add proximal and distal cap points to backbone
        chiv_ps = jnp.concatenate(
            [
                (chiv_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
                chiv_ps,
                (
                    chiv_ps[:, -1]
                    + jnp.array(
                        [
                            chiv_ps[0, -1],
                            -jnp.sin(chiv_ps[0, -1]) * robot.ldc[-1],
                            jnp.cos(chiv_ps[0, -1]) * robot.ldc[-1],
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
            color=self.backbone_color,
            thickness=self.backbone_thickness,
        )

        # Add cap points to left rod
        chiL_ps = jnp.concatenate(
            [
                (chiL_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
                chiL_ps,
                (
                    chiL_ps[:, -1]
                    + jnp.array(
                        [
                            chiL_ps[0, -1],
                            -jnp.sin(chiL_ps[0, -1]) * robot.ldc[-1],
                            jnp.cos(chiL_ps[0, -1]) * robot.ldc[-1],
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
            color=self.rod_color,
            thickness=self.rod_thickness,
        )

        # Add cap points to right rod
        chiR_ps = jnp.concatenate(
            [
                (chiR_ps[:, 0] - jnp.array([0.0, 0.0, robot.lpc[0]])).reshape(3, 1),
                chiR_ps,
                (
                    chiR_ps[:, -1]
                    + jnp.array(
                        [
                            chiR_ps[0, -1],
                            -jnp.sin(chiR_ps[0, -1]) * robot.ldc[-1],
                            jnp.cos(chiR_ps[0, -1]) * robot.ldc[-1],
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
            color=self.rod_color,
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
                [0, -robot.pcudim[i, 0] / 2, -robot.pcudim[i, 1] / 2]
            )
            platform_ulc = chip_ps[i, :] + platform_R @ jnp.array(
                [0, -robot.pcudim[i, 0] / 2, +robot.pcudim[i, 1] / 2]
            )
            platform_urc = chip_ps[i, :] + platform_R @ jnp.array(
                [0, +robot.pcudim[i, 0] / 2, +robot.pcudim[i, 1] / 2]
            )
            platform_lrc = chip_ps[i, :] + platform_R @ jnp.array(
                [0, +robot.pcudim[i, 0] / 2, -robot.pcudim[i, 1] / 2]
            )
            platform_curve = jnp.stack(
                [platform_llc, platform_ulc, platform_urc, platform_lrc, platform_llc],
                axis=1,
            )
            cv2.fillPoly(
                img,
                [np.array(batched_chi2u(platform_curve))],
                color=self.platform_color,
            )

        return img

    def show(self, q: Array) -> None:
        """Display single frame in OpenCV window.

        Args:
            q: Robot configuration
        """
        img = self.render_frame(q)
        win = "Planar HSA"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(win)

    def render_sequence(
        self,
        t_list: Array,
        q_list: Array,
        playback_speed: float = 1.0,
        record_path: str = None,
    ) -> None:
        """Render animated sequence to video file.

        Args:
            t_list: Time stamps (T,)
            q_list: Configurations (T, DOF)
            playback_speed: Playback speed multiplier (>1 = faster)
            record_path: Path to save video file
        """
        if record_path is None:
            raise ValueError("record_path is required for render_sequence")

        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)

        t_list = np.array(t_list)
        q_list = np.array(q_list)

        # Compute fps from timestamps
        video_dt = np.mean(np.diff(t_list))
        actual_fps = float(playback_speed) * (1.0 / video_dt)

        print(f"Rendering video with dt={video_dt:.4f} and {len(t_list)} frames")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            str(record_path),
            fourcc,
            actual_fps,
            (self.width, self.height),
        )

        for time_idx in range(len(t_list)):
            img = self.render_frame(q_list[time_idx])
            video.write(img)

        video.release()
        print(f"Video saved to: {record_path}")


# Backward compatibility aliases
def draw_robot(
    robot: PlanarHSA,
    q: Array,
    width: int = 700,
    height: int = 700,
    num_points: int = 50,
    show: bool = False,
) -> np.ndarray:
    """Legacy function - use OpenCVPlanarHSARenderer instead."""
    renderer = OpenCVPlanarHSARenderer(robot, width, height, num_points)
    img = renderer.render_frame(q)
    if show:
        renderer.show(q)
    return img


def animate_robot(
    robot: PlanarHSA,
    filepath: PathLike,
    video_ts: Array,
    q_ts: Array,
    video_width: int = 700,
    video_height: int = 700,
) -> None:
    """Legacy function - use OpenCVPlanarHSARenderer instead."""
    renderer = OpenCVPlanarHSARenderer(robot, video_width, video_height)
    renderer.render_sequence(video_ts, q_ts, record_path=str(filepath))
