"""Shared OpenCV rendering utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from jax import Array

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.config.camera import CameraConfig
from soromox.rendering.config.colors import RendererColorConfig
from soromox.rendering.config.output import VideoEncodingConfig
from soromox.rendering.video_encoding import FFmpegVideoWriter


class BaseOpenCVRenderer(BaseSoftRobotRenderer):
    """Base class adding video recording for OpenCV renderers."""

    def _writer_label(self) -> str:
        return f"[{self.__class__.__name__}]"

    def _draw_ground(self, image, curve, origin_uv, ppm):
        """Draw the configured planar ground reference onto a BGR image.

        Args:
            image: BGR uint8 image with shape (height, width, 3).
            curve: Backbone positions with shape (points, 2), in metres.
            origin_uv: Pixel coordinates of the world origin.
            ppm: Projection scale in pixels per metre.

        Returns:
            BGR uint8 image with the ground reference and configured opacity.
        """
        self._fit_scene_bounds(curve)
        ground = self.config.scene.ground
        if not ground.visible or not (ground.surface or ground.grid):
            return image
        for center, normal, size in self._resolve_ground_planes(
            np.asarray(curve)[None]
        ):
            tangent = np.array([-normal[1], normal[0]])
            length = np.linalg.norm(tangent)
            if length < 1e-9:
                continue
            points = center[:2] + np.array([[-0.5], [0.5]]) * size * tangent / length
            pixels = (points * ppm).astype(np.int32)
            pixels[:, 1] *= -1
            pixels += np.asarray(origin_uv, dtype=np.int32)
            overlay = image.copy()
            cv2.line(
                overlay,
                tuple(pixels[0]),
                tuple(pixels[1]),
                tuple(
                    int(c * 255)
                    for c in (
                        ground.color if ground.surface else ground.grid_major_color
                    )[::-1]
                ),
                2,
            )
            image = cv2.addWeighted(
                overlay, ground.opacity, image, 1 - ground.opacity, 0
            )
        return image

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        *,
        record_path: str,
        playback_speed: float = 1.0,
        video_config: VideoEncodingConfig | None = None,
        base_offsets: Array | None = None,
        camera_config: CameraConfig | None = None,
        color_config: RendererColorConfig | None = None,
    ) -> None:
        """Render animated sequence to video file using ffmpeg.

        Falls back to OpenCV VideoWriter if ffmpeg is unavailable.

        Args:
            ts: Time stamps of shape (T,)
            q_ts: Configurations of shape (T, DOF)
            record_path: Path to save video file
            playback_speed: Playback speed multiplier (>0)
            video_config: Complete video encoding override.
            base_offsets: Optional positional offset for every exported frame.
            camera_config: Camera override, approximated by the planar projection.
            color_config: Complete sRGB robot color override.
        """
        if record_path is None:
            raise ValueError("record_path is required for render_sequence")

        label = self._writer_label()
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)

        ts_np = np.array(ts)
        q_np = np.array(q_ts)

        video_dt = np.mean(np.diff(ts_np))
        actual_fps = float(playback_speed) * (1.0 / video_dt)

        print(f"{label} Rendering video with dt={video_dt:.4f} and {len(ts_np)} frames")

        width, height = int(self.width), int(self.height)
        video_writer: FFmpegVideoWriter | None = None
        cv_video: cv2.VideoWriter | None = None
        try:
            # Resolve encoding defaults without changing the renderer configuration.
            if video_config is None:
                video_config = self.config.output.video
            video_writer = FFmpegVideoWriter(
                str(record_path),
                width,
                height,
                actual_fps,
                input_pix_fmt="bgr24",
                video_config=video_config,
            )
            print(
                f"{label} Writing video via ffmpeg to: {record_path} (fps≈{actual_fps:.2f})"
            )
        except FileNotFoundError:
            print(
                f"{label} ffmpeg not found; falling back to OpenCV VideoWriter (mp4v)"
            )
        except Exception as exc:
            print(
                f"{label} ffmpeg failed to start ({exc}); falling back to OpenCV VideoWriter (mp4v)"
            )

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cv_video = cv2.VideoWriter(
                str(record_path), fourcc, actual_fps, (width, height)
            )
            if not cv_video.isOpened():
                print(f"{label} OpenCV VideoWriter failed to open; skipping write.")
                cv_video = None

        curves = np.stack([np.asarray(self.compute_backbone_curve(q)) for q in q_np])
        offset = self._single_base_offset(base_offsets, target_dim=curves.shape[-1])
        if offset is not None:
            curves = curves + offset
        self._fit_scene_bounds(curves)
        self._appearance_bounds_locked = True
        previous_mode = getattr(self, "_rendering_mode", "static")
        self._rendering_mode = "animated"
        try:
            for frame_idx in range(len(ts_np)):
                img = self.render_frame(
                    q_np[frame_idx],
                    color_config=color_config,
                    camera_config=camera_config,
                    base_offsets=base_offsets,
                )
                if video_writer is not None:
                    video_writer.write(img)
                elif cv_video is not None:
                    cv_video.write(img)
        finally:
            self._appearance_bounds_locked = False
            self._rendering_mode = previous_mode
            if video_writer is not None:
                video_writer.close()
            elif cv_video is not None:
                cv_video.release()

        if video_writer is not None:
            if video_writer.stderr_log:
                print(f"{label} ffmpeg stderr: {video_writer.stderr_log}")
            print(f"Video saved to: {record_path}")
        elif cv_video is not None:
            print(f"Video saved to: {record_path}")
        else:
            print(f"{label} No video was written.")
