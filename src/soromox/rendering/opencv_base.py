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

    def _warn_simple_appearance(self, mode: str) -> None:
        """Explain the planar drawing style once per rendering mode.

        Args:
            mode: Rendering operation used to deduplicate warnings.

        Returns:
            None. Reports that scene appearance and camera settings are ignored.
        """
        self._warn_appearance(
            mode,
            [
                "scene appearance and camera settings; using a white background "
                "and the planar pixel projection"
            ],
        )

    def _blank_frame(self) -> np.ndarray:
        """Create the white canvas used by every OpenCV rendering path.

        Returns:
            BGR uint8 image of shape (height, width, 3), filled with white.
            Scene backgrounds, ground planes and lighting are ignored.
        """
        return np.full((self.height, self.width, 3), 255, dtype=np.uint8)

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
