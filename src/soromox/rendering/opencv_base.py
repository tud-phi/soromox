"""Shared OpenCV rendering utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from jax import Array

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.video_encoding import FFmpegVideoWriter, VideoEncodingConfig


class BaseOpenCVRenderer(BaseSoftRobotRenderer):
    """Base class adding video recording for OpenCV renderers."""

    def _writer_label(self) -> str:
        return f"[{self.__class__.__name__}]"

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        playback_speed: float = 1.0,
        record_path: str | None = None,
        video_config: VideoEncodingConfig | None = None,
    ) -> None:
        """Render animated sequence to video file using ffmpeg.

        Falls back to OpenCV VideoWriter if ffmpeg is unavailable.

        Args:
            ts: Time stamps of shape (T,)
            q_ts: Configurations of shape (T, DOF)
            playback_speed: Playback speed multiplier (>0)
            record_path: Path to save video file
            video_config: Optional ffmpeg encoding configuration
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
            # Use default config with yuv420p for OpenCV (better compatibility)
            if video_config is None:
                video_config = VideoEncodingConfig(pix_fmt="yuv420p")
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

        for frame_idx in range(len(ts_np)):
            img = self.render_frame(q_np[frame_idx])
            if video_writer is not None:
                video_writer.write(img)
            elif cv_video is not None:
                cv_video.write(img)

        if video_writer is not None:
            video_writer.close()
            if video_writer.stderr_log:
                print(f"{label} ffmpeg stderr: {video_writer.stderr_log}")
            print(f"Video saved to: {record_path}")
        elif cv_video is not None:
            cv_video.release()
            print(f"Video saved to: {record_path}")
        else:
            print(f"{label} No video was written.")
