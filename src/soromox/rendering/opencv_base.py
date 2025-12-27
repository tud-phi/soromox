"""Shared OpenCV rendering utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from jax import Array

from soromox.rendering.base import BaseSoftRobotRenderer

FFMPEG_VIDEO_CODEC = "libx264"
FFMPEG_PIXEL_FORMAT = "yuv420p"


class _BGRFFmpegVideoWriter:
    """Minimal ffmpeg pipe for BGR frames."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        self._stderr_log: str | None = None
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-vcodec",
                FFMPEG_VIDEO_CODEC,
                "-pix_fmt",
                FFMPEG_PIXEL_FORMAT,
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            if self.proc.stderr:
                try:
                    err = self.proc.stderr.read()
                    if err:
                        self._stderr_log = err.decode("utf-8", errors="ignore")
                except Exception:
                    pass
        finally:
            self.proc.wait()

    @property
    def stderr_log(self) -> str | None:
        return self._stderr_log


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
    ) -> None:
        """Render animated sequence to video file using ffmpeg (H.264, yuv420p).

        Falls back to OpenCV VideoWriter if ffmpeg is unavailable.
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
        video_writer: _BGRFFmpegVideoWriter | None = None
        cv_video: cv2.VideoWriter | None = None
        try:
            video_writer = _BGRFFmpegVideoWriter(
                str(record_path), width, height, actual_fps
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
