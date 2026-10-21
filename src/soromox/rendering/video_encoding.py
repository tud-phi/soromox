"""Shared video encoding utilities for renderers.

Provides unified FFmpeg video encoding configuration and writer classes
that can be used by both Open3D and OpenCV renderers.
"""

from __future__ import annotations

__all__ = [
    "FFmpegVideoWriter",
]

import subprocess

import numpy as np

from soromox.rendering.config.output import VideoEncodingConfig


class FFmpegVideoWriter:
    """Minimal ffmpeg pipe for video frames.

    Supports both RGB and BGR input formats, making it suitable for
    use with both Open3D (RGB) and OpenCV (BGR) renderers.

    Attributes:
        path: Output video file path
        _stderr_log: Captured stderr output from ffmpeg process
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        input_pix_fmt: str = "rgb24",
        video_config: VideoEncodingConfig | None = None,
    ):
        """Initialize FFmpeg video writer.

        Args:
            path: Output video file path
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            input_pix_fmt: Input pixel format ("rgb24" or "bgr24")
            video_config: Optional encoding configuration
        """
        self.path = path
        self._stderr_log: str | None = None
        cfg = video_config or VideoEncodingConfig()
        args = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            input_pix_fmt,
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            cfg.codec,
        ]
        if cfg.preset:
            args += ["-preset", cfg.preset]
        if cfg.crf is not None:
            args += ["-crf", str(cfg.crf)]
        if cfg.tune:
            args += ["-tune", cfg.tune]
        if cfg.profile:
            args += ["-profile:v", cfg.profile]
        if cfg.bitrate:
            args += ["-b:v", cfg.bitrate]
        if cfg.gop is not None:
            args += ["-g", str(cfg.gop)]
        if cfg.pix_fmt:
            args += ["-pix_fmt", cfg.pix_fmt]
        if cfg.extra_args:
            args += [str(arg) for arg in cfg.extra_args]
        args.append(path)
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video stream.

        Args:
            frame: Frame array of shape (height, width, 3) with dtype uint8
        """
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        """Close the video writer and finalize the output file."""
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
        """Get captured stderr output from ffmpeg process."""
        return self._stderr_log
