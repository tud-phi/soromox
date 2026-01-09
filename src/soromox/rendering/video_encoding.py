"""Shared video encoding utilities for renderers.

Provides unified FFmpeg video encoding configuration and writer classes
that can be used by both Open3D and OpenCV renderers.
"""

from __future__ import annotations

__all__ = [
    "VideoEncodingConfig",
    "FFmpegVideoWriter",
]

import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoEncodingConfig:
    """Configuration for FFmpeg video encoding.

    Attributes:
        codec: Video codec (e.g., "libx264", "libx265")
        pix_fmt: Output pixel format (e.g., "yuv444p", "yuv420p")
        preset: Encoding preset (e.g., "veryslow", "slow", "medium", "fast")
        crf: Constant Rate Factor (0-51, lower = higher quality)
        tune: Tuning preset (e.g., "animation", "film", "grain")
        profile: Codec profile (e.g., "high", "baseline")
        bitrate: Target bitrate (e.g., "5M", "1000k")
        gop: Group of Pictures size (keyframe interval)
        extra_args: Additional FFmpeg arguments as tuple of strings
    """

    codec: str = "libx264"
    pix_fmt: str = "yuv444p"
    preset: str | None = "veryslow"
    crf: int | None = 12
    tune: str | None = "animation"
    profile: str | None = None
    bitrate: str | None = None
    gop: int | None = None
    extra_args: tuple[str, ...] = ()


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
