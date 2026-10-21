"""Rendered image dimensions and video encoding settings."""

from dataclasses import dataclass, field

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


@dataclass
class RenderOutputConfig:
    """Image dimensions and video encoding defaults.

    Attributes:
        width: Positive image width in pixels.
        height: Positive image height in pixels.
        video: Default video encoding configuration.
    """

    width: int = 800
    height: int = 600
    video: VideoEncodingConfig = field(default_factory=VideoEncodingConfig)

    def __post_init__(self) -> None:
        """Validate integer image dimensions.

        Returns:
            None.

        Raises:
            ValueError: A dimension is not a positive integer.
        """
        for name in ("width", "height"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
