"""Rendering module for soft robot visualization.

This module provides various backends for visualizing robots:
- BaseSoftRobotRenderer: Abstract base class for all renderers
- MatplotlibRenderer: Generic renderer using Matplotlib (2D and 3D)
- Open3DRenderer: Generic 3D renderer using Open3D
- ViserRenderer: Web-based 3D renderer using Viser (interactive browser visualization)
- OpenCVPlanarHSARenderer: Specialized renderer for PlanarHSA robots
- OpenCVPlanarRenderer: Generic renderer for planar robots
"""

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import (
    BackboneColorConfig,
    ColorLegend,
    RendererColorConfig,
    get_color_theme,
    list_builtin_palettes,
    list_builtin_themes,
)
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer
from soromox.rendering.video_encoding import VideoEncodingConfig

# Open3D renderer is optional (requires open3d package)
try:
    from soromox.rendering.open3d_renderer import Open3DRenderer
except ImportError:
    Open3DRenderer = None

# Viser renderer is optional (requires viser package)
try:
    from soromox.rendering.viser_renderer import ViserRenderer
except ImportError:
    ViserRenderer = None

# Specialized OpenCV renderers
from soromox.rendering.opencv_planar_renderer import OpenCVPlanarRenderer
from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer

__all__ = [
    # Base class
    "BaseSoftRobotRenderer",
    # Configuration
    "CameraConfig",
    "BackboneColorConfig",
    "RendererColorConfig",
    "ColorLegend",
    "list_builtin_palettes",
    "list_builtin_themes",
    "get_color_theme",
    "VideoEncodingConfig",
    # Generic renderers
    "MatplotlibRenderer",
    "Open3DRenderer",
    "ViserRenderer",
    # Specialized renderers
    "OpenCVPlanarHSARenderer",
    "OpenCVPlanarRenderer",
]
