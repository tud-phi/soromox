"""Rendering module for soft robot visualization.

This module provides various backends for visualizing robots:
- BaseSoftRobotRenderer: Abstract base class for all renderers
- MatplotlibRenderer: Generic renderer using Matplotlib (2D and 3D)
- Open3DRenderer: Generic 3D renderer using Open3D
- ViserRenderer: Web-based 3D renderer using Viser (interactive browser visualization)
- UMArmViserRenderer: Viser renderer specialized for the McKibben-actuated UMArm
- OpenCVPlanarHSARenderer: Specialized renderer for PlanarHSA robots
- OpenCVPlanarRenderer: Generic renderer for planar robots
"""

from soromox.rendering.actuators import (
    ActuatorVisualLayer,
    BatchedActuatorVisualLayer,
    TrajectoryActuatorVisualLayer,
)
from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import (
    ActuatorStyleConfig,
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

try:
    from soromox.rendering.umarm import UMArmViserRenderer
except ImportError:
    UMArmViserRenderer = None

# Specialized OpenCV renderers are optional (require opencv-python)
try:
    from soromox.rendering.opencv_planar_renderer import OpenCVPlanarRenderer
except ImportError:
    OpenCVPlanarRenderer = None

try:
    from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer
except ImportError:
    OpenCVPlanarHSARenderer = None

__all__ = [
    # Base class
    "BaseSoftRobotRenderer",
    # Configuration
    "CameraConfig",
    "ActuatorStyleConfig",
    "BackboneColorConfig",
    "RendererColorConfig",
    "ColorLegend",
    "list_builtin_palettes",
    "list_builtin_themes",
    "get_color_theme",
    "VideoEncodingConfig",
    "ActuatorVisualLayer",
    "BatchedActuatorVisualLayer",
    "TrajectoryActuatorVisualLayer",
    # Generic renderers
    "MatplotlibRenderer",
    "Open3DRenderer",
    "ViserRenderer",
    "UMArmViserRenderer",
    # Specialized renderers
    "OpenCVPlanarHSARenderer",
    "OpenCVPlanarRenderer",
]
