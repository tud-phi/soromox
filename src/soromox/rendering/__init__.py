"""Rendering module for soft robot visualization.

This module provides various backends for visualizing robots:
- BaseSoftRobotRenderer: Abstract base class for all renderers
- MatplotlibRenderer: Generic renderer using Matplotlib (2D and 3D)
- Open3DRenderer: Generic 3D renderer using Open3D
- OpenCVPlanarHSARenderer: Specialized renderer for PlanarHSA robots
- OpenCVPlanarRenderer: Generic renderer for planar robots
"""

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer

# Open3D renderer is optional (requires open3d package)
try:
    from soromox.rendering.open3d_renderer import Open3DRenderer
except ImportError:
    Open3DRenderer = None

# Specialized OpenCV renderers
from soromox.rendering.opencv_planar_renderer import OpenCVPlanarRenderer
from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer

__all__ = [
    # Base class
    "BaseSoftRobotRenderer",
    # Generic renderers
    "MatplotlibRenderer",
    "Open3DRenderer",
    # Specialized renderers
    "OpenCVPlanarHSARenderer",
    "OpenCVPlanarRenderer",
]
