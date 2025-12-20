"""Rendering module for continuum soft robot visualization.

This module provides various backends for visualizing robots:
- BaseRobotRenderer: Abstract base class for all renderers
- MatplotlibRenderer: Generic renderer using Matplotlib (2D and 3D)
- Open3DRenderer: Generic 3D renderer using Open3D
- OpenCVPlanarHSARenderer: Specialized renderer for PlanarHSA robots
- OpenCVPlanarPCSRenderer: Specialized renderer for PlanarPCS robots
"""

from soromox.rendering.base import BaseContinuumSoftRobotRenderer
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer

# Open3D renderer is optional (requires open3d package)
try:
    from soromox.rendering.open3d_renderer import Open3DRenderer
except ImportError:
    Open3DRenderer = None

# Specialized OpenCV renderers
from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer
from soromox.rendering.planar_pcs.opencv_renderer import OpenCVPlanarPCSRenderer

# Legacy function aliases (backward compatibility)
from soromox.rendering.planar_hsa.opencv_renderer import draw_robot, animate_robot
from soromox.rendering.planar_pcs.opencv_renderer import render_planar_pcs

__all__ = [
    # Base class
    "BaseContinuumSoftRobotRenderer",
    # Generic renderers
    "MatplotlibRenderer",
    "Open3DRenderer",
    # Specialized renderers
    "OpenCVPlanarHSARenderer",
    "OpenCVPlanarPCSRenderer",
    # Legacy aliases
    "draw_robot",
    "animate_robot",
    "render_planar_pcs",
]
