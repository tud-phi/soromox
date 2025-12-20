"""Rendering module for continuum soft robot visualization.

This module provides various backends for visualizing robots:
- BaseRobotRenderer: Abstract base class for all renderers
- MatplotlibRenderer: Generic renderer using Matplotlib (2D and 3D)
- Open3DRenderer: Generic 3D renderer using Open3D
- OpenCVPlanarHSARenderer: Specialized renderer for PlanarHSA robots
- OpenCVPlanarPCSRenderer: Specialized renderer for PlanarPCS robots
"""

from soromox.rendering.base import BaseContinuumSoftRobotRenderer
from soromox.rendering.continuum_soft_robot import MatplotlibRenderer

# Open3D renderer is optional (requires open3d package)
try:
    from soromox.rendering.continuum_soft_robot import Open3DRenderer
except ImportError:
    Open3DRenderer = None

# Specialized OpenCV renderers
from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer
from soromox.rendering.planar_pcs.opencv_renderer import OpenCVPlanarPCSRenderer

__all__ = [
    # Base class
    "BaseContinuumSoftRobotRenderer",
    # Generic renderers
    "MatplotlibRenderer",
    "Open3DRenderer",
    # Specialized renderers
    "OpenCVPlanarHSARenderer",
    "OpenCVPlanarPCSRenderer",
]
