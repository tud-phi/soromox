"""Compatibility re-exports for moved renderers."""

from soromox.rendering.matplotlib_renderer import MatplotlibRenderer
from soromox.rendering.open3d_renderer import Open3DRenderer

__all__ = ["MatplotlibRenderer", "Open3DRenderer"]
