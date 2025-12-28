"""Viser-based web visualization for soft robots.

This subpackage provides a web-based 3D visualization backend using Viser.
Access the visualization via a web browser at the specified host:port.

Main classes:
- ViserRenderer: Main renderer class extending BaseSoftRobotRenderer
- LiveModeController: Controller for live robot state updates
- GUIManager: Manages GUI panels and controls
- VideoRecorder: Records frames to video files
"""

from soromox.rendering.viser.viser_renderer import ViserRenderer

__all__ = [
    "ViserRenderer",
]
