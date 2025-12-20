# Rendering

Visualization and rendering utilities for robot systems.

## Overview

The rendering module provides a class-based architecture for visualizing continuum soft robots. All renderers inherit from a common base class `BaseContinuumSoftRobotRenderer` which provides cached forward kinematics and a unified interface.

### Architecture

```text
BaseContinuumSoftRobotRenderer (abstract base)
├── MatplotlibRenderer         # Generic, works with any robot (2D/3D)
├── Open3DRenderer             # Generic, works with any 3D robot
├── OpenCVPlanarHSARenderer    # Specialized for PlanarHSA
└── OpenCVPlanarPCSRenderer    # Specialized for PlanarPCS
```

### Quick Start

=== "Matplotlib (any robot)"

    ```python
    from soromox.rendering import MatplotlibRenderer

    # Create renderer
    renderer = MatplotlibRenderer(robot, num_points=50)

    # Single frame
    renderer.show(q)

    # Interactive animation with slider
    renderer.animate(ts, q_ts, mode="slider")

    # Auto-playing animation
    renderer.animate(ts, q_ts, mode="animation", interval=50)
    ```

=== "Open3D (3D robots)"

    ```python
    from soromox.rendering import Open3DRenderer

    # Create renderer
    renderer = Open3DRenderer(robot, num_points=80)

    # Single interactive frame
    renderer.show(q)

    # Animated playback with keyboard controls
    renderer.render_sequence(ts, q_ts, playback_speed=1.0, loop=True)

    # Headless capture
    img = renderer.render_frame(q)
    ```

=== "OpenCV (specialized)"

    ```python
    from soromox.rendering.planar_pcs import OpenCVPlanarPCSRenderer

    # Create renderer for planar PCS
    renderer = OpenCVPlanarPCSRenderer(robot, width=800, height=600)

    # Render single frame to numpy array
    img = renderer.render_frame(q)

    # Create video
    renderer.render_sequence(ts, q_ts, output_path="video.mp4")
    ```

### Keyboard Controls (Open3D)

When using `Open3DRenderer.render_sequence()`, the following keyboard controls are available:

| Key | Action |
| --- | ------ |
| `Space` | Play/Pause |
| `→` / `←` | Next/Previous frame |
| `H` | Go to first frame |
| `S` | Save snapshot |
| `R` | Reset camera view |
| `C` | Capture camera parameters |
| `L` | Load saved camera view |
| `V` | Print camera parameters |
| `Q` / `Esc` | Quit |

## Base Class

::: soromox.rendering.base.BaseContinuumSoftRobotRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

## Matplotlib Renderer

::: soromox.rendering.matplotlib_renderer.MatplotlibRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

## Open3D Renderer

::: soromox.rendering.open3d_renderer.Open3DRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

## OpenCV Planar PCS Renderer

::: soromox.rendering.planar_pcs.opencv_renderer.OpenCVPlanarPCSRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

## OpenCV Planar HSA Renderer

::: soromox.rendering.planar_hsa.opencv_renderer.OpenCVPlanarHSARenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source

## Animation Utilities

::: soromox.rendering.animation
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members_order: source
