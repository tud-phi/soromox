# Rendering

Visualization and rendering utilities for robot systems.

## Overview

The rendering module provides a class-based architecture for visualizing soft robots. All renderers inherit from a common base class `BaseSoftRobotRenderer` which provides cached forward kinematics and a unified interface.

### Architecture

```text
BaseSoftRobotRenderer (abstract base)
├── MatplotlibRenderer         # Generic, works with any robot (2D/3D)
├── Open3DRenderer             # Generic, works with any 3D robot
├── OpenCVPlanarHSARenderer    # Specialized for PlanarHSA
└── OpenCVPlanarRenderer       # Generic for planar robots
```

### Quick Start

=== "Matplotlib (any robot)"

    ```python
    from soromox.rendering import MatplotlibRenderer

    # Create renderer
    renderer = MatplotlibRenderer(robot, num_points=50, robot_colors="plasma")

    # Single frame
    renderer.show(q, robot_colors="plasma")

    # Interactive animation with slider
    renderer.animate(ts, q_ts, mode="slider", robot_colors="plasma")

    # Auto-playing animation
    renderer.animate(ts, q_ts, mode="animation", interval=50, robot_colors="plasma")
    ```

=== "Open3D (3D robots)"

    ```python
    from soromox.rendering import Open3DRenderer

    # Create renderer (defaults to legacy viewer backend)
    renderer = Open3DRenderer(
        robot,
        num_points=80,
        viewer_backend="legacy",
        seg_colors="coolwarm",    # colormap or (N,3)/(N,4) array with optional alpha
        robot_colors="viridis",   # colormap or (N,3)/(N,4) array for batched robots
    )

    # Single interactive frame (supports base offsets / spheres if provided)
    renderer.show(q, backend="legacy")

    # Animated playback with keyboard controls
    renderer.render_sequence(ts, q_ts, playback_speed=1.0, loop=True, backend="legacy")

    # Save directly to a video or frame directory (requires ffmpeg for video)
    renderer.render_sequence(
        ts,
        q_ts,
        playback_speed=1.0,
        record_path="trajectory.mp4",  # or a directory for PNG frames
        backend="gui",  # optional: use GUI SceneWidget if installed
    )

    # Headless capture (returns uint8 image array)
    img = renderer.render_frame(
        q,
        static_spheres_positions=targets_xyz,  # optional features
        static_spheres_radii=targets_r,
        robot_colors="plasma",
    )
    ```

=== "OpenCV (planar)"

    ```python
    from soromox.rendering import OpenCVPlanarRenderer

    # Create renderer for planar robots
    renderer = OpenCVPlanarRenderer(robot, width=800, height=600)

    # Render single frame to numpy array
    img = renderer.render_frame(q)

    # Create video
    renderer.render_sequence(ts, q_ts, record_path="video.mp4")
    ```

### Colors

- `seg_colors` (Open3D): optional colormap name or array of shape (N,3)/(N,4); defaults to the `coolwarm` colormap.
- `robot_colors` (Matplotlib/Open3D): optional colormap name or array of shape (N,3)/(N,4) cycled across robots; `"same"` uses the first segment color. Alpha is honored where supported.

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

::: soromox.rendering.base.BaseSoftRobotRenderer
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

## OpenCV Planar Renderer

::: soromox.rendering.opencv_planar_renderer.OpenCVPlanarRenderer
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
