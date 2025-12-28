# Rendering

Visualization and rendering utilities for robot systems.

## Overview

The rendering module provides a class-based architecture for visualizing soft robots. All renderers inherit from a common base class `BaseSoftRobotRenderer` which provides cached forward kinematics and a unified interface.

### Architecture

```text
BaseSoftRobotRenderer (abstract base)
├── MatplotlibRenderer         # Generic, works with any robot (2D/3D)
├── Open3DRenderer             # Generic, works with any 3D robot
├── ViserRenderer              # Web-based interactive 3D visualization
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

=== "Viser (web-based 3D)"

    ```python
    from soromox.rendering import ViserRenderer

    # Create renderer - opens browser at localhost:8080
    renderer = ViserRenderer(robot, port=8080, num_points=80)

    # Single interactive frame
    renderer.show(q)  # Opens browser automatically

    # Animated playback with GUI controls
    renderer.render_sequence(
        ts, q_ts,
        playback_speed=1.0,
        loop=True,
        autoplay=True,
    )

    # Multiple robots in grid layout
    renderer.show(q_batch, robot_colors="viridis")

    # Multiple robots overlayed with transparency
    renderer.render_sequence(
        ts, q_ts_batched,  # (N, T, DOF)
        overlay_mode="overlay",
        robot_alphas=[1.0, 0.5, 0.3],
    )

    # Live mode - callback-based
    ctrl = renderer.start_live_mode(callback=lambda t: compute_q(t), dt=0.02)
    # ... later ...
    ctrl.stop()

    # Live mode - streaming
    ctrl = renderer.start_live_mode()
    for q in simulation_loop():
        ctrl.push_state(q)
    ctrl.stop()

    # Record to video
    renderer.render_sequence(
        ts, q_ts,
        record_path="trajectory.mp4",
    )

    # High-quality rendering with professional lighting
    renderer = ViserRenderer(
        robot,
        port=8080,
        # Mesh quality
        sphere_resolution=3,      # Smoother spheres (1=low, 2=medium, 3=good, 4=high)
        cylinder_sections=64,     # High-quality cylinders
        # Lighting
        directional_light_intensity=0.9,
        ambient_light_intensity=0.3,
        cast_shadows=True,
        # Materials & Shading
        material="standard",      # or "toon3", "toon5" for cartoon styles
        flat_shading=False,       # Use smooth shading
        wireframe=False,          # Solid rendering
    )

    # Cartoon/toon rendering style
    renderer = ViserRenderer(
        robot,
        material="toon3",         # 3-level toon shading
        flat_shading=True,        # Enhance cartoon effect
    )
    ```

### Colors

- `seg_colors` (Open3D/Viser): optional colormap name or array of shape (N,3)/(N,4); defaults to the `coolwarm` colormap.
- `robot_colors` (Matplotlib/Open3D/Viser): optional colormap name or array of shape (N,3)/(N,4) cycled across robots; `"same"` uses the first segment color. Alpha is honored where supported.

### Visual Quality Settings (Viser)

ViserRenderer provides extensive visual quality controls:

**Lighting:**

- `enable_default_lights` (bool): Enable Viser's default lighting (default: True)
- `add_directional_light` (bool): Add custom directional (key) light (default: True)
- `directional_light_intensity` (float): Directional light intensity, 0-1+ (default: 0.8)
- `directional_light_direction` (tuple): Light direction vector (x, y, z) (default: (-0.5, -1.0, -0.5))
- `add_ambient_light` (bool): Add custom ambient (fill) light (default: True)
- `ambient_light_intensity` (float): Ambient light intensity, 0-1+ (default: 0.4)
- `cast_shadows` (bool): Enable shadow casting (default: True)

**Materials & Shading:**

- `material` (str): Material type - "standard" (PBR), "toon3" (3-level cartoon), "toon5" (5-level cartoon) (default: "standard")
- `flat_shading` (bool): Use flat shading instead of smooth Gouraud shading (default: False)
- `wireframe` (bool): Render geometry as wireframe (default: False)

**Mesh Quality:**

- `sphere_resolution` (int): Icosphere subdivision level 1-4 (1=low, 2=medium, 3=good, 4=high quality) (default: 3)
- `cylinder_sections` (int): Number of cylinder cross-section segments for swept backbone (default: 48)

**Shadow Control:**

- `backbone_cast_shadow` (bool): Enable shadow casting for backbone geometry (default: True)
- `sphere_cast_shadow` (bool): Enable shadow casting for spheres (default: True)

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

## Viser Renderer

The `ViserRenderer` provides web-based interactive 3D visualization using [Viser](https://github.com/nerfstudio-project/viser). It opens in your browser and supports:

- **Live visualization**: Real-time robot state updates
- **GUI controls**: Play/pause, timeline slider, speed control
- **Multiple robots**: Grid layout or transparent overlay
- **Dynamic spheres**: Visualize setpoints, obstacles, targets
- **Video export**: Record animations to MP4

::: soromox.rendering.viser.viser_renderer.ViserRenderer
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
