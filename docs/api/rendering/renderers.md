# Rendering

Visualization and rendering utilities for robot systems.

Composable actuators expose semantic geometry through
`robot.actuator_visual_layers(...)`; see
[Threadlike actuation](../actuation/threadlike.md#rendering).

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

### Choosing a renderer

The renderers overlap in features, but each has a sweet spot. Use this table as a
quick guide and then jump to the corresponding quick-start tab.

| Renderer | Best for | Strengths | Limitations / notes |
| --- | --- | --- | --- |
| `MatplotlibRenderer` | Static plots, notebooks, quick debugging | Visualizes the backbone of any robot (2D/3D), easy to embed in notebooks | Limited interactivity; does not visualize the robot geometry; slower for long animations |
| `Open3DRenderer` | Interactive 3D inspection, offline capture | Rich 3D camera controls, interactive playback, headless frame capture | Requires Open3D; 3D only; no Python 3.13+ wheels yet |
| `ViserRenderer` | Live demos, web-based 3D, remote viewing | Browser UI, live streaming, multi-robot layouts, high-quality lighting | Requires running a local web server; 3D only |
| `OpenCVPlanarRenderer` | Fast planar animations, video export | Lightweight, fast 2D output, simple video recording | Planar robots only; minimal interactivity |
| `OpenCVPlanarHSARenderer` | PlanarHSA-specific visuals | Adds HSA-specific geometry and styling | PlanarHSA only |

### Quick Start

=== "Matplotlib (any robot)"

    ```python
    from soromox.rendering import BackboneColorConfig, MatplotlibRenderer, RendererColorConfig

    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(robot_palette="plasma")
    )

    # Create renderer
    renderer = MatplotlibRenderer(robot, num_points=50, color_config=color_config)

    # Single frame
    renderer.show(q, color_config=color_config)

    # Interactive animation with slider
    renderer.animate(ts, q_ts, mode="slider", color_config=color_config)

    # Auto-playing animation
    renderer.animate(ts, q_ts, mode="animation", interval=50, color_config=color_config)
    ```

=== "Open3D (3D robots)"

    !!! note
        Open3D does not currently support Python 3.13+ (no wheels available yet).
        Use Python 3.12 or earlier for the Open3D renderer.

    ```python
    from soromox.rendering import BackboneColorConfig, Open3DRenderer, RendererColorConfig

    # Create renderer
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_palette="soromox:ember",  # colormap name or (S,3)/(S,4) array
            robot_palette="soromox:okabe-ito",  # per-robot palette for batched layouts
        )
    )
    renderer = Open3DRenderer(
        robot,
        num_points=80,
        color_config=color_config,
    )

    # Single interactive frame (supports base offsets / spheres if provided)
    renderer.show(q)

    # Animated playback with keyboard controls
    renderer.render_sequence(ts, q_ts, playback_speed=1.0, loop=True)

    # Save directly to a video or frame directory (requires ffmpeg for video)
    renderer.render_sequence(
        ts,
        q_ts,
        playback_speed=1.0,
        record_path="trajectory.mp4",  # or a directory for PNG frames
    )

    # Headless capture (returns uint8 image array)
    img = renderer.render_frame(
        q,
        static_spheres_positions=targets_xyz,  # optional features
        static_spheres_radii=targets_r,
        color_config=color_config,
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
    from soromox.rendering import BackboneColorConfig, RendererColorConfig, ViserRenderer

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
    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(robot_palette="viridis")
    )
    renderer.show(q_batch, color_config=color_config)

    # Multiple robots overlayed (robot color alpha sets default opacity)
    renderer.render_sequence(
        ts, q_ts_batched,  # (N, T, DOF)
        multi_robot_layout="overlay",
        color_config=RendererColorConfig(
            backbone=BackboneColorConfig(
                robot_colors=[
                    [0.2, 0.6, 0.9, 0.35],
                    [0.9, 0.3, 0.2, 0.5],
                    [0.2, 0.8, 0.4, 0.7],
                ]
            )
        ),
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

Colors are configured via `RendererColorConfig` + `BackboneColorConfig` and resolved
using a consistent hierarchy. More specific inputs override less specific ones:

- `robot_palette` → `segment_palette` → `point_palette`
- `robot_colors` → `segment_colors` → `point_colors`
- `robot_segment_colors` → `robot_point_colors`

Alpha in per-robot colors is propagated to more specific colors when those omit alpha.
Open3D's interactive viewer approximates alpha by blending with the background.
Open3D backbone geometry is per-segment, so point palettes are averaged into segments.

Pass `color_config=...` to renderer constructors or per-call methods (`show`,
`render_frame`, `render_sequence`, `animate`) to override styles.

Built-in publication-friendly palettes and themes are available:

- `list_builtin_palettes()` includes `soromox:okabe-ito`, `soromox:tol-bright`,
  `soromox:tol-muted`, `soromox:ember`, `soromox:glacier`, `soromox:slate`
- `get_color_theme("soromox:paper")` and `list_builtin_themes()` return presets

Example:

```python
from soromox.rendering import ViserRenderer, get_color_theme

renderer = ViserRenderer(robot, color_config=get_color_theme("soromox:paper"))
```

For legend data, use the tiny helper:

- `renderer.get_color_legend(num_robots=..., color_config=...)`

Shape hints:

- `robot_colors`: `(N, 3/4)` or broadcastable
- `segment_colors`: `(S, 3/4)` (segments)
- `point_colors`: `(P, 3/4)` (backbone points)
- `robot_segment_colors`: `(N, S, 3/4)`
- `robot_point_colors`: `(N, P, 3/4)`

Use `segment_palette` for lengthwise segment coloring and `point_palette` for
per-point gradients along the backbone.
Set `segment_palette=None` to fall back to solid per-robot colors.

`RendererColorConfig` also provides renderer-specific colors:

- `base_plate_color` (Open3D, Viser)
- `actuators.default_color` and `actuators.scalar_colormap`

### Camera configuration

Use `CameraConfig` with Open3D and Viser to control field of view and view
placement. By default, the camera auto-positions based on the scene bounds.

```python
from soromox.rendering import CameraConfig

camera = CameraConfig(
    fov=60.0,
    position=(0.6, -0.6, 0.4),
    look_at=(0.0, 0.0, 0.1),
)
renderer.show(q, camera_config=camera)
```

Matplotlib uses `camera_config` to set 3D view angles; 2D views ignore it.

### Recording and video encoding

All renderers support `record_path` in `render_sequence`, but the backends differ:

- Matplotlib relies on Matplotlib animation writers (ffmpeg required for mp4).
- Open3D and Viser use ffmpeg and accept `video_config=VideoEncodingConfig(...)`.
- Open3D also writes PNG frames when `record_path` is a directory, with
  `record_every_n` and `record_prefix` (Viser also supports `record_every_n`).
- OpenCV renderers require `record_path`, use ffmpeg when available, and fall back
  to OpenCV VideoWriter.

Example (Open3D/Viser):

```python
from soromox.rendering import VideoEncodingConfig

renderer.render_sequence(
    ts,
    q_ts,
    record_path="trajectory.mp4",
    video_config=VideoEncodingConfig(crf=18, pix_fmt="yuv420p"),
)
```

### Multi-robot layouts

Matplotlib, Open3D, and Viser accept batched inputs:

- `q` with shape `(N, DOF)` for `show()` and `render_frame()`
- `q_ts` with shape `(N, T, DOF)` for `render_sequence()` or `animate()`

Use `base_offsets` or `grid_spacing` to control placement. Viser additionally
supports `multi_robot_layout="overlay"` to stack robots at the same base pose.

### Actuators and helper geometry

If the robot exposes `actuator_visual_layers(q, s_points, *, actuator_inputs=None)`,
actuator rendering can be enabled with `render_actuators=True`. The hook should
return one or more `ActuatorVisualLayer` objects with points shaped
`(num_actuators_in_layer, num_points, dim)`.

For large batches or trajectories, robots can optionally provide
`actuator_visual_layers_batched(q_batch, s_points, *, actuator_inputs=None)` with
points shaped `(N, num_actuators_in_layer, num_points, dim)` or
`actuator_visual_layers_trajectory(q_ts, s_points, *, actuator_inputs=None)` with
points shaped `(N, T, num_actuators_in_layer, num_points, dim)`. The base
renderer uses these hooks directly when available and falls back to the single
hook otherwise.

Open3D and Viser also support helper spheres:

- Static: `static_spheres_positions`, `static_spheres_radii`, `static_spheres_colors`
- Dynamic: `dynamic_spheres_positions`, `dynamic_spheres_radii`,
  `dynamic_spheres_colors`

### Geometry styles (Open3D and Viser)

Set `backbone_style="discrete"` for per-point spheres or `"swept"` for
cylinders/boxes/ellipses based on the robot cross-section geometry. Base plate
sizing is controlled with `base_plate_radius_scale` and `base_plate_thickness`.

### GUI plots (Viser)

`ViserRenderer.render_sequence()` can add Plotly panels:

- `plot_configurations=True`
- `plot_actuator_positions=True`
- `custom_plots={"My Plot": (figure, aspect)}`

Plot panels require `plotly`. `ViserRenderer.render_frame()` also requires at
least one connected client to capture images. Plot panels only support `q_ts`
with shape `(T, DOF)` (no batched layouts).

Viser also exposes `add_dynamic_sphere()`, `update_dynamic_sphere()`, and
`remove_dynamic_sphere()` for custom markers.

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
      docstring_section_style: table
      members_order: source

## Matplotlib Renderer

::: soromox.rendering.matplotlib_renderer.MatplotlibRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

## Open3D Renderer

The `Open3DRenderer` provides high-quality 3D visualization using the [Open3D](http://www.open3d.org/) library. It offers:

- **Interactive 3D viewer**: Mouse-based camera controls
- **High-quality rendering**: Advanced shading and lighting
- **Real-time updates**: Efficient geometry updates
- **Screenshot export**: Save high-resolution images

::: soromox.rendering.open3d_renderer.Open3DRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

### References

Zhou, Q. Y., Park, J., & Koltun, V. (2018). Open3D: A modern library for 3D data processing. *arXiv preprint arXiv:1801.09847*.

[Open3D Documentation](http://www.open3d.org/docs/)

## Viser Renderer

The `ViserRenderer` provides web-based interactive 3D visualization using [Viser](https://github.com/nerfstudio-project/viser). It opens in your browser and supports:

- **Live visualization**: Real-time robot state updates
- **GUI controls**: Play/pause, timeline slider, speed control
- **Multiple robots**: Grid layout or transparent overlay
- **Dynamic spheres**: Visualize setpoints, obstacles, targets
- **Plot panels**: Embed Plotly figures in the GUI
- **Tendons**: Render tendon paths when available
- **Video export**: Record animations to MP4

::: soromox.rendering.viser_renderer.ViserRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

### References

Yi, B., Kim, C. M., Kerr, J., Wu, G., Feng, R., Zhang, A., Kulhanek, J., Choi, H., Ma, Y., Tancik, M., & Kanazawa, A. (2025). Viser: Imperative, Web-based 3D Visualization in Python. *arXiv preprint arXiv:2507.22885*.

- [arXiv Paper](https://arxiv.org/abs/2507.22885)
- [Viser GitHub Repository](https://github.com/nerfstudio-project/viser)

## OpenCV Planar Renderer

::: soromox.rendering.opencv_planar_renderer.OpenCVPlanarRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

## OpenCV Planar HSA Renderer

::: soromox.rendering.planar_hsa.opencv_renderer.OpenCVPlanarHSARenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source
