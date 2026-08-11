# Shared Renderer Configuration

This page documents camera, base, ground-plane, and color settings shared by
multiple SoRoMoX renderers.

## Camera Configuration

### CameraConfig

The `CameraConfig` class provides unified camera configuration across renderers (Matplotlib, Open3D, Viser).

```python
from soromox.rendering import CameraConfig

camera = CameraConfig(
    fov=60.0,                           # Field of view in degrees
    position=(0.6, -0.6, 0.4),          # Camera position (x, y, z)
    look_at=(0.0, 0.0, 0.1),            # Point camera looks at
    up=(0.0, 0.0, 1.0),                 # Camera up vector (default: Z-up)
    distance_factor=2.0,                # Multiplier for auto-positioning
)

renderer.show(q, camera_config=camera)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fov` | float | 75.0 | Field of view in degrees |
| `position` | tuple | None | Explicit camera position (x, y, z), None for auto |
| `look_at` | tuple | None | Point camera looks at, None for scene center |
| `up` | tuple | (0, 0, 1) | Camera up vector |
| `distance_factor` | float | 10.0 | Multiplier for auto-positioning distance |
| `position_offset` | tuple | (0.8, -0.8, 0.5) | Direction vector for camera placement |

Matplotlib applies `fov` and the viewing direction from `position` to
`look_at`; its axes limits determine the remaining framing. Open3D and Viser
also use the explicit camera distance.

::: soromox.rendering.camera_config.CameraConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

---

## Robot Base and Ground Plane

All renderers obtain the robot origin and orientation from the model's
`base_pose` and `base_transform`. This keeps the rendered backbone, base, and
reference geometry in the same world frame without a renderer-specific pose
override.

| Setting | Purpose | Availability |
| --- | --- | --- |
| `base_plate_radius_scale` | Scales the rendered base-plate radius relative to the robot cross section | Open3D and Viser |
| `base_plate_thickness` | Sets the base-plate thickness in meters | Open3D and Viser |
| `show_ground_plane` | Shows or hides the base-aligned reference plane | Matplotlib, Open3D, and Viser |
| `ground_plane_size` | Sets the reference-plane side length in meters; `None` uses a robot-scaled default | Matplotlib, Open3D, and Viser |

```python
from soromox.rendering import ViserRenderer

renderer = ViserRenderer(
    robot,
    base_plate_radius_scale=2.0,
    base_plate_thickness=0.06,
    show_ground_plane=True,
    ground_plane_size=0.6,
)
```

Matplotlib draws a lightweight base marker. Open3D and Viser render base-plate
geometry and place the ground plane immediately behind it along the model's
base axis. Viser uses its native grid primitive. Specialized Viser renderers,
including I-SUPPORT and UMArm, inherit these settings.

---

## Color Configuration

### Color Hierarchy

Colors are configured via `RendererColorConfig` + `BackboneColorConfig` and resolved using a consistent hierarchy. More specific inputs override less specific ones:

```
robot_palette → segment_palette → point_palette
robot_colors → segment_colors → point_colors
robot_segment_colors → robot_point_colors
```

Alpha values in per-robot colors propagate to more specific colors when those omit alpha.

### RendererColorConfig

Main color configuration container for all renderers.

```python
from soromox.rendering import ActuatorStyleConfig, BackboneColorConfig, RendererColorConfig

color_config = RendererColorConfig(
    backbone=BackboneColorConfig(
        robot_palette="viridis",
        segment_palette="soromox:ember",
    ),
    base_plate_color=(0.5, 0.5, 0.5),
    ground_plane_color=(0.94, 0.95, 0.96),
    ground_plane_grid_color=(0.72, 0.75, 0.78),
    actuators=ActuatorStyleConfig(
        default_color=(0.8, 0.2, 0.2),
        kind_colors={"tendon": (0.85, 0.2, 0.15)},
        kind_radii={"tendon": 5e-4},
    ),
)

renderer.show(q, color_config=color_config)
```

::: soromox.rendering.color_config.RendererColorConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

### BackboneColorConfig

Backbone-specific color configuration with palette and explicit color support.

```python
from soromox.rendering import BackboneColorConfig

backbone_config = BackboneColorConfig(
    robot_palette="plasma",                    # Colormap for multiple robots
    segment_palette="soromox:ember",           # Per-segment colors
    robot_colors=[(0.2, 0.6, 0.9, 0.5)],       # Explicit robot colors with alpha
)
```

::: soromox.rendering.color_config.BackboneColorConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

---

## Built-in Palettes and Themes

### Available Palettes

SoRoMoX includes publication-friendly color palettes:

| Palette | Description |
|---------|-------------|
| `soromox:okabe-ito` | Colorblind-friendly palette |
| `soromox:tol-bright` | Paul Tol's bright qualitative palette |
| `soromox:tol-muted` | Paul Tol's muted qualitative palette |
| `soromox:ember` | Warm gradient (orange to red) |
| `soromox:glacier` | Cool gradient (blue to cyan) |
| `soromox:slate` | Neutral gray gradient |

You can also use any Matplotlib colormap name (e.g., `"viridis"`, `"plasma"`, `"coolwarm"`).

```python
from soromox.rendering import list_builtin_palettes

print(list_builtin_palettes())
```

### Color Themes

Pre-configured themes for consistent styling:

```python
from soromox.rendering import get_color_theme, list_builtin_themes

# List available themes
print(list_builtin_themes())

# Use a theme
theme = get_color_theme("soromox:paper")
renderer = ViserRenderer(robot, color_config=theme)
```

---

## Color Shape Reference

When providing explicit colors, use these shapes:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `robot_colors` | (N, 3/4) | Per-robot colors (RGB or RGBA) |
| `segment_colors` | (S, 3/4) | Per-segment colors |
| `point_colors` | (P, 3/4) | Per-backbone-point colors |
| `robot_segment_colors` | (N, S, 3/4) | Per-robot, per-segment colors |
| `robot_point_colors` | (N, P, 3/4) | Per-robot, per-point colors |

---

## Color Legend

For creating legends in plots:

```python
legend = renderer.get_color_legend(num_robots=3, color_config=color_config)
# Returns ColorLegend with robot labels and colors
```

::: soromox.rendering.color_config.ColorLegend
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

---

## Multi-Robot Layouts

Matplotlib, Open3D, and Viser accept batched robot configurations:

- `q` with shape `(N, DOF)` for `show()` and `render_frame()`;
- `q_ts` with shape `(N, T, DOF)` for sequence rendering or animation.

Use `base_offsets` to place each robot explicitly or `grid_spacing` to control
automatically generated layouts. Viser additionally supports
`multi_robot_layout="overlay"` to render robots at a common base pose. Per-robot
colors and alpha values can distinguish overlaid configurations.

Open3D automatically merges each robot's backbone primitives into one dynamic
mesh when an animated scene contains multiple robots. This removes most backend
geometry registrations; set `merge_backbone_meshes=True` to force merging for
one robot or `False` to disable it for profiling or compatibility. Viser always
uses its instanced or color-grouped backbone representation because browser
scene-handle and message counts dominate that backend. Automatically sized
Viser ground planes are centered on the rendered bases and expanded to cover
the layout.

```python
renderer.render_sequence(
    ts,
    q_ts_batched,
    multi_robot_layout="overlay",
    color_config=color_config,
)
```

---

## Recording and Video Encoding

All renderer families accept `record_path` for sequence output, while their
capture mechanisms differ:

- Matplotlib uses its animation writers and requires FFmpeg for MP4 output.
- Open3D and Viser use FFmpeg with `VideoEncodingConfig`.
- Open3D writes PNG frames when `record_path` names a directory.
- Viser can capture synchronized browser snapshots with `snapshot_paths`.
- OpenCV uses FFmpeg when available and otherwise falls back to
  `cv2.VideoWriter`.

```python
from soromox.rendering import VideoEncodingConfig

renderer.render_sequence(
    ts,
    q_ts,
    record_path="trajectory.mp4",
    video_config=VideoEncodingConfig(crf=18, pix_fmt="yuv420p"),
)
```

Use `record_every_n` to subsample captured frames where supported. Recording
depends on the backend, so consult the relevant [renderer API](renderers.md)
for capture-only options.

::: soromox.rendering.video_encoding.VideoEncodingConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source
