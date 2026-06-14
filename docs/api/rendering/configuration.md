# Camera & Colors Configuration

This page documents the camera and color configuration options for SoRoMoX renderers.

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
| `fov` | float | 60.0 | Field of view in degrees |
| `position` | tuple | None | Explicit camera position (x, y, z), None for auto |
| `look_at` | tuple | None | Point camera looks at, None for scene center |
| `up` | tuple | (0, 0, 1) | Camera up vector |
| `distance_factor` | float | 2.0 | Multiplier for auto-positioning distance |
| `position_offset` | tuple | None | Direction vector for camera placement |

::: soromox.rendering.camera_config.CameraConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_section_style: table
      members_order: source

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
    actuators=ActuatorStyleConfig(default_color=(0.8, 0.2, 0.2)),
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
