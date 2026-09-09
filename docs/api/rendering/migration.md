# Renderer migration guide

The shared configuration API replaces appearance, geometry and output arguments
on renderer constructors. This is a breaking change: update constructors and
direct configuration imports when upgrading from SoRoMoX 0.4.1.

## Move constructor settings into RendererConfig

For an existing `robot`, a constructor using the previous API might look like:

```python
from soromox.rendering import Open3DRenderer

renderer = Open3DRenderer(
    robot,
    width=1920,
    height=1200,
    num_points=80,
    cross_section_resolution=20,
    background_color=(1.0, 1.0, 1.0),
    show_ground_plane=True,
    ground_plane_size=1.2,
)
```

Supply those settings through configuration sections:

```python
from soromox.rendering import (
    GeometryConfig,
    GroundPlaneConfig,
    Open3DRenderer,
    RendererConfig,
    RenderOutputConfig,
    SceneConfig,
)

config = RendererConfig(
    scene=SceneConfig.technical(
        background=(1.0, 1.0, 1.0),
        ground=GroundPlaneConfig(
            visible=True, surface=True, alignment="base", size=1.2,
        ),
    ),
    geometry=GeometryConfig(num_points=80, cross_section_resolution=20),
    output=RenderOutputConfig(width=1920, height=1200),
)
renderer = Open3DRenderer(robot, config=config)
```

This explicitly selects base-aligned filled ground planes and the previous
Open3D dimensions and sample counts. Lighting and floor placement can differ
from the old renderer. Omit these explicit values to use the new defaults:
800 × 600 pixels, 80 backbone samples, 48 cross-section samples, and a technical
scene with a world-aligned grid and no filled ground surface. Spatial ground
uses +Z; planar ground uses +Y. To hide the ground, use
`GroundPlaneConfig(visible=False)`.

The same sections work with Viser, Matplotlib and OpenCV. Backend controls such
as Viser's `host`, `port` and `open_browser`, or OpenCV's `length_scale`, are
still constructor arguments. The renderer copies its configuration, so edit
the configuration before constructing the renderer.

| Previous setting | New location |
| --- | --- |
| `width`, `height` | `config.output.width`, `config.output.height` |
| `background_color` | `config.scene.background` |
| `color_config` on the constructor | `config.colors` |
| `num_points`, `cross_section_resolution`, `backbone_style` | Matching fields in `config.geometry` |
| Base plate dimensions, actuator line width, layout `grid_spacing` | Matching fields in `config.geometry` |
| `show_ground_plane`, `ground_plane_size` | `config.scene.ground.visible`, `.size` |
| Ground colors in `RendererColorConfig` | `config.scene.ground.color`, `.grid_color`, `.grid_major_color` |
| Viser `camera_fov` | `config.camera.fov` |
| Viser `material`, `flat_shading`, `wireframe` | `config.scene.material.shading`, `.flat_shading`, `.wireframe` |
| Viser light and shadow constructor arguments | `config.scene.lights`, `.ambient`, `.shadows` and geometry shadow flags |

Light intensities now use directional lux and point-light candela. Previous
Viser relative intensities cannot be copied directly. Start with a scene preset
and adjust it; `PointLightConfig.from_lumens()` converts isotropic luminous flux
to candela. Public colors use sRGB floats in `[0, 1]`, including light colors.
See [shared configuration](configuration.md) for units and backend limitations.

## Update direct imports

Imports from `soromox.rendering` remain supported. Direct configuration imports
use the following modules; the old camera and color modules have been removed.

| Previous import module | New module |
| --- | --- |
| `soromox.rendering.camera_config` | `soromox.rendering.config.camera` |
| `soromox.rendering.color_config` | `soromox.rendering.config.colors` |
| `VideoEncodingConfig` from `soromox.rendering.video_encoding` | `soromox.rendering.config.output` |

`FFmpegVideoWriter` is in `soromox.rendering.video_encoding`. Public configuration
types can also be imported together from `soromox.rendering.config`.

## Keep per-call overrides explicit

Camera, color and video overrides replace the corresponding complete default
section for that call. To adjust only selected fields, derive an override from
the defaults with `dataclasses.replace`:

```python
from dataclasses import replace

camera = replace(renderer.config.camera, fov=45.0)
image = renderer.render_frame(q, camera_config=camera)

video = replace(renderer.config.output.video, crf=18)
renderer.render_sequence(ts, q_ts, record_path="motion.mp4", video_config=video)
```

Trajectories, `base_offsets`, recording paths and playback controls are operation
arguments. OpenCV sequence options are keyword-only, and `record_path` is required:

```python
opencv_renderer.render_sequence(
    ts, q_ts, record_path="motion.mp4", playback_speed=1.0,
)
```

## Separate Open3D export from animated playback

`show(q)` opens the modern static viewer. `render_frame(q)` and
`render_sequence(ts, q_ts, record_path=...)` use modern image rendering. Recorded
sequences export synchronously and return when the file is complete; they do
not open an interactive playback window. `close_when_recording_done` and viewer
keyboard controls no longer control exports.

To export and then inspect an animation, make two calls:

```python
renderer.render_sequence(ts, q_ts, record_path="motion.mp4")
renderer.render_sequence(ts, q_ts)
```

Sequences without `record_path` use the legacy animated viewer and warn about
appearance differences. Run modern static GUI viewing and legacy animated
viewing in separate processes because of the current Open3D lifecycle limitation.

For exports, `record_every_n=n` selects every nth frame and divides the output
FPS by n, preserving nominal duration. `playback_speed` scales output FPS.
Nonuniform timestamps use the median interval and produce a warning; resample
the trajectory first when exact timing is required. See the
[Open3D installation instructions](../../installation.md) for the required
development build.
