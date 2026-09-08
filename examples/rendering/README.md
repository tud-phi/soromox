# Open3D and Viser studio examples

Run from the repository root with the rendering dependencies installed:

```bash
python examples/rendering/open3d_studio.py
python examples/rendering/open3d_studio.py --count 1 --output examples/rendering/figures/open3d_studio_single.png
python examples/rendering/open3d_studio.py --video-output examples/rendering/videos/open3d_studio.mp4
```

The default output is `figures/open3d_studio.png` beside the script, at
1920 × 1080. Use `--width`, `--height`, and `--output` to change the export.
Install the [Open3D development build](../../tools/open3d/README.md) first. Image
and video exports use `OffscreenRenderer`; macOS uses the patched Metal build.
Other platforms need a working graphics driver. Video export needs `ffmpeg`.

The example uses `Open3DRenderer` with `Open3DRenderConfig.studio()`: matte
pastel materials, a grey floor curved into a backdrop, directional shadows,
ambient occlusion, environment illumination, and a local fill light. Set
`render_config=Open3DRenderConfig.studio(roughness=0.8, sun_intensity=50000)`
on your renderer to adapt it. `RendererColorConfig` sets the colors; `CameraConfig` sets the camera. The backdrop uses world Z as up; `floor_z` sets its height.
The shadows use variance shadow maps; the fill light does not cast shadows.
This approximates a studio photograph, with some shadow-map softness and bias
artifacts possible. It does not reproduce Cycles' path-traced area lights.

The upright GVS model uses the Section Va soft tentacle's two link lengths
(305 and 55 mm) and tapered radii (15.41 → 6.42 → 4.80 mm), with constant
strain on each link. The five configurations are prescribed visual examples,
not equilibria, fitted measurements, or reruns of the system-identification
experiment. Geometry comes from the robot's forward kinematics and shared
cross-section lofting helpers. `--video-output` exports a prescribed two-second
motion using the same renderer settings. `--interactive` opens a modern static viewer after
exporting the image.

`render_frame()` and `render_sequence(..., record_path=...)` use modern rendering.
Recordings run synchronously and return after export, independently of playback
controls. `record_every_n` selects frames and reduces the output FPS by the same
factor; `playback_speed` scales FPS. Nonuniform timestamps produce a warning
because video uses the median interval; resample first for exact timing.

`show()` uses the modern GUI with the same materials, lighting and backdrop.
It supports camera orbit, pan, zoom, reset, save/restore and modern snapshots.
Sequences without `record_path` use the legacy viewer and warn about approximate
materials, lighting, transparency, shadows and ambient occlusion. Camera fitting
for exports uses the whole trajectory and excludes the backdrop.

The visual reference is the [EgoHumans teaser](https://rawalkhirodkar.github.io/egohumans/).
Blender Cycles is the most likely renderer: the authors publish Blender scenes
and their [rendering script](https://github.com/rawalkhirodkar/egohumans/blob/main/egohumans/lib/utils/blender.py#L293)
explicitly sets `bpy.context.scene.render.engine = 'CYCLES'`. This establishes
their use of Cycles, but does not prove which engine generated the exact teaser.

## Viser comparison

```bash
python examples/rendering/viser_studio.py --port 8087
```

Open `http://localhost:8087` and press **Save studio PNG** to capture the
connected browser's view at 1920 × 1080. The default output is
`figures/viser_studio.png`; `--output` changes its destination. Drag to orbit
the scene before saving if a different view is wanted. Stop with Ctrl+C.

This uses the `ViserRenderer.show()` geometry pipeline, the same five
GVS poses and camera as the Open3D study, a custom curved backdrop, and Viser
directional/ambient/point lights. Viser 1.0.26 was used for the browser capture.
The preset adapts palette values to Viser's sRGB color API and tunes light
intensities for its renderer. It uses real cast shadows, but does not reproduce
Open3D's ambient occlusion or material response exactly. Capped link boundaries are visible near the tentacle tips.

The prototype imports the robot definition and palette from `open3d_studio.py`,
so it requires both examples' dependencies, including Open3D. Rendering and PNG
capture themselves are performed by Viser in the browser. The server binds only
to the local machine, and the scene uses no remote HDR environment map.
