# Shared scene presets in Open3D and Viser

Install the [rendering dependencies](../../tools/open3d/README.md). From the
repository root, generate the six presets in either backend:

```bash
python examples/rendering/preset_gallery.py --backend open3d
python examples/rendering/preset_gallery.py --backend viser
```

For Viser, open the printed local URL. The connected browser captures all six
images automatically. Each command writes 1920 × 1080 PNGs to `figures/presets/`.
The [preset gallery](../../docs/api/rendering/presets.md) compares the two backends. Use `--preset`,
`--count`, `--width`, `--height`, `--output-dir` or `--port` to customize a run.
The model, five prescribed poses, placement, palette and camera are shared in
`tentacle_scene.py`.

To inspect one preset or export a prescribed two-second motion:

```bash
python examples/rendering/preset_gallery.py --backend open3d --preset clay --count 1
python examples/rendering/preset_gallery.py --backend open3d --preset neutral --interactive
python examples/rendering/preset_gallery.py --backend open3d --preset neutral --video-output studio.mp4
python examples/rendering/preset_gallery.py --backend viser --preset bright --interactive
```

`--interactive` requires one preset. In Viser, the scene stays available until
Ctrl+C; browser camera controls and Save Canvas can be used to inspect and capture
other views. `--video-output` uses Open3D; browser video recording is available
through `ViserRenderer.render_sequence()`.

`--write-manifest` optionally saves the resolved configuration, backend version
and reproduction command beside the images. These generated JSON files are local
diagnostics and are ignored by Git. The twelve reviewed preset PNGs are versioned under
`docs/assets/rendering/presets/` to illustrate each preset. Generated example
captures stay local; use `--output-dir docs/assets/rendering/presets` when
deliberately updating the published gallery.

Open3D uses the modern renderer for images, synchronous videos and static
`show()`. Animated interactive previews use efficient legacy geometry updates
and warn about approximate appearance. Viser uses PBR geometry for static scenes
and editable mesh geometry for playback and live visualization.

`RendererConfig` composes scene, camera, colors, geometry and output settings.
The [configuration guide](../../docs/api/rendering/configuration.md) describes
physical light units, preset scaling, world floors and backend approximations.

The upright GVS model uses the Section Va soft tentacle's two link lengths
(0.305 m and 0.055 m) and tapered radii. The configurations are prescribed visual
examples, rather than simulated equilibria or fitted measurements. Geometry
comes from forward kinematics and shared cross-section lofting utilities.

The technical preset displays a world grid without a filled ground slab. Set
`config.scene.ground.surface = True` to add the surface. Grid spacing is in
metres, with major lines every five cells by default. Studio presets use a curved
backdrop, one directional key and point-light fills. Viser uses hemisphere
illumination to approximate Open3D's environment lighting.

Static Viser capture supplies the camera pose explicitly and waits for stable
images while the browser loads its meshes. The patched Open3D build uses Filmic
tone mapping to preserve neutral highlights. Lighting and shading still
differ between backends; the gallery describes these differences.

## Core renderer comparison

Generate the neutral studio tentacle used in the rendering overview:

```bash
python examples/rendering/backend_gallery.py --backend matplotlib
python examples/rendering/backend_gallery.py --backend open3d
python examples/rendering/backend_gallery.py --backend viser
python examples/rendering/backend_gallery.py --backend opencv
```

The default output is `docs/assets/rendering/`. OpenCV uses a planar PCS
counterpart with matching link lengths and prescribed bending coordinates.
Matplotlib displays colored lines on a white background with standard axes and
ignores scene appearance presets. OpenCV approximates the studio as colored
lines and a ground reference.
