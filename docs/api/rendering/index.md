# Rendering

Visualization and rendering utilities for soft robot systems.

## Overview

The rendering module provides a class-based architecture for visualizing soft robots. All renderers inherit from a common base class `BaseSoftRobotRenderer` which provides cached forward kinematics and a unified interface.

### Core renderer gallery

The Matplotlib, Open3D, and Viser samples show the same three-segment spatial
PCS configuration with the paper color theme, a dark base, and a base-aligned
ground plane. The OpenCV sample uses its three-segment planar counterpart with
the same dark-base and warm-backbone styling.

<div class="grid cards soromox-gallery" markdown>

-   **Matplotlib**

    ![Spatial continuum robot rendered in three dimensions with Matplotlib](../../assets/rendering/matplotlib.png)

    Polished static figures and notebook-friendly inspection for planar and
    spatial robots.

    [Matplotlib renderer details](renderers.md#matplotlib-renderer)

-   **Open3D**

    ![Spatial continuum robot rendered with Open3D](../../assets/rendering/open3d.png)

    Interactive spatial inspection with geometry, camera control, and offline
    frame or video capture.

    [Open3D renderer details](renderers.md#open3d-renderer)

-   **Viser**

    ![Spatial continuum robot rendered in the Viser browser viewport](../../assets/rendering/viser.png)

    Browser-based 3D visualization for live demos, playback, and multi-robot
    scenes.

    [Viser renderer details](renderers.md#viser-renderer)

-   **OpenCV Planar**

    ![Planar continuum robot frame rendered with OpenCV](../../assets/rendering/opencv-planar.png)

    Lightweight real-time planar frames and fast video export.

    [OpenCV Planar renderer details](renderers.md#opencv-planar-renderer)

</div>

System-specific renderer designs are documented alongside their robots: the
[Planar HSA OpenCV renderer](../systems/hsa/planar-hsa.md),
[I-SUPPORT Viser renderer](../systems/pcs/isupport.md), and
[UMArm Viser renderer](../systems/articulated/mckibben-umarm.md).

## Choosing a Renderer

| Renderer | Best For | Strengths | Limitations |
|----------|----------|-----------|-------------|
| `MatplotlibRenderer` | Static plots, notebooks, debugging | Works with any robot, easy to embed | Limited interactivity, slower animations |
| `Open3DRenderer` | Interactive 3D inspection | Rich camera controls, headless capture | Requires Open3D, 3D only |
| `ViserRenderer` | Live demos, web-based visualization | Browser UI, live streaming, multi-robot | Requires web server, 3D only |
| `OpenCVPlanarRenderer` | Fast 2D animations, video export | Lightweight, fast output | Planar robots only |
| `OpenCVPlanarHSARenderer` | PlanarHSA-specific visuals | HSA-specific geometry | PlanarHSA only |

## Documentation

### [Renderer API](renderers.md)

Complete API reference for all renderer classes:

- `BaseSoftRobotRenderer` - Abstract base class
- `MatplotlibRenderer` - Matplotlib-based visualization
- `Open3DRenderer` - Open3D-based 3D visualization
- `ViserRenderer` - Web-based visualization
- `OpenCVPlanarRenderer` - OpenCV-based 2D visualization
- `OpenCVPlanarHSARenderer` - HSA-specific visualization

### [Shared Configuration](configuration.md)

Settings shared by multiple renderer backends:

- `CameraConfig` - Camera positioning and field of view
- Robot base, base-plate, and ground-plane behavior
- Multi-robot layouts and spacing
- Recording and video encoding
- `RendererColorConfig` - Color configuration hierarchy
- `BackboneColorConfig` - Backbone-specific colors
- Built-in color palettes and themes

## Quick Start

All renderers accept a robot model at construction and expose a single-frame
entry point. For example:

```python
from soromox.rendering import MatplotlibRenderer

renderer = MatplotlibRenderer(robot, num_points=50)
renderer.show(q)
```

Choose another backend from the table above, then follow its
[API reference](renderers.md) for interactive playback, live mode, or video
capture.
