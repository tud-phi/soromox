# Rendering

Visualization and rendering utilities for soft robot systems.

## Overview

The rendering module provides a class-based architecture for visualizing soft robots. All renderers inherit from a common base class `BaseSoftRobotRenderer` which provides cached forward kinematics and a unified interface.

### Architecture

```text
BaseSoftRobotRenderer (abstract base)
├── MatplotlibRenderer         # Generic, works with any robot (2D/3D)
├── Open3DRenderer             # Interactive 3D visualization
├── ViserRenderer              # Web-based interactive 3D visualization
├── OpenCVPlanarRenderer       # Fast 2D rendering
└── OpenCVPlanarHSARenderer    # Specialized for PlanarHSA
```

## Choosing a Renderer

| Renderer | Best For | Strengths | Limitations |
|----------|----------|-----------|-------------|
| `MatplotlibRenderer` | Static plots, notebooks, debugging | Works with any robot, easy to embed | Limited interactivity, slower animations |
| `Open3DRenderer` | Interactive 3D inspection | Rich camera controls, headless capture | Requires Open3D, 3D only |
| `ViserRenderer` | Live demos, web-based visualization | Browser UI, live streaming, multi-robot | Requires web server, 3D only |
| `OpenCVPlanarRenderer` | Fast 2D animations, video export | Lightweight, fast output | Planar robots only |
| `OpenCVPlanarHSARenderer` | PlanarHSA-specific visuals | HSA-specific geometry | PlanarHSA only |

## Documentation

### [Renderers](renderers.md)

Complete API reference for all renderer classes:

- `BaseSoftRobotRenderer` - Abstract base class
- `MatplotlibRenderer` - Matplotlib-based visualization
- `Open3DRenderer` - Open3D-based 3D visualization
- `ViserRenderer` - Web-based visualization
- `OpenCVPlanarRenderer` - OpenCV-based 2D visualization
- `OpenCVPlanarHSARenderer` - HSA-specific visualization

### [Camera & Colors](configuration.md)

Configuration options for camera positioning and color schemes:

- `CameraConfig` - Camera positioning and field of view
- `RendererColorConfig` - Color configuration hierarchy
- `BackboneColorConfig` - Backbone-specific colors
- Built-in color palettes and themes

## Quick Start

=== "Matplotlib (any robot)"

    ```python
    from soromox.rendering import MatplotlibRenderer

    renderer = MatplotlibRenderer(robot, num_points=50)
    renderer.show(q)  # Single frame
    renderer.animate(ts, q_ts, mode="slider")  # Interactive
    ```

=== "Open3D (3D robots)"

    ```python
    from soromox.rendering import Open3DRenderer

    renderer = Open3DRenderer(robot, num_points=80)
    renderer.show(q)  # Interactive viewer
    renderer.render_sequence(ts, q_ts, record_path="video.mp4")
    ```

=== "Viser (web-based)"

    ```python
    from soromox.rendering import ViserRenderer

    renderer = ViserRenderer(robot, port=8080)
    renderer.show(q)  # Opens browser
    renderer.render_sequence(ts, q_ts, loop=True)
    ```

=== "OpenCV (planar)"

    ```python
    from soromox.rendering import OpenCVPlanarRenderer

    renderer = OpenCVPlanarRenderer(robot, width=800, height=600)
    img = renderer.render_frame(q)
    renderer.render_sequence(ts, q_ts, record_path="video.mp4")
    ```
