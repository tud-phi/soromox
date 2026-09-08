"""Serve and capture the Open3D studio composition using ViserRenderer.

Run this script, open http://localhost:8080, then press Save studio PNG.
The image is rendered by the connected browser at 1920 x 1080.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
from open3d_studio import PALETTE, make_tentacle
from PIL import Image

from soromox.rendering import (
    BackboneColorConfig,
    CameraConfig,
    RendererColorConfig,
    ViserRenderer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "viser_studio.png",
    )
    args = parser.parse_args()
    q = np.array(
        [
            [0, 0.6, 2.2, 0, 0, 0, 0, -1.0, -4.5, 0, 0, 0],
            [0, -1.0, -3.1, 0, 0, 0, 0, 1.5, 5.0, 0, 0, 0],
            [0, 0.5, 3.6, 0, 0, 0, 0, -1.5, -7.5, 0, 0, 0],
            [0, -0.8, -2.0, 0, 0, 0, 0, 1.0, -5.0, 0, 0, 0],
            [0, 0.7, 2.8, 0, 0, 0, 0, -1.0, 5.0, 0, 0, 0],
        ]
    )
    offsets = np.array([[(i - 2) * 0.23, 0.015 * (i % 2), 0.012] for i in range(5)])
    renderer = ViserRenderer(
        make_tentacle(),
        host="127.0.0.1",
        port=args.port,
        open_browser=False,
        num_points=155,
        cross_section_resolution=64,
        show_ground_plane=False,
        base_plate_thickness=0.012,
        base_plate_radius_scale=0.028 / 0.01541,
        color_config=RendererColorConfig(
            # Open3D's material values are linear; Viser's color API is sRGB.
            backbone=BackboneColorConfig(
                robot_colors=1.055 * np.asarray(PALETTE) ** (1 / 2.4) - 0.055,
                segment_palette=None,
            ),
            base_plate_color=(0.16, 0.18, 0.18),
        ),
        enable_default_lights=False,
        add_directional_light=True,
        directional_light_direction=(-0.25, 0.15, -1.0),
        directional_light_intensity=1.2,
        directional_light_color=(255, 247, 240),
        ambient_light_intensity=0.8,
        background_color=(0.48, 0.48, 0.48),
    )
    renderer.show(
        q,
        base_offsets=offsets,
        render_actuators=False,
        blocking=False,
        camera_config=CameraConfig(
            fov=35,
            position=(0.138, -1.15, 0.62),
            look_at=(0, 0, 0.16),
            up=(0, 0, 1),
        ),
    )
    server = renderer.server
    server.scene.world_axes.visible = False
    # Avoid a network dependency on a remotely hosted HDR environment map.
    server.scene.configure_environment_map(None)
    angles = np.linspace(0, np.pi / 2, 80)
    profile = [(-3.0, 0.0), (0.5, 0.0)]
    profile.extend(zip(0.5 + 0.6 * np.sin(angles[1:]), 0.6 * (1 - np.cos(angles[1:]))))
    profile.append((1.1, 3.0))
    vertices = [[x, y, z] for y, z in profile for x in (-3.0, 3.0)]
    faces = [
        face
        for i in range(len(profile) - 1)
        for face in ([2 * i, 2 * i + 1, 2 * i + 3], [2 * i, 2 * i + 3, 2 * i + 2])
    ]
    backdrop = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    backdrop.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[100, 100, 100, 255],
            roughnessFactor=1.0,
            metallicFactor=0.0,
        )
    )
    server.scene.add_mesh_trimesh(
        "/studio/backdrop",
        backdrop,
        cast_shadow=False,
        receive_shadow=True,
    )
    server.scene.add_light_point(
        "/studio/fill",
        color=(255, 250, 245),
        position=(-0.3, -0.4, 1.1),
        intensity=2.5,
        distance=4.0,
        cast_shadow=False,
    )

    @server.gui.add_button("Save studio PNG").on_click
    def save(event):
        if event.client is None:
            return
        pixels = event.client.camera.get_render(
            height=1080, width=1920, transport_format="png"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.output)
        print(f"Saved {args.output.resolve()}", flush=True)

    print(f"Open http://localhost:{args.port} and press Save studio PNG.", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        renderer.stop()


if __name__ == "__main__":
    main()
