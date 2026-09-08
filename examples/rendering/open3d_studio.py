"""Render upright soft tentacles in a pastel Open3D studio.

Run from the repository root:
    .venv/bin/python examples/rendering/open3d_studio.py

This is a static visual study, using prescribed GVS configurations and the
Section Va tentacle dimensions, rather than simulated equilibrium poses.
Open3D's Filament renderer supplies lighting, shadows and ambient occlusion.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import open3d as o3d

from soromox.rendering import (
    BackboneColorConfig,
    CameraConfig,
    Open3DRenderConfig,
    Open3DRenderer,
    RendererColorConfig,
)
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.utils.geometry.poses import spatial_mounting_pose

jax.config.update("jax_enable_x64", True)

PALETTE = [
    (0.76, 0.77, 0.36),
    (0.36, 0.70, 0.60),
    (0.65, 0.38, 0.78),
    (0.27, 0.48, 0.85),
    (0.78, 0.28, 0.43),
]


def make_tentacle():
    """Construct two tapered GVS links with the Section Va external dimensions.

    Returns:
        Upright GVS tentacle with two fixed joints and constant six-component
        strain coordinates per link. Lengths and radii are specified in metres.
    """
    links = [
        LinkSpec.circular(
            length=length,
            radius=LinearProfile(base=base, tip=tip),
            density=1500.0,
            young_modulus=5.05e5,
            poisson_ratio=0.45,
            reference_strain=[0, 0, 0, 1, 0, 0],
        )
        for length, base, tip in [
            (0.305, 0.01541, 0.00642),
            (0.055, 0.00642, 0.00480),
        ]
    ]
    return GVS.from_segments(
        [
            GVSSegment(
                link=link,
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[1] * 6,
                    basis_order=[0] * 6,
                ),
                num_gauss_points=8,
            )
            for link in links
        ],
        base_pose=spatial_mounting_pose("upright"),
    )


POSES = np.array(
    [
        [0, 0.6, 2.2, 0, 0, 0, 0, -1.0, -4.5, 0, 0, 0],
        [0, -1.0, -3.1, 0, 0, 0, 0, 1.5, 5.0, 0, 0, 0],
        [0, 0.5, 3.6, 0, 0, 0, 0, -1.5, -7.5, 0, 0, 0],
        [0, -0.8, -2.0, 0, 0, 0, 0, 1.0, -5.0, 0, 0, 0],
        [0, 0.7, 2.8, 0, 0, 0, 0, -1.0, 5.0, 0, 0, 0],
    ]
)


def main():
    """Parse command-line options and export the prescribed studio scene.

    Returns:
        None. Writes a PNG, optionally exports an MP4, and optionally opens a
        modern static viewer. Output paths and dimensions come from CLI options.

    Raises:
        RuntimeError: Open3D capture, image writing or video encoding fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "open3d_studio.png",
    )
    parser.add_argument(
        "--video-output", type=Path, help="Also export a short prescribed motion as MP4"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the modern static viewer after exporting",
    )
    parser.add_argument("--count", type=int, choices=(1, 5), default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    indices = np.arange(5) if args.count == 5 else np.array([2])
    poses = POSES[indices]
    offsets = np.array(
        [
            [(index - 2) * 0.23 if args.count == 5 else 0.0, 0.015 * (index % 2), 0.012]
            for index in indices
        ]
    )
    colors = RendererColorConfig(
        backbone=BackboneColorConfig(robot_colors=np.array(PALETTE)[indices]),
        base_plate_color=(0.16, 0.18, 0.18),
    )
    renderer = Open3DRenderer(
        make_tentacle(),
        width=args.width,
        height=args.height,
        num_points=155,
        cross_section_resolution=48,
        base_plate_radius_scale=0.028 / 0.01541,
        base_plate_thickness=0.012,
        color_config=colors,
        render_config=Open3DRenderConfig.studio(),
    )
    distance = 1.15 if args.count == 5 else 0.70
    camera = CameraConfig(
        fov=35.0,
        look_at=(0, 0, 0.16),
        position=(0.12 * distance, -distance, 0.16 + 0.40 * distance),
    )
    image = renderer.render_frame(poses, base_offsets=offsets, camera_config=camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(args.output), o3d.geometry.Image(image)):
        raise RuntimeError(f"Could not save {args.output}")
    print(f"Saved {args.output.resolve()}")
    if args.video_output is not None:
        ts = np.arange(60) / 30.0
        # Prescribed motion, with the first frame equal to the still-image poses.
        trajectory = (
            poses[:, None, :] * (1 + 0.15 * np.sin(2 * np.pi * ts))[None, :, None]
        )
        renderer.render_sequence(
            ts,
            trajectory,
            base_offsets=offsets,
            camera_config=camera,
            record_path=str(args.video_output),
        )
        print(f"Saved {args.video_output.resolve()}")
    if args.interactive:
        renderer.show(poses, base_offsets=offsets, camera_config=camera)


if __name__ == "__main__":
    main()
