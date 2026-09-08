"""Render upright soft tentacles in a pastel Open3D studio.

Run from the repository root:
    .venv/bin/python examples/rendering/open3d_studio.py

This is a static visual study, using prescribed GVS configurations and the
Section Va tentacle dimensions, rather than simulated equilibrium poses.
Open3D's Filament renderer supplies lighting, shadows and ambient occlusion.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d

from soromox.rendering import evaluate_cross_sections, loft_cross_sections
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
    """Two tapered GVS links with the Section Va external dimensions (metres)."""
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


def tentacle_mesh(robot, q, offset):
    """Use the shared cross-section loft, preserving taper and material frames."""
    # Include the link junction exactly; the external radius is continuous here.
    stations = np.r_[np.linspace(0, 0.305, 125), np.linspace(0.305, 0.36, 30)[1:]]
    transforms = np.asarray(
        robot.forward_kinematics_abscissa_batched(jnp.asarray(q), jnp.asarray(stations))
    )
    sections = evaluate_cross_sections(robot, q, stations)
    mesh = o3d.geometry.TriangleMesh()
    for index in range(len(stations) - 1):
        vertices, faces = loft_cross_sections(
            transforms[index, :3, 3],
            transforms[index + 1, :3, 3],
            transforms[index, :3, :3],
            transforms[index + 1, :3, :3],
            sections[index],
            sections[index + 1],
            64,
            cap_start=index == 0,
            cap_end=index == len(stations) - 2,
        )
        mesh += o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    mesh.translate(offset)
    return mesh


def material(color, roughness=0.72):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [*color, 1.0]
    mat.base_roughness = roughness
    mat.base_metallic = 0.0
    mat.base_reflectance = 0.35
    return mat


def build_scene(scene, count):
    robot = make_tentacle()
    # Curvatures in 1/m; no extension or shear. Each row gives two link strains.
    poses = [
        [0, 0.6, 2.2, 0, 0, 0, 0, -1.0, -4.5, 0, 0, 0],
        [0, -1.0, -3.1, 0, 0, 0, 0, 1.5, 5.0, 0, 0, 0],
        [0, 0.5, 3.6, 0, 0, 0, 0, -1.5, -7.5, 0, 0, 0],
        [0, -0.8, -2.0, 0, 0, 0, 0, 1.0, -5.0, 0, 0, 0],
        [0, 0.7, 2.8, 0, 0, 0, 0, -1.0, 5.0, 0, 0, 0],
    ]
    indices = range(5) if count == 5 else [2]
    for index in indices:
        x = (index - 2) * 0.23 if count == 5 else 0.0
        offset = [x, 0.015 * (index % 2), 0.012]
        scene.add_geometry(
            f"tentacle_{index}",
            tentacle_mesh(robot, poses[index], offset),
            material(PALETTE[index]),
        )
        base = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.028, height=0.012, resolution=96
        )
        base.compute_vertex_normals()
        base.translate([offset[0], offset[1], 0.006])
        scene.add_geometry(f"mount_{index}", base, material((0.16, 0.18, 0.18)))

    # Sweep the floor into a curved studio wall so there is no horizon seam.
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
    floor = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    floor.compute_vertex_normals()
    scene.add_geometry("floor", floor, material((0.48, 0.48, 0.48), 1.0))
    scene.set_background([0.48, 0.48, 0.48, 1.0])
    scene.show_skybox(False)
    scene.set_lighting(scene.SOFT_SHADOWS, [-0.5, 0.3, -1.0])
    scene.scene.set_sun_light([-0.25, 0.15, -1.0], [1.0, 0.97, 0.94], 60000)
    scene.scene.enable_sun_light(True)
    scene.scene.set_indirect_light_intensity(60000)
    scene.scene.add_point_light(
        "softbox_fill", [1.0, 0.98, 0.96], [-0.3, -0.4, 1.1], 35000, 4.0, False
    )
    scene.view.set_post_processing(True)
    scene.view.set_ambient_occlusion(True)
    scene.view.set_antialiasing(True)
    scene.view.set_shadowing(True, o3d.visualization.rendering.View.ShadowType.VSM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "open3d_studio.png",
    )
    parser.add_argument("--count", type=int, choices=(1, 5), default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    # macOS wheels lack EGL headless rendering; use a native graphics context.
    window = None
    if sys.platform == "darwin":
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        window = app.create_window("SoRoMoX studio", args.width, args.height)
        scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    else:
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            args.width, args.height
        )
        scene = renderer.scene
    build_scene(scene, args.count)
    distance = 1.15 if args.count == 5 else 0.70
    scene.camera.look_at(
        [0, 0, 0.16], [0.12 * distance, -distance, 0.16 + 0.40 * distance], [0, 0, 1]
    )
    scene.camera.set_projection(
        35.0,
        args.width / args.height,
        0.01,
        250.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if window is not None:
        image = app.render_to_image(scene, args.width, args.height)
    else:
        image = renderer.render_to_image()
    if not o3d.io.write_image(str(args.output), image):
        raise RuntimeError(f"Could not save {args.output}")
    print(f"Saved {args.output.resolve()}")
    if window is not None:
        window.close()
        app.run_one_tick()


if __name__ == "__main__":
    main()
