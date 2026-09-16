"""Shared prescribed tentacle geometry, poses and camera for renderer comparisons."""

import jax
import numpy as np

from soromox.rendering import (
    BackboneColorConfig,
    CameraConfig,
    GeometryConfig,
    RendererColorConfig,
    RendererConfig,
    RenderOutputConfig,
    SceneConfig,
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
BASE_PLATE_THICKNESS = 0.012


def make_tentacle(mounting="upright"):
    """Construct two tapered GVS links with the Section Va external dimensions.

    Args:
        mounting: Upright or hanging spatial mounting.

    Returns:
        GVS tentacle with two fixed joints and constant six-component
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
        base_pose=spatial_mounting_pose(mounting),
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

PRESETS = ("technical", "neutral", "bright", "dark", "flat", "clay")


def make_comparison(
    preset="neutral", *, count=5, width=1920, height=1080, mounting="upright"
):
    """Prepare one preset with identical robot geometry and camera framing.

    Args:
        preset: Name from PRESETS.
        mounting: Upright or hanging; rotates scenery and camera together.
        count: One central tentacle or all five prescribed tentacles.
        width: Export width in pixels.
        height: Export height in pixels.

    Returns:
        Tuple of RendererConfig, configurations (N, 12), and base offsets (N, 3).

    Raises:
        ValueError: Preset name or robot count is invalid.
    """
    if mounting not in ("upright", "hanging"):
        raise ValueError("mounting must be upright or hanging")
    if preset not in PRESETS or count not in (1, 5):
        raise ValueError("Choose a known preset and a robot count of one or five")
    indices = np.arange(5) if count == 5 else np.array([2])
    offsets = np.array(
        [[(i - 2) * 0.23 if count == 5 else 0, 0.015 * (i % 2), 0.0] for i in indices]
    )
    distance = 1.15 if count == 5 else 0.70
    config = (
        RendererConfig.clay()
        if preset == "clay"
        else RendererConfig(
            scene=SceneConfig.technical()
            if preset == "technical"
            else SceneConfig.flat()
            if preset == "flat"
            else SceneConfig.studio(style=preset)
        )
    )
    config.scene.ground.height_reference = "base_mounting_face"
    config.camera = CameraConfig(
        fov=35,
        position=(0.0, -distance, 0.16 + 0.4 * distance),
        look_at=(0, 0, 0.16),
    )
    config.geometry = GeometryConfig(
        num_points=155,
        cross_section_resolution=48,
        base_plate_radius_scale=0.028 / 0.01541,
        base_plate_thickness=BASE_PLATE_THICKNESS,
    )
    config.output = RenderOutputConfig(width=width, height=height)
    if preset != "clay":
        config.colors = RendererColorConfig(
            backbone=BackboneColorConfig(
                robot_colors=np.array(PALETTE)[indices], segment_palette=None
            ),
            base_plate_color=(0.16, 0.18, 0.18),
        )
    if mounting == "hanging":
        rotation = np.diag([-1.0, 1.0, -1.0])
        config.scene.ground.normal = (0.0, 0.0, -1.0)
        config.camera.position = tuple(rotation @ config.camera.position)
        config.camera.look_at = tuple(rotation @ config.camera.look_at)
        offsets = offsets @ rotation.T
    return config, POSES[indices], offsets
