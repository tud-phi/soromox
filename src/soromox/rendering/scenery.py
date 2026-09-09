"""Backend-independent scene bounds, floor placement and backdrop meshes."""

from dataclasses import replace

import numpy as np

from soromox.rendering.config import (
    DirectionalLightConfig,
    GroundPlaneConfig,
    SceneConfig,
)


def srgb_to_linear(color):
    """Convert sRGB components to linear light, preserving array shape.

    Args:
        color: Scalar or array of normalized sRGB components.

    Returns:
        NumPy array containing linear-light components.
    """
    c = np.asarray(color, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(color):
    """Convert linear-light components to sRGB, preserving array shape.

    Args:
        color: Scalar or array of normalized linear-light components.

    Returns:
        NumPy array containing sRGB components.
    """
    c = np.asarray(color, dtype=float)
    return np.where(
        c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055
    )


def plane_basis(normal):
    """Construct an orthonormal floor basis with a stable world orientation.

    Args:
        normal: Nonzero three-vector normal to the plane.

    Returns:
        Matrix whose columns are width, depth and normalized normal. Depth is
        projected world +Y, with a +X-width fallback at normals parallel to Y.
        The fallback is a coordinate singularity, not a continuous frame choice.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    # Project world +Y into the plane to preserve backdrop depth continuously
    # across the XZ plane, including upright and hanging mounts.
    u = np.cross([0.0, 1.0, 0.0], n)
    if np.linalg.norm(u) < 1e-12:
        # Depth is undefined at +/-Y. Retain the canonical planar convention
        # with width along +X; no normal-only frame can be continuous everywhere.
        u = np.array([1.0, 0.0, 0.0])
        u -= (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.column_stack((u, v, n))


def _eased_backdrop_profile(config) -> np.ndarray:
    """Sample a bend with independent reach, height and curvature easing.

    Args:
        config: Validated BackdropConfig. All lengths are in scene extents.

    Returns:
        Array of (depth, height) coordinates, including a flat floor and wall.
        Tangents meet the floor and wall continuously. Flat sections use similar
        spacing near the bend to avoid bias in computed vertex normals.
    """
    vertical = (
        config.radius if config.vertical_radius is None else config.vertical_radius
    )
    t = np.linspace(0, 1, 257)
    theta = (
        np.pi / 2 * (t - config.curvature_easing * np.sin(2 * np.pi * t) / (2 * np.pi))
    )
    tangent = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    curve = np.vstack(
        (np.zeros(2), np.cumsum((tangent[:-1] + tangent[1:]) / 2, axis=0))
    )
    curve *= np.array([config.radius, vertical]) / curve[-1]
    curve[:, 0] += config.wall_offset
    step = np.linalg.norm(np.diff(curve, axis=0), axis=1).mean()
    # Bound scenery memory for unusually large floor/wall dimensions.
    floor_steps = int(
        np.clip(np.ceil((config.depth + config.wall_offset) / step), 1, 4096)
    )
    wall_steps = int(np.clip(np.ceil((config.height - vertical) / step), 1, 4096))
    floor_y = np.linspace(-config.depth, config.wall_offset, floor_steps + 1)
    wall_z = np.linspace(vertical, config.height, wall_steps + 1)
    wall = (
        np.column_stack(
            (np.full(wall_steps, config.wall_offset + config.radius), wall_z[1:])
        )
        if config.height > vertical
        else np.empty((0, 2))
    )
    return np.vstack(
        (
            np.column_stack((floor_y[:-1], np.zeros(floor_steps))),
            curve,
            wall,
        )
    )


def backdrop_mesh(scene: SceneConfig, center, extent, normal, *, ground_height=None):
    """Build a curved floor/wall from shared dimensions and floor orientation.

    Args:
        scene: Scene configuration supplying ground and backdrop parameters.
        center: Robot scene center as a world three-vector.
        extent: Positive robot scene extent in metres.
        normal: World floor normal.
        ground_height: Optional resolved floor height. Defaults to the scene
            configuration's explicit world height.

    Returns:
        Tuple of vertices with shape (N, 3) and triangle indices with shape (M, 3).
    """
    cfg = scene.backdrop
    basis = plane_basis(normal)
    n = basis[:, 2]
    if ground_height is None:
        if scene.ground.height_reference != "world":
            raise ValueError(
                "ground_height is required when height_reference is not world"
            )
        ground_height = scene.ground.height
    height = ground_height
    origin = np.asarray(center) - np.dot(center, n) * n + height * n
    if cfg.vertical_radius is None and cfg.curvature_easing == 0:
        angles = np.linspace(0, np.pi / 2, 80)
        profile = [(-cfg.depth, 0), (cfg.wall_offset, 0)]
        profile.extend(
            zip(
                cfg.wall_offset + cfg.radius * np.sin(angles[1:]),
                cfg.radius * (1 - np.cos(angles[1:])),
            )
        )
        profile.append((cfg.wall_offset + cfg.radius, cfg.height))
    else:
        profile = _eased_backdrop_profile(cfg)
    vertices = np.array(
        [[x, y, z] for y, z in profile for x in (-cfg.width / 2, cfg.width / 2)]
    )
    vertices = (vertices * extent) @ basis.T + origin
    faces = np.array(
        [
            f
            for i in range(len(profile) - 1)
            for f in ([2 * i, 2 * i + 1, 2 * i + 3], [2 * i, 2 * i + 3, 2 * i + 2])
        ]
    )
    return vertices, faces


def ground_grid(config: GroundPlaneConfig, center, normal, size):
    """Create major and minor grid lines anchored to world coordinates.

    Args:
        config: Ground settings supplying spacing in metres and sRGB colors.
        center: Ground center as a world three-vector, in metres.
        normal: Nonzero world floor normal.
        size: Positive ground side length in metres.

    Returns:
        Tuple of world line endpoints, shape (lines, 2, 3), and sRGB colors,
        shape (lines, 3). Automatic spacing uses a 1/2/5 decimal step near
        one tenth of the ground size; explicit spacing is never stretched.
    """
    basis = plane_basis(normal)
    center = np.asarray(center, dtype=float)
    spacing = config.grid_spacing
    if spacing is None:
        power = 10 ** np.floor(np.log10(size / 10))
        candidates = power * np.array([1, 2, 5, 10])
        spacing = candidates[np.argmin(np.abs(np.log(candidates / (size / 10))))]
    half = size / 2
    local_center = center @ basis
    segments, colors = [], []
    for axis in (0, 1):
        first = int(np.ceil((local_center[axis] - half) / spacing))
        last = int(np.floor((local_center[axis] + half) / spacing))
        for index in range(first, last + 1):
            points = np.zeros((2, 3))
            points[:, axis] = index * spacing - local_center[axis]
            points[:, 1 - axis] = [-half, half]
            points[:, 2] = 2e-4 * max(size, 1.0)
            segments.append(points @ basis.T + center)
            colors.append(
                config.grid_major_color
                if index % config.grid_major_every == 0
                else config.grid_color
            )
    return np.asarray(segments).reshape(-1, 2, 3), np.asarray(colors).reshape(-1, 3)


def resolved_lights(scene: SceneConfig, normal):
    """Resolve ground-relative lights into world coordinates without mutation.

    The same right-handed basis orients the backdrop and preset lights. Explicit
    world lights retain their coordinates. Ground-relative point positions are
    measured from the world origin, matching the preset's reference arrangement.
    """
    basis = plane_basis(normal)
    for light in scene.lights:
        if light.reference == "world":
            yield light
        else:
            field = (
                "direction" if isinstance(light, DirectionalLightConfig) else "position"
            )
            yield replace(
                light,
                reference="world",
                **{field: tuple(basis @ getattr(light, field))},
            )
