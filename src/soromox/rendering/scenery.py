"""Backend-independent scene bounds, floor placement and backdrop meshes."""

import numpy as np

from soromox.rendering.renderer_config import GroundPlaneConfig, SceneConfig


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
        Matrix whose columns are the two plane axes and normalized normal.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.array([1.0, 0.0, 0.0])
    if abs(u @ n) > 0.95:
        u = np.array([0.0, 1.0, 0.0])
    u -= (u @ n) * n
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(n, u), n))


def backdrop_mesh(scene: SceneConfig, center, extent, normal):
    """Build a curved floor/wall from shared dimensions and floor orientation.

    Args:
        scene: Scene configuration supplying ground and backdrop parameters.
        center: Robot scene center as a world three-vector.
        extent: Positive robot scene extent in metres.
        normal: World floor normal.

    Returns:
        Tuple of vertices with shape (N, 3) and triangle indices with shape (M, 3).
    """
    cfg = scene.backdrop
    basis = plane_basis(normal)
    n = basis[:, 2]
    origin = np.asarray(center) - np.dot(center, n) * n + scene.ground.height * n
    angles = np.linspace(0, np.pi / 2, 80)
    profile = [(-cfg.depth, 0), (cfg.wall_offset, 0)]
    profile.extend(
        zip(
            cfg.wall_offset + cfg.radius * np.sin(angles[1:]),
            cfg.radius * (1 - np.cos(angles[1:])),
        )
    )
    profile.append((cfg.wall_offset + cfg.radius, cfg.height))
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
