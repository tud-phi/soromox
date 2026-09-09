"""Shared circular mounting geometry for the Open3D and Viser renderers."""

import numpy as np


def base_plate_mesh(
    radius: float, height: float, style: str, resolution: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a disk, beveled disk, truncated cone or flared collar.

    Args:
        radius: Maximum footprint radius in metres, finite and nonnegative.
        height: Total mount height in metres, finite and nonnegative.
        style: ``disk``, ``beveled_disk``, ``truncated_cone`` or ``flared_collar``.
        resolution: Angular samples around the mount, at least three.

    Returns:
        Vertices of shape ``(N, 3)`` in metres and triangles of shape ``(M, 3)``.
        The mount is centered at the origin with its axis along +Z and its top
        at ``height / 2``. Profile corners have separate vertices so the side
        shading is smooth around the circumference and crisp across bevels.

    Raises:
        ValueError: The style, dimensions or resolution is invalid.
    """
    profiles = {
        "disk": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "beveled_disk": [(0, 0), (0.9, 0), (1, 0.16), (1, 0.80), (0.88, 1), (0, 1)],
        "truncated_cone": [
            (0, 0),
            (0.94, 0),
            (1, 0.08),
            (0.67, 0.94),
            (0.62, 1),
            (0, 1),
        ],
        "flared_collar": [
            (0, 0),
            (0.92, 0),
            (1, 0.07),
            (1, 0.19),
            (0.92, 0.25),
            (0.66, 0.77),
            (0.66, 0.93),
            (0.60, 1),
            (0, 1),
        ],
    }
    if style not in profiles:
        raise ValueError(f"Unknown base_plate_style: {style}")
    if not np.isfinite([radius, height]).all() or radius < 0 or height < 0:
        raise ValueError("base radius and height must be finite and nonnegative")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, (int, np.integer))
        or resolution < 3
    ):
        raise ValueError("base resolution must be an integer >= 3")
    profile = np.asarray(profiles[style]) * [radius, height]
    bands = np.stack((profile[:-1], profile[1:]), axis=1)
    angles = np.arange(resolution) * (2 * np.pi / resolution)
    vertices = np.empty((len(bands), 2, resolution, 3))
    vertices[..., 0] = bands[..., 0, None] * np.cos(angles)
    vertices[..., 1] = bands[..., 0, None] * np.sin(angles)
    vertices[..., 2] = bands[..., 1, None] - height / 2
    indices = np.arange(resolution)
    following = (indices + 1) % resolution
    faces = np.stack(
        (
            np.stack((indices, following, resolution + following), axis=-1),
            np.stack((indices, resolution + following, resolution + indices), axis=-1),
        ),
        axis=0,
    )
    faces = faces[None] + (2 * resolution * np.arange(len(bands)))[:, None, None, None]
    # Omit the degenerate half of each fan where a profile endpoint is on-axis.
    faces = faces[bands[..., 0] > 0].reshape(-1, 3)
    return vertices.reshape(-1, 3), faces.astype(np.int32)
