"""Mount geometry dimensions, orientation and configuration contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.base_geometry import base_plate_mesh
from soromox.rendering.config import GeometryConfig


@pytest.mark.parametrize(
    "style", ["disk", "beveled_disk", "truncated_cone", "flared_collar"]
)
def test_base_shapes_have_requested_dimensions_and_outward_faces(style):
    vertices, faces = base_plate_mesh(0.032, 0.024, style, 48)
    assert_allclose(vertices[:, 2].min(), -0.012)
    assert_allclose(vertices[:, 2].max(), 0.012)
    assert_allclose(np.max(np.linalg.norm(vertices[:, :2], axis=1)), 0.032)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.all(np.linalg.norm(normals, axis=1) > 0)
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0)
    assert (
        np.einsum(
            "ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        )
        / 6
        > 0
    )
    assert GeometryConfig(base_plate_style=style).base_plate_style == style


def test_default_mount_is_flared_collar_and_rejects_unknown_shapes():
    assert GeometryConfig().base_plate_style == "flared_collar"
    with pytest.raises(ValueError, match="base_plate_style"):
        GeometryConfig(base_plate_style="unknown")
    with pytest.raises(ValueError, match="base_plate_style"):
        base_plate_mesh(1, 1, "unknown")


@pytest.mark.parametrize(
    "style", ["disk", "beveled_disk", "truncated_cone", "flared_collar"]
)
def test_open3d_mount_keeps_center_when_aligned(style):
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import _make_base_plate

    center = np.array([0.2, -0.4, 0.7])
    mesh = _make_base_plate(
        center, 0.032, 0.024, normal_xyz=np.array([1, 0, 0]), style=style
    )
    relative = np.asarray(mesh.vertices) - center
    assert_allclose([relative[:, 0].min(), relative[:, 0].max()], [-0.012, 0.012])
    assert_allclose(np.max(np.linalg.norm(relative[:, 1:], axis=1)), 0.032)
