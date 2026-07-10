import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("open3d")

from soromox.rendering.open3d_renderer import (  # noqa: E402
    _cross_section_ring_offsets,
    _make_swept_cross_section_segment,
    _swept_segment_vertices,
)
from soromox.systems.soft_robot import CrossSectionGeometry  # noqa: E402


def test_swept_cylinder_rings_follow_material_frame_not_curve_chord():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    frame = np.eye(3)

    mesh = _make_swept_cross_section_segment(
        p0,
        p1,
        frame,
        frame,
        CrossSectionGeometry.CIRCULAR,
        np.array([0.2]),
        (1.0, 0.0, 0.0),
        resolution=8,
    )

    first_ring = np.asarray(mesh.vertices)[:8]
    assert_allclose(first_ring[:, 0], 0.0, atol=1e-12)
    assert np.ptp(first_ring[:, 1]) > 0.39


def test_swept_segment_applies_each_endpoint_material_frame():
    offsets = _cross_section_ring_offsets(
        CrossSectionGeometry.RECTANGULAR, np.array([0.4, 0.2]), resolution=8
    )
    frame0 = np.eye(3)
    frame1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    p0 = np.zeros(3)
    p1 = np.array([1.0, 0.0, 0.0])

    vertices = _swept_segment_vertices(p0, p1, frame0, frame1, offsets)

    assert_allclose(vertices[:4], p0 + offsets @ frame0.T, atol=1e-12)
    assert_allclose(vertices[4:], p1 + offsets @ frame1.T, atol=1e-12)


def test_swept_segment_terminal_cap_closes_tip_with_outward_faces():
    p0 = np.zeros(3)
    p1 = np.array([1.0, 0.0, 0.0])
    frame = np.eye(3)

    mesh = _make_swept_cross_section_segment(
        p0,
        p1,
        frame,
        frame,
        CrossSectionGeometry.CIRCULAR,
        np.array([0.2]),
        (1.0, 0.0, 0.0),
        resolution=8,
        cap_end=True,
    )

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    cap_triangles = triangles[-8:]
    assert_allclose(vertices[-1], p1, atol=1e-12)
    assert np.all(cap_triangles[:, 0] == 16)
    first_cap = vertices[cap_triangles[0]]
    normal = np.cross(first_cap[1] - first_cap[0], first_cap[2] - first_cap[0])
    assert np.dot(normal, frame[:, 0]) > 0.0
