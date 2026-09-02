"""Tests for renderer-neutral cross-section contours and swept meshes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from soromox.rendering.cross_sections import (
    CrossSection,
    cross_section_sweep_layout,
    evaluate_cross_sections,
    loft_cross_sections,
    register_cross_section_contour,
)
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.systems.components import CrossSectionGeometry


def test_sweep_layout_duplicates_link_boundaries_without_cross_link_edges():
    layout = cross_section_sweep_layout([0.6, 0.4], num_points=8)

    assert_array_equal(layout.segment_starts, [0, 4])
    assert_array_equal(layout.segment_ends, [4, 8])
    assert layout.abscissae[3] < 0.6 < layout.abscissae[4]
    assert layout.abscissae[3] == pytest.approx(0.6, abs=1e-6)
    assert layout.abscissae[4] == pytest.approx(0.6, abs=1e-6)
    assert np.all(np.diff(layout.abscissae) > 0.0)


def test_sweep_layout_requires_two_stations_per_link():
    with pytest.raises(
        ValueError, match="at least two backbone points per positive-length link"
    ):
        cross_section_sweep_layout([0.5, 0.5], num_points=3)


def test_sweep_layout_skips_zero_length_articulated_chain_entries():
    layout = cross_section_sweep_layout([0.0, 0.6, 0.0, 0.4, 0.0], num_points=8)

    assert_array_equal(layout.segment_starts, [0, 0, 4, 4, 8])
    assert_array_equal(layout.segment_ends, [0, 4, 4, 8, 8])
    assert layout.abscissae[0] > 0.0
    assert layout.abscissae[3] < 0.6 < layout.abscissae[4]
    assert layout.abscissae[-1] == pytest.approx(1.0)
    assert layout.edge_caps() == {
        0: (False, False),
        1: (False, False),
        2: (False, True),
        4: (True, False),
        5: (False, False),
        6: (False, True),
    }


@pytest.mark.parametrize("lengths", [[0.0], [0.0, 0.0], [-0.1, 0.2]])
def test_sweep_layout_rejects_invalid_segment_lengths(lengths):
    with pytest.raises(ValueError, match="segment_lengths"):
        cross_section_sweep_layout(lengths, num_points=4)


@pytest.mark.parametrize(
    ("geometry", "dimensions", "expected_y_span", "expected_z_span"),
    [
        (CrossSectionGeometry.CIRCULAR, [0.2], 0.4, 0.4),
        (CrossSectionGeometry.ELLIPTICAL, [0.3, 0.1], 0.6, 0.2),
        (CrossSectionGeometry.RECTANGULAR, [0.2, 0.6], 0.6, 0.2),
    ],
)
def test_builtin_cross_section_contours_preserve_geometry(
    geometry, dimensions, expected_y_span, expected_z_span
):
    contour = CrossSection(geometry, np.asarray(dimensions)).contour(12)

    assert contour.shape == (12, 3)
    assert_allclose(contour[:, 0], 0.0, atol=0.0)
    assert np.ptp(contour[:, 1]) == pytest.approx(expected_y_span)
    assert np.ptp(contour[:, 2]) == pytest.approx(expected_z_span)


@pytest.mark.parametrize(
    ("geometry", "dimensions", "primitive", "scale_xyz", "aligned"),
    [
        (CrossSectionGeometry.CIRCULAR, [0.2], "sphere", [0.2, 0.2, 0.2], False),
        (
            CrossSectionGeometry.ELLIPTICAL,
            [0.3, 0.1],
            "ellipsoid",
            [0.3, 0.1, 0.3],
            True,
        ),
        (
            CrossSectionGeometry.RECTANGULAR,
            [0.2, 0.6],
            "box",
            [0.6, 0.2, 0.6],
            True,
        ),
    ],
)
def test_builtin_discrete_markers_are_backend_neutral(
    geometry, dimensions, primitive, scale_xyz, aligned
):
    marker = CrossSection(geometry, np.asarray(dimensions)).discrete_marker()

    assert marker.primitive == primitive
    assert_allclose(marker.scale_xyz, scale_xyz)
    assert marker.align_with_material_frame is aligned


def test_rectangular_contour_contains_corners_and_straight_edges():
    contour = CrossSection(
        CrossSectionGeometry.RECTANGULAR, np.array([0.2, 0.6])
    ).contour(10)
    yz = contour[:, 1:]
    expected_corners = np.array([[-0.3, -0.1], [0.3, -0.1], [0.3, 0.1], [-0.3, 0.1]])

    for corner in expected_corners:
        assert np.any(np.all(np.isclose(yz, corner), axis=1))
    assert np.all(np.isclose(np.abs(yz[:, 0]), 0.3) | np.isclose(np.abs(yz[:, 1]), 0.1))


@pytest.mark.parametrize(
    ("geometry", "base_dimensions", "tip_dimensions"),
    [
        (CrossSectionGeometry.CIRCULAR, [0.2], [0.1]),
        (CrossSectionGeometry.ELLIPTICAL, [0.3, 0.2], [0.15, 0.05]),
        (CrossSectionGeometry.RECTANGULAR, [0.4, 0.2], [0.2, 0.1]),
    ],
)
def test_loft_uses_independent_endpoint_contours(
    geometry, base_dimensions, tip_dimensions
):
    section0 = CrossSection(geometry, np.asarray(base_dimensions))
    section1 = CrossSection(geometry, np.asarray(tip_dimensions))
    p0 = np.zeros(3)
    p1 = np.array([1.0, 0.0, 0.0])
    frame0 = np.eye(3)
    frame1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    vertices, faces = loft_cross_sections(
        p0,
        p1,
        frame0,
        frame1,
        section0,
        section1,
        resolution=8,
        cap_end=True,
    )

    assert_allclose(vertices[:8], p0 + section0.contour(8) @ frame0.T)
    assert_allclose(vertices[8:16], p1 + section1.contour(8) @ frame1.T)
    assert_allclose(vertices[-1], p1)
    assert faces.shape == (24, 3)
    assert_array_equal(faces[-8:, 0], np.full(8, 16))


def test_loft_can_close_both_sides_of_a_link_interface():
    section = CrossSection(CrossSectionGeometry.CIRCULAR, np.array([0.2]))
    vertices, faces = loft_cross_sections(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.eye(3),
        np.eye(3),
        section,
        section,
        resolution=8,
        cap_start=True,
        cap_end=True,
    )

    assert vertices.shape == (18, 3)
    assert faces.shape == (32, 3)
    assert_allclose(vertices[-2], [0.0, 0.0, 0.0])
    assert_allclose(vertices[-1], [1.0, 0.0, 0.0])
    start_face = vertices[faces[16]]
    end_face = vertices[faces[-1]]
    start_normal = np.cross(
        start_face[1] - start_face[0], start_face[2] - start_face[0]
    )
    end_normal = np.cross(end_face[1] - end_face[0], end_face[2] - end_face[0])
    assert start_normal[0] < 0.0
    assert end_normal[0] > 0.0


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64")
def test_soft_robot_cross_sections_use_vectorized_mixed_profile_evaluation():
    material = {
        "young_modulus": 1e6,
        "shear_modulus": 1e6 / 3.0,
        "density": 1000.0,
        "reference_strain": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
    links = (
        LinkSpec.rectangular(
            length=0.4,
            height=LinearProfile(0.2, 0.1),
            width=0.3,
            **material,
        ),
        LinkSpec.elliptical(
            length=0.3,
            semi_major=0.15,
            semi_minor=LinearProfile(0.1, 0.05),
            **material,
        ),
        LinkSpec.circular(
            length=0.2,
            radius=LinearProfile(0.08, 0.04),
            **material,
        ),
    )
    segments = tuple(
        GVSSegment(
            link=link,
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="monomial",
                strain_selector=[1, 0, 0, 0, 0, 0],
                basis_order=[0, 0, 0, 0, 0, 0],
            ),
            num_gauss_points=5,
        )
        for link in links
    )
    robot = GVS.from_segments(segments)

    sections = evaluate_cross_sections(
        robot,
        np.zeros(robot.num_coordinates),
        [0.2, 0.55, 0.8],
    )

    assert [section.geometry for section in sections] == [
        CrossSectionGeometry.RECTANGULAR,
        CrossSectionGeometry.ELLIPTICAL,
        CrossSectionGeometry.CIRCULAR,
    ]
    assert_allclose(sections[0].dimensions, [0.15, 0.3], rtol=1e-6)
    assert_allclose(sections[1].dimensions, [0.15, 0.075], rtol=1e-6)
    assert_allclose(sections[2].dimensions, [0.06, 0.06], rtol=1e-6)


def test_additional_geometry_can_register_a_contour_without_renderer_changes():
    geometry = 10_001

    def diamond_contour(dimensions, resolution):
        assert resolution == 4
        half_width, half_height = dimensions
        return np.array(
            [
                [0.0, half_width, 0.0],
                [0.0, 0.0, half_height],
                [0.0, -half_width, 0.0],
                [0.0, 0.0, -half_height],
            ]
        )

    register_cross_section_contour(geometry, diamond_contour)
    contour = CrossSection(geometry, np.array([0.3, 0.2])).contour(4)

    assert_allclose(
        contour,
        [
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, -0.3, 0.0],
            [0.0, 0.0, -0.2],
        ],
    )
