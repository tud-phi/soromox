"""Compare rendered material sections with the mechanics of the same geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.cross_sections import CrossSection, evaluate_cross_sections
from soromox.systems import (
    GVS,
    GVSSegment,
    IsotropicMaterialParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
    StrandSection,
)
from soromox.systems.components import CrossSectionGeometry, section_properties

jax.config.update("jax_enable_x64", True)


def polygon_properties(contour):
    """Integrate area and moments about material y/z over a polygon boundary."""
    y, z = np.asarray(contour)[:, 1:].T
    yn, zn = np.roll(y, -1), np.roll(z, -1)
    cross = y * zn - yn * z
    area = cross.sum() / 2
    iy = np.sum(cross * (z * z + z * zn + zn * zn)) / 12
    iz = np.sum(cross * (y * y + y * yn + yn * yn)) / 12
    return np.array([iy + iz, iy, iz, area])


@pytest.mark.parametrize("dimensions", [[0.02, 0.06], [0.06, 0.02]])
def test_rectangular_rendered_moments_match_shared_mechanics(dimensions):
    section = CrossSection(CrossSectionGeometry.RECTANGULAR, dimensions)
    measured = polygon_properties(section.contours(10)[0])
    assert_allclose(measured, section_properties(section.geometry, dimensions))
    # Primitive axes map to material (y, z, x), just like the swept contour.
    assert_allclose(section.discrete_marker().scale_xyz[:2], dimensions)


@pytest.mark.parametrize("updated", [False, True])
def test_rendered_taper_matches_gvs_inertia_bending_and_damping(updated):
    length, density, young, shear, damping = 0.3, 1200.0, 1e6, 4e5, 100.0
    robot = GVS.from_segments(
        [
            GVSSegment(
                link=LinkSpec.rectangular(
                    length=length,
                    height=LinearProfile(0.02, 0.03),
                    width=LinearProfile(0.06, 0.04),
                    density=density,
                    young_modulus=young,
                    shear_modulus=shear,
                    material_damping_coefficient=damping,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[1, 1, 1, 1, 1, 1],
                    basis_order=[0, 0, 0, 0, 0, 0],
                ),
                num_gauss_points=5,
            )
        ]
    )
    if updated:
        section = robot.params.link.cross_section
        robot = robot.update_link_params(
            cross_section=section.replace(
                coefficients=section.coefficients * jnp.array([[2.0, 1.5, 0.5, 0.75]])
            )
        ).with_isotropic_material(
            IsotropicMaterialParams(
                young_modulus=young,
                shear_modulus=shear,
                material_damping_coefficient=damping,
            )
        )
    count = int(robot.num_integration_points[0])
    sections = evaluate_cross_sections(
        robot,
        jnp.zeros(robot.num_coordinates),
        length * np.asarray(robot.integration_points[0, :count]),
    )
    properties = np.stack([polygon_properties(s.contours(12)[0]) for s in sections])
    mass_diagonal = density * properties[:, [0, 1, 2, 3, 3, 3]]
    assert_allclose(
        np.diagonal(robot.mass_matrices[0, :count], axis1=-2, axis2=-1),
        mass_diagonal,
        rtol=1e-6,
    )
    integrated_moments = (
        length * np.asarray(robot.integration_weights[0, :count]) @ properties[:, 1:3]
    )
    assert_allclose(
        np.diag(robot.params.link.stiffness[0])[1:3],
        young * integrated_moments,
        rtol=1e-6,
    )
    assert_allclose(
        np.diag(robot.params.link.damping[0])[1:3],
        3 * damping * integrated_moments,
        rtol=1e-6,
    )


@pytest.mark.parametrize("wall", [None, 0.001])
def test_rectangular_strands_share_gvs_axes_and_rendered_moments(wall):
    section = StrandSection(
        CrossSectionGeometry.RECTANGULAR, jnp.array([0.02, 0.06]), wall
    )
    loops = CrossSection.from_strand_section(section).contours(12)
    measured = polygon_properties(loops[0])
    if wall is not None:
        measured -= polygon_properties(loops[1])
    assert_allclose(np.ptp(loops[0][:, 1:], axis=0), [0.02, 0.06])
    assert_allclose(section.radial_extent, 0.01)
    assert_allclose(
        section.mechanical_properties()["strand_second_moments"],
        measured[1:3],
        rtol=1e-6,
    )
