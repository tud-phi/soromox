"""Hollow GVS geometry drives inertia, constitutive matrices and rendering."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.cross_sections import evaluate_cross_sections
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    IsotropicMaterialParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    PlanarPCS,
    StrainBasisSpec,
)
from soromox.systems.components.cross_sections import (
    CrossSectionGeometry,
    section_properties,
    section_torsion_constant,
)
from soromox.systems.components.inertia import sectional_inertias

jax.config.update("jax_enable_x64", True)

LENGTH, DENSITY, YOUNG, SHEAR, DAMPING = 0.3, 1200.0, 1e6, 4e5, 100.0
MATERIAL = IsotropicMaterialParams(
    young_modulus=YOUNG, shear_modulus=SHEAR, material_damping_coefficient=DAMPING
)


def link_spec(kind, *, wall=0.002, tapered=False):
    kwargs = {
        "length": LENGTH,
        "density": DENSITY,
        "young_modulus": YOUNG,
        "shear_modulus": SHEAR,
        "material_damping_coefficient": DAMPING,
        "reference_strain": [0, 0, 0, 1, 0, 0],
    }
    if kind == "circular":
        return LinkSpec.circular(
            radius=LinearProfile(0.03, 0.02) if tapered else 0.03,
            wall_thickness=wall,
            **kwargs,
        )
    if kind == "rectangular":
        return LinkSpec.rectangular(
            height=LinearProfile(0.04, 0.03) if tapered else 0.04,
            width=LinearProfile(0.06, 0.05) if tapered else 0.06,
            wall_thickness=wall,
            **kwargs,
        )
    return LinkSpec.elliptical(semi_major=0.03, semi_minor=0.02, **kwargs)


def model(*links):
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
                num_gauss_points=5,
            )
            for link in links
        ]
    )


def expected_properties(kind, outer, wall):
    """Independent analytic area/moments, torsion and effective shear areas."""
    if kind == "circular":
        radius = outer[..., 0]
        inner = radius - wall
        area = np.pi * (radius**2 - inner**2)
        iy = iz = np.pi / 4 * (radius**4 - inner**4)
        torsion = iy + iz
        ratio2 = (inner / radius) ** 2
        shear_y = shear_z = (
            area * 6 * (1 + ratio2) ** 2 / (7 * (1 + ratio2) ** 2 + 20 * ratio2)
        )
    else:
        height, width = outer[..., 0], outer[..., 1]
        hi, wi = height - 2 * wall, width - 2 * wall
        area = height * width - hi * wi
        iy, iz = (
            (height * width**3 - hi * wi**3) / 12,
            (width * height**3 - wi * hi**3) / 12,
        )
        hm, wm = height - wall, width - wall
        torsion = 4 * (hm * wm) ** 2 / (2 * (hm + wm) / wall)
        shear_y, shear_z = 2 * height * wall, 2 * width * wall
    return area, iy, iz, torsion, shear_y, shear_z


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
@pytest.mark.parametrize("tapered", [False, True])
def test_hollow_geometry_drives_all_sectional_mechanics(kind, tapered):
    wall = LinearProfile(0.002, 0.001) if tapered else 0.002
    robot = model(link_spec(kind, wall=wall, tapered=tapered))
    count = int(robot.num_integration_points[0])
    points = np.asarray(robot.integration_points[0, :count])
    sections = evaluate_cross_sections(robot, jnp.zeros(6), LENGTH * points)
    outer = np.stack([section.outer_dimensions for section in sections])
    thickness = 0.002 - 0.001 * points if tapered else np.full_like(points, 0.002)
    factor = 1 if kind == "circular" else 2
    assert_allclose(
        np.stack([s.inner_dimensions for s in sections]),
        outer - factor * thickness[:, None],
    )
    area, iy, iz, torsion, sy, sz = expected_properties(kind, outer, thickness)
    expected_mass = DENSITY * np.stack([iy + iz, iy, iz, area, area, area], axis=-1)
    assert_allclose(
        np.diagonal(robot.mass_matrices[0, :count], axis1=-2, axis2=-1), expected_mass
    )
    weights = LENGTH * np.asarray(robot.integration_weights[0, :count])
    stiffness = np.stack(
        [SHEAR * torsion, YOUNG * iy, YOUNG * iz, YOUNG * area, SHEAR * sy, SHEAR * sz],
        axis=-1,
    )
    damping = DAMPING * np.stack([torsion, 3 * iy, 3 * iz, 3 * area, sy, sz], axis=-1)
    assert_allclose(
        robot.params.link.stiffness[0], np.diag(weights @ stiffness), atol=1e-12
    )
    assert_allclose(
        robot.params.link.damping[0], np.diag(weights @ damping), atol=1e-12
    )
    assert all(s.contours(12).shape == (2, 12, 3) for s in sections)
    solid = model(link_spec(kind, wall=None, tapered=tapered))
    assert np.all(
        np.diagonal(robot.mass_matrices[0, :count], axis1=-2, axis2=-1)
        < np.diagonal(solid.mass_matrices[0, :count], axis1=-2, axis2=-1)
    )


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
def test_wall_updates_match_reconstruction_and_are_differentiable(kind):
    robot = model(link_spec(kind))
    traces = []

    @eqx.filter_jit
    def update(wall):
        traces.append(True)
        section = robot.params.link.cross_section
        updated = robot.update_link_params(
            cross_section=section.replace(
                coefficients=section.coefficients.at[0, -1].set(wall)
            )
        ).with_isotropic_material(MATERIAL)
        return (
            updated.mass_matrices,
            updated.K,
            updated.D_active,
            updated.cross_section_geometry(jnp.zeros(6), LENGTH / 2),
        )

    update(jnp.array(0.002))
    mass, stiffness, damping, geometry = update(jnp.array(0.003))
    assert len(traces) == 1
    rebuilt = model(link_spec(kind, wall=0.003))
    assert_allclose(mass, rebuilt.mass_matrices)
    assert_allclose(stiffness, rebuilt.K)
    assert_allclose(damping, rebuilt.D_active)
    for actual, expected in zip(
        geometry, rebuilt.cross_section_geometry(jnp.zeros(6), LENGTH / 2), strict=True
    ):
        assert_allclose(actual, expected)
    derivative = jax.grad(lambda t: update(t)[0][0, 0, 3, 3])(jnp.array(0.002))
    expected = DENSITY * (
        2 * np.pi * (0.03 - 0.002)
        if kind == "circular"
        else 2 * (0.04 + 0.06 - 4 * 0.002)
    )
    assert_allclose(derivative, expected, rtol=1e-10)


def test_mixed_solid_hollow_families_use_one_geometry_query():
    robot = model(
        link_spec("circular", wall=None),
        link_spec("rectangular"),
        link_spec("circular"),
        link_spec("elliptical"),
    )
    sections = evaluate_cross_sections(
        robot, jnp.zeros(robot.num_coordinates), LENGTH * np.array([0.5, 1.5, 2.5, 3.5])
    )
    assert [s.inner_dimensions is not None for s in sections] == [
        False,
        True,
        True,
        False,
    ]
    assert [s.geometry for s in sections] == [0, 1, 0, 2]


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
def test_axial_dynamics_use_material_area_after_wall_update(kind):
    robot = model(link_spec(kind))
    section = robot.params.link.cross_section
    robot = robot.update_link_params(
        cross_section=section.replace(
            coefficients=section.coefficients.at[0, -1].set(0.003)
        )
    ).with_isotropic_material(MATERIAL)
    solid = model(link_spec(kind, wall=None))
    q = jnp.zeros(6)
    _, outer, _ = robot.cross_section_geometry(q, LENGTH / 2)
    area = expected_properties(kind, np.asarray(outer), 0.003)[0]
    solid_area = np.pi * 0.03**2 if kind == "circular" else 0.04 * 0.06
    mass, _, gravity = robot.dynamics_terms(q, q, backend="jax")
    solid_mass, _, solid_gravity = solid.dynamics_terms(q, q, backend="jax")
    # Uniform axial strain gives each material point velocity proportional to s.
    assert_allclose(mass[3, 3], DENSITY * area * LENGTH**3 / 3, rtol=1e-10)
    assert_allclose(mass[3, 3] / solid_mass[3, 3], area / solid_area)
    # Gravity acts only on translational mass, so all generalized loads scale by A.
    assert_allclose(gravity, solid_gravity * area / solid_area, atol=1e-12)


@pytest.mark.parametrize(
    "kind,wall",
    [
        ("circular", 0.03),
        ("rectangular", 0.02),
        ("circular", -0.001),
        ("rectangular", 0.0),
    ],
)
def test_invalid_cavities_are_rejected_at_construction(kind, wall):
    with pytest.raises(ValueError):
        link_spec(kind, wall=wall)


@pytest.mark.parametrize("kind,wall", [("circular", 0.025), ("rectangular", 0.016)])
def test_tapered_cavity_must_fit_at_tip_on_construction_and_update(kind, wall):
    with pytest.raises(ValueError, match="positive cavity"):
        link_spec(kind, wall=wall, tapered=True)
    robot = model(link_spec(kind, tapered=True))
    section = robot.params.link.cross_section
    with pytest.raises(ValueError, match="positive cavity"):
        robot.update_link_params(
            cross_section=section.replace(
                coefficients=section.coefficients.at[0, -1].set(wall)
            )
        )


@pytest.mark.parametrize("cls", [PCS, PlanarPCS])
def test_solid_only_systems_reject_hollow_link_specs(cls):
    with pytest.raises(ValueError, match="require GVS"):
        cls.from_links([link_spec("circular")])


def test_geometry_updates_preserve_explicit_constitutive_and_inertia_overrides():
    robot = model(link_spec("rectangular"))
    custom_inertia = jnp.eye(6)[None]
    robot = robot.update_link_params(sectional_inertia=custom_inertia)
    section = robot.params.link.cross_section
    updated = robot.update_link_params(
        cross_section=section.replace(
            coefficients=section.coefficients.at[0, -1].set(0.003)
        )
    )
    assert_allclose(updated.K, robot.K)
    assert_allclose(updated.D_active, robot.D_active)
    assert not np.allclose(updated.mass_matrices, robot.mass_matrices)
    assert_allclose(
        sectional_inertias(updated.params.link, updated.inner_mass_matrices),
        np.broadcast_to(custom_inertia[:, None], updated.inner_mass_matrices.shape),
    )


@pytest.mark.parametrize(
    "geometry,dimensions,expected",
    [
        (CrossSectionGeometry.RECTANGULAR, [1.0, 1.0], 0.140577014955),
        (CrossSectionGeometry.ELLIPTICAL, [2.0, 1.0], 8 * np.pi / 5),
    ],
)
def test_non_circular_torsion_is_distinct_from_polar_mass_moment(
    geometry, dimensions, expected
):
    assert_allclose(
        section_torsion_constant(geometry, jnp.array(dimensions)), expected, rtol=1e-8
    )
    assert not np.isclose(
        expected, section_properties(geometry, jnp.array(dimensions))[0]
    )
    kind = (
        "rectangular" if geometry == CrossSectionGeometry.RECTANGULAR else "elliptical"
    )
    robot = model(
        replace(
            link_spec(kind, wall=None), cross_section_coefficients=tuple(dimensions)
        )
    )
    assert_allclose(robot.K[0, 0], LENGTH * SHEAR * expected, rtol=1e-8)
    assert_allclose(
        robot.mass_matrices[0, 0, 0, 0],
        DENSITY * section_properties(geometry, jnp.array(dimensions))[0],
    )


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="Warp dynamics requires a GPU"
)
def test_hollow_link_dynamics_match_warp_after_wall_update():
    robot = model(link_spec("rectangular"))
    section = robot.params.link.cross_section
    robot = robot.update_link_params(
        cross_section=section.replace(
            coefficients=section.coefficients.at[0, -1].set(0.003)
        )
    ).with_isotropic_material(MATERIAL)
    q = jnp.array([0.1, -0.2, 0.3, 0.01, 0.02, -0.01])
    for actual, expected in zip(
        robot.dynamics_terms(q, 0.1 * q, backend="warp"),
        robot.dynamics_terms(q, 0.1 * q, backend="jax"),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=2e-8, atol=2e-11)
