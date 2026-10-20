"""Shared strand geometry, analytic properties and parameter updates."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from systems.test_gvs_hollow_links import numerical_rectangle_shear

from soromox.systems import (
    HelicoidalHelixParams,
    RodConstitutiveLaw,
    StrandSection,
)

jax.config.update("jax_enable_x64", True)


def helix_kwargs():
    return {
        "length": 0.2,
        "radius_base": 0.025,
        "radius_tip": 0.02,
        "num_turns": 2.5,
        "density": 1220.0,
        "young_modulus": 30e6,
        "shear_modulus": 10e6,
    }


def test_nonsquare_hollow_properties_follow_strand_axes():
    width, height, t = 0.010, 0.002, 0.0001
    p = HelicoidalHelixParams.rectangular_hollow(
        width=width, height=height, wall_thickness=t, **helix_kwargs()
    )
    # Independent values for a 10 x 2 mm tube with a 0.1 mm wall.
    assert_allclose(p.strand_area, 2.36e-6)
    assert_allclose(
        p.strand_second_moments, [2.548786666666667e-11, 1.903866666666667e-12]
    )
    assert_allclose(
        p.strand_properties.shear_areas,
        numerical_rectangle_shear(
            np.array([height, width]), np.array([height - 2 * t, width - 2 * t])
        ),
        rtol=1e-9,
    )
    mid_area = 0.0099 * 0.0019
    perimeter = 2 * (0.0099 + 0.0019)
    assert_allclose(
        p.strand_torsion_constant,
        4 * mid_area**2 * t / perimeter + perimeter * t**3 / 3,
    )
    assert_allclose(p.strand_section.outer_dimensions, [height, width])
    assert_allclose(p.strand_section.inner_dimensions, [0.0018, 0.0098])


@pytest.mark.parametrize(
    "width,height,t",
    [
        (0, 0.002, 0.0001),
        (0.01, -1, 0.0001),
        (0.01, 0.002, 0),
        (np.nan, 0.002, 0.0001),
        (0.01, np.inf, 0.0001),
        (0.01, 0.002, np.nan),
        (0.01, 0.002, 0.001),
        (0.01, 0.002, 0.002),
    ],
)
def test_invalid_hollow_dimensions(width, height, t):
    with pytest.raises(ValueError):
        StrandSection.rectangular_hollow(width=width, height=height, wall_thickness=t)


def test_geometry_updates_refresh_properties_and_preserve_explicit_overrides():
    p = HelicoidalHelixParams.circular(strand_radius=0.002, **helix_kwargs())
    section = StrandSection.rectangular_hollow(
        width=0.004, height=0.002, wall_thickness=0.0002
    )
    changed = p.replace(
        strand_section=section, strand_shear_factors_override=[0.7, 0.8]
    )
    assert_allclose(changed.strand_area, 0.004 * 0.002 - 0.0036 * 0.0016)
    assert_allclose(changed.strand_shear_factors, [0.7, 0.8])
    density_only = changed.replace(density=1500.0)
    assert_allclose(density_only.strand_shear_factors, [0.7, 0.8])
    assert_allclose(
        density_only.strand_section.outer_dimensions, section.outer_dimensions
    )
    enlarged = changed.replace(
        strand_section=section.replace(
            outer_dimensions=2 * section.outer_dimensions,
            wall_thickness=2 * section.wall_thickness,
        )
    )
    assert_allclose(enlarged.strand_area, 4 * changed.strand_area)
    assert_allclose(enlarged.strand_second_moments, 16 * changed.strand_second_moments)
    assert_allclose(
        enlarged.strand_torsion_constant, 16 * changed.strand_torsion_constant
    )
    assert_allclose(
        enlarged.strand_shear_factors,
        [0.7, 0.8],
    )


def test_section_dimensions_are_differentiable_and_reuse_compilation():
    traces = []

    @eqx.filter_jit
    def mass(section):
        traces.append(True)
        p = HelicoidalHelixParams.from_section(section, **helix_kwargs())
        return p.sectional_inertia(jnp.asarray(0.1))[3, 3]

    section = StrandSection.circular(0.002)
    first = mass(section)
    second = mass(section.replace(outer_dimensions=[0.004]))
    assert_allclose(second, 4 * first)
    assert len(traces) == 1
    derivative = jax.grad(lambda radius: mass(StrandSection.circular(radius)))(0.002)
    assert_allclose(derivative, 2 * first / 0.002)


def test_property_only_inputs_remain_supported():
    p = HelicoidalHelixParams(
        strand_area_override=1e-6,
        strand_second_moments_override=jnp.array([1e-12, 2e-12]),
        strand_torsion_constant_override=2e-12,
        strand_shear_factors_override=jnp.array([0.8, 0.9]),
        **helix_kwargs(),
    )
    assert p.strand_section is None
    assert_allclose(p.replace(density=1300.0).strand_area, p.strand_area)


def test_hollow_circular_properties_and_wall_derivative():
    p = HelicoidalHelixParams.circular_hollow(
        strand_radius=0.002, wall_thickness=0.0005, **helix_kwargs()
    )
    assert_allclose(p.strand_area, np.pi * (0.002**2 - 0.0015**2))
    assert_allclose(
        p.strand_second_moments, np.full(2, np.pi * (0.002**4 - 0.0015**4) / 4)
    )
    assert_allclose(p.strand_torsion_constant, np.sum(p.strand_second_moments))
    assert_allclose(p.strand_section.inner_dimensions, [0.0015])
    assert np.all(np.asarray(p.strand_shear_factors) > 0.5)
    assert np.all(np.asarray(p.strand_shear_factors) < 6 / 7)
    derivative = jax.jit(
        jax.grad(
            lambda t: (
                HelicoidalHelixParams.circular_hollow(
                    strand_radius=0.002, wall_thickness=t, **helix_kwargs()
                ).strand_area
            )
        )
    )(jnp.array(0.0005))
    assert_allclose(derivative, 2 * np.pi * 0.0015)


def test_solid_rectangle_properties_and_square_torsion():
    p = HelicoidalHelixParams.rectangular(width=0.004, height=0.002, **helix_kwargs())
    assert p.strand_section.inner_dimensions is None
    assert_allclose(p.strand_area, 8e-6)
    assert_allclose(
        p.strand_second_moments, [1.066666666666667e-11, 2.666666666666667e-12]
    )
    assert_allclose(p.strand_shear_factors, [5 / 6, 5 / 6])
    swapped = HelicoidalHelixParams.rectangular(
        width=0.002, height=0.004, **helix_kwargs()
    )
    assert_allclose(swapped.strand_second_moments, p.strand_second_moments[::-1])
    assert_allclose(swapped.strand_torsion_constant, p.strand_torsion_constant)
    square = HelicoidalHelixParams.rectangular(width=1.0, height=1.0, **helix_kwargs())
    assert_allclose(square.strand_torsion_constant, 7 / 48, rtol=1e-8)


@pytest.mark.parametrize(
    "radius,wall",
    [
        (0, 0.001),
        (0.002, 0),
        (0.002, -0.001),
        (0.002, 0.002),
        (0.002, 0.003),
        (0.002, np.nan),
    ],
)
def test_invalid_circular_tube_dimensions(radius, wall):
    with pytest.raises(ValueError):
        StrandSection.circular_hollow(radius=radius, wall_thickness=wall)


@pytest.mark.parametrize("hollow", [False, True])
def test_material_updates_refresh_cowper_shear_and_derivatives(hollow):
    section = (
        StrandSection.circular_hollow(radius=0.002, wall_thickness=0.0005)
        if hollow
        else StrandSection.circular(0.002)
    )
    p = HelicoidalHelixParams.from_section(section, **helix_kwargs())

    def factors(nu):
        return p.replace(
            shear_modulus=p.young_modulus / (2 * (1 + nu))
        ).strand_shear_factors

    def reference(nu):
        m2 = (0.0015 / 0.002) ** 2 if hollow else 0.0
        return (
            6
            * (1 + nu)
            * (1 + m2) ** 2
            / ((7 + 6 * nu) * (1 + m2) ** 2 + (20 + 12 * nu) * m2)
        )

    compiled = jax.jit(factors)
    for nu in [0.0, 0.3, 0.49]:
        assert_allclose(compiled(jnp.asarray(nu)), np.full(2, reference(nu)))
    derivative = jax.jit(jax.jacfwd(factors))(jnp.array(0.3))
    assert_allclose(
        derivative,
        np.full(2, (reference(0.300001) - reference(0.299999)) / 2e-6),
        rtol=1e-8,
    )
    assert_allclose(
        p.replace(young_modulus=0.8 * p.young_modulus).strand_shear_factors,
        np.full(2, reference(0.2)),
    )


def test_calibrations_persist_and_can_be_cleared():
    p = HelicoidalHelixParams.circular(
        strand_radius=0.002,
        strand_area_override=1e-5,
        strand_second_moments_override=[2e-12, 3e-12],
        strand_torsion_constant_override=4e-12,
        strand_shear_factors_override=[0.7, 0.8],
        **helix_kwargs(),
    )
    changed = p.replace(
        strand_section=StrandSection.rectangular(width=0.003, height=0.002),
        young_modulus=1.5 * p.young_modulus,
    )
    assert_allclose(changed.strand_properties.area, 1e-5)
    assert_allclose(changed.strand_properties.second_moments, [2e-12, 3e-12])
    assert_allclose(changed.strand_properties.torsion_constant, 4e-12)
    assert_allclose(changed.strand_properties.shear_areas, [7e-6, 8e-6])
    cleared = changed.replace(
        strand_area_override=None,
        strand_second_moments_override=None,
        strand_torsion_constant_override=None,
        strand_shear_factors_override=None,
    )
    assert_allclose(cleared.strand_area, 6e-6)
    assert_allclose(cleared.strand_shear_factors, [5 / 6, 5 / 6])
    assert_allclose(p.strand_shear_factors, [0.7, 0.8])


def test_strand_law_updates_and_rejection():
    p = HelicoidalHelixParams.rectangular(width=0.004, height=0.002, **helix_kwargs())
    uniform = p.replace(strand_constitutive_law=RodConstitutiveLaw())
    assert_allclose(uniform.strand_shear_factors, [1, 1])
    assert_allclose(
        uniform.strand_torsion_constant, jnp.sum(uniform.strand_second_moments)
    )
    assert not np.isclose(
        uniform.strand_torsion_constant, p.strand_torsion_constant, atol=0
    )
    with pytest.raises(ValueError, match="UNIFORM"):
        uniform.replace(
            strand_section=StrandSection.circular_hollow(
                radius=0.002, wall_thickness=0.0005
            )
        )
    with pytest.raises(TypeError, match="strand_constitutive_law"):
        p.replace(strand_constitutive_law="saint_venant")
    with pytest.raises(ValueError, match="four strand overrides"):
        p.replace(strand_section=None)
    with pytest.raises(ValueError, match="Cowper"):
        HelicoidalHelixParams.circular(
            strand_radius=0.002, **(helix_kwargs() | {"shear_modulus": 1e6})
        )


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
@pytest.mark.parametrize("hollow", [False, True])
def test_strand_properties_match_gvs_material_projection(kind, hollow):
    from dataclasses import replace

    from systems.test_gvs_hollow_links import LENGTH, SHEAR, YOUNG, link_spec, model

    wall = 0.002 if hollow else None
    section = StrandSection(
        geometry=link_spec(kind, wall=wall).cross_section_geometry,
        outer_dimensions=jnp.array([0.03] if kind == "circular" else [0.04, 0.06]),
        wall_thickness=wall,
    )
    kwargs = helix_kwargs() | {"young_modulus": YOUNG, "shear_modulus": SHEAR}
    helix = HelicoidalHelixParams.from_section(section, **kwargs)
    robot = model(
        replace(
            link_spec(kind, wall=wall), constitutive_law=helix.strand_constitutive_law
        )
    )
    properties = helix.strand_properties
    iy, iz = properties.second_moments
    sy, sz = properties.shear_areas
    expected = jnp.array(
        [
            SHEAR * properties.torsion_constant,
            YOUNG * iy,
            YOUNG * iz,
            YOUNG * properties.area,
            SHEAR * sy,
            SHEAR * sz,
        ]
    )
    assert_allclose(robot.K, LENGTH * jnp.diag(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
def test_strand_cavity_collapse_recovers_solid_properties(kind):
    if kind == "circular":
        solid = StrandSection.circular(0.002)
        hollow = StrandSection.circular_hollow(
            radius=0.002, wall_thickness=0.002 * (1 - 1e-9)
        )
    else:
        solid = StrandSection.rectangular(height=0.002, width=0.004)
        hollow = StrandSection.rectangular_hollow(
            height=0.002, width=0.004, wall_thickness=0.001 * (1 - 1e-9)
        )
    p = HelicoidalHelixParams.from_section(solid, **helix_kwargs())
    limit = p.replace(strand_section=hollow)
    for actual, expected in zip(
        limit.strand_properties, p.strand_properties, strict=True
    ):
        assert_allclose(actual, expected, rtol=2e-8)


@pytest.mark.parametrize(
    "field", ["length", "density", "young_modulus", "damping_time_constant"]
)
def test_required_parameters_cannot_be_cleared(field):
    p = HelicoidalHelixParams.circular(strand_radius=0.002, **helix_kwargs())
    with pytest.raises(ValueError, match=field):
        p.replace(**{field: None})
