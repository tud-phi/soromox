"""Shared strand geometry, analytic properties and parameter updates."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import HelicoidalHelixParams, StrandSection

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
    assert_allclose(p.strand_area * p.strand_shear_factors, [4e-7, 2e-6])
    mid_area = 0.0099 * 0.0019
    perimeter = 2 * (0.0099 + 0.0019)
    assert_allclose(p.strand_torsion_constant, 4 * mid_area**2 * t / perimeter)
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
    changed = p.replace(strand_section=section, strand_shear_factors=[0.7, 0.8])
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
        section.mechanical_properties()["strand_shear_factors"],
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
        strand_area=1e-6,
        strand_second_moments=jnp.array([1e-12, 2e-12]),
        strand_torsion_constant=2e-12,
        strand_shear_factors=jnp.array([0.8, 0.9]),
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
            lambda t: StrandSection.circular_hollow(
                radius=0.002,
                wall_thickness=t,
            ).mechanical_properties()["strand_area"]
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
    swapped = StrandSection.rectangular(
        width=0.002, height=0.004
    ).mechanical_properties()
    assert_allclose(swapped["strand_second_moments"], p.strand_second_moments[::-1])
    assert_allclose(swapped["strand_torsion_constant"], p.strand_torsion_constant)
    square = StrandSection.rectangular(width=1.0, height=1.0).mechanical_properties()
    assert_allclose(square["strand_torsion_constant"], 0.140577014955, rtol=1e-8)


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
