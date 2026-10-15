"""Strand inertia against independent volume moments and mixed GVS mechanics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from soromox.systems import (
    GVSSegment,
    Helicoidal,
    HelicoidalHelixParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.systems.components.inertia import sectional_inertias

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def helix():
    return HelicoidalHelixParams.circular(
        length=0.2,
        radius_base=0.025,
        radius_tip=0.025,
        num_turns=3.0,
        strand_radius=0.002,
        density=1220.0,
        young_modulus=30e6,
        shear_modulus=30e6 / 2.96,
    )


def _model(helix, **kwargs):
    segments = []
    for length, density, active, points in (
        (0.0183, 800.0, (False,) * 6, 5),
        (float(helix.length), 1.0, (True,) * 4 + (False,) * 2, 12),
    ):
        dofs = sum(active)
        segments.append(
            GVSSegment(
                link=LinkSpec.circular(
                    length=length,
                    radius=0.025,
                    density=density,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    stiffness=jnp.eye(dofs),
                    damping=jnp.zeros((dofs, dofs)),
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="legendre", strain_selector=active, basis_order=(0,) * 6
                ),
                num_gauss_points=points,
            )
        )
    return Helicoidal.from_segments(segments, constitutive_params={1: helix}, **kwargs)


def test_uniform_circular_strands_match_analytic_moments(helix):
    rho, area = float(helix.density), float(helix.strand_area)
    r, length = float(helix.radius_base), float(helix.length)
    rate = 2 * np.pi * float(helix.num_turns) / length
    factor = np.sqrt(1 + (r * rate) ** 2)
    count = 2 * helix.helices_per_direction
    moment = float(helix.strand_second_moments[0])
    mass = count * rho * area * factor
    axial = mass * r**2 + count * rho * factor * moment * (1 + 1 / factor**2)
    transverse = mass * r**2 / 2 + count * rho * factor * moment * (
        1 + (r * rate / factor) ** 2 / 2
    )
    expected = np.diag([axial, transverse, transverse, mass, mass, mass])
    actual = helix.sectional_inertia(jnp.array([0.0, 0.031, length]))
    assert_allclose(
        actual, np.broadcast_to(expected, actual.shape), rtol=1e-12, atol=1e-16
    )
    assert_allclose(
        helix.sectional_inertia(jnp.array(0.0)), expected, rtol=1e-12, atol=1e-16
    )


def test_tapered_non_circular_strands_match_point_volume_moments(helix):
    a, b = 0.003, 0.0015
    area = np.pi * a * b
    # One strand in each family retains COM coupling and anisotropic inertia.
    params = HelicoidalHelixParams(
        length=helix.length,
        radius_base=0.015,
        radius_tip=0.035,
        num_turns=helix.num_turns,
        density=helix.density,
        young_modulus=helix.young_modulus,
        shear_modulus=helix.shear_modulus,
        strand_area=area,
        strand_second_moments=jnp.array([area * b * b / 4, area * a * a / 4]),
        strand_torsion_constant=1e-10,
        strand_shear_factors=jnp.ones(2),
        helices_per_direction=1,
        phase=0.2,
        winding_family_phase_offset=0.7,
    )
    positions = np.array([[0.0, 0.04], [0.13, 0.2]])
    expected = []
    length, rho = float(params.length), float(params.density)
    slope = (0.035 - 0.015) / length
    rate = 2 * np.pi * float(params.num_turns) / length
    for s in positions.ravel():
        matrix = np.zeros((6, 6))
        r = 0.015 + slope * s
        for phase, omega in ((0.2, -rate), (0.9, rate)):
            theta = phase + omega * s
            radial = np.array([0, np.cos(theta), np.sin(theta)])
            derivative = np.array(
                [
                    1,
                    slope * np.cos(theta) - r * omega * np.sin(theta),
                    slope * np.sin(theta) + r * omega * np.cos(theta),
                ]
            )
            factor = np.linalg.norm(derivative)
            tangent = derivative / factor
            y = radial - np.dot(radial, tangent) * tangent
            y /= np.linalg.norm(y)
            z = np.cross(tangent, y)
            for uy, uz in (
                (a / np.sqrt(2), 0),
                (-a / np.sqrt(2), 0),
                (0, b / np.sqrt(2)),
                (0, -b / np.sqrt(2)),
            ):
                x, yy, zz = r * radial + uy * y + uz * z
                velocity = np.column_stack(
                    ([[0, zz, -yy], [-zz, 0, x], [yy, -x, 0]], np.eye(3))
                )
                matrix += rho * area * factor / 4 * velocity.T @ velocity
        expected.append(matrix)
    actual = params.sectional_inertia(jnp.asarray(positions))
    assert_allclose(
        actual, np.array(expected).reshape(actual.shape), rtol=1e-12, atol=1e-15
    )
    # Elastic torsion constant is independent of the mass polar moment.
    assert_allclose(
        params.replace(strand_torsion_constant=2e-10).sectional_inertia(
            jnp.asarray(positions)
        ),
        actual,
    )
    nodes, weights = np.polynomial.legendre.leggauss(16)
    sampled_mass = params.sectional_inertia(jnp.asarray(length * (nodes + 1) / 2))[
        :, 3, 3
    ]
    integrated = length / 2 * weights @ np.asarray(sampled_mass)
    expected_mass = quad(
        lambda s: (
            2 * rho * area * np.sqrt(1 + slope**2 + (rate * (0.015 + slope * s)) ** 2)
        ),
        0,
        length,
        epsabs=1e-13,
    )[0]
    assert_allclose(integrated, expected_mass, rtol=1e-12)


def test_density_updates_and_native_cylinder_dynamics(helix):
    robot = _model(helix)
    assert_allclose(robot.params.link.sectional_inertia_mask, [False, True])
    inertia = sectional_inertias(robot.params.link, robot.inner_mass_matrices)
    radius, rho = 0.025, 800.0
    area = np.pi * radius**2
    expected = np.diag(
        rho
        * np.array(
            [
                np.pi * radius**4 / 2,
                np.pi * radius**4 / 4,
                np.pi * radius**4 / 4,
                area,
                area,
                area,
            ]
        )
    )
    assert_allclose(inertia[0], np.broadcast_to(expected, inertia[0].shape), rtol=1e-12)
    denser_cylinder = robot.update_link_params(
        density=robot.params.link.density.at[0].multiply(2)
    )
    changed_sections = sectional_inertias(
        denser_cylinder.params.link, denser_cylinder.inner_mass_matrices
    )
    assert_allclose(changed_sections[0], 2 * inertia[0])
    assert_allclose(changed_sections[1], inertia[1])
    traces = []

    @eqx.filter_jit
    def prepare_and_evaluate(params):
        traces.append(True)
        updated = robot.with_helix_inertia({1: params})
        q = jnp.array([0.1, -0.2, 0.3, 0.01])
        return updated, updated.dynamics_terms(q, 0.1 * q)

    first, before = prepare_and_evaluate(helix)
    second, after = prepare_and_evaluate(helix.replace(density=2 * helix.density))
    assert len(traces) == 1
    # The fixed proximal cylinder has zero kinetic energy and generalized gravity.
    for a, b in zip(after, before, strict=True):
        assert_allclose(a, 2 * b, rtol=1e-10, atol=1e-12)
    assert_allclose(first.K, second.K)
    assert_allclose(first.D_active, second.D_active)
    derivative = jax.grad(
        lambda density: robot.with_helix_inertia(
            {1: helix.replace(density=density)}
        ).inertia_matrix(jnp.zeros(4))[3, 3]
    )(helix.density)
    assert_allclose(
        derivative, robot.inertia_matrix(jnp.zeros(4))[3, 3] / helix.density, rtol=1e-12
    )
    if jax.default_backend() == "gpu":
        q = jnp.array([0.1, -0.2, 0.3, 0.01])
        for a, b in zip(
            second.dynamics_terms(q, 0.1 * q, backend="warp"), after, strict=True
        ):
            assert_allclose(a, b, rtol=2e-8, atol=2e-11)


def test_explicit_overrides_and_validation(helix):
    robot = _model(
        helix,
        mass_per_length=jnp.array([0.3, 0.4]),
        rotational_inertia_per_length=jnp.ones((2, 3)) * 0.002,
    )
    sections = sectional_inertias(robot.params.link, robot.inner_mass_matrices)
    assert_allclose(sections[:, :, 3, 3], np.broadcast_to([[0.3], [0.4]], (2, 12)))
    assert_allclose(robot.params.link.sectional_inertia_mask, [True, True])
    for mask in (jnp.ones(2), jnp.ones(3, dtype=bool)):
        with pytest.raises(ValueError, match="sectional_inertia_mask"):
            robot.update_link_params(sectional_inertia_mask=mask)
    with pytest.raises(ValueError, match="requires sectional_inertia"):
        robot.update_link_params(sectional_inertia=None)
    with pytest.raises(ValueError, match="together"):
        _model(helix, mass_per_length=jnp.ones(2))
    with pytest.raises(ValueError, match="length must match"):
        robot.with_helix_inertia({1: helix.replace(length=0.3)})
    assert robot.with_helix_inertia({}) is robot
