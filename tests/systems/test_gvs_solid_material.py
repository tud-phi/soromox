"""Independent regression checks for material-derived solid GVS matrices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVS,
    GVSSegment,
    IsotropicMaterialParams,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


def _properties(kind, dimensions):
    """Area moments from the original solid-section geometry formulas."""
    a = dimensions[..., 0]
    if kind == "circular":
        area = np.pi * a**2
        iy = iz = np.pi * a**4 / 4
    else:
        b = dimensions[..., 1]
        if kind == "rectangular":
            area, iy, iz = a * b, a * b**3 / 12, b * a**3 / 12
        else:
            area, iy, iz = np.pi * a * b, np.pi * a * b**3 / 4, np.pi * b * a**3 / 4
    return area, iy, iz


def _torsion(kind, dimensions, polar):
    """Independent Saint-Venant reference for this PR's noncircular law."""
    if kind == "circular":
        return polar
    a, b = dimensions.T
    if kind == "elliptical":
        return np.pi * a**3 * b**3 / (a**2 + b**2)
    long, short = np.maximum(a, b), np.minimum(a, b)
    odd = np.arange(1, 20000, 2, dtype=float)
    correction = np.sum(
        np.tanh(np.pi * long[:, None] * odd / (2 * short[:, None])) / odd**5,
        axis=-1,
    )
    return long * short**3 / 3 * (1 - 192 * short * correction / (np.pi**5 * long))


@pytest.mark.parametrize("kind", ["circular", "rectangular", "elliptical"])
@pytest.mark.parametrize("tapered", [False, True])
def test_gvs_solid_material_matrices_at_construction_and_after_updates(kind, tapered):
    """Pin all six stiffness/damping entries, including Saint-Venant torsion."""
    length, density = 0.3, 1200.0
    start = np.array([0.02] if kind == "circular" else [0.02, 0.06])
    end = (
        start * (np.array([1.5]) if kind == "circular" else [1.5, 0.7])
        if tapered
        else start
    )
    names = {
        "circular": ("radius",),
        "rectangular": ("height", "width"),
        "elliptical": ("semi_major", "semi_minor"),
    }[kind]
    dimensions = {
        name: LinearProfile(a, b) if tapered else float(a)
        for name, a, b in zip(names, start, end, strict=True)
    }
    young, shear, damping = 1e6, 4e5, 100.0
    robot = GVS.from_segments(
        [
            GVSSegment(
                link=getattr(LinkSpec, kind)(
                    length=length,
                    density=density,
                    young_modulus=young,
                    shear_modulus=shear,
                    material_damping_coefficient=damping,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                    **dimensions,
                ),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[1] * 6,
                    basis_order=[0] * 6,
                ),
                num_gauss_points=5,
            )
        ]
    )
    # Independent quadrature integrates the degree-four section polynomials.
    points, weights = np.polynomial.legendre.leggauss(8)
    points, weights = (points + 1) / 2, weights / 2
    for updated in (False, True):
        if updated:
            start, end = 1.2 * start, 1.2 * end
            young, shear, damping = 1.4e6, 5e5, 170.0
            section = robot.params.link.cross_section
            robot = robot.update_link_params(
                cross_section=section.replace(coefficients=1.2 * section.coefficients)
            ).with_isotropic_material(
                IsotropicMaterialParams(
                    young_modulus=young,
                    shear_modulus=shear,
                    material_damping_coefficient=damping,
                )
            )
        area, iy, iz = _properties(kind, start + points[:, None] * (end - start))
        polar = iy + iz
        torsion = _torsion(kind, start + points[:, None] * (end - start), polar)
        stiffness = np.stack(
            [
                shear * torsion,
                young * iy,
                young * iz,
                young * area,
                shear * area,
                shear * area,
            ],
            axis=-1,
        )
        damping_diagonal = damping * np.stack(
            [torsion, 3 * iy, 3 * iz, 3 * area, area, area], axis=-1
        )
        assert_allclose(
            robot.params.link.stiffness[0],
            np.diag(length * weights @ stiffness),
            rtol=1e-8,
        )
        assert_allclose(
            robot.params.link.damping[0],
            np.diag(length * weights @ damping_diagonal),
            rtol=1e-8,
        )
        count = int(robot.num_integration_points[0])
        mass_points = np.asarray(robot.integration_points[0, :count])
        area, iy, iz = _properties(kind, start + mass_points[:, None] * (end - start))
        mass = density * np.stack([iy + iz, iy, iz, area, area, area], axis=-1)
        assert_allclose(
            robot.mass_matrices[0, :count],
            jax.vmap(jnp.diag)(jnp.asarray(mass)),
            rtol=1e-8,
        )
