"""Sectional limits, independent references and material projection regressions."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from systems.test_gvs_hollow_links import (
    LENGTH,
    link_spec,
    model,
    numerical_rectangle_shear,
)

from soromox.systems import (
    GVSLinkStructure,
    IsotropicMaterialParams,
    RodConstitutiveLaw,
    TorsionRodModel,
    TransverseShearRodModel,
)
from soromox.systems.components import CrossSectionGeometry as Geometry
from soromox.systems.components.cross_sections import section_properties

jax.config.update("jax_enable_x64", True)
EFFECTIVE = RodConstitutiveLaw(
    transverse_shear=TransverseShearRodModel.EFFECTIVE_AREA,
    torsion=TorsionRodModel.SAINT_VENANT,
)


@pytest.mark.parametrize("nu", [-0.5, 0.0, 0.3, 0.49, 0.5])
def test_cowper_annulus_has_solid_and_thin_wall_limits(nu):
    outer = jnp.array([0.02])
    area = np.pi * outer[0] ** 2
    solid = EFFECTIVE.shear_areas(Geometry.CIRCULAR, outer, poisson_ratio=nu)
    assert_allclose(solid / area, 6 * (1 + nu) / (7 + 6 * nu))
    almost_solid = EFFECTIVE.shear_areas(
        Geometry.CIRCULAR, outer, outer * 1e-7, poisson_ratio=nu
    )
    assert_allclose(almost_solid, solid, rtol=1e-12)
    inner = outer * (1 - 1e-6)
    thin_area = section_properties(Geometry.CIRCULAR, outer, inner)[3]
    thin = EFFECTIVE.shear_areas(Geometry.CIRCULAR, outer, inner, poisson_ratio=nu)
    assert_allclose(thin / thin_area, 2 * (1 + nu) / (4 + 3 * nu), rtol=1e-10)
    assert_allclose(
        EFFECTIVE.torsion_constant(Geometry.CIRCULAR, outer, inner),
        np.pi / 2 * (outer[0] ** 4 - inner[0] ** 4),
    )


@pytest.mark.parametrize("ratio", [1.0, 2.0, 5.0, 20.0])
@pytest.mark.parametrize("fraction", [1e-4, 0.01, 0.1, 0.4, 0.499999])
def test_rectangular_energy_areas_match_independent_integration(ratio, fraction):
    outer = np.array([0.02, 0.02 * ratio])
    inner = outer - 2 * fraction * 0.02
    areas = EFFECTIVE.shear_areas(Geometry.RECTANGULAR, outer, inner)
    assert_allclose(areas, numerical_rectangle_shear(outer, inner), rtol=1e-9)
    area = section_properties(Geometry.RECTANGULAR, outer, inner)[3]
    assert np.all((areas > 0) & (areas <= area))
    assert_allclose(
        EFFECTIVE.shear_areas(Geometry.RECTANGULAR, outer[::-1], inner[::-1]),
        areas[::-1],
    )


@pytest.mark.parametrize("ratio", [1.0, 2.0, 10.0])
def test_rectangular_shear_and_marshall_have_continuous_solid_limits(ratio):
    outer = jnp.array([0.02, 0.02 * ratio])
    inner = outer - 0.02 * (1 - 1e-10)
    solid = EFFECTIVE.shear_areas(Geometry.RECTANGULAR, outer)
    assert_allclose(solid, np.full(2, np.prod(outer) * 5 / 6))
    assert_allclose(
        EFFECTIVE.shear_areas(Geometry.RECTANGULAR, outer, inner), solid, rtol=3e-9
    )
    assert_allclose(
        EFFECTIVE.torsion_constant(Geometry.RECTANGULAR, outer, inner),
        EFFECTIVE.torsion_constant(Geometry.RECTANGULAR, outer),
        rtol=3e-9,
    )


@pytest.mark.parametrize(
    "ratio,relative_error",
    [(1.0, 0.03739099), (2.0, -0.0206547), (5.0, -0.0188229), (10.0, -0.0107795)],
)
def test_marshall_solid_error_against_series(ratio, relative_error):
    outer = jnp.array([1.0, ratio])
    # Independent high-order Saint-Venant reference, used only for validation
    # of Marshall's engineering extension to solid rectangles.
    n = np.arange(1, 20000, 2)
    exact = (
        ratio
        / 3
        * (
            1
            - 192
            / (np.pi**5 * ratio)
            * np.sum(np.tanh(n * np.pi * ratio / 2) / n.astype(float) ** 5)
        )
    )
    actual = EFFECTIVE.torsion_constant(Geometry.RECTANGULAR, outer)
    assert_allclose(actual / exact - 1, relative_error, atol=8e-8)


@pytest.mark.parametrize("fraction", [0.01, 0.05, 0.1])
def test_marshall_thin_wall_correction(fraction):
    outer = jnp.array([1.0, 1.0])
    inner = outer - 2 * fraction
    mid_area = (1 - fraction) ** 2
    perimeter = 4 * (1 - fraction)
    bredt = 4 * fraction * mid_area**2 / perimeter
    actual = EFFECTIVE.torsion_constant(Geometry.RECTANGULAR, outer, inner)
    assert_allclose(actual / bredt - 1, perimeter**2 * fraction**2 / (12 * mid_area**2))


def test_static_law_validation_and_hollow_rejection():
    with pytest.raises(TypeError, match="transverse_shear"):
        RodConstitutiveLaw(transverse_shear="effective_area")
    for kind in ["circular", "rectangular"]:
        with pytest.raises(ValueError, match="UNIFORM"):
            replace(link_spec(kind), constitutive_law=RodConstitutiveLaw())
    with pytest.raises(ValueError, match="SAINT_VENANT"):
        replace(
            link_spec("rectangular"),
            constitutive_law=RodConstitutiveLaw(
                transverse_shear=TransverseShearRodModel.EFFECTIVE_AREA
            ),
        )
    with pytest.raises(ValueError, match="circular and rectangular"):
        replace(link_spec("elliptical"), constitutive_law=EFFECTIVE)
    with pytest.raises(ValueError, match="UNIFORM"):
        GVSLinkStructure(
            cross_section_geometry=Geometry.CIRCULAR,
            cross_section_profile_types=("constant", "constant"),
            cross_section_profile_parameter_counts=(1, 1),
            hollow=True,
        )


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
@pytest.mark.parametrize(
    "invalid_law,message",
    [
        (RodConstitutiveLaw(), "UNIFORM"),
        (
            RodConstitutiveLaw(transverse_shear=TransverseShearRodModel.EFFECTIVE_AREA),
            "POLAR_MOMENT",
        ),
    ],
)
def test_updates_reject_incompatible_hollow_laws_even_under_jit(
    kind, invalid_law, message
):
    robot = model(link_spec(kind))
    # Static structure choices are revalidated even if changed through a low-level
    # dataclass replacement that bypasses a LinkSpec factory.
    invalid_link = object.__new__(GVSLinkStructure)
    for name in robot.structure.segments[0].link.__dataclass_fields__:
        object.__setattr__(
            invalid_link, name, getattr(robot.structure.segments[0].link, name)
        )
    object.__setattr__(invalid_link, "constitutive_law", invalid_law)
    invalid_structure = replace(
        robot.structure,
        segments=(replace(robot.structure.segments[0], link=invalid_link),),
    )
    with pytest.raises(ValueError, match=message):
        robot.params.validate_for_update_against_structure(invalid_structure)

    with pytest.raises(ValueError, match=message):
        eqx.filter_jit(
            lambda params: params.validate_for_update_against_structure(
                invalid_structure
            )
        )(robot.params)


@pytest.mark.parametrize("kind", ["circular", "rectangular", "elliptical"])
def test_default_solid_matrices_preserve_released_uniform_shear_and_polar_torsion(kind):
    robot = model(link_spec(kind, wall=None))
    geometry, outer, _ = robot.cross_section_geometry(jnp.zeros(6), LENGTH / 2)
    geometry = Geometry(int(geometry))
    ix, iy, iz, area = section_properties(geometry, outer)
    torsion = ix  # Released GVS law uses the polar area moment for every solid.
    expected_k = LENGTH * jnp.array(
        [4e5 * torsion, 1e6 * iy, 1e6 * iz, 1e6 * area, 4e5 * area, 4e5 * area]
    )
    expected_d = (
        LENGTH * 100 * jnp.array([torsion, 3 * iy, 3 * iz, 3 * area, area, area])
    )
    assert_allclose(robot.K, jnp.diag(expected_k), atol=1e-10)
    assert_allclose(robot.D_active, jnp.diag(expected_d), atol=1e-10)


@pytest.mark.parametrize("wall", [None, 0.002])
def test_cowper_material_updates_rebuild_shear_and_damping_with_gradients(wall):
    robot = model(replace(link_spec("circular", wall=wall), constitutive_law=EFFECTIVE))

    def matrices(nu):
        return robot.link_matrices_from_material(
            IsotropicMaterialParams(
                young_modulus=1e6, poisson_ratio=nu, material_damping_coefficient=100.0
            )
        )

    compiled = eqx.filter_jit(matrices)
    k, d = compiled(jnp.array(0.3))
    e_g = robot.link_matrices_from_material(
        IsotropicMaterialParams(
            young_modulus=1e6,
            shear_modulus=1e6 / 2.6,
            material_damping_coefficient=100.0,
        )
    )
    for actual, expected in zip((k, d), e_g, strict=True):
        assert_allclose(actual, expected, atol=1e-10)
    rebuilt = model(
        replace(
            link_spec("circular", wall=wall),
            constitutive_law=EFFECTIVE,
            shear_modulus=None,
            poisson_ratio=0.3,
        )
    )
    assert_allclose(k, rebuilt.params.link.stiffness, atol=1e-10)
    assert_allclose(d, rebuilt.params.link.damping, atol=1e-10)
    derivative = jax.grad(lambda nu: matrices(nu)[0][0, 4, 4])(jnp.array(0.3))
    finite_difference = (
        matrices(0.300001)[0][0, 4, 4] - matrices(0.299999)[0][0, 4, 4]
    ) / 2e-6
    assert_allclose(derivative, finite_difference, rtol=1e-8)
    assert not np.allclose(d, matrices(0.0)[1])
    with pytest.raises(ValueError, match="Cowper"):
        robot.with_isotropic_material(
            IsotropicMaterialParams(young_modulus=1e6, shear_modulus=1e5)
        )


def test_explicit_matrices_do_not_need_poisson_ratio():
    spec = replace(
        link_spec("circular"),
        young_modulus=None,
        shear_modulus=None,
        material_damping_coefficient=None,
        stiffness=jnp.eye(6),
        damping=2 * jnp.eye(6),
    )
    robot = model(spec)
    assert_allclose(robot.K, jnp.eye(6))
    assert_allclose(robot.D_active, 2 * jnp.eye(6))
    with pytest.raises(ValueError, match="elastic moduli"):
        model(replace(spec, damping=None, material_damping_coefficient=100.0))


def test_rectangular_geometry_derivatives_match_finite_differences():
    outer = jnp.array([0.02, 0.04])

    def properties(t):
        inner = outer - 2 * t
        return jnp.concatenate(
            (
                EFFECTIVE.shear_areas(Geometry.RECTANGULAR, outer, inner),
                EFFECTIVE.torsion_constant(Geometry.RECTANGULAR, outer, inner)[None],
            )
        )

    derivative = jax.jit(jax.jacfwd(properties))(jnp.array(0.003))
    assert_allclose(
        derivative, (properties(0.0030001) - properties(0.0029999)) / 2e-7, rtol=1e-8
    )


def test_mixed_links_use_their_own_law_and_poisson_ratio():
    uniform = link_spec("circular", wall=None)
    corrected = replace(link_spec("circular", wall=None), constitutive_law=EFFECTIVE)
    robot = model(uniform, corrected)
    material = IsotropicMaterialParams(
        young_modulus=jnp.array([1e6, 2e6]),
        shear_modulus=jnp.array([1e5, 8e5]),
        material_damping_coefficient=jnp.array([100.0, 200.0]),
    )
    # The uniform model keeps accepting independently calibrated E and G.
    # Only Cowper's model needs a physically admissible inferred Poisson ratio.
    k, d = robot.link_matrices_from_material(material)
    area = np.pi * 0.03**2
    factor = 6 * 1.25 / 8.5
    assert_allclose(k[:, 4, 4], LENGTH * area * np.array([1e5, 8e5 * factor]))
    assert_allclose(d[:, 4, 4], LENGTH * area * np.array([100, 200 * factor]))
    with pytest.raises(ValueError, match="Cowper"):
        model(replace(corrected, shear_modulus=1e5))


@pytest.mark.parametrize("hollow", [False, True])
def test_closed_over_poisson_array_allows_compiled_geometry_updates(hollow):
    nu = jnp.array(0.3)

    def areas(radius):
        outer = jnp.reshape(radius, (1,))
        inner = outer / 2 if hollow else None
        return EFFECTIVE.shear_areas(Geometry.CIRCULAR, outer, inner, poisson_ratio=nu)

    compiled = jax.jit(areas)
    for radius in [0.02, 0.03]:
        assert_allclose(compiled(jnp.array(radius)), areas(jnp.array(radius)))
    derivative = jax.jit(jax.jacfwd(areas))(jnp.array(0.02))
    # With fixed cavity ratio and material, effective area scales as radius².
    assert_allclose(derivative, 2 * areas(jnp.array(0.02)) / 0.02)


@pytest.mark.parametrize("nu", [-1.0, 0.5001, float("nan")])
def test_invalid_eager_poisson_ratios_remain_rejected(nu):
    with pytest.raises(ValueError, match="Cowper shear requires"):
        EFFECTIVE.shear_areas(
            Geometry.CIRCULAR, jnp.array([0.02]), poisson_ratio=jnp.array(nu)
        )


@pytest.mark.parametrize("geometry", list(Geometry))
@pytest.mark.parametrize("hollow", [False, True])
@pytest.mark.parametrize("shear", list(TransverseShearRodModel))
@pytest.mark.parametrize("torsion", list(TorsionRodModel))
def test_law_geometry_support_matrix_including_jit(geometry, hollow, shear, torsion):
    """All public model combinations obey the same restrictions at every layer."""
    law = RodConstitutiveLaw(transverse_shear=shear, torsion=torsion)
    supported = (
        not hollow
        or (
            shear == TransverseShearRodModel.EFFECTIVE_AREA
            and torsion == TorsionRodModel.SAINT_VENANT
        )
    ) and (
        geometry != Geometry.ELLIPTICAL
        or (not hollow and shear == TransverseShearRodModel.UNIFORM)
    )
    outer = jnp.array([0.03] if geometry == Geometry.CIRCULAR else [0.04, 0.06])
    inner = (
        outer - (0.002 if geometry == Geometry.CIRCULAR else 0.004) if hollow else None
    )
    count = len(outer) + int(hollow)

    def structure():
        return GVSLinkStructure(
            cross_section_geometry=geometry,
            cross_section_profile_types=("constant",) * count,
            cross_section_profile_parameter_counts=(1,) * count,
            hollow=hollow,
            constitutive_law=law,
        )

    def areas(x):
        return law.shear_areas(geometry, x, inner)

    def twist(x):
        return law.torsion_constant(geometry, x, inner)

    calls = (
        lambda: law.validate_geometry(geometry, hollow),
        structure,
        lambda: areas(outer),
        lambda: twist(outer),
        lambda: jax.jit(areas)(outer),
        lambda: jax.jit(twist)(outer),
    )
    for call in calls:
        if supported:
            call()
        else:
            with pytest.raises(ValueError):
                call()


def test_torsion_choice_requires_enum():
    with pytest.raises(TypeError, match="torsion"):
        RodConstitutiveLaw(torsion="saint_venant")


@pytest.mark.parametrize(
    "geometry,outer",
    [
        (Geometry.CIRCULAR, [[0.02], [0.03]]),
        (Geometry.RECTANGULAR, [[0.02, 0.04], [0.03, 0.05]]),
    ],
)
def test_shared_properties_batch_and_jit(geometry, outer):
    outer = jnp.array(outer)
    inner = outer * 0.5 if geometry == Geometry.CIRCULAR else outer - 0.004
    properties = eqx.filter_jit(
        lambda dims: EFFECTIVE.cross_section_properties(
            geometry, dims, inner, poisson_ratio=0.3
        )
    )(outer)
    _, iy, iz, area = section_properties(geometry, outer, inner)
    assert_allclose(properties.area, area)
    assert_allclose(properties.second_moments, jnp.stack((iy, iz), axis=-1))
    assert_allclose(
        properties.shear_areas,
        EFFECTIVE.shear_areas(geometry, outer, inner, poisson_ratio=0.3),
    )
    assert_allclose(
        properties.torsion_constant, EFFECTIVE.torsion_constant(geometry, outer, inner)
    )
    calibrated = EFFECTIVE.cross_section_properties(
        geometry,
        outer,
        inner,
        poisson_ratio=2.0,
        shear_factors_override=jnp.array([0.7, 0.8]),
    )
    assert_allclose(calibrated.shear_areas, area[:, None] * jnp.array([0.7, 0.8]))


@pytest.mark.parametrize("factors", [[0.8], [0.8, 0], [np.nan, 0.8], [0.8, -0.2]])
def test_shared_properties_reject_invalid_calibrated_shear(factors):
    with pytest.raises(ValueError, match="shear_factors_override"):
        EFFECTIVE.cross_section_properties(
            Geometry.CIRCULAR,
            jnp.array([0.02]),
            shear_factors_override=jnp.array(factors),
        )
