"""Helix preparation: independent reference, beam limits and runtime updates."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVSSegment,
    Helicoidal,
    HelicoidalHelixParams,
    JointSpec,
    LinkSpec,
    StrainBasisSpec,
)
from soromox.systems.gvs._helicoidal_reduction import _fit_backbone, reduce_helix

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def helix():
    return HelicoidalHelixParams.circular(
        length=0.2,
        radius_base=0.025,
        radius_tip=0.025,
        num_turns=3.0,
        strand_radius=0.002,
        strand_shear_factors_override=jnp.full(2, 0.9),
        density=1220.0,
        young_modulus=30e6,
        shear_modulus=30e6 / 2.96,
        damping_time_constant=0.02,
    )


@pytest.fixture(scope="module")
def reduced(helix):
    return reduce_helix(helix)


def test_reference_compliance_at_soft_ends(reduced):
    # Independent upstream assembly at 76c659d, demo_cyl M1. Integration split
    # at PCS element boundaries; platform compliance transported 9.15 mm to
    # the soft tip with T C T.T, then reordered from z- to x-longitudinal.
    # Round insignificant off-diagonals from the less-conditioned full solve.
    expected = np.diag(
        [
            0.6142624906366,
            30.96294847091,
            30.96294952807,
            0.01082371265765,
            0.4024188539729,
            0.4024188609311,
        ]
    )
    expected[1, 5] = expected[5, 1] = -3.080745132300
    expected[2, 4] = expected[4, 2] = 3.080745062313
    assert_allclose(reduced.tip_compliance, expected, rtol=4e-7, atol=2e-8)
    assert reduced.relative_asymmetry < 1e-7
    assert reduced.backbone.admissible_strains == (True, True, True, True, False, False)
    assert reduced.shear_compliance_residual < 0


def test_material_scaling_and_rotation(helix, reduced):
    changed = reduce_helix(
        helix.replace(
            young_modulus=2 * helix.young_modulus,
            shear_modulus=2 * helix.shear_modulus,
            phase=0.41,
        )
    )
    # Threefold winding symmetry makes the diagonal fit azimuth invariant.
    assert_allclose(
        changed.tip_compliance, reduced.tip_compliance / 2, rtol=2e-7, atol=2e-8
    )
    active = np.array(reduced.backbone.admissible_strains)
    assert_allclose(
        changed.backbone.stiffness_base[active],
        2 * reduced.backbone.stiffness_base[active],
        rtol=2e-7,
    )
    assert_allclose(
        changed.backbone.stiffness_tip[active],
        2 * reduced.backbone.stiffness_tip[active],
        rtol=2e-7,
    )


def test_fit_recovers_uniform_timoshenko_beam():
    length, gj, ei, ea, ga = 0.3, 0.7, 0.2, 40.0, 12.0
    tip = np.diag(
        [
            length / gj,
            length / ei,
            length / ei,
            length / ea,
            length**3 / (3 * ei) + length / ga,
            length**3 / (3 * ei) + length / ga,
        ]
    )
    tip[1, 5] = tip[5, 1] = -(length**2) / (2 * ei)
    tip[2, 4] = tip[4, 2] = length**2 / (2 * ei)
    law, residual = _fit_backbone(tip, length, 0.02)
    assert law.admissible_strains == (True,) * 6
    assert_allclose(residual, length / ga)
    assert_allclose(law.stiffness_base, [gj, ei, ei, ea, ga, ga], rtol=1e-12)
    assert_allclose(law.stiffness_tip, law.stiffness_base, rtol=1e-12)
    # Tip moments without force-induced bending rotation cannot support the fit.
    tip[1, 5] = tip[2, 4] = 0
    with pytest.raises(ValueError, match="bending-compliance"):
        _fit_backbone(tip, length, 0.02)


def make_model(helix, active=(True, True, True, True, False, False), **kwargs):
    def segment(length, selector):
        dof = sum(selector)
        return GVSSegment(
            link=LinkSpec.circular(
                length=length,
                radius=0.025,
                density=1.0,
                reference_strain=[0, 0, 0, 1, 0, 0],
                stiffness=jnp.eye(dof),
                damping=0.01 * jnp.eye(dof),
            ),
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="legendre", strain_selector=selector, basis_order=(0,) * 6
            ),
            num_gauss_points=5,
        )

    # Include an independently specified deformable link to catch accidental
    # overwrites, plus a rigid connector with zero coordinates.
    segments = [
        segment(0.03, (True, False, False, False, False, False)),
        segment(0.0183, (False,) * 6),
        segment(0.2, active),
    ]
    return Helicoidal.from_segments(
        segments,
        constitutive_params={2: helix},
        **kwargs,
    )


def test_install_and_update_without_recompilation(helix, reduced):
    model = make_model(helix)
    traces = []

    @eqx.filter_jit
    def dynamics(robot):
        traces.append(True)
        q = jnp.full(robot.num_internal_dofs, 0.01)
        return robot.dynamics_terms(q, 0.1 * q), robot.K, robot.D_active

    before = dynamics(model)
    changed = model.with_constitutive_params(
        {
            2: helix.replace(
                young_modulus=2 * helix.young_modulus,
                shear_modulus=2 * helix.shear_modulus,
            )
        }
    )
    after = dynamics(changed)
    assert len(traces) == 1
    for old, new in zip(before[0], after[0], strict=True):
        assert_allclose(new, old)
    assert_allclose(changed.params.link.stiffness[:2], model.params.link.stiffness[:2])
    assert_allclose(changed.params.link.damping[:2], model.params.link.damping[:2])
    assert_allclose(
        changed.params.link.stiffness[2], 2 * model.params.link.stiffness[2], rtol=2e-7
    )
    assert_allclose(
        changed.params.link.damping[2], 2 * model.params.link.damping[2], rtol=2e-7
    )
    # Rigid connector contributes no stiffness, independent link retains its law.
    assert_allclose(model.params.link.stiffness[0, 0, 0], 1.0)
    assert_allclose(model.params.link.stiffness[1], 0.0)


def test_incompatible_structure_is_rejected(helix, reduced):
    with pytest.raises(ValueError, match="inadmissible"):
        make_model(helix, active=(True,) * 6)
    with pytest.raises(ValueError, match="inadmissible"):
        make_model(reduced.backbone, active=(True,) * 6)
    model = make_model(helix)
    with pytest.raises(ValueError, match="length must match"):
        model.with_constitutive_params({2: helix.replace(length=0.3)})
    with pytest.raises(ValueError, match="segment index"):
        model.with_constitutive_params({3: helix})
    with pytest.raises(TypeError, match="HelicoidalHelixParams"):
        model.with_constitutive_params({2: object()})


def test_shared_inputs_reduce_once_and_match_sequential_updates(helix, monkeypatch):
    model = make_model(helix)
    model = model.update_link_params(length=jnp.array([0.2, 0.0183, 0.2]))
    # Two different strain bases share one lattice description.
    expected = model.with_constitutive_params({0: helix}).with_constitutive_params(
        {2: helix}
    )
    calls = []
    original = HelicoidalHelixParams.to_backbone

    def counted_reduce(params):
        calls.append(params)
        return original(params)

    monkeypatch.setattr(HelicoidalHelixParams, "to_backbone", counted_reduce)
    actual = model.with_constitutive_params({0: helix, 2: helix})
    assert len(calls) == 1
    assert_allclose(actual.K, expected.K)
    assert_allclose(actual.D_active, expected.D_active)
    assert actual.with_constitutive_params({}) is actual

    changed = helix.replace(young_modulus=1.5 * helix.young_modulus)
    actual.with_constitutive_params({0: helix, 2: changed})
    assert len(calls) == 3  # Each distinct input is prepared in the new call.


@pytest.mark.parametrize(
    "field,value",
    [
        ("strand_area_override", -1),
        ("density", 0),
        ("radius_tip", 0),
        ("fusion_stiffness_multiplier", np.nan),
        ("damping_time_constant", -1),
        ("strand_second_moments_override", [1]),
    ],
)
def test_invalid_params(helix, field, value):
    with pytest.raises(ValueError, match=field):
        helix.replace(**{field: value})


@pytest.mark.parametrize(
    "value",
    [
        [0.02],
        [[0.02] * 6],
        [0.02] * 5,
        [0.02] * 5 + [-1],
        [0.02] * 5 + [np.inf],
        [np.nan] * 6,
    ],
)
def test_per_strain_damping_validation(helix, value):
    with pytest.raises(ValueError, match="damping_time_constant"):
        helix.replace(damping_time_constant=value)


def test_per_strain_backbone_damping_survives_helix_reduction(helix, reduced):
    tau = jnp.array([0.002, 0.02, 0.03, 0.0, 0.04, 0.05])
    changed = helix.replace(damping_time_constant=tau)
    backbone = changed.to_backbone()
    assert_allclose(backbone.damping_time_constant, tau)
    assert_allclose(backbone.stiffness_base, reduced.backbone.stiffness_base)
    assert_allclose(backbone.stiffness_tip, reduced.backbone.stiffness_tip)
    assert backbone.admissible_strains == reduced.backbone.admissible_strains
    model = make_model(changed)
    expected = np.zeros_like(model.params.link.damping[2])
    expected[:4, :4] = np.diag(tau[:4] * np.diag(model.params.link.stiffness[2])[:4])
    assert_allclose(model.params.link.damping[2], expected, atol=1e-13)


def test_preparation_requires_eager_inputs(helix):
    with pytest.raises(ValueError, match="outside JAX transformations"):
        eqx.filter_jit(lambda p: p.to_backbone().stiffness_base)(helix)
    model = make_model(helix)
    with pytest.raises(ValueError, match="outside JAX transformations"):
        eqx.filter_jit(lambda p: model.with_constitutive_params({2: p}))(helix)


def test_preparation_without_global_x64(helix, reduced):
    with jax.enable_x64(False):
        params = jax.tree.map(lambda a: a.astype(jnp.float32), helix)
        actual = reduce_helix(params)
        assert not jax.config.x64_enabled
        assert actual.tip_compliance.dtype == jnp.float32
        assert_allclose(
            actual.tip_compliance, reduced.tip_compliance, rtol=3e-6, atol=2e-7
        )


def test_mixed_inputs_and_explicit_conversion_agree(helix, reduced):
    model = make_model(helix).update_link_params(length=jnp.array([0.2, 0.0183, 0.2]))
    backbone = helix.to_backbone()
    converted = model.with_constitutive_params({0: backbone, 2: backbone})
    mixed = model.with_constitutive_params({0: backbone, 2: helix})
    assert_allclose(mixed.K, converted.K, rtol=1e-12)
    assert_allclose(mixed.D_active, converted.D_active, rtol=1e-12)
    assert_allclose(backbone.stiffness_base, reduced.backbone.stiffness_base)
    assert_allclose(make_model(backbone).K, make_model(helix).K, rtol=1e-12)
    updated = backbone.replace(stiffness_tip=1.2 * backbone.stiffness_tip)
    assert updated.admissible_strains == backbone.admissible_strains
    with pytest.raises(ValueError, match="inadmissible"):
        make_model(updated, active=(True,) * 6)
    with pytest.raises(ValueError, match="fixed"):
        backbone.replace(admissible_strains=(True,) * 6)


def test_public_reduction_diagnostics(helix, reduced):
    diagnostics = helix.reduction_diagnostics()
    assert set(diagnostics) == {
        "tip_compliance",
        "shear_compliance_residual",
        "relative_asymmetry",
    }
    assert_allclose(diagnostics["tip_compliance"], reduced.tip_compliance)
    assert_allclose(
        diagnostics["shear_compliance_residual"], reduced.shear_compliance_residual
    )
    assert_allclose(diagnostics["relative_asymmetry"], reduced.relative_asymmetry)


@pytest.mark.parametrize("kind", ["circular", "rectangular"])
@pytest.mark.parametrize("hollow", [False, True])
def test_geometry_law_and_calibrated_properties_reduce_identically(helix, kind, hollow):
    from soromox.systems import StrandCrossSection

    if kind == "circular":
        section = (
            StrandCrossSection.circular_hollow(radius=0.002, wall_thickness=0.0002)
            if hollow
            else StrandCrossSection.circular(0.002)
        )
    else:
        section = (
            StrandCrossSection.rectangular_hollow(
                height=0.002, width=0.004, wall_thickness=0.0002
            )
            if hollow
            else StrandCrossSection.rectangular(height=0.002, width=0.004)
        )
    automatic = helix.replace(
        strand_cross_section=section, strand_shear_factors_override=None
    )
    props = automatic.strand_cross_section_properties
    calibrated = automatic.replace(
        strand_cross_section=None,
        strand_area_override=props.area,
        strand_second_moments_override=props.second_moments,
        strand_torsion_constant_override=props.torsion_constant,
        strand_shear_factors_override=props.shear_factors,
    )
    actual, expected = reduce_helix(automatic), reduce_helix(calibrated)
    assert np.linalg.eigvalsh(actual.tip_compliance).min() > 0
    assert actual.backbone.admissible_strains == expected.backbone.admissible_strains
    assert_allclose(
        actual.tip_compliance, expected.tip_compliance, rtol=2e-8, atol=2e-8
    )
    assert_allclose(
        actual.backbone.stiffness_base, expected.backbone.stiffness_base, rtol=2e-8
    )
    assert_allclose(
        actual.backbone.stiffness_tip, expected.backbone.stiffness_tip, rtol=2e-8
    )
    assert_allclose(
        automatic.sectional_inertia(jnp.array([0.0, 0.1, 0.2])),
        calibrated.sectional_inertia(jnp.array([0.0, 0.1, 0.2])),
    )


def test_default_cowper_shear_enters_lattice_compliance(helix, reduced):
    automatic = helix.replace(strand_shear_factors_override=None)
    expected_factor = 6 * (1 + 0.48) / (7 + 6 * 0.48)
    assert_allclose(automatic.strand_shear_factors, expected_factor)
    result = reduce_helix(automatic)
    assert np.linalg.eigvalsh(result.tip_compliance).min() > 0
    assert np.linalg.norm(result.tip_compliance - reduced.tip_compliance) > 1e-7
    assert_allclose(
        result.tip_compliance,
        reduce_helix(
            helix.replace(strand_shear_factors_override=jnp.full(2, expected_factor))
        ).tip_compliance,
    )
