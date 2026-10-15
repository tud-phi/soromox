"""General sectional and attached-body inertia across spatial PCS/GVS backends."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
    spatial_inertia,
)

jax.config.update("jax_enable_x64", True)


def make_model(family, floating=False):
    dimension = 3 if family == "planar" else 6
    links = [
        LinkSpec.circular(
            length=length,
            radius=0.02,
            density=1000.0,
            stiffness=jnp.eye(dimension),
            damping=jnp.eye(dimension) * 0.01,
            reference_strain=[0.0, 1.0, 0.0]
            if dimension == 3
            else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        for length in (0.12, 0.08)
    ]
    if family == "planar":
        model = PlanarPCS.from_links(
            links,
            structure=PlanarPCSStructure(num_gauss_points=5),
            floating_base=floating,
        )
    elif family == "pcs":
        model = PCS.from_links(
            links, structure=PCSStructure(num_gauss_points=5), floating_base=floating
        )
    else:
        segments = [
            GVSSegment(
                link=link,
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="legendre", strain_selector=[True] * 6, basis_order=[0] * 6
                ),
                num_gauss_points=5,
            )
            for link in links
        ]
        model = GVS.from_segments(segments, floating_base=floating)
    section = spatial_inertia(
        jnp.array([0.2, 0.3]),
        jnp.array([[0.001, 0.004, -0.002], [-0.002, 0.003, 0.005]]),
        jnp.tile(jnp.diag(jnp.array([2e-4, 1e-4, 1.5e-4]))[None], (2, 1, 1)),
    )
    tip = spatial_inertia(
        jnp.array([0.04, 0.08]),
        jnp.array([[0.01, -0.012, 0.003], [0.015, 0.02, -0.01]]),
        jnp.tile(jnp.diag(jnp.array([1e-5, 2e-5, 3e-5]))[None], (2, 1, 1)),
    )
    if dimension == 3:
        section = spatial_inertia(
            jnp.array([0.2, 0.3]),
            jnp.array([[0.004, -0.002], [0.003, 0.005]]),
            jnp.array([1e-4, 2e-4]),
        )
        tip = spatial_inertia(
            jnp.array([0.04, 0.08]),
            jnp.array([[-0.012, 0.003], [0.02, -0.01]]),
            jnp.array([2e-5, 3e-5]),
        )
    return model.update_link_params(sectional_inertia=section, tip_inertia=tip)


@pytest.mark.parametrize("family", ["pcs", "gvs", "planar"])
@pytest.mark.parametrize("floating", [False, True])
def test_full_inertia_energy_consistency_and_backend_agreement(family, floating):
    model = make_model(family, floating)
    internal = jnp.linspace(-0.12, 0.16, model.num_internal_dofs)
    if floating:
        base = (
            jnp.array([0.15, 0.02, -0.04])
            if family == "planar"
            else jnp.array([0.1, -0.15, 0.05, 1.0, 0.02, -0.03, 0.04])
        )
        q = model.normalize_configuration(jnp.concatenate((base, internal)))
    else:
        q = internal
    v = jnp.linspace(0.07, -0.11, model.num_velocities)
    M, b, G = model.dynamics_terms(q, v, backend="jax")
    assert_allclose(M, model.inertia_matrix(q), atol=1e-12)
    C = model.coriolis_matrix(q, v)
    assert_allclose(b, C @ v, atol=1e-11)
    qdot = model.configuration_derivative(q, v)
    mapping = jax.jacfwd(lambda vel: model.configuration_derivative(q, vel))(
        jnp.zeros_like(v)
    )
    assert_allclose(G, mapping.T @ jax.grad(model._gravitational_energy)(q), atol=1e-10)
    Mdot = jax.jvp(model.inertia_matrix, (q,), (qdot,))[1]
    assert_allclose(v @ b, 0.5 * v @ Mdot @ v, atol=1e-11)
    assert np.linalg.eigvalsh(M).min() > 0
    if jax.default_backend() == "gpu":
        for actual, expected in zip(
            model.dynamics_terms(q, v, backend="warp"), (M, b, G)
        ):
            assert_allclose(actual, expected, rtol=2e-8, atol=2e-11)


@pytest.mark.parametrize("family", ["pcs", "gvs", "planar"])
def test_updates_reuse_compilation_and_zero_tip_can_be_activated(family):
    model = make_model(family)
    zero = model.update_link_params(
        tip_inertia=jnp.zeros_like(model.params.link.tip_inertia)
    )
    traces = []

    @eqx.filter_jit
    def evaluate(robot):
        traces.append(True)
        q = jnp.full(robot.num_coordinates, 0.03)
        return robot.dynamics_terms(q, q * 0.1)

    baseline = evaluate(zero)
    changed = evaluate(model)
    assert len(traces) == 1
    assert np.linalg.norm(changed[0] - baseline[0]) > 0
    update = model.update_link_params(
        sectional_inertia=1.3 * model.params.link.sectional_inertia
    )
    evaluate(update)
    assert len(traces) == 1


@pytest.mark.parametrize("family", ["pcs", "gvs", "planar"])
def test_sampled_section_inertia_and_parameter_gradients(family):
    model = make_model(family)
    sections = (
        model.params.link.sectional_inertia[:, None]
        * jnp.linspace(0.7, 1.3, 5)[None, :, None, None]
    )
    model = model.update_link_params(sectional_inertia=sections)
    q = jnp.full(model.num_coordinates, 0.04)
    v = jnp.linspace(-0.1, 0.13, model.num_velocities)
    expected = model.dynamics_terms(q, v, backend="jax")
    if jax.default_backend() == "gpu":
        actual = model.dynamics_terms(q, v, backend="warp")
        for x, y in zip(actual, expected):
            assert_allclose(x, y, rtol=2e-8, atol=2e-11)

    def kinetic(scale):
        updated = model.update_link_params(sectional_inertia=scale * sections)
        return 0.5 * v @ updated.inertia_matrix(q) @ v

    derivative = jax.grad(kinetic)(jnp.array(1.0))
    assert_allclose(derivative, kinetic(1.0) - kinetic(0.0), rtol=1e-10, atol=1e-14)


def test_spatial_inertia_matches_rigid_body_kinetic_energy():
    for planar in (False, True):
        mass = jnp.array([0.2, 0.5])
        center = jnp.array([0.01, -0.02]) if planar else jnp.array([0.01, -0.02, 0.03])
        rotation = (
            jnp.array(2e-4) if planar else jnp.diag(jnp.array([1e-4, 2e-4, 3e-4]))
        )
        matrix = spatial_inertia(mass, center, rotation)
        if planar:
            velocity = jnp.array([0.7, 0.2, -0.1])
            center_velocity = velocity[1:] + velocity[0] * jnp.array(
                [-center[1], center[0]]
            )
            rotational_energy = rotation * velocity[0] ** 2
        else:
            velocity = jnp.array([0.7, -0.5, 0.3, 0.2, -0.1, 0.4])
            center_velocity = velocity[3:] + jnp.cross(velocity[:3], center)
            rotational_energy = velocity[:3] @ rotation @ velocity[:3]
        assert_allclose(
            jnp.einsum("i,nij,j->n", velocity, matrix, velocity),
            rotational_energy + mass * (center_velocity @ center_velocity),
            atol=1e-14,
        )


def test_gvs_full_inertia_with_joint_rigid_link_and_quadrature_padding():
    link = LinkSpec.circular(
        length=0.1,
        radius=0.02,
        density=1000.0,
        young_modulus=1e5,
        poisson_ratio=0.3,
        reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    segments = [
        GVSSegment(
            link=link,
            joint=JointSpec(type="revolute", axis="z"),
            basis=StrainBasisSpec(
                type="legendre", strain_selector=[True] * 6, basis_order=[0] * 6
            ),
            num_gauss_points=5,
        ),
        GVSSegment(
            link=link,
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="legendre", strain_selector=[False] * 6, basis_order=[0] * 6
            ),
            num_gauss_points=7,
        ),
    ]
    model = GVS.from_segments(segments)
    tip = spatial_inertia(
        jnp.array([0.02, 0.05]), jnp.array([0.01, 0.02, -0.01]), jnp.eye(3) * 1e-5
    )
    model = model.update_link_params(tip_inertia=tip)
    q = jnp.linspace(-0.1, 0.2, model.num_coordinates)
    v = jnp.linspace(0.07, -0.13, model.num_velocities)
    M, bias, gravity = model.dynamics_terms(q, v, backend="jax")
    assert_allclose(gravity, jax.grad(model._gravitational_energy)(q), atol=1e-11)
    assert_allclose(
        v @ bias, 0.5 * v @ jax.jvp(model.inertia_matrix, (q,), (v,))[1] @ v, atol=1e-11
    )
    if jax.default_backend() == "gpu":
        for actual, expected in zip(
            model.dynamics_terms(q, v, backend="warp"), (M, bias, gravity)
        ):
            assert_allclose(actual, expected, rtol=2e-8, atol=2e-11)


@pytest.mark.parametrize("family", ["pcs", "gvs", "planar"])
def test_partial_sectional_overrides_preserve_geometry_density_updates(family):
    from soromox.systems.components.inertia import sectional_inertias

    model = make_model(family)
    partial = model.update_link_params(sectional_inertia_mask=jnp.array([True, False]))
    # Use the shared resolver on known section geometry to test planar/spatial
    # broadcasting independently of each backend's cached quadrature layout.
    d = model.params.link.reference_strain.shape[1]
    geometry = jnp.broadcast_to(jnp.eye(d), (2, 5, d, d))
    resolved = sectional_inertias(partial.params.link, geometry)
    assert_allclose(
        resolved[0], jnp.broadcast_to(model.params.link.sectional_inertia[0], (5, d, d))
    )
    assert_allclose(resolved[1], geometry[1])
    changed = partial.update_link_params(density=2 * partial.params.link.density)
    q = jnp.zeros(partial.num_internal_dofs)
    all_geometric = model.update_link_params(sectional_inertia=None)
    # Only the unmasked second link responds to density updates.
    second_only = all_geometric.update_link_params(
        density=all_geometric.params.link.density.at[1].multiply(2)
    )
    assert_allclose(
        changed.inertia_matrix(q) - partial.inertia_matrix(q),
        second_only.inertia_matrix(q) - all_geometric.inertia_matrix(q),
        rtol=1e-10,
        atol=1e-12,
    )
