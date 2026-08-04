# ruff: noqa: E402
import equinox as eqx
import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    pcs_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)

from soromox.actuation import (
    BaseThreadlikeRoutingParams,
    IdentityActuator,
    IdentityTransmission,
    ThreadlikeActuator,
    ThreadlikeImpedance,
    ThreadlikeRouting,
)
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    StrainBasisSpec,
)
from soromox.systems.pcs import PlanarPCSStructure


def _spatial_pcs(*, actuators=None, passive_elements=()):
    body = pcs_params(
        base_pose=spatial_base_pose(),
        length=jnp.array([0.1]),
        radius=jnp.array([0.02]),
        density=jnp.array([1000.0]),
        gravity=jnp.zeros((3,)),
        young_modulus=jnp.array([1e3]),
        shear_modulus=jnp.array([5e2]),
        damping_matrix=1e-3 * jnp.eye(6),
    )
    return PCS(
        body,
        PCSStructure(num_gauss_points=5),
        actuators=actuators,
        passive_elements=passive_elements,
    )


def _planar_pcs(*, actuators=None, passive_elements=()):
    body = planar_pcs_params(
        base_pose=planar_base_pose(),
        length=jnp.array([0.1]),
        radius=jnp.array([0.02]),
        density=jnp.array([1000.0]),
        gravity=jnp.zeros((2,)),
        young_modulus=jnp.array([1e3]),
        shear_modulus=jnp.array([5e2]),
        damping_matrix=1e-3 * jnp.eye(3),
    )
    return PlanarPCS(
        body,
        PlanarPCSStructure(num_gauss_points=5),
        actuators=actuators,
        passive_elements=passive_elements,
    )


def _gvs(*, actuators=None, passive_elements=()):
    segment = GVSSegment(
        link=LinkSpec.circular(
            young_modulus=3e5,
            shear_modulus=3e5 / 2.9,
            density=1300.0,
            material_damping_coefficient=1e4,
            length=0.1,
            radius=0.015,
            reference_strain=[0, 0, 0, 1, 0, 0],
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="monomial",
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        ),
        num_gauss_points=5,
    )
    return GVS.from_segments(
        [segment],
        gravity=jnp.zeros((3,)),
        actuators=actuators,
        passive_elements=passive_elements,
    )


def _routing(*, count=2):
    y_offsets = jnp.linspace(-0.01, 0.01, count)
    y_gradients = jnp.linspace(0.0, 0.01, count)
    return ThreadlikeRouting.linear(
        intercept=jnp.stack(
            (jnp.zeros((count,)), y_offsets, jnp.zeros((count,))), axis=-1
        ),
        slope=jnp.stack(
            (jnp.zeros((count,)), y_gradients, jnp.zeros((count,))), axis=-1
        ),
        start_segment_index=(0,) * count,
        end_segment_index=(0,) * count,
    )


class SinusoidalRoutingParams(BaseThreadlikeRoutingParams):
    amplitude: jax.Array
    frequency: jax.Array
    start_segment_index: tuple[int, ...] = eqx.field(static=True)
    end_segment_index: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_paths(self) -> int:
        return int(self.amplitude.shape[0])

    def validate(self) -> None:
        if self.frequency.shape != self.amplitude.shape:
            raise ValueError("frequency must match amplitude shape.")


def _sinusoidal_offset(params: SinusoidalRoutingParams, s):
    y = params.amplitude * jnp.sin(params.frequency * s)
    return jnp.asarray([0.0, y, 0.0])


def _sinusoidal_derivative(params: SinusoidalRoutingParams, s):
    y_dot = params.amplitude * params.frequency * jnp.cos(params.frequency * s)
    return jnp.asarray([0.0, y_dot, 0.0])


def test_linear_routing_uses_full_material_frame_vectors():
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.02, 0.03], [0.0, 0.05, 0.06]]),
        slope=jnp.array([0.0, 0.2, 0.3]),
        end_segment_index=(0, 0),
    )

    assert routing.params.intercept.shape == (2, 3)
    assert routing.params.slope.shape == (2, 3)
    assert_allclose(
        routing.offset(routing.params, jnp.array(0.5)),
        routing.params.intercept + 0.5 * routing.params.slope,
    )


@pytest.mark.parametrize(
    ("intercept", "slope"),
    [
        (jnp.array([0.01, 0.02, 0.03]), jnp.zeros(3)),
        (jnp.array([0.0, 0.02, 0.03]), jnp.array([0.01, 0.0, 0.0])),
    ],
)
def test_linear_routing_rejects_longitudinal_components(intercept, slope):
    with pytest.raises(ValueError, match="local-x components must be zero"):
        ThreadlikeRouting.linear(intercept=intercept, slope=slope)


def test_identity_unactuated_and_mixed_construction():
    identity = _spatial_pcs()
    assert identity.num_actuators == identity.num_dofs
    assert isinstance(identity.actuators[0].transmission, IdentityTransmission)
    assert_allclose(identity.actuation_matrix(jnp.zeros(identity.num_dofs)), jnp.eye(6))

    unactuated = _spatial_pcs(actuators=())
    assert unactuated.num_actuators == 0
    assert unactuated.actuation_matrix(jnp.zeros(unactuated.num_dofs)).shape == (6, 0)

    routing = _routing(count=1)
    mixed = _spatial_pcs(
        actuators=(
            ThreadlikeActuator.tendons(routing),
            ThreadlikeActuator.push_rods(routing),
        )
    )
    q = jnp.zeros(mixed.num_dofs)
    raw = mixed._threadlike_moment_matrix(q, routing)
    assert_allclose(mixed.actuation_matrix(q), jnp.concatenate((-raw, raw), axis=1))
    assert [metadata.kind for metadata in mixed.actuator_input_metadata] == [
        "tendon",
        "push_rod",
    ]


def test_identity_actuator_rejects_channel_count_mismatch_during_construction():
    with pytest.raises(ValueError, match="one channel per robot degree of freedom"):
        _spatial_pcs(actuators=IdentityActuator(3))


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs, _gvs])
def test_coordinate_jacobian_equals_moment_matrix_transpose(factory):
    actuator = ThreadlikeActuator.tendons(_routing())
    robot = factory(actuators=actuator)
    q = jnp.linspace(-0.02, 0.03, robot.num_dofs)
    jacobian = jax.jacrev(robot.actuator_coordinates)(q)
    assert_allclose(jacobian, robot.actuation_matrix(q).T, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize(
    ("factory", "axial_index"),
    [(_spatial_pcs, 3), (_planar_pcs, 1), (_gvs, 3)],
)
def test_collapsed_threadlike_tangent_has_finite_reverse_mode(factory, axial_index):
    """Cover normalization and path density at an exactly zero routing tangent."""
    actuator = ThreadlikeActuator.tendons(_routing(count=1))
    robot = factory(actuators=actuator)
    q = jnp.zeros(robot.num_dofs).at[axial_index].set(-1.0)

    coordinate_jacobian = jax.jacrev(robot.actuator_coordinates)(q)
    matrix_jacobian = jax.jacrev(robot.actuation_matrix)(q)

    assert jnp.isfinite(coordinate_jacobian).all()
    assert jnp.isfinite(matrix_jacobian).all()


def test_custom_nonlinear_routing_uses_the_common_host_contract():
    params = SinusoidalRoutingParams(
        amplitude=jnp.array([0.01]),
        frequency=jnp.array([8.0]),
        start_segment_index=(0,),
        end_segment_index=(0,),
    )
    routing = ThreadlikeRouting(
        params=params,
        offset_fn=_sinusoidal_offset,
        derivative_fn=_sinusoidal_derivative,
    )
    robot = _spatial_pcs(actuators=ThreadlikeActuator.tendons(routing))
    q = jnp.linspace(-0.01, 0.02, robot.num_dofs)
    assert_allclose(
        jax.jacrev(robot.actuator_coordinates)(q),
        robot.actuation_matrix(q).T,
        rtol=2e-7,
        atol=2e-9,
    )


def test_modality_signs_pressure_work_pair_and_power():
    routing = _routing(count=1)
    tendon_robot = _spatial_pcs(actuators=ThreadlikeActuator.tendons(routing))
    rod_robot = _spatial_pcs(actuators=ThreadlikeActuator.push_rods(routing))
    muscle_robot = _spatial_pcs(actuators=ThreadlikeActuator.muscles(routing))
    pressure_robot = _spatial_pcs(
        actuators=ThreadlikeActuator.pressure_chambers(
            routing, effective_areas=jnp.array([2e-4])
        )
    )
    q = jnp.linspace(-0.01, 0.02, tendon_robot.num_dofs)
    qd = jnp.linspace(0.03, -0.02, tendon_robot.num_dofs)
    length = tendon_robot.actuators[0].path_lengths(tendon_robot, q)

    assert_allclose(tendon_robot.actuator_coordinates(q), -length)
    assert_allclose(rod_robot.actuator_coordinates(q), length)
    assert_allclose(muscle_robot.actuator_coordinates(q), -length)
    assert_allclose(pressure_robot.actuator_coordinates(q), 2e-4 * length)
    assert tendon_robot.actuator_input_metadata[0].units == ("N",)
    assert pressure_robot.actuator_input_metadata[0].units == ("Pa",)
    assert_allclose(
        tendon_robot.actuation_force(q, jnp.array([1.0])),
        -rod_robot.actuation_force(q, jnp.array([1.0])),
    )

    effort = jnp.array([3.0])
    generalized_power = qd @ tendon_robot.actuation_force(q, effort, qd=qd)
    actuator_power = effort @ tendon_robot.actuator_velocities(q, qd)
    assert_allclose(generalized_power, actuator_power, rtol=1e-9, atol=1e-11)


def test_tendon_reference_fixture_preserves_generalized_force():
    """Lock the reference generalized force for positive tendon tensions."""
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([[0.0, 0.02, 0.0], [0.0, -0.01, 0.015]]),
        start_segment_index=(0, 0),
        end_segment_index=(1, 0),
    )
    body = pcs_params(
        base_pose=spatial_base_pose(),
        length=jnp.array([0.1, 0.1]),
        radius=jnp.array([0.02, 0.02]),
        density=jnp.array([1000.0, 1000.0]),
        gravity=jnp.zeros((3,)),
        young_modulus=jnp.array([1e3, 1e3]),
        shear_modulus=jnp.array([5e2, 5e2]),
        damping_matrix=1e-3 * jnp.eye(12),
    )
    robot = PCS(
        body,
        PCSStructure(num_gauss_points=5),
        actuators=ThreadlikeActuator.tendons(routing),
    )
    q = jnp.zeros(robot.num_dofs)

    legacy_raw_matrix = jnp.zeros((12, 2))
    legacy_raw_matrix = legacy_raw_matrix.at[2, 0].set(-0.002)
    legacy_raw_matrix = legacy_raw_matrix.at[3, 0].set(0.1)
    legacy_raw_matrix = legacy_raw_matrix.at[1, 1].set(0.0015)
    legacy_raw_matrix = legacy_raw_matrix.at[2, 1].set(0.001)
    legacy_raw_matrix = legacy_raw_matrix.at[3, 1].set(0.1)
    legacy_raw_matrix = legacy_raw_matrix.at[8, 0].set(-0.002)
    legacy_raw_matrix = legacy_raw_matrix.at[9, 0].set(0.1)

    assert_allclose(robot.actuators[0].path_lengths(robot, q), jnp.array([0.2, 0.1]))
    assert_allclose(robot.actuation_matrix(q), -legacy_raw_matrix, atol=1e-12)
    legacy_negative_input = jnp.array([-0.7, -0.2])
    new_positive_tension = -legacy_negative_input
    assert_allclose(
        robot.actuation_force(q, new_positive_tension),
        legacy_raw_matrix @ legacy_negative_input,
        atol=1e-12,
    )


def test_pcs_and_gvs_threadlike_transmissions_agree_for_constant_strain_basis():
    routing = _routing(count=2)
    actuator = ThreadlikeActuator.tendons(routing)
    body = _spatial_pcs().params
    pcs = PCS(
        body,
        PCSStructure(
            num_gauss_points=5,
            strain_selector=jnp.array([True, True, True, True, False, False]),
        ),
        actuators=actuator,
    )
    gvs = _gvs(actuators=actuator)
    q = jnp.array([0.01, -0.02, 0.03, 0.015])

    assert_allclose(
        pcs.actuator_coordinates(q),
        gvs.actuator_coordinates(q),
        rtol=5e-8,
        atol=5e-10,
    )
    assert_allclose(
        pcs.actuation_matrix(q),
        gvs.actuation_matrix(q),
        rtol=5e-8,
        atol=5e-10,
    )


def test_passive_impedance_superposition_and_parameter_updates():
    routing = _routing(count=1)
    base = _spatial_pcs(actuators=())
    rest_length = base._threadlike_path_lengths(jnp.zeros(base.num_dofs), routing)
    impedance = ThreadlikeImpedance(
        routing=routing,
        stiffness=jnp.array([50.0]),
        damping=jnp.array([2.0]),
        rest_length=rest_length - 0.01,
    )
    robot = _spatial_pcs(actuators=(), passive_elements=(impedance,))
    q = jnp.linspace(-0.01, 0.02, robot.num_dofs)
    qd = jnp.linspace(0.02, -0.01, robot.num_dofs)

    path_jacobian = jax.jacrev(lambda q_: impedance.path_lengths(robot, q_))(q)
    assert_allclose(impedance.path_velocities(robot, q, qd), path_jacobian @ qd)
    assert impedance.path_poses(robot, q, robot.length).shape == (1, 3)

    energy_gradient = jax.grad(robot.passive_elastic_energy)(q)
    assert_allclose(
        energy_gradient, robot.passive_elastic_force(q), rtol=1e-7, atol=1e-9
    )
    assert_allclose(
        robot.elastic_force(q),
        robot.K_active @ q + robot.passive_elastic_force(q),
    )

    updated = robot.update_passive_element_params(0, stiffness=jnp.array([75.0]))
    assert_allclose(updated.passive_elements[0].params.stiffness, jnp.array([75.0]))
    assert_allclose(robot.passive_elements[0].params.stiffness, jnp.array([50.0]))


def test_gvs_forward_dynamics_includes_passive_threadlike_damping():
    routing = _routing(count=1)
    base = _gvs(actuators=())
    q = jnp.zeros((base.num_dofs,))
    qd = jnp.array([0.4, -0.2, 0.3, -0.1])
    rest_length = base._threadlike_path_lengths(q, routing)
    impedance = ThreadlikeImpedance(
        routing=routing,
        stiffness=jnp.zeros((1,)),
        damping=jnp.array([2.0]),
        rest_length=rest_length,
    )
    robot = _gvs(actuators=(), passive_elements=(impedance,))
    y = jnp.concatenate((q, qd))

    actual_qdd = robot.forward_dynamics(jnp.array(0.0), y)[robot.num_dofs :]
    inertia, coriolis_force, gravity_force = robot.dynamics_terms(q, qd)
    expected_qdd = jnp.linalg.solve(
        inertia,
        -coriolis_force
        - gravity_force
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ qd,
    )

    assert jnp.linalg.norm(robot.passive_damping_matrix(q)) > 0.0
    assert_allclose(actual_qdd, expected_qdd, rtol=1e-9, atol=1e-9)


def test_actuator_nested_update_and_topology_rejection():
    actuator = ThreadlikeActuator.tendons(_routing(count=1))
    robot = _spatial_pcs(actuators=actuator)
    params = robot.actuators[0].params
    routing = params.transmission.routing.replace(
        intercept=jnp.array([[0.0, 0.02, 0.0]])
    )
    transmission = params.transmission.replace(routing=routing)
    updated = robot.update_actuator_params(0, transmission=transmission)
    assert_allclose(
        updated.actuators[0].params.transmission.routing.intercept,
        jnp.array([[0.0, 0.02, 0.0]]),
    )
    assert_allclose(
        robot.actuators[0].params.transmission.routing.intercept,
        jnp.array([[0.0, -0.01, 0.0]]),
    )

    with pytest.raises(ValueError, match="topology"):
        params.transmission.routing.replace(end_segment_index=(1,))

    body_updated = robot.update_link_params(stiffness=2.0 * robot.params.link.stiffness)
    assert_allclose(
        body_updated.params.link.stiffness, 2.0 * robot.params.link.stiffness
    )
    assert_allclose(
        body_updated.actuators[0].params.transmission.routing.intercept,
        robot.actuators[0].params.transmission.routing.intercept,
    )


def test_visual_layers_are_semantic_unstyled_and_carry_inputs():
    actuator = ThreadlikeActuator.muscles(_routing(count=2))
    robot = _spatial_pcs(actuators=actuator)
    layers = robot.actuator_visual_layers(
        jnp.zeros(robot.num_dofs),
        jnp.linspace(0.0, robot.length, 8),
        actuator_inputs=jnp.array([2.0, 3.0]),
    )
    assert len(layers) == 1
    assert layers[0].kind == "muscle"
    assert layers[0].points.shape == (2, 8, 3)
    assert layers[0].paths.shape == (2, 8, 3)
    assert layers[0].radius is None
    assert layers[0].radii is None
    assert layers[0].colors is None
    assert not hasattr(actuator.params, "radius")
    assert not hasattr(actuator.params, "colors")
    assert_allclose(layers[0].scalar_fields["input"], jnp.array([2.0, 3.0]))
    assert_allclose(layers[0].scalar_values["input"], jnp.array([2.0, 3.0]))


def test_planar_host_rejects_out_of_plane_routing():
    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([0.0, 0.01, 0.005]),
        end_segment_index=0,
    )
    with pytest.raises(ValueError, match="local-y"):
        _planar_pcs(actuators=ThreadlikeActuator.tendons(routing))


def test_routing_spans_are_contiguous_inclusive_and_validated_by_the_host():
    with pytest.raises(ValueError, match="start_segment_index"):
        ThreadlikeRouting.linear(
            intercept=jnp.array([0.0, 0.01, 0.0]),
            start_segment_index=1,
            end_segment_index=0,
        )

    routing = ThreadlikeRouting.linear(
        intercept=jnp.array([0.0, 0.01, 0.0]),
        start_segment_index=0,
        end_segment_index=1,
    )
    with pytest.raises(ValueError, match="num_segments"):
        _spatial_pcs(actuators=ThreadlikeActuator.tendons(routing))
