import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    linear_tendon_routing,
    passive_tendon_params,
    pcs_params,
    pendulum_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
    tendon_actuated_pcs_params,
    tendon_actuated_planar_pcs_params,
)

from soromox.actuation.tendon_actuation import (
    linear_routing,
    linear_routing_arc_length_derivative,
)
from soromox.systems import (
    PCS,
    ArticulatedSoftRobotParams,
    BaseTendonRoutingParams,
    PCSParams,
    Pendulum,
    PendulumParams,
    TendonActuatedPCS,
    TendonActuatedPlanarPCS,
)

jax.config.update("jax_enable_x64", True)


def _pendulum_params():
    return pendulum_params(
        mass=jnp.array([1.0, 1.2], dtype=jnp.float64),
        moment_inertia=jnp.array([0.1, 0.12], dtype=jnp.float64),
        length=jnp.array([0.5, 0.4], dtype=jnp.float64),
        center_of_mass_length=jnp.array([0.25, 0.2], dtype=jnp.float64),
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
        joint_stiffness=jnp.eye(2, dtype=jnp.float64),
        joint_damping=0.1 * jnp.eye(2, dtype=jnp.float64),
    )


def _pcs_params(num_segments: int = 2):
    segment_lengths = 0.1 * jnp.ones((num_segments,), dtype=jnp.float64)
    return pcs_params(
        length=segment_lengths,
        radius=0.02 * jnp.ones((num_segments,), dtype=jnp.float64),
        density=1000.0 * jnp.ones((num_segments,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((num_segments,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((num_segments,), dtype=jnp.float64),
        gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        damping_matrix=jnp.eye(6 * num_segments, dtype=jnp.float64),
    )


def _planar_pcs_params(num_segments: int = 2):
    segment_lengths = 0.1 * jnp.ones((num_segments,), dtype=jnp.float64)
    return planar_pcs_params(
        length=segment_lengths,
        radius=0.02 * jnp.ones((num_segments,), dtype=jnp.float64),
        density=1000.0 * jnp.ones((num_segments,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((num_segments,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((num_segments,), dtype=jnp.float64),
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
        damping_matrix=jnp.eye(3 * num_segments, dtype=jnp.float64),
        base_pose=planar_base_pose(),
    )


class QuadraticTendonRoutingParams(BaseTendonRoutingParams):
    y_offset: jax.Array
    y_quadratic: jax.Array
    z_offset: jax.Array
    z_quadratic: jax.Array
    attachment_segment_index: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_tendons(self) -> int:
        return int(self.y_offset.shape[0])

    @property
    def attachment_segment_indices(self) -> tuple[int, ...]:
        return self.attachment_segment_index

    def validate(self) -> None:
        expected_shape = (self.num_tendons,)
        for name in ("y_quadratic", "z_offset", "z_quadratic"):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}.")
        if len(self.attachment_segment_index) != self.num_tendons:
            raise ValueError("attachment_segment_index must match num_tendons.")


def quadratic_routing(params: QuadraticTendonRoutingParams, s):
    y = params.y_offset + params.y_quadratic * s**2
    z = params.z_offset + params.z_quadratic * s**2
    return jnp.stack([jnp.zeros_like(y), y, z], axis=-1)


def quadratic_routing_derivative(params: QuadraticTendonRoutingParams, s):
    y = 2.0 * params.y_quadratic * s
    z = 2.0 * params.z_quadratic * s
    return jnp.stack([jnp.zeros_like(y), y, z], axis=-1)


def test_params_are_pytrees_and_replace_is_immutable():
    params = _pcs_params()
    leaves = jax.tree.leaves(params)

    assert any(leaf.shape == (2,) for leaf in leaves)

    updated = params.replace(length=2.0 * params.length)
    assert_allclose(params.length, jnp.array([0.1, 0.1]))
    assert_allclose(updated.length, jnp.array([0.2, 0.2]))

    with pytest.raises(KeyError, match="Unknown parameter field"):
        params.replace(not_a_field=jnp.array([1.0]))
    with pytest.raises(KeyError, match="segment_lengths"):
        params.replace(segment_lengths=jnp.array([0.1, 0.1]))


def test_system_update_rejects_static_shape_changes():
    robot = PCS(params=_pcs_params(num_segments=2))

    updated = robot.update_params(length=jnp.array([0.12, 0.13]))
    assert_allclose(updated.segment_length, jnp.array([0.12, 0.13]))
    assert_allclose(robot.segment_length, jnp.array([0.1, 0.1]))

    with pytest.raises(ValueError, match="length"):
        robot.update_params(length=jnp.array([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match="radius"):
        robot.update_params(radius=jnp.array([0.03]))

    with pytest.raises(KeyError, match="Unknown parameter field"):
        robot.update_params(unknown=jnp.array([0.0]))


def test_planar_pcs_params_validate_base_pose_shape():
    params = planar_pcs_params(
        length=jnp.array([0.1], dtype=jnp.float64),
        radius=jnp.array([0.02], dtype=jnp.float64),
        density=jnp.array([1000.0], dtype=jnp.float64),
        young_modulus=jnp.array([1e6], dtype=jnp.float64),
        shear_modulus=jnp.array([1e5], dtype=jnp.float64),
        damping_matrix=jnp.eye(3, dtype=jnp.float64),
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
        base_pose=jnp.array([0.0], dtype=jnp.float64),
    )

    with pytest.raises(ValueError, match="base_pose"):
        params.validate()


def test_spatial_params_validate_base_pose_quaternion_norm():
    with pytest.raises(ValueError, match="quaternion"):
        _pcs_params().replace(
            base_pose=jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=jnp.float64)
        )


def test_planar_params_validate_base_pose_finite_values():
    planar = planar_pcs_params(
        length=jnp.array([0.1], dtype=jnp.float64),
        radius=jnp.array([0.02], dtype=jnp.float64),
        density=jnp.array([1000.0], dtype=jnp.float64),
        young_modulus=jnp.array([1e6], dtype=jnp.float64),
        shear_modulus=jnp.array([1e5], dtype=jnp.float64),
        damping_matrix=jnp.eye(3, dtype=jnp.float64),
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
        base_pose=jnp.array([jnp.nan, 1.0, 2.0], dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="finite"):
        planar.validate()


def test_tendon_attachment_indices_are_static_topology():
    body = _pcs_params(num_segments=2)
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([1], dtype=jnp.int32),
    )
    robot = TendonActuatedPCS(
        params=tendon_actuated_pcs_params(body=body, active_tendon_routing=routing)
    )

    assert routing.attachment_segment_index == (1,)
    assert routing.routing_for_tendon(0).attachment_segment_index == 1
    assert not any(
        getattr(leaf, "dtype", None) == jnp.dtype(jnp.int32)
        for leaf in jax.tree.leaves(routing)
    )

    with pytest.raises(ValueError, match="topology"):
        routing.replace(attachment_segment_index=jnp.array([0], dtype=jnp.int32))

    changed_attachment = linear_tendon_routing(
        y_intercept=routing.y_intercept,
        y_slope=routing.y_slope,
        z_intercept=routing.z_intercept,
        z_slope=routing.z_slope,
        attachment_segment_index=jnp.array([0], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="topology"):
        robot.with_params(
            robot.params.replace(active_tendon_routing=changed_attachment)
        )


def test_tendon_system_accepts_base_routing_subclasses():
    body = _pcs_params(num_segments=1)
    routing = QuadraticTendonRoutingParams(
        y_offset=jnp.array([0.005], dtype=jnp.float64),
        y_quadratic=jnp.array([0.01], dtype=jnp.float64),
        z_offset=jnp.array([0.0], dtype=jnp.float64),
        z_quadratic=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=(0,),
    )
    robot = TendonActuatedPCS(
        params=tendon_actuated_pcs_params(body=body, active_tendon_routing=routing),
        active_tendon_routing_basis={
            "d_s": quadratic_routing,
            "dd_s_ds": quadratic_routing_derivative,
        },
    )

    assert robot.num_actuators == 1
    assert robot.active_tendon_routing is routing
    assert robot.actuation_matrix(jnp.zeros((robot.num_dofs,))).shape == (
        robot.num_dofs,
        1,
    )


def test_planar_tendon_params_use_body_wrapper_and_reject_topology_changes():
    body = _planar_pcs_params(num_segments=2)
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([1], dtype=jnp.int32),
    )
    robot = TendonActuatedPlanarPCS(
        params=tendon_actuated_planar_pcs_params(
            body=body,
            active_tendon_routing=routing,
        )
    )

    updated_body = body.replace(radius=0.9 * body.radius)
    updated_robot = robot.with_params(robot.params.replace(body=updated_body))
    assert_allclose(updated_robot.r, 0.9 * robot.r)
    assert robot.params.body is body

    changed_attachment = linear_tendon_routing(
        y_intercept=routing.y_intercept,
        y_slope=routing.y_slope,
        z_intercept=routing.z_intercept,
        z_slope=routing.z_slope,
        attachment_segment_index=jnp.array([0], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="topology"):
        robot.with_params(
            robot.params.replace(active_tendon_routing=changed_attachment)
        )


def test_same_shape_param_updates_do_not_retrace_under_filter_jit():
    params = _pendulum_params()
    robot = Pendulum(params=params)
    q = jnp.array([0.2, -0.1], dtype=jnp.float64)
    trace_count = {"value": 0}

    @eqx.filter_jit
    def potential_energy(current_params, current_q):
        trace_count["value"] += 1
        return robot.with_params(current_params).potential_energy(current_q)

    potential_energy(params, q)
    potential_energy(params.replace(mass=params.mass + 0.1), q)
    potential_energy(params.replace(gravity=params.gravity.at[1].set(-9.7)), q)

    assert trace_count["value"] == 1


def test_grad_differentiates_through_typed_params():
    params = _pendulum_params()
    robot = Pendulum(params=params)
    q = jnp.array([0.25, -0.15], dtype=jnp.float64)

    def energy_for_first_mass(mass_0):
        current = params.replace(mass=params.mass.at[0].set(mass_0))
        return robot.with_params(current).potential_energy(q)

    grad_value = jax.grad(energy_for_first_mass)(params.mass[0])
    assert jnp.isfinite(grad_value)


def test_linear_tendon_routing_supports_distinct_batched_tendons():
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.01, -0.02], dtype=jnp.float64),
        y_slope=jnp.array([0.1, 0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.0, 0.03], dtype=jnp.float64),
        z_slope=jnp.array([0.0, -0.2], dtype=jnp.float64),
        attachment_segment_index=jnp.array([0, 1], dtype=jnp.int32),
    )

    assert routing.num_tendons == 2
    positions = linear_routing(routing, jnp.array(0.5, dtype=jnp.float64))
    derivatives = linear_routing_arc_length_derivative(
        routing, jnp.array(0.5, dtype=jnp.float64)
    )

    assert positions.shape == (2, 3)
    assert derivatives.shape == (2, 3)
    assert_allclose(positions[0], jnp.array([0.0, 0.06, 0.0]))
    assert_allclose(positions[1], jnp.array([0.0, -0.02, -0.07]))

    single = routing.routing_for_tendon(1)
    assert single.y_intercept.shape == ()
    assert_allclose(linear_routing(single, 0.5), positions[1])


def test_passive_tendon_params_store_per_tendon_impedance():
    params = passive_tendon_params(
        stiffness=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64),
        damping=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        rest_length_offset=jnp.array([0.0, -0.01, 0.02], dtype=jnp.float64),
    )

    assert params.num_tendons == 3
    updated = params.replace(damping=params.damping + 1.0)
    assert_allclose(params.damping, jnp.array([0.1, 0.2, 0.3]))
    assert_allclose(updated.damping, jnp.array([1.1, 1.2, 1.3]))

    with pytest.raises(ValueError, match="rest_length_offset"):
        params.replace(rest_length_offset=jnp.array([0.0, 0.0]))


def test_removed_plural_typed_param_names_fail():
    pcs_kwargs = {
        "radius": jnp.array([0.01], dtype=jnp.float64),
        "density": jnp.array([1000.0], dtype=jnp.float64),
        "young_modulus": jnp.array([1e6], dtype=jnp.float64),
        "shear_modulus": jnp.array([1e5], dtype=jnp.float64),
        "damping_matrix": jnp.eye(6, dtype=jnp.float64),
        "gravity": jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        "base_pose": spatial_base_pose(),
        "reference_strain": jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    }
    with pytest.raises(TypeError, match="segment_lengths"):
        PCSParams(
            segment_lengths=jnp.array([0.1], dtype=jnp.float64),
            **pcs_kwargs,
        )

    pendulum_kwargs = {
        "base_pose": planar_base_pose(),
        "mass": jnp.array([1.0], dtype=jnp.float64),
        "moment_inertia": jnp.array([0.1], dtype=jnp.float64),
        "gravity": jnp.array([0.0, -9.81], dtype=jnp.float64),
        "joint_stiffness": jnp.zeros((1, 1), dtype=jnp.float64),
        "joint_damping": jnp.zeros((1, 1), dtype=jnp.float64),
        "radius": jnp.array([0.02], dtype=jnp.float64),
    }
    with pytest.raises(TypeError, match="link_lengths"):
        PendulumParams(
            link_lengths=jnp.array([0.5], dtype=jnp.float64),
            center_of_mass_length=jnp.array([0.25], dtype=jnp.float64),
            joint_rest_configuration=jnp.zeros((1,), dtype=jnp.float64),
            **pendulum_kwargs,
        )
    with pytest.raises(TypeError, match="center_of_mass_lengths"):
        PendulumParams(
            length=jnp.array([0.5], dtype=jnp.float64),
            center_of_mass_lengths=jnp.array([0.25], dtype=jnp.float64),
            joint_rest_configuration=jnp.zeros((1,), dtype=jnp.float64),
            **pendulum_kwargs,
        )
    with pytest.raises(TypeError, match="joint_stiffness_reference"):
        PendulumParams(
            length=jnp.array([0.5], dtype=jnp.float64),
            center_of_mass_length=jnp.array([0.25], dtype=jnp.float64),
            joint_stiffness_reference=jnp.zeros((1,), dtype=jnp.float64),
            **pendulum_kwargs,
        )

    articulated_kwargs = {
        "base_pose": spatial_base_pose(),
        "parent_to_joint_transform": jnp.eye(4, dtype=jnp.float64)[None, :, :],
        "tip_position": jnp.array([[0.5, 0.0, 0.0]], dtype=jnp.float64),
        "center_of_mass_position": jnp.array([[0.25, 0.0, 0.0]], dtype=jnp.float64),
        "mass": jnp.array([1.0], dtype=jnp.float64),
        "center_of_mass_inertia": jnp.eye(3, dtype=jnp.float64)[None, :, :],
        "gravity": jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        "joint_stiffness": jnp.zeros((1, 1), dtype=jnp.float64),
        "joint_damping": jnp.zeros((1, 1), dtype=jnp.float64),
        "joint_rest_configuration": jnp.zeros((1,), dtype=jnp.float64),
        "radius": jnp.array([0.02], dtype=jnp.float64),
    }
    with pytest.raises(TypeError, match="joint_screws"):
        ArticulatedSoftRobotParams(
            joint_screws=jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            **articulated_kwargs,
        )

    pendulum_params_new = PendulumParams(
        length=jnp.array([0.5], dtype=jnp.float64),
        center_of_mass_length=jnp.array([0.25], dtype=jnp.float64),
        joint_rest_configuration=jnp.zeros((1,), dtype=jnp.float64),
        **pendulum_kwargs,
    )
    with pytest.raises(KeyError, match="link_lengths"):
        pendulum_params_new.replace(link_lengths=jnp.array([0.5]))
    with pytest.raises(KeyError, match="joint_stiffness_reference"):
        pendulum_params_new.replace(joint_stiffness_reference=jnp.zeros((1,)))

    articulated_params_new = ArticulatedSoftRobotParams(
        joint_screw=jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        **articulated_kwargs,
    )
    with pytest.raises(KeyError, match="joint_screws"):
        articulated_params_new.replace(
            joint_screws=jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        )
