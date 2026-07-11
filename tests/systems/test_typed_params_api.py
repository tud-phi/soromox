from dataclasses import fields

import equinox as eqx
import jax
import pytest
from jax import Array, vmap
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    articulated_params,
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
    ISupport,
    ISupportParams,
    ISupportStructure,
    PCSParams,
    Pendulum,
    PendulumParams,
    PlanarPCS,
    PlanarPCSParams,
    TendonActuatedPCS,
    TendonActuatedPlanarPCS,
)
from soromox.utils.array_math import blk_diag
from soromox.utils.geometry import poses

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


def _expected_spatial_material_damping(
    length: Array, radius: Array, coefficient: Array
) -> Array:
    length = jnp.asarray(length, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)
    coefficient = jnp.asarray(coefficient, dtype=jnp.float64)
    if coefficient.ndim == 0:
        coefficient = jnp.full_like(length, coefficient)
    area = jnp.pi * radius**2
    second_moment = jnp.stack(
        [jnp.pi * radius**4 / 2.0, jnp.pi * radius**4 / 4.0, jnp.pi * radius**4 / 4.0],
        axis=1,
    )
    damping_diag = jnp.stack(
        [
            second_moment[:, 0],
            3.0 * second_moment[:, 1],
            3.0 * second_moment[:, 2],
            3.0 * area,
            area,
            area,
        ],
        axis=1,
    )
    return blk_diag(
        vmap(jnp.diag)(length[:, None] * coefficient[:, None] * damping_diag)
    )


def _expected_planar_material_damping(
    length: Array, radius: Array, coefficient: Array
) -> Array:
    length = jnp.asarray(length, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)
    coefficient = jnp.asarray(coefficient, dtype=jnp.float64)
    if coefficient.ndim == 0:
        coefficient = jnp.full_like(length, coefficient)
    area = jnp.pi * radius**2
    second_moment = jnp.pi * radius**4 / 4.0
    damping_diag = jnp.stack([3.0 * second_moment, 3.0 * area, area], axis=1)
    return blk_diag(
        vmap(jnp.diag)(length[:, None] * coefficient[:, None] * damping_diag)
    )


def _constructor_kwargs_without_environment(params):
    return {
        field.name: getattr(params, field.name)
        for field in fields(params)
        if field.name not in {"base_pose", "gravity"}
    }


def _articulated_params():
    return articulated_params(
        joint_screw=jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        tip_position=jnp.array([[0.5, 0.0, 0.0]]),
        center_of_mass_position=jnp.array([[0.25, 0.0, 0.0]]),
        mass=jnp.array([1.0]),
        center_of_mass_inertia=jnp.eye(3)[None, :, :],
        gravity=jnp.zeros(3),
    )


@pytest.mark.parametrize(
    ("params_type", "source", "expected_pose", "expected_gravity"),
    [
        (
            PlanarPCSParams,
            _planar_pcs_params,
            jnp.array([jnp.pi / 2, 0.0, 0.0]),
            jnp.array([0.0, -9.81]),
        ),
        (
            PendulumParams,
            _pendulum_params,
            jnp.array([jnp.pi / 2, 0.0, 0.0]),
            jnp.array([0.0, -9.81]),
        ),
        (
            PCSParams,
            _pcs_params,
            jnp.array([jnp.sqrt(0.5), 0.0, -jnp.sqrt(0.5), 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, -9.81]),
        ),
        (
            ArticulatedSoftRobotParams,
            _articulated_params,
            jnp.array([jnp.sqrt(0.5), 0.0, -jnp.sqrt(0.5), 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, -9.81]),
        ),
    ],
)
def test_soft_robot_params_materialize_upright_environment_defaults(
    params_type, source, expected_pose, expected_gravity
):
    params = params_type(**_constructor_kwargs_without_environment(source()))

    assert_allclose(params.base_pose, expected_pose)
    assert_allclose(params.gravity, expected_gravity)


@pytest.mark.parametrize(
    ("mounting", "expected_direction"),
    [
        ("horizontal", jnp.array([1.0, 0.0])),
        ("upright", jnp.array([0.0, 1.0])),
        ("hanging", jnp.array([0.0, -1.0])),
    ],
)
def test_planar_mounting_constructors_map_local_x_to_world_direction(
    mounting, expected_direction
):
    params = getattr(PlanarPCSParams, mounting)(
        **_constructor_kwargs_without_environment(_planar_pcs_params()),
        base_position=jnp.array([1.0, 2.0]),
    )
    transform = poses.planar_pose_to_transform(params.base_pose)

    assert_allclose(
        transform[:2, :2] @ jnp.array([1.0, 0.0]),
        expected_direction,
        atol=1e-12,
    )
    assert_allclose(params.base_pose[1:], jnp.array([1.0, 2.0]))
    assert_allclose(params.gravity, jnp.array([0.0, -9.81]))


@pytest.mark.parametrize(
    ("mounting", "expected_direction"),
    [
        ("horizontal", jnp.array([1.0, 0.0, 0.0])),
        ("upright", jnp.array([0.0, 0.0, 1.0])),
        ("hanging", jnp.array([0.0, 0.0, -1.0])),
    ],
)
def test_spatial_mounting_constructors_map_local_x_to_world_direction(
    mounting, expected_direction
):
    params = getattr(PCSParams, mounting)(
        **_constructor_kwargs_without_environment(_pcs_params()),
        base_position=jnp.array([1.0, 2.0, 3.0]),
    )
    transform = poses.quaternion_pose_to_transform(params.base_pose)

    assert_allclose(
        transform[:3, :3] @ jnp.array([1.0, 0.0, 0.0]),
        expected_direction,
        atol=1e-12,
    )
    assert_allclose(params.base_pose[4:], jnp.array([1.0, 2.0, 3.0]))
    assert_allclose(params.gravity, jnp.array([0.0, 0.0, -9.81]))


def test_environment_defaults_can_be_overridden_and_restored():
    custom_pose = jnp.array([0.25, 1.0, 2.0])
    custom_gravity = jnp.array([1.0, -2.0])
    params = PlanarPCSParams(
        **_constructor_kwargs_without_environment(_planar_pcs_params()),
        base_pose=custom_pose,
        gravity=custom_gravity,
    )
    assert_allclose(params.base_pose, custom_pose)
    assert_allclose(params.gravity, custom_gravity)

    restored = params.replace(base_pose=None, gravity=None)
    assert_allclose(restored.base_pose, jnp.array([jnp.pi / 2, 0.0, 0.0]))
    assert_allclose(restored.gravity, jnp.array([0.0, -9.81]))

    robot = Pendulum(params=_pendulum_params())
    updated_robot = robot.update_params(base_pose=None, gravity=None)
    assert_allclose(updated_robot.params.base_pose, jnp.array([jnp.pi / 2, 0.0, 0.0]))
    assert_allclose(updated_robot.params.gravity, jnp.array([0.0, -9.81]))


def test_named_mounting_rejects_competing_base_pose():
    with pytest.raises(TypeError, match="does not accept base_pose"):
        PlanarPCSParams.upright(
            **_constructor_kwargs_without_environment(_planar_pcs_params()),
            base_pose=jnp.zeros(3),
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


def test_pcs_material_damping_coefficient_builds_full_matrix():
    length = jnp.array([0.1, 0.2], dtype=jnp.float64)
    radius = jnp.array([0.02, 0.03], dtype=jnp.float64)
    coefficient = jnp.array([2.0, 3.0], dtype=jnp.float64)
    params = PCSParams(
        base_pose=spatial_base_pose(),
        length=length,
        radius=radius,
        density=1000.0 * jnp.ones((2,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((2,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((2,), dtype=jnp.float64),
        material_damping_coefficient=coefficient,
        gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64), 2
        ),
    )
    robot = PCS(params=params)
    expected = _expected_spatial_material_damping(length, radius, coefficient)

    assert_allclose(robot.D_full, expected)
    assert_allclose(robot.damping_matrix(jnp.zeros((robot.num_dofs,))), expected)


def test_planar_pcs_material_damping_coefficient_builds_full_matrix():
    length = jnp.array([0.1, 0.2], dtype=jnp.float64)
    radius = jnp.array([0.02, 0.03], dtype=jnp.float64)
    coefficient = jnp.array(2.0, dtype=jnp.float64)
    params = PlanarPCSParams(
        base_pose=planar_base_pose(),
        length=length,
        radius=radius,
        density=1000.0 * jnp.ones((2,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((2,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((2,), dtype=jnp.float64),
        material_damping_coefficient=coefficient,
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
        reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64), 2),
    )
    robot = PlanarPCS(params=params)
    expected = _expected_planar_material_damping(length, radius, coefficient)

    assert_allclose(robot.D_full, expected)
    assert_allclose(robot.damping_matrix(jnp.zeros((robot.num_dofs,))), expected)


def test_pcs_damping_input_validation():
    kwargs = {
        "base_pose": spatial_base_pose(),
        "length": jnp.array([0.1], dtype=jnp.float64),
        "radius": jnp.array([0.02], dtype=jnp.float64),
        "density": jnp.array([1000.0], dtype=jnp.float64),
        "young_modulus": jnp.array([1e6], dtype=jnp.float64),
        "shear_modulus": jnp.array([1e5], dtype=jnp.float64),
        "gravity": jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        "reference_strain": jnp.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64
        ),
    }

    with pytest.raises(ValueError, match="Exactly one"):
        PCSParams(**kwargs).validate()
    with pytest.raises(ValueError, match="Exactly one"):
        PCSParams(
            **kwargs,
            damping_matrix=jnp.eye(6, dtype=jnp.float64),
            material_damping_coefficient=jnp.array([1.0], dtype=jnp.float64),
        ).validate()
    with pytest.raises(ValueError, match="material_damping_coefficient"):
        PCSParams(
            **kwargs,
            material_damping_coefficient=jnp.array([1.0, 2.0], dtype=jnp.float64),
        ).validate()
    with pytest.raises(ValueError, match="damping_matrix"):
        PCSParams(
            **kwargs,
            damping_matrix=jnp.eye(5, dtype=jnp.float64),
        ).validate()


def test_material_damping_updates_and_matrix_switching():
    params = _pcs_params(num_segments=1).replace(
        damping_matrix=None,
        material_damping_coefficient=jnp.array([1.0], dtype=jnp.float64),
    )
    robot = PCS(params=params)
    expected = _expected_spatial_material_damping(
        params.length, params.radius, params.material_damping_coefficient
    )
    assert_allclose(robot.D_full, expected)

    updated = robot.update_params(
        material_damping_coefficient=jnp.array([2.0], dtype=jnp.float64)
    )
    assert_allclose(updated.D_full, 2.0 * expected)

    damping_matrix = 0.5 * jnp.eye(6, dtype=jnp.float64)
    matrix_updated = updated.update_params(
        material_damping_coefficient=None,
        damping_matrix=damping_matrix,
    )
    assert_allclose(matrix_updated.D_full, damping_matrix)

    coefficient_updated = matrix_updated.update_params(
        damping_matrix=None,
        material_damping_coefficient=jnp.array([3.0], dtype=jnp.float64),
    )
    assert_allclose(coefficient_updated.D_full, 3.0 * expected)


def test_existing_damping_matrix_path_is_unchanged():
    params = _pcs_params(num_segments=2)
    robot = PCS(params=params)

    assert_allclose(robot.D_full, params.damping_matrix)


def test_tendon_pcs_inherits_material_damping_path():
    body = _pcs_params(num_segments=1).replace(
        damping_matrix=None,
        material_damping_coefficient=jnp.array([2.0], dtype=jnp.float64),
    )
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([0], dtype=jnp.int32),
    )
    robot = TendonActuatedPCS(
        params=tendon_actuated_pcs_params(body=body, active_tendon_routing=routing)
    )

    expected = _expected_spatial_material_damping(
        body.length, body.radius, body.material_damping_coefficient
    )
    assert_allclose(robot.D_full, expected)


def test_planar_tendon_pcs_inherits_material_damping_path():
    body = _planar_pcs_params(num_segments=1).replace(
        damping_matrix=None,
        material_damping_coefficient=jnp.array([2.0], dtype=jnp.float64),
    )
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([0], dtype=jnp.int32),
    )
    robot = TendonActuatedPlanarPCS(
        params=tendon_actuated_planar_pcs_params(
            body=body,
            active_tendon_routing=routing,
        )
    )

    expected = _expected_planar_material_damping(
        body.length, body.radius, body.material_damping_coefficient
    )
    assert_allclose(robot.D_full, expected)


def test_isupport_inherits_material_damping_path():
    params = ISupportParams(
        base_pose=spatial_base_pose(),
        length=jnp.array([0.1], dtype=jnp.float64),
        radius=jnp.array([0.02], dtype=jnp.float64),
        density=jnp.array([1000.0], dtype=jnp.float64),
        gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        young_modulus=jnp.array([2e3], dtype=jnp.float64),
        shear_modulus=jnp.array([1e3], dtype=jnp.float64),
        material_damping_coefficient=jnp.array([2.0], dtype=jnp.float64),
        reference_strain=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64),
        chamber_inner_radius=jnp.array([0.002], dtype=jnp.float64),
        chamber_outer_radius=jnp.array([0.004], dtype=jnp.float64),
        chamber_distance=jnp.array([0.01], dtype=jnp.float64),
        chamber_azimuth_angles=(2.0 * jnp.pi * jnp.arange(3, dtype=jnp.float64) / 3.0)[
            None, :
        ],
    )
    robot = ISupport(params=params, structure=ISupportStructure(num_gauss_points=1))

    I_i = robot._local_second_moment_of_area(jnp.array(0))
    A_i = robot._local_cross_sectional_area(jnp.array(0))
    expected_diag = (
        params.length[0]
        * params.material_damping_coefficient[0]
        * jnp.array(
            [I_i[0], 3.0 * I_i[1], 3.0 * I_i[2], 3.0 * A_i, A_i, A_i],
            dtype=jnp.float64,
        )
    )
    assert_allclose(robot.D_full, jnp.diag(expected_diag))


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
