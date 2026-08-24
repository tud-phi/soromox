from dataclasses import fields

import equinox as eqx
import jax
import pytest
from jax import Array, vmap
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    articulated_params,
    pcs_params,
    pendulum_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)

from soromox.actuation import (
    ArticulatedTendonImpedance,
)
from soromox.systems import (
    PCS,
    ArticulatedSoftRobotParams,
    PCSParams,
    Pendulum,
    PendulumParams,
    PlanarPCSParams,
)
from soromox.systems.params import BaseSystemParams
from soromox.utils.array_math import blk_diag
from soromox.utils.geometry import poses

jax.config.update("jax_enable_x64", True)


class SplitValidationParams(BaseSystemParams):
    """Test parameter whose value validation cannot run on tracers."""

    value: Array

    def validate_structure(self) -> None:
        if self.value.shape != (1,):
            raise ValueError("value must have shape (1,).")

    def validate_values(self) -> None:
        if not bool(jnp.all(self.value > 0.0)):
            raise ValueError("value must be positive.")


class SplitCompatibilityValidationParams(SplitValidationParams):
    """Test parameter with structure-dependent concrete value checks."""

    def validate_structure_compatibility(
        self, structure: tuple[tuple[int, ...], float]
    ) -> None:
        expected_shape, _ = structure
        if self.value.shape != expected_shape:
            raise ValueError("value has an incompatible shape.")

    def validate_value_compatibility(
        self, structure: tuple[tuple[int, ...], float]
    ) -> None:
        _, minimum = structure
        if not bool(jnp.all(self.value >= minimum)):
            raise ValueError("value is below the compatible minimum.")


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


def test_validation_hooks_separate_traced_structure_from_eager_values():
    params = SplitValidationParams(value=jnp.ones((1,), dtype=jnp.float64))

    with pytest.raises(ValueError, match="positive"):
        params.replace(value=-jnp.ones((1,), dtype=jnp.float64))

    @jax.jit
    def traced_replace(value):
        return params.replace(value=value).value

    assert_allclose(traced_replace(-jnp.ones((1,), dtype=jnp.float64)), [-1.0])
    with pytest.raises(ValueError, match="shape"):
        traced_replace(jnp.ones((2,), dtype=jnp.float64))


def test_closed_over_concrete_parameter_validation_is_safe_inside_jit():
    params = SplitValidationParams(value=jnp.ones((1,), dtype=jnp.float64))

    @jax.jit
    def validated_sum(offset):
        params.validate_for_update()
        return jnp.sum(params.value) + offset

    assert_allclose(validated_sum(jnp.array(2.0)), 3.0)


def test_closed_over_concrete_compatibility_validation_is_safe_inside_jit():
    params = SplitCompatibilityValidationParams(value=jnp.ones((1,), dtype=jnp.float64))

    @jax.jit
    def validated_sum(offset):
        params.validate_for_update_against_structure(((1,), 0.0))
        return jnp.sum(params.value) + offset

    assert_allclose(validated_sum(jnp.array(2.0)), 3.0)


def test_closed_over_parameter_structure_validation_remains_active_inside_jit():
    params = SplitValidationParams(value=jnp.ones((1,), dtype=jnp.float64))
    invalid = eqx.tree_at(
        lambda current: current.value,
        params,
        jnp.ones((2,), dtype=jnp.float64),
    )

    @jax.jit
    def validated_sum(offset):
        invalid.validate_for_update()
        return jnp.sum(invalid.value) + offset

    with pytest.raises(ValueError, match="shape"):
        validated_sum(jnp.array(2.0))


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


def test_params_are_pytrees_and_replace_is_immutable():
    params = _pcs_params()
    leaves = jax.tree.leaves(params)

    assert any(leaf.shape == (2,) for leaf in leaves)

    updated = params.replace(link=params.link.replace(length=2.0 * params.link.length))
    assert_allclose(params.link.length, jnp.array([0.1, 0.1]))
    assert_allclose(updated.link.length, jnp.array([0.2, 0.2]))

    with pytest.raises(KeyError, match="Unknown parameter field"):
        params.replace(not_a_field=jnp.array([1.0]))
    with pytest.raises(KeyError, match="segment_lengths"):
        params.replace(segment_lengths=jnp.array([0.1, 0.1]))


def test_system_update_rejects_static_shape_changes():
    robot = PCS(params=_pcs_params(num_segments=2))

    updated = robot.update_link_params(length=jnp.array([0.12, 0.13]))
    assert_allclose(updated.segment_length, jnp.array([0.12, 0.13]))
    assert_allclose(robot.segment_length, jnp.array([0.1, 0.1]))

    with pytest.raises(ValueError, match="shape"):
        robot.update_link_params(length=jnp.array([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match="cross-section|coefficients"):
        robot.update_link_params(
            cross_section=robot.params.link.cross_section.replace(
                coefficients=jnp.array([[0.03]])
            )
        )

    with pytest.raises(KeyError, match="Unknown parameter field"):
        robot.update_params(unknown=jnp.array([0.0]))


def test_canonical_link_damping_updates_without_material_duplication():
    params = _pcs_params(num_segments=2)
    robot = PCS(params=params)
    expected = blk_diag(params.link.damping)
    assert_allclose(robot.D_full, expected)

    updated = robot.update_link_params(damping=2.0 * params.link.damping)
    assert_allclose(updated.D_full, 2.0 * expected)
    assert not hasattr(updated.params, "material_damping_coefficient")
    assert not hasattr(updated.params, "damping_matrix")


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


def test_articulated_tendon_impedance_stores_per_tendon_mechanics():
    impedance = ArticulatedTendonImpedance.from_routing(
        jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]]),
        stiffness=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64),
        damping=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        coordinate_offset=jnp.array([0.0, -0.01, 0.02], dtype=jnp.float64),
    )
    params = impedance.params

    updated = params.replace(damping=params.damping + 1.0)
    assert_allclose(params.damping, jnp.array([0.1, 0.2, 0.3]))
    assert_allclose(updated.damping, jnp.array([1.1, 1.2, 1.3]))

    with pytest.raises(ValueError, match="damping"):
        params.replace(damping=jnp.array([0.0, 0.0]))


def test_removed_plural_typed_param_names_fail():
    valid_pcs_params = _pcs_params(num_segments=1)
    with pytest.raises(TypeError, match="segment_lengths"):
        PCSParams(
            segment_lengths=jnp.array([0.1], dtype=jnp.float64),
            link=valid_pcs_params.link,
            gravity=valid_pcs_params.gravity,
            base_pose=valid_pcs_params.base_pose,
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
