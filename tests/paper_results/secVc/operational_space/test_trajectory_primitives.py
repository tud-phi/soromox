import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import pytest

from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    rotation_vector_to_rotation_matrix,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
    / "trajectory_primitives.py"
)

spec = importlib.util.spec_from_file_location(
    "example_operational_space_trajectory_primitives", MODULE_PATH
)
assert spec is not None
assert spec.loader is not None
trajectory_primitives = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trajectory_primitives
spec.loader.exec_module(trajectory_primitives)


@dataclass
class FakeOperationalSpaceDynamics:
    is_planar: bool = False

    def __post_init__(self):
        self.n_points = 1
        self.rotation_representation = RotationRepresentation.ROTATION_VECTOR
        if self.is_planar:
            self.n_velocity_dim = 3
            self.n_pose_dim = 3
            self.n_orientation_dim = 1
            self.n_angular_velocity_dim = 1
            self.x0_full = jnp.array([0.0, 0.0, 0.0])
        else:
            self.n_velocity_dim = 6
            self.n_pose_dim = 6
            self.n_orientation_dim = 3
            self.n_angular_velocity_dim = 3
            self.x0_full = jnp.zeros((6,))

    def operational_space_poses(self, q):
        return self.x0_full


def _make_reference(config):
    osd = FakeOperationalSpaceDynamics()
    return trajectory_primitives.make_operational_space_reference_trajectory(
        osd=osd,
        q0=jnp.zeros((1,)),
        ts=jnp.linspace(0.0, 4.0, 9),
        config=config,
    )


def test_default_config_uses_larger_ranges_and_four_second_period():
    config = trajectory_primitives.TaskSpaceTrajectoryConfig()

    assert config.period == 4.0
    assert config.ramp_time == 0.0
    assert config.position_amplitude_x == 0.08
    assert config.position_amplitude_y == 0.04
    assert config.position_amplitude_z == 0.0
    assert config.line_amplitude == 0.08
    assert config.circle_radius == 0.06
    assert config.helix_radius == 0.06
    assert config.helix_pitch == 0.06
    assert config.orientation_amplitude == 1.0


def test_make_end_effector_task_selector_supports_spatial_and_planar_modes():
    spatial_position = trajectory_primitives.make_end_effector_task_selector(
        "position", 6
    )
    spatial_pose = trajectory_primitives.make_end_effector_task_selector("pose", 6)
    planar_position = trajectory_primitives.make_end_effector_task_selector(
        "position", 3
    )
    planar_pose = trajectory_primitives.make_end_effector_task_selector("pose", 3)

    assert spatial_position.tolist() == [False, False, False, True, True, True]
    assert spatial_pose.tolist() == [True, True, True, True, True, True]
    assert planar_position.tolist() == [False, True, True]
    assert planar_pose.tolist() == [True, True, True]


@pytest.mark.parametrize(
    "primitive", ["point-to-point", "line", "planar-circle", "figure-eight"]
)
def test_spatial_position_primitives_are_defined_in_world_xy_plane(primitive):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="position",
        position_primitive=primitive,
        period=4.0,
    )
    reference = _make_reference(config)

    for t in (0.0, 0.5, 1.0, 2.0):
        pose = reference.x_des_fn(jnp.array(t))
        assert jnp.allclose(pose[5], 0.0)


def test_spatial_helix_advances_along_world_z_axis():
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="position",
        position_primitive="helix",
        period=4.0,
        ramp_time=0.0,
    )
    reference = _make_reference(config)

    pose = reference.x_des_fn(jnp.array(1.0))

    assert jnp.allclose(pose[3], -config.helix_radius)
    assert jnp.allclose(pose[4], config.helix_radius)
    assert jnp.allclose(pose[5], config.helix_pitch / 4.0)


@pytest.mark.parametrize("primitive", ["line", "planar-circle", "figure-eight"])
def test_periodic_spatial_primitives_repeat_by_default(primitive):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="position",
        position_primitive=primitive,
        period=4.0,
    )
    reference = _make_reference(config)

    for t in (0.25, 1.0, 2.75):
        pose = reference.x_des_fn(jnp.array(t))
        next_period_pose = reference.x_des_fn(jnp.array(t + config.period))
        assert jnp.allclose(next_period_pose[3:], pose[3:], atol=1e-12)


@pytest.mark.parametrize("primitive", trajectory_primitives.POSITION_PRIMITIVES)
def test_position_primitives_return_finite_full_spatial_pose_references(primitive):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="position",
        position_primitive=primitive,
        period=4.0,
    )
    reference = _make_reference(config)

    for t in (0.0, 1.0, 2.0):
        pose = reference.x_des_fn(jnp.array(t))
        assert pose.shape == (6,)
        assert jnp.all(jnp.isfinite(pose))


def test_fixed_orientation_preserves_initial_orientation():
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="line",
        orientation_primitive="fixed",
    )
    reference = _make_reference(config)

    initial_orientation = reference.x_des_fn(jnp.array(0.0))[:3]
    later_orientation = reference.x_des_fn(jnp.array(1.0))[:3]

    assert jnp.allclose(later_orientation, initial_orientation)


def test_sinusoid_orientation_changes_smoothly():
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="line",
        orientation_primitive="sinusoid",
        period=4.0,
    )
    reference = _make_reference(config)

    initial_orientation = reference.x_des_fn(jnp.array(0.0))[:3]
    quarter_period_orientation = reference.x_des_fn(jnp.array(1.0))[:3]

    assert jnp.all(jnp.isfinite(quarter_period_orientation))
    assert not jnp.allclose(quarter_period_orientation, initial_orientation)


@pytest.mark.parametrize("surface", trajectory_primitives.SURFACES)
def test_surface_following_aligns_local_x_with_surface_normal(surface):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="line",
        orientation_primitive="surface-following",
        surface=surface,
        period=4.0,
    )
    reference = _make_reference(config)
    pose = reference.x_des_fn(jnp.array(1.0))
    R = rotation_vector_to_rotation_matrix(pose[:3])
    position = pose[3:]

    normal0 = jnp.array([1.0, 0.0, 0.0])
    if surface == "plane":
        expected_normal = normal0
    elif surface == "cylinder":
        axis = jnp.array([0.0, -1.0, 0.0])
        origin = -config.surface_radius * normal0
        radial = position - origin - jnp.dot(position - origin, axis) * axis
        expected_normal = radial / jnp.linalg.norm(radial)
    else:
        center = -config.surface_radius * normal0
        expected_normal = position - center
        expected_normal = expected_normal / jnp.linalg.norm(expected_normal)

    assert jnp.allclose(R[:, 0], expected_normal, atol=1e-6)


@pytest.mark.parametrize("surface", trajectory_primitives.SURFACES)
def test_surface_following_positions_remain_on_selected_surface(surface):
    osd = FakeOperationalSpaceDynamics()
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        position_primitive="figure-eight",
        orientation_primitive="surface-following",
        surface=surface,
        period=4.0,
    )
    reference = trajectory_primitives.make_operational_space_reference_trajectory(
        osd=osd,
        q0=jnp.zeros((1,)),
        ts=jnp.linspace(0.0, 4.0, 9),
        config=config,
    )
    surface_geometry = trajectory_primitives.make_surface_geometry_metadata(
        osd=osd,
        q0=jnp.zeros((1,)),
        config=config,
    )

    for t in (0.0, 0.75, 1.5, 2.25, 3.0):
        position = reference.x_des_fn(jnp.array(t))[3:]
        if surface == "plane":
            signed_distance = jnp.dot(
                position - surface_geometry.plane_point,
                surface_geometry.plane_normal,
            )
            assert jnp.allclose(signed_distance, 0.0, atol=1e-12)
        elif surface == "cylinder":
            center_to_point = position - surface_geometry.cylinder_origin
            radial = (
                center_to_point
                - jnp.dot(center_to_point, surface_geometry.cylinder_axis)
                * surface_geometry.cylinder_axis
            )
            assert jnp.allclose(
                jnp.linalg.norm(radial),
                config.surface_radius,
                atol=1e-12,
            )
        else:
            assert jnp.allclose(
                jnp.linalg.norm(position - surface_geometry.sphere_center),
                config.surface_radius,
                atol=1e-12,
            )


def test_surface_following_rejects_planar_pose_tracking():
    osd = FakeOperationalSpaceDynamics(is_planar=True)
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        orientation_primitive="surface-following",
    )

    with pytest.raises(ValueError, match="Surface-following"):
        trajectory_primitives.make_operational_space_reference_trajectory(
            osd=osd,
            q0=jnp.zeros((1,)),
            ts=jnp.linspace(0.0, 1.0, 3),
            config=config,
        )
