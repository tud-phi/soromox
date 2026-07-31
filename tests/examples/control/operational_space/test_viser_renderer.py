import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from soromox.utils.geometry.rotations import RotationRepresentation

MODULE_DIR = (
    Path(__file__).resolve().parents[4] / "examples" / "control" / "operational_space"
)

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import trajectory_primitives  # noqa: E402

VISER_RENDERER_PATH = MODULE_DIR / "viser_operational_space_renderer.py"
spec = importlib.util.spec_from_file_location(
    "example_operational_space_viser_renderer",
    VISER_RENDERER_PATH,
)
assert spec is not None
assert spec.loader is not None
viser_operational_space_renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = viser_operational_space_renderer
spec.loader.exec_module(viser_operational_space_renderer)


@dataclass
class FakeOperationalSpaceDynamics:
    is_planar: bool = False

    def __post_init__(self):
        self.n_points = 1
        self.rotation_representation = RotationRepresentation.ROTATION_VECTOR
        self.n_velocity_dim = 6
        self.n_pose_dim = 6
        self.n_orientation_dim = 3
        self.n_angular_velocity_dim = 3
        self.x0_full = jnp.zeros((6,))

    def operational_space_poses(self, q):
        return self.x0_full


def _surface_geometry(surface):
    config = trajectory_primitives.TaskSpaceTrajectoryConfig(
        tracking="pose",
        orientation_primitive="surface-following",
        surface=surface,
        ramp_time=0.0,
    )
    return trajectory_primitives.make_surface_geometry_metadata(
        osd=FakeOperationalSpaceDynamics(),
        q0=jnp.zeros((1,)),
        config=config,
    )


@pytest.mark.parametrize("surface", trajectory_primitives.SURFACES)
def test_surface_meshes_are_finite_and_triangular(surface):
    surface_geometry = _surface_geometry(surface)
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.01, 0.0],
            [0.04, -0.01, 0.0],
            [0.06, 0.0, 0.01],
        ],
        dtype=np.float64,
    )

    mesh = viser_operational_space_renderer.make_surface_mesh(
        surface_geometry,
        reference_positions,
        resolution=16,
    )

    assert mesh.vertices.ndim == 2
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.ndim == 2
    assert mesh.faces.shape[1] == 3
    assert np.all(np.isfinite(mesh.vertices))
    assert np.all(mesh.faces >= 0)
    assert np.all(mesh.faces < len(mesh.vertices))


def test_fading_trail_colors_progress_from_faded_to_current():
    colors = viser_operational_space_renderer.make_fading_trail_colors(
        6,
        current_color=(1.0, 0.0, 0.0),
        faded_color=(0.8, 0.8, 0.8),
    )[:, 0, :]

    current = np.array([255, 0, 0], dtype=np.float64)
    distances = np.linalg.norm(colors.astype(np.float64) - current, axis=1)

    assert colors.shape == (6, 3)
    assert np.all(np.diff(distances) <= 0.0)
    assert distances[-1] < distances[0]


def test_operational_space_camera_config_is_finite_yz_aligned_close_view():
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.05, 0.2],
            [0.0, 0.1, 0.18],
        ],
        dtype=np.float64,
    )
    actual_positions = reference_positions + np.array([0.01, -0.02, 0.0])

    camera = viser_operational_space_renderer.make_operational_space_camera_config(
        reference_positions,
        actual_positions,
    )

    position = np.asarray(camera.position)
    look_at = np.asarray(camera.look_at)

    assert camera.fov == 36.0
    assert position.shape == (3,)
    assert look_at.shape == (3,)
    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(look_at))
    assert position[2] > look_at[2]
    all_points = np.concatenate(
        [reference_positions, actual_positions, np.zeros((1, 3))],
        axis=0,
    )
    raw_center_z = 0.5 * (all_points[:, 2].min() + all_points[:, 2].max())
    assert look_at[2] > raw_center_z
    assert position[0] == look_at[0]


def test_target_cross_segments_center_on_goal_position():
    position = np.array([1.0, 2.0, 3.0])
    radius = 0.25

    segments = viser_operational_space_renderer.make_target_cross_segments(
        position,
        radius,
    )

    assert segments.shape == (3, 2, 3)
    assert np.allclose(np.mean(segments, axis=1), position)
    assert np.allclose(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1), 0.5)


def test_target_ring_segments_stay_centered_on_goal_position():
    position = np.array([1.0, 2.0, 3.0])
    radius = 0.25

    segments = viser_operational_space_renderer.make_target_ring_segments(
        position,
        radius,
        num_points=12,
    )

    assert segments.shape == (36, 2, 3)
    assert np.all(np.isfinite(segments))
    assert np.allclose(
        np.linalg.norm(segments.reshape(-1, 3) - position, axis=1),
        radius,
    )


def test_target_marker_segments_lie_in_tangent_plane():
    position = np.array([1.0, 2.0, 3.0])
    normal = np.array([1.0, 0.0, 0.0])
    radius = 0.25

    segments = viser_operational_space_renderer.make_target_marker_segments(
        position,
        normal,
        radius,
        num_points=16,
    )

    assert segments.shape == (18, 2, 3)
    assert np.all(np.isfinite(segments))
    assert np.allclose((segments.reshape(-1, 3) - position) @ normal, 0.0)


def test_transparent_robot_color_config_preserves_requested_alpha():
    config = viser_operational_space_renderer.make_transparent_robot_color_config(
        num_points=7,
        opacity=0.35,
    )

    point_colors = np.asarray(config.backbone.point_colors)

    assert point_colors.shape == (7, 4)
    assert np.allclose(point_colors[:, 3], 0.35)


@pytest.mark.parametrize("surface", trajectory_primitives.SURFACES)
def test_surface_geometry_initial_normal_matches_reference_convention(surface):
    surface_geometry = _surface_geometry(surface)
    initial_position = jnp.zeros((3,))

    normal = trajectory_primitives.surface_normal_at(
        initial_position,
        surface_geometry,
    )

    assert jnp.allclose(normal, jnp.array([1.0, 0.0, 0.0]), atol=1e-6)
