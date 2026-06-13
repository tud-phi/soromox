import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer
from soromox.rendering.opencv_planar_renderer import OpenCVPlanarRenderer
from soromox.systems.soft_robot import CrossSectionGeometry
from soromox.utils import lie_algebra as lie


class DummyPlanarRobot:
    is_planar = True
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])

    def __init__(self, base_pose: jnp.ndarray):
        self.base_pose = jnp.asarray(base_pose)
        self.base_transform = lie.transform_from_planar_pose_SE2(self.base_pose)

    def forward_kinematics_batched(self, q, s_ps):
        theta = self.base_pose[0]
        direction = jnp.array([jnp.cos(theta), jnp.sin(theta)])
        xy = self.base_pose[1:3] + s_ps[:, None] * direction
        theta_ps = jnp.full((s_ps.shape[0], 1), theta, dtype=s_ps.dtype)
        return jnp.concatenate([theta_ps, xy], axis=1)

    def forward_kinematics(self, q, s):
        return self.forward_kinematics_batched(q, jnp.asarray([s]))[0]

    def cross_section_geometry(self, q, s):
        return CrossSectionGeometry.CIRCULAR, jnp.array([0.02])


class DummyRenderer(BaseSoftRobotRenderer):
    def render_frame(self, q):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def render_sequence(self, ts, q_ts, playback_speed=1.0, record_path=None, **kwargs):
        return None

    def show(self, q, **kwargs):
        return None


def test_renderer_exposes_base_pose_transform_and_axis():
    robot = DummyPlanarRobot(jnp.array([jnp.pi / 2, 1.0, 2.0]))
    renderer = DummyRenderer(robot)

    assert_allclose(renderer.base_pose, jnp.array([jnp.pi / 2, 1.0, 2.0]))
    assert_allclose(renderer.base_transform, robot.base_transform)
    assert_allclose(renderer._base_position(dim=3), np.array([1.0, 2.0, 0.0]))
    assert_allclose(
        renderer._base_tangent_axis(dim=3), np.array([0.0, 1.0, 0.0]), atol=1e-7
    )


def test_renderer_normalizes_base_offsets_to_curve_dimension():
    robot = DummyPlanarRobot(jnp.array([0.0, 0.0, 0.0]))
    renderer = DummyRenderer(robot)

    offsets = renderer._normalize_base_offsets(
        jnp.array([[0.1, -0.2], [0.3, 0.4]]),
        num_robots=2,
        target_dim=3,
    )

    assert offsets.shape == (2, 3)
    assert_allclose(offsets, jnp.array([[0.1, -0.2, 0.0], [0.3, 0.4, 0.0]]))


def test_renderer_can_slice_configured_base_offsets_to_batch_size():
    robot = DummyPlanarRobot(jnp.array([0.0, 0.0, 0.0]))
    renderer = DummyRenderer(robot)

    offsets = renderer._normalize_base_offsets(
        jnp.array([[0.1, -0.2], [0.3, 0.4], [0.5, 0.6]]),
        num_robots=2,
        target_dim=2,
        allow_extra_rows=True,
    )

    assert offsets.shape == (2, 2)
    assert_allclose(offsets, jnp.array([[0.1, -0.2], [0.3, 0.4]]))


def test_renderer_rejects_extra_explicit_base_offsets_by_default():
    robot = DummyPlanarRobot(jnp.array([0.0, 0.0, 0.0]))
    renderer = DummyRenderer(robot)

    with pytest.raises(ValueError, match="number of robots"):
        renderer._normalize_base_offsets(
            jnp.array([[0.1, -0.2], [0.3, 0.4]]),
            num_robots=1,
            target_dim=2,
        )


def test_matplotlib_2d_base_marker_is_perpendicular_to_base_tangent():
    base = np.array([1.0, 2.0])
    tangent = np.array([0.0, 1.0])

    marker = MatplotlibRenderer._base_plate_marker_points(
        base, tangent, marker_diameter=0.4, dim=2
    )

    assert marker.shape == (2, 2)
    assert_allclose(marker.mean(axis=0), base)
    assert_allclose(marker[1] - marker[0], np.array([-0.4, 0.0]))
    assert_allclose(np.dot(marker[1] - marker[0], tangent), 0.0, atol=1e-12)


def test_matplotlib_3d_base_marker_lies_in_plate_face_plane():
    base = np.array([1.0, 2.0, 3.0])
    normal = np.array([1.0, 0.0, 0.0])

    marker = MatplotlibRenderer._base_plate_marker_points(
        base, normal, marker_diameter=0.4, dim=3
    )

    assert marker.shape == (65, 3)
    assert_allclose(marker[0], marker[-1], atol=1e-12)
    assert_allclose((marker - base) @ normal, np.zeros(marker.shape[0]), atol=1e-12)
    assert_allclose(np.linalg.norm(marker - base, axis=1), 0.2, atol=1e-12)


def test_opencv_planar_base_marker_uses_base_pose_and_offset():
    robot = DummyPlanarRobot(jnp.array([0.0, 0.2, 0.1]))
    renderer = OpenCVPlanarRenderer(
        robot,
        width=100,
        height=100,
        num_points=5,
        base_color=(0, 255, 0),
        backbone_color=(0, 0, 0),
        length_scale=2.0,
        origin_uv=(50, 50),
    )

    img = renderer.render_frame(jnp.array([]), base_offsets=jnp.array([0.1, -0.1]))

    # ppm = height / (length_scale * L_max) = 50. Base xy + offset = [0.3, 0.0].
    expected_uv = (65, 50)
    assert tuple(img[expected_uv[1], expected_uv[0]]) == (0, 255, 0)
