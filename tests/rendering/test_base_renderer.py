import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.matplotlib_renderer import MatplotlibRenderer
from soromox.rendering.opencv_planar_renderer import OpenCVPlanarRenderer
from soromox.systems.soft_robot import CrossSectionGeometry
from soromox.utils.geometry import poses


class DummyPlanarRobot:
    is_planar = True
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])

    def __init__(self, base_pose: jnp.ndarray):
        self.base_pose = jnp.asarray(base_pose)
        self.base_transform = poses.planar_pose_to_transform(self.base_pose)

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


class DummySpatialRobot:
    is_planar = False
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])

    def __init__(self, base_pose: jnp.ndarray):
        self.base_pose = jnp.asarray(base_pose)
        self.base_transform = poses.quaternion_pose_to_transform(self.base_pose)

    def forward_kinematics_batched(self, q, s_ps):
        transforms = jnp.repeat(self.base_transform[None, :, :], s_ps.shape[0], axis=0)
        offsets = s_ps[:, None] * self.base_transform[:3, 0]
        return transforms.at[:, :3, 3].add(offsets)

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


class FakeMatplotlib3DAxis:
    def __init__(self):
        self.view = None

    def set_xlim(self, *args):
        return None

    def set_ylim(self, *args):
        return None

    def set_zlim(self, *args):
        return None

    def set_xlabel(self, *args):
        return None

    def set_ylabel(self, *args):
        return None

    def set_zlabel(self, *args):
        return None

    def view_init(self, *, elev, azim):
        self.view = (float(elev), float(azim))


class FakeOpen3DVisualizer:
    def __init__(self):
        self.reset_view_point_arg = None
        self.polled = False
        self.updated = False

    def reset_view_point(self, arg):
        self.reset_view_point_arg = arg

    def poll_events(self):
        self.polled = True

    def update_renderer(self):
        self.updated = True


class FakeOpen3DViewControl:
    def __init__(self):
        self.front = None
        self.lookat = None
        self.up = None
        self.zoom = None

    def set_front(self, front):
        self.front = np.asarray(front, dtype=np.float64)

    def set_lookat(self, lookat):
        self.lookat = np.asarray(lookat, dtype=np.float64)

    def set_up(self, up):
        self.up = np.asarray(up, dtype=np.float64)

    def set_zoom(self, zoom):
        self.zoom = float(zoom)


class FakeViserCamera:
    def __init__(self):
        self.position = None
        self.look_at = None
        self.up_direction = None
        self.fov = None


class FakeViserClient:
    def __init__(self):
        self.camera = FakeViserCamera()


class FakeViserServer:
    def __init__(self, clients):
        self._clients = clients
        self.on_client_connect_callback = None

    def get_clients(self):
        return self._clients

    def on_client_connect(self, callback):
        self.on_client_connect_callback = callback
        return callback


def test_renderer_exposes_base_pose_transform_and_axis():
    robot = DummyPlanarRobot(jnp.array([jnp.pi / 2, 1.0, 2.0]))
    renderer = DummyRenderer(robot)

    assert_allclose(renderer.base_pose, jnp.array([jnp.pi / 2, 1.0, 2.0]))
    assert_allclose(renderer.base_transform, robot.base_transform)
    assert_allclose(renderer._base_position(dim=3), np.array([1.0, 2.0, 0.0]))
    assert_allclose(
        renderer._base_tangent_axis(dim=3), np.array([0.0, 1.0, 0.0]), atol=1e-7
    )


def test_camera_auto_position_rotates_default_offset_by_base_pose():
    config = CameraConfig(distance_factor=2.0, position_offset=(1.0, 0.0, 0.0))
    center = np.array([10.0, 20.0, 30.0])
    base_transform = poses.planar_pose_to_transform(
        jnp.array([jnp.pi / 2, 1.0, 2.0])
    )

    camera_pos, look_at = config.compute_auto_position(
        center,
        max_extent=0.5,
        reference_transform=np.asarray(base_transform),
    )

    assert_allclose(camera_pos, np.array([10.0, 21.0, 30.0]), atol=1e-12)
    assert_allclose(look_at, center, atol=1e-12)


def test_camera_explicit_position_remains_world_space_with_base_pose():
    config = CameraConfig(
        position=(1.0, 2.0, 3.0),
        distance_factor=2.0,
        position_offset=(1.0, 0.0, 0.0),
    )
    center = np.array([10.0, 20.0, 30.0])
    base_transform = poses.planar_pose_to_transform(
        jnp.array([jnp.pi / 2, 1.0, 2.0])
    )

    camera_pos, look_at = config.compute_auto_position(
        center,
        max_extent=0.5,
        reference_transform=np.asarray(base_transform),
    )

    assert_allclose(camera_pos, np.array([1.0, 2.0, 3.0]), atol=1e-12)
    assert_allclose(look_at, center, atol=1e-12)


def test_camera_up_vector_is_world_space():
    config = CameraConfig(up=(0.0, 0.0, 1.0))

    up = config.compute_up()

    assert_allclose(up, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_matplotlib_3d_default_camera_uses_base_pose_orientation():
    robot = DummySpatialRobot(jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]))
    renderer = MatplotlibRenderer(robot)
    axis = FakeMatplotlib3DAxis()

    renderer._setup_axes(
        axis,
        width_m=3.0,
        scene_center=np.zeros(3),
        scene_extent=1.0,
        camera_config=None,
    )

    assert axis.view is not None
    expected_view = np.array(
        [
            np.degrees(np.arctan2(8.0, np.hypot(8.0, -5.0))),
            np.degrees(np.arctan2(-5.0, 8.0)),
        ]
    )
    assert_allclose(np.array(axis.view), expected_view, atol=1e-12)


def test_open3d_interactive_camera_front_points_from_eye_to_target():
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import Open3DRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = Open3DRenderer(robot)
    vis = FakeOpen3DVisualizer()
    ctrl = FakeOpen3DViewControl()
    scene_data = type(
        "SceneDataStub",
        (),
        {"curves": np.array([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])},
    )()

    renderer._setup_interactive_camera(
        vis,
        ctrl,
        scene_data,
        camera_config=CameraConfig(distance_factor=1.0, position_offset=(1.0, 0.0, 0.0)),
    )

    assert_allclose(ctrl.front, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert_allclose(ctrl.lookat, np.array([0.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(ctrl.up, np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert vis.reset_view_point_arg is True
    assert vis.polled
    assert vis.updated


def test_viser_camera_uses_shared_up_and_radian_fov():
    pytest.importorskip("viser")
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    client = FakeViserClient()
    server = FakeViserServer({"client": client})
    renderer._server = server
    curves = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    renderer._setup_camera(
        curves,
        camera_config=CameraConfig(
            fov=60.0,
            distance_factor=1.0,
            position_offset=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
    )

    assert_allclose(client.camera.position, np.array([1.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(client.camera.look_at, np.array([0.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(client.camera.up_direction, np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert_allclose(client.camera.fov, np.pi / 3.0, atol=1e-12)
    assert server.on_client_connect_callback is not None


def test_viser_camera_accepts_full_trajectory_curves_for_bounds():
    pytest.importorskip("viser")
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    client = FakeViserClient()
    renderer._server = FakeViserServer({"client": client})
    curves = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            ]
        ]
    )

    renderer._setup_camera(
        curves,
        camera_config=CameraConfig(distance_factor=1.0, position_offset=(1.0, 0.0, 0.0)),
    )

    assert_allclose(client.camera.look_at, np.array([1.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(client.camera.position, np.array([3.0, 0.5, 0.0]), atol=1e-12)


def test_viser_default_camera_uses_backend_specific_distance():
    pytest.importorskip("viser")
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    client = FakeViserClient()
    renderer._server = FakeViserServer({"client": client})
    curves = np.array([[[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]])

    renderer._setup_camera(curves, camera_config=None)

    max_extent = 2.0
    center = np.array([1.0, 0.5, 0.0])
    expected_position = center + np.array([0.8, -0.8, 0.5]) * max_extent
    assert_allclose(client.camera.look_at, center, atol=1e-12)
    assert_allclose(client.camera.position, expected_position, atol=1e-12)
    assert_allclose(client.camera.fov, np.deg2rad(75.0), atol=1e-12)


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
