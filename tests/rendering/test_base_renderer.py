import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering.actuators import (
    ActuatorVisualLayer,
    BatchedActuatorVisualLayer,
    TrajectoryActuatorVisualLayer,
)
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


class DummyActuatedPlanarRobot(DummyPlanarRobot):
    def actuator_visual_layers(self, q, s_points, *, actuator_inputs=None):
        del q, actuator_inputs
        points = jnp.stack(
            [
                s_points,
                jnp.full_like(s_points, 0.25),
            ],
            axis=1,
        )[None, ...]
        return (ActuatorVisualLayer(name="offset_cable", kind="cable", points=points),)


class DummyActuatedSpatialRobot(DummySpatialRobot):
    def actuator_visual_layers(self, q, s_points, *, actuator_inputs=None):
        del q
        tendon = jnp.stack(
            [
                s_points,
                jnp.full_like(s_points, 0.1),
                jnp.zeros_like(s_points),
            ],
            axis=1,
        )[None, ...]
        muscles = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]],
            ]
        )
        pressure = (
            jnp.zeros((2,))
            if actuator_inputs is None
            else jnp.asarray(actuator_inputs).reshape((2,))
        )
        return (
            ActuatorVisualLayer(name="sampled_tendon", kind="tendon", points=tendon),
            ActuatorVisualLayer(
                name="mckibben_pair",
                kind="muscle",
                points=muscles,
                scalar_fields={"pressure": pressure},
            ),
        )


class DummyBatchedActuatedSpatialRobot(DummySpatialRobot):
    def __init__(self, base_pose: jnp.ndarray):
        super().__init__(base_pose)
        self.single_calls = 0
        self.batched_calls = 0

    def actuator_visual_layers(self, q, s_points, *, actuator_inputs=None):
        del q, s_points, actuator_inputs
        self.single_calls += 1
        raise AssertionError("single actuator hook should not be used")

    def actuator_visual_layers_batched(self, q_batch, s_points, *, actuator_inputs=None):
        self.batched_calls += 1
        num_robots = int(q_batch.shape[0])
        base_paths = jnp.stack(
            [
                jnp.stack(
                    [
                        s_points,
                        jnp.full_like(s_points, 0.1),
                        jnp.zeros_like(s_points),
                    ],
                    axis=1,
                ),
                jnp.stack(
                    [
                        s_points,
                        jnp.full_like(s_points, 0.2),
                        jnp.zeros_like(s_points),
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )
        points = jnp.broadcast_to(base_paths, (num_robots, *base_paths.shape))
        pressure = (
            jnp.zeros((num_robots, 2))
            if actuator_inputs is None
            else jnp.asarray(actuator_inputs).reshape((num_robots, 2))
        )
        return (
            BatchedActuatorVisualLayer(
                name="batched_cables",
                kind="cable",
                points=points,
                scalar_fields={"pressure": pressure},
            ),
        )


class DummyTrajectoryActuatedSpatialRobot(DummySpatialRobot):
    def __init__(self, base_pose: jnp.ndarray):
        super().__init__(base_pose)
        self.trajectory_calls = 0

    def actuator_visual_layers_trajectory(
        self, q_ts, s_points, *, actuator_inputs=None
    ):
        self.trajectory_calls += 1
        num_robots, num_steps = int(q_ts.shape[0]), int(q_ts.shape[1])
        path = jnp.stack(
            [
                s_points,
                jnp.full_like(s_points, 0.3),
                jnp.zeros_like(s_points),
            ],
            axis=1,
        )[None, ...]
        points = jnp.broadcast_to(path, (num_robots, num_steps, *path.shape))
        pressure = (
            jnp.zeros((num_robots, num_steps, 1))
            if actuator_inputs is None
            else jnp.asarray(actuator_inputs).reshape((num_robots, num_steps, 1))
        )
        return (
            TrajectoryActuatorVisualLayer(
                name="trajectory_muscle",
                kind="muscle",
                points=points,
                scalar_fields={"pressure": pressure},
            ),
        )


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


class FakeViserLineHandle:
    def __init__(self, *, name, points, colors, line_width):
        self.name = name
        self.points = points
        self.colors = colors
        self.line_width = line_width
        self.remove_count = 0

    def remove(self):
        self.remove_count += 1


class FakeViserScene:
    def __init__(self):
        self.line_segments = []

    def add_line_segments(self, *, name, points, colors, line_width):
        handle = FakeViserLineHandle(
            name=name,
            points=points,
            colors=colors,
            line_width=line_width,
        )
        self.line_segments.append(handle)
        return handle


class FakeViserActuatorServer:
    def __init__(self):
        self.scene = FakeViserScene()


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


def test_viser_actuator_geometry_updates_existing_line_handles():
    pytest.importorskip("viser")
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummyActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = ViserRenderer(robot, auto_start=False)
    renderer._server = FakeViserActuatorServer()
    renderer._scene_handles = SceneHandles()

    renderer._build_actuator_geometry(
        jnp.zeros((1, 0)),
        np.zeros((1, 3)),
        1,
    )

    first_handles = tuple(renderer._scene_handles.actuator_lines)
    assert len(first_handles) == 3
    assert len(renderer._server.scene.line_segments) == 3

    renderer._build_actuator_geometry(
        jnp.zeros((1, 0)),
        np.array([[1.0, 0.0, 0.0]]),
        1,
        actuator_inputs=jnp.array([[1.0, 2.0]]),
    )

    assert tuple(renderer._scene_handles.actuator_lines) == first_handles
    assert len(renderer._server.scene.line_segments) == 3
    assert all(handle.remove_count == 0 for handle in first_handles)
    assert_allclose(first_handles[0].points[0, 0], np.array([1.0, 0.1, 0.0]))


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


def test_renderer_computes_actuator_visual_layers_single_and_batched():
    robot = DummyActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = DummyRenderer(robot, num_points=5)

    single = renderer.compute_actuator_visual_layers(jnp.array([]))

    assert len(single) == 2
    assert single[0].points.shape == (1, 5, 3)
    assert single[1].points.shape == (2, 2, 3)

    batched = renderer.compute_actuator_visual_layers_batched(
        jnp.zeros((2, 0)),
        jnp.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]]),
        actuator_inputs=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert len(batched) == 2
    assert batched[0].points.shape == (2, 1, 5, 3)
    assert batched[1].points.shape == (2, 2, 2, 3)
    assert_allclose(batched[0].points[1, 0, 0], jnp.array([1.0, -0.9, 0.5]))
    assert_allclose(
        batched[1].scalar_fields["pressure"],
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_renderer_uses_batched_actuator_visual_fast_path():
    robot = DummyBatchedActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = DummyRenderer(robot, num_points=5)

    layers = renderer.compute_actuator_visual_layers_batched(
        jnp.zeros((3, 0)),
        jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [2.0, 0.0, -0.5],
            ]
        ),
        actuator_inputs=jnp.arange(6.0).reshape(3, 2),
    )

    assert robot.batched_calls == 1
    assert robot.single_calls == 0
    assert len(layers) == 1
    assert layers[0].points.shape == (3, 2, 5, 3)
    assert_allclose(layers[0].points[1, 0, 0], jnp.array([1.0, -0.9, 0.5]))
    assert_allclose(layers[0].scalar_fields["pressure"], jnp.arange(6.0).reshape(3, 2))


def test_renderer_computes_trajectory_actuator_visual_layers():
    robot = DummyActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = DummyRenderer(robot, num_points=4)

    layers = renderer.compute_actuator_visual_layers_trajectory(
        jnp.zeros((2, 3, 0)),
        jnp.zeros((2, 3)),
        actuator_inputs=jnp.arange(12.0).reshape(2, 3, 2),
    )

    assert len(layers) == 2
    assert layers[0].points.shape == (2, 3, 1, 4, 3)
    assert layers[1].points.shape == (2, 3, 2, 2, 3)
    assert_allclose(
        layers[1].scalar_fields["pressure"], jnp.arange(12.0).reshape(2, 3, 2)
    )


def test_renderer_uses_trajectory_actuator_visual_fast_path():
    robot = DummyTrajectoryActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = DummyRenderer(robot, num_points=4)

    layers = renderer.compute_actuator_visual_layers_trajectory(
        jnp.zeros((2, 3, 0)),
        jnp.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]]),
        actuator_inputs=jnp.arange(6.0).reshape(2, 3, 1),
    )

    assert robot.trajectory_calls == 1
    assert len(layers) == 1
    assert layers[0].points.shape == (2, 3, 1, 4, 3)
    assert_allclose(layers[0].points[1, 2, 0, 0], jnp.array([1.0, -0.7, 0.5]))
    assert_allclose(
        layers[0].scalar_fields["pressure"], jnp.arange(6.0).reshape(2, 3, 1)
    )


def test_matplotlib_actuator_overlay_changes_rendered_frame():
    robot = DummyActuatedSpatialRobot(
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    renderer = MatplotlibRenderer(robot, width=240, height=180, num_points=8)

    without_actuators = renderer.render_frame(jnp.array([]), render_actuators=False)
    with_actuators = renderer.render_frame(jnp.array([]), render_actuators=True)

    assert without_actuators.shape == with_actuators.shape
    assert np.any(without_actuators != with_actuators)


def test_opencv_planar_actuator_overlay_draws_configured_color():
    robot = DummyActuatedPlanarRobot(jnp.array([0.0, 0.0, 0.0]))
    renderer = OpenCVPlanarRenderer(
        robot,
        width=100,
        height=100,
        num_points=5,
        actuator_color=(0, 0, 255),
        actuator_thickness=2,
        length_scale=2.0,
        origin_uv=(20, 50),
    )

    img = renderer.render_frame(jnp.array([]), render_actuators=True)

    assert np.any(np.all(img == np.array([0, 0, 255], dtype=np.uint8), axis=-1))


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
