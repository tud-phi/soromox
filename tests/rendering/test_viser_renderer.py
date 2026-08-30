"""Tests for the generic Viser soft-robot renderer."""

from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

from soromox.rendering.actuators import ActuatorVisualLayer
from soromox.rendering.camera_config import CameraConfig
from soromox.systems.components import CrossSectionGeometry
from soromox.utils.geometry import poses


class DummySpatialRobot:
    is_planar = False
    floating_base = False
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])

    def __init__(self, base_pose: jnp.ndarray):
        self.fixed_base_pose = jnp.asarray(base_pose)
        self.base_transform = poses.quaternion_pose_to_transform(self.fixed_base_pose)

    def forward_kinematics_abscissa_batched(self, q, s_ps):
        transforms = jnp.repeat(self.base_transform[None, :, :], s_ps.shape[0], axis=0)
        offsets = s_ps[:, None] * self.base_transform[:3, 0]
        return transforms.at[:, :3, 3].add(offsets)

    def forward_kinematics(self, q, s):
        return self.forward_kinematics_abscissa_batched(q, jnp.asarray([s]))[0]

    def cross_section_geometry(self, q, s):
        return CrossSectionGeometry.CIRCULAR, jnp.array([0.02])


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


class FakeViserCamera:
    def __init__(self):
        self.position = None
        self.look_at = None
        self.up_direction = None
        self.fov = None
        self.render = None

    def get_render(self, *, height, width):
        del height, width
        return self.render


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

    def get_port(self):
        return 8080


class FakeViserLineHandle:
    def __init__(self, *, name, points, colors, line_width):
        self.name = name
        self.points = points
        self.colors = colors
        self.line_width = line_width
        self.remove_count = 0

    def remove(self):
        self.remove_count += 1


class FakeViserGeometryHandle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.remove_count = 0

    def remove(self):
        self.remove_count += 1


class FakeViserScene:
    def __init__(self):
        self.line_segments = []
        self.icospheres = []
        self.trimeshes = []
        self.simple_meshes = []
        self.batched_meshes = []
        self.grids = []

    def add_grid(self, **kwargs):
        handle = FakeViserGeometryHandle(**kwargs)
        self.grids.append(handle)
        return handle

    def add_line_segments(self, *, name, points, colors, line_width):
        handle = FakeViserLineHandle(
            name=name,
            points=points,
            colors=colors,
            line_width=line_width,
        )
        self.line_segments.append(handle)
        return handle

    def add_icosphere(self, **kwargs):
        handle = FakeViserGeometryHandle(**kwargs)
        self.icospheres.append(handle)
        return handle

    def add_mesh_trimesh(self, **kwargs):
        handle = FakeViserGeometryHandle(**kwargs)
        self.trimeshes.append(handle)
        return handle

    def add_mesh_simple(self, **kwargs):
        handle = FakeViserGeometryHandle(**kwargs)
        self.simple_meshes.append(handle)
        return handle

    def add_batched_meshes_simple(self, **kwargs):
        handle = FakeViserGeometryHandle(**kwargs)
        self.batched_meshes.append(handle)
        return handle


class FakeViserActuatorServer:
    def __init__(self):
        self.scene = FakeViserScene()
        self.atomic_calls = 0

    @contextmanager
    def atomic(self):
        self.atomic_calls += 1
        yield

    def get_port(self):
        return 8080


class FakeViserGuiHandle:
    def __init__(self, *, value=None, icon=None):
        self.value = value
        self.icon = icon
        self.update_callback = None
        self.click_callback = None

    def on_update(self, callback):
        self.update_callback = callback
        return callback

    def on_click(self, callback):
        self.click_callback = callback
        return callback

    def trigger_update(self, value):
        self.value = value
        event = type("FakeViserGuiEvent", (), {"target": self})()
        self.update_callback(event)


class FakeViserGuiFolder:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeViserGui:
    def __init__(self):
        self.handles = {}

    def add_folder(self, name):
        del name
        return FakeViserGuiFolder()

    def add_button(self, name, *, icon):
        handle = FakeViserGuiHandle(icon=icon)
        self.handles[name] = handle
        return handle

    def add_slider(self, name, *, min, max, step, initial_value):
        del min, max, step
        handle = FakeViserGuiHandle(value=initial_value)
        self.handles[name] = handle
        return handle

    def add_checkbox(self, name, *, initial_value):
        handle = FakeViserGuiHandle(value=initial_value)
        self.handles[name] = handle
        return handle

    def add_text(self, name, *, initial_value, disabled):
        del disabled
        handle = FakeViserGuiHandle(value=initial_value)
        self.handles[name] = handle
        return handle


class FakeViserPlaybackServer:
    def __init__(self):
        self.gui = FakeViserGui()


def test_viser_camera_uses_shared_up_and_radian_fov():
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
        camera_config=CameraConfig(
            distance_factor=1.0, position_offset=(1.0, 0.0, 0.0)
        ),
    )

    assert_allclose(client.camera.look_at, np.array([1.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(client.camera.position, np.array([3.0, 0.5, 0.0]), atol=1e-12)


def test_viser_capture_rejects_missing_frame():
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    client = FakeViserClient()
    renderer._server = FakeViserServer({"client": client})

    with pytest.raises(RuntimeError, match="did not return a frame"):
        renderer._capture_viser_frame(client, timeout=0.1)


def test_viser_sequence_hooks_and_capture_reuse(monkeypatch, tmp_path):
    import soromox.rendering.viser_renderer as viser_module

    class HookedRenderer(viser_module.ViserRenderer):
        def __init__(self, *args, **kwargs):
            self.events = []
            self.current_frame = None
            super().__init__(*args, **kwargs)

        def _after_sequence_scene_built(self):
            self.events.append(("scene", None))

        def _after_sequence_frame_updated(self, frame_idx):
            self.current_frame = frame_idx
            self.events.append(("overlay", frame_idx))

    class FakeWriter:
        instances = []

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.frames = []
            self.closed = False
            self.__class__.instances.append(self)

        def write(self, frame):
            self.frames.append(np.asarray(frame).copy())

        def close(self):
            self.closed = True

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = HookedRenderer(
        robot,
        width=4,
        height=3,
        auto_start=False,
        open_browser=False,
    )
    client = FakeViserClient()
    renderer._server = FakeViserServer({"client": client})
    renderer._running = True

    curves = np.zeros((1, 2, 3), dtype=np.float64)
    frames = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
    renderer.compute_backbone_curves_and_frames_batched = lambda q, offsets: (
        curves,
        frames,
    )
    renderer.compute_backbone_curves_batched = lambda q, offsets: curves
    renderer._clear_scene = lambda: None
    renderer._build_robot_geometry = lambda *args, **kwargs: renderer.events.append(
        ("robot", 0)
    )
    renderer._build_static_spheres = lambda positions, radii, colors: (
        renderer.events.append(("static", None))
    )
    renderer._build_dynamic_spheres = lambda positions, radii, colors: (
        renderer.events.append(("dynamic", None))
    )
    renderer._setup_camera = lambda *args, **kwargs: None
    renderer._setup_lighting = lambda: None
    renderer._setup_playback_gui = lambda **kwargs: None
    renderer._update_playback_gui = lambda state: None

    def update_frame(frame_idx, *args, **kwargs):
        del args, kwargs
        renderer.events.append(("robot", frame_idx))

    renderer._update_frame = update_frame

    captured_indices = []

    def capture_frame(client_arg, *, timeout):
        del timeout
        captured_indices.append(renderer.current_frame)
        renderer.events.append(("capture", renderer.current_frame))
        frame = np.full(
            (renderer.height, renderer.width, 3),
            renderer.current_frame,
            dtype=np.uint8,
        )
        return frame, client_arg

    renderer._capture_viser_frame = capture_frame
    monkeypatch.setattr(viser_module, "FFmpegVideoWriter", FakeWriter)

    snapshot_one = tmp_path / "frame_1.png"
    snapshot_two = tmp_path / "frame_2.png"
    renderer.render_sequence(
        ts=np.array([0.0, 0.001, 0.002]),
        q_ts=np.zeros((3, 0)),
        record_path=str(tmp_path / "video.mp4"),
        record_every_n=2,
        stop_when_recording_done=True,
        render_actuators=False,
        static_spheres_positions=np.zeros((1, 3)),
        static_spheres_radii=np.array([0.01]),
        static_spheres_colors=np.array([[0.1, 0.2, 0.3]]),
        dynamic_spheres_positions=np.zeros((1, 3, 3)),
        dynamic_spheres_radii=np.array([0.02]),
        dynamic_spheres_colors=np.array([[0.4, 0.5, 0.6]]),
        snapshot_paths={np.int64(1): snapshot_one, 2: snapshot_two},
    )

    assert captured_indices == [0, 1, 2]
    assert (
        renderer.events.index(("robot", 0))
        < renderer.events.index(("scene", None))
        < renderer.events.index(("overlay", 0))
        < renderer.events.index(("capture", 0))
    )
    for frame_idx in (1, 2):
        assert (
            renderer.events.index(("robot", frame_idx))
            < renderer.events.index(("overlay", frame_idx))
            < renderer.events.index(("capture", frame_idx))
        )

    writer = FakeWriter.instances[-1]
    assert writer.closed
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 2]
    assert int(np.asarray(Image.open(snapshot_one))[0, 0, 0]) == 1
    assert int(np.asarray(Image.open(snapshot_two))[0, 0, 0]) == 2


def test_viser_recording_captures_synchronized_batched_geometry_for_every_frame(
    monkeypatch, tmp_path
):
    import soromox.rendering.viser_renderer as viser_module

    class AnimatingActuatedRobot(DummySpatialRobot):
        def forward_kinematics_abscissa_batched(self, q, s_points):
            transforms = jnp.broadcast_to(jnp.eye(4), (s_points.size, 4, 4))
            positions = jnp.stack((s_points, q[0] * s_points, q[1] * s_points), axis=-1)
            return transforms.at[:, :3, 3].set(positions)

        def actuator_visual_layers(self, q, s_points, *, actuator_inputs=None):
            del actuator_inputs
            points = jnp.stack(
                (
                    s_points,
                    jnp.full_like(s_points, q[0]),
                    jnp.full_like(s_points, q[1]),
                ),
                axis=-1,
            )[None]
            return (
                ActuatorVisualLayer(
                    name="animated_tendon",
                    kind="tendon",
                    points=points,
                ),
            )

    class FakeWriter:
        instances = []

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.frames = []
            self.closed = False
            self.__class__.instances.append(self)

        def write(self, frame):
            self.frames.append(np.asarray(frame).copy())

        def close(self):
            self.closed = True

    robot = AnimatingActuatedRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = viser_module.ViserRenderer(
        robot,
        width=4,
        height=3,
        num_points=4,
        auto_start=False,
        open_browser=False,
        backbone_style="discrete",
        show_ground_plane=False,
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._running = True
    renderer._setup_camera = lambda *args, **kwargs: None
    renderer._setup_lighting = lambda: None
    renderer._setup_playback_gui = lambda **kwargs: None
    renderer._update_playback_gui = lambda state: None
    renderer._wait_for_recording_client = lambda timeout: object()

    ts = np.array([0.0, 0.001, 0.002, 0.003])
    q_ts = np.array(
        [
            [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6]],
            [[0.0, 0.0], [-0.1, 0.3], [-0.2, 0.6], [-0.3, 0.9]],
        ],
        dtype=np.float64,
    )
    offsets = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]])
    dynamic_trajectories = np.array(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.4, 0.0], [1.0, 0.6, 0.0]],
        ]
    )
    s_points = np.linspace(0.0, 1.0, renderer.num_points)
    captured_states = []

    def capture_frame(client, *, timeout):
        del timeout
        frame_idx = renderer._animation_state.frame_idx
        backbone_positions = np.stack(
            [
                handles[0].batched_positions
                for handles in renderer._scene_handles.backbone_points
            ]
        )
        actuator_points = renderer._scene_handles.actuator_lines[0].points.copy()
        dynamic_positions = renderer._scene_handles.dynamic_spheres[
            0
        ].batched_positions.copy()

        expected_backbone = []
        expected_actuators = []
        for robot_idx in range(2):
            q = q_ts[robot_idx, frame_idx]
            expected_backbone.append(
                np.stack((s_points, q[0] * s_points, q[1] * s_points), axis=-1)
                + offsets[robot_idx]
            )
            curve = (
                np.stack(
                    (
                        s_points,
                        np.full_like(s_points, q[0]),
                        np.full_like(s_points, q[1]),
                    ),
                    axis=-1,
                )
                + offsets[robot_idx]
            )
            expected_actuators.append(np.stack((curve[:-1], curve[1:]), axis=1))

        assert_allclose(backbone_positions, np.stack(expected_backbone), atol=1e-7)
        assert_allclose(
            actuator_points,
            np.concatenate(expected_actuators, axis=0),
            atol=1e-7,
        )
        assert_allclose(
            dynamic_positions,
            dynamic_trajectories[:, frame_idx],
            atol=1e-7,
        )
        captured_states.append(frame_idx)
        return (
            np.full((renderer.height, renderer.width, 3), frame_idx, dtype=np.uint8),
            client,
        )

    renderer._capture_viser_frame = capture_frame
    monkeypatch.setattr(viser_module, "FFmpegVideoWriter", FakeWriter)

    renderer.render_sequence(
        ts,
        q_ts,
        base_offsets=offsets,
        dynamic_spheres_positions=dynamic_trajectories,
        dynamic_spheres_radii=np.array([0.02, 0.03]),
        dynamic_spheres_colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        record_path=str(tmp_path / "synchronized.mp4"),
        stop_when_recording_done=True,
    )

    assert captured_states == [0, 1, 2, 3]
    writer = FakeWriter.instances[-1]
    assert writer.closed
    assert [int(frame[0, 0, 0]) for frame in writer.frames] == [0, 1, 2, 3]
    assert len(server.scene.batched_meshes) == 3
    assert len(server.scene.line_segments) == 1


@pytest.mark.parametrize(
    "snapshot_paths, message",
    [
        ({-1: "negative.png"}, "within the rendered sequence"),
        ({3: "past_end.png"}, "within the rendered sequence"),
        ({1.0: "integral_float.png"}, "must be an integer"),
        ({1.5: "fractional.png"}, "must be an integer"),
        ({True: "boolean.png"}, "must be an integer"),
        ({np.bool_(False): "numpy_boolean.png"}, "must be an integer"),
        ({0: "not_png.jpg"}, "must be a PNG"),
        ({0: "same.png", 1: "same.png"}, "must be unique"),
    ],
)
def test_viser_snapshot_paths_are_validated_before_rendering(
    snapshot_paths,
    message,
):
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    renderer._server = FakeViserServer({})

    with pytest.raises(ValueError, match=message):
        renderer.render_sequence(
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 0)),
            snapshot_paths=snapshot_paths,
            blocking=False,
        )


def test_viser_capture_timeout_is_explicit():
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    client = FakeViserClient()
    renderer._server = FakeViserServer({"client": client})

    def block_render(*, height, width):
        del height, width
        import time

        time.sleep(0.1)
        return np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)

    client.camera.get_render = block_render
    with pytest.raises(RuntimeError, match="Timed out while capturing"):
        renderer._capture_viser_frame(client, timeout=0.001)


def test_viser_discrete_backbone_batches_positions_colors_and_robot_radius():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False, backbone_style="discrete")
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    curves = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    material_frames = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
    point_colors = np.ones((1, 2, 4), dtype=np.float64)

    renderer._build_robot_geometry(
        curves,
        point_colors,
        material_frames=material_frames,
        base_plate_color=(0.15, 0.15, 0.15),
    )

    assert server.scene.icospheres == []
    assert len(server.scene.batched_meshes) == 1
    handle = server.scene.batched_meshes[0]
    assert_allclose(handle.batched_positions, curves[0], atol=1e-12)
    assert_allclose(
        handle.batched_scales,
        np.full(2, renderer._robot_radius),
        atol=1e-12,
    )
    assert_array_equal(handle.batched_colors, np.full((2, 3), 255))


def test_viser_batched_instance_colors_preserve_regular_handle_srgb_appearance():
    from soromox.rendering.viser_renderer import (
        _viser_batched_colors_and_opacities,
    )

    colors, opacities = _viser_batched_colors_and_opacities(
        np.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.7, 0.2, 0.5]])
    )

    assert_array_equal(colors, np.array([[201, 3, 3], [3, 114, 8]], dtype=np.uint8))
    assert_allclose(opacities, np.array([1.0, 0.5], dtype=np.float32))


def test_viser_defaults_to_swept_backbone():
    from soromox.rendering.viser_renderer import ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)

    assert renderer._backbone_style == "swept"


def test_viser_ground_plane_uses_native_grid():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(
        robot,
        auto_start=False,
        ground_plane_size=0.4,
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()

    renderer._add_ground_plane()

    assert len(server.scene.grids) == 1
    assert server.scene.grids[0].name == "/ground"
    assert server.scene.grids[0].width == pytest.approx(0.4)
    assert server.scene.grids[0].height == pytest.approx(0.4)
    assert renderer._scene_handles.ground_planes == server.scene.grids


def test_viser_automatic_ground_plane_covers_batched_base_layout():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()

    renderer._add_ground_plane(np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]))

    ground = server.scene.grids[0]
    assert ground.width == pytest.approx(3.35)
    assert ground.height == pytest.approx(3.35)
    assert_allclose(ground.position, [-0.06, 0.0, 0.0])


def test_viser_swept_backbone_uses_material_frame_rings():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(
        robot, auto_start=False, backbone_style="swept", cylinder_sections=8
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    curves = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    material_frames = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()

    renderer._build_robot_geometry(
        curves,
        np.ones((1, 2, 4), dtype=np.float64),
        material_frames=material_frames,
        base_plate_color=(0.15, 0.15, 0.15),
    )

    first_ring = server.scene.simple_meshes[0].vertices[:8]
    assert_allclose(first_ring[:, 0], 0.0, atol=1e-7)
    assert np.ptp(first_ring[:, 1]) > renderer._robot_radius


def test_viser_swept_body_owns_closed_tip_and_updates_atomically():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(
        robot, auto_start=False, backbone_style="swept", cylinder_sections=8
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    frames = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
    colors = np.ones((1, 2, 4), dtype=np.float64)
    initial_curve = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    updated_curve = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])

    renderer._build_robot_geometry(
        initial_curve,
        colors,
        material_frames=frames,
        base_plate_color=(0.15, 0.15, 0.15),
    )
    renderer._update_robot_geometry(updated_curve, frames)

    tip_mesh = server.scene.simple_meshes[0]
    body_tip_ring = tip_mesh.vertices[8:16]
    cap_triangles = tip_mesh.faces[-8:]
    assert server.atomic_calls == 1
    assert_allclose(body_tip_ring.mean(axis=0), updated_curve[0, -1], atol=1e-7)
    assert_allclose(tip_mesh.vertices[-1], updated_curve[0, -1], atol=1e-7)
    assert np.all(cap_triangles[:, 0] == 16)
    assert server.scene.icospheres == []


def test_viser_frame_update_uses_one_atomic_transaction():
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(
        robot,
        auto_start=False,
        backbone_style="swept",
        cylinder_sections=8,
        show_ground_plane=False,
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    q_ts = jnp.zeros((1, 2, 0))
    offsets = jnp.zeros((1, 3))
    curves, frames = renderer.compute_backbone_curves_and_frames_batched(
        q_ts[:, 0], offsets
    )
    colors = renderer.resolve_backbone_colors(1).per_robot_point_rgba
    renderer._build_robot_geometry(
        np.asarray(curves),
        colors,
        material_frames=np.asarray(frames),
        base_plate_color=renderer.color_config.base_plate_color,
    )

    renderer._update_frame(
        1,
        q_ts,
        offsets,
        colors,
        base_plate_color=renderer.color_config.base_plate_color,
        render_actuators=False,
    )

    assert server.atomic_calls == 1


def test_viser_swept_batches_match_segment_geometry_across_animation_frames():
    from soromox.rendering.viser_renderer import (
        SceneHandles,
        ViserRenderer,
        _oriented_tube_segment_mesh,
    )

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(
        robot,
        auto_start=False,
        backbone_style="swept",
        cylinder_sections=8,
    )
    server = FakeViserActuatorServer()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    frames = np.broadcast_to(np.eye(3), (1, 4, 3, 3)).copy()
    colors = np.array(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    curves = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.1, 0.0], [1.2, 0.1, 0.0]],
                [[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [0.8, 0.2, 0.1], [1.2, 0.3, 0.2]],
                [[0.0, 0.0, 0.0], [0.3, -0.1, 0.1], [0.7, -0.2, 0.2], [1.1, -0.3, 0.3]],
            ]
        ]
    )

    renderer._build_robot_geometry(
        curves[:, 0],
        colors,
        material_frames=frames,
        base_plate_color=(0.15, 0.15, 0.15),
    )

    def assert_batches_match(frame_idx):
        if frame_idx:
            renderer._update_robot_geometry(curves[:, frame_idx], frames)
        for batch in renderer._scene_handles.swept_backbone_batches[0]:
            expected_vertices = []
            expected_faces = []
            vertex_offset = 0
            for segment_idx in batch.segment_indices:
                vertices, faces = _oriented_tube_segment_mesh(
                    curves[0, frame_idx, segment_idx],
                    curves[0, frame_idx, segment_idx + 1],
                    frames[0, segment_idx],
                    frames[0, segment_idx + 1],
                    renderer._robot_radius,
                    renderer._cylinder_sections,
                    cap_end=segment_idx == curves.shape[2] - 2,
                )
                expected_vertices.append(vertices)
                expected_faces.append(faces + vertex_offset)
                vertex_offset += len(vertices)
            assert_allclose(batch.handle.vertices, np.concatenate(expected_vertices))
            assert_array_equal(batch.handle.faces, np.concatenate(expected_faces))

    assert len(server.scene.simple_meshes) == 2
    assert_batches_match(0)
    assert_batches_match(1)
    assert_batches_match(2)


def test_viser_default_camera_uses_backend_specific_distance():
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
    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    robot = DummyActuatedSpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    renderer._server = FakeViserActuatorServer()
    renderer._scene_handles = SceneHandles()

    renderer._build_actuator_geometry(
        jnp.zeros((1, 0)),
        np.zeros((1, 3)),
        1,
    )

    first_handles = tuple(renderer._scene_handles.actuator_lines)
    assert len(first_handles) == 2
    assert len(renderer._server.scene.line_segments) == 2

    renderer._build_actuator_geometry(
        jnp.zeros((1, 0)),
        np.array([[1.0, 0.0, 0.0]]),
        1,
        actuator_inputs=jnp.array([[1.0, 2.0]]),
    )

    assert tuple(renderer._scene_handles.actuator_lines) == first_handles
    assert len(renderer._server.scene.line_segments) == 2
    assert all(handle.remove_count == 0 for handle in first_handles)
    assert_allclose(first_handles[0].points[0, 0], np.array([1.0, 0.1, 0.0]))


def test_viser_paused_frame_slider_seeks_rendered_frame():
    from soromox.rendering.viser_renderer import AnimationState, ViserRenderer

    robot = DummySpatialRobot(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    renderer = ViserRenderer(robot, auto_start=False)
    renderer._server = FakeViserPlaybackServer()
    renderer._animation_state = AnimationState(
        frame_idx=0,
        playing=False,
        num_frames=4,
        ts=np.array([0.0, 0.1, 0.2, 0.3]),
    )
    sought_frames = []

    renderer._setup_playback_gui(on_frame_seek=sought_frames.append)
    renderer._gui_handles["frame_slider"].trigger_update(2)

    assert renderer._animation_state.frame_idx == 2
    assert sought_frames == [2]
    assert renderer._gui_handles["time_text"].value == "t = 0.20 s"

    renderer._gui_handles["frame_slider"].trigger_update(99)

    assert renderer._animation_state.frame_idx == 3
    assert sought_frames == [2, 3]
    assert renderer._gui_handles["frame_slider"].value == 3
    assert renderer._gui_handles["time_text"].value == "t = 0.30 s"
