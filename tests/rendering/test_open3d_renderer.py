from unittest.mock import Mock

import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("open3d")

from soromox.rendering import open3d_renderer as open3d_renderer_module  # noqa: E402
from soromox.rendering.camera_config import CameraConfig  # noqa: E402
from soromox.rendering.open3d_renderer import (  # noqa: E402
    Open3DRenderer,
    _cross_section_ring_offsets,
    _make_polyline_lineset,
    _make_polylines_lineset,
    _make_sphere,
    _make_spheres_mesh,
    _make_swept_cross_section_segment,
    _merge_triangle_meshes,
    _refresh_merged_triangle_mesh,
    _swept_segment_vertices,
    _update_polylines_lineset,
)
from soromox.systems.components import CrossSectionGeometry  # noqa: E402


class _AnimatingSpatialRobot:
    is_planar = False
    length = jnp.array(1.0)
    segment_length = jnp.array([1.0])
    base_pose = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    base_transform = jnp.eye(4)

    def forward_kinematics_abscissa_batched(self, q, s_points):
        angles = q[0] * s_points
        cosines = jnp.cos(angles)
        sines = jnp.sin(angles)
        rotations = jnp.stack(
            (
                jnp.stack((cosines, -sines, jnp.zeros_like(angles)), axis=-1),
                jnp.stack((sines, cosines, jnp.zeros_like(angles)), axis=-1),
                jnp.stack(
                    (
                        jnp.zeros_like(angles),
                        jnp.zeros_like(angles),
                        jnp.ones_like(angles),
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        )
        positions = jnp.stack((s_points, q[1] * s_points**2, q[2] * s_points), axis=-1)
        transforms = jnp.broadcast_to(jnp.eye(4), (s_points.size, 4, 4))
        transforms = transforms.at[:, :3, :3].set(rotations)
        return transforms.at[:, :3, 3].set(positions)

    def cross_section_geometry(self, q, s):
        del q, s
        return CrossSectionGeometry.CIRCULAR, jnp.array([0.05])


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


def test_swept_cylinder_rings_follow_material_frame_not_curve_chord():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 1.0, 0.0])
    frame = np.eye(3)

    mesh = _make_swept_cross_section_segment(
        p0,
        p1,
        frame,
        frame,
        CrossSectionGeometry.CIRCULAR,
        np.array([0.2]),
        (1.0, 0.0, 0.0),
        resolution=8,
    )

    first_ring = np.asarray(mesh.vertices)[:8]
    assert_allclose(first_ring[:, 0], 0.0, atol=1e-12)
    assert np.ptp(first_ring[:, 1]) > 0.39


def test_swept_segment_applies_each_endpoint_material_frame():
    offsets = _cross_section_ring_offsets(
        CrossSectionGeometry.RECTANGULAR, np.array([0.4, 0.2]), resolution=8
    )
    frame0 = np.eye(3)
    frame1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    p0 = np.zeros(3)
    p1 = np.array([1.0, 0.0, 0.0])

    vertices = _swept_segment_vertices(p0, p1, frame0, frame1, offsets)

    assert_allclose(vertices[:4], p0 + offsets @ frame0.T, atol=1e-12)
    assert_allclose(vertices[4:], p1 + offsets @ frame1.T, atol=1e-12)


def test_swept_segment_terminal_cap_closes_tip_with_outward_faces():
    p0 = np.zeros(3)
    p1 = np.array([1.0, 0.0, 0.0])
    frame = np.eye(3)

    mesh = _make_swept_cross_section_segment(
        p0,
        p1,
        frame,
        frame,
        CrossSectionGeometry.CIRCULAR,
        np.array([0.2]),
        (1.0, 0.0, 0.0),
        resolution=8,
        cap_end=True,
    )

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    cap_triangles = triangles[-8:]
    assert_allclose(vertices[-1], p1, atol=1e-12)
    assert np.all(cap_triangles[:, 0] == 16)
    first_cap = vertices[cap_triangles[0]]
    normal = np.cross(first_cap[1] - first_cap[0], first_cap[2] - first_cap[0])
    assert np.dot(normal, frame[:, 0]) > 0.0


def test_static_spheres_are_batched_without_changing_geometry_or_colors():
    centers = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]])
    radii = np.array([0.25, 0.5])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    individual = [
        _make_sphere(center, radius, tuple(color), resolution=4)
        for center, radius, color in zip(centers, radii, colors, strict=True)
    ]
    expected = _merge_triangle_meshes(individual)

    batched = _make_spheres_mesh(centers, radii, colors, resolution=4)

    vertices_per_sphere = len(individual[0].vertices)
    vertices = np.asarray(batched.vertices).reshape(2, vertices_per_sphere, 3)
    vertex_colors = np.asarray(batched.vertex_colors).reshape(2, vertices_per_sphere, 3)
    assert len(batched.vertices) == 2 * vertices_per_sphere
    assert_allclose(vertices.mean(axis=1), centers, atol=1e-12)
    assert_allclose(vertex_colors[:, 0], colors)
    assert_allclose(
        np.asarray(batched.vertices), np.asarray(expected.vertices), atol=1e-15
    )
    assert_allclose(np.asarray(batched.triangles), np.asarray(expected.triangles))
    assert_allclose(
        np.asarray(batched.vertex_normals), np.asarray(expected.vertex_normals)
    )
    assert_allclose(
        np.asarray(batched.vertex_colors), np.asarray(expected.vertex_colors)
    )


def test_batched_actuator_lines_match_individual_lines_across_animation_frames():
    trajectories = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [1.0, 0.2, 0.0]],
                [[0.0, 0.2, 0.0], [0.5, 0.3, 0.1], [1.0, 0.4, 0.2]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.4, -0.1, 0.0], [0.8, -0.2, 0.1]],
                [[0.0, 0.3, 0.0], [0.4, 0.2, 0.2], [0.8, 0.1, 0.4]],
            ],
        ]
    )
    frame_colors = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.5, 0.0]],
        ]
    )
    batched = _make_polylines_lineset(trajectories[:, 0], frame_colors[:, 0])

    for frame_idx in range(2):
        if frame_idx:
            _update_polylines_lineset(
                batched,
                trajectories[:, frame_idx],
                frame_colors[:, frame_idx],
            )
        individual = [
            _make_polyline_lineset(points, tuple(color))
            for points, color in zip(
                trajectories[:, frame_idx], frame_colors[:, frame_idx], strict=True
            )
        ]
        point_count = trajectories.shape[2]
        expected_points = np.concatenate(
            [np.asarray(line_set.points) for line_set in individual]
        )
        expected_lines = np.concatenate(
            [
                np.asarray(line_set.lines) + line_idx * point_count
                for line_idx, line_set in enumerate(individual)
            ]
        )
        expected_colors = np.concatenate(
            [np.asarray(line_set.colors) for line_set in individual]
        )
        assert_allclose(np.asarray(batched.points), expected_points)
        assert_allclose(np.asarray(batched.lines), expected_lines)
        assert_allclose(np.asarray(batched.colors), expected_colors)


def test_batched_actuator_lines_accept_read_only_jax_arrays():
    polylines = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [1.0, 0.2, 0.0]],
            [[0.0, 0.2, 0.0], [0.5, 0.3, 0.1], [1.0, 0.4, 0.2]],
        ]
    )
    colors = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    line_set = _make_polylines_lineset(polylines, colors)
    _update_polylines_lineset(line_set, polylines + 0.1, colors)

    assert_allclose(
        np.asarray(line_set.points), np.asarray(polylines + 0.1).reshape(-1, 3)
    )


def test_dynamic_sphere_batch_matches_individual_meshes_at_every_frame():
    class FakeVisualizer:
        def __init__(self):
            self.added = []
            self.updated = []

        def add_geometry(self, geometry):
            self.added.append(geometry)

        def update_geometry(self, geometry):
            self.updated.append(geometry)

    renderer = Open3DRenderer(
        _AnimatingSpatialRobot(),
        num_points=4,
        sphere_resolution=3,
        show_ground_plane=False,
    )
    renderer._warned_dynamic_geometry = True
    trajectories = np.array(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.2, 0.1]],
            [[1.0, 0.0, 0.0], [0.9, -0.1, 0.2], [0.8, -0.2, 0.4]],
        ]
    )
    radii = np.array([0.1, 0.2])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scene = renderer._prepare_scene_data(
        np.array([0.0, 0.5, 1.0]),
        np.zeros((1, 3, 3)),
        base_offsets=np.zeros((1, 3)),
        color_config=None,
        render_actuators=False,
        actuator_inputs=None,
        static_spheres_positions=None,
        static_spheres_radii=None,
        static_spheres_colors=None,
        dynamic_spheres_positions=trajectories,
        dynamic_spheres_radii=radii,
        dynamics_spheres_colors=colors,
    )
    visualizer = FakeVisualizer()
    handles = renderer._build_scene(visualizer, scene)
    assert handles.dynamic_sphere_batch is not None

    for frame_idx in range(3):
        if frame_idx:
            renderer._update_scene(visualizer, scene, handles, frame_idx)
        expected = _merge_triangle_meshes(
            [
                _make_sphere(center, radius, tuple(color), resolution=12)
                for center, radius, color in zip(
                    trajectories[:, frame_idx], radii, colors, strict=True
                )
            ]
        )
        actual = handles.dynamic_sphere_batch.mesh
        assert_allclose(
            np.asarray(actual.vertices), np.asarray(expected.vertices), atol=1e-15
        )
        assert_allclose(np.asarray(actual.triangles), np.asarray(expected.triangles))
        assert_allclose(
            np.asarray(actual.vertex_normals),
            np.asarray(expected.vertex_normals),
            atol=1e-15,
        )
        assert_allclose(
            np.asarray(actual.vertex_colors), np.asarray(expected.vertex_colors)
        )


def test_merged_dynamic_mesh_refreshes_vertices_and_normals():
    first = _make_sphere((0.0, 0.0, 0.0), 0.25, resolution=4)
    second = _make_sphere((1.0, 0.0, 0.0), 0.5, resolution=4)
    merged = _merge_triangle_meshes([first, second])
    first_vertex_count = len(first.vertices)

    first.translate((0.5, 0.25, -0.1))
    _refresh_merged_triangle_mesh(merged, [first, second])

    assert len(merged.vertices) == len(first.vertices) + len(second.vertices)
    assert_allclose(
        np.asarray(merged.vertices)[:first_vertex_count], np.asarray(first.vertices)
    )
    assert_allclose(
        np.asarray(merged.vertex_normals)[:first_vertex_count],
        np.asarray(first.vertex_normals),
    )


def test_merged_backbone_matches_unmerged_geometry_across_animation_frames():
    class FakeVisualizer:
        def __init__(self):
            self.added = []
            self.updated = []

        def add_geometry(self, geometry):
            self.added.append(geometry)

        def update_geometry(self, geometry):
            self.updated.append(geometry)

    renderer = Open3DRenderer(
        _AnimatingSpatialRobot(),
        width=64,
        height=64,
        num_points=6,
        sphere_resolution=3,
        recompute_normals=False,
        show_ground_plane=False,
    )
    renderer._warned_dynamic_geometry = True
    q_ts = np.zeros((2, 2, 3), dtype=np.float64)
    q_ts[0, 1] = (0.15, -0.1, 0.05)
    q_ts[1, 1] = (-0.05, 0.2, -0.1)
    offsets = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
    scene = renderer._prepare_scene_data(
        np.array([0.0, 1.0]),
        q_ts,
        base_offsets=offsets,
        color_config=None,
        render_actuators=False,
        actuator_inputs=None,
        static_spheres_positions=None,
        static_spheres_radii=None,
        static_spheres_colors=None,
        dynamic_spheres_positions=None,
        dynamic_spheres_radii=None,
        dynamics_spheres_colors=None,
    )

    renderer.merge_backbone_meshes = False
    legacy_visualizer = FakeVisualizer()
    legacy_handles = renderer._build_scene(legacy_visualizer, scene, frame_idx=0)
    renderer.merge_backbone_meshes = True
    merged_visualizer = FakeVisualizer()
    merged_handles = renderer._build_scene(merged_visualizer, scene, frame_idx=0)

    def combined_arrays(handles, robot_idx):
        meshes = [
            cached.mesh
            for group in handles.backbone_meshes[robot_idx]
            for cached in group
        ]
        offsets = np.cumsum([0, *[len(mesh.vertices) for mesh in meshes[:-1]]])
        return {
            "vertices": np.concatenate([np.asarray(mesh.vertices) for mesh in meshes]),
            "triangles": np.concatenate(
                [
                    np.asarray(mesh.triangles) + offset
                    for mesh, offset in zip(meshes, offsets, strict=True)
                ]
            ),
            "normals": np.concatenate(
                [np.asarray(mesh.vertex_normals) for mesh in meshes]
            ),
            "colors": np.concatenate(
                [np.asarray(mesh.vertex_colors) for mesh in meshes]
            ),
        }

    def assert_scenes_match(frame_idx):
        if frame_idx > 0:
            renderer._update_scene(
                legacy_visualizer,
                scene,
                legacy_handles,
                frame_idx,
            )
            renderer._update_scene(
                merged_visualizer,
                scene,
                merged_handles,
                frame_idx,
            )

        for robot_idx, merged in enumerate(merged_handles.merged_backbone_meshes):
            assert merged is not None
            expected = combined_arrays(legacy_handles, robot_idx)
            assert_allclose(np.asarray(merged.vertices), expected["vertices"])
            assert_allclose(np.asarray(merged.triangles), expected["triangles"])
            assert_allclose(np.asarray(merged.vertex_normals), expected["normals"])
            assert_allclose(np.asarray(merged.vertex_colors), expected["colors"])

    assert_scenes_match(0)
    assert_scenes_match(1)


def test_open3d_backbone_merging_defaults_to_multi_robot_scenes():
    renderer = Open3DRenderer(_AnimatingSpatialRobot(), show_ground_plane=False)

    assert renderer.merge_backbone_meshes is None
    assert renderer._should_merge_backbone_meshes(1) is False
    assert renderer._should_merge_backbone_meshes(2) is True

    renderer.merge_backbone_meshes = False
    assert renderer._should_merge_backbone_meshes(64) is False
    renderer.merge_backbone_meshes = True
    assert renderer._should_merge_backbone_meshes(1) is True


def test_open3d_interactive_camera_front_points_from_eye_to_target():
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import Open3DRenderer

    robot = _AnimatingSpatialRobot()
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
        camera_config=CameraConfig(
            distance_factor=1.0, position_offset=(1.0, 0.0, 0.0)
        ),
    )

    assert_allclose(ctrl.front, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert_allclose(ctrl.lookat, np.array([0.0, 0.5, 0.0]), atol=1e-12)
    assert_allclose(ctrl.up, np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert vis.reset_view_point_arg is True
    assert vis.polled
    assert vis.updated


def test_open3d_visualizer_reports_window_creation_failure(monkeypatch):
    renderer = Open3DRenderer(_AnimatingSpatialRobot())
    visualizer = Mock()
    visualizer.create_window.return_value = False
    visualizer_factory = Mock(return_value=visualizer)
    monkeypatch.setattr(
        open3d_renderer_module.o3d.visualization,
        "VisualizerWithKeyCallback",
        visualizer_factory,
    )

    with pytest.raises(RuntimeError, match="failed to create the visualization window"):
        renderer._create_visualizer("test")

    visualizer.get_render_option.assert_not_called()


def test_open3d_defaults_to_swept_backbone():
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import Open3DRenderer

    robot = _AnimatingSpatialRobot()
    renderer = Open3DRenderer(robot)

    assert renderer._backbone_mode == "swept"


def test_open3d_ground_plane_has_surface_and_grid():
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import _make_ground_plane

    center = np.array([0.2, -0.1, 0.3])
    normal = np.array([1.0, 2.0, 3.0])
    size = 0.4
    grid_divisions = 4
    plane, grid = _make_ground_plane(
        center_xyz=center,
        normal_xyz=normal,
        size=size,
        plane_color=(0.9, 0.9, 0.9),
        grid_color=(0.7, 0.7, 0.7),
        grid_divisions=grid_divisions,
    )

    plane_vertices = np.asarray(plane.vertices)
    grid_points = np.asarray(grid.points)
    grid_lines = np.asarray(grid.lines)
    unit_normal = normal / np.linalg.norm(normal)
    grid_offset = 2e-4 * max(size, 1.0)

    assert plane_vertices.shape == (4, 3)
    assert np.asarray(plane.triangles).shape == (2, 3)
    assert grid_points.shape == (4 * (grid_divisions + 1), 3)
    assert grid_lines.shape == (2 * (grid_divisions + 1), 2)
    assert_array_equal(
        grid_lines,
        np.arange(grid_points.shape[0], dtype=np.int32).reshape(-1, 2),
    )
    assert_allclose((plane_vertices - center) @ unit_normal, 0.0, atol=1e-12)
    assert_allclose(
        (grid_points - center) @ unit_normal,
        grid_offset,
        atol=1e-12,
    )


def test_open3d_closes_window_when_recording_finishes(monkeypatch):
    pytest.importorskip("open3d")
    from soromox.rendering.open3d_renderer import Open3DRenderer, RecordingConfig

    robot = _AnimatingSpatialRobot()
    renderer = Open3DRenderer(robot)
    vis = Mock()
    vis.poll_events.side_effect = [True, True, True]
    ctrl = Mock()
    ctrl.convert_to_pinhole_camera_parameters.return_value = Mock()
    scene_data = type(
        "SceneDataStub",
        (),
        {"ts": np.array([0.0, 1.0]), "num_frames": 2},
    )()

    monkeypatch.setattr(renderer, "_create_visualizer", Mock(return_value=(vis, ctrl)))
    monkeypatch.setattr(renderer, "_build_scene", Mock(return_value=Mock()))
    monkeypatch.setattr(renderer, "_setup_interactive_camera", Mock())
    monkeypatch.setattr(
        renderer, "_frame_intervals_from_ts", Mock(return_value=np.zeros(2))
    )
    monkeypatch.setattr(
        renderer,
        "_init_recorder",
        Mock(return_value=(None, None, "rollout.mp4")),
    )
    monkeypatch.setattr(renderer, "_register_key_callbacks", Mock())
    update_scene = Mock()
    monkeypatch.setattr(renderer, "_update_scene", update_scene)

    renderer._run_viewer(
        scene_data,
        playback_speed=1.0,
        autoplay=True,
        loop=False,
        record_cfg=RecordingConfig(
            path="rollout.mp4",
            close_when_done=True,
        ),
        window_name="test",
    )

    assert update_scene.call_count == 2
    vis.destroy_window.assert_called_once_with()


def test_open3d_quit_callback_requests_close_without_destroying_window():
    renderer = Open3DRenderer(_AnimatingSpatialRobot())
    callbacks: dict[int, object] = {}
    vis = Mock()
    vis.register_key_callback.side_effect = callbacks.__setitem__
    state = {"idx": 0, "playing": True, "last_tick": 0.0, "dt_seq": np.zeros(1)}

    renderer._register_key_callbacks(
        vis=vis,
        ctrl=Mock(),
        state=state,
        update_frame=Mock(),
        save_frame_fn=Mock(),
        initial_cam=Mock(),
        saved_cam_holder=[Mock()],
        print_prefix="[Open3D]",
    )

    for key in (81, 256):  # Q, Esc
        assert callbacks[key](vis) is False

    assert state["playing"] is False
    assert vis.close.call_count == 2
    vis.destroy_window.assert_not_called()
