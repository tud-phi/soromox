# ruff: noqa: E402
from __future__ import annotations

from contextlib import contextmanager

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("viser")

from soromox.rendering import ISupportViserRenderer, ISupportVisualConfig
from soromox.rendering.color_config import validate_rgb
from soromox.rendering.isupport.viser_renderer import ISupportLiveModeController
from soromox.rendering.viser_renderer import SceneHandles
from soromox.systems import ISupport, ISupportParams, ISupportStructure


class _FakeHandle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.remove_count = 0

    def remove(self):
        self.remove_count += 1


class _FakeScene:
    def __init__(self):
        self.simple_meshes = []
        self.labels = []

    def add_mesh_simple(self, **kwargs):
        handle = _FakeHandle(**kwargs)
        self.simple_meshes.append(handle)
        return handle

    def add_label(self, **kwargs):
        handle = _FakeHandle(**kwargs)
        self.labels.append(handle)
        return handle


class _FakeGuiHandle(_FakeHandle):
    def on_update(self, callback):
        self.update_callback = callback
        return callback

    def update(self, value):
        self.value = value
        self.update_callback(type("Event", (), {"target": self})())


class _FakeGui:
    def __init__(self):
        self.images = []

    @contextmanager
    def add_folder(self, _name):
        yield

    def add_checkbox(self, _name, *, initial_value):
        return _FakeGuiHandle(value=initial_value)

    def add_text(self, _name, *, initial_value, disabled):
        return _FakeGuiHandle(value=initial_value, disabled=disabled)

    def add_image(self, image, *, format):
        handle = _FakeGuiHandle(image=image, format=format)
        self.images.append(handle)
        return handle


class _FakeServer:
    def __init__(self):
        self.scene = _FakeScene()
        self.gui = _FakeGui()
        self.atomic_calls = 0

    @contextmanager
    def atomic(self):
        self.atomic_calls += 1
        yield


def _make_robot(*, connectors: bool = True) -> ISupport:
    lengths = jnp.array([0.10, 0.12])
    params = ISupportParams(
        base_pose=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        length=lengths,
        radius=jnp.array([0.03, 0.03]),
        density=jnp.array([1000.0, 1000.0]),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=jnp.array([2.0e3, 2.0e3]),
        shear_modulus=jnp.array([1.0e3, 1.0e3]),
        damping_matrix=1.0e-3 * jnp.eye(12),
        reference_strain=jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 2),
        chamber_inner_radius=jnp.array([0.005, 0.005]),
        chamber_outer_radius=jnp.array([0.00779, 0.00779]),
        chamber_distance=jnp.array([0.020, 0.020]),
        chamber_azimuth_angles=jnp.stack(
            [2 * jnp.pi * jnp.arange(3) / 3, 0.15 + 2 * jnp.pi * jnp.arange(3) / 3]
        ),
        pcs_segment_lengths=jnp.array([0.04, 0.06, 0.12]),
        rigid_connector_lengths=(
            jnp.array([0.01, 0.02, 0.03]) if connectors else jnp.zeros((3,))
        ),
    )
    return ISupport(
        params=params,
        structure=ISupportStructure(
            num_gauss_points=1,
            pcs_segment_counts=(2, 1),
            rigid_connector_selector=(connectors, connectors, connectors),
        ),
    )


def _renderer(robot: ISupport, **kwargs) -> ISupportViserRenderer:
    return ISupportViserRenderer(
        robot,
        auto_start=False,
        open_browser=False,
        cylinder_sections=12,
        **kwargs,
    )


def _curves_and_frames(renderer, q, offsets=None):
    q = jnp.asarray(q)
    if q.ndim == 1:
        q = q[None, :]
    if offsets is None:
        offsets = jnp.zeros((q.shape[0], 3))
    curves, frames = renderer.compute_backbone_curves_and_frames_batched(q, offsets)
    return np.asarray(curves), np.asarray(frames)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"bellows_amplitude": 1.0}, "bellows_amplitude"),
        ({"chamber_sections": 5}, "chamber_sections"),
        ({"samples_per_fold": 1}, "samples_per_fold"),
        ({"spacer_opacity": 1.1}, "spacer_opacity"),
        ({"interface_radius": 0.0}, "interface_radius"),
        ({"pressure_range": (1.0, 1.0)}, "pressure_range"),
        ({"pressure_colormap_start": 1.0}, "pressure_colormap_start"),
        ({"pressure_label_offset": -0.01}, "pressure_label_offset"),
        ({"pressure_colormap": "not-a-colormap"}, "colormap"),
    ],
)
def test_visual_config_validation(updates, message):
    with pytest.raises(ValueError, match=message):
        ISupportVisualConfig(**updates)


@pytest.mark.parametrize(
    "color",
    [(-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5), (0.5, np.nan, 0.5)],
)
def test_shared_rgb_validation_rejects_invalid_colors(color):
    with pytest.raises(ValueError, match="test_color"):
        validate_rgb(color, name="test_color")


def test_renderer_rejects_non_isupport_robot():
    with pytest.raises(TypeError, match="requires an ISupport"):
        ISupportViserRenderer(object(), auto_start=False)


def test_renderer_rejects_invalid_visual_config_before_starting_server():
    with pytest.raises(TypeError, match="visual_config"):
        ISupportViserRenderer(
            _make_robot(connectors=False),
            visual_config=object(),
            auto_start=True,
        )


def test_layout_tracks_split_pneumatic_segments_and_connectors():
    renderer = _renderer(_make_robot(connectors=True))

    assert len(renderer._pneumatic_specs) == 2
    assert_allclose(
        [(spec.s_start, spec.s_end) for spec in renderer._pneumatic_specs],
        [(0.01, 0.11), (0.13, 0.25)],
        atol=1e-12,
    )
    assert [spec.thickness for spec in renderer._interface_specs] == pytest.approx(
        [0.01, 0.02, 0.03]
    )
    assert all(spec.modeled for spec in renderer._interface_specs)

    first_spacers = renderer._pneumatic_specs[0].spacer_sample_indices
    assert_allclose(
        renderer._sample_s[list(first_spacers)],
        0.01 + 0.10 * np.arange(1, 7) / 7.0,
        atol=1e-12,
    )


def test_layout_adds_visual_only_interfaces_without_connectors():
    config = ISupportVisualConfig(fallback_interface_thickness=0.004)
    renderer = _renderer(
        _make_robot(connectors=False),
        visual_config=config,
    )

    assert len(renderer._interface_specs) == 3
    assert not any(spec.modeled for spec in renderer._interface_specs)
    assert [spec.thickness for spec in renderer._interface_specs] == pytest.approx(
        [0.004, 0.004, 0.004]
    )


def test_chambers_follow_model_angles_ellipse_and_bellows():
    renderer = _renderer(_make_robot(connectors=True))
    q = jnp.zeros((renderer.robot.num_dofs,))
    curves, _ = _curves_and_frames(renderer, q)
    poses = renderer._sample_world_poses(q[None, :], curves)[0]
    spec = renderer._pneumatic_specs[0]
    sections = renderer.visual_config.chamber_sections

    chamber_centers = []
    for chamber_idx in range(3):
        vertices = renderer._chamber_vertices(poses, spec, chamber_idx)
        chamber_centers.append(vertices[:sections].mean(axis=0))
    chamber_centers = np.asarray(chamber_centers)
    backbone_center = poses[spec.ring_slice.start, :3, 3]
    offsets = chamber_centers - backbone_center
    assert_allclose(np.linalg.norm(offsets, axis=1), 0.020, atol=1e-7)
    unit_offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    assert_allclose(unit_offsets @ unit_offsets.T, 1.5 * np.eye(3) - 0.5, atol=1e-6)

    vertices = renderer._chamber_vertices(poses, spec, 0)
    first_ring = vertices[:sections]
    center = first_ring.mean(axis=0)
    rotation = poses[spec.ring_slice.start, :3, :3]
    local_offset = np.asarray(renderer.robot.local_chamber_offsets(0)[0])
    radial_local = local_offset / np.linalg.norm(local_offset)
    radial = rotation @ radial_local
    tangential = rotation @ np.cross(np.array([1.0, 0.0, 0.0]), radial_local)
    radial_extent = np.max(np.abs((first_ring - center) @ radial))
    tangential_extent = np.max(np.abs((first_ring - center) @ tangential))
    assert radial_extent / tangential_extent == pytest.approx(
        renderer.visual_config.chamber_aspect_ratio, rel=0.02
    )

    bulge_ring_idx = renderer.visual_config.samples_per_fold // 2
    bulge_ring = vertices[bulge_ring_idx * sections : (bulge_ring_idx + 1) * sections]
    waist_radius = np.max(np.linalg.norm(first_ring - center, axis=1))
    bulge_radius = np.max(np.linalg.norm(bulge_ring - bulge_ring.mean(axis=0), axis=1))
    assert bulge_radius > 1.2 * waist_radius


def test_pressure_colormap_clamps_and_handles_missing_values():
    renderer = _renderer(_make_robot(connectors=True))
    config = renderer.visual_config

    from matplotlib import colormaps

    cmap = colormaps[config.pressure_colormap]
    expected_low = tuple(
        int(round(255.0 * component))
        for component in cmap(config.pressure_colormap_start)[:3]
    )
    expected_mid = tuple(
        int(round(255.0 * component))
        for component in cmap(
            config.pressure_colormap_start
            + (1.0 - config.pressure_colormap_start) * 0.5
        )[:3]
    )
    expected_high = tuple(int(round(255.0 * component)) for component in cmap(1.0)[:3])

    assert renderer._pressure_color(-1.0e5) == expected_low
    assert renderer._pressure_color(0.0) == expected_low
    assert renderer._pressure_color(1.5e5) == expected_mid
    assert renderer._pressure_color(3.0e5) == expected_high
    assert renderer._pressure_color(5.0e5) == expected_high
    assert renderer._pressure_color(np.nan) == (145, 145, 145)


def test_pressure_shape_normalization():
    renderer = _renderer(_make_robot(connectors=True))
    pressures = np.arange(renderer.robot.num_actuators, dtype=float)

    static = renderer._static_pressures(pressures, num_robots=2)
    assert static.shape == (2, renderer.robot.num_actuators)
    assert_allclose(static[0], pressures)
    sequence = renderer._sequence_pressures(pressures, num_robots=2, num_frames=4)
    assert sequence.shape == (2, 4, renderer.robot.num_actuators)
    assert_allclose(sequence[1, 3], pressures)

    with pytest.raises(ValueError, match="pressures must have shape"):
        renderer._static_pressures(np.zeros((2, 2, 2)), num_robots=2)
    with pytest.raises(ValueError, match="sequence pressures"):
        renderer._sequence_pressures(np.zeros((2, 2)), num_robots=2, num_frames=4)


def test_build_and_update_preserve_custom_handle_identity():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    q = jnp.zeros((renderer.robot.num_dofs,))
    curves, frames = _curves_and_frames(renderer, q)
    renderer._current_geometry_q = q[None, :]

    renderer._build_robot_geometry(
        curves,
        np.ones((1, renderer.num_points, 4)),
        material_frames=frames,
        base_plate_color=(0.1, 0.1, 0.1),
    )

    handles = renderer._robot_visual_handles[0]
    assert len(handles.chamber_handles) == 6
    assert len(handles.spacer_handles) == 12
    assert len(handles.interface_handles) == 3
    assert len(renderer._server.scene.simple_meshes) == 21
    first_handle = handles.chamber_handles[0]
    first_vertices = first_handle.vertices.copy()

    q_next = q.at[1].set(2.0)
    curves_next, frames_next = _curves_and_frames(renderer, q_next)
    renderer._current_geometry_q = q_next[None, :]
    renderer._update_robot_geometry(curves_next, frames_next)

    assert renderer._robot_visual_handles[0].chamber_handles[0] is first_handle
    assert renderer._server.atomic_calls == 1
    assert not np.allclose(first_handle.vertices, first_vertices)


def test_pressure_labels_and_colors_update_in_place():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    q = jnp.zeros((renderer.robot.num_dofs,))
    curves, frames = _curves_and_frames(renderer, q)
    renderer._current_geometry_q = q[None, :]
    renderer._current_pressures = np.array([[2.0e4, 1.5e5, 3.0e5, np.nan, 0.0, 4.0e5]])

    renderer._build_robot_geometry(
        curves,
        np.ones((1, renderer.num_points, 4)),
        material_frames=frames,
        base_plate_color=(0.1, 0.1, 0.1),
    )

    robot_handles = renderer._robot_visual_handles[0]
    assert robot_handles.pressure_label_handles == []
    assert renderer._server.scene.labels == []
    assert robot_handles.chamber_handles[3].color == (145, 145, 145)
    pressure_color = robot_handles.chamber_handles[0].color

    renderer._setup_pressure_gui()
    checkbox = renderer._gui_handles["show_pressure_labels"]
    colorbar = renderer._gui_handles["pressure_colorbar"]
    assert checkbox.value is False
    assert colorbar.format == "png"
    assert colorbar.image.ndim == 3
    assert colorbar.image.shape[-1] == 4
    checkbox.update(True)
    labels = robot_handles.pressure_label_handles
    assert len(labels) == 6
    assert labels[0].text == "Segment 1 · Chamber 1\n20.0 kPa"
    assert labels[3].text == "Segment 2 · Chamber 1\n--"
    assert all(label.visible for label in labels)
    assert robot_handles.chamber_handles[0].color == pressure_color
    first_label = labels[0]
    first_position = np.asarray(first_label.position)

    q_next = q.at[1].set(2.0)
    curves_next, frames_next = _curves_and_frames(renderer, q_next)
    renderer._current_geometry_q = q_next[None, :]
    renderer._update_robot_geometry(curves_next, frames_next)

    assert renderer._robot_visual_handles[0].pressure_label_handles[0] is first_label
    assert not np.allclose(first_label.position, first_position)
    assert first_label.visible

    checkbox.update(False)
    assert robot_handles.pressure_label_handles == []
    assert first_label.remove_count == 1


def test_geometry_without_pressures_uses_default_chamber_style():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    q = jnp.zeros((renderer.robot.num_dofs,))
    curves, frames = _curves_and_frames(renderer, q)
    renderer._current_geometry_q = q[None, :]

    renderer._build_robot_geometry(
        curves,
        np.ones((1, renderer.num_points, 4)),
        material_frames=frames,
        base_plate_color=(0.1, 0.1, 0.1),
    )

    robot_handles = renderer._robot_visual_handles[0]
    assert robot_handles.pressure_label_handles == []
    assert robot_handles.chamber_handles[0].color == renderer._color_tuple(
        renderer.visual_config.chamber_color
    )


def test_render_frame_forces_labels_only_when_pressures_are_supplied(monkeypatch):
    renderer = _renderer(_make_robot(connectors=True))
    q = jnp.zeros((renderer.robot.num_dofs,))
    captured = []

    def fake_render_frame(self, q, **kwargs):
        captured.append((self._current_pressures, self._force_pressure_labels, kwargs))
        return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "soromox.rendering.viser_renderer.ViserRenderer.render_frame",
        fake_render_frame,
    )
    renderer.render_frame(q)
    renderer.render_frame(q, pressures=np.full(renderer.robot.num_actuators, 2.0e4))

    assert captured[0][0] is None
    assert captured[0][1] is False
    assert captured[1][0].shape == (1, renderer.robot.num_actuators)
    assert captured[1][1] is True


def test_batched_geometry_applies_base_offsets():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    q = jnp.zeros((2, renderer.robot.num_dofs))
    offsets = jnp.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
    curves, frames = _curves_and_frames(renderer, q, offsets)
    renderer._current_geometry_q = q

    renderer._build_robot_geometry(
        curves,
        np.ones((2, renderer.num_points, 4)),
        material_frames=frames,
        base_plate_color=(0.1, 0.1, 0.1),
    )

    vertices_0 = renderer._robot_visual_handles[0].chamber_handles[0].vertices
    vertices_1 = renderer._robot_visual_handles[1].chamber_handles[0].vertices
    assert_allclose(
        vertices_1 - vertices_0,
        np.broadcast_to(np.asarray(offsets[1]), vertices_0.shape),
        atol=1e-7,
    )


def test_live_controller_updates_custom_geometry_in_place():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    controller = ISupportLiveModeController(renderer)
    q = np.zeros((renderer.robot.num_dofs,))

    controller._process_state(q)
    first_handle = renderer._robot_visual_handles[0].chamber_handles[0]
    controller._process_state(q + 0.01)

    assert renderer._current_geometry_q.shape == (1, renderer.robot.num_dofs)
    assert renderer._robot_visual_handles[0].chamber_handles[0] is first_handle
    assert renderer._server.atomic_calls == 1


def test_live_controller_pushes_pressures_with_state():
    renderer = _renderer(_make_robot(connectors=True))
    renderer._server = _FakeServer()
    renderer._scene_handles = SceneHandles()
    controller = ISupportLiveModeController(renderer)
    controller.start()
    q = np.zeros((renderer.robot.num_dofs,))
    pressures = np.arange(renderer.robot.num_actuators) * 1.0e5

    controller.push_state_with_pressures(q, pressures)
    assert renderer._robot_visual_handles[0].pressure_label_handles == []
    renderer._set_pressure_labels_visible(True)
    controller.stop()

    assert_allclose(renderer._current_pressures[0], pressures)
    labels = renderer._robot_visual_handles[0].pressure_label_handles
    assert labels[4].text == "Segment 2 · Chamber 2\n400.0 kPa"


def test_sequence_frame_selects_matching_pressures(monkeypatch):
    renderer = _renderer(_make_robot(connectors=True))
    q_ts = jnp.zeros((3, renderer.robot.num_dofs))
    pressure_ts = np.arange(18, dtype=float).reshape(3, 6) * 1.0e4

    def fake_render_sequence(self, ts, q_ts, **kwargs):
        return None

    def fake_update_frame(self, frame_idx, *args, **kwargs):
        return None

    monkeypatch.setattr(
        "soromox.rendering.viser_renderer.ViserRenderer.render_sequence",
        fake_render_sequence,
    )
    monkeypatch.setattr(
        "soromox.rendering.viser_renderer.ViserRenderer._update_frame",
        fake_update_frame,
    )
    renderer.render_sequence(np.arange(3), q_ts, pressures=pressure_ts)
    renderer._update_frame(2, q_ts[None, ...])

    assert_allclose(renderer._current_pressures[0], pressure_ts[2])
    assert renderer._pressure_frame_idx == 2
