"""Helix geometry, material-frame motion, and Viser handle lifecycle."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs.simulate_helicoidal import build_robot
from soromox.rendering import (
    HelicoidalModuleVisual,
    HelicoidalViserRenderer,
    HelicoidalVisualConfig,
)
from soromox.rendering.viser_renderer import SceneHandles

jax.config.update("jax_enable_x64", True)


class Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.removed = False

    def remove(self):
        self.removed = True


class Server:
    def __init__(self):
        self.scene = self
        self.meshes = []
        self.atomic_calls = 0

    def add_mesh_simple(self, **kwargs):
        handle = Handle(**kwargs)
        self.meshes.append(handle)
        return handle

    @contextmanager
    def atomic(self):
        self.atomic_calls += 1
        yield


@pytest.fixture(scope="module")
def robot():
    return build_robot(num_gauss_points=5)


def renderer_for(robot):
    return HelicoidalViserRenderer(
        robot,
        modules=tuple(HelicoidalModuleVisual(i) for i in (1, 4, 7)),
        auto_start=False,
        open_browser=False,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"turns": 0},
        {"turns": np.nan},
        {"strand_radius": -1},
        {"strands_per_family": 1.5},
        {"segment_index": -1},
        {"phase": np.inf},
    ],
)
def test_invalid_module(kwargs):
    with pytest.raises(ValueError):
        HelicoidalModuleVisual(**({"segment_index": 0} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples_per_turn": 3},
        {"strand_sections": 2},
        {"samples_per_turn": 8.5},
        {"positive_color": (1.1, 0, 0)},
    ],
)
def test_invalid_visual_config(kwargs):
    with pytest.raises(ValueError):
        HelicoidalVisualConfig(**kwargs)


def test_module_selection_validation(robot):
    for modules in ((), (HelicoidalModuleVisual(8),), (HelicoidalModuleVisual(1),) * 2):
        with pytest.raises(ValueError):
            HelicoidalViserRenderer(robot, modules=modules, auto_start=False)


def test_helix_radius_turns_tube_thickness_and_mesh_indices(robot):
    renderer = renderer_for(robot)
    q = jnp.zeros((1, robot.num_coordinates))
    curves, frames = renderer.compute_backbone_curves_and_frames_batched(
        q, jnp.zeros((1, 3))
    )
    curve, frame = np.asarray(curves[0]), np.asarray(frames[0])
    meshes = list(renderer._meshes(curve, frame))
    assert len(meshes) == 3 * (6 + 2) + 5
    starts, ends = renderer._segment_bounds(renderer.num_points)
    for module in renderer.modules:
        start, end = starts[module.segment_index], ends[module.segment_index]
        assert end - start >= 1 + 3 * renderer.visual_config.samples_per_turn
        name = f"link_{module.segment_index}/family_-1/strand_0"
        vertices = next(mesh[1] for mesh in meshes if mesh[0] == name)
        rings = vertices[:-2].reshape(
            end - start, renderer.visual_config.strand_sections, 3
        )
        centers = rings.mean(axis=1, dtype=np.float64)
        offset = centers - curve[start:end]
        local = np.einsum("nji,nj->ni", frame[start:end], offset)
        assert_allclose(np.linalg.norm(local[:, 1:], axis=1), 0.025, atol=3e-8)
        theta = np.unwrap(np.arctan2(local[:, 2], local[:, 1]))
        assert_allclose(theta[-1] - theta[0], -6 * np.pi, atol=2e-5)
        assert_allclose(
            np.linalg.norm(rings - centers[:, None], axis=-1), 0.002, atol=5e-8
        )
    for _, vertices, faces, _ in meshes:
        assert np.isfinite(vertices).all()
        assert faces.min() >= 0 and faces.max() < len(vertices)
        triangles = vertices[faces]
        assert np.all(
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            )
            > 0
        )


def test_batched_deformation_updates_handles_and_rebuild_removes_old_meshes(robot):
    renderer = renderer_for(robot)
    renderer._server = Server()
    renderer._scene_handles = SceneHandles()
    renderer._add_ground_plane = lambda *args: None
    q = jnp.zeros((2, robot.num_coordinates))
    offsets = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.1, -0.2]])
    curves, frames = map(
        np.asarray, renderer.compute_backbone_curves_and_frames_batched(q, offsets)
    )
    colors = np.ones((*curves.shape[:2], 4))
    renderer._build_robot_geometry(
        curves, colors, material_frames=frames, base_plate_color=(0.5, 0.5, 0.5)
    )
    handles = renderer._scene_handles.backbone_points
    old_vertices = [h.vertices.copy() for h in handles[0]]
    for first, second in zip(handles[0], handles[1]):
        assert_allclose(
            second.vertices - first.vertices,
            np.broadcast_to(offsets[1], first.vertices.shape),
            atol=1e-7,
        )
    q_next = q.at[:, 7].set(1.2).at[:, 6].set(0.4)
    curves_next, frames_next = map(
        np.asarray, renderer.compute_backbone_curves_and_frames_batched(q_next, offsets)
    )
    renderer._update_robot_geometry(curves_next, frames_next)
    assert len(renderer._server.meshes) == 58
    assert renderer._server.atomic_calls == 1
    assert any(not np.allclose(h.vertices, v) for h, v in zip(handles[0], old_vertices))
    for handle, expected in zip(
        handles[0], renderer._meshes(curves_next[0], frames_next[0])
    ):
        assert_allclose(handle.vertices, expected[1])
    renderer._update_robot_geometry(curves_next, frames_next, atomic=False)
    assert renderer._server.atomic_calls == 1
    renderer._build_robot_geometry(
        curves[:1],
        colors[:1],
        material_frames=frames[:1],
        base_plate_color=(0.5, 0.5, 0.5),
    )
    assert all(h.removed for group in handles for h in group)
    assert renderer._scene_handles.num_robots == 1
