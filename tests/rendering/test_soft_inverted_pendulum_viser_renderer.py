"""Tests for Viser motion rendering of the soft cart pendulum."""

from contextlib import contextmanager

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs.soft_inverted_pendulum import (
    SoftInvertedPendulumConfig,
    make_cart_pendulum,
)
from examples.simulation.gvs.soft_inverted_pendulum_viser_renderer import (
    SoftCartPendulumViserRenderer,
)
from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer


class _GeometryHandle:
    """Minimal mutable Viser geometry handle used by the hook test."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Scene:
    """Collect boxes and spheres added by the example renderer."""

    def __init__(self):
        self.boxes = []
        self.spheres = []

    def add_box(self, **kwargs):
        handle = _GeometryHandle(**kwargs)
        self.boxes.append(handle)
        return handle

    def add_icosphere(self, **kwargs):
        handle = _GeometryHandle(**kwargs)
        self.spheres.append(handle)
        return handle


class _Server:
    """Minimal Viser server exposing a scene and atomic updates."""

    def __init__(self):
        self.scene = _Scene()
        self.atomic_updates = 0

    @contextmanager
    def atomic(self):
        self.atomic_updates += 1
        yield


def test_cart_renderer_tracks_sequence_and_moves_all_cart_geometry(monkeypatch):
    robot = make_cart_pendulum(SoftInvertedPendulumConfig())
    renderer = SoftCartPendulumViserRenderer(robot, auto_start=False)
    q_ts = jnp.array([[0.0, 0.1, -0.1], [0.4, 0.2, -0.1]])

    monkeypatch.setattr(ViserRenderer, "render_sequence", lambda *args, **kwargs: None)
    renderer.render_sequence(jnp.array([0.0, 0.1]), q_ts)
    assert_allclose(renderer._cart_positions, [0.0, 0.4])

    server = _Server()
    renderer._server = server
    renderer._scene_handles = SceneHandles()
    renderer._after_sequence_scene_built()
    assert len(server.scene.boxes) == 2
    assert len(server.scene.spheres) == 2
    assert set(renderer._scene_handles.custom_primitives) == {
        "soft_cart_rail",
        "soft_cart_body",
        "soft_cart_wheel_0",
        "soft_cart_wheel_1",
    }

    renderer._after_sequence_frame_updated(1)
    assert_allclose(renderer._cart_handle.position, [0.0, 0.4, -0.06])
    assert_allclose(renderer._wheel_handles[0].position, [0.0, 0.305, -0.135])
    assert_allclose(renderer._wheel_handles[1].position, [0.0, 0.495, -0.135])
    assert server.atomic_updates == 1


def test_cart_renderer_rejects_non_scalar_cart_trajectory():
    robot = make_cart_pendulum(SoftInvertedPendulumConfig())
    renderer = SoftCartPendulumViserRenderer(robot, auto_start=False)
    with pytest.raises(ValueError, match="shape"):
        renderer.render_sequence(jnp.array([0.0]), jnp.zeros((1, 2)))
