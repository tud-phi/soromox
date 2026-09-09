"""Preset gallery composition and mounting regressions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.rendering import RendererConfig, SceneConfig

spec = importlib.util.spec_from_file_location(
    "tentacle_scene",
    Path(__file__).resolve().parents[2] / "examples/rendering/tentacle_scene.py",
)
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


@pytest.mark.parametrize("preset", gallery.PRESETS)
def test_gallery_preserves_preset_ground_appearance(preset):
    expected = (
        RendererConfig.clay().scene
        if preset == "clay"
        else SceneConfig.technical()
        if preset == "technical"
        else SceneConfig.flat()
        if preset == "flat"
        else SceneConfig.studio(preset)
    )
    for mounting in ("upright", "hanging"):
        config, _, _ = gallery.make_comparison(preset, mounting=mounting)
        assert config.scene.ground.color == expected.ground.color
        assert config.scene.ground.visible == expected.ground.visible
        assert config.scene.ground.surface == expected.ground.surface
        assert config.scene.ground.grid == expected.ground.grid


def test_hanging_gallery_rotates_robot_and_camera_as_one_scene():
    rotation = np.diag([-1, 1, -1])
    upright, q, offsets = gallery.make_comparison()
    hanging, hanging_q, hanging_offsets = gallery.make_comparison(mounting="hanging")
    assert upright.camera.position[0] == upright.camera.look_at[0] == 0
    assert hanging.camera.position[0] == hanging.camera.look_at[0] == 0
    assert_allclose(hanging_q, q)
    assert_allclose(hanging_offsets, offsets @ rotation.T)
    assert_allclose(hanging.camera.position, rotation @ upright.camera.position)
    assert_allclose(hanging.camera.look_at, rotation @ upright.camera.look_at)
    assert_allclose(hanging.camera.up, upright.camera.up)
    up_robot = gallery.make_tentacle()
    down_robot = gallery.make_tentacle("hanging")
    up_pose = np.asarray(up_robot.base_transform)
    down_pose = np.asarray(down_robot.base_transform)
    assert_allclose(down_pose[:3, :3], rotation @ up_pose[:3, :3], atol=1e-14)
