"""Public configuration imports and migrated example construction."""

import ast
import runpy
from pathlib import Path

import numpy as np

from soromox import rendering
from soromox.rendering import config


def test_config_namespace_exports_the_public_renderer_settings():
    for name in config.__all__:
        assert getattr(config, name) is getattr(rendering, name)
    settings = config.RendererConfig.clay()
    assert isinstance(settings.output.video, config.VideoEncodingConfig)
    assert type(settings.scene).__module__ == "soromox.rendering.config.scene"
    assert type(settings.camera).__module__ == "soromox.rendering.config.camera"
    assert type(settings.colors).__module__ == "soromox.rendering.config.colors"
    assert type(settings.output).__module__ == "soromox.rendering.config.output"


def test_pendulum_example_constructs_opencv_renderer_with_black_backbones():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/simulation/pendulum/simulate_pendulum.py"
    )
    namespace = runpy.run_path(str(path))
    namespace["robot"] = namespace["Pendulum"](params=namespace["params"])
    call = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OpenCVPlanarRenderer"
    )
    # Execute the actual constructor expression without running the long rollout.
    renderer = eval(compile(ast.Expression(call), str(path), "eval"), namespace)
    assert (renderer.width, renderer.height) == (
        namespace["video_width"],
        namespace["video_height"],
    )
    colors = renderer.resolve_backbone_colors(1).per_robot_point_rgba
    np.testing.assert_array_equal(colors[..., :3], 0)
