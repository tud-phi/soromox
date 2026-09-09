"""Shared configuration behavior and backend adaptation contracts."""

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_base_renderer import DummyPlanarRobot

from soromox.rendering import (
    CameraConfig,
    GeometryConfig,
    GroundPlaneConfig,
    MatplotlibRenderer,
    OpenCVPlanarRenderer,
    PointLightConfig,
    RendererColorConfig,
    RendererConfig,
    RenderOutputConfig,
    SceneConfig,
    ViserRenderer,
)
from soromox.rendering.scenery import (
    backdrop_mesh,
    linear_to_srgb,
    plane_basis,
    srgb_to_linear,
)


def robot():
    """Return a simple planar fixture with a translated base."""
    result = DummyPlanarRobot(jnp.array([0.0, 0.2, 0.3]))
    result.num_coordinates = 1
    return result


def test_independent_editable_defaults_and_construction_snapshot():
    a, b = RendererConfig(), RendererConfig()
    a.scene.ground.height = 0.4
    a.geometry.num_points = 12
    renderer = OpenCVPlanarRenderer(robot(), config=a)
    a.geometry.num_points = 20
    a.scene.ground.height = 0.8
    assert b.scene.ground.height == 0
    assert b.geometry.num_points == 80
    assert b.geometry.cross_section_resolution == 48
    assert (b.output.width, b.output.height) == (800, 600)
    assert renderer.num_points == 12
    assert renderer.config.scene.ground.height == 0.4


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("geometry", "num_points", 1),
        ("output", "width", 0),
        ("scene", "tone_mapping", "invalid"),
        ("camera", "exposure_ev100", float("nan")),
    ],
)
def test_edits_are_validated_when_constructing(section, field, value):
    config = RendererConfig()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        OpenCVPlanarRenderer(robot(), config=config)


def test_photometry_and_preset_scaling_do_not_rescale_explicit_lights():
    light = PointLightConfig.from_lumens(4 * np.pi * 15, position=(1.0, 2.0, 3.0))
    assert light.intensity_candela == pytest.approx(15)
    small, large = (
        SceneConfig.studio(scene_extent=1),
        SceneConfig.studio(scene_extent=2),
    )
    assert_allclose(
        np.array(large.lights[1].position), 2 * np.array(small.lights[1].position)
    )
    assert large.lights[1].intensity_candela == pytest.approx(
        4 * small.lights[1].intensity_candela
    )
    config = RendererConfig(scene=SceneConfig.studio(scene_extent=20, lights=(light,)))
    renderer = OpenCVPlanarRenderer(robot(), config=config)
    assert renderer.config.scene.lights[0] == light
    with pytest.raises(ValueError):
        PointLightConfig.from_lumens(-1)


def test_world_base_and_explicit_ground_orientation():
    renderer = OpenCVPlanarRenderer(robot())
    curves = np.array([[[0.2, 0.3], [1.0, 0.3]], [[2.0, 0.8], [3.0, 0.8]]])
    renderer._fit_scene_bounds(curves)
    world = renderer._resolve_ground_planes(curves)
    assert len(world) == 1
    assert_allclose(world[0][1], [0, 1, 0])
    assert world[0][0][1] == 0
    renderer.config.scene.ground = GroundPlaneConfig(alignment="base", height=0.1)
    bases = renderer._resolve_ground_planes(curves, [[0, 1, 0], [1, 0, 0]])
    assert len(bases) == 2
    assert_allclose(bases[1][0], [2.04, 0.8, 0])
    renderer.config.scene.ground = GroundPlaneConfig(normal=(0, 0, 2), height=0.7)
    assert_allclose(renderer._resolve_ground_planes(curves)[0][0][2], 0.7)


def test_trajectory_helpers_and_backdrop_bounds():
    renderer = OpenCVPlanarRenderer(robot())
    renderer._fit_scene_bounds(
        np.array([[[0, 0, 0], [1, 0, 0]], [[7, 2, 1], [8, 2, 1]]])
    )
    renderer._expand_scene_bounds_for_spheres([[12, 1, 1]], [0.5])
    center, extent = renderer._appearance_center.copy(), renderer._appearance_extent
    assert center[0] + extent / 2 >= 12.5
    vertices, faces = backdrop_mesh(SceneConfig.studio(), center, extent, [0, 0, 1])
    assert len(faces) > 100
    assert vertices[:, 2].min() == 0
    assert_allclose(renderer._appearance_center, center)
    assert renderer._appearance_extent == extent
    basis = plane_basis([1, 2, 3])
    assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_clay_overrides_robot_palette_but_per_call_colors_replace_section():
    renderer = OpenCVPlanarRenderer(robot(), config=RendererConfig.clay())
    clay = renderer.resolve_backbone_colors(2).per_robot_point_rgba
    assert_allclose(
        clay[..., :3], np.broadcast_to([0.72, 0.65, 0.56], clay[..., :3].shape)
    )
    override = RendererColorConfig()
    colors = renderer.resolve_backbone_colors(2, color_config=override)
    assert not np.allclose(colors.per_robot_point_rgba, clay)
    assert renderer.color_config.robot_override == (0.72, 0.65, 0.56)


def test_warning_is_once_per_render_mode():
    renderer = OpenCVPlanarRenderer(
        robot(), config=RendererConfig(scene=SceneConfig.studio())
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        renderer._warn_simple_appearance("static")
        renderer._warn_simple_appearance("static")
        renderer._warn_simple_appearance("animated")
    assert len(seen) == 2
    assert "shadows" in str(seen[0].message)


def test_viser_lights_exposure_orientation_and_reinitialization():
    class Scene:
        def __init__(self):
            self.active = {}
            self.world_axes = SimpleNamespace(visible=True)

        def add(self, name, **kwargs):
            self.active[name] = kwargs
            return SimpleNamespace(remove=lambda: self.active.pop(name, None))

        add_light_directional = add
        add_light_point = add
        add_light_ambient = add
        add_light_hemisphere = add

        def configure_default_lights(self, **kwargs):
            pass

        def configure_environment_map(self, value):
            pass

        def set_background_image(self, value, **kwargs):
            pass

    renderer = ViserRenderer(
        robot(), config=RendererConfig(scene=SceneConfig.studio()), auto_start=False
    )
    scene = Scene()
    renderer._server = SimpleNamespace(scene=scene)
    renderer._scene_handles = SimpleNamespace(lights=[])
    renderer._active_camera = CameraConfig(exposure_ev100=14)
    with pytest.warns(UserWarning):
        renderer._setup_lighting()
    first = dict(scene.active)
    renderer._setup_lighting()
    assert scene.active == first
    assert len(scene.active) == len(renderer.config.scene.lights) + 1
    key = scene.active["/lights/configured_0"]
    assert key["intensity"] == pytest.approx(
        renderer.config.scene.lights[0].illuminance_lux / 50000 * 2
    )
    assert np.dot(key["position"], renderer.config.scene.lights[0].direction) < 0
    assert renderer.config.camera.exposure_ev100 == 15
    renderer._server = None


@pytest.mark.parametrize(
    "preset", ["technical", "neutral", "bright", "dark", "flat", "clay"]
)
@pytest.mark.parametrize("backend", [OpenCVPlanarRenderer, MatplotlibRenderer])
def test_simple_backends_accept_every_preset(preset, backend):
    config = (
        RendererConfig.clay()
        if preset == "clay"
        else RendererConfig(
            scene=SceneConfig.technical()
            if preset == "technical"
            else SceneConfig.flat()
            if preset == "flat"
            else SceneConfig.studio(preset)
        )
    )
    config.output = RenderOutputConfig(width=120, height=100)
    config.geometry = GeometryConfig(num_points=10)
    renderer = backend(robot(), config=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pixels = renderer.render_frame(jnp.zeros(1), render_actuators=False)
    assert pixels.shape == (100, 120, 3)
    assert pixels.dtype == np.uint8
    assert np.std(pixels) > 0


def test_srgb_conversion_round_trip():
    color = np.linspace(0, 1, 101)
    assert_allclose(linear_to_srgb(srgb_to_linear(color)), color, atol=1e-14)


def test_open3d_converts_point_flux_and_applies_exposure_without_mutating(monkeypatch):
    pytest.importorskip("open3d")
    from test_viser_renderer import DummySpatialRobot

    from soromox.rendering import Open3DRenderer

    config = RendererConfig(
        scene=SceneConfig(lights=(PointLightConfig(intensity_candela=10),))
    )
    renderer = Open3DRenderer(
        DummySpatialRobot(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])), config=config
    )
    renderer._scene_bounds = lambda data: (np.zeros(3), 1.0)
    calls = []

    class Scene:
        SOFT_SHADOWS = 0

        def __init__(self):
            self.scene = self
            self.view = self

        def __getattr__(self, name):
            return lambda *args: calls.append((name, args))

    with pytest.warns(UserWarning, match="exposure"):
        renderer._configure_modern_scene(Scene(), None, CameraConfig(exposure_ev100=14))
    point = next(args for name, args in calls if name == "add_point_light")
    assert point[3] == pytest.approx(10 * 4 * np.pi * 2)
    assert renderer.config.scene.lights[0].intensity_candela == 10
    assert renderer.config.camera.exposure_ev100 == 15


@pytest.mark.parametrize("global_dpi", [72, 100, 300])
def test_matplotlib_video_override_replaces_output_section(
    monkeypatch, tmp_path, global_dpi
):
    import soromox.rendering.matplotlib_renderer as module
    from soromox.rendering import VideoEncodingConfig

    captured = []

    class Writer:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs["video_config"])
            self.proc = SimpleNamespace(returncode=0)

        def write(self, frame):
            assert frame.shape == (100, 120, 3)

        def close(self):
            pass

    monkeypatch.setattr(module, "FFmpegVideoWriter", Writer)
    monkeypatch.setitem(module.plt.rcParams, "figure.dpi", global_dpi)
    default = VideoEncodingConfig(crf=20)
    override = VideoEncodingConfig(crf=8)
    renderer = MatplotlibRenderer(
        robot(),
        config=RendererConfig(
            scene=SceneConfig.flat(),
            output=RenderOutputConfig(width=120, height=100, video=default),
        ),
    )
    renderer.render_sequence(
        jnp.array([0.0, 0.1]),
        jnp.zeros((2, 1)),
        record_path=str(tmp_path / "test.mp4"),
        video_config=override,
        render_actuators=False,
    )
    assert captured == [override]
    assert renderer.config.output.video.crf == 20


def test_opencv_sequence_keeps_trajectory_bounds_and_releases_writer(
    monkeypatch, tmp_path
):
    import soromox.rendering.opencv_base as module

    renderer = OpenCVPlanarRenderer(
        robot(), config=RendererConfig(scene=SceneConfig.flat())
    )
    bounds, writers = [], []
    override = RendererColorConfig(base_plate_color=(1.0, 0.0, 0.0))

    class Writer:
        stderr_log = ""
        closed = False

        def __init__(self, *args, **kwargs):
            writers.append(self)

        def write(self, image):
            pass

        def close(self):
            self.closed = True

    def curve(q):
        return np.array([[float(q[0]), 0.0], [float(q[0]) + 1.0, 1.0]])

    def frame(q, **kwargs):
        assert kwargs["color_config"] is override
        renderer._fit_scene_bounds(curve(q))
        bounds.append((renderer._appearance_center.copy(), renderer._appearance_extent))
        return np.zeros((600, 800, 3), dtype=np.uint8)

    monkeypatch.setattr(module, "FFmpegVideoWriter", Writer)
    monkeypatch.setattr(renderer, "compute_backbone_curve", curve)
    monkeypatch.setattr(renderer, "render_frame", frame)
    renderer.render_sequence(
        np.array([0.0, 0.1]),
        np.array([[0.0], [4.0]]),
        record_path=str(tmp_path / "bounds.mp4"),
        color_config=override,
    )
    assert_allclose(bounds[0][0], bounds[1][0])
    assert bounds[0][1] == bounds[1][1]
    assert bounds[0][1] > 5.0
    assert writers[0].closed
    assert not renderer._appearance_bounds_locked


def test_grid_spacing_is_exact_and_anchored_across_scene_sizes():
    from soromox.rendering.scenery import ground_grid

    config = GroundPlaneConfig(surface=False, grid_spacing=0.1, grid_major_every=5)
    for center, size in [([0.03, 0.07, 0], 1.03), ([0.08, 0.02, 0], 1.17)]:
        points, colors = ground_grid(config, center, [0, 0, 1], size)
        constant_x = np.isclose(points[:, 0, 0], points[:, 1, 0])
        x = points[constant_x, 0, 0]
        assert_allclose(np.diff(x), 0.1)
        assert_allclose(x / 0.1, np.rint(x / 0.1), atol=1e-12)
        assert np.any(np.all(np.isclose(colors, config.grid_major_color), axis=1))
        assert np.any(np.all(np.isclose(colors, config.grid_color), axis=1))
    assert not RendererConfig().scene.ground.surface
    assert SceneConfig.studio().ground.surface


def test_preset_fills_use_point_lights_and_scale_with_extent():
    from soromox.rendering import DirectionalLightConfig

    for factory in [SceneConfig.technical, SceneConfig.studio]:
        small, large = factory(scene_extent=1), factory(scene_extent=2)
        assert (
            sum(isinstance(light, DirectionalLightConfig) for light in small.lights)
            == 1
        )
        for a, b in zip(small.lights[1:], large.lights[1:]):
            assert isinstance(a, PointLightConfig)
            assert_allclose(np.asarray(b.position), 2 * np.asarray(a.position))
            assert b.intensity_candela == pytest.approx(4 * a.intensity_candela)
    assert (
        sum(
            isinstance(light, DirectionalLightConfig)
            for light in RendererConfig.clay().scene.lights
        )
        == 1
    )


def test_viser_static_capture_uses_explicit_pose_and_waits_for_meshes(monkeypatch):
    from test_viser_renderer import DummySpatialRobot, FakeViserClient, FakeViserServer

    import soromox.rendering.viser_renderer as module

    camera = CameraConfig(position=(1.0, -2.0, 1.0), look_at=(0.0, 0.0, 0.1))
    renderer = ViserRenderer(
        DummySpatialRobot(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
        config=RendererConfig(
            camera=camera, output=RenderOutputConfig(width=80, height=60)
        ),
        auto_start=False,
    )
    client = FakeViserClient()
    renderer._server = FakeViserServer({0: client})
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return np.full((60, 80, 3), 0 if len(calls) == 1 else 128, dtype=np.uint8)

    client.get_render = capture
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(renderer, "_setup_lighting", lambda: None)
    monkeypatch.setattr(renderer, "_clear_scene", lambda: None)
    monkeypatch.setattr(renderer, "_build_robot_geometry", lambda *a, **kw: None)
    pixels = renderer.render_frame(jnp.zeros(1), render_actuators=False)
    assert np.all(pixels == 128)
    assert len(calls) == 3
    assert_allclose(calls[0]["position"], camera.position)
    assert calls[0]["fov"] == pytest.approx(np.deg2rad(camera.fov))
    forward = module.viser.transforms.SO3(calls[0]["wxyz"]).as_matrix()[:, 2]
    expected = np.asarray(camera.look_at) - np.asarray(camera.position)
    assert_allclose(forward, expected / np.linalg.norm(expected))
    renderer._server = None


def test_batched_base_ground_planes_normalize_axes_and_preserve_integer_curves():
    renderer = OpenCVPlanarRenderer(robot())
    renderer.config.scene.ground = GroundPlaneConfig(
        alignment="base", height=0.2, size=2
    )
    curves = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    originals = curves.copy()
    planes = renderer._resolve_ground_planes(curves, [[0, 4, 0], [3, 0, 4]])
    centers, normals, sizes = map(np.asarray, zip(*planes))
    assert_allclose(normals, [[0, 1, 0], [0.6, 0, 0.8]])
    assert_allclose(centers, [[1, 2.14, 0], [5.084, 6, 0.112]])
    assert_allclose(sizes, [2, 2])
    assert_allclose(curves, originals)
    fallback = renderer._resolve_ground_planes(curves)
    assert_allclose(fallback[0][1], fallback[1][1])


def test_technical_tone_mapping_preserves_white_without_changing_studio_defaults():
    """Keep the technical background white and studio shading filmic by default."""
    assert SceneConfig.technical().tone_mapping == "linear"
    assert SceneConfig.technical(tone_mapping="aces").tone_mapping == "aces"
    assert SceneConfig.studio().tone_mapping == "backend-default"
