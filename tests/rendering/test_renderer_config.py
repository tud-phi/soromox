"""Shared configuration behavior and backend adaptation contracts."""

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_base_renderer import DummyPlanarRobot, DummyRenderer, DummySpatialRobot

from soromox.rendering import (
    BackdropConfig,
    CameraConfig,
    DirectionalLightConfig,
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


def test_invalid_ground_height_reference_is_rejected():
    config = RendererConfig()
    config.scene.ground.height_reference = "invalid"
    with pytest.raises(ValueError, match="height_reference"):
        OpenCVPlanarRenderer(robot(), config=config)


def test_base_mounting_face_ground_rejects_nonvertical_fixed_base_robot():
    with pytest.raises(ValueError, match="parallel or antiparallel"):
        DummyRenderer(
            DummySpatialRobot(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
            config=RendererConfig(
                scene=SceneConfig(
                    ground=GroundPlaneConfig(height_reference="base_mounting_face")
                )
            ),
        )


def test_base_mounting_face_ground_rejects_floating_base_robot():
    floating_robot = DummySpatialRobot(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    floating_robot.floating_base = True
    with pytest.raises(ValueError, match="floating-base"):
        DummyRenderer(
            floating_robot,
            config=RendererConfig(
                scene=SceneConfig(
                    ground=GroundPlaneConfig(height_reference="base_mounting_face")
                )
            ),
        )


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
    spatial_renderer = DummyRenderer(
        DummySpatialRobot(
            jnp.array([0.0, -np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0, 0.0, 0.0])
        ),
        config=RendererConfig(
            scene=SceneConfig(
                ground=GroundPlaneConfig(height_reference="base_mounting_face")
            )
        ),
    )
    spatial_curves = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    spatial_renderer._fit_scene_bounds(spatial_curves)
    grounded_spatial = spatial_renderer._resolve_ground_planes(spatial_curves)
    assert grounded_spatial[0][0][2] == pytest.approx(-0.06)
    grounded = spatial_renderer._resolve_ground_planes(spatial_curves[:1])
    assert grounded[0][0][2] == pytest.approx(-0.06)
    spatial_renderer.config.scene.ground = GroundPlaneConfig(
        height_reference="base_mounting_face", height=0.1
    )
    grounded_offset = spatial_renderer._resolve_ground_planes(spatial_curves[:1])
    assert grounded_offset[0][0][2] == pytest.approx(-0.06 + 0.1)
    with pytest.raises(ValueError, match="share one height"):
        spatial_renderer._resolve_ground_planes(
            np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]],
                ]
            )
        )
    hanging_renderer = DummyRenderer(
        DummySpatialRobot(
            jnp.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0, 0.0, 0.0])
        ),
        config=RendererConfig(
            scene=SceneConfig(
                ground=GroundPlaneConfig(height_reference="base_mounting_face")
            )
        ),
    )
    hanging = hanging_renderer._resolve_ground_planes(spatial_curves[:1])
    assert hanging[0][0][2] == pytest.approx(0.06)
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
    grounded_scene = SceneConfig.studio(
        ground=GroundPlaneConfig(height_reference="base_mounting_face")
    )
    grounded_vertices, _ = backdrop_mesh(
        grounded_scene, center, extent, [0, 0, 1], ground_height=-0.06
    )
    assert grounded_vertices[:, 2].min() == pytest.approx(-0.06)
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
    assert "white background" in str(seen[0].message)


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


@pytest.mark.parametrize("normal", [[0, 0, 1], [0, 0, -1], [0, 1, 0], [1, 2, 3]])
def test_eased_backdrop_dimensions_orientation_and_smooth_joins(normal):
    """Preserve world floor height, bend dimensions and continuous join tangents."""
    scene = SceneConfig.studio()
    scene.ground.height = 0.17
    center, extent = np.array([0.4, -0.2, 0.3]), 2.0
    vertices, faces = backdrop_mesh(scene, center, extent, normal)
    basis = plane_basis(normal)
    origin = center - (center @ basis[:, 2]) * basis[:, 2] + 0.17 * basis[:, 2]
    local = (vertices - origin) @ basis / extent
    profile = local[::2, 1:]
    assert_allclose([local[:, 0].min(), local[:, 0].max()], [-2.5, 2.5])
    assert_allclose(profile[0], [-2.5, 0], atol=1e-14)
    assert_allclose(profile[-1], [0.42, 2.5])
    assert np.all(np.diff(profile, axis=0) >= -1e-14)
    start = np.argmin(np.linalg.norm(profile - [0.02, 0], axis=1))
    end = start + 256
    assert_allclose(profile[end], [0.42, 0.23])
    slope_start = profile[start + 1] - profile[start]
    slope_end = profile[end] - profile[end - 1]
    assert abs(slope_start[1] / slope_start[0]) < 0.002
    assert abs(slope_end[0] / slope_end[1]) < 0.002
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vertical_radius": 0},
        {"vertical_radius": float("nan")},
        {"vertical_radius": 3},
        {"curvature_easing": -0.1},
        {"curvature_easing": 1.1},
        {"curvature_easing": float("nan")},
        {"radius": 3, "curvature_easing": 1},
    ],
)
def test_backdrop_rejects_invalid_eased_dimensions(kwargs):
    with pytest.raises(ValueError):
        BackdropConfig(**kwargs)


def test_clay_shares_studio_settings_with_independent_configuration_objects():
    neutral, clay = SceneConfig.studio(), RendererConfig.clay().scene
    assert neutral.backdrop.vertical_radius == 0.23
    assert neutral.backdrop.curvature_easing == 0.8
    assert neutral.ambient.strength == 1.05
    assert_allclose(neutral.lights[0].direction, [-0.25, 0.55, -1])
    assert neutral.lights[1].intensity_candela == pytest.approx(212500 / (4 * np.pi))
    assert neutral.lights[2].intensity_candela == pytest.approx(112500 / (4 * np.pi))
    assert clay.backdrop == neutral.backdrop
    assert clay.ambient == neutral.ambient
    assert clay.lights[:3] == neutral.lights
    assert len(clay.lights) == 4
    assert clay.lights[3].intensity_candela == pytest.approx(100000 / (4 * np.pi))
    assert clay.material.roughness == 0.9
    clay.backdrop.radius = 0.6
    clay.lights[0].illuminance_lux = 10
    assert neutral.backdrop.radius == 0.4
    assert neutral.lights[0].illuminance_lux == 70000


@pytest.mark.parametrize(
    "style,key,ambient,front,rear,direction",
    [
        ("bright", 85000, 1.2, 400000, 250000, (-0.25, 0.15, -1)),
        ("dark", 100000, 0.4, 300000, 250000, (-0.25, 0.9, -0.55)),
    ],
)
def test_studio_variants_share_backdrop_and_retain_distinct_lighting(
    style, key, ambient, front, rear, direction
):
    """Changing the studio sweep preserves bright and dark lighting identities."""
    scene = SceneConfig.studio(style)
    assert scene.backdrop == SceneConfig.studio().backdrop
    assert scene.ambient.strength == ambient
    assert scene.lights[0].illuminance_lux == key
    assert_allclose(scene.lights[0].direction, direction)
    assert scene.lights[1].intensity_candela == pytest.approx(front / (4 * np.pi))
    assert scene.lights[2].intensity_candela == pytest.approx(rear / (4 * np.pi))


@pytest.mark.parametrize(
    "preset", ["technical", "neutral", "bright", "dark", "flat", "clay"]
)
def test_opencv_presets_use_identical_white_canvas_without_scenery(preset):
    """Scene settings cannot recolor OpenCV backgrounds or add ground lines."""
    scene = (
        RendererConfig.clay().scene
        if preset == "clay"
        else SceneConfig.technical()
        if preset == "technical"
        else SceneConfig.flat()
        if preset == "flat"
        else SceneConfig.studio(preset)
    )

    def capture(scene):
        renderer = OpenCVPlanarRenderer(
            robot(),
            config=RendererConfig(
                scene=scene,
                output=RenderOutputConfig(width=160, height=120),
                geometry=GeometryConfig(num_points=10),
            ),
        )
        with pytest.warns(UserWarning, match="white background"):
            return renderer.render_frame(jnp.zeros(1), render_actuators=False)

    pixels = capture(scene)
    np.testing.assert_array_equal(pixels, capture(SceneConfig.flat()))
    np.testing.assert_array_equal(pixels[0, 0], [255, 255, 255])
    assert np.any(pixels != 255)


def test_opencv_video_frames_ignore_studio_background(monkeypatch, tmp_path):
    """Video uses the same white canvas and emits one warning for the operation."""
    import soromox.rendering.opencv_base as module

    frames = []

    class Writer:
        stderr_log = ""

        def __init__(self, *args, **kwargs):
            assert kwargs["input_pix_fmt"] == "bgr24"

        def write(self, image):
            frames.append(image.copy())

        def close(self):
            pass

    monkeypatch.setattr(module, "FFmpegVideoWriter", Writer)
    renderer = OpenCVPlanarRenderer(
        robot(),
        config=RendererConfig(scene=SceneConfig.studio("dark")),
    )
    with pytest.warns(UserWarning, match="white background") as caught:
        renderer.render_sequence(
            np.array([0.0, 0.1]),
            np.zeros((2, 1)),
            record_path=str(tmp_path / "white.mp4"),
        )
    assert len(caught) == 1
    assert len(frames) == 2
    for frame in frames:
        np.testing.assert_array_equal(frame[0, 0], [255, 255, 255])


def test_hsa_opencv_ignores_studio_scenery():
    """The specialized HSA renderer also preserves a plain white canvas."""
    from pathlib import Path

    from soromox.rendering.planar_hsa.opencv_renderer import OpenCVPlanarHSARenderer
    from soromox.systems import PlanarHSA, PlanarHSAParams, PlanarHSAStructure

    params_path = (
        Path(__file__).resolve().parents[2]
        / "assets/robot_parameters/planar_hsa/fpu_control.npz"
    )
    hsa = PlanarHSA(
        params=PlanarHSAParams.from_npz(params_path),
        structure=PlanarHSAStructure(),
        base_pose=jnp.zeros(3),
    )
    images = []
    for scene in (SceneConfig.studio("dark"), SceneConfig.flat()):
        renderer = OpenCVPlanarHSARenderer(
            hsa,
            config=RendererConfig(
                scene=scene, output=RenderOutputConfig(width=240, height=180)
            ),
        )
        with pytest.warns(UserWarning, match="white background"):
            images.append(renderer.render_frame(jnp.zeros(hsa.num_coordinates)))
    np.testing.assert_array_equal(images[0], images[1])
    np.testing.assert_array_equal(images[0][0, 0], [255, 255, 255])
    assert np.any(images[0] != 255)


@pytest.mark.parametrize(
    "preset", ["technical", "neutral", "bright", "dark", "flat", "clay"]
)
def test_preset_lights_and_backdrop_rotate_together_for_hanging(preset):
    from soromox.rendering.scenery import resolved_lights

    scene = (
        RendererConfig.clay().scene
        if preset == "clay"
        else SceneConfig.technical()
        if preset == "technical"
        else SceneConfig.flat()
        if preset == "flat"
        else SceneConfig.studio(preset)
    )
    rotation = np.diag([-1, 1, -1])
    upright = list(resolved_lights(scene, (0, 0, 1)))
    hanging = list(resolved_lights(scene, (0, 0, -1)))
    for original, rotated in zip(upright, hanging):
        field = "position" if isinstance(original, PointLightConfig) else "direction"
        assert_allclose(getattr(rotated, field), rotation @ getattr(original, field))
    if scene.backdrop.enabled:
        center = np.array([0.2, -0.1, 0.4])
        vertices, faces = backdrop_mesh(scene, center, 1.2, (0, 0, 1))
        inverted, inverted_faces = backdrop_mesh(
            scene, rotation @ center, 1.2, (0, 0, -1)
        )
        assert_allclose(inverted, vertices @ rotation.T)
        assert_allclose(inverted_faces, faces)
        # Floor faces point toward the robot in both mounting orientations.
        triangle = inverted[inverted_faces[0]]
        assert np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2] < 0


def test_explicit_world_lights_do_not_rotate_with_ground():
    from soromox.rendering.scenery import resolved_lights

    light = PointLightConfig(position=(1, 2, 3))
    scene = SceneConfig.studio(lights=(light,))
    assert list(resolved_lights(scene, (0, 0, -1))) == [light]
    assert scene.lights == (light,)


@pytest.mark.parametrize("light_type", [PointLightConfig, DirectionalLightConfig])
def test_light_reference_validation(light_type):
    with pytest.raises(ValueError, match="reference"):
        light_type(reference="invalid")


@pytest.mark.parametrize("x", [0.01, 0.4, 0.96, 1.0])
def test_ground_basis_does_not_flip_across_horizontal_normals(x):
    from soromox.rendering.scenery import resolved_lights

    # Avoid the unavoidable reference-axis singularity at exactly +/-Y.
    y = np.sqrt(1 - x**2)
    normals = ([x, y, 1e-7], [x, y, -1e-7])
    bases = [plane_basis(n) for n in normals]
    assert_allclose(bases[0], bases[1], atol=3e-5)
    scene = SceneConfig.studio()
    lights = [list(resolved_lights(scene, n)) for n in normals]
    for left, right in zip(*lights):
        field = "position" if isinstance(left, PointLightConfig) else "direction"
        assert_allclose(getattr(left, field), getattr(right, field), atol=3e-5)
    meshes = [backdrop_mesh(scene, np.zeros(3), 1.0, n)[0] for n in normals]
    assert_allclose(meshes[0], meshes[1], atol=1e-4)


@pytest.mark.parametrize(
    "normal", [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [1, 2, -3]]
)
def test_ground_basis_is_right_handed_with_canonical_mounts(normal):
    basis = plane_basis(normal)
    assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1)
    assert_allclose(basis[:, 2], np.asarray(normal) / np.linalg.norm(normal))
    if normal == [0, 0, 1]:
        assert_allclose(basis, np.eye(3))
    elif normal == [0, 0, -1]:
        assert_allclose(basis, np.diag([-1, 1, -1]))
    elif normal[0] == normal[2] == 0:
        assert_allclose(basis[:, 0], [1, 0, 0])
