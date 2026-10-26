"""Resolved routing geometry and its shared rendering entry points."""

from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import planar_pcs_params

from soromox.actuation import (
    CircularLumenParams,
    PiecewiseLinearRoutingDiscretization,
    RoutingSpanParams,
    ShortestPathRouting,
    ShortestPathRoutingParams,
    StraightSpanRouting,
    ThreadlikeActuator,
    ThreadlikeRouting,
)
from soromox.rendering import (
    GeometryConfig,
    MatplotlibRenderer,
    RendererConfig,
    RenderOutputConfig,
    actuator_visual_layers,
)
from soromox.rendering import actuation as visual_adapter
from soromox.systems import (
    GVS,
    GVSSegment,
    Helicoidal,
    JointSpec,
    LinkSpec,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def robot():
    return PlanarPCS(
        planar_pcs_params(
            length=jnp.array([0.1, 0.1]),
            radius=jnp.array([0.02, 0.02]),
            density=jnp.full(2, 1000.0),
            young_modulus=jnp.full(2, 1000.0),
            shear_modulus=jnp.full(2, 500.0),
            gravity=jnp.zeros(2),
            damping_matrix=jnp.zeros((6, 6)),
        ),
        PlanarPCSStructure(num_gauss_points=5),
        base_pose=jnp.array([1.2, 0.01, 0.01]),
        backend="jax",
    )


def make_routing(strategy):
    # Path 0 has two spans; path 1 has one. Both have guides offset from the axis.
    offsets = jnp.array([[0, 0.003, 0], [0, 0.003, 0], [0, -0.003, 0]])
    spans = RoutingSpanParams(offsets, offsets, (0, 0, 1), (0, 1, 0), (0, 1, 0))
    if strategy is StraightSpanRouting:
        return strategy(spans)
    return strategy(
        ShortestPathRoutingParams(spans, CircularLumenParams.constant(0.01, 2)),
        PiecewiseLinearRoutingDiscretization(elements_per_segment=2),
    )


@pytest.fixture(scope="module", params=[StraightSpanRouting, ShortestPathRouting])
def case(robot, request):
    routing = make_routing(request.param)
    q = jnp.array([0.4, 0, 0, -0.2, 0, 0])
    solution = eqx.filter_jit(routing.solve_routing)(robot, q)
    assert bool(solution.converged)
    return routing, q, solution


def no_solve(*args, **kwargs):
    raise AssertionError("Rendering an existing solution must not solve routing")


def renderer_config():
    return RendererConfig(
        geometry=GeometryConfig(num_points=6, cross_section_resolution=8),
        output=RenderOutputConfig(width=240, height=160),
    )


def test_resolved_paths_preserve_endpoints_and_unequal_span_counts(
    robot, case, monkeypatch
):
    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    layers = actuator_visual_layers(
        robot, q, jnp.linspace(0, 1, 6), routing=routing, routing_solution=solution
    )
    assert [layer.points.shape for layer in layers] == [(1, 12, 3), (1, 6, 3)]
    assert [layer.name for layer in layers] == ["routing/path_0", "routing/path_1"]
    for index in range(2):
        expected = routing.span_positions(
            robot, q, solution, index, jnp.linspace(0, 1, 6)
        )
        assert_allclose(
            layers[0].points[0, 6 * index : 6 * (index + 1)], expected, atol=1e-12
        )
    direct = routing.span_positions(robot, q, solution, 2, jnp.array([0.0, 1.0]))
    assert_allclose(layers[1].points[0, [0, -1]], direct, atol=1e-12)


def test_omitted_solution_solves_once_for_all_paths(robot, case, monkeypatch):
    routing, q, solution = case
    solve = Mock(return_value=solution)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", solve)
    assert (
        len(actuator_visual_layers(robot, q, jnp.linspace(0, 1, 6), routing=routing))
        == 2
    )
    solve.assert_called_once_with(robot, q, routing)


@pytest.mark.parametrize("custom_hook", [False, True])
def test_installed_layers_and_saved_routes_coexist(robot, monkeypatch, custom_hook):
    fixed = ThreadlikeActuator.tendons(
        ThreadlikeRouting.linear(
            intercept=jnp.array([[0, 0.002, 0]]),
            slope=jnp.zeros((1, 3)),
            start_segment_index=(0,),
            end_segment_index=(1,),
        )
    )
    robot = eqx.tree_at(lambda model: model.actuators, robot, (fixed,))
    routing = make_routing(StraightSpanRouting)
    q = jnp.array([0.4, 0, 0, -0.2, 0, 0])
    solution = eqx.filter_jit(routing.solve_routing)(robot, q)
    stations = jnp.linspace(0, 0.2, 6)
    inputs = jnp.array([2.0])
    installed = actuator_visual_layers(robot, q, stations, actuator_inputs=inputs)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    combined = actuator_visual_layers(
        robot,
        q,
        stations,
        actuator_inputs=inputs,
        routing=routing,
        routing_solution=solution,
    )
    assert len(combined) == 3
    assert_allclose(combined[0].points, installed[0].points)
    assert_allclose(combined[0].scalar_fields["input"], inputs)
    hook = Mock(return_value=installed)
    if custom_hook:
        monkeypatch.setattr(PlanarPCS, "actuator_visual_layers", hook, raising=False)
    renderer = MatplotlibRenderer(robot, config=renderer_config())
    layers = renderer.compute_actuator_visual_layers(
        q, actuator_inputs=inputs, routing=routing, routing_solution=solution
    )
    assert len(layers) == 3
    assert_allclose(layers[0].points, installed[0].points)
    for actual, expected in zip(layers[1:], combined[1:], strict=True):
        assert_allclose(actual.points, expected.points[..., :2])
    if custom_hook:
        hook.assert_called_once()


def test_existing_solutions_skip_optimizer_in_public_frame_rendering(
    robot, case, monkeypatch
):
    from soromox.rendering import OpenCVPlanarRenderer

    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    for cls in (MatplotlibRenderer, OpenCVPlanarRenderer):
        renderer = cls(robot, config=renderer_config())
        plain = renderer.render_frame(q)
        routed = renderer.render_frame(q, routing=routing, routing_solution=solution)
        assert routed.shape == (160, 240, 3)
        assert np.any(plain != routed)
        hidden = renderer.render_frame(
            q, routing=routing, routing_solution=solution, render_actuators=False
        )
        np.testing.assert_array_equal(plain, hidden)


def test_batch_and_trajectory_solutions_keep_order_and_offsets(robot, monkeypatch):
    routing = make_routing(StraightSpanRouting)
    qs = jnp.array([[0.2, 0, 0, -0.4, 0, 0], [0.8, 0, 0, -0.1, 0, 0]])
    solutions = eqx.filter_jit(jax.vmap(lambda q: routing.solve_routing(robot, q)))(qs)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = MatplotlibRenderer(robot, config=renderer_config())
    offsets = jnp.array([[0.3, 0.2], [-0.2, 0.4]])
    batched = renderer.compute_actuator_visual_layers_batched(
        qs, offsets, routing=routing, routing_solution=solutions
    )
    for index in range(2):
        single = jax.tree.map(lambda x, index=index: x[index], solutions)
        expected = actuator_visual_layers(
            robot,
            qs[index],
            jnp.linspace(0, 1, 6),
            routing=routing,
            routing_solution=single,
        )
        assert_allclose(
            batched[0].points[index],
            expected[0].points[..., :2] + offsets[index],
            atol=1e-12,
        )
    # Two robots, two frames, distinct solutions at every position.
    qts = jnp.stack([qs, qs[::-1]])
    sts = jax.tree.map(lambda x: jnp.stack([x, x[::-1]]), solutions)
    layers = renderer.compute_actuator_visual_layers_trajectory(
        qts, offsets, routing=routing, routing_solution=sts
    )
    assert layers[0].points.shape == (2, 2, 1, 12, 2)
    assert_allclose(
        layers[0].points[0, 1] - offsets[0],
        layers[0].points[1, 0] - offsets[1],
        atol=1e-12,
    )
    # Omitting singleton robot/time axes is supported, broadcasting is not.
    renderer.compute_actuator_visual_layers_trajectory(
        qs[None], offsets[:1], routing=routing, routing_solution=solutions
    )
    with pytest.raises(ValueError, match="not broadcast"):
        renderer.compute_actuator_visual_layers_batched(
            qs,
            offsets,
            routing=routing,
            routing_solution=jax.tree.map(lambda x: x[0], solutions),
        )


def test_invalid_solution_is_rejected(robot, case):
    routing, q, solution = case
    invalid = eqx.tree_at(lambda s: s.spans[0].converged, solution, jnp.array(False))
    with pytest.raises(ValueError, match="unconverged"):
        actuator_visual_layers(
            robot, q, jnp.linspace(0, 1, 80), routing=routing, routing_solution=invalid
        )
    with pytest.raises(ValueError, match="at least two"):
        actuator_visual_layers(
            robot, q, jnp.linspace(0, 1, 1), routing=routing, routing_solution=solution
        )
    with pytest.raises(ValueError, match="routing is required"):
        MatplotlibRenderer(robot).compute_actuator_visual_layers(
            q, routing_solution=solution
        )


@pytest.mark.parametrize("strategy", [StraightSpanRouting, ShortestPathRouting])
def test_joint_bridge_and_floating_base_geometry(strategy, monkeypatch):
    link = LinkSpec.circular(
        young_modulus=1e3,
        shear_modulus=5e2,
        density=1000.0,
        length=0.1,
        radius=0.02,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    basis = StrainBasisSpec(
        type="monomial", strain_selector=[0, 0, 1, 0, 0, 0], basis_order=[0] * 6
    )
    robot = GVS.from_segments(
        [
            GVSSegment(
                link=link, joint=JointSpec.fixed(), basis=basis, num_gauss_points=5
            ),
            GVSSegment(
                link=link,
                joint=JointSpec(type="revolute", axis="z"),
                basis=basis,
                num_gauss_points=5,
            ),
        ],
        floating_base=True,
        gravity=jnp.zeros(3),
        backend="jax",
    )
    routing = make_routing(strategy)
    q = robot.pack_configuration(
        jnp.array([0.0, 0.4, 0.0]),
        base_pose=jnp.array([0.0, 0.0, 0.0, 1.0, 0.5, -0.2, 0.1]),
    )
    solution = eqx.filter_jit(routing.solve_routing)(robot, q)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    points = actuator_visual_layers(
        robot, q, jnp.linspace(0, 1, 6), routing=routing, routing_solution=solution
    )[0].points[0]
    assert_allclose(points[0], [0.5, -0.197, 0.1], atol=1e-12)
    assert_allclose(
        jnp.linalg.norm(points[6] - points[5]), solution.guides[0].length, atol=1e-12
    )
    assert float(solution.guides[0].length) > 0


def test_viser_updates_and_live_frames_reuse_solutions(robot, case, monkeypatch):
    from rendering.test_viser_renderer import FakeViserActuatorServer

    from soromox.rendering.viser_renderer import (
        LiveModeController,
        SceneHandles,
        ViserRenderer,
    )

    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = ViserRenderer(robot, auto_start=False, config=renderer_config())
    renderer._server = FakeViserActuatorServer()
    renderer._scene_handles = SceneHandles()
    renderer._setup_camera = Mock()
    renderer._build_actuator_geometry(
        q[None], np.zeros((1, 3)), 1, routing=routing, routing_solution=solution
    )
    handles = tuple(renderer._scene_handles.actuator_lines)
    assert len(handles) == 2
    expected = actuator_visual_layers(
        robot, q, jnp.linspace(0, 1, 6), routing=routing, routing_solution=solution
    )[0].points[0]
    assert_allclose(handles[0].points[:, 0], expected[:-1], atol=1e-7)
    controller = LiveModeController(renderer)
    controller.start()
    try:
        controller.push_state(np.asarray(q), routing=routing, routing_solution=solution)
    finally:
        controller.stop()
    assert tuple(renderer._scene_handles.actuator_lines) == handles


def test_open3d_scene_reuses_trajectory_solutions(robot, case, monkeypatch):
    from soromox.rendering import Open3DRenderer

    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = Open3DRenderer(robot, config=renderer_config())
    solutions = jax.tree.map(lambda x: jnp.stack([x, x]), solution)
    scene = renderer._prepare_scene_data(
        jnp.array([0.0, 1.0]),
        jnp.stack([q, q]),
        base_offsets=None,
        color_config=None,
        routing=routing,
        routing_solution=solutions,
        static_spheres_positions=None,
        static_spheres_radii=None,
        static_spheres_colors=None,
        dynamic_spheres_positions=None,
        dynamic_spheres_radii=None,
        dynamics_spheres_colors=None,
    )
    assert scene.actuator_layers[0].points.shape == (1, 2, 1, 12, 3)
    expected = actuator_visual_layers(
        robot, q, jnp.linspace(0, 1, 6), routing=routing, routing_solution=solution
    )[0].points
    assert_allclose(scene.actuator_layers[0].points[0, 0], expected, atol=1e-12)


def test_viser_sequence_selects_matching_solution_for_each_frame(robot, monkeypatch):
    from rendering.test_viser_renderer import FakeViserActuatorServer

    from soromox.rendering.viser_renderer import SceneHandles, ViserRenderer

    routing = make_routing(StraightSpanRouting)
    qs = jnp.array([[0.2, 0, 0, -0.4, 0, 0], [0.8, 0, 0, -0.1, 0, 0]])
    solutions = eqx.filter_jit(jax.vmap(lambda q: routing.solve_routing(robot, q)))(qs)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = ViserRenderer(
        robot, auto_start=False, open_browser=False, config=renderer_config()
    )
    renderer._server = FakeViserActuatorServer()
    renderer._scene_handles = SceneHandles()
    renderer._clear_scene = Mock()
    renderer._setup_camera = Mock()
    renderer._setup_lighting = Mock()
    renderer._setup_playback_gui = lambda on_frame_seek: on_frame_seek(1)
    renderer.render_sequence(
        jnp.array([0.0, 1.0]),
        qs,
        routing=routing,
        routing_solution=solutions,
        blocking=False,
        autoplay=False,
        plot_configurations=False,
        plot_actuator_positions=False,
    )
    expected = actuator_visual_layers(
        robot,
        qs[1],
        jnp.linspace(0, 1, 6),
        routing=routing,
        routing_solution=jax.tree.map(lambda x: x[1], solutions),
    )[0].points[0]
    handle = renderer._scene_handles.actuator_lines[0]
    assert_allclose(handle.points[:, 0], expected[:-1], atol=1e-7)


def test_helicoidal_renderer_inherits_resolved_routing(monkeypatch):
    from rendering.test_viser_renderer import FakeViserActuatorServer

    from soromox.rendering import HelicoidalModuleVisual, HelicoidalViserRenderer
    from soromox.rendering.viser_renderer import SceneHandles

    robot = Helicoidal.from_segments(
        [
            GVSSegment(
                link=LinkSpec.circular(
                    length=0.1,
                    radius=0.02,
                    density=1000.0,
                    young_modulus=1e3,
                    shear_modulus=5e2,
                    reference_strain=[0, 0, 0, 1, 0, 0],
                ),
                joint=JointSpec.fixed(),
                basis=StrainBasisSpec(
                    type="monomial",
                    strain_selector=[0, 0, 1, 0, 0, 0],
                    basis_order=[0] * 6,
                ),
                num_gauss_points=5,
            )
        ],
        backend="jax",
    )
    routing = StraightSpanRouting(
        RoutingSpanParams([[0.0, 0.001, 0.0]], [[0.0, 0.001, 0.0]], (0,), (0,), (0,))
    )
    q = jnp.zeros(robot.num_coordinates)
    solution = eqx.filter_jit(routing.solve_routing)(robot, q)
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = HelicoidalViserRenderer(
        robot,
        modules=(HelicoidalModuleVisual(0),),
        auto_start=False,
        config=renderer_config(),
    )
    renderer._server = FakeViserActuatorServer()
    renderer._scene_handles = SceneHandles()
    renderer._build_actuator_geometry(
        q[None], np.zeros((1, 3)), 1, routing=routing, routing_solution=solution
    )
    handles = renderer._scene_handles.actuator_lines
    assert any(
        "routing/path_0" in key for key in renderer._scene_handles.actuator_line_keys
    )
    expected = actuator_visual_layers(
        robot,
        q,
        jnp.linspace(0, 1, renderer.num_points),
        routing=routing,
        routing_solution=solution,
    )[0].points[0]
    assert_allclose(handles[-1].points[:, 0], expected[:-1], atol=1e-7)


def test_matplotlib_sequence_reuses_solution_trajectory(robot, case, monkeypatch):
    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    renderer = MatplotlibRenderer(robot, config=renderer_config())
    draw = Mock()
    monkeypatch.setattr(renderer, "_animate_backbone_curves", draw)
    renderer.render_sequence(
        jnp.array([0.0, 1.0]),
        jnp.stack([q, q]),
        routing=routing,
        routing_solution=jax.tree.map(lambda x: jnp.stack([x, x]), solution),
    )
    assert draw.call_args.kwargs["actuator_layers"][0].points.shape == (1, 2, 1, 12, 2)


def test_opencv_sequence_reuses_solution_trajectory(robot, case, monkeypatch, tmp_path):
    from soromox.rendering import OpenCVPlanarRenderer, opencv_base

    routing, q, solution = case
    monkeypatch.setattr(visual_adapter, "_solve_visual_routing", no_solve)
    writer = Mock(stderr_log=None)
    monkeypatch.setattr(opencv_base, "FFmpegVideoWriter", Mock(return_value=writer))
    renderer = OpenCVPlanarRenderer(robot, config=renderer_config())
    renderer.render_sequence(
        jnp.array([0.0, 1.0]),
        jnp.stack([q, q]),
        routing=routing,
        routing_solution=jax.tree.map(lambda x: jnp.stack([x, x]), solution),
        record_path=str(tmp_path / "routing.mp4"),
    )
    assert writer.write.call_count == 2
    writer.close.assert_called_once()
    expected = renderer.render_frame(q, routing=routing, routing_solution=solution)
    np.testing.assert_array_equal(writer.write.call_args.args[0], expected)
