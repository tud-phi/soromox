"""Complete native Warp actuation, including joint guides and CUDA replay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_implicit_joint_routing import _guided_routing, _joint_robot
from test_relative_routing_kinematics import _robot

from soromox.actuation import RoutingSpanParams, StraightSpanRouting

jax.config.update("jax_enable_x64", True)


def _case(kind, floating):
    """Use shared intervals, interleaved paths, and an offset moving joint."""
    robot = _robot(kind, floating)
    offsets = jnp.array([[0, -0.003, 0], [0, 0.004, 0], [0, -0.003, 0], [0, 0.004, 0]])
    routing = StraightSpanRouting(
        RoutingSpanParams(
            offsets,
            offsets.at[:2, 1].add(0.001),
            (0, 1, 0, 1),
            (0, 0, 2, 2),
            (1, 1, 2, 2),
        )
    )
    internal = jnp.linspace(-0.12, 0.18, robot.num_internal_dofs)
    pose = (
        jnp.array([0.3, 0.4, -0.2])
        if robot.is_planar
        else jnp.array([0, 0, 0.3, 0.9, 0.4, -0.2, 0.1])
    )
    q = robot.pack_configuration(internal, base_pose=pose) if floating else internal
    return robot, routing, q


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["planar", "spatial", "gvs", "gvs_scaled"])
@pytest.mark.parametrize("floating", [False, True])
def test_native_pipeline(device, kind, floating, monkeypatch):
    """Match JAX forces, overwrite outputs, and replay changed state and tensions."""
    wp = pytest.importorskip("warp")
    if device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from soromox.execution.warp.actuation.threadlike import StraightSpanActuation

    robot, routing, q = _case(kind, floating)
    qs = jnp.stack([q, q.at[-1].add(0.03)])
    tensions = jnp.array([[1.2, -0.3], [0.1, 2.0]])
    reference = jax.jit(
        jax.vmap(lambda values: routing.actuation_matrix(robot, values, backend="jax"))
    )
    expected = np.asarray(reference(qs))
    pipeline = StraightSpanActuation(robot, routing, batch_size=2, device=device)
    values = wp.array(np.asarray(qs), dtype=wp.float64, device=device)
    loads = wp.array(np.asarray(tensions), dtype=wp.float64, device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native actuation must not evaluate JAX kinematics")

    monkeypatch.setattr(type(robot), "relative_segment_kinematics", forbidden)
    monkeypatch.setattr(type(robot), "joint_kinematics", forbidden)
    assert_allclose(pipeline.actuation_matrix(values).numpy(), expected, atol=2e-11)
    force = pipeline.actuation_force(values, loads)
    assert_allclose(
        force.numpy(), np.einsum("evp,ep->ev", expected, tensions), atol=2e-11
    )
    assert_allclose(force.numpy()[:, : robot.num_base_velocities], 0, atol=0)
    force.fill_(np.nan)
    if device == "cuda":
        with wp.ScopedCapture(device=device) as capture:
            pipeline.actuation_force(values, loads)
        # Reuse both model/workspace buffers and the captured graph with new inputs.
        wp.copy(values, wp.array(np.asarray(qs[::-1]), dtype=wp.float64, device=device))
        wp.copy(
            loads, wp.array(np.asarray(tensions * 2), dtype=wp.float64, device=device)
        )
        wp.capture_launch(capture.graph)
        target = np.einsum("evp,ep->ev", expected[::-1], tensions * 2)
    else:
        pipeline.actuation_force(values, loads)
        target = np.einsum("evp,ep->ev", expected, tensions)
    assert_allclose(force.numpy(), target, atol=2e-11)
    with pytest.raises(ValueError, match="tensions"):
        pipeline.actuation_force(
            values, wp.zeros((2, 1), dtype=wp.float64, device=device)
        )


@pytest.mark.parametrize("kind", ["planar", "spatial", "gvs", "gvs_scaled"])
def test_native_public_dispatch(kind, monkeypatch):
    """Public matrix and force APIs use native kinematics and support mapped loads."""
    pytest.importorskip("warp")
    robot, routing, q = _case(kind, True)
    tensions = jnp.array([1.2, -0.3])
    expected = routing.actuation_matrix(robot, q, backend="jax")
    force_reference = routing.actuation_force(robot, q, tensions, backend="jax")

    def forbidden(*args, **kwargs):
        raise AssertionError("Warp forward execution called JAX kinematics")

    with monkeypatch.context() as patch:
        patch.setattr(type(robot), "relative_segment_kinematics", forbidden)
        patch.setattr(type(robot), "joint_kinematics", forbidden)
        matrix = jax.jit(
            lambda values: routing.actuation_matrix(robot, values, backend="warp")
        )(q)
        forces = jax.jit(
            jax.vmap(
                lambda load: routing.actuation_force(robot, q, load, backend="warp")
            )
        )(jnp.stack([tensions, 2 * tensions]))
        solution = jax.jit(
            lambda values: routing.solve_routing(robot, values, backend="warp")
        )(q)
    assert_allclose(matrix, expected, atol=2e-11)
    assert_allclose(
        forces, jnp.stack([force_reference, 2 * force_reference]), atol=2e-11
    )
    assert_allclose(
        -routing.path_velocity_jacobian(robot, q, solution).T, expected, atol=2e-11
    )


def test_native_force_derivatives_and_empty_batches():
    """Differentiate forces with respect to state and loads and flatten nested maps."""
    pytest.importorskip("warp")
    robot, routing, q = _case("planar", False)
    tensions = jnp.array([1.0, 2.0])

    def force(values, loads, backend):
        return routing.actuation_force(robot, values, loads, backend=backend)

    for arg in (0, 1):
        actual = jax.jit(
            jax.jacrev(lambda values, loads: force(values, loads, "warp"), argnums=arg)
        )(q, tensions)
        expected = jax.jit(
            jax.jacrev(lambda values, loads: force(values, loads, "jax"), argnums=arg)
        )(q, tensions)
        assert_allclose(actual, expected, atol=1e-11)
    for shape in ((0,), (2, 3), (2, 0)):

        def evaluate(values, loads):
            return force(values, loads, "warp")

        for _ in shape:
            evaluate = jax.vmap(evaluate)
        actual = jax.jit(evaluate)(
            jnp.broadcast_to(q, (*shape, q.size)),
            jnp.broadcast_to(tensions, (*shape, tensions.size)),
        )
        assert actual.shape == (*shape, robot.num_velocities)
        assert_allclose(
            actual,
            jnp.broadcast_to(force(q, tensions, "jax"), actual.shape),
            atol=1e-11,
        )


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
@pytest.mark.parametrize("offset", [0.0, 0.004])
@pytest.mark.parametrize("floating", [False, True])
def test_native_joint_force_singularities(kind, offset, floating):
    """Preserve centered-guide invariance and reject collapsed moving connections."""
    pytest.importorskip("warp")
    robot = _joint_robot(kind, floating)
    routing = _guided_routing(offset, StraightSpanRouting)
    tensions = jnp.ones(routing.spans.num_paths)
    for motion in (0.0, 0.01):
        q = jnp.array([0.12, motion, -0.08])
        if floating:
            q = robot.pack_configuration(
                q, base_pose=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            )
        expected = routing.actuation_force(robot, q, tensions, backend="jax")
        actual = jax.jit(
            lambda values: routing.actuation_force(
                robot, values, tensions, backend="warp"
            )
        )(q)
        assert_allclose(actual, expected, atol=1e-11, equal_nan=True)


def test_cached_derivatives_honor_rejected_spans(straight_backend):
    """A cached derivative cannot override an explicitly rejected span."""
    import equinox as eqx

    robot, routing, q = _case("planar", False)
    solution = routing.solve_routing(robot, q, backend=straight_backend)
    rejected = eqx.tree_at(
        lambda value: value.spans[0].converged, solution, jnp.array(False)
    )
    jacobian = routing.path_length_jacobian(robot, q, rejected)
    assert jnp.all(jnp.isnan(jacobian[0]))
    assert jnp.all(jnp.isfinite(jacobian[1]))


def test_native_empty_paths_and_input_validation():
    """An empty routing produces zero force without launching any span kernels."""
    wp = pytest.importorskip("warp")
    from soromox.execution.warp.actuation.threadlike import StraightSpanActuation

    robot, _, q = _case("planar", False)
    routing = StraightSpanRouting(
        RoutingSpanParams(jnp.empty((0, 3)), jnp.empty((0, 3)), (), (), ())
    )
    pipeline = StraightSpanActuation(robot, routing, device="cpu")
    values = wp.array(np.asarray(q)[None], dtype=wp.float64, device="cpu")
    loads = wp.empty((1, 0), dtype=wp.float64, device="cpu")
    assert_allclose(pipeline.actuation_force(values, loads).numpy(), 0, atol=0)
    assert pipeline.actuation_matrix(values).shape == (1, robot.num_velocities, 0)
    assert_allclose(
        routing.actuation_force(robot, q, jnp.empty(0), backend="warp"), 0, atol=0
    )
    with pytest.raises(ValueError, match="q must"):
        pipeline.actuation_force(
            wp.zeros((1, q.size), dtype=wp.float32, device="cpu"), loads
        )
    with pytest.raises(ValueError, match="tensions"):
        routing.actuation_force(robot, q, jnp.ones(1), backend="warp")


def test_native_fp32_guide_storage():
    """Promote stored guide data before passing it to the FP64 native boundary."""
    pytest.importorskip("warp")
    robot, routing, q = _case("planar", False)
    routing = routing.with_params(
        routing.params.replace(
            start_offset=routing.spans.start_offset.astype(jnp.float32),
            end_offset=routing.spans.end_offset.astype(jnp.float32),
        )
    )
    assert_allclose(
        jax.jit(lambda values: routing.actuation_matrix(robot, values, backend="warp"))(
            q
        ),
        routing.actuation_matrix(robot, q, backend="jax"),
        atol=1e-11,
    )


def test_native_auto_falls_back_for_other_hosts(monkeypatch):
    """Preserve the generic SoftRobot JAX interface for hosts without native data."""
    robot, routing, q = _case("planar", False)

    class Host:
        backend = "auto"

        def __getattr__(self, name):
            if name in ("B_xi", "B_Z1"):
                raise AttributeError(name)
            return getattr(robot, name)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    expected = routing.actuation_matrix(robot, q, backend="jax")
    assert_allclose(routing.actuation_matrix(Host(), q), expected)
    with pytest.raises(NotImplementedError, match="not enabled"):
        routing.actuation_matrix(Host(), q, backend="warp")
