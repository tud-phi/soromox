"""Warp-native threadlike actuation contracts and production-gate tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    PCS,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)
from soromox.systems.execution.warp import loader
from soromox.systems.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
    GVSThreadlikeShapes,
)
from soromox.systems.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
    PCSThreadlikeShapes,
)


def _link(*, planar: bool, length: float) -> LinkSpec:
    """Create one deterministic circular continuum link."""

    reference_strain = (
        [0.0, 1.0, 0.0] if planar else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    return LinkSpec.circular(
        length=length,
        radius=0.015,
        density=1050.0,
        young_modulus=5e5,
        shear_modulus=2e5,
        material_damping_coefficient=5e-4,
        reference_strain=reference_strain,
    )


def _routing(
    num_segments: int, num_paths: int, *, planar: bool
) -> ThreadlikeRouting:
    """Create deterministic, non-identical linear routes over all segments."""

    angles = 2.0 * jnp.pi * (jnp.arange(num_paths) + 0.25) / max(num_paths, 1)
    zeros = jnp.zeros_like(angles)
    intercepts = jnp.stack(
        (
            zeros,
            0.012 * jnp.cos(angles),
            zeros if planar else 0.012 * jnp.sin(angles),
        ),
        axis=-1,
    )
    slopes = jnp.stack(
        (
            zeros,
            0.004 * jnp.sin(angles + 0.3),
            zeros if planar else 0.004 * jnp.cos(angles + 0.3),
        ),
        axis=-1,
    )
    return ThreadlikeRouting.linear(
        intercept=intercepts,
        slope=slopes,
        start_segment_index=(0,) * num_paths,
        end_segment_index=(num_segments - 1,) * num_paths,
    )


def _build_system(family: str, num_segments: int, num_paths: int):
    """Construct a small fully actuated continuum model for executor tests."""

    planar = family == "planar_pcs"
    actuator = ThreadlikeActuator.tendons(
        _routing(num_segments, num_paths, planar=planar)
    )
    if family == "planar_pcs":
        return PlanarPCS.from_links(
            [_link(planar=True, length=0.12) for _ in range(num_segments)],
            structure=PlanarPCSStructure(num_gauss_points=5),
            gravity=jnp.asarray([0.0, 9.81]),
            actuators=actuator,
            backend="jax",
        )
    if family == "pcs":
        return PCS.from_links(
            [_link(planar=False, length=0.1) for _ in range(num_segments)],
            structure=PCSStructure(num_gauss_points=5),
            gravity=jnp.asarray([0.0, 0.0, -9.81]),
            actuators=actuator,
            backend="jax",
        )
    if family == "gvs":
        segments = [
            GVSSegment(
                link=_link(planar=False, length=0.1),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=[1, 1, 1, 1, 1, 1],
                    basis_order=[0, 0, 0, 0, 0, 0],
                ),
                num_gauss_points=5,
            )
            for _ in range(num_segments)
        ]
        return GVS.from_segments(
            segments,
            gravity=jnp.asarray([0.0, 0.0, -9.81]),
            actuators=actuator,
            backend="jax",
        )
    raise ValueError(f"Unknown system family: {family}")


def _rebuild_with_actuators(model, actuators):
    """Reconstruct one benchmark model with an alternate static layout."""

    if isinstance(model, PlanarPCS):
        structure = PlanarPCSStructure(
            num_gauss_points=model.num_gauss_points,
            scale_rotational_basis_by_length=(
                model.scale_rotational_basis_by_length
            ),
        )
    elif isinstance(model, PCS):
        structure = PCSStructure(
            num_gauss_points=model.num_gauss_points,
            scale_rotational_basis_by_length=(
                model.scale_rotational_basis_by_length
            ),
        )
    else:
        structure = model.structure
    return type(model)(
        params=model.params,
        structure=structure,
        actuators=actuators,
        backend="jax",
    )


def _one_path_routing(
    *, planar: bool, start: int, end: int, offset: float, slope: float
) -> ThreadlikeRouting:
    return ThreadlikeRouting.linear(
        intercept=jnp.array([0.0, offset, 0.0 if planar else -0.5 * offset]),
        slope=jnp.array([0.0, slope, 0.0 if planar else 0.25 * slope]),
        start_segment_index=start,
        end_segment_index=end,
    )


def _build_with_quadrature(family: str, count: int):
    route = _one_path_routing(
        planar=family == "planar_pcs", start=0, end=1, offset=0.01, slope=0.012
    )
    actuator = ThreadlikeActuator.tendons(route)
    if family == "gvs":
        segments = [
            GVSSegment(
                link=_link(planar=False, length=0.1),
                joint=JointSpec(type="fixed"),
                basis=StrainBasisSpec(
                    type="legendre",
                    strain_selector=[1, 1, 1, 1, 1, 1],
                    basis_order=[0, 0, 0, 0, 0, 0],
                ),
                num_gauss_points=count,
            )
            for _ in range(2)
        ]
        return GVS.from_segments(segments, actuators=actuator, backend="jax")
    base = _build_system(family, 2, 1)
    structure_type = PlanarPCSStructure if family == "planar_pcs" else PCSStructure
    return type(base)(
        params=base.params,
        structure=structure_type(num_gauss_points=count),
        actuators=actuator,
        backend="jax",
    )


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_low_level_threadlike_executors_match_jax_on_cpu_and_cuda(
    family: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Match matrices, fused forces, column order, and ``A @ u``."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / f"{family}-threadlike"))
    model = _build_system(family, 2, 4)
    key = "gvs" if family == "gvs" else "pcs"
    q = jnp.stack(
        [
            jnp.linspace(-0.03, 0.02, model.num_dofs),
            jnp.linspace(0.01, -0.025, model.num_dofs),
            jnp.linspace(0.005, 0.031, model.num_dofs),
        ]
    )
    controls = jnp.reshape(
        jnp.linspace(-0.7, 1.1, q.shape[0] * model.num_actuators),
        (q.shape[0], model.num_actuators),
    )
    if family == "gvs":
        operands = GVSThreadlikeOperands.from_model(model)
    else:
        operands = PCSThreadlikeOperands.from_model(model)
    matrix = loader.execute_actuation(key, "matrix", operands, q)
    force = loader.execute_actuation(key, "force", operands, q, controls)
    expected_matrix = jax.vmap(model._actuation_matrix)(q)
    expected_force = jax.vmap(model._actuation_force)(
        q, controls, jnp.zeros_like(q)
    )
    assert_allclose(matrix, expected_matrix, rtol=2e-10, atol=2e-11)
    assert_allclose(force, expected_force, rtol=2e-10, atol=2e-11)
    assert_allclose(force, jnp.einsum("eda,ea->ed", matrix, controls))

    qd = jnp.linspace(-0.4, 0.3, model.num_dofs)
    assert_allclose(
        qd @ matrix[0] @ controls[0],
        qd @ force[0],
        rtol=2e-10,
        atol=2e-11,
    )


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_caller_owned_launchers_match_jax(
    family: str,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Fill caller-owned buffers through the raw Warp launcher contracts."""

    cache_path = tmp_path / f"{family}-{device.replace(':', '-')}-raw-launch"
    monkeypatch.setenv("WARP_CACHE_PATH", str(cache_path))
    wp = pytest.importorskip("warp")
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(cache_path))
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Warp CUDA device is unavailable.")

    model = _build_system(family, 2, 4)
    q = jnp.stack(
        (
            jnp.linspace(-0.03, 0.02, model.num_dofs),
            jnp.linspace(0.01, -0.025, model.num_dofs),
        )
    )
    controls = jnp.reshape(
        jnp.linspace(-0.7, 1.1, q.shape[0] * model.num_actuators),
        (q.shape[0], model.num_actuators),
    )

    def warp_array(value, dtype):
        return wp.array(np.asarray(value), dtype=dtype, device=device)

    with wp.ScopedDevice(device):
        if family == "gvs":
            from soromox.systems.execution.warp.gvs import (
                launch_gvs_threadlike_actuation_force,
                launch_gvs_threadlike_actuation_matrix,
            )

            operands = GVSThreadlikeOperands.from_model(model)
            shapes = GVSThreadlikeShapes.from_operands(
                operands, batch_size=q.shape[0]
            )
            routing = operands.routing
            common = (
                warp_array(q, wp.float64),
                warp_array(operands.link_local_to_global, wp.int32),
                warp_array(operands.link_basis, wp.float64),
                warp_array(operands.reference_strain, wp.float64),
                warp_array(operands.integration_points, wp.float64),
                warp_array(operands.integration_weights, wp.float64),
                warp_array(operands.segment_lengths, wp.float64),
                warp_array(operands.segment_starts, wp.float64),
                warp_array(routing.intercepts, wp.float64),
                warp_array(routing.slopes, wp.float64),
                warp_array(routing.start_segments, wp.int32),
                warp_array(routing.end_segments, wp.int32),
                warp_array(routing.coordinate_scales, wp.float64),
                warp_array(
                    [int(operands.scale_rotational_basis_by_length)], wp.int32
                ),
                warp_array([operands.global_eps], wp.float64),
            )
            matrix_output = wp.empty(
                shapes.matrix_output(), dtype=wp.float64, device=device
            )
            force_output = wp.empty(
                shapes.force_output(), dtype=wp.float64, device=device
            )
            launch_gvs_threadlike_actuation_matrix(*common, matrix_output)
            launch_gvs_threadlike_actuation_force(
                common[0],
                warp_array(controls, wp.float64),
                *common[1:],
                force_output,
            )
        else:
            from soromox.systems.execution.warp.pcs import (
                launch_planar_threadlike_actuation_force,
                launch_planar_threadlike_actuation_matrix,
                launch_spatial_threadlike_actuation_force,
                launch_spatial_threadlike_actuation_matrix,
            )

            operands = PCSThreadlikeOperands.from_model(model)
            shapes = PCSThreadlikeShapes.from_operands(
                operands, batch_size=q.shape[0]
            )
            routing = operands.routing
            spatial_dim = 3 if operands.is_planar else 6
            common = (
                warp_array(q, wp.float64),
                warp_array(operands.active_strain_indices, wp.int32),
                warp_array(operands.active_strain_scales, wp.float64),
                warp_array(
                    operands.reference_strain.reshape(
                        operands.num_segments, spatial_dim
                    ),
                    wp.float64,
                ),
                warp_array(operands.local_points, wp.float64),
                warp_array(operands.quadrature_weights, wp.float64),
                warp_array(operands.segment_lengths, wp.float64),
                warp_array(operands.segment_starts, wp.float64),
                warp_array(routing.intercepts, wp.float64),
                warp_array(routing.slopes, wp.float64),
                warp_array(routing.start_segments, wp.int32),
                warp_array(routing.end_segments, wp.int32),
                warp_array(routing.coordinate_scales, wp.float64),
                warp_array([operands.global_eps], wp.float64),
            )
            matrix_output = wp.empty(
                shapes.matrix_output(), dtype=wp.float64, device=device
            )
            force_output = wp.empty(
                shapes.force_output(), dtype=wp.float64, device=device
            )
            matrix_launcher = (
                launch_planar_threadlike_actuation_matrix
                if operands.is_planar
                else launch_spatial_threadlike_actuation_matrix
            )
            force_launcher = (
                launch_planar_threadlike_actuation_force
                if operands.is_planar
                else launch_spatial_threadlike_actuation_force
            )
            matrix_launcher(*common, matrix_output)
            force_launcher(
                common[0],
                warp_array(controls, wp.float64),
                *common[1:],
                force_output,
            )
        wp.synchronize_device(device)

    expected_matrix = jax.vmap(model._actuation_matrix)(q)
    expected_force = jax.vmap(model._actuation_force)(
        q, controls, jnp.zeros_like(q)
    )
    assert_allclose(matrix_output.numpy(), expected_matrix, rtol=2e-10, atol=2e-11)
    assert_allclose(force_output.numpy(), expected_force, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_ordered_groups_modalities_and_segment_spans(
    family: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Preserve ordered columns, modality scales, and inactive link spans."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / f"{family}-groups"))
    base = _build_system(family, 2, 1)
    planar = family == "planar_pcs"
    routes = (
        _one_path_routing(planar=planar, start=0, end=0, offset=0.008, slope=0.01),
        _one_path_routing(planar=planar, start=1, end=1, offset=-0.01, slope=-0.02),
        _one_path_routing(planar=planar, start=0, end=1, offset=0.012, slope=0.015),
        _one_path_routing(planar=planar, start=1, end=1, offset=-0.006, slope=0.005),
    )
    model = _rebuild_with_actuators(
        base,
        (
            ThreadlikeActuator.tendons(routes[0]),
            ThreadlikeActuator.push_rods(routes[1]),
            ThreadlikeActuator.muscles(routes[2]),
            ThreadlikeActuator.pressure_chambers(
                routes[3], effective_areas=jnp.array([2.5e-4])
            ),
        ),
    )
    q = jnp.stack(
        [
            jnp.linspace(-0.025, 0.015, model.num_dofs),
            jnp.linspace(0.02, -0.01, model.num_dofs),
        ]
    )
    controls = jnp.array([[1.0, -0.3, 0.7, 2.0], [-0.2, 0.9, 1.1, -0.4]])
    key = "gvs" if family == "gvs" else "pcs"
    operands = (
        GVSThreadlikeOperands.from_model(model)
        if family == "gvs"
        else PCSThreadlikeOperands.from_model(model)
    )
    actual_matrix = loader.execute_actuation(key, "matrix", operands, q)
    actual_force = loader.execute_actuation(
        key, "force", operands, q, controls
    )
    expected_matrix = jax.vmap(model._actuation_matrix)(q)
    assert_allclose(actual_matrix, expected_matrix, rtol=2e-10, atol=2e-11)
    assert_allclose(
        actual_force,
        jnp.einsum("eda,ea->ed", expected_matrix, controls),
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(operands.routing.coordinate_scales, [-1.0, 1.0, -1.0, 2.5e-4])


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_collapsed_tangents_emit_exact_zeros(
    family: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Match JAX safe normalization when routed tangents collapse."""

    pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / f"{family}-collapsed"))
    base = _build_system(family, 2, 1)
    route = _one_path_routing(
        planar=family == "planar_pcs", start=0, end=1, offset=0.01, slope=0.0
    )
    model = _rebuild_with_actuators(base, ThreadlikeActuator.tendons(route))
    model = model.update_link_params(
        reference_strain=jnp.zeros_like(model.params.link.reference_strain)
    )
    q = jnp.zeros((2, model.num_dofs), dtype=jnp.float64)
    key = "gvs" if family == "gvs" else "pcs"
    operands = (
        GVSThreadlikeOperands.from_model(model)
        if family == "gvs"
        else PCSThreadlikeOperands.from_model(model)
    )
    actual = loader.execute_actuation(key, "matrix", operands, q)
    assert_allclose(actual, 0.0, rtol=0.0, atol=0.0)
    assert_allclose(actual, jax.vmap(model._actuation_matrix)(q), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
@pytest.mark.parametrize("quadrature", [3, 7])
def test_runtime_quadrature_counts_match_jax(
    family: str,
    quadrature: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Keep kernel source shape-generic across supported quadrature counts."""

    if family == "gvs" and quadrature < 5:
        pytest.skip("GVS requires at least five Gauss points.")
    pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH", str(tmp_path / f"{family}-quadrature-{quadrature}")
    )
    model = _build_with_quadrature(family, quadrature)
    q = jnp.stack(
        [
            jnp.linspace(-0.02, 0.03, model.num_dofs),
            jnp.linspace(0.015, -0.01, model.num_dofs),
        ]
    )
    key = "gvs" if family == "gvs" else "pcs"
    operands = (
        GVSThreadlikeOperands.from_model(model)
        if family == "gvs"
        else PCSThreadlikeOperands.from_model(model)
    )
    actual = loader.execute_actuation(key, "matrix", operands, q)
    assert operands.num_quadrature == quadrature
    assert_allclose(
        actual,
        jax.vmap(model._actuation_matrix)(q),
        rtol=2e-10,
        atol=2e-11,
    )


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_threadlike_operand_and_shape_contracts(family: str) -> None:
    """Expose body quadrature, routing, maps, scales, and output shapes."""

    model = _build_system(family, 2, 4)
    if family == "gvs":
        operands = GVSThreadlikeOperands.from_model(model)
        shapes = GVSThreadlikeShapes.from_operands(operands, batch_size=7)
        assert operands.link_basis.shape[:3] == (2, 5, 6)
        assert operands.link_local_to_global.shape[0] == 2
    else:
        operands = PCSThreadlikeOperands.from_model(model)
        shapes = PCSThreadlikeShapes.from_operands(operands, batch_size=7)
        assert operands.local_points.shape == (2, 5)
        assert operands.active_strain_indices.shape[0] == 2
    assert operands.routing.intercepts.shape == (4, 3)
    assert operands.routing.slopes.shape == (4, 3)
    assert operands.routing.coordinate_scales.shape == (4,)
    assert shapes.matrix_output() == (7, model.num_dofs, 4)
    assert shapes.force_output() == (7, model.num_dofs)


@pytest.mark.parametrize("factory", [PCSThreadlikeShapes, GVSThreadlikeShapes])
@pytest.mark.parametrize(
    ("batch_size", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_threadlike_shapes_reject_invalid_batches(
    factory, batch_size, error
) -> None:
    model = _build_system("gvs" if factory is GVSThreadlikeShapes else "pcs", 2, 4)
    operands = (
        GVSThreadlikeOperands.from_model(model)
        if factory is GVSThreadlikeShapes
        else PCSThreadlikeOperands.from_model(model)
    )
    with pytest.raises(error, match="batch_size"):
        factory.from_operands(operands, batch_size=batch_size)


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_public_gate_falls_back_to_jax_and_explains_explicit_warp(family: str) -> None:
    """Keep production methods on JAX after a failed benchmark gate."""

    model = _build_system(family, 2, 4)
    q = jnp.linspace(-0.02, 0.03, model.num_dofs)
    controls = jnp.linspace(0.4, 1.0, model.num_actuators)
    assert_allclose(
        model.actuation_matrix(q, backend="auto"), model._actuation_matrix(q)
    )
    assert_allclose(
        model.actuation_force(q, controls, backend="auto"),
        model._actuation_force(q, controls, jnp.zeros_like(q)),
    )
    with pytest.raises(NotImplementedError, match="low-level Warp-native"):
        model.actuation_matrix(q, backend="warp")
    with pytest.raises(NotImplementedError, match="low-level Warp-native"):
        model.actuation_force(q, controls, backend="warp")


def test_warp_actuation_vmap_consolidates_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map environments into one canonical PCS executor invocation."""

    model = _build_system("pcs", 2, 4)
    q = jnp.zeros((5, model.num_dofs), dtype=jnp.float64)
    calls: list[tuple[int, ...]] = []

    def fake_batch(model_, q_):
        calls.append(q_.shape)
        return jnp.zeros(
            (q_.shape[0], model_.num_dofs, model_.num_actuators), dtype=q_.dtype
        )

    monkeypatch.setattr(loader, "_execute_pcs_actuation_matrix_batch", fake_batch)
    evaluator = loader.get_actuation_evaluator("pcs", "matrix")
    result = jax.vmap(evaluator, in_axes=(None, 0))(model, q)
    assert result.shape == (5, model.num_dofs, model.num_actuators)
    assert calls[-1] == (5, model.num_dofs)


def test_warp_force_vmap_consolidates_shared_and_mapped_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broadcast shared configurations or controls into one fused launch."""

    model = _build_system("pcs", 2, 4)
    q = jnp.zeros((5, model.num_dofs), dtype=jnp.float64)
    controls = jnp.ones((5, model.num_actuators), dtype=jnp.float64)
    qd = jnp.zeros_like(q)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_batch(model_, q_, controls_):
        calls.append((q_.shape, controls_.shape))
        return jnp.zeros((q_.shape[0], model_.num_dofs), dtype=q_.dtype)

    monkeypatch.setattr(loader, "_execute_pcs_actuation_force_batch", fake_batch)
    evaluator = loader.get_actuation_evaluator("pcs", "force")

    mapped_q = jax.vmap(evaluator, in_axes=(None, 0, None, 0))(
        model, q, controls[0], qd
    )
    assert mapped_q.shape == (5, model.num_dofs)
    assert calls[-1] == (
        (5, model.num_dofs),
        (5, model.num_actuators),
    )

    mapped_controls = jax.vmap(evaluator, in_axes=(None, None, 0, None))(
        model, q[0], controls, qd[0]
    )
    assert mapped_controls.shape == (5, model.num_dofs)
    assert calls[-1] == (
        (5, model.num_dofs),
        (5, model.num_actuators),
    )


@pytest.mark.parametrize("family", ["planar_pcs", "pcs", "gvs"])
def test_public_actuation_autodiff_uses_jax(family: str) -> None:
    """Preserve JVP, forward/reverse Jacobians, and control gradients."""

    model = _build_system(family, 2, 4)
    q = jnp.linspace(-0.02, 0.03, model.num_dofs)
    u = jnp.linspace(0.4, 1.0, model.num_actuators)
    direction = jnp.linspace(0.1, -0.2, model.num_dofs)
    def matrix(q_):
        return model.actuation_matrix(q_, backend="jax")
    public_jvp = jax.jvp(matrix, (q,), (direction,))[1]
    protected_jvp = jax.jvp(model._actuation_matrix, (q,), (direction,))[1]
    assert_allclose(public_jvp, protected_jvp)
    assert_allclose(
        jax.jacfwd(matrix)(q),
        jax.jacrev(matrix)(q),
        rtol=2e-9,
        atol=2e-10,
    )
    gradient = jax.grad(
        lambda controls: jnp.sum(
            model.actuation_force(q, controls, backend="jax")
        )
    )(u)
    expected = jnp.sum(model._actuation_matrix(q), axis=0)
    assert_allclose(gradient, expected, rtol=2e-9, atol=2e-10)


def test_actuation_batch_validation_rejects_empty_and_non_fp64() -> None:
    with pytest.raises(ValueError, match="non-empty batch"):
        loader._validate_actuation_batch(
            jnp.zeros((0, 2), dtype=jnp.float64), None, "test"
        )
    with pytest.raises(TypeError, match="requires float64"):
        loader._validate_actuation_batch(
            jnp.zeros((1, 2), dtype=jnp.float32), None, "test"
        )
    with pytest.raises(TypeError, match="controls"):
        loader._validate_actuation_batch(
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 1), dtype=jnp.float32),
            "test",
        )
