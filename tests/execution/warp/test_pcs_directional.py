"""Matrix-free PCS inertial-Jacobian product tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.execution.warp.pcs.operands import (
    PCSKinematicsOperands,
    PCSKinematicsShapes,
)


def _warp_array(wp: Any, value: Any, *, device: Any) -> Any:
    """Copy one array-like value into dimension-preserving Warp storage."""

    array = np.asarray(value)
    dtype = wp.int32 if np.issubdtype(array.dtype, np.integer) else wp.float64
    return wp.array(array, dtype=dtype, device=device)


def _assert_directional_products_match_dense(
    model: Any,
    *,
    wp: Any,
    device: Any,
) -> None:
    operands = PCSKinematicsOperands.from_model(model)
    is_planar = operands.is_planar
    spatial_dim = 3 if is_planar else 6
    batch_size = 2
    dof_count = model.num_internal_dofs
    q_host = np.stack(
        (
            np.linspace(-0.031, 0.027, dof_count),
            np.linspace(0.023, -0.019, dof_count),
        )
    )
    qd_host = np.stack((0.41 * q_host[0], -0.37 * q_host[1]))
    segment_starts_host = np.asarray(operands.segment_starts)
    total_length = float(segment_starts_host[-1])
    boundary = float(segment_starts_host[1])
    samples = np.asarray(
        [
            total_length,
            0.39 * total_length,
            boundary,
            0.0,
            0.83 * total_length,
            0.39 * total_length,
        ]
    )
    s_host = np.stack((samples, samples[::-1]))

    q = _warp_array(wp, q_host, device=device)
    qd = _warp_array(wp, qd_host, device=device)
    s = _warp_array(wp, s_host, device=device)
    active_indices = _warp_array(wp, operands.active_strain_indices, device=device)
    active_scales = _warp_array(wp, operands.active_strain_scales, device=device)
    reference_strain = _warp_array(
        wp,
        np.asarray(operands.reference_strain).reshape(
            operands.num_segments,
            spatial_dim,
        ),
        device=device,
    )
    segment_lengths = _warp_array(wp, operands.segment_lengths, device=device)
    segment_starts = _warp_array(wp, operands.segment_starts, device=device)
    base_pose = _warp_array(wp, operands.base_pose, device=device)
    epsilons = wp.array(
        [operands.global_eps, operands.tangent_eps],
        dtype=wp.float64,
        device=device,
    )
    shapes = PCSKinematicsShapes.from_operands(
        operands,
        batch_size=batch_size,
        num_samples=s_host.shape[1],
    )

    from soromox.execution.warp.pcs import (
        launch_planar_forward_kinematics_and_jvp,
        launch_planar_inertial_jacobian_vjp,
        launch_planar_kinematics,
        launch_spatial_forward_kinematics_and_jvp,
        launch_spatial_inertial_jacobian_vjp,
        launch_spatial_kinematics,
    )

    dense_workspace = shapes.workspace()
    dense_segment_strain = wp.empty(
        dense_workspace["segment_strain"], dtype=wp.float64, device=device
    )
    dense_segment_pose = wp.empty(
        dense_workspace["segment_pose"], dtype=wp.float64, device=device
    )
    segment_jacobian = wp.empty(
        dense_workspace["segment_jacobian"], dtype=wp.float64, device=device
    )
    dense_poses = wp.empty(shapes.pose_output(), dtype=wp.float64, device=device)
    jacobians = wp.empty(shapes.jacobian_output(), dtype=wp.float64, device=device)
    common = (
        q,
        s,
        active_indices,
        active_scales,
        reference_strain,
        segment_lengths,
        segment_starts,
        base_pose,
    )
    if is_planar:
        launch_planar_kinematics(
            *common,
            epsilons,
            dense_segment_strain,
            dense_segment_pose,
            segment_jacobian,
            dense_poses,
            jacobians,
        )
    else:
        launch_spatial_kinematics(
            *common,
            dense_segment_strain,
            dense_segment_pose,
            segment_jacobian,
            dense_poses,
            jacobians,
        )

    jvp_workspace = shapes.jvp_workspace()
    jvp_segment_strain = wp.empty(
        jvp_workspace["segment_strain"], dtype=wp.float64, device=device
    )
    jvp_segment_pose = wp.empty(
        jvp_workspace["segment_pose"], dtype=wp.float64, device=device
    )
    segment_twist = wp.empty(
        jvp_workspace["segment_twist"], dtype=wp.float64, device=device
    )
    directional_poses = wp.empty(shapes.pose_output(), dtype=wp.float64, device=device)
    twists = wp.empty(shapes.twist_output(), dtype=wp.float64, device=device)
    directional_common = (
        q,
        qd,
        s,
        active_indices,
        active_scales,
        reference_strain,
        segment_lengths,
        segment_starts,
        base_pose,
    )
    if is_planar:
        launch_planar_forward_kinematics_and_jvp(
            *directional_common,
            epsilons,
            jvp_segment_strain,
            jvp_segment_pose,
            segment_twist,
            directional_poses,
            twists,
        )
    else:
        launch_spatial_forward_kinematics_and_jvp(
            *directional_common,
            jvp_segment_strain,
            jvp_segment_pose,
            segment_twist,
            directional_poses,
            twists,
        )
    jacobian_host = jacobians.numpy()
    expected_twists = np.einsum("bsij,bj->bsi", jacobian_host, qd_host)
    assert_allclose(
        directional_poses.numpy(), dense_poses.numpy(), rtol=0.0, atol=3e-12
    )
    assert_allclose(twists.numpy(), expected_twists, rtol=3e-11, atol=3e-12)

    rng = np.random.default_rng(59 + spatial_dim)
    wrench_host = rng.normal(size=shapes.twist_output())
    wrenches = _warp_array(wp, wrench_host, device=device)
    vjp_workspace = shapes.vjp_workspace()
    vjp_segment_strain = wp.empty(
        vjp_workspace["segment_strain"], dtype=wp.float64, device=device
    )
    vjp_segment_pose = wp.empty(
        vjp_workspace["segment_pose"], dtype=wp.float64, device=device
    )
    segment_wrench = wp.empty(
        vjp_workspace["segment_wrench"], dtype=wp.float64, device=device
    )
    generalized_force = wp.empty(
        shapes.generalized_force_output(), dtype=wp.float64, device=device
    )
    vjp_common = (
        q,
        s,
        wrenches,
        active_indices,
        active_scales,
        reference_strain,
        segment_lengths,
        segment_starts,
        base_pose,
    )
    if is_planar:
        launch_planar_inertial_jacobian_vjp(
            *vjp_common,
            epsilons,
            vjp_segment_strain,
            vjp_segment_pose,
            segment_wrench,
            generalized_force,
        )
    else:
        launch_spatial_inertial_jacobian_vjp(
            *vjp_common,
            vjp_segment_strain,
            vjp_segment_pose,
            segment_wrench,
            generalized_force,
        )
    expected_force = np.einsum("bsij,bsi->bj", jacobian_host, wrench_host)
    assert_allclose(generalized_force.numpy(), expected_force, rtol=3e-11, atol=3e-12)
    assert_allclose(
        np.einsum("bsi,bsi->b", expected_twists, wrench_host),
        np.einsum("bi,bi->b", qd_host, generalized_force.numpy()),
        rtol=3e-11,
        atol=3e-12,
    )

    wrenches.zero_()
    if is_planar:
        launch_planar_inertial_jacobian_vjp(
            *vjp_common,
            epsilons,
            vjp_segment_strain,
            vjp_segment_pose,
            segment_wrench,
            generalized_force,
        )
    else:
        launch_spatial_inertial_jacobian_vjp(
            *vjp_common,
            vjp_segment_strain,
            vjp_segment_pose,
            segment_wrench,
            generalized_force,
        )
    assert_allclose(generalized_force.numpy(), 0.0, rtol=0.0, atol=0.0)


def test_spatial_pcs_directional_products_match_dense_jacobians(
    make_pcs_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match spatial PCS JVPs, VJPs, and virtual-work duality."""

    wp = pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "pcs-directional-cache"))
    model = make_pcs_model("jax")
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    for device in devices:
        with wp.ScopedDevice(device):
            _assert_directional_products_match_dense(model, wp=wp, device=device)


def test_planar_pcs_directional_products_match_dense_jacobians(
    make_planar_pcs_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match PlanarPCS JVPs, VJPs, and virtual-work duality."""

    wp = pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH",
        str(tmp_path / "planar-pcs-directional-cache"),
    )
    model = make_planar_pcs_model("jax")
    devices = [wp.get_device("cpu")]
    if wp.is_cuda_available():
        devices.append(wp.get_device("cuda:0"))
    for device in devices:
        with wp.ScopedDevice(device):
            _assert_directional_products_match_dense(model, wp=wp, device=device)


def test_planar_floating_directional_composition_matches_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match planar floating JVP/VJP composition to augmented Jacobians."""

    wp = pytest.importorskip("warp")
    monkeypatch.setenv(
        "WARP_CACHE_PATH",
        str(tmp_path / "planar-floating-directional-cache"),
    )
    from soromox.execution.warp.common.floating_kinematics import (
        launch_planar_floating_jacobian_composition,
        launch_planar_floating_kinematics_jvp_composition,
        launch_planar_floating_vjp_assembly,
        launch_planar_floating_vjp_composition,
    )

    device = wp.get_device("cpu")
    wp.set_device(device)
    batch_size, sample_count, internal_dofs = 2, 5, 7
    base_pose_host = np.asarray([[0.31, 0.2, -0.1], [-0.27, -0.3, 0.4]])
    relative_pose_host = np.zeros((batch_size, sample_count, 3))
    for environment in range(batch_size):
        for sample in range(sample_count):
            relative_pose_host[environment, sample] = np.asarray(
                [
                    0.07 * (1 + environment + sample),
                    0.03 * sample,
                    -0.02 * environment + 0.01 * sample,
                ]
            )
    rng = np.random.default_rng(79)
    internal_jacobian_host = rng.normal(
        size=(batch_size, sample_count, 3, internal_dofs)
    )
    base_velocity_host = rng.normal(size=(batch_size, 3))
    internal_velocity_host = rng.normal(size=(batch_size, internal_dofs))
    full_velocity_host = np.concatenate(
        (base_velocity_host, internal_velocity_host), axis=1
    )
    internal_twist_host = np.einsum(
        "bsij,bj->bsi", internal_jacobian_host, internal_velocity_host
    )

    base_pose = _warp_array(wp, base_pose_host, device=device)
    relative_pose = _warp_array(wp, relative_pose_host, device=device)
    internal_jacobian = _warp_array(wp, internal_jacobian_host, device=device)
    full_jacobian = wp.empty(
        (batch_size, sample_count, 3, 3 + internal_dofs),
        dtype=wp.float64,
        device=device,
    )
    launch_planar_floating_jacobian_composition(
        base_pose,
        relative_pose,
        internal_jacobian,
        full_jacobian,
    )

    full_velocity = _warp_array(wp, full_velocity_host, device=device)
    internal_twist = _warp_array(wp, internal_twist_host, device=device)
    poses = wp.empty_like(relative_pose)
    twists = wp.empty((batch_size, sample_count, 3), dtype=wp.float64, device=device)
    launch_planar_floating_kinematics_jvp_composition(
        base_pose,
        full_velocity,
        relative_pose,
        internal_twist,
        poses,
        twists,
    )
    full_jacobian_host = full_jacobian.numpy()
    assert_allclose(
        twists.numpy(),
        np.einsum("bsij,bj->bsi", full_jacobian_host, full_velocity_host),
        rtol=3e-12,
        atol=3e-12,
    )

    wrench_host = rng.normal(size=(batch_size, sample_count, 3))
    wrenches = _warp_array(wp, wrench_host, device=device)
    relative_wrenches = wp.empty_like(wrenches)
    base_force = wp.empty((batch_size, 3), dtype=wp.float64, device=device)
    launch_planar_floating_vjp_composition(
        base_pose,
        relative_pose,
        wrenches,
        relative_wrenches,
        base_force,
    )
    internal_force_host = np.einsum(
        "bsij,bsi->bj", internal_jacobian_host, relative_wrenches.numpy()
    )
    internal_force = _warp_array(wp, internal_force_host, device=device)
    full_force = wp.empty(
        (batch_size, 3 + internal_dofs), dtype=wp.float64, device=device
    )
    launch_planar_floating_vjp_assembly(base_force, internal_force, full_force)
    assert_allclose(
        full_force.numpy(),
        np.einsum("bsij,bsi->bj", full_jacobian_host, wrench_host),
        rtol=3e-12,
        atol=3e-12,
    )
