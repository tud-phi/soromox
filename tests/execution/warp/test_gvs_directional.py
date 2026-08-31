"""Matrix-free GVS inertial-Jacobian product tests."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.execution.warp.gvs.executor import _local_dynamics_terms
from soromox.execution.warp.gvs.operands import (
    GVSKinematicsOperands,
    GVSKinematicsShapes,
    GVSOperands,
)


def _warp_array(wp: Any, value: Any, *, device: Any) -> Any:
    array = np.asarray(value)
    dtype = wp.int32 if np.issubdtype(array.dtype, np.integer) else wp.float64
    return wp.array(array, dtype=dtype, device=device)


def test_directional_gvs_products_match_dense_jacobians(
    make_gvs_model: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match batched JVP/VJP products and their virtual-work duality."""

    wp = pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "gvs-directional-cache"))
    from soromox.execution.warp.gvs import (
        launch_forward_kinematics_and_jacobians,
        launch_forward_kinematics_and_jvp,
        launch_inertial_jacobian_vjp,
    )

    model = make_gvs_model("jax")
    device = wp.get_device("cpu")
    wp.set_device(device)
    batch_size = 2
    q = jnp.stack(
        (
            jnp.linspace(-0.025, 0.031, model.num_internal_dofs),
            jnp.linspace(0.019, -0.017, model.num_internal_dofs),
        )
    )
    qd = jnp.stack((0.37 * q[0], -0.43 * q[1]))
    total_length = float(np.sum(model.segment_lengths))
    boundary = float(model.segment_lengths[0])
    samples = np.asarray(
        [total_length, 0.37 * total_length, boundary, 0.0, 0.81 * total_length, 0.37 * total_length]
    )
    s = np.stack((samples, samples[::-1]))

    dynamics_operands = GVSOperands.from_model(model, block_dim=1)
    kinematics_operands = GVSKinematicsOperands.from_model(model, block_dim=1)
    terms = _local_dynamics_terms(dynamics_operands, q, qd)
    dof_count = model.num_internal_dofs
    max_dof = model.max_dof
    segment_count = model.num_segments
    cell_count_value = kinematics_operands.num_cells

    q_wp = _warp_array(wp, q, device=device)
    qd_wp = _warp_array(wp, qd, device=device)
    s_wp = _warp_array(wp, s, device=device)
    joint_adjoint = _warp_array(
        wp, np.asarray(terms[0]).reshape(batch_size * segment_count * 6, 6), device=device
    )
    joint_tangent = _warp_array(
        wp,
        np.asarray(terms[2]).reshape(batch_size * segment_count * 6, dof_count),
        device=device,
    )
    joint_velocity = _warp_array(
        wp, np.asarray(terms[4]).reshape(batch_size * segment_count * 6, 1), device=device
    )
    cell_rows = batch_size * segment_count * cell_count_value * 6
    cell_adjoint = _warp_array(
        wp, np.asarray(terms[5]).reshape(cell_rows, 6), device=device
    )
    cell_tangent = _warp_array(
        wp, np.asarray(terms[6]).reshape(cell_rows, max_dof), device=device
    )
    cell_velocity = _warp_array(
        wp, np.asarray(terms[7]).reshape(cell_rows, 1), device=device
    )
    base_transform = _warp_array(wp, kinematics_operands.base_transform, device=device)
    link_local_to_global = _warp_array(
        wp, kinematics_operands.link_local_to_global, device=device
    )
    link_global_to_local = _warp_array(
        wp, kinematics_operands.link_global_to_local, device=device
    )
    link_basis_rows = _warp_array(wp, kinematics_operands.link_basis_rows, device=device)
    link_reference_z1 = _warp_array(
        wp,
        np.asarray(kinematics_operands.link_reference_z1).reshape(
            segment_count * cell_count_value, 6
        ),
        device=device,
    )
    link_reference_z2 = _warp_array(
        wp,
        np.asarray(kinematics_operands.link_reference_z2).reshape(
            segment_count * cell_count_value, 6
        ),
        device=device,
    )
    segment_lengths = _warp_array(wp, kinematics_operands.segment_lengths, device=device)
    segment_starts = _warp_array(wp, kinematics_operands.segment_starts, device=device)
    integration_points = _warp_array(
        wp, kinematics_operands.integration_points, device=device
    )
    basis_type_index = _warp_array(
        wp, kinematics_operands.basis_type_index, device=device
    )
    basis_active_params = _warp_array(
        wp, kinematics_operands.basis_active_params, device=device
    )
    basis_order_params = _warp_array(
        wp, kinematics_operands.basis_order_params, device=device
    )
    magnus_points = _warp_array(wp, kinematics_operands.magnus_points, device=device)
    scale_rotations = wp.array(
        [int(kinematics_operands.scale_rotational_basis_by_length)],
        dtype=wp.int32,
        device=device,
    )
    epsilons = wp.array(
        [kinematics_operands.global_eps, kinematics_operands.tangent_eps],
        dtype=wp.float64,
        device=device,
    )
    cell_count = wp.array([cell_count_value], dtype=wp.int32, device=device)

    shapes = GVSKinematicsShapes.from_operands(
        kinematics_operands, batch_size=batch_size, num_samples=s.shape[1]
    )
    dense_workspace = shapes.workspace()
    dense_node_pose = wp.empty(dense_workspace["node_pose"], dtype=wp.float64, device=device)
    node_jacobian = wp.empty(
        dense_workspace["node_jacobian"], dtype=wp.float64, device=device
    )
    dense_poses = wp.empty(shapes.pose_output(), dtype=wp.float64, device=device)
    jacobians = wp.empty(shapes.jacobian_output(), dtype=wp.float64, device=device)
    common = (
        q_wp,
        s_wp,
        joint_adjoint,
        joint_tangent,
        cell_adjoint,
        cell_tangent,
        base_transform,
        link_local_to_global,
        link_global_to_local,
        link_basis_rows,
        link_reference_z1,
        link_reference_z2,
        segment_lengths,
        segment_starts,
        integration_points,
        basis_type_index,
        basis_active_params,
        basis_order_params,
        magnus_points,
        scale_rotations,
        epsilons,
        cell_count,
    )
    launch_forward_kinematics_and_jacobians(
        *common,
        1,
        dense_node_pose,
        node_jacobian,
        dense_poses,
        jacobians,
    )

    jvp_workspace = shapes.jvp_workspace()
    jvp_node_pose = wp.empty(jvp_workspace["node_pose"], dtype=wp.float64, device=device)
    node_twist = wp.empty(jvp_workspace["node_twist"], dtype=wp.float64, device=device)
    directional_poses = wp.empty(shapes.pose_output(), dtype=wp.float64, device=device)
    twists = wp.empty(shapes.twist_output(), dtype=wp.float64, device=device)
    launch_forward_kinematics_and_jvp(
        q_wp,
        qd_wp,
        s_wp,
        joint_adjoint,
        joint_velocity,
        cell_adjoint,
        cell_velocity,
        base_transform,
        link_local_to_global,
        link_basis_rows,
        link_reference_z1,
        link_reference_z2,
        segment_lengths,
        segment_starts,
        integration_points,
        basis_type_index,
        basis_active_params,
        basis_order_params,
        magnus_points,
        scale_rotations,
        epsilons,
        cell_count,
        jvp_node_pose,
        node_twist,
        directional_poses,
        twists,
    )
    jacobian_host = jacobians.numpy()
    expected_twists = np.einsum("bsij,bj->bsi", jacobian_host, np.asarray(qd))
    assert_allclose(directional_poses.numpy(), dense_poses.numpy(), rtol=0.0, atol=3e-12)
    assert_allclose(twists.numpy(), expected_twists, rtol=3e-11, atol=3e-12)

    rng = np.random.default_rng(41)
    wrench_host = rng.normal(size=shapes.twist_output())
    wrenches = _warp_array(wp, wrench_host, device=device)
    vjp_workspace = shapes.vjp_workspace()
    vjp_node_pose = wp.empty(vjp_workspace["node_pose"], dtype=wp.float64, device=device)
    node_wrench = wp.empty(vjp_workspace["node_wrench"], dtype=wp.float64, device=device)
    edge_wrench = wp.empty(vjp_workspace["edge_wrench"], dtype=wp.float64, device=device)
    generalized_force = wp.empty(
        shapes.generalized_force_output(), dtype=wp.float64, device=device
    )
    launch_inertial_jacobian_vjp(
        *common[:2],
        wrenches,
        *common[2:],
        1,
        vjp_node_pose,
        node_wrench,
        edge_wrench,
        generalized_force,
    )
    expected_force = np.einsum("bsij,bsi->bj", jacobian_host, wrench_host)
    assert_allclose(generalized_force.numpy(), expected_force, rtol=3e-11, atol=3e-12)
    assert_allclose(
        np.einsum("bsi,bsi->b", expected_twists, wrench_host),
        np.einsum("bi,bi->b", np.asarray(qd), generalized_force.numpy()),
        rtol=3e-11,
        atol=3e-12,
    )

    wrenches.zero_()
    launch_inertial_jacobian_vjp(
        *common[:2],
        wrenches,
        *common[2:],
        1,
        vjp_node_pose,
        node_wrench,
        edge_wrench,
        generalized_force,
    )
    assert_allclose(generalized_force.numpy(), 0.0, rtol=0.0, atol=0.0)


def test_spatial_floating_directional_composition_matches_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Match common floating JVP/VJP composition to augmented Jacobians."""

    wp = pytest.importorskip("warp")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "floating-directional-cache"))
    from soromox.execution.warp.common.floating_kinematics import (
        launch_spatial_floating_jacobian_composition,
        launch_spatial_floating_kinematics_jvp_composition,
        launch_spatial_floating_vjp_assembly,
        launch_spatial_floating_vjp_composition,
    )

    device = wp.get_device("cpu")
    wp.set_device(device)
    batch_size, sample_count, internal_dofs = 2, 5, 7
    base_pose_host = np.asarray(
        [
            [0.1, -0.2, 0.05, 0.97, 0.3, -0.1, 0.2],
            [-0.15, 0.04, 0.12, 0.98, -0.2, 0.4, -0.3],
        ]
    )
    relative_pose_host = np.zeros((batch_size, sample_count, 4, 4))
    for environment in range(batch_size):
        for sample in range(sample_count):
            angle = 0.09 * (1 + environment + sample)
            cosine, sine = np.cos(angle), np.sin(angle)
            relative_pose_host[environment, sample] = np.asarray(
                [
                    [cosine, -sine, 0.0, 0.03 * sample],
                    [sine, cosine, 0.0, -0.02 * environment],
                    [0.0, 0.0, 1.0, 0.01 * (sample + environment)],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
    rng = np.random.default_rng(73)
    internal_jacobian_host = rng.normal(
        size=(batch_size, sample_count, 6, internal_dofs)
    )
    base_velocity_host = rng.normal(size=(batch_size, 6))
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
        (batch_size, sample_count, 6, 6 + internal_dofs),
        dtype=wp.float64,
        device=device,
    )
    launch_spatial_floating_jacobian_composition(
        base_pose, relative_pose, internal_jacobian, full_jacobian
    )

    full_velocity = _warp_array(wp, full_velocity_host, device=device)
    internal_twist = _warp_array(wp, internal_twist_host, device=device)
    poses = wp.empty_like(relative_pose)
    twists = wp.empty(
        (batch_size, sample_count, 6), dtype=wp.float64, device=device
    )
    launch_spatial_floating_kinematics_jvp_composition(
        base_pose, full_velocity, relative_pose, internal_twist, poses, twists
    )
    full_jacobian_host = full_jacobian.numpy()
    assert_allclose(
        twists.numpy(),
        np.einsum("bsij,bj->bsi", full_jacobian_host, full_velocity_host),
        rtol=3e-12,
        atol=3e-12,
    )

    wrench_host = rng.normal(size=(batch_size, sample_count, 6))
    wrenches = _warp_array(wp, wrench_host, device=device)
    relative_wrenches = wp.empty_like(wrenches)
    base_force = wp.empty((batch_size, 6), dtype=wp.float64, device=device)
    launch_spatial_floating_vjp_composition(
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
        (batch_size, 6 + internal_dofs), dtype=wp.float64, device=device
    )
    launch_spatial_floating_vjp_assembly(base_force, internal_force, full_force)
    assert_allclose(
        full_force.numpy(),
        np.einsum("bsij,bsi->bj", full_jacobian_host, wrench_host),
        rtol=3e-12,
        atol=3e-12,
    )
