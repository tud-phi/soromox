"""Architecture and public-surface tests for reusable Warp-native mechanics."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest


def test_gvs_public_names_are_discoverable_without_loading_warp() -> None:
    """Advertise stable symbols while preserving the optional dependency."""

    from soromox.execution.warp import gvs

    expected = {
        "GVSOperandSource",
        "GVSKinematicsOperands",
        "GVSKinematicsShapes",
        "GVSFloatingOperands",
        "GVSFloatingPipelineShapes",
        "GVSOperands",
        "GVSPipelineShapes",
        "GVSThreadlikeOperands",
        "GVSThreadlikeShapes",
        "cell_terms_kernel",
        "cooperative_joint_terms_kernel",
        "gvs_cooperative_node_states_kernel",
        "gvs_dynamics_and_threadlike_actuation_force",
        "joint_terms_kernel",
        "gvs_jacobian_samples_kernel",
        "gvs_node_poses_kernel",
        "gvs_node_states_kernel",
        "gvs_pose_samples_kernel",
        "gvs_samples_kernel",
        "gvs_threadlike_force_kernel",
        "gvs_threadlike_force_block_kernel",
        "gvs_threadlike_matrix_kernel",
        "floating_persistent_chain_kernel",
        "launch_cell_terms",
        "launch_cooperative_joint_terms",
        "launch_joint_terms",
        "launch_forward_kinematics",
        "launch_forward_kinematics_and_jacobians",
        "launch_floating_persistent_chain",
        "launch_gvs_forward_kinematics",
        "launch_gvs_dynamics_and_threadlike_actuation_force",
        "launch_gvs_inertial_jacobians",
        "launch_gvs_kinematics",
        "launch_gvs_threadlike_actuation_force",
        "launch_gvs_threadlike_actuation_matrix",
        "launch_inertial_jacobians",
        "launch_persistent_chain",
        "persistent_chain_kernel",
        "prepare_gvs_threadlike_strain_kernel",
    }
    assert expected == set(gvs.__all__)
    assert expected <= set(dir(gvs))


def test_public_gvs_launch_functions_have_documented_warp_contracts() -> None:
    """Expose direct preallocated-buffer launchers for external integrators."""

    pytest.importorskip("warp")
    from soromox.execution.warp.gvs import (
        cell_terms_kernel,
        cooperative_joint_terms_kernel,
        gvs_cooperative_node_states_kernel,
        gvs_jacobian_samples_kernel,
        gvs_node_poses_kernel,
        gvs_node_states_kernel,
        gvs_pose_samples_kernel,
        gvs_samples_kernel,
        gvs_threadlike_force_block_kernel,
        gvs_threadlike_force_kernel,
        gvs_threadlike_matrix_kernel,
        joint_terms_kernel,
        launch_cell_terms,
        launch_cooperative_joint_terms,
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_gvs_threadlike_actuation_force,
        launch_gvs_threadlike_actuation_matrix,
        launch_inertial_jacobians,
        launch_joint_terms,
        launch_persistent_chain,
        persistent_chain_kernel,
        prepare_gvs_threadlike_strain_kernel,
    )

    launchers = (
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_inertial_jacobians,
        launch_joint_terms,
        launch_cooperative_joint_terms,
        launch_cell_terms,
        launch_persistent_chain,
        launch_gvs_threadlike_actuation_matrix,
        launch_gvs_threadlike_actuation_force,
    )
    kernels = (
        gvs_cooperative_node_states_kernel,
        gvs_jacobian_samples_kernel,
        gvs_node_poses_kernel,
        gvs_node_states_kernel,
        gvs_pose_samples_kernel,
        gvs_samples_kernel,
        joint_terms_kernel,
        cooperative_joint_terms_kernel,
        cell_terms_kernel,
        persistent_chain_kernel,
        prepare_gvs_threadlike_strain_kernel,
        gvs_threadlike_matrix_kernel,
        gvs_threadlike_force_kernel,
        gvs_threadlike_force_block_kernel,
    )

    for launcher in launchers:
        documentation = inspect.getdoc(launcher)
        assert documentation is not None
        assert "Args:" in documentation
    assert launch_forward_kinematics is launch_gvs_forward_kinematics
    assert launch_inertial_jacobians is launch_gvs_inertial_jacobians
    assert launch_forward_kinematics_and_jacobians is launch_gvs_kinematics
    specialized = (
        launch_gvs_forward_kinematics,
        launch_gvs_inertial_jacobians,
        launch_gvs_kinematics,
        launch_joint_terms,
        launch_cooperative_joint_terms,
        launch_cell_terms,
        launch_persistent_chain,
    )
    for launcher in specialized:
        assert len(inspect.signature(launcher).parameters) >= 10
    for launcher in specialized[:3]:
        parameters = tuple(inspect.signature(launcher).parameters)
        assert parameters[:2] == ("q", "s")
    assert tuple(inspect.signature(launch_gvs_forward_kinematics).parameters)[:5] == (
        "q",
        "s",
        "joint_adjoint",
        "cell_adjoint",
        "base_transform",
    )
    state_prefix = (
        "q",
        "s",
        "joint_adjoint",
        "joint_tangent",
        "cell_adjoint",
        "cell_tangent_local",
        "base_transform",
        "link_local_to_global",
        "link_global_to_local",
    )
    assert (
        tuple(inspect.signature(launch_gvs_inertial_jacobians).parameters)[:9]
        == state_prefix
    )
    assert (
        tuple(inspect.signature(launch_gvs_kinematics).parameters)[:9] == state_prefix
    )
    for kernel in kernels:
        assert kernel is not None


def test_gvs_operand_contract_does_not_eagerly_load_kernel_modules() -> None:
    """Let orchestration inspect model operands without importing Warp kernels."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from soromox.execution.warp.gvs import GVSOperands; "
                "assert GVSOperands is not None; "
                "assert 'soromox.execution.warp.gvs.chain' "
                "not in sys.modules; "
                "assert 'soromox.execution.warp.gvs.cell' "
                "not in sys.modules; "
                "assert 'soromox.execution.warp.gvs.joint' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_pcs_public_names_are_discoverable_without_loading_kernels() -> None:
    """Advertise stable planar and spatial PCS integration symbols lazily."""

    from soromox.execution.warp import pcs

    expected = {
        "PCSOperandSource",
        "PCSKinematicsOperands",
        "PCSKinematicsShapes",
        "PCSFloatingOperands",
        "PCSFloatingPipelineShapes",
        "PCSOperands",
        "PCSPipelineShapes",
        "PCSThreadlikeOperands",
        "PCSThreadlikeShapes",
        "launch_forward_kinematics",
        "launch_forward_kinematics_and_jacobians",
        "launch_inertial_jacobians",
        "launch_planar_forward_kinematics",
        "launch_planar_dynamics_and_threadlike_actuation_force",
        "launch_planar_inertial_jacobians",
        "launch_planar_kinematics",
        "launch_planar_floating_persistent_chain",
        "launch_planar_local_operators",
        "launch_planar_persistent_chain",
        "launch_planar_threadlike_actuation_force",
        "launch_planar_threadlike_actuation_matrix",
        "launch_spatial_forward_kinematics",
        "launch_spatial_dynamics_and_threadlike_actuation_force",
        "launch_spatial_cooperative_inertial_jacobians",
        "launch_spatial_cooperative_kinematics",
        "launch_spatial_inertial_jacobians",
        "launch_spatial_kinematics",
        "launch_spatial_floating_persistent_chain",
        "launch_spatial_local_operators",
        "launch_spatial_persistent_chain",
        "launch_spatial_threadlike_actuation_force",
        "launch_spatial_threadlike_actuation_matrix",
        "planar_jacobian_samples_kernel",
        "planar_floating_persistent_chain_kernel",
        "planar_local_operators_kernel",
        "planar_persistent_chain_kernel",
        "planar_pose_samples_kernel",
        "planar_pose_segment_states_kernel",
        "planar_pose_step",
        "planar_samples_kernel",
        "planar_segment_states_kernel",
        "planar_threadlike_force_kernel",
        "planar_threadlike_matrix_kernel",
        "planar_dynamics_and_threadlike_actuation_force",
        "spatial_jacobian_samples_kernel",
        "spatial_floating_persistent_chain_kernel",
        "spatial_cooperative_segment_states_kernel",
        "spatial_local_operators_kernel",
        "spatial_operator_entry",
        "spatial_persistent_chain_kernel",
        "spatial_pose_samples_kernel",
        "spatial_pose_segment_states_kernel",
        "spatial_pose_step_entry",
        "spatial_samples_kernel",
        "spatial_segment_states_kernel",
        "spatial_threadlike_force_kernel",
        "spatial_threadlike_matrix_kernel",
        "spatial_dynamics_and_threadlike_actuation_force",
        "propagate_spatial_jacobian_column",
        "write_spatial_sample_jacobian",
    }
    assert expected == set(pcs.__all__)
    assert expected <= set(dir(pcs))


def test_public_pcs_launch_functions_have_documented_warp_contracts() -> None:
    """Expose preallocated-buffer launchers for both PCS dimensions."""

    pytest.importorskip("warp")
    from soromox.execution.warp.pcs import (
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_inertial_jacobians,
        launch_planar_forward_kinematics,
        launch_planar_inertial_jacobians,
        launch_planar_kinematics,
        launch_planar_local_operators,
        launch_planar_persistent_chain,
        launch_spatial_cooperative_inertial_jacobians,
        launch_spatial_cooperative_kinematics,
        launch_spatial_forward_kinematics,
        launch_spatial_inertial_jacobians,
        launch_spatial_kinematics,
        launch_spatial_local_operators,
        launch_spatial_persistent_chain,
        planar_jacobian_samples_kernel,
        planar_local_operators_kernel,
        planar_persistent_chain_kernel,
        planar_pose_samples_kernel,
        planar_pose_segment_states_kernel,
        planar_samples_kernel,
        planar_segment_states_kernel,
        spatial_cooperative_segment_states_kernel,
        spatial_jacobian_samples_kernel,
        spatial_local_operators_kernel,
        spatial_persistent_chain_kernel,
        spatial_pose_samples_kernel,
        spatial_pose_segment_states_kernel,
        spatial_samples_kernel,
        spatial_segment_states_kernel,
    )

    launchers = (
        launch_forward_kinematics,
        launch_forward_kinematics_and_jacobians,
        launch_inertial_jacobians,
        launch_planar_forward_kinematics,
        launch_planar_inertial_jacobians,
        launch_planar_kinematics,
        launch_planar_local_operators,
        launch_planar_persistent_chain,
        launch_spatial_forward_kinematics,
        launch_spatial_cooperative_inertial_jacobians,
        launch_spatial_cooperative_kinematics,
        launch_spatial_inertial_jacobians,
        launch_spatial_kinematics,
        launch_spatial_local_operators,
        launch_spatial_persistent_chain,
    )
    kernels = (
        planar_jacobian_samples_kernel,
        planar_local_operators_kernel,
        planar_persistent_chain_kernel,
        planar_pose_samples_kernel,
        planar_pose_segment_states_kernel,
        planar_samples_kernel,
        planar_segment_states_kernel,
        spatial_jacobian_samples_kernel,
        spatial_cooperative_segment_states_kernel,
        spatial_local_operators_kernel,
        spatial_persistent_chain_kernel,
        spatial_pose_samples_kernel,
        spatial_pose_segment_states_kernel,
        spatial_samples_kernel,
        spatial_segment_states_kernel,
    )
    for launcher in launchers:
        documentation = inspect.getdoc(launcher)
        assert documentation is not None
        assert "Args:" in documentation
    for launcher in launchers[:3]:
        parameters = tuple(inspect.signature(launcher).parameters)
        assert parameters[:2] == ("q", "s")
    specialized = launchers[3:]
    for launcher in specialized:
        assert len(inspect.signature(launcher).parameters) >= 10
    kinematics_launchers = (
        launch_planar_forward_kinematics,
        launch_planar_inertial_jacobians,
        launch_planar_kinematics,
        launch_spatial_forward_kinematics,
        launch_spatial_cooperative_inertial_jacobians,
        launch_spatial_cooperative_kinematics,
        launch_spatial_inertial_jacobians,
        launch_spatial_kinematics,
    )
    kinematics_prefix = (
        "q",
        "s",
        "active_indices",
        "active_scales",
        "reference_strain",
        "segment_lengths",
        "segment_starts",
    )
    for launcher in kinematics_launchers:
        parameters = tuple(inspect.signature(launcher).parameters)
        assert parameters[:7] == kinematics_prefix
    for kernel in kernels:
        assert kernel is not None


def test_pcs_operand_contract_does_not_eagerly_load_kernel_modules() -> None:
    """Inspect PCS operands without importing optional dimension kernels."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from soromox.execution.warp.pcs import PCSOperands; "
                "assert PCSOperands is not None; "
                "assert 'soromox.execution.warp.pcs.planar_kernels' "
                "not in sys.modules; "
                "assert 'soromox.execution.warp.pcs.spatial_kernels' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
