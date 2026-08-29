"""Public-surface tests for reusable Warp-native GVS mechanics."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest


def test_gvs_public_names_are_discoverable_without_loading_warp() -> None:
    """Advertise stable symbols while preserving the optional dependency."""

    from soromox.systems.execution.warp import gvs

    expected = {
        "GVSOperandSource",
        "GVSKinematicsOperands",
        "GVSKinematicsShapes",
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
        "launch_cell_terms",
        "launch_cooperative_joint_terms",
        "launch_joint_terms",
        "launch_forward_kinematics",
        "launch_forward_kinematics_and_jacobians",
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
    from soromox.systems.execution.warp.gvs import (
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


def test_operand_contract_does_not_eagerly_load_kernel_modules() -> None:
    """Let orchestration inspect model operands without importing Warp kernels."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from soromox.systems.execution.warp.gvs import GVSOperands; "
                "assert GVSOperands is not None; "
                "assert 'soromox.systems.execution.warp.gvs.chain' "
                "not in sys.modules; "
                "assert 'soromox.systems.execution.warp.gvs.cell' "
                "not in sys.modules; "
                "assert 'soromox.systems.execution.warp.gvs.joint' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
