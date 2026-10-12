"""Public Warp-native GVS kinematics and dynamics building blocks.

The kernels and launch functions in this module operate entirely on
``warp.array`` objects. They do not require JAX and do not allocate output
buffers, making them suitable for external CUDA-graph-capturable integrators
such as a Newton solver. Callers own the buffers and must preserve the shapes,
dtypes, coordinate maps, and launch ordering described by each function.

Soromox system users should normally call :meth:`GVS.dynamics_terms
<soromox.systems.GVS.dynamics_terms>` or :meth:`GVS.forward_dynamics
<soromox.systems.GVS.forward_dynamics>` instead. The lower-level API is intended
for integration packages that need to compose the kinematics and dynamics
pipelines inside a larger Warp-native simulation loop.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GVSMixedChainData",
    "packed_mixed_chain_kernel",
    "GVSCellTermsData",
    "packed_cell_terms_kernel",
    "packed_cooperative_cell_terms_kernel",
    "homogeneous_cooperative_cell_terms_kernel",
    "GVSOperandSource",
    "GVSKinematicsOperands",
    "GVSKinematicsShapes",
    "GVSFloatingOperands",
    "GVSFloatingPipelineShapes",
    "GVSOperands",
    "GVSPipelineShapes",
    "GVSThreadlikeOperands",
    "GVSThreadlikeShapes",
    "homogeneous_cell_terms_kernel",
    "homogeneous_cooperative_joint_terms_kernel",
    "homogeneous_joint_terms_kernel",
    "homogeneous_gvs_cooperative_node_states_kernel",
    "homogeneous_gvs_cooperative_node_vjp_kernel",
    "homogeneous_gvs_jacobian_samples_kernel",
    "homogeneous_gvs_node_pose_twist_kernel",
    "homogeneous_gvs_node_poses_kernel",
    "homogeneous_gvs_node_states_kernel",
    "homogeneous_gvs_node_vjp_kernel",
    "homogeneous_gvs_pose_samples_kernel",
    "homogeneous_gvs_pose_twist_samples_kernel",
    "homogeneous_gvs_sample_vjp_kernel",
    "homogeneous_gvs_samples_kernel",
    "launch_homogeneous_cell_terms",
    "launch_homogeneous_cooperative_joint_terms",
    "launch_homogeneous_joint_terms",
    "launch_homogeneous_gvs_forward_kinematics",
    "launch_homogeneous_gvs_forward_kinematics_and_jvp",
    "launch_homogeneous_gvs_inertial_jacobian_vjp",
    "launch_homogeneous_gvs_inertial_jacobians",
    "launch_homogeneous_gvs_kinematics",
    "launch_homogeneous_gvs_dynamics_and_threadlike_actuation_force",
    "launch_homogeneous_floating_persistent_chain",
    "launch_homogeneous_persistent_chain",
    "homogeneous_floating_persistent_chain_kernel",
    "homogeneous_persistent_chain_kernel",
    "homogeneous_gvs_threadlike_force_kernel",
    "homogeneous_gvs_threadlike_force_block_kernel",
    "homogeneous_gvs_threadlike_matrix_kernel",
    "launch_homogeneous_gvs_threadlike_actuation_force",
    "launch_homogeneous_gvs_threadlike_actuation_matrix",
    "homogeneous_gvs_threadlike_strain_kernel",
    "gvs_dynamics_and_threadlike_actuation_force",
]

_PUBLIC_MODULES = {
    "GVSMixedChainData": "packed",
    "packed_mixed_chain_kernel": "packed",
    "GVSCellTermsData": "cell",
    "packed_cell_terms_kernel": "cell",
    "packed_cooperative_cell_terms_kernel": "cell",
    "homogeneous_cooperative_cell_terms_kernel": "cell",
    "GVSOperandSource": "operands",
    "GVSKinematicsOperands": "operands",
    "GVSKinematicsShapes": "operands",
    "GVSFloatingOperands": "operands",
    "GVSFloatingPipelineShapes": "operands",
    "GVSOperands": "operands",
    "GVSPipelineShapes": "operands",
    "GVSThreadlikeOperands": "actuation.operands",
    "GVSThreadlikeShapes": "actuation.operands",
    "homogeneous_cell_terms_kernel": "cell",
    "homogeneous_cooperative_joint_terms_kernel": "common.joints_cooperative",
    "homogeneous_joint_terms_kernel": "common.joints",
    "launch_homogeneous_cell_terms": "cell",
    "launch_homogeneous_cooperative_joint_terms": "common.joints_cooperative",
    "launch_homogeneous_joint_terms": "common.joints",
    "launch_homogeneous_floating_persistent_chain": "floating_chain",
    "launch_homogeneous_persistent_chain": "chain",
    "launch_homogeneous_gvs_dynamics_and_threadlike_actuation_force": "fused",
    "homogeneous_persistent_chain_kernel": "chain",
    "homogeneous_floating_persistent_chain_kernel": "floating_chain",
    "homogeneous_gvs_threadlike_force_kernel": "actuation.threadlike",
    "homogeneous_gvs_threadlike_force_block_kernel": "actuation.threadlike",
    "homogeneous_gvs_threadlike_matrix_kernel": "actuation.threadlike",
    "launch_homogeneous_gvs_threadlike_actuation_force": "actuation.threadlike",
    "launch_homogeneous_gvs_threadlike_actuation_matrix": "actuation.threadlike",
    "homogeneous_gvs_threadlike_strain_kernel": "actuation.threadlike",
    "gvs_dynamics_and_threadlike_actuation_force": "fused",
    "homogeneous_gvs_cooperative_node_states_kernel": "kinematics",
    "homogeneous_gvs_cooperative_node_vjp_kernel": "directional_kinematics",
    "homogeneous_gvs_jacobian_samples_kernel": "kinematics",
    "homogeneous_gvs_node_pose_twist_kernel": "directional_kinematics",
    "homogeneous_gvs_node_poses_kernel": "kinematics",
    "homogeneous_gvs_node_states_kernel": "kinematics",
    "homogeneous_gvs_node_vjp_kernel": "directional_kinematics",
    "homogeneous_gvs_pose_samples_kernel": "kinematics",
    "homogeneous_gvs_pose_twist_samples_kernel": "directional_kinematics",
    "homogeneous_gvs_sample_vjp_kernel": "directional_kinematics",
    "homogeneous_gvs_samples_kernel": "kinematics",
    "launch_homogeneous_gvs_forward_kinematics": "kinematics",
    "launch_homogeneous_gvs_forward_kinematics_and_jvp": "directional_kinematics",
    "launch_homogeneous_gvs_inertial_jacobian_vjp": "directional_kinematics",
    "launch_homogeneous_gvs_inertial_jacobians": "kinematics",
    "launch_homogeneous_gvs_kinematics": "kinematics",
}


def __getattr__(name: str) -> Any:
    """Load a public GVS Warp symbol without making Warp a core dependency.

    Args:
        name: Attribute requested from this package.

    Returns:
        The public operand type, kernel, or launch function named by ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the public GVS Warp API.
        ImportError: If a kernel is requested without ``warp-lang`` installed.
    """

    try:
        module_name = _PUBLIC_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    if module_name.startswith("common."):
        module = import_module(f"soromox.execution.warp.{module_name}")
    else:
        module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported Warp symbols in interactive discovery.

    Returns:
        Sorted module globals and public lazy exports.
    """

    return sorted((*globals(), *__all__))
