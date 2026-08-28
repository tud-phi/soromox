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
    "GVSOperandSource",
    "GVSKinematicsOperands",
    "GVSKinematicsShapes",
    "GVSOperands",
    "GVSPipelineShapes",
    "GVSThreadlikeOperands",
    "GVSThreadlikeShapes",
    "cell_terms_kernel",
    "cooperative_joint_terms_kernel",
    "joint_terms_kernel",
    "gvs_cooperative_node_states_kernel",
    "gvs_jacobian_samples_kernel",
    "gvs_node_poses_kernel",
    "gvs_node_states_kernel",
    "gvs_pose_samples_kernel",
    "gvs_samples_kernel",
    "launch_cell_terms",
    "launch_cooperative_joint_terms",
    "launch_joint_terms",
    "launch_forward_kinematics",
    "launch_forward_kinematics_and_jacobians",
    "launch_inertial_jacobians",
    "launch_gvs_forward_kinematics",
    "launch_gvs_inertial_jacobians",
    "launch_gvs_kinematics",
    "launch_gvs_dynamics_and_threadlike_actuation_force",
    "launch_persistent_chain",
    "persistent_chain_kernel",
    "gvs_threadlike_force_kernel",
    "gvs_threadlike_force_block_kernel",
    "gvs_threadlike_matrix_kernel",
    "launch_gvs_threadlike_actuation_force",
    "launch_gvs_threadlike_actuation_matrix",
    "prepare_gvs_threadlike_strain_kernel",
    "gvs_dynamics_and_threadlike_actuation_force",
]

_PUBLIC_MODULES = {
    "GVSOperandSource": "operands",
    "GVSKinematicsOperands": "operands",
    "GVSKinematicsShapes": "operands",
    "GVSOperands": "operands",
    "GVSPipelineShapes": "operands",
    "GVSThreadlikeOperands": "actuation.operands",
    "GVSThreadlikeShapes": "actuation.operands",
    "cell_terms_kernel": "cell",
    "cooperative_joint_terms_kernel": "common.joints_cooperative",
    "joint_terms_kernel": "common.joints",
    "launch_cell_terms": "cell",
    "launch_cooperative_joint_terms": "common.joints_cooperative",
    "launch_joint_terms": "common.joints",
    "launch_persistent_chain": "chain",
    "launch_gvs_dynamics_and_threadlike_actuation_force": "fused",
    "persistent_chain_kernel": "chain",
    "gvs_threadlike_force_kernel": "actuation.threadlike",
    "gvs_threadlike_force_block_kernel": "actuation.threadlike",
    "gvs_threadlike_matrix_kernel": "actuation.threadlike",
    "launch_gvs_threadlike_actuation_force": "actuation.threadlike",
    "launch_gvs_threadlike_actuation_matrix": "actuation.threadlike",
    "prepare_gvs_threadlike_strain_kernel": "actuation.threadlike",
    "gvs_dynamics_and_threadlike_actuation_force": "fused",
    "gvs_cooperative_node_states_kernel": "kinematics",
    "gvs_jacobian_samples_kernel": "kinematics",
    "gvs_node_poses_kernel": "kinematics",
    "gvs_node_states_kernel": "kinematics",
    "gvs_pose_samples_kernel": "kinematics",
    "gvs_samples_kernel": "kinematics",
    "launch_forward_kinematics": "kinematics",
    "launch_forward_kinematics_and_jacobians": "kinematics",
    "launch_inertial_jacobians": "kinematics",
    "launch_gvs_forward_kinematics": "kinematics",
    "launch_gvs_inertial_jacobians": "kinematics",
    "launch_gvs_kinematics": "kinematics",
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
        module = import_module(f"soromox.systems.execution.warp.{module_name}")
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
