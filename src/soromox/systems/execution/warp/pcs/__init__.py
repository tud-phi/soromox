"""Public Warp-native PlanarPCS and PCS kinematics and dynamics building blocks.

The package exposes a shared runtime operand and workspace-shape contract plus
dimension-specific raw kernels and direct launch functions. All launch
functions accept caller-owned ``warp.array`` inputs and outputs; they do not
allocate buffers or invoke JAX, which allows an external integrator to compose
them into a larger CUDA graph.

Ordinary Soromox users should call the system methods. This lower-level surface
is intended for integration packages that need PlanarPCS or PCS kinematics and
dynamics inside another Warp-native simulation loop.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PCSOperandSource",
    "PCSKinematicsOperands",
    "PCSKinematicsShapes",
    "PCSOperands",
    "PCSPipelineShapes",
    "PCSThreadlikeOperands",
    "PCSThreadlikeShapes",
    "launch_planar_local_operators",
    "launch_planar_dynamics_and_threadlike_actuation_force",
    "launch_planar_persistent_chain",
    "launch_spatial_local_operators",
    "launch_spatial_dynamics_and_threadlike_actuation_force",
    "launch_spatial_persistent_chain",
    "launch_planar_threadlike_actuation_force",
    "launch_planar_threadlike_actuation_matrix",
    "launch_spatial_threadlike_actuation_force",
    "launch_spatial_threadlike_actuation_matrix",
    "launch_forward_kinematics",
    "launch_forward_kinematics_and_jacobians",
    "launch_inertial_jacobians",
    "planar_samples_kernel",
    "planar_pose_step",
    "planar_segment_states_kernel",
    "planar_local_operators_kernel",
    "planar_persistent_chain_kernel",
    "spatial_local_operators_kernel",
    "spatial_persistent_chain_kernel",
    "spatial_cooperative_segment_states_kernel",
    "spatial_samples_kernel",
    "spatial_operator_entry",
    "spatial_pose_step_entry",
    "spatial_segment_states_kernel",
    "launch_planar_kinematics",
    "launch_planar_forward_kinematics",
    "launch_planar_inertial_jacobians",
    "launch_spatial_kinematics",
    "launch_spatial_cooperative_inertial_jacobians",
    "launch_spatial_cooperative_kinematics",
    "launch_spatial_forward_kinematics",
    "launch_spatial_inertial_jacobians",
    "planar_jacobian_samples_kernel",
    "planar_pose_samples_kernel",
    "planar_pose_segment_states_kernel",
    "spatial_jacobian_samples_kernel",
    "spatial_pose_samples_kernel",
    "spatial_pose_segment_states_kernel",
    "propagate_spatial_jacobian_column",
    "write_spatial_sample_jacobian",
    "planar_threadlike_force_kernel",
    "planar_threadlike_matrix_kernel",
    "spatial_threadlike_force_kernel",
    "spatial_threadlike_matrix_kernel",
    "planar_dynamics_and_threadlike_actuation_force",
    "spatial_dynamics_and_threadlike_actuation_force",
]

_PUBLIC_MODULES = {
    "PCSOperandSource": "operands",
    "PCSKinematicsOperands": "operands",
    "PCSKinematicsShapes": "operands",
    "PCSOperands": "operands",
    "PCSPipelineShapes": "operands",
    "PCSThreadlikeOperands": "actuation.operands",
    "PCSThreadlikeShapes": "actuation.operands",
    "launch_planar_local_operators": "planar_kernels",
    "launch_planar_dynamics_and_threadlike_actuation_force": "fused",
    "launch_planar_persistent_chain": "planar_kernels",
    "launch_spatial_local_operators": "spatial_kernels",
    "launch_spatial_dynamics_and_threadlike_actuation_force": "fused",
    "launch_spatial_persistent_chain": "spatial_kernels",
    "planar_local_operators_kernel": "planar_kernels",
    "planar_persistent_chain_kernel": "planar_kernels",
    "spatial_local_operators_kernel": "spatial_kernels",
    "spatial_persistent_chain_kernel": "spatial_kernels",
    "launch_planar_threadlike_actuation_force": "actuation.threadlike",
    "launch_planar_threadlike_actuation_matrix": "actuation.threadlike",
    "launch_spatial_threadlike_actuation_force": "actuation.threadlike",
    "launch_spatial_threadlike_actuation_matrix": "actuation.threadlike",
    "planar_threadlike_force_kernel": "actuation.threadlike",
    "planar_threadlike_matrix_kernel": "actuation.threadlike",
    "spatial_threadlike_force_kernel": "actuation.threadlike",
    "spatial_threadlike_matrix_kernel": "actuation.threadlike",
    "planar_dynamics_and_threadlike_actuation_force": "fused",
    "spatial_dynamics_and_threadlike_actuation_force": "fused",
    "launch_forward_kinematics": "kinematics",
    "launch_forward_kinematics_and_jacobians": "kinematics",
    "launch_inertial_jacobians": "kinematics",
    "launch_planar_kinematics": "planar_kinematics",
    "launch_planar_forward_kinematics": "planar_kinematics",
    "launch_planar_inertial_jacobians": "planar_kinematics",
    "planar_jacobian_samples_kernel": "planar_kinematics",
    "planar_pose_samples_kernel": "planar_kinematics",
    "planar_pose_segment_states_kernel": "planar_kinematics",
    "planar_pose_step": "planar_kinematics",
    "planar_samples_kernel": "planar_kinematics",
    "planar_segment_states_kernel": "planar_kinematics",
    "launch_spatial_kinematics": "spatial_kinematics",
    "launch_spatial_cooperative_inertial_jacobians": "spatial_kinematics",
    "launch_spatial_cooperative_kinematics": "spatial_kinematics",
    "launch_spatial_forward_kinematics": "spatial_kinematics",
    "launch_spatial_inertial_jacobians": "spatial_kinematics",
    "spatial_jacobian_samples_kernel": "spatial_kinematics",
    "spatial_cooperative_segment_states_kernel": "spatial_kinematics",
    "spatial_pose_samples_kernel": "spatial_kinematics",
    "spatial_pose_segment_states_kernel": "spatial_kinematics",
    "spatial_operator_entry": "spatial_kinematics",
    "spatial_pose_step_entry": "spatial_kinematics",
    "spatial_samples_kernel": "spatial_kinematics",
    "spatial_segment_states_kernel": "spatial_kinematics",
    "propagate_spatial_jacobian_column": "spatial_kinematics",
    "write_spatial_sample_jacobian": "spatial_kinematics",
}


def __getattr__(name: str) -> Any:
    """Load a public PCS Warp symbol without making Warp a core dependency.

    Args:
        name: Attribute requested from this package.

    Returns:
        The public operand type, kernel, or launch function named by ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the public PCS Warp API.
        ImportError: If a kernel is requested without ``warp-lang`` installed.
    """

    try:
        module_name = _PUBLIC_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
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
