"""Execution backends for accelerated system operations.

This public package defines backend selection and transformation semantics used
by supported Soromox system classes. Most users only need the ``backend``
constructor argument; :class:`ExecutionBackend` names its accepted values and
the family-specific backend parameter objects provide optional advanced
tuning.

Integrations that need direct Warp execution can import stable family-specific
building blocks from :mod:`soromox.execution.warp`. Importing this
module does not import the optional ``warp-lang`` dependency; Warp is loaded
only when an integration explicitly enters that namespace or selects a Warp
executor.
"""

from soromox.execution.catalog import (
    GVS_ACTUATION,
    GVS_DYNAMICS,
    GVS_KINEMATICS,
    PCS_ACTUATION,
    PCS_DYNAMICS,
    PCS_KINEMATICS,
    PLANAR_PCS_ACTUATION,
)
from soromox.execution.config import (
    DEFAULT_GVS_BLOCK_DIM,
    DEFAULT_GVS_LARGE_BLOCK_DIM,
    DEFAULT_PCS_BLOCK_DIM,
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    GVSBackendParams,
    PCSBackendParams,
    default_gvs_block_dim,
)
from soromox.execution.dispatch import (
    dispatch_actuation_force,
    dispatch_actuation_matrix,
    dispatch_dynamics_terms,
    dispatch_fused_dynamics_actuation_force,
    dispatch_kinematics,
    dispatch_kinematics_abscissa_batched,
)
from soromox.execution.transforms import (
    evaluate_forward_dynamics,
    evaluate_forward_kinematics,
    evaluate_inertial_jacobian,
    make_actuation_evaluator,
    make_dynamics_actuation_evaluator,
    make_dynamics_evaluator,
    make_kinematics_evaluators,
)
from soromox.execution.types import (
    AbscissaBatchedKinematicsEvaluator,
    ActuationCapabilities,
    ActuationForceEvaluator,
    ActuationMatrixEvaluator,
    ActuationModel,
    ActuationOperation,
    DynamicsActuationEvaluator,
    DynamicsActuationTerms,
    DynamicsCapabilities,
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
    ForwardDynamicsModel,
    KinematicsCapabilities,
    KinematicsEvaluator,
    KinematicsModel,
    KinematicsOperation,
    KinematicsResult,
    WarpExecutorKey,
)

__all__ = [
    "AbscissaBatchedKinematicsEvaluator",
    "ActuationCapabilities",
    "ActuationForceEvaluator",
    "ActuationMatrixEvaluator",
    "ActuationModel",
    "ActuationOperation",
    "DynamicsCapabilities",
    "DynamicsActuationEvaluator",
    "DynamicsActuationTerms",
    "DynamicsEvaluator",
    "DynamicsModel",
    "DynamicsTerms",
    "ExecutionBackend",
    "ForwardDynamicsModel",
    "GVS_ACTUATION",
    "GVSBackendParams",
    "GVS_DYNAMICS",
    "GVS_KINEMATICS",
    "KinematicsCapabilities",
    "KinematicsEvaluator",
    "KinematicsModel",
    "KinematicsOperation",
    "KinematicsResult",
    "PCS_ACTUATION",
    "PCS_DYNAMICS",
    "PCS_KINEMATICS",
    "PCSBackendParams",
    "PLANAR_PCS_ACTUATION",
    "DEFAULT_GVS_BLOCK_DIM",
    "DEFAULT_GVS_LARGE_BLOCK_DIM",
    "DEFAULT_PCS_BLOCK_DIM",
    "DEFAULT_PLANAR_PCS_BLOCK_DIM",
    "WarpExecutorKey",
    "dispatch_actuation_force",
    "dispatch_actuation_matrix",
    "dispatch_dynamics_terms",
    "dispatch_fused_dynamics_actuation_force",
    "dispatch_kinematics",
    "dispatch_kinematics_abscissa_batched",
    "default_gvs_block_dim",
    "evaluate_forward_dynamics",
    "evaluate_forward_kinematics",
    "evaluate_inertial_jacobian",
    "make_actuation_evaluator",
    "make_dynamics_actuation_evaluator",
    "make_dynamics_evaluator",
    "make_kinematics_evaluators",
]
