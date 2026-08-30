"""Single lazy import boundary for optional Warp dynamics executors."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib import import_module
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.transforms import (
    make_actuation_evaluator,
    make_dynamics_actuation_evaluator,
    make_dynamics_evaluator,
    make_kinematics_evaluators,
)
from soromox.systems.execution.types import (
    AbscissaBatchedKinematicsEvaluator,
    ActuationForceEvaluator,
    ActuationMatrixEvaluator,
    ActuationModel,
    ActuationOperation,
    DynamicsActuationEvaluator,
    DynamicsActuationTerms,
    DynamicsEvaluator,
    DynamicsModel,
    DynamicsTerms,
    ForwardDynamicsModel,
    KinematicsEvaluator,
    KinematicsModel,
    KinematicsOperation,
    KinematicsResult,
    WarpExecutorKey,
)
from soromox.systems.execution.warp.gvs.actuation.operands import (
    GVSThreadlikeOperands,
)
from soromox.systems.execution.warp.gvs.operands import (
    GVSFloatingKinematicsOperands,
    GVSFloatingOperands,
    GVSKinematicsOperands,
    GVSOperands,
)
from soromox.systems.execution.warp.pcs.actuation.operands import (
    PCSThreadlikeOperands,
)
from soromox.systems.execution.warp.pcs.operands import (
    PCSFloatingKinematicsOperands,
    PCSFloatingOperands,
    PCSKinematicsOperands,
    PCSOperands,
)

WarpExecutor = Callable[[Any, Array, Array], DynamicsTerms]
WarpDynamicsActuationExecutor = Callable[
    [Any, Any, Array, Array, Array], DynamicsActuationTerms
]
WarpKinematicsExecutor = Callable[
    [Any, Array, Array, KinematicsOperation], KinematicsResult
]
WarpActuationExecutor = Callable[..., Array]

_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems.execution.warp.gvs.executor",
    "pcs": "soromox.systems.execution.warp.pcs.executor",
}
_FUSED_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems.execution.warp.gvs.fused_executor",
    "pcs": "soromox.systems.execution.warp.pcs.fused_executor",
}
_KINEMATICS_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems.execution.warp.gvs.kinematics_executor",
    "pcs": "soromox.systems.execution.warp.pcs.kinematics_executor",
}
_ACTUATION_EXECUTOR_MODULES: dict[WarpExecutorKey, str] = {
    "gvs": "soromox.systems.execution.warp.gvs.actuation.executor",
    "pcs": "soromox.systems.execution.warp.pcs.actuation.executor",
}


@cache
def load_executor(key: WarpExecutorKey) -> WarpExecutor:
    """Load and cache one family executor behind the optional-dependency boundary.

    Args:
        key: Registered system-family executor key.

    Returns:
        The family's batch-shaped ``execute_dynamics_terms`` function.

    Raises:
        KeyError: If ``key`` is not registered.
        ImportError: If the executor needs ``warp-lang`` and it is not installed.
        ModuleNotFoundError: If loading fails because another module is missing;
            unrelated import failures are deliberately preserved.
    """

    module_name = _EXECUTOR_MODULES[key]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The requested dynamics executor requires the optional "
                "'warp-lang' dependency. Install it with "
                "`pip install soromox[warp]`."
            ) from error
        raise
    return cast(WarpExecutor, module.execute_dynamics_terms)


@cache
def load_dynamics_actuation_executor(
    key: WarpExecutorKey,
) -> WarpDynamicsActuationExecutor:
    """Load one fused family executor behind the optional-Warp boundary."""

    module_name = _FUSED_EXECUTOR_MODULES[key]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The requested fused dynamics executor requires the optional "
                "'warp-lang' dependency. Install it with "
                "`pip install soromox[warp]`."
            ) from error
        raise
    return cast(
        WarpDynamicsActuationExecutor,
        module.execute_dynamics_and_threadlike_actuation_force,
    )


@cache
def load_kinematics_executor(key: WarpExecutorKey) -> WarpKinematicsExecutor:
    """Load one family kinematics executor behind the optional-Warp boundary.

    Args:
        key: Registered GVS or PCS executor family.

    Returns:
        The family's canonical batch-shaped kinematics executor.

    Raises:
        ImportError: If the optional ``warp-lang`` dependency is unavailable.
        KeyError: If ``key`` is not a registered family.
    """

    module_name = _KINEMATICS_EXECUTOR_MODULES[key]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The requested kinematics executor requires the optional "
                "'warp-lang' dependency. Install it with "
                "`pip install soromox[warp]`."
            ) from error
        raise
    return cast(WarpKinematicsExecutor, module.execute_kinematics)


@cache
def load_actuation_executor(
    key: WarpExecutorKey, operation: ActuationOperation
) -> WarpActuationExecutor:
    """Load one linear-threadlike Warp actuation executor lazily."""

    module_name = _ACTUATION_EXECUTOR_MODULES[key]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "warp":
            raise ImportError(
                "The requested actuation executor requires the optional "
                "'warp-lang' dependency. Install it with "
                "`pip install soromox[warp]`."
            ) from error
        raise
    return cast(
        WarpActuationExecutor, getattr(module, f"execute_actuation_{operation}")
    )


def execute_dynamics_terms(
    key: WarpExecutorKey,
    operands: Any,
    q: Array,
    qd: Array,
) -> DynamicsTerms:
    """Invoke a lazily resolved family executor with prepared operands.

    Args:
        key: Registered family executor key.
        operands: Family-specific immutable runtime operand bundle.
        q: Batched FP64 generalized coordinates.
        qd: Batched FP64 generalized velocities.

    Returns:
        Batched inertia, convective-force, and gravity-force terms.

    Raises:
        ImportError: If the optional Warp dependency is unavailable.
    """

    return load_executor(key)(operands, q, qd)


def execute_dynamics_and_threadlike_actuation_force(
    key: WarpExecutorKey,
    dynamics_operands: Any,
    actuation_operands: Any,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Invoke one fused family dynamics-and-direct-effort executor."""

    return load_dynamics_actuation_executor(key)(
        dynamics_operands,
        actuation_operands,
        q,
        qd,
        controls,
    )


def execute_kinematics(
    key: WarpExecutorKey,
    operands: Any,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Invoke a lazily resolved family kinematics executor.

    Args:
        key: Registered GVS or PCS executor family.
        operands: Family-specific runtime operand bundle.
        q: Batched configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Canonically batch-shaped kinematics output.
    """

    return load_kinematics_executor(key)(operands, q, s, operation)


def execute_actuation(
    key: WarpExecutorKey,
    operation: ActuationOperation,
    operands: Any,
    q: Array,
    controls: Array | None = None,
) -> Array:
    """Invoke a lazily loaded canonical threadlike actuation executor."""

    executor = load_actuation_executor(key, operation)
    if operation == "matrix":
        return executor(operands, q)
    return executor(operands, q, controls)


def _validate_batch(q: Array, qd: Array, family_name: str) -> None:
    """Validate restrictions shared by the production Warp executors.

    Args:
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.
        family_name: Name included in dtype diagnostics.

    Raises:
        ValueError: If the leading batch dimension is empty.
        TypeError: If either input is not FP64.

    Returns:
        None.
    """

    if q.shape[0] == 0:
        raise ValueError("The Warp executor requires a non-empty batch.")
    if q.dtype != jnp.float64 or qd.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} dynamics executor requires float64 q "
            f"and qd; got {q.dtype} and {qd.dtype}."
        )


def _validate_kinematics_batch(q: Array, s: Array, family_name: str) -> None:
    """Validate canonical Warp kinematics inputs.

    Args:
        q: Batched configurations with shape ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        family_name: Human-readable family name used in errors.

    Returns:
        None.

    Raises:
        ValueError: If either canonical batch dimension is empty.
        TypeError: If either array is not FP64.
    """

    if q.shape[0] == 0 or s.shape[1] == 0:
        raise ValueError("The Warp kinematics executor requires non-empty batches.")
    if q.dtype != jnp.float64 or s.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} kinematics executor requires float64 q "
            f"and s; got {q.dtype} and {s.dtype}."
        )


def _validate_actuation_batch(
    q: Array, controls: Array | None, family_name: str
) -> None:
    """Validate canonical FP64 threadlike matrix and force inputs."""

    if q.shape[0] == 0:
        raise ValueError("The Warp actuation executor requires a non-empty batch.")
    if q.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} actuation executor requires float64 q; "
            f"got {q.dtype}."
        )
    if controls is not None and controls.dtype != jnp.float64:
        raise TypeError(
            f"The Warp {family_name} actuation executor requires float64 "
            f"controls; got {controls.dtype}."
        )


def _gvs_model_block_dim(model: Any) -> int:
    """Return the model-owned GVS lane count, or one for Warp CPU."""

    if jax.default_backend() != "gpu":
        return 1
    return cast(int, model.backend_params.warp_block_dim)


@eqx.filter_jit
def _execute_gvs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Build GVS operands and execute one compiled batch-shaped pipeline.

    Args:
        model: GVS-compatible dynamics model.
        q: FP64 coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 velocities with the same shape as ``q``.

    Returns:
        Batched ``(B, Cqd, G)`` terms.
    """

    _validate_batch(q, qd, "GVS")
    block_dim = _gvs_model_block_dim(model)
    operands = (
        GVSFloatingOperands.from_model(model, block_dim=block_dim)
        if model.floating_base
        else GVSOperands.from_model(model, block_dim=block_dim)
    )
    return execute_dynamics_terms("gvs", operands, q, qd)


@eqx.filter_jit
def _execute_pcs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Build PCS operands and execute one compiled batch-shaped pipeline.

    Args:
        model: PlanarPCS- or PCS-compatible dynamics model.
        q: FP64 coordinates with shape ``(batch_size, num_dofs)``.
        qd: FP64 velocities with the same shape as ``q``.

    Returns:
        Batched ``(B, Cqd, G)`` terms.
    """

    _validate_batch(q, qd, "PCS")
    operands = (
        PCSFloatingOperands.from_model(model)
        if model.floating_base
        else PCSOperands.from_model(model)
    )
    return execute_dynamics_terms("pcs", operands, q, qd)


@eqx.filter_jit
def _execute_gvs_dynamics_actuation_batch(
    model: ForwardDynamicsModel,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Build GVS operands and execute one fused canonical batch."""

    _validate_batch(q, qd, "GVS")
    _validate_actuation_batch(q, controls, "GVS")
    floating = model.floating_base
    dynamics = (
        GVSFloatingOperands.from_model(
            model,
            block_dim=_gvs_model_block_dim(model),
        )
        if floating
        else GVSOperands.from_model(
            model,
            block_dim=_gvs_model_block_dim(model),
        )
    )
    actuation_model = model._fixed_base_robot_at_pose() if floating else model
    actuation = GVSThreadlikeOperands.from_model(actuation_model)
    return execute_dynamics_and_threadlike_actuation_force(
        "gvs", dynamics, actuation, q, qd, controls
    )


@eqx.filter_jit
def _execute_pcs_dynamics_actuation_batch(
    model: ForwardDynamicsModel,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Build PCS operands and execute one fused canonical batch."""

    _validate_batch(q, qd, "PCS")
    _validate_actuation_batch(q, controls, "PCS")
    floating = model.floating_base
    dynamics = (
        PCSFloatingOperands.from_model(model)
        if floating
        else PCSOperands.from_model(model)
    )
    actuation_model = model._fixed_base_robot_at_pose() if floating else model
    return execute_dynamics_and_threadlike_actuation_force(
        "pcs",
        dynamics,
        PCSThreadlikeOperands.from_model(actuation_model),
        q,
        qd,
        controls,
    )


def _call_gvs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Call the replaceable GVS batch boundary.

    Args:
        model: GVS-compatible dynamics model.
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.

    Returns:
        Batched ``(B, Cqd, G)`` terms.

    Notes:
        Keeping this one-line boundary outside the evaluator closure permits
        focused instrumentation without coupling model code to Warp modules.
    """

    return _execute_gvs_batch(model, q, qd)


def _call_pcs_batch(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
    """Call the replaceable PCS batch boundary.

    Args:
        model: PlanarPCS- or PCS-compatible dynamics model.
        q: Batched generalized coordinates.
        qd: Batched generalized velocities.

    Returns:
        Batched ``(B, Cqd, G)`` terms.

    Notes:
        Keeping this one-line boundary outside the evaluator closure permits
        focused instrumentation without coupling model code to Warp modules.
    """

    return _execute_pcs_batch(model, q, qd)


def _call_gvs_dynamics_actuation_batch(
    model: ForwardDynamicsModel,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Call the replaceable fused GVS batch boundary."""

    return _execute_gvs_dynamics_actuation_batch(model, q, qd, controls)


def _call_pcs_dynamics_actuation_batch(
    model: ForwardDynamicsModel,
    q: Array,
    qd: Array,
    controls: Array,
) -> DynamicsActuationTerms:
    """Call the replaceable fused PCS batch boundary."""

    return _execute_pcs_dynamics_actuation_batch(model, q, qd, controls)


_GVS_EVALUATOR = make_dynamics_evaluator(_call_gvs_batch, family_name="GVS")
_PCS_EVALUATOR = make_dynamics_evaluator(_call_pcs_batch, family_name="PCS")
_GVS_DYNAMICS_ACTUATION_EVALUATOR = make_dynamics_actuation_evaluator(
    _call_gvs_dynamics_actuation_batch,
    family_name="GVS",
)
_PCS_DYNAMICS_ACTUATION_EVALUATOR = make_dynamics_actuation_evaluator(
    _call_pcs_dynamics_actuation_batch,
    family_name="PCS",
)


@eqx.filter_jit
def _execute_gvs_kinematics_batch(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Build GVS operands and execute one canonical kinematics batch.

    Args:
        model: GVS model satisfying the neutral kinematics contract.
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Canonically batch-shaped GVS kinematics output.
    """

    _validate_kinematics_batch(q, s, "GVS")
    block_dim = _gvs_model_block_dim(model)
    operands = (
        GVSFloatingKinematicsOperands.from_model(model, block_dim=block_dim)
        if model.floating_base
        else GVSKinematicsOperands.from_model(model, block_dim=block_dim)
    )
    return execute_kinematics("gvs", operands, q, s, operation)


@eqx.filter_jit
def _execute_pcs_kinematics_batch(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Build PCS operands and execute one canonical kinematics batch.

    Args:
        model: PCS or PlanarPCS model satisfying the neutral contract.
        q: Batched active configurations ``(E, D)``.
        s: Per-environment abscissae with shape ``(E, N)``.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Canonically batch-shaped PCS kinematics output.
    """

    _validate_kinematics_batch(q, s, "PCS")
    operands = (
        PCSFloatingKinematicsOperands.from_model(model)
        if model.floating_base
        else PCSKinematicsOperands.from_model(model)
    )
    return execute_kinematics("pcs", operands, q, s, operation)


def _call_gvs_kinematics_batch(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Forward a canonical GVS batch through the lazy executor boundary."""

    return _execute_gvs_kinematics_batch(model, q, s, operation)


def _call_pcs_kinematics_batch(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Forward a canonical PCS batch through the lazy executor boundary."""

    return _execute_pcs_kinematics_batch(model, q, s, operation)


_KINEMATICS_EVALUATORS = {
    (key, operation): make_kinematics_evaluators(
        _call_gvs_kinematics_batch if key == "gvs" else _call_pcs_kinematics_batch,
        family_name="GVS" if key == "gvs" else "PCS",
        operation=operation,
    )
    for key in ("gvs", "pcs")
    for operation in ("pose", "jacobian", "both")
}


@eqx.filter_jit
def _execute_gvs_actuation_matrix_batch(model: ActuationModel, q: Array) -> Array:
    _validate_actuation_batch(q, None, "GVS")
    return execute_actuation(
        "gvs", "matrix", GVSThreadlikeOperands.from_model(model), q
    )


@eqx.filter_jit
def _execute_gvs_actuation_force_batch(
    model: ActuationModel, q: Array, controls: Array
) -> Array:
    _validate_actuation_batch(q, controls, "GVS")
    return execute_actuation(
        "gvs", "force", GVSThreadlikeOperands.from_model(model), q, controls
    )


@eqx.filter_jit
def _execute_pcs_actuation_matrix_batch(model: ActuationModel, q: Array) -> Array:
    _validate_actuation_batch(q, None, "PCS")
    return execute_actuation(
        "pcs", "matrix", PCSThreadlikeOperands.from_model(model), q
    )


@eqx.filter_jit
def _execute_pcs_actuation_force_batch(
    model: ActuationModel, q: Array, controls: Array
) -> Array:
    _validate_actuation_batch(q, controls, "PCS")
    return execute_actuation(
        "pcs", "force", PCSThreadlikeOperands.from_model(model), q, controls
    )


def _call_gvs_actuation_matrix_batch(model: ActuationModel, q: Array) -> Array:
    return _execute_gvs_actuation_matrix_batch(model, q)


def _call_gvs_actuation_force_batch(
    model: ActuationModel, q: Array, controls: Array
) -> Array:
    return _execute_gvs_actuation_force_batch(model, q, controls)


def _call_pcs_actuation_matrix_batch(model: ActuationModel, q: Array) -> Array:
    return _execute_pcs_actuation_matrix_batch(model, q)


def _call_pcs_actuation_force_batch(
    model: ActuationModel, q: Array, controls: Array
) -> Array:
    return _execute_pcs_actuation_force_batch(model, q, controls)


_ACTUATION_EVALUATORS = {
    ("gvs", "matrix"): make_actuation_evaluator(
        _call_gvs_actuation_matrix_batch, family_name="GVS", operation="matrix"
    ),
    ("gvs", "force"): make_actuation_evaluator(
        _call_gvs_actuation_force_batch, family_name="GVS", operation="force"
    ),
    ("pcs", "matrix"): make_actuation_evaluator(
        _call_pcs_actuation_matrix_batch, family_name="PCS", operation="matrix"
    ),
    ("pcs", "force"): make_actuation_evaluator(
        _call_pcs_actuation_force_batch, family_name="PCS", operation="force"
    ),
}


def get_dynamics_evaluator(key: WarpExecutorKey) -> DynamicsEvaluator:
    """Return the transform-aware evaluator for a registered family.

    Args:
        key: Registered family executor key.

    Returns:
        Single-point evaluator whose custom ``vmap`` invokes one batched
        executor and whose derivative rule uses JAX.
    """

    if key == "gvs":
        return _GVS_EVALUATOR
    return _PCS_EVALUATOR


def get_dynamics_actuation_evaluator(
    key: WarpExecutorKey,
) -> DynamicsActuationEvaluator:
    """Return one transform-aware fused dynamics-and-actuation evaluator."""

    if key == "gvs":
        return _GVS_DYNAMICS_ACTUATION_EVALUATOR
    return _PCS_DYNAMICS_ACTUATION_EVALUATOR


def get_actuation_evaluator(
    key: WarpExecutorKey, operation: ActuationOperation
) -> ActuationMatrixEvaluator | ActuationForceEvaluator:
    """Return one transform-aware linear-threadlike actuation evaluator."""

    return _ACTUATION_EVALUATORS[(key, operation)]


def get_kinematics_evaluator(
    key: WarpExecutorKey, operation: KinematicsOperation
) -> KinematicsEvaluator:
    """Return a transform-aware single-point kinematics evaluator.

    Args:
        key: Registered GVS or PCS executor family.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Single-point evaluator with custom batching and JVP rules.
    """

    return _KINEMATICS_EVALUATORS[(key, operation)][0]


def get_abscissa_batched_kinematics_evaluator(
    key: WarpExecutorKey, operation: KinematicsOperation
) -> AbscissaBatchedKinematicsEvaluator:
    """Return a transform-aware abscissa-batch kinematics evaluator.

    Args:
        key: Registered GVS or PCS executor family.
        operation: Select poses, inertial Jacobians, or both.

    Returns:
        Abscissa-batch evaluator with environment-batch fusion and JVP rules.
    """

    return _KINEMATICS_EVALUATORS[(key, operation)][1]


__all__ = [
    "WarpActuationExecutor",
    "WarpDynamicsActuationExecutor",
    "WarpExecutor",
    "WarpKinematicsExecutor",
    "WarpExecutorKey",
    "execute_actuation",
    "execute_dynamics_terms",
    "execute_dynamics_and_threadlike_actuation_force",
    "execute_kinematics",
    "get_abscissa_batched_kinematics_evaluator",
    "get_actuation_evaluator",
    "get_dynamics_evaluator",
    "get_dynamics_actuation_evaluator",
    "get_kinematics_evaluator",
    "load_actuation_executor",
    "load_executor",
    "load_dynamics_actuation_executor",
    "load_kinematics_executor",
]
