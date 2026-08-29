"""Backend-neutral validation and selection for system execution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.transforms import (
    evaluate_forward_kinematics,
    evaluate_inertial_jacobian,
)
from soromox.systems.execution.types import (
    ActuationCapabilities,
    ActuationModel,
    DynamicsActuationTerms,
    DynamicsCapabilities,
    DynamicsModel,
    DynamicsTerms,
    ExecutionBackend,
    ForwardDynamicsModel,
    KinematicsCapabilities,
    KinematicsModel,
    KinematicsOperation,
    KinematicsResult,
)
from soromox.systems.execution.warp.actuation.threadlike import (
    supports_linear_threadlike_force,
    supports_linear_threadlike_matrix,
)
from soromox.systems.execution.warp.loader import (
    get_abscissa_batched_kinematics_evaluator,
    get_actuation_evaluator,
    get_dynamics_actuation_evaluator,
    get_dynamics_evaluator,
    get_kinematics_evaluator,
)


def _select_backend(
    model: DynamicsModel,
    requested: ExecutionBackend | None,
    capabilities: DynamicsCapabilities | KinematicsCapabilities | ActuationCapabilities,
    *,
    warp_supported: bool,
) -> ExecutionBackend:
    """Resolve a requested backend against device and model capabilities.

    Args:
        model: System supplying the configured backend and static dimensions.
        requested: Optional per-call override. ``None`` uses ``model.backend``.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this concrete model instance is implemented by
            the family's Warp executor.

    Returns:
        The concrete backend to execute, either ``"jax"`` or ``"warp"``.

    Raises:
        ValueError: If the requested or configured backend name is invalid.
        NotImplementedError: If Warp was requested explicitly for an unsupported
            device or model instance.
    """

    configured = model.backend if requested is None else requested
    if configured not in ("auto", "jax", "warp"):
        raise ValueError(
            f"backend must be one of 'auto', 'jax', or 'warp', got {requested!r}."
        )

    device = jax.default_backend()
    selected: ExecutionBackend = configured
    if selected == "auto":
        selected = "warp" if device == "gpu" else "jax"
    warp_device_supported = device == "gpu" or (
        device == "cpu" and capabilities.warp_cpu_supported
    )
    if selected == "warp" and not warp_device_supported:
        if configured == "warp":
            raise NotImplementedError(
                f"The Warp {capabilities.family_name} executor is not supported "
                f"on the active {device.upper()} device."
            )
        selected = "jax"
    if model.num_dofs == 0:
        selected = "jax"

    if selected == "warp" and not warp_supported:
        if configured == "warp":
            raise NotImplementedError(
                f"The Warp {capabilities.family_name} executor is "
                "not enabled for this system."
            )
        selected = "jax"
    return selected


def _explicit_backend(
    model: ActuationModel, requested: ExecutionBackend | None
) -> ExecutionBackend:
    """Return the configured name before automatic device resolution."""

    return model.backend if requested is None else requested


def dispatch_actuation_matrix(
    model: ActuationModel,
    q: Array,
    *,
    backend: ExecutionBackend | None,
    capabilities: ActuationCapabilities,
) -> Array:
    """Dispatch one scalar linear-threadlike actuation-matrix request."""

    q = jnp.asarray(q)
    if q.shape != (model.num_dofs,):
        raise ValueError(f"q must have shape ({model.num_dofs},), got {q.shape}.")
    eligible = supports_linear_threadlike_matrix(model)
    configured = _explicit_backend(model, backend)
    if configured == "warp" and not eligible:
        raise NotImplementedError(
            f"Warp {capabilities.family_name} actuation_matrix requires only "
            "built-in linear ThreadlikeActuator transmissions."
        )
    if configured == "warp" and eligible and not capabilities.matrix_enabled:
        raise NotImplementedError(
            f"High-level Warp {capabilities.family_name} actuation_matrix "
            "dispatch did not pass the GPU production gate; only the public "
            "low-level Warp-native threadlike integration API is available."
        )
    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=eligible and capabilities.matrix_enabled,
    )
    if selected == "jax":
        return model._actuation_matrix(q)
    evaluator = get_actuation_evaluator(capabilities.warp_executor, "matrix")
    return evaluator(model, q)  # type: ignore[operator]


def dispatch_actuation_force(
    model: ActuationModel,
    q: Array,
    u: Array,
    qd: Array | None,
    *,
    backend: ExecutionBackend | None,
    capabilities: ActuationCapabilities,
) -> Array:
    """Dispatch one scalar fused linear-threadlike generalized-force request."""

    q = jnp.asarray(q)
    u = jnp.asarray(u)
    if q.shape != (model.num_dofs,):
        raise ValueError(f"q must have shape ({model.num_dofs},), got {q.shape}.")
    if u.shape != (model.num_actuators,):
        raise ValueError(f"u must have shape ({model.num_actuators},), got {u.shape}.")
    if qd is None:
        qd = jnp.zeros_like(q)
    else:
        qd = jnp.asarray(qd)
        if qd.shape != q.shape:
            raise ValueError(f"qd must have shape {q.shape}, got {qd.shape}.")

    eligible = supports_linear_threadlike_force(model)
    configured = _explicit_backend(model, backend)
    if configured == "warp" and not eligible:
        raise NotImplementedError(
            f"Warp {capabilities.family_name} actuation_force requires only "
            "built-in linear ThreadlikeActuator transmissions with DirectEffort."
        )
    if configured == "warp" and eligible and not capabilities.force_enabled:
        raise NotImplementedError(
            f"High-level Warp {capabilities.family_name} actuation_force "
            "dispatch did not pass the GPU production gate; only the public "
            "low-level Warp-native threadlike integration API is available."
        )
    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=eligible and capabilities.force_enabled,
    )
    if selected == "jax":
        return model._actuation_force(q, u, qd)
    evaluator = get_actuation_evaluator(capabilities.warp_executor, "force")
    return evaluator(model, q, u, qd)  # type: ignore[operator]


def dispatch_fused_dynamics_actuation_force(
    model: ForwardDynamicsModel,
    q: Array,
    qd: Array,
    u: Array,
    *,
    backend: ExecutionBackend | None,
    capabilities: DynamicsCapabilities,
) -> DynamicsActuationTerms | None:
    """Use a benchmark-qualified fused Warp primal when it is eligible.

    Returning ``None`` preserves the independently dispatched dynamics and
    actuation paths. An explicit Warp request therefore continues to permit
    JAX actuation for custom transmissions or effort laws.
    """

    if (
        not capabilities.fused_threadlike_force_enabled
        or not supports_linear_threadlike_force(model)
    ):
        return None
    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=True,
    )
    if selected != "warp":
        return None
    evaluator = get_dynamics_actuation_evaluator(capabilities.warp_executor)
    return evaluator(model, q, qd, u)


def dispatch_dynamics_terms(
    model: DynamicsModel,
    q: Array,
    qd: Array,
    *,
    backend: ExecutionBackend | None,
    capabilities: DynamicsCapabilities,
    warp_supported: bool = True,
) -> DynamicsTerms:
    """Validate inputs and invoke one system's selected dynamics assembly.

    Scalar inputs retain scalar outputs. A single leading environment dimension
    is also accepted. JAX batches are evaluated with :func:`jax.vmap`; Warp
    batches pass through the evaluator's custom batching rule and therefore use
    one batch-shaped executor invocation. Derivative transformations applied to
    either form are handled by the evaluator and use ``_assemble_dynamics_terms``.

    Args:
        model: System implementing the neutral :class:`DynamicsModel` contract.
        q: Generalized coordinates with shape ``(num_dofs,)`` or
            ``(batch_size, num_dofs)``.
        qd: Generalized velocities with the same shape and dtype as ``q``.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this concrete model instance can use the family
            Warp executor. This distinguishes an exact supported class from an
            arbitrary subclass with different equations.

    Returns:
        A tuple ``(B, Cqd, G)``. Each result has the same optional leading batch
        dimension as the inputs.

    Raises:
        ValueError: If either state has an invalid shape, their shapes differ,
            or the backend name is invalid.
        NotImplementedError: If Warp is explicitly requested for an unsupported
            device or model instance.
        ImportError: If Warp is selected but the optional dependency is absent.
        TypeError: If the selected Warp executor does not support the state
            dtype.
    """

    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    if q.ndim not in (1, 2) or q.shape[-1:] != (model.num_dofs,):
        raise ValueError(
            "q must have shape (num_dofs,) or (batch_size, num_dofs); "
            f"expected (..., {model.num_dofs}), got {q.shape}."
        )
    if qd.shape != q.shape:
        raise ValueError(f"qd must have shape {q.shape}, got {qd.shape}.")

    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=warp_supported,
    )
    if selected == "jax":
        if q.ndim == 1:
            return model._assemble_dynamics_terms(q, qd)
        return jax.vmap(model._assemble_dynamics_terms)(q, qd)
    warp_evaluator = get_dynamics_evaluator(capabilities.warp_executor)
    if q.ndim == 1:
        return warp_evaluator(model, q, qd)
    return jax.vmap(warp_evaluator, in_axes=(None, 0, 0))(model, q, qd)


def _reference_kinematics_result(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Evaluate one kinematics request with the differentiable JAX reference."""

    if operation == "pose":
        return evaluate_forward_kinematics(model, q, s)
    if operation == "jacobian":
        return evaluate_inertial_jacobian(model, q, s)
    return (
        evaluate_forward_kinematics(model, q, s),
        evaluate_inertial_jacobian(model, q, s),
    )


def _reference_kinematics_abscissa_batched_result(
    model: KinematicsModel,
    q: Array,
    s: Array,
    operation: KinematicsOperation,
) -> KinematicsResult:
    """Evaluate a spatial batch with the model's specialized JAX traversal."""

    if operation == "pose":
        return model._forward_kinematics_abscissa_batched(q, s)
    if operation == "jacobian":
        return model._jacobian_inertialframe_abscissa_batched(q, s)
    return (
        model._forward_kinematics_abscissa_batched(q, s),
        model._jacobian_inertialframe_abscissa_batched(q, s),
    )


def dispatch_kinematics(
    model: KinematicsModel,
    q: Array,
    s: Array,
    *,
    operation: KinematicsOperation,
    backend: ExecutionBackend | None,
    capabilities: KinematicsCapabilities,
    warp_supported: bool = True,
) -> KinematicsResult:
    """Dispatch one unbatched configuration at one scalar abscissa.

    Args:
        model: Continuum system implementing the neutral kinematics contract.
        q: Generalized-coordinate vector with shape ``(D,)`` for one
            configuration.
        s: Scalar abscissa.
        operation: Select poses, inertial Jacobians, or their fused evaluation.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this exact model type supports the family Warp
            executor.

    Returns:
        A pose, inertial Jacobian, or their tuple at ``s``.

    Raises:
        ValueError: If an input shape or backend name is invalid.
        NotImplementedError: If explicit Warp execution is unsupported.
        ImportError: If Warp is selected but unavailable.
        TypeError: If Warp receives a dtype other than FP64.
    """

    q = jnp.asarray(q)
    s = jnp.asarray(s)
    if q.shape != (model.num_dofs,):
        raise ValueError(f"q must have shape ({model.num_dofs},), got {q.shape}.")
    if s.ndim != 0:
        raise ValueError(f"s must be scalar, got shape {s.shape}.")

    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=warp_supported,
    )

    if selected == "jax":
        return _reference_kinematics_result(model, q, s, operation)

    scalar_evaluator = get_kinematics_evaluator(capabilities.warp_executor, operation)
    return scalar_evaluator(model, q, s)


def dispatch_kinematics_abscissa_batched(
    model: KinematicsModel,
    q: Array,
    s: Array,
    *,
    operation: KinematicsOperation,
    backend: ExecutionBackend | None,
    capabilities: KinematicsCapabilities,
    warp_supported: bool = True,
) -> KinematicsResult:
    """Dispatch kinematics for one configuration and an abscissa batch.

    Environment batching is deliberately expressed with :func:`jax.vmap` over
    this function's public system wrapper. The Warp evaluator's custom batching
    rule collapses that mapped axis into one canonical ``(E,D)``/``(E,N)``
    executor invocation.

    Args:
        model: Continuum system implementing the neutral kinematics contract.
        q: Generalized-coordinate vector with shape ``(D,)`` for one
            configuration.
        s: Curvilinear abscissae with shape ``(N,)``.
        operation: Select poses, inertial Jacobians, or their fused evaluation.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.
        capabilities: Static support declared for the system family.
        warp_supported: Whether this exact model type supports the family Warp
            executor.

    Returns:
        Poses, inertial Jacobians, or their tuple with leading dimension ``N``.

    Raises:
        ValueError: If an input shape or backend name is invalid.
        NotImplementedError: If explicit Warp execution is unsupported.
        ImportError: If Warp is selected but unavailable.
        TypeError: If Warp receives a dtype other than FP64.
    """

    q = jnp.asarray(q)
    s = jnp.asarray(s)
    if q.shape != (model.num_dofs,):
        raise ValueError(f"q must have shape ({model.num_dofs},), got {q.shape}.")
    if s.ndim != 1:
        raise ValueError(f"s must have shape (num_samples,), got {s.shape}.")

    selected = _select_backend(
        model,
        backend,
        capabilities,
        warp_supported=warp_supported,
    )
    if selected == "jax":
        return _reference_kinematics_abscissa_batched_result(model, q, s, operation)

    abscissa_batched_evaluator = get_abscissa_batched_kinematics_evaluator(
        capabilities.warp_executor, operation
    )
    return abscissa_batched_evaluator(model, q, s)


__all__ = [
    "dispatch_actuation_force",
    "dispatch_actuation_matrix",
    "dispatch_dynamics_terms",
    "dispatch_kinematics",
    "dispatch_kinematics_abscissa_batched",
]
