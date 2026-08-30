"""JAX transformation rules shared by optional execution backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.execution.types import (
    AbscissaBatchedKinematicsEvaluator,
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
    KinematicsEvaluator,
    KinematicsModel,
    KinematicsOperation,
    KinematicsResult,
)

BatchExecutor = Callable[[DynamicsModel, Array, Array], DynamicsTerms]
DynamicsActuationBatchExecutor = Callable[
    [ForwardDynamicsModel, Array, Array, Array], DynamicsActuationTerms
]
KinematicsBatchExecutor = Callable[
    [KinematicsModel, Array, Array, KinematicsOperation], KinematicsResult
]
ActuationMatrixBatchExecutor = Callable[[ActuationModel, Array], Array]
ActuationForceBatchExecutor = Callable[[ActuationModel, Array, Array], Array]


def make_actuation_evaluator(
    execute_batch: ActuationMatrixBatchExecutor | ActuationForceBatchExecutor,
    *,
    family_name: str,
    operation: ActuationOperation,
) -> ActuationMatrixEvaluator | ActuationForceEvaluator:
    """Adapt a canonical actuation batch to scalar and autodiff semantics."""

    if operation == "matrix":

        @jax.custom_batching.custom_vmap
        def matrix_primal(model: ActuationModel, q: Array) -> Array:
            return execute_batch(model, q[None, :])[0]

        @matrix_primal.def_vmap
        def matrix_vmap(
            axis_size: int,
            in_batched: tuple[Any, bool],
            model: ActuationModel,
            q: Array,
        ) -> tuple[Array, bool]:
            model_batched, q_batched = in_batched
            if any(jax.tree.leaves(model_batched)):
                raise ValueError(
                    f"Batching over {family_name} model parameters is unsupported."
                )
            if not q_batched:
                q = jnp.broadcast_to(q, (axis_size, *q.shape))
            return execute_batch(model, q), True

        @eqx.filter_custom_jvp
        def matrix_evaluator(model: ActuationModel, q: Array) -> Array:
            return matrix_primal(model, q)

        @matrix_evaluator.def_jvp
        def matrix_jvp(
            primals: tuple[ActuationModel, Array],
            tangents: tuple[Any, Array | None],
        ) -> tuple[Array, Array]:
            return eqx.filter_jvp(
                lambda model_, q_: model_._actuation_matrix(q_), primals, tangents
            )

        return matrix_evaluator

    @jax.custom_batching.custom_vmap
    def force_primal(model: ActuationModel, q: Array, u: Array, qd: Array) -> Array:
        del qd
        return execute_batch(model, q[None, :], u[None, :])[0]

    @force_primal.def_vmap
    def force_vmap(
        axis_size: int,
        in_batched: tuple[Any, bool, bool, bool],
        model: ActuationModel,
        q: Array,
        u: Array,
        qd: Array,
    ) -> tuple[Array, bool]:
        model_batched, q_batched, u_batched, _qd_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is unsupported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not u_batched:
            u = jnp.broadcast_to(u, (axis_size, *u.shape))
        del qd
        return execute_batch(model, q, u), True

    @eqx.filter_custom_jvp
    def force_evaluator(model: ActuationModel, q: Array, u: Array, qd: Array) -> Array:
        return force_primal(model, q, u, qd)

    @force_evaluator.def_jvp
    def force_jvp(
        primals: tuple[ActuationModel, Array, Array, Array],
        tangents: tuple[Any, Array | None, Array | None, Array | None],
    ) -> tuple[Array, Array]:
        return eqx.filter_jvp(
            lambda model_, q_, u_, qd_: model_._actuation_force(q_, u_, qd_),
            primals,
            tangents,
        )

    return force_evaluator


@jax.custom_batching.custom_vmap
def _evaluate_forward_kinematics_primal(
    model: KinematicsModel, q: Array, s: Array
) -> Array:
    """Evaluate the scalar protected pose implementation."""

    return model._absolute_forward_kinematics(q, s)


@_evaluate_forward_kinematics_primal.def_vmap
def _evaluate_forward_kinematics_vmap(
    axis_size: int,
    in_batched: tuple[Any, bool, bool],
    model: KinematicsModel,
    q: Array,
    s: Array,
) -> tuple[Array, bool]:
    """Use the model's specialized traversal for spatial vectorization."""

    model_batched, q_batched, s_batched = in_batched
    if any(jax.tree.leaves(model_batched)):
        raise ValueError("Batching over kinematics model parameters is unsupported.")
    if not q_batched and not s_batched:
        result = model._absolute_forward_kinematics(q, s)
    elif not q_batched and s_batched:
        result = model._absolute_forward_kinematics_abscissa_batched(q, s)
    elif q_batched and not s_batched:
        result = jax.vmap(model._absolute_forward_kinematics, in_axes=(0, None))(q, s)
    else:
        result = jax.vmap(model._absolute_forward_kinematics)(q, s)
    return result, q_batched or s_batched


@eqx.filter_custom_jvp
def evaluate_forward_kinematics(model: KinematicsModel, q: Array, s: Array) -> Array:
    """Evaluate one forward-kinematics request with its established JVP.

    Args:
        model: System implementing the neutral kinematics contract.
        q: Generalized coordinates for one environment.
        s: One backbone coordinate.

    Returns:
        The model-specific pose at ``s``.
    """

    return _evaluate_forward_kinematics_primal(model, q, s)


@evaluate_forward_kinematics.def_jvp
def _evaluate_forward_kinematics_jvp(
    primals: tuple[KinematicsModel, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[Array, Array]:
    """Route pose derivatives through the model's established JAX rule."""

    model, q, s = primals
    _model_tangent, qd, sd = tangents
    return model._forward_kinematics_jvp(q, s, qd, sd)


@jax.custom_batching.custom_vmap
def _evaluate_inertial_jacobian_primal(
    model: KinematicsModel, q: Array, s: Array
) -> Array:
    """Evaluate one inertial Jacobian with specialized spatial batching.

    Args:
        model: System implementing the neutral kinematics contract.
        q: Generalized coordinates for one environment.
        s: One backbone coordinate.

    Returns:
        The inertial-frame Jacobian at ``s``.
    """

    return model._absolute_inertial_jacobian(q, s)


@_evaluate_inertial_jacobian_primal.def_vmap
def _evaluate_inertial_jacobian_vmap(
    axis_size: int,
    in_batched: tuple[Any, bool, bool],
    model: KinematicsModel,
    q: Array,
    s: Array,
) -> tuple[Array, bool]:
    """Use the model's specialized traversal for spatial vectorization."""

    model_batched, q_batched, s_batched = in_batched
    if any(jax.tree.leaves(model_batched)):
        raise ValueError("Batching over kinematics model parameters is unsupported.")
    if not q_batched and not s_batched:
        result = model._absolute_inertial_jacobian(q, s)
    elif not q_batched and s_batched:
        result = model._absolute_inertial_jacobian_abscissa_batched(q, s)
    elif q_batched and not s_batched:
        result = jax.vmap(model._absolute_inertial_jacobian, in_axes=(0, None))(q, s)
    else:
        result = jax.vmap(model._absolute_inertial_jacobian)(q, s)
    return result, q_batched or s_batched


@eqx.filter_custom_jvp
def evaluate_inertial_jacobian(model: KinematicsModel, q: Array, s: Array) -> Array:
    """Evaluate one inertial Jacobian through the differentiable JAX path."""

    return _evaluate_inertial_jacobian_primal(model, q, s)


@evaluate_inertial_jacobian.def_jvp
def _evaluate_inertial_jacobian_jvp(
    primals: tuple[KinematicsModel, Array, Array],
    tangents: tuple[Any, Array | None, Array | None],
) -> tuple[Array, Array]:
    """Differentiate the model's established scalar JAX Jacobian."""

    return eqx.filter_jvp(
        lambda model, q, s: model._absolute_inertial_jacobian(q, s),
        primals,
        tangents,
    )


def make_kinematics_evaluators(
    execute_batch: KinematicsBatchExecutor,
    *,
    family_name: str,
    operation: KinematicsOperation,
) -> tuple[KinematicsEvaluator, AbscissaBatchedKinematicsEvaluator]:
    """Adapt a canonical ``(E,D)``/``(E,N)`` executor to public JAX semantics."""

    def slice_environment_sample(result: KinematicsResult) -> KinematicsResult:
        return jax.tree.map(lambda value: value[0, 0], result)

    def slice_environment(result: KinematicsResult) -> KinematicsResult:
        return jax.tree.map(lambda value: value[0], result)

    @jax.custom_batching.custom_vmap
    def scalar_primal(model: KinematicsModel, q: Array, s: Array) -> KinematicsResult:
        result = execute_batch(model, q[None, :], s.reshape(1, 1), operation)
        return slice_environment_sample(result)

    @scalar_primal.def_vmap
    def scalar_vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: KinematicsModel,
        q: Array,
        s: Array,
    ) -> tuple[KinematicsResult, Any]:
        model_batched, q_batched, s_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if q_batched:
            if not s_batched:
                s = jnp.broadcast_to(s, (axis_size,))
            result = execute_batch(model, q, s[:, None], operation)
            result = jax.tree.map(lambda value: value[:, 0], result)
        else:
            # Keep the spatial batch behind its own custom batching boundary.
            # An enclosing environment vmap can then merge q's mapped axis
            # into the same canonical ``(E,D)``/``(E,N)`` executor call.
            result = abscissa_batched_primal(model, q, s)
        return result, jax.tree.map(lambda _: True, result)

    @eqx.filter_custom_jvp
    def scalar_evaluator(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return scalar_primal(model, q, s)

    @scalar_evaluator.def_jvp
    def scalar_jvp(
        primals: tuple[KinematicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[KinematicsResult, KinematicsResult]:
        model, q, s = primals
        _model_tangent, qd, sd = tangents
        if operation == "pose":
            return model._forward_kinematics_jvp(q, s, qd, sd)
        if operation == "jacobian":
            return eqx.filter_jvp(
                lambda model_, q_, s_: model_._absolute_inertial_jacobian(q_, s_),
                primals,
                tangents,
            )
        pose, pose_tangent = model._forward_kinematics_jvp(q, s, qd, sd)
        jacobian, jacobian_tangent = eqx.filter_jvp(
            lambda model_, q_, s_: model_._absolute_inertial_jacobian(q_, s_),
            primals,
            tangents,
        )
        return (pose, jacobian), (pose_tangent, jacobian_tangent)

    @jax.custom_batching.custom_vmap
    def abscissa_batched_primal(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return slice_environment(
            execute_batch(model, q[None, :], s[None, :], operation)
        )

    @abscissa_batched_primal.def_vmap
    def abscissa_batched_vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: KinematicsModel,
        q: Array,
        s: Array,
    ) -> tuple[KinematicsResult, Any]:
        model_batched, q_batched, s_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not s_batched:
            s = jnp.broadcast_to(s, (axis_size, *s.shape))
        result = execute_batch(model, q, s, operation)
        return result, jax.tree.map(lambda _: True, result)

    @eqx.filter_custom_jvp
    def abscissa_batched_evaluator(
        model: KinematicsModel, q: Array, s: Array
    ) -> KinematicsResult:
        return abscissa_batched_primal(model, q, s)

    @abscissa_batched_evaluator.def_jvp
    def abscissa_batched_jvp(
        primals: tuple[KinematicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[KinematicsResult, KinematicsResult]:
        def jax_abscissa_batched(
            model_: KinematicsModel, q_: Array, s_: Array
        ) -> KinematicsResult:
            if operation == "pose":
                return model_._absolute_forward_kinematics_abscissa_batched(q_, s_)
            if operation == "jacobian":
                return model_._absolute_inertial_jacobian_abscissa_batched(q_, s_)
            return (
                model_._absolute_forward_kinematics_abscissa_batched(q_, s_),
                model_._absolute_inertial_jacobian_abscissa_batched(q_, s_),
            )

        return eqx.filter_jvp(jax_abscissa_batched, primals, tangents)

    return scalar_evaluator, abscissa_batched_evaluator


def make_dynamics_evaluator(
    execute_batch: BatchExecutor, *, family_name: str
) -> DynamicsEvaluator:
    """Adapt a batch-shaped executor to JAX scalar transformation semantics.

    The returned callable accepts one environment. Applying :func:`jax.vmap`
    does not replicate batch-one executor launches: the custom batching rule
    moves the mapped axis into a single call to ``execute_batch``. Applying
    forward- or reverse-mode differentiation replaces the forward-only executor
    with ``model._assemble_dynamics_terms`` and differentiates that JAX
    implementation. This is an execution-routing rule, not an alternative
    analytical dynamics derivative.

    Args:
        execute_batch: Family implementation accepting ``q`` and ``qd`` with
            shape ``(batch_size, num_dofs)`` and returning batched terms.
        family_name: Human-readable family name used in batching errors.

    Returns:
        A scalar-semantics dynamics evaluator with custom ``vmap`` and JVP
        behavior.
    """

    @jax.custom_batching.custom_vmap
    def execute_primal(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Execute a scalar request through a temporary one-item batch.

        Args:
            model: System model supplied to the family executor.
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Scalar inertia, Coriolis/centrifugal, and gravity terms.
        """

        batched = execute_batch(model, q[None, :], qd[None, :])
        return jax.tree.map(lambda value: value[0], batched)

    @execute_primal.def_vmap
    def vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool],
        model: DynamicsModel,
        q: Array,
        qd: Array,
    ) -> tuple[DynamicsTerms, tuple[bool, bool, bool]]:
        """Combine mapped environments into one family executor call.

        Args:
            axis_size: Size of the mapped axis.
            in_batched: Boolean PyTree indicating which inputs carry that axis.
            model: Unbatched system model.
            q: Scalar or mapped generalized coordinates.
            qd: Scalar or mapped generalized velocities.

        Returns:
            Batched dynamics terms and flags marking every output as batched.

        Raises:
            ValueError: If a caller attempts to batch model parameters.
        """

        model_batched, q_batched, qd_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not qd_batched:
            qd = jnp.broadcast_to(qd, (axis_size, *qd.shape))
        return execute_batch(model, q, qd), (True, True, True)

    @eqx.filter_custom_jvp
    def evaluate_terms(model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Evaluate scalar forward-only dynamics terms.

        Args:
            model: System model supplied to the family executor.
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Scalar inertia, Coriolis/centrifugal, and gravity terms.
        """

        return execute_primal(model, q, qd)

    @evaluate_terms.def_jvp
    def jvp_rule(
        primals: tuple[DynamicsModel, Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[DynamicsTerms, DynamicsTerms]:
        """Differentiate the model's JAX dynamics assembly.

        Args:
            primals: Model, generalized coordinates, and velocities.
            tangents: Equinox-filtered tangents corresponding to ``primals``.

        Returns:
            JAX-backed primal terms and their directional derivatives.
        """

        return eqx.filter_jvp(
            lambda model, q, qd: model._assemble_dynamics_terms(q, qd),
            primals,
            tangents,
        )

    return evaluate_terms


def make_dynamics_actuation_evaluator(
    execute_batch: DynamicsActuationBatchExecutor,
    *,
    family_name: str,
) -> DynamicsActuationEvaluator:
    """Adapt fused batch execution to scalar and transformation semantics.

    Args:
        execute_batch: Family executor accepting canonical ``(E, D)`` states
            and ``(E, A)`` controls.
        family_name: Human-readable name used in batching diagnostics.

    Returns:
        Scalar evaluator whose mapped primal uses one batch invocation and
        whose derivatives use the protected JAX implementations.
    """

    def reference(
        model: ForwardDynamicsModel,
        q: Array,
        qd: Array,
        u: Array,
    ) -> DynamicsActuationTerms:
        inertia, coriolis_qd, gravity = model._assemble_dynamics_terms(q, qd)
        return inertia, coriolis_qd, gravity, model._actuation_force(q, u, qd)

    @jax.custom_batching.custom_vmap
    def execute_primal(
        model: ForwardDynamicsModel,
        q: Array,
        qd: Array,
        u: Array,
    ) -> DynamicsActuationTerms:
        batched = execute_batch(model, q[None, :], qd[None, :], u[None, :])
        return jax.tree.map(lambda value: value[0], batched)

    @execute_primal.def_vmap
    def vmap_rule(
        axis_size: int,
        in_batched: tuple[Any, bool, bool, bool],
        model: ForwardDynamicsModel,
        q: Array,
        qd: Array,
        u: Array,
    ) -> tuple[DynamicsActuationTerms, tuple[bool, bool, bool, bool]]:
        model_batched, q_batched, qd_batched, u_batched = in_batched
        if any(jax.tree.leaves(model_batched)):
            raise ValueError(
                f"Batching over {family_name} model parameters is not supported."
            )
        if not q_batched:
            q = jnp.broadcast_to(q, (axis_size, *q.shape))
        if not qd_batched:
            qd = jnp.broadcast_to(qd, (axis_size, *qd.shape))
        if not u_batched:
            u = jnp.broadcast_to(u, (axis_size, *u.shape))
        return execute_batch(model, q, qd, u), (True, True, True, True)

    @eqx.filter_custom_jvp
    def evaluate_terms(
        model: ForwardDynamicsModel,
        q: Array,
        qd: Array,
        u: Array,
    ) -> DynamicsActuationTerms:
        return execute_primal(model, q, qd, u)

    @evaluate_terms.def_jvp
    def jvp_rule(
        primals: tuple[ForwardDynamicsModel, Array, Array, Array],
        tangents: tuple[Any, Array | None, Array | None, Array | None],
    ) -> tuple[DynamicsActuationTerms, DynamicsActuationTerms]:
        return eqx.filter_jvp(reference, primals, tangents)

    return evaluate_terms


@eqx.filter_custom_jvp
def evaluate_forward_dynamics(
    model: ForwardDynamicsModel,
    t: Array,
    y: Array,
    actuation_args: tuple | None,
    backend: ExecutionBackend | None,
    capabilities: DynamicsCapabilities | None = None,
) -> Array:
    """Evaluate compiled forward dynamics with transform-aware backend routing.

    This boundary does not implement a new analytical derivative. Its custom
    JVP exists solely to prevent a compiled primal calculation from staging a
    forward-only Warp callback before JAX selects a derivative rule. Primal
    calls use the model's configured backend; forward- and reverse-mode
    transformations evaluate the same forward-dynamics equations with JAX term
    assembly.

    Args:
        model: System implementing the forward-dynamics execution contract.
        t: Current integration time.
        y: State vector accepted by the system's forward dynamics.
        actuation_args: Optional actuation and external-force arguments forwarded
            unchanged to the system.
        backend: Optional per-call backend override. ``None`` uses the model's
            configured backend.
        capabilities: Optional family capabilities enabling a benchmark-gated
            fused dynamics-and-threadlike-actuation primal path.

    Returns:
        The state derivative produced by the model's compiled forward-dynamics
        implementation.
    """

    return _assemble_forward_dynamics(
        model,
        t,
        y,
        actuation_args,
        backend=backend,
        capabilities=capabilities,
    )


@evaluate_forward_dynamics.def_jvp
def _evaluate_forward_dynamics_jvp(
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
) -> tuple[Array, Array]:
    """Differentiate forward dynamics using the model's JAX term assembly.

    Args:
        primals: Model, time, state, optional actuation arguments, and backend
            supplied to :func:`evaluate_forward_dynamics`.
        tangents: Tangents corresponding to ``primals`` as filtered by Equinox.

    Returns:
        A pair containing the JAX-backed primal state derivative and its
        directional derivative.
    """

    if len(primals) == 5:
        model, t, y, actuation_args, _backend = primals
        model_tangent, t_tangent, y_tangent, actuation_tangent, _backend_tangent = (
            tangents
        )
        capabilities = None
        capabilities_tangent = None
    else:
        model, t, y, actuation_args, _backend, capabilities = primals
        (
            model_tangent,
            t_tangent,
            y_tangent,
            actuation_tangent,
            _backend_tangent,
            capabilities_tangent,
        ) = tangents
    return eqx.filter_jvp(
        lambda model_, t_, y_, actuation_args_, capabilities_: (
            _assemble_forward_dynamics(
                model_,
                t_,
                y_,
                actuation_args_,
                backend="jax",
                capabilities=capabilities_,
            )
        ),
        (model, t, y, actuation_args, capabilities),
        (
            model_tangent,
            t_tangent,
            y_tangent,
            actuation_tangent,
            capabilities_tangent,
        ),
    )


@eqx.filter_jit
def _assemble_forward_dynamics(
    model: ForwardDynamicsModel,
    t: Array,
    y: Array,
    actuation_args: tuple | None,
    *,
    backend: ExecutionBackend | None,
    capabilities: DynamicsCapabilities | None,
) -> Array:
    """Assemble generalized forces and solve one forward-dynamics request.

    This shared implementation keeps the public system methods small and makes
    the force convention identical for GVS, PCS, and PlanarPCS. Passive forces,
    damping, and the inertia solve remain JAX operations. Actuation follows the
    same backend request but independently falls back to JAX unless its fused
    high-level Warp path is enabled.

    Args:
        model: System implementing the forward-dynamics execution contract.
        t: Current integration time. The supported autonomous systems accept it
            for solver compatibility and do not otherwise use it.
        y: State vector ``[q, qd]`` for one environment.
        actuation_args: Optional tuple ``(u,)`` or ``(u, tau_ext)``. Missing
            actuation and external force values default to zero.
        backend: Backend used to assemble dynamics terms. ``None`` uses the
            model's configured backend.
        capabilities: Optional family support for fused primal execution.

    Returns:
        State time derivative ``[qd, qdd]`` with the same shape as ``y``.

    Raises:
        ValueError: If ``actuation_args`` has a length other than one or two.
    """

    del t
    q, qd, auxiliary_state = model.split_state(y)
    if actuation_args is None:
        u, tau_ext = None, None
    elif len(actuation_args) == 1:
        u = actuation_args[0]
        tau_ext = None
    elif len(actuation_args) == 2:
        u, tau_ext = actuation_args
    else:
        raise ValueError("actuation_args must be a tuple of length 1 or 2.")

    if u is None:
        u = jnp.zeros((model.num_actuators,), dtype=y.dtype)
    if tau_ext is None:
        tau_ext = jnp.zeros((model.num_velocities,), dtype=y.dtype)

    elastic = model.elastic_force(q)
    fused = None
    if capabilities is not None:
        from soromox.systems.execution.dispatch import (
            dispatch_fused_dynamics_actuation_force,
        )

        fused = dispatch_fused_dynamics_actuation_force(
            model,
            q,
            qd,
            u,
            backend=backend,
            capabilities=capabilities,
        )
    if fused is None:
        inertia, coriolis_qd, gravity = model.dynamics_terms(q, qd, backend=backend)
        requested = model.backend if backend is None else backend
        actuation_backend: ExecutionBackend | None = (
            "auto" if requested == "warp" else backend
        )
        actuation = model.actuation_force(q, u, qd=qd, backend=actuation_backend)
    else:
        inertia, coriolis_qd, gravity, actuation = fused
    rhs = (
        actuation
        + tau_ext
        - coriolis_qd
        - gravity
        - elastic
        - model.damping_matrix(q) @ qd
    )
    qdd = model._solve_inertia(inertia, rhs)
    qdot = model.configuration_derivative(q, qd)
    if model.num_auxiliary_states == 0:
        return jnp.concatenate([qdot, qdd])
    auxiliary_dot = jnp.zeros_like(auxiliary_state)
    return jnp.concatenate([qdot, qdd, auxiliary_dot])


__all__ = [
    "ActuationForceBatchExecutor",
    "ActuationMatrixBatchExecutor",
    "BatchExecutor",
    "DynamicsActuationBatchExecutor",
    "KinematicsBatchExecutor",
    "evaluate_forward_dynamics",
    "make_actuation_evaluator",
    "make_dynamics_evaluator",
    "make_dynamics_actuation_evaluator",
    "make_kinematics_evaluators",
]
