"""Shared types and structural contracts for system execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from jax import Array

ExecutionBackend = Literal["auto", "jax", "warp"]
WarpExecutorKey = Literal["gvs", "pcs"]
DynamicsTerms = tuple[Array, Array, Array]
DynamicsActuationTerms = tuple[Array, Array, Array, Array]
KinematicsOperation = Literal["pose", "jacobian", "both"]
KinematicsResult = Array | tuple[Array, Array]
ActuationOperation = Literal["matrix", "force"]


class DynamicsModel(Protocol):
    """Structural interface required by dynamics execution policy.

    System implementations satisfy this protocol implicitly; they do not
    inherit from an execution-layer base class. This keeps the neutral dispatch
    package independent of concrete model modules and prevents circular
    imports.

    Attributes:
        backend: Model-level execution preference used when a call does not
            supply an override.
        num_dofs: Number of active generalized coordinates in the model.
    """

    backend: ExecutionBackend
    num_dofs: int

    def _assemble_dynamics_terms(self, q: Array, qd: Array) -> DynamicsTerms:
        """Assemble differentiable dynamics terms with the JAX implementation.

        Args:
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Inertia, Coriolis/centrifugal, and gravity terms ``(B, Cqd, G)``.
        """

        ...


class ForwardDynamicsModel(DynamicsModel, Protocol):
    """Model contract required by transform-aware forward dynamics.

    Implementations provide a compiled forward calculation whose ``backend``
    argument controls only dynamics-term assembly. The shared transformation
    boundary invokes this method with the configured backend for primal calls
    and with JAX for derivative calls.

    The protocol deliberately extends :class:`DynamicsModel` rather than
    importing ``GVS``, ``PCS``, or ``PlanarPCS``. Every supported system can
    therefore share the same transformation rule without creating a dependency
    from the execution layer back to the system classes.
    """

    num_actuators: int

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> DynamicsTerms:
        """Assemble dynamics terms using the selected execution backend.

        Args:
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.
            backend: Optional per-call backend override.

        Returns:
            Inertia, Coriolis/centrifugal, and gravity terms ``(B, Cqd, G)``.
        """

        ...

    def elastic_force(self, q: Array) -> Array:
        """Return the generalized elastic force for ``q``.

        Args:
            q: Generalized coordinates for one environment.

        Returns:
            Generalized elastic force vector.
        """

        ...

    def damping_matrix(self, q: Array) -> Array:
        """Return the generalized damping matrix for ``q``.

        Args:
            q: Generalized coordinates for one environment.

        Returns:
            Generalized damping matrix.
        """

        ...

    def actuation_force(
        self,
        q: Array,
        u: Array,
        *,
        qd: Array,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Return the generalized actuator force.

        Args:
            q: Generalized coordinates for one environment.
            u: Actuator inputs.
            qd: Generalized velocities for the same environment.
            backend: Optional per-call backend override.

        Returns:
            Generalized actuator force vector.
        """

        ...

    def _actuation_force(self, q: Array, u: Array, qd: Array) -> Array:
        """Return the differentiable JAX generalized actuator force."""

        ...

    def _solve_inertia(self, inertia: Array, rhs: Array) -> Array:
        """Solve the generalized inertia system.

        Args:
            inertia: Generalized inertia matrix.
            rhs: Generalized force right-hand side.

        Returns:
            Generalized acceleration vector.
        """

        ...


class DynamicsEvaluator(Protocol):
    """Callable contract for a transform-aware dynamics evaluator.

    Evaluators present scalar inputs to model code while retaining an internal
    batch-shaped executor. Their custom batching rule combines a mapped leading
    dimension into one executor call, and their derivative rule substitutes
    the model's differentiable JAX assembly.
    """

    def __call__(self, model: DynamicsModel, q: Array, qd: Array) -> DynamicsTerms:
        """Evaluate dynamics terms for one environment.

        Args:
            model: System model satisfying :class:`DynamicsModel`.
            q: Generalized coordinates for one environment.
            qd: Generalized velocities for the same environment.

        Returns:
            Inertia, Coriolis/centrifugal, and gravity terms ``(B, Cqd, G)``.
        """

        ...


class DynamicsActuationEvaluator(Protocol):
    """Scalar dynamics-and-actuation evaluator backed by one batched executor."""

    def __call__(
        self,
        model: ForwardDynamicsModel,
        q: Array,
        qd: Array,
        u: Array,
    ) -> DynamicsActuationTerms:
        """Evaluate inertia, convective, gravity, and actuator-force terms."""

        ...


class KinematicsModel(Protocol):
    """Structural contract for accelerated continuum-system kinematics."""

    backend: ExecutionBackend
    num_dofs: int
    is_planar: bool

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """Return the differentiable JAX pose at one backbone coordinate."""

        ...

    def _forward_kinematics_abscissa_batched(self, q: Array, s: Array) -> Array:
        """Return poses from the model's specialized spatial JAX traversal."""

        ...

    def _forward_kinematics_jvp(
        self,
        q: Array,
        s: Array,
        qd: Array | None,
        sd: Array | None,
    ) -> tuple[Array, Array]:
        """Return the established JAX/custom-JVP pose rule."""

        ...

    def _jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """Return the differentiable JAX inertial Jacobian at one coordinate."""

        ...

    def _jacobian_inertialframe_abscissa_batched(self, q: Array, s: Array) -> Array:
        """Return Jacobians from the specialized spatial JAX traversal."""

        ...


class KinematicsEvaluator(Protocol):
    """Single-point callable backed by a batch-shaped kinematics executor."""

    def __call__(self, model: KinematicsModel, q: Array, s: Array) -> KinematicsResult:
        """Evaluate one configuration and one backbone coordinate."""

        ...


class AbscissaBatchedKinematicsEvaluator(Protocol):
    """Spatial-batch callable backed by the canonical kinematics executor."""

    def __call__(self, model: KinematicsModel, q: Array, s: Array) -> KinematicsResult:
        """Evaluate one configuration at a one-dimensional coordinate batch."""

        ...


class ActuationModel(Protocol):
    """Structural contract for transform-aware actuation execution."""

    backend: ExecutionBackend
    num_dofs: int
    num_actuators: int

    def _actuation_matrix(self, q: Array) -> Array:
        """Return the differentiable scalar JAX transmission matrix."""

        ...

    def _actuation_force(self, q: Array, u: Array, qd: Array) -> Array:
        """Return the differentiable scalar JAX generalized actuator force."""

        ...


class ActuationMatrixEvaluator(Protocol):
    """Scalar matrix evaluator backed by one canonical batched executor."""

    def __call__(self, model: ActuationModel, q: Array) -> Array:
        """Return one transmission matrix with shape ``(D, A)``."""

        ...


class ActuationForceEvaluator(Protocol):
    """Scalar fused-force evaluator backed by one canonical batched executor."""

    def __call__(self, model: ActuationModel, q: Array, u: Array, qd: Array) -> Array:
        """Return one generalized actuator force with shape ``(D,)``."""

        ...


@dataclass(frozen=True)
class DynamicsCapabilities:
    """Describe optional dynamics support for one system family.

    Attributes:
        family_name: Human-readable family name used in validation errors.
        warp_executor: Lazy-loader key for the family's Warp implementation.
        warp_cpu_supported: Whether the Warp executor may run when JAX's default
            backend is CPU. When false, automatic selection falls back to JAX
            and an explicit Warp request raises an error on CPU.
        fused_threadlike_force_enabled: Whether eligible linear-threadlike
            forward dynamics may combine dynamics and direct-effort actuation
            in one family executor invocation.
    """

    family_name: str
    warp_executor: WarpExecutorKey
    warp_cpu_supported: bool = False
    fused_threadlike_force_enabled: bool = False


@dataclass(frozen=True)
class KinematicsCapabilities:
    """Describe optional Warp kinematics support for one system family.

    Attributes:
        family_name: Human-readable system-family name used in errors.
        warp_executor: Lazy executor registry key for the family.
        warp_cpu_supported: Whether explicit Warp execution is available on
            CPU as well as CUDA.
    """

    family_name: str
    warp_executor: WarpExecutorKey
    warp_cpu_supported: bool = True


@dataclass(frozen=True)
class ActuationCapabilities:
    """Record benchmark-gated high-level threadlike Warp support.

    The low-level Warp-native operand and launcher API is independent of these
    flags and remains available even when a public operation is disabled.
    """

    family_name: str
    warp_executor: WarpExecutorKey
    matrix_enabled: bool
    force_enabled: bool
    warp_cpu_supported: bool = False


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
    "KinematicsCapabilities",
    "KinematicsEvaluator",
    "KinematicsModel",
    "KinematicsOperation",
    "KinematicsResult",
    "WarpExecutorKey",
]
