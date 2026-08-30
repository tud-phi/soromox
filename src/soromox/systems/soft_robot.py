__all__ = [
    "SoftRobot",
]

import copy
from abc import abstractmethod
from typing import Any, Self

import equinox as eqx
import jax
import numpy as np
from jax import Array, grad, jacfwd, jvp, lax, vmap
from jax import numpy as jnp
from jax.scipy.linalg import cho_solve

from soromox.actuation.core import (
    Actuator,
    ActuatorMetadata,
    IdentityActuator,
    PassiveElement,
)
from soromox.autodiff import custom_jvp_enabled
from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.params import (
    validate_planar_base_pose,
    validate_quaternion_base_pose,
)
from soromox.utils.array_math import broadcast_leading_axes
from soromox.utils.geometry import poses
from soromox.utils.geometry.rotations import quaternion_multiply
from soromox.utils.lie_algebra import se2, se3, so3


class SoftRobot(DynamicalSystem):
    """
    Abstract base class for soft robot systems.

    This class extends DynamicalSystem with interfaces specific to soft robots,
    including methods for forward kinematics and Jacobians parameterized by
    arc-length or position along the robot backbone.

    All soft robot implementations (PCS, PlanarPCS, Pendulum, etc.) should
    inherit from this class and implement the abstract methods.

    The key distinction from DynamicalSystem is that soft robots support:
    - Continuous parameterization along their structure (via parameter `s`)
    - Forward kinematics and Jacobians at arbitrary points along the backbone
    - Dynamical matrices (inertia, Coriolis, damping, etc.) in configuration space
    - Energy computation methods

    Concrete parameterized models expose ``params`` and implement
    ``with_params(...)`` and ``update_params(...)``. A complete parameter
    replacement must return a new model, refresh every parameter-dependent
    runtime array and cache, and preserve the model's static structure. It must
    not mutate the original model or rely on the constructor-time
    :meth:`precompute` hook.

    Attributes:
        num_actuators (int): Number of actuators.
        global_eps (float): Global epsilon for numerical computations.
        floating_base: Whether configuration and velocity contain runtime base
            state.
        fixed_base_pose: Fixed mounting pose, or ``None`` for floating models.
        num_gauss_points (int | Array | None): Requested nonzero
            Gauss-Legendre quadrature point count. May be scalar for systems
            with a uniform grid or an array for systems with per-segment grids.
        num_integration_points (int | Array | None): Stored integration point
            count, including any zero-weight boundary nodes used internally.
            May be scalar or per-segment.
        integration_points (Array | None): Quadrature nodes used for numerical
            integration, typically on the normalized interval [0, 1].
        integration_weights (Array | None): Quadrature weights corresponding
            to `integration_points`.
    """

    # global epsilon for numerical computations
    global_eps: float  # Global epsilon for numerical computations
    floating_base: bool = eqx.field(static=True)
    num_internal_dofs: int = eqx.field(static=True)
    fixed_base_pose: Array | None
    num_gauss_points: int | Array | None
    num_integration_points: int | Array | None
    integration_points: Array | None
    integration_weights: Array | None
    actuators: tuple[Actuator, ...]
    passive_elements: tuple[PassiveElement, ...]

    @property
    def tangent_eps(self) -> Array:
        """Epsilon value for Lie algebra tangent computations.

        Returns:
            Array: Epsilon value for Lie algebra tangent computations.
        """
        return jnp.sqrt(self.global_eps)

    def __init__(
        self,
        eps: float | None = None,
        base_pose: Array | None = None,
        floating_base: bool = False,
        **kwargs: Any,
    ):
        """Initialize the SoftRobot.

        Args:
            eps (float): Optional global epsilon value for numerical computations.
                If not provided, defaults to 10x machine epsilon for float64.
            base_pose: Optional base frame pose coordinates. Planar robots
                expect shape ``(3,)`` with ``[theta, x, y]``. Spatial robots
                expect shape ``(7,)`` with ``[qw, qx, qy, qz, x, y, z]``.
                Spatial quaternions are scalar-first Hamilton quaternions,
                normalized before use, and must have nonzero finite norm. If
                omitted, the upright pose is used: the backbone points along
                world +y for planar robots and world +z for spatial robots.
            floating_base: Whether the base pose and velocity are runtime state.
                Floating robots reject ``base_pose`` because their initial pose
                must be supplied to :meth:`pack_configuration`.
            **kwargs: Additional keyword arguments (unused, kept for API compatibility).
        """
        # Note: We don't call super().__init__() here because Equinox modules
        # work like dataclasses - fields are set directly rather than through
        # parent __init__ calls. Child classes must set the dimensions and
        # number of actuators before installing actuator models.
        if not isinstance(floating_base, bool):
            raise TypeError("floating_base must be a bool.")
        if floating_base and base_pose is not None:
            raise ValueError(
                "base_pose configures a fixed mounting and cannot be supplied "
                "when floating_base=True; pass the initial pose to "
                "pack_configuration instead."
            )
        self.floating_base = floating_base
        if floating_base:
            self.fixed_base_pose = None
        else:
            resolved_pose = (
                (
                    poses.planar_mounting_pose("upright")
                    if self.is_planar
                    else poses.spatial_mounting_pose("upright")
                )
                if base_pose is None
                else jnp.asarray(base_pose)
            )
            if self.is_planar:
                validate_planar_base_pose("base_pose", resolved_pose)
            else:
                validate_quaternion_base_pose("base_pose", resolved_pose, (7,))
            self.fixed_base_pose = jnp.asarray(resolved_pose, dtype=jnp.float64)
        self.num_gauss_points = None
        self.num_integration_points = None
        self.integration_points = None
        self.integration_weights = None
        self.actuators = ()
        self.passive_elements = ()
        if eps is not None:
            self.global_eps = eps
        else:
            self.global_eps = 1e1 * float(jnp.finfo(jnp.float64).eps)

    @property
    def num_base_coordinates(self) -> int:
        """Return the configuration-storage dimension of the base.

        Returns:
            Zero for fixed robots, three for planar floating robots, and seven
            for spatial floating robots.
        """
        return (3 if self.is_planar else 7) if self.floating_base else 0

    @property
    def num_base_velocities(self) -> int:
        """Return the generalized-velocity dimension of the base.

        Returns:
            Zero for fixed robots, three for planar floating robots, and six
            for spatial floating robots.
        """
        return (3 if self.is_planar else 6) if self.floating_base else 0

    @property
    def num_coordinates(self) -> int:
        """Return the total configuration dimension.

        Returns:
            ``nq``, including runtime base coordinates when enabled.
        """
        return self.num_base_coordinates + self.num_internal_dofs

    @property
    def num_velocities(self) -> int:
        """Return the total generalized-velocity dimension.

        Returns:
            ``nv``, including runtime base velocities when enabled.
        """
        return self.num_base_velocities + self.num_internal_dofs

    @property
    def num_auxiliary_states(self) -> int:
        """Return the number of trailing first-order auxiliary states.

        Returns:
            The hysteresis-state dimension, or zero for systems without
            auxiliary dynamics.
        """
        return int(getattr(self, "num_hysteresis", 0))

    @property
    def state_size(self) -> int:
        """Return the flattened first-order state dimension.

        Returns:
            ``num_coordinates + num_velocities + num_auxiliary_states``.
        """
        return self.num_coordinates + self.num_velocities + self.num_auxiliary_states

    @property
    def _identity_base_pose(self) -> Array:
        """Return the identity pose for mounting-independent recurrences.

        Returns:
            Planar ``[theta, x, y]`` or spatial scalar-first quaternion and
            position coordinates.
        """
        if self.is_planar:
            return jnp.zeros((3,), dtype=jnp.float64)
        return jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def _kinematic_base_pose(self) -> Array:
        """Return the pose consumed by internal model recurrences.

        Returns:
            The fixed mounting for fixed robots and identity for floating
            robots, whose runtime base is composed by an outer adapter.
        """
        if self.floating_base:
            return self._identity_base_pose
        assert self.fixed_base_pose is not None
        return self.fixed_base_pose

    def _check_trailing_dimension(self, name: str, value: Array, size: int) -> Array:
        """Convert an input to an array and validate its trailing dimension.

        Args:
            name: User-facing input name for validation errors.
            value: Scalar or batched input to validate.
            size: Required trailing-axis size.

        Returns:
            The input converted to a JAX array.

        Raises:
            ValueError: If the input is scalar or has the wrong trailing size.
        """
        array = jnp.asarray(value)
        if array.ndim == 0 or array.shape[-1] != size:
            raise ValueError(
                f"{name} must have trailing dimension {size}, got {array.shape}."
            )
        return array

    def pack_configuration(
        self, q_internal: Array, *, base_pose: Array | None = None
    ) -> Array:
        """Pack internal coordinates and a runtime base pose.

        Args:
            q_internal: Internal coordinates with trailing dimension
                ``num_internal_dofs``.
            base_pose: Required runtime pose for floating robots. Fixed robots
                reject this argument.

        Returns:
            Total configuration with trailing dimension ``num_coordinates``.

        Raises:
            ValueError: If a required base pose is missing, a fixed robot is
                given a runtime pose, or a trailing dimension is invalid.
        """
        q_internal = self._check_trailing_dimension(
            "q_internal", q_internal, self.num_internal_dofs
        )
        if not self.floating_base:
            if base_pose is not None:
                raise ValueError("base_pose is only valid for a floating-base robot.")
            return q_internal
        if base_pose is None:
            raise ValueError("base_pose is required for a floating-base robot.")
        base_pose = self._check_trailing_dimension(
            "base_pose", base_pose, self.num_base_coordinates
        )
        base_pose, q_internal = broadcast_leading_axes(base_pose, q_internal)
        return self.normalize_configuration(
            jnp.concatenate([base_pose, q_internal], -1)
        )

    def split_configuration(self, q: Array) -> tuple[Array | None, Array]:
        """Split total configuration into base and internal coordinates.

        Args:
            q: Configuration with trailing dimension ``num_coordinates``.

        Returns:
            ``(q_base, q_internal)``. ``q_base`` is ``None`` for fixed robots.
        """
        q = self._check_trailing_dimension("q", q, self.num_coordinates)
        if not self.floating_base:
            return None, q
        return q[..., : self.num_base_coordinates], q[..., self.num_base_coordinates :]

    def pack_velocity(
        self, qd_internal: Array, *, base_velocity: Array | None = None
    ) -> Array:
        """Pack internal and floating-base generalized velocities.

        Args:
            qd_internal: Internal generalized velocity with trailing dimension
                ``num_internal_dofs``.
            base_velocity: Optional world-frame base velocity. Omission uses
                zero for floating robots; fixed robots reject this argument.

        Returns:
            Total generalized velocity with trailing dimension
            ``num_velocities``.

        Raises:
            ValueError: If a fixed robot is given a base velocity or a trailing
                dimension is invalid.
        """
        qd_internal = self._check_trailing_dimension(
            "qd_internal", qd_internal, self.num_internal_dofs
        )
        if not self.floating_base:
            if base_velocity is not None:
                raise ValueError(
                    "base_velocity is only valid for a floating-base robot."
                )
            return qd_internal
        if base_velocity is None:
            base_velocity = jnp.zeros(
                (*qd_internal.shape[:-1], self.num_base_velocities),
                dtype=qd_internal.dtype,
            )
        base_velocity = self._check_trailing_dimension(
            "base_velocity", base_velocity, self.num_base_velocities
        )
        base_velocity, qd_internal = broadcast_leading_axes(base_velocity, qd_internal)
        return jnp.concatenate([base_velocity, qd_internal], -1)

    def split_velocity(self, v: Array) -> tuple[Array | None, Array]:
        """Split total velocity into base and internal components.

        Args:
            v: Generalized velocity with trailing dimension ``num_velocities``.

        Returns:
            ``(v_base, qd_internal)``. ``v_base`` is ``None`` for fixed robots.
        """
        v = self._check_trailing_dimension("v", v, self.num_velocities)
        if not self.floating_base:
            return None, v
        return v[..., : self.num_base_velocities], v[..., self.num_base_velocities :]

    def pack_state(
        self, q: Array, v: Array, auxiliary_state: Array | None = None
    ) -> Array:
        """Pack configuration, velocity, and optional auxiliary state.

        Args:
            q: Configuration with trailing dimension ``num_coordinates``.
            v: Velocity with trailing dimension ``num_velocities``.
            auxiliary_state: Optional state with trailing dimension
                ``num_auxiliary_states``. Omission uses zeros.

        Returns:
            State ``[q, v, auxiliary]`` with broadcast leading axes and trailing
            dimension ``state_size``.
        """
        q = self._check_trailing_dimension("q", q, self.num_coordinates)
        v = self._check_trailing_dimension("v", v, self.num_velocities)
        if auxiliary_state is None:
            auxiliary_state = jnp.zeros(
                (*q.shape[:-1], self.num_auxiliary_states), dtype=q.dtype
            )
        auxiliary_state = self._check_trailing_dimension(
            "auxiliary_state", auxiliary_state, self.num_auxiliary_states
        )
        q, v, auxiliary_state = broadcast_leading_axes(q, v, auxiliary_state)
        return jnp.concatenate([q, v, auxiliary_state], axis=-1)

    def split_state(self, y: Array) -> tuple[Array, Array, Array]:
        """Split a state with unequal configuration and velocity dimensions.

        Args:
            y: State with trailing dimension ``state_size``.

        Returns:
            ``(q, v, auxiliary_state)`` with preserved leading axes.
        """
        y = self._check_trailing_dimension("y", y, self.state_size)
        q_end = self.num_coordinates
        v_end = q_end + self.num_velocities
        return y[..., :q_end], y[..., q_end:v_end], y[..., v_end:]

    def normalize_configuration(self, q: Array) -> Array:
        """Normalize a floating spatial quaternion without changing hemisphere.

        Args:
            q: Configuration with trailing dimension ``num_coordinates``.

        Returns:
            The unchanged input for fixed or planar robots, or a configuration
            whose leading scalar-first quaternion has unit norm.

        Raises:
            ValueError: If a concrete spatial quaternion is zero or non-finite.
        """
        q = self._check_trailing_dimension("q", q, self.num_coordinates)
        if not self.floating_base or self.is_planar:
            return q
        quaternion = q[..., :4]
        norm_sq = jnp.sum(quaternion * quaternion, axis=-1, keepdims=True)
        if not isinstance(norm_sq, jax.core.Tracer):
            concrete_norm_sq = np.asarray(norm_sq)
            if not np.all(np.isfinite(concrete_norm_sq)):
                raise ValueError("The floating-base quaternion must be finite.")
            if np.any(concrete_norm_sq == 0.0):
                raise ValueError("The floating-base quaternion must have nonzero norm.")
        regular = jnp.isfinite(norm_sq) & (norm_sq > 0.0)
        safe_norm_sq = jnp.where(regular, norm_sq, jnp.ones_like(norm_sq))
        normalized = quaternion / jnp.sqrt(safe_norm_sq)
        identity = jnp.zeros_like(quaternion).at[..., 0].set(1.0)
        normalized = jnp.where(regular, normalized, identity)
        return jnp.concatenate([normalized, q[..., 4:]], axis=-1)

    def project_state(self, y: Array) -> Array:
        """Project a state onto the robot's configuration manifold.

        Fixed and planar systems return the original state without additional
        array operations. Spatial floating systems normalize the leading
        scalar-first quaternion while preserving velocity and auxiliary state.

        Args:
            y: State with trailing dimension ``state_size``.

        Returns:
            State with a unit floating-base quaternion when applicable.
        """
        if not self.floating_base or self.is_planar:
            return y
        q, v, auxiliary_state = self.split_state(y)
        return self.pack_state(self.normalize_configuration(q), v, auxiliary_state)

    def configuration_derivative(self, q: Array, v: Array) -> Array:
        """Map generalized velocity to the derivative of stored coordinates.

        Args:
            q: Configuration with trailing dimension ``num_coordinates``.
            v: Velocity with trailing dimension ``num_velocities``.

        Returns:
            ``qdot`` with trailing dimension ``num_coordinates``. Spatial base
            angular velocity is world-frame and left-multiplies the quaternion.
        """
        q = self._check_trailing_dimension("q", q, self.num_coordinates)
        v = self._check_trailing_dimension("v", v, self.num_velocities)
        q, v = broadcast_leading_axes(q, v)
        if not self.floating_base:
            return v
        v_base, qd_internal = self.split_velocity(v)
        assert v_base is not None
        if self.is_planar:
            return jnp.concatenate([v_base, qd_internal], axis=-1)
        omega_quaternion = jnp.concatenate(
            [jnp.zeros_like(v_base[..., :1]), v_base[..., :3]], axis=-1
        )
        quaternion_dot = 0.5 * quaternion_multiply(omega_quaternion, q[..., :4])
        return jnp.concatenate([quaternion_dot, v_base[..., 3:6], qd_internal], axis=-1)

    def retract_configuration(self, q: Array, delta_v: Array) -> Array:
        """Retract a velocity-space increment onto configuration space.

        Args:
            q: Configuration with trailing dimension ``num_coordinates``.
            delta_v: Tangent increment with trailing dimension
                ``num_velocities``.

        Returns:
            Updated configuration. Spatial rotation increments left-multiply
            the scalar-first quaternion and the result is normalized.
        """
        q = self._check_trailing_dimension("q", q, self.num_coordinates)
        delta_v = self._check_trailing_dimension(
            "delta_v", delta_v, self.num_velocities
        )
        q, delta_v = broadcast_leading_axes(q, delta_v)
        if not self.floating_base:
            return q + delta_v
        delta_base, delta_internal = self.split_velocity(delta_v)
        assert delta_base is not None
        if self.is_planar:
            return q + jnp.concatenate([delta_base, delta_internal], axis=-1)
        rotation = delta_base[..., :3]
        angle = jnp.linalg.norm(rotation, axis=-1, keepdims=True)
        half = 0.5 * angle
        safe_angle = jnp.where(angle > 0.0, angle, jnp.ones_like(angle))
        scale = jnp.where(
            angle > 1e-8,
            jnp.sin(half) / safe_angle,
            0.5 - angle * angle / 48.0,
        )
        increment = jnp.concatenate([jnp.cos(half), scale * rotation], axis=-1)
        quaternion = quaternion_multiply(increment, q[..., :4])
        result = jnp.concatenate(
            [
                quaternion,
                q[..., 4:7] + delta_base[..., 3:6],
                q[..., 7:] + delta_internal,
            ],
            axis=-1,
        )
        return self.normalize_configuration(result)

    def base_transform_from_configuration(self, q: Array) -> Array:
        """Convert the runtime or fixed base pose to homogeneous transforms.

        Args:
            q: Total configuration for floating robots or internal
                configuration for fixed robots.

        Returns:
            A planar ``(..., 3, 3)`` or spatial ``(..., 4, 4)`` transform.
        """
        if self.floating_base:
            base_pose, _ = self.split_configuration(q)
            assert base_pose is not None
        else:
            self._check_trailing_dimension("q", q, self.num_coordinates)
            assert self.fixed_base_pose is not None
            base_pose = self.fixed_base_pose
        converter = (
            poses.planar_pose_to_transform
            if self.is_planar
            else poses.quaternion_pose_to_transform
        )
        if base_pose.ndim == 1:
            return converter(base_pose)
        flat = base_pose.reshape((-1, base_pose.shape[-1]))
        transforms = vmap(converter)(flat)
        return transforms.reshape((*base_pose.shape[:-1], *transforms.shape[-2:]))

    def with_fixed_base_pose(self, pose: Array) -> Self:
        """Return a fixed robot with a different mounting pose.

        Args:
            pose: Planar pose with shape ``(3,)`` or spatial scalar-first
                quaternion pose with shape ``(7,)``.

        Returns:
            A functionally updated robot with mounting-dependent caches refreshed.

        Raises:
            ValueError: If the robot is floating or the pose is invalid.
        """
        if self.floating_base:
            raise ValueError("A floating-base robot has no fixed base pose.")
        pose = jnp.asarray(pose, dtype=jnp.float64)
        expected_shape = (3,) if self.is_planar else (7,)
        if pose.shape != expected_shape:
            raise ValueError(
                f"pose must have shape {expected_shape}, got {pose.shape}."
            )
        if not isinstance(pose, jax.core.Tracer):
            if self.is_planar:
                validate_planar_base_pose("pose", pose)
            else:
                validate_quaternion_base_pose("pose", pose, (7,))
        updated = eqx.tree_at(lambda robot: robot.fixed_base_pose, self, pose)
        transform = (
            poses.planar_pose_to_transform(pose)
            if self.is_planar
            else poses.quaternion_pose_to_transform(pose)
        )
        if hasattr(updated, "g0"):
            updated = eqx.tree_at(lambda robot: robot.g0, updated, transform)
        if hasattr(updated, "gravity_base") and updated.gravity_base is not None:
            adjoint_inverse = (
                se2.adjoint_inverse if self.is_planar else se3.adjoint_inverse
            )
            gravity_base = adjoint_inverse(transform) @ updated.g
            updated = eqx.tree_at(
                lambda robot: robot.gravity_base, updated, gravity_base
            )
        return updated

    def precompute(self) -> None:
        """Optionally initialize state-independent cached quantities.

        Subclasses with cached mass, stiffness, damping, basis, or quadrature
        data can override this method and call it during construction. Models
        without such caches can inherit this no-op implementation. Immutable
        parameter updates must instead refresh dependent caches functionally on
        the model returned by ``with_params(...)``; they must not call this
        potentially mutating constructor hook.
        """
        return None

    def _fixed_base_robot_at_pose(self, base_pose: Array | None = None) -> Self:
        """
        Return a fixed-base copy mounted at the requested pose.

        Floating systems store their base pose in the runtime configuration,
        while existing internal-coordinate algorithms expect a statically
        mounted robot. This method creates the latter without reconstructing
        physical parameters or caches. The result deliberately excludes base
        motion and floating/internal dynamic coupling, and is intended only for
        coordinate transformations and controllers operating on internal
        coordinates.

        Args:
            base_pose: Mounting for the returned robot. Floating robots use
                identity when omitted. Fixed robots return themselves when it
                is omitted.

        Returns:
            The original fixed robot when no change is required, or a shallow
            fixed-base copy whose mounting-dependent transforms are refreshed.
        """
        if not self.floating_base:
            if base_pose is None:
                return self
            return self.with_fixed_base_pose(base_pose)
        pose = self._identity_base_pose if base_pose is None else jnp.asarray(base_pose)
        fixed_robot = copy.copy(self)
        object.__setattr__(fixed_robot, "floating_base", False)
        object.__setattr__(fixed_robot, "fixed_base_pose", pose)
        transform = (
            poses.planar_pose_to_transform(pose)
            if self.is_planar
            else poses.quaternion_pose_to_transform(pose)
        )
        if hasattr(fixed_robot, "g0"):
            object.__setattr__(fixed_robot, "g0", transform)
        if (
            hasattr(fixed_robot, "gravity_base")
            and fixed_robot.gravity_base is not None
        ):
            adjoint_inverse = (
                se2.adjoint_inverse if self.is_planar else se3.adjoint_inverse
            )
            object.__setattr__(
                fixed_robot,
                "gravity_base",
                adjoint_inverse(transform) @ fixed_robot.g,
            )
        return fixed_robot

    def _configure_actuation(
        self,
        actuators: Actuator | tuple[Actuator, ...] | None,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
    ) -> None:
        """Install immutable actuator/passive components after DOFs are known."""
        if actuators is None:
            normalized_actuators = (IdentityActuator(self.num_internal_dofs),)
        elif isinstance(actuators, Actuator):
            normalized_actuators = (actuators,)
        else:
            normalized_actuators = tuple(actuators)
        if any(not isinstance(actuator, Actuator) for actuator in normalized_actuators):
            raise TypeError("actuators must contain only Actuator instances.")

        if passive_elements is None:
            normalized_passive = ()
        elif isinstance(passive_elements, PassiveElement):
            normalized_passive = (passive_elements,)
        else:
            normalized_passive = tuple(passive_elements)
        if any(
            not isinstance(element, PassiveElement) for element in normalized_passive
        ):
            raise TypeError(
                "passive_elements must contain only PassiveElement instances."
            )

        for component in (*normalized_actuators, *normalized_passive):
            component.validate_for_robot(self)

        self.actuators = normalized_actuators
        self.passive_elements = normalized_passive
        self.num_actuators = sum(actuator.num_channels for actuator in self.actuators)

    @property
    def actuator_input_metadata(self) -> tuple[ActuatorMetadata, ...]:
        """Metadata groups in the same order used to concatenate controls."""
        return tuple(actuator.metadata for actuator in self.actuators)

    def with_actuator_params(self, index: int, params) -> Self:
        """Return a robot with one actuator's complete parameter object replaced."""
        if not 0 <= index < len(self.actuators):
            raise IndexError(f"actuator index {index} is out of range.")
        actuators = list(self.actuators)
        actuators[index] = actuators[index].with_params(params)
        actuators[index].validate_for_robot(self)
        return eqx.tree_at(lambda robot: robot.actuators, self, tuple(actuators))

    def update_actuator_params(self, index: int, **updates: Any) -> Self:
        """Return a robot with selected fields of one actuator's params replaced."""
        if not 0 <= index < len(self.actuators):
            raise IndexError(f"actuator index {index} is out of range.")
        return self.with_actuator_params(
            index, self.actuators[index].params.replace(**updates)
        )

    def with_passive_element_params(self, index: int, params) -> Self:
        """Return a robot with one passive element's complete params replaced."""
        if not 0 <= index < len(self.passive_elements):
            raise IndexError(f"passive element index {index} is out of range.")
        elements = list(self.passive_elements)
        elements[index] = elements[index].with_params(params)
        elements[index].validate_for_robot(self)
        return eqx.tree_at(lambda robot: robot.passive_elements, self, tuple(elements))

    def update_passive_element_params(self, index: int, **updates: Any) -> Self:
        """Return a robot with selected passive-element parameter fields replaced."""
        if not 0 <= index < len(self.passive_elements):
            raise IndexError(f"passive element index {index} is out of range.")
        return self.with_passive_element_params(
            index, self.passive_elements[index].params.replace(**updates)
        )

    @property
    def length(self) -> Array:
        """Total backbone length of the robot (scalar)."""
        return jnp.sum(jnp.atleast_1d(jnp.asarray(self.segment_length)))

    @property
    @abstractmethod
    def segment_length(self) -> Array:
        """Per-segment backbone lengths (1D array)."""
        ...

    @property
    @abstractmethod
    def is_planar(self) -> bool:
        """Return True for planar (SE(2)) robots, False for spatial (SE(3))."""
        ...

    @property
    def supports_articulated_tendon_routing(self) -> bool:
        """Whether joint indices describe a serial articulated routing topology."""
        return False

    @property
    def base_transform(self) -> Array:
        """Return the homogeneous transform of the fixed/internal base pose.

        Planar robots consume ``[theta, x, y]`` and return an SE(2) matrix with
        shape ``(3, 3)``. Spatial robots consume
        ``[qw, qx, qy, qz, x, y, z]`` and return an SE(3) matrix with shape
        ``(4, 4)``. Spatial quaternions are scalar-first Hamilton quaternions.
        """
        pose = self._kinematic_base_pose
        if self.is_planar:
            return poses.planar_pose_to_transform(jnp.asarray(pose))
        return poses.quaternion_pose_to_transform(jnp.asarray(pose))

    @abstractmethod
    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Return integer cross-section tag and geometry parameters at position s.

        Tag values:
            - CrossSectionGeometry.CIRCULAR
            - CrossSectionGeometry.RECTANGULAR
            - CrossSectionGeometry.ELLIPTICAL

        Rectangular parameters use ``[height, width]`` order, matching
        :func:`soromox.systems.components.section_properties`. Elliptical
        parameters use ``[semi_major, semi_minor]`` order.
        """
        ...

    # -----------------------------------------
    # Arc-length parameterized kinematics
    # -----------------------------------------

    def forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics at a point s along the robot.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            s: Position parameter along the robot structure. The meaning depends
                on the specific robot type:
                - For continuum robots (PCS, PlanarPCS): arc-length in [0, L_total]
                - For articulated robots (Pendulum): can be link index or fraction

        Returns:
            chi: Pose at point s. The shape and meaning depend on the robot type:
                - For 3D robots (PCS): SE(3) transformation matrix, shape (4, 4)
                - For planar robots (PlanarPCS, Pendulum): [theta, x, y], shape (3,)
        """
        if self.floating_base:
            return self._floating_forward_kinematics(q, s)
        if custom_jvp_enabled():
            return SoftRobot._forward_kinematics_custom_jvp(self, q, s)
        return self._forward_kinematics(q, s)

    def _floating_forward_kinematics(self, q: Array, s: Array) -> Array:
        """Compose relative model kinematics with the runtime base pose.

        Args:
            q: Total floating configuration with shape ``(num_coordinates,)``.
            s: Scalar evaluation abscissa.

        Returns:
            Absolute planar pose or spatial homogeneous transform at ``s``.
        """
        base_pose, q_internal = self.split_configuration(q)
        assert base_pose is not None
        relative_robot = self._fixed_base_robot_at_pose()
        relative_pose = relative_robot._forward_kinematics(q_internal, s)
        base_transform = self.base_transform_from_configuration(q)
        if self.is_planar:
            relative_transform = self._homogeneous_pose(relative_pose)
            return poses.planar_pose_from_transform(
                base_transform @ relative_transform, self.global_eps
            )
        return base_transform @ relative_pose

    def _absolute_forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Evaluate absolute forward kinematics.

        System methods use JAX by convention. This semantic helper differs from
        :meth:`_forward_kinematics` only by composing the runtime base when the
        robot is floating. Accelerated backends also use it for derivative
        substitution.

        Args:
            q: Fixed internal or floating total configuration.
            s: Scalar evaluation abscissa.

        Returns:
            The absolute model pose at ``s``.
        """
        if self.floating_base:
            return self._floating_forward_kinematics(q, s)
        return self._forward_kinematics(q, s)

    def _absolute_inertial_jacobian(self, q: Array, s: Array) -> Array:
        """
        Evaluate an absolute inertial-frame Jacobian.

        Fixed models return their internal-coordinate Jacobian. Floating models
        rotate those columns into the world frame and prepend the runtime base
        columns. Accelerated backends also use this method for derivative
        substitution.

        Args:
            q: Fixed internal or floating total configuration.
            s: Scalar evaluation abscissa.

        Returns:
            The fixed or augmented inertial-frame Jacobian at ``s``.
        """
        if not self.floating_base:
            return self._jacobian_inertialframe(q, s)
        _, q_internal = self.split_configuration(q)
        relative_robot = self._fixed_base_robot_at_pose()
        relative_pose = relative_robot._forward_kinematics(q_internal, s)
        relative_transform = self._homogeneous_pose(relative_pose)
        jacobian_internal = relative_robot._jacobian_inertialframe(q_internal, s)
        return self._floating_jacobians_from_relative_inertial(
            q, relative_transform, jacobian_internal
        )

    def _absolute_forward_kinematics_abscissa_batched(
        self, q: Array, s: Array
    ) -> Array:
        """
        Evaluate absolute forward kinematics at multiple abscissae.

        Args:
            q: Fixed internal or floating total configuration.
            s: One-dimensional batch of evaluation abscissae.

        Returns:
            Absolute poses with a leading abscissa axis.
        """
        if not self.floating_base:
            return self._forward_kinematics_abscissa_batched(q, s)
        _, q_internal = self.split_configuration(q)
        relative_robot = self._fixed_base_robot_at_pose()
        relative_poses = relative_robot._forward_kinematics_abscissa_batched(
            q_internal, s
        )
        return self._compose_runtime_base_poses(q, relative_poses)

    def _absolute_inertial_jacobian_abscissa_batched(self, q: Array, s: Array) -> Array:
        """
        Evaluate absolute inertial-frame Jacobians at multiple abscissae.

        Args:
            q: Fixed internal or floating total configuration.
            s: One-dimensional batch of evaluation abscissae.

        Returns:
            Fixed or augmented Jacobians with a leading abscissa axis.
        """
        if not self.floating_base:
            return self._jacobian_inertialframe_abscissa_batched(q, s)
        _, q_internal = self.split_configuration(q)
        relative_robot = self._fixed_base_robot_at_pose()
        relative_poses = relative_robot._forward_kinematics_abscissa_batched(
            q_internal, s
        )
        relative_transforms = vmap(self._homogeneous_pose)(relative_poses)
        jacobians_internal = (
            relative_robot._jacobian_inertialframe_abscissa_batched(q_internal, s)
        )
        return self._floating_jacobians_from_relative_inertial(
            q, relative_transforms, jacobians_internal
        )

    def _compose_runtime_base_poses(self, q: Array, relative_poses: Array) -> Array:
        """Left-compose the runtime base with batched relative poses.

        Args:
            q: Total floating configuration with shape ``(num_coordinates,)``.
            relative_poses: Relative planar poses or spatial transforms with one
                leading sample axis.

        Returns:
            Absolute poses with the same sample axis and pose representation.
        """
        base_transform = self.base_transform_from_configuration(q)
        if self.is_planar:
            relative_transforms = vmap(poses.planar_pose_to_transform)(relative_poses)
            absolute = jnp.einsum("ab,nbc->nac", base_transform, relative_transforms)
            return vmap(
                lambda transform: poses.planar_pose_from_transform(
                    transform, self.global_eps
                )
            )(absolute)
        return jnp.einsum("ab,nbc->nac", base_transform, relative_poses)

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """Evaluate model-relative forward kinematics at one abscissa.

        Concrete systems override this protected hook with their specialized
        internal-coordinate recurrence. Public methods compose a runtime base
        separately when floating-base state is enabled.

        Args:
            q: Internal generalized coordinates with shape
                ``(num_internal_dofs,)``.
            s: Scalar backbone abscissa.

        Returns:
            A planar pose vector or spatial homogeneous transform relative to
            the model's kinematic base.

        Raises:
            NotImplementedError: If a concrete system does not implement the
                kinematic recurrence.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_kinematics."
        )

    @eqx.filter_custom_jvp
    @staticmethod
    def _forward_kinematics_custom_jvp(robot: "SoftRobot", q: Array, s: Array) -> Array:
        """Custom-JVP entry point for the public forward_kinematics wrapper."""
        return robot._forward_kinematics(q, s)

    @_forward_kinematics_custom_jvp.def_jvp
    def _forward_kinematics_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[Array, Array]:
        robot, q, s = primals
        _, qd, sd = tangents

        return robot._forward_kinematics_jvp(q, s, qd, sd)

    def _forward_kinematics_jvp(
        self, q: Array, s: Array, qd: Array | None, sd: Array | None
    ) -> tuple[Array, Array]:
        """
        Evaluate the FK custom-JVP tangent with the narrowest needed hooks.

        The arc-length-only tangent uses the analytical
        ``forward_kinematics_and_arc_length_derivative`` hook when available,
        because systems often have a compact closed-form expression for
        ``d pose / ds``. Configuration-only and mixed ``q``/``s`` tangents
        intentionally use JAX autodiff through ``_forward_kinematics``. Earlier
        specialized FK velocity paths duplicated substantial kinematics code
        and did not show enough benchmark benefit to justify the maintenance
        cost.
        """
        forward_kinematics = self._absolute_forward_kinematics
        if qd is not None:
            if sd is None:
                return jvp(lambda q_: forward_kinematics(q_, s), (q,), (qd,))
            return jvp(
                lambda q_, s_: forward_kinematics(q_, s_),
                (q, s),
                (qd, sd),
            )

        if sd is None:
            pose = forward_kinematics(q, s)
            return pose, jnp.zeros_like(pose)

        if self.floating_base:
            return jvp(lambda s_: forward_kinematics(q, s_), (s,), (sd,))

        pose, poses = self.forward_kinematics_and_arc_length_derivative(q, s)
        return pose, poses * sd

    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics at all segment or link tips.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            chi_tips: Poses at the robot tips. The shape and meaning depend on
                the robot type:
                - For 3D robots (PCS, GVS): shape (num_segments, 4, 4)
                - For planar robots (PlanarPCS, Pendulum): shape (num_segments, 3)
        """
        s_tips = jnp.cumsum(jnp.atleast_1d(jnp.asarray(self.segment_length)))
        return vmap(lambda s: self.forward_kinematics(q, s))(s_tips)

    def forward_kinematics_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics at multiple points along the robot.

        Default implementation uses vmap over forward_kinematics.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            chi_ps: Poses at all points, shape depends on robot type.
        """
        return vmap(lambda s: self.forward_kinematics(q, s))(s_ps)

    def forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the forward kinematics at ``s``.

        The returned tangent has the same shape and representation as
        ``forward_kinematics(q, s)``.
        """
        return self._forward_kinematics_arc_length_derivative(q, s)

    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Protected arc-length derivative hook for forward kinematics.

        Subclasses can override this with an analytical expression. The default
        implementation differentiates the protected primal hook with respect to
        the scalar arc-length parameter ``s``.
        """
        _, pose_s = jvp(
            lambda s_: self._forward_kinematics(q, s_),
            (s,),
            (jnp.ones_like(jnp.asarray(s)),),
        )
        return pose_s

    def forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute forward kinematics and its arc-length derivative at ``s``.

        Subclasses can override the protected hook to share intermediate
        kinematic quantities between the primal pose and ``d pose / ds``.
        """
        return self._forward_kinematics_and_arc_length_derivative(q, s)

    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """Protected hook for fused FK and arc-length derivative evaluation."""
        pose = self._forward_kinematics(q, s)
        poses = self._forward_kinematics_arc_length_derivative(q, s)
        return pose, poses

    def jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot.

        The Jacobian maps configuration space velocities to operational space
        (Cartesian/task space) velocities at point s.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            s: Position parameter along the robot structure.

        Returns:
            J: Jacobian matrix of shape (n_pose_dim, num_velocities), where n_pose_dim
                depends on the robot type:
                - For 3D robots (PCS): 6 (angular velocity + linear velocity)
                - For planar robots (PlanarPCS, Pendulum): 3 (omega_z, v_x, v_y)
        """
        if self.floating_base:
            return self._floating_jacobian(q, s)
        if custom_jvp_enabled():
            return SoftRobot._jacobian_custom_jvp(self, q, s)
        return self._jacobian(q, s)

    def _floating_jacobian(self, q: Array, s: Array) -> Array:
        """Assemble a world-frame Jacobian for total velocity.

        Args:
            q: Total floating configuration with shape ``(num_coordinates,)``.
            s: Scalar evaluation abscissa.

        Returns:
            Jacobian with shape ``(spatial_dimension, num_velocities)`` and
            world-frame angular-first velocity convention.
        """
        base_pose, q_internal = self.split_configuration(q)
        assert base_pose is not None
        relative_pose = self._forward_kinematics(q_internal, s)
        J_internal_relative = self._jacobian(q_internal, s)
        base_transform = self.base_transform_from_configuration(q)
        if self.is_planar:
            relative_transform = self._homogeneous_pose(relative_pose)
            world_transform = base_transform @ relative_transform
            R_base = base_transform[:2, :2]
            J_internal = jnp.concatenate(
                [J_internal_relative[:1], R_base @ J_internal_relative[1:]], axis=0
            )
            r = world_transform[:2, 2] - base_pose[1:3]
            J_base = jnp.array(
                [[1.0, 0.0, 0.0], [-r[1], 1.0, 0.0], [r[0], 0.0, 1.0]],
                dtype=q.dtype,
            )
        else:
            world_transform = base_transform @ relative_pose
            R_base = base_transform[:3, :3]
            J_internal = jnp.concatenate(
                [R_base @ J_internal_relative[:3], R_base @ J_internal_relative[3:]],
                axis=0,
            )
            r = world_transform[:3, 3] - base_pose[4:7]
            J_base = jnp.block(
                [
                    [jnp.eye(3, dtype=q.dtype), jnp.zeros((3, 3), dtype=q.dtype)],
                    [-so3.skew(r), jnp.eye(3, dtype=q.dtype)],
                ]
            )
        return jnp.concatenate([J_base, J_internal], axis=1)

    def _jacobian(self, q: Array, s: Array) -> Array:
        """
        Protected primal Jacobian hook.

        Default implementation differentiates the protected forward-kinematics
        hook. Subclasses with analytical Jacobians should override this hook.
        """
        pose = self._forward_kinematics(q, s)
        dpose_dq = jacfwd(lambda q_: self._forward_kinematics(q_, s))(q)

        if self.is_planar:
            return self._planar_pose_jacobian(pose, dpose_dq)

        return self._spatial_pose_jacobian(pose, dpose_dq)

    @eqx.filter_custom_jvp
    @staticmethod
    def _jacobian_custom_jvp(robot: "SoftRobot", q: Array, s: Array) -> Array:
        """Custom-JVP entry point for the public jacobian wrapper."""
        return robot._jacobian(q, s)

    @_jacobian_custom_jvp.def_jvp
    def _jacobian_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[Array, Array]:
        robot, q, s = primals
        _, qd, sd = tangents

        return robot._jacobian_jvp(q, s, qd, sd)

    def _jacobian_jvp(
        self, q: Array, s: Array, qd: Array | None, sd: Array | None
    ) -> tuple[Array, Array]:
        """
        Evaluate the Jacobian custom-JVP tangent with specialized hooks.

        The mixed ``q``/``s`` tangent intentionally falls back to JAX autodiff
        through ``_jacobian``. Benchmarks showed that maintaining a
        separate fused custom-JVP path for this case does not pay for its added
        complexity.
        """
        if qd is None and sd is None:
            J = self._jacobian(q, s)
            return J, jnp.zeros_like(J)

        if qd is None:
            J, Js = self.jacobian_and_arc_length_derivative(q, s)
            return J, Js * sd

        if sd is None:
            return self.jacobian_and_time_derivative(q, qd, s)

        return jvp(
            lambda q_, s_: self._jacobian(q_, s_),
            (q, s),
            (qd, sd),
        )

    def jacobian_tips(self, q: Array) -> Array:
        """
        Compute inertial-frame Jacobians at all segment or link tips.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            J_tips: Inertial-frame Jacobians at the robot tips, with shape
                (num_tips, n_pose_dim, num_velocities).
        """
        s_tips = jnp.cumsum(jnp.atleast_1d(jnp.asarray(self.segment_length)))
        return vmap(lambda s: self.jacobian(q, s))(s_tips)

    def jacobian_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian at multiple points along the robot.

        Default implementation uses vmap over jacobian.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            J_ps: Jacobians at all points, shape (N, n_pose_dim, num_velocities).
        """
        return vmap(lambda s: self.jacobian(q, s))(s_ps)

    def jacobian_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the Jacobian at ``s``.

        The returned derivative has the same shape and frame convention as
        ``jacobian(q, s)``.
        """
        return self._jacobian_arc_length_derivative(q, s)

    def _jacobian_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Protected arc-length derivative hook for the Jacobian.

        Subclasses can override this with an analytical expression. The default
        implementation differentiates the protected Jacobian hook with respect
        to the scalar arc-length parameter ``s``.
        """
        _, Js = jvp(
            lambda s_: self._jacobian(q, s_),
            (s,),
            (jnp.ones_like(jnp.asarray(s)),),
        )
        return Js

    def jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its arc-length derivative at ``s``.

        Returns:
            J: Jacobian matrix of shape (n_pose_dim, num_velocities).
            Js: Arc-length derivative of the Jacobian with the same shape as ``J``.
        """
        return self._jacobian_and_arc_length_derivative(q, s)

    def _jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """Protected hook for fused Jacobian and arc-length derivative evaluation."""
        J = self._jacobian(q, s)
        Js = self._jacobian_arc_length_derivative(q, s)
        return J, Js

    def jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time derivative at a point s along the robot.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            qd: Generalized velocities of shape (num_velocities,).
            s: Position parameter along the robot structure.

        Returns:
            J: Jacobian matrix of shape (n_pose_dim, num_velocities).
            Jd: Time derivative of the Jacobian, shape (n_pose_dim, num_velocities).
        """
        if self.floating_base:
            qdot = self.configuration_derivative(q, qd)
            return jvp(lambda q_: self._floating_jacobian(q_, s), (q,), (qdot,))
        return self._jacobian_and_time_derivative(q, qd, s)

    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Protected primal Jacobian-derivative hook.

        Default implementation differentiates the protected Jacobian hook to
        avoid recursively invoking the public custom-JVP Jacobian wrapper.
        """
        J, Jd = jvp(lambda q_: self._jacobian(q_, s), (q,), (qd,))
        return J, Jd

    def jacobian_and_time_derivative_abscissa_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its derivative at multiple points along the robot.

        Default implementation uses vmap over jacobian_and_time_derivative.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            qd: Generalized velocities of shape (num_velocities,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            J_ps: Jacobians at all points, shape (N, n_pose_dim, num_velocities).
            Jd_ps: Jacobian time derivatives at all points, shape (N, n_pose_dim, num_velocities).
        """
        return vmap(lambda s: self.jacobian_and_time_derivative(q, qd, s))(s_ps)

    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the body-frame Jacobian at a point ``s`` along the robot.

        The default implementation converts the public inertial-frame Jacobian
        to the local pose frame using the rotation-only convention used by the
        dynamics integrands.
        """
        pose = self.forward_kinematics(q, s)
        J = self.jacobian(q, s)
        return self._inertial_jacobian_to_bodyframe(pose, J)

    def jacobian_bodyframe_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute body-frame Jacobians at multiple points along the robot.
        """
        return vmap(lambda s: self.jacobian_bodyframe(q, s))(s_ps)

    def jacobian_and_time_derivative_bodyframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute a body-frame Jacobian and its time derivative at ``s``.

        The default differentiates ``jacobian_bodyframe`` with respect to the
        generalized coordinates.
        """
        J, Jd = jvp(lambda q_: self.jacobian_bodyframe(q_, s), (q,), (qd,))
        return J, Jd

    def jacobian_and_time_derivative_bodyframe_abscissa_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute body-frame Jacobians and time derivatives at many points.
        """
        return vmap(lambda s: self.jacobian_and_time_derivative_bodyframe(q, qd, s))(
            s_ps
        )

    def integration_kinematics(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Evaluate poses, body-frame Jacobians, and Jacobian derivatives at
        interior integration nodes.

        The default implementation is intentionally unfused. It samples
        ``integration_points[..., 1:-1]`` for every segment, maps normalized
        nodes to arclength, and calls the public kinematics/Jacobian APIs.
        Subclasses can override this method with a fused implementation.

        Returns:
            Tuple ``(g_ps, J_ps, Jd_ps)`` with leading axes
            ``(num_segments, num_inner_points)``.
        """
        if self.integration_points is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define integration_points or override "
                "integration_kinematics."
            )

        segment_lengths = jnp.atleast_1d(jnp.asarray(self.segment_length))
        segment_starts = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=segment_lengths.dtype),
                jnp.cumsum(segment_lengths)[:-1],
            ]
        )
        points = jnp.asarray(self.integration_points)

        if points.ndim == 1:
            inner_points = points[1:-1]
            s_ps = (
                segment_starts[:, None]
                + segment_lengths[:, None] * inner_points[None, :]
            )
        elif points.ndim == 2:
            inner_points = points[:, 1:-1]
            s_ps = segment_starts[:, None] + segment_lengths[:, None] * inner_points
        else:
            raise ValueError(
                "integration_points must have shape (num_points,) or "
                "(num_segments, num_points)."
            )

        s_flat = s_ps.reshape(-1)
        g_flat = vmap(lambda s: self._homogeneous_forward_kinematics(q, s))(s_flat)
        J_flat, Jd_flat = self.jacobian_and_time_derivative_bodyframe_abscissa_batched(
            q, qd, s_flat
        )

        leading_shape = s_ps.shape
        g_ps = g_flat.reshape(*leading_shape, *g_flat.shape[1:])
        J_ps = J_flat.reshape(*leading_shape, *J_flat.shape[1:])
        Jd_ps = Jd_flat.reshape(*leading_shape, *Jd_flat.shape[1:])
        return g_ps, J_ps, Jd_ps

    def _floating_base_body_kinematics(
        self, q: Array, v: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Initialize a floating-base body-frame recurrence.

        The returned Jacobian maps the world-frame velocity at the base origin
        into the base frame. The convective vector is the exact contraction of
        its configuration derivative with the current base velocity. Concrete
        systems propagate these four values through their existing local
        recurrences and accumulate inertial terms immediately.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            v: Total floating-base generalized velocity with shape
                ``(num_velocities,)``.

        Returns:
            Tuple ``(J_base, Jd_v_base, velocity_base, gravity_base)`` expressed
            in the runtime base frame.
        """
        base_velocity, _ = self.split_velocity(v)
        assert base_velocity is not None
        transform = self.base_transform_from_configuration(q)
        jacobian = self._floating_base_body_jacobian(q)
        if self.is_planar:
            rotation_transpose = transform[:2, :2].T
            velocity = jacobian @ base_velocity
            rotation_generator = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=q.dtype)
            jacobian_dot_velocity = jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=q.dtype),
                    -velocity[0] * rotation_generator @ velocity[1:],
                ]
            )
            gravity = jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=q.dtype),
                    rotation_transpose @ jnp.asarray(self.g[-2:], dtype=q.dtype),
                ]
            )
        else:
            rotation_transpose = transform[:3, :3].T
            velocity = jacobian @ base_velocity
            jacobian_dot_velocity = jnp.concatenate(
                [
                    jnp.zeros((3,), dtype=q.dtype),
                    -jnp.cross(velocity[:3], velocity[3:]),
                ]
            )
            gravity = jnp.concatenate(
                [
                    jnp.zeros((3,), dtype=q.dtype),
                    rotation_transpose @ jnp.asarray(self.g[-3:], dtype=q.dtype),
                ]
            )
        return jacobian, jacobian_dot_velocity, velocity, gravity

    def _floating_base_body_jacobian(self, q: Array) -> Array:
        """
        Return the root body Jacobian for world-frame base velocity.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.

        Returns:
            Root body Jacobian with shape ``(3, 3)`` for planar systems or
            ``(6, 6)`` for spatial systems.
        """
        transform = self.base_transform_from_configuration(q)
        if self.is_planar:
            jacobian = jnp.zeros((3, 3), dtype=q.dtype)
            jacobian = jacobian.at[0, 0].set(1.0)
            return jacobian.at[1:, 1:].set(transform[:2, :2].T)

        rotation_transpose = transform[:3, :3].T
        jacobian = jnp.zeros((6, 6), dtype=q.dtype)
        jacobian = jacobian.at[:3, :3].set(rotation_transpose)
        return jacobian.at[3:, 3:].set(rotation_transpose)

    def _floating_inertial_jacobians(
        self,
        q: Array,
        relative_transforms: Array,
        jacobians_internal: Array,
    ) -> Array:
        """
        Add base columns and rotate body Jacobians into the inertial frame.

        The relative transforms and internal body Jacobians are intended to be
        outputs of a system's existing fused batched recurrence. No kinematic
        quantity is recomputed by this adapter.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            relative_transforms: Base-relative transforms with arbitrary
                leading dimensions.
            jacobians_internal: Internal body Jacobians with matching leading
                dimensions.

        Returns:
            Inertial-frame total Jacobians with matching leading dimensions and
            ``num_velocities`` columns.
        """
        spatial_dimension = 3 if self.is_planar else 6
        transform_dimension = 3 if self.is_planar else 4
        leading_shape = relative_transforms.shape[:-2]
        transforms_flat = relative_transforms.reshape(
            -1, transform_dimension, transform_dimension
        )
        jacobians_internal_flat = jacobians_internal.reshape(
            -1, spatial_dimension, self.num_internal_dofs
        )
        adjoint_inverse = se2.adjoint_inverse if self.is_planar else se3.adjoint_inverse
        relative_adjoint_inverses = vmap(adjoint_inverse)(transforms_flat)
        jacobian_base = self._floating_base_body_jacobian(q)
        jacobians_base = jnp.einsum(
            "nij,jk->nik", relative_adjoint_inverses, jacobian_base
        )
        jacobians_body = jnp.concatenate(
            [jacobians_base, jacobians_internal_flat], axis=-1
        )

        base_transform = self.base_transform_from_configuration(q)
        world_transforms = jnp.einsum("ij,njk->nik", base_transform, transforms_flat)
        if self.is_planar:
            rotations = world_transforms[:, :2, :2]
            rotation_actions = jnp.zeros(
                (world_transforms.shape[0], 3, 3), dtype=q.dtype
            )
            rotation_actions = rotation_actions.at[:, 0, 0].set(1.0)
            rotation_actions = rotation_actions.at[:, 1:, 1:].set(rotations)
        else:
            rotations = world_transforms[:, :3, :3]
            rotation_actions = jnp.zeros(
                (world_transforms.shape[0], 6, 6), dtype=q.dtype
            )
            rotation_actions = rotation_actions.at[:, :3, :3].set(rotations)
            rotation_actions = rotation_actions.at[:, 3:, 3:].set(rotations)
        jacobians_world = jnp.einsum("nij,njk->nik", rotation_actions, jacobians_body)
        return jacobians_world.reshape(
            *leading_shape, spatial_dimension, self.num_velocities
        )

    def _floating_jacobians_from_relative_inertial(
        self,
        q: Array,
        relative_transforms: Array,
        jacobians_internal: Array,
    ) -> Array:
        """
        Augment base-relative inertial Jacobians with floating-base columns.

        This adapter is used after a fixed-base JAX or Warp recurrence has
        already rotated the internal Jacobian columns into the base coordinate
        frame. It rotates those columns into the world frame and constructs the
        base columns without repeating the relative kinematic recurrence.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            relative_transforms: Base-relative transforms with arbitrary
                leading dimensions.
            jacobians_internal: Base-frame inertial Jacobians with matching
                leading dimensions and ``num_internal_dofs`` columns.

        Returns:
            World-frame Jacobians with matching leading dimensions and
            ``num_velocities`` columns.
        """
        spatial_dimension = 3 if self.is_planar else 6
        transform_dimension = 3 if self.is_planar else 4
        leading_shape = relative_transforms.shape[:-2]
        transforms_flat = relative_transforms.reshape(
            -1, transform_dimension, transform_dimension
        )
        jacobians_internal_flat = jacobians_internal.reshape(
            -1, spatial_dimension, self.num_internal_dofs
        )

        adjoint_inverse = se2.adjoint_inverse if self.is_planar else se3.adjoint_inverse
        relative_adjoint_inverses = vmap(adjoint_inverse)(transforms_flat)
        root_jacobian = self._floating_base_body_jacobian(q)
        jacobians_base_body = jnp.einsum(
            "nij,jk->nik", relative_adjoint_inverses, root_jacobian
        )

        base_transform = self.base_transform_from_configuration(q)
        base_rotation = (
            base_transform[:2, :2] if self.is_planar else base_transform[:3, :3]
        )
        world_transforms = jnp.einsum("ij,njk->nik", base_transform, transforms_flat)
        if self.is_planar:
            base_rotation_action = jnp.zeros((3, 3), dtype=q.dtype)
            base_rotation_action = base_rotation_action.at[0, 0].set(1.0)
            base_rotation_action = base_rotation_action.at[1:, 1:].set(base_rotation)
            world_rotation_actions = jnp.zeros(
                (world_transforms.shape[0], 3, 3), dtype=q.dtype
            )
            world_rotation_actions = world_rotation_actions.at[:, 0, 0].set(1.0)
            world_rotation_actions = world_rotation_actions.at[:, 1:, 1:].set(
                world_transforms[:, :2, :2]
            )
        else:
            base_rotation_action = jnp.zeros((6, 6), dtype=q.dtype)
            base_rotation_action = base_rotation_action.at[:3, :3].set(base_rotation)
            base_rotation_action = base_rotation_action.at[3:, 3:].set(base_rotation)
            world_rotation_actions = jnp.zeros(
                (world_transforms.shape[0], 6, 6), dtype=q.dtype
            )
            world_rotation_actions = world_rotation_actions.at[:, :3, :3].set(
                world_transforms[:, :3, :3]
            )
            world_rotation_actions = world_rotation_actions.at[:, 3:, 3:].set(
                world_transforms[:, :3, :3]
            )

        jacobians_base_world = jnp.einsum(
            "nij,njk->nik", world_rotation_actions, jacobians_base_body
        )
        jacobians_internal_world = jnp.einsum(
            "ij,njk->nik", base_rotation_action, jacobians_internal_flat
        )
        jacobians_world = jnp.concatenate(
            [jacobians_base_world, jacobians_internal_world], axis=-1
        )
        return jacobians_world.reshape(
            *leading_shape, spatial_dimension, self.num_velocities
        )

    def _floating_world_positions(self, q: Array, relative_positions: Array) -> Array:
        """
        Transform base-relative points into the inertial frame.

        This helper deliberately accepts positions rather than poses or spatial
        inertias so gravitational energy does not construct unused Jacobians or
        matrices.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            relative_positions: Base-relative points with trailing dimension two
                for planar systems or three for spatial systems.

        Returns:
            Inertial-frame points with the same leading dimensions as
            ``relative_positions``.
        """
        transform = self.base_transform_from_configuration(q)
        linear_dimension = 2 if self.is_planar else 3
        rotation = transform[:linear_dimension, :linear_dimension]
        translation = transform[:linear_dimension, linear_dimension]
        return jnp.einsum("ij,...j->...i", rotation, relative_positions) + translation

    def _floating_gravitational_energy_from_points(
        self, q: Array, relative_positions: Array, masses: Array
    ) -> Array:
        """
        Compute floating-base gravitational energy from point masses.

        Quadrature weights must already be folded into ``masses``. This direct
        contraction is intended for system-specific energy paths and never
        constructs Jacobians or spatial inertia matrices.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            relative_positions: Base-relative mass locations with trailing
                dimension two or three.
            masses: Scalar masses broadcastable to the leading point axes.

        Returns:
            Absolute gravitational potential energy as a scalar.
        """
        positions = self._floating_world_positions(q, relative_positions)
        linear_dimension = positions.shape[-1]
        gravity = jnp.asarray(self.g[-linear_dimension:], dtype=q.dtype)
        return -jnp.sum(masses * (positions @ gravity))

    def _augment_floating_body_kinematics(
        self,
        q: Array,
        v: Array,
        relative_transforms: Array,
        jacobians_internal: Array,
        jacobian_dot_velocity_internal: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Add floating-base columns to existing body-frame kinematics.

        This helper consumes quantities already produced by a system's fused
        recurrence. It performs no kinematic propagation and is therefore
        suitable for discrete mass terms such as HSA platforms and payloads.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            v: Total generalized velocity with shape ``(num_velocities,)``.
            relative_transforms: Base-relative body transforms with arbitrary
                leading dimensions.
            jacobians_internal: Body Jacobians with the same leading dimensions
                and trailing shape
                ``(spatial_dimension, num_internal_dofs)``.
            jacobian_dot_velocity_internal: Internal convective contractions
                with trailing shape ``(spatial_dimension,)``.

        Returns:
            Tuple ``(J, Jd_v, velocity, gravity)`` with the original leading
            dimensions and total generalized-velocity columns.
        """
        _, qd_internal = self.split_velocity(v)
        spatial_dimension = 3 if self.is_planar else 6
        transform_dimension = 3 if self.is_planar else 4
        leading_shape = relative_transforms.shape[:-2]
        transforms_flat = relative_transforms.reshape(
            -1, transform_dimension, transform_dimension
        )
        jacobians_internal_flat = jacobians_internal.reshape(
            -1, spatial_dimension, self.num_internal_dofs
        )
        jacobian_dot_velocity_internal_flat = jacobian_dot_velocity_internal.reshape(
            -1, spatial_dimension
        )
        (
            jacobian_base,
            jacobian_dot_velocity_base,
            velocity_base,
            gravity_base,
        ) = self._floating_base_body_kinematics(q, v)
        adjoint_inverse_function = (
            se2.adjoint_inverse if self.is_planar else se3.adjoint_inverse
        )
        if self.is_planar:

            def small_adjoint_action(left: Array, right: Array) -> Array:
                return se2.small_adjoint(left) @ right
        else:
            small_adjoint_action = se3.small_adjoint_action
        adjoint_inverses = vmap(adjoint_inverse_function)(transforms_flat)
        jacobians_base = jnp.einsum("nij,jk->nik", adjoint_inverses, jacobian_base)
        velocities_base = jnp.einsum("nij,j->ni", adjoint_inverses, velocity_base)
        velocities_internal = jnp.einsum(
            "nij,j->ni", jacobians_internal_flat, qd_internal
        )
        jacobian_dot_velocity_base_samples = jnp.einsum(
            "nij,j->ni", adjoint_inverses, jacobian_dot_velocity_base
        ) - vmap(small_adjoint_action)(velocities_internal, velocities_base)
        jacobians = jnp.concatenate([jacobians_base, jacobians_internal_flat], axis=-1)
        jacobian_dot_velocity = (
            jacobian_dot_velocity_base_samples + jacobian_dot_velocity_internal_flat
        )
        velocities = velocities_base + velocities_internal
        gravity = jnp.einsum("nij,j->ni", adjoint_inverses, gravity_base)
        return (
            jacobians.reshape(*leading_shape, spatial_dimension, self.num_velocities),
            jacobian_dot_velocity.reshape(*leading_shape, spatial_dimension),
            velocities.reshape(*leading_shape, spatial_dimension),
            gravity.reshape(*leading_shape, spatial_dimension),
        )

    def _assemble_floating_dynamics_terms(
        self, q: Array, v: Array
    ) -> tuple[Array, Array, Array]:
        """
        Assemble statically specialized floating-base dynamics terms.

        Concrete systems must reuse their fused local recurrence and accumulate
        the augmented system without invoking a generic sample/JVP pipeline.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            v: Total generalized velocity with shape ``(num_velocities,)``.

        Returns:
            Tuple ``(M, Cv, G)`` in total generalized velocities.

        Raises:
            NotImplementedError: If floating-base dynamics have not been
                implemented by the concrete system.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fused floating-base dynamics."
        )

    def _floating_gravitational_energy(self, q: Array) -> Array:
        """
        Compute absolute floating-base gravitational potential energy.

        Concrete systems must implement a direct position/mass contraction and
        must not route this operation through augmented dynamics assembly.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.

        Returns:
            Absolute gravitational potential energy as a scalar.

        Raises:
            NotImplementedError: If the concrete system does not implement a
                direct floating-base energy path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement floating-base gravitational "
            "energy."
        )

    def _floating_internal_inertia_addition(self, q_internal: Array) -> Array:
        """
        Return non-body inertia attached directly to internal coordinates.

        Args:
            q_internal: Internal generalized coordinates with shape
                ``(num_internal_dofs,)``.

        Returns:
            Internal inertia addition with shape
            ``(num_internal_dofs, num_internal_dofs)``.
        """
        return jnp.zeros(
            (self.num_internal_dofs, self.num_internal_dofs), dtype=q_internal.dtype
        )

    def _floating_coriolis_matrix(self, q: Array, v: Array) -> Array:
        """
        Return a matrix realization satisfying ``C(q, v) @ v == Cv``.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.
            v: Total generalized velocity with shape ``(num_velocities,)``.

        Returns:
            Coriolis matrix with shape ``(num_velocities, num_velocities)``.
        """
        return 0.5 * jacfwd(
            lambda velocity: self._assemble_floating_dynamics_terms(q, velocity)[1]
        )(v)

    # -----------------------------------------
    # Kinematics implementation helpers
    # -----------------------------------------

    def _homogeneous_forward_kinematics(self, q: Array, s: Array) -> Array:
        """Return the forward-kinematics pose as an SE(2) or SE(3) matrix."""
        return self._homogeneous_pose(self.forward_kinematics(q, s))

    def _homogeneous_pose(self, pose: Array) -> Array:
        """Convert the public pose representation to a homogeneous matrix."""
        if self.is_planar:
            if pose.ndim == 1 and pose.shape[0] == 3:
                return poses.planar_pose_to_transform(pose)
            if pose.shape == (3, 3):
                return pose
            raise ValueError(
                "Planar forward_kinematics must return [theta, x, y] with shape (3,) "
                "or an SE(2) matrix with shape (3, 3)."
            )

        if pose.shape != (4, 4):
            raise ValueError(
                "Spatial forward_kinematics must return an SE(3) matrix with "
                "shape (4, 4)."
            )
        return pose

    def _inertial_jacobian_to_bodyframe(self, pose: Array, J: Array) -> Array:
        """
        Rotate an inertial-frame Jacobian into the local body frame.

        The convention matches the PCS/GVS dynamics code: angular and linear
        velocity components are rotated, with no translational adjoint coupling.
        """
        if self.is_planar:
            if pose.ndim == 1 and pose.shape[0] == 3:
                theta = pose[0]
                R = jnp.array(
                    [
                        [jnp.cos(theta), -jnp.sin(theta)],
                        [jnp.sin(theta), jnp.cos(theta)],
                    ],
                    dtype=pose.dtype,
                )
            elif pose.shape == (3, 3):
                R = pose[:2, :2]
            else:
                raise ValueError(
                    "Planar forward_kinematics must return [theta, x, y] with "
                    "shape (3,) or an SE(2) matrix with shape (3, 3)."
                )
            return jnp.concatenate([J[:1], R.T @ J[1:]], axis=0)

        if pose.shape != (4, 4):
            raise ValueError(
                "Spatial forward_kinematics must return an SE(3) matrix with "
                "shape (4, 4)."
            )
        R = pose[:3, :3]
        return jnp.concatenate([R.T @ J[:3], R.T @ J[3:]], axis=0)

    def _planar_pose_jacobian(self, pose: Array, dpose_dq: Array) -> Array:
        """
        Convert an autodiff pose derivative to a planar inertial-frame Jacobian.

        The standard planar pose representation in soromox is ``[theta, x, y]``.
        A homogeneous SE(2) matrix is also accepted for subclasses that expose one.
        """
        if pose.ndim == 1 and pose.shape[0] == 3:
            return dpose_dq

        if pose.shape == (3, 3):
            R = pose[:2, :2]
            dR_dq = dpose_dq[:2, :2, :]
            omega_hat = jnp.einsum("abn,cb->acn", dR_dq, R)
            omega_z = 0.5 * (omega_hat[1, 0, :] - omega_hat[0, 1, :])
            v = dpose_dq[:2, 2, :]
            return jnp.concatenate([omega_z[None, :], v], axis=0)

        raise ValueError(
            "Planar forward_kinematics must return [theta, x, y] with shape (3,) "
            "or an SE(2) matrix with shape (3, 3)."
        )

    def _spatial_pose_jacobian(self, pose: Array, dpose_dq: Array) -> Array:
        """
        Convert an autodiff SE(3) pose derivative to an inertial-frame Jacobian.

        For ``g = [R, p]``, each column is ``[omega, v]`` with
        ``omega = vee(dR_dq @ R.T)`` and ``v = dp_dq``.
        """
        if pose.shape != (4, 4):
            raise ValueError(
                "Spatial forward_kinematics must return an SE(3) matrix with "
                "shape (4, 4)."
            )

        R = pose[:3, :3]
        dR_dq = dpose_dq[:3, :3, :]
        omega_hat = jnp.einsum("abn,cb->acn", dR_dq, R)
        omega = 0.5 * jnp.stack(
            [
                omega_hat[2, 1, :] - omega_hat[1, 2, :],
                omega_hat[0, 2, :] - omega_hat[2, 0, :],
                omega_hat[1, 0, :] - omega_hat[0, 1, :],
            ],
            axis=0,
        )
        v = dpose_dq[:3, 3, :]
        return jnp.concatenate([omega, v], axis=0)

    def _pose_tangent_from_inertial_velocity(self, pose: Array, eta: Array) -> Array:
        """Convert an inertial velocity vector to a tangent of the pose representation."""
        if self.is_planar:
            return self._planar_pose_tangent_from_inertial_velocity(pose, eta)

        return self._spatial_pose_tangent_from_inertial_velocity(pose, eta)

    def _planar_pose_tangent_from_inertial_velocity(
        self, pose: Array, eta: Array
    ) -> Array:
        """Convert ``[omega_z, v_x, v_y]`` to a tangent for planar pose output."""
        if pose.ndim == 1 and pose.shape[0] == 3:
            return eta

        if pose.shape == (3, 3):
            omega = eta[0]
            v = eta[1:]
            R = pose[:2, :2]
            omega_hat = jnp.array(
                [[0.0, -omega], [omega, 0.0]],
                dtype=pose.dtype,
            )
            tangent = jnp.zeros_like(pose)
            tangent = tangent.at[:2, :2].set(omega_hat @ R)
            tangent = tangent.at[:2, 2].set(v)
            return tangent

        raise ValueError(
            "Planar forward_kinematics must return [theta, x, y] with shape (3,) "
            "or an SE(2) matrix with shape (3, 3)."
        )

    def _spatial_pose_tangent_from_inertial_velocity(
        self, pose: Array, eta: Array
    ) -> Array:
        """Convert ``[omega, v]`` to a tangent for SE(3) pose output."""
        if pose.shape != (4, 4):
            raise ValueError(
                "Spatial forward_kinematics must return an SE(3) matrix with "
                "shape (4, 4)."
            )

        omega = eta[:3]
        v = eta[3:]
        omega_hat = jnp.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ],
            dtype=pose.dtype,
        )
        tangent = jnp.zeros_like(pose)
        tangent = tangent.at[:3, :3].set(omega_hat @ pose[:3, :3])
        tangent = tangent.at[:3, 3].set(v)
        return tangent

    # -----------------------------------------
    # Dynamical matrices
    # -----------------------------------------

    @abstractmethod
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the generalized inertia (mass) matrix.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            M: Inertia matrix of shape (num_velocities, num_velocities).
        """
        ...

    @abstractmethod
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the Coriolis/centrifugal matrix.

        Implementations are expected to use the Christoffel/energy-consistent
        convention associated with ``inertia_matrix(q)``: ``C(q, qd) @ qd`` is
        the convective force and ``M_dot(q, qd) - 2 * C(q, qd)`` is
        skew-symmetric. This structure is important for energy balance,
        passivity, model-based control, and derivative-based APIs. Optimized
        ``dynamics_terms`` overrides should return a ``Cqd`` vector consistent
        with the same inertia matrix.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            qd: Generalized velocities of shape (num_velocities,).

        Returns:
            C: Coriolis matrix of shape (num_velocities, num_velocities).
        """
        ...

    @abstractmethod
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            D: Damping matrix of shape (num_velocities, num_velocities).
        """
        ...

    def stiffness_matrix(self) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Returns:
            K: Stiffness matrix of shape (num_velocities, num_velocities).
        """
        ...

    @abstractmethod
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic (stiffness) force.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            tau_el: Elastic force of shape (num_velocities,).
        """
        ...

    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            G: Gravitational force of shape (num_velocities,).
        """
        if self.floating_base:
            zeros = jnp.zeros((self.num_velocities,), dtype=q.dtype)
            return self._assemble_floating_dynamics_terms(q, zeros)[2]
        return self._gravitational_force(q)

    def potential_force(self, q: Array) -> Array:
        """
        Compute the total conservative generalized force.

        This is the sum of gravitational and elastic forces.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            tau_pot: Potential force of shape (num_velocities,).
        """
        return self.gravitational_force(q) + self.elastic_force(q)

    def _gravitational_force(self, q: Array) -> Array:
        """
        Protected gravitational-force hook.

        The default implementation differentiates the protected
        :meth:`_gravitational_energy` hook. Subclasses may override this method
        with an analytical or fused implementation. Differentiating the
        protected energy hook avoids recursively invoking the public
        custom-JVP energy wrapper.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.

        Returns:
            G: Gravitational generalized force of shape ``(num_velocities,)``.
        """
        return grad(lambda q_: self._gravitational_energy(q_))(q)

    def actuator_coordinates(self, q: Array) -> Array:
        """
        Return the installed transmissions' work-conjugate coordinates.

        Coordinates are concatenated in actuator and channel order, matching
        the columns of :meth:`actuation_matrix` and the entries returned by
        :meth:`actuator_efforts`. Their physical units may differ between
        actuator families, such as length for tendons or volume for pressure
        actuation.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.

        Returns:
            y_a: Actuator coordinates of shape ``(num_actuators,)``. An
                unactuated robot returns an empty array.
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            return self._fixed_base_robot_at_pose().actuator_coordinates(q_internal)
        if not self.actuators:
            return jnp.zeros((0,), dtype=q.dtype)
        return jnp.concatenate(
            tuple(actuator.coordinates(self, q) for actuator in self.actuators)
        )

    def actuator_velocities(self, q: Array, qd: Array) -> Array:
        """
        Return the installed transmissions' coordinate velocities.

        Velocities are concatenated in actuator and channel order and satisfy
        ``yd_a = A(q).T @ qd``, where ``A(q)`` is the concatenated actuator
        transmission matrix.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.
            qd: Generalized velocities of shape ``(num_velocities,)``.

        Returns:
            yd_a: Actuator-coordinate velocities of shape
                ``(num_actuators,)``. An unactuated robot returns an empty
                array.
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            _, qd_internal = self.split_velocity(qd)
            return self._fixed_base_robot_at_pose().actuator_velocities(
                q_internal, qd_internal
            )
        if not self.actuators:
            return jnp.zeros((0,), dtype=q.dtype)
        return jnp.concatenate(
            tuple(actuator.velocities(self, q, qd) for actuator in self.actuators)
        )

    def _actuation_matrix(self, q: Array) -> Array:
        """Return the protected differentiable JAX transmission matrix."""

        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            internal = self._fixed_base_robot_at_pose()._actuation_matrix(q_internal)
            return jnp.pad(internal, ((self.num_base_velocities, 0), (0, 0)))

        if not self.actuators:
            return jnp.zeros((self.num_velocities, 0), dtype=q.dtype)
        return jnp.concatenate(
            tuple(
                actuator.transmission.moment_matrix(self, q)
                for actuator in self.actuators
            ),
            axis=1,
        )

    def actuation_matrix(self, q: Array, *, backend: str | None = None) -> Array:
        """
        Return the actuator transmission matrix ``A(q)``.

        ``A(q)`` is the concatenated moment-arm matrix of the installed
        transmissions. It maps work-conjugate actuator efforts ``e`` to
        generalized forces,

        ``tau_u = A(q) @ e``.

        Equivalently, ``A(q).T`` is the Jacobian of the actuator coordinates
        with respect to the generalized coordinates.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            backend: Optional execution-backend override. The base
                implementation supports JAX execution.

        Returns:
            A: Actuator transmission matrix of shape
                (num_velocities, num_actuators).
        """
        if backend not in (None, "auto", "jax"):
            raise NotImplementedError(
                "This system exposes only the JAX actuation_matrix implementation."
            )
        return self._actuation_matrix(q)

    def actuator_efforts(
        self,
        q: Array,
        u: Array,
        qd: Array | None = None,
        *,
        actuation_matrix: Array | None = None,
    ) -> Array:
        """
        Map user controls to work-conjugate actuator efforts.

        Controls and returned efforts follow the order of the installed
        actuators. Effort models evaluate the actuator coordinates or
        velocities only when those quantities are required. If an effort model
        requires velocity and ``qd`` is omitted, zero generalized velocity is
        used.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            u: Actuator controls of shape (num_actuators,).
            qd: Optional generalized velocities of shape (num_velocities,).
            actuation_matrix: Optional precomputed actuator transmission matrix
                with shape ``(num_velocities, num_actuators)``. It is reused for
                velocity-dependent effort laws.

        Returns:
            e: Work-conjugate actuator efforts of shape (num_actuators,).

        Raises:
            ValueError: If ``u`` or ``actuation_matrix`` has an incompatible
                shape.
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            qd_internal = None
            if qd is not None:
                _, qd_internal = self.split_velocity(qd)
            internal_matrix = (
                None
                if actuation_matrix is None
                else actuation_matrix[self.num_base_velocities :]
            )
            return self._fixed_base_robot_at_pose().actuator_efforts(
                q_internal,
                u,
                qd=qd_internal,
                actuation_matrix=internal_matrix,
            )
        u = jnp.asarray(u)
        if u.shape != (self.num_actuators,):
            raise ValueError(
                f"u must have shape ({self.num_actuators},), got {u.shape}."
            )
        if actuation_matrix is not None and actuation_matrix.shape != (
            self.num_velocities,
            self.num_actuators,
        ):
            raise ValueError(
                "actuation_matrix must have shape "
                f"({self.num_velocities}, {self.num_actuators}), got "
                f"{actuation_matrix.shape}."
            )
        if not self.actuators:
            return jnp.zeros((0,), dtype=u.dtype)
        efforts = []
        start = 0
        for actuator in self.actuators:
            stop = start + actuator.num_channels
            actuator_transmission_matrix = (
                None if actuation_matrix is None else actuation_matrix[:, start:stop]
            )
            efforts.append(
                actuator.efforts(
                    self,
                    q,
                    u[start:stop],
                    qd=qd,
                    transmission_matrix=actuator_transmission_matrix,
                )
            )
            start = stop
        return jnp.concatenate(tuple(efforts))

    def _actuation_force(self, q: Array, u: Array, qd: Array | None = None) -> Array:
        """Return the protected differentiable JAX generalized actuator force."""

        actuation_matrix = self._actuation_matrix(q)
        effort = self.actuator_efforts(
            q,
            u,
            qd=qd,
            actuation_matrix=actuation_matrix,
        )
        return actuation_matrix @ effort

    def actuation_force(
        self,
        q: Array,
        u: Array,
        qd: Array | None = None,
        *,
        backend: str | None = None,
    ) -> Array:
        """
        Compute the generalized actuation force.

        The installed effort models first map the ordered controls ``u`` to
        work-conjugate actuator efforts ``e``. The actuator transmission matrix
        then maps those efforts to generalized forces as

        ``tau_u = A(q) @ e``.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            u: Actuator controls of shape (num_actuators,).
            qd: Optional generalized velocities of shape (num_velocities,).
            backend: Optional execution-backend override. The base
                implementation supports JAX execution.

        Returns:
            tau_u: Generalized actuation force of shape (num_velocities,).

        Raises:
            ValueError: If ``u`` does not have shape (num_actuators,).
        """
        if backend not in (None, "auto", "jax"):
            raise NotImplementedError(
                "This system exposes only the JAX actuation_force implementation."
            )
        return self._actuation_force(q, u, qd)

    def passive_elastic_force(self, q: Array) -> Array:
        """
        Return the installed passive elements' elastic generalized force.

        This method sums only composable passive-element contributions. System
        implementations add the result to their intrinsic body elasticity when
        evaluating :meth:`elastic_force`.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.

        Returns:
            tau_passive: Passive elastic generalized force of shape
                ``(num_velocities,)``. A robot without passive elements returns
                zeros.
        """
        if self.floating_base:
            # Internal callers may already have extracted the material coordinates.
            total_input = q.shape[-1] == self.num_coordinates
            q_internal = (
                self.split_configuration(q)[1]
                if total_input
                else self._check_trailing_dimension(
                    "q_internal", q, self.num_internal_dofs
                )
            )
            internal = self._fixed_base_robot_at_pose().passive_elastic_force(
                q_internal
            )
            return (
                jnp.pad(internal, (self.num_base_velocities, 0))
                if total_input
                else internal
            )
        force = jnp.zeros((self.num_velocities,), dtype=q.dtype)
        for element in self.passive_elements:
            force = force + element.elastic_force(self, q)
        return force

    def passive_damping_matrix(self, q: Array) -> Array:
        """
        Return the installed passive elements' generalized damping matrix.

        This method sums only composable passive-element contributions. System
        implementations add the result to their intrinsic body damping when
        evaluating :meth:`damping_matrix`.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.

        Returns:
            D_passive: Passive damping matrix of shape
                ``(num_velocities, num_velocities)``. A robot without passive elements
                returns zeros.
        """
        if self.floating_base:
            total_input = q.shape[-1] == self.num_coordinates
            q_internal = (
                self.split_configuration(q)[1]
                if total_input
                else self._check_trailing_dimension(
                    "q_internal", q, self.num_internal_dofs
                )
            )
            internal = self._fixed_base_robot_at_pose().passive_damping_matrix(
                q_internal
            )
            return (
                jnp.pad(
                    internal,
                    ((self.num_base_velocities, 0), (self.num_base_velocities, 0)),
                )
                if total_input
                else internal
            )
        matrix = jnp.zeros((self.num_velocities, self.num_velocities), dtype=q.dtype)
        for element in self.passive_elements:
            matrix = matrix + element.damping_matrix(self, q)
        return matrix

    def passive_elastic_energy(self, q: Array) -> Array:
        """
        Return the installed passive elements' stored elastic energy.

        This method sums only composable passive-element contributions. System
        implementations add the result to their intrinsic body strain energy
        when evaluating elastic or potential energy.

        Args:
            q: Generalized coordinates of shape ``(num_coordinates,)``.

        Returns:
            U_passive: Scalar passive elastic energy. A robot without passive
                elements returns zero.
        """
        if self.floating_base:
            total_input = q.shape[-1] == self.num_coordinates
            q_internal = (
                self.split_configuration(q)[1]
                if total_input
                else self._check_trailing_dimension(
                    "q_internal", q, self.num_internal_dofs
                )
            )
            return self._fixed_base_robot_at_pose().passive_elastic_energy(q_internal)
        energy = jnp.zeros((), dtype=q.dtype)
        for element in self.passive_elements:
            energy = energy + element.elastic_energy(self, q)
        return energy

    # -----------------------------------------
    # Energy methods
    # -----------------------------------------

    def kinetic_energy(self, q: Array, qd: Array) -> Array:
        """
        Compute the kinetic energy of the system.

        Default implementation: T = 0.5 * qd^T * M(q) * qd

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            qd: Generalized velocities of shape (num_velocities,).

        Returns:
            T: Kinetic energy (scalar).
        """
        if custom_jvp_enabled() and not self.floating_base:
            return SoftRobot._kinetic_energy_custom_jvp(self, q, qd)
        return self._kinetic_energy(q, qd)

    def _kinetic_energy(self, q: Array, qd: Array) -> Array:
        """
        Protected primal kinetic-energy hook for custom autodiff wrappers.

        Default implementation: T = 0.5 * qd^T * M(q) * qd.
        """
        M = self.inertia_matrix(q)
        T = 0.5 * qd @ M @ qd
        return T

    @eqx.filter_custom_jvp
    @staticmethod
    def _kinetic_energy_custom_jvp(robot: "SoftRobot", q: Array, qd: Array) -> Array:
        """Custom-JVP entry point for the public kinetic_energy wrapper."""
        return robot._kinetic_energy(q, qd)

    @_kinetic_energy_custom_jvp.def_jvp
    def _kinetic_energy_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[Array, Array]:
        robot, q, qd = primals
        _, q_tangent, qdd = tangents

        T = robot._kinetic_energy(q, qd)
        Td = robot._kinetic_energy_jvp_tangent(q, qd, q_tangent, qdd)
        return T, Td

    def _kinetic_energy_jvp_tangent(
        self,
        q: Array,
        qd: Array,
        q_tangent: Array | None,
        qdd: Array | None,
    ) -> Array:
        """Compute the kinetic-energy tangent from analytical dynamics terms."""
        Td = jnp.zeros_like(self._kinetic_energy(q, qd))

        if q_tangent is not None:
            Td = Td + self._kinetic_energy_q_jvp_tangent(q, qd, q_tangent)

        if qdd is not None:
            Td = Td + self._kinetic_energy_qd_jvp_tangent(q, qd, qdd)

        return Td

    def _kinetic_energy_q_jvp_tangent(
        self, q: Array, qd: Array, q_tangent: Array
    ) -> Array:
        """
        Compute the kinetic-energy contribution from a configuration tangent.
        """
        coriolis_cross = self._kinetic_energy_coriolis_cross_force(q, qd, q_tangent)
        return qd @ coriolis_cross

    def _kinetic_energy_qd_jvp_tangent(self, q: Array, qd: Array, qdd: Array) -> Array:
        """
        Compute the kinetic-energy contribution from a velocity tangent.
        """
        M = self._kinetic_energy_inertia_matrix(q, qd)
        return qd @ M @ qdd

    def _kinetic_energy_inertia_matrix(self, q: Array, qd: Array) -> Array:
        """
        Return the inertia matrix used by the kinetic-energy custom JVP.

        Systems that override ``dynamics_terms`` can reuse that assembled
        inertia matrix instead of materializing it through the public matrix API.
        """
        M, _, _ = self.dynamics_terms(q, qd)
        return M

    def _kinetic_energy_coriolis_cross_force(
        self, q: Array, qd: Array, qd_tangent: Array
    ) -> Array:
        """
        Return ``C(q, qd_tangent) @ qd`` for the configuration tangent.

        The cross term is obtained by a velocity-direction JVP through
        ``dynamics_terms``:
        ``d(C(v) @ v)[qd, qd_tangent] / 2 = C(q, qd_tangent) @ qd``
        for the standard Christoffel/energy-consistent Coriolis convention.
        """
        _, coriolis_cross = jvp(
            lambda velocity: self.dynamics_terms(q, velocity)[1],
            (qd,),
            (qd_tangent,),
        )
        return 0.5 * coriolis_cross

    def gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy of the system.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            U_g: Gravitational potential energy (scalar).
        """
        if self.floating_base:
            return self._floating_gravitational_energy(q)
        if custom_jvp_enabled():
            return SoftRobot._gravitational_energy_custom_jvp(self, q)
        return self._gravitational_energy(q)

    def _gravitational_energy(self, q: Array) -> Array:
        """
        Protected primal gravitational-energy hook for custom autodiff wrappers.

        Subclasses that inherit the public SoftRobot gravitational_energy method
        should override this hook.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _gravitational_energy."
        )

    @eqx.filter_custom_jvp
    @staticmethod
    def _gravitational_energy_custom_jvp(robot: "SoftRobot", q: Array) -> Array:
        """Custom-JVP entry point for the public gravitational_energy wrapper."""
        return robot._gravitational_energy(q)

    @_gravitational_energy_custom_jvp.def_jvp
    def _gravitational_energy_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array],
        tangents: tuple[Any, Array | None],
    ) -> tuple[Array, Array]:
        robot, q = primals
        _, qd = tangents

        U = robot._gravitational_energy(q)
        if qd is None:
            Ud = jnp.zeros_like(U)
        else:
            Ud = robot.gravitational_force(q) @ qd
        return U, Ud

    def elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in the system.

        Default implementation: U_el = 0.5 * q^T * K * q

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            U_el: Elastic potential energy (scalar).
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            return self._elastic_energy(q_internal)
        if custom_jvp_enabled():
            return SoftRobot._elastic_energy_custom_jvp(self, q)
        return self._elastic_energy(q)

    def _elastic_energy(self, q: Array) -> Array:
        """
        Protected primal elastic-energy hook for custom autodiff wrappers.

        Default implementation: U_el = 0.5 * q^T * K * q.
        """
        S = self.stiffness_matrix()
        U_el = 0.5 * q @ S @ q
        return U_el

    @eqx.filter_custom_jvp
    @staticmethod
    def _elastic_energy_custom_jvp(robot: "SoftRobot", q: Array) -> Array:
        """Custom-JVP entry point for the public elastic_energy wrapper."""
        return robot._elastic_energy(q)

    @_elastic_energy_custom_jvp.def_jvp
    def _elastic_energy_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array],
        tangents: tuple[Any, Array | None],
    ) -> tuple[Array, Array]:
        robot, q = primals
        _, qd = tangents

        U_el = robot._elastic_energy(q)
        if qd is None:
            Ueld = jnp.zeros_like(U_el)
        else:
            Ueld = robot.elastic_force(q) @ qd
        return U_el, Ueld

    def potential_energy(self, q: Array) -> Array:
        """
        Compute the total potential energy of the system.

        This is the sum of gravitational and elastic energy.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).

        Returns:
            U: Total potential energy (scalar).
        """
        if self.floating_base:
            return self.gravitational_energy(q) + self.elastic_energy(q)
        if custom_jvp_enabled():
            return SoftRobot._potential_energy_custom_jvp(self, q)
        return self._potential_energy(q)

    def _potential_energy(self, q: Array) -> Array:
        """Protected primal potential-energy hook for custom autodiff wrappers."""
        U_g = self._gravitational_energy(q)
        U_el = self._elastic_energy(q)
        U = U_g + U_el
        return U

    def _potential_energy_gradient(self, q: Array) -> Array:
        """Gradient of total potential energy with respect to q."""
        return self.potential_force(q)

    @eqx.filter_custom_jvp
    @staticmethod
    def _potential_energy_custom_jvp(robot: "SoftRobot", q: Array) -> Array:
        """Custom-JVP entry point for the public potential_energy wrapper."""
        return robot._potential_energy(q)

    @_potential_energy_custom_jvp.def_jvp
    def _potential_energy_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array],
        tangents: tuple[Any, Array | None],
    ) -> tuple[Array, Array]:
        robot, q = primals
        _, qd = tangents

        U = robot._potential_energy(q)
        if qd is None:
            Ud = jnp.zeros_like(U)
        else:
            Ud = robot._potential_energy_gradient(q) @ qd
        return U, Ud

    def total_energy(self, q: Array, qd: Array) -> Array:
        """
        Compute the total energy of the system.

        This is the sum of kinetic and potential energy.

        Args:
            q: Generalized coordinates of shape (num_coordinates,).
            qd: Generalized velocities of shape (num_velocities,).

        Returns:
            E: Total energy (scalar).
        """
        if self.floating_base:
            return self.kinetic_energy(q, qd) + self.potential_energy(q)
        if custom_jvp_enabled():
            return SoftRobot._total_energy_custom_jvp(self, q, qd)
        return self._total_energy(q, qd)

    def _total_energy(self, q: Array, qd: Array) -> Array:
        """Protected primal total-energy hook for custom autodiff wrappers."""
        T = self._kinetic_energy(q, qd)
        U = self._potential_energy(q)
        E = T + U
        return E

    @eqx.filter_custom_jvp
    @staticmethod
    def _total_energy_custom_jvp(robot: "SoftRobot", q: Array, qd: Array) -> Array:
        """Custom-JVP entry point for the public total_energy wrapper."""
        return robot._total_energy(q, qd)

    @_total_energy_custom_jvp.def_jvp
    def _total_energy_custom_jvp_jvp(
        primals: tuple["SoftRobot", Array, Array],
        tangents: tuple[Any, Array | None, Array | None],
    ) -> tuple[Array, Array]:
        robot, q, qd = primals
        _, q_tangent, qdd = tangents

        E = robot._total_energy(q, qd)
        Ed = robot._total_energy_jvp_tangent(q, qd, q_tangent, qdd)
        return E, Ed

    def _total_energy_jvp_tangent(
        self,
        q: Array,
        qd: Array,
        q_tangent: Array | None,
        qdd: Array | None,
    ) -> Array:
        """Compute the total-energy tangent from analytical energy gradients."""
        Ed = self._kinetic_energy_jvp_tangent(q, qd, q_tangent, qdd)

        if q_tangent is not None:
            Ed = Ed + self._potential_energy_jvp_tangent(q, q_tangent)

        return Ed

    def _potential_energy_jvp_tangent(self, q: Array, q_tangent: Array) -> Array:
        """Compute the potential-energy contribution from a configuration tangent."""
        return self._potential_energy_gradient(q) @ q_tangent

    # -----------------------------------------
    # Dynamics
    # -----------------------------------------

    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Return forward-dynamics terms ``(M, Cqd, G)``.

        ``Cqd`` is the convective force vector ``C(q, qd) @ qd``. The default
        implementation is intentionally unfused and calls the public matrix and
        force APIs separately. Overrides must keep ``M`` and ``Cqd``
        energy-consistent with ``inertia_matrix`` and ``coriolis_matrix``.
        """
        if self.floating_base:
            return self._assemble_floating_dynamics_terms(q, qd)
        M = self.inertia_matrix(q)
        Cqd = self.coriolis_matrix(q, qd) @ qd
        G = self.gravitational_force(q)
        return M, Cqd, G

    def _solve_inertia(self, inertia: Array, rhs: Array) -> Array:
        """
        Solve an active-coordinate inertia system for an acceleration vector.

        Soft-robot inertia matrices are symmetric positive definite. The CPU
        path uses a Cholesky factorization for lower latency on the small dense
        systems common in this package, while accelerator platforms retain the
        generic dense solve. This is one platform-level dispatch shared by all
        soft-robot implementations and does not use model-size heuristics.

        Args:
            inertia: Active-coordinate inertia matrix with shape
                ``(self.num_velocities, self.num_velocities)``.
            rhs: Generalized right-hand side with shape
                ``(self.num_velocities,)``.

        Returns:
            Acceleration vector satisfying ``inertia @ acceleration == rhs``,
            with shape ``(self.num_velocities,)``.
        """

        def generic(matrix: Array, vector: Array) -> Array:
            return jnp.linalg.solve(matrix, vector)

        def cholesky(matrix: Array, vector: Array) -> Array:
            symmetric = 0.5 * (matrix + matrix.T)
            factor = jnp.linalg.cholesky(symmetric)
            return cho_solve((factor, True), vector)

        return lax.platform_dependent(
            inertia,
            rhs,
            cpu=cholesky,
            cuda=generic,
            rocm=generic,
            default=generic,
        )

    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Compute the forward dynamics of the system.

        Given the current state and actuation inputs, compute the state derivative.

        Args:
            t: Current time (scalar).
            y: State ``[q, v, auxiliary]`` with shape ``(state_size,)``.
            actuation_args: Optional tuple containing actuation inputs:
                - (u,): Control input u
                - (u, tau_ext): Control input u and external forces tau_ext

        Returns:
            State derivative ``[qdot, vdot, auxiliary_dot]`` with shape
            ``(state_size,)``.
        """
        del t

        q, qd, auxiliary_state = self.split_state(y)

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
            u = jnp.zeros((self.num_actuators,), dtype=q.dtype)
        if tau_ext is None:
            tau_ext = jnp.zeros_like(qd)

        M, Cqd, G = self.dynamics_terms(q, qd)
        K = self.elastic_force(q)
        D = self.damping_matrix(q)
        tau_u = self.actuation_force(q, u, qd=qd)

        rhs = tau_u + tau_ext - Cqd - G - K - D @ qd
        qdd = self._solve_inertia(M, rhs)

        qdot = self.configuration_derivative(q, qd)
        if self.num_auxiliary_states == 0:
            return jnp.concatenate([qdot, qdd])
        auxiliary_dot = jnp.zeros_like(auxiliary_state)
        return jnp.concatenate([qdot, qdd, auxiliary_dot])
