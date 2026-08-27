__all__ = [
    "SoftRobot",
]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
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
from soromox.utils.geometry import poses

if TYPE_CHECKING:
    from soromox.rendering.actuators import ActuatorVisualLayer


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
        num_dofs (int): Number of degrees of freedom (configuration variables).
        num_actuators (int): Number of actuators.
        global_eps (float): Global epsilon for numerical computations.
        base_pose (Array): Base frame pose coordinates for the robot. Planar
            robots use shape ``(3,)`` with coordinates ``[theta, x, y]``,
            where ``theta`` is a right-handed angle in radians about the
            out-of-plane z-axis. Spatial robots use shape ``(7,)`` with
            coordinates ``[qw, qx, qy, qz, x, y, z]``. Spatial quaternions are
            scalar-first Hamilton quaternions, normalized before use, and
            represent the base-frame orientation; translations are inserted
            directly. Configured spatial quaternions must have nonzero finite
            norm. When omitted, the base pose is upright: the backbone points
            along world +y for planar robots and world +z for spatial robots.
            In an explicit zero-rotation pose, the backbone is aligned with +x.
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
    base_pose: Array
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
            **kwargs: Additional keyword arguments (unused, kept for API compatibility).
        """
        # Note: We don't call super().__init__() here because Equinox modules
        # work like dataclasses - fields are set directly rather than through
        # parent __init__ calls. Child classes must set num_dofs and num_actuators.
        if base_pose is None:
            if self.is_planar:
                base_pose = jnp.array([jnp.pi / 2, 0.0, 0.0], dtype=jnp.float64)
            else:
                sqrt_half = jnp.sqrt(jnp.asarray(0.5, dtype=jnp.float64))
                base_pose = jnp.array(
                    [sqrt_half, 0.0, -sqrt_half, 0.0, 0.0, 0.0, 0.0],
                    dtype=jnp.float64,
                )
        if self.is_planar:
            validate_planar_base_pose("base_pose", base_pose)
        else:
            validate_quaternion_base_pose("base_pose", base_pose, (7,))
        self.base_pose = jnp.asarray(base_pose, dtype=jnp.float64)
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

    def _configure_actuation(
        self,
        actuators: Actuator | tuple[Actuator, ...] | None,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
    ) -> None:
        """Install immutable actuator/passive components after DOFs are known."""
        if actuators is None:
            normalized_actuators = (IdentityActuator(self.num_dofs),)
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
        """Return the homogeneous transform represented by ``base_pose``.

        Planar robots consume ``[theta, x, y]`` and return an SE(2) matrix with
        shape ``(3, 3)``. Spatial robots consume
        ``[qw, qx, qy, qz, x, y, z]`` and return an SE(3) matrix with shape
        ``(4, 4)``. Spatial quaternions are scalar-first Hamilton quaternions.
        """
        if self.is_planar:
            return poses.planar_pose_to_transform(jnp.asarray(self.base_pose))
        return poses.quaternion_pose_to_transform(jnp.asarray(self.base_pose))

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
            q: Generalized coordinates of shape (num_dofs,).
            s: Position parameter along the robot structure. The meaning depends
                on the specific robot type:
                - For continuum robots (PCS, PlanarPCS): arc-length in [0, L_total]
                - For articulated robots (Pendulum): can be link index or fraction

        Returns:
            chi: Pose at point s. The shape and meaning depend on the robot type:
                - For 3D robots (PCS): SE(3) transformation matrix, shape (4, 4)
                - For planar robots (PlanarPCS, Pendulum): [theta, x, y], shape (3,)
        """
        if custom_jvp_enabled():
            return SoftRobot._forward_kinematics_custom_jvp(self, q, s)
        return self._forward_kinematics(q, s)

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Protected primal forward-kinematics hook for custom autodiff wrappers.

        Subclasses that inherit the public SoftRobot forward_kinematics method
        should override this hook.
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
        if qd is not None:
            if sd is None:
                return jvp(lambda q_: self._forward_kinematics(q_, s), (q,), (qd,))
            return jvp(
                lambda q_, s_: self._forward_kinematics(q_, s_),
                (q, s),
                (qd, sd),
            )

        if sd is None:
            pose = self._forward_kinematics(q, s)
            return pose, jnp.zeros_like(pose)

        pose, poses = self.forward_kinematics_and_arc_length_derivative(q, s)
        return pose, poses * sd

    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics at all segment or link tips.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            chi_tips: Poses at the robot tips. The shape and meaning depend on
                the robot type:
                - For 3D robots (PCS, GVS): shape (num_segments, 4, 4)
                - For planar robots (PlanarPCS, Pendulum): shape (num_segments, 3)
        """
        s_tips = jnp.cumsum(jnp.atleast_1d(jnp.asarray(self.segment_length)))
        return vmap(lambda s: self.forward_kinematics(q, s))(s_tips)

    def forward_kinematics_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics at multiple points along the robot.

        Default implementation uses vmap over forward_kinematics.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
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
            q: Generalized coordinates of shape (num_dofs,).
            s: Position parameter along the robot structure.

        Returns:
            J: Jacobian matrix of shape (n_pose_dim, num_dofs), where n_pose_dim
                depends on the robot type:
                - For 3D robots (PCS): 6 (angular velocity + linear velocity)
                - For planar robots (PlanarPCS, Pendulum): 3 (omega_z, v_x, v_y)
        """
        if custom_jvp_enabled():
            return SoftRobot._jacobian_custom_jvp(self, q, s)
        return self._jacobian(q, s)

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
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_tips: Inertial-frame Jacobians at the robot tips, with shape
                (num_tips, n_pose_dim, num_dofs).
        """
        s_tips = jnp.cumsum(jnp.atleast_1d(jnp.asarray(self.segment_length)))
        return vmap(lambda s: self.jacobian(q, s))(s_tips)

    def jacobian_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian at multiple points along the robot.

        Default implementation uses vmap over jacobian.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            J_ps: Jacobians at all points, shape (N, n_pose_dim, num_dofs).
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
            J: Jacobian matrix of shape (n_pose_dim, num_dofs).
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
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).
            s: Position parameter along the robot structure.

        Returns:
            J: Jacobian matrix of shape (n_pose_dim, num_dofs).
            Jd: Time derivative of the Jacobian, shape (n_pose_dim, num_dofs).
        """
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

    def jacobian_and_time_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its derivative at multiple points along the robot.

        Default implementation uses vmap over jacobian_and_time_derivative.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            J_ps: Jacobians at all points, shape (N, n_pose_dim, num_dofs).
            Jd_ps: Jacobian time derivatives at all points, shape (N, n_pose_dim, num_dofs).
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

    def jacobian_bodyframe_batched(self, q: Array, s_ps: Array) -> Array:
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

    def jacobian_and_time_derivative_bodyframe_batched(
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
        J_flat, Jd_flat = self.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_flat
        )

        leading_shape = s_ps.shape
        g_ps = g_flat.reshape(*leading_shape, *g_flat.shape[1:])
        J_ps = J_flat.reshape(*leading_shape, *J_flat.shape[1:])
        Jd_ps = Jd_flat.reshape(*leading_shape, *Jd_flat.shape[1:])
        return g_ps, J_ps, Jd_ps

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
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            M: Inertia matrix of shape (num_dofs, num_dofs).
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
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            C: Coriolis matrix of shape (num_dofs, num_dofs).
        """
        ...

    @abstractmethod
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            D: Damping matrix of shape (num_dofs, num_dofs).
        """
        ...

    def stiffness_matrix(self) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Returns:
            K: Stiffness matrix of shape (num_dofs, num_dofs).
        """
        ...

    @abstractmethod
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic (stiffness) force.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            tau_el: Elastic force of shape (num_dofs,).
        """
        ...

    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            G: Gravitational force of shape (num_dofs,).
        """
        return self._gravitational_force(q)

    def potential_force(self, q: Array) -> Array:
        """
        Compute the total conservative generalized force.

        This is the sum of gravitational and elastic forces.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            tau_pot: Potential force of shape (num_dofs,).
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
            q: Generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            G: Gravitational generalized force of shape ``(num_dofs,)``.
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
            q: Generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            y_a: Actuator coordinates of shape ``(num_actuators,)``. An
                unactuated robot returns an empty array.
        """
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
            q: Generalized coordinates of shape ``(num_dofs,)``.
            qd: Generalized velocities of shape ``(num_dofs,)``.

        Returns:
            yd_a: Actuator-coordinate velocities of shape
                ``(num_actuators,)``. An unactuated robot returns an empty
                array.
        """
        if not self.actuators:
            return jnp.zeros((0,), dtype=q.dtype)
        return jnp.concatenate(
            tuple(actuator.velocities(self, q, qd) for actuator in self.actuators)
        )

    def actuation_matrix(self, q: Array) -> Array:
        """
        Return the actuator transmission matrix ``A(q)``.

        ``A(q)`` is the concatenated moment-arm matrix of the installed
        transmissions. It maps work-conjugate actuator efforts ``e`` to
        generalized forces,

        ``tau_u = A(q) @ e``.

        Equivalently, ``A(q).T`` is the Jacobian of the actuator coordinates
        with respect to the generalized coordinates.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            A: Actuator transmission matrix of shape
                (num_dofs, num_actuators).
        """
        if not self.actuators:
            return jnp.zeros((self.num_dofs, 0), dtype=q.dtype)
        return jnp.concatenate(
            tuple(
                actuator.transmission.moment_matrix(self, q)
                for actuator in self.actuators
            ),
            axis=1,
        )

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
            q: Generalized coordinates of shape (num_dofs,).
            u: Actuator controls of shape (num_actuators,).
            qd: Optional generalized velocities of shape (num_dofs,).
            actuation_matrix: Optional precomputed actuator transmission matrix
                with shape ``(num_dofs, num_actuators)``. It is reused for
                velocity-dependent effort laws.

        Returns:
            e: Work-conjugate actuator efforts of shape (num_actuators,).

        Raises:
            ValueError: If ``u`` or ``actuation_matrix`` has an incompatible
                shape.
        """
        u = jnp.asarray(u)
        if u.shape != (self.num_actuators,):
            raise ValueError(
                f"u must have shape ({self.num_actuators},), got {u.shape}."
            )
        if actuation_matrix is not None and actuation_matrix.shape != (
            self.num_dofs,
            self.num_actuators,
        ):
            raise ValueError(
                "actuation_matrix must have shape "
                f"({self.num_dofs}, {self.num_actuators}), got "
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

    def actuation_force(self, q: Array, u: Array, qd: Array | None = None) -> Array:
        """
        Compute the generalized actuation force.

        The installed effort models first map the ordered controls ``u`` to
        work-conjugate actuator efforts ``e``. The actuator transmission matrix
        then maps those efforts to generalized forces as

        ``tau_u = A(q) @ e``.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            u: Actuator controls of shape (num_actuators,).
            qd: Optional generalized velocities of shape (num_dofs,).

        Returns:
            tau_u: Generalized actuation force of shape (num_dofs,).

        Raises:
            ValueError: If ``u`` does not have shape (num_actuators,).
        """
        actuation_matrix = self.actuation_matrix(q)
        effort = self.actuator_efforts(
            q,
            u,
            qd=qd,
            actuation_matrix=actuation_matrix,
        )
        return actuation_matrix @ effort

    def passive_elastic_force(self, q: Array) -> Array:
        """
        Return the installed passive elements' elastic generalized force.

        This method sums only composable passive-element contributions. System
        implementations add the result to their intrinsic body elasticity when
        evaluating :meth:`elastic_force`.

        Args:
            q: Generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            tau_passive: Passive elastic generalized force of shape
                ``(num_dofs,)``. A robot without passive elements returns
                zeros.
        """
        force = jnp.zeros((self.num_dofs,), dtype=q.dtype)
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
            q: Generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            D_passive: Passive damping matrix of shape
                ``(num_dofs, num_dofs)``. A robot without passive elements
                returns zeros.
        """
        matrix = jnp.zeros((self.num_dofs, self.num_dofs), dtype=q.dtype)
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
            q: Generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            U_passive: Scalar passive elastic energy. A robot without passive
                elements returns zero.
        """
        energy = jnp.zeros((), dtype=q.dtype)
        for element in self.passive_elements:
            energy = energy + element.elastic_energy(self, q)
        return energy

    def actuator_visual_layers(
        self,
        q: Array,
        s_points: Array,
        *,
        actuator_inputs: Array | None = None,
    ) -> tuple["ActuatorVisualLayer", ...]:
        """
        Return renderer-facing geometry for installed actuation components.

        The returned layers describe semantic geometry and optional scalar
        fields for supported active actuators and passive elements. Visual
        styling such as colors, radii, and line widths remains the renderer's
        responsibility. Unsupported components do not produce a layer.

        Args:
            q: Generalized coordinates of shape ``(num_dofs,)``.
            s_points: Backbone sample coordinates with shape
                ``(num_points,)`` used for routed continuum geometry.
            actuator_inputs: Optional actuator inputs of shape
                ``(num_actuators,)``. When supplied, supported adapters attach
                input-dependent scalar fields such as pressure or axial force.

        Returns:
            layers: Tuple of semantic actuator visual layers. The tuple is
                empty when no installed component has a renderer adapter.
        """
        from soromox.rendering.actuation import actuator_visual_layers

        return actuator_visual_layers(
            self, q, s_points, actuator_inputs=actuator_inputs
        )

    # -----------------------------------------
    # Energy methods
    # -----------------------------------------

    def kinetic_energy(self, q: Array, qd: Array) -> Array:
        """
        Compute the kinetic energy of the system.

        Default implementation: T = 0.5 * qd^T * M(q) * qd

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            T: Kinetic energy (scalar).
        """
        if custom_jvp_enabled():
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
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U_g: Gravitational potential energy (scalar).
        """
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
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U_el: Elastic potential energy (scalar).
        """
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
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U: Total potential energy (scalar).
        """
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
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            E: Total energy (scalar).
        """
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
                ``(self.num_dofs, self.num_dofs)``.
            rhs: Generalized right-hand side with shape ``(self.num_dofs,)``.

        Returns:
            Acceleration vector satisfying ``inertia @ acceleration == rhs``,
            with shape ``(self.num_dofs,)``.
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
            y: State vector containing [q, qd], shape (2 * num_dofs,).
            actuation_args: Optional tuple containing actuation inputs:
                - (u,): Control input u
                - (u, tau_ext): Control input u and external forces tau_ext

        Returns:
            yd: State derivative [qd, qdd], shape (2 * num_dofs,).
        """
        del t

        q, qd = jnp.split(y, 2)

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
            tau_ext = jnp.zeros_like(q)

        M, Cqd, G = self.dynamics_terms(q, qd)
        K = self.elastic_force(q)
        D = self.damping_matrix(q)
        tau_u = self.actuation_force(q, u, qd=qd)

        rhs = tau_u + tau_ext - Cqd - G - K - D @ qd
        qdd = self._solve_inertia(M, rhs)

        return jnp.concatenate([qd, qdd])
