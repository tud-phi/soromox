__all__ = ["SoftRobot"]

from abc import abstractmethod

from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.dynamical_system import DynamicalSystem


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

    Attributes:
        num_dofs (int): Number of degrees of freedom (configuration variables).
        num_actuators (int): Number of actuators.
        global_eps (float): Global epsilon for numerical computations.
    """

    # global epsilon for numerical computations
    global_eps: float  # Global epsilon for numerical computations

    @property
    def tangent_eps(self) -> Array:
        """Epsilon value for Lie algebra tangent computations.

        Returns:
            Array: Epsilon value for Lie algebra tangent computations.
        """
        return jnp.sqrt(self.global_eps)

    def __init__(self, eps: float | None = None, **kwargs):
        """Initialize the SoftRobot.

        Args:
            eps (float): Optional global epsilon value for numerical computations.
                If not provided, defaults to 10x machine epsilon for float64.
            **kwargs: Additional keyword arguments (unused, kept for API compatibility).
        """
        # Note: We don't call super().__init__() here because Equinox modules
        # work like dataclasses - fields are set directly rather than through
        # parent __init__ calls. Child classes must set num_dofs and num_actuators.
        if eps is not None:
            self.global_eps = eps
        else:
            self.global_eps = 1e1 * float(jnp.finfo(jnp.float64).eps)

    # -----------------------------------------
    # Arc-length parameterized kinematics
    # -----------------------------------------

    @abstractmethod
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
        ...

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

    @abstractmethod
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
        ...

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

    @abstractmethod
    def jacobian_and_derivative(
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
        ...

    def jacobian_and_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its derivative at multiple points along the robot.

        Default implementation uses vmap over jacobian_and_derivative.
        Subclasses may override this for more efficient batch computation.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).
            s_ps: Array of position parameters, shape (N,).

        Returns:
            J_ps: Jacobians at all points, shape (N, n_pose_dim, num_dofs).
            Jd_ps: Jacobian derivatives at all points, shape (N, n_pose_dim, num_dofs).
        """
        return vmap(lambda s: self.jacobian_and_derivative(q, qd, s))(s_ps)

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

    @abstractmethod
    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            G: Gravitational force of shape (num_dofs,).
        """
        ...

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
        M = self.inertia_matrix(q)
        T = 0.5 * qd @ M @ qd
        return T

    @abstractmethod
    def gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy of the system.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U_G: Gravitational potential energy (scalar).
        """
        ...

    def elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic potential energy stored in the system.

        Default implementation: U_k = 0.5 * q^T * K * q

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U_k: Elastic potential energy (scalar).
        """
        S = self.stiffness_matrix()
        U_k = 0.5 * q @ S @ q
        return U_k

    def potential_energy(self, q: Array) -> Array:
        """
        Compute the total potential energy of the system.

        This is the sum of gravitational and elastic energy.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            U: Total potential energy (scalar).
        """
        U_g = self.gravitational_energy(q)
        U_k = self.elastic_energy(q)
        U = U_g + U_k
        return U

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
        T = self.kinetic_energy(q, qd)
        U = self.potential_energy(q)
        E = T + U
        return E

    # -----------------------------------------
    # Dynamics
    # -----------------------------------------

    @abstractmethod
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
        ...
