__all__ = ["ActuationSpaceDynamics"]

from typing import TYPE_CHECKING

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp

from soromox.systems.soft_robot import SoftRobot

if TYPE_CHECKING:
    from soromox.control.reference_trajectory import ReferenceTrajectory


class ActuationSpaceDynamics(eqx.Module):
    """
    Actuation space dynamics for soft robots.

    This class transforms the configuration-space dynamics of a soft robot to actuation
    space using the Jacobian of the coordinate transformation. The actuation space is
    defined by work-conjugate actuator coordinates and unactuated coordinates
    that span the null space of the actuation matrix.

    The actuation space dynamics follow the formulation:
        M_y @ ydd + eta_y @ yd + h_y = [tau, 0_{n-m}]

    where:
        - M_y: Actuation space inertia matrix
        - eta_y: Actuation space Coriolis/centrifugal matrix
        - h_y: Sum of actuation space forces (gravitational, elastic, damping)
        - tau: Applied actuator forces (only on the first num_actuators coordinates)
        - y: Actuation space coordinates = [actuated_coords, unactuated_coords]

    The actuated coordinates are obtained from a (generally nonlinear) map from
    configuration space. They are accessed through the robot's
    `actuator_coordinates` method. The Jacobian of this
    map is the transpose of the actuation matrix:
        J_actuated = dy_a/dq = A_at^T

    The unactuated coordinates can be specified by the user or computed via basis
    expansion using QR decomposition to find a set of linearly independent coordinates.

    Attributes:
        robot: The complete soft-robot system supplied at construction. For a
            floating system, this retains the runtime base coordinates.
        fixed_base_robot: Fixed-base copy used for internal-coordinate dynamics.
            For floating systems, controllers remount this copy at the current
            runtime base pose. It deliberately excludes base motion and
            floating/internal dynamic coupling.
        H_unactuated: Matrix of shape (num_dofs - num_actuators, num_dofs) that maps
            configurations to unactuated coordinates. If None, computed via basis expansion.
        n_actuated: Number of actuated coordinates (equals num_actuators).
        n_unactuated: Number of unactuated coordinates (equals num_dofs - num_actuators).

    Notes:
        The new actuation matrix in actuation space is:
            A_y = [[I_{num_actuators}], [0_{num_dofs - num_actuators, num_actuators}]]
        where only the first num_actuators coordinates are directly actuated.

        The robot must implement `actuator_coordinates(q)` and return one
        independent work coordinate per actuator channel.

    References:
        Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024).
        Input decoupling of lagrangian systems via coordinate transformation:
        General characterization and its application to soft robotics.
        IEEE Transactions on Robotics, 40, 2098-2110.

        Pustina, P. (2025). Analysis and control of the underactuation in continuum
        soft robots: a kinematic independent approach.
        PhD Thesis, Sapienza University of Rome.

        Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
        into Learned Models for Control. Dissertation, Delft University of Technology.
        https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da
    """

    robot: SoftRobot
    fixed_base_robot: SoftRobot
    H_unactuated: Array  # (n_unactuated, num_dofs), maps q to unactuated coords
    n_actuated: int = eqx.field(static=True)
    n_unactuated: int = eqx.field(static=True)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        robot: SoftRobot,
        H_unactuated: Array | None = None,
    ):
        """
        Initialize the actuation space dynamics transformer.

        Args:
            robot: A soft robot implementing the SoftRobot interface.
                Must have `num_actuators`, `num_internal_dofs`,
                `actuation_matrix`, and `actuator_coordinates`
                attributes/methods.
            H_unactuated: Optional matrix of shape (num_dofs - num_actuators, num_dofs) that
                maps configurations to unactuated coordinates. If None, the unactuated
                coordinate mapping is computed via basis expansion using QR decomposition.

        Raises:
            ValueError: If H_unactuated has incorrect shape.
            ValueError: If the robot does not have the required attributes.
        """
        self.robot = robot
        self.fixed_base_robot = robot._fixed_base_robot_at_pose()

        # Get dimensions from robot
        num_dofs = self.fixed_base_robot.num_internal_dofs
        num_actuators = robot.num_actuators
        n_unactuated = num_dofs - num_actuators

        self.n_actuated = num_actuators
        self.n_unactuated = n_unactuated

        if n_unactuated < 0:
            raise ValueError(
                f"num_actuators ({num_actuators}) cannot exceed num_dofs ({num_dofs})"
            )

        # Get the actuation matrix at a reference configuration for basis expansion
        # A_at has shape (num_dofs, num_actuators)
        q_ref = jnp.zeros(num_dofs)
        A_at = self.fixed_base_robot.actuation_matrix(q_ref)
        rank = int(jnp.linalg.matrix_rank(A_at))
        if rank != num_actuators:
            raise ValueError(
                "ActuationSpaceDynamics requires independent actuator coordinates "
                f"at the reference configuration; expected rank {num_actuators}, "
                f"got {rank}."
            )

        # Compute or validate unactuated coordinate mapping
        if H_unactuated is None:
            # Compute via basis expansion using QR decomposition
            H_unactuated = self._compute_unactuated_coordinates_mapping(A_at)
        else:
            H_unactuated = jnp.asarray(H_unactuated)
            expected_shape = (n_unactuated, num_dofs)
            if H_unactuated.shape != expected_shape:
                raise ValueError(
                    f"H_unactuated must have shape {expected_shape}, "
                    f"got {H_unactuated.shape}"
                )

        self.H_unactuated = H_unactuated

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _compute_unactuated_coordinates_mapping(self, A_at: Array) -> Array:
        """
        Compute the unactuated coordinate mapping via basis expansion using QR decomposition.

        This method finds a set of coordinates that are linearly independent from the
        actuated coordinates. The approach uses QR decomposition to
        find the orthogonal complement of the column space of the actuation matrix.

        Args:
            A_at: Actuation matrix of shape (num_dofs, num_actuators).

        Returns:
            H_unactuated: Matrix of shape (n_unactuated, num_dofs) mapping configurations
                to unactuated coordinates.
        """
        num_dofs = A_at.shape[0]
        num_actuators = A_at.shape[1]
        n_unactuated = num_dofs - num_actuators

        if n_unactuated == 0:
            # Fully actuated system, no unactuated coordinates
            return jnp.zeros((0, num_dofs))

        # Perform basis expansion via QR decomposition
        # QR decomposition of A_at gives Q @ R = A_at
        # The last (n_unactuated) columns of Q span the orthogonal complement
        Q_complete, _ = jnp.linalg.qr(A_at, mode="complete")

        # The orthogonal complement spans the unactuated directions
        # These are the last n_unactuated columns of Q_complete
        extra = Q_complete[:, num_actuators:]  # (num_dofs, n_unactuated)

        # The mapping from q to unactuated coordinates is extra.T
        H_unactuated = extra.T  # (n_unactuated, num_dofs)

        return H_unactuated

    # =========================================================================
    # Coordinate transformations
    # =========================================================================

    @eqx.filter_jit
    def actuated_coordinates(self, q: Array) -> Array:
        """
        Compute the actuated coordinates from configuration coordinates.

        For tendon-actuated robots, this returns the tendon lengths, which are a
        nonlinear function of the configuration q. The robot must implement
        an `actuator_coordinates(q)` method.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            y_a: Actuated coordinates of shape (n_actuated,).
        """
        return self.fixed_base_robot.actuator_coordinates(q)

    @eqx.filter_jit
    def unactuated_coordinates(self, q: Array) -> Array:
        """
        Compute the unactuated coordinates from configuration coordinates.

        The unactuated coordinates span the null space of the actuation, i.e.,
        directions that cannot be directly controlled by the actuators.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            y_u: Unactuated coordinates of shape (n_unactuated,).
        """
        return self.H_unactuated @ q

    @eqx.filter_jit
    def actuated_unactuated_coordinates(self, q: Array) -> Array:
        """
        Compute the concatenated actuated and unactuated coordinates.

        This transforms the full configuration space coordinates to actuation space
        coordinates: y = [y_actuated, y_unactuated].

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            y: Actuation space coordinates of shape (num_dofs,).
                The first n_actuated entries are actuated (e.g., tendon lengths),
                the remaining n_unactuated entries are unactuated.
        """
        y_actuated = self.actuated_coordinates(q)
        y_unactuated = self.unactuated_coordinates(q)
        return jnp.concatenate([y_actuated, y_unactuated])

    # =========================================================================
    # Jacobians
    # =========================================================================

    @eqx.filter_jit
    def jacobian_actuated(self, q: Array) -> Array:
        """
        Compute the Jacobian of the map from configuration space to actuated coordinates.

        For tendon-actuated robots, this is the transpose of the actuation matrix:
            J_actuated = A_at^T = d(actuated_coordinates)/dq

        where the actuation matrix A_at maps tendon forces to generalized forces.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_a: Jacobian of shape (n_actuated, num_dofs).
        """
        A_at = self.fixed_base_robot.actuation_matrix(q)
        return A_at.T

    @eqx.filter_jit
    def jacobian_unactuated(self, q: Array) -> Array:
        """
        Compute the Jacobian of the map from configuration space to unactuated coordinates.

        For linear unactuated coordinate mappings (as computed via basis expansion),
        this is simply H_unactuated.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_u: Jacobian of shape (n_unactuated, num_dofs).
        """
        return self.H_unactuated

    @eqx.filter_jit
    def jacobian(self, q: Array) -> Array:
        """
        Compute the full Jacobian from configuration space to actuation space.

        The Jacobian is constructed by stacking the actuated and unactuated Jacobians:
            J = [[J_actuated], [J_unactuated]]

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J: Full Jacobian of shape (num_dofs, num_dofs).
        """
        J_actuated = self.jacobian_actuated(q)
        J_unactuated = self.jacobian_unactuated(q)
        return jnp.vstack([J_actuated, J_unactuated])

    @eqx.filter_jit
    def jacobian_inverse(self, q: Array) -> Array:
        """
        Compute the inverse Jacobian from actuation space to configuration space.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_inv: Inverse Jacobian of shape (num_dofs, num_dofs).
        """
        J = self.jacobian(q)
        return jnp.linalg.inv(J)

    @eqx.filter_jit
    def dynamically_consistent_pseudoinverse(self, q: Array) -> Array:
        """
        Compute the dynamically-consistent pseudo-inverse of the Jacobian.

        The dynamically-consistent pseudo-inverse is defined as:
            J_bar = M^{-1} @ J^T @ Lambda

        where Lambda = (J @ M^{-1} @ J^T)^{-1} is the actuation space inertia.

        For square Jacobians (our case), this simplifies but we compute it
        in the general form for consistency.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_bar: Dynamically-consistent pseudo-inverse of shape (num_dofs, num_dofs).
        """
        J = self.jacobian(q)
        M = self.fixed_base_robot.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)

        # For square J: Lambda = (J @ M^{-1} @ J^T)^{-1}
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # J_bar = M^{-1} @ J^T @ Lambda
        J_bar = M_inv @ J.T @ Lambda

        return J_bar

    # =========================================================================
    # Dynamical matrices in actuation space
    # =========================================================================

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the actuation space inertia matrix.

        The actuation space inertia matrix is defined as:
            M_y = J^{-T} @ M @ J^{-1}

        where M is the configuration space inertia matrix and J is the Jacobian
        from configuration to actuation space.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            M_y: Actuation space inertia matrix of shape (num_dofs, num_dofs).
        """
        J_inv = self.jacobian_inverse(q)
        M = self.fixed_base_robot.inertia_matrix(q)

        # M_y = J^{-T} @ M @ J^{-1}
        M_y = J_inv.T @ M @ J_inv

        return M_y

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the actuation space Coriolis/centrifugal matrix.

        The actuation space Coriolis matrix is defined as:
            eta_y = M_y @ (J @ M^{-1} @ C) @ J^{-1}

        where:
            - M_y is the actuation space inertia matrix
            - J is the Jacobian from configuration to actuation space
            - M is the configuration space inertia matrix
            - C is the configuration space Coriolis matrix

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            eta_y: Actuation space Coriolis matrix of shape (num_dofs, num_dofs).
        """
        J = self.jacobian(q)
        J_inv = jnp.linalg.inv(J)
        M = self.fixed_base_robot.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)
        C = self.fixed_base_robot.coriolis_matrix(q, qd)

        # Actuation space inertia
        M_y = J_inv.T @ M @ J_inv

        # Coriolis matrix: eta_y = M_y @ (J @ M^{-1} @ C) @ J^{-1}
        eta_y = M_y @ (J @ M_inv @ C) @ J_inv

        return eta_y

    @eqx.filter_jit
    def coriolis_force(self, q: Array, qd: Array) -> Array:
        """
        Compute the actuation space Coriolis/centrifugal force.

        The actuation space Coriolis force is defined as:
            tau_c_y = eta_y @ yd

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            tau_c_y: Actuation space Coriolis force of shape (num_dofs,).
        """
        eta_y = self.coriolis_matrix(q, qd)
        J = self.jacobian(q)
        yd = J @ qd

        return eta_y @ yd

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the actuation space damping matrix.

        The actuation space damping matrix is defined as:
            D_y = J^{-T} @ D @ J^{-1}

        where D is the configuration space damping matrix.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            D_y: Actuation space damping matrix of shape (num_dofs, num_dofs).
        """
        J_inv = self.jacobian_inverse(q)
        D = self.fixed_base_robot.damping_matrix(q)

        # D_y = J^{-T} @ D @ J^{-1}
        D_y = J_inv.T @ D @ J_inv

        return D_y

    # =========================================================================
    # Forces in actuation space
    # =========================================================================

    @eqx.filter_jit
    def damping_force(self, q: Array, qd: Array) -> Array:
        """
        Compute the actuation space damping force.

        The actuation space damping force is defined as:
            tau_d_y = D_y @ yd = J^{-T} @ D @ qd

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            tau_d_y: Actuation space damping force of shape (num_dofs,).
        """
        J_inv = self.jacobian_inverse(q)
        D = self.fixed_base_robot.damping_matrix(q)

        return J_inv.T @ D @ qd

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the actuation space elastic force.

        The actuation space elastic force is defined as:
            tau_el_y = J^{-T} @ tau_el

        where tau_el is the configuration space elastic force.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            tau_el_y: Actuation space elastic force of shape (num_dofs,).
        """
        J_inv = self.jacobian_inverse(q)
        tau_el = self.fixed_base_robot.elastic_force(q)

        return J_inv.T @ tau_el

    @eqx.filter_jit
    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the actuation space gravitational force.

        The actuation space gravitational force is defined as:
            G_y = J^{-T} @ G

        where G is the configuration space gravitational force.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            G_y: Actuation space gravitational force of shape (num_dofs,).
        """
        J_inv = self.jacobian_inverse(q)
        G = self.fixed_base_robot.gravitational_force(q)

        return J_inv.T @ G

    @eqx.filter_jit
    def potential_force(self, q: Array) -> Array:
        """
        Compute the total conservative actuation space force.

        This is the sum of gravitational and elastic forces.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            tau_pot_y: Conservative actuation space force of shape (num_dofs,).
        """
        return self.gravitational_force(q) + self.elastic_force(q)

    @eqx.filter_jit
    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Return actuation-space dynamics terms ``(M_y, Cyd, G_y)``.

        ``Cyd`` is the actuation-space Coriolis/centrifugal force vector.
        ``G_y`` is the actuation-space gravitational force, matching the
        ``SoftRobot.dynamics_terms`` convention of returning gravity separately
        from elastic forces.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            M_y: Actuation space inertia matrix of shape (num_dofs, num_dofs).
            Cyd: Actuation space Coriolis/centrifugal force of shape (num_dofs,).
            G_y: Actuation space gravitational force of shape (num_dofs,).
        """
        J_inv = self.jacobian_inverse(q)
        J_inv_T = J_inv.T
        M, Cqd, G = self.fixed_base_robot.dynamics_terms(q, qd)

        # For yd = J @ qd, the existing actuation-space Coriolis definition
        # contracts to J^{-T} @ (C @ qd). Reusing the robot's contracted term
        # avoids materializing C and recomputing M, J, and their inverses.
        M_y = J_inv_T @ M @ J_inv
        Cyd = J_inv_T @ Cqd
        G_y = J_inv_T @ G
        return M_y, Cyd, G_y

    # =========================================================================
    # Actuation matrix in actuation space
    # =========================================================================

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix in actuation space.

        In actuation space, the actuation matrix has a special structure:
            A_y = [[I_{num_actuators}], [0_{n_unactuated, num_actuators}]]

        This means that actuator forces directly affect only the first num_actuators
        coordinates (the actuated ones), while the unactuated coordinates receive
        no direct actuation input.

        Args:
            q: Generalized coordinates of shape (num_dofs,) (unused, for API consistency).

        Returns:
            A_y: Actuation matrix in actuation space of shape (num_dofs, num_actuators).
        """
        num_dofs = self.fixed_base_robot.num_internal_dofs
        num_actuators = self.fixed_base_robot.num_actuators

        # Build the actuation matrix: [[I], [0]]
        A_y = jnp.vstack(
            [
                jnp.eye(num_actuators),
                jnp.zeros((num_dofs - num_actuators, num_actuators)),
            ]
        )

        return A_y

    # =========================================================================
    # Reference trajectory conversion
    # =========================================================================

    def convert_reference_trajectory(
        self, reference_trajectory: "ReferenceTrajectory"
    ) -> "ReferenceTrajectory":
        """
        Convert a configuration-space reference trajectory to actuation space.

        This method transforms the entries of the reference trajectory (positions,
        velocities, accelerations) from configuration space (q, qd, qdd) to
        actuation space (y, yd, ydd) using the coordinate transformations.

        Only entries that are present (not None) in the input reference trajectory
        are transformed and included in the output. The ReferenceTrajectory
        __post_init__ will auto-derive missing entries as needed.

        The transformation is:
            y = [actuated_coordinates(q), unactuated_coordinates(q)]
            yd = J(q) @ qd
            ydd = J(q) @ qdd + Jd(q, qd) @ qd

        where J is the Jacobian of the coordinate transformation and Jd is its
        time derivative.

        Args:
            reference_trajectory: Reference trajectory in configuration space,
                containing desired configurations (x_des_fn), velocities (xd_des_fn),
                and accelerations (xdd_des_fn) as functions of time.

        Returns:
            ReferenceTrajectory in actuation space with transformed entries.
            Only entries that were present in the input are explicitly set;
            missing entries will be auto-derived by ReferenceTrajectory.__post_init__.
        """
        # Import here to avoid circular dependency
        from soromox.control.reference_trajectory import ReferenceTrajectory

        # Helper to transform position: q -> y
        def transform_position(q: Array) -> Array:
            return self.actuated_unactuated_coordinates(q)

        # Helper to transform velocity: (q, qd) -> yd = J(q) @ qd
        def transform_velocity(q: Array, qd: Array) -> Array:
            J = self.jacobian(q)
            return J @ qd

        # Helper to transform acceleration: (q, qd, qdd) -> ydd = J @ qdd + Jd @ qd
        def transform_acceleration(q: Array, qd: Array, qdd: Array) -> Array:
            J = self.jacobian(q)

            # Compute Jd @ qd using automatic differentiation
            # d/dt[J(q)] @ qd can be computed via JVP
            def J_times_qd(q_inner: Array) -> Array:
                return self.jacobian(q_inner) @ qd

            Jd_times_qd = jax.jvp(J_times_qd, (q,), (qd,))[1]
            return J @ qdd + Jd_times_qd

        # Build output dictionary with only non-None entries
        output_kwargs: dict = {"ts": reference_trajectory.ts}

        # Transform position function if present
        if reference_trajectory.x_des_fn is not None:

            def y_des_fn(t: Array) -> Array:
                q = reference_trajectory.x_des_fn(t)  # type: ignore[misc]
                return transform_position(q)

            output_kwargs["x_des_fn"] = y_des_fn

        # Transform position time series if present
        if reference_trajectory.x_des_ts is not None:
            output_kwargs["x_des_ts"] = jax.vmap(transform_position)(
                reference_trajectory.x_des_ts
            )

        # Transform velocity function if present
        if (
            reference_trajectory.xd_des_fn is not None
            and reference_trajectory.x_des_fn is not None
        ):

            def yd_des_fn(t: Array) -> Array:
                q = reference_trajectory.x_des_fn(t)  # type: ignore[misc]
                qd = reference_trajectory.xd_des_fn(t)  # type: ignore[misc]
                return transform_velocity(q, qd)

            output_kwargs["xd_des_fn"] = yd_des_fn

        # Transform velocity time series if present
        if (
            reference_trajectory.xd_des_ts is not None
            and reference_trajectory.x_des_ts is not None
        ):
            output_kwargs["xd_des_ts"] = jax.vmap(transform_velocity)(
                reference_trajectory.x_des_ts, reference_trajectory.xd_des_ts
            )

        # Transform acceleration function if present
        if (
            reference_trajectory.xdd_des_fn is not None
            and reference_trajectory.xd_des_fn is not None
            and reference_trajectory.x_des_fn is not None
        ):

            def ydd_des_fn(t: Array) -> Array:
                q = reference_trajectory.x_des_fn(t)  # type: ignore[misc]
                qd = reference_trajectory.xd_des_fn(t)  # type: ignore[misc]
                qdd = reference_trajectory.xdd_des_fn(t)  # type: ignore[misc]
                return transform_acceleration(q, qd, qdd)

            output_kwargs["xdd_des_fn"] = ydd_des_fn

        # Transform acceleration time series if present
        if (
            reference_trajectory.xdd_des_ts is not None
            and reference_trajectory.xd_des_ts is not None
            and reference_trajectory.x_des_ts is not None
        ):
            output_kwargs["xdd_des_ts"] = jax.vmap(transform_acceleration)(
                reference_trajectory.x_des_ts,
                reference_trajectory.xd_des_ts,
                reference_trajectory.xdd_des_ts,
            )

        # Preserve metadata from original trajectory
        output_kwargs["rotation_representation"] = (
            reference_trajectory.rotation_representation
        )
        output_kwargs["n_points"] = reference_trajectory.n_points
        output_kwargs["is_planar"] = reference_trajectory.is_planar

        return ReferenceTrajectory(**output_kwargs)
