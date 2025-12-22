__all__ = ["OperationalSpaceDynamics"]

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp
from typing import Optional, Tuple

from soromox.systems.soft_robot import SoftRobot


class OperationalSpaceDynamics(eqx.Module):
    """
    Operational space dynamics for soft robots.

    This class transforms the configuration-space dynamics of a soft robot to operational
    (task/Cartesian) space using the dynamically-consistent pseudo-inverse of the Jacobian.
    The operational space is defined by specifying points along the robot backbone and
    selecting which axes of the Cartesian space are active.

    The operational space dynamics follow the formulation:
        Lambda @ xdd + mu @ xd + p = F

    where:
        - Lambda: Operational space inertia matrix (dynamically-consistent)
        - mu: Operational space Coriolis/centrifugal matrix
        - p: Sum of operational space forces (gravitational, elastic, damping)
        - F: Applied operational space force
        - x: Operational space coordinates (stacked poses at the selected points)

    The transformation uses the dynamically-consistent pseudo-inverse:
        J_bar = M^{-1} @ J^T @ Lambda

    where Lambda = (J @ M^{-1} @ J^T)^{-1} is the operational space inertia matrix.

    Attributes:
        robot: The soft robot (e.g., PCS, PlanarPCS, Pendulum) to transform.
        s_ps: Array of shape (N,) specifying points along the backbone where the
            operational space is defined. Each value should be in [0, L_total].
        task_selector: Boolean array specifying which pose axes are active. The shape
            depends on the robot type and number of points:
            - For 3D robots (PCS): shape (N, 6) or (6*N,) selecting from
              [omega_x, omega_y, omega_z, v_x, v_y, v_z] per point
            - For planar robots (PlanarPCS, Pendulum): shape (N, 3) or (3*N,) selecting from
              [omega_z, v_x, v_y] per point
        n_operational_space: Total dimension of the operational space after selection.
        n_pose_dim: Dimension of a single pose (6 for 3D, 3 for planar).
        n_points: Number of points along the backbone.

    Notes:
        The dynamically-consistent pseudo-inverse preserves the relationship between
        operational space forces and configuration space accelerations through the
        mass matrix, ensuring physical consistency of the transformed dynamics.

    References:
        Khatib, O. (1987). A unified approach for motion and force control of robot
        manipulators: The operational space formulation. IEEE Journal on Robotics
        and Automation, 3(1), 43-53.

        Natale, C. (2003). Interaction control of robot manipulators: Six-degrees-of-freedom
        tasks. Springer Science & Business Media.
    """

    robot: SoftRobot
    s_ps: Array  # Points along the backbone, shape (N,)
    task_selector: Array  # Boolean selector for which axes are active
    B_task: Array  # Task selection basis matrix, shape (n_pose_dim * N, n_operational_space)
    n_operational_space: int = eqx.field(static=True)
    n_pose_dim: int = eqx.field(static=True)
    n_points: int = eqx.field(static=True)

    def __init__(
        self,
        robot: SoftRobot,
        s_ps: Array,
        task_selector: Optional[Array] = None,
    ):
        """
        Initialize the operational space dynamics transformer.

        Args:
            robot: A soft robot implementing the SoftRobot interface.
                Examples include PCS, PlanarPCS, TendonActuatedPCS, Pendulum, etc.
            s_ps: Array of shape (N,) specifying the arc-length positions along the
                backbone where the operational space is defined. Values should be
                in the interval [0, L_total] where L_total is the total robot length.
            task_selector: Optional boolean array specifying which pose components
                are active in the operational space. Can be:
                - Shape (n_pose_dim,): Applied uniformly to all points
                - Shape (N, n_pose_dim): Different selection per point
                - Shape (N * n_pose_dim,): Flattened selection
                If None, all pose components at all points are active.

        Raises:
            ValueError: If s_ps is empty or has invalid shape.
            ValueError: If task_selector has incompatible shape.
        """
        self.robot = robot
        self.s_ps = jnp.atleast_1d(jnp.asarray(s_ps))

        if self.s_ps.ndim != 1 or self.s_ps.size == 0:
            raise ValueError(
                f"s_ps must be a 1D array with at least one element, got shape {self.s_ps.shape}"
            )

        n_points = self.s_ps.shape[0]
        self.n_points = n_points

        # Determine pose dimension from the robot by checking the Jacobian shape
        # For PCS: 6 (omega_x, omega_y, omega_z, v_x, v_y, v_z)
        # For PlanarPCS/Pendulum: 3 (omega_z, v_x, v_y)
        n_pose_dim = self._infer_pose_dimension()
        self.n_pose_dim = n_pose_dim

        total_pose_dim = n_points * n_pose_dim

        # Handle task_selector
        if task_selector is None:
            # All axes active
            task_selector = jnp.ones(total_pose_dim, dtype=bool)
        else:
            task_selector = jnp.asarray(task_selector)

            # Handle different input shapes
            if task_selector.shape == (n_pose_dim,):
                # Same selection for all points - broadcast
                task_selector = jnp.tile(task_selector, n_points)
            elif task_selector.shape == (n_points, n_pose_dim):
                # Per-point selection - flatten
                task_selector = task_selector.flatten()
            elif task_selector.shape == (total_pose_dim,):
                # Already flattened
                pass
            else:
                raise ValueError(
                    f"task_selector must have shape ({n_pose_dim},), ({n_points}, {n_pose_dim}), "
                    f"or ({total_pose_dim},), got {task_selector.shape}"
                )

            if not jnp.issubdtype(task_selector.dtype, jnp.bool_):
                raise TypeError(
                    f"task_selector must be a boolean array, got {task_selector.dtype}"
                )

        self.task_selector = task_selector
        self.n_operational_space = int(jnp.sum(task_selector))

        if self.n_operational_space == 0:
            raise ValueError(
                "task_selector must have at least one True value to define a non-empty operational space"
            )

        # Build the task selection basis matrix
        # B_task maps from full pose space to selected operational space
        # Shape: (total_pose_dim, n_operational_space)
        self.B_task = self._compute_task_basis(task_selector)

    def _infer_pose_dimension(self) -> int:
        """
        Infer the pose dimension from the robot's Jacobian.

        Returns:
            n_pose_dim: 6 for 3D robots (PCS), 3 for planar robots (PlanarPCS, Pendulum).
        """
        # Create a dummy configuration to evaluate the Jacobian
        n_dof = self.robot.num_dofs
        q_dummy = jnp.zeros(n_dof)
        s_dummy = self.s_ps[0]

        # Evaluate the Jacobian to get its shape
        J = self.robot.jacobian(q_dummy, s_dummy)
        return J.shape[0]

    def _compute_task_basis(self, task_selector: Array) -> Array:
        """
        Compute the task selection basis matrix.

        Args:
            task_selector: Boolean array of shape (total_pose_dim,).

        Returns:
            B_task: Selection matrix of shape (total_pose_dim, n_operational_space).
        """
        total_pose_dim = self.n_points * self.n_pose_dim
        n_selected = int(jnp.sum(task_selector))

        # Build selection matrix: columns correspond to selected dimensions
        selected_indices = jnp.where(task_selector, size=n_selected)[0]
        B_task = jnp.zeros((total_pose_dim, n_selected))
        B_task = B_task.at[selected_indices, jnp.arange(n_selected)].set(1.0)

        return B_task

    @eqx.filter_jit
    def _stacked_jacobian_full(self, q: Array) -> Array:
        """
        Compute the full stacked Jacobian for all points (before task selection).

        Uses the robot's jacobian_batched method if available for efficiency.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_full: Stacked Jacobian of shape (n_points * n_pose_dim, num_dofs).
        """
        # Use batched method from the robot
        J_ps = self.robot.jacobian_batched(q, self.s_ps)
        # J_ps has shape (n_points, n_pose_dim, num_dofs)

        # Stack to get (n_points * n_pose_dim, num_dofs)
        J_full = J_ps.reshape(-1, self.robot.num_dofs)

        return J_full

    @eqx.filter_jit
    def _stacked_jacobian_and_derivative_full(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the full stacked Jacobian and its time derivative for all points.

        Uses the robot's jacobian_and_derivative_batched method if available.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            J_full: Stacked Jacobian of shape (n_points * n_pose_dim, num_dofs).
            Jd_full: Stacked Jacobian derivative of shape (n_points * n_pose_dim, num_dofs).
        """
        # Use batched method from the robot
        J_ps, Jd_ps = self.robot.jacobian_and_derivative_batched(q, qd, self.s_ps)
        # J_ps, Jd_ps have shape (n_points, n_pose_dim, num_dofs)

        # Stack to get (n_points * n_pose_dim, num_dofs)
        J_full = J_ps.reshape(-1, self.robot.num_dofs)
        Jd_full = Jd_ps.reshape(-1, self.robot.num_dofs)

        return J_full, Jd_full

    @eqx.filter_jit
    def jacobian(self, q: Array) -> Array:
        """
        Compute the operational space Jacobian (with task selection applied).

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J: Operational space Jacobian of shape (n_operational_space, num_dofs).
        """
        J_full = self._stacked_jacobian_full(q)
        # Apply task selection: J = B_task^T @ J_full
        J = self.B_task.T @ J_full
        return J

    @eqx.filter_jit
    def jacobian_and_derivative(self, q: Array, qd: Array) -> Tuple[Array, Array]:
        """
        Compute the operational space Jacobian and its time derivative.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            J: Operational space Jacobian of shape (n_operational_space, num_dofs).
            Jd: Time derivative of J, shape (n_operational_space, num_dofs).
        """
        J_full, Jd_full = self._stacked_jacobian_and_derivative_full(q, qd)
        # Apply task selection
        J = self.B_task.T @ J_full
        Jd = self.B_task.T @ Jd_full
        return J, Jd

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the operational space inertia matrix.

        The operational space inertia matrix is defined as:
            Lambda = (J @ M^{-1} @ J^T)^{-1}

        where M is the configuration space inertia matrix and J is the Jacobian.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            Lambda: Operational space inertia matrix of shape
                (n_operational_space, n_operational_space).
        """
        J = self.jacobian(q)
        M = self.robot.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)

        # Lambda = (J @ M^{-1} @ J^T)^{-1}
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        return Lambda

    @eqx.filter_jit
    def dynamically_consistent_pseudoinverse(self, q: Array) -> Array:
        """
        Compute the dynamically-consistent pseudo-inverse of the Jacobian.

        The dynamically-consistent pseudo-inverse is defined as:
            J_bar = M^{-1} @ J^T @ Lambda

        where Lambda = (J @ M^{-1} @ J^T)^{-1} is the operational space inertia.

        This pseudo-inverse has the property that:
            J @ J_bar = I  (in operational space)
            J_bar @ J != I  (generally, in configuration space)

        and it minimizes the kinetic energy norm of the configuration space motion.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_bar: Dynamically-consistent pseudo-inverse of shape
                (num_dofs, n_operational_space).
        """
        J = self.jacobian(q)
        M = self.robot.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)

        # Lambda = (J @ M^{-1} @ J^T)^{-1}
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # J_bar = M^{-1} @ J^T @ Lambda
        J_bar = M_inv @ J.T @ Lambda

        return J_bar

    @eqx.filter_jit
    def jacobian_and_dynamically_consistent_pseudoinverse(
        self, q: Array
    ) -> Tuple[Array, Array]:
        """
        Compute both the Jacobian and its dynamically-consistent pseudo-inverse.

        This method is more efficient than calling jacobian() and
        dynamically_consistent_pseudoinverse() separately as it avoids
        redundant computation of the inertia matrix and its inverse.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J: Operational space Jacobian of shape (n_operational_space, num_dofs).
            J_bar: Dynamically-consistent pseudo-inverse of shape
                (num_dofs, n_operational_space).
        """
        J = self.jacobian(q)
        M = self.robot.inertia_matrix(q)
        M_inv = jnp.linalg.inv(M)

        # Lambda = (J @ M^{-1} @ J^T)^{-1}
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # J_bar = M^{-1} @ J^T @ Lambda
        J_bar = M_inv @ J.T @ Lambda

        return J, J_bar

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space Coriolis/centrifugal matrix.

        The operational space Coriolis matrix is defined as:
            mu = Lambda @ (J @ M^{-1} @ C - Jd) @ J_bar

        where:
            - Lambda is the operational space inertia matrix
            - J is the Jacobian
            - M is the configuration space inertia matrix
            - C is the configuration space Coriolis matrix
            - Jd is the time derivative of the Jacobian
            - J_bar is the dynamically-consistent pseudo-inverse

        This formulation ensures that (d/dt)(Lambda) - 2*mu is skew-symmetric,
        analogous to the skew-symmetry property of the configuration space dynamics.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            mu: Operational space Coriolis matrix of shape
                (n_operational_space, n_operational_space).
        """
        J, Jd = self.jacobian_and_derivative(q, qd)
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        M_inv = jnp.linalg.inv(M)

        # Compute Lambda
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # Compute J_bar
        J_bar = M_inv @ J.T @ Lambda

        # mu = Lambda @ (J @ M^{-1} @ C - Jd) @ J_bar
        mu = Lambda @ (J @ M_inv @ C - Jd) @ J_bar

        return mu

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the operational space damping matrix.

        The operational space damping matrix is defined as:
            D_x = J_bar^T @ D @ J_bar

        where D is the configuration space damping matrix and J_bar is the
        dynamically-consistent pseudo-inverse.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            D_x: Operational space damping matrix of shape
                (n_operational_space, n_operational_space).
        """
        J_bar = self.dynamically_consistent_pseudoinverse(q)
        D = self.robot.damping_matrix(q)

        # D_x = J_bar^T @ D @ J_bar
        D_x = J_bar.T @ D @ J_bar

        return D_x

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the operational space elastic force.

        The operational space elastic force is defined as:
            f_el_x = J_bar^T @ tau_el

        where tau_el is the configuration space elastic force and J_bar is the
        dynamically-consistent pseudo-inverse.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            f_el_x: Operational space elastic force of shape (n_operational_space,).
        """
        J_bar = self.dynamically_consistent_pseudoinverse(q)
        tau_el = self.robot.elastic_force(q)

        # f_el_x = J_bar^T @ tau_el
        f_el_x = J_bar.T @ tau_el

        return f_el_x

    @eqx.filter_jit
    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the operational space gravitational force.

        The operational space gravitational force is defined as:
            G_x = J_bar^T @ G

        where G is the configuration space gravitational force and J_bar is the
        dynamically-consistent pseudo-inverse.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            G_x: Operational space gravitational force of shape (n_operational_space,).
        """
        J_bar = self.dynamically_consistent_pseudoinverse(q)
        G = self.robot.gravitational_force(q)

        # G_x = J_bar^T @ G
        G_x = J_bar.T @ G

        return G_x

    @eqx.filter_jit
    def damping_force(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space damping force.

        The operational space damping force is defined as:
            f_d_x = D_x @ xd = J_bar^T @ D @ qd

        where D is the configuration space damping matrix.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            f_d_x: Operational space damping force of shape (n_operational_space,).
        """
        J_bar = self.dynamically_consistent_pseudoinverse(q)
        D = self.robot.damping_matrix(q)

        # f_d_x = J_bar^T @ D @ qd
        f_d_x = J_bar.T @ D @ qd

        return f_d_x

    @eqx.filter_jit
    def coriolis_force(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space Coriolis/centrifugal force.

        The operational space Coriolis force is defined as:
            f_c_x = mu @ xd

        where mu is the operational space Coriolis matrix and xd is the
        operational space velocity.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            f_c_x: Operational space Coriolis force of shape (n_operational_space,).
        """
        mu = self.coriolis_matrix(q, qd)
        J = self.jacobian(q)
        xd = J @ qd

        # f_c_x = mu @ xd
        f_c_x = mu @ xd

        return f_c_x

    @eqx.filter_jit
    def operational_space_velocity(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space velocity.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            xd: Operational space velocity of shape (n_operational_space,).
        """
        J = self.jacobian(q)
        return J @ qd

    @eqx.filter_jit
    def null_space_projector(self, q: Array) -> Array:
        """
        Compute the null-space projector matrix.

        The null-space projector projects configuration space motions into the
        null space of the operational space Jacobian:
            N = I - J_bar @ J

        Motions in the null space do not affect the operational space coordinates.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            N: Null-space projector matrix of shape (num_dofs, num_dofs).
        """
        J, J_bar = self.jacobian_and_dynamically_consistent_pseudoinverse(q)
        n_dof = self.robot.num_dofs
        N = jnp.eye(n_dof) - J_bar @ J
        return N
