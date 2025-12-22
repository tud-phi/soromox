__all__ = ["OperationalSpaceDynamics"]


import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.soft_robot import SoftRobot
from soromox.utils.rotations import (
    RotationRepresentation,
    angle_error,
    quaternion_to_rotation_matrix,
    quaternion_to_rotation_vector,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_error,
    rotation_matrix_to_6d,
    rotation_matrix_to_quaternion,
    rotation_vector_to_rotation_matrix,
)


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
        rotation_representation: The representation used for orientation in the
            operational space coordinates. Options are:
            - ROTATION_VECTOR (default): 3D rotation vector (axis-angle)
            - QUATERNION: 4D unit quaternion
            - ROTATION_MATRIX_6D: 6D continuous representation
            Note: For planar robots, this is ignored as orientation is always a scalar angle.
        n_operational_space: Total dimension of the operational space after selection.
        n_pose_dim: Dimension of a single pose (depends on rotation_representation for 3D).
        n_points: Number of points along the backbone.

    Notes:
        The dynamically-consistent pseudo-inverse preserves the relationship between
        operational space forces and configuration space accelerations through the
        mass matrix, ensuring physical consistency of the transformed dynamics.

        For control applications, use `compute_pose_error()` to compute the geometric
        orientation error rather than naive subtraction of coordinates. The geometric
        error takes the shortest path and is suitable for PID control.

    References:
        Khatib, O. (1987). A unified approach for motion and force control of robot
        manipulators: The operational space formulation. IEEE Journal on Robotics
        and Automation, 3(1), 43-53.

        Natale, C. (2003). Interaction control of robot manipulators: Six-degrees-of-freedom
        tasks. Springer Science & Business Media.

        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019).
        On the continuity of rotation representations in neural networks. CVPR 2019.
    """

    robot: SoftRobot
    s_ps: Array  # Points along the backbone, shape (N,)
    task_selector: Array  # Boolean selector for which axes are active
    B_task: Array  # Task selection basis matrix, shape (n_velocity_dim * N, n_operational_space)
    rotation_representation: RotationRepresentation = eqx.field(static=True)
    n_operational_space: int = eqx.field(static=True)
    n_velocity_dim: int = eqx.field(
        static=True
    )  # Velocity space dim (always 6 for 3D, 3 for planar)
    n_pose_dim: int = eqx.field(
        static=True
    )  # Pose dim (varies with rotation repr for 3D)
    n_orientation_dim: int = eqx.field(static=True)  # Dim of orientation in chosen repr
    n_points: int = eqx.field(static=True)
    is_planar: bool = eqx.field(
        static=True, default=False
    )  # True for planar robots (2D)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        robot: SoftRobot,
        s_ps: Array,
        task_selector: Array | None = None,
        rotation_representation: RotationRepresentation = RotationRepresentation.ROTATION_VECTOR,
    ):
        """
        Initialize the operational space dynamics transformer.

        Args:
            robot: A soft robot implementing the SoftRobot interface.
                Examples include PCS, PlanarPCS, TendonActuatedPCS, Pendulum, etc.
            s_ps: Array of shape (N,) specifying the arc-length positions along the
                backbone where the operational space is defined. Values should be
                in the interval [0, L_total] where L_total is the total robot length.
            task_selector: Optional boolean array specifying which VELOCITY components
                are active in the operational space. The task selector operates on
                the velocity space (Jacobian rows), which is always:
                - 6D for 3D robots: [angular_velocity (3), linear_velocity (3)]
                - 3D for planar robots: [angular_velocity (1), linear_velocity (2)]
                Can be:
                - Shape (n_velocity_dim,): Applied uniformly to all points
                - Shape (N, n_velocity_dim): Different selection per point
                - Shape (N * n_velocity_dim,): Flattened selection
                If None, all velocity components at all points are active.
                NOTE: The task selector dimension is INDEPENDENT of the rotation
                representation because dynamics operate in velocity space.
            rotation_representation: The representation to use for orientations
                when reporting pose coordinates. This affects:
                - The output of `operational_space_coordinates()`
                - The expected input format for `compute_pose_error()`
                But it does NOT affect the dynamics (Jacobian, inertia, etc.) which
                always operate in velocity space.
                Options are:
                - ROTATION_VECTOR (default): 3D axis-angle, compact
                - QUATERNION: 4D unit quaternion
                - ROTATION_MATRIX_6D: 6D continuous representation (Zhou et al. 2019)
                This is ignored for planar robots (which always use a scalar angle).

        Raises:
            ValueError: If s_ps is empty or has invalid shape.
            ValueError: If task_selector has incompatible shape.
        """
        self.robot = robot
        self.s_ps = jnp.atleast_1d(jnp.asarray(s_ps))
        self.rotation_representation = rotation_representation

        if self.s_ps.ndim != 1 or self.s_ps.size == 0:
            raise ValueError(
                f"s_ps must be a 1D array with at least one element, got shape {self.s_ps.shape}"
            )

        n_points = self.s_ps.shape[0]
        self.n_points = n_points

        # Determine if planar robot by checking the Jacobian shape
        # For PCS: 6 rows → 3D robot
        # For PlanarPCS/Pendulum: 3 rows → planar robot
        jacobian_rows = self._infer_jacobian_rows()
        self.is_planar = jacobian_rows == 3

        # The velocity dimension is fixed by the Jacobian (angular vel + linear vel)
        # This is independent of how we represent orientation
        self.n_velocity_dim = jacobian_rows  # 6 for 3D, 3 for planar

        # Determine orientation dimension based on representation
        # This only affects how coordinates are reported, not the dynamics
        if self.is_planar:
            # Planar robots always use scalar angle
            self.n_orientation_dim = 1
            n_pose_dim = 3  # [theta, x, y]
        else:
            # 3D robots: orientation dimension depends on representation
            if rotation_representation == RotationRepresentation.ROTATION_VECTOR:
                self.n_orientation_dim = 3
            elif rotation_representation == RotationRepresentation.QUATERNION:
                self.n_orientation_dim = 4
            elif rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
                self.n_orientation_dim = 6
            else:
                raise ValueError(
                    f"Unknown rotation representation: {rotation_representation}"
                )
            n_pose_dim = self.n_orientation_dim + 3  # orientation + position

        self.n_pose_dim = n_pose_dim

        # Task selector operates on the VELOCITY space (Jacobian rows), not pose space
        # This is because dynamics are formulated in terms of velocities
        total_velocity_dim = n_points * self.n_velocity_dim

        # Handle task_selector
        if task_selector is None:
            # All axes active
            task_selector = jnp.ones(total_velocity_dim, dtype=bool)
        else:
            task_selector = jnp.asarray(task_selector)

            # Handle different input shapes
            if task_selector.shape == (self.n_velocity_dim,):
                # Same selection for all points - broadcast
                task_selector = jnp.tile(task_selector, n_points)
            elif task_selector.shape == (n_points, self.n_velocity_dim):
                # Per-point selection - flatten
                task_selector = task_selector.flatten()
            elif task_selector.shape == (total_velocity_dim,):
                # Already flattened
                pass
            else:
                raise ValueError(
                    f"task_selector must have shape ({self.n_velocity_dim},), "
                    f"({n_points}, {self.n_velocity_dim}), or ({total_velocity_dim},), "
                    f"got {task_selector.shape}"
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

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _infer_jacobian_rows(self) -> int:
        """
        Infer the number of Jacobian rows from the robot.

        Returns:
            n_rows: 6 for 3D robots (PCS), 3 for planar robots (PlanarPCS, Pendulum).
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

        The task selector operates on the velocity space (Jacobian rows), which
        is always 6D for 3D robots and 3D for planar robots, regardless of the
        chosen rotation representation.

        Args:
            task_selector: Boolean array of shape (total_velocity_dim,).

        Returns:
            B_task: Selection matrix of shape (total_velocity_dim, n_operational_space).
        """
        total_velocity_dim = self.n_points * self.n_velocity_dim
        n_selected = int(jnp.sum(task_selector))

        # Build selection matrix: columns correspond to selected dimensions
        selected_indices = jnp.where(task_selector, size=n_selected)[0]
        B_task = jnp.zeros((total_velocity_dim, n_selected))
        B_task = B_task.at[selected_indices, jnp.arange(n_selected)].set(1.0)

        return B_task

    @eqx.filter_jit
    def _stacked_jacobian_full(self, q: Array) -> Array:
        """
        Compute the full stacked Jacobian for all points (before task selection).

        Uses the robot's jacobian_batched method if available for efficiency.

        The Jacobian relates configuration velocities to operational space velocities.
        It has n_velocity_dim rows per point (6 for 3D, 3 for planar), which is
        independent of the rotation_representation (that only affects pose coordinates).

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            J_full: Stacked Jacobian of shape (n_points * n_velocity_dim, num_dofs).
        """
        # Use batched method from the robot
        J_ps = self.robot.jacobian_batched(q, self.s_ps)
        # J_ps has shape (n_points, n_velocity_dim, num_dofs)

        # Stack to get (n_points * n_velocity_dim, num_dofs)
        J_full = J_ps.reshape(-1, self.robot.num_dofs)

        return J_full

    @eqx.filter_jit
    def _stacked_jacobian_and_derivative_full(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array]:
        """
        Compute the full stacked Jacobian and its time derivative for all points.

        Uses the robot's jacobian_and_derivative_batched method if available.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            J_full: Stacked Jacobian of shape (n_points * n_velocity_dim, num_dofs).
            Jd_full: Stacked Jacobian derivative of shape (n_points * n_velocity_dim, num_dofs).
        """
        # Use batched method from the robot
        J_ps, Jd_ps = self.robot.jacobian_and_derivative_batched(q, qd, self.s_ps)
        # J_ps, Jd_ps have shape (n_points, n_pose_dim, num_dofs)

        # Stack to get (n_points * n_pose_dim, num_dofs)
        J_full = J_ps.reshape(-1, self.robot.num_dofs)
        Jd_full = Jd_ps.reshape(-1, self.robot.num_dofs)

        return J_full, Jd_full

    def _extract_pose_from_fk(self, fk_result: Array) -> Array:
        """
        Extract the pose vector from the forward kinematics result.

        For 3D robots (PCS), forward_kinematics returns a 4x4 SE(3) matrix.
        We extract the pose using the configured rotation representation:
        - ROTATION_VECTOR: [omega_x, omega_y, omega_z, p_x, p_y, p_z] (6D)
        - QUATERNION: [q_x, q_y, q_z, q_w, p_x, p_y, p_z] (7D)
        - ROTATION_MATRIX_6D: [r6d_0, ..., r6d_5, p_x, p_y, p_z] (9D)

        For planar robots (PlanarPCS, Pendulum), forward_kinematics returns
        a 3-vector [theta, x, y], which we return directly.

        Args:
            fk_result: Forward kinematics result from the robot.

        Returns:
            pose: Pose vector of shape (n_pose_dim,).
        """
        if self.is_planar:
            # Planar robot: fk_result is [theta, x, y]
            return fk_result
        else:
            # 3D robot: fk_result is a 4x4 SE(3) matrix
            # Extract position from the last column
            p = fk_result[:3, 3]
            # Extract rotation matrix
            R = fk_result[:3, :3]

            # Convert to the selected representation
            if self.rotation_representation == RotationRepresentation.ROTATION_VECTOR:
                quat = rotation_matrix_to_quaternion(R)
                orientation = quaternion_to_rotation_vector(quat)
            elif self.rotation_representation == RotationRepresentation.QUATERNION:
                orientation = rotation_matrix_to_quaternion(R)
            elif (
                self.rotation_representation
                == RotationRepresentation.ROTATION_MATRIX_6D
            ):
                orientation = rotation_matrix_to_6d(R)
            else:
                raise ValueError(
                    f"Unknown rotation representation: {self.rotation_representation}"
                )

            # Concatenate orientation and position
            return jnp.concatenate([orientation, p])

    def _compute_single_pose_error(
        self, pose_current: Array, pose_desired: Array
    ) -> Array:
        """
        Compute the geometric error for a single pose.

        For planar robots, this wraps the angle error to [-π, π].
        For 3D robots, this computes the proper geometric orientation error
        regardless of the rotation representation used.

        Args:
            pose_current: Current pose vector of shape (n_pose_dim,).
            pose_desired: Desired pose vector of shape (n_pose_dim,).

        Returns:
            error: Pose error vector. For 3D robots, always returns 6D error
                   [rotation_vector_error, position_error] regardless of
                   the rotation representation, since the geometric error
                   naturally lives in the 3D tangent space.
        """
        if self.is_planar:
            # Planar: pose = [theta, x, y]
            theta_current = pose_current[0]
            theta_desired = pose_desired[0]
            pos_current = pose_current[1:]
            pos_desired = pose_desired[1:]

            # Geometric angle error (shortest path)
            theta_error = angle_error(theta_current, theta_desired)
            pos_error = pos_desired - pos_current

            return jnp.concatenate([jnp.array([theta_error]), pos_error])
        else:
            # 3D robot: extract orientation and position
            n_orient = self.n_orientation_dim
            orient_current = pose_current[:n_orient]
            orient_desired = pose_desired[:n_orient]
            pos_current = pose_current[n_orient:]
            pos_desired = pose_desired[n_orient:]

            # Convert to rotation matrices for geometric error computation
            if self.rotation_representation == RotationRepresentation.ROTATION_VECTOR:
                R_current = rotation_vector_to_rotation_matrix(orient_current)
                R_desired = rotation_vector_to_rotation_matrix(orient_desired)
            elif self.rotation_representation == RotationRepresentation.QUATERNION:
                R_current = quaternion_to_rotation_matrix(orient_current)
                R_desired = quaternion_to_rotation_matrix(orient_desired)
            elif (
                self.rotation_representation
                == RotationRepresentation.ROTATION_MATRIX_6D
            ):
                R_current = rotation_6d_to_rotation_matrix(orient_current)
                R_desired = rotation_6d_to_rotation_matrix(orient_desired)
            else:
                raise ValueError(
                    f"Unknown rotation representation: {self.rotation_representation}"
                )

            # Compute geometric orientation error (always 3D rotation vector)
            orient_error = rotation_matrix_error(R_current, R_desired)
            pos_error = pos_desired - pos_current

            return jnp.concatenate([orient_error, pos_error])

    def _compute_single_velocity_error(
        self, vel_current: Array, vel_desired: Array
    ) -> Array:
        """
        Compute the velocity-space error for a single point.

        For planar robots, this wraps the angular velocity error (though for velocities
        wrapping is typically not needed). For 3D robots, this is just the difference.

        Note: This is for velocity errors which live in the tangent space where
        naive subtraction IS correct (unlike orientation errors).

        Args:
            vel_current: Current velocity of shape (n_velocity_dim,).
            vel_desired: Desired velocity of shape (n_velocity_dim,).

        Returns:
            error: Velocity error of shape (n_velocity_dim,).
        """
        # Velocity errors can use naive subtraction (tangent space)
        return vel_desired - vel_current

    @eqx.filter_jit
    def compute_pose_error(self, pose_current: Array, pose_desired: Array) -> Array:
        """
        Compute the geometric pose error between current and desired poses.

        This method computes the proper geometric error for orientations, which is
        essential for PID-based control. Unlike naive subtraction (x_des - x), this:
        - Takes the shortest angular path for orientations
        - Properly wraps angles for planar systems
        - Works correctly regardless of the rotation representation

        IMPORTANT: The returned error is always in the velocity space (6D for 3D robots,
        3D for planar), not the pose space. This is because:
        1. The dynamics (Jacobian, inertia) operate in velocity space
        2. Orientation errors naturally live in the tangent space (same as angular velocity)
        3. Control gains K_x and D_x are sized for the operational space (velocity-based)

        Args:
            pose_current: Current pose in the chosen representation. Shape depends on
                representation: (n_points * n_pose_dim,) where n_pose_dim is:
                - Planar: 3 (theta, x, y)
                - 3D + ROTATION_VECTOR: 6 (omega, p)
                - 3D + QUATERNION: 7 (q, p)
                - 3D + ROTATION_MATRIX_6D: 9 (r6d, p)
            pose_desired: Desired pose in the same representation.

        Returns:
            error: Geometric pose error in velocity space (always n_velocity_dim per point).
                   Shape: (n_operational_space,) after task selection.

        Example:
            >>> # For a planar robot, angles wrap correctly:
            >>> osd = OperationalSpaceDynamics(robot, s_ps)
            >>> pose_current = jnp.array([jnp.deg2rad(10), 0.0, 0.0])
            >>> pose_desired = jnp.array([jnp.deg2rad(350), 0.0, 0.0])
            >>> error = osd.compute_pose_error(pose_current, pose_desired)
            >>> # error[0] ≈ -0.349 (≈ -20°, not +340°)

        Note:
            The input poses should be FULL poses (all points, all dimensions), not
            task-selected. This is different from the velocity-based operational
            space coordinates. Use `operational_space_coordinates()` to get the
            full pose.
        """
        # Reshape to per-point poses
        poses_current = pose_current.reshape(self.n_points, self.n_pose_dim)
        poses_desired = pose_desired.reshape(self.n_points, self.n_pose_dim)

        # Compute geometric error for each point (returns velocity-space error)
        errors = vmap(self._compute_single_pose_error)(poses_current, poses_desired)

        # errors is (n_points, n_velocity_dim) - always 6D for 3D, 3D for planar
        error_full = errors.flatten()

        # Apply task selection (operates on velocity space)
        return self.B_task.T @ error_full

    # =========================================================================
    # Kinematics: coordinates, velocity, and Jacobians
    # =========================================================================

    @eqx.filter_jit
    def operational_space_coordinates(self, q: Array) -> Array:
        """
        Compute the full operational space pose coordinates from forward kinematics.

        This method evaluates the forward kinematics at the specified points
        along the backbone (s_ps) and extracts the pose components in the chosen
        rotation representation.

        IMPORTANT: This returns the FULL pose (all points, all dimensions) without
        task selection. Task selection only applies to velocity-space quantities
        (Jacobian, dynamics). The pose dimension depends on the rotation representation:

        For 3D robots, the pose at each point depends on rotation_representation:
        - ROTATION_VECTOR: [omega_x, omega_y, omega_z, p_x, p_y, p_z] (6D per point)
        - QUATERNION: [q_x, q_y, q_z, q_w, p_x, p_y, p_z] (7D per point)
        - ROTATION_MATRIX_6D: [r6d_0, ..., r6d_5, p_x, p_y, p_z] (9D per point)

        For planar robots, the pose at each point is [theta, x, y] (3D per point).

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            x: Full operational space coordinates of shape (n_points * n_pose_dim,).
               Note: This is NOT n_operational_space because task selection does
               not apply to pose coordinates.

        Note:
            For computing errors for control, use `compute_pose_error()` which
            properly handles the geometric orientation error and applies task
            selection to the velocity-space error output.
        """
        # Compute forward kinematics at all points
        fk_results = self.robot.forward_kinematics_batched(q, self.s_ps)

        # Extract pose vectors from forward kinematics results
        poses = vmap(self._extract_pose_from_fk)(fk_results)

        # Flatten to get full pose vector
        pose_full = poses.flatten()

        return pose_full

    @eqx.filter_jit
    def operational_space_velocity(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space velocity (with task selection applied).

        The velocity is computed as xd = J @ qd where J is the task-selected
        Jacobian. The velocity lives in the velocity space which is:
        - 6D per point for 3D robots: [angular_velocity (3), linear_velocity (3)]
        - 3D per point for planar robots: [angular_velocity (1), linear_velocity (2)]

        After task selection, the dimension is n_operational_space.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            xd: Operational space velocity of shape (n_operational_space,).
                Note: This is in velocity space (always 6D per point for 3D before
                task selection), regardless of the rotation_representation.
        """
        J = self.jacobian(q)
        return J @ qd

    @eqx.filter_jit
    def jacobian(self, q: Array) -> Array:
        """
        Compute the operational space Jacobian (with task selection applied).

        The Jacobian maps configuration velocities to operational space velocities:
            xd = J @ qd

        The Jacobian operates in velocity space which is always 6D per point for
        3D robots (3 angular + 3 linear velocity) regardless of rotation_representation.
        Task selection then reduces this to n_operational_space dimensions.

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
    def jacobian_and_derivative(self, q: Array, qd: Array) -> tuple[Array, Array]:
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

    # =========================================================================
    # Dynamically-consistent pseudo-inverse and null-space
    # =========================================================================

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
    ) -> tuple[Array, Array]:
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

    # =========================================================================
    # Dynamical matrices
    # =========================================================================

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

    # =========================================================================
    # Forces
    # =========================================================================

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
        J, Jd = self.jacobian_and_derivative(q, qd)
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        M_inv = jnp.linalg.inv(M)

        # Compute Lambda
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # Compute the Coriolis force acting on the operational space
        f_c_x = Lambda @ (J @ M_inv @ C - Jd) @ qd

        return f_c_x
