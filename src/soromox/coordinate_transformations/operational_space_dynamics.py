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
        task_selector: Boolean array specifying which velocity axes are active. The shape
            depends on the robot type and number of points:
            - For 3D robots (PCS): shape (N, 6) or (6*N,) selecting from
              [omega_x, omega_y, omega_z, v_x, v_y, v_z] per point
            - For planar robots (PlanarPCS, Pendulum): shape (N, 3) or (3*N,) selecting from
              [omega_z, v_x, v_y] per point
            IMPORTANT: For 3D robots, rotation components (omega_x, omega_y, omega_z) must
            be either ALL selected or ALL unselected for each point. This ensures consistency
            between velocity and pose spaces.
        rotation_representation: The representation used for orientation in the
            operational space coordinates. Options are:
            - ROTATION_VECTOR (default): 3D rotation vector (axis-angle)
            - QUATERNION: 4D unit quaternion
            - ROTATION_MATRIX_6D: 6D continuous representation
            Note: For planar robots, this is ignored as orientation is always a scalar angle.
        n_operational_space: Total dimension of the operational space after task selection.
            This is the same for both velocity space and pose space when using ROTATION_VECTOR.
            For other rotation representations, pose dimension differs.
        n_pose_operational_space: Dimension of task-selected pose space. Equal to
            n_operational_space for ROTATION_VECTOR and planar robots, but differs
            for QUATERNION (+1 per point with rotation) and ROTATION_MATRIX_6D (+3 per point).
        n_pose_dim: Dimension of a single full pose (depends on rotation_representation for 3D).
        n_points: Number of points along the backbone.

    Notes:
        The dynamically-consistent pseudo-inverse preserves the relationship between
        operational space forces and configuration space accelerations through the
        mass matrix, ensuring physical consistency of the transformed dynamics.

        All methods consistently apply task selection:
        - `operational_space_poses()` returns full poses (for trajectories and error computation)
        - `operational_space_coordinates()` returns task-selected pose coordinates
        - `jacobian()` and dynamics methods return task-selected quantities
        - `compute_pose_error()` takes full poses and returns full error
        - `compute_task_pose_error()` takes full poses and returns task-selected error

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
    task_selector: Array  # Boolean selector for which velocity axes are active
    B_task: Array  # Task selection basis matrix for velocity and acceleration space
    B_task_pose: Array  # Task selection basis matrix for pose space
    rotation_selected_per_point: (
        Array  # Boolean array (n_points,) indicating if rotation is selected
    )
    rotation_representation: RotationRepresentation = eqx.field(static=True)
    n_operational_space: int = eqx.field(static=True)  # Velocity space dimension
    n_pose_operational_space: int = eqx.field(static=True)  # Pose space dimension
    n_velocity_dim: int = eqx.field(
        static=True
    )  # Velocity space dim per point (always 6 for 3D, 3 for planar)
    n_pose_dim: int = eqx.field(
        static=True
    )  # Full pose dim per point (varies with rotation repr for 3D)
    n_orientation_dim: int = eqx.field(static=True)  # Dim of orientation in chosen repr
    n_angular_velocity_dim: int = eqx.field(
        static=True
    )  # Dim of angular velocity (3 for 3D, 1 for planar)
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
                IMPORTANT: For 3D robots, the rotation components (omega_x, omega_y, omega_z)
                must be either ALL selected or ALL unselected for each point. This ensures
                consistency between velocity and pose space representations.
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
            ValueError: If rotation components are partially selected (3D robots only).
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
        self.n_angular_velocity_dim = 1 if self.is_planar else 3

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

        # Validate rotation selection: all-or-nothing per point (for 3D robots only)
        # This ensures consistency between velocity and pose space
        rotation_selected_per_point = self._validate_rotation_selection(task_selector)
        self.rotation_selected_per_point = rotation_selected_per_point

        # Build the velocity-space task selection basis matrix
        # B_task maps from full velocity space to selected operational space
        # Shape: (total_velocity_dim, n_operational_space)
        self.B_task = self._compute_task_basis(task_selector)

        # Build the pose-space task selection basis matrix
        # B_task_pose maps from full pose space to selected pose operational space
        pose_task_selector, n_pose_operational_space = self._compute_pose_task_selector(
            task_selector, rotation_selected_per_point
        )
        self.B_task_pose = self._compute_pose_task_basis(pose_task_selector)
        self.n_pose_operational_space = n_pose_operational_space

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
        Compute the task selection basis matrix for velocity space.

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

    def _validate_rotation_selection(self, task_selector: Array) -> Array:
        """
        Validate that rotation components are selected all-or-nothing per point.

        For 3D robots, the rotation components (omega_x, omega_y, omega_z at indices 0, 1, 2)
        must be either ALL selected or ALL unselected for each point. This ensures
        consistency between velocity and pose space representations.

        For planar robots, there's only one rotation component so no constraint is needed.

        Args:
            task_selector: Boolean array of shape (total_velocity_dim,).

        Returns:
            rotation_selected: Boolean array of shape (n_points,) indicating whether
                rotation is selected for each point.

        Raises:
            ValueError: If rotation components are partially selected for any point.
        """
        import numpy as np  # Use numpy for validation at construction time

        task_selector_np = np.asarray(task_selector)
        task_selector_reshaped = task_selector_np.reshape(
            self.n_points, self.n_velocity_dim
        )

        rotation_selected = np.zeros(self.n_points, dtype=bool)

        for i in range(self.n_points):
            if self.is_planar:
                # Planar: only 1 rotation component (omega_z at index 0)
                rotation_selected[i] = task_selector_reshaped[i, 0]
            else:
                # 3D: rotation components are indices 0, 1, 2
                rot_selector = task_selector_reshaped[i, :3]
                all_rot = np.all(rot_selector)
                no_rot = not np.any(rot_selector)

                if not (all_rot or no_rot):
                    raise ValueError(
                        f"For point {i} (s={float(self.s_ps[i]):.4f}), rotation components "
                        f"must be either ALL selected or ALL unselected. "
                        f"Got task_selector for [omega_x, omega_y, omega_z] = {rot_selector.tolist()}. "
                        f"This constraint ensures consistency between velocity and pose spaces."
                    )

                rotation_selected[i] = all_rot

        return jnp.array(rotation_selected)

    def _compute_pose_task_selector(
        self, task_selector: Array, rotation_selected_per_point: Array
    ) -> tuple[Array, int]:
        """
        Compute the pose-space task selector from the velocity-space selector.

        The pose space may have a different dimension than velocity space due to
        different rotation representations:
        - Velocity space: angular velocity is always 3D (or 1D for planar)
        - Pose space: orientation dim depends on representation (3, 4, or 6 for 3D)

        Args:
            task_selector: Boolean array of shape (total_velocity_dim,).
            rotation_selected_per_point: Boolean array of shape (n_points,).

        Returns:
            pose_task_selector: Boolean array of shape (total_pose_dim,).
            n_pose_operational_space: Dimension of task-selected pose space.
        """
        import numpy as np

        task_selector_np = np.asarray(task_selector)
        rotation_selected_np = np.asarray(rotation_selected_per_point)
        task_selector_reshaped = task_selector_np.reshape(
            self.n_points, self.n_velocity_dim
        )

        # Build pose-space task selector
        pose_selectors = []

        for i in range(self.n_points):
            if self.is_planar:
                # Planar: pose = [theta, x, y], velocity = [omega_z, v_x, v_y]
                # Direct 1-to-1 mapping
                pose_selectors.append(task_selector_reshaped[i, :])
            else:
                # 3D: velocity = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
                # Pose = [orientation (n_orientation_dim), position (3)]
                point_pose_selector = []

                # Orientation: all selected if rotation is in task, all False otherwise
                if rotation_selected_np[i]:
                    point_pose_selector.extend([True] * self.n_orientation_dim)
                else:
                    point_pose_selector.extend([False] * self.n_orientation_dim)

                # Position: direct mapping from v_x, v_y, v_z (indices 3, 4, 5)
                point_pose_selector.extend(task_selector_reshaped[i, 3:6].tolist())

                pose_selectors.append(np.array(point_pose_selector))

        pose_task_selector = np.concatenate(pose_selectors)
        n_pose_operational_space = int(np.sum(pose_task_selector))

        return jnp.array(pose_task_selector, dtype=bool), n_pose_operational_space

    def _compute_pose_task_basis(self, pose_task_selector: Array) -> Array:
        """
        Compute the task selection basis matrix for pose space.

        Args:
            pose_task_selector: Boolean array of shape (total_pose_dim,).

        Returns:
            B_task_pose: Selection matrix of shape (total_pose_dim, n_pose_operational_space).
        """
        total_pose_dim = self.n_points * self.n_pose_dim
        n_selected = int(jnp.sum(pose_task_selector))

        # Build selection matrix: columns correspond to selected dimensions
        selected_indices = jnp.where(pose_task_selector, size=n_selected)[0]
        B_task_pose = jnp.zeros((total_pose_dim, n_selected))
        B_task_pose = B_task_pose.at[selected_indices, jnp.arange(n_selected)].set(1.0)

        return B_task_pose

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
    def _stacked_jacobian_and_time_derivative_full(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array]:
        """
        Compute the full stacked Jacobian and its time derivative for all points.

        Uses the robot's jacobian_and_time_derivative_batched method if available.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            J_full: Stacked Jacobian of shape (n_points * n_velocity_dim, num_dofs).
            Jd_full: Stacked Jacobian time derivative of shape (n_points * n_velocity_dim, num_dofs).
        """
        # Use batched method from the robot
        J_ps, Jd_ps = self.robot.jacobian_and_time_derivative_batched(q, qd, self.s_ps)
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
        Compute the FULL geometric pose error between current and desired poses.

        This method computes the proper geometric error for orientations, which is
        essential for PID-based control. Unlike naive subtraction (x_des - x), this:
        - Takes the shortest angular path for orientations
        - Properly wraps angles for planar systems
        - Works correctly regardless of the rotation representation

        The returned error is in velocity space (always 6D per point for 3D robots,
        3D per point for planar robots), NOT task-selected. Use `compute_task_pose_error`
        if you need the task-selected error.

        Args:
            pose_current: Current full pose of shape (n_points * n_pose_dim,).
                For 3D robots: n_pose_dim is 6 (ROTATION_VECTOR), 7 (QUATERNION),
                or 9 (ROTATION_MATRIX_6D). For planar robots: n_pose_dim is 3.
            pose_desired: Desired full pose of shape (n_points * n_pose_dim,).

        Returns:
            error: Full geometric pose error in velocity space.
                   Shape: (n_points * n_velocity_dim,).

        Raises:
            ValueError: If input poses have incorrect shape.

        Example:
            >>> # Get FULL current pose
            >>> x = osd.operational_space_poses(q)  # shape (6,) for 1 point
            >>> # Define FULL desired pose
            >>> x_des = jnp.array([0., 0., 0., 0.0, 0.0, 0.2])  # [rot, pos]
            >>> # Compute FULL error (all 6 components)
            >>> error = osd.compute_pose_error(x, x_des)  # shape (6,)

        See Also:
            compute_task_pose_error: Returns task-selected pose error.
        """
        # Validate input shapes
        expected_shape = (self.n_points * self.n_pose_dim,)
        if pose_current.shape != expected_shape:
            raise ValueError(
                f"pose_current must have shape {expected_shape}, "
                f"got {pose_current.shape}. Use operational_space_poses() "
                f"to get the full pose."
            )
        if pose_desired.shape != expected_shape:
            raise ValueError(
                f"pose_desired must have shape {expected_shape}, "
                f"got {pose_desired.shape}. The reference trajectory must provide "
                f"full poses (n_points * n_pose_dim dimensions)."
            )

        # Reshape to per-point poses
        poses_current = pose_current.reshape(self.n_points, self.n_pose_dim)
        poses_desired = pose_desired.reshape(self.n_points, self.n_pose_dim)

        # Compute geometric error for each point (returns velocity-space error)
        # _compute_single_pose_error returns n_velocity_dim error per point
        errors = vmap(self._compute_single_pose_error)(poses_current, poses_desired)

        # errors is (n_points, n_velocity_dim) - always 6D for 3D, 3D for planar
        return errors.flatten()

    @eqx.filter_jit
    def compute_task_pose_error(
        self, pose_current: Array, pose_desired: Array
    ) -> Array:
        """
        Compute the task-selected geometric pose error between current and desired poses.

        This method calls `compute_pose_error` to get the full geometric error, then
        applies task selection to return only the error components that are part of
        the task.

        IMPORTANT:
        - Input poses must be FULL poses (all points, all dimensions), not task-selected.
        - Use `operational_space_poses(q)` to get the current full pose.
        - The reference trajectory should also provide full poses (some components
          may be ignored by the controller based on task selection).

        Args:
            pose_current: Current full pose of shape (n_points * n_pose_dim,).
                For 3D robots: n_pose_dim is 6 (ROTATION_VECTOR), 7 (QUATERNION),
                or 9 (ROTATION_MATRIX_6D). For planar robots: n_pose_dim is 3.
            pose_desired: Desired full pose of shape (n_points * n_pose_dim,).

        Returns:
            error: Task-selected geometric pose error in velocity space.
                   Shape: (n_operational_space,).

        Example:
            >>> # Create operational space dynamics with position-only task
            >>> osd = OperationalSpaceDynamics(robot, s_ps,
            ...     task_selector=jnp.array([False, False, False, True, True, True]))
            >>> # Get FULL current pose (includes rotation even though not in task)
            >>> x = osd.operational_space_poses(q)  # shape (6,)
            >>> # Define FULL desired pose
            >>> x_des = jnp.array([0., 0., 0., 0.0, 0.0, 0.2])  # [rot, pos]
            >>> # Compute task-selected error (only position error)
            >>> error = osd.compute_task_pose_error(x, x_des)  # shape (3,)

        See Also:
            compute_pose_error: Returns full pose error without task selection.
        """
        # Compute full pose error
        error_full = self.compute_pose_error(pose_current, pose_desired)

        # Apply task selection (operates on velocity space)
        return self.B_task.T @ error_full

    # =========================================================================
    # Kinematics: coordinates, velocity, and Jacobians
    # =========================================================================

    @eqx.filter_jit
    def operational_space_coordinates(self, q: Array) -> Array:
        """
        Compute the task-selected operational space pose coordinates from forward kinematics.

        This method evaluates the forward kinematics at the specified points
        along the backbone (s_ps) and extracts the pose components in the chosen
        rotation representation, then applies task selection.

        Task selection is applied consistently with the velocity-space selection:
        - If rotation is selected for a point, all orientation components are included
        - Position components are included according to the task selector

        The pose dimension depends on the rotation representation and task selection:

        For 3D robots, the FULL pose at each point depends on rotation_representation:
        - ROTATION_VECTOR: [rx, ry, rz, p_x, p_y, p_z] (6D per point)
        - QUATERNION: [q_x, q_y, q_z, q_w, p_x, p_y, p_z] (7D per point)
        - ROTATION_MATRIX_6D: [r6d_0, ..., r6d_5, p_x, p_y, p_z] (9D per point)

        For planar robots, the pose at each point is [theta, x, y] (3D per point).

        After task selection, only the selected components are returned.

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            x: Task-selected operational space coordinates of shape
               (n_pose_operational_space,).

        Note:
            For computing errors for control, use `compute_task_pose_error()` which
            properly handles the geometric orientation error.
        """
        # Get full pose and apply task selection
        pose_full = self.operational_space_poses(q)
        return self.B_task_pose.T @ pose_full

    @eqx.filter_jit
    def operational_space_poses(self, q: Array) -> Array:
        """
        Compute the FULL operational space poses without task selection.

        This method returns the complete pose at each point along the backbone,
        regardless of the task selector. Use this for:
        - Defining reference trajectories (which should be in full pose space)
        - Computing pose errors with `compute_pose_error()` or `compute_task_pose_error()`
        - Visualization

        For 3D robots, the pose at each point depends on rotation_representation:
        - ROTATION_VECTOR: [rx, ry, rz, p_x, p_y, p_z] (6D per point)
        - QUATERNION: [q_x, q_y, q_z, q_w, p_x, p_y, p_z] (7D per point)
        - ROTATION_MATRIX_6D: [r6d_0, ..., r6d_5, p_x, p_y, p_z] (9D per point)

        For planar robots, the pose at each point is [theta, x, y] (3D per point).

        Args:
            q: Generalized coordinates of shape (num_dofs,).

        Returns:
            poses: Full operational space poses of shape (n_points * n_pose_dim,).

        See Also:
            operational_space_coordinates: Returns task-selected coordinates.
            compute_pose_error: Returns full pose error.
            compute_task_pose_error: Returns task-selected pose error.
        """
        # Compute forward kinematics at all points
        fk_results = self.robot.forward_kinematics_batched(q, self.s_ps)

        # Extract pose vectors from forward kinematics results
        poses = vmap(self._extract_pose_from_fk)(fk_results)

        # Flatten to get full pose vector
        return poses.flatten()

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
    def jacobian_and_time_derivative(self, q: Array, qd: Array) -> tuple[Array, Array]:
        """
        Compute the operational space Jacobian and its time derivative.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            J: Operational space Jacobian of shape (n_operational_space, num_dofs).
            Jd: Time derivative of J, shape (n_operational_space, num_dofs).
        """
        J_full, Jd_full = self._stacked_jacobian_and_time_derivative_full(q, qd)
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
    def mu_x(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space Coriolis/centrifugal matrix mu_x that
            connects the operational-space Coriolis forces with the configuration-space velocities qd.

        This Coriolis matrix of shape (n_operational_space, num_dofs) is defined as:
            mu_x = Lambda @ (J @ M^{-1} @ C - Jd)

        where:
            - Lambda is the operational space inertia matrix
            - J is the Jacobian
            - M is the configuration space inertia matrix
            - C is the configuration space Coriolis matrix
            - Jd is the time derivative of the Jacobian

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).

        Returns:
            mu_x: Operational space Coriolis matrix of shape
                (n_operational_space, num_dofs) that connects the operational-space Coriolis forces with the configuration-space velocities.
        """
        J, Jd = self.jacobian_and_time_derivative(q, qd)
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        M_inv = jnp.linalg.inv(M)

        # Compute Lambda
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # Compute the Coriolis force acting on the operational space with factorized qd
        mu_x = Lambda @ (J @ M_inv @ C - Jd)

        return mu_x

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the operational space Coriolis/centrifugal matrix mu_xx that
            connects the operational-space Coriolis forces with the operational-space velocities xd.

        The operational space Coriolis matrix is defined as:
            mu_xx = Lambda @ (J @ M^{-1} @ C - Jd) @ J_bar

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
            mu_xx: Operational space Coriolis matrix of shape
                (n_operational_space, n_operational_space) that connects the operational-space Coriolis forces with the operational-space velocities.
        """
        Jbar = self.dynamically_consistent_pseudoinverse(q)
        mu_x = self.mu_x(q, qd)

        # mu_xx = mu_x @ Jbar
        mu_xx = mu_x @ Jbar

        return mu_xx

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
        J, Jd = self.jacobian_and_time_derivative(q, qd)
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        M_inv = jnp.linalg.inv(M)

        # Compute Lambda
        Lambda_inv = J @ M_inv @ J.T
        Lambda = jnp.linalg.inv(Lambda_inv)

        # Compute the Coriolis force acting on the operational space
        f_c_x = Lambda @ (J @ M_inv @ C - Jd) @ qd

        return f_c_x
