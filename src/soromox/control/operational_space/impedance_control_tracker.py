__all__ = ["ImpedanceControlTracker"]

import warnings
from typing import Any

import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import (
    ActuationScenario,
    analyze_actuation_matrix,
)
from soromox.control.operational_space.base_controller import (
    OperationalSpaceBaseController,
)
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.operational_space_dynamics import (
    OperationalSpaceDynamics,
)
from soromox.systems.system_state import SystemState


class ImpedanceControlTracker(OperationalSpaceBaseController):
    """
    Operational-space impedance controller for soft robots.

    This controller implements partial feedback linearization to cancel the original
    task dynamics and replace them with a new set of linear dynamics in operational
    space. The closed-loop dynamics in operational space become:

        Λ ẍ + D_x ẋ + K_x (x - x^d) = 0

    where Λ is the operational space inertia matrix, D_x is the desired damping,
    K_x is the desired stiffness, and x^d is the reference position.

    The control law is:

        τ = A^{-1}(q) [
            J^T(q) J_bar^T(q) (τ_el(q) + D(q) q̇)     (Cancel elastic & damping forces on task)
            + G(q)                                    (Cancel gravity)
            + J^T(q) μ_x(q,q̇) (I - J_bar(q) J(q)) q̇   (Cancel null-space Coriolis coupling to task)
            + J^T(q) (K_x (x^d - x) - D_x ẋ)         (PD for shaping operational space impedance)
        ]

    where:
        - A(q) is the actuation matrix (must be square and invertible)
        - J(q) is the operational space Jacobian
        - J_bar(q) = M^{-1} J^T Λ is the dynamically-consistent pseudo-inverse
        - Λ = (J M^{-1} J^T)^{-1} is the operational space inertia matrix
        - τ_el(q) is the configuration-space elastic force
        - D(q) is the configuration-space damping matrix
        - G(q) is the configuration-space gravitational force
        - μ(q, q̇) is the operational space Coriolis matrix
        - K_x is the operational space stiffness gain matrix
        - D_x is the operational space damping gain matrix
        - x^d is the desired operational space position
        - x is the current operational space position
        - ẋ is the current operational space velocity

    Assumptions:
        (a) Full actuation: n = m (number of DOFs equals number of actuators)
        (b) The actuation matrix A(q) ∈ ℝ^{n×n} is invertible
        (c) The operational space has lower or equal dimensionality than the
            configuration space (o ≤ n)
        (d) The null space is asymptotically stable, typically satisfied when
            K(q) = S q with S ≻ 0 and D ≻ 0 (positive definite stiffness and damping)

    Attributes:
        operational_space_dynamics: The OperationalSpaceDynamics instance.
        reference_trajectory: The desired trajectory in operational space.
        K_x: Operational space stiffness gain matrix, shape (o, o) or (o,).
        D_x: Operational space damping gain matrix, shape (o, o) or (o,).

    References:
        Khatib, O. (1987). A unified approach for motion and force control of robot
        manipulators: The operational space formulation. IEEE Journal on Robotics
        and Automation, 3(1), 43-53.

        Ott, C. (2008). Cartesian impedance control of redundant and flexible-joint robots. Springer.

        Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
        Model-based dynamic feedback control of a planar soft robot: trajectory
        tracking and interaction with the environment. The International Journal
        of Robotics Research, 39(4-5), 490-513.

        Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
        into Learned Models for Control. Dissertation, Delft University of Technology.
        https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da
    """

    K_x: Array  # Operational space stiffness gain
    D_x: Array  # Operational space damping gain

    def __init__(
        self,
        operational_space_dynamics: OperationalSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        K_x: float | Array,
        D_x: float | Array,
    ):
        """
        Initialize the operational-space impedance controller.

        Args:
            operational_space_dynamics: The OperationalSpaceDynamics instance that
                defines the task space and provides transformations between
                configuration and operational space.
            reference_trajectory: The desired trajectory in operational space.
                Must provide x_des_fn (desired position) and xd_des_fn (desired
                velocity) as functions of time. The trajectory dimension must match
                the operational space dimension (n_operational_space).
            K_x: Operational space stiffness gain. Can be:
                - A scalar (float): applied uniformly to all operational space dimensions.
                - A 1-d array of shape (o,): diagonal stiffness.
                - A 2-d array of shape (o, o): full stiffness matrix.
                Should be positive definite for stability.
            D_x: Operational space damping gain. Same format options as K_x.
                Should be positive definite for stability.

        Raises:
            ValueError: If the actuation matrix is not square or not invertible.
        """
        self.operational_space_dynamics = operational_space_dynamics
        self.reference_trajectory = reference_trajectory
        self.robot = operational_space_dynamics.robot

        # Convert gains to arrays
        n_op = operational_space_dynamics.n_operational_space
        self.K_x = self._process_gain(K_x, n_op, "K_x")
        self.D_x = self._process_gain(D_x, n_op, "D_x")

        # Check that the actuation matrix is square and invertible
        self._check_actuation()

    def _process_gain(self, gain: float | Array, n_op: int, name: str) -> Array:
        """
        Process a gain parameter into the appropriate array format.

        Args:
            gain: The gain value (scalar, 1-d, or 2-d array).
            n_op: The operational space dimension.
            name: The name of the gain for error messages.

        Returns:
            The gain as a JAX array.
        """
        gain = jnp.atleast_1d(jnp.asarray(gain))

        if gain.ndim == 0 or (gain.ndim == 1 and gain.shape[0] == 1):
            # Scalar: expand to diagonal
            gain = jnp.full((n_op,), float(gain.flatten()[0]))

        return gain

    def _check_actuation(self) -> None:
        """
        Check that the actuation matrix is square and invertible.

        Raises:
            ValueError: If the actuation matrix is not square or not invertible.
        """
        robot = self.operational_space_dynamics.robot
        scenario, shape, rank = analyze_actuation_matrix(robot)
        n_dof, n_actuators = shape

        if scenario != ActuationScenario.FULL_ACTUATION:
            raise ValueError(
                f"ImpedanceControlTracker requires a square actuation matrix "
                f"(full actuation with n = m). Got {n_actuators} actuators and "
                f"{n_dof} DOFs. For under- or over-actuated systems, consider "
                f"alternative control strategies."
            )

        if rank < n_dof:
            raise ValueError(
                f"ImpedanceControlTracker requires an invertible actuation matrix. "
                f"The actuation matrix has rank {rank} but should have full rank {n_dof}."
            )

        # Warn about configuration-dependent actuation
        warnings.warn(
            "ImpedanceControlTracker: For full theoretical guarantees, the actuation "
            "matrix should be configuration-independent (constant). If A(q) is "
            "configuration-dependent, the stability analysis may not hold exactly.",
            UserWarning,
            stacklevel=3,
        )

    def __call__(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the operational-space impedance control action.

        This method implements the full operational-space impedance control law
        that cancels elastic, damping, gravitational, and Coriolis forces while
        injecting the desired operational-space impedance behavior.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            u_control: The control input, shape (num_actuators,).
            control_state_dot: None (this controller is stateless).
        """
        t = system_state.t
        y = system_state.y

        # Extract configuration and velocity
        q, qd = jnp.split(y, 2)

        # Get operational space dynamics instance and robot
        osd = self.operational_space_dynamics
        n_dof = self.robot.num_dofs

        # Get reference trajectory at current time
        # IMPORTANT: The reference trajectory should provide FULL poses (all points,
        # all dimensions), not task-selected pose components. Some components may be ignored
        # by the controller based on task selection.
        # - x_des_full: shape (n_points * n_pose_dim,) - full pose
        # - xd_des_full: shape (n_points * n_velocity_dim,) - full velocity
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        x_des_full = self.reference_trajectory.x_des_fn(t)
        xd_des_full = self.reference_trajectory.xd_des_fn(t)

        # Apply task selection to velocity trajectory
        # (velocity space is where the Jacobian and dynamics operate)
        xd_des = osd.B_task.T @ xd_des_full

        # Compute operational space quantities
        J, Jd = osd.jacobian_and_derivative(q, qd)
        J_bar = osd.dynamically_consistent_pseudoinverse(q)

        # Current operational space FULL pose (for error computation) and task-selected velocity
        x_full = osd.operational_space_poses(q)
        xd = J @ qd  # Task-selected velocity

        # Null-space projector: N = I - J_bar @ J
        N = jnp.eye(n_dof) - J_bar @ J

        # Configuration-space forces
        tau_el = self.robot.elastic_force(q)  # Elastic force
        D_config = self.robot.damping_matrix(q)  # Damping matrix
        G = self.robot.gravitational_force(q)  # Gravitational force

        # ===== Compute control torque components =====

        # 1. Cancel elastic and damping forces projected to task space
        # The elastic and damping forces in configuration space that couple to the task
        # are projected using J_bar^T: f_x = J_bar^T @ tau_config
        # To cancel them, we apply: tau_cancel = J^T @ J_bar^T @ (tau_el + D @ qd)
        tau_task_forces = tau_el + D_config @ qd
        tau_cancel_task = J.T @ (J_bar.T @ tau_task_forces)

        # 2. Cancel gravity
        tau_cancel_gravity = G

        # 3. Cancel null-space Coriolis coupling to task
        # The null-space Coriolis coupling term cancels the effect of null-space
        # motions on the task dynamics through the Coriolis matrix.
        # From the formula: J^T @ Lambda @ (J @ M^{-1} @ C - Jd) @ N @ qd
        # tau_cancel_coriolis = J.T @ mu_x @ qd_null
        # This projects the null-space Coriolis force to the task space and then back.
        mu_x = osd.mu_x(q, qd)
        qd_null = N @ qd  # Null-space velocity
        tau_cancel_coriolis = J.T @ mu_x @ qd_null

        # 4. PD control in operational space
        # tau_pd = J^T @ (K_x @ e_x + D_x @ ed_x)
        #
        # IMPORTANT: For orientation, we use the geometric error (shortest path)
        # computed via osd.compute_pose_error(), not naive subtraction (x_des - x).
        # This ensures:
        #   - Proper angle wrapping for planar systems
        #   - Geodesic (shortest path) error for 3D orientations
        #   - Correct behavior near angle wrap-around points
        #
        # The velocity error (xd_des - xd) uses naive subtraction because
        # angular velocities live in the tangent space where subtraction is valid.

        # Position/orientation error in operational space (geometric error)
        # Note: compute_task_pose_error takes FULL poses and returns task-selected error
        e_x = osd.compute_task_pose_error(x_full, x_des_full)
        # Velocity error in operational space (tangent space, naive subtraction is OK)
        ed_x = xd_des - xd

        # PD term: K_x @ e_x + D_x @ ed_x
        tau_pd = J.T @ (
            self._apply_gain(self.K_x, e_x) + self._apply_gain(self.D_x, ed_x)
        )

        # Total generalized torque
        tau_control = (
            tau_cancel_task + tau_cancel_gravity + tau_cancel_coriolis + tau_pd
        )

        # Get the actuation matrix and compute actuator input
        A = self.robot.actuation_matrix(q)
        u_control = jnp.linalg.inv(A) @ tau_control

        return u_control, None
