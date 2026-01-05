__all__ = ["SynergisticController"]

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


class SynergisticController(OperationalSpaceBaseController):
    """
    Synergistic controller for soft robots.

    The control law is:

        τ = (J(q) M^{-1}(q) A(q))^{-1} J(q) M^{-1}(q) J^T(q) (K_x (x^d - x) - D_x ẋ)

    where:
        - A(q) is the actuation matrix
        - M(q) is the inertia matrix
        - J(q) is the operational space Jacobian
        - K_x is the operational space proportional gain matrix
        - D_x is the operational space derivative gain matrix
        - x^d is the desired operational space position
        - x is the current operational space position
        - ẋ is the current operational space velocity

    Assumptions:
        (a) Under-actuation: The actuation space has lower dimensionality than
            the configuration space (m < n). If full actuation, consider using, for example, the impedance control tracker.
        (b) The operational space has equal dimensionality of the actuation
            space (o = m)
        (c) The matrix J(q) M^{-1}(q) A(q) ∈ ℝ^{m×m} is full-rank

    Attributes:
        operational_space_dynamics: The OperationalSpaceDynamics instance.
        reference_trajectory: The desired trajectory in operational space.
        K_x: Operational space proportional gain matrix, shape (o, o) or (o,).
        D_x: Operational space derivative gain matrix, shape (o, o) or (o,).

    References:
        Della Santina, C., Pallottino, L., Rus, D., & Bicchi, A. (2019). Exact
        task execution in highly under-actuated soft limbs: an operational
        space based approach. IEEE Robotics and Automation Letters, 4(3),
        2508-2515.
    """

    K_x: Array  # Operational space proportional gain
    D_x: Array  # Operational space derivative gain

    def __init__(
        self,
        operational_space_dynamics: OperationalSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        K_x: float | Array,
        D_x: float | Array,
    ):
        """
        Initialize the synergistic controller.

        Args:
            operational_space_dynamics: The OperationalSpaceDynamics instance that
                defines the task space and provides transformations between
                configuration and operational space.
            reference_trajectory: The desired trajectory in operational space.
                Must provide x_des_fn (desired position) and xd_des_fn (desired
                velocity) as functions of time. The trajectory dimension must match
                the operational space dimension (n_operational_space).
            K_x: Operational space proportional gain. Can be:
                - A scalar (float): applied uniformly to all operational space dimensions.
                - A 1-d array of shape (o,): diagonal stiffness.
                - A 2-d array of shape (o, o): full stiffness matrix.
                Should be positive definite.
            D_x: Operational space derivative gain. Same format options as K_x.
                Should be positive definite.

        Raises:
            ValueError: If either assumption (a) or (b) is not met.
        """
        self.operational_space_dynamics = operational_space_dynamics
        self.reference_trajectory = reference_trajectory
        self.robot = operational_space_dynamics.robot

        # Convert gains to arrays
        n_op = operational_space_dynamics.n_operational_space
        self.K_x = self._process_gain(K_x, n_op, "K_x")
        self.D_x = self._process_gain(D_x, n_op, "D_x")

        # Check that the actuation matrix is square and invertible
        self._check_assumptions()

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

    def _check_assumptions(self) -> None:
        """
        Check that assumptions (a) and (b) hold.

        Raises:
            ValueError: If either (a) or (b) is not met.
        """
        robot = self.operational_space_dynamics.robot
        scenario, shape, _ = analyze_actuation_matrix(robot)
        n_dof, n_actuators = shape
        n_op = self.operational_space_dynamics.n_operational_space

        if scenario == ActuationScenario.FULL_ACTUATION:
            raise ValueError(
                f"SynergisticController requires a rectangular actuation matrix "
                f"(under-actuation with m < n). Got {n_actuators} actuators and "
                f"{n_dof} DOFs. For fully- or over-actuated systems, consider "
                f"alternative control strategies."
            )

        if n_op != n_actuators:
            raise ValueError(
                f"SynergisticController requires the operational and actuation spaces to "
                f"have the same dimension (o = m). Got m = {n_actuators}, o = {n_op}."
            )

    def __call__(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the synergistic control action.

        This method implements the synergistic control law to track
        the desired trajectory.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            tau_control: The control input, shape (num_actuators,).
            control_state_dot: None (this controller is stateless).
        """
        t = system_state.t
        y = system_state.y

        # Extract configuration and velocity
        q, qd = jnp.split(y, 2)

        # Get operational space dynamics instance and robot
        osd = self.operational_space_dynamics

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
        J = osd.jacobian(q)

        # Current operational space FULL pose (for error computation) and task-selected velocity
        x_full = osd.operational_space_poses(q)
        xd = J @ qd  # Task-selected velocity

        # Compute configuration-space quantities
        A = self.robot.actuation_matrix(q)
        M_inv = jnp.linalg.inv(self.robot.inertia_matrix(q))

        # Dynamically-consisted synergistic projector
        P_AM = jnp.linalg.inv(J @ M_inv @ A) @ J @ M_inv

        # PD control in operational space
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
        tau_control = P_AM @ tau_pd

        return tau_control, None
