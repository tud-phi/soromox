__all__ = ["SynergisticController"]


import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import (
    ActuationScenario,
    analyze_actuation_matrix,
)
from soromox.control.operational_space.base_controller import (
    OperationalSpaceBaseController,
)
from soromox.control.pid_control import (
    PIDControl,
    PIDControllerState,
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

        tau = P_AM @ J(q).T @ (Kp e_x + Ki integral_e_x + Kd e_dx)

        P_AM = (J(q) M(q)^{-1} A(q))^{-1} J(q) M(q)^{-1}
        e_x = x_des - x
        e_dx = xd_des - xd
        integral_e_x = integral of e_x dt

    where:
        - A(q) is the actuation matrix
        - M(q) is the inertia matrix
        - J(q) is the operational space Jacobian
        - Kp is the operational space proportional gain matrix
        - Ki is the operational space integral gain matrix
        - Kd is the operational space derivative gain matrix
        - e_x is the operational space pose error (geometric error for orientation)
        - e_dx is the operational space velocity error
        - x is the current operational space pose
        - x_des is the desired operational space pose
        - xd is the current operational space velocity
        - xd_des is the desired operational space velocity

    Assumptions:
        (a) Under-actuation: The actuation space has lower dimensionality than
            the configuration space (m < n). If full actuation, consider using,
            for example, the impedance control tracker.
        (b) The operational space has equal dimensionality of the actuation
            space (o = m)
        (c) The matrix J(q) M^{-1}(q) A(q) ∈ ℝ^{m×m} is full-rank

    Attributes:
        operational_space_dynamics: The OperationalSpaceDynamics instance.
        reference_trajectory: The desired trajectory in operational space.
        pid_control: The task-space PIDControl instance containing the gains and saturation.

    References:
        Della Santina, C., Pallottino, L., Rus, D., & Bicchi, A. (2019). Exact
        task execution in highly under-actuated soft limbs: an operational
        space based approach. IEEE Robotics and Automation Letters, 4(3),
        2508-2515.
    """

    pid_control: PIDControl

    def __init__(
        self,
        operational_space_dynamics: OperationalSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
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
            pid_control: A task-space PIDControl instance containing the control gains
                (Kp, Ki, Kd of dimensions o) and optional saturation function.

        Raises:
            ValueError: If either assumption (a) or (b) is not met.
        """
        self.operational_space_dynamics = operational_space_dynamics
        self.reference_trajectory = reference_trajectory
        self.robot = operational_space_dynamics.robot
        self.pid_control = pid_control

        # Check that the actuation matrix is square and invertible
        self._check_assumptions()

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

    def __call__(
        self, system_state: SystemState
    ) -> tuple[Array, PIDControllerState | None]:
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
            control_state_dot: PIDControllerState derivative if tracking integral
                error, otherwise None.
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

        # PID control in operational space
        # tau_pid = J^T @ (Kp @ e_x + Ki integral(e_x) + Kd @ ed_x)
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

        # Get integral error from control state (or zeros if not tracking)
        control_state: PIDControllerState | None = system_state.control_state
        if control_state is None:
            control_state = PIDControllerState.zero(osd.n_operational_space)

        integral_error = control_state.integral_error
        tau_pid, integral_error_dot = self.pid_control(e_x, ed_x, integral_error)

        # Build control state derivative
        if control_state is not None:
            control_state_dot = PIDControllerState(integral_error=integral_error_dot)
        else:
            control_state_dot = None

        # Total generalized torque
        tau_control = P_AM @ J.T @ tau_pid

        return tau_control, control_state_dot
