__all__ = ["PIDController"]

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.control.closed_form_model_based_controller import (
    ClosedFormModelBasedController,
)
from soromox.control.operational_space.base_controller import (
    OperationalSpaceBaseController,
)
from soromox.control.pid_control import PIDControl, PIDControllerState
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.operational_space_dynamics import (
    OperationalSpaceDynamics,
)
from soromox.systems.system_state import SystemState


class PIDController(OperationalSpaceBaseController, ClosedFormModelBasedController):
    """
    PID controller in operational space for soft robots.

    This controller implements PID control in the robot's operational space, computing
    the force in operational space required to reach x_des, based on tracking errors
    in the selected operational coordinates only.

    Control Law
    -----------

    The control law is:

        f_x = Kp @ e_x + Ki @ integral_error_x + Kd @ ed_x

    where:
        - e_x = x_des - x is the operational coordinate error (shape: n_o)
        - ed_x = xd_des - xd is the operational velocity error (shape: n_o)
        - integral_error_x = integral of sat(e_x) over time
        - sat() is an optional saturation function for anti-windup
        - f_x is the operational space force (shape: n_o)

    Usage
    -----

    Attributes:
        robot: The soft robot system to be controlled (inherited from BaseController,
            should be set to operational_space_dynamics.robot).
        operational_space_dynamics: The OperationalSpaceDynamics instance.
        reference_trajectory: The desired trajectory in operational space.
        pid_control: The PIDControl instance containing gains and saturation.
            Note: Gains should be sized for the operational coordinates (n_o).
    """

    pid_control: PIDControl

    def __init__(
        self,
        operational_space_dynamics: OperationalSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the operational-space PID controller.

        Args:
            operational_space_dynamics: The OperationalSpaceDynamics instance that
                provides coordinate transformations and contains the robot.
            reference_trajectory: The desired trajectory in operational space.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The gains should
                be sized for the actuated coordinates (n_o x n_o or n_o).
        """
        self.operational_space_dynamics = operational_space_dynamics
        self.robot = operational_space_dynamics.robot
        self.pid_control = pid_control
        self.reference_trajectory = reference_trajectory

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> tuple[Array, PIDControllerState | None]:
        """
        Compute the PID feedback control term in operational space.

        This method computes the PID control action based on the tracking error
        in the selected operational coordinates only (selected via task-selector).

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)
                - control_state: PIDControllerState containing integral error
                    of shape (n_o,), or None if not using integral control.

        Returns:
            f_x: The operational space forcing action, shape (n_o,).
            control_state_dot: Time derivative of the PIDControllerState, or None
                if no control state is being tracked.
        """
        t = system_state.t
        y = system_state.y

        osd, q, qd = self._controller_dynamics_and_state(
            y, self.operational_space_dynamics
        )

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

        # Compute operational space Jacobian
        J = osd.jacobian(q)

        # Current operational space FULL pose (for error computation) and task-selected velocity
        x_full = osd.operational_space_poses(q)
        xd = J @ qd  # Task-selected velocity

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
            integral_error = jnp.zeros(osd.n_operational_space)
        else:
            integral_error = control_state.integral_error

        # Compute PID output
        f_x, integral_error_dot = self.pid_control(e_x, ed_x, integral_error)

        # Build control state derivative
        if control_state is not None:
            control_state_dot = PIDControllerState(integral_error=integral_error_dot)
        else:
            control_state_dot = None

        return f_x, control_state_dot

    def update_gains(self, gains: dict[str, Array]) -> "PIDController":
        """
        This function updates the gains of the PID controller.

        Args:
            gains (dict[str, Array]): proportional, integral, and derivative gains

        Returns:
            updated_self (PIDController): self object with updated gains
        """
        updated_pid_control = self.pid_control.update_gains(gains)

        updated_self = eqx.tree_at(lambda x: x.pid_control, self, updated_pid_control)

        return updated_self
