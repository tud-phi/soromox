__all__ = ["PIDController"]

import warnings

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import (
    ActuationScenario,
    analyze_actuation_matrix,
)
from soromox.control.closed_form_model_based_controller import (
    ClosedFormModelBasedController,
)
from soromox.control.pid_control import PIDControl, PIDControllerState
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class PIDController(ClosedFormModelBasedController):
    """
    PID controller in configuration space for soft robots.

    This controller implements PID control in the robot's configuration space,
    computing control inputs based on tracking errors in generalized coordinates.

    The control law is:
        tau = Kp @ e + Ki @ integral_error + Kd @ ed
        u = A(q).T @ tau

    where:
        - e = q_des - q is the configuration error
        - ed = qd_des - qd is the velocity error
        - integral_error = integral of sat(e) over time
        - sat() is an optional saturation function for anti-windup
        - A(q) is the state-dependent actuation matrix of the robot
        - tau is the generalized force in configuration space
        - u is the actuator input

    The integral error dynamics are:
        d(integral_error)/dt = sat(e)

    where sat() can be the identity (no saturation), tanh, or a custom function.
    The saturation function provides anti-windup protection by limiting the
    growth of the integral term when errors are large.

    Note:
        This base PID controller uses u = A.T @ tau, which naturally handles
        all actuation scenarios (full, over, and underactuation). However,
        subclasses that use the actuation matrix inverse require additional
        considerations for non-square actuation matrices.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.
    """

    pid_control: PIDControl

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the PID controller.

        Args:
            robot: The dynamical system (robot) to be controlled.
                Must have an `actuation_matrix(q)` method.
            reference_trajectory: The desired trajectory to track.
                Must provide x_des_fn (desired configuration) and
                xd_des_fn (desired velocity) as functions of time.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function.
        """
        self.robot = robot
        self.reference_trajectory = reference_trajectory
        self.pid_control = pid_control
        self._check_pid_actuation()

    def _check_pid_actuation(self) -> None:
        """
        Check actuation matrix properties for the base PID controller.

        The base PID controller uses u = A.T @ tau, which works with all
        actuation scenarios, but we emit informational warnings for
        non-standard cases.
        """
        try:
            scenario, shape, rank = analyze_actuation_matrix(self.robot)
            n_dof, n_actuators = shape

            if scenario == ActuationScenario.FULL_ACTUATION:
                if rank < n_dof:
                    warnings.warn(
                        f"PIDController: The actuation matrix is singular "
                        f"(rank {rank} < {n_dof}). This may lead to reduced "
                        f"controllability in some directions.",
                        UserWarning,
                        stacklevel=3,
                    )
            elif scenario == ActuationScenario.UNDERACTUATION:
                warnings.warn(
                    f"PIDController: The system is underactuated ({n_actuators} "
                    f"actuators < {n_dof} DOFs). The base PID controller uses "
                    f"u = A.T @ tau which provides feedback in actuator space, but "
                    f"cannot fully control all DOFs. Consider using actuation-space "
                    f"controllers for better underactuated control.",
                    UserWarning,
                    stacklevel=3,
                )
        except Exception:
            # If we can't analyze, continue silently
            pass

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> tuple[Array, PIDControllerState | None]:
        """
        Compute the PID feedback control term.

        This method computes the PID control action based on the tracking error
        between the current state and the reference trajectory.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)
                - control_state: PIDControllerState containing integral error,
                    or None if not using integral control.

        Returns:
            u_feedback: The feedback control input, shape (num_actuators,).
            control_state_dot: Time derivative of the PIDControllerState, or None
                if no control state is being tracked.
        """
        t = system_state.t
        y = system_state.y

        # Split state into configuration and velocity
        robot, q, qd = self._controller_model_and_state(y)

        # Get reference trajectory at current time
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)
        qd_des = self.reference_trajectory.xd_des_fn(t)

        # Compute tracking errors
        e = q_des - q  # Configuration error
        ed = qd_des - qd  # Velocity error

        # Get integral error from control state (or zeros if not tracking)
        control_state: PIDControllerState | None = system_state.control_state
        if control_state is None:
            integral_error = jnp.zeros_like(q)
        else:
            integral_error = control_state.integral_error

        # Compute PID output in configuration space
        tau, integral_error_dot = self.pid_control(e, ed, integral_error)

        # Get the actuation matrix and compute actuator input
        A = robot.actuation_matrix(q)
        u_feedback = A.T @ tau

        # Build control state derivative
        if control_state is not None:
            control_state_dot = PIDControllerState(integral_error=integral_error_dot)
        else:
            control_state_dot = None

        return u_feedback, control_state_dot

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
