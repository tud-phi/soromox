__all__ = ["PIDController"]

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_space.base_controller import ActuationSpaceBaseController
from soromox.control.closed_form_model_based_controller import (
    ClosedFormModelBasedController,
)
from soromox.control.pid_control import PIDControl, PIDControllerState
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.systems.system_state import SystemState


class PIDController(ActuationSpaceBaseController, ClosedFormModelBasedController):
    """
    PID controller in actuation space for soft robots.

    This controller implements PID control in the robot's actuation space, computing
    control inputs based on tracking errors in the actuated coordinates only. By
    working in actuation space, the controller benefits from the collocated form
    where the actuation matrix is an identity (for the actuated coordinates).

    Key Features
    ------------

    1. **Direct control of actuated coordinates**: The PID operates only on the first
       n_a (actuated) coordinates in actuation space. These coordinates directly
       correspond to actuator inputs via the identity actuation structure.

    2. **No actuation matrix inversion**: Unlike configuration-space controllers,
       there is no need to invert or pseudo-invert the actuation matrix A(q) in the control law.
       The control input tau is directly the actuator force/torque.

    3. **Automatic reference trajectory conversion**: If a configuration-space
       reference trajectory is provided, it is automatically converted to actuation
       space using the coordinate transformations.

    Control Law
    -----------

    The control law is:

        tau = Kp @ e_a + Ki @ integral_error_a + Kd @ ed_a

    where:
        - e_a = y_a_des - y_a is the actuated coordinate error (shape: n_a)
        - ed_a = yd_a_des - yd_a is the actuated velocity error (shape: n_a)
        - integral_error_a = integral of sat(e_a) over time
        - sat() is an optional saturation function for anti-windup
        - tau is the actuator input (shape: n_a)

    The subscript "_a" denotes the actuated portion (first n_a entries) of the
    actuation-space coordinates.

    Zero Dynamics Assumption
    ------------------------

    In any underactuated control setting—regardless of whether the controller
    operates in configuration space, operational space, or actuation space—a key
    assumption is that the zero dynamics (dynamics of unactuated coordinates when
    actuated coordinates are perfectly tracked) are stable. We emphasize this here
    because actuation-space controllers are particularly well-suited for underactuated
    robots. For soft robots with elastic restoring forces and damping, this assumption
    is typically satisfied. If the zero dynamics are unstable, the unactuated
    coordinates may diverge even with perfect tracking on the actuated coordinates.

    Usage
    -----

    The controller can be initialized with either:

    1. A configuration-space reference trajectory (will be automatically converted)
    2. An actuation-space reference trajectory (used directly)

    Set `reference_in_configuration_space=True` (default) if providing a
    configuration-space trajectory, or `False` if providing an actuation-space
    trajectory directly.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory in actuation space.
        reference_trajectory_config_space: The original configuration-space trajectory,
            or None if trajectory was provided in actuation space. Used by subclasses
            for model-based control terms that need to evaluate dynamics at q_des.
        actuation_space_dynamics: The ActuationSpaceDynamics instance for
            coordinate transformations.
        pid_control: The PIDControl instance containing gains and saturation.
            Note: Gains should be sized for the actuated coordinates (n_a).
    """

    pid_control: PIDControl

    def __init__(
        self,
        actuation_space_dynamics: ActuationSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
        reference_in_configuration_space: bool = True,
    ):
        """
        Initialize the actuation-space PID controller.

        Args:
            actuation_space_dynamics: The ActuationSpaceDynamics instance that
                provides coordinate transformations and contains the robot.
            reference_trajectory: The desired trajectory to track. If
                `reference_in_configuration_space=True`, this should be a
                configuration-space trajectory which will be converted to
                actuation space. Otherwise, it should already be in actuation space.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The gains should
                be sized for the actuated coordinates (n_a x n_a or n_a).
            reference_in_configuration_space: If True (default), the reference
                trajectory is in configuration space and will be converted to
                actuation space. If False, the reference trajectory is already
                in actuation space.

        Note:
            When `reference_in_configuration_space=True`, the original configuration-
            space trajectory is stored in `reference_trajectory_config_space` for use
            by model-based controllers that need to evaluate dynamics at q_des. This
            is necessary because the mapping from actuation space y back to configuration
            space q is generally nonlinear and not trivially invertible (e.g., for
            PCS with configuration-dependent tendon routing).
        """
        self.actuation_space_dynamics = actuation_space_dynamics
        self.robot = actuation_space_dynamics.robot
        self.pid_control = pid_control

        # Convert reference trajectory if needed, but also store the original
        # configuration-space trajectory for model-based terms
        if reference_in_configuration_space:
            # Store original config-space trajectory for model-based evaluation
            self.reference_trajectory_config_space = reference_trajectory
            # Convert to actuation space for PID feedback
            self.reference_trajectory = (
                self.actuation_space_dynamics.convert_reference_trajectory(
                    reference_trajectory
                )
            )
        else:
            # No config-space trajectory available
            self.reference_trajectory_config_space = None
            self.reference_trajectory = reference_trajectory

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> tuple[Array, PIDControllerState | None]:
        """
        Compute the PID feedback control term in actuation space.

        This method computes the PID control action based on the tracking error
        in the actuated coordinates only (first n_a coordinates in actuation space).

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)
                - control_state: PIDControllerState containing integral error
                    of shape (n_a,), or None if not using integral control.

        Returns:
            tau: The actuator control input, shape (num_actuators,).
            control_state_dot: Time derivative of the PIDControllerState, or None
                if no control state is being tracked.
        """
        t = system_state.t
        y = system_state.y
        asd = self.actuation_space_dynamics
        n_a = asd.n_actuated

        # Split state into configuration and velocity
        q, qd = jnp.split(y, 2)

        # Transform current state to actuation space
        y_act = asd.actuated_unactuated_coordinates(q)
        J = asd.jacobian(q)
        yd_act = J @ qd

        # Extract actuated portion (first n_a entries)
        y_a = y_act[:n_a]
        yd_a = yd_act[:n_a]

        # Get reference trajectory at current time (in actuation space)
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        y_des = self.reference_trajectory.x_des_fn(t)
        yd_des = self.reference_trajectory.xd_des_fn(t)

        # Extract actuated portion of reference
        y_a_des = y_des[:n_a]
        yd_a_des = yd_des[:n_a]

        # Compute tracking errors on actuated coordinates
        e_a = y_a_des - y_a  # Actuated coordinate error
        ed_a = yd_a_des - yd_a  # Actuated velocity error

        # Get integral error from control state (or zeros if not tracking)
        control_state: PIDControllerState | None = system_state.control_state
        if control_state is None:
            integral_error_a = jnp.zeros(n_a)
        else:
            integral_error_a = control_state.integral_error

        # Compute PID output - this is directly the actuator input tau
        # Since the actuation matrix in actuation space is [I; 0], no inversion needed
        tau, integral_error_dot_a = self.pid_control(e_a, ed_a, integral_error_a)

        # Build control state derivative
        if control_state is not None:
            control_state_dot = PIDControllerState(integral_error=integral_error_dot_a)
        else:
            control_state_dot = None

        return tau, control_state_dot

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
