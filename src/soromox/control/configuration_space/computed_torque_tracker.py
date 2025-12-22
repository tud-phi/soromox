__all__ = ["ComputedTorqueTracker"]

import warnings
from typing import Any, Optional, Tuple

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


class ComputedTorqueTracker(ClosedFormModelBasedController):
    """
    Computed torque trajectory tracker for soft robots.

    This controller implements the computed torque control method, a model-based
    nonlinear control technique that achieves exact input-output linearization of
    robot dynamics through feedback. The method cancels all static and dynamic
    forces at the current robot motion (configuration and velocity) and applies
    a desired acceleration computed from PID feedback on the tracking error.

    The computed torque method, also known as inverse dynamics control, was first
    developed for rigid robot manipulators. It achieves exact linearization of the
    nonlinear robot dynamics by computing the control torque required to produce
    a desired acceleration, given the current state. This transforms the closed-loop
    dynamics into a linear double integrator, enabling straightforward trajectory
    tracking through linear feedback (e.g., PID control).

    The control law is:
        qdd_ref = qdd_des + PID(e, ed, integral_error)
        tau = M(q) @ qdd_ref + C(q, qd) @ qd + G(q) + tau_el(q) + D(q) @ qd
        u = inv(A(q)) @ tau  (or pinv for overactuated systems)

    where:
        - M(q) is the inertia (mass) matrix at the current configuration
        - C(q, qd) is the Coriolis matrix at the current configuration and velocity
        - G(q) is the gravitational force at the current configuration
        - tau_el(q) is the elastic force at the current configuration
        - D(q) is the damping matrix at the current configuration
        - qdd_des is the desired acceleration from the reference trajectory
        - e = q_des - q is the configuration error
        - ed = qd_des - qd is the velocity error
        - qdd_ref is the reference acceleration that includes PID feedback
        - A(q) is the state-dependent actuation matrix of the robot
        - inv(A(q)) is the inverse of A(q) (or pseudo-inverse for overactuation)

    With perfect model knowledge, the closed-loop dynamics become:
        qdd = qdd_des + PID(e, ed, integral_error)

    which represents a linear error system. Appropriate choice of PID gains then
    ensures asymptotic stability and desired transient response.

    Note:
        This controller is the recommended choice for fully actuated systems with
        configuration-dependent actuation matrices, as it achieves full feedback
        linearization. Other configuration-space controllers may have reduced
        theoretical guarantees when the actuation matrix depends on configuration.

        The controller requires the actuation matrix to have sufficient rank:
        - For full actuation: A must be square and invertible.
        - For overactuation: A must have rank at least equal to the number of DOFs.
        - Underactuation is not supported as the system cannot be fully linearized.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Slotine, J.-J. E., & Li, W. (1987). On the Adaptive Control of Robot
        Manipulators. The International Journal of Robotics Research, 6(3), 49-59.
        https://doi.org/10.1177/027836498700600303

        Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2020). Robot Modeling
        and Control (2nd ed.). Wiley.
    """

    pid_control: PIDControl
    _actuation_scenario: ActuationScenario

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the computed torque trajectory tracker.

        Args:
            robot: The soft robot system to be controlled.
                Must have `actuation_matrix(q)`, `inertia_matrix(q)`,
                `coriolis_matrix(q, qd)`, `gravitational_force(q)`,
                `elastic_force(q)`, and `damping_matrix(q)` methods.
            reference_trajectory: The desired trajectory to track.
                Must provide x_des_fn (desired configuration), xd_des_fn
                (desired velocity), and xdd_des_fn (desired acceleration)
                as functions of time.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The PID output
                is used to compute the reference acceleration for the inverse
                dynamics computation.

        Raises:
            ValueError: If the system is underactuated, if the actuation matrix
                is singular (for full actuation), or if the rank is insufficient
                (for overactuation).
        """
        self.robot = robot
        self.reference_trajectory = reference_trajectory
        self.pid_control = pid_control
        self._actuation_scenario = self._check_actuation()

    def _check_actuation(self) -> ActuationScenario:
        """
        Check actuation matrix properties for computed torque control.

        Returns:
            The detected actuation scenario.

        Raises:
            ValueError: If the system is underactuated, singular (for full actuation),
                or has insufficient rank (for overactuation).
        """
        scenario, shape, rank = analyze_actuation_matrix(self.robot)
        n_dof, n_actuators = shape

        if scenario == ActuationScenario.UNDERACTUATION:
            raise ValueError(
                f"ComputedTorqueTracker requires at least as many actuators as DOFs, "
                f"but the system is underactuated ({n_actuators} actuators < {n_dof} DOFs). "
                f"Computed torque control cannot fully linearize underactuated systems. "
                f"Consider using actuation-space controllers or regulators designed for "
                f"underactuated systems. For deeper analysis, see: Borja (2022), "
                f"Pustina (2022), Della Santina (2023), and Pustina (2025)."
            )

        if scenario == ActuationScenario.FULL_ACTUATION:
            if rank < n_dof:
                raise ValueError(
                    f"ComputedTorqueTracker requires an invertible actuation matrix, but "
                    f"the actuation matrix is singular (rank {rank} < {n_dof}). "
                    f"The actuation matrix must be full rank (positive-definite) for "
                    f"computed torque control."
                )

        if scenario == ActuationScenario.OVERACTUATION:
            if rank < n_dof:
                raise ValueError(
                    f"ComputedTorqueTracker: The actuation matrix has rank {rank}, which is "
                    f"less than the number of DOFs ({n_dof}). Even with overactuation "
                    f"({n_actuators} actuators > {n_dof} DOFs), the actuation matrix "
                    f"must have rank at least {n_dof} to control all DOFs."
                )
            warnings.warn(
                f"ComputedTorqueTracker: The system is overactuated ({n_actuators} "
                f"actuators > {n_dof} DOFs). Overactuation has been rarely investigated "
                f"in literature. The pseudo-inverse will be used to compute actuator inputs, "
                f"which minimizes the actuator effort while achieving the desired generalized "
                f"forces. This should generally work well as the residual can be minimized "
                f"to (almost) zero.",
                UserWarning,
                stacklevel=3,
            )

        return scenario

    def model_based_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the model-based feedforward control term for computed torque control.

        This method computes the inverse dynamics torque required to cancel all
        static and dynamic forces at the current motion and achieve the desired
        acceleration from the reference trajectory.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            u_model: The model-based control input, shape (num_actuators,).
            control_state_dot: None (this term is stateless).
        """
        t = system_state.t
        y = system_state.y

        # Get current configuration and velocity
        q, qd = jnp.split(y, 2)

        # Get desired trajectory at current time
        assert self.reference_trajectory.xdd_des_fn is not None
        qdd_des = self.reference_trajectory.xdd_des_fn(t)

        # Compute dynamics matrices at current configuration and velocity
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        G = self.robot.gravitational_force(q)
        tau_el = self.robot.elastic_force(q)
        D = self.robot.damping_matrix(q)

        # Compute the inverse dynamics: cancel all forces and apply desired acceleration
        # tau = M(q) @ qdd_des + C(q, qd) @ qd + G(q) + tau_el(q) + D(q) @ qd
        tau_model = M @ qdd_des + C @ qd + G + tau_el + D @ qd

        # Get the actuation matrix at current configuration
        A = self.robot.actuation_matrix(q)

        # Compute actuator input using inverse (or pseudo-inverse for overactuation)
        if self._actuation_scenario == ActuationScenario.OVERACTUATION:
            u_model = jnp.linalg.pinv(A) @ tau_model
        else:
            u_model = jnp.linalg.inv(A) @ tau_model

        return u_model, None

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[PIDControllerState]]:
        """
        Compute the PID-based feedback control term for computed torque control.

        This method computes the feedback torque from the PID controller acting
        on the tracking error. The PID output represents an acceleration correction
        that is mapped through the inertia matrix and actuation matrix to obtain
        the actuator inputs.

        The feedback term implements:
            qdd_fb = PID(e, ed, integral_error)
            tau_fb = M(q) @ qdd_fb
            u_fb = inv(A(q)) @ tau_fb

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
        q, qd = jnp.split(y, 2)

        # Get reference trajectory at current time
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)
        qd_des = self.reference_trajectory.xd_des_fn(t)

        # Compute tracking errors
        e = q_des - q  # Configuration error
        ed = qd_des - qd  # Velocity error

        # Get integral error from control state (or zeros if not tracking)
        control_state: Optional[PIDControllerState] = system_state.control_state
        if control_state is None:
            integral_error = jnp.zeros_like(q)
        else:
            integral_error = control_state.integral_error

        # Compute PID output: this gives an acceleration correction qdd_fb
        qdd_fb, integral_error_dot = self.pid_control(e, ed, integral_error)

        # Get inertia matrix at current configuration
        M = self.robot.inertia_matrix(q)

        # Compute feedback torque: tau_fb = M(q) @ qdd_fb
        tau_fb = M @ qdd_fb

        # Get the actuation matrix and compute actuator input
        A = self.robot.actuation_matrix(q)
        if self._actuation_scenario == ActuationScenario.OVERACTUATION:
            u_feedback = jnp.linalg.pinv(A) @ tau_fb
        else:
            u_feedback = jnp.linalg.inv(A) @ tau_fb

        # Build control state derivative
        if control_state is not None:
            control_state_dot = PIDControllerState(integral_error=integral_error_dot)
        else:
            control_state_dot = None

        return u_feedback, control_state_dot

