__all__ = ["FeedforwardCompensationTracker"]

from typing import Any

import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import emit_actuation_warnings
from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class FeedforwardCompensationTracker(PIDController):
    """
    Feedforward compensation trajectory tracker for soft robots.

    This controller extends the PIDController by adding a model-based feedforward
    term that evaluates the complete inverse dynamics at the *desired* trajectory.
    It computes the inertial forces, Coriolis forces, gravitational forces, elastic
    forces, and damping forces all evaluated at the desired configuration, desired
    velocity, and desired acceleration.

    The control law is:
        tau_model = B(q_des) @ qdd_des + C(q_des, qd_des) @ qd_des + G(q_des) + tau_el(q_des) + D(q_des) @ qd_des
        u_model = inv(A(q)) @ tau_model  (or pinv if A is not square)
        u_control = u_model + u_feedback

    where:
        - B(q_des) is the inertia matrix at the desired configuration
        - C(q_des, qd_des) is the Coriolis matrix at the desired configuration and velocity
        - G(q_des) is the gravitational force at the desired configuration
        - tau_el(q_des) is the elastic force at the desired configuration
        - D(q_des) is the damping matrix at the desired configuration
        - qd_des is the desired velocity
        - qdd_des is the desired acceleration
        - A(q) is the state-dependent actuation matrix of the robot
        - inv(A(q)) is the inverse of A(q) if square, otherwise the pseudo-inverse
        - u_feedback is the PID feedback term from the parent class

    Note:
        This controller is suitable for trajectory tracking but may have limited
        robustness to model uncertainties since all dynamics are evaluated at the
        desired trajectory rather than the actual state.

    Warning:
        For full theoretical guarantees, the actuation matrix should be
        configuration-independent. If A(q) is configuration-dependent, consider
        using the ComputedTorqueTracker (with full feedback linearization) or
        actuation-space controllers.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Della Santina, C., Duriez, C., & Rus, D. (2023). Model-based control of
        soft robots: A survey of the state of the art and open challenges.
        IEEE Control Systems Magazine, 43(3), 30-65.

        Kelly, R., & Salgado, R. (1994). PD control with computed feedforward of
        robot manipulators: A design procedure. IEEE Transactions on Robotics
        and Automation, 10(4), 566-571.
    """

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the feedforward compensation trajectory tracker.

        Args:
            robot: The soft robot system to be controlled.
                Must have `actuation_matrix(q)`, `dynamics_terms(q, qd)`,
                `elastic_force(q)`, and `damping_matrix(q)` methods.
            reference_trajectory: The desired trajectory to track.
                Must provide x_des_fn (desired configuration), xd_des_fn
                (desired velocity), and xdd_des_fn (desired acceleration)
                as functions of time.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function.

        Raises:
            ValueError: If the actuation matrix is singular (for full actuation)
                or has insufficient rank (for overactuation).
        """
        # Skip parent's _check_pid_actuation by calling grandparent's __init__
        self.robot = robot
        self.reference_trajectory = reference_trajectory
        self.pid_control = pid_control
        self._check_actuation()

    def _check_actuation(self) -> None:
        """Check actuation matrix properties and emit appropriate warnings."""
        emit_actuation_warnings(
            controller_name="FeedforwardCompensationTracker",
            robot=self.robot,
            is_regulator=False,
            is_computed_torque=False,
            stacklevel=4,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for trajectory tracking.

        This method computes the control input required to track the desired
        trajectory by evaluating the complete inverse dynamics at the desired
        configuration, velocity, and acceleration. The generalized torques are
        mapped to actuator inputs using the inverse of the actuation matrix
        (or pseudo-inverse if A is not square).

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

        # Get current configuration (for actuation matrix)
        q, _ = jnp.split(y, 2)

        # Get desired trajectory at current time
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        assert self.reference_trajectory.xdd_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)
        qd_des = self.reference_trajectory.xd_des_fn(t)
        qdd_des = self.reference_trajectory.xdd_des_fn(t)

        # Compute the coupled dynamics terms at the desired trajectory. The
        # returned Coriolis term is already contracted with qd_des.
        B_des, Cqd_des, G_des = self.robot.dynamics_terms(q_des, qd_des)
        tau_el_des = self.robot.elastic_force(q_des)
        D_des = self.robot.damping_matrix(q_des)

        # Compute inverse dynamics at desired trajectory
        # tau = B @ qdd + C @ qd + G + tau_el + D @ qd
        tau_model = (
            B_des @ qdd_des + Cqd_des + G_des + tau_el_des + D_des @ qd_des
        )

        # Get the actuation matrix at current configuration
        A = self.robot.actuation_matrix(q)

        # Compute actuator input using inverse (or pseudo-inverse) of actuation matrix
        if A.shape[0] == A.shape[1]:
            u_model = jnp.linalg.inv(A) @ tau_model
        else:
            u_model = jnp.linalg.pinv(A) @ tau_model

        return u_model, None
