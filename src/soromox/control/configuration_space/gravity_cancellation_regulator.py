__all__ = ["GravityCancellationRegulator"]

from typing import Any

import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import emit_actuation_warnings
from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class GravityCancellationRegulator(PIDController):
    """
    Gravity cancellation regulator for soft robots.

    This controller extends the PIDController by adding a model-based feedforward
    term that compensates for the gravitational forces at the current configuration
    and elastic forces at the desired configuration. Unlike the PotentialShapingRegulator,
    this controller evaluates gravity at the current state rather than the desired state,
    providing real-time gravity compensation.

    Note:
        This regulator is specialized for setpoint regulation or quasi-static
        trajectories as it does not consider the dynamical (e.g., inertial, Coriolis, damping, etc.)
        forces of the soft robot.

    Warning:
        For full theoretical guarantees, the actuation matrix should be
        configuration-independent. If A(q) is configuration-dependent, consider
        using the ComputedTorqueTracker (with full feedback linearization) or
        actuation-space controllers.

    The control law is:
        tau_model = G(q) + tau_el(q_des)
        u_model = inv(A(q)) @ tau_model  (or pinv if A is not square)
        u_control = u_model + u_feedback

    where:
        - G(q) is the gravitational force at the current configuration
        - tau_el(q_des) is the elastic force at the desired configuration
        - A(q) is the state-dependent actuation matrix of the robot
        - inv(A(q)) is the inverse of A(q) if square, otherwise the pseudo-inverse
        - u_feedback is the PID feedback term from the parent class

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
        Model-based dynamic feedback control of a planar soft robot: trajectory
        tracking and interaction with the environment. The International Journal
        of Robotics Research, 39.

        Borja, P., Della Santina, C., & Albu-Schäffer, A. (2022). Energy-shaping
        control of soft continuum manipulators with in-plane disturbances.
        The International Journal of Robotics Research, 41(1), 62-81.

        Pustina, P., Della Santina, C., & De Luca, A. (2025). Feedback regulation
        of elastically decoupled underactuated soft robots. IEEE Transactions
        on Robotics.
    """

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the gravity cancellation regulator.

        Args:
            robot: The soft robot system to be controlled.
                Must have `actuation_matrix(q)`, `gravitational_force(q)`,
                and `elastic_force(q)` methods.
            reference_trajectory: The desired trajectory to track.
                Must provide x_des_fn (desired configuration) as a function of time.
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
            controller_name="GravityCancellationRegulator",
            robot=self.robot,
            is_regulator=True,
            is_computed_torque=False,
            stacklevel=4,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for gravity cancellation.

        This method computes the control input required to compensate for the
        gravitational forces at the current configuration and elastic forces at
        the desired configuration. The generalized torques are mapped to actuator
        inputs using the inverse of the actuation matrix (or pseudo-inverse if
        A is not square).

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

        # Get current configuration
        robot, q, _ = self._controller_model_and_state(y)

        # Get desired configuration at current time
        assert self.reference_trajectory.x_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)

        # Compute gravitational force at current configuration
        tau_gravity = robot.gravitational_force(q)

        # Compute elastic force at desired configuration
        tau_elastic = robot.elastic_force(q_des)

        # Total generalized torque for gravity cancellation
        tau_model = tau_gravity + tau_elastic

        # Get the actuation matrix at current configuration
        A = robot.actuation_matrix(q)

        # Compute actuator input using inverse (or pseudo-inverse) of actuation matrix
        # Use standard inverse if A is square, otherwise use pseudo-inverse
        if A.shape[0] == A.shape[1]:
            u_model = jnp.linalg.inv(A) @ tau_model
        else:
            u_model = jnp.linalg.pinv(A) @ tau_model

        return u_model, None
