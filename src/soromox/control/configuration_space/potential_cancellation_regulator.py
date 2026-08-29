__all__ = ["PotentialCancellationRegulator"]

from typing import Any

import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import emit_actuation_warnings
from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class PotentialCancellationRegulator(PIDController):
    """
    Potential cancellation regulator for soft robots.

    This controller extends the PIDController by adding a model-based feedforward
    term that cancels the gravitational and elastic forces at the current
    configuration. This provides real-time cancellation of potential energy forces.

    This regulator is closely related to PD+ regulators in rigid robotics, which
    augment PD feedback with gravity compensation evaluated at the current
    configuration. The PD+ approach was shown to be globally asymptotically stable
    for rigid robot manipulators by Paden & Panja (1988) and Kelly & Carelli (1996).

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
        tau_model = G(q) + tau_el(q)
        u_model = inv(A(q)) @ tau_model  (or pinv if A is not square)
        u_control = u_model + u_feedback

    where:
        - G(q) is the gravitational force at the current configuration
        - tau_el(q) is the elastic force at the current configuration
        - A(q) is the state-dependent actuation matrix of the robot
        - inv(A(q)) is the inverse of A(q) if square, otherwise the pseudo-inverse
        - u_feedback is the PID feedback term from the parent class

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Paden, B., & Panja, R. (1988). Globally asymptotically stable 'PD+'
        controller for robot manipulators. International Journal of Control,
        47(6), 1697-1712.

        Kelly, R., & Carelli, R. (1996). A class of nonlinear PD-type controllers
        for robot manipulators. Journal of Robotic Systems, 13(12), 793-802.

        Patterson, Z. J., Sologuren, E., Della Santina, C., & Rus, D. (2024).
        Design and Control of Modular Soft-Rigid Hybrid Manipulators with
        Self-Contact. IEEE Transactions on Robotics.

        Pustina, P. (2024). Analysis and control of the underactuation in
        continuum soft robots: a kinematic independent approach. PhD Thesis,
        Sapienza University of Rome.

        Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
        into Learned Models for Control. Dissertation, Delft University of Technology.
        https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da

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
        Initialize the potential cancellation regulator.

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
            controller_name="PotentialCancellationRegulator",
            robot=self.robot,
            is_regulator=True,
            is_computed_torque=False,
            stacklevel=4,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for potential cancellation.

        This method computes the control input required to compensate for the
        gravitational and elastic forces at the current configuration. The
        generalized torques are mapped to actuator inputs using the inverse
        of the actuation matrix (or pseudo-inverse if A is not square).

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            u_model: The model-based control input, shape (num_actuators,).
            control_state_dot: None (this term is stateless).
        """
        y = system_state.y

        # Get current configuration
        robot, q, _ = self._controller_model_and_state(y)

        # Compute gravitational force at current configuration
        tau_gravity = robot.gravitational_force(q)

        # Compute elastic force at current configuration
        tau_elastic = robot.elastic_force(q)

        # Total generalized torque for potential cancellation
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
