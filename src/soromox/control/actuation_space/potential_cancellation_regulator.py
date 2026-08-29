__all__ = ["PotentialCancellationRegulator"]

from typing import Any

from jax import Array

from soromox.control.actuation_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.systems.system_state import SystemState


class PotentialCancellationRegulator(PIDController):
    """
    Potential cancellation regulator in actuation space for soft robots.

    This controller extends the actuation-space PIDController by adding a model-based
    feedforward term that cancels the gravitational and elastic forces at the current
    configuration, all evaluated in actuation space. This provides real-time
    cancellation of potential energy forces.

    This regulator is closely related to PD+ regulators in rigid robotics, which
    augment PD feedback with gravity compensation evaluated at the current
    configuration. The PD+ approach was shown to be globally asymptotically stable
    for rigid robot manipulators by Paden & Panja (1988) and Kelly & Carelli (1996).

    Note:
        This regulator is specialized for setpoint regulation or quasi-static
        trajectories as it does not consider the dynamical (e.g., inertial,
        Coriolis, damping, etc.) forces of the soft robot.

    Control Law
    -----------

    The control law is:

        tau_model_y = G_y(y) + tau_el_y(y)

        tau = tau_model_y[:n_a] + tau_feedback

    where:
        - G_y(y) is the actuation-space gravitational force at current configuration
        - tau_el_y(y) is the actuation-space elastic force at current configuration
        - n_a is the number of actuators
        - tau is the actuator input (shape: n_a)
        - tau_feedback is the PID feedback term from the parent class

    Only the first n_a rows of the actuation-space forces contribute to the control
    input, corresponding to the actuated coordinates.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory in actuation space.
        actuation_space_dynamics: The ActuationSpaceDynamics instance for
            coordinate transformations.
        pid_control: The PIDControl instance containing gains and saturation.

    References:
        Paden, B., & Panja, R. (1988). Globally asymptotically stable 'PD+'
        controller for robot manipulators. International Journal of Control,
        47(6), 1697-1712.

        Kelly, R., & Carelli, R. (1996). A class of nonlinear PD-type controllers
        for robot manipulators. Journal of Robotic Systems, 13(12), 793-802.

        Pustina, P. (2025). Analysis and control of the underactuation in continuum
        soft robots: a kinematic independent approach.
        PhD Thesis, Sapienza University of Rome.

        Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
        into Learned Models for Control. Dissertation, Delft University of Technology.
        https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da
    """

    def __init__(
        self,
        actuation_space_dynamics: ActuationSpaceDynamics,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
        reference_in_configuration_space: bool = True,
    ):
        """
        Initialize the actuation-space potential cancellation regulator.

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
        """
        super().__init__(
            actuation_space_dynamics=actuation_space_dynamics,
            reference_trajectory=reference_trajectory,
            pid_control=pid_control,
            reference_in_configuration_space=reference_in_configuration_space,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for potential cancellation.

        This method computes the control input required to compensate for the
        gravitational and elastic forces at the current configuration, all in
        actuation space. Only the first n_a (actuated) rows of the actuation-space
        forces contribute to the output.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            tau_model: The model-based control input, shape (num_actuators,).
            control_state_dot: None (this term is stateless).
        """
        y = system_state.y
        asd, q, _ = self._controller_dynamics_and_state(y)
        n_a = asd.n_actuated

        # Compute gravitational force at current configuration in actuation space
        G_y = asd.gravitational_force(q)

        # Compute elastic force at current configuration in actuation space
        tau_el_y = asd.elastic_force(q)

        # Total actuation-space force for potential cancellation
        tau_model_y = G_y + tau_el_y

        # Extract only the first n_a (actuated) rows as the control input
        tau_model = tau_model_y[:n_a]

        return tau_model, None
