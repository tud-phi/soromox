__all__ = ["PotentialCompensationRegulator"]

from typing import Any

from jax import Array

from soromox.control.actuation_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.systems.system_state import SystemState


class PotentialCompensationRegulator(PIDController):
    """
    Potential compensation regulator in actuation space for soft robots.

    This controller extends the actuation-space PIDController by adding a model-based
    feedforward term that compensates for the gravitational and elastic forces at the
    desired configuration, all evaluated in actuation space. This is known as
    "potential shaping" as it effectively reshapes the potential energy landscape
    of the system to have a minimum at the desired configuration.

    Note:
        This regulator is specialized for setpoint regulation or quasi-static
        trajectories as it does not consider the dynamical (e.g., inertial,
        Coriolis, damping, etc.) forces of the soft robot.

    Important:
        This controller requires the reference trajectory to be provided in
        configuration space (`reference_in_configuration_space=True`). The
        configuration-space trajectory is needed to evaluate forces at q_des,
        since the inverse mapping from actuation space y back to configuration
        space q is generally nonlinear and not trivially invertible.

    Control Law
    -----------

    The control law is:

        tau_model_y = G_y(q_des) + tau_el_y(q_des)

        tau = tau_model_y[:n_a] + tau_feedback

    where:
        - G_y(q_des) is the actuation-space gravitational force at desired configuration
        - tau_el_y(q_des) is the actuation-space elastic force at desired configuration
        - n_a is the number of actuators
        - tau is the actuator input (shape: n_a)
        - tau_feedback is the PID feedback term from the parent class

    Only the first n_a rows of the actuation-space forces contribute to the control
    input, corresponding to the actuated coordinates.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory in actuation space.
        reference_trajectory_config_space: The original configuration-space trajectory
            (required for evaluating forces at q_des).
        actuation_space_dynamics: The ActuationSpaceDynamics instance for
            coordinate transformations.
        pid_control: The PIDControl instance containing gains and saturation.

    References:
        Kelly, R., & Salgado, R. (1994). PD control with computed feedforward of
        robot manipulators: A design procedure. IEEE Transactions on Robotics
        and Automation, 10(4), 566-571.

        Borja, P., Della Santina, C., & Albu-Schäffer, A. (2022). Energy-shaping
        control of soft continuum manipulators with in-plane disturbances.
        The International Journal of Robotics Research, 41(1), 62-81.

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
        Initialize the actuation-space potential compensation regulator.

        Args:
            actuation_space_dynamics: The ActuationSpaceDynamics instance that
                provides coordinate transformations and contains the robot.
            reference_trajectory: The desired trajectory to track. Must be in
                configuration space (set `reference_in_configuration_space=True`).
                The configuration-space trajectory is required to evaluate forces
                at q_des.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The gains should
                be sized for the actuated coordinates (n_a x n_a or n_a).
            reference_in_configuration_space: Must be True for this controller.
                The configuration-space trajectory is required to evaluate forces
                at the desired configuration q_des.

        Raises:
            ValueError: If `reference_in_configuration_space=False`. This controller
                requires the configuration-space trajectory to evaluate forces.
        """
        if not reference_in_configuration_space:
            raise ValueError(
                "PotentialCompensationRegulator requires "
                "`reference_in_configuration_space=True` because the model-based "
                "term needs to evaluate forces at q_des. The inverse mapping "
                "from actuation space y to configuration space q is generally "
                "nonlinear and not trivially invertible."
            )

        super().__init__(
            actuation_space_dynamics=actuation_space_dynamics,
            reference_trajectory=reference_trajectory,
            pid_control=pid_control,
            reference_in_configuration_space=reference_in_configuration_space,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for potential compensation.

        This method computes the control input required to compensate for the
        gravitational and elastic forces at the desired configuration, all in
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
        t = system_state.t
        asd, _, _ = self._controller_dynamics_and_state(
            system_state.y, self.actuation_space_dynamics
        )
        n_a = asd.n_actuated

        # Get desired configuration from the configuration-space reference trajectory
        ref_config = self.reference_trajectory_config_space
        assert ref_config is not None, "Configuration-space reference required"
        assert ref_config.x_des_fn is not None
        q_des = ref_config.x_des_fn(t)

        # Compute gravitational force at DESIRED configuration in actuation space
        G_y = asd.gravitational_force(q_des)

        # Compute elastic force at DESIRED configuration in actuation space
        tau_el_y = asd.elastic_force(q_des)

        # Total actuation-space force for potential compensation
        tau_model_y = G_y + tau_el_y

        # Extract only the first n_a (actuated) rows as the control input
        tau_model = tau_model_y[:n_a]

        return tau_model, None
