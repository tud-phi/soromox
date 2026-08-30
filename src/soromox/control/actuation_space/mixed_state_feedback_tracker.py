__all__ = ["MixedStateFeedbackTracker"]

import warnings
from typing import Any

from jax import Array

from soromox.control.actuation_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.systems.system_state import SystemState


class MixedStateFeedbackTracker(PIDController):
    """
    Mixed state feedback trajectory tracker in actuation space for soft robots.

    This controller extends the actuation-space PIDController by adding a model-based
    feedforward term that uses a hybrid evaluation strategy in actuation space: the
    dynamical matrices (inertia, Coriolis, gravitational, damping) are evaluated at
    the *current* state, while the desired acceleration and velocity are used for
    computing the inertial, Coriolis, and damping forces. Only the elastic forces
    are evaluated at the *desired* configuration.

    This mixed strategy often provides better control performance with good model
    knowledge compared to pure feedforward compensation since it uses the actual
    state for the configuration-dependent matrices while still providing feedforward
    action through the desired velocities and accelerations.

    Important:
        This controller requires the reference trajectory to be provided in
        configuration space (`reference_in_configuration_space=True`). The
        configuration-space trajectory is needed to evaluate elastic forces at q_des,
        since the inverse mapping from actuation space y back to configuration
        space q is generally nonlinear and not trivially invertible.

    Control Law
    -----------

    The control law is:

        tau_model_y = M_y(q) @ ydd_des + eta_y(q, qd) @ yd_des
                      + G_y(q) + tau_el_y(q_des) + D_y(q) @ yd_des

        tau = tau_model_y[:n_a] + tau_feedback

    where:
        - M_y(q) is the actuation-space inertia matrix at current configuration
        - eta_y(q, qd) is the actuation-space Coriolis matrix at current state
        - G_y(q) is the actuation-space gravitational force at current configuration
        - tau_el_y(q_des) is the actuation-space elastic force at desired configuration
        - D_y(q) is the actuation-space damping matrix at current configuration
        - yd_des, ydd_des are the desired velocity and acceleration in actuation space
        - n_a is the number of actuators
        - tau is the actuator input (shape: n_a)
        - tau_feedback is the PID feedback term from the parent class

    Only the first n_a rows of the actuation-space forces contribute to the control
    input, corresponding to the actuated coordinates.

    Underactuation Warning
    ----------------------

    In underactuated settings (n_u > 0), trajectory tracking is an open research
    problem. The unactuated coordinates evolve according to the zero dynamics and
    cannot be independently controlled. The reference trajectory for the unactuated
    coordinates must be dynamically feasible, meaning it must correspond to a valid
    solution of the system's equations of motion. If the reference trajectory is not
    feasible, tracking performance may be significantly degraded.

    Coriolis Matrix Warning
    -----------------------

    For the theoretical stability guarantees to hold, the Coriolis matrix must be
    derived using the Christoffel symbols of the first kind, ensuring that the
    matrix N = dM/dt - 2*C is skew-symmetric. This property is essential for
    passivity-based stability proofs.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory in actuation space.
        reference_trajectory_config_space: The original configuration-space trajectory
            (required for evaluating elastic forces at q_des).
        actuation_space_dynamics: The ActuationSpaceDynamics instance for
            coordinate transformations.
        pid_control: The PIDControl instance containing gains and saturation.

    References:
        Kelly, R., & Carelli, R. (1996). A class of nonlinear PD-type controllers
        for robot manipulators. Journal of Robotic Systems, 13(12), 793-802.

        Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
        Model-based dynamic feedback control of a planar soft robot: trajectory
        tracking and interaction with the environment. The International Journal
        of Robotics Research, 39.

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
        Initialize the actuation-space mixed state feedback trajectory tracker.

        Args:
            actuation_space_dynamics: The ActuationSpaceDynamics instance that
                provides coordinate transformations and contains the robot.
            reference_trajectory: The desired trajectory to track. Must be in
                configuration space (set `reference_in_configuration_space=True`).
                The configuration-space trajectory is required to evaluate elastic
                forces at q_des.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The gains should
                be sized for the actuated coordinates (n_a x n_a or n_a).
            reference_in_configuration_space: Must be True for this controller.
                The configuration-space trajectory is required to evaluate elastic
                forces at the desired configuration q_des.

        Raises:
            ValueError: If `reference_in_configuration_space=False`. This controller
                requires the configuration-space trajectory to evaluate elastic forces.
        """
        if not reference_in_configuration_space:
            raise ValueError(
                "MixedStateFeedbackTracker requires "
                "`reference_in_configuration_space=True` because the model-based "
                "term needs to evaluate elastic forces at q_des. The inverse mapping "
                "from actuation space y to configuration space q is generally "
                "nonlinear and not trivially invertible."
            )

        super().__init__(
            actuation_space_dynamics=actuation_space_dynamics,
            reference_trajectory=reference_trajectory,
            pid_control=pid_control,
            reference_in_configuration_space=reference_in_configuration_space,
        )
        self._warn_underactuation_tracking()
        self._warn_coriolis_christoffel()

    def _warn_underactuation_tracking(self) -> None:
        """Emit a warning about trajectory tracking in underactuated settings."""
        asd = self.actuation_space_dynamics
        if asd.n_unactuated > 0:
            warnings.warn(
                f"MixedStateFeedbackTracker: The system is underactuated "
                f"({asd.n_actuated} actuators, {asd.n_unactuated} unactuated DOFs). "
                f"Trajectory tracking in underactuated settings is an open research "
                f"problem. The reference trajectory for unactuated coordinates must be "
                f"dynamically feasible (consistent with the zero dynamics). "
                f"Consider using regulation (setpoint control) for more robust behavior.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_coriolis_christoffel(self) -> None:
        """Emit a warning about Coriolis matrix derivation requirements."""
        warnings.warn(
            "MixedStateFeedbackTracker: For theoretical stability guarantees to hold, "
            "the Coriolis matrix C(q, qd) must be derived using Christoffel symbols "
            "of the first kind, ensuring that N = dM/dt - 2*C is skew-symmetric. "
            "If the robot's coriolis_matrix method uses a different derivation "
            "(e.g., direct Jacobian time derivatives), stability guarantees may not apply.",
            UserWarning,
            stacklevel=3,
        )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term for trajectory tracking.

        This method computes the control input using a mixed evaluation strategy
        in actuation space:
        - Dynamical matrices (M_y, eta_y, G_y, D_y) are evaluated at current state
        - The desired acceleration and velocity are used for inertial/Coriolis/damping forces
        - Elastic forces are evaluated at the desired configuration q_des

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            tau_model: The model-based control input, shape (num_actuators,).
            control_state_dot: None (this term is stateless).
        """
        t = system_state.t
        y = system_state.y
        asd, q, qd = self._controller_dynamics_and_state(
            y, self.actuation_space_dynamics
        )
        n_a = asd.n_actuated

        # Get desired configuration from the configuration-space reference trajectory
        ref_config = self.reference_trajectory_config_space
        assert ref_config is not None, "Configuration-space reference required"
        assert ref_config.x_des_fn is not None
        q_des = ref_config.x_des_fn(t)

        # Get desired trajectory in actuation space at current time
        assert self.reference_trajectory.xd_des_fn is not None
        assert self.reference_trajectory.xdd_des_fn is not None
        yd_des = self.reference_trajectory.xd_des_fn(t)
        ydd_des = self.reference_trajectory.xdd_des_fn(t)

        # Compute actuation-space dynamics matrices at CURRENT state
        M_y = asd.inertia_matrix(q)
        eta_y = asd.coriolis_matrix(q, qd)  # Coriolis at current state
        G_y = asd.gravitational_force(q)
        D_y = asd.damping_matrix(q)

        # Compute elastic force at DESIRED configuration in actuation space
        tau_el_y = asd.elastic_force(q_des)

        # Compute the model-based actuation-space torque
        # Inertial, Coriolis, and damping use current matrices with desired velocity/acceleration
        # Elastic force is evaluated at desired configuration
        tau_model_y = M_y @ ydd_des + eta_y @ yd_des + G_y + tau_el_y + D_y @ yd_des

        # Extract only the first n_a (actuated) rows as the control input
        tau_model = tau_model_y[:n_a]

        return tau_model, None
