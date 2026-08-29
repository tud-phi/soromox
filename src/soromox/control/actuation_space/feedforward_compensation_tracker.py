__all__ = ["FeedforwardCompensationTracker"]

import warnings
from typing import Any

from jax import Array

from soromox.control.actuation_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.systems.system_state import SystemState


class FeedforwardCompensationTracker(PIDController):
    """
    Feedforward compensation trajectory tracker in actuation space for soft robots.

    This controller extends the actuation-space PIDController by adding a model-based
    feedforward term that evaluates the complete inverse dynamics at the *desired*
    configuration. It computes the inertial forces, Coriolis forces, gravitational
    forces, elastic forces, and damping forces all evaluated at the desired
    configuration q_des, then transforms them to actuation space.

    By working in actuation space, this controller benefits from the collocated form
    where the actuation matrix becomes an identity for the actuated coordinates.
    The model-based term is computed by:
    1. Evaluating the actuation-space dynamics at q_des (from the configuration-space
       reference trajectory)
    2. Using the actuation-space velocities yd_des and accelerations ydd_des
    3. Extracting only the first n_a (actuated) rows for the control input

    Important:
        This controller requires the reference trajectory to be provided in
        configuration space (`reference_in_configuration_space=True`). The
        configuration-space trajectory is needed to evaluate dynamics at q_des,
        since the inverse mapping from actuation space y back to configuration
        space q is generally nonlinear and not trivially invertible.

    Control Law
    -----------

    The control law is:

        tau_model_y = M_y(q_des) @ ydd_des + eta_y(q_des, qd_des) @ yd_des
                      + G_y(q_des) + tau_el_y(q_des) + D_y(q_des) @ yd_des

        tau = tau_model_y[:n_a] + tau_feedback

    where:
        - M_y is the actuation-space inertia matrix evaluated at q_des
        - eta_y is the actuation-space Coriolis matrix evaluated at (q_des, qd_des)
        - G_y is the actuation-space gravitational force evaluated at q_des
        - tau_el_y is the actuation-space elastic force evaluated at q_des
        - D_y is the actuation-space damping matrix evaluated at q_des
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

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory in actuation space.
        reference_trajectory_config_space: The original configuration-space trajectory
            (required for evaluating dynamics at q_des).
        actuation_space_dynamics: The ActuationSpaceDynamics instance for
            coordinate transformations.
        pid_control: The PIDControl instance containing gains and saturation.

    References:
        Della Santina, C., Duriez, C., & Rus, D. (2023). Model-based control of
        soft robots: A survey of the state of the art and open challenges.
        IEEE Control Systems Magazine, 43(3), 30-65.

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
        Initialize the actuation-space feedforward compensation trajectory tracker.

        Args:
            actuation_space_dynamics: The ActuationSpaceDynamics instance that
                provides coordinate transformations and contains the robot.
            reference_trajectory: The desired trajectory to track. Must be in
                configuration space (set `reference_in_configuration_space=True`).
                The configuration-space trajectory is required to evaluate dynamics
                at q_des.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function. The gains should
                be sized for the actuated coordinates (n_a x n_a or n_a).
            reference_in_configuration_space: Must be True for this controller.
                The configuration-space trajectory is required to evaluate dynamics
                at the desired configuration q_des.

        Raises:
            ValueError: If `reference_in_configuration_space=False`. This controller
                requires the configuration-space trajectory to evaluate dynamics.
        """
        if not reference_in_configuration_space:
            raise ValueError(
                "FeedforwardCompensationTracker requires "
                "`reference_in_configuration_space=True` because the model-based "
                "term needs to evaluate dynamics at q_des. The inverse mapping from "
                "actuation space y to configuration space q is generally nonlinear "
                "and not trivially invertible."
            )

        super().__init__(
            actuation_space_dynamics=actuation_space_dynamics,
            reference_trajectory=reference_trajectory,
            pid_control=pid_control,
            reference_in_configuration_space=reference_in_configuration_space,
        )
        self._warn_underactuation_tracking()

    def _warn_underactuation_tracking(self) -> None:
        """Emit a warning about trajectory tracking in underactuated settings."""
        asd = self.actuation_space_dynamics
        if asd.n_unactuated > 0:
            warnings.warn(
                f"FeedforwardCompensationTracker: The system is underactuated "
                f"({asd.n_actuated} actuators, {asd.n_unactuated} unactuated DOFs). "
                f"Trajectory tracking in underactuated settings is an open research "
                f"problem. The reference trajectory for unactuated coordinates must be "
                f"dynamically feasible (consistent with the zero dynamics). "
                f"Consider using regulation (setpoint control) for more robust behavior.",
                UserWarning,
                stacklevel=3,
            )

    def model_based_term(self, system_state: SystemState) -> tuple[Array, Any | None]:
        """
        Compute the model-based feedforward control term in actuation space.

        This method computes the control input required to track the desired
        trajectory by evaluating the complete inverse dynamics at the desired
        configuration q_des. Only the first n_a (actuated) rows of the actuation-
        space forces contribute to the output.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            tau_model: The model-based control input, shape (num_actuators,).
            control_state_dot: None (this term is stateless).
        """
        t = system_state.t
        asd, _, _ = self._controller_dynamics_and_state(system_state.y)
        n_a = asd.n_actuated

        # Get desired configuration-space trajectory at current time
        # This is the original trajectory before conversion to actuation space
        ref_config = self.reference_trajectory_config_space
        assert ref_config is not None, "Configuration-space reference required"
        assert ref_config.x_des_fn is not None
        assert ref_config.xd_des_fn is not None
        assert ref_config.xdd_des_fn is not None
        q_des = ref_config.x_des_fn(t)
        qd_des = ref_config.xd_des_fn(t)

        # Get desired trajectory in actuation space at current time
        assert self.reference_trajectory.xd_des_fn is not None
        assert self.reference_trajectory.xdd_des_fn is not None
        yd_des = self.reference_trajectory.xd_des_fn(t)
        ydd_des = self.reference_trajectory.xdd_des_fn(t)

        # Compute the coupled actuation-space dynamics at the desired state.
        # The returned Coriolis term is already contracted with yd_des because
        # yd_des = J(q_des) @ qd_des for the converted reference trajectory.
        M_y_des, Cyd_des, G_y_des = asd.dynamics_terms(q_des, qd_des)
        tau_el_y_des = asd.elastic_force(q_des)
        D_y_des = asd.damping_matrix(q_des)

        # Compute the full actuation-space inverse dynamics at desired trajectory
        # Using yd_des and ydd_des (actuation-space velocities/accelerations)
        tau_model_y = (
            M_y_des @ ydd_des + Cyd_des + G_y_des + tau_el_y_des + D_y_des @ yd_des
        )

        # Extract only the first n_a (actuated) rows as the control input
        # In actuation space, A_y = [[I], [0]], so tau directly affects actuated coords
        tau_model = tau_model_y[:n_a]

        return tau_model, None
