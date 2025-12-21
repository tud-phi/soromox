__all__ = ["GravityCancellationRegulator"]

from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.dynamical_system import DynamicalSystem
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
        robot: The dynamical system (robot) to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
        Model-based dynamic feedback control of a planar soft robot: trajectory
        tracking and interaction with the environment. The International Journal
        of Robotics Research, 39.
    """

    def __init__(
        self,
        robot: DynamicalSystem,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the gravity cancellation regulator.

        Args:
            robot: The dynamical system (robot) to be controlled.
                Must have `actuation_matrix(q)`, `gravitational_force(q)`,
                and `elastic_force(q)` methods.
            reference_trajectory: The desired trajectory to track.
                Must provide x_des_fn (desired configuration) as a function of time.
            pid_control: A PIDControl instance containing the control gains
                (Kp, Ki, Kd) and optional saturation function.
        """
        super().__init__(robot, reference_trajectory, pid_control)

    def model_based_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
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
        q, _ = jnp.split(y, 2)

        # Get desired configuration at current time
        assert self.reference_trajectory.x_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)

        # Compute gravitational force at current configuration
        tau_gravity = self.robot.gravitational_force(q)

        # Compute elastic force at desired configuration
        tau_elastic = self.robot.elastic_force(q_des)

        # Total generalized torque for gravity cancellation
        tau_model = tau_gravity + tau_elastic

        # Get the actuation matrix at current configuration
        A = self.robot.actuation_matrix(q)

        # Compute actuator input using inverse (or pseudo-inverse) of actuation matrix
        # Use standard inverse if A is square, otherwise use pseudo-inverse
        if A.shape[0] == A.shape[1]:
            u_model = jnp.linalg.inv(A) @ tau_model
        else:
            u_model = jnp.linalg.pinv(A) @ tau_model

        return u_model, None

