__all__ = ["MixedStateFeedbackTracker"]

import warnings
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import emit_actuation_warnings
from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class MixedStateFeedbackTracker(PIDController):
    """
    Mixed state feedback trajectory tracker for soft robots.

    This controller extends the PIDController by adding a model-based feedforward
    term that uses a hybrid evaluation strategy: the dynamical matrices (inertia,
    Coriolis, gravitational, damping) are evaluated at the *current* state (q, qd),
    while the desired acceleration and velocity are used for computing the inertial,
    Coriolis, and damping forces. Only the elastic forces are evaluated at the
    *desired* configuration.

    This mixed strategy often provides with good model knowledge better control performance compared to pure feedforward
    compensation since it uses the actual state for the configuration-dependent
    matrices while still providing feedforward action through the desired
    velocities and accelerations.

    The control law is:
        tau_model = B(q) @ qdd_des + C(q, qd) @ qd_des + G(q) + tau_el(q_des) + D(q) @ qd_des
        u_model = inv(A(q)) @ tau_model  (or pinv if A is not square)
        u_control = u_model + u_feedback

    where:
        - B(q) is the inertia matrix at the current configuration
        - C(q, qd) is the Coriolis matrix at the current configuration and current velocity
        - G(q) is the gravitational force at the current configuration
        - tau_el(q_des) is the elastic force at the desired configuration
        - D(q) is the damping matrix at the current configuration
        - qd_des is the desired velocity
        - qdd_des is the desired acceleration
        - A(q) is the state-dependent actuation matrix of the robot
        - inv(A(q)) is the inverse of A(q) if square, otherwise the pseudo-inverse
        - u_feedback is the PID feedback term from the parent class

    Note:
        The key difference from FeedforwardCompensationTracker is that the inertia,
        Coriolis, gravitational, and damping matrices are evaluated at the current
        state (q, qd), providing state feedback through the dynamical matrices.
        Only the elastic forces use the desired configuration to provide the
        correct equilibrium point. The Coriolis term C(q, qd) @ qd_des and the
        damping term D(q) @ qd_des mix current state evaluation with desired velocity.

    Warning:
        For full theoretical guarantees, the actuation matrix should be
        configuration-independent. If A(q) is configuration-dependent, consider
        using the ComputedTorqueTracker (with full feedback linearization) or
        actuation-space controllers.

        For the theoretical stability guarantees to hold, the Coriolis matrix
        C(q, qd) must be derived using the Christoffel symbols of the first kind,
        ensuring that the matrix N = dB/dt - 2*C is skew-symmetric. This property
        is essential for the passivity-based stability proofs. If the Coriolis
        matrix is computed using other methods (e.g., direct Jacobian time
        derivatives), the skew-symmetry property may not hold and the stability
        guarantees may be invalidated.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
        pid_control: The PIDControl instance containing the gains and saturation.

    References:
        Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
        Model-based dynamic feedback control of a planar soft robot: trajectory
        tracking and interaction with the environment. The International Journal
        of Robotics Research, 39.

        Kelly, R., & Carelli, R. (1996). A class of nonlinear PD-type controllers
        for robot manipulators. Journal of Robotic Systems, 13(12), 793-802.
    """

    def __init__(
        self,
        robot: SoftRobot,
        reference_trajectory: ReferenceTrajectory,
        pid_control: PIDControl,
    ):
        """
        Initialize the mixed state feedback trajectory tracker.

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
        self._warn_coriolis_christoffel()

    def _check_actuation(self) -> None:
        """Check actuation matrix properties and emit appropriate warnings."""
        emit_actuation_warnings(
            controller_name="MixedStateFeedbackTracker",
            robot=self.robot,
            is_regulator=False,
            is_computed_torque=False,
            stacklevel=4,
        )

    def _warn_coriolis_christoffel(self) -> None:
        """Emit a warning about Coriolis matrix derivation requirements."""
        warnings.warn(
            "MixedStateFeedbackTracker: For theoretical stability guarantees to hold, "
            "the Coriolis matrix C(q, qd) must be derived using Christoffel symbols "
            "of the first kind, ensuring that N = dB/dt - 2*C is skew-symmetric. "
            "If the robot's coriolis_matrix method uses a different derivation "
            "(e.g., direct Jacobian time derivatives), stability guarantees may not apply.",
            UserWarning,
            stacklevel=3,
        )

    def model_based_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the model-based feedforward control term for trajectory tracking.

        This method computes the control input using a mixed evaluation strategy:
        - Dynamical matrices (B, C, G, D) are evaluated at the current state (q, qd)
        - The desired acceleration and velocity are used for inertial/Coriolis/damping forces
        - Elastic forces are evaluated at the desired configuration

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
        assert self.reference_trajectory.x_des_fn is not None
        assert self.reference_trajectory.xd_des_fn is not None
        assert self.reference_trajectory.xdd_des_fn is not None
        q_des = self.reference_trajectory.x_des_fn(t)
        qd_des = self.reference_trajectory.xd_des_fn(t)
        qdd_des = self.reference_trajectory.xdd_des_fn(t)

        # Compute dynamics matrices at CURRENT state (q, qd)
        B = self.robot.inertia_matrix(q)
        # Coriolis matrix at current configuration and CURRENT velocity
        C = self.robot.coriolis_matrix(q, qd)
        # Gravitational force at current configuration
        G = self.robot.gravitational_force(q)
        # Damping matrix at current configuration
        D = self.robot.damping_matrix(q)

        # Elastic force at DESIRED configuration
        tau_el_des = self.robot.elastic_force(q_des)

        # Compute the model-based torque
        # Inertial, Coriolis, and damping use current matrices with desired velocity/acceleration
        # Elastic force is evaluated at desired configuration
        tau_model = (
            B @ qdd_des
            + C @ qd_des
            + G
            + tau_el_des
            + D @ qd_des
        )

        # Get the actuation matrix at current configuration
        A = self.robot.actuation_matrix(q)

        # Compute actuator input using inverse (or pseudo-inverse) of actuation matrix
        if A.shape[0] == A.shape[1]:
            u_model = jnp.linalg.inv(A) @ tau_model
        else:
            u_model = jnp.linalg.pinv(A) @ tau_model

        return u_model, None

