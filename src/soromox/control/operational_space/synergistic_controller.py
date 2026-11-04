__all__ = ["SynergisticController"]


import jax.numpy as jnp
from jax import Array

from soromox.control.actuation_matrix_utils import (
    ActuationScenario,
    analyze_actuation_matrix,
)
from soromox.control.operational_space.pid_controller import (
    PIDController,
)
from soromox.control.pid_control import (
    PIDControllerState,
)
from soromox.systems.system_state import SystemState


class SynergisticController(PIDController):
    """
    Synergistic controller for soft robots.

    The control law is:

        tau = P_AM @ J(q).T @ (Kp e_x + Ki integral_e_x + Kd e_dx)

        P_AM = (J(q) M(q)^{-1} A(q))^{-1} J(q) M(q)^{-1}
        e_x = x_des - x
        e_dx = xd_des - xd
        integral_e_x = integral of e_x dt

    where:
        - A(q) is the actuation matrix
        - M(q) is the inertia matrix
        - J(q) is the operational space Jacobian
        - Kp is the operational space proportional gain matrix
        - Ki is the operational space integral gain matrix
        - Kd is the operational space derivative gain matrix
        - e_x is the operational space pose error (geometric error for orientation)
        - e_dx is the operational space velocity error
        - x is the current operational space pose
        - x_des is the desired operational space pose
        - xd is the current operational space velocity
        - xd_des is the desired operational space velocity

    Assumptions:
        (A) Under-actuation: The actuation space has lower dimensionality than
            the configuration space (m < n). If full actuation, consider using,
            for example, the impedance control tracker.
        (B) The operational space has equal dimensionality of the actuation
            space (o = m).
        (C) The matrix J(q) M^{-1}(q) A(q) ∈ ℝ^{m×m} is full-rank.

    Attributes:
        robot: The soft robot system to be controlled (inherited from BaseController,
            should be set to operational_space_dynamics.robot).
        operational_space_dynamics: The OperationalSpaceDynamics instance.
        reference_trajectory: The desired trajectory in operational space.
        pid_control: The PIDControl instance containing gains and saturation.
            Note: Gains should be sized for the operational coordinates (n_o).

    References:
        Della Santina, C., Pallottino, L., Rus, D., & Bicchi, A. (2019). Exact
        task execution in highly under-actuated soft limbs: an operational
        space based approach. IEEE Robotics and Automation Letters, 4(3),
        2508-2515.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the synergistic controller.

        Raises:
            ValueError: If either assumption (A) or (B) is not met.
        """
        super().__init__(*args, **kwargs)

        # Check wheter the assumptions are met
        self._check_assumptions()

    def _check_assumptions(self) -> None:
        """
        Check that assumptions (A) and (B) hold.

        Raises:
            ValueError: If either (A) or (B) is not met.
        """
        robot = self.operational_space_dynamics.robot
        scenario, shape, _ = analyze_actuation_matrix(robot)
        n_dof, n_actuators = shape
        n_op = self.operational_space_dynamics.n_operational_space

        if scenario == ActuationScenario.FULL_ACTUATION:
            raise ValueError(
                f"SynergisticController requires a rectangular actuation matrix "
                f"(under-actuation with m < n). Got {n_actuators} actuators and "
                f"{n_dof} DOFs. For fully- or over-actuated systems, consider "
                f"alternative control strategies."
            )

        if n_op != n_actuators:
            raise ValueError(
                f"SynergisticController requires the operational and actuation spaces to "
                f"have the same dimension (o = m). Got m = {n_actuators}, o = {n_op}."
            )

    def __call__(
        self, system_state: SystemState
    ) -> tuple[Array, PIDControllerState | None]:
        """
        Compute the synergistic control action.

        This method implements the synergistic control law to track
        the desired trajectory.

        Args:
            system_state: The current state of the system, containing:
                - t: Current simulation time
                - y: Robot state vector [q, qd] (configuration and velocity)

        Returns:
            tau_control: The control input, shape (num_actuators,).
            control_state_dot: PIDControllerState derivative if tracking integral
                error, otherwise None.
        """
        # Inertia and actuation matrices of the mounted fixed-base robot, with the
        # inertia matrix from the fused dynamics terms
        robot, q, qd = self._controller_robot_and_state(system_state.y)
        M, _, _ = robot.dynamics_terms(q, qd)
        A = robot.actuation_matrix(q)

        # Compute PID output together with the task Jacobian, evaluated once
        tau_pid, control_state_dot, J = self._feedback_and_task_jacobian(system_state)

        # Dynamically-consistent synergistic projection
        # tau = (J M^-1 A)^-1 J M^-1 J^T tau_pid, with one solve of the inertia
        # system for both M^-1 A and M^-1 J^T tau_pid instead of explicit inverses
        X = robot._solve_inertia(
            M, jnp.concatenate([A, (J.T @ tau_pid)[:, None]], axis=1)
        )
        tau_control = jnp.linalg.solve(J @ X[:, :-1], J @ X[:, -1])

        return tau_control, control_state_dot
