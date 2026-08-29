__all__ = ["OperationalSpaceBaseController"]

from abc import ABC

import equinox as eqx
from jax import Array

from soromox.control.base_controller import BaseController
from soromox.coordinate_transformations.operational_space_dynamics import (
    OperationalSpaceDynamics,
)


class OperationalSpaceBaseController(BaseController, ABC):
    """
    Abstract base class for operational-space controllers.

    Operational-space controllers compute actuation inputs based on the current
    system state and a reference trajectory defined in operational (task) space.
    The controller uses an OperationalSpaceDynamics instance to transform between
    configuration space and operational space.

    The reference trajectory should be specified in operational space coordinates
    (e.g., end-effector positions/orientations), and the controller computes the
    corresponding actuator inputs to track this trajectory.

    Attributes:
        robot: The soft robot system to be controlled (inherited from BaseController,
            should be set to operational_space_dynamics.robot).
        reference_trajectory: The desired trajectory to track in operational space.
        operational_space_dynamics: The OperationalSpaceDynamics instance that
            defines the task space and provides the necessary transformations
            (Jacobians, dynamically-consistent pseudo-inverse, etc.).

    Note:
        Subclasses should set `self.robot = operational_space_dynamics.robot` in
        their `__init__` method to ensure the robot attribute is properly initialized.

    References:
        Khatib, O. (1987). A unified approach for motion and force control of robot
        manipulators: The operational space formulation. IEEE Journal on Robotics
        and Automation, 3(1), 43-53.

        Natale, C. (2003). Interaction control of robot manipulators: Six-degrees-of-freedom
        tasks. Springer Science & Business Media.
    """

    operational_space_dynamics: OperationalSpaceDynamics

    def _controller_dynamics_and_state(
        self, y: Array
    ) -> tuple[OperationalSpaceDynamics, Array, Array]:
        """Return current-pose fixed dynamics and internal controller state.

        Args:
            y: Complete system state with trailing dimension
                ``robot.state_size``.

        Returns:
            A tuple containing an operational-space dynamics view mounted at
            the current base pose, internal coordinates, and internal
            velocities.
        """
        robot, q_internal, qd_internal = super()._controller_model_and_state(y)
        dynamics = self.operational_space_dynamics
        if dynamics.robot is not robot:
            dynamics = eqx.tree_at(lambda value: value.robot, dynamics, robot)
        return dynamics, q_internal, qd_internal
