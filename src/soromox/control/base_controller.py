__all__ = ["BaseController"]

from abc import ABC, abstractmethod
import equinox as eqx
from typing import Any, Optional, Tuple

from jax import Array

from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.dynamical_system import DynamicalSystem
from soromox.systems.system_state import SystemState


class BaseController(eqx.Module, ABC):
    """
    Abstract base class for controllers.

    Controllers compute actuation inputs based on the current system state
    and a reference trajectory. They can maintain internal state (e.g.,
    integrator terms for PID control) that evolves over time.

    Args:
        robot: The dynamical system (robot) to be controlled.
        reference_trajectory: The desired trajectory to track.
    """

    robot: DynamicalSystem
    reference_trajectory: ReferenceTrajectory

    @abstractmethod
    def __call__(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the control action given the current system state.

        This method is called at each control step to compute the actuation
        input based on the current state and the reference trajectory.

        Implementations should be JAX-compatible (jittable) for use with
        the rollout methods in DynamicalSystem. Avoid Python control flow
        that depends on array values; use jax.lax primitives instead.

        Args:
            system_state: The current state of the system, containing:
                - t (Array): Current simulation time
                - y (Array): Robot state vector (typically, configuration and velocity)
                - u (Optional[Array]): Previous actuation input (optional)
                - control_state (Optional[Any]): Internal controller state (optional)

        Returns:
            u_control (Array): The control input to apply, shape (num_actuators,).
            control_state_dot (Optional[Any]): Time derivative of the internal controller
                state as a PyTree (e.g., integrator error for integral control), or None
                if the controller is stateless.
        """
        ...
