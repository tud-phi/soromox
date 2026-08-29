__all__ = ["BaseController"]

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState


class BaseController(eqx.Module, ABC):
    """
    Abstract base class for controllers.

    Controllers compute actuation inputs based on the current system state
    and a reference trajectory. They can maintain internal state (e.g.,
    integrator terms for PID control) that evolves over time.

    Attributes:
        robot: The soft robot system to be controlled.
        reference_trajectory: The desired trajectory to track.
    """

    robot: SoftRobot
    reference_trajectory: ReferenceTrajectory

    def _controller_model_and_state(self, y: Array) -> tuple[SoftRobot, Array, Array]:
        """Return the evaluation model and configuration-space state.

        Args:
            y: Complete system state. Soft robots use their explicit state
                parser; legacy controller test doubles may provide ``[q, qd]``.

        Returns:
            A tuple containing the fixed evaluation model, configuration, and
            velocity used by controller equations.
        """
        parser = getattr(self.robot, "controller_model_and_state", None)
        if parser is not None:
            return parser(y)
        q, qd = jnp.split(y, 2)
        return self.robot, q, qd

    @staticmethod
    def _apply_gain(gain: Array, x: Array) -> Array:
        """
        Apply a gain (scalar, diagonal vector, or matrix) to a vector.

        This utility method handles different gain representations:
        - 1-d array: Element-wise multiplication (diagonal gain)
        - 2-d array: Matrix-vector multiplication (full gain matrix)

        Args:
            gain: The gain to apply (1-d or 2-d array).
            x: The input vector.

        Returns:
            The result of applying the gain to the input.

        Raises:
            ValueError: If gain is not 1-d or 2-d.
        """
        if gain.ndim == 1:
            # Diagonal or scalar (broadcasted): element-wise multiplication
            return gain * x
        elif gain.ndim == 2:
            # Full matrix: matrix-vector multiplication
            return gain @ x
        else:
            raise ValueError(f"Gain must be 1-d or 2-d, got {gain.ndim}-d")

    @abstractmethod
    def __call__(self, system_state: SystemState) -> tuple[Array, Any | None]:
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
                - environment_state (Optional[Any]): Environment state (optional)

        Returns:
            u_control (Array): The control input to apply, shape (num_actuators,).
            control_state_dot (Optional[Any]): Time derivative of the internal controller
                state as a PyTree (e.g., integrator error for integral control), or None
                if the controller is stateless.
        """
        ...
