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

    def _controller_robot_and_state(self, y: Array) -> tuple[SoftRobot, Array, Array]:
        """
        Return the fixed-base robot and internal state used by a controller.

        Floating-base controllers intentionally retain the established
        internal-coordinate control laws. The returned robot has floating-base
        dynamics disabled and is mounted at the current runtime base pose. Its
        kinematics and gravity therefore reflect that pose, while base velocity,
        acceleration, reaction motion, and floating/internal coupling are not
        part of the controller approximation.

        Args:
            y: Complete system state with trailing dimension
                ``robot.state_size``. Legacy controller test doubles may still
                provide a two-block ``[q, qd]`` state.

        Returns:
            The fixed-base robot mounted at the current base pose, followed by
            internal coordinates and velocities.
        """
        if isinstance(self.robot, SoftRobot):
            q, v, _ = self.robot.split_state(y)
            base_pose, q_internal = self.robot.split_configuration(q)
            _, qd_internal = self.robot.split_velocity(v)
            fixed_base_robot = self.robot._fixed_base_robot_at_pose(base_pose)
            return fixed_base_robot, q_internal, qd_internal
        q, qd = jnp.split(y, 2)
        return self.robot, q, qd

    def _controller_dynamics_and_state(
        self, y: Array, dynamics: Any
    ) -> tuple[Any, Array, Array]:
        """
        Mount transformed internal dynamics at the current base pose.

        Args:
            y: Complete system state.
            dynamics: Actuation- or operational-space dynamics object whose
                ``fixed_base_robot`` field contains its internal-coordinate
                fixed-base approximation.

        Returns:
            The dynamics object with ``fixed_base_robot`` mounted at the
            current base pose, followed by internal coordinates and velocities.
        """
        fixed_base_robot, q_internal, qd_internal = self._controller_robot_and_state(y)
        if (
            hasattr(dynamics, "fixed_base_robot")
            and dynamics.fixed_base_robot is not fixed_base_robot
        ):
            dynamics = eqx.tree_at(
                lambda value: value.fixed_base_robot,
                dynamics,
                fixed_base_robot,
            )
        return dynamics, q_internal, qd_internal

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
