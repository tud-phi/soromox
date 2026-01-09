__all__ = ["ClosedFormModelBasedController"]

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from soromox.control.base_controller import BaseController
from soromox.systems.system_state import SystemState


class ClosedFormModelBasedController(BaseController):
    """
    Base class for closed-form model-based controllers.

    This controller combines a model-based feedforward term with an error-based
    feedback term. Subclasses can override either or both terms; unimplemented
    terms default to zero.

    The control law is:
        u_control = model_based_term + error_based_feedback_term
    """

    def model_based_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the model-based feedforward control term.

        This term typically computes the control input required to achieve
        the desired trajectory based on the system dynamics model (e.g.,
        inverse dynamics, gravity compensation).

        The default implementation returns zero control input. Subclasses
        can override this method to provide model-based feedforward control.

        Args:
            system_state: The current state of the system.

        Returns:
            u_model (Array): Model-based control input, shape (num_actuators,).
            control_state_dot (Optional[Any]): Time derivative of the internal
                controller state contributed by this term, or None.
        """
        return jnp.zeros((self.robot.num_actuators,)), None

    def error_based_feedback_term(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the error-based feedback control term.

        This term typically computes a corrective control input based on
        the tracking error (e.g., PD control, PID control).

        The default implementation returns zero control input. Subclasses
        can override this method to provide error-based feedback control.

        Args:
            system_state: The current state of the system.

        Returns:
            u_feedback (Array): Feedback control input, shape (num_actuators,).
            control_state_dot (Optional[Any]): Time derivative of the internal
                controller state contributed by this term, or None.
        """
        return jnp.zeros((self.robot.num_actuators,)), None

    def __call__(
        self, system_state: SystemState
    ) -> Tuple[Array, Optional[Any]]:
        """
        Compute the combined control action.

        Combines the model-based feedforward term and the error-based feedback
        term by summing both the control inputs and the control state derivatives.

        Args:
            system_state: The current state of the system.

        Returns:
            u_control (Array): Combined control input, shape (num_actuators,).
            control_state_dot (Optional[Any]): Combined time derivative of the
                internal controller state, or None if both terms are stateless.
        """
        u_model, control_state_dot_model = self.model_based_term(system_state)
        u_feedback, control_state_dot_feedback = self.error_based_feedback_term(
            system_state
        )

        u_control = u_model + u_feedback

        # Combine control state derivatives
        if control_state_dot_model is None and control_state_dot_feedback is None:
            control_state_dot = None
        elif control_state_dot_model is None:
            control_state_dot = control_state_dot_feedback
        elif control_state_dot_feedback is None:
            control_state_dot = control_state_dot_model
        else:
            # Both are non-None, add them element-wise using tree_map
            control_state_dot = jax.tree.map(
                lambda a, b: a + b,
                control_state_dot_model,
                control_state_dot_feedback,
            )

        return u_control, control_state_dot
