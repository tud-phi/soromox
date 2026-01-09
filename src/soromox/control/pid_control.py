__all__ = ["PIDControl", "PIDControllerState"]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

# Supported saturation function names
SaturationFnName = Literal["identity", "tanh"]


@jax.tree_util.register_pytree_node_class
@dataclass
class PIDControllerState:
    """
    State container for the PID controller.

    This dataclass holds all internal state variables that evolve over time
    during closed-loop control. It is registered as a JAX pytree to enable
    use with JAX transformations and ODE solvers.

    Attributes:
        integral_error: Accumulated integral of the (possibly saturated) error.
            Shape: (num_dofs,)
    """

    integral_error: Array

    def tree_flatten(self):
        """Flatten the state for JAX pytree operations."""
        children = (self.integral_error,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the state from JAX pytree operations."""
        (integral_error,) = children
        return cls(integral_error=integral_error)

    @classmethod
    def zero(cls, num_dofs: int) -> "PIDControllerState":
        """
        Create a zero-initialized controller state.

        Args:
            num_dofs: Number of degrees of freedom.

        Returns:
            PIDControllerState with zero integral error.
        """
        return cls(integral_error=jnp.zeros((num_dofs,)))


class PIDControl(eqx.Module):
    """
    PID controller with configurable gains and optional integral error saturation.

    This class implements a standard PID control law:
        u = Kp @ e + Ki @ integral_error + Kd @ ed

    where:
        - e is the tracking error
        - ed is the error derivative
        - integral_error is the accumulated integral of the (possibly saturated) error

    The integral error dynamics are:
        integral_error_dot = sat(e)

    where sat() is a saturation function that can be used for anti-windup.

    Attributes:
        Kp: Proportional gain (scalar, diagonal vector, or matrix).
        Ki: Integral gain (scalar, diagonal vector, or matrix).
        Kd: Derivative gain (scalar, diagonal vector, or matrix).
        gamma: Scaling parameter for built-in saturation functions (e.g., "tanh").

    References:
        Pustina, P., Borja, P., Della Santina, C., & De Luca, A. (2022).
        P-satI-D shape regulation of soft robots. IEEE Robotics and Automation Letters, 8(1), 1-8.
    """

    Kp: Array
    Ki: Array
    Kd: Array
    gamma: Array
    _saturation_fn_name: str = eqx.field(static=True)
    _custom_saturation_fn: Callable[[Array], Array] | None = eqx.field(static=True)

    def __init__(
        self,
        Kp: float | Array,
        Ki: float | Array,
        Kd: float | Array,
        saturation_fn: SaturationFnName | Callable[[Array], Array] | None = None,
        gamma: float | Array = 1.0,
    ):
        """
        Initialize the PID controller.

        Args:
            Kp: Proportional gain. Can be:
                - A scalar (float or 0-d array): applied uniformly to all error components.
                - A 1-d array (diagonal): element-wise multiplication with error.
                - A 2-d array (matrix): full matrix multiplication with error.
            Ki: Integral gain. Same format options as Kp.
            Kd: Derivative gain. Same format options as Kp.
            saturation_fn: Optional saturation function for anti-windup. Can be:
                - None or "identity": No saturation (default).
                - "tanh": Uses `tanh(gamma * e)` where gamma is specified separately.
                - A callable `f(e) -> saturated_e`: Custom saturation function.
            gamma: Scaling parameter for built-in saturation functions (e.g., "tanh").
                Can be a scalar, diagonal vector, or matrix to stretch/compress
                the error before saturation. Only used when saturation_fn is a string.
                Defaults to 1.0.
        """
        self.Kp = jnp.atleast_1d(jnp.asarray(Kp))
        self.Ki = jnp.atleast_1d(jnp.asarray(Ki))
        self.Kd = jnp.atleast_1d(jnp.asarray(Kd))
        self.gamma = jnp.atleast_1d(jnp.asarray(gamma))

        if saturation_fn is None or saturation_fn == "identity":
            self._saturation_fn_name = "identity"
            self._custom_saturation_fn = None
        elif saturation_fn == "tanh":
            self._saturation_fn_name = "tanh"
            self._custom_saturation_fn = None
        elif callable(saturation_fn):
            self._saturation_fn_name = "custom"
            self._custom_saturation_fn = saturation_fn
        else:
            raise ValueError(
                f"Unknown saturation function: {saturation_fn}. "
                f"Supported options: 'identity', 'tanh', or a callable."
            )

    def _apply_saturation(self, e: Array) -> Array:
        """
        Apply the saturation function to the error.

        Args:
            e: Error vector.

        Returns:
            Saturated error.
        """
        if self._saturation_fn_name == "identity":
            return e
        elif self._saturation_fn_name == "tanh":
            return self._tanh_saturation(e, self.gamma)
        else:  # custom
            assert self._custom_saturation_fn is not None
            return self._custom_saturation_fn(e)

    @staticmethod
    def _tanh_saturation(e: Array, gamma: Array) -> Array:
        """
        Tanh saturation function: tanh(gamma * e).

        Args:
            e: Error vector.
            gamma: Scaling parameter (scalar, vector, or matrix).

        Returns:
            Saturated error.
        """
        if gamma.ndim == 1:
            return jnp.tanh(gamma * e)
        elif gamma.ndim == 2:
            return jnp.tanh(gamma @ e)
        else:
            return jnp.tanh(gamma * e)

    def _apply_gain(self, gain: Array, x: Array) -> Array:
        """
        Apply a gain (scalar, diagonal vector, or matrix) to a vector.

        Args:
            gain: The gain to apply (1-d or 2-d array).
            x: The input vector.

        Returns:
            The result of applying the gain to the input.
        """
        if gain.ndim == 1:
            # Diagonal or scalar (broadcasted): element-wise multiplication
            return gain * x
        elif gain.ndim == 2:
            # Full matrix: matrix-vector multiplication
            return gain @ x
        else:
            raise ValueError(f"Gain must be 1-d or 2-d, got {gain.ndim}-d")

    def __call__(
        self,
        e: Array,
        ed: Array,
        integral_error: Array,
    ) -> tuple[Array, Array]:
        """
        Compute the PID control output and integral error derivative.

        Args:
            e: Tracking error vector, shape (n,).
            ed: Error derivative vector, shape (n,).
            integral_error: Current integral of error, shape (n,).

        Returns:
            u: Control output, shape (n,) or (m,) depending on gain shapes.
            integral_error_dot: Time derivative of integral error, shape (n,).
                This is the (possibly saturated) error that should be integrated.
        """
        # Compute PID terms
        p_term = self._apply_gain(self.Kp, e)
        i_term = self._apply_gain(self.Ki, integral_error)
        d_term = self._apply_gain(self.Kd, ed)

        # Compute control output
        u = p_term + i_term + d_term

        # Compute integral error derivative with optional saturation
        integral_error_dot = self._apply_saturation(e)

        return u, integral_error_dot
