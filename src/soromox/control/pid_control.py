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
        gamma: Inverse error scale for built-in saturation functions (e.g.,
            ``"tanh"``). Its reciprocal has the same units as the corresponding
            error and sets the asymptotic magnitude of the saturated error.

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
                - "tanh": Uses ``solve(gamma, tanh(gamma @ e))`` for a matrix
                  ``gamma`` and ``tanh(gamma * e) / gamma`` otherwise. This
                  preserves the units and small-error slope of ``e``.
                - A callable `f(e) -> saturated_e`: Custom saturation function.
            gamma: Inverse error scale for built-in saturation functions. Can be
                a positive scalar, positive diagonal vector, or symmetric
                positive-definite matrix. For a scalar or vector, ``1 / gamma``
                sets the componentwise saturation magnitude. Only used when
                ``saturation_fn="tanh"``. Defaults to 1.0.
        """
        self.Kp = jnp.atleast_1d(jnp.asarray(Kp))
        self.Ki = jnp.atleast_1d(jnp.asarray(Ki))
        self.Kd = jnp.atleast_1d(jnp.asarray(Kd))
        self.gamma = jnp.atleast_1d(jnp.asarray(gamma))

        if saturation_fn is None or saturation_fn == "identity":
            self._saturation_fn_name = "identity"
            self._custom_saturation_fn = None
        elif saturation_fn == "tanh":
            if self.gamma.ndim == 1:
                if bool(jnp.any(self.gamma <= 0)):
                    raise ValueError(
                        "Gamma must be strictly positive for tanh saturation."
                    )
            elif self.gamma.ndim == 2:
                if self.gamma.shape[0] != self.gamma.shape[1]:
                    raise ValueError(
                        "A matrix gamma must be square for tanh saturation."
                    )
                if not bool(jnp.allclose(self.gamma, self.gamma.T)):
                    raise ValueError(
                        "A matrix gamma must be symmetric positive definite for "
                        "tanh saturation."
                    )
                if not bool(jnp.all(jnp.linalg.eigvalsh(self.gamma) > 0)):
                    raise ValueError(
                        "A matrix gamma must be symmetric positive definite for "
                        "tanh saturation."
                    )
            else:
                raise ValueError(
                    "Gamma must be a scalar-like vector or matrix for tanh "
                    f"saturation, got {self.gamma.ndim}-d."
                )
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
        Apply a unit-preserving hyperbolic-tangent saturation.

        For scalar or diagonal ``gamma``, the saturation is

        ``tanh(gamma * e) / gamma``.

        Thus, ``gamma * e`` is dimensionless, the output has the same units as
        ``e``, and the Jacobian at the origin is the identity. For a matrix
        ``gamma``, the equivalent transformed-coordinate expression is
        ``solve(gamma, tanh(gamma @ e))``.

        Args:
            e: Error vector.
            gamma: Inverse error scale (scalar, vector, or matrix).

        Returns:
            Saturated error.
        """
        if gamma.ndim == 1:
            if gamma.shape not in ((1,), e.shape):
                raise ValueError(
                    "A vector gamma must be scalar-like or match the error shape; "
                    f"got gamma shape {gamma.shape} and error shape {e.shape}."
                )
            return jnp.tanh(gamma * e) / gamma
        elif gamma.ndim == 2:
            if e.ndim != 1 or gamma.shape != (e.shape[0], e.shape[0]):
                raise ValueError(
                    "A matrix gamma must match the error size of a one-dimensional "
                    "error vector; "
                    f"got gamma shape {gamma.shape} for error shape {e.shape}."
                )
            return jnp.linalg.solve(gamma, jnp.tanh(gamma @ e))
        else:
            raise ValueError(
                f"Gamma must be a scalar-like vector or matrix, got {gamma.ndim}-d."
            )

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

    def update_gains(self, gains: dict[str, Array]) -> "PIDControl":
        """
        This function updates the gains of the PIDControl object.

        Args:
            gains (dict[str, Array]): proportional, integral, and derivative gains

        Returns:
            updated_self (PIDControl): self object with updated parameters
        """
        updated_self = self
        valid_keys = {"Kp", "Ki", "Kd"}

        for key in gains:
            if key not in valid_keys:
                raise KeyError(
                    f"Attempted to update unknown gain parameter '{key}'. "
                    f"Valid keys are: {list(valid_keys)}"
                )
        updated_Kp = gains.get("Kp", self.Kp)
        updated_Ki = gains.get("Ki", self.Ki)
        updated_Kd = gains.get("Kd", self.Kd)

        updated_self = eqx.tree_at(
            lambda x: (x.Kp, x.Ki, x.Kd),
            updated_self,
            (updated_Kp, updated_Ki, updated_Kd),
        )

        return updated_self
