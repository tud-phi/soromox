__all__ = ["ReferenceTrajectory"]

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array


def _make_interp_fn(ts: Array, vals: Array) -> Callable[[Array], Array]:
    """
    Create a jittable linear interpolation function.

    Parameters
    ----------
    ts : Array
        Time steps array, shape (T,).
    vals : Array
        Values at each time step, shape (T, dim).

    Returns
    -------
    Callable[[Array], Array]
        Function that takes a scalar time t and returns interpolated value.
    """

    def interp_fn(t: Array) -> Array:
        # Handle single time point case
        n = ts.shape[0]

        # Find index: ts[idx-1] <= t < ts[idx]
        idx = jnp.searchsorted(ts, t)
        idx = jnp.clip(idx, 1, n - 1)

        # Get bounding time points
        t0 = ts[idx - 1]
        t1 = ts[idx]

        # Compute interpolation weight
        # Handle case where t0 == t1 (avoid division by zero)
        dt = t1 - t0
        alpha = jnp.where(dt > 0, (t - t0) / dt, 0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        # Linear interpolation
        return (1.0 - alpha) * vals[idx - 1] + alpha * vals[idx]

    return interp_fn


def _derive_fn(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """
    Create a function that computes the time derivative of fn.

    Uses JAX automatic differentiation to compute the derivative.

    Parameters
    ----------
    fn : Callable[[Array], Array]
        Function that takes scalar time t and returns a value of shape (dim,).

    Returns
    -------
    Callable[[Array], Array]
        Function that takes scalar time t and returns the time derivative.
    """

    def derivative_fn(t: Array) -> Array:
        # For a function f: R -> R^n (scalar to vector),
        # jacfwd computes df/dt which has shape (n,) - the same as the output
        return jax.jacfwd(fn)(t)

    return derivative_fn


@jax.tree_util.register_pytree_node_class
@dataclass
class ReferenceTrajectory:
    """
    Container for reference trajectory with both discrete and continuous representations.

    This class provides flexible specification of reference trajectories for controllers.
    Users can provide either discrete time-series data or continuous functions, and the
    class will automatically generate the other representation.

    Parameters
    ----------
    ts : Array
        Time steps array, shape (T,).
    x_des_ts : Optional[Array]
        Desired positions at discrete time steps, shape (T, dim).
        If not provided, will be computed from x_des_fn.
    x_des_fn : Optional[Callable[[Array], Array]]
        Function that returns desired position at any time t.
        If not provided, will be created via linear interpolation of x_des_ts.
    xd_des_ts : Optional[Array]
        Desired velocities at discrete time steps, shape (T, dim).
        If not provided, will be computed from xd_des_fn or derived from x_des_fn.
    xd_des_fn : Optional[Callable[[Array], Array]]
        Function that returns desired velocity at any time t.
        If not provided, will be created via interpolation or derived from x_des_fn.
    xdd_des_ts : Optional[Array]
        Desired accelerations at discrete time steps, shape (T, dim).
        If not provided, will be computed from xdd_des_fn or derived from xd_des_fn.
    xdd_des_fn : Optional[Callable[[Array], Array]]
        Function that returns desired acceleration at any time t.
        If not provided, will be created via interpolation or derived from xd_des_fn.

    Examples
    --------
    # From discrete time series
    >>> ts = jnp.linspace(0, 1, 100)
    >>> x_des_ts = jnp.sin(ts)[:, None]  # shape (100, 1)
    >>> ref = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
    >>> ref.x_des_fn(0.5)  # Interpolated value at t=0.5

    # From continuous function
    >>> def x_fn(t):
    ...     return jnp.array([jnp.sin(t)])
    >>> ref = ReferenceTrajectory(ts=ts, x_des_fn=x_fn)
    >>> ref.x_des_ts  # Evaluated at all ts
    >>> ref.xd_des_fn(0.5)  # Auto-derived velocity at t=0.5
    """

    ts: Array

    # Position reference
    x_des_ts: Optional[Array] = None
    x_des_fn: Optional[Callable[[Array], Array]] = None

    # Velocity reference (first derivative)
    xd_des_ts: Optional[Array] = None
    xd_des_fn: Optional[Callable[[Array], Array]] = None

    # Acceleration reference (second derivative)
    xdd_des_ts: Optional[Array] = None
    xdd_des_fn: Optional[Callable[[Array], Array]] = None

    def __post_init__(self):
        # Validate that at least one position representation is provided
        if self.x_des_ts is None and self.x_des_fn is None:
            raise ValueError(
                "At least one of x_des_ts or x_des_fn must be provided"
            )

        # Position: discrete -> continuous
        if self.x_des_ts is not None and self.x_des_fn is None:
            object.__setattr__(
                self, "x_des_fn", _make_interp_fn(self.ts, self.x_des_ts)
            )

        # Position: continuous -> discrete
        if self.x_des_fn is not None and self.x_des_ts is None:
            object.__setattr__(
                self, "x_des_ts", jax.vmap(self.x_des_fn)(self.ts)
            )

        # Velocity: handle auto-derivation and conversion
        if self.xd_des_ts is None and self.xd_des_fn is None:
            # Derive from position function
            object.__setattr__(self, "xd_des_fn", _derive_fn(self.x_des_fn))
            object.__setattr__(
                self, "xd_des_ts", jax.vmap(self.xd_des_fn)(self.ts)
            )
        elif self.xd_des_ts is not None and self.xd_des_fn is None:
            # discrete -> continuous
            object.__setattr__(
                self, "xd_des_fn", _make_interp_fn(self.ts, self.xd_des_ts)
            )
        elif self.xd_des_fn is not None and self.xd_des_ts is None:
            # continuous -> discrete
            object.__setattr__(
                self, "xd_des_ts", jax.vmap(self.xd_des_fn)(self.ts)
            )

        # Acceleration: handle auto-derivation and conversion
        if self.xdd_des_ts is None and self.xdd_des_fn is None:
            # Derive from velocity function
            object.__setattr__(self, "xdd_des_fn", _derive_fn(self.xd_des_fn))
            object.__setattr__(
                self, "xdd_des_ts", jax.vmap(self.xdd_des_fn)(self.ts)
            )
        elif self.xdd_des_ts is not None and self.xdd_des_fn is None:
            # discrete -> continuous
            object.__setattr__(
                self, "xdd_des_fn", _make_interp_fn(self.ts, self.xdd_des_ts)
            )
        elif self.xdd_des_fn is not None and self.xdd_des_ts is None:
            # continuous -> discrete
            object.__setattr__(
                self, "xdd_des_ts", jax.vmap(self.xdd_des_fn)(self.ts)
            )

    def tree_flatten(self):
        """Flatten the ReferenceTrajectory for JAX PyTree operations."""
        # Children: arrays that participate in JAX transformations
        children = [self.ts, self.x_des_ts, self.xd_des_ts, self.xdd_des_ts]

        # Aux data: functions (not traced by JAX)
        aux = {
            "x_des_fn": self.x_des_fn,
            "xd_des_fn": self.xd_des_fn,
            "xdd_des_fn": self.xdd_des_fn,
        }

        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten a ReferenceTrajectory from JAX PyTree operations."""
        ts, x_des_ts, xd_des_ts, xdd_des_ts = children

        # Create instance without triggering __post_init__ validation
        # since all fields are already populated
        obj = object.__new__(cls)
        object.__setattr__(obj, "ts", ts)
        object.__setattr__(obj, "x_des_ts", x_des_ts)
        object.__setattr__(obj, "x_des_fn", aux["x_des_fn"])
        object.__setattr__(obj, "xd_des_ts", xd_des_ts)
        object.__setattr__(obj, "xd_des_fn", aux["xd_des_fn"])
        object.__setattr__(obj, "xdd_des_ts", xdd_des_ts)
        object.__setattr__(obj, "xdd_des_fn", aux["xdd_des_fn"])

        return obj
