__all__ = ["ReferenceTrajectory"]

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from soromox.utils.geometry.rotations import RotationRepresentation
from soromox.utils.twist import twist_callable_from_pose_callable_factory


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
        n = ts.shape[0]

        # Handle single time point case: return the single value for any time
        if n == 1:
            return vals[0]

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

    When working with poses that include orientation (for 3D robots), you can specify
    a `rotation_representation` to enable proper velocity derivation. This ensures that
    angular velocities are computed correctly from orientation changes (not just naive
    differentiation).

    Parameters
    ----------
    ts : Array
        Time steps array, shape (T,).
    x_des_ts : Optional[Array]
        Desired poses at discrete time steps, shape (T, dim).
        If not provided, will be computed from x_des_fn.
    x_des_fn : Optional[Callable[[Array], Array]]
        Function that returns desired pose at any time t.
        If not provided, will be created via linear interpolation of x_des_ts.
    xd_des_ts : Optional[Array]
        Desired velocities (twists) at discrete time steps, shape (T, dim).
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
    rotation_representation : Optional[RotationRepresentation]
        The rotation representation used in the pose. If provided, velocity derivation
        will properly convert orientation derivatives to angular velocities.
        - ROTATION_VECTOR: 3D rotation vector (axis-angle)
        - QUATERNION: 4D unit quaternion
        - ROTATION_MATRIX_6D: 6D continuous representation
        If None, simple differentiation is used (appropriate for position-only
        trajectories or planar robots).
    n_points : int
        Number of points in the pose (default 1). Used with rotation_representation
        for proper velocity derivation.
    is_planar : bool
        Whether the trajectory is for a planar robot. If True, velocity derivation
        uses simple differentiation even if rotation_representation is set.

    Examples
    --------
    From discrete time series (position-only):

    ```python
    ts = jnp.linspace(0, 1, 100)
    x_des_ts = jnp.sin(ts)[:, None]  # shape (100, 1)
    reference = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
    reference.x_des_fn(0.5)
    ```

    From a continuous function with proper twist derivation:

    ```python
    def x_fn(t):
        return jnp.array([0.0, 0.0, t * 0.1, 0.0, 0.0, 0.2 + 0.01 * t])

    reference = ReferenceTrajectory(
        ts=ts,
        x_des_fn=x_fn,
        rotation_representation=RotationRepresentation.ROTATION_VECTOR,
    )
    reference.xd_des_fn(0.5)
    ```
    """

    ts: Array

    # Pose reference
    x_des_ts: Array | None = None
    x_des_fn: Callable[[Array], Array] | None = None

    # Velocity (twist) reference (first derivative)
    xd_des_ts: Array | None = None
    xd_des_fn: Callable[[Array], Array] | None = None

    # Acceleration reference (second derivative)
    xdd_des_ts: Array | None = None
    xdd_des_fn: Callable[[Array], Array] | None = None

    # Rotation representation for proper twist derivation
    rotation_representation: RotationRepresentation | None = None
    n_points: int = 1
    is_planar: bool = False

    def __post_init__(self):
        # Validate that at least one position representation is provided
        if self.x_des_ts is None and self.x_des_fn is None:
            raise ValueError("At least one of x_des_ts or x_des_fn must be provided")

        # Position: discrete -> continuous
        if self.x_des_ts is not None and self.x_des_fn is None:
            object.__setattr__(
                self, "x_des_fn", _make_interp_fn(self.ts, self.x_des_ts)
            )

        # Position: continuous -> discrete
        if self.x_des_fn is not None and self.x_des_ts is None:
            object.__setattr__(self, "x_des_ts", jax.vmap(self.x_des_fn)(self.ts))

        # Velocity: handle auto-derivation and conversion
        if self.xd_des_ts is None and self.xd_des_fn is None:
            # Derive from position function
            xd_fn = self._create_velocity_fn()
            object.__setattr__(self, "xd_des_fn", xd_fn)
            object.__setattr__(self, "xd_des_ts", jax.vmap(self.xd_des_fn)(self.ts))
        elif self.xd_des_ts is not None and self.xd_des_fn is None:
            # discrete -> continuous
            object.__setattr__(
                self, "xd_des_fn", _make_interp_fn(self.ts, self.xd_des_ts)
            )
        elif self.xd_des_fn is not None and self.xd_des_ts is None:
            # continuous -> discrete
            object.__setattr__(self, "xd_des_ts", jax.vmap(self.xd_des_fn)(self.ts))

        # Acceleration: handle auto-derivation and conversion
        if self.xdd_des_ts is None and self.xdd_des_fn is None:
            # Derive from velocity function
            object.__setattr__(self, "xdd_des_fn", _derive_fn(self.xd_des_fn))
            object.__setattr__(self, "xdd_des_ts", jax.vmap(self.xdd_des_fn)(self.ts))
        elif self.xdd_des_ts is not None and self.xdd_des_fn is None:
            # discrete -> continuous
            object.__setattr__(
                self, "xdd_des_fn", _make_interp_fn(self.ts, self.xdd_des_ts)
            )
        elif self.xdd_des_fn is not None and self.xdd_des_ts is None:
            # continuous -> discrete
            object.__setattr__(self, "xdd_des_ts", jax.vmap(self.xdd_des_fn)(self.ts))

    def _create_velocity_fn(self) -> Callable[[Array], Array]:
        """
        Create the velocity (twist) function from the pose function.

        If rotation_representation is set, uses proper geometric velocity derivation.
        Otherwise, uses simple differentiation.
        """
        if self.rotation_representation is None and not self.is_planar:
            # Simple differentiation for position-only or unknown representation
            return _derive_fn(self.x_des_fn)

        # Get pose dimension from evaluating at t=0
        sample_pose = self.x_des_fn(self.ts[0])
        n_pose_dim = sample_pose.shape[0] // self.n_points

        # Determine orientation dimension
        if self.is_planar:
            n_orientation_dim = 1  # theta for planar
        elif self.rotation_representation == RotationRepresentation.ROTATION_VECTOR:
            n_orientation_dim = 3
        elif self.rotation_representation == RotationRepresentation.QUATERNION:
            n_orientation_dim = 4
        elif self.rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
            n_orientation_dim = 6
        else:
            n_orientation_dim = 3  # default

        return twist_callable_from_pose_callable_factory(
            pose_fn=self.x_des_fn,
            n_points=self.n_points,
            n_pose_dim=n_pose_dim,
            n_orientation_dim=n_orientation_dim,
            rotation_representation=self.rotation_representation,
            is_planar=self.is_planar,
        )

    def tree_flatten(self):
        """Flatten the ReferenceTrajectory for JAX PyTree operations."""
        # Children: arrays that participate in JAX transformations
        children = [self.ts, self.x_des_ts, self.xd_des_ts, self.xdd_des_ts]

        # Aux data: functions and static fields (not traced by JAX)
        aux = {
            "x_des_fn": self.x_des_fn,
            "xd_des_fn": self.xd_des_fn,
            "xdd_des_fn": self.xdd_des_fn,
            "rotation_representation": self.rotation_representation,
            "n_points": self.n_points,
            "is_planar": self.is_planar,
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
        object.__setattr__(
            obj, "rotation_representation", aux["rotation_representation"]
        )
        object.__setattr__(obj, "n_points", aux["n_points"])
        object.__setattr__(obj, "is_planar", aux["is_planar"])

        return obj
