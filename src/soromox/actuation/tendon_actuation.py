__all__ = ["linear_routing", "linear_routing_arc_length_derivative"]
from jax import numpy as jnp


def linear_routing(tendon_routing_params, s):
    """
    Routing of linear tendons as a function of ``s``.

    ``LinearTendonRoutingParams`` stores a leading tendon axis. System classes
    evaluate this function through ``jax.vmap`` so each tendon may have distinct
    intercepts, slopes, and attachment segment. Direct batched calls are also
    supported when the parameter fields and ``s`` are broadcast-compatible.

    Args:
        tendon_routing_params: Single-tendon or batched ``LinearTendonRoutingParams``.
        s: Abscissa value or broadcast-compatible array.

    Returns:
        Position of the tendon at ``s`` w.r.t. the centerline. The final axis stores
        ``[0, y, z]``.
    """

    ry = jnp.asarray(tendon_routing_params.y_intercept)
    my = jnp.asarray(tendon_routing_params.y_slope)
    rz = jnp.asarray(tendon_routing_params.z_intercept)
    mz = jnp.asarray(tendon_routing_params.z_slope)
    s = jnp.asarray(s)
    y = ry + my * s
    z = rz + mz * s
    return jnp.stack([jnp.zeros_like(y), y, z], axis=-1)


def linear_routing_arc_length_derivative(tendon_routing_params, s):
    """
    Arc-length derivative of the linear tendon routing as a function of ``s``.

    The final axis stores the derivative of ``[0, y, z]``. As with
    ``linear_routing``, system classes evaluate this per tendon through
    ``jax.vmap`` and direct broadcast-compatible batched calls are supported.

    Args:
        tendon_routing_params: Single-tendon or batched ``LinearTendonRoutingParams``.
        s: Abscissa value or broadcast-compatible array. It is accepted so custom
            routing derivatives can share the same signature.

    Returns:
        Arc-length derivative of the tendon routing.
    """

    my = jnp.asarray(tendon_routing_params.y_slope)
    mz = jnp.asarray(tendon_routing_params.z_slope)
    s = jnp.asarray(s)
    zeros = jnp.zeros_like(my + 0.0 * s)
    return jnp.stack([zeros, my + zeros, mz + zeros], axis=-1)
