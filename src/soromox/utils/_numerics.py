"""Internal numerical helpers shared across JAX utilities."""

__all__ = ["eps_for_dtype"]

import jax.numpy as jnp
from jax import Array


def eps_for_dtype(eps: float | Array | None, dtype: jnp.dtype) -> Array:
    """Return an epsilon scalar as a JAX array with the requested dtype.

    Args:
        eps: Optional epsilon value. If ``None``, machine epsilon for ``dtype``
            is used.
        dtype: JAX floating-point dtype for the returned scalar.

    Returns:
        Scalar JAX array with dtype ``dtype``.
    """
    if eps is None:
        return jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    return jnp.asarray(eps, dtype=dtype)
