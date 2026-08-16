"""Numerical helpers shared across JAX utilities.

Singular-point conventions used by the ``safe_*`` helpers:

- The value is the analytic limit where one exists, otherwise zero.
- The derivative is defined to be zero at the singular point.
- Away from the singular point the helpers match their plain counterparts
  (``jnp.sqrt``, ``jnp.linalg.norm``, ``/``) in both value and derivative.

Choosing between them:

- ``jnp.linalg.norm(x, axis=k) ** 2`` is exactly ``jnp.sum(x**2, axis=k)``.
  Prefer the latter whenever a norm is only ever squared: it needs no
  threshold and is finite everywhere.
- Use :func:`safe_norm` only when the norm's own value is needed.

Guarding a singular expression only works on its *input*, never on its output.
``jnp.where`` and ``lax.select`` evaluate both sides and their transpose
multiplies the unused side by a materialized zero, so a ``NaN`` produced there
survives as ``0 * NaN``::

    jnp.where(x > 0, jnp.sqrt(x), 0.0)       # broken at x == 0
    x_safe = jnp.where(x > 0, x, 1.0)        # safe: no branch is ever singular
    jnp.where(x > 0, jnp.sqrt(x_safe), 0.0)

For a scalar predicate, ``lax.cond`` evaluates and differentiates only the
taken branch. Under ``vmap``, however, batched predicates may lower to a
select operation and both branch computations can execute. Branch inputs must
therefore still be sanitized when a condition is vectorized.

The fallbacks hide *where* a model reaches a singular configuration. To find
that out, wrap the call in ``soromox.autodiff.strict_singularities_mode()``: the
helpers then behave like the plain operations they replace, so the singular
input surfaces as ``NaN`` or ``inf``.
"""

__all__ = [
    "eps_for_dtype",
    "safe_sqrt",
    "safe_norm",
    "safe_divide",
    "safe_normalize",
]

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from soromox.autodiff import strict_singularities_enabled


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


@jax.custom_jvp
def _safe_sqrt(x: Array) -> Array:
    """Return ``sqrt(max(x, 0))`` with a zero derivative at the origin."""
    return jnp.sqrt(jnp.maximum(x, jnp.zeros_like(x)))


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    """Differentiate :func:`_safe_sqrt`, returning a zero slope at the origin."""
    (x,) = primals
    (x_dot,) = tangents
    primal_out = _safe_sqrt(x)
    positive = x > 0
    denominator = jnp.where(positive, 2.0 * primal_out, jnp.ones_like(primal_out))
    tangent_out = jnp.where(positive, x_dot / denominator, jnp.zeros_like(primal_out))
    return primal_out, tangent_out


def safe_sqrt(x: Array) -> Array:
    """Return ``sqrt(max(x, 0))`` with a zero derivative at the origin.

    ``jnp.sqrt`` has an infinite derivative at zero, which becomes ``NaN`` as
    soon as it is multiplied by a cotangent.

    Under ``strict_singularities_mode`` this is plain ``jnp.sqrt``, so a
    negative or zero argument surfaces instead of being clamped.

    Args:
        x: Real-valued array. Negative entries are treated as zero.

    Returns:
        Array with the same shape as ``x``.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from soromox.utils import safe_sqrt

        jax.grad(safe_sqrt)(jnp.array(0.0))  # 0.0 rather than inf
        ```
    """
    if strict_singularities_enabled():
        return jnp.sqrt(x)
    return _safe_sqrt(x)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _safe_norm(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Return the Euclidean norm of ``x`` with a zero derivative at the origin."""
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims))


@_safe_norm.defjvp
def _safe_norm_jvp(
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[Array, Array]:
    """Differentiate :func:`_safe_norm`, returning a zero slope at the origin."""
    (x,) = primals
    (x_dot,) = tangents
    primal_out = _safe_norm(x, axis, keepdims)
    positive = primal_out > 0
    denominator = jnp.where(positive, primal_out, jnp.ones_like(primal_out))
    directional = jnp.sum(x * x_dot, axis=axis, keepdims=keepdims) / denominator
    tangent_out = jnp.where(positive, directional, jnp.zeros_like(primal_out))
    return primal_out, tangent_out


def safe_norm(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Return the Euclidean norm of ``x`` with a zero derivative at the origin.

    ``jnp.linalg.norm`` differentiates to ``x / norm(x)``, which is ``0 / 0`` at
    the zero vector. Singular configurations such as a straight backbone or a
    collapsed routing path reach that point exactly.

    Under ``strict_singularities_mode`` the zero vector differentiates to
    ``NaN``, as the plain norm does.

    Args:
        x: Real-valued array.
        axis: Static axis or axes to reduce over. If ``None``, the norm is taken
            over the flattened array.
        keepdims: Static flag. If ``True``, reduced axes are retained with
            length one, which is convenient when dividing ``x`` by its norm.

    Returns:
        Array containing the Euclidean norm, reduced over ``axis``.

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        from soromox.utils import safe_norm

        jax.grad(safe_norm)(jnp.zeros(3))  # [0. 0. 0.] rather than [nan nan nan]
        ```
    """
    if strict_singularities_enabled():
        return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims))
    return _safe_norm(x, axis, keepdims)


def safe_divide(
    numerator: Array,
    denominator: Array,
    eps: float | Array,
    fallback: float | Array = 0.0,
) -> Array:
    """Divide by a denominator that may vanish, keeping derivatives finite.

    The denominator is sanitized before the division, so neither branch ever
    evaluates a singular quotient.

    Args:
        numerator: Dividend array.
        denominator: Divisor array. Entries with magnitude at or below ``eps``
            select ``fallback`` instead of the quotient.
        eps: Small nonnegative threshold below which the denominator is treated
            as zero.
        fallback: Value returned where the denominator vanishes. Broadcast
            against the quotient shape.

    Under ``strict_singularities_mode`` this is a plain division, so a
    vanishing denominator surfaces as ``inf`` or ``NaN``.

    Returns:
        Array containing ``numerator / denominator`` away from the singularity
        and ``fallback`` at it.

    Notes:
        The two ``jnp.where`` calls are safe here because the denominator is
        sanitized *before* division. Thus both evaluated branches remain finite.
    """
    numerator = jnp.asarray(numerator)
    denominator = jnp.asarray(denominator)

    if strict_singularities_enabled():
        return numerator / denominator

    eps_arr = eps_for_dtype(eps, denominator.dtype)

    regular = jnp.abs(denominator) > eps_arr
    denominator_safe = jnp.where(regular, denominator, jnp.ones_like(denominator))
    quotient = numerator / denominator_safe

    return jnp.where(regular, quotient, jnp.broadcast_to(fallback, quotient.shape))


def safe_normalize(
    x: Array,
    axis: int | tuple[int, ...] | None = None,
    eps: float | Array | None = None,
    fallback: Array | None = None,
) -> Array:
    """Scale ``x`` to unit length, keeping the zero vector finite.

    Args:
        x: Real-valued array to normalize.
        axis: Static axis or axes holding the vector components. If ``None``,
            the whole array is treated as a single vector.
        eps: Small nonnegative norm threshold. Defaults to machine epsilon for
            the dtype of ``x``.
        fallback: Value returned where the norm vanishes. Defaults to zeros,
            which keeps the result finite without inventing a direction.

    Returns:
        Array with the same shape as ``x``.

    Note:
        Under ``strict_singularities_mode`` the zero vector surfaces as ``NaN``
        rather than ``fallback``.
    """
    x = jnp.asarray(x)
    eps_arr = eps_for_dtype(eps, x.dtype)

    norm = safe_norm(x, axis, axis is not None)

    if fallback is None:
        fallback = jnp.zeros_like(x)

    return safe_divide(x, norm, eps_arr, fallback=fallback)
