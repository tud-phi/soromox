"""Routines for numerical differentiation."""

import functools
from collections.abc import Callable

from jax import Array
from jax import numpy as jnp

__all__ = ["approx_derivative"]


def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
    """Adjust finite-difference steps to respect variable bounds.

    Args:
        x0: Point at which to estimate the derivative, with shape ``(n,)``.
        h: Desired absolute finite-difference steps, with shape ``(n,)``.
        num_steps: Number of ``h`` steps required in one direction. For
            example, ``2`` means that the scheme evaluates ``f(x0 + 2 * h)``
            or ``f(x0 - 2 * h)``.
        scheme: Whether the scheme requires one or both directions. The
            ``"1-sided"`` scheme covers forward and backward differences,
            while ``"2-sided"`` covers central differences.
        lb: Lower bounds on the independent variables.
        ub: Upper bounds on the independent variables.

    Returns:
        A tuple containing the adjusted step sizes and a Boolean mask
        indicating which entries use a one-sided scheme. The mask is
        meaningful only for ``scheme="2-sided"``.

    Raises:
        ValueError: If ``scheme`` is not ``"1-sided"`` or ``"2-sided"``.
    """
    if scheme == "1-sided":
        use_one_sided = jnp.ones_like(h, dtype=bool)
    elif scheme == "2-sided":
        h = jnp.abs(h)
        use_one_sided = jnp.zeros_like(h, dtype=bool)
    else:
        raise ValueError('`scheme` must be "1-sided" or "2-sided".')

    if jnp.all((lb == -jnp.inf) & (ub == jnp.inf)):
        return h, use_one_sided

    h_total = h * num_steps
    h_adjusted = h.copy()

    lower_dist = x0 - lb
    upper_dist = ub - x0

    if scheme == "1-sided":
        x = x0 + h_total
        violated = (x < lb) | (x > ub)
        fitting = jnp.abs(h_total) <= jnp.maximum(lower_dist, upper_dist)
        h_adjusted = jnp.where(violated & fitting, -h_adjusted, h_adjusted)

        forward = (upper_dist >= lower_dist) & ~fitting
        h_adjusted = jnp.where(forward, upper_dist / num_steps, h_adjusted)
        backward = (upper_dist < lower_dist) & ~fitting
        h_adjusted = jnp.where(backward, -lower_dist / num_steps, h_adjusted)
    elif scheme == "2-sided":
        central = (lower_dist >= h_total) & (upper_dist >= h_total)

        forward = (upper_dist >= lower_dist) & ~central
        h_adjusted = jnp.where(
            forward, jnp.minimum(h, 0.5 * upper_dist / num_steps), h_adjusted
        )
        use_one_sided = jnp.where(forward, True, use_one_sided)

        backward = (upper_dist < lower_dist) & ~central
        h_adjusted = jnp.where(
            backward,
            -jnp.minimum(h, 0.5 * lower_dist / num_steps),
            h_adjusted,
        )
        use_one_sided = jnp.where(backward, True, use_one_sided)

        min_dist = jnp.minimum(upper_dist, lower_dist) / num_steps
        adjusted_central = ~central & (jnp.abs(h_adjusted) <= min_dist)
        h_adjusted = jnp.where(adjusted_central, min_dist, h_adjusted)
        use_one_sided = jnp.where(adjusted_central, False, use_one_sided)

    return h_adjusted, use_one_sided


@functools.lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """Choose a relative step size for a finite-difference method.

    The step is selected from the smallest floating-point dtype among the
    input and function output. The method-specific power of machine epsilon
    then balances truncation and round-off error.

    Args:
        x0_dtype: Dtype of the parameter vector.
        f0_dtype: Dtype of the function evaluation.
        method: Finite-difference method, either ``"2-point"`` or
            ``"3-point"``.

    Returns:
        The relative step size for ``method``.

    Raises:
        RuntimeError: If ``method`` is not supported.
    """
    epsilon = jnp.finfo(jnp.float64).eps

    x0_is_fp = False
    if jnp.issubdtype(x0_dtype, jnp.inexact):
        epsilon = jnp.finfo(x0_dtype).eps
        x0_itemsize = jnp.dtype(x0_dtype).itemsize
        x0_is_fp = True

    if jnp.issubdtype(f0_dtype, jnp.inexact):
        f0_itemsize = jnp.dtype(f0_dtype).itemsize
        if x0_is_fp and f0_itemsize < x0_itemsize:
            epsilon = jnp.finfo(f0_dtype).eps

    if method == "2-point":
        return epsilon**0.5
    if method == "3-point":
        return epsilon ** (1 / 3)
    raise RuntimeError("Unknown step method, should be one of {'2-point', '3-point'}")


def _compute_absolute_step(rel_step, x0, f0, method):
    """Compute absolute finite-difference steps from a relative step.

    Args:
        rel_step: Relative step size, or ``None`` to select it automatically.
        x0: Parameter vector.
        f0: Function value at ``x0``.
        method: Finite-difference method, either ``"2-point"`` or
            ``"3-point"``.

    Returns:
        An array of absolute finite-difference steps. The dtype follows the
        dtypes used to select the machine epsilon.
    """
    # Use 1 rather than 0 for the sign of a zero input.
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1

    rstep = _eps_for_method(x0.dtype, f0.dtype, method)

    if rel_step is None:
        abs_step = rstep * sign_x0 * jnp.maximum(1.0, jnp.abs(x0))
    else:
        # Do not multiply by max(1, abs(x0)) for an explicitly requested step.
        abs_step = rel_step * sign_x0 * jnp.abs(x0)

        # Avoid a zero step when rel_step or x0 is zero.
        dx = (x0 + abs_step) - x0
        abs_step = jnp.where(
            dx == 0, rstep * sign_x0 * jnp.maximum(1.0, jnp.abs(x0)), abs_step
        )

    return abs_step


def _prepare_bounds(bounds, x0):
    """Broadcast scalar bounds and convert bounds to JAX arrays.

    Args:
        bounds: A ``(lower, upper)`` pair. Each entry may be a scalar or an
            array with the same shape as ``x0``. Unbounded entries should use
            ``-jnp.inf`` and ``jnp.inf``.
        x0: Parameter vector whose shape determines scalar-bound expansion.

    Returns:
        A tuple ``(lower, upper)`` of JAX arrays with the same shape as
        ``x0`` when scalar bounds are provided.

    Examples:
        ```python
        lower, upper = _prepare_bounds(
            [(0, 1, 2), (1, 2, jnp.inf)],
            [0.5, 1.5, 2.5],
        )
        # lower = [0, 1, 2] and upper = [1, 2, inf]
        ```
    """
    lb, ub = (jnp.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = jnp.resize(lb, x0.shape)

    if ub.ndim == 0:
        ub = jnp.resize(ub, x0.shape)

    return lb, ub


def approx_derivative(
    fun: Callable[..., Array | float],
    x0: Array | float,
    method: str = "3-point",
    rel_step: Array | float | None = None,
    abs_step: Array | float | None = None,
    f0: Array | float | None = None,
    bounds: tuple[Array | float, Array | float] = (-jnp.inf, jnp.inf),
    args: tuple = (),
    kwargs: dict | None = None,
) -> Array:
    """Compute a finite-difference approximation of a function's Jacobian.

    If a function maps from ``R^n`` to ``R^m``, the Jacobian has shape
    ``(m, n)`` and contains the partial derivative of output ``i`` with
    respect to input ``j`` at entry ``(i, j)``.

    Args:
        fun: Function whose derivatives should be estimated. It receives a
            one-dimensional array of shape ``(n,)`` and returns a scalar or a
            one-dimensional array of shape ``(m,)``.
        x0: Point at which to estimate the derivatives. Scalars are converted
            to a one-dimensional array.
        method: Finite-difference method. ``"2-point"`` uses a first-order
            forward or backward difference. ``"3-point"`` uses a central
            difference in the interior and a second-order one-sided difference
            near bounds.
        rel_step: Relative step size. If ``None``, an appropriate step is
            selected from the input and output dtypes.
        abs_step: Absolute step size. If provided, it takes precedence over
            ``rel_step`` and is adjusted to fit within ``bounds``.
        f0: Optional value of ``fun(x0)``. Providing it avoids evaluating the
            function at ``x0``.
        bounds: Lower and upper bounds on the independent variables. Each
            bound may be a scalar or an array matching ``x0``. Defaults to no
            bounds.
        args: Additional positional arguments passed to ``fun``.
        kwargs: Additional keyword arguments passed to ``fun``.

    Returns:
        A dense Jacobian. For a scalar-valued function the result has shape
        ``(n,)``; otherwise it has shape ``(m, n)``.

    Raises:
        ValueError: If ``method`` is unsupported, ``x0`` is not one
            dimensional, the bounds have incompatible shapes, or ``x0`` lies
            outside the bounds.

    Notes:
        If a requested step is indistinguishable from ``x0`` in the working
        dtype, an automatic step is substituted for that entry. For the
        ``"3-point"`` method, central differences are used when both sides fit
        within the bounds; otherwise a second-order one-sided difference is
        used.

        Scalar-valued functions return a one-dimensional gradient for
        consistency with common autodiff APIs. Use ``jnp.atleast_2d`` when a
        two-dimensional Jacobian is required in all cases.

    References:
        - W. H. Press et al., *Numerical Recipes: The Art of Scientific
          Computing*, 3rd edition, section 5.7.
        - B. Fornberg, "Generation of Finite Difference Formulas on
          Arbitrarily Spaced Grids," *Mathematics of Computation*, 51, 1988.

    Examples:
        ```python
        import jax.numpy as jnp

        from soromox.utils.numerical_jacobian import approx_derivative

        def f(x, c1, c2):
            return jnp.array(
                [
                    x[0] * jnp.sin(c1 * x[1]),
                    x[0] * jnp.cos(c2 * x[1]),
                ]
            )

        x0 = jnp.array([1.0, 0.5 * jnp.pi])
        jacobian = approx_derivative(f, x0, args=(1, 2))
        # jacobian is approximately [[1, 0], [-1, 0]]
        ```

        Bounds can limit the region of function evaluation:

        ```python
        def g(x):
            return x**2 if x >= 1 else x

        x0 = 1.0
        left_derivative = approx_derivative(g, x0, bounds=(-jnp.inf, 1.0))
        right_derivative = approx_derivative(g, x0, bounds=(1.0, jnp.inf))
        # left_derivative is [1] and right_derivative is [2]
        ```
    """
    x0 = jnp.atleast_1d(jnp.asarray(x0))

    if kwargs is None:
        kwargs = {}
    if method not in ["2-point", "3-point"]:
        raise ValueError(f"Unknown method '{method}'. ")

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = _prepare_bounds(bounds, x0)

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    def fun_wrapped(x):
        return fun(x, *args, **kwargs)

    if f0 is None:
        f0 = fun_wrapped(x0)
    f0 = jnp.atleast_1d(jnp.asarray(f0))

    if jnp.any((x0 < lb) | (x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    if abs_step is None:
        h = _compute_absolute_step(rel_step, x0, f0, method)
    else:
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = abs_step

        # Fall back to a relative step if the requested step is not
        # distinguishable from x0 in the working dtype.
        dx = (x0 + h) - x0
        h = jnp.where(
            dx == 0,
            _eps_for_method(x0.dtype, f0.dtype, method)
            * sign_x0
            * jnp.maximum(1.0, jnp.abs(x0)),
            h,
        )

    if method == "2-point":
        h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "1-sided", lb, ub)
    else:
        h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "2-sided", lb, ub)

    return _dense_difference(fun_wrapped, x0, f0, h, use_one_sided, method)


def _dense_difference(fun, x0, f0, h, use_one_sided, method):
    """Evaluate dense finite differences for prepared steps.

    Args:
        fun: Function to differentiate.
        x0: Evaluation point.
        f0: Function value at ``x0`` with shape ``(m,)``.
        h: Absolute finite-difference steps with shape ``(n,)``.
        use_one_sided: Boolean mask selecting one-sided differences.
        method: Finite-difference method, either ``"2-point"`` or
            ``"3-point"``.

    Returns:
        The dense Jacobian or gradient assembled from the finite differences.
    """
    m = f0.shape[-1]
    n = x0.shape[-1]

    J_T_rows = []
    for i in range(h.size):
        if method == "2-point":
            x1 = x0 + jnp.concatenate(
                [
                    jnp.zeros((i,)),
                    h[i : i + 1],
                    jnp.zeros((n - i - 1,)),
                ],
                axis=-1,
            )
            dx = h[i]
            df = fun(x1) - f0
        elif method == "3-point" and use_one_sided[i]:
            dx01 = jnp.concatenate(
                [
                    jnp.zeros((i,)),
                    h[i : i + 1],
                    jnp.zeros((n - i - 1,)),
                ],
                axis=-1,
            )
            x1 = x0 + dx01
            x2 = x0 + 2 * dx01
            dx = 2 * h[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = -3.0 * f0 + 4 * f1 - f2
        elif method == "3-point" and not use_one_sided[i]:
            dx02 = jnp.concatenate(
                [
                    jnp.zeros((i,)),
                    h[i : i + 1],
                    jnp.zeros((n - i - 1,)),
                ],
                axis=-1,
            )
            x1 = x0 - dx02
            x2 = x0 + dx02
            dx = 2 * h[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1
        else:
            raise RuntimeError("Never be here.")

        J_T_rows.append(df / dx)

    J_T = jnp.stack(J_T_rows, axis=0)

    if m == 1:
        J_T = jnp.ravel(J_T)

    return J_T.T
