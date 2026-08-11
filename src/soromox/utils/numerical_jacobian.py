"""Routines for numerical differentiation."""

__all__ = ["approx_derivative"]
import functools

import jax.numpy as jnp


def _adjust_scheme_to_bounds(x0, h, num_steps, scheme, lb, ub):
    """Adjust final difference scheme to the presence of bounds.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Point at which we wish to estimate derivative.
    h : ndarray, shape (n,)
        Desired absolute finite difference steps.
    num_steps : int
        Number of `h` steps in one direction required to implement finite
        difference scheme. For example, 2 means that we need to evaluate
        f(x0 + 2 * h) or f(x0 - 2 * h)
    scheme : {'1-sided', '2-sided'}
        Whether steps in one or both directions are required. In other
        words '1-sided' applies to forward and backward schemes, '2-sided'
        applies to center schemes.
    lb : ndarray, shape (n,)
        Lower bounds on independent variables.
    ub : ndarray, shape (n,)
        Upper bounds on independent variables.

    Returns
    -------
    h_adjusted : ndarray, shape (n,)
        Adjusted absolute step sizes. Step size decreases only if a sign flip
        or switching to one-sided scheme doesn't allow to take a full step.
    use_one_sided : ndarray of bool, shape (n,)
        Whether to switch to one-sided scheme. Informative only for
        ``scheme='2-sided'``.
    """
    if scheme == "1-sided":
        use_one_sided = jnp.ones_like(h, dtype=bool)
    elif scheme == "2-sided":
        h = jnp.abs(h)
        use_one_sided = jnp.zeros_like(h, dtype=bool)
    else:
        raise ValueError("`scheme` must be '1-sided' or '2-sided'.")

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
        h_adjusted[violated & fitting] *= -1

        forward = (upper_dist >= lower_dist) & ~fitting
        h_adjusted[forward] = upper_dist[forward] / num_steps
        backward = (upper_dist < lower_dist) & ~fitting
        h_adjusted[backward] = -lower_dist[backward] / num_steps
    elif scheme == "2-sided":
        central = (lower_dist >= h_total) & (upper_dist >= h_total)

        forward = (upper_dist >= lower_dist) & ~central
        h_adjusted[forward] = jnp.minimum(
            h[forward], 0.5 * upper_dist[forward] / num_steps
        )
        use_one_sided[forward] = True

        backward = (upper_dist < lower_dist) & ~central
        h_adjusted[backward] = -jnp.minimum(
            h[backward], 0.5 * lower_dist[backward] / num_steps
        )
        use_one_sided[backward] = True

        min_dist = jnp.minimum(upper_dist, lower_dist) / num_steps
        adjusted_central = ~central & (jnp.abs(h_adjusted) <= min_dist)
        h_adjusted[adjusted_central] = min_dist[adjusted_central]
        use_one_sided[adjusted_central] = False

    return h_adjusted, use_one_sided


@functools.lru_cache
def _eps_for_method(x0_dtype, f0_dtype, method):
    """
    Calculates relative EPS step to use for a given data type
    and numdiff step method.

    Progressively smaller steps are used for larger floating point types.

    Parameters
    ----------
    f0_dtype: jnp.dtype
        dtype of function evaluation

    x0_dtype: jnp.dtype
        dtype of parameter vector

    method: {'2-point', '3-point'}

    Returns
    -------
    EPS: float
        relative step size. May be jnp.float16, jnp.float32, jnp.float64

    Notes
    -----
    The default relative step will be jnp.float64. However, if x0 or f0 are
    smaller floating point types (jnp.float16, jnp.float32), then the smallest
    floating point type is chosen.
    """
    # the default EPS value
    EPS = jnp.finfo(jnp.float64).eps

    x0_is_fp = False
    if jnp.issubdtype(x0_dtype, jnp.inexact):
        # if you're a floating point type then over-ride the default EPS
        EPS = jnp.finfo(x0_dtype).eps
        x0_itemsize = jnp.dtype(x0_dtype).itemsize
        x0_is_fp = True

    if jnp.issubdtype(f0_dtype, jnp.inexact):
        f0_itemsize = jnp.dtype(f0_dtype).itemsize
        # choose the smallest itemsize between x0 and f0
        if x0_is_fp and f0_itemsize < x0_itemsize:
            EPS = jnp.finfo(f0_dtype).eps

    if method in ["2-point"]:
        return EPS**0.5
    elif method in ["3-point"]:
        return EPS ** (1 / 3)
    else:
        raise RuntimeError(
            "Unknown step method, should be one of {'2-point', '3-point'}"
        )


def _compute_absolute_step(rel_step, x0, f0, method):
    """
    Computes an absolute step from a relative step for finite difference
    calculation.

    Parameters
    ----------
    rel_step: None or array-like
        Relative step for the finite difference calculation
    x0 : jnp.ndarray
        Parameter vector
    f0 : jnp.ndarray or scalar
    method : {'2-point', '3-point'}

    Returns
    -------
    h : float
        The absolute step size

    Notes
    -----
    `h` will always be jnp.float64. However, if `x0` or `f0` are
    smaller floating point dtypes (e.g. jnp.float32), then the absolute
    step size will be calculated from the smallest floating point size.
    """
    # this is used instead of jnp.sign(x0) because we need
    # sign_x0 to be 1 when x0 == 0.
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1

    rstep = _eps_for_method(x0.dtype, f0.dtype, method)

    if rel_step is None:
        abs_step = rstep * sign_x0 * jnp.maximum(1.0, jnp.abs(x0))
    else:
        # User has requested specific relative steps.
        # Don't multiply by max(1, abs(x0) because if x0 < 1 then their
        # requested step is not used.
        abs_step = rel_step * sign_x0 * jnp.abs(x0)

        # however we don't want an abs_step of 0, which can happen if
        # rel_step is 0, or x0 is 0. Instead, substitute a realistic step
        dx = (x0 + abs_step) - x0
        abs_step = jnp.where(
            dx == 0, rstep * sign_x0 * jnp.maximum(1.0, jnp.abs(x0)), abs_step
        )

    return abs_step


def _prepare_bounds(bounds, x0):
    """
    Prepares new-style bounds from a two-tuple specifying the lower and upper
    limits for values in x0. If a value is not bound then the lower/upper bound
    will be expected to be -jnp.inf/jnp.inf.

    Examples
    --------
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
    fun,
    x0,
    method="3-point",
    rel_step=None,
    abs_step=None,
    f0=None,
    bounds=(-jnp.inf, jnp.inf),
    args=(),
    kwargs=None,
):
    """Compute finite difference approximation of the derivatives of a
    vector-valued function.

    If a function maps from R^n to R^m, its derivatives form m-by-n matrix
    called the Jacobian, where an element (i, j) is a partial derivative of
    f[i] with respect to x[j].

    Parameters
    ----------
    fun : callable
        Function of which to estimate the derivatives. The argument x
        passed to this function is ndarray of shape (n,) (never a scalar
        even if n=1). It must return 1-D array_like of shape (m,) or a scalar.
    x0 : array_like of shape (n,) or float
        Point at which to estimate the derivatives. Float will be converted
        to a 1-D array.
    method : {'3-point', '2-point'}, optional
        Finite difference method to use:
            - '2-point' - use the first order accuracy forward or backward
                          difference.
            - '3-point' - use central difference in interior points and the
                          second order accuracy forward or backward difference
                          near the boundary.
    rel_step : None or array_like, optional
        Relative step size to use. If None (default) the absolute step size is
        computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``, with
        `rel_step` being selected automatically, see Notes. Otherwise
        ``h = rel_step * sign(x0) * abs(x0)``. For ``method='3-point'`` the
        sign of `h` is ignored. The calculated step size is possibly adjusted
        to fit into the bounds.
    abs_step : array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `abs_step` is ignored. By default
        relative steps are used, only if ``abs_step is not None`` are absolute
        steps used.
    f0 : None or array_like, optional
        If not None it is assumed to be equal to ``fun(x0)``, in this case
        the ``fun(x0)`` is not called. Default is None.
    bounds : tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each bound must match the size of `x0` or be a scalar, in the latter
        case the bound will be the same for all variables. Use it to limit the
        range of function evaluation.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------
    J : {ndarray}
        Finite difference approximation of the Jacobian matrix.
        It returns a dense ndarray with shape (m, n) is returned. If
        m=1 it is returned as a 1-D gradient array with shape (n,).

    See Also
    --------
    check_derivative : Check correctness of a function computing derivatives.

    Notes
    -----
    If `rel_step` is not provided, it assigned as ``EPS**(1/s)``, where EPS is
    determined from the smallest floating point dtype of `x0` or `fun(x0)`,
    ``jnp.finfo(x0.dtype).eps``, s=2 for '2-point' method and
    s=3 for '3-point' method. Such relative step approximately minimizes a sum
    of truncation and round-off errors, see [1]_. Relative steps are used by
    default. However, absolute steps are used when ``abs_step is not None``.
    If any of the absolute or relative steps produces an indistinguishable
    difference from the original `x0`, ``(x0 + dx) - x0 == 0``, then a
    automatic step size is substituted for that particular entry.

    A finite difference scheme for '3-point' method is selected automatically.
    The well-known central difference scheme is used for points sufficiently
    far from the boundary, and 3-point forward or backward scheme is used for
    points near the boundary. Both schemes have the second-order accuracy in
    terms of Taylor expansion. Refer to [2]_ for the formulas of 3-point
    forward and backward difference schemes.

    For dense differencing when m=1 Jacobian is returned with a shape (n,),
    on the other hand when n=1 Jacobian is returned with a shape (m, 1).
    Our motivation is the following: a) It handles a case of gradient
    computation (m=1) in a conventional way. b) It clearly separates these two
    different cases. b) In all cases jnp.atleast_2d can be called to get 2-D
    Jacobian with correct dimensions.

    References
    ----------
    .. [1] W. H. Press et. al. "Numerical Recipes. The Art of Scientific
           Computing. 3rd edition", sec. 5.7.

    .. [3] B. Fornberg, "Generation of Finite Difference Formulas on
           Arbitrarily Spaced Grids", Mathematics of Computation 51, 1988.

    Examples
    --------
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

    Bounds can be used to limit the region of function evaluation.
    In the example below we compute left and right derivative at point 1.0.

    ```python
    def g(x):
        return x**2 if x >= 1 else x

    x0 = 1.0
    left_derivative = approx_derivative(g, x0, bounds=(-jnp.inf, 1.0))
    right_derivative = approx_derivative(g, x0, bounds=(1.0, jnp.inf))
    # left_derivative is [1] and right_derivative is [2]
    ```
    """
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
        f = fun(x, *args, **kwargs)
        return f

    if f0 is None:
        f0 = fun_wrapped(x0)

    if jnp.any((x0 < lb) | (x0 > ub)):
        raise ValueError("`x0` violates bound constraints.")

    # by default we use rel_step
    if abs_step is None:
        h = _compute_absolute_step(rel_step, x0, f0, method)
    else:
        # user specifies an absolute step
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = abs_step

        # cannot have a zero step. This might happen if x0 is very large
        # or small. In which case fall back to relative step.
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
    elif method == "3-point":
        h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "2-sided", lb, ub)

    return _dense_difference(fun_wrapped, x0, f0, h, use_one_sided, method)


def _dense_difference(fun, x0, f0, h, use_one_sided, method):
    m = f0.shape[-1]
    n = x0.shape[-1]

    J_T_rows = []
    for i in range(h.size):
        if method == "2-point":
            x1 = x0 + jnp.concatenate(
                [
                    jnp.zeros(
                        (i,),
                    ),
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

    J = J_T.T

    return J
