__all__ = ["gauss_quadrature", "scale_gaussian_quadrature"]
from jax import numpy as jnp
from jax import lax

# For documentation purposes
from jax import Array
from typing import Tuple


def gauss_quadrature(N_GQ: int, a: Array = jnp.zeros(()), b: Array = jnp.ones(())) -> Tuple[Array, Array, int]:
    """
    Computes the Legendre-Gauss nodes and weights on the interval [0, 1]
    using Legendre-Gauss Quadrature with truncation order N_GQ.

    Args:
        N_GQ (int): order of the truncature.
        a (Array, optional): The lower bound of the interval. Default is 0.0.
        b (Array, optional): The upper bound of the interval. Default is 1.0.

    Returns:
        Xs (Array): The Gauss nodes on [a, b].
        Ws (Array): The Gauss weights on [a, b].
        nGauss (int): The number of Gauss points including boundary points, i.e., N_GQ + 2.
    """

    N = N_GQ - 1
    N1 = N + 1
    N2 = N + 2

    xu = jnp.linspace(-1, 1, N1)

    # Initial guess
    y = jnp.cos((2 * jnp.arange(N + 1) + 1) * jnp.pi / (2 * N + 2)) + (
        0.27 / N1
    ) * jnp.sin(jnp.pi * xu * N / N2)

    def legendre_iteration(y: Array) -> Array:
        """
        Perform one Newton–Raphson update for the Legendre roots.

        Builds Legendre polynomials up to order `N1` at the current points `y`,
        evaluates their derivative via the stable identity using `N2`, and
        returns the refined estimate `y_new = y - P_N(y)/P_N'(y)`.

        Args:
            y (Array): Current estimates of the roots (shape: [...]).

        Returns:
            y_next (Array): Updated root estimates after a single iteration (same shape as `y`).
        """
        _L_ls = [jnp.ones_like(y), y]
        for k in range(2, N1 + 1):
            Lk = ((2 * k - 1) * y * _L_ls[-1] - (k - 1) * _L_ls[-2]) / k
            _L_ls.append(Lk)
        _L = jnp.stack(_L_ls, axis=1)
        Lp = N2 * (_L[:, N1 - 1] - y * _L[:, N1]) / (1 - y**2)
        y_next = y - _L[:, N1] / Lp
        return y_next

    def convergence_condition(y: Array) -> Array:
        """
        Convergence predicate for the Newton iterations in Gauss–Legendre.

        Computes the next Newton update and checks if the maximum absolute
        change across all entries is larger than machine epsilon (float32).
        Used as the loop condition in `lax.while_loop`.

        Args:
            y (Array): Current estimates of the roots (shape: [...]).

        Returns:
            needs_more_its (Array): Boolean array of shape () that contains True if another iteration is required; False otherwise.
        """
        L_ls_ = [jnp.ones_like(y), y]
        for k in range(2, N1 + 1):
            Lk = ((2 * k - 1) * y * L_ls_[-1] - (k - 1) * L_ls_[-2]) / k
            L_ls_.append(Lk)
        L_ = jnp.stack(L_ls_, axis=1)
        Lp = N2 * (L_[:, N1 - 1] - y * L_[:, N1]) / (1 - y**2)
        y_new = y - L_[:, N1] / Lp
        needs_more_its = jnp.max(jnp.abs(y_new - y)) > jnp.finfo(jnp.float32).eps
        return needs_more_its

    y = lax.while_loop(
        convergence_condition, legendre_iteration, y
    )

    # Linear map from [-1, 1] to [a, b]
    Xs = (a * (1 - y) + b * (1 + y)) / 2
    Xs = jnp.flip(Xs)

    # Add the boundary points
    Xs = jnp.concatenate([jnp.array([a]), Xs, jnp.array([b])])

    # Compute the weights
    L_ls = [jnp.ones_like(y), y]
    for k in range(2, N1 + 1):
        Lk = ((2 * k - 1) * y * L_ls[-1] - (k - 1) * L_ls[-2]) / k
        L_ls.append(Lk)
    L = jnp.stack(L_ls, axis=1)
    Lp = N2 * (L[:, N1 - 1] - y * L[:, N1]) / (1 - y**2)
    Ws = (b - a) / ((1 - y**2) * Lp**2) * (N2 / N1) ** 2

    # Add the boundary points
    Ws = jnp.concatenate([jnp.zeros((1, )), Ws, jnp.zeros((1, ))], axis=0)

    return Xs, Ws, N_GQ + 2


def scale_gaussian_quadrature(
    Xs: Array, Ws: Array, a: Array = jnp.zeros(()), b: Array = jnp.ones(())
) -> Tuple[Array, Array]:
    """
    Scale the Gauss nodes and weights from [0, 1] to the interval [a, b].

    Args:
        Xs (Array): The Gauss nodes on [0, 1].
        Ws (Array): The Gauss weights on [0, 1].
        a (Array): The lower bound of the interval.
        b (Array): The upper bound of the interval.

    Returns:
        Xs_scaled (Array): The scaled Gauss nodes on [a, b].
        Ws_scaled (Array): The scaled Gauss weights on [a, b].
    """
    Xs_scaled = a + (b - a) * Xs
    Ws_scaled = Ws * (b - a)
    return Xs_scaled, Ws_scaled
