__all__ = [
    "forward_dynamics", "nonlinear_state_space", 
    "concatenate_params_syms", 
    "compute_strain_basis", 
    "gauss_quadrature", "scale_gaussian_quadrature"
]
from copy import deepcopy
from functools import partial
import jax
from jax import Array, jit
from jax import numpy as jnp
import sympy as sp
from typing import Callable, Dict, List, Tuple, Union


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def forward_dynamics(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    qd: Array,
    tau: Array,
):
    """
    Compute the forward dynamics of a Lagrangian system.
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, alpha]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            and alpha is the actuation matrix of shape (n_q, n_tau).
        params: Dictionary with robot parameters
        q: configuration vector of shape (n_q, )
        qd: configuration velocity vector of shape (n_q, )
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        qdd: configuration acceleration vector of shape (n_q, )
    """
    B, C, G, K, D, alpha = dynamical_matrices_fn(params, q, qd)

    # inverse of B
    B_inv = jnp.linalg.inv(B)

    # compute the acceleration
    qdd = B_inv @ (alpha @ tau - C @ qd - G - K - D @ qd)

    return qdd


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def nonlinear_state_space(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    x: Array,
    tau: Array,
) -> jnp.array:
    """
    Compute the nonlinear state space dynamics of a Lagrangian system (i.e. the ODE function).
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, A]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            and alpha is the actuation matrix of shape (n_q, n_tau).
        params: Dictionary with robot parameters
        x: state vector of shape (2 * n_q, ) containing the configuration and velocity vectors
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        xd: state derivative vector of shape (2 * n_q, ) containing the velocity and acceleration vectors
    """
    n_q = x.shape[0] // 2
    qdd = forward_dynamics(dynamical_matrices_fn, params, x[:n_q], x[n_q:], tau)
    xd = jnp.concatenate([x[n_q:], qdd])
    return xd


def concatenate_params_syms(
    params_syms: Dict[str, Union[sp.Symbol, List[sp.Symbol]]],
) -> List[sp.Symbol]:
    # concatenate the robot params symbols
    params_syms_cat = []
    for params_key, params_sym in sorted(params_syms.items()):
        if type(params_sym) in [list, tuple]:
            params_syms_cat += params_sym
        else:
            params_syms_cat.append(params_sym)
    return params_syms_cat


def compute_strain_basis(
    strain_selector: Array,
) -> Array:
    """
    Compute constant strain basis based on boolean strain selector.
    Args:
        strain_selector (Array):
            boolean array of shape (n_xi, ) specifying which strain components are active
    Returns:
        strain_basis (Array):
            strain basis matrix of shape (n_xi, n_q) where n_q is the number of configuration variables
            and n_xi is the number of strains
    """
    n_q = strain_selector.sum().item()
    strain_basis = jnp.zeros((strain_selector.shape[0], n_q))
    strain_basis_cumsum = jnp.cumsum(strain_selector)
    for i in range(strain_selector.shape[0]):
        j = int(strain_basis_cumsum[i].item()) - 1
        if strain_selector[i].item() is True:
            strain_basis = strain_basis.at[i, j].set(1.0)
    return strain_basis

def gauss_quadrature(N_GQ: int, a=0.0, b=1.0) -> Tuple[Array, Array, int]:
    """
    Computes the Legendre-Gauss nodes and weights on the interval [0, 1]
    using Legendre-Gauss Quadrature with truncation order N_GQ.

    Args:
        N_GQ (int): order of the truncature.
        a (float, optional): The lower bound of the interval. Default is 0.0.
        b (float, optional): The upper bound of the interval. Default is 1.0.

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

    def legendre_iteration(y):
        L = [jnp.ones_like(y), y]
        for k in range(2, N1 + 1):
            Lk = ((2 * k - 1) * y * L[-1] - (k - 1) * L[-2]) / k
            L.append(Lk)
        L = jnp.stack(L, axis=1)
        Lp = N2 * (L[:, N1 - 1] - y * L[:, N1]) / (1 - y**2)
        return y - L[:, N1] / Lp

    def convergence_condition(y):
        L = [jnp.ones_like(y), y]
        for k in range(2, N1 + 1):
            Lk = ((2 * k - 1) * y * L[-1] - (k - 1) * L[-2]) / k
            L.append(Lk)
        L = jnp.stack(L, axis=1)
        Lp = N2 * (L[:, N1 - 1] - y * L[:, N1]) / (1 - y**2)
        y_new = y - L[:, N1] / Lp
        return jnp.max(jnp.abs(y_new - y)) > jnp.finfo(jnp.float32).eps

    y = jax.lax.while_loop(  # TODO
        convergence_condition, legendre_iteration, y
    )

    # Linear map from [-1, 1] to [a, b]
    Xs = (a * (1 - y) + b * (1 + y)) / 2
    Xs = jnp.flip(Xs)

    # Add the boundary points
    Xs = jnp.concatenate([jnp.array([a]), Xs, jnp.array([b])])

    # Compute the weights
    L = [jnp.ones_like(y), y]
    for k in range(2, N1 + 1):
        Lk = ((2 * k - 1) * y * L[-1] - (k - 1) * L[-2]) / k
        L.append(Lk)
    L = jnp.stack(L, axis=1)
    Lp = N2 * (L[:, N1 - 1] - y * L[:, N1]) / (1 - y**2)
    Ws = (b - a) / ((1 - y**2) * Lp**2) * (N2 / N1) ** 2

    # Add the boundary points
    Ws = jnp.concatenate([jnp.array([0.0]), Ws, jnp.array([0.0])])

    return Xs, Ws, N_GQ + 2


def scale_gaussian_quadrature(
    Xs: Array, Ws: Array, a: float = 0.0, b: float = 1.0
) -> Tuple[Array, Array]:
    """
    Scale the Gauss nodes and weights from [0, 1] to the interval [a, b].

    Args:
        Xs (Array): The Gauss nodes on [0, 1].
        Ws (Array): The Gauss weights on [0, 1].
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.

    Returns:
        Xs_scaled (Array): The scaled Gauss nodes on [a, b].
        Ws_scaled (Array): The scaled Gauss weights on [a, b].
    """
    Xs_scaled = a + (b - a) * Xs
    Ws_scaled = Ws * (b - a)
    return Xs_scaled, Ws_scaled
