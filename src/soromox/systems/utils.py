from copy import deepcopy
import jax

from jax import numpy as jnp
import sympy as sp

# For documentation purposes
from jax import Array
from typing import Dict, List, Tuple, Union


def substitute_params_into_all_symbolic_expressions(
    sym_exps: Dict, params: Dict[str, Array]
) -> Dict:
    """
    Substitute robot parameters into symbolic expressions.
    Args:
        sym_exps: dictionary with entries
            params_syms: dictionary of list with symbols for parameters
            state_syms: dictionary of lists with symbols for state variables
            exps: dictionary of symbolic expressions
        params: dictionary of robot parameters
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]
    # symbolic expressions
    exps = deepcopy(sym_exps["exps"])

    for exp_key, exp_val in exps.items():
        if issubclass(type(exp_val), list):
            for exp_item_idx, exp_item_val in enumerate(exp_val):
                exps[exp_key][exp_item_idx] = (
                    substitute_params_into_single_symbolic_expression(
                        exp_item_val, params_syms, params
                    )
                )
        else:
            exps[exp_key] = substitute_params_into_single_symbolic_expression(
                exp_val, params_syms, params
            )

    return exps


def substitute_params_into_single_symbolic_expression(
    sym_exp: sp.Expr,
    params_syms: Dict[str, List[sp.Symbol]],
    params: Dict[str, Array],
) -> sp.Expr:
    """
    Substitute robot parameters into a single symbolic expression.
    Args:
        sym_exp: symbolic expression
        params_syms: Dictionary of list with symbols for parameters
        params: Dictionary of jax arrays with numerical values for parameters

    Returns:
        sym_exp: symbolic expression with parameters substituted
    """
    for param_key, param_sym in params_syms.items():
        if issubclass(type(param_sym), list):
            for idx, param_sym_item in enumerate(param_sym):
                if param_sym_item in sym_exp.free_symbols:
                    sym_exp = sym_exp.subs(
                        param_sym_item, params[param_key].flatten()[idx]
                    )
        else:
            if param_sym in sym_exp.free_symbols:
                sym_exp = sym_exp.subs(param_sym, params[param_key])

    return sym_exp


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


def compute_planar_stiffness_matrix(
    l: Array, A: Array, Ib: Array, E: Array, G: Array
) -> Array:
    """
    Compute the stiffness matrix of a planar system for a rod aligned along the x-axis.
    
    Args:
        l: length of the segment of shape ()
        A: cross-sectional area of shape ()
        Ib: second moment of area of shape ()
        E: Elastic modulus of shape ()
        G: Shear modulus of shape ()

    Returns:
        S: stiffness matrix of shape (3, 3)
    """
    S = l * jnp.diag(
        jnp.stack(
            [
                E * Ib, # bending Z
                A * E, # axial X
                4 / 3 * A * G, # shear Y
            ], 
            axis=0
        )
    )

    return S


def compute_spatial_stiffness_matrix(
    l: Array, A: Array, Ib: Array, J: Array, E: Array, G: Array
) -> Array:
    """
    Compute the stiffness matrix of a spatial system for a rod aligned along the x-axis.
    Args:
        l: length of the segment of shape ()
        A: cross-sectional area of shape ()
        Ib: second moment of area of shape ()
        J: polar moment of inertia of shape ()
        E: Elastic modulus of shape ()
        G: Shear modulus of shape ()

    Returns:
        S: stiffness matrix of shape (3, 3)
    """
    S = l * jnp.diag(
        jnp.stack(
            [
                G * J,  # torsion X
                E * Ib,  # bending Y
                E * Ib,  # bending Z
                A * E,  # axial X
                4 / 3 * A * G,  # shear Y
                4 / 3 * A * G,  # shear Z
            ],
            axis=0,
        )
    )

    return S


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
