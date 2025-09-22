__all__ = [
    "concatenate_params_syms", 
    "compute_strain_basis",
]
from functools import partial
import jax
from jax import Array, jit
from jax import numpy as jnp
import sympy as sp
from typing import Callable, Dict, List, Tuple, Union


def concatenate_params_syms(
    params_syms: Dict[str, Union[sp.Symbol, List[sp.Symbol]]],
) -> List[sp.Symbol]:
    # concatenate the robot params symbols
    params_syms_cat: List[sp.Symbol] = []
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
