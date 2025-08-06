import dill
import jax
from jax import Array, debug, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

from .utils import concatenate_params_syms


def factory(filepath: Union[str, Path]) -> Tuple[Callable, Callable]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
    Returns:
        forward_kinematics_fn: function that returns the p vector of shape (3, n_q) with the positions
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

    @jit
    def select_params_for_lambdify(params: Dict[str, Array]) -> List[Array]:
        """
        Select the parameters for lambdify
        Args:
            params: Dictionary of robot parameters
        Returns:
            params_for_lambdify: list of with each robot parameter
        """
        params_for_lambdify = []
        for params_key, params_vals in sorted(params.items()):
            if params_key in params_syms.keys():
                for param in params_vals:
                    params_for_lambdify.append(param)
        return params_for_lambdify

    # symbols of state variables
    state_syms = sym_exps["state_syms"]
    # symbolic expressions
    exps = sym_exps["exps"]

    # concatenate the robot params symbols
    params_syms_cat = concatenate_params_syms(params_syms)

    # number of degrees of freedom
    n_q = len(sym_exps["state_syms"]["q"])

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["q"] + sym_exps["state_syms"]["qd"]

    # lambdify symbolic expressions
    chi_lambda_ls = []
    # iterate through symbolic expressions for each segment
    for chi_exp in sym_exps["exps"]["chi_ls"]:
        chi_lambda = sp.lambdify(
            params_syms_cat + sym_exps["state_syms"]["q"], chi_exp, "jax"
        )
        chi_lambda_ls.append(chi_lambda)

    # lambdify symbolic expressions
    B_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["q"], sym_exps["exps"]["B"], "jax"
    )
    C_lambda = sp.lambdify(
        params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
    )
    G_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["q"], sym_exps["exps"]["G"], "jax"
    )

    @jit
    def forward_kinematics_fn(
        params: Dict[str, Array], q: Array, link_idx: Array
    ) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            link_idx: index of link to evaluate with shape ()
        Returns:
            chi: pose of the tip of the link in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)

        chi = lax.switch(link_idx, chi_lambda_ls, *params_for_lambdify, *q).squeeze()

        return chi

    # actuation matrix
    alpha = jnp.identity(n_q)

    @jit
    def dynamical_matrices_fn(
        params: Dict[str, Array], q: Array, qd: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            qd: generalized velocities of shape (n_q, )
        Returns:
            B: mass / inertia matrix of shape (n_q, n_q)
            C: coriolis / centrifugal matrix of shape (n_q, n_q)
            G: gravity vector of shape (n_q, )
            K: elastic vector of shape (n_q, )
            D: dissipative matrix of shape (n_q, n_q)
            alpha: actuation matrix of shape (n_q, n_tau)
        """
        # elastic and dissipative matrices
        K = params.get("K", jnp.zeros((n_q, n_q)))
        D = params.get("D", jnp.zeros((n_q, n_q)))

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)

        B = B_lambda(*params_for_lambdify, *q)
        C = C_lambda(*params_for_lambdify, *q, *qd)
        G = G_lambda(*params_for_lambdify, *q).squeeze()

        # compute elastic matrices as K(q) = K q
        K = K @ q

        return B, C, G, K, D, alpha

    return forward_kinematics_fn, dynamical_matrices_fn


@jit
def normalize_joint_angles(q: Array) -> Array:
    """
    Normalize the joint angles `q` to the interval [-pi, pi].
    """
    q_norm = jnp.mod(q + jnp.pi, 2 * jnp.pi) - jnp.pi
    return q_norm
