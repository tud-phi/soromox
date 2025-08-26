__all__ = ["Pendulum"]
import dill
import equinox as eqx
from jax import Array, jit, lax
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional

# Diffrax (for time integration helpers)
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    PIDController,
    ConstantStepSize,
    AbstractSolver,
)

from .base_system import BaseSystem
from .utils import concatenate_params_syms


class Pendulum(BaseSystem):
    """
    N-link planar pendulum with symbolic kinematics and dynamics.

    Loads symbolic expressions (generated via symbolic_derivation.pendulum)
    and exposes JAX-compiled forward kinematics and dynamics, following
    the class-based style used by other systems.
    """

    # Static fields
    num_links: int = eqx.field(static=True)

    # Parameter cache used for lambdified calls
    params_for_lambdify: List[Array]

    # Lambdified symbolic expressions
    chi_lambda_ls: List[Callable]
    B_lambda: Callable
    C_lambda: Callable
    G_lambda: Callable

    # Optional linear elasticity/damping (defaults to zero)
    K_mat: Array
    D_mat: Array

    def __init__(
        self,
        sym_exp_filepath: Union[str, Path],
        params: Dict[str, Array],
    ) -> None:
        """
        Initialize the Pendulum model from pre-computed symbolic expressions.

        Args:
            sym_exp_filepath (Union[str, Path]):
                Path to the `.dill` file with symbolic expressions.
            params (Dict[str, Array]):
                Dictionary of robot parameters. Expected keys include:
                - "m": masses of each link, shape (n,)
                - "I": inertias of each link, shape (n,)
                - "L": lengths of each link, shape (n,)
                - "Lc": COM distances of each link, shape (n,)
                - "g": planar gravity vector [gx, gy], shape (2,)
                Optional:
                - "K": stiffness matrix, shape (n, n)
                - "D": damping matrix, shape (n, n)
        """
        # Load symbolic expressions
        sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

        # Parameter symbols and state symbols
        params_syms = sym_exps["params_syms"]
        state_syms = sym_exps["state_syms"]

        # rename "Lc" to "lc" for consistency in params_syms
        params_syms["Lc"] = params_syms.pop("lc")

        # Number of links = DoFs
        self.num_links = len(state_syms["q"])  # n_q

        # Build list of parameters in the order required by lambdified functions
        params_for_lambdify: List[Array] = []
        for k, vals in sorted(params.items()):
            if k in params_syms.keys():
                # flatten to ensure scalar unpack works in lambdify calls
                for v in jnp.asarray(vals).flatten():
                    params_for_lambdify.append(v)
        self.params_for_lambdify = params_for_lambdify

        # Concatenate sympy symbols
        params_syms_cat = concatenate_params_syms(params_syms)
        state_syms_cat = state_syms["q"] + state_syms["qd"]

        # Lambdify kinematics for each link tip pose
        chi_lambda_ls: List[Callable] = []
        for chi_exp in sym_exps["exps"]["chi_ls"]:
            chi_lambda = sp.lambdify(
                params_syms_cat + state_syms["q"], chi_exp, "jax"
            )
            chi_lambda_ls.append(chi_lambda)
        self.chi_lambda_ls = chi_lambda_ls

        # Lambdify dynamics
        self.B_lambda = sp.lambdify(
            params_syms_cat + state_syms["q"], sym_exps["exps"]["B"], "jax"
        )
        self.C_lambda = sp.lambdify(
            params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
        )
        self.G_lambda = sp.lambdify(
            params_syms_cat + state_syms["q"], sym_exps["exps"]["G"], "jax"
        )

        # Optional linear elasticity and damping matrices
        n_q = self.num_links
        self.K_mat = jnp.asarray(params.get("K", jnp.zeros((n_q, n_q))))
        self.D_mat = jnp.asarray(params.get("D", jnp.zeros((n_q, n_q))))

    def update_params(self, params: Dict[str, Array]) -> "Pendulum":
        """
        Update internal parameters.

        Supports updating physical parameters (m, I, L, Lc, g) and linear
        stiffness/damping matrices (K, D). Returns an updated eqx tree for
        JIT-friendliness (functional style).

        Args:
            params (Dict[str, Array]): Dictionary of parameters to update.

        Returns:
            updated (Pendulum): A new instance with updated fields.
        """
        # Rebuild params_for_lambdify list
        new_params_for_lambdify: List[Array] = []
        # Conservative approach: keep ordering consistent by sorting keys
        for v in params.values():
            pass  # placeholder to satisfy potential empty loop in tracing
        # Since lambdify ordering depends on original params_syms, for safety
        # require caller to pass the full set of physical params when changing
        # them. Otherwise, only update K/D.
        updated = self
        if any(k in params for k in ["m", "I", "L", "Lc", "g"]):
            # If physical parameters change, rebuild the flattened list from
            # available values; missing keys would lead to wrong ordering, so
            # we expect the caller to provide all physical keys.
            for k, vals in sorted(params.items()):
                if k in ["m", "I", "L", "Lc", "g"]:
                    for vv in jnp.asarray(vals).flatten():
                        new_params_for_lambdify.append(vv)
            updated = eqx.tree_at(
                lambda x: x.params_for_lambdify, updated, new_params_for_lambdify
            )
        if "K" in params:
            updated = eqx.tree_at(
                lambda x: x.K_mat, updated, jnp.asarray(params["K"])
            )
        if "D" in params:
            updated = eqx.tree_at(
                lambda x: x.D_mat, updated, jnp.asarray(params["D"])
            )
        return updated

    @eqx.filter_jit
    def forward_kinematics(self, q: Array, link_idx: Array) -> Array:
        """
        Compute the forward kinematics at a given link.

        Args:
            q (Array): Joint angles of shape (n_q,).
            link_idx (Array): Link index (0-based), scalar array.
                Use `n_q - 1` to query the end-effector.

        Returns:
            chi (Array): Pose [p_x, p_y, theta], shape (3,).
        """
        q_list = [q[i] for i in range(self.num_links)]
        chi = lax.switch(
            link_idx, self.chi_lambda_ls, *self.params_for_lambdify, *q_list
        )
        return chi.squeeze()

    # -------------------------------
    # Standardized dynamics interface
    # -------------------------------

    def mass_matrix(self, q: Array) -> Array:
        """
        Compute the mass (inertia) matrix of the pendulum.

        Args:
            q (Array): Joint angles of shape (n_q,).

        Returns:
            B (Array): Inertia matrix of shape (n_q, n_q).
        """
        q_list = [q[i] for i in range(self.num_links)]
        return self.B_lambda(*self.params_for_lambdify, *q_list)

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the Coriolis/centrifugal matrix of the pendulum.

        Args:
            q (Array): Joint angles of shape (n_q,).
            qd (Array): Joint velocities of shape (n_q,).

        Returns:
            C (Array): Coriolis matrix of shape (n_q, n_q).
        """
        q_list = [q[i] for i in range(self.num_links)]
        qd_list = [qd[i] for i in range(self.num_links)]
        return self.C_lambda(*self.params_for_lambdify, *q_list, *qd_list)

    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational generalized forces.

        Args:
            q (Array): Joint angles of shape (n_q,).

        Returns:
            G (Array): Gravity vector of shape (n_q,).
        """
        q_list = [q[i] for i in range(self.num_links)]
        return self.G_lambda(*self.params_for_lambdify, *q_list).squeeze()

    def stiffness_matrix(self) -> Array:
        """
        Return the linear stiffness matrix.

        Returns:
            K (Array): Stiffness matrix of shape (n_q, n_q).
        """
        return self.K_mat

    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic generalized forces.

        Args:
            q (Array): Joint angles of shape (n_q,).

        Returns:
            Kq (Array): Elastic force K @ q of shape (n_q,).
        """
        return self.K_mat @ q

    def damping_matrix(self) -> Array:
        """
        Return the linear viscous damping matrix.

        Returns:
            D (Array): Damping matrix of shape (n_q, n_q).
        """
        return self.D_mat

    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix (identity for fully actuated joints).

        Args:
            q (Array): Joint angles of shape (n_q,).

        Returns:
            A (Array): Actuation matrix of shape (n_q, n_q).
        """
        return jnp.eye(self.num_links)

    def actuation_force(self, q: Array, u: Array) -> Array:
        """
        Compute the generalized actuation forces.

        Args:
            q (Array): Joint angles of shape (n_q,).
            u (Array): Actuation inputs (torques) of shape (n_q,).

        Returns:
            tau_u (Array): Generalized forces of shape (n_q,).
        """
        return u

    @eqx.filter_jit
    def dynamical_matrices(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the standard dynamical matrices of the pendulum.

        Args:
            q (Array): Joint angles of shape (n_q,).
            qd (Array): Joint velocities of shape (n_q,).

        Returns:
            B (Array): Inertia matrix, shape (n_q, n_q).
            C (Array): Coriolis/centrifugal matrix, shape (n_q, n_q).
            G (Array): Gravity vector, shape (n_q,).
            Kq (Array): Elastic force K @ q, shape (n_q,).
            D (Array): Damping matrix, shape (n_q, n_q).
            A (Array): Actuation matrix, shape (n_q, n_q).
        """
        B = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        K = self.elastic_force(q)
        D = self.damping_matrix()
        A = self.actuation_matrix(q)
        return B, C, G, K, D, A
    
    def forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: Optional[Tuple] = None,
    ) -> Array:
        """
        Forward dynamics function in state space.

        Computes yd = [qd, qdd] given the current state and actuation.

        Args:
            t (Array): Current time.
            y (Array): State vector [q, qd] of shape (2 * n_q,).
            actuation_args (Optional[Tuple]):
                None -> u=0, tau_ext=0;
                (u,) -> tau_ext=0;
                (u, tau_ext) -> both provided.

        Returns:
            yd (Array): Time derivative of the state, shape (2 * n_q,).
        """
        q, qd = jnp.split(y, 2)

        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u, tau_ext = actuation_args[0], None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be None, (u,), or (u, tau_ext)")

        if u is None:
            u = jnp.zeros((self.num_links,))
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_links,))

        B = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        D = self.damping_matrix()
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        qdd = jnp.linalg.inv(B) @ (tau_u + tau_ext - C @ qd - G - tau_el - D @ qd)
        return jnp.concatenate([qd, qdd])


@jit
def normalize_joint_angles(q: Array) -> Array:
    """
    Normalize joint angles to the interval [-pi, pi].

    Args:
        q (Array): Joint angles.

    Returns:
        q_norm (Array): Normalized joint angles.
    """
    q_norm = jnp.mod(q + jnp.pi, 2 * jnp.pi) - jnp.pi
    return q_norm
