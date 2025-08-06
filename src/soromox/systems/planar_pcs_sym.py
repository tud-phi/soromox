import dill
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
import numpy as onp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional

from .utils import (
    concatenate_params_syms,
    compute_strain_basis,
    compute_planar_stiffness_matrix,
)
from soromox.math_utils import blk_diag


def factory(
    filepath: Union[str, Path],
    strain_selector: Array = None,
    xi_ref: Optional[Array] = None,
    stiffness_fn: Optional[Callable] = None,
    actuation_mapping_fn: Optional[Callable] = None,
    global_eps: float = 1e-6,
) -> Tuple[
    Array,
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[
        [Dict[str, Array], Array, Array],
        Tuple[Array, Array, Array, Array, Array, Array],
    ],
    Dict[str, Callable],
]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
        strain_selector: array of shape (n_xi, ) with boolean values indicating which components of the
                strain are active / non-zero
        xi_ref: array of shape (3 * num_segments) with the reference strains of the rod
        stiffness_fn: function to compute the stiffness matrix of the system. Should have the signature
            stiffness_fn(params: Dict[str, Array], B_xi, formulate_in_strain_space: bool) -> Array
        actuation_mapping_fn: function to compute the actuation matrix that maps the actuation space to the
            configuration space.
        global_eps: small number to avoid singularities (e.g., division by zero)
    Returns:
        B_xi: strain basis matrix of shape (3 * num_segments, n_q)
        forward_kinematics_fn: function that returns the p vector of shape (3, n_q) with the positions
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
        auxiliary_fns: dictionary with auxiliary functions
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

    def select_params_for_lambdify_fn(params: Dict[str, Array]) -> List[Array]:
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
                for param in params_vals.flatten():
                    params_for_lambdify.append(param)
        return params_for_lambdify

    # concatenate the robot params symbols
    params_syms_cat = concatenate_params_syms(params_syms)

    # number of degrees of freedom
    n_xi = len(sym_exps["state_syms"]["xi"])

    # compute the strain basis
    if strain_selector is None:
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    else:
        assert strain_selector.shape == (n_xi,)
    B_xi = compute_strain_basis(strain_selector)

    # initialize the reference strain
    if xi_ref is None:
        xi_ref = jnp.zeros((n_xi,))
        # by default, set the axial reference strain (local y-axis) along the entire rod to 1.0
        ref_strain_reshaped = xi_ref.reshape((-1, 3))
        ref_strain_reshaped = ref_strain_reshaped.at[:, -2].set(1.0)
        
        xi_ref = ref_strain_reshaped.flatten()
    else:
        assert xi_ref.shape == (n_xi,)

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"]

    # lambdify symbolic expressions
    chi_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chi_exp in sym_exps["exps"]["chi_sms"]:
        chi_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            chi_exp,
            "jax",
        )
        chi_lambda_sms.append(chi_lambda)
    J_lambda_sms = []
    for J_exp in sym_exps["exps"]["J_sms"]:
        J_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            J_exp,
            "jax",
        )
        J_lambda_sms.append(J_lambda)
    Jd_lambda_sms = []
    for Jd_exp in sym_exps["exps"]["Jd_sms"]:
        Jd_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + sym_exps["state_syms"]["xid"]
            + [sym_exps["state_syms"]["s"]],
            Jd_exp,
            "jax",
        )
        Jd_lambda_sms.append(Jd_lambda)

    B_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["B"], "jax"
    )
    C_lambda = sp.lambdify(
        params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
    )
    G_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["G"], "jax"
    )
    U_g_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["U_g"], "jax"
    )

    compute_stiffness_matrix_for_all_segments_fn = vmap(compute_planar_stiffness_matrix)

    def apply_eps_to_bend_strains(xi: Array, _eps: float) -> Array:
        """
        Add a small number to the bending strain to avoid singularities
        """
        if _eps == None:
            return xi
        else:
            xi_reshaped = xi.reshape((-1, 3))

            xi_bend_sign = jnp.sign(xi_reshaped[:, 0])
            # set zero sign to 1 (i.e. positive)
            xi_bend_sign = jnp.where(xi_bend_sign == 0, 1, xi_bend_sign)
            # add eps to the bending strain (i.e. the first column)
            sigma_b_epsed = lax.select(
                jnp.abs(xi_reshaped[:, 0]) < _eps,
                xi_bend_sign * _eps,
                xi_reshaped[:, 0],
            )
            xi_epsed = jnp.stack(
                [
                    sigma_b_epsed,
                    xi_reshaped[:, 1],
                    xi_reshaped[:, 2],
                ],
                axis=1,
            )

            # old implementation:
            # xi_epsed = xi_reshaped
            # xi_epsed = xi_epsed.at[:, 0].add(xi_bend_sign * _eps)

            # flatten the array
            xi_epsed = xi_epsed.flatten()

        return xi_epsed

    def classify_segment(params: Dict[str, Array], s: Array) -> Tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment
        Args:
            params: Dictionary of robot parameters
            s: point coordinate along the robot in the interval [0, L].
        Returns:
            segment_idx: index of the segment
            s_segment: point coordinate along the segment in the interval [0, l_segment
        """
        # cumsum of the segment lengths
        l_cum = jnp.cumsum(params["l"])
        # add zero to the beginning of the array
        l_cum_padded = jnp.concatenate([jnp.array([0.0]), l_cum], axis=0)
        # determine in which segment the point is located
        # use argmax to find the last index where the condition is true
        segment_idx = (
            l_cum.shape[0] - 1 - jnp.argmax((s >= l_cum_padded[:-1])[::-1]).astype(int)
        )
        # point coordinate along the segment in the interval [0, l_segment]
        s_segment = s - l_cum_padded[segment_idx]

        return segment_idx, s_segment

    if stiffness_fn is None:

        def stiffness_fn(
            params: Dict[str, Array],
            B_xi: Array,
            formulate_in_strain_space: bool = False,
        ) -> Array:
            """
            Compute the stiffness matrix of the system.
            Args:
                params: Dictionary of robot parameters
                B_xi: Strain basis matrix
                formulate_in_strain_space: whether to formulate the elastic matrix in the strain space
            Returns:
                S: elastic matrix of shape (n_q, n_q) if formulate_in_strain_space is False or (n_xi, n_xi) otherwise
            """
            # length of the segments
            l = params["l"]
            # cross-sectional area and second moment of area
            A = jnp.pi * params["r"] ** 2
            Ib = A**2 / (4 * jnp.pi)

            # elastic and shear modulus
            E, G = params["E"], params["G"]
            # stiffness matrix of shape (num_segments, 3, 3)
            S_sms = compute_stiffness_matrix_for_all_segments_fn(l, A, Ib, E, G)
            # we define the elastic matrix of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
            S = blk_diag(S_sms)

            if not formulate_in_strain_space:
                S = B_xi.T @ S @ B_xi

            return S

    if actuation_mapping_fn is None:

        def actuation_mapping_fn(
            forward_kinematics_fn: Callable,
            jacobian_fn: Callable,
            params: Dict[str, Array],
            B_xi: Array,
            xi_ref: Array,
            q: Array,
        ) -> Array:
            """
            Returns the actuation matrix that maps the actuation space to the configuration space.
            Assumes the fully actuated and identity actuation matrix.
            Args:
                forward_kinematics_fn: function to compute the forward kinematics
                jacobian_fn: function to compute the Jacobian
                params: dictionary with robot parameters
                B_xi: strain basis matrix
                xi_ref: reference strains as array of shape (n_xi,)
                q: configuration of the robot
            Returns:
                A: actuation matrix of shape (n_xi, n_xi) where n_xi is the number of strains.
            """
            A = B_xi.T @ jnp.identity(n_xi) @ B_xi

            return A

    def forward_kinematics_fn(
        params: Dict[str, Array], q: Array, s: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            chi: pose of the backbone point in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = xi_ref + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # classify the point along the robot to the corresponding segment
        segment_idx, s_segment = classify_segment(params, s)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chi = lax.switch(
            segment_idx, chi_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()

        return chi

    def jacobian_fn(
        params: Dict[str, Array], q: Array, s: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            J: Jacobian matrix of shape (3, n_q) of the backbone point in Cartesian-space
                Relates the configuration-space velocity qd to the Cartesian-space velocity chid,
                where chid = J @ qd. Chid consists of [p_xd, p_yd, thetad]
        """
        # map the configuration to the strains
        xi = xi_ref + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # classify the point along the robot to the corresponding segment
        segment_idx, s_segment = classify_segment(params, s)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        J = lax.switch(
            segment_idx, J_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()

        # apply the strain basis to the Jacobian
        J = J @ B_xi

        return J

    def dynamical_matrices_fn(
        params: Dict[str, Array], q: Array, qd: Array, eps: float = 1e4 * global_eps
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            qd: generalized velocities of shape (n_q, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            B: mass / inertia matrix of shape (n_q, n_q)
            C: coriolis / centrifugal matrix of shape (n_q, n_q)
            G: gravity vector of shape (n_q, )
            K: elastic vector of shape (n_q, )
            D: dissipative matrix of shape (n_q, n_q)
            alpha: actuation matrix of shape (n_q, n_tau)
        """
        # map the configuration to the strains
        xi = xi_ref + B_xi @ q
        xid = B_xi @ qd

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # compute the stiffness matrix
        K = stiffness_fn(params, B_xi, formulate_in_strain_space=True)
        # compute the actuation matrix
        A = actuation_mapping_fn(
            forward_kinematics_fn, jacobian_fn, params, B_xi, xi_ref, q
        )

        # dissipative matrix from the parameters
        D = params.get("D", jnp.zeros((n_xi, n_xi)))

        params_for_lambdify = select_params_for_lambdify_fn(params)

        B = B_xi.T @ B_lambda(*params_for_lambdify, *xi_epsed) @ B_xi
        C_xi = C_lambda(*params_for_lambdify, *xi_epsed, *xid)
        C = B_xi.T @ C_xi @ B_xi
        G = B_xi.T @ G_lambda(*params_for_lambdify, *xi_epsed).squeeze()

        # apply the strain basis to the elastic and dissipative matrices
        K = B_xi.T @ K @ (xi - xi_ref)  # evaluate K(xi) = K @ xi
        D = B_xi.T @ D @ B_xi

        # apply the strain basis to the actuation matrix
        alpha = A

        return B, C, G, K, D, alpha

    def kinetic_energy_fn(params: Dict[str, Array], q: Array, qd: Array) -> Array:
        """
        Compute the kinetic energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            qd: generalized velocities of shape (n_q, )
        Returns:
            T: kinetic energy of shape ()
        """
        B, C, G, K, D, alpha = dynamical_matrices_fn(params, q=q, qd=qd)

        # kinetic energy
        T = (0.5 * qd.T @ B @ qd).squeeze()

        return T

    def potential_energy_fn(
        params: Dict[str, Array], q: Array, eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the potential energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            U: potential energy of shape ()
        """
        # map the configuration to the strains
        xi = xi_ref + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # compute the stiffness matrix
        K = stiffness_fn(params, B_xi, formulate_in_strain_space=True)
        # elastic energy
        U_K = 0.5 * (xi - xi_ref).T @ K @ (xi - xi_ref)  # evaluate K(xi) = K @ xi

        # gravitational potential energy
        params_for_lambdify = select_params_for_lambdify_fn(params)
        U_G = U_g_lambda(*params_for_lambdify, *xi_epsed)

        # total potential energy
        U = (U_G + U_K).squeeze()

        return U

    def energy_fn(params: Dict[str, Array], q: Array, qd: Array) -> Array:
        """
        Compute the total energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            qd: generalized velocities of shape (n_q, )
        Returns:
            E: total energy of shape ()
        """
        T = kinetic_energy_fn(params, q, qd)
        U = potential_energy_fn(params, q)
        E = T + U

        return E

    def operational_space_dynamical_matrices_fn(
        params: Dict[str, Array],
        q: Array,
        qd: Array,
        s: Array,
        B: Array,
        C: Array,
        operational_space_selector: Tuple = (True, True, True),
        eps: float = 1e4 * global_eps,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the dynamics in operational space.
        The implementation is based on Chapter 7.8 of "Modelling, Planning and Control of Robotics" by
        Siciliano, Sciavicco, Villani, Oriolo.
        Args:
            params: dictionary of parameters
            q: generalized coordinates of shape (n_q,)
            qd: generalized velocities of shape (n_q,)
            s: point coordinate along the robot in the interval [0, L].
            B: inertia matrix in the generalized coordinates of shape (n_q, n_q)
            C: coriolis matrix derived with Christoffer symbols in the generalized coordinates of shape (n_q, n_q)
            operational_space_selector: tuple of shape (3,) to select the operational space variables.
                For examples, (True, True, False) selects only the positional components of the operational space.
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            Lambda: inertia matrix in the operational space of shape (3, 3)
            mu: matrix with corioli and centrifugal terms in the operational space of shape (3, 3)
            J: Jacobian of the Cartesian pose with respect to the generalized coordinates of shape (3, n_q)
            Jd: time-derivative of the Jacobian of the end-effector pose with respect to the generalized coordinates
                of shape (3, n_q)
            JB_pinv: Dynamically-consistent pseudo-inverse of the Jacobian. Allows the mapping of torques
                from the generalized coordinates to the operational space: f = JB_pinv.T @ tau_q
                Shape (n_q, 3)
        """
        ## map the configuration to the strains
        xi = xi_ref + B_xi @ q
        xid = B_xi @ qd
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # classify the point along the robot to the corresponding segment
        segment_idx, s_segment = classify_segment(params, s)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J = lax.switch(
            segment_idx, J_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()
        Jd = lax.switch(
            segment_idx,
            Jd_lambda_sms,
            *params_for_lambdify,
            *xi_epsed,
            *xid,
            s_segment,
        ).squeeze()
        # apply the operational_space_selector and strain basis to the J and Jd
        J = J[operational_space_selector, :] @ B_xi
        Jd = Jd[operational_space_selector, :] @ B_xi

        # inverse of the inertia matrix in the configuration space
        B_inv = jnp.linalg.inv(B)

        Lambda = jnp.linalg.inv(
            J @ B_inv @ J.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            J @ B_inv @ C - Jd
        )  # coriolis and centrifugal matrix in the operational space

        JB_pinv = (
            B_inv @ J.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, J, Jd, JB_pinv

    auxiliary_fns = {
        "apply_eps_to_bend_strains": apply_eps_to_bend_strains,
        "classify_segment": classify_segment,
        "stiffness_fn": stiffness_fn,
        "actuation_mapping_fn": actuation_mapping_fn,
        "jacobian_fn": jacobian_fn,
        "kinetic_energy_fn": kinetic_energy_fn,
        "potential_energy_fn": potential_energy_fn,
        "energy_fn": energy_fn,
        "operational_space_dynamical_matrices_fn": operational_space_dynamical_matrices_fn,
    }

    return B_xi, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns
