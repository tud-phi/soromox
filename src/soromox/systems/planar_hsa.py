import dill
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .utils import (
    concatenate_params_syms,
    compute_strain_basis,
)


def factory(
    sym_exp_filepath: Union[str, Path],
    strain_selector: Array = None,
    global_eps: float = 1e-6,
    consider_hysteresis: bool = False,
) -> Tuple[
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Dict,
]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        sym_exp_filepath: path to file containing symbolic expressions
        strain_selector: array of shape (n_xi, ) with boolean values indicating which components of the
                strain are active / non-zero
        global_eps: small number to avoid singularities (e.g., division by zero)
        consider_hysteresis: If True, Bouc-Wen is used to model hysteresis. Otherwise, hysteresis will be neglected.
    Returns:
        forward_kinematics_virtual_backbone_fn: function that returns the chi vector of shape (3, n_q) with the
            positions and orientations of the virtual backbone
        forward_kinematics_end_effector_fn: function that returns the pose of the end effector of shape (3, )
        jacobian_end_effector_fn: function that returns the Jacobian of the end effector of shape (3, n_q)
        inverse_kinematics_end_effector_fn: function that returns the generalized coordinates for a given end-effector pose
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
        sys_helpers: dictionary of helper functions / variables. Includes for examples:
            B_xi: strain basis matrix of shape (3 * num_segments, n_q)
            forward_kinematics_rod_fn: function that returns the chi vector of shape (3, n_q) with the
                positions and orientations of the rod
            forward_kinematics_platform_fn: function that returns the chi vector of shape (3, n_q) with the positions
                and orientations of the platform
            operational_space_dynamical_matrices_fn: function that returns Lambda, mu, and JB_pinv describing
                the dynamics in operational space
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

    num_segments = len(params_syms["l"])
    num_rods_per_segment = len(params_syms["rout"]) // num_segments

    @jit
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

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"]

    # lambdify symbolic expressions
    chiv_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chiv_exp in sym_exps["exps"]["chiv_sms"]:
        chiv_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            chiv_exp,
            "jax",
        )
        chiv_lambda_sms.append(chiv_lambda)

    chir_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chir_exp in sym_exps["exps"]["chir_sms"]:
        chir_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            chir_exp,
            "jax",
        )
        chir_lambda_sms.append(chir_lambda)

    chip_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chip_exp in sym_exps["exps"]["chip_sms"]:
        chip_lambda = sp.lambdify(
            params_syms_cat + sym_exps["state_syms"]["xi"],
            chip_exp,
            "jax",
        )
        chip_lambda_sms.append(chip_lambda)

    # end-effector kinematics
    chiee_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"],
        sym_exps["exps"]["chiee"],
        "jax",
    )
    Jee_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"],
        sym_exps["exps"]["Jee"],
        "jax",
    )
    Jeed_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"],
        sym_exps["exps"]["Jeed"],
        "jax",
    )

    B_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["B"], "jax"
    )
    C_lambda = sp.lambdify(
        params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
    )
    G_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["G"], "jax"
    )
    Shat_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["Shat"], "jax")
    K_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["K"], "jax"
    )
    D_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["D"], "jax")
    alpha_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
        sym_exps["exps"]["alpha"],
        "jax",
    )

    @jit
    def beta_fn(params: Dict[str, Array], vxi: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the physical rods
        Args:
            params: Dictionary of robot parameters
            vxi: strains of the virtual backbone of shape (n_xi, )
        Returns:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        """
        # strains of the virtual rod
        vxi = vxi.reshape((num_segments, 1, -1))

        pxi = jnp.repeat(vxi, num_rods_per_segment, axis=1)
        psigma_a = (
            pxi[:, :, 2]
            + params["roff"] * jnp.repeat(vxi, num_rods_per_segment, axis=1)[..., 0]
        )
        pxi = pxi.at[:, :, 2].set(psigma_a)

        return pxi

    @jit
    def beta_inv_fn(params: Dict[str, Array], pxi: Array) -> Array:
        """
        Map the strains in the physical rods to the strains of the virtual backbone
        Args:
            params: Dictionary of robot parameters
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        Returns:
            vxi: strains of the virtual backbone of shape (n_xi, )
        """
        vxi = jnp.mean(pxi, axis=1)
        vxi = vxi.at[:, 2].set(
            vxi[:, 2] - jnp.mean(params["roff"] * pxi[..., 0], axis=1)
        )
        vxi = vxi.flatten()

        return vxi

    @jit
    def ref_strains_fn(params: Dict[str, Array]) -> Array:
        """
        Compute the reference strains of the virtual backbone
        Args:
            params: Dictionary of robot parameters

        Returns:
            vxi_ref: reference strains of the virtual backbone of shape (n_xi, )
        """
        # ref strains of the physical rods
        pxi_ref = jnp.zeros((num_segments, num_rods_per_segment, 3))
        pxi_ref = pxi_ref.at[:, :, 0].set(params["kappa_b_ref"])
        pxi_ref = pxi_ref.at[:, :, 1].set(params["sigma_sh_ref"])
        pxi_ref = pxi_ref.at[:, :, 2].set(params["sigma_a_ref"])

        # map the ref strains from the physical rods to the virtual backbone
        vxi_ref = beta_inv_fn(params, pxi_ref)
        return vxi_ref

    @jit
    def configuration_to_strains_fn(params: Dict[str, Array], q: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the virtual backbone
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
        Returns:
            xi: strains of the virtual backbone of shape (n_xi, )
        """
        # ref strains of the virtual backbone
        xi_ref = ref_strains_fn(params)

        # map the configuration to the strains
        xi = xi_ref + B_xi @ q

        return xi

    @jit
    def apply_eps_to_bend_strains_fn(xi: Array, _eps: float) -> Array:
        """
        Add a small number to the bending strain to avoid singularities
        """
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

    @jit
    def forward_kinematics_virtual_backbone_fn(
        params: Dict[str, Array], q: Array, s: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the forward kinematics the virtual backbone
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
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

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

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chi = lax.switch(
            segment_idx, chiv_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()

        return chi

    @jit
    def forward_kinematics_rod_fn(
        params: Dict[str, Array],
        q: Array,
        s: Array,
        rod_idx: Array,
        eps: float = global_eps,
    ) -> Array:
        """
        Evaluate the forward kinematics of the physical rods
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
            rod_idx: index of the rod. If there are two rods per segment, then rod_idx can be 0 or 1.
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            chir: pose of the rod centerline point in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

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

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chir_lambda_sms_idx = segment_idx * num_rods_per_segment + rod_idx
        chir = lax.switch(
            chir_lambda_sms_idx,
            chir_lambda_sms,
            *params_for_lambdify,
            *xi_epsed,
            s_segment,
        ).squeeze()

        return chir

    @jit
    def forward_kinematics_platform_fn(
        params: Dict[str, Array], q: Array, segment_idx: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the forward kinematics the platform
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            segment_idx: index of the segment
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            chip: pose of the CoG of the platform in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chip = lax.switch(
            segment_idx, chip_lambda_sms, *params_for_lambdify, *xi_epsed
        ).squeeze()

        return chip

    @jit
    def forward_kinematics_end_effector_fn(
        params: Dict[str, Array], q: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the forward kinematics of the end-effector
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)
        # evaluate the symbolic expression
        chiee = chiee_lambda(*params_for_lambdify, *xi_epsed).squeeze()

        return chiee

    @jit
    def jacobian_end_effector_fn(
        params: Dict[str, Array], q: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluate the Jacobian of the end-effector
        Args:
           params: Dictionary of robot parameters
           q: generalized coordinates of shape (n_q, )
           eps: small number to avoid singularities (e.g., division by zero)
        Returns:
           Jee: the Jacobian of the end-effector pose with respect to the generalized coordinates.
                Jee is an array of shape (3, n_q).
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)
        # evaluate the symbolic expression
        Jee = Jee_lambda(*params_for_lambdify, *xi_epsed)

        return Jee

    @jit
    def inverse_kinematics_end_effector_fn(
        params: Dict[str, Array], chiee: Array, eps: float = global_eps
    ) -> Array:
        """
        Evaluates the inverse kinematics for a given end-effector pose.
            Important: only works for one segment!
        Args:
           params: Dictionary of robot parameters
           chiee: pose of the end-effector in Cartesian-space of shape (3, )
           eps: small number to avoid singularities (e.g., division by zero)
        Returns:
           q: generalized coordinates of shape (n_q, )
        """
        assert num_segments == 1, "Inverse kinematics only works for one segment!"

        # height of platform
        hp = params["pcudim"][0, 1]
        # length of the proximal rod caps
        lpc = params["lpc"][0]
        # length of the distal rod caps
        ldc = params["ldc"][0]
        # offset of the end-effector from the distal surface of the platform
        chiee_off = params["chiee_off"]

        # transformation from the base to the proximal end of the virtual backbone
        T_b_to_pe = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, lpc],
                [0.0, 0.0, 1.0],
            ]
        )

        # transformation from the base to the end-effector
        T_b_to_ee = jnp.array(
            [
                [jnp.cos(chiee[2]), -jnp.sin(chiee[2]), chiee[0]],
                [jnp.sin(chiee[2]), jnp.cos(chiee[2]), chiee[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        # transformation from the distal end of the virtual backbone to the end-effector
        T_de_to_ee = jnp.array(
            [
                [jnp.cos(chiee_off[2]), -jnp.sin(chiee_off[2]), chiee_off[0]],
                [jnp.sin(chiee_off[2]), jnp.cos(chiee_off[2]), ldc + hp + chiee_off[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        # compute the transformation from the proximal to the distal end of the virtual backbone
        T_pe_to_de = jnp.linalg.inv(T_b_to_pe) @ T_b_to_ee @ jnp.linalg.inv(T_de_to_ee)

        # compute the SE(2) pose from the transformation matrix
        vchi_pe_to_de = jnp.array(
            [
                T_pe_to_de[0, 2],
                T_pe_to_de[1, 2],
                jnp.arctan2(T_pe_to_de[1, 0], T_pe_to_de[0, 0]),
            ]
        )

        # extract the x and y position and the orientation
        px, py, th = vchi_pe_to_de[0], vchi_pe_to_de[1], vchi_pe_to_de[2]

        # add small eps for numerical stability
        th_sign = jnp.sign(th)
        # set zero sign to 1 (i.e. positive)
        th_sign = jnp.where(th_sign == 0, 1, th_sign)
        # add eps to the bending strain (i.e. the first column)
        th_epsed = th + th_sign * eps

        # compute the inverse kinematics for the virtual backbone
        vxi = (
            th_epsed
            / (2 * params["l"].sum())
            * jnp.array(
                [
                    2,
                    py - (px * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                    -px - (py * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                ]
            )
        )

        # ref strains of the virtual backbone
        vxi_ref = ref_strains_fn(params)

        # map the strains to the generalized coordinates
        q = jnp.linalg.pinv(B_xi) @ (vxi - vxi_ref)

        return q

    @jit
    def dynamical_matrices_fn(
        params: Dict[str, Array],
        q: Array,
        qd: Array,
        z: Optional[Array] = None,
        phi: Array = jnp.zeros((num_segments * num_rods_per_segment,)),
        eps: float = 1e4 * global_eps,
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            qd: generalized velocities of shape (n_q, )
            z: state variables of the hysteresis model of shape (n_z, )
            phi: motor positions / twist angles of shape (num_segments * num_rods_per_segment, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            B: mass / inertia matrix of shape (n_q, n_q)
            C: coriolis / centrifugal matrix of shape (n_q, n_q)
            G: gravity vector of shape (n_q, )
            K: elastic vector of shape (n_q, )
            D: dissipative matrix of shape (n_q, n_q)
            alpha: actuation vector acting on the generalized coordinates of shape (n_q, )
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        xid = B_xi @ qd

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # evaluate the symbolic expressions
        params_for_lambdify = select_params_for_lambdify_fn(params)
        B = B_lambda(*params_for_lambdify, *xi_epsed)
        C_xi = C_lambda(*params_for_lambdify, *xi_epsed, *xid)
        G = G_lambda(*params_for_lambdify, *xi_epsed).squeeze()
        K = K_lambda(*params_for_lambdify, *xi).squeeze()
        Shat = Shat_lambda(*params_for_lambdify)
        D = D_lambda(*params_for_lambdify)
        alpha = alpha_lambda(*params_for_lambdify, *xi, *phi).squeeze()

        if consider_hysteresis is True:
            # hysteresis parameters
            B_z = params["hysteresis"]["basis"]
            hyst_alpha = params["hysteresis"]["alpha"]
            # add the post-yield potential forces
            K = hyst_alpha * K + (1 - hyst_alpha) * Shat @ (B_z @ z)

            # TODO: add post-yield potential forces (i.e., hysteresis effects) to the actuation vector

        # apply the strain basis
        B = B_xi.T @ B @ B_xi
        C = B_xi.T @ C_xi @ B_xi
        G = B_xi.T @ G
        K = B_xi.T @ K
        D = B_xi.T @ D @ B_xi
        alpha = B_xi.T @ alpha

        return B, C, G, K, D, alpha

    def operational_space_dynamical_matrices_fn(
        params: Dict[str, Array],
        q: Array,
        qd: Array,
        B: Array,
        C: Array,
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
            B: inertia matrix in the generalized coordinates of shape (n_q, n_q)
            C: coriolis matrix derived with Christoffer symbols in the generalized coordinates of shape (n_q, n_q)
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            Lambda: inertia matrix in the operational space of shape (n_x, n_x)
            mu: matrix with corioli and centrifugal terms in the operational space of shape (n_x, n_x)
            Jee: Jacobian of the end-effector pose with respect to the generalized coordinates of shape (3, n_q)
            Jeed: time-derivative of the Jacobian of the end-effector pose with respect to the generalized coordinates
            JeeB_pinv: Dynamically-consistent pseudo-inverse of the Jacobian. Allows the mapping of torques
                from the generalized coordinates to the operational space: f = JB_pinv.T @ tau_q
                Shape (n_q, n_x)
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        xid = B_xi @ qd
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        # end-effector Jacobian and its time-derivative
        Jee = Jee_lambda(*params_for_lambdify, *xi_epsed)
        Jeed = Jeed_lambda(*params_for_lambdify, *xi_epsed, *xid)

        # inverse of the inertia matrix in the configuration space
        B_inv = jnp.linalg.inv(B)

        Lambda = jnp.linalg.inv(
            Jee @ B_inv @ Jee.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            Jee @ B_inv @ C - Jeed
        )  # coriolis and centrifugal matrix in the operational space

        JeeB_pinv = (
            B_inv @ Jee.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, Jee, Jeed, JeeB_pinv

    sys_helpers = {
        "eps": global_eps,
        "select_params_for_lambdify_fn": select_params_for_lambdify_fn,
        "beta_fn": beta_fn,
        "beta_inv_fn": beta_inv_fn,
        "ref_strains_fn": ref_strains_fn,
        "B_xi": B_xi,
        "configuration_to_strains_fn": configuration_to_strains_fn,
        "apply_eps_to_bend_strains_fn": apply_eps_to_bend_strains_fn,
        "forward_kinematics_rod_fn": forward_kinematics_rod_fn,
        "forward_kinematics_platform_fn": forward_kinematics_platform_fn,
        "operational_space_dynamical_matrices_fn": operational_space_dynamical_matrices_fn,
    }

    return (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    )


def ode_factory(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    control_fn: Optional[Callable] = None,
    consider_underactuation_model: bool = True,
    consider_hysteresis: bool = False,
) -> Callable[[float, Array], Array]:
    """
    Make an ODE function of the form ode_fn(t, x) -> xd.
    This function assumes a constant torque input (i.e. zero-order hold).
    Args:
        dynamical_matrices_fn: Callable that returns B, C, G, K, D, alpha_fn. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd, z, phi) -> Tuple[B, C, G, K, D, A]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            alpha_fn is a function to compute the actuation vector of shape (n_q). It has the following signature:
                alpha_fn(phi) -> tau_q where phi is the twist angle vector of shape (n_phi, )
        params: Dictionary with robot parameters
        control_fn: Callable that returns the forcing function of the form control_fn(t, x) -> phi. If consider_underactuation_model is True,
            then phi is an array of shape (n_q, ) with the configuration-space torques. If consider_underactuation_model is False,
            then phi is an array of shape (n_phi, ) with the motor positions / twist angles of the proximal end of the rods.
        consider_underactuation_model: If True, the underactuation model is considered. Otherwise, the fully-actuated
            model is considered with the identity matrix as the actuation matrix.
        consider_hysteresis: If True, Bouc-Wen is used to model hysteresis. Otherwise, hysteresis will be neglected.
    Returns:
        ode_fn: ODE function of the form ode_fn(t, x) -> xd
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    @jit
    def ode_fn(t: float, x: Array, u: Array) -> Array:
        """
        ODE of the dynamical Lagrangian system.
        Args:
            t: time
            x: state vector of shape (2 * n_q + n_z, ) where n_q is the number of configuration variables and n_z is the
                number of state variables of the hysteresis model. The state vector is of the form [q, qd, z]
            u: input to the system.
                - if consider_underactuation_model is True, then this is an array of shape (n_phi) with
                    motor positions / twist angles of the proximal end of the rods
                - if consider_underactuation_model is False, then this is an array of shape (n_q) with
                    the generalized torques
        Returns:
            xd: time-derivative of the state vector of shape (2 * n_q, )
        """
        if consider_hysteresis is True:
            hys_params = params["hysteresis"]
            B_z = hys_params["basis"]

            n_z = B_z.shape[1]
            n_q = (x.shape[0] - n_z) // 2
            q, qd, z = x[:n_q], x[n_q : 2 * n_q], x[2 * n_q :]

            zd = (B_z.T @ qd) * (
                hys_params["A"]
                - jnp.abs(z) ** hys_params["n"]
                * (
                    hys_params["gamma"]
                    + hys_params["beta"] * jnp.sign((B_z.T @ qd) * z)
                )
            )
        else:
            n_q = x.shape[0] // 2
            q, qd = x[:n_q], x[n_q:]
            z = None

        if control_fn is not None:
            u = u + control_fn(t, x)

        if consider_underactuation_model is True:
            phi = u
            B, C, G, K, D, tau_q = dynamical_matrices_fn(params, q, qd, z=z, phi=phi)
        else:
            B, C, G, K, D, _ = dynamical_matrices_fn(
                params, q, qd, z=z, phi=jnp.zeros((num_rods,))
            )
            tau_q = u

        # inverse of B
        B_inv = jnp.linalg.inv(B)

        # compute the acceleration
        qdd = B_inv @ (tau_q - C @ qd - G - K - D @ qd)

        if consider_hysteresis is True:
            xd = jnp.concatenate([qd, qdd, zd])
        else:
            xd = jnp.concatenate([x[n_q:], qdd])

        return xd

    return ode_fn
