__all__ = ["PlanarHSA"]
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dill
import equinox as eqx
import sympy as sp
from jax import Array, jacfwd, lax
from jax import numpy as jnp

from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.basic import (
    compute_strain_basis,
    concatenate_params_syms,
)


class PlanarHSA(SoftRobot):
    """
    A kinematic and dynamic model for planar Handed Shearing Auxetics (HSA) robots.

    This class implements the geometric and dynamic modeling of planar HSA robots
    using a piecewise constant strain assumption. It supports computation of forward
    kinematics, inverse kinematics, Jacobians, and dynamical matrices. The model
    accounts for hysteresis effects using the Bouc-Wen model when enabled.

    Based on the publication:
        Stölzle, M., Rus, D., & Della Santina, C. (2023, November). An experimental
        study of model-based control for planar handed shearing auxetics robots.
        In International Symposium on Experimental Robotics (pp. 153-167).
        Cham: Springer Nature Switzerland.
        https://link.springer.com/chapter/10.1007/978-3-031-63596-0_14

    Attributes:
        num_segments: Number of segments along the robot.
        num_rods_per_segment: Number of physical rods per segment.
        num_dofs: Number of degrees of freedom (active strain components).
        num_actuators: Number of actuators in the robot.
        consider_underactuation: Whether to consider underactuation in the model.
        consider_hysteresis: Whether to consider hysteresis effects in the model.
        num_hysteresis: Number of hysteresis state variables.
        chiv_lambda_sms: Lambda functions for virtual backbone forward kinematics per segment.
        chir_lambda_sms: Lambda functions for physical rod forward kinematics per segment.
        chip_lambda_sms: Lambda functions for platform forward kinematics per segment.
        chiee_lambda: Lambda function for end-effector forward kinematics.
        Jee_lambda: Lambda function for end-effector Jacobian.
        Jeed_lambda: Lambda function for end-effector Jacobian time derivative.
        B_lambda: Lambda function for inertia matrix computation.
        C_lambda: Lambda function for Coriolis matrix computation.
        G_lambda: Lambda function for gravitational force computation.
        Shat_lambda: Lambda function for nominal stiffness matrix computation.
        K_lambda: Lambda function for elastic force computation.
        D_lambda: Lambda function for damping matrix computation.
        alpha_lambda: Lambda function for actuation force computation.
        B_xi: Strain basis matrix for mapping active strain components.
        kappa_b_ref: Reference bending curvatures for each rod. Shape: (num_segments, num_rods_per_segment).
        sigma_sh_ref: Reference shear strains for each rod. Shape: (num_segments, num_rods_per_segment).
        sigma_a_ref: Reference axial strains for each rod. Shape: (num_segments, num_rods_per_segment).
        L: Segment lengths. Shape: (num_segments,).
        L_cum: Cumulative segment lengths. Shape: (num_segments + 1,).
        Lmax: Total robot length (sum of all segments).
        roff: Rod offset from centerline. Shape: (num_segments, num_rods_per_segment).
        pcudim: Platform dimensions (width, height, depth). Shape: (num_segments, 3).
        lpc: Length of rigid proximal rod caps. Shape: (num_segments,).
        ldc: Length of rigid distal rod caps. Shape: (num_segments,).
        chiee_off: End-effector offset transformation [theta, p_x, p_y]. Shape: (3,).
        B_hyst: Hysteresis basis matrix. Shape: (num_dofs, num_hysteresis).
        hyst_alpha: Bouc-Wen hysteresis parameter: ratio of post-yield to pre-yield stiffness.
        hyst_A: Bouc-Wen hysteresis parameter A.
        hyst_n: Bouc-Wen hysteresis parameter n.
        hyst_beta: Bouc-Wen hysteresis parameter beta.
        hyst_gamma: Bouc-Wen hysteresis parameter gamma.
        params_for_lambdify: Flattened parameter list for symbolic function evaluation.

    Notes:
    -----
    - The strain vector is composed of 3 components per segment:
      [kappa_b, sigma_sh, sigma_a] representing bending curvature,
      shear strain, and axial strain respectively.
    - The robot uses a virtual backbone representation with physical
      rod mapping for accurate modeling of HSA mechanics.
    - Hysteresis modeling is optional and uses the Bouc-Wen model
      when consider_hysteresis=True.

    References:
        Stölzle, M., Rus, D., & Della Santina, C. (2023). An experimental study of
        model-based control for planar handed shearing auxetics robots. In
        International Symposium on Experimental Robotics (pp. 153-167). Springer.
    """

    # static settings
    num_segments: int = eqx.field(static=True)
    num_rods_per_segment: int = eqx.field(static=True)
    consider_underactuation: bool = eqx.field(static=True)
    consider_hysteresis: bool = eqx.field(static=True)
    num_hysteresis: int = eqx.field(static=True)

    chiv_lambda_sms: list[Callable]
    chir_lambda_sms: list[Callable]
    chip_lambda_sms: list[Callable]
    Jv_lambda_sms: list[Callable]  # Jacobian of the virtual backbone at each segment

    # kinematic lambda functions
    chiee_lambda: Callable
    Jee_lambda: Callable
    Jeed_lambda: Callable

    # energy lambda functions
    U_g_lambda: Callable  # gravitational potential energy

    # dynamic lambda functions
    B_lambda: Callable
    C_lambda: Callable
    G_lambda: Callable
    Shat_lambda: Callable
    K_lambda: Callable
    D_lambda: Callable
    alpha_lambda: Callable

    # strain basis
    B_xi: Array

    # reference strains
    kappa_b_ref: Array
    sigma_sh_ref: Array
    sigma_a_ref: Array

    # geometric parameters of the robot
    L: Array  # Array of segment lengths
    L_cum: Array  # Cumulative length of the robot as array of size (num_segments, )
    Lmax: Array  # Maximum length of the robot (sum of all segments)
    roff: Array
    pcudim: Array
    lpc: Array
    ldc: Array
    chiee_off: Array

    # hysteresis parameters
    B_hyst: Array
    hyst_alpha: Array
    hyst_A: Array
    hyst_n: Array
    hyst_beta: Array
    hyst_gamma: Array

    # parameters for lambdify
    params_for_lambdify: list[Array]

    def __init__(
        self,
        sym_exp_filepath: str | Path,
        params: dict[str, Array],
        strain_selector: Array | None = None,
        consider_underactuation: bool = True,
        consider_hysteresis: bool = False,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PlanarHSA system.

        Args:
            sym_exp_filepath: path to file containing symbolic expressions
            params: dictionary with robot parameters
            strain_selector: array of shape (num_dofs, ) with boolean values indicating which components of the
                    strain are active / non-zero
            consider_underactuation (bool): If True, the underactuation model is considered. Otherwise, the fully-actuated
                    model is considered with the identity matrix as the actuation matrix.
            consider_hysteresis: If True, Bouc-Wen is used to model hysteresis. Otherwise, hysteresis will be neglected.
            eps: small number to avoid singularities (e.g., division by zero)
            **kwargs: Additional keyword arguments for SoftRobot.__init__.
        """
        super().__init__(eps=eps, **kwargs)

        # Load saved symbolic data
        try:
            sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Symbolic expressions file not found. Please generate the symbolic expressions first."
            )

        # Symbols for robot parameters
        try:
            params_syms = sym_exps["params_syms"]
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'params_syms'. Please generate the symbolic expressions first."
            )

        try:
            params_for_lambdify = []
            for params_key, params_vals in sorted(params.items()):
                if params_key in params_syms.keys():
                    for param in params_vals.flatten():
                        params_for_lambdify.append(param)
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain the required parameters. Please generate the symbolic expressions first."
            )
        self.params_for_lambdify = params_for_lambdify

        try:
            L = params["L"]
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'L'. Please generate the symbolic expressions first."
            )
        self.L = L

        try:
            # cumsum of the segment lengths
            L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'L'. Please generate the symbolic expressions first."
            )
        self.L_cum = L_cum

        # Maximum length of the robot
        self.Lmax = L_cum[-1]

        # Number of segments
        try:
            num_segments = len(params_syms["L"])
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'L'. Please generate the symbolic expressions first."
            )
        self.num_segments = num_segments

        # Number of rods per segment
        try:
            num_rods_per_segment = len(params_syms["rout"]) // num_segments
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'rout'. Please generate the symbolic expressions first."
            )
        self.num_rods_per_segment = num_rods_per_segment

        # =================================================
        # Parameters
        # =====================

        # concatenate the robot params symbols
        params_syms_cat = concatenate_params_syms(params_syms)

        # Number of degrees of freedom
        try:
            num_dofs = len(sym_exps["state_syms"]["xi"])
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'state_syms'. Please generate the symbolic expressions first."
            )
        self.num_dofs = num_dofs

        # set number of actuators
        self.consider_underactuation = consider_underactuation
        if self.consider_underactuation:
            # the number of actuators equals the number of HSA rods
            self.num_actuators = num_rods_per_segment * num_segments
        else:
            # the number of actuators equals the number of degrees of freedom as we consider an identity actuation matrix
            self.num_actuators = self.num_dofs

        # Hysteresis
        self.consider_hysteresis = consider_hysteresis

        # ================================================================
        # Robot parameters
        self._set_params(params, consider_hysteresis, num_dofs)

        # compute the strain basis
        if strain_selector is None:
            strain_selector = jnp.ones((num_dofs,), dtype=bool)
        else:
            if not isinstance(strain_selector, (list, jnp.ndarray)):
                raise TypeError(
                    f"strain_selector must be a list or an array, got {type(strain_selector).__name__}"
                )
            strain_selector = jnp.asarray(strain_selector)
            if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
                raise TypeError(
                    f"strain_selector must be a boolean array, got {strain_selector.dtype}"
                )
            if strain_selector.size != num_dofs:
                raise ValueError(
                    f"strain_selector must have {num_dofs} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(num_dofs)
        self.B_xi = compute_strain_basis(strain_selector)

        # concatenate the list of state symbols
        try:
            state_syms_cat = (
                sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xid"]
            )
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain 'state_syms'. Please generate the symbolic expressions first."
            )

        # =================================================
        # lambdify symbolic expressions

        chiv_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chiv_exp in sym_exps["exps"]["chiv_sms"]:
                chiv_lambda = sp.lambdify(
                    params_syms_cat
                    + sym_exps["state_syms"]["xi"]
                    + [sym_exps["state_syms"]["s"]],
                    chiv_exp,
                    "jax",
                )
                chiv_lambda_sms.append(chiv_lambda)
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does ['exps']['chiv_sms']. Please generate the symbolic expressions first."
            )
        self.chiv_lambda_sms = chiv_lambda_sms

        chir_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chir_exp in sym_exps["exps"]["chir_sms"]:
                chir_lambda = sp.lambdify(
                    params_syms_cat
                    + sym_exps["state_syms"]["xi"]
                    + [sym_exps["state_syms"]["s"]],
                    chir_exp,
                    "jax",
                )
                chir_lambda_sms.append(chir_lambda)
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain ['exps']['chir_sms']. Please generate the symbolic expressions first."
            )
        self.chir_lambda_sms = chir_lambda_sms

        chip_lambda_sms = []
        # iterate through symbolic expressions for each segment
        try:
            for chip_exp in sym_exps["exps"]["chip_sms"]:
                chip_lambda = sp.lambdify(
                    params_syms_cat + sym_exps["state_syms"]["xi"],
                    chip_exp,
                    "jax",
                )
                chip_lambda_sms.append(chip_lambda)
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain ['exps']['chip_sms']. Please generate the symbolic expressions first."
            )
        self.chip_lambda_sms = chip_lambda_sms

        # lambdify the Jacobians of the virtual backbone for each segment
        Jv_lambda_sms = []
        try:
            for Jv_exp in sym_exps["exps"]["Jv_sms"]:
                Jv_lambda = sp.lambdify(
                    params_syms_cat
                    + sym_exps["state_syms"]["xi"]
                    + [sym_exps["state_syms"]["s"]],
                    Jv_exp,
                    "jax",
                )
                Jv_lambda_sms.append(Jv_lambda)
        except KeyError:
            raise KeyError(
                "Symbolic expressions file does not contain ['exps']['Jv_sms']. Please generate the symbolic expressions first."
            )
        self.Jv_lambda_sms = Jv_lambda_sms

        # end-effector kinematics
        try:
            chiee_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["chiee"],
                "jax",
            )
        except ValueError:
            raise ValueError(
                "Fail to lambdify chiee. Check the symbolic expressions file."
            )
        self.chiee_lambda = chiee_lambda

        try:
            Jee_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["Jee"],
                "jax",
            )
        except ValueError:
            raise ValueError(
                "Fail to lambdify Jee. Check the symbolic expressions file."
            )
        self.Jee_lambda = Jee_lambda

        try:
            Jeed_lambda = sp.lambdify(
                params_syms_cat
                + sym_exps["state_syms"]["xi"]
                + sym_exps["state_syms"]["xid"],
                sym_exps["exps"]["Jeed"],
                "jax",
            )
        except ValueError:
            raise ValueError(
                "Fail to lambdify Jeed. Check the symbolic expressions file."
            )
        self.Jeed_lambda = Jeed_lambda

        # dynamical matrices
        try:
            B_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["B"],
                "jax",
            )
        except ValueError:
            raise ValueError("Fail to lambdify B. Check the symbolic expressions file.")
        self.B_lambda = B_lambda

        try:
            C_lambda = sp.lambdify(
                params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
            )
        except ValueError:
            raise ValueError("Fail to lambdify C. Check the symbolic expressions file.")
        self.C_lambda = C_lambda

        try:
            G_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["G"],
                "jax",
            )
        except ValueError:
            raise ValueError("Fail to lambdify G. Check the symbolic expressions file.")
        self.G_lambda = G_lambda

        # gravitational potential energy
        try:
            U_g_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["U_g"],
                "jax",
            )
        except ValueError:
            raise ValueError(
                "Fail to lambdify U_g. Check the symbolic expressions file."
            )
        self.U_g_lambda = U_g_lambda

        try:
            Shat_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["Shat"], "jax")
        except ValueError:
            raise ValueError(
                "Fail to lambdify Shat. Check the symbolic expressions file."
            )
        self.Shat_lambda = Shat_lambda

        try:
            K_lambda = sp.lambdify(
                params_syms_cat + sym_exps["state_syms"]["xi"],
                sym_exps["exps"]["K"],
                "jax",
            )
        except ValueError:
            raise ValueError("Fail to lambdify K. Check the symbolic expressions file.")
        self.K_lambda = K_lambda

        try:
            D_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["D"], "jax")
        except ValueError:
            raise ValueError("Fail to lambdify D. Check the symbolic expressions file.")
        self.D_lambda = D_lambda

        try:
            alpha_lambda = sp.lambdify(
                params_syms_cat
                + sym_exps["state_syms"]["xi"]
                + sym_exps["state_syms"]["phi"],
                sym_exps["exps"]["alpha"],
                "jax",
            )
        except ValueError:
            raise ValueError(
                "Fail to lambdify alpha. Check the symbolic expressions file."
            )
        self.alpha_lambda = alpha_lambda

    @property
    def is_planar(self) -> bool:
        """Planar HSA is a 2D model."""
        return True

    @property
    def length(self) -> Array:
        """Total backbone length."""
        return self.Lmax

    @property
    def segment_length(self) -> Array:
        """Per-segment backbone lengths."""
        return jnp.asarray(self.L)

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Circular cross-section using max rod offset for the segment."""
        segment_idx, _ = self.classify_segment(s)
        roff = jnp.asarray(self.roff)
        radius = jnp.max(jnp.abs(roff[segment_idx]))
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([radius])

    def _set_params(
        self, params: dict[str, Array], consider_hysteresis: bool, num_dofs: int
    ) -> None:
        """
        Set the parameters of the PlanarHSA model.

        Args:
            params (Dict[str, Array]):
                Dictionary containing the robot parameters to update:
                - "roff": Array of shape (num_segments, num_rods_per_segment)
                    offset [m] of each rod from the centerline.
                    The rows correspond to the segments.
                - "kappa_b_ref": Array of shape (num_segments, num_rods_per_segment)
                    bending reference curvatures of each rod
                - "sigma_sh_ref": Array of shape (num_segments, num_rods_per_segment)
                    shear reference curvatures of each rod
                - "sigma_a_ref": Array of shape (num_segments, num_rods_per_segment)
                    axial reference strains of each rod
                - "pcudim": Array of shape (num_segments, 3)
                    width, height, depth of each segment's platform [m]
                - "lpc": Array of shape (num_segments,)
                    length of the rigid proximal of the rods connecting to the base [m]
                - "ldc": Array of shape (num_segments,)
                    length of the rigid distal of the rods connecting to the platform [m]
                - "chiee_off": Array of shape (3,)
                    rigid offset transformation from the distal end of the platform to the end-effector [m]
                    in the form [theta, p_x, p_y]
                - "hysteresis": Dictionary containing hysteresis parameters if consider_hysteresis is True
                    - "basis":
                        Basis for the hysteresis model
                    - Bouc-Wen model parameters:
                        - "alpha": [-] Ratio of post-yield and pre-yield stiffness
                        - "A": TODO
                        - "n": [-] TODO
                        - "beta": [-] TODO
                        - "gamma": [-] TODO
        """
        try:
            roff = params["roff"]
        except KeyError:
            raise KeyError(
                "Parameter 'roff' not found in the parameters dictionary. Please provide it."
            )
        self.roff = roff
        try:
            kappa_b_ref = params["kappa_b_ref"]
        except KeyError:
            raise KeyError(
                "Parameter 'kappa_b_ref' not found in the parameters dictionary. Please provide it."
            )
        self.kappa_b_ref = kappa_b_ref
        try:
            sigma_sh_ref = params["sigma_sh_ref"]
        except KeyError:
            raise KeyError(
                "Parameter 'sigma_sh_ref' not found in the parameters dictionary. Please provide it."
            )
        self.sigma_sh_ref = sigma_sh_ref
        try:
            sigma_a_ref = params["sigma_a_ref"]
        except KeyError:
            raise KeyError(
                "Parameter 'sigma_a_ref' not found in the parameters dictionary. Please provide it."
            )
        self.sigma_a_ref = sigma_a_ref

        try:
            pcudim = params["pcudim"]
        except KeyError:
            raise KeyError(
                "Parameter 'pcudim' not found in the parameters dictionary. Please provide it."
            )
        self.pcudim = pcudim
        try:
            lpc = params["lpc"]
        except KeyError:
            raise KeyError(
                "Parameter 'lpc' not found in the parameters dictionary. Please provide it."
            )
        self.lpc = lpc
        try:
            ldc = params["ldc"]
        except KeyError:
            raise KeyError(
                "Parameter 'ldc' not found in the parameters dictionary. Please provide it."
            )
        self.ldc = ldc
        try:
            chiee_off = params["chiee_off"]
        except KeyError:
            raise KeyError(
                "Parameter 'chiee_off' not found in the parameters dictionary. Please provide it."
            )
        self.chiee_off = chiee_off

        if consider_hysteresis:
            try:
                hyst_params = params["hysteresis"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' parameters. Please generate the symbolic expressions first."
                )
            try:
                B_hyst = hyst_params["basis"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' basis. Please generate the symbolic expressions first."
                )
            self.B_hyst = B_hyst

            try:
                num_hysteresis = B_hyst.shape[1]
            except AttributeError:
                raise AttributeError(
                    "Symbolic expressions file does not contain 'hysteresis' basis. Please generate the symbolic expressions first."
                )
            self.num_hysteresis = num_hysteresis

            try:
                hyst_alpha = params["hysteresis"]["alpha"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' alpha. Please generate the symbolic expressions first."
                )
            self.hyst_alpha = hyst_alpha

            try:
                hyst_A = hyst_params["A"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' A. Please generate the symbolic expressions first."
                )
            self.hyst_A = hyst_A

            try:
                hyst_n = hyst_params["n"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' n. Please generate the symbolic expressions first."
                )
            self.hyst_n = hyst_n

            try:
                hyst_beta = hyst_params["beta"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' beta. Please generate the symbolic expressions first."
                )
            self.hyst_beta = hyst_beta

            try:
                hyst_gamma = hyst_params["gamma"]
            except KeyError:
                raise KeyError(
                    "Symbolic expressions file does not contain 'hysteresis' gamma. Please generate the symbolic expressions first."
                )
            self.hyst_gamma = hyst_gamma
        else:
            self.num_hysteresis = 0
            self.B_hyst = jnp.zeros((num_dofs, 0))
            self.hyst_alpha = jnp.zeros((num_dofs,))
            self.hyst_A = jnp.zeros((1,))
            self.hyst_n = jnp.zeros((1,))
            self.hyst_beta = jnp.zeros((1,))
            self.hyst_gamma = jnp.zeros((1,))

    def update_params(self, params: dict[str, Array]) -> None:
        """
        Update the parameters of the PlanarHSA model.

        Args:
            params (Dict[str, Array]):
                Dictionary containing the robot parameters to update:
                - "roff": Array of shape (num_segments, num_rods_per_segment)
                    offset [m] of each rod from the centerline.
                    The rows correspond to the segments.
                - "kappa_b_ref": Array of shape (num_segments, num_rods_per_segment)
                    bending reference curvatures of each rod
                - "sigma_sh_ref": Array of shape (num_segments, num_rods_per_segment)
                    shear reference curvatures of each rod
                - "sigma_a_ref": Array of shape (num_segments, num_rods_per_segment)
                    axial reference strains of each rod
                - "pcudim": Array of shape (num_segments, 3)
                    width, height, depth of each segment's platform [m]
                - "lpc": Array of shape (num_segments,)
                    length of the rigid proximal of the rods connecting to the base [m]
                - "ldc": Array of shape (num_segments,)
                    length of the rigid distal of the rods connecting to the platform [m]
                - "chiee_off": Array of shape (3,)
                    rigid offset transformation from the distal end of the platform to the end-effector [m]
                    in the form [theta, p_x, p_y]
                - "hysteresis": Dictionary containing hysteresis parameters if consider_hysteresis is True
                    - "basis":
                        Basis for the hysteresis model
                    - Bouc-Wen model parameters:
                        - "alpha": [-] Ratio of post-yield and pre-yield stiffness
                        - "A": TODO
                        - "n": [-] TODO
                        - "beta": [-] TODO
                        - "gamma": [-] TODO
        """
        # Apply updates sequentially
        updated_self = self

        if "roff" in params:
            roff = params["roff"]
            updated_self = eqx.tree_at(lambda x: x.roff, updated_self, roff)

        if "kappa_b_ref" in params:
            kappa_b_ref = params["kappa_b_ref"]
            updated_self = eqx.tree_at(
                lambda x: x.kappa_b_ref, updated_self, kappa_b_ref
            )

        if "sigma_sh_ref" in params:
            sigma_sh_ref = params["sigma_sh_ref"]
            updated_self = eqx.tree_at(
                lambda x: x.sigma_sh_ref, updated_self, sigma_sh_ref
            )

        if "sigma_a_ref" in params:
            sigma_a_ref = params["sigma_a_ref"]
            updated_self = eqx.tree_at(
                lambda x: x.sigma_a_ref, updated_self, sigma_a_ref
            )

        if "pcudim" in params:
            pcudim = params["pcudim"]
            updated_self = eqx.tree_at(lambda x: x.pcudim, updated_self, pcudim)

        if "lpc" in params:
            lpc = params["lpc"]
            updated_self = eqx.tree_at(lambda x: x.lpc, updated_self, lpc)

        if "ldc" in params:
            ldc = params["ldc"]
            updated_self = eqx.tree_at(lambda x: x.ldc, updated_self, ldc)

        if "chiee_off" in params:
            chiee_off = params["chiee_off"]
            updated_self = eqx.tree_at(lambda x: x.chiee_off, updated_self, chiee_off)

        if self.consider_hysteresis:
            if "hysteresis" in params:
                hyst_params = params["hysteresis"]
                updated_self = eqx.tree_at(
                    lambda x: x.hyst_params, updated_self, hyst_params
                )

            if "basis" in hyst_params:
                B_hyst = hyst_params["basis"]
                updated_self = eqx.tree_at(lambda x: x.B_hyst, updated_self, B_hyst)

            if "alpha" in hyst_params:
                hyst_alpha = hyst_params["alpha"]
                updated_self = eqx.tree_at(
                    lambda x: x.hyst_alpha, updated_self, hyst_alpha
                )

            if "A" in hyst_params:
                hyst_A = hyst_params["A"]
                updated_self = eqx.tree_at(lambda x: x.hyst_A, updated_self, hyst_A)

            if "n" in hyst_params:
                hyst_n = hyst_params["n"]
                updated_self = eqx.tree_at(lambda x: x.hyst_n, updated_self, hyst_n)

            if "beta" in hyst_params:
                hyst_beta = hyst_params["beta"]
                updated_self = eqx.tree_at(
                    lambda x: x.hyst_beta, updated_self, hyst_beta
                )

            if "gamma" in hyst_params:
                hyst_gamma = hyst_params["gamma"]
                updated_self = eqx.tree_at(
                    lambda x: x.hyst_gamma, updated_self, hyst_gamma
                )

    @eqx.filter_jit
    def classify_segment(self, s: Array) -> tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_local (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local

    @eqx.filter_jit
    def strain(self, q: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the virtual backbone
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            xi: strains of the virtual backbone of shape (num_dofs, )
        """
        # reference strains of the virtual backbone
        xi_ref = self.ref_strains()

        # map the configuration to the strains
        xi = self.B_xi @ q + xi_ref

        return xi

    @eqx.filter_jit
    def beta(self, vxi: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the physical rods
        Args:
            vxi: strains of the virtual backbone of shape (num_dofs, )
        Returns:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        """
        # strains of the virtual rod
        vxi = vxi.reshape((self.num_segments, 1, -1))

        pxi = jnp.repeat(vxi, self.num_rods_per_segment, axis=1)
        psigma_a = (
            pxi[:, :, 2]
            + self.roff * jnp.repeat(vxi, self.num_rods_per_segment, axis=1)[..., 0]
        )
        pxi = pxi.at[:, :, 2].set(psigma_a)

        return pxi

    @eqx.filter_jit
    def beta_inv(self, pxi: Array) -> Array:
        """
        Map the strains in the physical rods to the strains of the virtual backbone
        Args:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        Returns:
            vxi: strains of the virtual backbone of shape (num_dofs, )
        """
        vxi = jnp.mean(pxi, axis=1)
        vxi = vxi.at[:, 2].set(vxi[:, 2] - jnp.mean(self.roff * pxi[..., 0], axis=1))
        vxi = vxi.flatten()

        return vxi

    @eqx.filter_jit
    def ref_strains(self) -> Array:
        """
        Compute the ref strains of the virtual backbone

        Returns:
            vxi_ref: ref strains of the virtual backbone of shape (num_dofs, )
        """
        # reference strains of the physical rods
        pxi_ref = jnp.zeros((self.num_segments, self.num_rods_per_segment, 3))
        pxi_ref = pxi_ref.at[:, :, 0].set(self.kappa_b_ref)
        pxi_ref = pxi_ref.at[:, :, 1].set(self.sigma_sh_ref)
        pxi_ref = pxi_ref.at[:, :, 2].set(self.sigma_a_ref)

        # map the reference strains from the physical rods to the virtual backbone
        vxi_ref = self.beta_inv(pxi_ref)
        return vxi_ref

    @eqx.filter_jit
    def apply_eps_to_bend_strains(self, xi: Array, eps: float | None = None) -> Array:
        """
        Add a small number to the bending strain to avoid singularities
        Args:
            xi: strains of the virtual backbone of shape (num_dofs, )
            eps: small number to add to the bending strain (optional). By default, it will be initialized to as self.global_eps
        """
        # initialize eps if not provided
        if eps is None:
            eps = self.global_eps

        xi_reshaped = xi.reshape((-1, 3))

        xi_bend_sign = jnp.sign(xi_reshaped[:, 0])
        # set zero sign to 1 (i.e. positive)
        xi_bend_sign = jnp.where(xi_bend_sign == 0, 1, xi_bend_sign)
        # add eps to the bending strain (i.e. the first column)
        sigma_b_epsed = lax.select(
            jnp.abs(xi_reshaped[:, 0]) < eps,
            xi_bend_sign * eps,
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

        # flatten the array
        xi_epsed = xi_epsed.flatten()

        return xi_epsed

    @eqx.filter_jit
    def forward_kinematics_virtual_backbone(self, q: Array, s: Array) -> Array:
        """
        Evaluate the forward kinematics the virtual backbone
        Args:
            q: generalized coordinates of shape (num_dofs, )
            s: point coordinate along the rod in the interval [0, L].
        Returns:
            chi: pose of the backbone point in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        # determine in which segment the point is located
        segment_idx, s_local = self.classify_segment(s)

        chi = lax.switch(
            segment_idx,
            self.chiv_lambda_sms,
            *self.params_for_lambdify,
            *xi_epsed,
            s_local,
        ).squeeze()

        chi = jnp.roll(
            chi, 1
        )  # shift from [p_x, p_y, theta] (symbolic derivation def) to [theta, p_x, p_y] (SE(2) convention)

        return chi

    forward_kinematics = forward_kinematics_virtual_backbone

    @eqx.filter_jit
    def forward_kinematics_rod(
        self,
        q: Array,
        s: Array,
        rod_idx: Array,
    ) -> Array:
        """
        Evaluate the forward kinematics of the physical rods
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (num_dofs, )
            s: point coordinate along the rod in the interval [0, L].
            rod_idx: index of the rod. If there are two rods per segment, then rod_idx can be 0 or 1.
        Returns:
            chir: pose of the rod centerline point in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        # determine in which segment the point is located
        segment_idx, s_local = self.classify_segment(s)

        chir_lambda_sms_idx = segment_idx * self.num_rods_per_segment + rod_idx
        chir = lax.switch(
            chir_lambda_sms_idx,
            self.chir_lambda_sms,
            *self.params_for_lambdify,
            *xi_epsed,
            s_local,
        ).squeeze()

        chir = jnp.roll(
            chir, 1
        )  # shift from [p_x, p_y, theta] (symbolic derivation def) to [theta, p_x, p_y] (SE(2) convention)

        return chir

    @eqx.filter_jit
    def forward_kinematics_platform(self, q: Array, segment_idx: Array) -> Array:
        """
        Evaluate the forward kinematics the platform
        Args:
            q: generalized coordinates of shape (num_dofs, )
            segment_idx: index of the segment
        Returns:
            chip: pose of the CoG of the platform in Cartesian-space with shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        chip = lax.switch(
            segment_idx, self.chip_lambda_sms, *self.params_for_lambdify, *xi_epsed
        ).squeeze()

        chip = jnp.roll(
            chip, 1
        )  # shift from [p_x, p_y, theta] (symbolic derivation def) to [theta, p_x, p_y] (SE(2) convention)

        return chip

    @eqx.filter_jit
    def forward_kinematics_end_effector(self, q: Array) -> Array:
        """
        Evaluate the forward kinematics of the end-effector
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
                Consists of [theta, p_x, p_y]
                where theta is the planar orientation with respect to the x-axis,
                p_x is the x-position, p_y is the y-position,
        """
        # map the configuration to the strains
        xi = self.strain(q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        # evaluate the symbolic expression
        chiee = self.chiee_lambda(*self.params_for_lambdify, *xi_epsed).squeeze()

        chiee = jnp.roll(
            chiee, 1
        )  # shift from [p_x, p_y, theta] (symbolic derivation def) to [theta, p_x, p_y] (SE(2) convention)

        return chiee

    @eqx.filter_jit
    def jacobian_end_effector(self, q: Array) -> Array:
        """
        Evaluate the Jacobian of the end-effector
        Args:
            q: generalized coordinates of shape (num_dofs, )
        Returns:
            Jee: the Jacobian of the end-effector pose with respect to the generalized coordinates.
                Jee is an array of shape (3, num_dofs).
        """
        # map the configuration to the strains
        xi = self.strain(q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        # evaluate the symbolic expression
        Jee = self.Jee_lambda(*self.params_for_lambdify, *xi_epsed)

        return Jee

    @eqx.filter_jit
    def jacobian_virtual_backbone(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the virtual backbone forward kinematics at a point s.

        The Jacobian maps configuration space velocities to operational space
        (Cartesian/task space) velocities at point s using the symbolic expressions.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            s: Point coordinate along the robot in the interval [0, L].

        Returns:
            J: Jacobian matrix of shape (3, num_dofs), mapping velocities in
                configuration space to velocities in operational space [omega_z, v_x, v_y].
        """
        # map the configuration to the strains
        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi)

        # determine in which segment the point is located
        segment_idx, s_local = self.classify_segment(s)

        # evaluate the symbolic Jacobian expression
        # The symbolic Jacobian is with respect to strains xi, shape (3, num_dofs)
        # with rows [p_x, p_y, theta] from symbolic derivation
        J_xi = lax.switch(
            segment_idx,
            self.Jv_lambda_sms,
            *self.params_for_lambdify,
            *xi_epsed,
            s_local,
        )

        # Reorder rows from [p_x, p_y, theta] to [theta, p_x, p_y] (SE(2) convention)
        J_xi = jnp.roll(J_xi, 1, axis=0)

        # Chain rule: J_q = J_xi @ B_xi (since xi = B_xi @ q + xi_ref)
        J = J_xi @ self.B_xi

        return J

    # Alias for consistency: jacobian is the same as jacobian_virtual_backbone
    jacobian = jacobian_virtual_backbone

    @eqx.filter_jit
    def jacobian_and_derivative_virtual_backbone(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time derivative of the virtual backbone at point s.

        Args:
            q: Generalized coordinates of shape (num_dofs,).
            qd: Generalized velocities of shape (num_dofs,).
            s: Point coordinate along the robot in the interval [0, L].

        Returns:
            J: Jacobian matrix of shape (3, num_dofs).
            Jd: Time derivative of the Jacobian, shape (3, num_dofs).
        """
        # Compute the Jacobian using the symbolic expression
        J = self.jacobian_virtual_backbone(q, s)

        # Compute the Jacobian derivative: d/dt(J(q(t))) = d(J)/dq @ qd
        # Define a function that computes the Jacobian for a given q
        def jacobian_fn(q: Array) -> Array:
            return self.jacobian_virtual_backbone(q, s)

        # Compute dJ/dq (shape: 3, num_dofs, num_dofs)
        dJ_dq = jacfwd(jacobian_fn, argnums=0)(q)

        # Contract with qd to get Jd: Jd[i, j] = sum_k (dJ[i, j]/dq[k] * qd[k])
        Jd = jnp.einsum("ijk,k->ij", dJ_dq, qd)

        return J, Jd

    # Alias for consistency: jacobian_and_derivative is the same as jacobian_and_derivative_virtual_backbone
    jacobian_and_derivative = jacobian_and_derivative_virtual_backbone

    @eqx.filter_jit
    def inverse_kinematics_end_effector(self, chiee: Array) -> Array:
        """
        Evaluates the inverse kinematics for a given end-effector pose.
            Important: only works for one segment!
        Args:
            params: Dictionary of robot parameters
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            q: generalized coordinates of shape (num_dofs, )
        """
        assert self.num_segments == 1, "Inverse kinematics only works for one segment!"

        # height of platform
        hp = self.pcudim[0, 1]
        # length of the proximal rod caps
        lpc = self.lpc[0]
        # length of the distal rod caps
        ldc = self.ldc[0]
        # offset of the end-effector from the distal surface of the platform
        chiee_off = self.chiee_off

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
                [jnp.cos(chiee[0]), -jnp.sin(chiee[0]), chiee[1]],
                [jnp.sin(chiee[0]), jnp.cos(chiee[0]), chiee[2]],
                [0.0, 0.0, 1.0],
            ]
        )

        # transformation from the distal end of the virtual backbone to the end-effector
        T_de_to_ee = jnp.array(
            [
                [jnp.cos(chiee_off[0]), -jnp.sin(chiee_off[0]), chiee_off[1]],
                [jnp.sin(chiee_off[0]), jnp.cos(chiee_off[0]), ldc + hp + chiee_off[2]],
                [0.0, 0.0, 1.0],
            ]
        )

        # compute the transformation from the proximal to the distal end of the virtual backbone
        T_pe_to_de = jnp.linalg.inv(T_b_to_pe) @ T_b_to_ee @ jnp.linalg.inv(T_de_to_ee)

        # compute the SE(2) pose from the transformation matrix
        vchi_pe_to_de = jnp.array(
            [
                jnp.arctan2(T_pe_to_de[1, 0], T_pe_to_de[0, 0]),
                T_pe_to_de[0, 2],
                T_pe_to_de[1, 2],
            ]
        )

        # extract the x and y position and the orientation
        th, px, py = vchi_pe_to_de[0], vchi_pe_to_de[1], vchi_pe_to_de[2]

        # add small eps for numerical stability
        th_sign = jnp.sign(th)
        # set zero sign to 1 (i.e. positive)
        th_sign = jnp.where(th_sign == 0, 1, th_sign)
        # add eps to the bending strain (i.e. the first column)
        th_epsed = th + th_sign * self.global_eps

        # compute the inverse kinematics for the virtual backbone
        vxi = (
            th_epsed
            / (2 * self.Lmax)
            * jnp.array(
                [
                    2,
                    py - (px * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                    -px - (py * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                ]
            )
        )

        # reference strains of the virtual backbone
        vxi_ref = self.ref_strains()

        # map the strains to the generalized coordinates
        q = jnp.linalg.pinv(self.B_xi) @ (vxi - vxi_ref)

        return q

    @eqx.filter_jit
    def _inertia_full_matrix(self, q: Array, eps: float | None = None) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            B_full (Array): Full inertia matrix of shape (num_dofs_max, num_dofs_max).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi, eps)

        B_full = self.B_lambda(*self.params_for_lambdify, *xi_epsed)

        return B_full

    @eqx.filter_jit
    def inertia_matrix(self, q: Array, eps: float | None = None) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            B (Array): Inertia matrix of shape (num_dofs, num_dofs).
        """
        if eps is None:
            eps = 1e4 * self.global_eps

        B_full = self._inertia_full_matrix(q, eps)

        B = self.B_xi.T @ B_full @ self.B_xi

        return B

    @eqx.filter_jit
    def _coriolis_full_matrix(
        self, q: Array, qd: Array, eps: float | None = None
    ) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            C_full (Array): Full Coriolis matrix of shape (num_dofs_max, num_dofs_max).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        xi = self.strain(q)
        xid = self.B_xi @ qd

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi, eps)

        C_full = self.C_lambda(*self.params_for_lambdify, *xi_epsed, *xid)

        return C_full

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array, eps: float | None = None) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            C (Array): Coriolis matrix of shape (num_dofs, num_dofs).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        C_full = self._coriolis_full_matrix(q, qd, eps)

        C = self.B_xi.T @ C_full @ self.B_xi

        return C

    @eqx.filter_jit
    def _gravitational_full_force(self, q: Array, eps: float | None = None) -> Array:
        """
        Compute the full gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            G (Array): Full gravitational vector of shape (num_dofs_max,).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi, eps)

        G_full = self.G_lambda(*self.params_for_lambdify, *xi_epsed).squeeze()

        return G_full

    @eqx.filter_jit
    def gravitational_force(self, q: Array, eps: float | None = None) -> Array:
        """
        Compute the gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            eps (float): small number to avoid singularities (e.g., division by zero). By default, it will be initialized to 1e4 * self.global_eps.

        Returns:
            G (Array): Gravitational vector of shape (num_dofs,).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        G_full = self._gravitational_full_force(q, eps)

        G = self.B_xi.T @ G_full

        return G

    @eqx.filter_jit
    def _stiffness_full_vector(self, q: Array) -> Array:
        """
        Compute the full stiffness vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            K_full (Array): Full stiffness vector of shape (num_dofs_max, ).
        """
        xi = self.strain(q)

        K_full = self.K_lambda(*self.params_for_lambdify, *xi).squeeze()

        return K_full

    @eqx.filter_jit
    def elastic_force(self, q: Array, z: Array) -> Array:
        """
        Compute the elastic force vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            z (Array): hysteresis state vector of shape (num_hysteresis, )

        Returns:
            K (Array): Stiffness vector of shape (num_dofs, ).
        """
        K_full = self._stiffness_full_vector(q)

        if self.consider_hysteresis is True:
            Shat = self.Shat()
            # add the post-yield potential forces
            K_full = self.hyst_alpha * K_full + (1 - self.hyst_alpha) * Shat @ (
                self.B_hyst @ z
            )

            # TODO: add post-yield potential forces (i.e., hysteresis effects) to the actuation vector

        K = self.B_xi.T @ K_full

        return K

    @eqx.filter_jit
    def _damping_full_matrix(self) -> Array:
        """
        Compute the full damping matrix of the robot.

        Returns:
            D (Array): Full damping matrix of shape (num_dofs_max, num_dofs_max).
        """
        D_full = self.D_lambda(*self.params_for_lambdify)

        return D_full

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            D (Array): Damping matrix of shape (num_dofs, num_dofs).
        """
        D_full = self._damping_full_matrix()

        D = self.B_xi.T @ D_full @ self.B_xi

        return D

    @eqx.filter_jit
    def _actuation_full_matrix(self, q: Array, phi: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            phi (Array): motor positions / twist angles of shape (num_segments * num_rods_per_segment, )

        Returns:
            alpha (Array): Actuation matrix of shape (num_dofs, num_dofs).
        """
        xi = self.strain(q)

        alpha = self.alpha_lambda(*self.params_for_lambdify, *xi, *phi).squeeze()

        return alpha

    @eqx.filter_jit
    def actuation_force(self, q: Array, phi: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            phi (Array): motor positions / twist angles of shape (num_segments * num_rods_per_segment, )

        Returns:
            alpha (Array): Actuation matrix of shape (num_dofs, num_dofs).
        """
        alpha = self._actuation_full_matrix(q, phi)

        # apply the strain basis
        alpha = self.B_xi.T @ alpha

        return alpha

    @eqx.filter_jit
    def Shat(self) -> Array:
        """
        Compute the nominal stiffness of the robot.

        Returns:
            Array: Nominal stiffness matrix of shape (num_dofs, num_dofs).
        """
        Shat = self.Shat_lambda(*self.params_for_lambdify)

        return Shat

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Compute the stiffness matrix of the robot in configuration space.

        Returns:
            K (Array): Stiffness matrix of shape (num_dofs, num_dofs).
        """
        Shat_full = self.Shat()
        K = self.B_xi.T @ Shat_full @ self.B_xi
        return K

    # -----------------------------------------
    # Energy methods
    # -----------------------------------------

    @eqx.filter_jit
    def gravitational_energy(self, q: Array, eps: float | None = None) -> Array:
        """
        Compute the gravitational potential energy of the robot.

        This uses the symbolic expression U_g derived from the gravitational
        potential energy of all rods, platforms, and payload.

        Args:
            q (Array): Generalized coordinates of shape (num_dofs,).
            eps (float): Small number to avoid singularities. Defaults to 1e4 * global_eps.

        Returns:
            U_g (Array): Gravitational potential energy [J] (scalar).
        """
        # initialize eps if not provided
        if eps is None:
            eps = 1e4 * self.global_eps

        xi = self.strain(q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = self.apply_eps_to_bend_strains(xi, eps)

        # evaluate the symbolic expression for gravitational potential energy
        U_g = self.U_g_lambda(*self.params_for_lambdify, *xi_epsed).squeeze()

        return U_g

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple[Array, Array | None]
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration, velocity, and possibly hysteresis state.
                Shape is (2 * num_dofs + num_hysteresis,).
            actuation_args (Tuple): Additional arguments for the actuation function.
                - u (Array): Actuation input.
                    If consider_underactuation is True, this is an array of shape (num_actuators, )
                    with motor positions / twist angles of the proximal end of the rods.
                    If consider_underactuation is False, this is an array of shape (num_dofs, )
                    with the configuration-space torques.
                - tau_ext (Array, optional): External generalized forces, shape (num_dofs,).

        Returns:
            yd: Time derivative of the state vector of shape (2 * num_dofs + num_hysteresis, ).
        """
        u, tau_ext = actuation_args
        if tau_ext is None:
            tau_ext = jnp.zeros((self.num_dofs,), dtype=y.dtype)

        if self.consider_hysteresis:
            q, qd, z = jnp.split(y, [self.num_dofs, 2 * self.num_dofs])
        else:
            q, qd = jnp.split(y, [self.num_dofs])
            z = jnp.zeros((self.num_hysteresis,), dtype=y.dtype)

        if self.consider_hysteresis:
            zd = (self.B_hyst.T @ qd) * (
                self.hyst_A
                - jnp.abs(z) ** self.hyst_n
                * (
                    self.hyst_gamma
                    + self.hyst_beta * jnp.sign((self.B_hyst.T @ qd) * z)
                )
            )
        else:
            zd = jnp.zeros((self.num_hysteresis,), dtype=y.dtype)

        if self.consider_underactuation is True:
            phi = u
            B = self.inertia_matrix(q)
            C = self.coriolis_matrix(q, qd)
            G = self.gravitational_force(q)
            tau_el = self.elastic_force(q, z)
            D = self.damping_matrix(q)
            alpha = self.actuation_force(q, phi)

        else:
            B = self.inertia_matrix(q)
            C = self.coriolis_matrix(q, qd)
            G = self.gravitational_force(q)
            tau_el = self.elastic_force(q, z)
            D = self.damping_matrix(q)

            phi = jnp.zeros((self.num_segments * self.num_rods_per_segment,))

            alpha = u

        # Inverse of the inertia matrix
        B_inv = jnp.linalg.inv(B)

        # Compute the acceleration
        qdd = B_inv @ (-C @ qd - G - tau_el - D @ qd + alpha + tau_ext)

        if self.consider_hysteresis:
            yd = jnp.concatenate([qd, qdd, zd])
        else:
            yd = jnp.concatenate([qd, qdd])

        return yd
