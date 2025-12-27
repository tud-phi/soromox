__all__ = ["TendonActuatedPCS"]
from collections.abc import Callable

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

import soromox.actuation.tendon_actuation as act
import soromox.utils.lie_algebra as lie
from soromox.utils.integration import scale_gaussian_quadrature

from .pcs import PCS


class TendonActuatedPCS(PCS):
    """
    Piecewise Constant Strain (PCS) model for 3D soft continuum robots with tendon actuation.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    In addition, it supports the implementation of (optional) passive tendons to be routed
    within the robot, with one end internally attached to the end of a segment (as for the
    active tendons) and with the other end externally attached to a spring-damper unit.
    This serves as additional degrees of freedom of the physical design, which can be
    optimized to shift some of the control effort from digital to mechanical (see reference
    below [Physical control]).

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot.
    n_p : int
        Number of passive tendons in the robot.
    g0 : Array
        Initial pose of the robot base as an SE(3) transformation matrix.
    g : Array
        Gravitational acceleration vector (embedded in a 6D vector).
        [0, 0, 0, g_x, g_y, g_z]
    L, r, E, G, rho, D : Array
        Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
    num_active_strains : int
        Number of active strain components (based on strain_selector).
    num_strains : int
        Total number of strain components (6 * num_segments).
    B_xi : Array
        Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
    xi_ref : Array
        Reference strain (reference configuration) of the robot.
    num_gauss_points : int
        Number of points used for numerical integration.
        Corresponds to the order of Gauss-Legendre quadrature + 2 (for the endpoints).
    Xs, Ws : Array
        Gauss-Legendre quadrature nodes and weights for numerical integration.
    active_tendon_routing_params : Dict[str, Array]
        Dictionary of arrays of length n_actuators representing the tendon parameters.
    active_d_s : Callable
        Function that returns the vector [d_x, d_y, d_z] of the active tendon
        position w.r.t. the central backbone within the cross-sectional plane at a given
        abscissa point s. The three entries represent the coordinate of the tendon
        position with respect to the local cross-sectional frame at s. For this reason,
        d_x is always equal to 0 (as the backbone is pointing in the local x-direction)
    active_dd_s_ds : Callable
        Function that returns the vector of the derivative over s of active_d_s.
    passive_tendon_routing_params : Dict[str, Array]
        Dictionary of arrays of length n_p representing the passive tendon parameters.
    passive_d_s : Callable
        As active_d_s for the passive tendons.
    passive_dd_s_ds : Callable
        As active_dd_s_ds for the passive tendons.
    K_pt: Array
        Stiffness matrix of the passive tendons (n_p, n_p).
    D_pt: Array
        Damping matrix of the passive tendons (n_p, n_p).
    l_pt0: Array
        Vector of the initial displacement of the passive tendons (n_p,).

    Notes:
    -----
    - The strain vector is composed of 6 components per segment:
      [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z].
      By default, the rod is assumed to be straight and aligned with the x-axis,
        so the reference strain is set to [0, 0, 0, 1, 0, 0].
        Thus:   - kappa_x corresponds to torsion around the x-axis,
                - kappa_y corresponds to bending around the y-axis,
                - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to axial strain along the x-axis,
                - sigma_y corresponds to shear along the y-axis,
                - sigma_z corresponds to shear along the z-axis.

    References:
    ----------
    - [PCS model] Renda, Federico, Frédéric Boyer, Jorge Dias, and Lakmal Seneviratne. "Discrete cosserat approach for multisection soft manipulator dynamics." IEEE Transactions on Robotics 34, no. 6 (2018): 1518-1533.
    - [Actuation matrix for tendon-actuated soft robots] Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020). A geometric variable-strain approach for static modeling of soft manipulators with tendon and fluidic actuation. IEEE Robotics and Automation Letters, 5(3), 4006-4013.
    - [Tendon lengths] Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024). Input decoupling of lagrangian systems via coordinate transformation: General characterization and its application to soft robotics. IEEE transactions on robotics, 40, 2098-2110.
    - [Physical control] Milana, E., Santina, C. D., Gorissen, B., & Rothemund, P. (2025). Physical control: A new avenue to achieve intelligence in soft robotics. Science Robotics, 10(102), eadw7660.

    """

    n_p: int
    active_tendon_routing_params: dict[str, Array]
    active_d_s: Callable = eqx.field(static=True)
    active_dd_s_ds: Callable = eqx.field(static=True)
    passive_tendon_routing_params: dict[str, Array]
    passive_d_s: Callable = eqx.field(static=True)
    passive_dd_s_ds: Callable = eqx.field(static=True)
    K_pt: Array
    D_pt: Array
    l_pt0: Array

    def __init__(
        self,
        num_segments: int,
        params: dict[str, Array],
        *args,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        active_tendon_routing_params: dict[str, Array] | None = None,
        passive_tendon_routing_basis: dict[str, Callable] | None = None,
        passive_tendon_routing_params: dict[str, Array] | None = None,
        passive_tendon_params: dict[str, Array] | None = None,
        **kwargs,
    ):
        """
        Initialize the TendonActuatedPCS class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "p0": (optional) List/Array of shape (6,)
                    Initial orientation and position in the inertial frame [rad, m]
                    [ψ, θ, φ, x0, y0, z0]
                        [ψ, θ, φ] are the Euler angles in the ZXZ convention:
                            ψ (psi): Rotation around Z axis (fixed axis)
                            θ (theta): Rotation around X' axis (after first rotation)
                            φ (phi): Rotation about Z' axis (after the first two rotations)
                        [x0, y0, z0]: Base position in the inertial frame
                    Defaults to [pi/2, pi/2, 0.0, 0.0, 0.0, 0.0].
                - "L": List/Array of num_segments floats
                    Length of each segment [m]
                - "r": List/Array of num_segments floats
                    Radius of each segment [m]
                - "rho": List/Array of num_segments floats
                    Density of each segment [kg/m^3]
                - "g": List/Array of 3 floats [gx, gy, gz]
                    Gravitational acceleration vector [m/s^2]
                - "E": List/Array of num_segments floats
                    Elastic modulus of each segment [Pa]
                - "G": List/Array of num_segments floats
                    Shear modulus of each segment [Pa]
                - "D": Array of shape (6*num_segments, 6*num_segments)
                    Damping matrix [Pa·s]
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (6 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_ref (Optional[Array], optional):
                Reference strain of shape (6 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local x-axis).
            active_tendon_routing_basis (Optional[Dict[str, Callable]]):
                Dictionary with the active tendon routing functions. If None, a linear routing is used.
                Expected keys and signatures:
                - "d_s": Callable[[Dict[str, Array], Array], Array]
                    Returns the 3D vector [d_x, d_y, d_z] giving the tendon position
                    with respect to the local cross-section at abscissa s. For typical routing,
                    d_x = 0 as the offset lies in the cross-sectional plane.
                - "dd_s_ds": Callable[[Dict[str, Array], Array], Array]
                    Returns the derivative over s of d_s.
            active_tendon_routing_params (Optional[Dict[str, Array]]):
                Dictionary describing the active tendon routing parameters for each actuator (length n_actuators).
                When using the default linear routing, the following keys are expected:
                - "ry": Array (n_actuators,)
                    y-intercept of the tendon line at the base [m].
                - "my": Array (n_actuators,)
                    Slope in the x–y plane [-].
                - "rz": Array (n_actuators,)
                    z-intercept of the tendon line at the base [m].
                - "mz": Array (n_actuators,)
                    Slope in the x–z plane [-].
                - "idx_seg_att": Array (n_actuators,)
                    Attachment segment index for each tendon (0-based, inclusive). The tendon contributes
                    along the backbone up to and including this segment’s distal end.
            passive_tendon_routing_basis (Optional[Dict[str, Callable]]):
                Same as active_tendon_routing_basis for the passive tendons.
            passive_tendon_routing_params (Optional[Dict[str, Array]]):
                Same as active_tendon_routing_params for the passive tendons.
            passive_tendon_params (optional[Dict[str, Array]]):
                Dictionary containing the parameters of the spring-damper attached to each passive tendon:
                - "k_pt": List/Array of n_p floats
                    Stiffness coefficients of the springs attached to the passive tendons (n_p,) [N / m]
                - "d_pt": List/Array of n_p floats
                    Damping coefficients of the dampers attached to the passive tendons (n_p,) [N s / m]
                - "l_pt0": List/Array of n_p floats
                    Initial displacement of the passive tendons (pull is negative) (n_p,) [m]

        """
        super().__init__(
            num_segments,
            params,
            *args,
            **kwargs,
        )

        # Set default active tendon routing basis to linear routing if not provided
        if active_tendon_routing_basis is None:
            active_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_derivative,
            }
        if active_tendon_routing_params is None:
            active_tendon_routing_params = {
                "ry": jnp.array([]),
                "rz": jnp.array([]),
                "my": jnp.array([]),
                "mz": jnp.array([]),
                "idx_seg_att": jnp.array([], dtype=jnp.int32),
            }
        self.active_d_s, self.active_dd_s_ds = self._set_tendon_routing_basis(
            active_tendon_routing_basis
        )
        self.active_tendon_routing_params = self._set_active_tendon_routing_params(
            active_tendon_routing_params, self.active_d_s
        )

        # Set default passive tendon routing basis to linear routing if not provided
        if passive_tendon_routing_basis is None:
            passive_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_derivative,
            }
        if passive_tendon_routing_params is None:
            passive_tendon_routing_params = {
                "ry": jnp.array([]),
                "rz": jnp.array([]),
                "my": jnp.array([]),
                "mz": jnp.array([]),
                "idx_seg_att": jnp.array([], dtype=jnp.int32),
            }
        self.passive_d_s, self.passive_dd_s_ds = self._set_tendon_routing_basis(
            passive_tendon_routing_basis
        )
        self.passive_tendon_routing_params = self._set_passive_tendon_routing_params(
            passive_tendon_routing_params, self.passive_d_s
        )

        # Set physical parameters of the passive tendons
        if passive_tendon_params is None:
            passive_tendon_params = {
                "k_pt": jnp.array([]),
                "d_pt": jnp.array([]),
                "l_pt0": jnp.array([]),
            }
        self._set_passive_tendon_params(passive_tendon_params)

    def _set_passive_tendon_params(self, passive_tendon_params: dict[str, Array]):
        """
        This internal function stores as attributes of the class the physical
        parameters of the passive tendons specified by the user.

        Args:
            passive_tendon_params (Dict[str, Array]): parameters of the tendons
        """
        if not isinstance(passive_tendon_params, (dict)):
            raise TypeError(
                "The parameter 'tendon_routing_params' must be a dictionary of jax.Array."
            )
        for key, val in passive_tendon_params.items():
            if len(val) != self.n_p:
                raise ValueError(
                    f"The arrays in 'passive_tendon_params' must have the same length. Mismatch found in {key}."
                )
        self.K_pt = jnp.diag(jnp.array(passive_tendon_params["k_pt"]))
        self.D_pt = jnp.diag(jnp.array(passive_tendon_params["d_pt"]))
        self.l_pt0 = jnp.array(passive_tendon_params["l_pt0"])

    def _set_active_tendon_routing_params(
        self, tendon_routing_params: dict[str, Array], d_s: Callable
    ):
        """
        This internal function stores as attributes of the class the parameters of
        the active tendon routings specified by the user.

        Args:
            tendon_routing_params (Dict[str, Array]): parameters of the tendons
        """
        if not isinstance(tendon_routing_params, (dict)):
            raise TypeError(
                "The parameter 'tendon_routing_params' must be a dictionary of jax.Array."
            )
        self.num_actuators = len(list(tendon_routing_params.values())[0])
        for key, val in tendon_routing_params.items():
            if len(val) != self.num_actuators:
                raise ValueError(
                    f"The arrays in 'active_tendon_routing_params' must have the same length. Mismatch found in {key}."
                )
        for idx in tendon_routing_params["idx_seg_att"]:
            if idx >= self.num_segments:
                raise ValueError(
                    'The indexes of the segments of attachment (active_tendon_routing_params["idx_seg_att"]) must be strictly '
                    + "lower than the number of segments of the robot. Got {idx}; num_segments = {self.num_segments}."
                )
        if self._check_tendon_routing_in_body(tendon_routing_params, d_s):
            raise UserWarning("Active tendon(s) exit the robot body.")
        return tendon_routing_params

    def _set_passive_tendon_routing_params(
        self, tendon_routing_params: dict[str, Array], d_s: Callable
    ):
        """
        This internal function stores as attributes of the class the parameters of
        the passive tendon routings specified by the user.

        Args:
            tendon_routing_params (Dict[str, Array]): parameters of the tendons
        """
        if not isinstance(tendon_routing_params, (dict)):
            raise TypeError(
                "The parameter 'tendon_routing_params' must be a dictionary of jax.Array."
            )
        self.n_p = len(list(tendon_routing_params.values())[0])
        for key, val in tendon_routing_params.items():
            if len(val) != self.n_p:
                raise ValueError(
                    f"The arrays in 'passive_tendon_routing_params' must have the same length. Mismatch found in {key}."
                )
        for idx in tendon_routing_params["idx_seg_att"]:
            if idx >= self.num_segments:
                raise ValueError(
                    'The indexes of the segments of attachment (passive_tendon_routing_params["idx_seg_att"]) must be strictly '
                    + "lower than the number of segments of the robot. Got {idx}; num_segments = {self.num_segments}."
                )
        if self._check_tendon_routing_in_body(tendon_routing_params, d_s):
            raise UserWarning("Passive tendon(s) exit the robot body.")
        return tendon_routing_params

    def _set_tendon_routing_basis(self, tendon_routing_basis: dict[str, Callable]):
        """
        This internal function stores as attributes of the class the basis functions of
        the tendon routings specified by the user.

        Args:
            tendon_routing_basis (Dict[str, Callable]): basis functions of the tendons
        """
        return tendon_routing_basis["d_s"], tendon_routing_basis["dd_s_ds"]

    def _check_tendon_routing_in_body(self, tendon_routing_params, d_s):
        """
        Checks whether the tendons are correctly inside the robot body. This function
        computes the distance between the centerline of the cross-section and the tendons
        w.r.t. the local frame for a set of abscissa points along the body. If the any of
        such distances is greater than the radius of the respective segment, it yields
        true.

        Args:
            tendon_routing_params : Dict[str, Array]
                Dictionary of arrays of length n_actuators representing the tendon parameters.

        Returns:
            flag (bool): True if any of the tendons is out of the body
        """

        def r_s(s: Array):
            """
            Returns the radius of the body at the specified abscissa point s.

            Args:
                s (Array): abscissa point along the body

            Returns:
                r_s (Array): radius of the robot at s
            """
            cond = self.L_cum <= s
            idx = jnp.sum(cond) - 1
            return self.r[idx]

        s = jnp.linspace(0.0, self.L_cum[-1], 75)
        t = vmap(
            vmap(d_s, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0
        )(tendon_routing_params, s)  # (num_tendons, N, 3)
        d = t[:, :, 1:]  # (num_tendons, N, 2)
        r_tendons = jnp.linalg.norm(d, axis=2)  # (num_tendons, N)
        r_body = vmap(r_s)(s)  # (N,)
        check = r_tendons > r_body  # (num_tendons, N)
        flag = check.any()

        # idxs = jnp.argwhere(check)
        # tendon_idx = idxs[0,0]
        # s_val = idxs[1,0]

        return flag

    def update_active_tendon_routing_params(
        self, tendon_routing_params: dict[str, Array]
    ) -> "TendonActuatedPCS":
        """
        This function updates the parameters of the active tendons of the object.

        Args:
            tendon_routing_params (Dict[str, Array]): parameters of the tendons

        Returns:
            updated_self (TendonActuatedPCS): self object with updated parameters
        """
        updated_self = self
        updated_params = self.active_tendon_routing_params.copy()

        for key, val in tendon_routing_params.items():
            if key not in updated_params:
                raise KeyError(
                    f"Attempted to update unknown active tendon routing parameter '{key}'. "
                    f"Valid keys are: {list(updated_params.keys())}"
                )
            updated_params[key] = val

        updated_self = eqx.tree_at(
            lambda x: x.active_tendon_routing_params, updated_self, updated_params
        )

        return updated_self

    def update_passive_tendon_routing_params(
        self, tendon_routing_params: dict[str, Array]
    ) -> "TendonActuatedPCS":
        """
        This function updates the parameters of the passive tendons of the object.

        Args:
            tendon_routing_params (Dict[str, Array]): parameters of the tendons

        Returns:
            updated_self (TendonActuatedPCS): self object with updated parameters
        """
        updated_self = self
        updated_params = self.passive_tendon_routing_params.copy()

        for key, val in tendon_routing_params.items():
            if key not in updated_params:
                raise KeyError(
                    f"Attempted to update unknown passive tendon routing parameter '{key}'. "
                    f"Valid keys are: {list(updated_params.keys())}"
                )
            updated_params[key] = val

        updated_self = eqx.tree_at(
            lambda x: x.passive_tendon_routing_params, updated_self, updated_params
        )

        return updated_self

    def update_passive_tendon_params(
        self, tendon_params: dict[str, Array]
    ) -> "TendonActuatedPCS":
        """
        This function updates the physical parameters of the passive tendons of the object.

        Args:
            tendon_params (Dict[str, Array]): parameters of the tendons

        Returns:
            updated_self (TendonActuatedPCS): self object with updated parameters
        """
        updated_self = self

        if "k_pt" in tendon_params:
            k_pt = jnp.asarray(tendon_params["k_pt"])
            updated_self = eqx.tree_at(lambda x: x.K_pt, updated_self, jnp.diag(k_pt))

        if "d_pt" in tendon_params:
            d_pt = jnp.asarray(tendon_params["d_pt"])
            updated_self = eqx.tree_at(lambda x: x.D_pt, updated_self, jnp.diag(d_pt))

        if "l_pt0" in tendon_params:
            l_pt0 = jnp.asarray(tendon_params["l_pt0"])
            updated_self = eqx.tree_at(lambda x: x.l_pt0, updated_self, l_pt0)

        return updated_self

    @eqx.filter_jit
    def _local_actuation_basis(
        self,
        i: Array,
        xi_i: Array,
        s: Array,
        tendon_routing_params_k: dict[str, Array],
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
    ) -> Array:
        """
        Compute the actuation matrix contribution of one tendon k at s contained in segment i.

        Args:
            i (Array): index of the segment ()
            xi_i (Array): strain at segment i (6,)
            tendon_routing_params_k (Dict[str, Array]): parameters of the tendon k
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            Phi_a_k (Array): local actuation matrix contribution of one tendon of shape (6,).
        """
        attachment_segment_idx = tendon_routing_params_k["idx_seg_att"]  # ()
        is_tendon_active = attachment_segment_idx >= i  # ()

        # extract tendon routing evolution in the cross-sectional plane
        d_s = jnp.append(d_s_fn(tendon_routing_params_k, s), 1.0)  # (4,)
        dd_s = jnp.append(dd_s_ds_fn(tendon_routing_params_k, s), 1.0)  # (4,)

        term = (dd_s + lie.hat_SE3(xi_i) @ d_s)[:-1]  # (3,)
        norm = jnp.linalg.norm(term)  # ()
        t = term / norm  # (3,)

        Phi_a_k = is_tendon_active * jnp.hstack(
            [lie.tilde_SE3(d_s[:-1]) @ t, t]
        )  # (6,)

        return Phi_a_k

    @eqx.filter_jit
    def actuation_matrix(self, q: Array):
        return self._actuation_matrix(
            q, self.active_tendon_routing_params, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def jacobian_passive_tendon(self, q: Array):
        return self._actuation_matrix(
            q,
            self.passive_tendon_routing_params,
            self.passive_d_s,
            self.passive_dd_s_ds,
        ).T

    @eqx.filter_jit
    def _actuation_matrix(
        self,
        q: Array,
        tendon_routing_params: dict[str, Array],
        d_s: Callable,
        dd_s_ds: Callable,
    ) -> Array:
        """
        Compute the actuation matrix of the robot.
        This implementation is based on the formulation presented in:
        Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020). A geometric variable-strain approach for static modeling of soft manipulators with tendon and fluidic actuation. IEEE Robotics and Automation Letters, 5(3), 4006-4013.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, nt).
        """
        # extract strains
        xi = self.strain(q).reshape((self.num_segments, 6))

        # number of tendons
        nt = len(list(tendon_routing_params.values())[0])

        def A_segment_i(i: Array):
            """
            Compute the actuation matrix at the gaussian points of segment i of the robot.

            Args:
                i (Array): index of the segment ()

            Returns:
                A_i (Array): stack of actuation matrices of shape (num_gauss_points, 6, nt).
            """
            xi_i = xi[i]

            def A_point_j(j: Array):
                """
                Compute the actuation matrix at the abscissa point corresponding to the gaussian point j.

                Args:
                    j (Array): index of the gaussian point ()

                Returns:
                    A_j (Array): local actuation matrix of shape (6, nt).
                """
                # extract gaussian point and weight
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]

                # Vectorize the actuation basis computation for all tendons
                Phi_a_j = vmap(
                    self._local_actuation_basis,
                    in_axes=(None, None, None, 0, None, None),
                    out_axes=(-1),
                )(i, xi_i, Xs_j, tendon_routing_params, d_s, dd_s_ds)  # (6, nt)

                A_j = Phi_a_j

                return Ws_j * A_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )

            # Vectorize the actuation matrix computation for all gaussian points
            A_i = vmap(A_point_j)(
                jnp.arange(self.num_gauss_points)
            )  # (num_gauss_points, 6, nt)

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_point_j(j) for j in range(self.num_gauss_points)], axis=0)
            # print('A_blocks_i =\n', A_blocks_i.shape)

            return A_i

        # Vectorize the actuation matrix computation for all segments
        A_blocks = vmap(A_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, 6, nt)

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_segment_i(i) for i in range(self.num_segments)], axis=0)
        # print('A_blocks_tot =\n', A_blocks_tot.shape)

        A = jnp.sum(A_blocks, axis=(1,))  # Sum the Gauss points

        # reshape A to (6 * num_segments, nt)
        A = A.reshape((6 * self.num_segments, nt))

        # Project into the active strains space
        A = self.B_xi.T @ A  # (num_active_strains, nt)

        return A

    @eqx.filter_jit
    def active_tendon_length(self, q: Array) -> Array:
        return self._tendon_length(
            q, self.active_tendon_routing_params, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        return self._tendon_length(
            q,
            self.passive_tendon_routing_params,
            self.passive_d_s,
            self.passive_dd_s_ds,
        )

    tendon_length = active_tendon_length  # Alias for compatibility
    actuated_coordinates = active_tendon_length  # Alias for actuation space dynamics

    @eqx.filter_jit
    def _tendon_length(
        self,
        q: Array,
        tendon_routing_params: dict[str, Array],
        d_s: Callable,
        dd_s_ds: Callable,
    ) -> Array:
        """
        Compute the length of all tendons of the robot.
        This implementation is based on the formulation presented in:
        Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024). Input decoupling of lagrangian systems via coordinate transformation: General characterization and its application to soft robotics. IEEE transactions on robotics, 40, 2098-2110.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            l (Array): Lengths of all tendons of shape (num_actuators,).
        """
        # extract strains
        xi = self.strain(q).reshape((self.num_segments, 6))

        def tendon_length_density_segment_i(i: Array):
            """
            Compute the tendon length density at all Gauss points of segment i.

            Args:
                i (Array): segment index.

            Returns:
                dl_ds_i (Array): tendon length density values at Gauss points,
                    shape (num_gauss_points, num_actuators).
            """
            xi_i = xi[i]

            def tendon_length_density_point_j(j: Array):
                """
                Evaluate the tendon length density at the Gauss point j inside segment i.

                Args:
                    j (Array): Gauss point index.

                Returns:
                    dl_ds_j (Array): local tendon length density of shape (num_actuators,)
                        at Gauss point j.
                """
                # extract gaussian point and weight
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]

                def tendon_length_density_tendon_k(
                    tendon_routing_params_k: dict[str, Array],
                ) -> Array:
                    """
                    Compute the tendon length density contribution of one tendon at the
                    current Gauss point.

                    Args:
                        tendon_routing_params_k (Dict[str, Array]): parameters of the tendon.

                    Returns:
                        dl_ds_k (Array): scalar tendon length density contribution.
                    """
                    # extract tendon routing evolution in the cross-sectional plane
                    dd_s = jnp.append(
                        dd_s_ds(tendon_routing_params_k, Xs_j), 1.0
                    )  # (4,)

                    # compute local actuation basis
                    Phi_a_k = self._local_actuation_basis(
                        i, xi_i, Xs_j, tendon_routing_params_k, d_s, dd_s_ds
                    )  # (6,)

                    # compute tendon length density contribution
                    dl_ds_k = jnp.dot(
                        Phi_a_k.T,
                        xi_i + jnp.concat([jnp.zeros((3,)), dd_s[:3]], axis=-1),
                    )  # ()

                    return dl_ds_k

                # Vectorize the tendon length density computation for all tendons
                dl_ds_j = vmap(tendon_length_density_tendon_k)(
                    tendon_routing_params
                )  # (num_actuators,)

                return Ws_j * dl_ds_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )

            # Vectorize the tendon length density evaluation for all Gauss points
            dl_ds_i = vmap(tendon_length_density_point_j)(
                jnp.arange(self.num_gauss_points)
            )  # (num_gauss_points, num_actuators)

            return dl_ds_i

        # Vectorize the tendon length density evaluation for all segments
        dl_ds_blocks = vmap(tendon_length_density_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, num_actuators)

        l = jnp.sum(dl_ds_blocks, axis=(0, 1))  # Sum over the segments and Gauss points

        return l

    @eqx.filter_jit
    def forward_kinematics_active_tendons(self, q: Array, s: Array) -> Array:
        return self._forward_kinematics_tendons(
            q, s, self.active_tendon_routing_params, self.active_d_s
        )

    @eqx.filter_jit
    def forward_kinematics_passive_tendons(self, q: Array, s: Array) -> Array:
        return self._forward_kinematics_tendons(
            q, s, self.passive_tendon_routing_params, self.passive_d_s
        )

    @eqx.filter_jit
    def forward_kinematics_tendons(self, q: Array, s: Array) -> Array:
        """
        Forward kinematics for all tendons (active + passive) at abscissa s.

        Args:
            q: Generalized coordinates (num_active_strains,)
            s: Position along the backbone

        Returns:
            Cartesian tendon positions of shape (n_actuators, 3)
        """
        active_pos = self.forward_kinematics_active_tendons(q, s)
        passive_pos = self.forward_kinematics_passive_tendons(q, s)

        if active_pos.size == 0:
            return passive_pos
        if passive_pos.size == 0:
            return active_pos
        return jnp.concatenate([active_pos, passive_pos], axis=0)

    @eqx.filter_jit
    def _forward_kinematics_tendons(
        self, q: Array, s: Array, tendon_routing_params: dict[str, Array], d_s: Callable
    ) -> Array:
        """
        Compute the forward kinematics of the tendon actuators at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,)
            s (Array): point coordinate along the robot ()

        Returns:
            t_s (Array): cartesian position of the tendons at s, shape (n_actuators, 3)
        """

        def forward_kinematics_tendon_k(
            single_tendon_routing_params: dict[str, Array], q: Array, s: Array
        ) -> Array:
            """
            Compute the forward kinematics of one tendon actuator at a point s along the robot.
            If s is greater than the length of the tendon, the function returns the last valid
            position of tendon (i.e., its attachement point).

            Args:
                tendon_routing_params : Dict[str, Array]
                    Dictionary of arrays of length n_actuators representing the tendon parameter for tendon k.
                q (Array): generalized coordinates of shape (num_active_strains,)
                s (Array): point coordinate along the robot ()

            Returns:
                t_k_s (Array): cartesian position of the tendons at s, shape (3,)
            """
            lt = self.L_cum[single_tendon_routing_params["idx_seg_att"] + 1]  # ()
            s_val = jnp.clip(s, 0.0, lt)  # ()

            g_s = self.forward_kinematics(q, s_val)  # (4,4)
            t_k_s = g_s @ jnp.append(
                d_s(single_tendon_routing_params, s_val), 1.0
            )  # (4,)
            t_k_s = t_k_s[:-1]  # (3,)

            return t_k_s

        # Vectorize the forward kinematics computation for all tendons
        t_s = vmap(forward_kinematics_tendon_k, in_axes=(0, None, None))(
            tendon_routing_params, q, s
        )  # (n_actuators, 3)

        return t_s

    # ===========================================
    # Dynamical matrices computation
    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            tau_el_tot (Array): Total eElastic force of shape (num_active_strains,).
        """
        # Stiffness of the body
        tau_el = super().elastic_force(q)

        # Stiffness of the springs attached to the passive tendons
        J_pt = self.jacobian_passive_tendon(q)
        l_pt = self.passive_tendon_length(q)
        tau_el_pt = J_pt.T @ self.K_pt @ (l_pt - self.l_pt0)

        tau_el_tot = tau_el + tau_el_pt
        return tau_el_tot

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            D_tot (Array): Total damping matrix of shape (num_active_strains, num_active_strains).
        """
        # Damping of the body
        D = super().damping_matrix(q)

        # Damping of the dampers attached to the passive tendons
        # Note: jacobian_passive_tendon returns J_pt in active strain space (n_p, num_dofs),
        # so J_pt.T @ D_pt @ J_pt is already in active strain space (num_dofs, num_dofs)
        J_pt = self.jacobian_passive_tendon(q)
        D_pt = J_pt.T @ self.D_pt @ J_pt

        D_tot = D + D_pt
        return D_tot

    @eqx.filter_jit
    def elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_K_tot (float): Total elastic energy of the robot.
        """
        # Elastic energy of the body
        U_K = super().elastic_energy(q)

        # Elastic energy of the passive tendons
        l_pt = self.passive_tendon_length(q)
        U_K_pt = 0.5 * (l_pt - self.l_pt0).T @ self.K_pt @ (l_pt - self.l_pt0)

        U_K_tot = U_K + U_K_pt
        return U_K_tot
