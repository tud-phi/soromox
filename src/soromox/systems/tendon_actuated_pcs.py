__all__ = ["TendonActuatedPCS"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
from typing import Callable, Dict, List, Optional

import soromox.utils.lie_algebra as lie
import soromox.actuation.tendon_actuation as act

from soromox.systems.pcs import PCS
from soromox.utils.integration import scale_gaussian_quadrature


class TendonActuatedPCS(PCS):
    """
    Piecewise Constant Strain (PCS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
    ----------
    num_segments : int
        Number of segments (constant strain sections) along the robot.
    num_actuators : int
        Number of actuators (control inputs) for the robot.
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
    tendon_routing_params : Dict[str, Array]
        Dictionary of arrays of length n_actuators representing the tendon parameters.
    d_s : Callable
        Function that returns the vector [d_x, d_y, d_z] of the tendon
        position w.r.t. the central backbone within the cross-sectional plane at a given
        abscissa point s. The three entries represent the coordinate of the tendon
        position with respect to the local cross-sectional frame at s. For this reason,
        d_x is always equal to 0 (as the backbone is pointing in the local x-direction)
    dd_s_ds : Callable
        Function that returns the vector of the derivative over s of d_s.

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

    """

    tendon_routing_params: Dict[str, Array]
    d_s: Callable
    dd_s_ds: Callable

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        *args,
        tendon_routing_basis: Optional[Dict[str, Callable]] = None,
        tendon_routing_params: Dict[str, Array],
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
            tendon_routing_basis (Optional[Dict[str, Callable]]):
                Dictionary with the tendon routing functions. If None, a linear routing is used.
                Expected keys and signatures:
                - "d_s": Callable[[Dict[str, Array], Array], Array]
                    Returns the 3D vector [d_x, d_y, d_z] giving the tendon position
                    with respect to the local cross-section at abscissa s. For typical routing,
                    d_x = 0 as the offset lies in the cross-sectional plane.
                - "dd_s_ds": Callable[[Dict[str, Array], Array], Array]
                    Returns the derivative over s of d_s.
            tendon_routing_params (Dict[str, Array]):
                Dictionary describing the tendon routing parameters for each actuator (length n_actuators).
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
        """
        super().__init__(
            num_segments,
            params,
            *args,
            **kwargs,
        )

        # Set default tendon routing basis to linear routing if not provided
        if tendon_routing_basis is None:
            tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_derivative,
            }
        self._set_tendon_routing_basis(tendon_routing_basis)
        self._set_tendon_routing_params(tendon_routing_params)

    def _set_tendon_routing_params(self, tendon_routing_params: Dict[str, Array]):
        """
        This internal function stores as attributes of the class the parameters of
        the tendon routings specified by the user.

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
                    f"The arrays in 'tendon_routing_params' must have the same length. Mismatch found in {key}."
                )
        for idx in tendon_routing_params["idx_seg_att"]:
            if idx >= self.num_segments:
                raise ValueError(
                    f'The indexes of the segments of attachment (tendon_routing_params["idx_seg_att"]) must be strictly '
                    + "lower than the number of segments of the robot. Got {idx}; num_segments = {self.num_segments}."
                )
        if self._check_tendon_routing_in_body(tendon_routing_params):
            raise UserWarning(f"Tendon(s) exit the robot body.")
        self.tendon_routing_params = tendon_routing_params

    def _set_tendon_routing_basis(self, tendon_routing_basis: Dict[str, Callable]):
        """
        This internal function stores as attributes of the class the basis functions of
        the tendon routings specified by the user.

        Args:
            tendon_routing_basis (Dict[str, Callable]): basis functions of the tendons
        """
        self.d_s = tendon_routing_basis["d_s"]
        self.dd_s_ds = tendon_routing_basis["dd_s_ds"]

    def _check_tendon_routing_in_body(self, tendon_routing_params):
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
            vmap(self.d_s, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0
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

    def update_tendon_routing_params(
        self, tendon_routing_params: Dict[str, Array]
    ) -> "TendonActuatedPCS":
        """
        This function updates the parameters of the tendon routings of the object.

        Args:
            tendon_routing_params (Dict[str, Array]): parameters of the tendons

        Returns:
            updated self (TendonActuatedPCS): self object with updated parameters
        """
        updated_self = self

        # Recompute derived parameters
        tendon_routing_params = dict(tendon_routing_params)
        num_actuators = len(list(tendon_routing_params.values())[0])

        # update all fields
        updated_self = eqx.tree_at(
            lambda x: x.tendon_routing_params, updated_self, tendon_routing_params
        )
        updated_self = eqx.tree_at(
            lambda x: x.num_actuators, updated_self, num_actuators
        )

        return updated_self
    
    @eqx.filter_jit
    def _local_actuation_basis(self, i: Array, xi_i: Array, s: Array, tendon_routing_params_k: Dict[str, Array]) -> Array:
        """
        Compute the actuation matrix contribution of one tendon k at s contained in segment i.

        Args:
            xi_i (Array): strain at segment i (6,)
            tendon_routing_params_k (Dict[str, Array]): parameters of the tendon k
            s (Array): abscissa points (num_gauss_points,)
        
        Returns:
            Phi_a_k (Array): local actuation matrix contribution of one tendon of shape (6,).
        """
        attachment_segment_idx = tendon_routing_params_k["idx_seg_att"]  # ()
        cond = attachment_segment_idx >= i  # ()

        # extract tendon routing evolution in the cross-sectional plane
        d_s = jnp.append(self.d_s(tendon_routing_params_k, s), 1.0)  # (4,)
        dd_s = jnp.append(
            self.dd_s_ds(tendon_routing_params_k, s), 1.0
        )  # (4,)

        term = (dd_s + lie.hat_SE3(xi_i) @ d_s)[:-1]  # (3,)
        norm = jnp.linalg.norm(term)  # ()
        t = term / norm  # (3,)

        Phi_a_k = cond * jnp.hstack([lie.tilde_SE3(d_s[:-1]) @ t, t])  # (6,)

        return Phi_a_k

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        # extract strains
        xi = self.strain(q).reshape((self.num_segments, 6))

        def A_segment_i(i: Array):
            """
            Compute the actuation matrix at the gaussian points of segment i of the robot.

            Args:
                i (Array): index of the segment ()

            Returns:
                A_i (Array): stack of actuation matrices of shape (num_gauss_points, 6, num_actuators).
            """
            xi_i = xi[i]

            def A_point_j(j: Array):
                """
                Compute the actuation matrix at the abscissa point corresponding to the gaussian point j.

                Args:
                    j (Array): index of the gaussian point ()

                Returns:
                    A_j (Array): local actuation matrix of shape (6, num_actuators).
                """
                # extract gaussian point and weight
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]

                # Vectorize the actuation basis computation for all tendons
                Phi_a_j = vmap(self._local_actuation_basis, in_axes=(None, None, None, 0), out_axes=(-1))(
                    i, xi_i, Xs_j, self.tendon_routing_params
                )  # (6, num_actuators)

                A_j = Phi_a_j

                return Ws_j * A_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )

            # Vectorize the actuation matrix computation for all gaussian points
            A_i = vmap(A_point_j)(
                jnp.arange(self.num_gauss_points)
            )  # (num_gauss_points, 6, num_actuators)

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_point_j(j) for j in range(self.num_gauss_points)], axis=0)
            # print('A_blocks_i =\n', A_blocks_i.shape)

            return A_i

        # Vectorize the actuation matrix computation for all segments
        A_blocks = vmap(A_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, 6, num_actuators)

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_segment_i(i) for i in range(self.num_segments)], axis=0)
        # print('A_blocks_tot =\n', A_blocks_tot.shape)

        A = jnp.sum(A_blocks, axis=(1, ))  # Sum the Gauss points

        # reshape A to (6 * num_segments, num_actuators)
        A = A.reshape((-1, self.num_actuators))

        # Project into the active strains space
        A = self.B_xi.T @ A  # (num_active_strains, num_actuators)

        return A

    @eqx.filter_jit
    def forward_kinematics_tendons(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the tendon actuators at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,)
            s (Array): point coordinate along the robot ()

        Returns:
            t_s (Array): cartesian position of the tendons at s, shape (n_actuators, 3)
        """

        def forward_kinematics_tendon_k(
            single_tendon_routing_params: Dict[str, Array], q: Array, s: Array
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
                self.d_s(single_tendon_routing_params, s_val), 1.0
            )  # (4,)
            t_k_s = t_k_s[:-1]  # (3,)

            return t_k_s

        # Vectorize the forward kinematics computation for all tendons
        t_s = vmap(forward_kinematics_tendon_k, in_axes=(0, None, None))(
            self.tendon_routing_params, q, s
        )  # (n_actuators, 3)

        return t_s
