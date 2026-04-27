__all__ = ["TendonActuatedGVS"]
from collections.abc import Callable
from typing import Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

import soromox.actuation.tendon_actuation as act
import soromox.utils.lie_algebra as lie

from .core import GVS


class TendonActuatedGVS(GVS):
    """
      Geometric Variable Strain (GVS) model for 3D soft continuum robots with tendon actuation.

      This class implements the geometric and dynamic modeling of a 3D soft robot
      using the Cosserat rod theory with geometric variable strain parametrizations.
      It supports computation of forward kinematics, Jacobians, and dynamic matrices
      for robots with arbitrary combinations of link cross-sections, joint types, and
      strain basis functions. Additionally, it includes tendon actuation capabilities.

    Attributes:
        num_segments: Number of segments (links) in the robot.
        max_dof: Maximum number of degrees of freedom (DOFs) per link or joint.
        max_num_gauss_points: Maximum number of Gauss points per link used for integration.
        max_num_integration_points: Maximum number of integration points per link (= max_num_gauss_points + 2).
        num_dofs: Total number of DOFs for the robot (sum of joint + link DOFs).
        num_padded_dofs: Theoretical maximum number of DOFs (num_segments * 2 * max_dof).
        active_dof_map: Strain basis selection matrix mapping active DOFs to the full strain space.
        segment_lengths, segment_end_positions: Length of each link, and cumulative link lengths.
        num_integration_points: Number of integration/evaluation points for each link.
        dofs_per_segment: Number of DOFs for each link/joint pair (shape: num_segments x 2).
        integration_points, integration_weights: Gauss quadrature nodes and weights for each link.
        mass_matrices, stiffness_matrices, damping_matrices: Mass, stiffness, and damping matrices at integration points.
        B_joint, B_Xs, B_Z1, B_Z2: Basis matrices for joints and links at quadrature points and intermediate points.
        xi_ref_joint, xi_ref_Xs, xi_ref_Z1, xi_ref_Z2: Reference strain vectors for joints and links.
        joint_stiffness: Joint stiffness matrices.
        g: Gravitational acceleration vector in 6D wrench form.
        basis_type_index: Index of the strain basis type used for each segment.
        basis_active_params, basis_order_params: Parameters controlling the strain basis DOFs and orders.

      Notes
      -----
      - The GVS model generalizes PCS by allowing the strain distribution in each segment
        to be expressed in arbitrary basis functions (monomials, Fourier, Gaussian, etc.),
        rather than assuming it to be constant.
      - The strain vector per segment is composed of 6 components:
        [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z].
      - The joint and link strain contributions are treated separately, with their DOFs
        padded or truncated to `max_dof` for consistent computation.
      - Link cross-section geometry can vary along the length and supports circular,
        rectangular, and elliptical shapes.

    """

    active_tendon_routing_params: dict[str, Array]
    # B_xi_segments: Array
    active_d_s: Callable = eqx.field(static=True)
    active_dd_s_ds: Callable = eqx.field(static=True)

    def __init__(
        self,
        *args,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        active_tendon_routing_params: dict[str, Array],
        **kwargs: Any,
    ):
        """
        Initialize the TendonActuatedGVS class.

        Args:
            active_tendon_routing_basis (Optional[Dict[str, Callable]]):
                Dictionary with the tendon routing functions. If None, a linear routing is used.
                Expected keys and signatures:
                - "d_s": Callable[[Dict[str, Array], Array], Array]
                    Returns the 3D vector [d_x, d_y, d_z] giving the tendon position
                    with respect to the local cross-section at abscissa s. For typical routing,
                    d_x = 0 as the offset lies in the cross-sectional plane.
                - "dd_s_ds": Callable[[Dict[str, Array], Array], Array]
                    Returns the derivative over s of d_s.
            active_tendon_routing_params (Dict[str, Array]):
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
            *args,
            **kwargs,
        )

        # Set default tendon routing basis to linear routing if not provided
        if active_tendon_routing_basis is None:
            active_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_arc_length_derivative,
            }
        self._set_active_tendon_routing_basis(active_tendon_routing_basis)
        self._set_active_tendon_routing_params(active_tendon_routing_params)

    def _set_active_tendon_routing_params(
        self, active_tendon_routing_params: dict[str, Array]
    ):
        """
        This internal function stores as attributes of the class the parameters of
        the tendon routings specified by the user.

        Args:
            active_tendon_routing_params (Dict[str, Array]): parameters of the tendons
        """
        if not isinstance(active_tendon_routing_params, (dict)):
            raise TypeError(
                "The parameter 'active_tendon_routing_params' must be a dictionary of jax.Array."
            )
        self.num_actuators = len(list(active_tendon_routing_params.values())[0])
        for key, val in active_tendon_routing_params.items():
            if len(val) != self.num_actuators:
                raise ValueError(
                    f"The arrays in 'active_tendon_routing_params' must have the same length. Mismatch found in {key}."
                )
        for idx in active_tendon_routing_params["idx_seg_att"]:
            if idx >= self.num_segments:
                raise ValueError(
                    'The indexes of the segments of attachment (active_tendon_routing_params["idx_seg_att"]) must be strictly '
                    f"lower than the number of segments of the robot. Got {idx}; num_segments = {self.num_segments}."
                )
        # if self._check_tendon_routing_in_body(active_tendon_routing_params):
        #     raise UserWarning(f"Tendon(s) exit the robot body.")
        self.active_tendon_routing_params = active_tendon_routing_params

    def _set_active_tendon_routing_basis(
        self, active_tendon_routing_basis: dict[str, Callable]
    ):
        """
        This internal function stores as attributes of the class the basis functions of
        the tendon routings specified by the user.

        Args:
            active_tendon_routing_basis (Dict[str, Callable]): basis functions of the tendons
        """
        self.active_d_s = active_tendon_routing_basis["d_s"]
        self.active_dd_s_ds = active_tendon_routing_basis["dd_s_ds"]

    def update_active_tendon_routing_params(
        self, active_tendon_routing_params: dict[str, Array]
    ) -> "TendonActuatedGVS":
        """
        This function updates the parameters of the tendon routings of the object.

        Args:
            active_tendon_routing_params (Dict[str, Array]): parameters of the tendons

        Returns:
            updated self (TendonActuatedGVS): self object with updated parameters
        """
        updated_self = self

        # Recompute derived parameters
        active_tendon_routing_params = dict(active_tendon_routing_params)
        num_actuators = len(list(active_tendon_routing_params.values())[0])

        # update all fields
        updated_self = eqx.tree_at(
            lambda x: x.active_tendon_routing_params,
            updated_self,
            active_tendon_routing_params,
        )
        updated_self = eqx.tree_at(
            lambda x: x.num_actuators, updated_self, num_actuators
        )

        return updated_self

    @eqx.filter_jit
    def _local_actuation_basis_single(
        self,
        single_active_tendon_routing_params: dict[str, Array],
        q_i: Array,
        s: Array,
        i: Array,
        j: Array,
    ) -> Array:
        """
        Returns a (6,) vector: actuation basis contribution of a single tendon
        at segment i and abscissa index j (for abscissa value s).
        """
        attachment_segment_idx = single_active_tendon_routing_params[
            "idx_seg_att"
        ]  # ()
        cond = attachment_segment_idx >= i  # ()

        B_Xs_j = self.B_Xs[i, j]  # (6, max_dof)
        length_i = self.segment_lengths[i]
        if self.scale_rotational_basis_by_length:
            B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)

        xi_ref_j = self.xi_ref_Xs[i, j]
        xi_j = B_Xs_j @ q_i + xi_ref_j  # (6,)

        d_s = jnp.append(self.active_d_s(single_active_tendon_routing_params, s), 1.0)
        dd_s = jnp.append(
            self.active_dd_s_ds(single_active_tendon_routing_params, s), 1.0
        )

        term = (dd_s + lie.hat_SE3(xi_j) @ d_s)[:-1]
        norm = jnp.linalg.norm(term)
        t = term / norm

        return cond * jnp.hstack([lie.tilde_SE3(d_s[:-1]) @ t, t])

    @eqx.filter_jit
    def _local_actuation_basis(self, q_i: Array, s: Array, i: Array, j: Array) -> Array:
        """
        Vectorized wrapper: compute actuation basis Phi_a_s for all tendons at
        given (q_i, s, segment i, gauss index j). Returns (6, num_actuators).
        """
        # vmap over the first axis of active_tendon_routing_params, keep other args fixed
        Phi_a_s = vmap(
            self._local_actuation_basis_single,
            in_axes=(0, None, None, None, None),
            out_axes=1,
        )(self.active_tendon_routing_params, q_i, s, i, j)

        # print('Phi_a_s size=\n', Phi_a_s.shape) #debug
        return Phi_a_s

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        q_gathered = self._min_size_gathered(q)

        def A_segment_i(i: Array):
            """
            Compute the actuation matrix at the gaussian points of segment i of the robot.

            Args:
                i (Array): index of the segment ()

            Returns:
                A_i (Array): stack of actuation matrices of shape (num_gauss_points, num_active_strains, num_actuators).
            """

            length_i = self.segment_lengths[i]
            Xs_i = self.integration_points[i]  # (max_num_integration_points, 1, )
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            B_Xs_i = self.B_Xs[i]  # (max_num_integration_points, 6, max_dof)

            q_i = q_gathered[i, 1]
            A_joint_i = jnp.zeros((self.max_dof, self.num_actuators))

            def A_point_j(j: Array):
                """
                Compute the actuation matrix at the abscissa point corresponding to the gaussian point j.

                Args:
                    j (Array): index of the gaussian point ()

                Returns:
                    A_j (Array): local actuation matrix of shape (num_active_strains, num_actuators).
                """
                Xs_j = Xs_i[j]
                Ws_j = Ws_i[j]
                B_Xs_j = B_Xs_i[j]

                if self.scale_rotational_basis_by_length:
                    B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)

                Phi_a_j = self._local_actuation_basis(q_i, Xs_j * length_i, i, j)

                A_j = B_Xs_j.T @ Phi_a_j

                return Ws_j * A_j

            # Vectorize the actuation matrix computation for all gaussian points
            A_link_i = (
                jnp.sum(
                    vmap(A_point_j)(jnp.arange(1, self.max_num_integration_points - 1)),
                    axis=0,
                )
                * length_i
            )

            # # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_point_j(j) for j in range(self.max_num_gauss_points)], axis=0)
            # print('A_blocks_i =\n', A_blocks_i.shape)

            # return A_i # (max_dof, num_actuators)
            return jnp.stack([A_joint_i, A_link_i], axis=0)

        # Vectorize the actuation matrix computation for all segments
        A_blocks = vmap(A_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, num_active_strains, num_actuators)

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # A_blocks_tot = jnp.stack([A_segment_i(i) for i in range(self.num_segments)], axis=0)
        # print('A_blocks_tot =\n', A_blocks_tot.shape)

        A_blocks_flat = A_blocks.reshape(-1, self.max_dof, self.num_actuators)

        A_full = A_blocks_flat.reshape(
            -1, self.num_actuators
        )  # (num_segments*2*max_dof, num_actuators)

        A = self.active_dof_map.T @ A_full  # Map to active strains only

        # print('Actuation matrix A =\n', A) # debug
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
            single_active_tendon_routing_params: dict[str, Array], q: Array, s: Array
        ) -> Array:
            """
            Compute the forward kinematics of one tendon actuator at a point s along the robot.
            If s is greater than the length of the tendon, the function returns the last valid
            position of tendon (i.e., its attachement point).

            Args:
                single_active_tendon_routing_params (Dict[str, Array]): parameters of one tendon (6,)
                q (Array): generalized coordinates of shape (num_active_strains,)
                s (Array): point coordinate along the robot ()

            Returns:
                t_k_s (Array): cartesian position of the tendons at s, shape (3,)
            """
            lt = self.segment_end_positions[
                single_active_tendon_routing_params["idx_seg_att"] + 1
            ]  # ()
            s_val = jnp.clip(s, 0.0, lt)  # ()

            g_s = self.forward_kinematics(q, s_val)  # (4,4)
            t_k_s = g_s @ jnp.append(
                self.active_d_s(single_active_tendon_routing_params, s_val), 1.0
            )  # (4,)
            t_k_s = t_k_s[:-1]  # (3,)

            return t_k_s

        # Vectorize the forward kinematics computation for all tendons
        t_s = vmap(forward_kinematics_tendon_k, in_axes=(0, None, None))(
            self.active_tendon_routing_params, q, s
        )  # (n_actuators, 3)

        return t_s

    @eqx.filter_jit
    def tendon_length(self, q: Array) -> Array:
        """
        Compute the length of all tendons of the robot.
        This implementation is based on the formulation presented in:
        Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024). Input decoupling of lagrangian systems via coordinate transformation: General characterization and its application to soft robotics. IEEE transactions on robotics, 40, 2098-2110.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            l (Array): Lengths of all tendons of shape (num_actuators,).
        """
        q_gathered = self._min_size_gathered(q)

        def tendon_length_density_segment_i(i: Array):
            """
            Compute the tendon length density at all Gauss points of segment i.

            Args:
                i (Array): segment index.

            Returns:
                dl_ds_i (Array): tendon length density values at Gauss points,
                    shape (num_gauss_points, num_actuators).
            """
            length_i = self.segment_lengths[i]
            Xs_i = self.integration_points[i]  # (max_num_integration_points, 1, )
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            B_Xs_i = self.B_Xs[i]  # (max_num_integration_points, 6, max_dof)
            q_i = q_gathered[i, 1]

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
                Xs_j = Xs_i[j]
                Ws_j = Ws_i[j]
                B_Xs_j = B_Xs_i[j]

                if self.scale_rotational_basis_by_length:
                    B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)

                def tendon_length_density_tendon_k(
                    active_tendon_routing_params_k: dict[str, Array],
                ) -> Array:
                    """
                    Compute the tendon length density contribution of one tendon at the
                    current Gauss point.

                    Args:
                        active_tendon_routing_params_k (Dict[str, Array]): parameters of the tendon.

                    Returns:
                        dl_ds_k (Array): scalar tendon length density contribution.
                    """
                    # extract tendon routing evolution in the cross-sectional plane
                    dd_s = jnp.append(
                        self.active_dd_s_ds(
                            active_tendon_routing_params_k, Xs_j * length_i
                        ),
                        1.0,
                    )  # (4,)

                    # compute local actuation basis
                    Phi_a_k = self._local_actuation_basis_single(
                        active_tendon_routing_params_k, q_i, Xs_j * length_i, i, j
                    )  # (6,)

                    xi_ref_j = self.xi_ref_Xs[i, j]
                    xi_j = B_Xs_j @ q_i + xi_ref_j  # (6,)

                    # compute tendon length density contribution
                    dl_ds_k = jnp.dot(
                        Phi_a_k.T,
                        xi_j + jnp.concatenate([jnp.zeros((3,)), dd_s[:3]], axis=-1),
                    )  # ()

                    return dl_ds_k

                # Vectorize the tendon length density computation for all tendons
                dl_ds_j = vmap(tendon_length_density_tendon_k)(
                    self.active_tendon_routing_params
                )  # (num_actuators,)

                return Ws_j * dl_ds_j * length_i

            # Vectorize the tendon length density evaluation for all Gauss points
            dl_ds_i = vmap(tendon_length_density_point_j)(
                jnp.arange(self.max_num_integration_points - 1)
            )  # (num_gauss_points, num_actuators)

            return dl_ds_i

        # Vectorize the tendon length density evaluation for all segments
        dl_ds_blocks = vmap(tendon_length_density_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, num_actuators)

        l = jnp.sum(dl_ds_blocks, axis=(0, 1))  # Sum over the segments and Gauss points

        return l

    actuated_coordinates = tendon_length  # Alias for actuation space dynamics
