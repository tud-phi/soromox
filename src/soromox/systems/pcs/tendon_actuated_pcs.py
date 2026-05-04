__all__ = ["TendonActuatedPCS"]
from collections.abc import Callable
from typing import Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

import soromox.actuation.tendon_actuation as act
import soromox.utils.lie_algebra as lie
from soromox.systems.params import (
    BaseTendonRoutingParams,
    PassiveTendonParams,
)
from soromox.systems.pcs.params import TendonActuatedPCSParams
from soromox.systems.pcs.structures import PCSStructure
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
        num_segments: Number of segments (constant strain sections) along the robot.
        num_actuators: Number of actuators (control inputs) for the robot.
        n_p: Number of passive tendons in the robot.
        g0: Initial pose of the robot base as an SE(3) transformation matrix.
        g: Gravitational acceleration vector (embedded in a 6D vector).
            [0, 0, 0, g_x, g_y, g_z]
        L, r, E, G, rho, D: Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.
        active_tendon_routing: Batched routing params with one leading entry per active tendon.
        active_d_s: Function that returns the vector [d_x, d_y, d_z] of the active tendon
            position w.r.t. the central backbone within the cross-sectional plane at a given
            abscissa point s. The three entries represent the coordinate of the tendon
            position with respect to the local cross-sectional frame at s. For this reason,
            d_x is always equal to 0 (as the backbone is pointing in the local x-direction)
        active_dd_s_ds: Function that returns the vector of the derivative over s of active_d_s.
        passive_tendon_routing: Batched routing params with one leading entry per passive tendon.
        passive_d_s: As active_d_s for the passive tendons.
        passive_dd_s_ds: As active_dd_s_ds for the passive tendons.
        K_pt: Stiffness matrix of the passive tendons (n_p, n_p).
        D_pt: Damping matrix of the passive tendons (n_p, n_p).
        l_pt0: Vector of the initial displacement of the passive tendons (n_p,).

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
        Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete Cosserat
        Approach for Multisection Soft Manipulator Dynamics. IEEE Transactions on
        Robotics, 34(6), 1518-1533.

        Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020).
        A geometric variable-strain approach for static modeling of soft manipulators
        with tendon and fluidic actuation. IEEE Robotics and Automation Letters,
        5(3), 4006-4013.

        Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024).
        Input decoupling of Lagrangian systems via coordinate transformation: General
        characterization and its application to soft robotics. IEEE Transactions on
        Robotics, 40, 2098-2110.

        Milana, E., Della Santina, C., Gorissen, B., & Rothemund, P. (2025). Physical
        control: A new avenue to achieve intelligence in soft robotics. Science
        Robotics, 10(102), eadw7660.

    """

    n_p: int
    params: TendonActuatedPCSParams
    active_tendon_routing: BaseTendonRoutingParams
    active_d_s: Callable = eqx.field(static=True)
    active_dd_s_ds: Callable = eqx.field(static=True)
    passive_tendon_routing: BaseTendonRoutingParams
    passive_d_s: Callable = eqx.field(static=True)
    passive_dd_s_ds: Callable = eqx.field(static=True)
    K_pt: Array
    D_pt: Array
    l_pt0: Array

    def __init__(
        self,
        params: TendonActuatedPCSParams,
        structure: PCSStructure | None = None,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        passive_tendon_routing_basis: dict[str, Callable] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the tendon-actuated PCS class.

        Args:
            params: Typed dynamic parameters. ``params.active_tendon_routing`` and
                ``params.passive_tendon_routing`` are batched routing sets with one
                leading entry per tendon. ``params.passive_tendon`` stores per-passive
                tendon stiffness, damping, and rest-length offsets.
            structure: Static PCS layout. Changing the number of segments, active
                strains, or quadrature layout requires reconstruction.
            active_tendon_routing_basis: Routing functions for active tendons. The
                default is linear routing. Functions receive a single-tendon typed
                routing PyTree under ``jax.vmap``.
            passive_tendon_routing_basis: Routing functions for passive tendons.
            **kwargs: Additional keyword arguments for ``PCS``.
        """
        if not isinstance(params, TendonActuatedPCSParams):
            raise TypeError("params must be a TendonActuatedPCSParams instance.")
        super().__init__(params.body, structure=structure, **kwargs)
        self.params = params

        # Set default active tendon routing basis to linear routing if not provided
        if active_tendon_routing_basis is None:
            active_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_arc_length_derivative,
            }
        self.active_d_s, self.active_dd_s_ds = self._set_tendon_routing_basis(
            active_tendon_routing_basis
        )
        self.active_tendon_routing = self._set_active_tendon_routing(
            params.active_tendon_routing, self.active_d_s
        )

        # Set default passive tendon routing basis to linear routing if not provided
        if passive_tendon_routing_basis is None:
            passive_tendon_routing_basis = {
                "d_s": act.linear_routing,
                "dd_s_ds": act.linear_routing_arc_length_derivative,
            }
        self.passive_d_s, self.passive_dd_s_ds = self._set_tendon_routing_basis(
            passive_tendon_routing_basis
        )
        self.passive_tendon_routing = self._set_passive_tendon_routing(
            params.passive_tendon_routing, self.passive_d_s
        )

        # Set physical parameters of the passive tendons
        self._set_passive_tendon(params.passive_tendon)

    def _set_passive_tendon(self, passive_tendon: PassiveTendonParams) -> None:
        """
        This internal function stores as attributes of the class the physical
        parameters of the passive tendons specified by the user.

        Args:
            passive_tendon: Per-passive-tendon impedance parameters.
        """
        if not isinstance(passive_tendon, PassiveTendonParams):
            raise TypeError("passive_tendon must be a PassiveTendonParams instance.")
        if passive_tendon.num_tendons != self.n_p:
            raise ValueError(
                "passive_tendon length must match passive_tendon_routing length; "
                f"got {passive_tendon.num_tendons} and {self.n_p}."
            )
        self.K_pt = jnp.diag(jnp.asarray(passive_tendon.stiffness))
        self.D_pt = jnp.diag(jnp.asarray(passive_tendon.damping))
        self.l_pt0 = jnp.asarray(passive_tendon.rest_length_offset)

    def _set_active_tendon_routing(
        self, tendon_routing_params: BaseTendonRoutingParams, d_s: Callable
    ) -> BaseTendonRoutingParams:
        """
        This internal function stores as attributes of the class the parameters of
        the active tendon routings specified by the user.

        Args:
            tendon_routing_params: Batched routing params for all active tendons.
        """
        if not isinstance(tendon_routing_params, BaseTendonRoutingParams):
            raise TypeError(
                "active_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        tendon_routing_params.validate()
        self.num_actuators = tendon_routing_params.num_tendons
        tendon_routing_params.validate_attachment_segments(
            self.num_segments, "active_tendon_routing"
        )
        if self._check_tendon_routing_in_body(tendon_routing_params, d_s):
            raise UserWarning("Active tendon(s) exit the robot body.")
        return tendon_routing_params

    def _set_passive_tendon_routing(
        self, tendon_routing_params: BaseTendonRoutingParams, d_s: Callable
    ) -> BaseTendonRoutingParams:
        """
        This internal function stores as attributes of the class the parameters of
        the passive tendon routings specified by the user.

        Args:
            tendon_routing_params: Batched routing params for all passive tendons.
        """
        if not isinstance(tendon_routing_params, BaseTendonRoutingParams):
            raise TypeError(
                "passive_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        tendon_routing_params.validate()
        self.n_p = tendon_routing_params.num_tendons
        tendon_routing_params.validate_attachment_segments(
            self.num_segments, "passive_tendon_routing"
        )
        if self._check_tendon_routing_in_body(tendon_routing_params, d_s):
            raise UserWarning("Passive tendon(s) exit the robot body.")
        return tendon_routing_params

    def _set_tendon_routing_basis(self, tendon_routing_basis: dict[str, Callable]):
        """
        This internal function stores as attributes of the class the basis functions of
        the tendon routings specified by the user.

        Args:
            tendon_routing_basis: Basis functions for tendon routing.
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
            tendon_routing_params: Batched routing params with one leading entry per tendon.

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

    def with_params(self, params: TendonActuatedPCSParams) -> "TendonActuatedPCS":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, TendonActuatedPCSParams):
            raise TypeError("params must be a TendonActuatedPCSParams instance.")
        if type(params.active_tendon_routing) is not type(
            self.params.active_tendon_routing
        ):
            raise ValueError(
                "Changing active_tendon_routing type requires reconstruction."
            )
        if type(params.passive_tendon_routing) is not type(
            self.params.passive_tendon_routing
        ):
            raise ValueError(
                "Changing passive_tendon_routing type requires reconstruction."
            )
        if (
            params.active_tendon_routing.num_tendons
            != self.params.active_tendon_routing.num_tendons
        ):
            raise ValueError(
                "Changing the number of active tendons requires reconstruction."
            )
        if (
            params.passive_tendon_routing.num_tendons
            != self.params.passive_tendon_routing.num_tendons
        ):
            raise ValueError(
                "Changing the number of passive tendons requires reconstruction."
            )
        if (
            params.passive_tendon.num_tendons
            != params.passive_tendon_routing.num_tendons
        ):
            raise ValueError(
                "passive_tendon length must match passive_tendon_routing length."
            )
        params.active_tendon_routing.validate()
        params.passive_tendon_routing.validate()
        params.passive_tendon.validate()
        self.params.active_tendon_routing.assert_same_attachment_segments(
            params.active_tendon_routing, "active_tendon_routing"
        )
        self.params.passive_tendon_routing.assert_same_attachment_segments(
            params.passive_tendon_routing, "passive_tendon_routing"
        )

        updated_self = self._with_pcs_params(params.body, stored_params=params)
        return eqx.tree_at(
            lambda model: (
                model.active_tendon_routing,
                model.passive_tendon_routing,
                model.K_pt,
                model.D_pt,
                model.l_pt0,
            ),
            updated_self,
            (
                params.active_tendon_routing,
                params.passive_tendon_routing,
                jnp.diag(jnp.asarray(params.passive_tendon.stiffness)),
                jnp.diag(jnp.asarray(params.passive_tendon.damping)),
                jnp.asarray(params.passive_tendon.rest_length_offset),
            ),
        )

    def update_params(self, **updates: Array) -> "TendonActuatedPCS":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def _local_actuation_basis(
        self,
        i: Array,
        xi_i: Array,
        s: Array,
        tendon_routing_params_k: BaseTendonRoutingParams,
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
        attachment_segment_idx: Array | None = None,
    ) -> Array:
        """
        Compute the actuation matrix contribution of one tendon k at s contained in segment i.

        Args:
            i (Array): index of the segment ()
            xi_i (Array): strain at segment i (6,)
            tendon_routing_params_k: Single-tendon routing params.
            s (Array): abscissa points (num_gauss_points,)

        Returns:
            Phi_a_k (Array): local actuation matrix contribution of one tendon of shape (6,).
        """
        if attachment_segment_idx is None:
            attachment_segment_idx = tendon_routing_params_k.attachment_segment_index
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
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def jacobian_passive_tendon(self, q: Array):
        return self._actuation_matrix(
            q,
            self.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
        ).T

    @eqx.filter_jit
    def _actuation_matrix(
        self,
        q: Array,
        tendon_routing_params: BaseTendonRoutingParams,
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
        nt = tendon_routing_params.num_tendons

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
                    in_axes=(None, None, None, 0, None, None, 0),
                    out_axes=(-1),
                )(
                    i,
                    xi_i,
                    Xs_j,
                    tendon_routing_params,
                    d_s,
                    dd_s_ds,
                    tendon_routing_params.attachment_segment_index_array,
                )  # (6, nt)

                A_j = Phi_a_j

                return Ws_j * A_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[i],
                self.L_cum[i + 1],
            )

            # Vectorize the actuation matrix computation for all gaussian points
            A_i = vmap(A_point_j)(
                jnp.arange(self.num_integration_points)
            )  # (num_gauss_points, 6, nt)

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # A_blocks_i = jnp.stack([A_point_j(j) for j in range(self.num_integration_points)], axis=0)
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
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        return self._tendon_length(
            q,
            self.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
        )

    tendon_length = active_tendon_length  # Alias for compatibility
    actuated_coordinates = active_tendon_length  # Alias for actuation space dynamics

    @eqx.filter_jit
    def _tendon_length(
        self,
        q: Array,
        tendon_routing_params: BaseTendonRoutingParams,
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
                    tendon_routing_params_k: BaseTendonRoutingParams,
                    attachment_segment_idx: Array,
                ) -> Array:
                    """
                    Compute the tendon length density contribution of one tendon at the
                    current Gauss point.

                    Args:
                        tendon_routing_params_k: Single-tendon routing params.

                    Returns:
                        dl_ds_k (Array): scalar tendon length density contribution.
                    """
                    # extract tendon routing evolution in the cross-sectional plane
                    dd_s = jnp.append(
                        dd_s_ds(tendon_routing_params_k, Xs_j), 1.0
                    )  # (4,)

                    # compute local actuation basis
                    Phi_a_k = self._local_actuation_basis(
                        i,
                        xi_i,
                        Xs_j,
                        tendon_routing_params_k,
                        d_s,
                        dd_s_ds,
                        attachment_segment_idx,
                    )  # (6,)

                    # compute tendon length density contribution
                    dl_ds_k = jnp.dot(
                        Phi_a_k.T,
                        xi_i + jnp.concatenate([jnp.zeros((3,)), dd_s[:3]], axis=-1),
                    )  # ()

                    return dl_ds_k

                # Vectorize the tendon length density computation for all tendons
                dl_ds_j = vmap(tendon_length_density_tendon_k)(
                    tendon_routing_params,
                    tendon_routing_params.attachment_segment_index_array,
                )  # (num_actuators,)

                return Ws_j * dl_ds_j

            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[i],
                self.L_cum[i + 1],
            )

            # Vectorize the tendon length density evaluation for all Gauss points
            dl_ds_i = vmap(tendon_length_density_point_j)(
                jnp.arange(self.num_integration_points)
            )  # (num_gauss_points, num_actuators)

            return dl_ds_i

        # Vectorize the tendon length density evaluation for all segments
        dl_ds_blocks = vmap(tendon_length_density_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, num_gauss_points, num_actuators)

        tendon_lengths = jnp.sum(
            dl_ds_blocks, axis=(0, 1)
        )  # Sum over the segments and Gauss points

        return tendon_lengths

    @eqx.filter_jit
    def forward_kinematics_active_tendons(self, q: Array, s: Array) -> Array:
        return self._forward_kinematics_tendons(
            q, s, self.active_tendon_routing, self.active_d_s
        )

    @eqx.filter_jit
    def forward_kinematics_passive_tendons(self, q: Array, s: Array) -> Array:
        return self._forward_kinematics_tendons(
            q, s, self.passive_tendon_routing, self.passive_d_s
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
        self,
        q: Array,
        s: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s: Callable,
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
            single_tendon_routing_params: BaseTendonRoutingParams,
            attachment_segment_idx: Array,
            q: Array,
            s: Array,
        ) -> Array:
            """
            Compute the forward kinematics of one tendon actuator at a point s along the robot.
            If s is greater than the length of the tendon, the function returns the last valid
            position of tendon (i.e., its attachement point).

            Args:
                tendon_routing_params: Batched routing params with one leading entry per tendon.
                q (Array): generalized coordinates of shape (num_active_strains,)
                s (Array): point coordinate along the robot ()

            Returns:
                t_k_s (Array): cartesian position of the tendons at s, shape (3,)
            """
            lt = self.L_cum[attachment_segment_idx + 1]  # ()
            s_val = jnp.clip(s, 0.0, lt)  # ()

            g_s = self.forward_kinematics(q, s_val)  # (4,4)
            t_k_s = g_s @ jnp.append(
                d_s(single_tendon_routing_params, s_val), 1.0
            )  # (4,)
            t_k_s = t_k_s[:-1]  # (3,)

            return t_k_s

        # Vectorize the forward kinematics computation for all tendons
        t_s = vmap(forward_kinematics_tendon_k, in_axes=(0, 0, None, None))(
            tendon_routing_params,
            tendon_routing_params.attachment_segment_index_array,
            q,
            s,
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
    def _elastic_energy(self, q: Array) -> Array:
        """
        Compute the elastic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_K_tot (float): Total elastic energy of the robot.
        """
        # Elastic energy of the body
        U_K = super()._elastic_energy(q)

        # Elastic energy of the passive tendons
        l_pt = self.passive_tendon_length(q)
        U_K_pt = 0.5 * (l_pt - self.l_pt0).T @ self.K_pt @ (l_pt - self.l_pt0)

        U_K_tot = U_K + U_K_pt
        return U_K_tot
