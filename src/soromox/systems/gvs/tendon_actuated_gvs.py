__all__ = ["TendonActuatedGVS"]
from collections.abc import Callable
from typing import Any

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

import soromox.actuation.tendon_actuation as act
from soromox.systems.gvs.params import TendonActuatedGVSParams
from soromox.systems.gvs.structures import GVSStructure
from soromox.systems.params import (
    BaseTendonRoutingParams,
    LinearTendonRoutingParams,
    PassiveTendonParams,
)
from soromox.utils.lie_algebra import se3, so3

from .core import GVS
from .specs import GVSSegment


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

    params: TendonActuatedGVSParams
    active_tendon_routing: BaseTendonRoutingParams
    passive_tendon_routing: BaseTendonRoutingParams
    n_p: int
    # B_xi_segments: Array
    active_d_s: Callable = eqx.field(static=True)
    active_dd_s_ds: Callable = eqx.field(static=True)
    passive_d_s: Callable = eqx.field(static=True)
    passive_dd_s_ds: Callable = eqx.field(static=True)
    K_pt: Array
    D_pt: Array
    l_pt0: Array

    def __init__(
        self,
        params: TendonActuatedGVSParams,
        structure: GVSStructure,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        passive_tendon_routing_basis: dict[str, Callable] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the TendonActuatedGVS class.

        Args:
            params: Typed dynamic parameters. ``params.active_tendon_routing`` is a
                batched routing set with one leading entry per active tendon, so
                each tendon can have distinct routing values and attachment segment.
            structure: Static GVS segment structure and padded layout.
            active_tendon_routing_basis: Routing functions for active tendons. The
                default is linear routing. Functions receive a single-tendon typed
                routing PyTree under ``jax.vmap``.
            passive_tendon_routing_basis: Routing functions for passive tendons.
            **kwargs: Additional keyword arguments for ``GVS``.
        """

        if not isinstance(params, TendonActuatedGVSParams):
            raise TypeError("params must be a TendonActuatedGVSParams instance.")
        params.validate_against_structure(structure)
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
            params.active_tendon_routing
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
            params.passive_tendon_routing
        )
        self._set_passive_tendon(params.passive_tendon)

    def _set_active_tendon_routing(
        self, active_tendon_routing: BaseTendonRoutingParams
    ) -> BaseTendonRoutingParams:
        """
        This internal function stores as attributes of the class the parameters of
        the tendon routings specified by the user.

        Args:
            active_tendon_routing: Batched routing params for all active tendons.
        """
        if not isinstance(active_tendon_routing, BaseTendonRoutingParams):
            raise TypeError(
                "active_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        active_tendon_routing.validate()
        self.num_actuators = active_tendon_routing.num_tendons
        active_tendon_routing.validate_attachment_segments(
            self.num_segments, "active_tendon_routing"
        )
        # if self._check_tendon_routing_in_body(active_tendon_routing):
        #     raise UserWarning(f"Tendon(s) exit the robot body.")
        return active_tendon_routing

    def _set_passive_tendon_routing(
        self, passive_tendon_routing: BaseTendonRoutingParams
    ) -> BaseTendonRoutingParams:
        """
        Store the parameters for all passive tendon routings.

        Args:
            passive_tendon_routing: Batched routing params for all passive tendons.
        """
        if not isinstance(passive_tendon_routing, BaseTendonRoutingParams):
            raise TypeError(
                "passive_tendon_routing must be a BaseTendonRoutingParams instance."
            )
        passive_tendon_routing.validate()
        self.n_p = passive_tendon_routing.num_tendons
        passive_tendon_routing.validate_attachment_segments(
            self.num_segments, "passive_tendon_routing"
        )
        return passive_tendon_routing

    def _set_passive_tendon(self, passive_tendon: PassiveTendonParams) -> None:
        """Store per-passive-tendon stiffness, damping, and rest-length offsets."""
        if not isinstance(passive_tendon, PassiveTendonParams):
            raise TypeError("passive_tendon must be a PassiveTendonParams instance.")
        passive_tendon.validate()
        if passive_tendon.num_tendons != self.n_p:
            raise ValueError(
                "passive_tendon length must match passive_tendon_routing length; "
                f"got {passive_tendon.num_tendons} and {self.n_p}."
            )
        self.K_pt = jnp.diag(jnp.asarray(passive_tendon.stiffness))
        self.D_pt = jnp.diag(jnp.asarray(passive_tendon.damping))
        self.l_pt0 = jnp.asarray(passive_tendon.rest_length_offset)

    @classmethod
    def from_segments(
        cls,
        segments: list[GVSSegment] | tuple[GVSSegment, ...],
        *,
        gravity: Array,
        active_tendon_routing: BaseTendonRoutingParams,
        passive_tendon_routing: BaseTendonRoutingParams | None = None,
        passive_tendon: PassiveTendonParams | None = None,
        base_pose: Array | None = None,
        max_dof: int | None = None,
        max_num_gauss_points: int | None = None,
        scale_rotational_basis_by_length: bool = False,
        active_tendon_routing_basis: dict[str, Callable] | None = None,
        passive_tendon_routing_basis: dict[str, Callable] | None = None,
        **kwargs: Any,
    ) -> "TendonActuatedGVS":
        """Construct a tendon-actuated GVS model from segment and tendon specs."""
        body, structure = GVS.params_from_segments(
            segments,
            gravity=gravity,
            base_pose=base_pose,
            max_dof=max_dof,
            max_num_gauss_points=max_num_gauss_points,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        )
        if passive_tendon_routing is None:
            passive_tendon_routing = LinearTendonRoutingParams.empty()
        if passive_tendon is None:
            passive_tendon = PassiveTendonParams.empty()
        params = TendonActuatedGVSParams(
            body=body,
            active_tendon_routing=active_tendon_routing,
            passive_tendon_routing=passive_tendon_routing,
            passive_tendon=passive_tendon,
        )
        return cls(
            params=params,
            structure=structure,
            active_tendon_routing_basis=active_tendon_routing_basis,
            passive_tendon_routing_basis=passive_tendon_routing_basis,
            **kwargs,
        )

    def _set_tendon_routing_basis(
        self, tendon_routing_basis: dict[str, Callable]
    ) -> tuple[Callable, Callable]:
        """
        This internal function stores as attributes of the class the basis functions of
        the tendon routings specified by the user.

        Args:
            tendon_routing_basis: Basis functions for tendon routing.
        """
        return tendon_routing_basis["d_s"], tendon_routing_basis["dd_s_ds"]

    def with_params(self, params: TendonActuatedGVSParams) -> "TendonActuatedGVS":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, TendonActuatedGVSParams):
            raise TypeError("params must be a TendonActuatedGVSParams instance.")
        params.validate_against_structure(self.structure)
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
        updated_self = GVS.with_params(self, params.body)
        return eqx.tree_at(
            lambda model: (
                model.params,
                model.active_tendon_routing,
                model.passive_tendon_routing,
                model.K_pt,
                model.D_pt,
                model.l_pt0,
            ),
            updated_self,
            (
                params,
                params.active_tendon_routing,
                params.passive_tendon_routing,
                jnp.diag(jnp.asarray(params.passive_tendon.stiffness)),
                jnp.diag(jnp.asarray(params.passive_tendon.damping)),
                jnp.asarray(params.passive_tendon.rest_length_offset),
            ),
        )

    def update_params(self, **updates: Array) -> "TendonActuatedGVS":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    @eqx.filter_jit
    def _local_tendon_basis_single(
        self,
        single_tendon_routing_params: BaseTendonRoutingParams,
        attachment_segment_idx: Array,
        q_i: Array,
        s: Array,
        i: Array,
        j: Array,
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
    ) -> Array:
        """
        Returns a (6,) vector: actuation basis contribution of a single tendon
        at segment i and abscissa index j (for abscissa value s).
        """
        cond = attachment_segment_idx >= i  # ()

        B_Xs_j = self.B_Xs[i, j]  # (6, max_dof)
        length_i = self.segment_lengths[i]
        if self.scale_rotational_basis_by_length:
            B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)

        xi_ref_j = self.xi_ref_Xs[i, j]
        xi_j = B_Xs_j @ q_i + xi_ref_j  # (6,)

        d_s = jnp.append(d_s_fn(single_tendon_routing_params, s), 1.0)
        dd_s = jnp.append(dd_s_ds_fn(single_tendon_routing_params, s), 1.0)

        term = (dd_s + se3.hat(xi_j) @ d_s)[:-1]
        norm = jnp.linalg.norm(term)
        t = term / norm

        return cond * jnp.hstack([so3.skew(d_s[:-1]) @ t, t])

    @eqx.filter_jit
    def _local_actuation_basis_single(
        self,
        single_active_tendon_routing: BaseTendonRoutingParams,
        attachment_segment_idx: Array,
        q_i: Array,
        s: Array,
        i: Array,
        j: Array,
    ) -> Array:
        """Return the active actuation basis contribution of one tendon."""
        return self._local_tendon_basis_single(
            single_active_tendon_routing,
            attachment_segment_idx,
            q_i,
            s,
            i,
            j,
            self.active_d_s,
            self.active_dd_s_ds,
        )

    @eqx.filter_jit
    def _local_tendon_basis(
        self,
        q_i: Array,
        s: Array,
        i: Array,
        j: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
    ) -> Array:
        """
        Vectorized wrapper: compute actuation basis Phi_a_s for all tendons at
        given (q_i, s, segment i, gauss index j). Returns (6, num_tendons).
        """
        attachment_segment_indices = (
            tendon_routing_params.attachment_segment_index_array
        )
        Phi_a_s = vmap(
            self._local_tendon_basis_single,
            in_axes=(0, 0, None, None, None, None, None, None),
            out_axes=1,
        )(
            tendon_routing_params,
            attachment_segment_indices,
            q_i,
            s,
            i,
            j,
            d_s_fn,
            dd_s_ds_fn,
        )

        return Phi_a_s

    @eqx.filter_jit
    def _local_actuation_basis(self, q_i: Array, s: Array, i: Array, j: Array) -> Array:
        """Active-tendon wrapper retained for internal call sites."""
        return self._local_tendon_basis(
            q_i,
            s,
            i,
            j,
            self.active_tendon_routing,
            self.active_d_s,
            self.active_dd_s_ds,
        )

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        return self._actuation_matrix(
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def jacobian_passive_tendon(self, q: Array) -> Array:
        """Return the passive tendon Jacobian with shape (num_passive_tendons, num_dofs)."""
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
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
    ) -> Array:
        """Compute the actuation/Jacobian matrix for a batched tendon family."""
        q_gathered = self._min_size_gathered(q)
        num_tendons = tendon_routing_params.num_tendons

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
            A_joint_i = jnp.zeros((self.max_dof, num_tendons))

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

                Phi_a_j = self._local_tendon_basis(
                    q_i,
                    Xs_j * length_i,
                    i,
                    j,
                    tendon_routing_params,
                    d_s_fn,
                    dd_s_ds_fn,
                )

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

        A_blocks_flat = A_blocks.reshape(-1, self.max_dof, num_tendons)

        A_full = A_blocks_flat.reshape(
            -1, num_tendons
        )  # (num_segments*2*max_dof, num_tendons)

        A = self.active_dof_map.T @ A_full  # Map to active strains only

        return A

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
        Compute forward kinematics for all active and passive tendons.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,)
            s (Array): point coordinate along the robot ()

        Returns:
            t_s (Array): cartesian tendon positions at s.
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
        d_s_fn: Callable,
    ) -> Array:
        """Compute forward kinematics for one batched tendon family."""

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
                single_tendon_routing_params: Single-tendon routing params.
                q (Array): generalized coordinates of shape (num_active_strains,)
                s (Array): point coordinate along the robot ()

            Returns:
                t_k_s (Array): cartesian position of the tendons at s, shape (3,)
            """
            lt = self.segment_end_positions[attachment_segment_idx + 1]  # ()
            s_val = jnp.clip(s, 0.0, lt)  # ()

            g_s = self.forward_kinematics(q, s_val)  # (4,4)
            t_k_s = g_s @ jnp.append(d_s_fn(single_tendon_routing_params, s_val), 1.0)
            t_k_s = t_k_s[:-1]  # (3,)

            return t_k_s

        t_s = vmap(forward_kinematics_tendon_k, in_axes=(0, 0, None, None))(
            tendon_routing_params,
            tendon_routing_params.attachment_segment_index_array,
            q,
            s,
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
        return self._tendon_length(
            q, self.active_tendon_routing, self.active_d_s, self.active_dd_s_ds
        )

    @eqx.filter_jit
    def passive_tendon_length(self, q: Array) -> Array:
        """Return lengths of all passive tendons."""
        return self._tendon_length(
            q,
            self.passive_tendon_routing,
            self.passive_d_s,
            self.passive_dd_s_ds,
        )

    @eqx.filter_jit
    def _tendon_length(
        self,
        q: Array,
        tendon_routing_params: BaseTendonRoutingParams,
        d_s_fn: Callable,
        dd_s_ds_fn: Callable,
    ) -> Array:
        """Compute the length of all tendons in one batched tendon family."""
        q_gathered = self._min_size_gathered(q)
        num_tendons = tendon_routing_params.num_tendons

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
                        dd_s_ds_fn(tendon_routing_params_k, Xs_j * length_i),
                        1.0,
                    )  # (4,)

                    # compute local actuation basis
                    Phi_a_k = self._local_tendon_basis_single(
                        tendon_routing_params_k,
                        attachment_segment_idx,
                        q_i,
                        Xs_j * length_i,
                        i,
                        j,
                        d_s_fn,
                        dd_s_ds_fn,
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
                    tendon_routing_params,
                    tendon_routing_params.attachment_segment_index_array,
                )  # (num_tendons,)

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

        tendon_lengths = jnp.sum(dl_ds_blocks, axis=(0, 1)).reshape((num_tendons,))

        return tendon_lengths

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """Compute body elastic force plus passive tendon spring force."""
        tau_el = super().elastic_force(q)
        J_pt = self.jacobian_passive_tendon(q)
        l_pt = self.passive_tendon_length(q)
        tau_el_pt = J_pt.T @ self.K_pt @ (l_pt - self.l_pt0)
        return tau_el + tau_el_pt

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """Compute body damping plus passive tendon damping."""
        D = super().damping_matrix(q)
        J_pt = self.jacobian_passive_tendon(q)
        D_pt = J_pt.T @ self.D_pt @ J_pt
        return D + D_pt

    @eqx.filter_jit
    def _elastic_energy(self, q: Array) -> Array:
        """Compute body elastic energy plus passive tendon spring energy."""
        U_K = super()._elastic_energy(q)
        l_pt = self.passive_tendon_length(q)
        U_K_pt = 0.5 * (l_pt - self.l_pt0).T @ self.K_pt @ (l_pt - self.l_pt0)
        return U_K + U_K_pt

    actuated_coordinates = tendon_length  # Alias for actuation space dynamics
