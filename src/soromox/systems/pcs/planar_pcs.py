__all__ = ["PlanarPCS"]


from typing import Any

import equinox as eqx
from jax import Array, lax, vmap
from jax import numpy as jnp

from soromox.systems.pcs.params import PlanarPCSParams
from soromox.systems.pcs.structures import PlanarPCSStructure
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.basic import (
    compute_strain_basis,
)
from soromox.utils.geometry import poses
from soromox.utils.integration import (
    gauss_quadrature,
    scale_gaussian_quadrature,
    scale_interior_gaussian_quadrature,
)
from soromox.utils.lie_algebra import constant_strain, se2


class PlanarPCS(SoftRobot):
    """
    Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 2D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
        num_segments: Number of segments (constant strain sections) along the robot.
        num_actuators: Number of actuators (control inputs) for the robot.
        base_pose: Initial planar pose ``[theta, x, y]`` with ``theta`` in radians.
        g: Gravitational acceleration vector (embedded in a 3D vector).
            [0, g_x, g_y]
        L, r, E, G, rho, D: Physical properties of each segment (length, radius, elastic/shear modulus, etc.).
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (3 * num_segments, num_active_strains).
        scale_rotational_basis_by_length: If True, rotational strain-basis rows
            are divided by their segment length.
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.

    Notes:
    -----
    - The strain vector is composed of 3 components per segment:
      [kappa_z, sigma_x, sigma_y].
      By default, the rod is assumed to be straight and aligned with the x-axis,
        so the reference strain is set to [0, 1, 0].
        Thus:   - kappa_z corresponds to bending around the z-axis,
                - sigma_x corresponds to axial strain along the x-axis,
                - sigma_y corresponds to shear along the y-axis.

    References:
        Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete Cosserat
        Approach for Multisection Soft Manipulator Dynamics. IEEE Transactions on
        Robotics, 34(6), 1518-1533.

    """

    params: PlanarPCSParams

    # Robot parameters
    g: Array  # Gravitational acceleration vector

    L: Array  # Length of the segments
    L_cum: Array  # Cumulative length of the segments
    r: Array  # Radius of the segments
    rho: Array
    E: Array  # Young's modulus of the segments
    G: Array  # Shear modulus of the segments
    D: Array  # Damping coefficient of the segments

    num_segments: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)
    num_integration_points: int = eqx.field(static=True)
    num_strains: int = eqx.field(static=True)  # Number of strains (3 * num_segments)
    scale_rotational_basis_by_length: bool = eqx.field(static=True)

    xi_ref: Array  # Reference configuration strain
    B_xi_unscaled: Array  # Unscaled strain basis matrix
    B_xi: Array  # Strain basis matrix
    num_active_strains: Array  # Number of selected strains

    integration_points: Array
    integration_weights: Array
    M_segments: Array  # Cached per-segment mass matrices
    K_full: Array  # Cached full stiffness matrix
    K: Array  # Cached active-coordinate stiffness matrix
    D_full: Array  # Cached full damping matrix
    D_active: Array  # Cached active-coordinate damping matrix

    def __init__(
        self,
        params: PlanarPCSParams,
        structure: PlanarPCSStructure | None = None,
        **kwargs: Any,
    ):
        """Initialize the PlanarPCS class from typed dynamic parameters."""
        if not isinstance(params, PlanarPCSParams):
            raise TypeError("params must be a PlanarPCSParams instance.")
        params.validate()
        super().__init__(base_pose=params.base_pose, **kwargs)
        if structure is None:
            structure = PlanarPCSStructure()
        self.params = params
        self.scale_rotational_basis_by_length = bool(
            structure.scale_rotational_basis_by_length
        )

        # Number of segments
        num_segments = int(params.length.shape[0])
        if num_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {num_segments}")
        self.num_segments = num_segments

        num_strains = 3 * num_segments
        self.num_strains = num_strains

        # ================================================================
        # Robot parameters
        self._set_params(params)

        # ================================================================
        # Integration grid
        num_gauss_points = structure.num_gauss_points
        if not isinstance(num_gauss_points, int):
            raise TypeError(
                f"num_gauss_points must be an integer, got {type(num_gauss_points).__name__}"
            )
        if num_gauss_points < 1:
            raise ValueError(
                f"num_gauss_points must be at least 1, got {num_gauss_points}"
            )
        integration_points, integration_weights, num_integration_points = (
            gauss_quadrature(num_gauss_points, a=jnp.array(0.0), b=jnp.array(1.0))
        )
        self.integration_points = integration_points
        self.integration_weights = integration_weights
        self.num_gauss_points = num_gauss_points
        self.num_integration_points = num_integration_points

        # ================================================================
        # Strain basis matrix
        strain_selector = structure.strain_selector
        if strain_selector is None:
            strain_selector = jnp.ones(self.num_strains, dtype=bool)
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
            if strain_selector.size != self.num_strains:
                raise ValueError(
                    f"strain_selector must have {self.num_strains} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(self.num_strains)
        self.B_xi_unscaled = compute_strain_basis(strain_selector)
        self.B_xi = self._scaled_strain_basis(self.B_xi_unscaled)

        self.num_active_strains = jnp.sum(strain_selector)
        self.num_dofs = int(self.num_active_strains.item())

        reference_strain = jnp.asarray(params.reference_strain, dtype=jnp.float64)
        if reference_strain.size != self.num_strains:
            raise ValueError(
                "reference_strain must have "
                f"{self.num_strains} elements, got {reference_strain.size}."
            )
        self.xi_ref = reference_strain.reshape(self.num_strains)

        # Number of actuators
        self.num_actuators = int(self.num_active_strains.item())

        self.precompute()

    @property
    def is_planar(self) -> bool:
        """Planar PCS is a 2D model."""
        return True

    @property
    def segment_length(self) -> Array:
        """Per-segment backbone lengths."""
        return jnp.asarray(self.L)

    def _strain_basis_scaling_vector(self) -> Array:
        """Return per-strain coordinate scaling for the constant strain basis."""
        return jnp.stack(
            [1.0 / self.L, jnp.ones_like(self.L), jnp.ones_like(self.L)], axis=1
        ).reshape(self.num_strains)

    def _scaled_strain_basis(self, B_xi: Array) -> Array:
        """Apply optional length normalization to rotational strain coordinates."""
        if self.scale_rotational_basis_by_length:
            return self._strain_basis_scaling_vector()[:, None] * B_xi
        return B_xi

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Circular cross-section with segment radius."""
        segment_idx, _ = self.classify_segment(s)
        radius = jnp.asarray(self.r)[segment_idx]
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([radius])

    def _set_params(self, params: PlanarPCSParams) -> None:
        """Set cached runtime arrays from typed parameters."""
        self.base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)

        # Gravitational acceleration vector
        g = params.gravity
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 2:
            raise ValueError(f"gravity must be a vector of shape (2,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(1), g]
        )  # Add a zero for the orientation angle

        # Lengths of the segments
        L = params.length
        L = jnp.asarray(L, dtype=jnp.float64)
        if L.shape != (self.num_segments,):
            raise ValueError(
                f"length must have shape ({self.num_segments},), got {L.shape}"
            )
        self.L = L

        L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        self.L_cum = L_cum

        # Radius of the segments
        r = params.radius
        r = jnp.asarray(r, dtype=jnp.float64)
        if r.shape != (self.num_segments,):
            raise ValueError(
                f"radius must have shape ({self.num_segments},), got {r.shape}"
            )
        self.r = r

        # Densities of the segments
        rho = params.density
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if rho.shape != (self.num_segments,):
            raise ValueError(
                f"density must have shape ({self.num_segments},), got {rho.shape}"
            )
        self.rho = rho

        # Elastic modulus of the segments
        E = params.young_modulus
        E = jnp.asarray(E, dtype=jnp.float64)
        if E.shape != (self.num_segments,):
            raise ValueError(
                f"young_modulus must have shape ({self.num_segments},), got {E.shape}"
            )
        self.E = E

        # Shear modulus of the segments
        G = params.shear_modulus
        G = jnp.asarray(G, dtype=jnp.float64)
        if G.shape != (self.num_segments,):
            raise ValueError(
                f"shear_modulus must have shape ({self.num_segments},), got {G.shape}"
            )
        self.G = G

        # Damping matrix of the robot
        D = params.damping_matrix
        D = jnp.asarray(D, dtype=jnp.float64)
        expected_D_shape = (self.num_strains, self.num_strains)
        if D.shape != expected_D_shape:
            raise ValueError(
                f"damping_matrix must have shape {expected_D_shape}, got {D.shape}"
            )
        self.D = D

    def _current_body_params(self) -> PlanarPCSParams:
        """Return the planar PCS body params, including typed actuated wrappers."""
        if isinstance(self.params, PlanarPCSParams):
            return self.params
        body = getattr(self.params, "body", None)
        if isinstance(body, PlanarPCSParams):
            return body
        raise TypeError("model params do not contain PlanarPCSParams body fields.")

    def _with_planar_pcs_params(
        self, params: PlanarPCSParams, stored_params: Any | None = None
    ) -> "PlanarPCS":
        """Return a copy with planar PCS body caches refreshed."""
        current_params = self._current_body_params()
        if not isinstance(params, PlanarPCSParams):
            raise TypeError("params must be a PlanarPCSParams instance.")
        params.validate()
        if params.length.shape != current_params.length.shape:
            raise ValueError(
                "length shape changes the model structure; construct a new PlanarPCS."
            )
        if params.reference_strain.shape != current_params.reference_strain.shape:
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PlanarPCS."
            )

        base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)
        gravity = jnp.asarray(params.gravity, dtype=jnp.float64)
        segment_lengths = jnp.asarray(params.length, dtype=jnp.float64)
        radius = jnp.asarray(params.radius, dtype=jnp.float64)
        density = jnp.asarray(params.density, dtype=jnp.float64)
        young_modulus = jnp.asarray(params.young_modulus, dtype=jnp.float64)
        shear_modulus = jnp.asarray(params.shear_modulus, dtype=jnp.float64)
        damping_matrix = jnp.asarray(params.damping_matrix, dtype=jnp.float64)
        reference_strain = jnp.asarray(params.reference_strain, dtype=jnp.float64)

        expected_D_shape = (self.num_strains, self.num_strains)
        if damping_matrix.shape != expected_D_shape:
            raise ValueError(
                f"damping_matrix must have shape {expected_D_shape}, got {damping_matrix.shape}"
            )

        updated_self = eqx.tree_at(
            lambda m: (
                m.params,
                m.base_pose,
                m.g,
                m.L,
                m.L_cum,
                m.r,
                m.rho,
                m.E,
                m.G,
                m.D,
                m.xi_ref,
            ),
            self,
            (
                params if stored_params is None else stored_params,
                base_pose,
                jnp.concatenate([jnp.zeros(1, dtype=gravity.dtype), gravity]),
                segment_lengths,
                jnp.cumsum(
                    jnp.concatenate(
                        [jnp.zeros(1, dtype=segment_lengths.dtype), segment_lengths]
                    )
                ),
                radius,
                density,
                young_modulus,
                shear_modulus,
                damping_matrix,
                reference_strain.reshape(self.num_strains),
            ),
        )
        return updated_self._with_refreshed_precomputed_matrices()

    def with_params(self, params: PlanarPCSParams) -> "PlanarPCS":
        """Return an updated copy with a full typed parameter object."""
        return self._with_planar_pcs_params(params)

    def update_params(self, **updates: Array) -> "PlanarPCS":
        """Return an updated copy with selected typed parameter fields replaced."""
        if (
            "length" in updates
            and jnp.asarray(updates["length"]).shape != self.params.length.shape
        ):
            raise ValueError(
                "length shape changes the model structure; construct a new PlanarPCS."
            )
        if (
            "reference_strain" in updates
            and jnp.asarray(updates["reference_strain"]).shape
            != self.params.reference_strain.shape
        ):
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PlanarPCS."
            )
        return self.with_params(self.params.replace(**updates))

    def _precomputed_matrices(self) -> tuple[Array, Array, Array, Array, Array]:
        """Compute state-independent matrices cached by the model."""
        M_segments = vmap(self._compute_local_mass_matrix)(
            jnp.arange(self.num_segments)
        )
        K_full = self._compute_stiffness_full_matrix()
        K = self.B_xi.T @ K_full @ self.B_xi
        D_full = self.D
        D_active = self.B_xi.T @ D_full @ self.B_xi
        return M_segments, K_full, K, D_full, D_active

    def precompute(self) -> None:
        """Refresh state-independent matrices cached by the model."""
        object.__setattr__(self, "B_xi", self._scaled_strain_basis(self.B_xi_unscaled))
        (
            M_segments,
            K_full,
            K,
            D_full,
            D_active,
        ) = self._precomputed_matrices()
        object.__setattr__(self, "M_segments", M_segments)
        object.__setattr__(self, "K_full", K_full)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "D_full", D_full)
        object.__setattr__(self, "D_active", D_active)

    def _with_refreshed_precomputed_matrices(self) -> "PlanarPCS":
        """Return a copy with cached state-independent matrices refreshed."""
        B_xi = self._scaled_strain_basis(self.B_xi_unscaled)
        updated_self = eqx.tree_at(lambda m: m.B_xi, self, B_xi)
        return eqx.tree_at(
            lambda m: (m.M_segments, m.K_full, m.K, m.D_full, m.D_active),
            updated_self,
            updated_self._precomputed_matrices(),
        )

    @eqx.filter_jit
    def classify_segment(self, s: Array) -> tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(
            jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1
        ).astype(jnp.int32)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local

    @eqx.filter_jit
    def strain(self, q: Array) -> Array:
        """
        Compute the strain vector from the generalized coordinates.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            xi (Array): strain vector of shape (num_active_strains,)
        """
        xi = self.B_xi @ q + self.xi_ref

        return xi

    @eqx.filter_jit
    def chi(self, xi: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the robot.

        Args:
            xi (Array): strain vector of shape (3*num_segments,) where each row corresponds to a segment
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi_s (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = xi.reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)

        chi_0 = self._base_planar_pose(xi.dtype)

        # Iteration function
        def chi_i(chi_prev: Array, i: Array) -> tuple[Array, Array]:
            th_prev = chi_prev[0]
            p_prev = chi_prev[1:]

            # rotational/bending strain of the current segment
            kappa_i = xi[i, 0]
            # linear strains of the current segment
            sigmas_i = xi[i, 1:]

            l_i = jnp.where(i == segment_idx, s_local, self.L[i])

            # classify whether it is a small strain
            small_strain = jnp.abs(kappa_i * l_i) < self.global_eps

            # compute the orientation
            th = th_prev + kappa_i * l_i

            # compute the components of the rotation matrix
            int_cos_th = lax.cond(
                small_strain,
                lambda _: l_i * jnp.cos(th_prev),
                lambda _: (jnp.sin(th) - jnp.sin(th_prev)) / kappa_i,
                operand=None,
            )
            int_sin_th = lax.cond(
                small_strain,
                lambda _: l_i * jnp.sin(th_prev),
                lambda _: -(jnp.cos(th) - jnp.cos(th_prev)) / kappa_i,
                operand=None,
            )

            R = jnp.stack(
                [
                    jnp.stack([int_cos_th, -int_sin_th]),
                    jnp.stack([int_sin_th, int_cos_th]),
                ]
            )

            p = p_prev + R @ sigmas_i

            chi = jnp.concatenate([th[None], p])

            return chi, chi

        _, chi_list = lax.scan(f=chi_i, init=chi_0, xs=jnp.arange(self.num_segments))

        chi_s = chi_list[segment_idx]

        return chi_s

    @eqx.filter_jit
    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = self.strain(q)

        chi = self.chi(xi, s)

        return chi

    @eqx.filter_jit
    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the planar pose at ``s``.
        """
        _, chis = self._forward_kinematics_and_arc_length_derivative(q, s)
        return chis

    @eqx.filter_jit
    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the planar pose and its arc-length derivative at ``s``.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, _ = self.classify_segment(s)
        xi_i = xi[segment_idx]
        chi = self._forward_kinematics(q, s)
        theta = chi[0]
        R = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)],
            ],
            dtype=chi.dtype,
        )
        p_s = R @ xi_i[1:]
        return chi, jnp.concatenate([xi_i[:1], p_s])

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics of the robot at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            chi_tips (Array): forward kinematics at each segment tip, shape (num_segments, 3)
                where each row is [theta, x, y].
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        chi_base = self._base_planar_pose(xi.dtype)

        def integrate_segment(chi_prev: Array, i: Array) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            kappa_i = xi_i[0]  # rotational/bending strain of the current segment
            sigmas_i = xi_i[1:]  # linear strains of the current segment

            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)

            th_prev = chi_prev[0]
            p_prev = chi_prev[1:]

            # classify whether it is a small strain
            small_strain = jnp.abs(kappa_i * L_i) < self.global_eps

            # compute the orientation
            th = th_prev + kappa_i * L_i

            # compute the components of the rotation matrix
            int_cos_th = lax.cond(
                small_strain,
                lambda _: L_i * jnp.cos(th_prev),
                lambda _: (jnp.sin(th) - jnp.sin(th_prev)) / kappa_i,
                operand=None,
            )
            int_sin_th = lax.cond(
                small_strain,
                lambda _: L_i * jnp.sin(th_prev),
                lambda _: -(jnp.cos(th) - jnp.cos(th_prev)) / kappa_i,
                operand=None,
            )

            R = jnp.stack(
                [
                    jnp.stack([int_cos_th, -int_sin_th]),
                    jnp.stack([int_sin_th, int_cos_th]),
                ]
            )

            p = p_prev + R @ sigmas_i
            chi_tip = jnp.concatenate([th[None], p])

            return chi_tip, chi_tip

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, chi_tips = lax.scan(integrate_segment, chi_base, indices)

        return chi_tips

    @eqx.filter_jit
    def forward_kinematics_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a batch of arc-length positions.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): arc-length positions in [0, L] of shape (N,).

        Returns:
            chi_ps (Array): forward kinematics evaluated at all points, shape (N, 3)
                where each row is [theta, x, y].
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        chi_tips = self.forward_kinematics_tips(q)
        chi_base = self._base_planar_pose(xi.dtype)
        chi_bases = jnp.concatenate([chi_base[None, :], chi_tips[:-1]], axis=0)

        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)
        xi_ps = xi[segment_indices]
        chi_base_ps = chi_bases[segment_indices]

        def integrate_point(chi_prev: Array, xi_i: Array, arc_len: Array) -> Array:
            th_prev = chi_prev[0]
            p_prev = chi_prev[1:]

            kappa_i = xi_i[0]  # rotational/bending strain of the current segment
            sigmas_i = xi_i[1:]  # linear strains of the current segment

            # classify whether it is a small strain
            small_strain = jnp.abs(kappa_i * arc_len) < self.global_eps

            # compute the orientation
            th = th_prev + kappa_i * arc_len

            # compute the components of the rotation matrix
            int_cos_th = lax.cond(
                small_strain,
                lambda _: arc_len * jnp.cos(th_prev),
                lambda _: (jnp.sin(th) - jnp.sin(th_prev)) / kappa_i,
                operand=None,
            )
            int_sin_th = lax.cond(
                small_strain,
                lambda _: arc_len * jnp.sin(th_prev),
                lambda _: -(jnp.cos(th) - jnp.cos(th_prev)) / kappa_i,
                operand=None,
            )

            R = jnp.stack(
                [
                    jnp.stack([int_cos_th, -int_sin_th]),
                    jnp.stack([int_sin_th, int_cos_th]),
                ]
            )

            p = p_prev + R @ sigmas_i

            return jnp.concatenate([th[None], p])

        chi_ps = vmap(integrate_point)(chi_base_ps, xi_ps, s_local_ps)

        return chi_ps

    @eqx.filter_jit
    def _compute_relative_segment_poses(self, chi_tips: Array) -> Array:
        """
        Compute the relative pose differences of each segment tip w.r.t. the previous segment tip
        in a vectorized fashion.

        Args:
            chi_tips (Array): absolute tip poses of shape (num_segments, 3) where each row is [theta, x, y]
                              representing the pose of each segment tip in the global frame.

        Returns:
            chi_rel (Array): relative poses of shape (num_segments, 3) where each row is [delta_theta, delta_x, delta_y]
                            representing the relative pose change from the previous segment tip.
                            The first segment's relative pose is computed w.r.t. ``base_pose``.
        """
        # Ensure chi_tips has the correct shape
        chi_tips = chi_tips.reshape(self.num_segments, 3)

        # Create array of previous poses: base + all segment tips except the last
        chi_base = self._base_planar_pose(chi_tips.dtype)
        prev_poses = jnp.concatenate([chi_base[None, :], chi_tips[:-1]], axis=0)

        # Compute relative poses vectorized
        # For SE(2), the relative transformation between poses chi_prev and chi_curr is:
        # delta_chi = log(exp(chi_prev)^(-1) @ exp(chi_curr))
        # But for computational efficiency, we can compute this more directly:

        # Extract angles and positions
        theta_prev = prev_poses[:, 0]  # (num_segments,)
        p_prev = prev_poses[:, 1:]  # (num_segments, 2)

        theta_curr = chi_tips[:, 0]  # (num_segments,)
        p_curr = chi_tips[:, 1:]  # (num_segments, 2)

        # Relative angle is simply the difference
        delta_theta = theta_curr - theta_prev  # (num_segments,)

        # Relative position needs to be rotated to the previous frame
        cos_prev = jnp.cos(theta_prev)  # (num_segments,)
        sin_prev = jnp.sin(theta_prev)  # (num_segments,)

        # Relative position in global frame
        delta_p_global = p_curr - p_prev  # (num_segments, 2)

        # Transform relative position to the previous segment frame using rotation matrix transpose
        # R_prev^T @ delta_p_global for each segment
        delta_x = cos_prev * delta_p_global[:, 0] + sin_prev * delta_p_global[:, 1]
        delta_y = -sin_prev * delta_p_global[:, 0] + cos_prev * delta_p_global[:, 1]

        # Combine relative transformations
        chi_rel = jnp.column_stack([delta_theta, delta_x, delta_y])  # (num_segments, 3)

        return chi_rel

    @eqx.filter_jit
    def inverse_kinematics(self, chi_tips: Array) -> Array:
        """
        Execute inverse kinematics to find the generalized coordinates / configurations.
        Instead of an iterative IK procedure, we leverage here the closed-form solution for the mapping
        the relative tip poses of each segment distal end with respect to the proximal end of the segment
        to the planar constant strain parameters.

        Args:
            chi_tips (Array): the tip poses of shape (num_segments, 3)
                where each row is [theta, x, y] for the corresponding segment tip.

        Returns:
            q (Array): the generalized coordinates of shape (num_active_strains,)
        """
        # Compute the relative poses of each segment tip w.r.t. the previous segment tip
        chi_rel = self._compute_relative_segment_poses(chi_tips)

        def _inverse_kinematics_single_segment(chi_rel_i: Array, s_i: Array) -> Array:
            """
            Compute the inverse kinematics for a single segment given its relative tip pose and arc-length.

            Args:
                chi_rel_i (Array): relative tip pose of shape (3,) : [delta_theta, delta_x, delta_y]
                s_i (Array): arc-length position of the tip along the segment.

            Returns:
                xi_i (Array): strain parameters of shape (3,) : [kappa_z, sigma_x, sigma_y]
            """
            th, px, py = chi_rel_i

            # compute the inverse kinematics for the virtual backbone
            divisor = jnp.cos(th) - 1
            xi = lax.select(
                jnp.abs(th) < self.global_eps,
                jnp.array([0.0, px / s_i, py / s_i]),
                (
                    th
                    / (2 * s_i)
                    * jnp.stack(
                        [
                            2 * jnp.ones((), dtype=th.dtype),
                            py - (px * jnp.sin(th)) / divisor,
                            -px - (py * jnp.sin(th)) / divisor,
                        ]
                    )
                ),
            )
            return xi

        # define the local arc-length positions of each segment tip
        s_local = self.L

        # apply the closed-form inverse kinematics for each segment using the
        xi_sgs = vmap(_inverse_kinematics_single_segment)(chi_rel, s_local)
        # flatten to (3*num_segments,)
        xi = xi_sgs.reshape(-1)

        # use the pseudo-inverse of the strain basis matrix to map to generalized coordinates
        q = jnp.linalg.pinv(self.B_xi) @ (xi - self.xi_ref)

        return q

    @eqx.filter_jit
    def _final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 3, 3)

        Returns:
            J_final (Array): Jacobian for the selected strains of shape (3, num_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(3, self.num_strains)

        return J_final

    @eqx.filter_jit
    def _J_local_tips(self, q: Array) -> Array:
        """
        Compute the body-frame Jacobian at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian at each segment tip, shape (num_segments, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        def scan_body(J_prev: Array, i: Array) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)

            Ad_inv = constant_strain.adjoint_inverse_se2(xi_i, L_i, eps=self.global_eps)
            T = constant_strain.tangent_se2(xi_i, L_i, eps=self.tangent_eps)

            J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

            return J_next, J_next

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, J_target_tips = lax.scan(scan_body, zeros, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_target_tips)

        return J_local_tips

    @eqx.filter_jit
    def _J_local_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the body-frame Jacobian at a batch of arc-length positions.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): arc-length positions in [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        J_tips = self._J_local_tips(q)  # (num_segments, 3, num_strains)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]  # (N, 3, num_strains)
        xi_ps = xi[segment_indices]

        N = s_ps.shape[0]
        J_base_ps = J_base_ps.reshape(N, 3, self.num_segments, 3).transpose(
            0, 2, 1, 3
        )  # (N, num_segments, 3, 3)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            arc_len: Array,
            J_base: Array,
        ) -> Array:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.tangent_eps)

            return self._update_body_jacobian_step(J_base, i, Ad_inv, T)

        J_local_ps = vmap(integrate_segment)(
            segment_indices, xi_ps, s_local_ps, J_base_ps
        )

        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)

        return J_local_ps

    @eqx.filter_jit
    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_active_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        zero_slice = jnp.zeros((3, 3), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.global_eps)

            return self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

        def scan_body(
            carry: tuple[Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array], Array]:
            J_prev, J_target, done = carry

            def compute_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next = integrate_segment(J_prev, i, xi_i, arc_len)

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_target)
                done_next = jnp.logical_or(done, is_target)

                return (J_next, J_target_next, done_next), zero_slice

            def skip_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                return (J_prev, J_target, done), zero_slice

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        carry_init = (
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, J_target, _), _ = lax.scan(scan_body, carry_init, indices)

        J_local = self._final_size_jacobian(J_target) @ self.B_xi

        return J_local

    def _pcs_jacobian_step_terms(
        self, xi_i: Array, arc_len: Array
    ) -> tuple[Array, Array]:
        """
        Compute shared planar constant-strain Jacobian step terms.

        Args:
            xi_i: Constant SE(2) strain of segment ``i`` with shape ``(3,)``.
            arc_len: Local arc length used for this segment step.

        Returns:
            Tuple ``(Ad_inv, T)`` where ``Ad_inv`` is the inverse SE(2) adjoint
            of the local segment transform and ``T`` is the planar tangent
            operator. Both arrays have shape ``(3, 3)``.
        """
        Ad_inv = constant_strain.adjoint_inverse_se2(xi_i, arc_len, eps=self.global_eps)
        T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.global_eps)
        return Ad_inv, T

    def _pcs_jacobian_arc_length_step_terms(
        self, xi_i: Array, arc_len: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute shared PlanarPCS terms for ``J`` and ``dJ/ds`` propagation.

        Args:
            xi_i: Constant SE(2) strain of segment ``i``, shape ``(3,)``.
            arc_len: Local arc length used for the segment step.

        Returns:
            Tuple ``(Ad_inv, T, dAd_inv_ds, dT_ds)`` containing the primal
            Jacobian step terms and their analytical local arc-length
            derivatives.
        """
        Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, arc_len)
        dAd_inv_ds = -se2.small_adjoint(xi_i) @ Ad_inv
        dT_ds = constant_strain.adjoint_se2(xi_i, arc_len, eps=self.global_eps)
        return Ad_inv, T, dAd_inv_ds, dT_ds

    def _pcs_relative_pose(self, xi_i: Array, arc_len: Array) -> Array:
        """
        Compute the relative SE(2) transform for one segment slice.

        Args:
            xi_i: Constant planar strain of segment ``i``, shape ``(3,)``.
            arc_len: Local arc length of the slice.

        Returns:
            Relative SE(2) transform with the translation obtained by
            analytically integrating the constant planar strain over
            ``arc_len``.
        """
        kappa_i = xi_i[0]
        sigmas_i = xi_i[1:]
        theta = kappa_i * arc_len
        small_strain = jnp.abs(theta) < self.global_eps

        int_cos = lax.cond(
            small_strain,
            lambda _: arc_len,
            lambda _: jnp.sin(theta) / kappa_i,
            operand=None,
        )
        int_sin = lax.cond(
            small_strain,
            lambda _: jnp.zeros((), dtype=theta.dtype),
            lambda _: (1.0 - jnp.cos(theta)) / kappa_i,
            operand=None,
        )

        R = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)],
            ],
            dtype=xi_i.dtype,
        )
        V = jnp.array(
            [
                [int_cos, -int_sin],
                [int_sin, int_cos],
            ],
            dtype=xi_i.dtype,
        )
        p = V @ sigmas_i
        return jnp.block(
            [
                [R, p[:, None]],
                [
                    jnp.zeros((1, 2), dtype=xi_i.dtype),
                    jnp.ones((1, 1), dtype=xi_i.dtype),
                ],
            ]
        )

    def _compose_planar_pose(self, chi: Array, g_step: Array) -> Array:
        """
        Compose a planar pose vector with a relative SE(2) transform.

        Args:
            chi: Current absolute planar pose ``[theta, x, y]``.
            g_step: Relative SE(2) transform to apply from ``chi``.

        Returns:
            Absolute planar pose after the relative step, again represented as
            ``[theta, x, y]``.
        """
        theta = chi[0]
        R = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]],
            dtype=chi.dtype,
        )
        theta_next = theta + jnp.arctan2(g_step[1, 0], g_step[0, 0])
        p_next = chi[1:] + R @ g_step[:2, 2]
        return jnp.concatenate([theta_next[None], p_next])

    def _base_planar_pose(self, dtype: jnp.dtype) -> Array:
        """
        Return the absolute base pose used by PlanarPCS scans.

        Args:
            dtype: Desired floating-point dtype of the returned array.

        Returns:
            Planar base pose ``[theta0, x0, y0]`` with shape ``(3,)``.
        """
        return jnp.asarray(self.base_pose, dtype=dtype)

    def _update_body_jacobian_step(
        self, J_prev: Array, i: Array, Ad_inv: Array, T: Array
    ) -> Array:
        """
        Propagate the block body-frame Jacobian through one PlanarPCS step.

        Args:
            J_prev: Previous full block Jacobian, shape
                ``(num_segments, 3, 3)``.
            i: Current segment index.
            Ad_inv: Inverse adjoint of the current relative transform.
            T: Tangent operator of the current relative transform.

        Returns:
            Updated full block Jacobian with the active current-segment block
            set to ``Ad_inv @ T``.
        """
        J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_prev)
        return J_rot.at[i].set(Ad_inv @ T)

    def _update_body_jacobian_time_derivative_step(
        self,
        J_prev: Array,
        Jd_prev: Array,
        i: Array,
        xid_i: Array,
        Ad_inv: Array,
        T: Array,
        Td: Array,
    ) -> tuple[Array, Array]:
        """
        Propagate ``J`` and its time derivative through one PlanarPCS step.

        Args:
            J_prev: Previous full block body-frame Jacobian.
            Jd_prev: Previous full block body-frame Jacobian time derivative.
            i: Current segment index.
            xid_i: Current segment strain time derivative.
            Ad_inv: Inverse adjoint of the current relative transform.
            T: Tangent operator of the current relative transform.
            Td: Time derivative of ``T`` along ``xid_i``.

        Returns:
            Tuple ``(J_next, Jd_next)`` after the segment update.
        """
        J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)
        eta = lax.dynamic_index_in_dim(J_next, i, axis=0, keepdims=False) @ xid_i
        Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv
        Jd_rot = jnp.einsum("ij, njk->nik", Ad_inv, Jd_prev) + jnp.einsum(
            "ij, njk->nik", Ad_inv_dot, J_prev
        )
        Jd_next = Jd_rot.at[i].set(Ad_inv_dot @ T + Ad_inv @ Td)
        return J_next, Jd_next

    def _update_body_jacobian_arc_length_derivative_step(
        self,
        J_prev: Array,
        i: Array,
        Ad_inv: Array,
        T: Array,
        dAd_inv_ds: Array,
        dT_ds: Array,
    ) -> Array:
        """
        Propagate the arc-length derivative of the block body Jacobian.

        Args:
            J_prev: Previous full block body-frame Jacobian.
            i: Current segment index.
            Ad_inv: Inverse adjoint of the current relative transform.
            T: Tangent operator of the current relative transform.
            dAd_inv_ds: Arc-length derivative of ``Ad_inv``.
            dT_ds: Arc-length derivative of ``T``.

        Returns:
            Full block Jacobian arc-length derivative for the current step.
        """
        Js_next = jnp.einsum("ij, njk->nik", dAd_inv_ds, J_prev)
        return Js_next.at[i].set(dAd_inv_ds @ T + Ad_inv @ dT_ds)

    def _jacobian_bodyframe_with_pose(self, q: Array, s: Array) -> tuple[Array, Array]:
        """
        Compute body-frame Jacobian and planar pose in a single segment scan.

        Args:
            q: Active generalized coordinates, shape ``(num_dofs,)``.
            s: Backbone arc-length coordinate.

        Returns:
            Tuple ``(chi_s, J_body)`` where ``chi_s`` is represented as
            ``[theta, x, y]`` and ``J_body`` maps active velocities to
            body-frame planar twists at ``s``.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        chi0 = self._base_planar_pose(xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
            chi_prev, J_prev, chi_target, J_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, arc_len)
                chi_next = self._compose_planar_pose(
                    chi_prev, self._pcs_relative_pose(xi_i, arc_len)
                )
                J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

                is_target = i == segment_idx
                return (
                    chi_next,
                    J_next,
                    jnp.where(is_target, chi_next, chi_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                return (chi_prev, J_prev, chi_target, J_target, done), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, chi_target, J_target, _), _ = lax.scan(
            scan_body,
            (chi0, zeros, chi0, zeros, jnp.array(False, dtype=jnp.bool_)),
            indices,
        )
        return chi_target, self._final_size_jacobian(J_target) @ self.B_xi

    def _jacobian_and_arc_length_derivative_bodyframe_with_pose(
        self, q: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """
        Compute pose, body-frame Jacobian, and arc-length derivative together.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        chi0 = self._base_planar_pose(xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
            chi_prev, J_prev, chi_target, J_target, Js_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T, dAd_inv_ds, dT_ds = self._pcs_jacobian_arc_length_step_terms(
                    xi_i, arc_len
                )
                chi_next = self._compose_planar_pose(
                    chi_prev, self._pcs_relative_pose(xi_i, arc_len)
                )
                J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)
                Js_next = self._update_body_jacobian_arc_length_derivative_step(
                    J_prev, i, Ad_inv, T, dAd_inv_ds, dT_ds
                )

                is_target = i == segment_idx
                return (
                    chi_next,
                    J_next,
                    jnp.where(is_target, chi_next, chi_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.where(is_target, Js_next, Js_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
                return (chi_prev, J_prev, chi_target, J_target, Js_target, done), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, chi_target, J_target, Js_target, _), _ = lax.scan(
            scan_body,
            (
                chi0,
                zeros,
                chi0,
                zeros,
                zeros,
                jnp.array(False, dtype=jnp.bool_),
            ),
            indices,
        )

        J = self._final_size_jacobian(J_target) @ self.B_xi
        Js = self._final_size_jacobian(Js_target) @ self.B_xi
        return chi_target, J, Js

    def _jacobian_and_time_derivative_bodyframe_with_pose(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """Compute pose, body-frame Jacobian, and time derivative together."""
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        chi0 = self._base_planar_pose(xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
            chi_prev, J_prev, Jd_prev, chi_target, J_target, Jd_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, arc_len)
                Td = constant_strain.tangent_derivative_se2(
                    xi_i, xid_i, arc_len, eps=self.global_eps
                )
                chi_next = self._compose_planar_pose(
                    chi_prev, self._pcs_relative_pose(xi_i, arc_len)
                )
                J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                    J_prev, Jd_prev, i, xid_i, Ad_inv, T, Td
                )

                is_target = i == segment_idx
                return (
                    chi_next,
                    J_next,
                    Jd_next,
                    jnp.where(is_target, chi_next, chi_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.where(is_target, Jd_next, Jd_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
                return (
                    chi_prev,
                    J_prev,
                    Jd_prev,
                    chi_target,
                    J_target,
                    Jd_target,
                    done,
                ), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, _, chi_target, J_target, Jd_target, _), _ = lax.scan(
            scan_body,
            (
                chi0,
                zeros,
                zeros,
                chi0,
                zeros,
                zeros,
                jnp.array(False, dtype=jnp.bool_),
            ),
            indices,
        )

        J = self._final_size_jacobian(J_target) @ self.B_xi
        Jd = self._final_size_jacobian(Jd_target) @ self.B_xi
        return chi_target, J, Jd

    @eqx.filter_jit
    def jacobian_and_arc_length_derivative_bodyframe(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the body-frame Jacobian and its arc-length derivative at ``s``.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        zero_slice = jnp.zeros((3, 3), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.global_eps)

            return self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

        def integrate_segment_arc_length_derivative(
            J_base: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.global_eps)
            dAd_inv_ds = -se2.small_adjoint(xi_i) @ Ad_inv
            dT_ds = constant_strain.adjoint_se2(xi_i, arc_len, eps=self.global_eps)

            return self._update_body_jacobian_arc_length_derivative_step(
                J_base, i, Ad_inv, T, dAd_inv_ds, dT_ds
            )

        def scan_body(
            carry: tuple[Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array], Array]:
            J_prev, J_target, Js_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next = integrate_segment(J_prev, i, xi_i, arc_len)
                Js_next = integrate_segment_arc_length_derivative(
                    J_prev, i, xi_i, arc_len
                )

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_target)
                Js_target_next = jnp.where(is_target, Js_next, Js_target)
                done_next = jnp.logical_or(done, is_target)

                return (J_next, J_target_next, Js_target_next, done_next), zero_slice

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array], Array]:
                return (J_prev, J_target, Js_target, done), zero_slice

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, J_target, Js_target, _), _ = lax.scan(
            scan_body,
            (zeros, zeros, zeros, jnp.array(False, dtype=jnp.bool_)),
            indices,
        )

        J = self._final_size_jacobian(J_target) @ self.B_xi
        Js = self._final_size_jacobian(Js_target) @ self.B_xi
        return J, Js

    @eqx.filter_jit
    def jacobian_arc_length_derivative_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the body-frame Jacobian at ``s``.
        """
        _, Js = self.jacobian_and_arc_length_derivative_bodyframe(q, s)
        return Js

    @eqx.filter_jit
    def jacobian_bodyframe_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a batch of points s_ps along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
        """
        J_local_ps_full = self._J_local_batched(q, s_ps)  # shape (N, 3, num_strains)
        J_local_ps = jnp.einsum("ijk, kl->ijl", J_local_ps_full, self.B_xi)

        return J_local_ps

    def _rotation_adjoint_from_pose(self, chi: Array) -> Array:
        """Adjoint of the planar pose rotation, with zero translation."""
        theta = chi[0]
        zero = jnp.zeros((), dtype=theta.dtype)
        g = poses.planar_pose_to_transform(jnp.stack([theta, zero, zero]))
        return se2.adjoint(g)

    def _body_jacobian_to_inertial(self, chi: Array, J_local: Array) -> Array:
        """Rotate a PlanarPCS body-frame Jacobian into the inertial frame."""
        return self._rotation_adjoint_from_pose(chi) @ J_local

    @eqx.filter_jit
    def jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
        """
        chi, J_local = self._jacobian_bodyframe_with_pose(q, s)
        return self._body_jacobian_to_inertial(chi, J_local)

    @eqx.filter_jit
    def jacobian_and_arc_length_derivative_inertialframe(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the inertial-frame Jacobian and its arc-length derivative at ``s``.
        """
        chi, J_local, Js_local = (
            self._jacobian_and_arc_length_derivative_bodyframe_with_pose(q, s)
        )
        Ad_g = self._rotation_adjoint_from_pose(chi)
        zero = jnp.zeros((), dtype=chi.dtype)

        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, _ = self.classify_segment(s)
        eta_rot_s = jnp.stack([xi[segment_idx, 0], zero, zero])
        Ad_g_s = Ad_g @ se2.small_adjoint(eta_rot_s)

        J = Ad_g @ J_local
        Js = Ad_g_s @ J_local + Ad_g @ Js_local
        return J, Js

    @eqx.filter_jit
    def jacobian_arc_length_derivative_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the inertial-frame Jacobian at ``s``.
        """
        chi, J_local, Js_local = (
            self._jacobian_and_arc_length_derivative_bodyframe_with_pose(q, s)
        )
        Ad_g = self._rotation_adjoint_from_pose(chi)
        zero = jnp.zeros((), dtype=chi.dtype)

        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, _ = self.classify_segment(s)
        eta_rot_s = jnp.stack([xi[segment_idx, 0], zero, zero])
        Ad_g_s = Ad_g @ se2.small_adjoint(eta_rot_s)
        Js = Ad_g_s @ J_local + Ad_g @ Js_local
        return Js

    @eqx.filter_jit
    def jacobian_inertialframe_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a batch of points s_ps along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_global_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
        """
        J_local_ps = self.jacobian_bodyframe_batched(q, s_ps)

        chi_ps = self.forward_kinematics_batched(q, s_ps)  # shape (N, 3)
        thetas = chi_ps[:, 0]

        def lift_theta(th: Array) -> Array:
            zeros = jnp.zeros((), dtype=th.dtype)
            return poses.planar_pose_to_transform(jnp.stack([th, zeros, zeros]))

        g_rot_ps = vmap(lift_theta)(thetas)
        Ad_g_ps = vmap(se2.adjoint)(g_rot_ps)

        J_global_ps = jnp.einsum("nij, njk->nik", Ad_g_ps, J_local_ps)

        return J_global_ps

    @eqx.filter_jit
    def jacobian_tips(self, q: Array) -> Array:
        """
        Compute inertial-frame Jacobians at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_tips (Array): inertial-frame Jacobians at each segment tip, shape
                (num_segments, 3, num_active_strains).
        """
        J_local_tips = jnp.einsum("ijk,kl->ijl", self._J_local_tips(q), self.B_xi)
        chi_tips = self.forward_kinematics_tips(q)

        def rotate_pair(chi_i: Array, J_i: Array) -> Array:
            theta = chi_i[0]
            zeros = jnp.zeros((), dtype=theta.dtype)
            g_rot = poses.planar_pose_to_transform(jnp.stack([theta, zeros, zeros]))
            return se2.adjoint(g_rot) @ J_i

        return vmap(rotate_pair)(chi_tips, J_local_tips)

    @eqx.filter_jit
    def _J_Jd_local_tips(self, q: Array, qd: Array) -> tuple[Array, Array]:
        """
        Compute the body-frame Jacobian and its time derivative at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            Tuple[Array, Array]: Jacobian and Jacobian time derivative at each segment tip,
                both of shape (num_segments, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        Ad_inv_tips = vmap(
            lambda xi_i, L_i: constant_strain.adjoint_inverse_se2(
                xi_i, L_i, eps=self.global_eps
            )
        )(xi, self.L)
        T_tips = vmap(
            lambda xi_i, L_i: constant_strain.tangent_se2(
                xi_i, L_i, eps=self.tangent_eps
            )
        )(xi, self.L)
        Td_tips = vmap(
            lambda xi_i, xid_i, L_i: constant_strain.tangent_derivative_se2(
                xi_i, xid_i, L_i, eps=self.tangent_eps
            )
        )(xi, xid, self.L)

        def scan_body(
            carry: tuple[Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_prev = carry

            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)
            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)

            J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                J_prev, Jd_prev, i, xid_i, Ad_inv_i, T_i, Td_i
            )

            return (J_next, Jd_next), (J_next, Jd_next)

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        (_, _), (J_local_tips, Jd_local_tips) = lax.scan(
            scan_body, (zeros, zeros), indices
        )

        J_local_tips = vmap(self._final_size_jacobian)(J_local_tips)
        Jd_local_tips = vmap(self._final_size_jacobian)(Jd_local_tips)

        return J_local_tips, Jd_local_tips

    @eqx.filter_jit
    def _J_Jd_local_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the body-frame Jacobian and its time derivative at a batch of arc-length positions.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s_ps (Array): arc-length positions in [0, L] of shape (N,).

        Returns:
            Tuple[Array, Array]: Jacobians and their time derivatives evaluated at all points,
                both of shape (N, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)

        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        J_tips, Jd_tips = self._J_Jd_local_tips(q, qd)
        zeros_like_tip = jnp.zeros_like(J_tips[:1])

        J_bases = jnp.concatenate([zeros_like_tip, J_tips[:-1]], axis=0)
        Jd_bases = jnp.concatenate([zeros_like_tip, Jd_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        Jd_base_ps = Jd_bases[segment_indices]

        xi_ps = xi[segment_indices]
        xid_ps = xid[segment_indices]

        N = s_ps.shape[0]
        J_base_ps = J_base_ps.reshape(N, 3, self.num_segments, 3).transpose(0, 2, 1, 3)
        Jd_base_ps = Jd_base_ps.reshape(N, 3, self.num_segments, 3).transpose(
            0, 2, 1, 3
        )

        def integrate_segment(
            i: Array,
            xi_i: Array,
            xid_i: Array,
            arc_len: Array,
            J_base: Array,
            Jd_base: Array,
        ) -> tuple[Array, Array]:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.tangent_eps)
            Td = constant_strain.tangent_derivative_se2(
                xi_i, xid_i, arc_len, eps=self.tangent_eps
            )

            J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                J_base, Jd_base, i, xid_i, Ad_inv, T, Td
            )

            return J_next, Jd_next

        J_local_ps, Jd_local_ps = vmap(integrate_segment)(
            segment_indices, xi_ps, xid_ps, s_local_ps, J_base_ps, Jd_base_ps
        )

        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)
        Jd_local_ps = vmap(self._final_size_jacobian)(Jd_local_ps)

        return J_local_ps, Jd_local_ps

    @eqx.filter_jit
    def jacobian_and_time_derivative_bodyframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_active_strains)
            Jd_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (3, num_active_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)
        zero_slice = jnp.zeros((3, 3), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            Jd_prev: Array,
            i: Array,
            xi_i: Array,
            xid_i: Array,
            arc_len: Array,
        ) -> tuple[Array, Array]:
            Ad_inv = constant_strain.adjoint_inverse_se2(
                xi_i, arc_len, eps=self.global_eps
            )
            T = constant_strain.tangent_se2(xi_i, arc_len, eps=self.global_eps)
            Td = constant_strain.tangent_derivative_se2(
                xi_i, xid_i, arc_len, eps=self.global_eps
            )

            return self._update_body_jacobian_time_derivative_step(
                J_prev, Jd_prev, i, xid_i, Ad_inv, T, Td
            )

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
            J_prev, Jd_prev, J_target, Jd_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next, Jd_next = integrate_segment(
                    J_prev,
                    Jd_prev,
                    i,
                    xi_i,
                    xid_i,
                    arc_len,
                )

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_target)
                Jd_target_next = jnp.where(is_target, Jd_next, Jd_target)
                done_next = jnp.logical_or(done, is_target)

                return (
                    J_next,
                    Jd_next,
                    J_target_next,
                    Jd_target_next,
                    done_next,
                ), zero_slice

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
                return (J_prev, Jd_prev, J_target, Jd_target, done), zero_slice

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        carry_init = (
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.array(False, dtype=jnp.bool_),
        )

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, J_target, Jd_target, _), _ = lax.scan(scan_body, carry_init, indices)

        J_local_full = self._final_size_jacobian(J_target)
        Jd_local_full = self._final_size_jacobian(Jd_target)

        J_local = J_local_full @ self.B_xi
        Jd_local = Jd_local_full @ self.B_xi

        return J_local, Jd_local

    @eqx.filter_jit
    def jacobian_and_time_derivative_bodyframe_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a batch of points s_ps along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
            Jd_local_ps (Array): Time-derivative of the Jacobians, shape (N, 3, num_active_strains)
        """
        J_local_ps_full, Jd_local_ps_full = self._J_Jd_local_batched(q, qd, s_ps)

        J_local_ps = jnp.einsum("ijk, kl->ijl", J_local_ps_full, self.B_xi)
        Jd_local_ps = jnp.einsum("ijk, kl->ijl", Jd_local_ps_full, self.B_xi)

        return J_local_ps, Jd_local_ps

    @eqx.filter_jit
    def jacobian_and_time_derivative_inertialframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_active_strains)
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_active_strains)
        """
        J_local, Jd_local = self.jacobian_and_time_derivative_bodyframe(q, qd, s)
        chi = self._forward_kinematics(q, s)
        Ad_g = self._rotation_adjoint_from_pose(chi)

        eta_body = J_local @ qd
        zero_twist = jnp.zeros((), dtype=eta_body.dtype)
        eta_rot = jnp.stack([eta_body[0], zero_twist, zero_twist])
        Ad_g_dot = Ad_g @ se2.small_adjoint(eta_rot)

        J_global = jnp.einsum("ij, jk->ik", Ad_g, J_local)
        Jd_global = jnp.einsum("ij, jk->ik", Ad_g, Jd_local) + jnp.einsum(
            "ij, jk->ik", Ad_g_dot, J_local
        )

        return J_global, Jd_global

    def jacobian_and_time_derivative_inertialframe_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a batch of points s_ps along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_global_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
            Jd_global_ps (Array): Time-derivative of the Jacobians, shape (N, 3, num_active_strains)
        """
        J_local_ps, Jd_local_ps = self.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_ps
        )

        chi_ps = self.forward_kinematics_batched(q, s_ps)
        thetas = chi_ps[:, 0]

        def lift_theta(th: Array) -> Array:
            zeros = jnp.zeros((), dtype=th.dtype)
            return poses.planar_pose_to_transform(jnp.stack([th, zeros, zeros]))

        g_rot_ps = vmap(lift_theta)(thetas)
        Ad_g_ps = vmap(se2.adjoint)(g_rot_ps)

        eta_body_ps = jnp.einsum("ijk, k->ij", J_local_ps, qd)
        zeros = jnp.zeros_like(eta_body_ps[:, :1])
        eta_rot_ps = jnp.concatenate([eta_body_ps[:, :1], zeros, zeros], axis=1)
        Ad_g_dot_ps = jnp.einsum(
            "nij, njk->nik", Ad_g_ps, vmap(se2.small_adjoint)(eta_rot_ps)
        )

        J_global_ps = jnp.einsum("nij, njk->nik", Ad_g_ps, J_local_ps)
        Jd_global_ps = jnp.einsum("nij, njk->nik", Ad_g_ps, Jd_local_ps) + jnp.einsum(
            "nij, njk->nik", Ad_g_dot_ps, J_local_ps
        )

        return J_global_ps, Jd_global_ps

    @eqx.filter_jit
    def _jacobian(self, q: Array, s: Array) -> Array:
        """Protected SoftRobot hook for the inertial-frame Jacobian."""
        return self.jacobian_inertialframe(q, s)

    @eqx.filter_jit
    def _jacobian_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """Protected SoftRobot hook for the inertial-frame Jacobian arc-length derivative."""
        return self.jacobian_arc_length_derivative_inertialframe(q, s)

    @eqx.filter_jit
    def _jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """Protected SoftRobot hook for inertial-frame Jacobian arc-length derivative."""
        return self.jacobian_and_arc_length_derivative_inertialframe(q, s)

    @eqx.filter_jit
    def jacobian_batched(self, q: Array, s_ps: Array) -> Array:
        """Compute inertial-frame Jacobians at multiple arc-length positions."""
        return self.jacobian_inertialframe_batched(q, s_ps)

    @eqx.filter_jit
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """Protected SoftRobot hook for the inertial-frame Jacobian time derivative."""
        return self.jacobian_and_time_derivative_inertialframe(q, qd, s)

    @eqx.filter_jit
    def jacobian_and_time_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """Compute inertial-frame Jacobians and time derivatives at multiple arc-length positions."""
        return self.jacobian_and_time_derivative_inertialframe_batched(q, qd, s_ps)

    # ==========================================
    # Useful functions for the system

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        # Cross-sectional area
        A_i = jnp.pi * self.r[i] ** 2
        return A_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the local second moment of area for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        # Second moment of area
        I_i = jnp.pi * self.r[i] ** 4 / 4
        return I_i

    def _compute_local_mass_matrix(self, i: Array) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            M_i (Array): local mass matrix of shape (3, 3) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area
        I_i = self._local_second_moment_of_area(i)  # Second moment of area

        M_i = rho_i * jnp.diag(jnp.array([I_i, A_i, A_i]))
        return M_i

    @eqx.filter_jit
    def _local_mass_matrix(self, i: Array) -> Array:
        """
        Return the cached local mass matrix for the i-th segment.

        Args:
            i (Array): index of the segment

        Returns:
            M_i (Array): Local mass matrix of shape (3, 3) for the i-th segment.
        """
        return self.M_segments[i]

    # ===========================================
    # Dynamical matrices computation

    @eqx.filter_jit
    def _inertia_full_matrix(self, q: Array) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            B_full (Array): Full inertia matrix of shape (num_strains, num_strains).
        """
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )

        J_ps = self._J_local_batched(q, Xs_scaled.flatten())
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )

        def B_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def B_ij(j: Array) -> Array:
                Ws_ij = Ws_scaled[i][j]
                J_ij = J_ps[i, j]

                return Ws_ij * J_ij.T @ M_i @ J_ij

            B_blocks_i = vmap(B_ij)(jnp.arange(1, self.num_integration_points - 1))

            return B_blocks_i

        B_blocks_tot = vmap(B_i)(jnp.arange(self.num_segments))

        B_full = jnp.sum(
            B_blocks_tot, axis=(0, 1)
        )  # Sum over segments and Gauss points

        return B_full

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            B (Array): Inertia matrix of shape (num_active_strains, num_active_strains).
        """
        B_full = self._inertia_full_matrix(q)

        B = self.B_xi.T @ B_full @ self.B_xi

        return B

    @eqx.filter_jit
    def _coriolis_full_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            C_full (Array): Full Coriolis matrix of shape (num_strains, num_strains).
        """
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )

        J_ps, Jd_ps = self._J_Jd_local_batched(q, qd, Xs_scaled.flatten())
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_integration_points, *Jd_ps.shape[1:]
        )

        def C_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def C_ij(j: Array) -> Array:
                Ws_ij = Ws_scaled[i][j]
                J_ij = J_ps[i, j]
                Jd_ij = Jd_ps[i, j]

                return Ws_ij * (
                    J_ij.T
                    @ (M_i @ Jd_ij + se2.coadjoint(J_ij @ self.B_xi @ qd) @ M_i @ J_ij)
                )

            C_blocks_i = vmap(C_ij)(jnp.arange(1, self.num_integration_points - 1))

            return C_blocks_i

        C_blocks_tot = vmap(C_i)(jnp.arange(self.num_segments))

        C_full = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C_full

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            C (Array): Coriolis matrix of shape (num_active_strains, num_active_strains).
        """
        C_full = self._coriolis_full_matrix(q, qd)

        C = self.B_xi.T @ C_full @ self.B_xi

        return C

    @eqx.filter_jit
    def _gravitational_full_force(self, q: Array) -> Array:
        """
        Compute the full gravitational force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            G (Array): Full gravitational force of shape (num_strains,).
        """
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )

        chi_ps = self.forward_kinematics_batched(q, Xs_scaled.flatten())
        g_ps = vmap(poses.planar_pose_to_transform)(chi_ps.reshape(-1, 3))
        g_ps = g_ps.reshape(self.num_segments, self.num_integration_points, 3, 3)

        J_ps = self._J_local_batched(q, Xs_scaled.flatten())
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )

        def G_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def G_ij(j: Array) -> Array:
                Ws_ij = Ws_scaled[i][j]
                g_ij = g_ps[i, j]
                J_ij = J_ps[i, j]

                Ad_g_inv_ij = se2.adjoint_inverse(g_ij)

                return -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

            G_blocks_segment_i = vmap(G_ij)(
                jnp.arange(1, self.num_integration_points - 1)
            )

            return G_blocks_segment_i

        G_blocks_tot = vmap(G_i)(jnp.arange(self.num_segments))

        G_full = jnp.sum(
            G_blocks_tot, axis=(0, 1)
        )  # Sum over links and quadrature points

        return G_full

    @eqx.filter_jit
    def _gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            G (Array): Gravitational force of shape (num_active_strains,).
        """
        G_full = self._gravitational_full_force(q)

        G = self.B_xi.T @ G_full

        return G

    @eqx.filter_jit
    def _local_stiffness_matrix(self, i: Array) -> Array:
        """
        Compute the local stiffness matrix of a planar system for a rod aligned along the x-axis.

        Args:
            i (Array): index of the segment

        Returns:
            S_i (Array): Local stiffness matrix of shape (3, 3) for the i-th segment.
        """
        I_i = self._local_second_moment_of_area(i)  # Second moment of area
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

        S_i = self.L[i] * jnp.diag(
            jnp.stack(
                [
                    I_i * self.E[i],  # bending Z
                    A_i * self.E[i],  # axial X
                    A_i * self.G[i],  # shear Y
                ],
                axis=0,
            )
        )

        return S_i

    def _compute_stiffness_full_matrix(self) -> Array:
        """Compute the uncached full stiffness matrix from current parameters."""
        # stiffness matrix of shape (num_segments, 3, 3)
        S_sms = vmap(self._local_stiffness_matrix)(jnp.arange(self.num_segments))

        # we define the elastic matrix of shape (num_strains, num_strains) as K(xi) = K @ xi where K is equal to
        return blk_diag(S_sms)

    @eqx.filter_jit
    def _stiffness(self, formulate_in_strain_space: bool = False) -> Array:
        if formulate_in_strain_space:
            return self.K_full

        return self.K

    @eqx.filter_jit
    def _stiffness_full_matrix(self) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix of shape (num_strains, num_strains).
        """
        return self.K_full

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Returns:
            K (Array): Stiffness matrix of shape (num_active_strains, num_active_strains).
        """
        return self.K

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            tau_el (Array): Elastic force of shape (num_active_strains,).
        """
        return self.K @ q

    @eqx.filter_jit
    def _damping_full_matrix(self) -> Array:
        """
        Compute the full damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Full damping matrix of shape (num_strains, num_strains).
        """
        return self.D_full

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            D (Array): Damping matrix of shape (num_active_strains, num_active_strains).
        """
        return self.D_active

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            A (Array): Actuation matrix of shape (num_active_strains, num_actuators).
        """
        A = jnp.identity(self.num_actuators)
        return A

    @eqx.filter_jit
    def _gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            U_G (float): Gravitational energy of the robot.
        """
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )

        chi_ps = self.forward_kinematics_batched(q, Xs_scaled.flatten())
        chi_ps = chi_ps.reshape(self.num_segments, self.num_integration_points, 3)

        def U_G_i(i: Array) -> Array:
            rho_i = self.rho[i]
            A_i = self._local_cross_sectional_area(i)

            def U_G_ij(j: Array) -> Array:
                Ws_ij = Ws_scaled[i][j]
                chi_ij = chi_ps[i, j]
                p_j = chi_ij.at[0].set(0.0)

                return -Ws_ij * rho_i * A_i * jnp.dot(p_j, self.g)

            U_G_blocks_segment_i = vmap(U_G_ij)(
                jnp.arange(1, self.num_integration_points - 1)
            )

            return U_G_blocks_segment_i

        U_G_blocks_tot = vmap(U_G_i)(jnp.arange(self.num_segments))

        U_G = jnp.sum(U_G_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points

        return U_G

    @eqx.filter_jit
    def _active_J_Jd_local_tips_from_strain(
        self,
        xi: Array,
        xid: Array,
        B_segments: Array,
        convective_only_jd: bool = False,
    ) -> tuple[Array, Array]:
        """Compute active-coordinate local Jacobians at all segment tips."""
        Ad_inv_tips = vmap(
            lambda xi_i, L_i: constant_strain.adjoint_inverse_se2(
                xi_i, L_i, eps=self.global_eps
            )
        )(xi, self.L)
        T_tips = vmap(
            lambda xi_i, L_i: constant_strain.tangent_se2(
                xi_i, L_i, eps=self.tangent_eps
            )
        )(xi, self.L)
        Td_tips = vmap(
            lambda xi_i, xid_i, L_i: constant_strain.tangent_derivative_se2(
                xi_i, xid_i, L_i, eps=self.tangent_eps
            )
        )(xi, xid, self.L)

        zeros = jnp.zeros((3, self.num_dofs), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array], i: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_prev = carry

            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
            B_i = lax.dynamic_index_in_dim(B_segments, i, axis=0, keepdims=False)
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)

            Ad_inv_T_i = Ad_inv_i @ T_i
            J_segment = Ad_inv_T_i @ B_i
            J_next = Ad_inv_i @ J_prev + J_segment

            eta = Ad_inv_T_i @ xid_i
            Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv_i

            if convective_only_jd:
                Jd_segment = (Ad_inv_i @ Td_i) @ B_i
            else:
                Jd_segment = (Ad_inv_dot @ T_i + Ad_inv_i @ Td_i) @ B_i
            Jd_next = Ad_inv_i @ Jd_prev + Ad_inv_dot @ J_prev + Jd_segment

            return (J_next, Jd_next), (J_next, Jd_next)

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        (_, _), (J_tips, Jd_tips) = lax.scan(scan_body, (zeros, zeros), indices)

        return J_tips, Jd_tips

    @eqx.filter_jit
    def integration_kinematics(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Return integration-point kinematics in active generalized coordinates.

        The stored quadrature grid includes zero-weight endpoint nodes; this
        method ignores those endpoints and evaluates only the nonzero-weight
        interior quadrature nodes. The coordinates are generalized coordinates
        for active strain components only.

        Args:
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.

        Returns:
            Tuple ``(g_ps, J_ps, Jd_ps)``. ``g_ps`` contains SE(2) poses with
            shape ``(self.num_segments, self.num_gauss_points, 3, 3)``.
            ``J_ps`` contains body-frame Jacobians in active generalized
            coordinates with shape
            ``(self.num_segments, self.num_gauss_points, 3, self.num_dofs)``.
            ``Jd_ps`` contains their time derivatives with the same shape as
            ``J_ps``.
        """
        _, g_ps, J_ps, Jd_ps = self._integration_kinematics(q, qd)
        return g_ps, J_ps, Jd_ps

    @eqx.filter_jit
    def _integration_kinematics(
        self, q: Array, qd: Array, convective_only_jd: bool = False
    ) -> tuple[Array, Array, Array, Array]:
        """
        Return weights, poses, active Jacobians, and derivatives at integration points.

        Args:
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.
            convective_only_jd: If true, compute a Jacobian time derivative that is
                only guaranteed to be equivalent after multiplication by ``qd``.

        When ``convective_only_jd`` is true, the returned derivative is only
        guaranteed to be equivalent after multiplication by ``qd``. This is
        sufficient for the forward-dynamics ``C(q, qd) @ qd`` path and avoids
        assembling terms that vanish in that product.
        """
        xi = self.strain(q).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 3)
        B_segments = self.B_xi.reshape(self.num_segments, 3, self.num_dofs)

        Xs_scaled, Ws_scaled = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_scaled - self.L_cum[:-1, None]

        g0 = poses.planar_pose_to_transform(jnp.asarray(self.base_pose, dtype=xi.dtype))

        def scan_fk(g_base: Array, i: Array) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
            g_tip = g_base @ se2.exp(L_i * xi_i, eps=self.global_eps)
            return g_tip, g_base

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, g_bases = lax.scan(scan_fk, g0, indices)

        def segment_poses(g_base_i: Array, xi_i: Array, s_local_i: Array) -> Array:
            def pose_at_s(s_local_ij: Array) -> Array:
                g_rel = se2.exp(s_local_ij * xi_i, eps=self.global_eps)
                return g_base_i @ g_rel

            return vmap(pose_at_s)(s_local_i)

        g_ps = vmap(segment_poses)(g_bases, xi, s_local)

        J_tips, Jd_tips = self._active_J_Jd_local_tips_from_strain(
            xi, xid, B_segments, convective_only_jd=convective_only_jd
        )
        zeros_tip = jnp.zeros_like(J_tips[:1])
        J_bases = jnp.concatenate([zeros_tip, J_tips[:-1]], axis=0)
        Jd_bases = jnp.concatenate([zeros_tip, Jd_tips[:-1]], axis=0)

        def segment_jacobians(
            xi_i: Array,
            xid_i: Array,
            B_i: Array,
            J_base_i: Array,
            Jd_base_i: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array]:
            def jacobian_at_s(s_local_ij: Array) -> tuple[Array, Array]:
                Ad_inv = constant_strain.adjoint_inverse_se2(
                    xi_i, s_local_ij, eps=self.global_eps
                )
                T = constant_strain.tangent_se2(xi_i, s_local_ij, eps=self.tangent_eps)
                Td = constant_strain.tangent_derivative_se2(
                    xi_i, xid_i, s_local_ij, eps=self.tangent_eps
                )

                Ad_inv_T = Ad_inv @ T
                J_segment = Ad_inv_T @ B_i
                J_next = Ad_inv @ J_base_i + J_segment

                eta = Ad_inv_T @ xid_i
                Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv

                if convective_only_jd:
                    Jd_segment = (Ad_inv @ Td) @ B_i
                else:
                    Jd_segment = (Ad_inv_dot @ T + Ad_inv @ Td) @ B_i
                Jd_next = Ad_inv @ Jd_base_i + Ad_inv_dot @ J_base_i + Jd_segment

                return J_next, Jd_next

            return vmap(jacobian_at_s)(s_local_i)

        J_ps, Jd_ps = vmap(segment_jacobians)(
            xi, xid, B_segments, J_bases, Jd_bases, s_local
        )

        return Ws_scaled, g_ps, J_ps, Jd_ps

    @eqx.filter_jit
    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Assemble forward-dynamics terms in active generalized coordinates.

        The coordinates and returned generalized forces correspond only to the
        active strain components selected by the strain basis.

        Args:
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.

        Returns:
            Tuple ``(B, Cqd, G)``. ``B`` is the active-coordinate inertia
            matrix with shape ``(self.num_dofs, self.num_dofs)``. ``Cqd`` is
            the active Coriolis/centrifugal force vector with shape
            ``(self.num_dofs,)``. ``G`` is the active generalized gravity
            vector with shape ``(self.num_dofs,)``.
        """
        Ws_scaled, g_ps, J_ps, Jd_ps = self._integration_kinematics(
            q, qd, convective_only_jd=True
        )
        num_quad = self.num_gauss_points

        def dynamical_terms_i(i: Array) -> tuple[Array, Array, Array]:
            M_i = self.M_segments[i]

            def dynamical_terms_ij(j: Array) -> tuple[Array, Array, Array]:
                Ws_ij = Ws_scaled[i, j]
                g_ij = g_ps[i, j]
                J_ij = J_ps[i, j]
                Jd_ij = Jd_ps[i, j]

                Ad_g_inv_ij = se2.adjoint_inverse(g_ij)
                eta_ij = J_ij @ qd

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                Cqd_ij = Ws_ij * (
                    J_ij.T @ (M_i @ (Jd_ij @ qd) + se2.coadjoint(eta_ij) @ M_i @ eta_ij)
                )
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, Cqd_ij, G_ij

            return vmap(dynamical_terms_ij)(jnp.arange(num_quad))

        B_blocks_tot, Cqd_blocks_tot, G_blocks_tot = vmap(dynamical_terms_i)(
            jnp.arange(self.num_segments)
        )

        B = jnp.sum(B_blocks_tot, axis=(0, 1))
        Cqd = jnp.sum(Cqd_blocks_tot, axis=(0, 1))
        G = jnp.sum(G_blocks_tot, axis=(0, 1))

        return B, Cqd, G

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.

        Returns:
            yd (Array): Time derivative of the state vector.
        """
        # Split the state vector into configuration and velocity
        q, qd = jnp.split(y, 2)

        # split the actuation arguments if provided
        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u = actuation_args[0]
            tau_ext = None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        if u is None:
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[-1],))

        B, Cqd, G = self.dynamics_terms(q, qd)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        rhs = tau_u + tau_ext - Cqd - G - tau_el - self.damping_matrix(q) @ qd
        qdd = jnp.linalg.solve(B, rhs)

        yd = jnp.concatenate([qd, qdd])

        return yd
