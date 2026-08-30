__all__ = ["PlanarPCS"]


from typing import Any, Self

import equinox as eqx
from jax import Array, jvp, lax, vmap
from jax import numpy as jnp

from soromox.actuation.core import Actuator, PassiveElement
from soromox.actuation.threadlike import (
    BaseThreadlikeRoutingParams,
    ThreadlikeRouting,
)
from soromox.systems.components import (
    ContinuumLinkParams,
    CrossSectionGeometry,
    CrossSectionParams,
    IsotropicMaterialParams,
    LinkSpec,
)
from soromox.systems.execution import (
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    PCS_DYNAMICS,
    PCS_KINEMATICS,
    PLANAR_PCS_ACTUATION,
    ExecutionBackend,
    PCSBackendParams,
    dispatch_actuation_force,
    dispatch_actuation_matrix,
    dispatch_dynamics_terms,
    dispatch_kinematics,
    dispatch_kinematics_abscissa_batched,
    evaluate_forward_dynamics,
    uses_warp_kinematics,
)
from soromox.systems.pcs.params import PlanarPCSParams
from soromox.systems.pcs.structures import PlanarPCSStructure
from soromox.systems.soft_robot import SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.dof import build_active_dof_basis
from soromox.utils.geometry import poses
from soromox.utils.integration import (
    gauss_quadrature,
    scale_gaussian_quadrature,
    scale_interior_gaussian_quadrature,
)
from soromox.utils.lie_algebra import se2
from soromox.utils.lie_algebra.constant_strain import se2 as constant_strain_se2
from soromox.utils.numerics import safe_divide, safe_norm


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
        L, r, E, G, rho: Physical properties of each segment. ``r`` is the
            radius of the assumed solid circular cross-section.
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (3 * num_segments, num_active_strains).
        scale_rotational_basis_by_length: If True, rotational strain-basis rows
            are divided by their segment length.
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.
        backend_params: Optional tuning parameters for the PCS execution
            backend. The default uses 128 threads per planar Warp block.

    Notes:
    -----
    - The material-frame local x-axis is the rod's longitudinal axis (the
      undeformed backbone tangent). This convention is independent of the base
      pose, which may orient the local x-axis arbitrarily in the inertial frame.
    - The strain vector is composed of 3 components per segment:
      [kappa_z, sigma_x, sigma_y]. The default straight, unstretched reference
      strain is [0, 1, 0]. Thus, kappa_z is the out-of-plane bending strain,
      sigma_x is axial stretch along local x, and sigma_y is transverse shear.
    - Every segment is assumed to have a solid circular cross-section. Its
      radius determines the area and second moment used by the mass, material
      damping, and stiffness matrices.
    - ``material_damping_coefficient`` is a viscosity-like modulus in Pa*s
      (N*s/m^2). The assembled damping matrix also contains section geometry
      and segment-length factors, so its entries do not share one blanket unit.

    References:
        Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete Cosserat
        Approach for Multisection Soft Manipulator Dynamics. IEEE Transactions on
        Robotics, 34(6), 1518-1533.

    """

    # These fields are optional because PlanarHSA reuses this layout and
    # propagation infrastructure but has a distinct multi-rod material schema;
    # leaving PCS-only material fields unset avoids fabricating PCS parameters.
    params: PlanarPCSParams | None = eqx.field(default=None)

    # Robot parameters
    g: Array | None = eqx.field(default=None)  # Gravitational acceleration vector

    L: Array | None = eqx.field(default=None)  # Length of the segments
    L_cum: Array | None = eqx.field(default=None)  # Cumulative length of the segments
    r: Array | None = eqx.field(default=None)  # Radius of the segments
    rho: Array | None = eqx.field(default=None)

    num_segments: int = eqx.field(static=True, default=0)
    num_gauss_points: int = eqx.field(static=True, default=0)
    num_integration_points: int = eqx.field(static=True, default=0)
    num_strains: int = eqx.field(
        static=True, default=0
    )  # Number of strains (3 * num_segments)
    backend: ExecutionBackend = eqx.field(static=True, default="auto")
    backend_params: PCSBackendParams = eqx.field(
        static=True,
        default=PCSBackendParams(warp_block_dim=DEFAULT_PLANAR_PCS_BLOCK_DIM),
    )
    _segment_dof_ends: tuple[int, ...] = eqx.field(static=True, default=())
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)

    xi_ref: Array | None = eqx.field(default=None)  # Reference configuration strain
    B_xi_unscaled: Array | None = eqx.field(
        default=None
    )  # Unscaled strain basis matrix
    B_xi: Array | None = eqx.field(default=None)  # Strain basis matrix
    num_active_strains: Array | None = eqx.field(
        default=None
    )  # Number of selected strains

    integration_points: Array | None = eqx.field(default=None)
    integration_weights: Array | None = eqx.field(default=None)
    M_segments: Array | None = eqx.field(
        default=None
    )  # Cached per-segment mass matrices
    K_full: Array | None = eqx.field(default=None)  # Cached full stiffness matrix
    K_active: Array | None = eqx.field(
        default=None
    )  # Cached active-coordinate stiffness matrix
    D_full: Array | None = eqx.field(default=None)  # Cached full damping matrix
    D_active: Array | None = eqx.field(
        default=None
    )  # Cached active-coordinate damping matrix
    young_stiffness_operator: Array | None = eqx.field(default=None)
    shear_stiffness_operator: Array | None = eqx.field(default=None)
    material_damping_operator: Array | None = eqx.field(default=None)
    active_strain_indices: Array | None = eqx.field(default=None)
    active_strain_scales: Array | None = eqx.field(default=None)
    active_dof_ends: Array | None = eqx.field(default=None)
    dynamics_local_points: Array | None = eqx.field(default=None)
    weighted_mass_diagonals: Array | None = eqx.field(default=None)
    inertia_upper_rows: Array | None = eqx.field(default=None)
    inertia_upper_columns: Array | None = eqx.field(default=None)
    gravity_base: Array | None = eqx.field(default=None)

    @staticmethod
    def params_from_links(
        links: list[LinkSpec] | tuple[LinkSpec, ...],
        *,
        gravity: Array | None = None,
    ) -> PlanarPCSParams:
        """Build planar PCS parameters from shared link specifications.

        Args:
            links: Non-empty sequence of constant, circular link
                specifications. Each reference strain must have shape ``(3,)``.
            gravity: Optional planar gravity vector with shape ``(2,)``.

        Returns:
            Validated :class:`PlanarPCSParams` with canonical per-link
            stiffness and damping arrays of shape ``(num_links, 3, 3)``.

        Raises:
            ValueError: If no links are supplied; a link is not constant and
                circular; a reference strain has the wrong shape; a material
                source is incomplete; or an explicit matrix is not ``(3, 3)``.
        """
        if not links:
            raise ValueError("PlanarPCS requires at least one link.")
        specs = tuple(links)
        for index, spec in enumerate(specs):
            if spec.cross_section_geometry != CrossSectionGeometry.CIRCULAR:
                raise ValueError(
                    f"PlanarPCS link {index} must have a circular section."
                )
            if spec.cross_section_profile_types != ("constant",):
                raise ValueError(f"PlanarPCS link {index} must have a constant radius.")
            if jnp.asarray(spec.reference_strain).shape != (3,):
                raise ValueError(
                    f"PlanarPCS link {index} reference_strain must have shape (3,)."
                )
        length = jnp.asarray([spec.length for spec in specs])
        density = jnp.asarray([spec.density for spec in specs])
        radius = jnp.asarray([spec.cross_section_coefficients[0] for spec in specs])
        reference_strain = jnp.asarray(
            [spec.reference_strain for spec in specs], dtype=float
        )
        area = jnp.pi * radius**2
        moment = jnp.pi * radius**4 / 4.0
        zeros = jnp.zeros_like(length)
        young_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack([moment, area, zeros], axis=1)
        )
        shear_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack([zeros, zeros, area], axis=1)
        )
        damping_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack([3.0 * moment, 3.0 * area, area], axis=1)
        )
        stiffness_items = []
        damping_items = []
        for index, spec in enumerate(specs):
            if spec.stiffness is not None:
                stiffness = jnp.asarray(spec.stiffness)
                if stiffness.shape != (3, 3):
                    raise ValueError(
                        f"PlanarPCS link {index} stiffness must have shape (3, 3)."
                    )
            else:
                stiffness = (
                    spec.young_modulus * young_operator[index]
                    + spec._resolved_shear_modulus() * shear_operator[index]
                )
            if spec.damping is not None:
                damping = jnp.asarray(spec.damping)
                if damping.shape != (3, 3):
                    raise ValueError(
                        f"PlanarPCS link {index} damping must have shape (3, 3)."
                    )
            elif spec.material_damping_coefficient is not None:
                damping = spec.material_damping_coefficient * damping_operator[index]
            else:
                damping = jnp.zeros_like(damping_operator[index])
            stiffness_items.append(stiffness)
            damping_items.append(damping)
        return PlanarPCSParams(
            gravity=gravity,
            link=ContinuumLinkParams(
                length=length,
                density=density,
                reference_strain=reference_strain,
                cross_section=CrossSectionParams(coefficients=radius[:, None]),
                stiffness=jnp.stack(stiffness_items),
                damping=jnp.stack(damping_items),
            ),
        )

    @classmethod
    def from_links(
        cls,
        links: list[LinkSpec] | tuple[LinkSpec, ...],
        *,
        structure: PlanarPCSStructure | None = None,
        gravity: Array | None = None,
        base_pose: Array | None = None,
        **kwargs: Any,
    ) -> "PlanarPCS":
        """Construct a planar PCS model from shared link specifications.

        Args:
            links: Non-empty sequence of constant, circular link
                specifications.
            structure: Optional static planar PCS quadrature, strain-selection,
                and rotational-scaling configuration. Defaults to
                :class:`PlanarPCSStructure`.
            gravity: Optional planar gravity vector with shape ``(2,)``.
            base_pose: Optional planar base pose ``[theta, x, y]`` with shape
                ``(3,)``.
            **kwargs: Additional keyword arguments forwarded to the PlanarPCS
                constructor, such as actuators or passive elements.

        Returns:
            A fully initialized planar PCS model.

        Raises:
            TypeError: If a structure or constructor argument has an invalid
                type.
            ValueError: If a link specification, static structure, parameter
                array, or attached component is invalid.
        """
        return cls(
            params=cls.params_from_links(links, gravity=gravity),
            structure=structure,
            base_pose=base_pose,
            **kwargs,
        )

    def __init__(
        self,
        params: PlanarPCSParams,
        structure: PlanarPCSStructure | None = None,
        actuators: Actuator | tuple[Actuator, ...] | None = None,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
        backend: ExecutionBackend = "auto",
        backend_params: PCSBackendParams | None = None,
        base_pose: Array | None = None,
        floating_base: bool = False,
        **kwargs: Any,
    ):
        """Initialize a planar PCS model from typed parameters.

        Args:
            params: Canonical dynamic link, gravity, and base-pose parameters.
            structure: Optional static quadrature, active-strain, and
                rotational-scaling configuration.
            actuators: Optional actuator or tuple of actuators attached to the
                model.
            passive_elements: Optional passive element or tuple of passive
                elements. Pass ``None`` or an empty tuple to disable them.
            backend: Preferred execution backend for accelerated methods.
            backend_params: Optional tuning parameters for accelerated PCS
                execution. Defaults to a 128-thread persistent Warp block.
            **kwargs: Additional keyword arguments forwarded to
                :class:`BaseContinuumSoftRobot`.

        Raises:
            TypeError: If ``params``, a structure field, or an attached
                component has an invalid type.
            ValueError: If parameter shapes or values, the structure, or an
                attached component are invalid.
        """
        if not isinstance(params, PlanarPCSParams):
            raise TypeError("params must be a PlanarPCSParams instance.")
        params.validate()
        if backend not in ("auto", "jax", "warp"):
            raise ValueError(
                f"backend must be one of 'auto', 'jax', or 'warp', got {backend!r}."
            )
        resolved_base_pose = None if floating_base else base_pose
        super().__init__(
            base_pose=resolved_base_pose, floating_base=floating_base, **kwargs
        )
        if structure is None:
            structure = PlanarPCSStructure()
        self.params = params
        self.backend = backend
        if backend_params is None:
            backend_params = PCSBackendParams(
                warp_block_dim=DEFAULT_PLANAR_PCS_BLOCK_DIM
            )
        if not isinstance(backend_params, PCSBackendParams):
            raise TypeError("backend_params must be a PCSBackendParams instance.")
        self.backend_params = backend_params
        self.scale_rotational_basis_by_length = bool(
            structure.scale_rotational_basis_by_length
        )

        # Number of segments
        num_segments = int(params.link.length.shape[0])
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
        self.B_xi_unscaled = build_active_dof_basis(strain_selector)
        self.B_xi = self._scaled_strain_basis(self.B_xi_unscaled)

        self.num_active_strains = jnp.sum(strain_selector)
        self.num_internal_dofs = int(self.num_active_strains.item())
        active_dofs_per_segment = jnp.sum(
            strain_selector.reshape(self.num_segments, 3), axis=1
        )
        self._segment_dof_ends = tuple(
            int(value) for value in jnp.cumsum(active_dofs_per_segment).tolist()
        )

        reference_strain = jnp.asarray(params.link.reference_strain, dtype=jnp.float64)
        if reference_strain.size != self.num_strains:
            raise ValueError(
                "reference_strain must have "
                f"{self.num_strains} elements, got {reference_strain.size}."
            )
        self.xi_ref = reference_strain.reshape(self.num_strains)

        self._configure_actuation(actuators, passive_elements)

        self.precompute()

    @property
    def is_planar(self) -> bool:
        """Return whether the system is planar.

        Returns:
            Always ``True``.
        """
        return True

    @property
    def segment_length(self) -> Array:
        """Return the per-segment backbone lengths.

        Returns:
            Array with shape ``(num_segments,)``.
        """
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
        """Evaluate the circular cross-section at a backbone coordinate.

        Args:
            q: Generalized coordinates. PCS cross-section geometry is
                configuration-independent, but ``q`` is accepted for the common
                continuum-robot interface.
            s: Scalar global backbone coordinate.

        Returns:
            A tuple containing the circular :class:`CrossSectionGeometry` tag
            and a one-element array holding the selected segment radius.

        Raises:
            ValueError: If ``s`` cannot be assigned to a segment.
        """
        segment_idx, _ = self.classify_segment(s)
        radius = jnp.asarray(self.r)[segment_idx]
        tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
        return tag, jnp.array([radius])

    def _set_params(self, params: PlanarPCSParams) -> None:
        """Set cached runtime arrays from typed parameters."""
        # Gravitational acceleration vector
        g = params.gravity
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 2:
            raise ValueError(f"gravity must be a vector of shape (2,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(1), g]
        )  # Add a zero for the orientation angle

        # Lengths of the segments
        L = params.link.length
        L = jnp.asarray(L, dtype=jnp.float64)
        if L.shape != (self.num_segments,):
            raise ValueError(
                f"length must have shape ({self.num_segments},), got {L.shape}"
            )
        self.L = L

        L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        self.L_cum = L_cum

        # Radius of the segments
        r = jnp.asarray(params.link.cross_section.coefficients[:, 0], dtype=jnp.float64)
        self.r = r

        # Densities of the segments
        rho = params.link.density
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if rho.shape != (self.num_segments,):
            raise ValueError(
                f"density must have shape ({self.num_segments},), got {rho.shape}"
            )
        self.rho = rho

    def _compute_damping_full_matrix(self) -> Array:
        """Assemble canonical per-link generalized damping blocks."""
        return blk_diag(self._current_body_params().link.damping)

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
        params.validate_for_update()
        if params.link.length.shape != current_params.link.length.shape:
            raise ValueError(
                "length shape changes the model structure; construct a new PlanarPCS."
            )
        if (
            params.link.reference_strain.shape
            != current_params.link.reference_strain.shape
        ):
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PlanarPCS."
            )

        gravity = jnp.asarray(params.gravity, dtype=jnp.float64)
        segment_lengths = jnp.asarray(params.link.length, dtype=jnp.float64)
        radius = jnp.asarray(
            params.link.cross_section.coefficients[:, 0], dtype=jnp.float64
        )
        density = jnp.asarray(params.link.density, dtype=jnp.float64)
        reference_strain = jnp.asarray(params.link.reference_strain, dtype=jnp.float64)

        updated_self = eqx.tree_at(
            lambda m: (
                m.params,
                m.g,
                m.L,
                m.L_cum,
                m.r,
                m.rho,
                m.xi_ref,
            ),
            self,
            (
                params if stored_params is None else stored_params,
                jnp.concatenate([jnp.zeros(1, dtype=gravity.dtype), gravity]),
                segment_lengths,
                jnp.cumsum(
                    jnp.concatenate(
                        [jnp.zeros(1, dtype=segment_lengths.dtype), segment_lengths]
                    )
                ),
                radius,
                density,
                reference_strain.reshape(self.num_strains),
            ),
        )
        return updated_self._with_refreshed_precomputed_matrices()

    def with_params(self, params: PlanarPCSParams) -> "PlanarPCS":
        """Return a model copy using a complete parameter PyTree.

        Args:
            params: Complete replacement parameters with the same number of
                segments and reference-strain layout as this model.

        Returns:
            A new PlanarPCS model with refreshed geometry, mass,
            material-response, stiffness, and damping caches. The original
            model is unchanged.

        Raises:
            TypeError: If ``params`` is not a :class:`PlanarPCSParams`.
            ValueError: If the replacement is invalid or changes the static
                segment or strain layout.
        """
        return self._with_planar_pcs_params(params)

    def update_params(self, **updates: Any) -> "PlanarPCS":
        """Return a copy with selected top-level parameter fields replaced.

        Args:
            **updates: Fields of :class:`PlanarPCSParams` to replace, typically
                ``link`` or ``gravity``.

        Returns:
            A new validated PlanarPCS model containing the replacements.

        Raises:
            TypeError: If an unknown field is supplied or a replacement has an
                invalid type.
            ValueError: If the result is invalid or changes the static segment
                or strain layout.
        """
        if (
            "link" in updates
            and jnp.asarray(updates["link"].length).shape
            != self.params.link.length.shape
        ):
            raise ValueError(
                "length shape changes the model structure; construct a new PlanarPCS."
            )
        if (
            "link" in updates
            and jnp.asarray(updates["link"].reference_strain).shape
            != self.params.link.reference_strain.shape
        ):
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PlanarPCS."
            )
        return self.with_params(self.params.replace(**updates))

    def update_link_params(self, **updates: Any) -> "PlanarPCS":
        """Return a copy with selected continuum-link fields replaced.

        Args:
            **updates: Fields of :class:`ContinuumLinkParams` to replace, such
                as ``length``, ``density``, ``reference_strain``,
                ``cross_section``, ``stiffness``, or ``damping``.

        Returns:
            A new validated PlanarPCS model with refreshed dependent caches.

        Raises:
            TypeError: If an unknown link field is supplied.
            ValueError: If a replacement is invalid or changes the static
                segment or strain layout.
        """
        return self.with_params(
            self.params.replace(link=self.params.link.replace(**updates))
        )

    def link_matrices_from_material(
        self, material: IsotropicMaterialParams
    ) -> tuple[Array, Array]:
        """Map isotropic material variables to canonical planar link matrices.

        Args:
            material: Scalar or per-segment Young's modulus, either shear
                modulus or Poisson's ratio, and optional material damping
                coefficient. Scalars are broadcast over all segments.

        Returns:
            A tuple ``(stiffness, damping)`` whose arrays both have shape
            ``(num_segments, 3, 3)``.

        Raises:
            ValueError: If a material field is not scalar or does not have
                shape ``(num_segments,)``.
        """
        material = material.broadcast(self.num_segments)
        shear_modulus = material._resolved_shear_modulus()
        stiffness = (
            material.young_modulus[:, None, None] * self.young_stiffness_operator
            + shear_modulus[:, None, None] * self.shear_stiffness_operator
        )
        if material.material_damping_coefficient is None:
            damping = jnp.zeros_like(self.material_damping_operator)
        else:
            damping = (
                material.material_damping_coefficient[:, None, None]
                * self.material_damping_operator
            )
        return stiffness, damping

    def with_isotropic_material(self, material: IsotropicMaterialParams) -> Self:
        """Return a copy whose link matrices are built from isotropic material.

        The supplied material PyTree remains caller-owned and is not stored on
        the model.

        Args:
            material: Scalar or per-segment isotropic material variables.

        Returns:
            A new PlanarPCS model containing the generated canonical link
            stiffness and damping matrices.

        Raises:
            ValueError: If a material field cannot be broadcast to one value
                per segment.
        """
        stiffness, damping = self.link_matrices_from_material(material)
        return self.update_link_params(stiffness=stiffness, damping=damping)

    def _precomputed_matrices(self) -> tuple[Array, Array, Array, Array, Array]:
        """Compute state-independent matrices cached by the model."""
        M_segments = vmap(self._compute_local_mass_matrix)(
            jnp.arange(self.num_segments)
        )
        K_full = self._compute_stiffness_full_matrix()
        K_active = self.B_xi.T @ K_full @ self.B_xi
        D_full = self._compute_damping_full_matrix()
        D_active = self.B_xi.T @ D_full @ self.B_xi
        return M_segments, K_full, K_active, D_full, D_active

    def _dynamics_runtime_arrays(
        self, M_segments: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """Build compact state-independent planar dynamics operands.

        The arrays encode the active strain selection, quadrature grid,
        diagonal mass action, packed inertia layout, and base-frame gravity.
        They are cached on the model and shared by the JAX and Warp dynamics
        implementations, so backend execution does not repeat this structural
        preprocessing.

        Args:
            M_segments: Local planar spatial-inertia matrices with shape
                ``(num_segments, 3, 3)``.

        Returns:
            A tuple containing, in order:

            - active strain indices, shape ``(num_segments, 3)``;
            - active strain scales, shape ``(num_segments, 3)``;
            - cumulative active-DOF ends, shape ``(num_segments,)``;
            - segment-local operator points, shape
              ``(num_segments, num_gauss_points + 1)``;
            - quadrature-weighted mass diagonals, shape
              ``(num_segments, num_gauss_points, 3)``;
            - packed upper-inertia row indices;
            - matching packed upper-inertia column indices; and
            - gravity expressed in the base frame, shape ``(3,)``.
        """

        if self.num_internal_dofs == 0:
            active_indices = -jnp.ones((self.num_segments, 3), dtype=jnp.int32)
            active_scales = jnp.zeros((self.num_segments, 3), dtype=self.B_xi.dtype)
        else:
            basis = self.B_xi.reshape(self.num_segments, 3, self.num_internal_dofs)
            active_mask = jnp.any(basis != 0.0, axis=-1)
            active_indices = jnp.argmax(jnp.abs(basis), axis=-1).astype(jnp.int32)
            active_indices = jnp.where(active_mask, active_indices, -1)
            safe_indices = jnp.maximum(active_indices, 0)[..., None]
            active_scales = jnp.take_along_axis(basis, safe_indices, axis=-1)[..., 0]
            active_scales = jnp.where(active_mask, active_scales, 0.0)

        points, weights = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        local_points = points - self.L_cum[:-1, None]
        operator_points = jnp.concatenate([local_points, self.L[:, None]], axis=1)
        weighted_masses = (
            weights[..., None]
            * jnp.diagonal(M_segments, axis1=-2, axis2=-1)[:, None, :]
        )
        assembly_dimension = (
            self.num_velocities if self.floating_base else self.num_internal_dofs
        )
        upper_rows = jnp.asarray(
            [row for column in range(assembly_dimension) for row in range(column + 1)],
            dtype=jnp.int32,
        )
        upper_columns = jnp.asarray(
            [column for column in range(assembly_dimension) for _ in range(column + 1)],
            dtype=jnp.int32,
        )
        g0 = poses.planar_pose_to_transform(
            jnp.asarray(self._kinematic_base_pose, dtype=self.g.dtype)
        )
        gravity_base = se2.adjoint_inverse(g0) @ self.g
        return (
            active_indices,
            active_scales,
            jnp.asarray(self._segment_dof_ends, dtype=jnp.int32),
            operator_points,
            weighted_masses,
            upper_rows,
            upper_columns,
            gravity_base,
        )

    def precompute(self) -> None:
        """Refresh state-independent matrices cached by the model.

        Returns:
            ``None``. Material unit-response operators and the segment, full,
            and active-coordinate matrices are replaced in place.
        """
        object.__setattr__(self, "B_xi", self._scaled_strain_basis(self.B_xi_unscaled))
        young_operator, shear_operator, damping_operator = self._material_operators()
        object.__setattr__(self, "young_stiffness_operator", young_operator)
        object.__setattr__(self, "shear_stiffness_operator", shear_operator)
        object.__setattr__(self, "material_damping_operator", damping_operator)
        (
            M_segments,
            K_full,
            K_active,
            D_full,
            D_active,
        ) = self._precomputed_matrices()
        object.__setattr__(self, "M_segments", M_segments)
        object.__setattr__(self, "K_full", K_full)
        object.__setattr__(self, "K_active", K_active)
        object.__setattr__(self, "D_full", D_full)
        object.__setattr__(self, "D_active", D_active)
        runtime_arrays = self._dynamics_runtime_arrays(M_segments)
        for name, value in zip(
            (
                "active_strain_indices",
                "active_strain_scales",
                "active_dof_ends",
                "dynamics_local_points",
                "weighted_mass_diagonals",
                "inertia_upper_rows",
                "inertia_upper_columns",
                "gravity_base",
            ),
            runtime_arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def _with_refreshed_precomputed_matrices(self) -> "PlanarPCS":
        """Return a copy with cached state-independent matrices refreshed."""
        B_xi = self._scaled_strain_basis(self.B_xi_unscaled)
        updated_self = eqx.tree_at(lambda m: m.B_xi, self, B_xi)
        young_operator, shear_operator, damping_operator = (
            updated_self._material_operators()
        )
        updated_self = eqx.tree_at(
            lambda m: (
                m.young_stiffness_operator,
                m.shear_stiffness_operator,
                m.material_damping_operator,
            ),
            updated_self,
            (young_operator, shear_operator, damping_operator),
        )
        (
            M_segments,
            K_full,
            K_active,
            D_full,
            D_active,
        ) = updated_self._precomputed_matrices()
        updated_self = eqx.tree_at(
            lambda m: (m.M_segments, m.K_full, m.K_active, m.D_full, m.D_active),
            updated_self,
            (M_segments, K_full, K_active, D_full, D_active),
        )
        runtime_arrays = updated_self._dynamics_runtime_arrays(M_segments)
        return eqx.tree_at(
            lambda m: (
                m.active_strain_indices,
                m.active_strain_scales,
                m.active_dof_ends,
                m.dynamics_local_points,
                m.weighted_mass_diagonals,
                m.inertia_upper_rows,
                m.inertia_upper_columns,
                m.gravity_base,
            ),
            updated_self,
            runtime_arrays,
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

        Components use the rod material frame, whose local x-axis is the
        longitudinal backbone direction.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            xi (Array): strain vector of shape (num_active_strains,)
        """
        xi = self.B_xi @ q + self.xi_ref

        return xi

    @eqx.filter_jit
    def forward_kinematics(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Compute a planar pose at one curvilinear abscissa.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s: Scalar curvilinear abscissa.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            Planar pose ``[theta, x, y]`` with shape ``(3,)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics(
                    self,
                    q,
                    s,
                    operation="pose",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_pose = self._fixed_base_robot_at_pose().forward_kinematics(
                q_internal, s, backend=backend
            )
            return self._compose_runtime_base_poses(q, relative_pose[None])[0]
        return dispatch_kinematics(
            self,
            q,
            s,
            operation="pose",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def forward_kinematics_abscissa_batched(
        self,
        q: Array,
        s_ps: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Compute planar poses at a batch of curvilinear abscissae.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s_ps: Curvilinear abscissae with shape ``(N,)``.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            Planar poses with shape ``(N, 3)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics_abscissa_batched(
                    self,
                    q,
                    s_ps,
                    operation="pose",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_poses = (
                self._fixed_base_robot_at_pose().forward_kinematics_abscissa_batched(
                    q_internal, s_ps, backend=backend
                )
            )
            return self._compose_runtime_base_poses(q, relative_poses)
        return dispatch_kinematics_abscissa_batched(
            self,
            q,
            s_ps,
            operation="pose",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

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
        xi = self.strain(q).reshape(self.num_segments, 3)
        segment_idx, s_local = self.classify_segment(s)
        chi_0 = self._base_planar_pose(xi.dtype)

        def integrate_segment(chi_prev: Array, i: Array) -> tuple[Array, Array]:
            arc_len = jnp.where(i == segment_idx, s_local, self.L[i])
            chi_next = self._integrate_planar_pose(chi_prev, xi[i], arc_len)
            return chi_next, chi_next

        _, chi_list = lax.scan(
            f=integrate_segment,
            init=chi_0,
            xs=jnp.arange(self.num_segments),
        )
        return chi_list[segment_idx]

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
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            relative_tips = self._fixed_base_robot_at_pose().forward_kinematics_tips(
                q_internal
            )
            return self._compose_runtime_base_poses(q, relative_tips)
        xi = self.strain(q).reshape(self.num_segments, 3)

        chi_base = self._base_planar_pose(xi.dtype)

        def integrate_segment(chi_prev: Array, i: Array) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
            chi_tip = self._integrate_planar_pose(chi_prev, xi_i, L_i)

            return chi_tip, chi_tip

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, chi_tips = lax.scan(integrate_segment, chi_base, indices)

        return chi_tips

    @eqx.filter_jit
    def _forward_kinematics_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
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
            return self._integrate_planar_pose(chi_prev, xi_i, arc_len)

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
                            The first segment's relative pose is computed with
                            respect to the resolved fixed mounting.
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
            transform = poses.planar_pose_to_transform(chi_rel_i)
            integrated_strain = se2.log(transform, eps=self.global_eps)
            return safe_divide(integrated_strain, s_i, self.global_eps)

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

            Ad_inv, T = constant_strain_se2._operators(
                xi_i, L_i, self.global_eps, self.tangent_eps
            )

            J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

            return J_next, J_next

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, J_target_tips = lax.scan(scan_body, zeros, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_target_tips)

        return J_local_tips

    @eqx.filter_jit
    def _J_local_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
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
            Ad_inv, T = constant_strain_se2._operators(
                xi_i, arc_len, self.global_eps, self.tangent_eps
            )

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
            Ad_inv, T = constant_strain_se2._operators(
                xi_i, arc_len, self.global_eps, self.global_eps
            )

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
        return constant_strain_se2._operators(
            xi_i, arc_len, self.global_eps, self.global_eps
        )

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
        dT_ds = constant_strain_se2.adjoint(xi_i, arc_len, eps=self.global_eps)
        return Ad_inv, T, dAd_inv_ds, dT_ds

    def _integrate_planar_pose(self, chi: Array, xi_i: Array, arc_len: Array) -> Array:
        """Integrate one constant-strain slice into an absolute planar pose."""
        theta = chi[0]
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        p_step = se2._exp_translation(arc_len * xi_i, eps=self.global_eps)
        theta_next = theta + arc_len * xi_i[0]
        p_next = chi[1:] + jnp.stack(
            [
                cos_theta * p_step[0] - sin_theta * p_step[1],
                sin_theta * p_step[0] + cos_theta * p_step[1],
            ]
        )
        return jnp.concatenate([theta_next[None], p_next])

    def _base_planar_pose(self, dtype: jnp.dtype) -> Array:
        """
        Return the absolute base pose used by PlanarPCS scans.

        Args:
            dtype: Desired floating-point dtype of the returned array.

        Returns:
            Planar base pose ``[theta0, x0, y0]`` with shape ``(3,)``.
        """
        return jnp.asarray(self._kinematic_base_pose, dtype=dtype)

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
                chi_next = self._integrate_planar_pose(chi_prev, xi_i, arc_len)
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
                chi_next = self._integrate_planar_pose(chi_prev, xi_i, arc_len)
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

                Ad_inv, T, Td = constant_strain_se2._operators(
                    xi_i,
                    arc_len,
                    self.global_eps,
                    self.global_eps,
                    xid_i,
                )
                chi_next = self._integrate_planar_pose(chi_prev, xi_i, arc_len)
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
        """Compute a body-frame Jacobian and its arc-length derivative.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s: Scalar global backbone coordinate.

        Returns:
            A tuple ``(J, J_s)`` containing the body-frame Jacobian and its
            derivative with respect to global arc length. Both arrays have
            shape ``(3, num_active_strains)``.
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
            Ad_inv, T = constant_strain_se2._operators(
                xi_i, arc_len, self.global_eps, self.global_eps
            )

            return self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

        def integrate_segment_arc_length_derivative(
            J_base: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv, T = constant_strain_se2._operators(
                xi_i, arc_len, self.global_eps, self.global_eps
            )
            dAd_inv_ds = -se2.small_adjoint(xi_i) @ Ad_inv
            dT_ds = constant_strain_se2.adjoint(xi_i, arc_len, eps=self.global_eps)

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
        """Compute the body-frame Jacobian derivative with respect to arc length.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s: Scalar global backbone coordinate.

        Returns:
            Arc-length derivative with shape
            ``(3, num_active_strains)``.
        """
        _, Js = self.jacobian_and_arc_length_derivative_bodyframe(q, s)
        return Js

    @eqx.filter_jit
    def jacobian_bodyframe_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a batch of points s_ps along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
        """
        J_local_ps_full = self._J_local_abscissa_batched(
            q, s_ps
        )  # shape (N, 3, num_strains)
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
    def _jacobian_inertialframe(self, q: Array, s: Array) -> Array:
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
    def jacobian_inertialframe(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Compute an inertial Jacobian at one curvilinear abscissa.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s: Scalar curvilinear abscissa.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            Inertial Jacobian with shape ``(3, D)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics(
                    self,
                    q,
                    s,
                    operation="jacobian",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_pose, jacobian_internal = (
                self._fixed_base_robot_at_pose().forward_kinematics_and_jacobian_inertialframe(
                    q_internal, s, backend=backend
                )
            )
            relative_transform = poses.planar_pose_to_transform(relative_pose)
            return self._floating_jacobians_from_relative_inertial(
                q, relative_transform, jacobian_internal
            )

        return dispatch_kinematics(
            self,
            q,
            s,
            operation="jacobian",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def jacobian_inertialframe_abscissa_batched(
        self,
        q: Array,
        s_ps: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Compute inertial Jacobians at a batch of curvilinear abscissae.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s_ps: Curvilinear abscissae with shape ``(N,)``.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            Inertial-frame Jacobians with shape ``(N, 3, D)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics_abscissa_batched(
                    self,
                    q,
                    s_ps,
                    operation="jacobian",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_poses, jacobians_internal = (
                self._fixed_base_robot_at_pose().forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
                    q_internal, s_ps, backend=backend
                )
            )
            relative_transforms = vmap(poses.planar_pose_to_transform)(relative_poses)
            return self._floating_jacobians_from_relative_inertial(
                q, relative_transforms, jacobians_internal
            )

        return dispatch_kinematics_abscissa_batched(
            self,
            q,
            s_ps,
            operation="jacobian",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def forward_kinematics_and_jacobian_inertialframe(
        self,
        q: Array,
        s: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> tuple[Array, Array]:
        """Compute a planar pose and inertial Jacobian in one traversal.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s: Scalar curvilinear abscissa.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            A pose with shape ``(3,)`` and an inertial Jacobian with shape
            ``(3, D)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics(
                    self,
                    q,
                    s,
                    operation="both",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_pose, jacobian_internal = (
                self._fixed_base_robot_at_pose().forward_kinematics_and_jacobian_inertialframe(
                    q_internal, s, backend=backend
                )
            )
            relative_transform = poses.planar_pose_to_transform(relative_pose)
            return (
                self._compose_runtime_base_poses(q, relative_pose[None])[0],
                self._floating_jacobians_from_relative_inertial(
                    q, relative_transform, jacobian_internal
                ),
            )

        return dispatch_kinematics(
            self,
            q,
            s,
            operation="both",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
        self,
        q: Array,
        s_ps: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> tuple[Array, Array]:
        """Compute fused planar poses and Jacobians for an abscissa batch.

        Args:
            q: Active generalized strains with shape ``(D,)``.
            s_ps: Curvilinear abscissae with shape ``(N,)``.
            backend: Optional execution override. ``None`` uses the model's
                configured backend.

        Returns:
            Poses with shape ``(N, 3)`` and inertial Jacobians with shape
            ``(N, 3, D)``.
        """

        if self.floating_base:
            if uses_warp_kinematics(
                self,
                backend,
                PCS_KINEMATICS,
                warp_supported=type(self) is PlanarPCS,
            ):
                return dispatch_kinematics_abscissa_batched(
                    self,
                    q,
                    s_ps,
                    operation="both",
                    backend=backend,
                    capabilities=PCS_KINEMATICS,
                    warp_supported=True,
                )
            _, q_internal = self.split_configuration(q)
            relative_poses, jacobians_internal = (
                self._fixed_base_robot_at_pose().forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
                    q_internal, s_ps, backend=backend
                )
            )
            relative_transforms = vmap(poses.planar_pose_to_transform)(relative_poses)
            return (
                self._compose_runtime_base_poses(q, relative_poses),
                self._floating_jacobians_from_relative_inertial(
                    q, relative_transforms, jacobians_internal
                ),
            )

        return dispatch_kinematics_abscissa_batched(
            self,
            q,
            s_ps,
            operation="both",
            backend=backend,
            capabilities=PCS_KINEMATICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def jacobian_and_arc_length_derivative_inertialframe(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """Compute an inertial-frame Jacobian and its arc-length derivative.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s: Scalar global backbone coordinate.

        Returns:
            A tuple ``(J, J_s)`` containing the inertial-frame Jacobian and its
            derivative with respect to global arc length. Both arrays have
            shape ``(3, num_active_strains)``.
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
        """Compute the inertial Jacobian derivative with respect to arc length.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s: Scalar global backbone coordinate.

        Returns:
            Arc-length derivative with shape
            ``(3, num_active_strains)``.
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
    def _jacobian_inertialframe_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a batch of points s_ps along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_global_ps (Array): Jacobians evaluated at all points, shape (N, 3, num_active_strains)
        """
        J_local_ps = self.jacobian_bodyframe_abscissa_batched(q, s_ps)

        chi_ps = self._forward_kinematics_abscissa_batched(q, s_ps)  # shape (N, 3)
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
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            fixed_base_robot = self._fixed_base_robot_at_pose()
            relative_tips = fixed_base_robot.forward_kinematics_tips(q_internal)
            relative_transforms = vmap(poses.planar_pose_to_transform)(relative_tips)
            jacobians_internal = jnp.einsum(
                "ijk,kl->ijl", fixed_base_robot._J_local_tips(q_internal), self.B_xi
            )
            return self._floating_inertial_jacobians(
                q, relative_transforms, jacobians_internal
            )
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

        Ad_inv_tips, T_tips, Td_tips = vmap(
            lambda xi_i, xid_i, L_i: constant_strain_se2._operators(
                xi_i, L_i, self.global_eps, self.tangent_eps, xid_i
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
    def _J_Jd_local_abscissa_batched(
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
            Ad_inv, T, Td = constant_strain_se2._operators(
                xi_i,
                arc_len,
                self.global_eps,
                self.tangent_eps,
                xid_i,
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
            Ad_inv, T, Td = constant_strain_se2._operators(
                xi_i,
                arc_len,
                self.global_eps,
                self.global_eps,
                xid_i,
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
    def jacobian_and_time_derivative_bodyframe_abscissa_batched(
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
        J_local_ps_full, Jd_local_ps_full = self._J_Jd_local_abscissa_batched(
            q, qd, s_ps
        )

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

    def jacobian_and_time_derivative_inertialframe_abscissa_batched(
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
        J_local_ps, Jd_local_ps = (
            self.jacobian_and_time_derivative_bodyframe_abscissa_batched(q, qd, s_ps)
        )

        chi_ps = self._forward_kinematics_abscissa_batched(q, s_ps)
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
        return self._jacobian_inertialframe(q, s)

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
    def jacobian_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """Compute inertial-frame Jacobians at multiple arc-length positions.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s_ps: Backbone coordinates with shape ``(num_points,)``.

        Returns:
            Inertial-frame Jacobians with shape
            ``(num_points, 3, num_active_strains)``.
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            fixed_base_robot = self._fixed_base_robot_at_pose()
            relative_poses = fixed_base_robot._forward_kinematics_abscissa_batched(
                q_internal, s_ps
            )
            relative_transforms = vmap(poses.planar_pose_to_transform)(relative_poses)
            jacobians_internal = fixed_base_robot._J_local_abscissa_batched(
                q_internal, s_ps
            )
            return self._floating_inertial_jacobians(
                q, relative_transforms, jacobians_internal
            )
        return self._jacobian_inertialframe_abscissa_batched(q, s_ps)

    @eqx.filter_jit
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """Protected SoftRobot hook for the inertial-frame Jacobian time derivative."""
        return self.jacobian_and_time_derivative_inertialframe(q, qd, s)

    @eqx.filter_jit
    def jacobian_and_time_derivative_abscissa_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """Compute batched inertial Jacobians and their time derivatives.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            qd: Active generalized strain rates with shape
                ``(num_active_strains,)``.
            s_ps: Backbone coordinates with shape ``(num_points,)``.

        Returns:
            A tuple ``(J, J_dot)`` whose arrays both have shape
            ``(num_points, 3, num_active_strains)``.
        """
        if self.floating_base:
            qdot = self.configuration_derivative(q, qd)
            return jvp(
                lambda q_value: self.jacobian_abscissa_batched(q_value, s_ps),
                (q,),
                (qdot,),
            )
        return self.jacobian_and_time_derivative_inertialframe_abscissa_batched(
            q, qd, s_ps
        )

    # ==========================================
    # Useful functions for the system

    @eqx.filter_jit
    def _local_cross_sectional_area(self, i: Array) -> Array:
        """
        Compute the area of the i-th segment's solid circular cross-section.

        Args:
            i (Array): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        # Area of the assumed solid circular cross-section
        A_i = jnp.pi * self.r[i] ** 2
        return A_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute the transverse second moment of the solid circular section.

        The rod is longitudinally aligned with local x and bends about local z.

        Args:
            i (Array): index of the segment

        Returns:
            I_i (Array): local second moment of area of the i-th segment
        """
        # Transverse second moment of the assumed solid circular cross-section
        I_i = jnp.pi * self.r[i] ** 4 / 4
        return I_i

    def _compute_local_mass_matrix(self, i: Array) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        The rotational entry uses the solid circular cross-section's transverse
        second moment about local z; local x is the longitudinal direction.

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

        J_ps = self._J_local_abscissa_batched(q, Xs_scaled.flatten())
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
        if self.floating_base:
            zeros = jnp.zeros((self.num_velocities,), dtype=q.dtype)
            return self._assemble_dynamics_terms(q, zeros)[0]
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

        J_ps, Jd_ps = self._J_Jd_local_abscissa_batched(q, qd, Xs_scaled.flatten())
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
        if self.floating_base:
            return self._floating_coriolis_matrix(q, qd)
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

        chi_ps = self._forward_kinematics_abscissa_batched(q, Xs_scaled.flatten())
        g_ps = vmap(poses.planar_pose_to_transform)(chi_ps.reshape(-1, 3))
        g_ps = g_ps.reshape(self.num_segments, self.num_integration_points, 3, 3)

        J_ps = self._J_local_abscissa_batched(q, Xs_scaled.flatten())
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

    def _material_operators(self) -> tuple[Array, Array, Array]:
        """Return unit Young, shear, and material-damping link matrices."""
        area = jnp.pi * self.r**2
        moment = jnp.pi * self.r**4 / 4.0
        zeros = jnp.zeros_like(self.L)
        young = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack([moment, area, zeros], axis=1)
        )
        shear = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack([zeros, zeros, area], axis=1)
        )
        damping = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack([3.0 * moment, 3.0 * area, area], axis=1)
        )
        return young, shear, damping

    def _compute_stiffness_full_matrix(self) -> Array:
        """Assemble canonical per-link generalized stiffness blocks."""
        return blk_diag(self._current_body_params().link.stiffness)

    @eqx.filter_jit
    def _stiffness(self, formulate_in_strain_space: bool = False) -> Array:
        if formulate_in_strain_space:
            return self.K_full

        return self.K_active

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
        if self.floating_base:
            return jnp.pad(
                self.K_active,
                ((self.num_base_velocities, 0), (self.num_base_velocities, 0)),
            )
        return self.K_active

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            tau_el (Array): Elastic force of shape (num_active_strains,).
        """
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            internal = self.K_active @ q_internal + self.passive_elastic_force(
                q_internal
            )
            return jnp.pad(internal, (self.num_base_velocities, 0))
        return self.K_active @ q + self.passive_elastic_force(q)

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
        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            internal = self.D_active + self.passive_damping_matrix(q_internal)
            return jnp.pad(
                internal,
                ((self.num_base_velocities, 0), (self.num_base_velocities, 0)),
            )
        return self.D_active + self.passive_damping_matrix(q)

    def _elastic_energy(self, q: Array) -> Array:
        """Return body strain energy plus installed passive-element energy."""
        return 0.5 * q @ self.K_active @ q + self.passive_elastic_energy(q)

    def actuation_matrix(
        self, q: Array, *, backend: ExecutionBackend | None = None
    ) -> Array:
        """Return the actuator matrix through the selected execution backend."""

        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            internal = self._fixed_base_robot_at_pose().actuation_matrix(
                q_internal, backend=backend
            )
            return jnp.pad(internal, ((self.num_base_velocities, 0), (0, 0)))
        return dispatch_actuation_matrix(
            self, q, backend=backend, capabilities=PLANAR_PCS_ACTUATION
        )

    def actuation_force(
        self,
        q: Array,
        u: Array,
        qd: Array | None = None,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Return generalized actuator force through the selected backend."""

        if self.floating_base:
            _, q_internal = self.split_configuration(q)
            qd_internal = None
            if qd is not None:
                _, qd_internal = self.split_velocity(qd)
            internal = self._fixed_base_robot_at_pose().actuation_force(
                q_internal, u, qd_internal, backend=backend
            )
            return jnp.pad(internal, (self.num_base_velocities, 0))
        return dispatch_actuation_force(
            self,
            q,
            u,
            qd,
            backend=backend,
            capabilities=PLANAR_PCS_ACTUATION,
        )

    def _threadlike_local_basis(
        self,
        segment_index: Array,
        strain: Array,
        s: Array,
        routing: ThreadlikeRouting,
        path_params: BaseThreadlikeRoutingParams,
        start_segment_index: Array,
        end_segment_index: Array,
    ) -> Array:
        active = (start_segment_index <= segment_index) & (
            segment_index <= end_segment_index
        )
        offset = routing.offset(path_params, s)[1]
        offset_derivative = routing.derivative(path_params, s)[1]
        axial = strain[1] + offset * strain[0]
        shear = strain[2] + offset_derivative
        norm = safe_norm(jnp.stack([axial, shear]))
        axial_ratio = safe_divide(axial, norm, self.global_eps)
        shear_ratio = safe_divide(shear, norm, self.global_eps)
        return active * jnp.asarray([offset * axial_ratio, axial_ratio, shear_ratio])

    def _threadlike_moment_matrix(self, q: Array, routing: ThreadlikeRouting) -> Array:
        """Integrate raw routed-length moment arms in the planar PCS basis."""
        params = routing.params
        count = params.num_paths
        if count == 0:
            return jnp.zeros((self.num_internal_dofs, 0), dtype=q.dtype)
        strains = self.strain(q).reshape((self.num_segments, 3))

        def segment_matrix(segment_index: Array) -> Array:
            points, weights = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[segment_index],
                self.L_cum[segment_index + 1],
            )

            def point_matrix(point_index: Array) -> Array:
                basis = vmap(
                    self._threadlike_local_basis,
                    in_axes=(None, None, None, None, 0, 0, 0),
                    out_axes=1,
                )(
                    segment_index,
                    strains[segment_index],
                    points[point_index],
                    routing,
                    params,
                    params.start_segment_index_array,
                    params.end_segment_index_array,
                )
                return weights[point_index] * basis

            return jnp.sum(
                vmap(point_matrix)(jnp.arange(self.num_integration_points)), axis=0
            )

        full_matrix = vmap(segment_matrix)(jnp.arange(self.num_segments)).reshape(
            self.num_strains, count
        )
        return self.B_xi.T @ full_matrix

    def _threadlike_path_lengths(self, q: Array, routing: ThreadlikeRouting) -> Array:
        """Integrate raw planar threadlike path lengths."""
        params = routing.params
        if params.num_paths == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        strains = self.strain(q).reshape((self.num_segments, 3))

        def segment_density(segment_index: Array) -> Array:
            points, weights = scale_gaussian_quadrature(
                self.integration_points,
                self.integration_weights,
                self.L_cum[segment_index],
                self.L_cum[segment_index + 1],
            )

            def point_density(point_index: Array) -> Array:
                s = points[point_index]

                def path_density(
                    path_params: BaseThreadlikeRoutingParams,
                    start_segment_index: Array,
                    end_segment_index: Array,
                ) -> Array:
                    active = (start_segment_index <= segment_index) & (
                        segment_index <= end_segment_index
                    )
                    offset = routing.offset(path_params, s)[1]
                    derivative = routing.derivative(path_params, s)[1]
                    axial = (
                        strains[segment_index, 1] + offset * strains[segment_index, 0]
                    )
                    shear = strains[segment_index, 2] + derivative
                    return active * safe_norm(jnp.stack([axial, shear]))

                density = vmap(path_density)(
                    params,
                    params.start_segment_index_array,
                    params.end_segment_index_array,
                )
                return weights[point_index] * density

            return jnp.sum(
                vmap(point_density)(jnp.arange(self.num_integration_points)), axis=0
            )

        return jnp.sum(vmap(segment_density)(jnp.arange(self.num_segments)), axis=0)

    def _threadlike_path_positions(
        self, q: Array, s: Array, routing: ThreadlikeRouting
    ) -> Array:
        """Return planar positions of all routed paths at backbone coordinate ``s``."""
        params = routing.params
        if params.num_paths == 0:
            return jnp.zeros((0, 2), dtype=q.dtype)

        def path_position(
            path_params: BaseThreadlikeRoutingParams,
            start_segment_index: Array,
            end_segment_index: Array,
        ) -> Array:
            start = self.L_cum[start_segment_index]
            end = self.L_cum[end_segment_index + 1]
            s_clamped = jnp.clip(s, start, end)
            pose = self.forward_kinematics(q, s_clamped)
            normal = jnp.asarray([-jnp.sin(pose[0]), jnp.cos(pose[0])])
            offset = routing.offset(path_params, s_clamped)[1]
            return pose[1:] + offset * normal

        return vmap(path_position)(
            params,
            params.start_segment_index_array,
            params.end_segment_index_array,
        )

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

        chi_ps = self._forward_kinematics_abscissa_batched(q, Xs_scaled.flatten())
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
    def _floating_gravitational_energy(self, q: Array) -> Array:
        """
        Compute absolute floating-base gravitational potential energy.

        The quadrature recurrence evaluates positions only. It does not build
        Jacobians or spatial inertia matrices.

        Args:
            q: Total floating-base configuration with shape
                ``(num_coordinates,)``.

        Returns:
            Absolute gravitational potential energy as a scalar.
        """
        _, q_internal = self.split_configuration(q)
        points, weights = vmap(scale_gaussian_quadrature, in_axes=(None, None, 0, 0))(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        relative_poses = (
            self._fixed_base_robot_at_pose()._forward_kinematics_abscissa_batched(
                q_internal, points.reshape(-1)
            )
        )
        relative_positions = relative_poses.reshape(
            self.num_segments, self.num_integration_points, 3
        )[..., 1:]
        weighted_masses = (
            weights
            * self.rho[:, None]
            * vmap(self._local_cross_sectional_area)(
                jnp.arange(self.num_segments, dtype=jnp.int32)
            )[:, None]
        )
        return self._floating_gravitational_energy_from_points(
            q, relative_positions, weighted_masses
        )

    @eqx.filter_jit
    def _active_J_Jd_local_tips_from_strain(
        self,
        xi: Array,
        xid: Array,
        B_xi_segments: Array,
        qd: Array,
        convective_only_jd: bool = False,
        jacobian_initial: Array | None = None,
        jacobian_dot_velocity_initial: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """
        Propagate active-coordinate local Jacobians to every segment tip.

        The recurrence remains in each segment's body frame. It can propagate
        either the complete Jacobian time derivative or only its contraction
        with ``qd``. The returned inverse adjoints let dynamics-only callers
        propagate local quantities without constructing absolute poses.

        Args:
            xi: Segment strains with shape ``(self.num_segments, 3)``.
            xid: Segment strain rates with shape ``(self.num_segments, 3)``.
            B_xi_segments: Active strain bases with shape
                ``(self.num_segments, 3, self.num_internal_dofs)``.
            qd: Active generalized velocities, shape ``(self.num_internal_dofs,)``.
            convective_only_jd: If true, propagate ``Jd @ qd`` instead of the
                complete Jacobian time derivative.

        Returns:
            Tuple ``(J_tips, Jd_or_Jd_qd_tips, Ad_inv_tips)``. ``J_tips`` has
            shape ``(self.num_segments, 3, self.num_internal_dofs)``. The second array
            has the same shape when ``convective_only_jd`` is false and shape
            ``(self.num_segments, 3)`` otherwise. ``Ad_inv_tips`` contains the
            inverse adjoint of each segment-tip transform with shape
            ``(self.num_segments, 3, 3)``.
        """
        Ad_inv_tips, T_tips, Td_tips = vmap(
            lambda xi_i, xid_i, L_i: constant_strain_se2._operators(
                xi_i, L_i, self.global_eps, self.tangent_eps, xid_i
            )
        )(xi, xid, self.L)

        zeros = (
            jnp.zeros((3, self.num_internal_dofs), dtype=xi.dtype)
            if jacobian_initial is None
            else jacobian_initial
        )
        if convective_only_jd:
            Jd_or_Jd_qd_zeros = (
                jnp.zeros((3,), dtype=xi.dtype)
                if jacobian_dot_velocity_initial is None
                else jacobian_dot_velocity_initial
            )
        else:
            Jd_or_Jd_qd_zeros = zeros

        def scan_body(
            carry: tuple[Array, Array], i: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_or_Jd_qd_prev = carry

            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
            B_xi_i = lax.dynamic_index_in_dim(B_xi_segments, i, axis=0, keepdims=False)
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)

            Ad_inv_T_i = Ad_inv_i @ T_i
            J_segment = Ad_inv_T_i @ B_xi_i
            J_next = Ad_inv_i @ J_prev + J_segment

            eta = Ad_inv_T_i @ xid_i
            Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv_i

            if convective_only_jd:
                # The omitted local term Ad_inv_dot @ (T_i @ xid_i) equals
                # -ad_eta @ eta and is therefore exactly zero.
                Jd_qd_next = (
                    Ad_inv_i @ Jd_or_Jd_qd_prev
                    + Ad_inv_dot @ (J_prev @ qd)
                    + Ad_inv_i @ (Td_i @ xid_i)
                )
                Jd_or_Jd_qd_next = Jd_qd_next
            else:
                Jd_segment = (Ad_inv_dot @ T_i + Ad_inv_i @ Td_i) @ B_xi_i
                Jd_next = Ad_inv_i @ Jd_or_Jd_qd_prev + Ad_inv_dot @ J_prev + Jd_segment
                Jd_or_Jd_qd_next = Jd_next

            return (J_next, Jd_or_Jd_qd_next), (J_next, Jd_or_Jd_qd_next)

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        (_, _), (J_tips, Jd_or_Jd_qd_tips) = lax.scan(
            scan_body, (zeros, Jd_or_Jd_qd_zeros), indices
        )

        return J_tips, Jd_or_Jd_qd_tips, Ad_inv_tips

    @eqx.filter_jit
    def integration_kinematics(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Return integration-point kinematics in active generalized coordinates.

        The stored quadrature grid includes zero-weight endpoint nodes; this
        method ignores those endpoints and evaluates only the nonzero-weight
        interior quadrature nodes. The coordinates are generalized coordinates
        for active strain components only.

        Args:
            q: Active generalized coordinates, shape ``(self.num_internal_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_internal_dofs,)``.

        Returns:
            Tuple ``(g_ps, J_ps, Jd_ps)``. ``g_ps`` contains SE(2) poses with
            shape ``(self.num_segments, self.num_gauss_points, 3, 3)``.
            ``J_ps`` contains body-frame Jacobians in active generalized
            coordinates with shape
            ``(self.num_segments, self.num_gauss_points, 3, self.num_internal_dofs)``.
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
            q: Active generalized coordinates, shape ``(self.num_internal_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_internal_dofs,)``.
            convective_only_jd: If true, return ``Jd @ qd`` directly instead
                of materializing the complete Jacobian time derivative.

        When ``convective_only_jd`` is true, the fourth returned array has
        shape ``(num_segments, num_gauss_points, 3)`` and contains the exact
        contraction ``Jd @ qd`` required by forward dynamics. Otherwise it
        contains the full Jacobian derivatives with the same shape as ``J``.
        """
        if self.floating_base:
            q = self.normalize_configuration(q)
            _, q_internal = self.split_configuration(q)
            base_velocity, qd_internal = self.split_velocity(qd)
            assert base_velocity is not None
            g0 = self.base_transform_from_configuration(q)
            jacobian_base, jacobian_dot_qd_initial, _, _ = (
                self._floating_base_body_kinematics(q, qd)
            )
            jacobian_initial = jnp.pad(
                jacobian_base, ((0, 0), (0, self.num_internal_dofs))
            )
            if convective_only_jd:
                jacobian_derivative_initial = jacobian_dot_qd_initial
            else:
                rotation_transpose = g0[:2, :2].T
                rotation_generator = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=q.dtype)
                jacobian_derivative_base = jnp.zeros((3, 3), dtype=q.dtype)
                jacobian_derivative_base = jacobian_derivative_base.at[1:, 1:].set(
                    -base_velocity[0] * rotation_generator @ rotation_transpose
                )
                jacobian_derivative_initial = jnp.pad(
                    jacobian_derivative_base,
                    ((0, 0), (0, self.num_internal_dofs)),
                )
            velocity_coordinates = qd
        else:
            q_internal = q
            qd_internal = qd
            g0 = poses.planar_pose_to_transform(
                jnp.asarray(self._kinematic_base_pose, dtype=q.dtype)
            )
            jacobian_initial = jnp.zeros((3, self.num_internal_dofs), dtype=q.dtype)
            jacobian_derivative_initial = (
                jnp.zeros((3,), dtype=q.dtype)
                if convective_only_jd
                else jacobian_initial
            )
            velocity_coordinates = qd

        xi = self.strain(q_internal).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd_internal).reshape(self.num_segments, 3)
        basis_internal = self.B_xi.reshape(self.num_segments, 3, self.num_internal_dofs)
        B_xi_segments = (
            jnp.pad(
                basis_internal,
                ((0, 0), (0, 0), (self.num_base_velocities, 0)),
            )
            if self.floating_base
            else basis_internal
        )

        Xs_scaled, Ws_scaled = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_scaled - self.L_cum[:-1, None]

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

        J_tips, Jd_or_Jd_qd_tips, _ = self._active_J_Jd_local_tips_from_strain(
            xi,
            xid,
            B_xi_segments,
            velocity_coordinates,
            convective_only_jd=convective_only_jd,
            jacobian_initial=(jacobian_initial if self.floating_base else None),
            jacobian_dot_velocity_initial=(
                jacobian_derivative_initial if self.floating_base else None
            ),
        )
        J_bases = jnp.concatenate([jacobian_initial[None], J_tips[:-1]], axis=0)
        Jd_or_Jd_qd_bases = jnp.concatenate(
            [jacobian_derivative_initial[None], Jd_or_Jd_qd_tips[:-1]], axis=0
        )

        def segment_jacobians(
            xi_i: Array,
            xid_i: Array,
            B_xi_i: Array,
            J_base_i: Array,
            Jd_or_Jd_qd_base_i: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array]:
            def jacobian_at_s(s_local_ij: Array) -> tuple[Array, Array]:
                Ad_inv, T, Td = constant_strain_se2._operators(
                    xi_i,
                    s_local_ij,
                    self.global_eps,
                    self.tangent_eps,
                    xid_i,
                )

                Ad_inv_T = Ad_inv @ T
                J_segment = Ad_inv_T @ B_xi_i
                J_next = Ad_inv @ J_base_i + J_segment

                eta = Ad_inv_T @ xid_i
                Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv

                if convective_only_jd:
                    # The omitted local term Ad_inv_dot @ (T @ xid_i) equals
                    # -ad_eta @ eta and is therefore exactly zero.
                    Jd_qd_next = (
                        Ad_inv @ Jd_or_Jd_qd_base_i
                        + Ad_inv_dot @ (J_base_i @ velocity_coordinates)
                        + Ad_inv @ (Td @ xid_i)
                    )
                    Jd_or_Jd_qd_next = Jd_qd_next
                else:
                    Jd_segment = (Ad_inv_dot @ T + Ad_inv @ Td) @ B_xi_i
                    Jd_next = (
                        Ad_inv @ Jd_or_Jd_qd_base_i + Ad_inv_dot @ J_base_i + Jd_segment
                    )
                    Jd_or_Jd_qd_next = Jd_next

                return J_next, Jd_or_Jd_qd_next

            return vmap(jacobian_at_s)(s_local_i)

        J_ps, Jd_or_Jd_qd_ps = vmap(segment_jacobians)(
            xi, xid, B_xi_segments, J_bases, Jd_or_Jd_qd_bases, s_local
        )

        return Ws_scaled, g_ps, J_ps, Jd_or_Jd_qd_ps

    @eqx.filter_jit
    def _dynamics_integration_kinematics(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Return dynamics-only quadrature kinematics in local body frames.

        Unlike :meth:`_integration_kinematics`, this path does not construct
        absolute SE(2) poses. It propagates gravity between segment frames with
        inverse adjoints and directly computes the convective contraction
        ``Jd @ qd`` needed by :meth:`dynamics_terms`.

        Args:
            q: Active generalized coordinates, shape ``(self.num_internal_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_internal_dofs,)``.

        Returns:
            Tuple ``(weights, gravity, jacobians, jacobian_dot_qd)``.
            ``weights`` has shape
            ``(self.num_segments, self.num_gauss_points)``; ``gravity`` and
            ``jacobian_dot_qd`` have shape
            ``(self.num_segments, self.num_gauss_points, 3)``; and
            ``jacobians`` has shape
            ``(self.num_segments, self.num_gauss_points, 3, self.num_internal_dofs)``.
        """
        # Preserve the exact fixed-base hot path from PR #183. This static
        # branch returns before any augmented operands or shapes are created.
        if not self.floating_base:
            xi = self.strain(q).reshape(self.num_segments, 3)
            xid = (self.B_xi @ qd).reshape(self.num_segments, 3)
            B_xi_segments = self.B_xi.reshape(
                self.num_segments, 3, self.num_internal_dofs
            )

            Xs_scaled, Ws_scaled = vmap(
                scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
            )(
                self.integration_points,
                self.integration_weights,
                self.L_cum[:-1],
                self.L_cum[1:],
            )
            s_local = Xs_scaled - self.L_cum[:-1, None]

            J_tips, Jd_qd_tips, Ad_inv_tips = self._active_J_Jd_local_tips_from_strain(
                xi,
                xid,
                B_xi_segments,
                qd,
                convective_only_jd=True,
            )
            J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)
            Jd_qd_bases = jnp.concatenate(
                [jnp.zeros_like(Jd_qd_tips[:1]), Jd_qd_tips[:-1]], axis=0
            )

            g0 = poses.planar_pose_to_transform(
                jnp.asarray(self._kinematic_base_pose, dtype=xi.dtype)
            )
            gravity_base_initial = se2.adjoint_inverse(g0) @ self.g

            def scan_gravity(
                gravity_base: Array, Ad_inv_tip: Array
            ) -> tuple[Array, Array]:
                gravity_tip = Ad_inv_tip @ gravity_base
                return gravity_tip, gravity_base

            _, gravity_bases = lax.scan(scan_gravity, gravity_base_initial, Ad_inv_tips)

            def segment_kinematics(
                xi_i: Array,
                xid_i: Array,
                B_xi_i: Array,
                gravity_base_i: Array,
                J_base_i: Array,
                Jd_qd_base_i: Array,
                s_local_i: Array,
            ) -> tuple[Array, Array, Array]:
                def kinematics_at_s(
                    s_local_ij: Array,
                ) -> tuple[Array, Array, Array]:
                    Ad_inv, T, Td = constant_strain_se2._operators(
                        xi_i,
                        s_local_ij,
                        self.global_eps,
                        self.tangent_eps,
                        xid_i,
                    )
                    Ad_inv_T = Ad_inv @ T
                    J_segment = Ad_inv_T @ B_xi_i
                    J_next = Ad_inv @ J_base_i + J_segment

                    eta = Ad_inv_T @ xid_i
                    Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv
                    Jd_qd_next = (
                        Ad_inv @ Jd_qd_base_i
                        + Ad_inv_dot @ (J_base_i @ qd)
                        + Ad_inv @ (Td @ xid_i)
                    )
                    gravity_local = Ad_inv @ gravity_base_i
                    return gravity_local, J_next, Jd_qd_next

                return vmap(kinematics_at_s)(s_local_i)

            gravity_ps, J_ps, Jd_qd_ps = vmap(segment_kinematics)(
                xi,
                xid,
                B_xi_segments,
                gravity_bases,
                J_bases,
                Jd_qd_bases,
                s_local,
            )
            return Ws_scaled, gravity_ps, J_ps, Jd_qd_ps

        q = self.normalize_configuration(q)
        _, q_internal = self.split_configuration(q)
        _, qd_internal = self.split_velocity(qd)
        (
            jacobian_base,
            jacobian_dot_velocity_initial,
            _,
            gravity_base_initial,
        ) = self._floating_base_body_kinematics(q, qd)
        jacobian_initial = jnp.pad(jacobian_base, ((0, 0), (0, self.num_internal_dofs)))
        xi = self.strain(q_internal).reshape(self.num_segments, 3)
        xid = (self.B_xi @ qd_internal).reshape(self.num_segments, 3)
        basis_internal = self.B_xi.reshape(self.num_segments, 3, self.num_internal_dofs)
        B_xi_segments = jnp.pad(
            basis_internal,
            ((0, 0), (0, 0), (self.num_base_velocities, 0)),
        )

        Xs_scaled, Ws_scaled = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_scaled - self.L_cum[:-1, None]

        J_tips, Jd_qd_tips, Ad_inv_tips = self._active_J_Jd_local_tips_from_strain(
            xi,
            xid,
            B_xi_segments,
            qd,
            convective_only_jd=True,
            jacobian_initial=jacobian_initial,
            jacobian_dot_velocity_initial=jacobian_dot_velocity_initial,
        )
        J_bases = jnp.concatenate([jacobian_initial[None], J_tips[:-1]], axis=0)
        Jd_qd_bases = jnp.concatenate(
            [jacobian_dot_velocity_initial[None], Jd_qd_tips[:-1]], axis=0
        )

        def scan_gravity(gravity_base: Array, Ad_inv_tip: Array) -> tuple[Array, Array]:
            gravity_tip = Ad_inv_tip @ gravity_base
            return gravity_tip, gravity_base

        _, gravity_bases = lax.scan(scan_gravity, gravity_base_initial, Ad_inv_tips)

        def segment_kinematics(
            xi_i: Array,
            xid_i: Array,
            B_xi_i: Array,
            gravity_base_i: Array,
            J_base_i: Array,
            Jd_qd_base_i: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array, Array]:
            def kinematics_at_s(s_local_ij: Array) -> tuple[Array, Array, Array]:
                Ad_inv, T, Td = constant_strain_se2._operators(
                    xi_i,
                    s_local_ij,
                    self.global_eps,
                    self.tangent_eps,
                    xid_i,
                )
                Ad_inv_T = Ad_inv @ T
                J_segment = Ad_inv_T @ B_xi_i
                J_next = Ad_inv @ J_base_i + J_segment

                eta = Ad_inv_T @ xid_i
                Ad_inv_dot = -se2.small_adjoint(eta) @ Ad_inv
                Jd_qd_next = (
                    Ad_inv @ Jd_qd_base_i
                    + Ad_inv_dot @ (J_base_i @ qd)
                    + Ad_inv @ (Td @ xid_i)
                )
                gravity_local = Ad_inv @ gravity_base_i
                return gravity_local, J_next, Jd_qd_next

            return vmap(kinematics_at_s)(s_local_i)

        gravity_ps, J_ps, Jd_qd_ps = vmap(segment_kinematics)(
            xi,
            xid,
            B_xi_segments,
            gravity_bases,
            J_bases,
            Jd_qd_bases,
            s_local,
        )
        return Ws_scaled, gravity_ps, J_ps, Jd_qd_ps

    def dynamics_terms(
        self,
        q: Array,
        qd: Array,
        *,
        backend: ExecutionBackend | None = None,
    ) -> tuple[Array, Array, Array]:
        """Assemble planar PCS dynamics terms for one or many environments.

        ``backend=None`` uses the model's configured :attr:`backend`. Warp is
        selected for forward-only GPU execution with any positive quadrature
        count; CPU execution and all forward- or reverse-mode differentiation
        use the JAX implementation. Applying :func:`jax.vmap` to scalar calls
        invokes one batch-shaped Warp pipeline rather than mapping independent
        batch-one launches.

        The Warp implementation uses FP64 SE(2) constant-strain operators and
        one persistent cooperative chain block per environment. Its generated
        source is independent of the batch size and segment count. Install the
        optional dependency with ``pip install soromox[warp]`` before explicitly
        requesting it.

        Args:
            q: Active generalized coordinates with shape ``(num_dofs,)`` or
                ``(batch_size, num_dofs)``.
            qd: Active generalized velocities with the same shape as ``q``.
            backend: Optional per-call override: ``"auto"``, ``"jax"``, or
                ``"warp"``. ``None`` uses :attr:`backend`.

        Returns:
            Tuple ``(B, Cqd, G)`` with the same optional leading batch dimension
            as the inputs. ``B`` is the active inertia matrix, ``Cqd`` is the
            Coriolis/centrifugal force action, and ``G`` is generalized gravity.

        Raises:
            ValueError: If the backend or input shapes are invalid, or a Warp
                batch is empty.
            TypeError: If Warp is selected for non-FP64 states.
            ImportError: If Warp is selected but ``warp-lang`` is unavailable.
        """

        return dispatch_dynamics_terms(
            self,
            q,
            qd,
            backend=backend,
            capabilities=PCS_DYNAMICS,
            warp_supported=type(self) is PlanarPCS,
        )

    @eqx.filter_jit
    def _assemble_dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        """
        Assemble forward-dynamics terms in active generalized coordinates.

        The coordinates and returned generalized forces correspond only to the
        active strain components selected by the strain basis.

        Args:
            q: Active generalized coordinates, shape ``(self.num_internal_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_internal_dofs,)``.

        Returns:
            Tuple ``(B, Cqd, G)``. ``B`` is the active-coordinate inertia
            matrix with shape ``(self.num_internal_dofs, self.num_internal_dofs)``. ``Cqd`` is
            the active Coriolis/centrifugal force vector with shape
            ``(self.num_internal_dofs,)``. ``G`` is the active generalized gravity
            vector with shape ``(self.num_internal_dofs,)``.
        """
        weights, gravity, jacobians, jacobian_dot_qd = (
            self._dynamics_integration_kinematics(q, qd)
        )
        inertia = jnp.zeros(
            (self.num_velocities, self.num_velocities), dtype=jacobians.dtype
        )
        coriolis_qd = jnp.zeros((self.num_velocities,), dtype=jacobians.dtype)
        gravity_force = jnp.zeros((self.num_velocities,), dtype=jacobians.dtype)

        # This trace-time loop intentionally keeps each contraction at its causal
        # prefix width; scan/fori_loop require a fixed-width body and do extra work.
        for segment_index, dof_end in enumerate(self._segment_dof_ends):
            velocity_end = self.num_base_velocities + dof_end
            mass_diagonal = jnp.diagonal(self.M_segments[segment_index])
            jacobian = jacobians[segment_index, :, :, :velocity_end]
            weight = weights[segment_index]

            def mass_action(value: Array, diagonal: Array = mass_diagonal) -> Array:
                shape = (1, diagonal.shape[0]) + (1,) * (value.ndim - 2)
                return diagonal.reshape(shape) * value

            flattened_rows = self.num_gauss_points * 3
            jacobian_flat = jacobian.reshape(flattened_rows, velocity_end)
            weighted_mass_jacobian = (
                weight[:, None, None] * mass_action(jacobian)
            ).reshape(flattened_rows, velocity_end)
            inertia_segment = jacobian_flat.T @ weighted_mass_jacobian

            eta = jacobian @ qd[:velocity_end]
            momentum = mass_action(eta)
            coadjoint_momentum = vmap(se2.coadjoint_action)(eta, momentum)
            wrench = mass_action(jacobian_dot_qd[segment_index]) + coadjoint_momentum
            coriolis_segment = jacobian_flat.T @ (weight[:, None] * wrench).reshape(-1)
            gravity_segment = -jacobian_flat.T @ (
                weight[:, None] * mass_action(gravity[segment_index])
            ).reshape(-1)

            inertia = inertia.at[:velocity_end, :velocity_end].add(inertia_segment)
            coriolis_qd = coriolis_qd.at[:velocity_end].add(coriolis_segment)
            gravity_force = gravity_force.at[:velocity_end].add(gravity_segment)

        return inertia, coriolis_qd, gravity_force

    def forward_dynamics(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None = None,
        *,
        backend: ExecutionBackend | None = None,
    ) -> Array:
        """Compute the time derivative of a planar PCS state.

        Args:
            t: Current integration time. Planar PCS dynamics are autonomous, so
                the value is accepted for solver compatibility but is not
                otherwise used.
            y: State ``[q, v, auxiliary]`` with shape ``(state_size,)``.
            actuation_args: Optional tuple containing the actuation input ``u``
                and, optionally, an external generalized force ``tau_ext``. A
                one-element tuple is interpreted as ``(u,)`` and a two-element
                tuple as ``(u, tau_ext)``. Missing values default to zero.
            backend: Optional execution-backend override for the inertia,
                Coriolis, and gravity terms. ``None`` uses :attr:`backend`.

        Returns:
            State derivative ``[qd, qdd]`` with the same shape as ``y``.

        Raises:
            ValueError: If ``actuation_args`` does not contain one or two
                elements.
        """
        return evaluate_forward_dynamics(
            self, t, y, actuation_args, backend, PCS_DYNAMICS
        )
