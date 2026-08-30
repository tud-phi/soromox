__all__ = ["PCS"]


from typing import Any, Self

import equinox as eqx
from jax import Array, jvp, lax, vmap
from jax import numpy as jnp

from soromox.actuation.core import Actuator, PassiveElement
from soromox.actuation.threadlike import (
    BaseThreadlikeRoutingParams,
    ThreadlikeRouting,
)
from soromox.autodiff import custom_jvp_enabled
from soromox.systems.components import (
    ContinuumLinkParams,
    CrossSectionGeometry,
    CrossSectionParams,
    IsotropicMaterialParams,
    LinkSpec,
)
from soromox.systems.pcs.params import PCSParams
from soromox.systems.pcs.structures import PCSStructure
from soromox.systems.soft_robot import SoftRobot
from soromox.utils.array_math import blk_diag
from soromox.utils.dof import build_active_dof_basis
from soromox.utils.geometry import poses
from soromox.utils.integration import (
    gauss_quadrature,
    scale_gaussian_quadrature,
    scale_interior_gaussian_quadrature,
)
from soromox.utils.joint_motion_subspace import (
    joint_dSdq_qd,
    joint_motion_subspace_with_derivatives,
)
from soromox.utils.lie_algebra import se3, so3
from soromox.utils.lie_algebra.constant_strain import se3 as constant_strain_se3
from soromox.utils.numerics import safe_divide, safe_norm, safe_normalize


class PCS(SoftRobot):
    """
    Piecewise Constant Strain (PCS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory and piecewise constant strain assumption.
    It supports computation of forward kinematics, Jacobians, dynamical matrices.

    Attributes:
        num_segments: Number of segments (constant strain sections) along the robot.
        num_actuators: Number of actuators (control inputs) for the robot.
        g0: Initial pose of the robot base as an SE(3) transformation matrix.
        g: Gravitational acceleration vector (embedded in a 6D vector).
            [0, 0, 0, g_x, g_y, g_z]
        L, r, E, G, rho: Physical properties of each segment. ``r`` is the
            radius of the assumed solid circular cross-section.
        num_active_strains: Number of active strain components (based on strain_selector).
        num_strains: Total number of strain components (6 * num_segments).
        B_xi: Basis matrix for projecting active strains (6 * num_segments, num_active_strains).
        scale_rotational_basis_by_length: If True, rotational strain-basis rows
            are divided by their segment length.
        xi_ref: Reference strain (reference configuration) of the robot.
        num_gauss_points: Requested nonzero Gauss-Legendre quadrature nodes.
        num_integration_points: Stored integration nodes, including zero-weight endpoints.
        integration_points, integration_weights: Quadrature nodes and weights.

    Notes:
    -----
    - The material-frame local x-axis is the rod's longitudinal axis (the
      undeformed backbone tangent). This convention is independent of the base
      pose, which may orient the local x-axis arbitrarily in the inertial frame.
    - The strain vector is composed of 6 components per segment:
      [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]. The default
      straight, unstretched reference strain is [0, 0, 0, 1, 0, 0]. Thus,
      kappa_x is torsion about the longitudinal axis, kappa_y and kappa_z are
      bending strains, sigma_x is axial stretch, and sigma_y and sigma_z are
      transverse shear strains.
    - Every segment is assumed to have a solid circular cross-section. Its
      radius determines the area and second moments used by the mass, material
      damping, and stiffness matrices.
    - ``material_damping_coefficient`` is a viscosity-like modulus in Pa*s
      (N*s/m^2). The assembled damping matrix also contains section geometry
      and segment-length factors, so its entries do not share one blanket unit.

    References:
        Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete Cosserat
        Approach for Multisection Soft Manipulator Dynamics. IEEE Transactions on
        Robotics, 34(6), 1518-1533.

    """

    params: PCSParams

    # Robot parameters
    g0: Array  # Initial position and orientation of the rod
    g: Array  # Gravitational acceleration vector

    L: Array  # Length of the segments
    L_cum: Array  # Cumulative length of the segments
    r: Array  # Radius of the segments
    rho: Array

    num_segments: int = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)
    num_integration_points: int = eqx.field(static=True)
    num_strains: int = eqx.field(static=True)  # Number of strains (6 * num_segments)
    _segment_dof_ends: tuple[int, ...] = eqx.field(static=True)
    scale_rotational_basis_by_length: bool = eqx.field(static=True)

    xi_ref: Array  # Reference configuration strain
    B_xi_unscaled: Array  # Unscaled strain basis matrix
    B_xi: Array  # Strain basis matrix
    num_active_strains: Array  # Number of selected strains

    integration_points: Array
    integration_weights: Array
    M_segments: Array  # Cached per-segment mass matrices
    K_full: Array  # Cached full stiffness matrix
    K_active: Array  # Cached active-coordinate stiffness matrix
    D_full: Array  # Cached full damping matrix
    D_active: Array  # Cached active-coordinate damping matrix
    young_stiffness_operator: Array
    shear_stiffness_operator: Array
    material_damping_operator: Array

    @staticmethod
    def params_from_links(
        links: list[LinkSpec] | tuple[LinkSpec, ...],
        *,
        gravity: Array | None = None,
        base_pose: Array | None = None,
    ) -> PCSParams:
        """Build spatial PCS parameters from shared link specifications.

        Args:
            links: Non-empty sequence of constant, circular link
                specifications. Each reference strain must have shape ``(6,)``.
            gravity: Optional world-frame gravity vector with shape ``(3,)``.
            base_pose: Optional base translation and unit quaternion with shape
                ``(7,)``.

        Returns:
            Validated :class:`PCSParams` with canonical per-link stiffness and
            damping arrays of shape ``(num_links, 6, 6)``.

        Raises:
            ValueError: If no links are supplied; a link is not constant and
                circular; a reference strain has the wrong shape; a material
                source is incomplete; or an explicit matrix is not ``(6, 6)``.
        """
        if not links:
            raise ValueError("PCS requires at least one link.")
        specs = tuple(links)
        for index, spec in enumerate(specs):
            if spec.cross_section_geometry != CrossSectionGeometry.CIRCULAR:
                raise ValueError(f"PCS link {index} must have a circular section.")
            if spec.cross_section_profile_types != ("constant",):
                raise ValueError(f"PCS link {index} must have a constant radius.")
            if jnp.asarray(spec.reference_strain).shape != (6,):
                raise ValueError(
                    f"PCS link {index} reference_strain must have shape (6,)."
                )

        length = jnp.asarray([spec.length for spec in specs])
        density = jnp.asarray([spec.density for spec in specs])
        radius = jnp.asarray([spec.cross_section_coefficients[0] for spec in specs])
        reference_strain = jnp.asarray(
            [spec.reference_strain for spec in specs], dtype=float
        )
        area = jnp.pi * radius**2
        transverse = jnp.pi * radius**4 / 4.0
        polar = 2.0 * transverse
        young_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack(
                [
                    jnp.zeros_like(length),
                    transverse,
                    transverse,
                    area,
                    jnp.zeros_like(length),
                    jnp.zeros_like(length),
                ],
                axis=1,
            )
        )
        shear_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack(
                [
                    polar,
                    jnp.zeros_like(length),
                    jnp.zeros_like(length),
                    jnp.zeros_like(length),
                    area,
                    area,
                ],
                axis=1,
            )
        )
        damping_operator = length[:, None, None] * vmap(jnp.diag)(
            jnp.stack(
                [polar, 3.0 * transverse, 3.0 * transverse, 3.0 * area, area, area],
                axis=1,
            )
        )

        stiffness_items = []
        damping_items = []
        for index, spec in enumerate(specs):
            if spec.stiffness is not None:
                stiffness = jnp.asarray(spec.stiffness)
                if stiffness.shape != (6, 6):
                    raise ValueError(
                        f"PCS link {index} stiffness must have shape (6, 6)."
                    )
            else:
                stiffness = (
                    spec.young_modulus * young_operator[index]
                    + spec._resolved_shear_modulus() * shear_operator[index]
                )
            if spec.damping is not None:
                damping = jnp.asarray(spec.damping)
                if damping.shape != (6, 6):
                    raise ValueError(
                        f"PCS link {index} damping must have shape (6, 6)."
                    )
            elif spec.material_damping_coefficient is not None:
                damping = spec.material_damping_coefficient * damping_operator[index]
            else:
                damping = jnp.zeros_like(damping_operator[index])
            stiffness_items.append(stiffness)
            damping_items.append(damping)

        return PCSParams(
            base_pose=base_pose,
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
        structure: PCSStructure | None = None,
        gravity: Array | None = None,
        base_pose: Array | None = None,
        **kwargs: Any,
    ) -> "PCS":
        """Construct a spatial PCS model from shared link specifications.

        Args:
            links: Non-empty sequence of constant, circular link
                specifications.
            structure: Optional static PCS quadrature, strain-selection, and
                rotational-scaling configuration. Defaults to
                :class:`PCSStructure`.
            gravity: Optional world-frame gravity vector with shape ``(3,)``.
            base_pose: Optional base translation and unit quaternion with shape
                ``(7,)``.
            **kwargs: Additional keyword arguments forwarded to the PCS
                constructor, such as actuators or passive elements.

        Returns:
            A fully initialized spatial PCS model.

        Raises:
            TypeError: If a structure or constructor argument has an invalid
                type.
            ValueError: If a link specification, static structure, parameter
                array, or attached component is invalid.
        """
        return cls(
            params=cls.params_from_links(links, gravity=gravity, base_pose=base_pose),
            structure=structure,
            **kwargs,
        )

    def __init__(
        self,
        params: PCSParams,
        structure: PCSStructure | None = None,
        actuators: Actuator | tuple[Actuator, ...] | None = None,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
        **kwargs: Any,
    ):
        """Initialize a spatial PCS model from typed parameters.

        Args:
            params: Canonical dynamic link, gravity, and base-pose parameters.
            structure: Optional static quadrature, active-strain, and
                rotational-scaling configuration.
            actuators: Optional actuator or tuple of actuators attached to the
                model.
            passive_elements: Optional passive element or tuple of passive
                elements. Pass ``None`` or an empty tuple to disable them.
            **kwargs: Additional keyword arguments forwarded to
                :class:`BaseContinuumSoftRobot`.

        Raises:
            TypeError: If ``params``, a structure field, or an attached
                component has an invalid type.
            ValueError: If parameter shapes or values, the structure, or an
                attached component are invalid.
        """
        if not isinstance(params, PCSParams):
            raise TypeError("params must be a PCSParams instance.")
        params.validate()
        super().__init__(base_pose=params.base_pose, **kwargs)
        if structure is None:
            structure = PCSStructure()
        self.params = params
        self.scale_rotational_basis_by_length = bool(
            structure.scale_rotational_basis_by_length
        )

        # Number of segments
        num_segments = int(params.link.length.shape[0])
        if num_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {num_segments}")
        self.num_segments = num_segments

        num_strains = 6 * num_segments
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
            strain_selector = jnp.ones(num_strains, dtype=bool)
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
            if strain_selector.size != num_strains:
                raise ValueError(
                    f"strain_selector must have {num_strains} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(num_strains)
        self.B_xi_unscaled = build_active_dof_basis(strain_selector)
        self.B_xi = self._scaled_strain_basis(self.B_xi_unscaled)

        self.num_active_strains = jnp.sum(strain_selector)
        self.num_dofs = int(self.num_active_strains.item())
        active_dofs_per_segment = jnp.sum(
            strain_selector.reshape(self.num_segments, 6), axis=1
        )
        self._segment_dof_ends = tuple(
            int(value) for value in jnp.cumsum(active_dofs_per_segment).tolist()
        )

        reference_strain = jnp.asarray(params.link.reference_strain, dtype=jnp.float64)
        if reference_strain.size != num_strains:
            raise ValueError(
                "reference_strain must have "
                f"{num_strains} elements, got {reference_strain.size}."
            )
        self.xi_ref = reference_strain.reshape(num_strains)

        self._configure_actuation(actuators, passive_elements)

        self.precompute()

    @property
    def is_planar(self) -> bool:
        """Return whether the system is planar.

        Returns:
            Always ``False`` because spatial PCS uses SE(3) kinematics.
        """
        return False

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
            [
                1.0 / self.L,
                1.0 / self.L,
                1.0 / self.L,
                jnp.ones_like(self.L),
                jnp.ones_like(self.L),
                jnp.ones_like(self.L),
            ],
            axis=1,
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

    def _set_params(self, params: PCSParams) -> None:
        """Set cached runtime arrays from typed parameters."""
        # Initial position and orientation angle
        self.base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)
        p0 = self.base_pose
        p0 = jnp.asarray(p0, dtype=jnp.float64)
        if p0.size != 7:
            raise ValueError(f"base_pose must have shape (7,), got {p0.size}")
        self.g0 = poses.quaternion_pose_to_transform(p0)

        # Gravitational acceleration vector
        g = params.gravity
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 3:
            raise ValueError(f"gravity must be a vector of shape (3,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(3), g]
        )  # Add zeros for the orientation angles

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

    def _current_body_params(self) -> PCSParams:
        """Return the PCS body params, including for typed actuated wrappers."""
        if isinstance(self.params, PCSParams):
            return self.params
        body = getattr(self.params, "body", None)
        if isinstance(body, PCSParams):
            return body
        raise TypeError("model params do not contain PCSParams body fields.")

    def _with_pcs_params(
        self, params: PCSParams, stored_params: Any | None = None
    ) -> "PCS":
        """Return a copy with PCS body caches refreshed from typed parameters."""
        current_params = self._current_body_params()
        if not isinstance(params, PCSParams):
            raise TypeError("params must be a PCSParams instance.")
        params.validate_for_update()
        if params.link.length.shape != current_params.link.length.shape:
            raise ValueError(
                "length shape changes the model structure; construct a new PCS."
            )
        if (
            params.link.reference_strain.shape
            != current_params.link.reference_strain.shape
        ):
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PCS."
            )

        base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)
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
                m.base_pose,
                m.g0,
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
                base_pose,
                poses.quaternion_pose_to_transform(base_pose),
                jnp.concatenate([jnp.zeros(3, dtype=gravity.dtype), gravity]),
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

    def with_params(self, params: PCSParams) -> "PCS":
        """Return a model copy using a complete parameter PyTree.

        Args:
            params: Complete replacement parameters with the same number of
                segments and reference-strain layout as this model.

        Returns:
            A new PCS model with refreshed geometry, mass, material-response,
            stiffness, and damping caches. The original model is unchanged.

        Raises:
            TypeError: If ``params`` is not a :class:`PCSParams`.
            ValueError: If the replacement is invalid or changes the static
                segment or strain layout.
        """
        return self._with_pcs_params(params)

    def update_params(self, **updates: Any) -> "PCS":
        """Return a copy with selected top-level parameter fields replaced.

        Args:
            **updates: Fields of :class:`PCSParams` to replace, typically
                ``link``, ``gravity``, or ``base_pose``.

        Returns:
            A new validated PCS model containing the replacements.

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
                "length shape changes the model structure; construct a new PCS."
            )
        if (
            "link" in updates
            and jnp.asarray(updates["link"].reference_strain).shape
            != self.params.link.reference_strain.shape
        ):
            raise ValueError(
                "reference_strain shape changes the model structure; construct a new PCS."
            )
        return self.with_params(self.params.replace(**updates))

    def update_link_params(self, **updates: Any) -> "PCS":
        """Return a copy with selected continuum-link fields replaced.

        Args:
            **updates: Fields of :class:`ContinuumLinkParams` to replace, such
                as ``length``, ``density``, ``reference_strain``,
                ``cross_section``, ``stiffness``, or ``damping``.

        Returns:
            A new validated PCS model with refreshed dependent caches.

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
        """Map isotropic material variables to canonical PCS link matrices.

        Args:
            material: Scalar or per-segment Young's modulus, either shear
                modulus or Poisson's ratio, and optional material damping
                coefficient. Scalars are broadcast over all segments.

        Returns:
            A tuple ``(stiffness, damping)`` whose arrays both have shape
            ``(num_segments, 6, 6)``.

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
            A new PCS model containing the generated canonical link stiffness
            and damping matrices.

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

    def _with_refreshed_precomputed_matrices(self) -> "PCS":
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
        return eqx.tree_at(
            lambda m: (m.M_segments, m.K_full, m.K_active, m.D_full, m.D_active),
            updated_self,
            (M_segments, K_full, K_active, D_full, D_active),
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
    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        The translational strain components are interpreted in the material
        frame, with axial stretch along local x and shear along local y and z.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            g_s (Array): forward kinematics of the robot at point s, shape (4, 4) :
                [[  R,      p],
                [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        # Compute the point coordinate along the segment in the interval [0, L_i]
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(g_base_i: Array, i: Array) -> tuple[Array, Array]:
            """
            Compute the forward kinematics of the robot for the i-th segment. Either until the tip of the segment or until the point s_local.
            Args:
                g_base_i (Array): transformation matrix of the base of the segment, shape (4, 4).
                i (Array): index of the segment.

            Returns:
                g_i (Array): transformation matrix of the point s_local if i == segment_idx, otherwise the tip of the segment, shape (4, 4).
                g_i (Array): transformation matrix of the point s_local if i == segment_idx, otherwise the tip of the segment, shape (4, 4).
            """
            length_i = jnp.where(i == segment_idx, s_local, self.L[i])
            xi_i = xi[i]

            # Magnus expansion
            Magnus_i = length_i * xi_i

            # Compute the transformation matrix from the base of the segment until length_i using the exponential map
            g_step = se3.exp(Magnus_i, eps=self.global_eps)

            g_i = g_base_i @ g_step

            return g_i, g_i

        indices_link = jnp.arange(self.num_segments)

        g_ini = self.g0  # Initial position and orientation of the robot base

        _, g_ls = lax.scan(f=body_segment_i, init=g_ini, xs=indices_link)

        # # For debugging purposes, you can uncomment the following line to see the list of transformations
        # carry = g_ini
        # g_list = []
        # for i_segment in indices_link:
        #     g_tip_i, g_list_i = body_segment_i(carry, i_segment)
        #     carry = g_tip_i
        #     g_list.append(g_list_i)
        # g_list = jnp.array(g_list)

        g_s = g_ls[segment_idx]

        return g_s

    @eqx.filter_jit
    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the SE(3) pose at ``s``.

        For PCS, the local segment strain is constant, so
        ``d g(s) / ds = g(s) @ hat(xi_i)`` within segment ``i``.
        """
        _, gs = self._forward_kinematics_and_arc_length_derivative(q, s)
        return gs

    @eqx.filter_jit
    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the SE(3) pose and its arc-length derivative at ``s``.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, _ = self.classify_segment(s)
        xi_i = xi[segment_idx]
        g_s = self._forward_kinematics(q, s)
        return g_s, g_s @ se3.hat(xi_i)

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics of the robot at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            g_tips (Array): forward kinematics of the robot at all segment tips, shape (num_segments, 4, 4) :
                g_tip_i = [[  R,      p],
                           [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        def body_segment_i(g_base_i: Array, i: Array) -> tuple[Array, Array]:
            # Magnus expansion
            Magnus_Li = self.L[i] * xi[i]

            # Compute the transformation matrix from the base until the tip of the segment using the exponential map
            g_Li_rel = se3.exp(Magnus_Li, eps=self.global_eps)

            g_Li = g_base_i @ g_Li_rel

            return g_Li, g_Li

        indices_link = jnp.arange(self.num_segments)

        # Initial position and orientation of the robot base
        g_ini = self.g0

        _, g_tips = lax.scan(f=body_segment_i, init=g_ini, xs=indices_link)

        return g_tips

    @eqx.filter_jit
    def forward_kinematics_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            g_ps: forward kinematics of the robot at all points s_ps, shape (N, 4, 4) :
                g_si = [[  R,      p],
                        [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        # compute the forward kinematics at the tips
        g_tips = self.forward_kinematics_tips(q)

        # Compute the point coordinates along the segment in the interval [0, L_i]
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # collect the base transform for each segment (g0 for the first, previous tip otherwise)
        g_bases = jnp.concatenate([self.g0[None, :, :], g_tips[:-1]], axis=0)

        # compute the magnus expansion for each point
        magnus = s_local_ps.reshape(-1, 1) * xi[segment_indices]
        # compute the relative transform from the base of the segment to the point
        g_rel = vmap(lambda magnus_i: se3.exp(magnus_i, eps=self.global_eps))(magnus)
        # get the base transform for each point
        g_base_points = g_bases[segment_indices]

        # combine to get the absolute transforms
        g_ps = vmap(lambda g_base_i, g_rel_i: g_base_i @ g_rel_i)(g_base_points, g_rel)

        return g_ps

    @eqx.filter_jit
    def _compute_relative_segment_transforms(self, g_tips: Array) -> Array:
        """Return the SE(3) transforms from each segment base to its distal tip.

        The first entry is measured from the global base pose ``g0``; subsequent entries
        use the distal pose of the previous segment, mirroring the serial chain geometry.

        Args:
            g_tips (Array): absolute tip transforms of shape (num_segments, 4, 4).

        Returns:
            g_rels (Array): relative transforms of shape (num_segments, 4, 4).
        """
        g_tips = jnp.asarray(g_tips).reshape(self.num_segments, 4, 4)

        # prepend the base transform so every segment has its proximal pose available
        g_prev = jnp.concatenate([self.g0[None, :, :], g_tips[:-1]], axis=0)

        def rel_transform(g_prev_i: Array, g_curr_i: Array) -> Array:
            """
            Compute ``g_prev_i^{-1} @ g_curr_i`` without explicitly inverting ``g_prev_i``.
            This is more numerically stable and efficient than computing the inverse directly.
            Args:
                g_prev_i (Array): previous segment tip transform of shape (4, 4).
                g_curr_i (Array): current segment tip transform of shape (4, 4).

            Returns:
                g_rel (Array): relative transform of shape (4, 4).
            """
            R_prev = g_prev_i[:3, :3]
            p_prev = g_prev_i[:3, 3]

            R_curr = g_curr_i[:3, :3]
            p_curr = g_curr_i[:3, 3]

            R_rel = R_prev.T @ R_curr
            p_rel = R_prev.T @ (p_curr - p_prev)

            upper = jnp.concatenate([R_rel, p_rel[:, None]], axis=1)
            lower = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=g_curr_i.dtype)[None, :]
            g_rel = jnp.concatenate([upper, lower], axis=0)
            return g_rel

        g_rels = vmap(rel_transform)(g_prev, g_tips)
        return g_rels

    @eqx.filter_jit
    def inverse_kinematics(self, g_tips: Array) -> Array:
        """
        Recover generalized coordinates from the absolute tip poses of each segment.

        The routine converts each tip pose into a body-relative transform, extracts the
        corresponding twist via the SE(3) logarithmic map, and normalises by segment length
        to obtain the constant strain representation used by the PCS model.

        Args:
            g_tips (Array): homogeneous tip transforms of shape (num_segments, 4, 4).

        Returns:
            q (Array): generalized coordinates ``q`` of shape (num_active_strains,).
        """
        # Relative pose of each segment distal tip with respect to its proximal base
        g_rel = self._compute_relative_segment_transforms(g_tips)

        def _segment_inverse(g_rel_i: Array, length_i: Array) -> Array:
            """
            Map the relative tip pose of a segment to its constant strain twist.
            Args:
                g_rel_i (Array): relative transform of shape (4, 4).
                length_i (Array): length of the segment.

            Returns:
                xi (Array): strain twist of shape (6,).
            """
            xi_mag = se3.log(g_rel_i, eps=self.global_eps)
            return safe_divide(xi_mag, length_i, self.global_eps)

        # Convert every relative transform to segment twist parameters
        xi_segments = vmap(_segment_inverse)(g_rel, self.L)
        xi = xi_segments.reshape(-1)

        # Project the strain difference onto the active strain basis to obtain ``q``
        q = jnp.linalg.pinv(self.B_xi) @ (xi - self.xi_ref)

        return q

    @eqx.filter_jit
    def _final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 6, 6)

        Returns:
            J_selected (Array): Jacobian for the selected strains of shape (6, num_active_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(6, self.num_strains)

        return J_final

    @eqx.filter_jit
    def _J_local_tips(self, q: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian of the forward kinematics the tips of all segments, shape (num_segments, 6, num_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            J_prev: Array,
            i: Array,
        ) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)

            Ad_inv, T = constant_strain_se3._operators(
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
        Compute the Jacobian for the forward kinematics at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_strains)
        """
        # num points
        N = s_ps.shape[0]

        # compute the strain for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)

        # classify all points along the robot to the corresponding segment and compute local coordinates
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # compute the Jacobian at the tips
        J_tips = self._J_local_tips(q)  # shape (num_segments, 6, num_strains)

        # select the base Jacobian for each point (g0 for the first, previous tip otherwise)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        # select the other variables for each point
        xi_ps = xi[segment_indices]
        idx_ps = segment_indices

        # reshape for easier indexing
        J_base_ps = J_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            s_local: Array,
            J_base: Array,
        ) -> Array:
            Ad_inv, T = constant_strain_se3._operators(
                xi_i, s_local, self.global_eps, self.tangent_eps
            )

            return self._update_body_jacobian_step(J_base, i, Ad_inv, T)

        # vmap the segment integration over all points
        J_local_ps = vmap(integrate_segment)(idx_ps, xi_ps, s_local_ps, J_base_ps)

        # reshape back to (N, 6, num_strains)
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
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 6)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)
        zero_slice = jnp.zeros((6, 6), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            i: Array,
            xi_i: Array,
            arc_len: Array,
        ) -> Array:
            Ad_inv, T = constant_strain_se3._operators(
                xi_i, arc_len, self.global_eps, self.global_eps
            )

            return self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

        def scan_body(
            carry: tuple[Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array], Array]:
            J_prev, J_local, done = carry

            def compute_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                J_next = integrate_segment(J_prev, i, xi_i, arc_len)

                is_target = i == segment_idx
                J_target_next = jnp.where(is_target, J_next, J_local)
                done_next = jnp.logical_or(done, is_target)

                return (J_next, J_target_next, done_next), zero_slice

            def skip_branch(_: None) -> tuple[tuple[Array, Array, Array], Array]:
                return (J_prev, J_local, done), zero_slice

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
        Compute shared constant-strain terms for one PCS Jacobian step.

        Args:
            xi_i: Constant SE(3) strain of segment ``i``, shape ``(6,)``.
            arc_len: Local arc length over which the segment contribution is
                integrated. For segments before the target this is the full
                segment length; for the target segment this is ``s_local``.

        Returns:
            Tuple ``(Ad_inv, T)`` where ``Ad_inv`` is the inverse adjoint of the
            local segment transform and ``T`` is the SE(3) tangent operator.
            Both arrays have shape ``(6, 6)``.
        """
        return constant_strain_se3._operators(
            xi_i, arc_len, self.global_eps, self.global_eps
        )

    def _pcs_jacobian_arc_length_step_terms(
        self, xi_i: Array, arc_len: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute shared PCS terms for ``J`` and ``dJ/ds`` propagation.

        Args:
            xi_i: Constant SE(3) strain of segment ``i``, shape ``(6,)``.
            arc_len: Local arc length used for the segment step.

        Returns:
            Tuple ``(Ad_inv, T, dAd_inv_ds, dT_ds)``. ``Ad_inv`` and ``T`` are
            the primal Jacobian step terms; ``dAd_inv_ds`` and ``dT_ds`` are the
            analytical derivatives with respect to local arc length.
        """
        Ad_inv, T, adjoint = constant_strain_se3._operators(
            xi_i,
            arc_len,
            self.global_eps,
            self.global_eps,
            return_adjoint=True,
        )
        dAd_inv_ds = -se3.small_adjoint(xi_i) @ Ad_inv
        return Ad_inv, T, dAd_inv_ds, adjoint

    def _pcs_relative_pose(self, xi_i: Array, arc_len: Array) -> Array:
        """
        Compute the relative SE(3) pose for one constant-strain segment slice.

        Args:
            xi_i: Constant SE(3) strain of segment ``i``, shape ``(6,)``.
            arc_len: Local arc length of the slice.

        Returns:
            Relative homogeneous transform ``exp(arc_len * hat(xi_i))`` with
            shape ``(4, 4)``.
        """
        return se3.exp(arc_len * xi_i, eps=self.global_eps)

    def _update_body_jacobian_step(
        self, J_prev: Array, i: Array, Ad_inv: Array, T: Array
    ) -> Array:
        """
        Propagate the block body-frame Jacobian through one PCS segment step.

        Args:
            J_prev: Previous full block Jacobian, shape
                ``(num_segments, 6, 6)``.
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
        Propagate ``J`` and its time derivative through one PCS segment step.

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
        Ad_inv_dot = -se3.small_adjoint(eta) @ Ad_inv
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
        Compute body-frame Jacobian and SE(3) pose in a single segment scan.

        Args:
            q: Active generalized coordinates, shape ``(num_dofs,)``.
            s: Backbone arc-length coordinate.

        Returns:
            Tuple ``(g_s, J_body)`` where ``g_s`` is the pose at ``s`` and
            ``J_body`` maps active velocities to body-frame twists at ``s``.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
            g_prev, J_prev, g_target, J_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T = self._pcs_jacobian_step_terms(xi_i, arc_len)
                g_next = g_prev @ self._pcs_relative_pose(xi_i, arc_len)
                J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)

                is_target = i == segment_idx
                return (
                    g_next,
                    J_next,
                    jnp.where(is_target, g_next, g_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                return (g_prev, J_prev, g_target, J_target, done), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, g_target, J_target, _), _ = lax.scan(
            scan_body,
            (self.g0, zeros, self.g0, zeros, jnp.array(False, dtype=jnp.bool_)),
            indices,
        )
        return g_target, self._final_size_jacobian(J_target) @ self.B_xi

    def _jacobian_and_arc_length_derivative_bodyframe_with_pose(
        self, q: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """
        Compute pose, body-frame Jacobian, and arc-length derivative together.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
            g_prev, J_prev, g_target, J_target, Js_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T, dAd_inv_ds, dT_ds = self._pcs_jacobian_arc_length_step_terms(
                    xi_i, arc_len
                )
                g_next = g_prev @ self._pcs_relative_pose(xi_i, arc_len)
                J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)
                Js_next = self._update_body_jacobian_arc_length_derivative_step(
                    J_prev, i, Ad_inv, T, dAd_inv_ds, dT_ds
                )

                is_target = i == segment_idx
                return (
                    g_next,
                    J_next,
                    jnp.where(is_target, g_next, g_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.where(is_target, Js_next, Js_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], None]:
                return (g_prev, J_prev, g_target, J_target, Js_target, done), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, g_target, J_target, Js_target, _), _ = lax.scan(
            scan_body,
            (
                self.g0,
                zeros,
                self.g0,
                zeros,
                zeros,
                jnp.array(False, dtype=jnp.bool_),
            ),
            indices,
        )

        J = self._final_size_jacobian(J_target) @ self.B_xi
        Js = self._final_size_jacobian(Js_target) @ self.B_xi
        return g_target, J, Js

    def _jacobian_and_time_derivative_bodyframe_with_pose(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """Compute pose, body-frame Jacobian, and time derivative together."""
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
            g_prev, J_prev, Jd_prev, g_target, J_target, Jd_target, done = carry

            def compute_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
                xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
                xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
                L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
                arc_len = jnp.where(i == segment_idx, s_local, L_i)

                Ad_inv, T, Td = constant_strain_se3._operators(
                    xi_i,
                    arc_len,
                    self.global_eps,
                    self.global_eps,
                    xid_i,
                )
                g_next = g_prev @ self._pcs_relative_pose(xi_i, arc_len)

                J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                    J_prev, Jd_prev, i, xid_i, Ad_inv, T, Td
                )

                is_target = i == segment_idx
                return (
                    g_next,
                    J_next,
                    Jd_next,
                    jnp.where(is_target, g_next, g_target),
                    jnp.where(is_target, J_next, J_target),
                    jnp.where(is_target, Jd_next, Jd_target),
                    jnp.logical_or(done, is_target),
                ), None

            def skip_branch(
                _: None,
            ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], None]:
                return (
                    g_prev,
                    J_prev,
                    Jd_prev,
                    g_target,
                    J_target,
                    Jd_target,
                    done,
                ), None

            return lax.cond(done, skip_branch, compute_branch, operand=None)

        indices = jnp.arange(self.num_segments, dtype=segment_idx.dtype)
        (_, _, _, g_target, J_target, Jd_target, _), _ = lax.scan(
            scan_body,
            (
                self.g0,
                zeros,
                zeros,
                self.g0,
                zeros,
                zeros,
                jnp.array(False, dtype=jnp.bool_),
            ),
            indices,
        )

        J = self._final_size_jacobian(J_target) @ self.B_xi
        Jd = self._final_size_jacobian(Jd_target) @ self.B_xi
        return g_target, J, Jd

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
            shape ``(6, num_active_strains)``.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)
        zero_slice = jnp.zeros((6, 6), dtype=xi.dtype)

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

                Ad_inv, T, dAd_inv_ds, dT_ds = self._pcs_jacobian_arc_length_step_terms(
                    xi_i, arc_len
                )
                J_next = self._update_body_jacobian_step(J_prev, i, Ad_inv, T)
                Js_next = self._update_body_jacobian_arc_length_derivative_step(
                    J_prev, i, Ad_inv, T, dAd_inv_ds, dT_ds
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
            ``(6, num_active_strains)``.
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
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_active_strains)
        """
        J_local_ps_ = self._J_local_abscissa_batched(
            q, s_ps
        )  # shape (N, 6, num_strains)

        J_local_ps = jnp.einsum(
            "ijk, kl->ijl", J_local_ps_, self.B_xi
        )  # shape (N, 6, num_active_strains)

        return J_local_ps

    def _rotation_adjoint_from_pose(self, g_s: Array) -> Array:
        """Adjoint of the pose rotation, with zero translational coupling."""
        g_rot = jnp.block(
            [[g_s[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
        )
        return se3.adjoint(g_rot)

    def _body_jacobian_to_inertial(self, g_s: Array, J_local: Array) -> Array:
        """Rotate a PCS body-frame Jacobian into the inertial frame."""
        return self._rotation_adjoint_from_pose(g_s) @ J_local

    @eqx.filter_jit
    def jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
        """
        g_s, J_local = self._jacobian_bodyframe_with_pose(q, s)
        return self._body_jacobian_to_inertial(g_s, J_local)

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
            shape ``(6, num_active_strains)``.
        """
        g_s, J_local, Js_local = (
            self._jacobian_and_arc_length_derivative_bodyframe_with_pose(q, s)
        )
        Ad_g = self._rotation_adjoint_from_pose(g_s)

        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, _ = self.classify_segment(s)
        eta_rot_s = jnp.concatenate([xi[segment_idx, :3], jnp.zeros(3, dtype=xi.dtype)])
        Ad_g_s = Ad_g @ se3.small_adjoint(eta_rot_s)

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
            ``(6, num_active_strains)``.
        """
        g_s, J_local, Js_local = (
            self._jacobian_and_arc_length_derivative_bodyframe_with_pose(q, s)
        )
        Ad_g = self._rotation_adjoint_from_pose(g_s)

        xi = self.strain(q).reshape(self.num_segments, 6)
        segment_idx, _ = self.classify_segment(s)
        eta_rot_s = jnp.concatenate([xi[segment_idx, :3], jnp.zeros(3, dtype=xi.dtype)])
        Ad_g_s = Ad_g @ se3.small_adjoint(eta_rot_s)
        Js = Ad_g_s @ J_local + Ad_g @ Js_local
        return Js

    @eqx.filter_jit
    def jacobian_inertialframe_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a batch of points s_ps along the robot in the inertial frame.
        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_global_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_active_strains)
        """
        # compute the Jacobian in the body frame
        J_local_ps = self.jacobian_bodyframe_abscissa_batched(
            q, s_ps
        )  # shape (N, 6, num_active_strains)

        g_ps = self.forward_kinematics_abscissa_batched(q, s_ps)  # shape (N, 4, 4)
        # construct g with zero translation for the Adjoint transformation
        g_rot_ps = jnp.block(
            [
                [g_ps[:, :3, :3], jnp.zeros((g_ps.shape[0], 3, 1))],
                [jnp.zeros((g_ps.shape[0], 1, 3)), jnp.ones((g_ps.shape[0], 1, 1))],
            ]
        )  # shape (N, 4, 4)
        Ad_g_ps = vmap(se3.adjoint)(g_rot_ps)  # shape (N, 6, 6)

        J_global_ps = jnp.einsum(
            "nij, njk->nik", Ad_g_ps, J_local_ps
        )  # shape (N, 6, num_active_strains)

        return J_global_ps

    @eqx.filter_jit
    def jacobian_tips(self, q: Array) -> Array:
        """
        Compute inertial-frame Jacobians at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_tips (Array): inertial-frame Jacobians at each segment tip, shape
                (num_segments, 6, num_active_strains).
        """
        J_local_tips = jnp.einsum("ijk,kl->ijl", self._J_local_tips(q), self.B_xi)
        g_tips = self.forward_kinematics_tips(q)

        def rotate_pair(g_i: Array, J_i: Array) -> Array:
            R = g_i[:3, :3]
            g_rot = jnp.block(
                [
                    [R, jnp.zeros((3, 1), dtype=R.dtype)],
                    [jnp.zeros((1, 3), dtype=R.dtype), jnp.ones((1, 1), dtype=R.dtype)],
                ]
            )
            return se3.adjoint(g_rot) @ J_i

        return vmap(rotate_pair)(g_tips, J_local_tips)

    @eqx.filter_jit
    def _J_Jd_local_tips(self, q: Array, qd: Array) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 6, num_strains)
            Jd_local_tips (Array): Time-derivative of the Jacobian at point s, shape (num_segments, 6, num_strains)
        """
        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        # precompute the fused Lie-algebra expressions for all segment tips
        Ad_inv_tips, T_tips, Td_tips = vmap(
            lambda xi_i, xid_i, L_i: constant_strain_se3._operators(
                xi_i,
                L_i,
                self.global_eps,
                self.tangent_eps,
                xid_i,
            )
        )(xi, xid, self.L)

        # initialize zeros
        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(
            carry: tuple[Array, Array],
            i: Array,
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_prev = carry

            # extract the current segment variables
            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)

            J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                J_prev, Jd_prev, i, xid_i, Ad_inv_i, T_i, Td_i
            )

            return (J_next, Jd_next), (J_next, Jd_next)

        carry_init = (zeros, zeros)

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        (_, _), (J_local_tips, Jd_local_tips) = lax.scan(scan_body, carry_init, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_local_tips)
        Jd_local_tips = vmap(self._final_size_jacobian)(Jd_local_tips)

        return J_local_tips, Jd_local_tips

    @eqx.filter_jit
    def _J_Jd_local_abscissa_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_strains)
            Jd_local_ps (Array): Time-derivative of the Jacobians, shape (N, 6, num_strains)
        """
        # num points
        N = s_ps.shape[0]

        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        # classify all points along the robot to the corresponding segment and compute local coordinates
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # compute the Jacobian at the tips
        J_tips, Jd_tips = self._J_Jd_local_tips(
            q, qd
        )  # shape (num_segments, 6, num_strains)

        # select the base Jacobian for each point (g0 for the first, previous tip otherwise)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)
        Jd_bases = jnp.concatenate([jnp.zeros_like(Jd_tips[:1]), Jd_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        Jd_base_ps = Jd_bases[segment_indices]
        # select the other variables for each point
        xi_ps, xid_ps = xi[segment_indices], xid[segment_indices]
        idx_ps = segment_indices

        # reshape for easier indexing
        J_base_ps = J_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)
        Jd_base_ps = Jd_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            xid_i: Array,
            s_local: Array,
            J_base: Array,
            Jd_base: Array,
        ) -> tuple[Array, Array]:
            Ad_inv, T, Td = constant_strain_se3._operators(
                xi_i,
                s_local,
                self.global_eps,
                self.tangent_eps,
                xid_i,
            )

            J_next, Jd_next = self._update_body_jacobian_time_derivative_step(
                J_base, Jd_base, i, xid_i, Ad_inv, T, Td
            )

            return J_next, Jd_next

        # vmap the segment integration over all points
        J_local_ps, Jd_local_ps = vmap(integrate_segment)(
            idx_ps, xi_ps, xid_ps, s_local_ps, J_base_ps, Jd_base_ps
        )

        # reshape back to (N, 6, num_strains)
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
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
            Jd_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (6, num_active_strains)
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)

        segment_idx, s_local = self.classify_segment(s)

        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)
        zero_slice = jnp.zeros((6, 6), dtype=xi.dtype)

        def integrate_segment(
            J_prev: Array,
            Jd_prev: Array,
            i: Array,
            xi_i: Array,
            xid_i: Array,
            arc_len: Array,
        ) -> tuple[Array, Array]:
            Ad_inv, T, Td = constant_strain_se3._operators(
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

        J_local = self._final_size_jacobian(J_target) @ self.B_xi
        Jd_local = self._final_size_jacobian(Jd_target) @ self.B_xi

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
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_active_strains)
            Jd_local_ps (Array): Time-derivative of the Jacobians, shape (N, 6, num_active_strains)
        """
        J_local_ps_, Jd_local_ps_ = self._J_Jd_local_abscissa_batched(
            q, qd, s_ps
        )  # shape (N, 6, num_strains)

        J_local_ps = jnp.einsum(
            "ijk, kl->ijl", J_local_ps_, self.B_xi
        )  # shape (N, 6, num_active_strains)
        Jd_local_ps = jnp.einsum(
            "ijk, kl->ijl", Jd_local_ps_, self.B_xi
        )  # shape (N, 6, num_active_strains)

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
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_strains)
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (6, num_active_strains)
        """
        J_local, Jd_local = self.jacobian_and_time_derivative_bodyframe(q, qd, s)
        g_s = self._forward_kinematics(q, s)
        Ad_g = self._rotation_adjoint_from_pose(g_s)

        # compute the body twist eta
        eta_body = J_local @ qd

        # compute the time-derivative of the Adjoint transformation matrix
        omega = eta_body[:3]
        eta_rot = jnp.concatenate([omega, jnp.zeros(3, dtype=eta_body.dtype)])
        Ad_g_dot = Ad_g @ se3.small_adjoint(eta_rot)

        # rotate both J and Jd to the inertial frame
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
            J_global_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_active_strains)
            Jd_global_ps (Array): Time-derivative of the Jacobians, shape (N, 6, num_active_strains)
        """
        J_local_ps, Jd_local_ps = (
            self.jacobian_and_time_derivative_bodyframe_abscissa_batched(q, qd, s_ps)
        )  # shape (N, 6, num_active_strains)

        g_ps = self.forward_kinematics_abscissa_batched(q, s_ps)  # shape (N, 4, 4)
        # construct g with zero translation for the Adjoint transformation
        g_rot_ps = jnp.block(
            [
                [g_ps[:, :3, :3], jnp.zeros((g_ps.shape[0], 3, 1))],
                [jnp.zeros((g_ps.shape[0], 1, 3)), jnp.ones((g_ps.shape[0], 1, 1))],
            ]
        )  # shape (N, 4, 4)
        Ad_g_ps = vmap(se3.adjoint)(g_rot_ps)  # shape (N, 6, 6)

        # compute the body twist eta for all points
        eta_body_ps = jnp.einsum("ijk, k->ij", J_local_ps, qd)  # shape (N, 6)

        # compute the time-derivative of the Adjoint transformation matrix for all points
        omega_ps = eta_body_ps[:, :3]
        eta_rot_ps = jnp.concatenate(
            [omega_ps, jnp.zeros((omega_ps.shape[0], 3), dtype=omega_ps.dtype)],
            axis=1,
        )  # shape (N, 6)
        Ad_g_dot_ps = jnp.einsum(
            "nij, njk->nik", Ad_g_ps, vmap(se3.small_adjoint)(eta_rot_ps)
        )  # shape (N, 6, 6)

        # rotate both J and Jd to the inertial frame
        J_global_ps = jnp.einsum(
            "nij, njk->nik", Ad_g_ps, J_local_ps
        )  # shape (N, 6, num_active_strains)
        Jd_global_ps = jnp.einsum("nij, njk->nik", Ad_g_ps, Jd_local_ps) + jnp.einsum(
            "nij, njk->nik", Ad_g_dot_ps, J_local_ps
        )  # shape (N, 6, num_active_strains)
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
    def jacobian_abscissa_batched(self, q: Array, s_ps: Array) -> Array:
        """Compute inertial-frame Jacobians at multiple arc-length positions.

        Args:
            q: Active generalized strains with shape
                ``(num_active_strains,)``.
            s_ps: Backbone coordinates with shape ``(num_points,)``.

        Returns:
            Inertial-frame Jacobians with shape
            ``(num_points, 6, num_active_strains)``.
        """
        return self.jacobian_inertialframe_abscissa_batched(q, s_ps)

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
            ``(num_points, 6, num_active_strains)``.
        """
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
            i (Array): index of the segment as array of shape ()

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        # Cross-sectional area for a circular cross-section
        A_i = jnp.pi * self.r[i] ** 2
        return A_i

    @eqx.filter_jit
    def _local_second_moment_of_area(self, i: Array) -> Array:
        """
        Compute second moments for the i-th segment's solid circular section.

        The local x-axis is longitudinal, so ``I_xx`` is the polar second
        moment and ``I_yy = I_zz`` are the transverse second moments.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            I_i (Array): local second moment of area of the i-th segment as array of shape (3, ) where the entries correspond to [I_xx, I_yy, I_zz]
        """
        # Second moment of area for a circular cross-section
        I_i = jnp.array(
            [
                jnp.pi * self.r[i] ** 4 / 2,  # I_xx
                jnp.pi * self.r[i] ** 4 / 4,  # I_yy
                jnp.pi * self.r[i] ** 4 / 4,  # I_zz
            ]
        )
        return I_i

    def _compute_local_mass_matrix(self, i: Array) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        The rotational entries use the solid circular cross-section's moments
        expressed about the longitudinal local x-axis and transverse y/z axes.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            M_i (Array): local mass matrix of shape (6, 6) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self._local_cross_sectional_area(i)  # Cross-sectional area
        I_i = self._local_second_moment_of_area(
            i
        )  # Second moment of area as array of shape (3, )

        M_i = rho_i * jnp.diag(jnp.array([I_i[0], I_i[1], I_i[2], A_i, A_i, A_i]))
        return M_i

    @eqx.filter_jit
    def _local_mass_matrix(self, i: Array) -> Array:
        """
        Return the cached local mass matrix for the i-th segment.

        Args:
            i (Array): index of the segment as array of shape ()

        Returns:
            M_i (Array): local mass matrix of shape (6, 6) for the i-th segment
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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the jacobian for each quadrature point
        J_ps = self._J_local_abscissa_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def B_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def B_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th jacobian
                J_ij = J_ps[i, j]

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                return B_ij

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_i = vmap(B_ij)(jnp.arange(1, self.num_integration_points - 1))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # B_blocks_i = jnp.stack(
            #     [B_j(j) for j in range(self.num_integration_points)], axis=0
            # )

            return B_blocks_i

        B_blocks_tot = vmap(B_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # B_blocks_tot = jnp.stack([B_i(i) for i in range(self.num_segments)], axis=0)

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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the jacobian and its time-derivative for each quadrature point
        J_ps, Jd_ps = self._J_Jd_local_abscissa_batched(
            q, qd, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_integration_points, *Jd_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def C_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def C_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]
                Jd_ij = Jd_ps[i, j]

                C_ij = (
                    Ws_ij
                    * J_ij.T
                    @ (M_i @ Jd_ij + se3.coadjoint(J_ij @ self.B_xi @ qd) @ M_i @ J_ij)
                )
                return C_ij

            # we can skip the first and last quadrature points since their weight is zero
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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_abscissa_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_integration_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        # compute the jacobian for each quadrature point
        J_ps = self._J_local_abscissa_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_integration_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def G_i(i: Array) -> Array:
            M_i = self._local_mass_matrix(i)

            def G_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]

                # compute the inverse Adjoint transformation matrix at point s
                Ad_g_inv_ij = se3.adjoint_inverse(g_ij)

                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g
                return G_ij

            # we can skip the first and last quadrature points since their weight is zero
            G_blocks_segment_i = vmap(G_ij)(
                jnp.arange(1, self.num_integration_points - 1)
            )

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # G_blocks_segment_i = jnp.stack(
            #     [G_j(j) for j in range(self.num_integration_points)], axis=0
            # )

            return G_blocks_segment_i

        G_blocks_tot = vmap(G_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # G_blocks_tot = jnp.stack(
        #     [G_i(i) for i in range(self.num_segments)], axis=0
        # )

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
        transverse = jnp.pi * self.r**4 / 4.0
        polar = 2.0 * transverse
        zeros = jnp.zeros_like(self.L)
        young = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack([zeros, transverse, transverse, area, zeros, zeros], axis=1)
        )
        shear = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack([polar, zeros, zeros, zeros, area, area], axis=1)
        )
        damping = self.L[:, None, None] * vmap(jnp.diag)(
            jnp.stack(
                [polar, 3.0 * transverse, 3.0 * transverse, 3.0 * area, area, area],
                axis=1,
            )
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
        return self.D_active + self.passive_damping_matrix(q)

    def _elastic_energy(self, q: Array) -> Array:
        """Return body strain energy plus installed passive-element energy."""
        return 0.5 * q @ self.K_active @ q + self.passive_elastic_energy(q)

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
        """Return one spatial routed-path length gradient density."""
        active = (start_segment_index <= segment_index) & (
            segment_index <= end_segment_index
        )
        offset = jnp.append(routing.offset(path_params, s), 1.0)
        offset_derivative = jnp.append(routing.derivative(path_params, s), 1.0)
        tangent_unnormalized = (offset_derivative + se3.hat(strain) @ offset)[:-1]
        tangent = safe_normalize(tangent_unnormalized, eps=self.global_eps)
        basis = jnp.hstack([so3.skew(offset[:-1]) @ tangent, tangent])
        return active * basis

    def _threadlike_moment_matrix(self, q: Array, routing: ThreadlikeRouting) -> Array:
        """Integrate raw routed-length moment arms in the PCS strain basis."""
        params = routing.params
        count = params.num_paths
        if count == 0:
            return jnp.zeros((self.num_dofs, 0), dtype=q.dtype)
        strains = self.strain(q).reshape((self.num_segments, 6))

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
        """Integrate raw threadlike path lengths without signed work scaling."""
        params = routing.params
        if params.num_paths == 0:
            return jnp.zeros((0,), dtype=q.dtype)
        strains = self.strain(q).reshape((self.num_segments, 6))

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
                    offset = jnp.append(routing.offset(path_params, s), 1.0)
                    derivative = jnp.append(routing.derivative(path_params, s), 1.0)
                    tangent = (derivative + se3.hat(strains[segment_index]) @ offset)[
                        :-1
                    ]
                    return active * safe_norm(tangent)

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
        """Return spatial positions of all routed paths at backbone coordinate ``s``."""
        params = routing.params
        if params.num_paths == 0:
            return jnp.zeros((0, 3), dtype=q.dtype)

        def path_position(
            path_params: BaseThreadlikeRoutingParams,
            start_segment_index: Array,
            end_segment_index: Array,
        ) -> Array:
            start = self.L_cum[start_segment_index]
            end = self.L_cum[end_segment_index + 1]
            s_clamped = jnp.clip(s, start, end)
            pose = self.forward_kinematics(q, s_clamped)
            point = pose @ jnp.append(routing.offset(path_params, s_clamped), 1.0)
            return point[:-1]

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
        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_abscissa_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_integration_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        def U_G_i(i: Array) -> Array:
            rho_i = self.rho[i]
            A_i = self._local_cross_sectional_area(i)  # Cross-sectional area

            def U_G_ij(j: Array) -> Array:
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]

                p_j = jnp.concatenate(
                    [jnp.zeros(3), g_ij[:3, 3]]
                )  # Add zeros for the orientation angles
                U_G_ij = -Ws_ij * rho_i * A_i * jnp.dot(p_j, self.g)
                return U_G_ij

            # we can skip the first and last quadrature points since their weight is zero
            U_G_blocks_segment_i = vmap(U_G_ij)(
                jnp.arange(1, self.num_integration_points - 1)
            )

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # U_G_blocks_segment_i = jnp.stack(
            #     [U_G_j(j) for j in range(self.num_integration_points)], axis=0
            # )

            return U_G_blocks_segment_i

        U_G_blocks_tot = vmap(U_G_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # U_G_blocks_tot = jnp.stack(
        #     [U_G_i(i) for i in range(self.num_segments)], axis=0
        # )

        U_G = jnp.sum(U_G_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points

        return U_G

    @eqx.filter_jit
    def _active_J_Jd_local_tips_from_strain(
        self,
        xi: Array,
        xid: Array,
        prepared_adjoint_powers: constant_strain_se3._PreparedAdjointPowers,
        B_xi_segments: Array,
        qd: Array,
        convective_only_jd: bool = False,
    ) -> tuple[Array, Array, Array]:
        """
        Propagate active-coordinate local Jacobians to every segment tip.

        The recurrence remains in each segment's body frame and uses prepared
        constant-strain operators. It can propagate either the complete
        Jacobian time derivative or only its contraction with ``qd``. The
        returned inverse adjoints let dynamics-only callers propagate local
        quantities, such as gravity, without constructing absolute poses.

        Args:
            xi: Segment strains with shape ``(self.num_segments, 6)``.
            xid: Segment strain rates with shape ``(self.num_segments, 6)``.
            prepared_adjoint_powers: Prepared constant-strain operators for
                every segment.
            B_xi_segments: Active strain bases with shape
                ``(self.num_segments, 6, self.num_dofs)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.
            convective_only_jd: If true, propagate ``Jd @ qd`` instead of the
                complete Jacobian time derivative.

        Returns:
            Tuple ``(J_tips, Jd_or_Jd_qd_tips, Ad_inv_tips)``. ``J_tips`` has
            shape ``(self.num_segments, 6, self.num_dofs)``. The second array
            has the same shape when ``convective_only_jd`` is false and shape
            ``(self.num_segments, 6)`` otherwise. ``Ad_inv_tips`` contains the
            inverse adjoint of each segment-tip transform with shape
            ``(self.num_segments, 6, 6)``.
        """
        zeros = jnp.zeros((6, self.num_dofs), dtype=xi.dtype)
        operators_tips = vmap(
            lambda prepared_adjoint_powers_i, L_i: (
                constant_strain_se3._kinematic_operators_from_prepared(
                    prepared_adjoint_powers_i,
                    L_i,
                    self.global_eps,
                    self.tangent_eps,
                    convective_only=convective_only_jd,
                )
            )
        )(prepared_adjoint_powers, self.L)
        if convective_only_jd:
            Ad_inv_tips, Ad_inv_T_tips, Td_tips = operators_tips
            T_tips = None
            Jd_or_Jd_qd_zeros = jnp.zeros((6,), dtype=xi.dtype)
        else:
            Ad_inv_tips, Ad_inv_T_tips, T_tips, Td_tips = operators_tips
            Jd_or_Jd_qd_zeros = zeros

        def scan_body(
            carry: tuple[Array, Array], i: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            J_prev, Jd_or_Jd_qd_prev = carry

            xid_i = lax.dynamic_index_in_dim(xid, i, axis=0, keepdims=False)
            B_xi_i = lax.dynamic_index_in_dim(B_xi_segments, i, axis=0, keepdims=False)
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            Ad_inv_T_i = lax.dynamic_index_in_dim(
                Ad_inv_T_tips, i, axis=0, keepdims=False
            )
            Td_i = lax.dynamic_index_in_dim(Td_tips, i, axis=0, keepdims=False)

            J_segment = Ad_inv_T_i @ B_xi_i
            J_next = Ad_inv_i @ J_prev + J_segment

            eta = Ad_inv_T_i @ xid_i

            if convective_only_jd:
                Jd_qd_next = (
                    Ad_inv_i @ Jd_or_Jd_qd_prev
                    - se3.small_adjoint_action(eta, Ad_inv_i @ (J_prev @ qd))
                    + Ad_inv_i @ (Td_i @ xid_i)
                )
                Jd_or_Jd_qd_next = Jd_qd_next
            else:
                assert T_tips is not None
                Ad_inv_dot = -se3.small_adjoint(eta) @ Ad_inv_i
                T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)
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
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.

        Returns:
            Tuple ``(g_ps, J_ps, Jd_ps)``. ``g_ps`` contains SE(3) poses with
            shape ``(self.num_segments, self.num_gauss_points, 4, 4)``.
            ``J_ps`` contains body-frame Jacobians in active generalized
            coordinates with shape
            ``(self.num_segments, self.num_gauss_points, 6, self.num_dofs)``.
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
        Return weights, poses, active Jacobians, and derivative information.

        Args:
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.
            convective_only_jd: If true, return ``Jd @ qd`` directly instead
                of materializing the complete Jacobian time derivative.

        Returns:
            Tuple ``(weights, poses, jacobians, jacobian_derivatives)``.
            ``weights`` has shape
            ``(self.num_segments, self.num_gauss_points)``; ``poses`` has
            shape ``(self.num_segments, self.num_gauss_points, 4, 4)``; and
            ``jacobians`` has shape
            ``(self.num_segments, self.num_gauss_points, 6, self.num_dofs)``.
            If ``convective_only_jd`` is true, the fourth array contains
            ``Jd @ qd`` with shape
            ``(self.num_segments, self.num_gauss_points, 6)``. Otherwise it
            contains the complete Jacobian derivatives with the same shape as
            ``jacobians``.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)
        B_xi_segments = self.B_xi.reshape(self.num_segments, 6, self.num_dofs)
        prepared_adjoint_powers = vmap(constant_strain_se3._prepare_adjoint_powers)(
            xi, xid
        )

        Xs_inner, Ws_inner = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_inner - self.L_cum[:-1, None]

        def scan_fk(g_base: Array, i: Array) -> tuple[Array, Array]:
            xi_i = lax.dynamic_index_in_dim(xi, i, axis=0, keepdims=False)
            L_i = lax.dynamic_index_in_dim(self.L, i, axis=0, keepdims=False)
            g_tip = g_base @ se3.exp(L_i * xi_i, eps=self.global_eps)
            return g_tip, g_base

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, g_bases = lax.scan(scan_fk, self.g0, indices)

        def segment_poses(g_base_i: Array, xi_i: Array, s_local_i: Array) -> Array:
            def pose_at_s(s_local_ij: Array) -> Array:
                g_rel = se3.exp(s_local_ij * xi_i, eps=self.global_eps)
                return g_base_i @ g_rel

            return vmap(pose_at_s)(s_local_i)

        g_ps = vmap(segment_poses)(g_bases, xi, s_local)

        J_tips, Jd_or_Jd_qd_tips, _ = self._active_J_Jd_local_tips_from_strain(
            xi,
            xid,
            prepared_adjoint_powers,
            B_xi_segments,
            qd,
            convective_only_jd=convective_only_jd,
        )
        zeros_tip = jnp.zeros_like(J_tips[:1])
        Jd_or_Jd_qd_zeros_tip = jnp.zeros_like(Jd_or_Jd_qd_tips[:1])
        J_bases = jnp.concatenate([zeros_tip, J_tips[:-1]], axis=0)
        Jd_or_Jd_qd_bases = jnp.concatenate(
            [Jd_or_Jd_qd_zeros_tip, Jd_or_Jd_qd_tips[:-1]], axis=0
        )

        def segment_jacobians(
            xi_i: Array,
            xid_i: Array,
            prepared_adjoint_powers_i: constant_strain_se3._PreparedAdjointPowers,
            B_xi_i: Array,
            J_base_i: Array,
            Jd_or_Jd_qd_base_i: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array]:
            def jacobian_at_s(s_local_ij: Array) -> tuple[Array, Array]:
                operators = constant_strain_se3._kinematic_operators_from_prepared(
                    prepared_adjoint_powers_i,
                    s_local_ij,
                    self.global_eps,
                    self.tangent_eps,
                    convective_only=convective_only_jd,
                )
                if convective_only_jd:
                    Ad_inv, Ad_inv_T, Td = operators
                    T = None
                else:
                    Ad_inv, Ad_inv_T, T, Td = operators
                J_segment = Ad_inv_T @ B_xi_i
                J_next = Ad_inv @ J_base_i + J_segment

                eta = Ad_inv_T @ xid_i
                Ad_inv_dot = -se3.small_adjoint(eta) @ Ad_inv

                if convective_only_jd:
                    Jd_qd_next = (
                        Ad_inv @ Jd_or_Jd_qd_base_i
                        + Ad_inv_dot @ (J_base_i @ qd)
                        + Ad_inv @ (Td @ xid_i)
                    )
                    Jd_or_Jd_qd_next = Jd_qd_next
                else:
                    assert T is not None
                    Jd_segment = (Ad_inv_dot @ T + Ad_inv @ Td) @ B_xi_i
                    Jd_next = (
                        Ad_inv @ Jd_or_Jd_qd_base_i + Ad_inv_dot @ J_base_i + Jd_segment
                    )
                    Jd_or_Jd_qd_next = Jd_next

                return J_next, Jd_or_Jd_qd_next

            return vmap(jacobian_at_s)(s_local_i)

        J_ps, Jd_or_Jd_qd_ps = vmap(segment_jacobians)(
            xi,
            xid,
            prepared_adjoint_powers,
            B_xi_segments,
            J_bases,
            Jd_or_Jd_qd_bases,
            s_local,
        )

        return Ws_inner, g_ps, J_ps, Jd_or_Jd_qd_ps

    @eqx.filter_jit
    def _dynamics_integration_kinematics(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Return dynamics-only quadrature kinematics in local body frames.

        Unlike :meth:`_integration_kinematics`, this path does not construct
        absolute SE(3) poses. It propagates gravity between segment frames with
        inverse adjoints and directly computes the convective contraction
        ``Jd @ qd`` needed by :meth:`dynamics_terms`.

        Args:
            q: Active generalized coordinates, shape ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape ``(self.num_dofs,)``.

        Returns:
            Tuple ``(weights, gravity, jacobians, jacobian_dot_qd)``.
            ``weights`` has shape
            ``(self.num_segments, self.num_gauss_points)``; ``gravity`` and
            ``jacobian_dot_qd`` have shape
            ``(self.num_segments, self.num_gauss_points, 6)``; and
            ``jacobians`` has shape
            ``(self.num_segments, self.num_gauss_points, 6, self.num_dofs)``.
        """
        xi = self.strain(q).reshape(self.num_segments, 6)
        xid = (self.B_xi @ qd).reshape(self.num_segments, 6)
        B_xi_segments = self.B_xi.reshape(self.num_segments, 6, self.num_dofs)
        prepared_adjoint_powers = vmap(constant_strain_se3._prepare_adjoint_powers)(
            xi, xid
        )

        Xs_inner, Ws_inner = vmap(
            scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local = Xs_inner - self.L_cum[:-1, None]

        J_tips, Jd_qd_tips, Ad_inv_tips = self._active_J_Jd_local_tips_from_strain(
            xi,
            xid,
            prepared_adjoint_powers,
            B_xi_segments,
            qd,
            convective_only_jd=True,
        )
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)
        Jd_qd_bases = jnp.concatenate(
            [jnp.zeros_like(Jd_qd_tips[:1]), Jd_qd_tips[:-1]], axis=0
        )

        gravity_base_initial = se3.adjoint_inverse(self.g0) @ self.g

        def scan_gravity(gravity_base: Array, Ad_inv_tip: Array) -> tuple[Array, Array]:
            gravity_tip = Ad_inv_tip @ gravity_base
            return gravity_tip, gravity_base

        _, gravity_bases = lax.scan(scan_gravity, gravity_base_initial, Ad_inv_tips)

        def segment_kinematics(
            xid_i: Array,
            prepared_adjoint_powers_i: constant_strain_se3._PreparedAdjointPowers,
            B_xi_i: Array,
            gravity_base_i: Array,
            J_base_i: Array,
            Jd_qd_base_i: Array,
            s_local_i: Array,
        ) -> tuple[Array, Array, Array]:
            def kinematics_at_s(
                s_local_ij: Array,
            ) -> tuple[Array, Array, Array]:
                Ad_inv, Ad_inv_T, Td = (
                    constant_strain_se3._kinematic_operators_from_prepared(
                        prepared_adjoint_powers_i,
                        s_local_ij,
                        self.global_eps,
                        self.tangent_eps,
                        convective_only=True,
                    )
                )
                J_segment = Ad_inv_T @ B_xi_i
                J_next = Ad_inv @ J_base_i + J_segment

                eta = Ad_inv_T @ xid_i
                Jd_qd_next = (
                    Ad_inv @ Jd_qd_base_i
                    - se3.small_adjoint_action(eta, Ad_inv @ (J_base_i @ qd))
                    + Ad_inv @ (Td @ xid_i)
                )
                gravity_local = Ad_inv @ gravity_base_i
                return gravity_local, J_next, Jd_qd_next

            return vmap(kinematics_at_s)(s_local_i)

        gravity_ps, J_ps, Jd_qd_ps = vmap(segment_kinematics)(
            xid,
            prepared_adjoint_powers,
            B_xi_segments,
            gravity_bases,
            J_bases,
            Jd_qd_bases,
            s_local,
        )
        return Ws_inner, gravity_ps, J_ps, Jd_qd_ps

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
        weights, gravity, jacobians, jacobian_dot_qd = (
            self._dynamics_integration_kinematics(q, qd)
        )
        inertia = jnp.zeros((self.num_dofs, self.num_dofs), dtype=jacobians.dtype)
        coriolis_qd = jnp.zeros((self.num_dofs,), dtype=jacobians.dtype)
        gravity_force = jnp.zeros((self.num_dofs,), dtype=jacobians.dtype)

        # This trace-time loop intentionally keeps each contraction at its causal
        # prefix width; scan/fori_loop require a fixed-width body and do extra work.
        for segment_index, dof_end in enumerate(self._segment_dof_ends):
            mass_diagonal = jnp.diagonal(self.M_segments[segment_index])
            jacobian = jacobians[segment_index, :, :, :dof_end]
            weight = weights[segment_index]

            def mass_action(value: Array, diagonal: Array = mass_diagonal) -> Array:
                shape = (1, diagonal.shape[0]) + (1,) * (value.ndim - 2)
                return diagonal.reshape(shape) * value

            flattened_rows = self.num_gauss_points * 6
            jacobian_flat = jacobian.reshape(flattened_rows, dof_end)
            weighted_mass_jacobian = (
                weight[:, None, None] * mass_action(jacobian)
            ).reshape(flattened_rows, dof_end)
            inertia_segment = jacobian_flat.T @ weighted_mass_jacobian

            eta = jacobian @ qd[:dof_end]
            momentum = mass_action(eta)
            coadjoint_momentum = vmap(se3.coadjoint_action)(eta, momentum)
            wrench = mass_action(jacobian_dot_qd[segment_index]) + coadjoint_momentum
            coriolis_segment = jacobian_flat.T @ (weight[:, None] * wrench).reshape(-1)
            gravity_segment = -jacobian_flat.T @ (
                weight[:, None] * mass_action(gravity[segment_index])
            ).reshape(-1)

            inertia = inertia.at[:dof_end, :dof_end].add(inertia_segment)
            coriolis_qd = coriolis_qd.at[:dof_end].add(coriolis_segment)
            gravity_force = gravity_force.at[:dof_end].add(gravity_segment)

        return inertia, coriolis_qd, gravity_force

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
            actuation_args (tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.

        Returns:
            yd (Array): Time derivative of the state vector.
        """
        if custom_jvp_enabled():
            return PCS._forward_dynamics_custom_jvp(self, t, y, actuation_args)
        return self._forward_dynamics(t, y, actuation_args)

    def _forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """Evaluate the PCS forward-dynamics primal without a custom JVP."""
        del t

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
        tau_u = self.actuation_force(q, u, qd=qd)

        rhs = tau_u + tau_ext - Cqd - G - tau_el - self.damping_matrix(q) @ qd
        qdd = self._solve_inertia(B, rhs)

        yd = jnp.concatenate([qd, qdd])

        return yd

    @eqx.filter_custom_jvp
    @staticmethod
    def _forward_dynamics_custom_jvp(
        robot: "PCS",
        t: Array,
        y: Array,
        actuation_args: tuple | None,
    ) -> Array:
        """Custom-JVP entry point for PCS forward dynamics."""
        return robot._forward_dynamics(t, y, actuation_args)

    @_forward_dynamics_custom_jvp.def_jvp
    def _forward_dynamics_custom_jvp_jvp(
        primals: tuple["PCS", Array, Array, tuple | None],
        tangents: tuple[Any, Array | None, Array | None, tuple | None],
    ) -> tuple[Array, Array]:
        """
        Evaluate a hybrid analytical/autodiff forward-dynamics JVP.

        The inertial, Coriolis, and gravity directions use the analytical
        inverse-dynamics Jacobian passes. Actuator and passive-element force
        directions remain under JAX autodiff so configuration-dependent
        components retain their exact derivatives.
        """
        robot, t, y, actuation_args = primals
        _, _, y_tangent, actuation_args_tangent = tangents

        yd = robot._forward_dynamics(t, y, actuation_args)
        q, qd = jnp.split(y, 2)
        _, qdd = jnp.split(yd, 2)

        if y_tangent is None:
            q_tangent = jnp.zeros_like(q)
            qd_tangent = jnp.zeros_like(qd)
        else:
            q_tangent, qd_tangent = jnp.split(y_tangent, 2)

        if actuation_args is None:
            u = jnp.zeros((robot.num_actuators,), dtype=q.dtype)
            tau_ext = jnp.zeros_like(q)
            u_tangent = jnp.zeros_like(u)
            tau_ext_tangent = jnp.zeros_like(tau_ext)
        else:
            u = actuation_args[0]
            tau_ext = actuation_args[1] if len(actuation_args) == 2 else None
            if actuation_args_tangent is None:
                u_tangent = None
                tau_ext_tangent = None
            else:
                u_tangent = actuation_args_tangent[0]
                tau_ext_tangent = (
                    actuation_args_tangent[1]
                    if len(actuation_args_tangent) == 2
                    else None
                )

            if u is None:
                u = jnp.zeros((robot.num_actuators,), dtype=q.dtype)
            if tau_ext is None:
                tau_ext = jnp.zeros_like(q)
            if u_tangent is None:
                u_tangent = jnp.zeros_like(u)
            if tau_ext_tangent is None:
                tau_ext_tangent = jnp.zeros_like(tau_ext)

        def applied_force(
            q_value: Array,
            qd_value: Array,
            u_value: Array,
            tau_ext_value: Array,
        ) -> Array:
            return (
                robot.actuation_force(q_value, u_value, qd=qd_value)
                + tau_ext_value
                - robot.elastic_force(q_value)
                - robot.damping_matrix(q_value) @ qd_value
            )

        _, applied_force_tangent = jvp(
            applied_force,
            (q, qd, u, tau_ext),
            (q_tangent, qd_tangent, u_tangent, tau_ext_tangent),
        )

        if y_tangent is None:
            mass_matrix = robot.inertia_matrix(q)
            inverse_dynamics_tangent = jnp.zeros_like(q)
        else:
            (
                dID_dq,
                dID_dqd,
                mass_matrix,
                _,
                _,
                _,
                _,
            ) = robot.inverse_dynamics_backward_pass(q, qd, qdd)
            inverse_dynamics_tangent = dID_dq @ q_tangent + dID_dqd @ qd_tangent

        qdd_tangent = robot._solve_inertia(
            mass_matrix,
            applied_force_tangent - inverse_dynamics_tangent,
        )
        yd_tangent = jnp.concatenate([qd_tangent, qdd_tangent])
        return yd, yd_tangent

    @eqx.filter_jit
    def _local_id_jacobian_kinematics(
        self,
        segment_idx: Array,
        H: Array,
        q: Array,
        qd: Array,
        qdd: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """
        Compute the local kinematics used by the inverse-dynamics Jacobian passes.

        Args:
            segment_idx (Array): Index of the PCS segment containing the
                interval. Shape is ``()``.
            H (Array): Length of the local quadrature interval. Shape is
                ``()``.
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.
            qdd (Array): Active generalized accelerations. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, Array, Array, Array, Array]: Tuple
            ``(g_local, S, Sd, deta_dq, detad_dq)``, where ``g_local`` is the
            interval transformation with shape ``(4, 4)``. The remaining
            arrays have shape ``(6, self.num_dofs)`` and contain the local
            joint motion subspace, its time derivative, and the configuration
            derivatives of the local velocity and acceleration twists.
        """
        B_xi_segments = self.B_xi.reshape(
            self.num_segments,
            6,
            self.num_dofs,
        )
        xi_ref_segments = self.xi_ref.reshape(self.num_segments, 6)

        B_xi_i = B_xi_segments[segment_idx]
        xi_ref_i = xi_ref_segments[segment_idx]

        (
            g_local,
            S,
            Sd,
            deta_dq,
            dS_dq_qdd,
            dSd_dq_qd,
        ) = joint_motion_subspace_with_derivatives(
            H,
            B_xi_i,
            xi_ref_i,
            q,
            qd,
            qdd,
            self.global_eps,
        )
        detad_dq = dS_dq_qdd + dSd_dq_qd

        return g_local, S, Sd, deta_dq, detad_dq

    @eqx.filter_jit
    def _inverse_dynamics_jacobian_forward_pass(
        self,
        q: Array,
        qd: Array,
        qdd: Array,
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        """
        Propagate ID-Jacobian kinematics through all quadrature intervals.

        The propagated derivative quantities are ``J = d(eta) / d(qd)``,
        ``Jdot = d(J) / dt``, ``R = d(eta) / d(q)``,
        ``A = d(etad) / d(q)``, and
        ``Y = d(etad) / d(qd) = Jdot + R``.

        Local interval quantities are evaluated in parallel with vmap. A
        single parent-to-child scan then propagates the global quantities.
        All returned arrays use flattened segment-major interval ordering.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.
            qdd (Array): Active generalized accelerations. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, ...]: Tuple ``(segment_indices, H, mass_weights, g,
            eta, etad, J, Jdot, R, A, Y, Adginv, S)``. Let
            ``num_intervals = self.num_segments *
            (self.num_integration_points - 1)``. ``segment_indices``, ``H``,
            and ``mass_weights`` have shape ``(num_intervals,)``; ``g`` has
            shape ``(num_intervals, 4, 4)``; ``eta`` and ``etad`` have shape
            ``(num_intervals, 6)``; ``J``, ``Jdot``, ``R``, ``A``, ``Y``,
            and ``S`` have shape
            ``(num_intervals, 6, self.num_dofs)``; and ``Adginv`` has shape
            ``(num_intervals, 6, 6)``.
        """
        Xs_nodes, Ws_nodes = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.integration_points,
            self.integration_weights,
            self.L_cum[:-1],
            self.L_cum[1:],
        )
        s_local_nodes = Xs_nodes - self.L_cum[:-1, None]
        H_steps = jnp.diff(s_local_nodes, axis=1)

        n_steps = self.num_integration_points - 1
        segment_indices = jnp.repeat(
            jnp.arange(self.num_segments, dtype=jnp.int32),
            n_steps,
        )
        H_flat = H_steps.reshape(-1)
        mass_weights = Ws_nodes[:, 1:].reshape(-1)

        (
            g_local,
            S,
            Sd,
            deta_dq,
            detad_dq,
        ) = vmap(
            self._local_id_jacobian_kinematics,
            in_axes=(0, 0, None, None, None),
        )(
            segment_indices,
            H_flat,
            q,
            qd,
            qdd,
        )
        Adginv = vmap(se3.adjoint_inverse)(g_local)

        zeros_6n = jnp.zeros((6, self.num_dofs), dtype=q.dtype)
        zeros_6 = jnp.zeros((6,), dtype=q.dtype)
        carry_init = (
            self.g0,
            zeros_6,
            zeros_6,
            zeros_6n,
            zeros_6n,
            zeros_6n,
            zeros_6n,
        )

        def propagate_interval(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
            local: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[
            tuple[Array, Array, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array, Array, Array, Array, Array],
        ]:
            (
                g_base,
                eta_base,
                etad_base,
                J_base,
                Jdot_base,
                R_base,
                A_base,
            ) = carry
            g_local_i, Adginv_i, S_i, Sd_i, deta_dq_i, detad_dq_i = local

            eta_local = S_i @ qd
            etad_local = S_i @ qdd + Sd_i @ qd
            eta_plus = eta_base + eta_local  # Intermediate velocity twist
            etad_plus = etad_base + etad_local + se3.small_adjoint(eta_base) @ eta_local

            g_tip = g_base @ g_local_i
            eta_tip = Adginv_i @ eta_plus
            etad_tip = Adginv_i @ etad_plus

            J_pre = J_base + S_i
            Jdot_pre = Jdot_base + Sd_i - se3.small_adjoint(eta_local) @ J_pre
            J_tip = Adginv_i @ J_pre
            Jdot_tip = Adginv_i @ Jdot_pre

            R_local = se3.small_adjoint(eta_plus) @ S_i + deta_dq_i
            R_tip = Adginv_i @ (R_base + R_local)

            A_local = (
                se3.small_adjoint(etad_plus) @ S_i
                + se3.small_adjoint(eta_base) @ deta_dq_i
                + detad_dq_i
            )
            A_pre = A_base - se3.small_adjoint(eta_local) @ R_base + A_local
            A_tip = Adginv_i @ A_pre
            Y_tip = Jdot_tip + R_tip

            carry_next = (
                g_tip,
                eta_tip,
                etad_tip,
                J_tip,
                Jdot_tip,
                R_tip,
                A_tip,
            )
            output = (
                g_tip,
                eta_tip,
                etad_tip,
                J_tip,
                Jdot_tip,
                R_tip,
                A_tip,
                Y_tip,
            )
            return carry_next, output

        (
            _,
            (
                g,
                eta,
                etad,
                J,
                Jdot,
                R,
                A,
                Y,
            ),
        ) = lax.scan(
            propagate_interval,
            carry_init,
            (g_local, Adginv, S, Sd, deta_dq, detad_dq),
        )

        return (
            segment_indices,
            H_flat,
            mass_weights,
            g,
            eta,
            etad,
            J,
            Jdot,
            R,
            A,
            Y,
            Adginv,
            S,
        )

    @eqx.filter_jit
    def inverse_dynamics_backward_pass(
        self,
        q: Array,
        qd: Array,
        qdd: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """
        Recurse the composite wrench and its Jacobians from tip to base.

        At an interval, ``F_res`` is the resultant spatial wrench
        produced by that interval and every interval downstream toward the
        robot tip, transported into the current interval's base frame. It is
        ordered as ``[moment, force]`` and is projected into generalized
        coordinates by ``S.T @ F_res``.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.
            qdd (Array): Active generalized accelerations. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, Array, Array, Array, Array, Array, Array]: Tuple
            ``(dID_dq, dID_dqd, dID_dqdd, F_res, dF_res_dq,
            dF_res_dqd, dF_res_dqdd)``. The first three inverse-dynamics
            derivatives have shape ``(self.num_dofs, self.num_dofs)``;
            ``dID_dqdd`` is also the generalized inertia matrix. ``F_res``
            has shape ``(self.num_segments,
            self.num_integration_points - 1, 6)``. Its three derivatives
            have shape ``(self.num_segments,
            self.num_integration_points - 1, 6, self.num_dofs)``.
        """
        (
            segment_indices,
            H_steps,
            mass_weights,
            g,
            eta,
            etad,
            J,
            _,
            R,
            A,
            Y,
            Adginv,
            S,
        ) = self._inverse_dynamics_jacobian_forward_pass(q, qd, qdd)

        zeros_6 = jnp.zeros((6,), dtype=q.dtype)
        zeros_6n = jnp.zeros((6, self.num_dofs), dtype=q.dtype)
        zeros_nn = jnp.zeros((self.num_dofs, self.num_dofs), dtype=q.dtype)
        q_basis = jnp.eye(self.num_dofs, dtype=q.dtype)
        B_xi_segments = self.B_xi.reshape(
            self.num_segments,
            6,
            self.num_dofs,
        )
        xi_ref_segments = self.xi_ref.reshape(self.num_segments, 6)

        carry_init = (
            zeros_6,
            zeros_6n,
            zeros_6n,
            zeros_6n,
            zeros_nn,
            zeros_nn,
            zeros_nn,
        )

        def propagate_interval_backward(
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
            step_idx: Array,
        ) -> tuple[
            tuple[Array, Array, Array, Array, Array, Array, Array],
            tuple[Array, Array, Array, Array],
        ]:
            (
                F_res_child,
                dF_res_dq_child,
                dF_res_dqd_child,
                dF_res_dqdd_child,
                dID_dq,
                dID_dqd,
                dID_dqdd,
            ) = carry

            segment_idx = segment_indices[step_idx]
            M = mass_weights[step_idx] * self.M_segments[segment_idx]
            g_i = g[step_idx]
            eta_i = eta[step_idx]
            etad_i = etad[step_idx]
            J_i = J[step_idx]
            R_i = R[step_idx]
            A_i = A[step_idx]
            Y_i = Y[step_idx]
            Adginv_i = Adginv[step_idx]
            S_i = S[step_idx]

            M_eta = M @ eta_i
            gravity_i = se3.adjoint_inverse(g_i) @ self.g
            F_i = M @ etad_i + se3.coadjoint(eta_i) @ M_eta - M @ gravity_i
            N = se3.coadjoint(eta_i) @ M + se3.coadjoint_bar(M_eta)

            F_res_tip = F_res_child + F_i
            dF_res_dq_tip = (
                dF_res_dq_child
                + M @ A_i
                + N @ R_i
                - M @ se3.small_adjoint(gravity_i) @ J_i
            )
            dF_res_dqd_tip = dF_res_dqd_child + M @ Y_i + N @ J_i
            dF_res_dqdd_tip = dF_res_dqdd_child + M @ J_i

            coAdg = Adginv_i.T
            F_res = coAdg @ F_res_tip
            dF_res_dq = coAdg @ dF_res_dq_tip
            dF_res_dqd = coAdg @ dF_res_dqd_tip
            dF_res_dqdd = coAdg @ dF_res_dqdd_tip

            dF_res_dq = dF_res_dq + se3.coadjoint_bar(F_res) @ S_i

            B_xi_i = B_xi_segments[segment_idx]
            xi_ref_i = xi_ref_segments[segment_idx]
            H_i = H_steps[step_idx]

            def projected_wrench_row(q_unit: Array) -> Array:
                dS_dq_direction = joint_dSdq_qd(
                    H_i,
                    B_xi_i,
                    xi_ref_i,
                    q,
                    q_unit,
                    self.global_eps,
                )
                return F_res @ dS_dq_direction

            dSTF_res_dq = vmap(projected_wrench_row)(q_basis)

            dID_dq = dID_dq + S_i.T @ dF_res_dq + dSTF_res_dq
            dID_dqd = dID_dqd + S_i.T @ dF_res_dqd
            dID_dqdd = dID_dqdd + S_i.T @ dF_res_dqdd

            carry_next = (
                F_res,
                dF_res_dq,
                dF_res_dqd,
                dF_res_dqdd,
                dID_dq,
                dID_dqd,
                dID_dqdd,
            )
            output = (
                F_res,
                dF_res_dq,
                dF_res_dqd,
                dF_res_dqdd,
            )
            return carry_next, output

        num_intervals = segment_indices.shape[0]
        reverse_indices = jnp.arange(
            num_intervals - 1,
            -1,
            -1,
            dtype=jnp.int32,
        )
        (
            (
                _,
                _,
                _,
                _,
                dID_dq,
                dID_dqd,
                dID_dqdd,
            ),
            reverse_outputs,
        ) = lax.scan(
            propagate_interval_backward,
            carry_init,
            reverse_indices,
        )

        (
            F_res,
            dF_res_dq,
            dF_res_dqd,
            dF_res_dqdd,
        ) = (jnp.flip(value, axis=0) for value in reverse_outputs)
        step_shape = (self.num_segments, self.num_integration_points - 1)
        F_res = F_res.reshape(*step_shape, 6)
        dF_res_dq = dF_res_dq.reshape(*step_shape, 6, self.num_dofs)
        dF_res_dqd = dF_res_dqd.reshape(*step_shape, 6, self.num_dofs)
        dF_res_dqdd = dF_res_dqdd.reshape(*step_shape, 6, self.num_dofs)

        return (
            dID_dq,
            dID_dqd,
            dID_dqdd,
            F_res,
            dF_res_dq,
            dF_res_dqd,
            dF_res_dqdd,
        )

    @eqx.filter_jit
    def inverse_dynamics_derivatives(
        self,
        q: Array,
        qd: Array,
        qdd: Array,
    ) -> tuple[Array, Array]:
        """
        Return inverse-dynamics derivatives from the analytical PCS passes.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.
            qdd (Array): Active generalized accelerations. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, Array]: The derivatives ``(dID_dq, dID_dqd)``. Both
            arrays have shape ``(self.num_dofs, self.num_dofs)``.
        """
        dID_dq, dID_dqd, _, _, _, _, _ = self.inverse_dynamics_backward_pass(
            q,
            qd,
            qdd,
        )
        return dID_dq, dID_dqd

    @eqx.filter_jit
    def elastic_force_derivative_q(self, q: Array) -> Array:
        """
        Return the analytical configuration derivative of the elastic force.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.

        Returns:
            Array: Derivative ``d(tau_el) / d(q)`` with shape
            ``(self.num_dofs, self.num_dofs)``.
        """
        del q
        return self.stiffness_matrix()

    @eqx.filter_jit
    def damping_force_derivatives(
        self,
        q: Array,
        qd: Array,
    ) -> tuple[Array, Array]:
        """
        Return analytical derivatives of the generalized damping force.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, Array]: Derivatives of ``damping_matrix(q) @ qd``
            with respect to ``q`` and ``qd``, respectively. Both arrays have
            shape ``(self.num_dofs, self.num_dofs)``.
        """
        del qd
        return (
            jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype),
            self.damping_matrix(q),
        )

    @eqx.filter_jit
    def actuation_force_derivative_q(
        self,
        q: Array,
        u: Array,
    ) -> Array:
        """
        Return the analytical configuration derivative of the actuation force.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            u (Array): Actuation inputs. Shape is ``(self.num_actuators,)``.

        Returns:
            Array: Derivative ``d(tau_u) / d(q)`` with shape
            ``(self.num_dofs, self.num_dofs)``.
        """
        del u
        return jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)

    @eqx.filter_jit
    def actuation_force_derivative_u(
        self,
        q: Array,
    ) -> Array:
        """
        Return the analytical input derivative of the actuation force.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.

        Returns:
            Array: Derivative ``d(tau_u) / d(u)`` with shape
            ``(self.num_dofs, self.num_actuators)``.
        """
        return self.actuation_matrix(q)

    @eqx.filter_jit
    def forward_dynamics_derivatives(
        self,
        q: Array,
        qd: Array,
        qdd: Array,
        u: Array | None = None,
    ) -> tuple[Array, Array]:
        """
        Return analytical ``dqdd/dq`` and ``dqdd/dqd``.

        The derivatives follow from the unconstrained PCS dynamics solve:

        ``dqdd_dq = M \\ (dtau_dq - dID_dq)``
        ``dqdd_dqd = M \\ (dtau_dqd - dID_dqd)``

        For base PCS, ``dtau_dq = -K`` and ``dtau_dqd = -D`` because the
        actuation matrix, stiffness, and damping are configuration independent.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.
            qd (Array): Active generalized velocities. Shape is
                ``(self.num_dofs,)``.
            qdd (Array): Active generalized accelerations. Shape is
                ``(self.num_dofs,)``.
            u (Array, optional): Actuation inputs. Shape is
                ``(self.num_actuators,)``. Defaults to a zero input.

        Returns:
            tuple[Array, Array]: Derivatives ``(dqdd_dq, dqdd_dqd)``. Both
            arrays have shape ``(self.num_dofs, self.num_dofs)``.
        """
        if u is None:
            u = jnp.zeros((self.num_actuators,), dtype=q.dtype)

        (
            dID_dq,
            dID_dqd,
            mass_matrix,
            _,
            _,
            _,
            _,
        ) = self.inverse_dynamics_backward_pass(q, qd, qdd)

        d_elastic_dq = self.elastic_force_derivative_q(q)
        d_damping_dq, d_damping_dqd = self.damping_force_derivatives(q, qd)
        d_actuation_dq = self.actuation_force_derivative_q(q, u)

        dtau_dq = d_actuation_dq - d_elastic_dq - d_damping_dq
        dtau_dqd = -d_damping_dqd

        dqdd_dq = self._solve_inertia(mass_matrix, dtau_dq - dID_dq)
        dqdd_dqd = self._solve_inertia(mass_matrix, dtau_dqd - dID_dqd)

        return dqdd_dq, dqdd_dqd

    @eqx.filter_jit
    def forward_dynamics_input_derivatives(
        self,
        q: Array,
    ) -> tuple[Array, Array]:
        """
        Return analytical ``dqdd/du`` and ``dqdd/dtau_ext``.

        ``tau_ext`` is the direct generalized-force input used by
        ``forward_dynamics``. Its derivative is therefore ``M(q)^{-1}``.

        Args:
            q (Array): Active generalized coordinates. Shape is
                ``(self.num_dofs,)``.

        Returns:
            tuple[Array, Array]: Derivatives ``(dqdd_du, dqdd_dtau_ext)``.
            Their shapes are ``(self.num_dofs, self.num_actuators)`` and
            ``(self.num_dofs, self.num_dofs)``, respectively.
        """
        mass_matrix = self.inertia_matrix(q)
        dqdd_du = self._solve_inertia(
            mass_matrix,
            self.actuation_force_derivative_u(q),
        )
        dqdd_dtau_ext = self._solve_inertia(
            mass_matrix,
            jnp.eye(q.shape[0], dtype=q.dtype),
        )

        return dqdd_du, dqdd_dtau_ext

    @eqx.filter_jit
    def forward_dynamics_state_jacobian(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None = None,
    ) -> Array:
        """
        Return the analytical state Jacobian ``d([qd, qdd]) / d([q, qd])``.

        The returned blocks follow the unconstrained PCS state ordering.

        Args:
            t (Array): Current time. Shape is ``()``.
            y (Array): State ``[q, qd]``. Shape is
                ``(2 * self.num_dofs,)``.
            actuation_args (tuple, optional): Tuple containing ``u`` with
                shape ``(self.num_actuators,)`` and optionally ``tau_ext``
                with shape ``(self.num_dofs,)``. Defaults to zero inputs.

        Returns:
            Array: State Jacobian ``d([qd, qdd]) / d([q, qd])`` with shape
            ``(2 * self.num_dofs, 2 * self.num_dofs)``.
        """
        q, qd = jnp.split(y, 2)

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
            u = jnp.zeros((self.num_actuators,), dtype=q.dtype)
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[0],), dtype=q.dtype)

        yd = self.forward_dynamics(t, y, (u, tau_ext))
        _, qdd = jnp.split(yd, 2)
        dqdd_dq, dqdd_dqd = self.forward_dynamics_derivatives(q, qd, qdd, u)

        eye = jnp.eye(q.shape[0], dtype=q.dtype)
        zeros = jnp.zeros_like(eye)
        return jnp.block([[zeros, eye], [dqdd_dq, dqdd_dqd]])

    @eqx.filter_jit
    def forward_dynamics_jacobians(
        self,
        t: Array,
        y: Array,
        actuation_args: tuple | None = None,
    ) -> tuple[Array, Array, Array]:
        """
        Return state, actuation-input, and external-force Jacobians.

        The returned tuple is ``(dyd_dy, dyd_du, dyd_dtau_ext)`` where
        ``yd = forward_dynamics(t, y, actuation_args)``.

        Args:
            t (Array): Current time. Shape is ``()``.
            y (Array): State ``[q, qd]``. Shape is
                ``(2 * self.num_dofs,)``.
            actuation_args (tuple, optional): Tuple containing ``u`` with
                shape ``(self.num_actuators,)`` and optionally ``tau_ext``
                with shape ``(self.num_dofs,)``. Defaults to zero inputs.

        Returns:
            tuple[Array, Array, Array]: State, actuation-input, and external
            generalized-force Jacobians ``(dyd_dy, dyd_du, dyd_dtau_ext)``.
            Their shapes are ``(2 * self.num_dofs, 2 * self.num_dofs)``,
            ``(2 * self.num_dofs, self.num_actuators)``, and
            ``(2 * self.num_dofs, self.num_dofs)``, respectively.
        """
        q, qd = jnp.split(y, 2)

        if actuation_args is None:
            u = None
        elif len(actuation_args) == 1 or len(actuation_args) == 2:
            u = actuation_args[0]
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        if u is None:
            u = jnp.zeros((self.num_actuators,), dtype=q.dtype)

        dyd_dy = self.forward_dynamics_state_jacobian(t, y, actuation_args)
        dqdd_du, dqdd_dtau_ext = self.forward_dynamics_input_derivatives(q)

        zeros_du = jnp.zeros((q.shape[0], u.shape[0]), dtype=q.dtype)
        zeros_tau = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        dyd_du = jnp.concatenate([zeros_du, dqdd_du], axis=0)
        dyd_dtau_ext = jnp.concatenate([zeros_tau, dqdd_dtau_ext], axis=0)

        return dyd_dy, dyd_du, dyd_dtau_ext
