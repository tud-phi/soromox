__all__ = ["GVS"]
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, vmap

import soromox.utils.lie_algebra as lie
from soromox.systems.gvs._assembly import assign_gvs_runtime_arrays
from soromox.systems.gvs._runtime import SegmentRuntimeData
from soromox.systems.gvs.construction import params_and_structure_from_segments
from soromox.systems.gvs.joint_bases import (
    B_Cylindrical,
    B_Fixed,
    B_Free,
    B_Helical,
    B_Planar,
    B_Prismatic,
    B_Revolute,
    B_Spherical,
)
from soromox.systems.gvs.operands import GeometricOperand, JointOperand
from soromox.systems.gvs.params import GVSLinkParams, GVSParams
from soromox.systems.gvs.primitives import Basis, Joint, Link
from soromox.systems.gvs.specs import GVSSegment
from soromox.systems.gvs.strain_bases import (
    B_IMQ,
    B_Chebychev,
    B_Fourier,
    B_Gaussian,
    B_LegendrePolynomial,
    B_Monomial,
    dB_Chebychev,
    dB_Fourier,
    dB_Gaussian,
    dB_IMQ,
    dB_LegendrePolynomial,
    dB_Monomial,
)
from soromox.systems.gvs.structures import (
    GVSJointStructure,
    GVSLinkStructure,
    GVSStrainBasisStructure,
    GVSStructure,
)
from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot
from soromox.utils.integration import gauss_quadrature


class GVS(SoftRobot):
    """
    Geometric Variable Strain (GVS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory with geometric variable strain parametrizations.
    It supports computation of forward kinematics, Jacobians, and dynamic matrices
    for robots with arbitrary combinations of link cross-sections, joint types, and
    strain basis functions.

    Attributes:
        num_segments: Number of segments (links) in the robot.
        max_dof: Maximum number of degrees of freedom (DOFs) per link or joint.
        max_num_gauss_points: Maximum number of Gauss points per link used for integration.
        max_num_integration_points: Maximum number of integration points per link (= max_num_gauss_points + 2).
        num_dofs: Total number of DOFs for the robot (sum of joint + link DOFs).
        num_padded_dofs: Theoretical maximum number of DOFs (num_segments * 2 * max_dof).
        active_dof_map: Strain basis selection matrix mapping active DOFs to the full strain space.
        scale_rotational_basis_by_length: If True, apply scaling to the angular component of the strain basis matrix for improved numerical stability.
        segment_lengths, segment_end_positions: Length of each link, and cumulative link lengths.
        num_integration_points: Number of integration/evaluation points for each link.
        dofs_per_segment: Number of DOFs for each link/joint pair (shape: num_segments x 2).
        integration_points, integration_weights: Gauss quadrature nodes and weights for each link.
        mass_matrices, stiffness_matrices, damping_matrices: Mass, stiffness, and damping matrices at integration points.
        B_joint, B_Xs, B_Z1, B_Z2: Basis matrices for joints and links at quadrature points and intermediate points.
        xi_ref_joint, xi_ref_Xs, xi_ref_Z1, xi_ref_Z2: Reference strain vectors for joints and links.
        joint_stiffness: Joint stiffness matrices.
        K_full: Precomputed full stiffness matrix for the robot.
        D_full: Precomputed full damping matrix for the robot.
        g0: Initial pose of the robot base as an SE(3) transformation matrix.
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

    References:
        Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020).
        A geometric variable-strain approach for static modeling of soft manipulators
        with tendon and fluidic actuation. IEEE Robotics and Automation Letters,
        5(3), 4006-4013.

        Boyer, F., Lebastard, V., Candelier, F., & Renda, F. (2020). Dynamics of
        continuum and soft robots: A strain parameterization based approach. IEEE
        Transactions on Robotics, 37(3), 847-863.

        Mathew, A. T., Feliu-Talegon, D., Alkayas, A. Y., Boyer, F., & Renda, F.
        (2025). Reduced order modeling of hybrid soft-rigid robots using global,
        local, and state-dependent strain parameterization. The International
        Journal of Robotics Research, 44(1), 129-154.
    """

    # Static attributes
    num_segments: int = eqx.field(static=True)  # Number of links in the robot
    max_dof: int = eqx.field(
        static=True
    )  # Maximum number of DOFs for one link of the robot
    max_num_gauss_points: int = eqx.field(
        static=True
    )  # Maximum number of Gauss points for one link of the robot
    max_num_integration_points: int = eqx.field(
        static=True
    )  # Maximum number of integration points for one link of the robot
    num_gauss_points: Array

    num_dofs: int = eqx.field(
        static=True
    )  # Total number of DOFs for the robot = sum(dofs_per_segment)
    num_padded_dofs: int = eqx.field(
        static=True
    )  # Total number of DOFs for the robot, considering the maximum DOFs per link: num_segments * 2 * max_dof

    Z1: float = eqx.field(static=True, default=0.5 - math.sqrt(3) / 6)
    Z2: float = eqx.field(static=True, default=0.5 + math.sqrt(3) / 6)

    active_dof_map: Array  # Strain basis functions for the robot (6, max_dof)
    # Active selectors by segment/block.
    active_dof_map_blocks: Array
    scale_rotational_basis_by_length: (
        bool  # If True, apply length scaling to angular strain contributions
    )

    # Dynamic attributes
    segment_lengths: Array  # List of lengths for each link (num_segments, )
    segment_end_positions: Array  # Cumulative lengths of the links (num_segments + 1, )
    num_integration_points: (
        Array  # List of number of evaluation points for each link (num_segments, )
    )
    dofs_per_segment: Array  # List of number of DOFs for each link (num_segments, )
    integration_points: Array  # List of integration points for each link (num_segments, max_num_integration_points)
    integration_weights: Array  # List of weights for each link (num_segments, max_num_integration_points)
    # Length-scaled non-endpoint quadrature weights.
    inner_integration_weights: Array
    mass_matrices: Array
    inner_mass_matrices: Array
    stiffness_matrices: Array
    damping_matrices: Array

    B_joint: Array
    B_Xs: Array
    B_Z1: Array
    B_Z2: Array

    xi_ref_joint: Array
    xi_ref_Xs: Array
    xi_ref_Z1: Array
    xi_ref_Z2: Array

    joint_stiffness: Array  # Joint stiffness matrix (num_segments, max_dof, max_dof)
    # Precomputed full stiffness and damping matrices
    K_full: Array  # Full stiffness matrix (num_padded_dofs, num_padded_dofs)
    K: Array  # Active-coordinate stiffness matrix (num_dofs, num_dofs)
    D_full: Array  # Full damping matrix (num_padded_dofs, num_padded_dofs)
    D_active: Array  # Active-coordinate damping matrix (num_dofs, num_dofs)
    gather_indices: Array  # Indices for active-to-padded coordinate gather
    gather_mask: Array  # Valid-entry mask for active-to-padded coordinate gather

    g0: Array  # Initial base pose (4,4)
    g: Array  # Gravity vector (6,)

    # Addition to compute forward kinematics at s
    basis_type_index: Array  # Index of the basis type for each segment (num_segments,)
    basis_active_params: Array  # Parameters for the basis DOFs (num_segments, max_dof)
    basis_order_params: (
        Array  # Parameters for the basis orientation (num_segments, max_dof)
    )
    cross_section_geometry_index: Array  # CrossSectionGeometry index for each segment
    radius_params: Array  # Circular params (num_segments, 2)
    height_params: Array  # Rectangular height params (num_segments, 2)
    width_params: Array  # Rectangular width params (num_segments, 2)
    semi_major_params: Array  # Elliptical semi-major params (num_segments, 2)
    semi_minor_params: Array  # Elliptical semi-minor params (num_segments, 2)
    params: GVSParams
    structure: GVSStructure = eqx.field(static=True)

    def __init__(
        self,
        params: GVSParams,
        structure: GVSStructure,
        **kwargs: Any,
    ) -> None:
        """Initialize a GVS robot from typed params and static segment structure."""
        super().__init__(**kwargs)
        if not isinstance(params, GVSParams):
            raise TypeError("params must be a GVSParams instance.")
        if not isinstance(structure, GVSStructure):
            raise TypeError("structure must be a GVSStructure instance.")
        params.validate_against_structure(structure)
        self.params = params
        self.structure = structure
        self.scale_rotational_basis_by_length = bool(
            structure.scale_rotational_basis_by_length
        )
        assign_gvs_runtime_arrays(
            self,
            segments=structure.segments,
            params=params,
            max_dof=structure.max_dof,
            max_num_gauss_points=structure.max_num_gauss_points,
            g=params.gravity,
            p0=params.base_pose,
        )
        self.precompute()

    @staticmethod
    def params_from_segments(
        segments: list[GVSSegment] | tuple[GVSSegment, ...],
        *,
        gravity: Array,
        base_pose: Array | None = None,
        max_dof: int | None = None,
        max_num_gauss_points: int | None = None,
        scale_rotational_basis_by_length: bool = False,
    ) -> tuple[GVSParams, GVSStructure]:
        """Build typed GVS params and static structure from segment specs.

        This is the user-facing bridge between the natural GVS construction
        language, a sequence of ``GVSSegment`` objects, and the typed PyTree
        params used for JAX-friendly updates. Per-segment choices such as joint
        family, basis family, quadrature count, and cross-section family remain
        static in ``GVSStructure``. Numeric link, joint, and reference-strain
        values are copied into ``GVSParams``.
        """
        return params_and_structure_from_segments(
            segments,
            gravity=gravity,
            base_pose=base_pose,
            max_dof=max_dof,
            max_num_gauss_points=max_num_gauss_points,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        )

    @classmethod
    def from_segments(
        cls,
        segments: list[GVSSegment] | tuple[GVSSegment, ...],
        *,
        gravity: Array,
        base_pose: Array | None = None,
        max_dof: int | None = None,
        max_num_gauss_points: int | None = None,
        scale_rotational_basis_by_length: bool = False,
        **kwargs: Any,
    ) -> "GVS":
        """Construct a GVS model directly from user-facing segment specs."""
        params, structure = cls.params_from_segments(
            segments,
            gravity=gravity,
            base_pose=base_pose,
            max_dof=max_dof,
            max_num_gauss_points=max_num_gauss_points,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        )
        return cls(params=params, structure=structure, **kwargs)

    @property
    def is_planar(self) -> bool:
        """GVS is a spatial (3D) model."""
        return False

    @property
    def length(self) -> Array:
        """Total backbone length."""
        return jnp.sum(self.segment_lengths)

    @property
    def segment_length(self) -> Array:
        """Per-segment backbone lengths."""
        return jnp.asarray(self.segment_lengths)

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        """Cross-section geometry evaluated from stored link parameters."""
        segment_idx, s_local = self.classify_segment(s)
        length_i = self.segment_lengths[segment_idx]
        x = jnp.where(length_i > self.global_eps, s_local / length_i, 0.0)
        cross_section_geometry_idx = self.cross_section_geometry_index[segment_idx]
        cross_section_geometry_int = int(cross_section_geometry_idx)
        if cross_section_geometry_int == CrossSectionGeometry.CIRCULAR:
            params = self.radius_params[segment_idx]
            radius = Link.interpolate_param(x, params[0], params[1])
            tag = jnp.asarray(CrossSectionGeometry.CIRCULAR, dtype=jnp.int32)
            return tag, jnp.array([radius])
        if cross_section_geometry_int == CrossSectionGeometry.RECTANGULAR:
            h_params = self.height_params[segment_idx]
            w_params = self.width_params[segment_idx]
            height = Link.interpolate_param(x, h_params[0], h_params[1])
            width = Link.interpolate_param(x, w_params[0], w_params[1])
            tag = jnp.asarray(CrossSectionGeometry.RECTANGULAR, dtype=jnp.int32)
            return tag, jnp.array([width, height])
        a_params = self.semi_major_params[segment_idx]
        b_params = self.semi_minor_params[segment_idx]
        a_val = Link.interpolate_param(x, a_params[0], a_params[1])
        b_val = Link.interpolate_param(x, b_params[0], b_params[1])
        tag = jnp.asarray(CrossSectionGeometry.ELLIPTICAL, dtype=jnp.int32)
        return tag, jnp.array([a_val, b_val])

    def _build_segment_i(
        self,
        link_structure: GVSLinkStructure,
        joint_structure: GVSJointStructure,
        basis_structure: GVSStrainBasisStructure,
        num_gauss_points: int,
        max_dof: int,
        max_num_integration_points: int,
        *,
        length: Array,
        young_modulus: Array,
        poisson_ratio: Array,
        density: Array,
        damping_coefficient: Array,
        radius_initial: Array,
        radius_final: Array,
        height_initial: Array,
        height_final: Array,
        width_initial: Array,
        width_final: Array,
        semi_major_initial: Array,
        semi_major_final: Array,
        semi_minor_initial: Array,
        semi_minor_final: Array,
        reference_strain: Array,
        joint_stiffness: Array,
    ) -> SegmentRuntimeData:
        """
        Construct the discretized model data for a single segment.

        Args
        ----
        link_structure : GVSLinkStructure
            Static cross-section family for the link.
        joint_structure : GVSJointStructure
            Static joint type and kinematic choices.
        basis_structure : GVSStrainBasisStructure
            Static strain basis definition for variable strain representation.
        num_gauss_points : int
            Number of Gauss-Legendre quadrature points for integration.
        max_dof : int
            Maximum number of DOFs to pad/truncate joint/link basis matrices.
        max_num_integration_points : int
            Maximum number of integration points for padding.

        Returns
        -------
        SegmentRuntimeData
            Structured object containing:
            - Length, number of integration points, DOFs
            - Strain selection mask
            - Padded Gauss points/weights
            - Mass, stiffness, and damping matrices
            - Joint and link basis matrices at integration points and Z1/Z2 midpoints
            - Reference strains
            - Joint stiffness matrix

        Notes
        -----
        - Joint and link basis matrices are padded or truncated to `max_dof`
        to match the global storage format.
        - Link geometry properties are computed according to the section type
        (circular, rectangular, elliptical) and can vary linearly along the length.
        - The resulting data are suitable for direct concatenation into the
        full-robot arrays.
        """
        # === Joint attributes
        jointtype = joint_structure.type
        jointtype_idx = Joint.JOINTTYPE_MAP[jointtype]

        dof_joint = Joint.DICT_JOINT_TYPE_DOF[jointtype]
        K_joint = jnp.asarray(joint_stiffness)

        joint_operand = JointOperand(
            axis_idx=Joint.AXIS_MAP[joint_structure.axis],
            plane_idx=Joint.PLANE_MAP[joint_structure.plane],
            pitch=joint_structure.pitch,
        )

        B_joint_full = lax.switch(
            index=jointtype_idx,
            branches=[
                Joint.make_B_branch(B_fn=B_Revolute, dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Prismatic, dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Helical, dof_joint=1, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Cylindrical, dof_joint=2, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Planar, dof_joint=3, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Spherical, dof_joint=3, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Free, dof_joint=6, max_dof=max_dof),
                Joint.make_B_branch(B_fn=B_Fixed, dof_joint=0, max_dof=max_dof),
            ],
            operand=joint_operand,
        )
        strain_selector_joint = jnp.concatenate(
            [
                jnp.ones(dof_joint, dtype=bool),
                jnp.zeros(max_dof - dof_joint, dtype=bool),
            ]
        )

        xi_ref_joint = jnp.zeros((6,))
        K_joint = jnp.asarray(K_joint).reshape((dof_joint, dof_joint))
        K_joint_full = jnp.pad(
            K_joint,
            ((0, max_dof - dof_joint), (0, max_dof - dof_joint)),
            mode="constant",
        )  # shape (6, max_dof)

        # === Link attributes
        cross_section_geometry = link_structure.cross_section_geometry
        cross_section_geometry_idx = int(cross_section_geometry)

        E = jnp.asarray(young_modulus)
        nu = jnp.asarray(poisson_ratio)
        rho = jnp.asarray(density)
        eta = jnp.asarray(damping_coefficient)
        L = jnp.asarray(length)

        r_i = jnp.asarray(radius_initial)
        r_f = jnp.asarray(radius_final)
        h_i = jnp.asarray(height_initial)
        h_f = jnp.asarray(height_final)
        w_i = jnp.asarray(width_initial)
        w_f = jnp.asarray(width_final)
        a_i = jnp.asarray(semi_major_initial)
        a_f = jnp.asarray(semi_major_final)
        b_i = jnp.asarray(semi_minor_initial)
        b_f = jnp.asarray(semi_minor_final)

        G = E / (2 * (1 + nu))  # Shear modulus

        r_params = (r_i, r_f)
        h_params = (h_i, h_f)
        w_params = (w_i, w_f)
        a_params = (a_i, a_f)
        b_params = (b_i, b_f)

        # === Basis attributes
        basetype = basis_structure.type
        basistype_idx = Basis.BASISTYPE_MAP[basetype]
        Bdof = jnp.asarray(basis_structure.active).flatten()
        Bodr = jnp.asarray(basis_structure.orders).flatten()
        xi_ref = jnp.asarray(reference_strain).reshape(6, 1)

        dof_link = lax.switch(
            index=basistype_idx, branches=Basis.DOF_BRANCHES, operand=(Bdof, Bodr)
        )
        strain_selector_link = jnp.concatenate(
            [
                jnp.ones(dof_link, dtype=jnp.bool_),
                jnp.zeros(max_dof - dof_link, dtype=jnp.bool_),
            ]
        )

        # Gauss points and weights
        integration_points, integration_weights, num_integration_points_i = (
            gauss_quadrature(N_GQ=num_gauss_points)
        )

        deltas = (
            integration_points[1:] - integration_points[:-1]
        )  # Array of shape (num_integration_points - 1, )

        B_branches = [
            lambda operand: vmap(
                lambda X: B_Monomial(X, operand[1], operand[2], max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda X: B_LegendrePolynomial(X, operand[1], operand[2], max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda X: B_Chebychev(X, operand[1], operand[2], max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda X: B_Fourier(X, operand[1], operand[2], max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda X: B_Gaussian(X, operand[1], operand[2], max_dof)
            )(operand[0]),
            lambda operand: vmap(lambda X: B_IMQ(X, operand[1], operand[2], max_dof))(
                operand[0]
            ),
        ]

        B_Xs = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(integration_points, Bdof, Bodr),
        )
        B_Z1 = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(integration_points[:-1] + self.Z1 * deltas, Bdof, Bodr),
        )
        B_Z2 = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(integration_points[:-1] + self.Z2 * deltas, Bdof, Bodr),
        )

        # Compute the initial strain vectors at the integration points
        def xi_reffn(x: Array) -> Array:
            return xi_ref  # TODO: allow to have an expression for xi_ref

        xi_ref_Xs = vmap(xi_reffn)(integration_points).squeeze()
        xi_ref_Z1 = vmap(xi_reffn)(integration_points[:-1] + self.Z1 * deltas).squeeze()
        xi_ref_Z2 = vmap(xi_reffn)(integration_points[:-1] + self.Z2 * deltas).squeeze()

        # Compute the mass, stiffness, and damping matrices at the integration points
        geometric_operand = GeometricOperand(
            integration_points=integration_points,
            r_params=r_params,
            h_params=h_params,
            w_params=w_params,
            a_params=a_params,
            b_params=b_params,
        )
        Ix_p, Iy_p, Iz_p, A_p = lax.switch(
            index=cross_section_geometry_idx,
            branches=Link.geometric_branches(),
            operand=geometric_operand,
        )

        # Prepare the component vectors
        Ms_diag = jnp.stack([Ix_p, Iy_p, Iz_p, A_p, A_p, A_p], axis=1)  # Shape: (np, 6)
        Es_diag = jnp.stack(
            [G * Ix_p, E * Iy_p, E * Iz_p, E * A_p, G * A_p, G * A_p], axis=1
        )
        Gs_diag = jnp.stack([Ix_p, 3 * Iy_p, 3 * Iz_p, 3 * A_p, A_p, A_p], axis=1)

        Ms = rho * vmap(jnp.diag)(Ms_diag)  # Shape: (np, 6, 6)
        Es = vmap(jnp.diag)(Es_diag)
        Gs = eta * vmap(jnp.diag)(Gs_diag)

        # Pad the arrays to the maximum number of integration points and DOFs
        integration_points_full = jnp.pad(
            integration_points,
            (0, max_num_integration_points - num_integration_points_i),
            mode="constant",
        )
        integration_weights_full = jnp.pad(
            integration_weights,
            (0, max_num_integration_points - num_integration_points_i),
            mode="constant",
        )

        Ms_full = jnp.pad(
            Ms,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )
        Es_full = jnp.pad(
            Es,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )
        Gs_full = jnp.pad(
            Gs,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )

        B_Xs_full = jnp.pad(
            B_Xs,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )
        B_Z1_full = jnp.pad(
            B_Z1,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )
        B_Z2_full = jnp.pad(
            B_Z2,
            (
                (0, max_num_integration_points - num_integration_points_i),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
        )

        xi_ref_Xs_full = jnp.pad(
            xi_ref_Xs,
            ((0, max_num_integration_points - num_integration_points_i), (0, 0)),
            mode="constant",
        )
        xi_ref_Z1_full = jnp.pad(
            xi_ref_Z1,
            ((0, max_num_integration_points - num_integration_points_i), (0, 0)),
            mode="constant",
        )
        xi_ref_Z2_full = jnp.pad(
            xi_ref_Z2,
            ((0, max_num_integration_points - num_integration_points_i), (0, 0)),
            mode="constant",
        )

        dofs_joint_link = jnp.stack([dof_joint, dof_link])

        strain_selector_segment = jnp.concatenate(
            [strain_selector_joint, strain_selector_link], axis=0
        )

        return SegmentRuntimeData(
            L=L,
            num_integration_points=jnp.array(num_integration_points_i),
            dofs_joint_link=dofs_joint_link,
            strain_selector=strain_selector_segment,
            integration_points=integration_points_full,
            integration_weights=integration_weights_full,
            mass_matrices=Ms_full,
            stiffness_matrices=Es_full,
            damping_matrices=Gs_full,
            B_joint=B_joint_full,
            B_Xs=B_Xs_full,
            B_Z1=B_Z1_full,
            B_Z2=B_Z2_full,
            xi_ref_joint=xi_ref_joint,
            xi_ref_Xs=xi_ref_Xs_full,
            xi_ref_Z1=xi_ref_Z1_full,
            xi_ref_Z2=xi_ref_Z2_full,
            joint_stiffness=K_joint_full,
        )

    def precompute(self) -> None:
        """
        Precompute any necessary matrices or values for the simulation.

        This method can be expanded to include additional precomputations as needed.
        """
        dofs_flat = self.dofs_per_segment.reshape(-1)
        start_indices_flat = jnp.cumsum(jnp.pad(dofs_flat, (1, 0)))[:-1]
        start_indices = start_indices_flat.reshape(self.num_segments, 2)
        gather_indices = start_indices[..., None] + jnp.arange(self.max_dof)
        gather_mask = jnp.arange(self.max_dof) < self.dofs_per_segment[..., None]

        K_full = self._stiffness_full_matrix()
        D_full = self._damping_full_matrix()
        object.__setattr__(
            self,
            "active_dof_map_blocks",
            self.active_dof_map.reshape(
                self.num_segments, 2, self.max_dof, self.num_dofs
            ),
        )
        object.__setattr__(
            self,
            "inner_integration_weights",
            self.integration_weights[:, 1 : self.max_num_integration_points - 1]
            * self.segment_lengths[:, None],
        )
        object.__setattr__(
            self,
            "inner_mass_matrices",
            self.mass_matrices[:, 1 : self.max_num_integration_points - 1],
        )
        object.__setattr__(self, "gather_indices", gather_indices)
        object.__setattr__(self, "gather_mask", gather_mask)
        object.__setattr__(self, "K_full", K_full)
        object.__setattr__(
            self, "K", self.active_dof_map.T @ K_full @ self.active_dof_map
        )
        object.__setattr__(self, "D_full", D_full)
        object.__setattr__(
            self, "D_active", self.active_dof_map.T @ D_full @ self.active_dof_map
        )

    def _link_parameter_arrays(
        self, link: GVSLinkParams
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
    ]:
        """
        Build link-derived GVS arrays from dynamic link params.

        This helper recomputes only quantities that depend on the physical link
        parameters: length, cross-section geometry, local mass matrices, local
        stiffness matrices, and local damping matrices. It deliberately leaves the
        joint basis, link strain basis, reference strain, and active coordinate
        layout unchanged. That makes it suitable for parameter updates that keep
        the GVS discretization and generalized coordinates fixed.

        Returns:
            A tuple containing:

            - ``segment_lengths``: segment lengths, shape ``(self.num_segments,)``.
            - ``segment_end_positions``: cumulative segment endpoints including zero, shape
              ``(self.num_segments + 1,)``.
            - ``mass_matrices``: local mass matrices at stored quadrature nodes, shape
              ``(self.num_segments, self.max_num_integration_points, 6, 6)``.
            - ``stiffness_matrices``: local stiffness matrices at stored quadrature nodes,
              shape ``(self.num_segments, self.max_num_integration_points, 6, 6)``.
            - ``damping_matrices``: local damping matrices at stored quadrature nodes, shape
              ``(self.num_segments, self.max_num_integration_points, 6, 6)``.
            - ``cross_section_geometry_index``: cross-section enum indices, shape
              ``(self.num_segments,)``.
            - ``radius_params``: circular radius interpolation parameters, shape
              ``(self.num_segments, 2)``.
            - ``height_params``: rectangular height interpolation parameters, shape
              ``(self.num_segments, 2)``.
            - ``width_params``: rectangular width interpolation parameters, shape
              ``(self.num_segments, 2)``.
            - ``semi_major_params``: elliptical semi-major-axis interpolation
              parameters, shape ``(self.num_segments, 2)``.
            - ``semi_minor_params``: elliptical semi-minor-axis interpolation
              parameters, shape ``(self.num_segments, 2)``.

        Raises:
            ValueError: If link params do not contain one entry per segment.
        """
        link.validate()
        if link.length.shape != (self.num_segments,):
            raise ValueError(
                "link params must contain one entry per segment, "
                f"got {link.length.shape[0]} for {self.num_segments} segments."
            )

        segment_length_items = []
        mass_matrix_items = []
        stiffness_matrix_items = []
        damping_matrix_items = []
        cross_section_geometry_items = []
        radius_param_items = []
        height_param_items = []
        width_param_items = []
        semi_major_param_items = []
        semi_minor_param_items = []

        for i_segment, segment_structure in enumerate(self.structure.segments):
            cross_section_geometry_idx = int(
                segment_structure.link.cross_section_geometry
            )

            E = jnp.asarray(link.young_modulus[i_segment], dtype=jnp.float64)
            nu = jnp.asarray(link.poisson_ratio[i_segment], dtype=jnp.float64)
            rho = jnp.asarray(link.density[i_segment], dtype=jnp.float64)
            eta = jnp.asarray(link.damping_coefficient[i_segment], dtype=jnp.float64)
            L = jnp.asarray(link.length[i_segment], dtype=jnp.float64)

            r_params = (
                jnp.asarray(link.radius_initial[i_segment], dtype=jnp.float64),
                jnp.asarray(link.radius_final[i_segment], dtype=jnp.float64),
            )
            h_params = (
                jnp.asarray(link.height_initial[i_segment], dtype=jnp.float64),
                jnp.asarray(link.height_final[i_segment], dtype=jnp.float64),
            )
            w_params = (
                jnp.asarray(link.width_initial[i_segment], dtype=jnp.float64),
                jnp.asarray(link.width_final[i_segment], dtype=jnp.float64),
            )
            a_params = (
                jnp.asarray(link.semi_major_initial[i_segment], dtype=jnp.float64),
                jnp.asarray(link.semi_major_final[i_segment], dtype=jnp.float64),
            )
            b_params = (
                jnp.asarray(link.semi_minor_initial[i_segment], dtype=jnp.float64),
                jnp.asarray(link.semi_minor_final[i_segment], dtype=jnp.float64),
            )

            geometric_operand = GeometricOperand(
                integration_points=self.integration_points[i_segment],
                r_params=r_params,
                h_params=h_params,
                w_params=w_params,
                a_params=a_params,
                b_params=b_params,
            )
            Ix_p, Iy_p, Iz_p, A_p = lax.switch(
                index=cross_section_geometry_idx,
                branches=Link.geometric_branches(),
                operand=geometric_operand,
            )

            G = E / (2 * (1 + nu))
            Ms_diag = jnp.stack([Ix_p, Iy_p, Iz_p, A_p, A_p, A_p], axis=1)
            Es_diag = jnp.stack(
                [G * Ix_p, E * Iy_p, E * Iz_p, E * A_p, G * A_p, G * A_p],
                axis=1,
            )
            Gs_diag = jnp.stack([Ix_p, 3 * Iy_p, 3 * Iz_p, 3 * A_p, A_p, A_p], axis=1)

            segment_length_items.append(L)
            mass_matrix_items.append(rho * vmap(jnp.diag)(Ms_diag))
            stiffness_matrix_items.append(vmap(jnp.diag)(Es_diag))
            damping_matrix_items.append(eta * vmap(jnp.diag)(Gs_diag))
            cross_section_geometry_items.append(cross_section_geometry_idx)
            radius_param_items.append(jnp.stack(r_params))
            height_param_items.append(jnp.stack(h_params))
            width_param_items.append(jnp.stack(w_params))
            semi_major_param_items.append(jnp.stack(a_params))
            semi_minor_param_items.append(jnp.stack(b_params))

        segment_lengths = jnp.stack(segment_length_items)
        segment_end_positions = jnp.cumsum(
            jnp.concatenate(
                [jnp.zeros(1, dtype=segment_lengths.dtype), segment_lengths]
            )
        )

        return (
            segment_lengths,
            segment_end_positions,
            jnp.stack(mass_matrix_items),
            jnp.stack(stiffness_matrix_items),
            jnp.stack(damping_matrix_items),
            jnp.asarray(cross_section_geometry_items, dtype=jnp.int32),
            jnp.stack(radius_param_items),
            jnp.stack(height_param_items),
            jnp.stack(width_param_items),
            jnp.stack(semi_major_param_items),
            jnp.stack(semi_minor_param_items),
        )

    def _reference_strain_runtime_arrays(
        self, reference_strain: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Build padded GVS reference-strain arrays without changing layout."""
        reference_strain = jnp.asarray(reference_strain, dtype=jnp.float64)
        if reference_strain.shape != (self.num_segments, 6):
            raise ValueError(
                "reference_strain must have shape "
                f"({self.num_segments}, 6), got {reference_strain.shape}."
            )
        xs_mask = (
            jnp.arange(self.max_num_integration_points)[None, :]
            < self.num_integration_points[:, None]
        )
        z_mask = jnp.arange(self.max_num_integration_points - 1)[None, :] < (
            self.num_integration_points[:, None] - 1
        )
        xi_ref_joint = jnp.zeros_like(self.xi_ref_joint)
        xi_ref_Xs = reference_strain[:, None, :] * xs_mask[:, :, None]
        xi_ref_Z1 = reference_strain[:, None, :] * z_mask[:, :, None]
        xi_ref_Z2 = reference_strain[:, None, :] * z_mask[:, :, None]
        return xi_ref_joint, xi_ref_Xs, xi_ref_Z1, xi_ref_Z2

    def with_params(self, params: GVSParams) -> "GVS":
        """Return an updated copy with a full typed parameter object."""
        if not isinstance(params, GVSParams):
            raise TypeError("params must be a GVSParams instance.")
        params.validate_against_structure(self.structure)
        link_arrays = self._link_parameter_arrays(params.link)
        (
            xi_ref_joint,
            xi_ref_Xs,
            xi_ref_Z1,
            xi_ref_Z2,
        ) = self._reference_strain_runtime_arrays(params.reference_strain)

        joint_stiffness = jnp.asarray(params.joint_stiffness, dtype=jnp.float64)
        if joint_stiffness.shape != self.joint_stiffness.shape:
            raise ValueError(
                "joint_stiffness shape changes the padded GVS layout; "
                f"expected {self.joint_stiffness.shape}, got {joint_stiffness.shape}."
            )

        gravity = jnp.asarray(params.gravity, dtype=jnp.float64)
        if gravity.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")
        base_pose = jnp.asarray(params.base_pose, dtype=jnp.float64)
        if base_pose.shape != (6,):
            raise ValueError(f"base_pose must have shape (6,), got {base_pose.shape}.")

        updated_self = eqx.tree_at(
            lambda model: (
                model.params,
                model.segment_lengths,
                model.segment_end_positions,
                model.mass_matrices,
                model.stiffness_matrices,
                model.damping_matrices,
                model.cross_section_geometry_index,
                model.radius_params,
                model.height_params,
                model.width_params,
                model.semi_major_params,
                model.semi_minor_params,
                model.xi_ref_joint,
                model.xi_ref_Xs,
                model.xi_ref_Z1,
                model.xi_ref_Z2,
                model.joint_stiffness,
                model.g0,
                model.g,
            ),
            self,
            (
                params,
                *link_arrays,
                xi_ref_joint,
                xi_ref_Xs,
                xi_ref_Z1,
                xi_ref_Z2,
                joint_stiffness,
                lie.exp_SE3(base_pose),
                jnp.concatenate([jnp.zeros(3, dtype=gravity.dtype), gravity]),
            ),
        )
        K_full = updated_self._stiffness_full_matrix()
        D_full = updated_self._damping_full_matrix()
        return eqx.tree_at(
            lambda model: (
                model.inner_integration_weights,
                model.inner_mass_matrices,
                model.K_full,
                model.K,
                model.D_full,
                model.D_active,
            ),
            updated_self,
            (
                updated_self.integration_weights[
                    :, 1 : updated_self.max_num_integration_points - 1
                ]
                * updated_self.segment_lengths[:, None],
                updated_self.mass_matrices[
                    :, 1 : updated_self.max_num_integration_points - 1
                ],
                K_full,
                updated_self.active_dof_map.T @ K_full @ updated_self.active_dof_map,
                D_full,
                updated_self.active_dof_map.T @ D_full @ updated_self.active_dof_map,
            ),
        )

    def update_params(self, **updates: Array) -> "GVS":
        """Return an updated copy with selected typed parameter fields replaced."""
        return self.with_params(self.params.replace(**updates))

    # Gathering functions =========================================================
    @eqx.filter_jit
    def _min_size_gathered(self, vec_min_size_flat: Array) -> Array:
        """
        Function to split the joint coordinates into segments and pad them to the maximum DOF.

        Args:
            vec_min_size_flat (Array): shape (num_dofs, 1) or (num_dofs,) JAX array
                Joint coordinates or joint velocities.

        Returns:
            vec_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Padded joint coordinates.
        """
        vec = jnp.asarray(vec_min_size_flat).reshape(-1)  # Ensure q is a 1D array
        vec_padded = jnp.concatenate([vec, jnp.zeros(self.max_dof, dtype=vec.dtype)])
        vec_gathered = (
            jnp.take(vec_padded, self.gather_indices, mode="clip") * self.gather_mask
        )

        return vec_gathered

    @eqx.filter_jit
    def _max_size_gathered(self, vec_max_size: Array) -> Array:
        """
        Gather the configuration vector into the shape (num_segments, 2, max_dof).

        Args:
            vec_max_size (Array): configuration vector of shape (num_segments * 2 * max_dof, )

        Returns:
            vec_gathered (Array): gathered configuration vector of shape (num_segments, 2, max_dof)
        """
        vec_gathered = vec_max_size.reshape(self.num_segments, 2, self.max_dof)
        return vec_gathered

    @eqx.filter_jit
    def _eval_B_segment(self, i_segment: Array, X: Array) -> Array:
        """
        Evaluate the basis functions for a given segment at specified points.

        Args:
            i_segment (Array): Index of the segment to evaluate.
            X (Array): shape (len(X),) JAX array of points at which to evaluate the basis functions.

        Returns:
            Array: shape (len(X), 6, max_dof) JAX array containing the evaluated basis functions
                for the specified segment at the given points.
        """
        basistype_idx = self.basis_type_index[i_segment]
        Bdof = self.basis_active_params[i_segment]
        Bodr = self.basis_order_params[i_segment]
        B_branches = [
            lambda operand: vmap(
                lambda xx: B_Monomial(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: B_LegendrePolynomial(
                    xx, operand[1], operand[2], self.max_dof
                )
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: B_Chebychev(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: B_Fourier(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: B_Gaussian(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: B_IMQ(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
        ]
        return lax.switch(
            basistype_idx, B_branches, (X, Bdof, Bodr)
        )  # (len(X), 6, max_dof)

    @eqx.filter_jit
    def _eval_dB_segment(self, i_segment: Array, X: Array) -> Array:
        """
        Evaluate ``dB/dX`` for a given segment at normalized positions ``X``.

        Args:
            i_segment: Segment index.
            X: Normalized points in ``[0, 1]`` with shape ``(len(X),)``.

        Returns:
            Basis derivatives with shape ``(len(X), 6, self.max_dof)``.
        """
        basistype_idx = self.basis_type_index[i_segment]
        Bdof = self.basis_active_params[i_segment]
        Bodr = self.basis_order_params[i_segment]
        dB_branches = [
            lambda operand: vmap(
                lambda xx: dB_Monomial(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: dB_LegendrePolynomial(
                    xx, operand[1], operand[2], self.max_dof
                )
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: dB_Chebychev(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: dB_Fourier(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: dB_Gaussian(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
            lambda operand: vmap(
                lambda xx: dB_IMQ(xx, operand[1], operand[2], self.max_dof)
            )(operand[0]),
        ]
        return lax.switch(basistype_idx, dB_branches, (X, Bdof, Bodr))

    def _scaled_link_basis_pair(
        self, length: Array, B_Z1: Array, B_Z2: Array
    ) -> tuple[Array, Array]:
        """
        Apply optional rotational scaling to one pair of link strain bases.

        The GVS link integrator evaluates the strain basis at two internal points
        of each integration cell, denoted ``Z1`` and ``Z2``. Each basis maps the
        padded link coordinates ``q_i`` to a six-dimensional strain contribution:

        ``xi_Zk = B_Zk @ q_i + xi_ref_Zk`` for ``k in {1, 2}``.

        If ``scale_rotational_basis_by_length`` is enabled, the first three rows of
        each basis matrix, corresponding to rotational strain coordinates, are
        divided by the physical link length. The translational rows are left
        unchanged. This helper centralizes that scaling so every Jacobian path
        uses the same convention.

        Args:
            length: Physical length of the current link. Shape ``()``.
            B_Z1: Link strain basis at the first Magnus quadrature point, shape
                ``(6, self.max_dof)``.
            B_Z2: Link strain basis at the second Magnus quadrature point, shape
                ``(6, self.max_dof)``.

        Returns:
            A tuple ``(B_Z1_scaled, B_Z2_scaled)``. Each entry has shape
            ``(6, self.max_dof)``. If rotational scaling is disabled, these are
            the input basis matrices unchanged. If it is enabled, rows ``0:3`` are
            divided by ``length``.
        """
        if self.scale_rotational_basis_by_length:
            B_Z1 = B_Z1.at[:3, :].divide(length)
            B_Z2 = B_Z2.at[:3, :].divide(length)
        return B_Z1, B_Z2

    def _magnus_and_basis_from_strains(
        self,
        length: Array,
        H: Array,
        xi_Z1: Array,
        xi_Z2: Array,
        B_Z1: Array,
        B_Z2: Array,
    ) -> tuple[Array, Array, Array]:
        """
        Compute one second-order GVS Magnus increment and its basis Jacobian.

        This helper is the shared algebraic core for the GVS Jacobian and Jdot
        paths. It expects the strains and basis matrices for the two internal
        cell quadrature points to have already been assembled. If rotational
        basis scaling is active, callers must pass already-scaled bases and the
        corresponding strains.

        For normalized cell width ``H`` and physical link length ``length``, the
        two-point Magnus approximation used here is:

        ``c = sqrt(3) * length**2 * H**2 / 12``

        ``Magnus = length * H / 2 * (xi_Z1 + xi_Z2) + c * ad(xi_Z1) @ xi_Z2``

        Its coordinate Jacobian with respect to the padded link coordinate
        vector ``q_i`` is:

        ``B_Magnus = length * H / 2 * (B_Z1 + B_Z2)``
        ``+ c * (ad(xi_Z1) @ B_Z2 - ad(xi_Z2) @ B_Z1)``

        The returned ``B_Magnus`` maps padded link coordinate rates to the
        Magnus increment rate: ``Magnusd = B_Magnus @ qd_i``.

        Args:
            length: Physical length of the current link. Shape ``()``.
            H: Normalized cell width. This is either a full-cell width
                ``integration_points[j + 1] - integration_points[j]`` or a partial-cell width ``Hp``. Shape
                ``()``.
            xi_Z1: Strain vector at the first Magnus quadrature point, shape
                ``(6,)``.
            xi_Z2: Strain vector at the second Magnus quadrature point, shape
                ``(6,)``.
            B_Z1: Link strain basis at the first quadrature point, shape
                ``(6, self.max_dof)``.
            B_Z2: Link strain basis at the second quadrature point, shape
                ``(6, self.max_dof)``.

        Returns:
            Tuple ``(Magnus, B_Magnus, comm_coeff)``:
            ``Magnus`` is the se(3) increment for the cell, shape ``(6,)``.
            ``B_Magnus`` is ``d(Magnus) / d(q_i)``, shape
            ``(6, self.max_dof)``.
            ``comm_coeff`` is the scalar coefficient ``c`` above, shape ``()``.
        """
        ad_xi_Z1 = lie.adjoint_se3(xi_Z1)
        ad_xi_Z2 = lie.adjoint_se3(xi_Z2)
        comm_coeff = jnp.sqrt(3) * (length**2) * H**2 / 12

        Magnus = length * (H / 2) * (xi_Z1 + xi_Z2) + (comm_coeff * (ad_xi_Z1 @ xi_Z2))
        B_Magnus = length * (H / 2) * (B_Z1 + B_Z2) + (
            comm_coeff * (ad_xi_Z1 @ B_Z2 - ad_xi_Z2 @ B_Z1)
        )
        return Magnus, B_Magnus, comm_coeff

    def _magnus_jacobian_step_terms(
        self,
        length: Array,
        H: Array,
        q_i: Array,
        B_Z1: Array,
        B_Z2: Array,
        xi_ref_Z1: Array,
        xi_ref_Z2: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute local terms for propagating a body-frame Jacobian through a cell.

        This is the Jacobian-only helper for a full or partial GVS link cell. It
        assembles the two quadrature-point strains from the current link
        coordinates, computes the two-point Magnus increment, and returns the
        relative transform/tangent terms required by the recurrence:

        ``J_next = Ad_step_inv @ (J_prev + T_block)``

        where ``T_block`` is zero except for the current link block, which is
        ``T_step @ B_Magnus``. The helper deliberately does not compute
        Jdot-only quantities such as ``Td_step`` or ``B_Magnus_dot``; callers in
        J-only paths avoid that extra work.

        Args:
            length: Physical length of the current link. Shape ``()``.
            H: Normalized cell width. This may be a full cell width or a partial
                cell width. Shape ``()``.
            q_i: Padded generalized coordinates for the current link strain
                basis, shape ``(self.max_dof,)``.
            B_Z1: Link strain basis at the first Magnus quadrature point, shape
                ``(6, self.max_dof)``.
            B_Z2: Link strain basis at the second Magnus quadrature point, shape
                ``(6, self.max_dof)``.
            xi_ref_Z1: Reference strain at the first quadrature point, shape
                ``(6,)``.
            xi_ref_Z2: Reference strain at the second quadrature point, shape
                ``(6,)``.

        Returns:
            Tuple ``(g_step, T_step, Ad_step_inv, B_Magnus)``:
            ``g_step`` is the relative SE(3) transform for the cell, shape
            ``(4, 4)``; ``T_step`` is the tangent map of the Magnus increment,
            shape ``(6, 6)``; ``Ad_step_inv`` is ``Ad(g_step)^(-1)``, shape
            ``(6, 6)``; ``B_Magnus`` is ``d(Magnus) / d(q_i)``, shape
            ``(6, self.max_dof)``.
        """
        B_Z1, B_Z2 = self._scaled_link_basis_pair(length, B_Z1, B_Z2)
        xi_Z1 = B_Z1 @ q_i + xi_ref_Z1
        xi_Z2 = B_Z2 @ q_i + xi_ref_Z2

        Magnus, B_Magnus, _ = self._magnus_and_basis_from_strains(
            length, H, xi_Z1, xi_Z2, B_Z1, B_Z2
        )
        g_step = lie.exp_gn_SE3(Magnus, self.global_eps)
        T_step = lie.Tangent_gi_se3(Magnus, jnp.array(1.0), eps=self.tangent_eps)
        Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)
        return g_step, T_step, Ad_step_inv, B_Magnus

    def _magnus_jacobian_time_derivative_step_terms(
        self,
        length: Array,
        H: Array,
        q_i: Array,
        qd_i: Array,
        B_Z1: Array,
        B_Z2: Array,
        xi_ref_Z1: Array,
        xi_ref_Z2: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute local terms for propagating ``J`` and ``Jdot`` through a cell.

        This is the analytical derivative helper for a full or partial GVS link
        cell. It shares the same Magnus and ``B_Magnus`` algebra as
        ``_magnus_jacobian_step_terms`` and additionally computes the tangent-map
        derivative and the time derivative of the Magnus basis Jacobian.

        With ``xid_Zk = B_Zk @ qd_i`` and
        ``c = sqrt(3) * length**2 * H**2 / 12``, the basis-Jacobian time derivative is:

        ``B_Magnus_dot = c * (ad(xid_Z1) @ B_Z2 - ad(xid_Z2) @ B_Z1)``

        The returned terms are sufficient for the local product-rule recurrence:

        ``J_next = Ad_step_inv @ (J_prev + T_block)``

        ``Jdot_next = Ad_step_inv @ (Jdot_prev + Tdot_block)``
        ``+ Ad_step_inv_dot @ (J_prev + T_block)``

        where ``T_block = T_step @ B_Magnus`` and
        ``Tdot_block = Td_step @ B_Magnus + T_step @ B_Magnus_dot``. No runtime
        autodiff is used in this helper.

        Args:
            length: Physical length of the current link. Shape ``()``.
            H: Normalized cell width. This may be a full cell width or a partial
                cell width. Shape ``()``.
            q_i: Padded generalized coordinates for the current link strain
                basis, shape ``(self.max_dof,)``.
            qd_i: Padded generalized velocities for the current link strain
                basis, shape ``(self.max_dof,)``.
            B_Z1: Link strain basis at the first Magnus quadrature point, shape
                ``(6, self.max_dof)``.
            B_Z2: Link strain basis at the second Magnus quadrature point, shape
                ``(6, self.max_dof)``.
            xi_ref_Z1: Reference strain at the first quadrature point, shape
                ``(6,)``.
            xi_ref_Z2: Reference strain at the second quadrature point, shape
                ``(6,)``.

        Returns:
            Tuple ``(g_step, T_step, Td_step, Ad_step_inv, B_Magnus,
            B_Magnus_dot)``. ``g_step`` has shape ``(4, 4)``; ``T_step``,
            ``Td_step``, and ``Ad_step_inv`` have shape ``(6, 6)``;
            ``B_Magnus`` and ``B_Magnus_dot`` have shape
            ``(6, self.max_dof)``.
        """
        B_Z1, B_Z2 = self._scaled_link_basis_pair(length, B_Z1, B_Z2)
        xi_Z1 = B_Z1 @ q_i + xi_ref_Z1
        xi_Z2 = B_Z2 @ q_i + xi_ref_Z2
        xid_Z1 = B_Z1 @ qd_i
        xid_Z2 = B_Z2 @ qd_i

        Magnus, B_Magnus, comm_coeff = self._magnus_and_basis_from_strains(
            length, H, xi_Z1, xi_Z2, B_Z1, B_Z2
        )
        B_Magnus_dot = comm_coeff * (
            lie.adjoint_se3(xid_Z1) @ B_Z2 - lie.adjoint_se3(xid_Z2) @ B_Z1
        )
        Magnusd = B_Magnus @ qd_i

        g_step = lie.exp_gn_SE3(Magnus, self.global_eps)
        T_step = lie.Tangent_gi_se3(Magnus, jnp.array(1.0), eps=self.tangent_eps)
        Td_step = lie.Tangent_derivative_gi_se3(
            Magnus, Magnusd, jnp.array(1.0), eps=self.tangent_eps
        )
        Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)
        return g_step, T_step, Td_step, Ad_step_inv, B_Magnus, B_Magnus_dot

    def _magnus_arc_length_derivative_step_terms(
        self,
        length: Array,
        H: Array,
        q_i: Array,
        B_Z1: Array,
        B_Z2: Array,
        dB_Z1_dH: Array,
        dB_Z2_dH: Array,
        xi_ref_Z1: Array,
        xi_ref_Z2: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """
        Compute one partial-cell Magnus step and its derivative with respect to ``H``.

        ``H`` is the normalized partial-cell width. The basis derivative inputs
        are already differentiated with respect to ``H`` through the moving
        quadrature sample locations.
        """
        B_Z1, B_Z2 = self._scaled_link_basis_pair(length, B_Z1, B_Z2)
        dB_Z1_dH, dB_Z2_dH = self._scaled_link_basis_pair(length, dB_Z1_dH, dB_Z2_dH)

        xi_Z1 = B_Z1 @ q_i + xi_ref_Z1
        xi_Z2 = B_Z2 @ q_i + xi_ref_Z2
        xi_Z1_H = dB_Z1_dH @ q_i
        xi_Z2_H = dB_Z2_dH @ q_i

        ad_xi_Z1 = lie.adjoint_se3(xi_Z1)
        ad_xi_Z2 = lie.adjoint_se3(xi_Z2)
        ad_xi_Z1_H = lie.adjoint_se3(xi_Z1_H)
        ad_xi_Z2_H = lie.adjoint_se3(xi_Z2_H)

        comm_coeff = jnp.sqrt(3) * (length**2) * H**2 / 12
        comm_coeff_H = jnp.sqrt(3) * (length**2) * H / 6

        Magnus = length * (H / 2) * (xi_Z1 + xi_Z2) + (comm_coeff * (ad_xi_Z1 @ xi_Z2))
        Magnus_H = (
            length / 2 * (xi_Z1 + xi_Z2)
            + length * (H / 2) * (xi_Z1_H + xi_Z2_H)
            + comm_coeff_H * (ad_xi_Z1 @ xi_Z2)
            + comm_coeff * (ad_xi_Z1_H @ xi_Z2 + ad_xi_Z1 @ xi_Z2_H)
        )

        B_Magnus = length * (H / 2) * (B_Z1 + B_Z2) + (
            comm_coeff * (ad_xi_Z1 @ B_Z2 - ad_xi_Z2 @ B_Z1)
        )
        B_Magnus_H = (
            length / 2 * (B_Z1 + B_Z2)
            + length * (H / 2) * (dB_Z1_dH + dB_Z2_dH)
            + comm_coeff_H * (ad_xi_Z1 @ B_Z2 - ad_xi_Z2 @ B_Z1)
            + comm_coeff
            * (
                ad_xi_Z1_H @ B_Z2
                + ad_xi_Z1 @ dB_Z2_dH
                - ad_xi_Z2_H @ B_Z1
                - ad_xi_Z2 @ dB_Z1_dH
            )
        )

        g_step = lie.exp_gn_SE3(Magnus, self.global_eps)
        T_step = lie.Tangent_gi_se3(Magnus, jnp.array(1.0), eps=self.tangent_eps)
        T_step_H = lie.Tangent_derivative_gi_se3(
            Magnus, Magnus_H, jnp.array(1.0), eps=self.tangent_eps
        )
        Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)
        eta_step_H = Ad_step_inv @ (T_step @ Magnus_H)

        return (
            g_step,
            T_step,
            T_step_H,
            Ad_step_inv,
            B_Magnus,
            B_Magnus_H,
            eta_step_H,
        )

    def _joint_jacobian_step_terms(
        self, B_joint: Array, xi_ref_joint: Array, q_joint: Array
    ) -> tuple[Array, Array, Array]:
        """
        Compute joint-local terms for body-frame Jacobian propagation.

        This helper is the joint analogue of the link-cell Magnus helper used in
        the GVS Jacobian recurrences. It evaluates the current joint strain,
        maps it to an SE(3) transform, and computes the product that contributes
        the current joint block to the body-frame Jacobian.

        Args:
            B_joint: Padded joint basis for one segment, shape
                ``(6, self.max_dof)``.
            xi_ref_joint: Reference joint strain, shape ``(6,)``.
            q_joint: Padded generalized joint coordinates for the same segment,
                shape ``(self.max_dof,)``.

        Returns:
            Tuple ``(g_joint, Ad_joint_inv, T_joint_B)``:
            ``g_joint`` is the joint SE(3) transform, shape ``(4, 4)``;
            ``Ad_joint_inv`` is ``Adjoint_g_inv_SE3(g_joint)``, shape
            ``(6, 6)``; ``T_joint_B`` is the tangent-map/basis product
            ``T(g_joint) @ B_joint``, shape ``(6, self.max_dof)``.
        """
        xi_joint = B_joint @ q_joint + xi_ref_joint
        g_joint = lie.exp_gn_SE3(xi_joint, self.global_eps)
        T_joint = lie.Tangent_gi_se3(xi_joint, jnp.array(1.0), eps=self.tangent_eps)
        return g_joint, lie.Adjoint_g_inv_SE3(g_joint), T_joint @ B_joint

    def _joint_jacobian_time_derivative_step_terms(
        self, B_joint: Array, xi_ref_joint: Array, q_joint: Array, qd_joint: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute joint-local terms for coupled ``J`` and ``Jdot`` propagation.

        The returned values support the same product-rule recurrence used by
        the link-cell derivative helpers:

        ``J_next = Ad_inv @ (J_prev + T_joint_B_block)``

        ``Jdot_next = Ad_inv @ (Jdot_prev + Td_joint_B_block)``
        ``+ Ad_inv_dot @ (J_prev + T_joint_B_block)``

        Args:
            B_joint: Padded joint basis for one segment, shape
                ``(6, self.max_dof)``.
            xi_ref_joint: Reference joint strain, shape ``(6,)``.
            q_joint: Padded generalized joint coordinates for the segment,
                shape ``(self.max_dof,)``.
            qd_joint: Padded generalized joint velocities for the segment,
                shape ``(self.max_dof,)``.

        Returns:
            Tuple ``(g_joint, Ad_joint_inv, Ad_joint_inv_dot, T_joint_B,
            Td_joint_B, joint_velocity)``. ``g_joint`` has shape ``(4, 4)``;
            ``Ad_joint_inv`` and ``Ad_joint_inv_dot`` have shape ``(6, 6)``;
            ``T_joint_B`` and ``Td_joint_B`` have shape
            ``(6, self.max_dof)``; ``joint_velocity`` is
            ``T_joint_B @ qd_joint`` with shape ``(6,)``.
        """
        xi_joint = B_joint @ q_joint + xi_ref_joint
        xid_joint = B_joint @ qd_joint
        g_joint = lie.exp_gn_SE3(xi_joint, self.global_eps)
        T_joint = lie.Tangent_gi_se3(xi_joint, jnp.array(1.0), eps=self.tangent_eps)
        Td_joint = lie.Tangent_derivative_gi_se3(
            xi_joint, xid_joint, jnp.array(1.0), eps=self.tangent_eps
        )
        T_joint_B = T_joint @ B_joint
        joint_velocity = T_joint_B @ qd_joint
        Ad_joint_inv = lie.Adjoint_g_inv_SE3(g_joint)
        eta_joint_step = Ad_joint_inv @ joint_velocity
        Ad_joint_inv_dot = -lie.adjoint_se3(eta_joint_step) @ Ad_joint_inv
        return (
            g_joint,
            Ad_joint_inv,
            Ad_joint_inv_dot,
            T_joint_B,
            Td_joint @ B_joint,
            joint_velocity,
        )

    def _jacobian_blocks_to_flat(self, J_blocks: Array) -> Array:
        """
        Flatten segmented full-coordinate Jacobian blocks.

        Args:
            J_blocks: Jacobian blocks grouped as
                ``(self.num_segments, 2, 6, self.max_dof)``, where axis 1
                distinguishes joint and link coordinates.

        Returns:
            J_flat: Flattened full-coordinate Jacobian, shape
            ``(6, self.num_segments * 2 * self.max_dof)``.
        """
        return jnp.transpose(J_blocks, (2, 0, 1, 3)).reshape(
            6, self.num_segments * 2 * self.max_dof
        )

    def _flat_jacobian_to_blocks(self, J_flat: Array) -> Array:
        """
        Restore segmented blocks from a flattened full-coordinate Jacobian.

        Args:
            J_flat: Flattened full-coordinate Jacobian, shape
                ``(6, self.num_segments * 2 * self.max_dof)``.

        Returns:
            J_blocks: Jacobian blocks grouped as
            ``(self.num_segments, 2, 6, self.max_dof)``.
        """
        return jnp.transpose(
            J_flat.reshape(6, self.num_segments, 2, self.max_dof), (1, 2, 0, 3)
        )

    def _rotation_adjoint_from_pose(self, g: Array) -> Array:
        """
        Build the SE(3) adjoint associated with only the pose rotation.

        GVS body-frame Jacobians are rotated into the inertial frame using the
        orientation of the evaluated pose, not the full translational adjoint.
        This helper constructs the corresponding zero-translation SE(3) adjoint.

        Args:
            g: Homogeneous SE(3) pose, shape ``(4, 4)``.

        Returns:
            Ad_g_rot: Rotation-only adjoint matrix, shape ``(6, 6)``.
        """
        R = g[:3, :3]
        g_rot = jnp.block(
            [
                [R, jnp.zeros((3, 1), dtype=R.dtype)],
                [jnp.zeros((1, 3), dtype=R.dtype), jnp.ones((1, 1), dtype=R.dtype)],
            ]
        )
        return lie.Adjoint_g_SE3(g_rot)

    def _body_jacobian_to_inertial(self, g: Array, J_body: Array) -> Array:
        """
        Rotate a body-frame Jacobian into the inertial frame.

        Args:
            g: Homogeneous SE(3) pose at the Jacobian evaluation point, shape
                ``(4, 4)``.
            J_body: Body-frame Jacobian, shape ``(6, self.num_dofs)``.

        Returns:
            J_inertial: Inertial-frame Jacobian, shape
            ``(6, self.num_dofs)``.
        """
        return self._rotation_adjoint_from_pose(g) @ J_body

    def _body_jacobian_time_derivative_to_inertial(
        self, g: Array, J_body: Array, Jd_body: Array, qd: Array
    ) -> tuple[Array, Array]:
        """
        Rotate body-frame ``J`` and ``Jdot`` into the inertial frame.

        The derivative includes the time derivative of the rotation-only
        adjoint, computed from the angular part of ``J_body @ qd``.

        Args:
            g: Homogeneous SE(3) pose at the Jacobian evaluation point, shape
                ``(4, 4)``.
            J_body: Body-frame Jacobian, shape ``(6, self.num_dofs)``.
            Jd_body: Body-frame Jacobian time derivative, shape
                ``(6, self.num_dofs)``.
            qd: Active generalized velocities, shape
                ``(self.num_dofs,)``.

        Returns:
            Tuple ``(J_inertial, Jd_inertial)`` where both arrays have shape
            ``(6, self.num_dofs)``.
        """
        Ad_g = self._rotation_adjoint_from_pose(g)
        eta_body = J_body @ qd
        eta_rot = jnp.concatenate([eta_body[:3], jnp.zeros(3, dtype=eta_body.dtype)])
        Ad_g_dot = Ad_g @ lie.adjoint_se3(eta_rot)
        return Ad_g @ J_body, Ad_g @ Jd_body + Ad_g_dot @ J_body

    def _body_jacobian_arc_length_derivative_to_inertial(
        self, g: Array, J_body: Array, Js_body: Array, gs: Array
    ) -> tuple[Array, Array]:
        """
        Rotate body-frame ``J`` and ``dJ/ds`` into the inertial frame.
        """
        Ad_g = self._rotation_adjoint_from_pose(g)
        omega_hat_body_s = g[:3, :3].T @ gs[:3, :3]
        omega_body_s = 0.5 * jnp.array(
            [
                omega_hat_body_s[2, 1] - omega_hat_body_s[1, 2],
                omega_hat_body_s[0, 2] - omega_hat_body_s[2, 0],
                omega_hat_body_s[1, 0] - omega_hat_body_s[0, 1],
            ],
            dtype=g.dtype,
        )
        eta_rot_s = jnp.concatenate(
            [omega_body_s, jnp.zeros(3, dtype=omega_body_s.dtype)]
        )
        Ad_g_s = Ad_g @ lie.adjoint_se3(eta_rot_s)
        return Ad_g @ J_body, Ad_g_s @ J_body + Ad_g @ Js_body

    def _inertia_integrand(self, weight: Array, J: Array, M: Array) -> Array:
        """
        Compute one quadrature contribution to the generalized inertia matrix.

        Args:
            weight: Length-scaled quadrature weight, shape ``()``.
            J: Body-frame Jacobian at the quadrature node, shape
                ``(6, self.num_dofs)``.
            M: Local spatial mass matrix at the quadrature node, shape
                ``(6, 6)``.

        Returns:
            B_ij: Inertia contribution ``weight * J.T @ M @ J``, shape
            ``(self.num_dofs, self.num_dofs)``.
        """
        return weight * (J.T @ M @ J)

    def _coriolis_matrix_integrand(
        self, weight: Array, J: Array, Jd: Array, M: Array, qd: Array
    ) -> Array:
        """
        Compute one quadrature contribution to the Coriolis matrix.

        Args:
            weight: Length-scaled quadrature weight, shape ``()``.
            J: Body-frame Jacobian at the quadrature node, shape
                ``(6, self.num_dofs)``.
            Jd: Body-frame Jacobian time derivative at the quadrature node, shape
                ``(6, self.num_dofs)``.
            M: Local spatial mass matrix at the quadrature node, shape
                ``(6, 6)``.
            qd: Active generalized velocities, shape
                ``(self.num_dofs,)``.

        Returns:
            C_ij: Coriolis matrix contribution, shape
            ``(self.num_dofs, self.num_dofs)``.
        """
        eta = J @ qd
        return weight * (J.T @ (M @ Jd + lie.coadjoint_se3(eta) @ M @ J))

    def _coriolis_force_integrand(
        self, weight: Array, J: Array, Jd: Array, M: Array, qd: Array
    ) -> Array:
        """
        Compute one quadrature contribution to the convective force vector.

        This is the direct ``C(q, qd) @ qd`` integrand used by the optimized
        forward-dynamics path. It avoids materializing the full Coriolis matrix
        when only its product with ``qd`` is required.

        Args:
            weight: Length-scaled quadrature weight, shape ``()``.
            J: Body-frame Jacobian at the quadrature node, shape
                ``(6, self.num_dofs)``.
            Jd: Body-frame Jacobian time derivative at the quadrature node, shape
                ``(6, self.num_dofs)``.
            M: Local spatial mass matrix at the quadrature node, shape
                ``(6, 6)``.
            qd: Active generalized velocities, shape
                ``(self.num_dofs,)``.

        Returns:
            Cqd_ij: Convective force contribution, shape
            ``(self.num_dofs,)``.
        """
        eta = J @ qd
        return weight * (J.T @ (M @ (Jd @ qd) + lie.coadjoint_se3(eta) @ M @ eta))

    def _gravity_integrand(self, weight: Array, g: Array, J: Array, M: Array) -> Array:
        """
        Compute one quadrature contribution to generalized gravity.

        Args:
            weight: Length-scaled quadrature weight, shape ``()``.
            g: Homogeneous SE(3) pose at the quadrature node, shape ``(4, 4)``.
            J: Body-frame Jacobian at the quadrature node, shape
                ``(6, self.num_dofs)``.
            M: Local spatial mass matrix at the quadrature node, shape
                ``(6, 6)``.

        Returns:
            G_ij: Generalized gravity contribution, shape
            ``(self.num_dofs,)``.
        """
        return -weight * J.T @ M @ lie.Adjoint_g_inv_SE3(g) @ self.g

    # =========================================================================
    @eqx.filter_jit
    def _forward_kinematics_gauss(self, q_gathered: Array) -> Array:
        """
        Compute the forward kinematics transformation matrices at all significant points

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            g_list (Array): shape (num_segments, max_num_integration_points, 4, 4) JAX array
                Forward kinematics transformation matrices at all significant points
        """

        def body_segment_i(carry: Array, i_segment: Array) -> tuple[Array, Array]:
            """Propagate transforms for a segment and collect frames.

            Args:
                carry (Array): Current tip transform `g_tip`, shape (4, 4).
                i_segment (Array): Segment index.

            Returns:
                g_tip_link (Array): Updated tip transform after this segment, shape (4, 4).
                g_link (Array): Per-point transforms for this segment, shape (max_num_integration_points, 4, 4).
            """
            g_tip = carry

            # Joint =======================
            B_joint_i = self.B_joint[i_segment]  # shape (6, max_dof)
            xi_ref_joint_i = self.xi_ref_joint[i_segment]  # shape (6,)
            q_joint_i = q_gathered[i_segment, 0]  # shape (max_dof,)

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i  # shape (6,)

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # shape (4, 4)

            g_j = g_tip @ g_joint_i  # Start with the last transformation matrix

            # Link ========================
            Xs_i = self.integration_points[
                i_segment
            ]  # shape (max_num_integration_points,)
            xi_ref_Z1_i = self.xi_ref_Z1[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6)
            xi_ref_Z2_i = self.xi_ref_Z2[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6)
            length_i = self.segment_lengths[i_segment]  # shape (1,)
            B_Z1_i = self.B_Z1[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6, max_dof)
            B_Z2_i = self.B_Z2[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6, max_dof)

            q_i = q_gathered[i_segment, 1]

            def body_eval_points(carry, j_eval):
                """Advance one cell via Magnus and accumulate the frame.

                Args:
                    carry: Previous transform `g_j_prev`, shape (4, 4).
                    j_eval: Cell index (int).

                Returns:
                    g_j (Array): New transform after the cell, shape (4, 4) (carry).
                    g_j (Array): Same transform as scan output, shape (4, 4).
                """
                g_j_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                xi_ref_Z1_j = xi_ref_Z1_i[j_eval]
                xi_ref_Z2_j = xi_ref_Z2_i[j_eval]
                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                if self.scale_rotational_basis_by_length:
                    B_Z1_j = B_Z1_j.at[:3, :].divide(length_i)
                    B_Z2_j = B_Z2_j.at[:3, :].divide(length_i)

                xi_Z1_j = B_Z1_j @ q_i + xi_ref_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_ref_Z2_j

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_j = length_i * (H / 2) * (xi_Z1_j + xi_Z2_j) + (
                    jnp.sqrt(3) * (length_i**2) * H**2 / 12
                ) * (ad_xi_Z1_j @ xi_Z2_j)

                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)

                g_j = g_j_prev @ g_step

                return g_j, g_j

            indices_eval_points = jnp.arange(self.max_num_integration_points - 1)

            g_tip_link, g_link = lax.scan(
                f=body_eval_points,
                init=g_j,  # Start with the last transformation matrix
                xs=indices_eval_points,
            )

            # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = g_j
            # g_link = []
            # for x in indices_eval_points:
            #     g_j, g_j = body_eval_points(carry, x)
            #     carry = g_j
            #     g_link.append(g_j)
            # g_tip_link = g_j
            # g_link = jnp.array(g_link)

            g_link = jnp.concatenate((jnp.expand_dims(g_j, axis=0), g_link), axis=0)

            return g_tip_link, g_link

        indices_link = jnp.arange(0, self.num_segments)

        g_ini = self.g0

        _, g_list = lax.scan(f=body_segment_i, init=g_ini, xs=indices_link)

        # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = g_ini
        # g_list = []
        # for x in indices_link:
        #     g_tip_link, g_link = body_segment_i(carry, x)
        #     carry = g_tip_link
        #     g_list.append(g_link)
        # g_tip_final = g_tip_link
        # g_list = jnp.array(g_list)

        return g_list

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
            jnp.sum(s > self.segment_end_positions) - 1, 0, self.num_segments - 1
        )

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.segment_end_positions[segment_idx]

        return segment_idx, s_local

    @eqx.filter_jit
    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            g_s (Array): forward kinematics of the robot at point s, shape (4, 4) :
                [[  R,      p],
                [0, 0, 0,   1]] where R is the rotation matrix and p is the position vector.
        """
        q_gathered = self._min_size_gathered(q)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(carry: Array, i_segment: Array) -> tuple[Array, Array]:
            """Compose joint transform and integrate the link up to `s`.

            Args:
                carry: Current tip transform `g_tip`, shape (4, 4).
                i_segment: Segment index (int).

            Returns:
                g_out (Array): Updated tip transform (carry), shape (4, 4).
                g_out (Array): Same transform as scan output, shape (4, 4).
            """
            g_tip = carry

            # Joint =======================
            B_joint = self.B_joint[i_segment]
            xi_ref_joint = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]

            g_joint_i = lie.exp_gn_SE3(
                B_joint @ q_joint_i + xi_ref_joint, self.global_eps
            )

            g_j = g_tip @ g_joint_i

            # Link ========================
            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]

            q_i = q_gathered[i_segment, 1]

            # advance on complete cells if i < seg_idx
            def full_cell(carry: Array, j: Array) -> tuple[Array, None]:
                """Integrate one full cell `j` of the link using Magnus.

                Args:
                    carry: Previous transform, shape (4, 4).
                    j (Array): Cell index.

                Returns:
                    Tuple[Array, None]: Next transform (4, 4) and dummy output.
                """
                g_prev = carry

                H = Xs_i[j + 1] - Xs_i[j]
                B_Z1_j = B_Z1_i[j]
                B_Z2_j = B_Z2_i[j]
                if self.scale_rotational_basis_by_length:
                    B_Z1_j = B_Z1_j.at[:3, :].divide(length_i)
                    B_Z2_j = B_Z2_j.at[:3, :].divide(length_i)

                xi_Z1_j = (B_Z1_j @ q_i) + xi_ref_Z1_i[j]
                xi_Z2_j = (B_Z2_j @ q_i) + xi_ref_Z2_i[j]

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_j = (H / 2) * length_i * (xi_Z1_j + xi_Z2_j) + (
                    jnp.sqrt(3) * (length_i**2) * H**2 / 12
                ) * (ad_xi_Z1_j @ xi_Z2_j)
                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)

                return g_prev @ g_step, None

            # if segment before s: make entire link and return
            def do_full_link() -> Array:
                """Integrate the entire link (segment before the target `s`).

                Returns:
                    g_tip_link (Array): Tip transform after this link, shape (4, 4).
                """
                g_tip_link, _ = lax.scan(
                    full_cell, g_j, jnp.arange(self.max_num_integration_points - 1)
                )
                return g_tip_link

            # if segment is the one where s is located, we need to do a partial link
            def do_partial_link() -> Array:
                """Integrate up to location `s` within this segment.

                Consumes complete cells up to j-1, then a partial cell j.

                Returns:
                    g_out (Array): Tip transform at `s` within this segment, shape (4, 4).
                """
                x = s_local / length_i
                j = jnp.clip(
                    jnp.searchsorted(Xs_i, x) - 1,
                    0,
                    self.max_num_integration_points - 2,
                )
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                # compose complete cells up to j-1 without dynamic-length arange
                def full_cell_masked(carry: Array, idx: Array) -> tuple[Array, None]:
                    """Advance a cell only if `idx < j`, keep state otherwise.

                    Args:
                        carry (Array): Current transform, shape (4, 4).
                        idx (Array): Cell index being processed.

                    Returns:
                        Tuple[Array, None]: Next transform (or unchanged), and dummy output.
                    """
                    g_prev = carry
                    g_next, _ = full_cell(g_prev, idx)
                    return lax.select(idx < j, g_next, g_prev), None

                g_in, _ = lax.scan(
                    full_cell_masked,
                    g_j,
                    jnp.arange(self.max_num_integration_points - 1),
                )

                # enter the (partial) cell j
                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)
                Bp_Z1 = Bp[0]
                Bp_Z2 = Bp[1]

                if self.scale_rotational_basis_by_length:
                    Bp_Z1 = Bp_Z1.at[:3, :].divide(length_i)
                    Bp_Z2 = Bp_Z2.at[:3, :].divide(length_i)

                xi_Z1_j = (Bp_Z1 @ q_i) + self.xi_ref_Z1[i_segment][j]
                xi_Z2_j = (Bp_Z2 @ q_i) + self.xi_ref_Z2[i_segment][j]

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_p = (Hp / 2) * length_i * (xi_Z1_j + xi_Z2_j) + (
                    jnp.sqrt(3) * (length_i**2) * Hp**2 / 12
                ) * (ad_xi_Z1_j @ xi_Z2_j)

                g_p = lie.exp_gn_SE3(Magnus_p, self.global_eps)

                g_out = g_in @ g_p
                return g_out

            g_out = lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

            return g_out, g_out

        g0 = self.g0

        # we scan *at least* up to segment seg_idx; for subsequent segments, we don't change a thing
        def step(carry: Array, i: Array) -> tuple[Array, None]:
            """Scan over segments; freeze state after the target segment.

            Args:
                carry (Array): Current transform, shape (4, 4).
                i (Array): Segment index.

            Returns:
                Tuple[Array, None]: Next transform and dummy output.
            """
            g_curr = carry
            g_next, _ = body_segment_i(g_curr, i)
            return jnp.where(i <= segment_idx, g_next, g_curr), None

        g_s, _ = lax.scan(step, g0, jnp.arange(self.num_segments))

        return g_s

    @eqx.filter_jit
    def _forward_kinematics_arc_length_derivative(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the SE(3) pose at ``s``.
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
        q_gathered = self._min_size_gathered(q)
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(
            carry: tuple[Array, Array], i_segment: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            g_tip, g_s_tip = carry

            B_joint = self.B_joint[i_segment]
            xi_ref_joint = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]
            g_joint_i = lie.exp_gn_SE3(
                B_joint @ q_joint_i + xi_ref_joint, self.global_eps
            )
            g_j = g_tip @ g_joint_i

            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]
            q_i = q_gathered[i_segment, 1]

            def full_cell(carry: Array, j_eval: Array) -> tuple[Array, None]:
                g_prev = carry
                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                g_step, *_ = self._magnus_jacobian_step_terms(
                    length_i,
                    H,
                    q_i,
                    B_Z1_i[j_eval],
                    B_Z2_i[j_eval],
                    xi_ref_Z1_i[j_eval],
                    xi_ref_Z2_i[j_eval],
                )
                return g_prev @ g_step, None

            def do_full_link() -> tuple[Array, Array]:
                g_end, _ = lax.scan(
                    full_cell, g_j, jnp.arange(self.max_num_integration_points - 1)
                )
                return g_end, jnp.zeros_like(g_end)

            def do_partial_link() -> tuple[Array, Array]:
                x = s_local / length_i
                j = jnp.clip(
                    jnp.searchsorted(Xs_i, x) - 1,
                    0,
                    self.max_num_integration_points - 2,
                )
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                def full_cell_masked(carry: Array, idx: Array) -> tuple[Array, None]:
                    g_next, _ = full_cell(carry, idx)
                    return lax.select(idx < j, g_next, carry), None

                g_in, _ = lax.scan(
                    full_cell_masked,
                    g_j,
                    jnp.arange(self.max_num_integration_points - 1),
                )

                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)
                dBp = self._eval_dB_segment(i_segment, Xp)
                dBp_dH = jnp.stack([self.Z1 * dBp[0], self.Z2 * dBp[1]])

                (
                    g_step,
                    _T_step,
                    _T_step_H,
                    _Ad_step_inv,
                    _B_Magnus,
                    _B_Magnus_H,
                    eta_step_H,
                ) = self._magnus_arc_length_derivative_step_terms(
                    length_i,
                    Hp,
                    q_i,
                    Bp[0],
                    Bp[1],
                    dBp_dH[0],
                    dBp_dH[1],
                    xi_ref_Z1_i[j],
                    xi_ref_Z2_i[j],
                )

                g_out = g_in @ g_step
                eta_step_s = eta_step_H / length_i
                return g_out, g_out @ lie.hat_SE3(eta_step_s)

            return lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

        def step(
            carry: tuple[Array, Array], i: Array
        ) -> tuple[tuple[Array, Array], None]:
            g_curr, g_s_curr = carry
            g_next, g_s_next = body_segment_i(carry, i)
            return (
                jnp.where(i <= segment_idx, g_next, g_curr),
                jnp.where(i <= segment_idx, g_s_next, g_s_curr),
            ), None

        (g_s, g_s_derivative), _ = lax.scan(
            step,
            (self.g0, jnp.zeros_like(self.g0)),
            jnp.arange(self.num_segments),
        )

        return g_s, g_s_derivative

    @eqx.filter_jit
    def forward_kinematics_tips(self, q: Array) -> Array:
        """
        Compute the forward kinematics at all link tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            g_tips (Array): forward kinematics at each link tip, shape
                (num_segments, 4, 4).
        """
        q_gathered = self._min_size_gathered(q)
        return self._forward_kinematics_gauss(q_gathered)[:, -1]

    @eqx.filter_jit
    def forward_kinematics_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the forward kinematics of the robot at a batch of points along the backbone.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): points along the robot arclength in the interval [0, L], shape (N,).

        Returns:
            Array: shape (N, 4, 4) with the homogeneous transforms evaluated at each ``s``.
        """
        q_gathered = self._min_size_gathered(q)
        g_gauss = self._forward_kinematics_gauss(q_gathered)
        s_flat = jnp.asarray(s_ps).reshape(-1)

        def fk_single(s_val: Array) -> Array:
            i, s_local = self.classify_segment(s_val)

            length_i = self.segment_lengths[i]
            length_safe = jnp.where(jnp.abs(length_i) < self.global_eps, 1.0, length_i)
            x_norm = jnp.where(
                jnp.abs(length_i) < self.global_eps,
                0.0,
                jnp.clip(s_local / length_safe, 0.0, 1.0),
            )

            Xs_i = self.integration_points[i]
            g_nodes = g_gauss[i]
            q_link = q_gathered[i, 1]
            xi_ref_Z1_i = self.xi_ref_Z1[i]
            xi_ref_Z2_i = self.xi_ref_Z2[i]

            cell_idx = jnp.clip(
                jnp.searchsorted(Xs_i, x_norm, side="right") - 1,
                0,
                self.max_num_integration_points - 2,
            )

            distances = jnp.abs(Xs_i - x_norm)
            closest_idx = jnp.argmin(distances)
            base_idx = jnp.minimum(closest_idx, cell_idx)

            g_base = g_nodes[base_idx]
            x_base = Xs_i[base_idx]

            Hp_signed = x_norm - x_base
            ds = Hp_signed * length_i

            def compute_partial_cell(_) -> Array:
                Xp = jnp.array(
                    [x_base + self.Z1 * Hp_signed, x_base + self.Z2 * Hp_signed]
                )
                Bp = self._eval_B_segment(i, Xp)
                xi_Z1 = Bp[0] @ q_link + xi_ref_Z1_i[cell_idx]
                xi_Z2 = Bp[1] @ q_link + xi_ref_Z2_i[cell_idx]
                ad_xi_Z1 = lie.adjoint_se3(xi_Z1)
                Magnus = (ds / 2.0) * (xi_Z1 + xi_Z2) + (
                    jnp.sqrt(3.0) * ds * ds / 12.0
                ) * (ad_xi_Z1 @ xi_Z2)
                return lie.exp_gn_SE3(Magnus, self.global_eps)

            g_partial = lax.cond(
                jnp.logical_or(
                    jnp.abs(Hp_signed) <= self.global_eps,
                    jnp.abs(length_i) < self.global_eps,
                ),
                lambda _: jnp.eye(4, dtype=g_base.dtype),
                compute_partial_cell,
                operand=None,
            )

            return g_base @ g_partial

        g_ps = vmap(fk_single)(s_flat)

        return g_ps

    @eqx.filter_jit
    def _jacobian_gauss(self, q_gathered: Array) -> Array:
        """
        Compute the Jacobian matrices at all significant points of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            J_list (Array): shape (num_segments, max_num_integration_points, num_segments, 2, 6, max_dof) JAX array
                Jacobian matrices at all significant points
                J_list[i_segment, j_nip, k_link, type_contrib, s, d]
                    i_segment : link for which we calculate the Jacobian
                    j_nip : evaluation point index on i_segment
                    k_link : contributing link (i.e. the part of the kinematic chain that influences this point)
                    type_contrib ∈ {0,1} : type of contribution :
                        0 = articulation (joint)
                        1 = rigid segment (Magnus step)
                    s ∈ {0..5} : spatial components of viscinematics (twist)
                    d ∈ {0..max_dof-1} : partial derivative with respect to the d-th DoF of the link k_link
        """

        def body_segment_i(
            carry: Array, i_segment: Array
        ) -> tuple[tuple[Array, Array], Array]:
            """Accumulate transforms/Jacobians for a segment.

            Args:
                carry: Tuple (g_tip, J_tip) with shapes (4, 4) and
                    (num_segments, 2, 6, max_dof).
                i_segment: Segment index (int).

            Returns:
                Tuple[Tuple[Array, Array], Array]:
                    - (g_tip_link, J_tip_link): tip state with
                      shapes (4, 4) and (num_segments, 2, 6, max_dof).
                    - J_link: per-point Jacobians for this segment, shape
                      (max_num_integration_points, num_segments, 2, 6, max_dof).
            """
            g_tip, J_tip = carry

            # Joint ============================
            B_joint_i = self.B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]

            g_joint_i, Ad_g_joint_inv, T_joint_B_i = self._joint_jacobian_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i
            )

            T_g_joint_i_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_joint_B_i)
            )

            g_j = g_tip @ g_joint_i  # shape (4, 4)
            J_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, (J_tip + T_g_joint_i_B_joint_i)
            )  # shape (num_segments, 6, max_dof)

            # Link ========================
            Xs_i = self.integration_points[
                i_segment
            ]  # shape (max_num_integration_points,)
            xi_ref_Z1_i = self.xi_ref_Z1[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6)
            xi_ref_Z2_i = self.xi_ref_Z2[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6)
            length_i = self.segment_lengths[i_segment]  # shape (1,)
            B_Z1_i = self.B_Z1[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6, max_dof)
            B_Z2_i = self.B_Z2[
                i_segment
            ]  # shape (max_num_integration_points - 1, 6, max_dof)

            q_i = q_gathered[i_segment, 1]

            def body_eval_points(
                carry: Array, j_eval: Array
            ) -> tuple[tuple[Array, Array], Array]:
                """Advance one cell and update the Jacobian via Magnus terms.

                Args:
                    carry (Array): Tuple (g_prev, J_prev) with shapes (4, 4) and
                        (num_segments, 2, 6, max_dof).
                    j_eval (Array): Cell index (int).

                Returns:
                    Tuple[Tuple[Array, Array], Array]:
                        - (g_next, J_next): updated carry with same shapes as input.
                        - J_next: Jacobian, shape
                          (num_segments, 2, 6, max_dof).
                """
                g_prev, J_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                g_step, T_step, Ad_g_step_inv, B_Magnus_j = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        H,
                        q_i,
                        B_Z1_i[j_eval],
                        B_Z2_i[j_eval],
                        xi_ref_Z1_i[j_eval],
                        xi_ref_Z2_i[j_eval],
                    )
                )

                T_step_B_step = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )

                g_next = g_prev @ g_step
                J_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_g_step_inv, (J_prev + T_step_B_step)
                )  # shape (num_segments, 6, max_dof)

                return (g_next, J_next), J_next

            indices_eval_points = jnp.arange(self.max_num_integration_points - 1)

            (g_tip_link, J_tip_link), J_link = lax.scan(
                f=body_eval_points, init=(g_j, J_j), xs=indices_eval_points
            )

            # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = (g_j, J_j)
            # J_link = []
            # for x in indices_eval_points:
            #     (g_j, J_j), J_j_here = body_eval_points(carry, x)
            #     carry = (g_j, J_j)
            #     J_link.append(J_j_here)
            # (g_tip_link, J_tip_link) = carry
            # J_link = jnp.array(J_link)  # shape (max_num_integration_points - 1, num_segments, 6, max_dof)

            J_link = jnp.concatenate((jnp.expand_dims(J_j, axis=0), J_link), axis=0)
            return (g_tip_link, J_tip_link), J_link

        indices_link = jnp.arange(0, self.num_segments)

        g_init = self.g0

        J_init = jnp.zeros((self.num_segments, 2, 6, self.max_dof))

        _, J_list = lax.scan(f=body_segment_i, init=(g_init, J_init), xs=indices_link)

        # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = (g_init, J_init)
        # J_list = []
        # for x in indices_link:
        #     (g_tip_link, J_tip_link), J_j = body_segment_i(carry, x)
        #     carry = (g_tip_link, J_tip_link)
        #     J_list.append(J_j)
        # J_list = jnp.array(J_list)

        # (num_segments, max_num_integration_points, num_segments, 2, 6, max_dof) => (num_segments, max_num_integration_points, 6, num_segments*2*max_dof)
        J_reordered = jnp.transpose(J_list, (0, 1, 4, 2, 3, 5))
        J = J_reordered.reshape(
            self.num_segments,
            self.max_num_integration_points,
            6,
            self.num_segments * 2 * self.max_dof,
        )

        return J

    @eqx.filter_jit
    def _jacobian_bodyframe_terms(self, q: Array, s: Array) -> tuple[Array, Array]:
        """
        Compute the pose and body-frame Jacobian at ``s`` in one scan.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            g_s (Array): SE(3) pose at point s.
            J_local (Array): Body-frame Jacobian at point s.
        """
        q_gathered = self._min_size_gathered(q)

        # classify where s lies
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(
            carry: Array, i_segment: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            """Propagate body-frame Jacobian across a segment up to `s`.

            Args:
                carry (Array): Tuple (g_tip, J_tip) with shapes (4, 4) and
                    (num_segments, 2, 6, max_dof).
                i_segment (Array): Segment index (int).

            Returns:
                Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
                    - (g_pass, J_pass): tip state to pass forward.
                    - (g_pass, J_pass): same as scan output.
            """
            g_tip, J_tip = carry

            # Joint ============================
            B_joint_i = self.B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]

            g_joint_i, Ad_g_joint_inv, T_joint_B_i = self._joint_jacobian_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i
            )

            # contribution of this joint to its own block
            T_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_joint_B_i)
            )

            # Work directly in physical coordinates (consistent with FK scaling)
            g_j = g_tip @ g_joint_i
            J_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, (J_tip + T_g_joint_B_joint_i)
            )

            # Link ========================
            Xs_i = self.integration_points[i_segment]  # (max_num_integration_points,)
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]  # (max_num_integration_points-1, 6)
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]  # (max_num_integration_points-1, 6)
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]  # (max_num_integration_points-1, 6, max_dof)
            B_Z2_i = self.B_Z2[i_segment]  # (max_num_integration_points-1, 6, max_dof)

            q_i = q_gathered[i_segment, 1]

            def full_cell(
                carry: Array, j_eval: Array
            ) -> tuple[tuple[Array, Array], None]:
                """Consume a full cell; update g and J in body frame.

                Args:
                    carry (Array): Tuple (g_prev, J_prev) with shapes (4, 4) and
                        (num_segments, 2, 6, max_dof).
                    j_eval (Array): Cell index.

                Returns:
                    Tuple[Tuple[Array, Array], None]: Updated carry and dummy output.
                """
                g_prev, J_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                g_step, T_step, Ad_step_inv, B_Magnus_j = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        H,
                        q_i,
                        B_Z1_i[j_eval],
                        B_Z2_i[j_eval],
                        xi_ref_Z1_i[j_eval],
                        xi_ref_Z2_i[j_eval],
                    )
                )

                # add contribution for this link block
                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )

                g_next = g_prev @ g_step
                J_next = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, (J_prev + T_block))

                return (g_next, J_next), None

            # If this segment is before the target, consume all cells.
            def do_full_link() -> tuple[Array, Array]:
                """Segment before target `s`: consume all its cells.

                Returns:
                    Tuple[Array, Array]: (g_end, J_end), shapes (4, 4) and
                    (num_segments, 2, 6, max_dof).
                """
                (g_end, J_end), _ = lax.scan(
                    full_cell,
                    (g_j, J_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )
                return g_end, J_end

            # If this segment is the one containing s, step up to the cell j-1 (masked), then do a partial cell of size Hp.
            def do_partial_link() -> tuple[Array, Array]:
                """Segment containing `s`: consume up to j-1, then partial cell.

                Returns:
                    Tuple[Array, Array]: (g_out, J_out) at location `s`.
                """
                x = s_local / length_i
                j = jnp.clip(
                    jnp.searchsorted(Xs_i, x) - 1,
                    0,
                    self.max_num_integration_points - 2,
                )
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                # masked full cells up to j-1
                def full_cell_masked(
                    carry: Array, idx: Array
                ) -> tuple[tuple[Array, Array], None]:
                    """Advance only while idx < j; otherwise keep state unchanged.

                    Args:
                        carry: Tuple (g_prev, J_prev).
                        idx: Cell index being processed.

                    Returns:
                        Tuple[Tuple[Array, Array], None]: Updated or kept carry and dummy output.
                    """
                    (g_prev, J_prev), _ = full_cell(carry, idx)
                    # keep state unchanged once idx >= j
                    g_keep, J_keep = carry
                    g_out = lax.select(idx < j, g_prev, g_keep)
                    J_out = lax.select(idx < j, J_prev, J_keep)
                    return (g_out, J_out), None

                (g_in, J_in), _ = lax.scan(
                    full_cell_masked,
                    (g_j, J_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )

                # partial cell j with Hp
                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)
                g_step, T_step, Ad_step_inv, B_Magnus_p = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        Hp,
                        q_i,
                        Bp[0],
                        Bp[1],
                        xi_ref_Z1_i[j],
                        xi_ref_Z2_i[j],
                    )
                )

                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_p)
                )

                g_out = g_in @ g_step
                J_out = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, (J_in + T_block))
                return g_out, J_out

            g_pass, J_pass = lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

            return (g_pass, J_pass), (g_pass, J_pass)

        # walk the chain, but freeze state after we pass the segment that contains s
        def step(carry: Array, i: Array) -> tuple[tuple[Array, Array], None]:
            """Walk segments; freeze state after the one containing `s`.

            Args:
                carry: Tuple (g_curr, J_curr).
                i: Segment index.

            Returns:
                Tuple[Tuple[Array, Array], None]: Next carry and dummy output.
            """
            (g_curr, J_curr), _ = body_segment_i(carry, i)
            g_next = jnp.where(i <= segment_idx, g_curr, carry[0])
            J_next = jnp.where(i <= segment_idx, J_curr, carry[1])
            return (g_next, J_next), None

        g0 = self.g0

        J0 = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        (g_s, J_full), _ = lax.scan(step, (g0, J0), jnp.arange(self.num_segments))

        # J_full is (num_segments,2,6,max_dof) at location s in body frame.
        # Flatten to (6, num_segments*2*max_dof), then project to active strains.
        J_local = self._jacobian_blocks_to_flat(J_full) @ self.active_dof_map

        return g_s, J_local

    @eqx.filter_jit
    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.
        """
        _, J_local = self._jacobian_bodyframe_terms(q, s)
        return J_local

    @eqx.filter_jit
    def _jacobian_and_arc_length_derivative_bodyframe_terms(
        self, q: Array, s: Array
    ) -> tuple[Array, Array, Array, Array]:
        """
        Compute pose, body-frame Jacobian, and arc-length terms at ``s``.
        """
        q_gathered = self._min_size_gathered(q)
        segment_idx, s_local = self.classify_segment(s)
        gs0 = jnp.zeros((4, 4), dtype=q.dtype)

        def body_segment_i(
            carry: tuple[Array, Array, Array, Array], i_segment: Array
        ) -> tuple[Array, Array, Array, Array]:
            g_tip, J_tip, Js_tip, gs_tip = carry

            B_joint_i = self.B_joint[i_segment]
            xi_ref_joint_i = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]

            g_joint_i, Ad_g_joint_inv, T_joint_B_i = self._joint_jacobian_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i
            )
            T_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_joint_B_i)
            )

            g_j = g_tip @ g_joint_i
            J_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, J_tip + T_g_joint_B_joint_i
            )

            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]
            q_i = q_gathered[i_segment, 1]

            def full_cell(
                carry: tuple[Array, Array], j_eval: Array
            ) -> tuple[tuple[Array, Array], None]:
                g_prev, J_prev = carry
                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                g_step, T_step, Ad_step_inv, B_Magnus_j = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        H,
                        q_i,
                        B_Z1_i[j_eval],
                        B_Z2_i[j_eval],
                        xi_ref_Z1_i[j_eval],
                        xi_ref_Z2_i[j_eval],
                    )
                )

                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )

                g_next = g_prev @ g_step
                J_next = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, J_prev + T_block)
                return (g_next, J_next), None

            def do_full_link() -> tuple[Array, Array, Array, Array]:
                (g_end, J_end), _ = lax.scan(
                    full_cell,
                    (g_j, J_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )
                return g_end, J_end, jnp.zeros_like(J_end), gs0

            def do_partial_link() -> tuple[Array, Array, Array, Array]:
                x = s_local / length_i
                j = jnp.clip(
                    jnp.searchsorted(Xs_i, x) - 1,
                    0,
                    self.max_num_integration_points - 2,
                )
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                def full_cell_masked(
                    carry: tuple[Array, Array], idx: Array
                ) -> tuple[tuple[Array, Array], None]:
                    g_next, J_next = full_cell(carry, idx)[0]
                    g_keep, J_keep = carry
                    return (
                        lax.select(idx < j, g_next, g_keep),
                        lax.select(idx < j, J_next, J_keep),
                    ), None

                (g_in, J_in), _ = lax.scan(
                    full_cell_masked,
                    (g_j, J_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )

                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)
                dBp = self._eval_dB_segment(i_segment, Xp)
                dBp_dH = jnp.stack([self.Z1 * dBp[0], self.Z2 * dBp[1]])

                (
                    g_step,
                    T_step,
                    T_step_H,
                    Ad_step_inv,
                    B_Magnus_p,
                    B_Magnus_H_p,
                    eta_step_H,
                ) = self._magnus_arc_length_derivative_step_terms(
                    length_i,
                    Hp,
                    q_i,
                    Bp[0],
                    Bp[1],
                    dBp_dH[0],
                    dBp_dH[1],
                    xi_ref_Z1_i[j],
                    xi_ref_Z2_i[j],
                )

                T_block = jnp.zeros_like(J_in).at[i_segment, 1].set(T_step @ B_Magnus_p)
                T_block_H = (
                    jnp.zeros_like(J_in)
                    .at[i_segment, 1]
                    .set(T_step_H @ B_Magnus_p + T_step @ B_Magnus_H_p)
                )
                Ad_step_inv_H = -lie.adjoint_se3(eta_step_H) @ Ad_step_inv

                g_out = g_in @ g_step
                J_out = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, J_in + T_block)
                J_H = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv_H, J_in + T_block
                ) + jnp.einsum("ij,nmjk->nmik", Ad_step_inv, T_block_H)
                Js = J_H / length_i
                gs = g_out @ lie.hat_SE3(eta_step_H / length_i)
                return g_out, J_out, Js, gs

            return lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

        def step(
            carry: tuple[Array, Array, Array, Array], i: Array
        ) -> tuple[tuple[Array, Array, Array, Array], None]:
            g_curr, J_curr, Js_curr, gs_curr = carry
            g_next, J_next, Js_next, gs_next = body_segment_i(carry, i)
            return (
                jnp.where(i <= segment_idx, g_next, g_curr),
                jnp.where(i <= segment_idx, J_next, J_curr),
                jnp.where(i <= segment_idx, Js_next, Js_curr),
                jnp.where(i <= segment_idx, gs_next, gs_curr),
            ), None

        J0 = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        (g_s, J_full, Js_full, gs), _ = lax.scan(
            step,
            (self.g0, J0, J0, gs0),
            jnp.arange(self.num_segments),
        )

        J_body = self._jacobian_blocks_to_flat(J_full) @ self.active_dof_map
        Js_body = self._jacobian_blocks_to_flat(Js_full) @ self.active_dof_map
        return g_s, J_body, Js_body, gs

    @eqx.filter_jit
    def jacobian_and_arc_length_derivative_bodyframe(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the body-frame Jacobian and its arc-length derivative at ``s``.
        """
        _, J_body, Js_body, _ = (
            self._jacobian_and_arc_length_derivative_bodyframe_terms(q, s)
        )
        return J_body, Js_body

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
        Compute the body-frame Jacobian at multiple arclength locations.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): points along the backbone in [0, L], shape (N,).

        Returns:
            Array: Jacobians with shape (N, 6, num_active_strains).
        """
        q_gathered = self._min_size_gathered(q)
        J_gauss_full = self._jacobian_gauss(q_gathered)

        s_flat = jnp.asarray(s_ps).reshape(-1)

        def body_single(s_val: Array) -> Array:
            i, s_local = self.classify_segment(s_val)

            length_i = self.segment_lengths[i]
            length_safe = jnp.where(jnp.abs(length_i) < self.global_eps, 1.0, length_i)
            x_norm = jnp.where(
                jnp.abs(length_i) < self.global_eps,
                0.0,
                jnp.clip(s_local / length_safe, 0.0, 1.0),
            )

            Xs_i = self.integration_points[i]
            J_nodes = J_gauss_full[i]
            q_link = q_gathered[i, 1]
            xi_ref_Z1_i = self.xi_ref_Z1[i]
            xi_ref_Z2_i = self.xi_ref_Z2[i]

            cell_idx = jnp.clip(
                jnp.searchsorted(Xs_i, x_norm, side="right") - 1,
                0,
                self.max_num_integration_points - 2,
            )

            distances = jnp.abs(Xs_i - x_norm)
            closest_idx = jnp.argmin(distances)
            base_idx = jnp.minimum(closest_idx, cell_idx)

            J_base_flat = J_nodes[base_idx]
            J_blocks_base = self._flat_jacobian_to_blocks(J_base_flat)
            x_base = Xs_i[base_idx]

            Hp = x_norm - x_base

            def integrate_partial(_) -> Array:
                Xp = jnp.array(
                    [x_base + self.Z1 * Hp, x_base + self.Z2 * Hp], dtype=x_norm.dtype
                )
                Bp = self._eval_B_segment(i, Xp)
                _g_step, T_step, Ad_step_inv, B_Magnus = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        Hp,
                        q_link,
                        Bp[0],
                        Bp[1],
                        xi_ref_Z1_i[cell_idx],
                        xi_ref_Z2_i[cell_idx],
                    )
                )

                T_block = jnp.zeros_like(J_blocks_base).at[i, 1].set(T_step @ B_Magnus)

                return jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (J_blocks_base + T_block)
                )

            J_blocks = lax.cond(
                jnp.logical_or(
                    jnp.abs(Hp) <= self.global_eps,
                    jnp.abs(length_i) < self.global_eps,
                ),
                lambda _: J_blocks_base,
                integrate_partial,
                operand=None,
            )

            J_flat = self._jacobian_blocks_to_flat(J_blocks)
            return J_flat @ self.active_dof_map

        return vmap(body_single)(s_flat)

    @eqx.filter_jit
    def jacobian_and_time_derivative_bodyframe_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute body-frame Jacobians and time derivatives at multiple arclength locations.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): generalized velocities of shape (num_active_strains,).
            s_ps (Array): points along the backbone in [0, L], shape (N,).

        Returns:
            Tuple[Array, Array]: Jacobians and derivatives with shape
            (N, 6, num_active_strains).
        """
        q_gathered = self._min_size_gathered(q)
        qd_gathered = self._min_size_gathered(qd)
        J_gauss_full = self._jacobian_gauss(q_gathered)
        Jd_gauss_full = self._jacobian_time_derivative_gauss(q_gathered, qd_gathered)

        s_flat = jnp.asarray(s_ps).reshape(-1)

        def body_single(s_val: Array) -> tuple[Array, Array]:
            i, s_local = self.classify_segment(s_val)

            length_i = self.segment_lengths[i]
            length_safe = jnp.where(jnp.abs(length_i) < self.global_eps, 1.0, length_i)
            x_norm = jnp.where(
                jnp.abs(length_i) < self.global_eps,
                0.0,
                jnp.clip(s_local / length_safe, 0.0, 1.0),
            )

            Xs_i = self.integration_points[i]
            J_nodes = J_gauss_full[i]
            Jd_nodes = Jd_gauss_full[i]
            q_link = q_gathered[i, 1]
            qd_link = qd_gathered[i, 1]
            xi_ref_Z1_i = self.xi_ref_Z1[i]
            xi_ref_Z2_i = self.xi_ref_Z2[i]

            cell_idx = jnp.clip(
                jnp.searchsorted(Xs_i, x_norm, side="right") - 1,
                0,
                self.max_num_integration_points - 2,
            )

            distances = jnp.abs(Xs_i - x_norm)
            closest_idx = jnp.argmin(distances)
            base_idx = jnp.minimum(closest_idx, cell_idx)

            J_base_flat = J_nodes[base_idx]
            Jd_base_flat = Jd_nodes[base_idx]
            J_blocks_base = self._flat_jacobian_to_blocks(J_base_flat)
            Jd_blocks_base = self._flat_jacobian_to_blocks(Jd_base_flat)
            x_base = Xs_i[base_idx]

            Hp = x_norm - x_base

            def integrate_partial(_) -> tuple[Array, Array]:
                Xp = jnp.array(
                    [x_base + self.Z1 * Hp, x_base + self.Z2 * Hp], dtype=x_norm.dtype
                )
                Bp = self._eval_B_segment(i, Xp)
                (
                    _g_step,
                    T_step,
                    Td_step,
                    Ad_step_inv,
                    B_Magnus,
                    B_Magnus_dot,
                ) = self._magnus_jacobian_time_derivative_step_terms(
                    length_i,
                    Hp,
                    q_link,
                    qd_link,
                    Bp[0],
                    Bp[1],
                    xi_ref_Z1_i[cell_idx],
                    xi_ref_Z2_i[cell_idx],
                )

                T_block = jnp.zeros_like(J_blocks_base).at[i, 1].set(T_step @ B_Magnus)
                Td_block = (
                    jnp.zeros_like(Jd_blocks_base)
                    .at[i, 1]
                    .set(Td_step @ B_Magnus + T_step @ B_Magnus_dot)
                )

                eta_step = Ad_step_inv @ (T_step @ B_Magnus @ qd_link)
                Ad_step_inv_dot = -lie.adjoint_se3(eta_step) @ Ad_step_inv

                J_blocks = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, J_blocks_base + T_block
                )
                Jd_blocks = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, Jd_blocks_base + Td_block
                ) + jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv_dot, J_blocks_base + T_block
                )
                return J_blocks, Jd_blocks

            J_blocks, Jd_blocks = lax.cond(
                jnp.logical_or(
                    jnp.abs(Hp) <= self.global_eps,
                    jnp.abs(length_i) < self.global_eps,
                ),
                lambda _: (J_blocks_base, Jd_blocks_base),
                integrate_partial,
                operand=None,
            )

            J_flat = self._jacobian_blocks_to_flat(J_blocks)
            Jd_flat = self._jacobian_blocks_to_flat(Jd_blocks)
            return J_flat @ self.active_dof_map, Jd_flat @ self.active_dof_map

        return vmap(body_single)(s_flat)

    @eqx.filter_jit
    def jacobian_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_configurations,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_configurations)
        """
        g_s, J_local = self._jacobian_bodyframe_terms(q, s)
        return self._body_jacobian_to_inertial(g_s, J_local)

    @eqx.filter_jit
    def jacobian_and_arc_length_derivative_inertialframe(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the inertial-frame Jacobian and its arc-length derivative at ``s``.
        """
        g_s, J_body, Js_body, gs = (
            self._jacobian_and_arc_length_derivative_bodyframe_terms(q, s)
        )
        return self._body_jacobian_arc_length_derivative_to_inertial(
            g_s, J_body, Js_body, gs
        )

    @eqx.filter_jit
    def jacobian_arc_length_derivative_inertialframe(self, q: Array, s: Array) -> Array:
        """
        Compute the arc-length derivative of the inertial-frame Jacobian at ``s``.
        """
        g_s, J_body, Js_body, gs = (
            self._jacobian_and_arc_length_derivative_bodyframe_terms(q, s)
        )
        _, Js = self._body_jacobian_arc_length_derivative_to_inertial(
            g_s, J_body, Js_body, gs
        )
        return Js

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
    def _jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the inertial-frame Jacobian at a point along the robot.

        This public adapter satisfies the SoftRobot interface while preserving
        the explicit body-frame and inertial-frame GVS Jacobian methods.
        """
        return self.jacobian_inertialframe(q, s)

    @eqx.filter_jit
    def jacobian_tips(self, q: Array) -> Array:
        """
        Compute inertial-frame Jacobians at all link tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_tips (Array): inertial-frame Jacobians at each link tip, shape
                (num_segments, 6, num_dofs).
        """
        q_gathered = self._min_size_gathered(q)
        J_local_tips = self._jacobian_gauss(q_gathered)[:, -1] @ self.active_dof_map
        g_tips = self.forward_kinematics_tips(q)

        return vmap(self._body_jacobian_to_inertial)(g_tips, J_local_tips)

    @eqx.filter_jit
    def jacobian_inertialframe_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the inertial-frame Jacobian at multiple arclength locations.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): points along the backbone in [0, L], shape (N,).

        Returns:
            Array: Jacobians with shape (N, 6, num_active_strains).
        """
        J_body = self.jacobian_bodyframe_batched(q, s_ps)  # (N, 6, num_active_strains)
        g_ps = self.forward_kinematics_batched(q, s_ps)  # (N, 4, 4)

        return vmap(self._body_jacobian_to_inertial)(g_ps, J_body)

    @eqx.filter_jit
    def jacobian_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute inertial-frame Jacobians at multiple arclength locations.

        This public adapter satisfies the batched SoftRobot interface while
        preserving the explicit GVS body-frame and inertial-frame methods.

        Args:
            q (Array): generalized coordinates of shape
                ``(num_active_strains,)``.
            s_ps (Array): arclength locations in ``[0, self.segment_lengths]``, shape
                ``(N,)``.

        Returns:
            J_global (Array): inertial-frame Jacobians with shape
                ``(N, 6, num_active_strains)``.
        """
        return self.jacobian_inertialframe_batched(q, s_ps)

    @eqx.filter_jit
    def _jacobian_time_derivative_gauss(
        self, q_gathered: Array, qd_gathered: Array
    ) -> Array:
        """
        Compute the Jacobian time derivative matrices at all significant points of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

            qd_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint velocities.

        Returns:
            Jd_list (Array): shape (num_segments, max_num_integration_points, 6, num_segments*2*max_dof) JAX array
                Jacobian time derivative matrices at all significant points
        """

        def body_segment_i(
            carry: tuple[Array, Array, Array, Array], i_segment: Array
        ) -> tuple[tuple[Array, Array, Array, Array], Array]:
            """Accumulate transforms/Jacobians/Jdot for one segment."""
            g_tip, J_tip, Jd_tip, eta_tip = carry

            # Joint ============================
            B_joint_i = self.B_joint[i_segment]
            xi_ref_joint_i = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]
            qd_joint_i = qd_gathered[i_segment, 0]

            (
                g_joint_i,
                Ad_g_joint_inv,
                Ad_g_joint_inv_dot,
                T_joint_B_i,
                Td_joint_B_i,
                joint_velocity_i,
            ) = self._joint_jacobian_time_derivative_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i, qd_joint_i
            )

            T_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_joint_B_i)
            )
            Td_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(Td_joint_B_i)
            )

            eta_j = Ad_g_joint_inv @ (eta_tip + joint_velocity_i)

            g_j = g_tip @ g_joint_i
            J_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, J_tip + T_g_joint_B_joint_i
            )
            Jd_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, Jd_tip + Td_g_joint_B_joint_i
            ) + jnp.einsum(
                "ij,nmjk->nmik",
                Ad_g_joint_inv_dot,
                J_tip + T_g_joint_B_joint_i,
            )

            # Link ========================
            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]

            q_i = q_gathered[i_segment, 1]
            qd_i = qd_gathered[i_segment, 1]

            def body_eval_points(
                carry: tuple[Array, Array, Array, Array], j_eval: Array
            ) -> tuple[tuple[Array, Array, Array, Array], Array]:
                """Advance one integration cell and update Jdot analytically."""
                g_prev, J_prev, Jd_prev, eta_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                (
                    g_step,
                    T_step,
                    Td_step,
                    Ad_step_inv,
                    B_Magnus_j,
                    B_Magnus_dot_j,
                ) = self._magnus_jacobian_time_derivative_step_terms(
                    length_i,
                    H,
                    q_i,
                    qd_i,
                    B_Z1_i[j_eval],
                    B_Z2_i[j_eval],
                    xi_ref_Z1_i[j_eval],
                    xi_ref_Z2_i[j_eval],
                )

                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )
                Td_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(Td_step @ B_Magnus_j + T_step @ B_Magnus_dot_j)
                )

                eta_step = Ad_step_inv @ (T_step @ B_Magnus_j @ qd_i)
                eta_next = Ad_step_inv @ (eta_prev + T_step @ B_Magnus_j @ qd_i)
                Ad_step_inv_dot = -lie.adjoint_se3(eta_step) @ Ad_step_inv

                g_next = g_prev @ g_step
                J_next = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, J_prev + T_block)
                Jd_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, Jd_prev + Td_block
                ) + jnp.einsum("ij,nmjk->nmik", Ad_step_inv_dot, J_prev + T_block)

                return (g_next, J_next, Jd_next, eta_next), Jd_next

            (g_tip_link, J_tip_link, Jd_tip_link, eta_tip_link), Jd_link = lax.scan(
                f=body_eval_points,
                init=(g_j, J_j, Jd_j, eta_j),
                xs=jnp.arange(self.max_num_integration_points - 1),
            )

            Jd_link = jnp.concatenate((jnp.expand_dims(Jd_j, axis=0), Jd_link), axis=0)
            return (g_tip_link, J_tip_link, Jd_tip_link, eta_tip_link), Jd_link

        J_init = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        Jd_init = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        eta_init = jnp.zeros((6,))

        _, Jd_list = lax.scan(
            f=body_segment_i,
            init=(self.g0, J_init, Jd_init, eta_init),
            xs=jnp.arange(self.num_segments),
        )

        Jd_reordered = jnp.transpose(Jd_list, (0, 1, 4, 2, 3, 5))
        return Jd_reordered.reshape(
            self.num_segments,
            self.max_num_integration_points,
            6,
            self.num_segments * 2 * self.max_dof,
        )

    @eqx.filter_jit
    def _jacobian_and_time_derivative_bodyframe_terms(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array, Array]:
        """
        Compute pose, body-frame Jacobian, and time derivative at ``s``.
        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): generalized velocities of shape (num_dofs,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
            Jd_local (Array): Jacobian time derivative of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
        """
        q_gathered = self._min_size_gathered(q)
        qd_gathered = self._min_size_gathered(qd)

        # Segment containing s and its local arclength.
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(
            carry: tuple[Array, Array, Array, Array], i_segment: Array
        ) -> tuple[
            tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]
        ]:
            """Propagate body-frame Jdot across a segment up to `s`.

            Args:
                carry (Array): Tuple (g_tip, J_tip, Jd_tip, eta_tip).
                i_segment (Array): Segment index.

            Returns:
                Tuple[Tuple[Array, Array, Array, Array], Tuple[Array, Array, Array, Array]]:
                    - (g_pass, J_pass, Jd_pass, eta_pass): scaled tip state to pass forward.
                    - (g_pass, J_pass, Jd_pass, eta_pass): same as scan output.
            """
            g_tip, J_tip, Jd_tip, eta_tip = carry

            # Joint ============================
            B_joint_i = self.B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]  # (max_dof,)
            qd_joint_i = qd_gathered[i_segment, 0]  # (max_dof,)

            (
                g_joint_i,
                Ad_g_joint_inv,
                Ad_g_joint_inv_dot,
                T_joint_B_i,
                Td_joint_B_i,
                joint_velocity_i,
            ) = self._joint_jacobian_time_derivative_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i, qd_joint_i
            )

            # contribution of this joint to its own block
            T_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_joint_B_i)
            )
            Td_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(Td_joint_B_i)
            )

            # forward kinematics of joint
            g_j = g_tip @ g_joint_i
            # bodyframe Jacobian of joint
            J_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, (J_tip + T_g_joint_B_joint_i)
            )  # (num_segments, 2, 6, 6)

            # compute the bodyframe velocity
            eta_j = Ad_g_joint_inv @ (eta_tip + joint_velocity_i)  # (6,)

            # compute the bodyframe Jacobian time derivative
            Jd_j = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, Jd_tip + Td_g_joint_B_joint_i
            ) + jnp.einsum(
                "ij,nmjk->nmik",
                Ad_g_joint_inv_dot,
                J_tip + T_g_joint_B_joint_i,
            )

            # Link ========================
            Xs_i = self.integration_points[i_segment]  # (max_num_integration_points,)
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]  # (max_num_integration_points-1,6)
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]  # (max_num_integration_points-1,6)
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]  # (max_num_integration_points-1,6,max_dof)
            B_Z2_i = self.B_Z2[i_segment]  # (max_num_integration_points-1,6,max_dof)

            q_i = q_gathered[i_segment, 1]
            qd_i = qd_gathered[i_segment, 1]

            def full_cell(
                carry: tuple[Array, Array, Array, Array], j_eval: Array
            ) -> tuple[tuple[Array, Array, Array, Array], None]:
                """Consume a full cell; update g, Jdot and convective term.

                Args:
                    carry (Array): Tuple (g_prev, J_prev, Jd_prev, eta_prev).
                    j_eval (Array): Cell index (int).

                Returns:
                    Tuple[Tuple[Array, Array, Array, Array], None]: Updated carry and dummy output.
                """
                g_prev, J_prev, Jd_prev, eta_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]
                (
                    g_step,
                    T_step,
                    Td_step,
                    Ad_step_inv,
                    B_Magnus_j,
                    B_Magnus_dot_j,
                ) = self._magnus_jacobian_time_derivative_step_terms(
                    length_i,
                    H,
                    q_i,
                    qd_i,
                    B_Z1_i[j_eval],
                    B_Z2_i[j_eval],
                    xi_ref_Z1_i[j_eval],
                    xi_ref_Z2_i[j_eval],
                )

                # contribution of this link cell to its own block
                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )
                Td_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(Td_step @ B_Magnus_j + T_step @ B_Magnus_dot_j)
                )

                g_next = g_prev @ g_step
                J_next = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, (J_prev + T_block))
                eta_step = Ad_step_inv @ (T_step @ B_Magnus_j @ qd_i)
                eta_next = Ad_step_inv @ (eta_prev + T_step @ B_Magnus_j @ qd_i)

                # compute the bodyframe Jacobian time derivative
                Ad_step_inv_dot = -lie.adjoint_se3(eta_step) @ Ad_step_inv
                Jd_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (Jd_prev + Td_block)
                ) + jnp.einsum("ij,nmjk->nmik", Ad_step_inv_dot, J_prev + T_block)

                return (g_next, J_next, Jd_next, eta_next), None

            # Case 1: segment entirely before s → consume every cell
            def do_full_link() -> tuple[Array, Array, Array, Array]:
                """Segment before target `s`: integrate all its cells (Jdot).

                Returns:
                    Tuple[Array, Array, Array, Array]: (g_end, J_end, Jd_end, eta_end).
                """
                (g_end, J_end, Jd_end, eta_end), _ = lax.scan(
                    full_cell,
                    (g_j, J_j, Jd_j, eta_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )
                return g_end, J_end, Jd_end, eta_end

            # Case 2: segment containing s → full cells up to j-1 then partial cell Hp
            def do_partial_link() -> tuple[Array, Array, Array, Array]:
                """Segment containing `s`: consume up to j-1, then partial cell.

                Returns:
                    Tuple[Array, Array, Array, Array]: (g_out, J_out, Jd_out, eta_out) at `s`.
                """
                x = s_local / length_i
                j = jnp.clip(
                    jnp.searchsorted(Xs_i, x) - 1,
                    0,
                    self.max_num_integration_points - 2,
                )
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                # consume up to j-1 (masked)
                def full_cell_masked(
                    carry: tuple[Array, Array, Array, Array], idx: Array
                ) -> tuple[tuple[Array, Array, Array, Array], None]:
                    """Advance only for idx < j; keep state otherwise.

                    Args:
                        carry (Array): Tuple (g_p, J_p, Jd_p, eta_p).
                        idx (Array): Cell index (int).

                    Returns:
                        Tuple[Tuple[Array, Array, Array, Array], None]: Updated or kept carry; dummy output.
                    """
                    (g_p, J_p, Jd_p, eta_p), _ = full_cell(carry, idx)
                    g_keep, J_keep, Jd_keep, eta_keep = carry
                    return (
                        lax.select(idx < j, g_p, g_keep),
                        lax.select(idx < j, J_p, J_keep),
                        lax.select(idx < j, Jd_p, Jd_keep),
                        lax.select(idx < j, eta_p, eta_keep),
                    ), None

                (g_in, J_in, Jd_in, eta_in), _ = lax.scan(
                    full_cell_masked,
                    (g_j, J_j, Jd_j, eta_j),
                    jnp.arange(self.max_num_integration_points - 1),
                )

                # partial cell j of size Hp
                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)
                (
                    g_step,
                    T_step,
                    Td_step,
                    Ad_step_inv,
                    B_Magnus_p,
                    B_Magnus_dot_p,
                ) = self._magnus_jacobian_time_derivative_step_terms(
                    length_i,
                    Hp,
                    q_i,
                    qd_i,
                    Bp[0],
                    Bp[1],
                    xi_ref_Z1_i[j],
                    xi_ref_Z2_i[j],
                )

                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_p)
                )
                Td_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(Td_step @ B_Magnus_p + T_step @ B_Magnus_dot_p)
                )

                g_out = g_in @ g_step
                J_out = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, (J_in + T_block))
                eta_step = Ad_step_inv @ (T_step @ B_Magnus_p @ qd_i)
                eta_out = Ad_step_inv @ (eta_in + T_step @ B_Magnus_p @ qd_i)

                # compute the bodyframe Jacobian time derivative
                Ad_step_inv_dot = -lie.adjoint_se3(eta_step) @ Ad_step_inv
                Jd_out = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (Jd_in + Td_block)
                ) + jnp.einsum("ij,nmjk->nmik", Ad_step_inv_dot, J_in + T_block)

                return g_out, J_out, Jd_out, eta_out

            g_out, J_out, Jd_out, eta_out = lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

            return (g_out, J_out, Jd_out, eta_out), (g_out, J_out, Jd_out, eta_out)

        # Traverse the chain; freeze the state after the segment containing s
        def step(
            carry: tuple[Array, Array, Array, Array], i: Array
        ) -> tuple[tuple[Array, Array, Array, Array], None]:
            """Walk segments; freeze state after the segment containing `s`.

            Args:
                carry (Array): Tuple (g_curr, J_curr, Jd_curr, eta_curr).
                i (Array): Segment index (int).

            Returns:
                Tuple[Tuple[Array, Array, Array, Array], None]: Next carry and dummy output.
            """
            (g_curr, J_curr, Jd_curr, eta_curr), _ = body_segment_i(carry, i)
            g_next = jnp.where(i <= segment_idx, g_curr, carry[0])
            J_next = jnp.where(i <= segment_idx, J_curr, carry[1])
            Jd_next = jnp.where(i <= segment_idx, Jd_curr, carry[2])
            eta_next = jnp.where(i <= segment_idx, eta_curr, carry[3])
            return (g_next, J_next, Jd_next, eta_next), None

        g0 = self.g0

        J0 = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        Jd0 = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        eta0 = jnp.zeros((6,))
        (g_s, J_full, Jd_full, eta_full), _ = lax.scan(
            step, (g0, J0, Jd0, eta0), jnp.arange(self.num_segments)
        )

        J_local = self._jacobian_blocks_to_flat(J_full) @ self.active_dof_map
        Jd_local = self._jacobian_blocks_to_flat(Jd_full) @ self.active_dof_map

        return g_s, J_local, Jd_local

    @eqx.filter_jit
    def jacobian_and_time_derivative_bodyframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian time derivative of the forward kinematics at a point s along the robot in the body frame.
        """
        _, J_local, Jd_local = self._jacobian_and_time_derivative_bodyframe_terms(
            q, qd, s
        )
        return J_local, Jd_local

    @eqx.filter_jit
    def jacobian_and_time_derivative_inertialframe(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_active_configurations,).
            qd (Array): time-derivative of the generalized coordinates of shape (num_active_configurations,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (6, num_active_configurations)
            Jd_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (6, num_active_configurations)
        """
        g_s, J_local, Jd_local = self._jacobian_and_time_derivative_bodyframe_terms(
            q, qd, s
        )
        return self._body_jacobian_time_derivative_to_inertial(
            g_s, J_local, Jd_local, qd
        )

    @eqx.filter_jit
    def jacobian_and_time_derivative_inertialframe_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute inertial-frame Jacobians and time derivatives at multiple arclengths.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            qd (Array): generalized velocities of shape (num_active_strains,).
            s_ps (Array): points along the backbone in [0, L], shape (N,).

        Returns:
            Tuple[Array, Array]: Jacobians and derivatives with shape
            (N, 6, num_active_strains).
        """
        J_body, Jd_body = self.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_ps
        )
        g_ps = self.forward_kinematics_batched(q, s_ps)

        return vmap(
            self._body_jacobian_time_derivative_to_inertial, in_axes=(0, 0, 0, None)
        )(g_ps, J_body, Jd_body, qd)

    @eqx.filter_jit
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time derivative at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): generalized velocities of shape (num_dofs,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J (Array): Jacobian matrix of shape (6, num_dofs).
            Jd (Array): Time derivative of the Jacobian, shape (6, num_dofs).
        """
        return self.jacobian_and_time_derivative_inertialframe(q, qd, s)

    @eqx.filter_jit
    def jacobian_and_time_derivative_batched(
        self, q: Array, qd: Array, s_ps: Array
    ) -> tuple[Array, Array]:
        """
        Compute inertial-frame Jacobians and time derivatives at multiple points.

        This public adapter satisfies the batched SoftRobot interface while
        preserving the explicit GVS body-frame and inertial-frame methods.

        Args:
            q (Array): generalized coordinates of shape
                ``(num_active_strains,)``.
            qd (Array): generalized velocities of shape
                ``(num_active_strains,)``.
            s_ps (Array): arclength locations in ``[0, self.segment_lengths]``, shape
                ``(N,)``.

        Returns:
            Tuple[Array, Array]: inertial-frame Jacobians and time derivatives.
            Each array has shape ``(N, 6, num_active_strains)``.
        """
        return self.jacobian_and_time_derivative_inertialframe_batched(q, qd, s_ps)

    def _active_selector_blocks(self) -> Array:
        """
        Return active-coordinate selectors grouped by segment and local block.

        ``self.active_dof_map`` maps active GVS coordinates to the padded full
        coordinate layout. For local active-coordinate assembly, it is useful to
        view that matrix as per-segment joint/link selector blocks.

        Returns:
            selector_blocks: Active selectors with shape
            ``(self.num_segments, 2, self.max_dof, self.num_dofs)``.
            Axis 1 is ``0`` for joint coordinates and ``1`` for link coordinates.
        """
        return self.active_dof_map_blocks

    def _inner_quadrature_weights(self) -> Array:
        """
        Return length-scaled quadrature weights without endpoint nodes.

        GVS quadrature stores the two link endpoints, but their weights are zero.
        Dynamics assembly only needs the interior nodes and scales each segment's
        normalized weights by its physical length.

        Returns:
            weights: Interior quadrature weights, shape
            ``(self.num_segments, self.max_num_integration_points - 2)``.
        """
        return self.inner_integration_weights

    def _inner_mass_matrices(self) -> Array:
        """
        Return local mass matrices without endpoint nodes.

        Returns:
            Ms_inner: Local spatial mass matrices at interior quadrature nodes,
            shape ``(self.num_segments, self.max_num_integration_points - 2, 6, 6)``.
        """
        return self.inner_mass_matrices

    @eqx.filter_jit
    def _integration_pose_jacobians(self, q: Array) -> tuple[Array, Array, Array]:
        """
        Return active pose/Jacobian data at interior quadrature nodes.

        The zero-weight endpoint nodes are omitted from the returned arrays. The
        segment-tip cell is still advanced internally so the next segment receives
        the correct base pose and Jacobian. This helper is intentionally
        J-only; callers that need ``Jdot`` should use
        ``integration_kinematics``.

        Args:
            q: Active generalized coordinates, shape
                ``(self.num_dofs,)``.

        Returns:
            Tuple ``(weights, g_quads, J_quads)``. ``weights`` contains
            length-scaled interior quadrature weights with shape
            ``(self.num_segments, self.max_num_integration_points - 2)``. ``g_quads`` contains
            poses at those nodes with shape
            ``(self.num_segments, self.max_num_integration_points - 2, 4, 4)``. ``J_quads``
            contains active-coordinate body-frame Jacobians with shape
            ``(self.num_segments, self.max_num_integration_points - 2, 6, self.num_dofs)``.
        """
        q_gathered = self._min_size_gathered(q)
        selector_blocks = self._active_selector_blocks()
        weights = self._inner_quadrature_weights()

        def body_segment_i(
            carry: tuple[Array, Array], i_segment: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            g_tip, J_tip = carry

            B_joint_i = self.B_joint[i_segment]
            xi_ref_joint_i = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]
            selector_joint_i = selector_blocks[i_segment, 0]

            g_joint_i, Ad_g_joint_inv, T_joint_B_i = self._joint_jacobian_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i
            )

            T_joint_active = T_joint_B_i @ selector_joint_i
            g_j = g_tip @ g_joint_i
            J_j = Ad_g_joint_inv @ (J_tip + T_joint_active)

            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]
            q_i = q_gathered[i_segment, 1]
            selector_link_i = selector_blocks[i_segment, 1]

            def body_eval_point(
                carry_link: tuple[Array, Array], j_eval: Array
            ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
                g_prev, J_prev = carry_link
                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                g_step, T_step, Ad_step_inv, B_Magnus_j = (
                    self._magnus_jacobian_step_terms(
                        length_i,
                        H,
                        q_i,
                        B_Z1_i[j_eval],
                        B_Z2_i[j_eval],
                        xi_ref_Z1_i[j_eval],
                        xi_ref_Z2_i[j_eval],
                    )
                )

                T_active = T_step @ B_Magnus_j @ selector_link_i
                g_next = g_prev @ g_step
                J_next = Ad_step_inv @ (J_prev + T_active)
                return (g_next, J_next), (g_next, J_next)

            (g_tip_link, J_tip_link), (g_link_nodes, J_link_nodes) = lax.scan(
                body_eval_point,
                (g_j, J_j),
                jnp.arange(self.max_num_integration_points - 1),
            )

            return (g_tip_link, J_tip_link), (
                g_link_nodes[: self.max_num_integration_points - 2],
                J_link_nodes[: self.max_num_integration_points - 2],
            )

        J_init = jnp.zeros((6, self.num_dofs), dtype=self.active_dof_map.dtype)
        (_, _), (g_quads, J_quads) = lax.scan(
            body_segment_i, (self.g0, J_init), jnp.arange(self.num_segments)
        )

        return weights, g_quads, J_quads

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
            Tuple ``(g_quads, J_quads, Jd_quads)``. ``g_quads`` contains SE(3)
            poses with shape
            ``(self.num_segments, self.max_num_integration_points - 2, 4, 4)``.
            ``J_quads`` contains body-frame Jacobians in active generalized
            coordinates with shape
            ``(self.num_segments, self.max_num_integration_points - 2, 6,
            self.num_dofs)``.
            ``Jd_quads`` contains their time derivatives with the same shape as
            ``J_quads``.
        """
        _, g_quads, J_quads, Jd_quads = self._integration_kinematics(q, qd)
        return g_quads, J_quads, Jd_quads

    @eqx.filter_jit
    def _integration_kinematics(
        self, q: Array, qd: Array, convective_only_jd: bool = False
    ) -> tuple[Array, Array, Array, Array]:
        """
        Return active kinematic data at interior quadrature nodes.

        This is the active-coordinate counterpart of the public/full Gauss
        kinematics path. It keeps the segment and quadrature axes explicit,
        projects local joint/link contributions into active coordinates as they
        are assembled, and drops zero-weight endpoint outputs. When
        ``convective_only_jd`` is true, the returned derivative is only
        guaranteed to be equivalent after multiplication by ``qd``. This is
        sufficient for the forward-dynamics ``C(q, qd) @ qd`` path and avoids
        assembling terms that vanish in that product.

        Args:
            q: Active generalized coordinates, shape
                ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape
                ``(self.num_dofs,)``.
            convective_only_jd: If true, compute a Jacobian time derivative that is
                only guaranteed to be equivalent after multiplication by ``qd``.

        Returns:
            Tuple ``(weights, g_quads, J_quads, Jd_quads)``. ``weights`` has
            shape ``(self.num_segments, self.max_num_integration_points - 2)``; ``g_quads`` has
            shape ``(self.num_segments, self.max_num_integration_points - 2, 4, 4)``;
            ``J_quads`` and ``Jd_quads`` both have shape
            ``(self.num_segments, self.max_num_integration_points - 2, 6, self.num_dofs)``.
        """
        q_gathered = self._min_size_gathered(q)
        qd_gathered = self._min_size_gathered(qd)
        selector_blocks = self._active_selector_blocks()
        weights = self._inner_quadrature_weights()

        def body_segment_i(
            carry: tuple[Array, Array, Array, Array], i_segment: Array
        ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array]]:
            g_tip, J_tip, Jd_tip, eta_tip = carry

            B_joint_i = self.B_joint[i_segment]
            xi_ref_joint_i = self.xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]
            qd_joint_i = qd_gathered[i_segment, 0]
            selector_joint_i = selector_blocks[i_segment, 0]

            (
                g_joint_i,
                Ad_g_joint_inv,
                Ad_g_joint_inv_dot,
                T_joint_B_i,
                Td_joint_B_i,
                joint_velocity,
            ) = self._joint_jacobian_time_derivative_step_terms(
                B_joint_i, xi_ref_joint_i, q_joint_i, qd_joint_i
            )
            eta_j = Ad_g_joint_inv @ (eta_tip + joint_velocity)

            T_joint_active = T_joint_B_i @ selector_joint_i
            Td_joint_active = Td_joint_B_i @ selector_joint_i

            g_j = g_tip @ g_joint_i
            J_j = Ad_g_joint_inv @ (J_tip + T_joint_active)
            if convective_only_jd:
                Jd_j = (
                    Ad_g_joint_inv @ (Jd_tip + Td_joint_active)
                    + Ad_g_joint_inv_dot @ J_tip
                )
            else:
                Jd_j = Ad_g_joint_inv @ (
                    Jd_tip + Td_joint_active
                ) + Ad_g_joint_inv_dot @ (J_tip + T_joint_active)

            Xs_i = self.integration_points[i_segment]
            xi_ref_Z1_i = self.xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.xi_ref_Z2[i_segment]
            length_i = self.segment_lengths[i_segment]
            B_Z1_i = self.B_Z1[i_segment]
            B_Z2_i = self.B_Z2[i_segment]
            q_i = q_gathered[i_segment, 1]
            qd_i = qd_gathered[i_segment, 1]
            selector_link_i = selector_blocks[i_segment, 1]

            def body_eval_point(
                carry_link: tuple[Array, Array, Array, Array], j_eval: Array
            ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array]]:
                g_prev, J_prev, Jd_prev, eta_prev = carry_link
                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                (
                    g_step,
                    T_step,
                    Td_step,
                    Ad_step_inv,
                    B_Magnus_j,
                    B_Magnus_dot_j,
                ) = self._magnus_jacobian_time_derivative_step_terms(
                    length_i,
                    H,
                    q_i,
                    qd_i,
                    B_Z1_i[j_eval],
                    B_Z2_i[j_eval],
                    xi_ref_Z1_i[j_eval],
                    xi_ref_Z2_i[j_eval],
                )

                link_velocity = T_step @ B_Magnus_j @ qd_i
                eta_step = Ad_step_inv @ link_velocity
                eta_next = Ad_step_inv @ (eta_prev + link_velocity)
                Ad_step_inv_dot = -lie.adjoint_se3(eta_step) @ Ad_step_inv

                T_active = T_step @ B_Magnus_j @ selector_link_i
                Td_active = (
                    Td_step @ B_Magnus_j + T_step @ B_Magnus_dot_j
                ) @ selector_link_i

                g_next = g_prev @ g_step
                J_next = Ad_step_inv @ (J_prev + T_active)
                if convective_only_jd:
                    Jd_next = (
                        Ad_step_inv @ (Jd_prev + Td_active) + Ad_step_inv_dot @ J_prev
                    )
                else:
                    Jd_next = Ad_step_inv @ (Jd_prev + Td_active) + Ad_step_inv_dot @ (
                        J_prev + T_active
                    )

                return (g_next, J_next, Jd_next, eta_next), (
                    g_next,
                    J_next,
                    Jd_next,
                )

            (
                (
                    g_tip_link,
                    J_tip_link,
                    Jd_tip_link,
                    eta_tip_link,
                ),
                (g_link_nodes, J_link_nodes, Jd_link_nodes),
            ) = lax.scan(
                body_eval_point,
                (g_j, J_j, Jd_j, eta_j),
                jnp.arange(self.max_num_integration_points - 1),
            )

            return (g_tip_link, J_tip_link, Jd_tip_link, eta_tip_link), (
                g_link_nodes[: self.max_num_integration_points - 2],
                J_link_nodes[: self.max_num_integration_points - 2],
                Jd_link_nodes[: self.max_num_integration_points - 2],
            )

        J_init = jnp.zeros((6, self.num_dofs), dtype=self.active_dof_map.dtype)
        Jd_init = jnp.zeros((6, self.num_dofs), dtype=self.active_dof_map.dtype)
        eta_init = jnp.zeros((6,), dtype=self.active_dof_map.dtype)
        (_, _, _, _), (g_quads, J_quads, Jd_quads) = lax.scan(
            body_segment_i,
            (self.g0, J_init, Jd_init, eta_init),
            jnp.arange(self.num_segments),
        )

        return weights, g_quads, J_quads, Jd_quads

    @eqx.filter_jit
    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        """
        Assemble forward-dynamics terms directly in active coordinates.

        This helper returns exactly the quadrature-derived terms needed by
        ``forward_dynamics``. In particular, it assembles the convective force
        vector ``C(q, qd) @ qd`` directly instead of materializing the full
        Coriolis matrix. The coordinates and returned generalized forces
        correspond only to the active strain components selected by the GVS
        segment bases.

        Args:
            q: Active generalized coordinates, shape
                ``(self.num_dofs,)``.
            qd: Active generalized velocities, shape
                ``(self.num_dofs,)``.

        Returns:
            Tuple ``(B, Cqd, G)``. ``B`` is the active inertia matrix with shape
            ``(self.num_dofs, self.num_dofs)``. ``Cqd`` is the
            active convective force vector with shape ``(self.num_dofs,)``.
            ``G`` is the active generalized gravity vector with shape
            ``(self.num_dofs,)``.
        """
        weights, g_quads, J_quads, Jd_quads = self._integration_kinematics(
            q, qd, convective_only_jd=True
        )
        Ms_inner = self._inner_mass_matrices()

        def segment_terms(i: Array) -> tuple[Array, Array, Array]:
            weights_i = weights[i]
            g_i = g_quads[i]
            J_i = J_quads[i]
            Jd_i = Jd_quads[i]
            Ms_i = Ms_inner[i]

            def point_terms(j: Array) -> tuple[Array, Array, Array]:
                weight_ij = weights_i[j]
                g_ij = g_i[j]
                J_ij = J_i[j]
                Jd_ij = Jd_i[j]
                M_ij = Ms_i[j]

                B_ij = self._inertia_integrand(weight_ij, J_ij, M_ij)
                Cqd_ij = self._coriolis_force_integrand(
                    weight_ij, J_ij, Jd_ij, M_ij, qd
                )
                G_ij = self._gravity_integrand(weight_ij, g_ij, J_ij, M_ij)
                return B_ij, Cqd_ij, G_ij

            return vmap(point_terms)(jnp.arange(self.max_num_integration_points - 2))

        B_blocks, Cqd_blocks, G_blocks = vmap(segment_terms)(
            jnp.arange(self.num_segments)
        )
        return (
            jnp.sum(B_blocks, axis=(0, 1)),
            jnp.sum(Cqd_blocks, axis=(0, 1)),
            jnp.sum(G_blocks, axis=(0, 1)),
        )

    # ===========================================
    # Dynamical matrices computation

    @eqx.filter_jit
    def _inertia_full_matrix(self, q_gathered: Array) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            B_full (Array): Full inertia matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """

        V_J = self._jacobian_gauss(
            q_gathered
        )  # (num_segments, max_num_integration_points, 6, num_segments * 2 * max_dof)

        # Define function for each quadrature point
        def B_i(i: Array) -> Array:
            """Assemble inertia contributions for one segment by quadrature.

            Args:
                i (Array): Segment index (int).

            Returns:
                B_blocks_i (Array): Blocks over quadrature points, shape
                    (max_num_integration_points, num_segments*2*max_dof, num_segments*2*max_dof).
            """
            length_i = self.segment_lengths[i]
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            J_i = V_J[i]  # (max_num_integration_points, 6, num_segments * 2 * max_dof)
            Ms_i = self.mass_matrices[i]  # (max_num_integration_points, 6, 6)

            def B_ij(j: Array) -> Array:
                """Inertia block at a single quadrature point.

                Args:
                    j (Array): Quadrature point index (int).

                Returns:
                    B_ij (Array): Block of shape (num_segments*2*max_dof, num_segments*2*max_dof).
                """
                Ws_ij = Ws_i[j]
                J_ij = J_i[j]  # (6, num_segments * 2 * max_dof)
                Ms_ij = Ms_i[j]  # (6, 6)

                B_ij = Ws_ij * (
                    J_ij.T @ Ms_ij @ J_ij
                )  # (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
                return B_ij

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_i = (
                vmap(B_ij)(jnp.arange(1, self.max_num_integration_points - 1))
                * length_i
            )  # (max_num_integration_points - 2, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            # # For debugging purposes, we can use a list comprehension
            # B_blocks_segment_i = jnp.stack([B_eval_points(i_eval) for i_eval in range(self.max_num_integration_points)])  # (max_num_integration_points, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            return B_blocks_i

        B_blocks_tot = vmap(
            B_i
        )(
            jnp.arange(self.num_segments)
        )  # (num_segments, max_num_integration_points, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        # # For debugging purposes, we can use a list comprehension
        # B_blocks_tot = jnp.stack([B_segment_i(i_segment) for i_segment in range(self.num_segments)])  # (num_segments, max_num_integration_points, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        # Sum over all segments and quadrature points
        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))

        return B_full

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            B (Array): Inertia matrix, shape (num_dofs, num_dofs)
        """
        weights, _, J_quads = self._integration_pose_jacobians(q)
        Ms_inner = self._inner_mass_matrices()

        def segment_terms(i: Array) -> Array:
            def point_term(j: Array) -> Array:
                J_ij = J_quads[i, j]
                return self._inertia_integrand(weights[i, j], J_ij, Ms_inner[i, j])

            return vmap(point_term)(jnp.arange(self.max_num_integration_points - 2))

        return jnp.sum(vmap(segment_terms)(jnp.arange(self.num_segments)), axis=(0, 1))

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).
            qd (Array): generalized velocities of shape (num_dofs,).

        Returns:
            C (Array): Coriolis matrix, shape (num_dofs, num_dofs)
        """
        weights, _, J_quads, Jd_quads = self._integration_kinematics(q, qd)
        Ms_inner = self._inner_mass_matrices()

        def segment_terms(i: Array) -> Array:
            def point_term(j: Array) -> Array:
                J_ij = J_quads[i, j]
                Jd_ij = Jd_quads[i, j]
                M_ij = Ms_inner[i, j]
                return self._coriolis_matrix_integrand(
                    weights[i, j], J_ij, Jd_ij, M_ij, qd
                )

            return vmap(point_term)(jnp.arange(self.max_num_integration_points - 2))

        return jnp.sum(vmap(segment_terms)(jnp.arange(self.num_segments)), axis=(0, 1))

    @eqx.filter_jit
    def _gravitational_full_force(self, q_gathered: Array) -> Array:
        """
        Compute the full gravitational force vector of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            G_full (Array): Full gravitational force vector, shape (num_segments * 2 * max_dof, 1)
        """

        # Get forward kinematics at all quadrature points
        V_g = self._forward_kinematics_gauss(
            q_gathered
        )  # (num_segments, max_num_integration_points, 4, 4)

        # Get Jacobian at all quadrature points
        V_J = self._jacobian_gauss(
            q_gathered
        )  # (num_segments, max_num_integration_points, 6, num_segments * 2 * max_dof)

        def G_i(i: Array) -> Array:
            """Assemble gravitational force contributions for one segment.

            Args:
                i (Array): Segment index (int).

            Returns:
                G_blocks_i (Array): Blocks over quadrature points, shape (max_num_integration_points, num_segments*2*max_dof, 1).
            """
            length_i = self.segment_lengths[i]
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            g_i = V_g[i]  # (max_num_integration_points, 4, 4)
            J_i = V_J[i]  # (max_num_integration_points, 6, num_segments * 2 * max_dof)
            Ms_i = self.mass_matrices[i]  # (max_num_integration_points, 6, 6)

            def G_ij(j: Array) -> Array:
                """Gravitational block at a single quadrature point.

                Args:
                    j (Array): Quadrature point index (int).

                Returns:
                    G_ij (Array): Block of shape (num_segments*2*max_dof, 1).
                """
                Ws_ij = Ws_i[j]  # ()
                g_ij = g_i[j]  # (4, 4)
                Ad_g_inv_ij = lie.Adjoint_g_inv_SE3(g_ij)  # (6, 6)
                J_ij = J_i[j]  # (6, num_segments * 2 * max_dof)
                M_ij = Ms_i[j]  # (6, 6)

                G_ij = (
                    -Ws_ij * J_ij.T @ M_ij @ Ad_g_inv_ij @ self.g
                )  # (num_segments * 2 * max_dof, 1)
                return G_ij

            # we can skip the first and last quadrature points since their weight is zero
            G_blocks_i = (
                vmap(G_ij)(jnp.arange(1, self.max_num_integration_points - 1))
                * length_i
            )  # (max_num_integration_points - 2, num_segments * 2 * max_dof, 1)

            # For debugging purposes, we can use a list comprehension
            # G_blocks_segment_i = jnp.stack([G_eval_points(i_eval) for i_eval in range(self.max_num_integration_points)])

            return G_blocks_i

        G_blocks_tot = vmap(G_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, max_num_integration_points, num_segments * 2 * max_dof, 1)

        # For debugging purposes, we can use a list comprehension
        # G_blocks_tot = jnp.stack([G_segment_i(i_segment) for i_segment in range(self.num_segments)])  # (num_segments, max_num_integration_points, num_segments * 2 * max_dof, 1)

        G_full = jnp.sum(
            G_blocks_tot, axis=(0, 1)
        )  # Sum over links and quadrature points

        return G_full

    @eqx.filter_jit
    def _gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            G (Array): Gravitational force vector, shape (num_dofs, 1)
        """
        weights, g_quads, J_quads = self._integration_pose_jacobians(q)
        Ms_inner = self._inner_mass_matrices()

        def segment_terms(i: Array) -> Array:
            def point_term(j: Array) -> Array:
                return self._gravity_integrand(
                    weights[i, j], g_quads[i, j], J_quads[i, j], Ms_inner[i, j]
                )

            return vmap(point_term)(jnp.arange(self.max_num_integration_points - 2))

        return jnp.sum(vmap(segment_terms)(jnp.arange(self.num_segments)), axis=(0, 1))

    @eqx.filter_jit
    def _stiffness_full_matrix(self) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """

        def K_i(i: Array) -> Array:
            """Assemble stiffness contributions for one segment by quadrature.

            Args:
                i (Array): Segment index (int).

            Returns:
                K_blocks_i (Array): Two blocks (joint/link) of shape (2, max_dof, max_dof).
            """
            # Joint ==============================
            K_joint_i = jnp.zeros(
                (self.max_dof, self.max_dof)
            )  # self.joint_stiffness[i_segment]  # (max_dof, max_dof) TODO

            # Link ===============================
            length_i = self.segment_lengths[i]
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            Es_i = self.stiffness_matrices[i]  # (max_num_integration_points, 6, 6)
            B_Xs_i = self.B_Xs[i]  # (max_num_integration_points, 6, max_dof)

            def K_ij(j: Array) -> Array:
                """
                Stiffness block at a single quadrature point.

                Args:
                    j (Array): Quadrature point index (int).

                Returns:
                    K_ij (Array): Block of shape (max_dof, max_dof).
                """
                Ws_ij = Ws_i[j]
                Es_ij = Es_i[j]  # (6, 6)
                B_Xs_ij = B_Xs_i[j]  # (6, max_dof)

                if self.scale_rotational_basis_by_length:
                    B_Xs_ij = B_Xs_ij.at[:3, :].divide(length_i)

                return Ws_ij * (B_Xs_ij.T @ Es_ij @ B_Xs_ij)

            # we can skip the first and last quadrature points since their weight is zero
            K_link_i = (
                jnp.sum(
                    vmap(K_ij)(jnp.arange(1, self.max_num_integration_points - 1)),
                    axis=0,
                )
                * length_i
            )  # (max_num_integration_points - 2, max_dof, max_dof)

            # Create a (2, max_dof, max_dof) array with K_joint_i and K_segment_i
            K_blocks_i = jnp.stack([K_joint_i, K_link_i], axis=0)
            return K_blocks_i

        K_blocks_tot = vmap(K_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, 2, max_dof, max_dof)

        # Assume that K_blocks is of the form (num_segments, 2, max_dof, max_dof)
        K_blocks_flat = K_blocks_tot.reshape(
            -1, self.max_dof, self.max_dof
        )  # (num_segments * 2, max_dof, max_dof)

        # Convert to list of matrices
        K_blocks_list = [K_blocks_flat[i] for i in range(K_blocks_flat.shape[0])]

        # Building the diagonal matrix in blocks
        K_full = jax.scipy.linalg.block_diag(*K_blocks_list)

        return K_full

    @eqx.filter_jit
    def stiffness_matrix(self) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Returns:
            K (Array): Stiffness matrix, shape (num_dofs, num_dofs)
        """
        return self.K

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            tau_el (Array): Elastic force of shape (num_dofs,).
        """
        return self.K @ q

    @eqx.filter_jit
    def _damping_full_matrix(self) -> Array:
        """
        Compute the full damping matrix of the robot.

        Returns:
            D_full (Array): Full damping matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """

        def D_i(i: Array) -> Array:
            """Assemble damping contributions for one segment by quadrature.

            Args:
                i (Array): Segment index (int).

            Returns:
                D_blocks_i (Array): Two blocks (joint/link) of shape (2, max_dof, max_dof).
            """
            # Joint ==============================
            D_joint_i = jnp.zeros(
                (self.max_dof, self.max_dof)
            )  # Initialize joint stiffness matrix

            # Link ===============================
            length_i = self.segment_lengths[i]
            Ws_i = self.integration_weights[i]  # (max_num_integration_points, 1, )
            Gs_i = self.damping_matrices[i]  # (max_num_integration_points, 6, 6)
            B_Xs_i = self.B_Xs[i]  # (max_num_integration_points, 6, max_dof)

            def D_ij(j: Array) -> Array:
                """Damping block at a single quadrature point.

                Args:
                    j (Array): Quadrature point index (int).

                Returns:
                    D_ij (Array): Block of shape (max_dof, max_dof).
                """
                Ws_j = Ws_i[j]
                Gs_j = Gs_i[j]  # (6, 6)
                B_Xs_j = B_Xs_i[j]  # (6, max_dof)

                if self.scale_rotational_basis_by_length:
                    B_Xs_j = B_Xs_j.at[:3, :].divide(length_i)

                return Ws_j * (B_Xs_j.T @ Gs_j @ B_Xs_j)

            # we can skip the first and last quadrature points since their weight is zero
            D_link_i = (
                jnp.sum(
                    vmap(D_ij)(jnp.arange(1, self.max_num_integration_points - 1)),
                    axis=0,
                )
                * length_i
            )  # (max_num_integration_points - 2, max_dof, max_dof)

            # Create a (2, max_dof, max_dof) array with D_joint_i and D_segment_i
            D_blocks_i = jnp.stack([D_joint_i, D_link_i], axis=0)
            return D_blocks_i

        D_blocks_tot = vmap(D_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, 2, max_dof, max_dof)

        # Assume that D_blocks is of the form (num_segments, 2, max_dof, max_dof)
        D_blocks_flat = D_blocks_tot.reshape(
            -1, self.max_dof, self.max_dof
        )  # (num_segments * 2, max_dof, max_dof)

        # Convert to list of matrices
        D_blocks_list = [D_blocks_flat[i] for i in range(D_blocks_flat.shape[0])]

        # Building the diagonal matrix in blocks
        D_full = jax.scipy.linalg.block_diag(*D_blocks_list)

        return D_full

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            D (Array): Damping matrix, shape (num_dofs, num_dofs)
        """
        return self.D_active

    @eqx.filter_jit
    def _gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy by Gauss quadrature.

        This evaluates
        ``U_g = -sum_i integral rho_i(s) A_i(s) g_linear.dot(p_i(s)) ds``
        over the link centerlines. The mass density per arclength is recovered
        from the translational block of the local mass matrix stored in
        ``mass_matrices``. The implementation uses the same cached Gauss nodes and
        forward-kinematics integration path as the gravitational force, and does
        not use runtime autodiff.

        Args:
            q (Array): generalized coordinates of shape ``(num_dofs,)``.

        Returns:
            U_g (Array): Gravitational potential energy as a scalar array.
        """
        q_gathered = self._min_size_gathered(q)
        g_list = self._forward_kinematics_gauss(q_gathered)
        gravity_linear = self.g[3:]

        def U_G_i(i: Array) -> Array:
            """Compute the gravitational energy contribution of one segment."""
            length_i = self.segment_lengths[i]
            weights = self.integration_weights[
                i, 1 : self.max_num_integration_points - 1
            ]
            mass_density = (
                jnp.trace(
                    self.mass_matrices[
                        i, 1 : self.max_num_integration_points - 1, 3:, 3:
                    ],
                    axis1=-2,
                    axis2=-1,
                )
                / 3.0
            )
            positions = g_list[i, 1 : self.max_num_integration_points - 1, :3, 3]
            height_along_gravity = positions @ gravity_linear
            return -length_i * jnp.sum(weights * mass_density * height_along_gravity)

        return jnp.sum(vmap(U_G_i)(jnp.arange(self.num_segments)))

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_dofs,).

        Returns:
            A (Array): Actuation matrix of shape (num_dofs, num_dofs)
        """
        A = jnp.identity(self.num_dofs)
        return A

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

        rhs = tau_u + tau_ext - Cqd - G - tau_el - self.D_active @ qd
        qdd = jnp.linalg.solve(B, rhs)

        yd = jnp.concatenate([qd, qdd])

        return yd
