__all__ = ["GVS"]
import math
import warnings
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, vmap

import soromox.utils.lie_algebra as lie
from soromox.systems.gvs.attributes import (
    BasisAttributes,
    JointAttributes,
    LinkAttributes,
)
from soromox.systems.gvs.data_classes import SegmentData
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
from soromox.systems.gvs.primitives import Basis, Joint, Link
from soromox.systems.gvs.strain_bases import (
    B_IMQ,
    B_Chebychev,
    B_Fourier,
    B_Gaussian,
    B_LegendrePolynomial,
    B_Monomial,
)
from soromox.systems.soft_robot import SoftRobot
from soromox.utils.basic import compute_strain_basis
from soromox.utils.integration import gauss_quadrature


class GVS(SoftRobot):
    """
    Generalized Variable Strain (GVS) model for 3D soft continuum robots.

    This class implements the geometric and dynamic modeling of a 3D soft robot
    using the Cosserat rod theory with generalized variable strain parametrizations.
    It supports computation of forward kinematics, Jacobians, and dynamic matrices
    for robots with arbitrary combinations of link cross-sections, joint types, and
    strain basis functions.

    Attributes
    ----------
    num_segments : int
        Number of segments (links) in the robot.
    max_dof : int
        Maximum number of degrees of freedom (DOFs) per link or joint.
    max_nGauss : int
        Maximum number of Gauss points per link used for integration.
    max_nip : int
        Maximum number of integration points per link (= max_nGauss + 2).
    dof_tot_system : int
        Total number of DOFs for the robot (sum of joint + link DOFs).
    dof_tot_max : int
        Theoretical maximum number of DOFs (num_segments * 2 * max_dof).
    B_select : Array
        Strain basis selection matrix mapping active DOFs to the full strain space.
    V_L, V_L_cum : Array
        Length of each link, and cumulative link lengths.
    V_nip : Array
        Number of integration/evaluation points for each link.
    V_dof : Array
        Number of DOFs for each link/joint pair (shape: num_segments x 2).
    V_Xs, V_Ws : Array
        Gauss quadrature nodes and weights for each link.
    V_Ms, V_Es, V_Gs : Array
        Mass, stiffness, and damping matrices at integration points.
    V_B_joint, V_B_Xs, V_B_Z1, V_B_Z2 : Array
        Basis matrices for joints and links at quadrature points and intermediate points.
    V_xi_ref_joint, V_xi_ref_Xs, V_xi_ref_Z1, V_xi_ref_Z2 : Array
        Reference strain vectors for joints and links.
    V_K_joint : Array
        Joint stiffness matrices.
    g : Array
        Gravitational acceleration vector in 6D wrench form.
    V_basistype_idx : Array
        Index of the strain basis type used for each segment.
    V_Bdof_params, V_Bodr_params : Array
        Parameters controlling the strain basis DOFs and orders.
    g0 : Array
        Initial pose of the robot base as an SE(3) transformation matrix.
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
    ----------
    - Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020). A geometric variable-strain approach for static modeling of soft manipulators with tendon and fluidic actuation. IEEE Robotics and Automation Letters, 5(3), 4006-4013.
    - Boyer, F., Lebastard, V., Candelier, F., & Renda, F. (2020). Dynamics of continuum and soft robots: A strain parameterization based approach. IEEE Transactions on Robotics, 37(3), 847-863.
    - Mathew, A. T., Feliu-Talegon, D., Alkayas, A. Y., Boyer, F., & Renda, F. (2025). Reduced order modeling of hybrid soft-rigid robots using global, local, and state-dependent strain parameterization. The International Journal of Robotics Research, 44(1), 129-154.
    """

    # Static attributes
    num_segments: int = eqx.field(static=True)  # Number of links in the robot
    max_dof: int = eqx.field(
        static=True
    )  # Maximum number of DOFs for one link of the robot
    max_nGauss: int = eqx.field(
        static=True
    )  # Maximum number of Gauss points for one link of the robot
    max_nip: int = eqx.field(
        static=True
    )  # Maximum number of integration points for one link of the robot

    dof_tot_system: int = eqx.field(
        static=True
    )  # Total number of DOFs for the robot = sum(V_dof)
    dof_tot_max: int = eqx.field(
        static=True
    )  # Total number of DOFs for the robot, considering the maximum DOFs per link: num_segments * 2 * max_dof

    Z1: float = eqx.field(static=True, default=0.5 - math.sqrt(3) / 6)
    Z2: float = eqx.field(static=True, default=0.5 + math.sqrt(3) / 6)

    B_select: Array  # Strain basis functions for the robot (6, max_dof)

    # Dynamic attributes
    V_L: Array  # List of lengths for each link (num_segments, )
    V_L_cum: Array  # Cumulative lengths of the links (num_segments + 1, )
    V_nip: Array  # List of number of evaluation points for each link (num_segments, )
    V_dof: Array  # List of number of DOFs for each link (num_segments, )
    V_Xs: Array  # List of integration points for each link (num_segments, max_nip)
    V_Ws: Array  # List of weights for each link (num_segments, max_nip)
    V_Ms: Array
    V_Es: Array
    V_Gs: Array

    V_B_joint: Array
    V_B_Xs: Array
    V_B_Z1: Array
    V_B_Z2: Array

    V_xi_ref_joint: Array
    V_xi_ref_Xs: Array
    V_xi_ref_Z1: Array
    V_xi_ref_Z2: Array

    V_K_joint: Array  # Joint stiffness matrix (num_segments, max_dof, max_dof)
    g: Array  # Gravity vector (6,)
    p0: Array
    g0: Array

    # Addition to compute forward kinematics at s
    V_basistype_idx: Array  # Index of the basis type for each segment (num_segments,)
    V_Bdof_params: Array  # Parameters for the basis DOFs (num_segments, max_dof)
    V_Bodr_params: Array  # Parameters for the basis orientation (num_segments, max_dof)

    def __init__(
        self,
        links_list: list[LinkAttributes,],
        joints_list: list[JointAttributes,],
        basis_list: list[BasisAttributes,],
        n_gauss_list: list[int],
        gravity_vector: list[float],
        max_dof: int | None = None,
        max_nGauss: int | None = None,
        p0: Array | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the GVS class.

        Args
        ----
        links_list : List[LinkAttributes]
            List of link property objects (one per segment) containing geometric and
            material attributes:
            - section: Type of cross-section ('Circular', 'Rectangular', 'Elliptical').
            - E: Young's modulus [N/m²].
            - nu: Poisson's ratio [-1, 0.5].
            - rho: Density [kg/m³].
            - eta: Material damping [N·s/m].
            - L: Length of the link [m].
            - One of the following geometric parameters must be provided based on the section type:
                - r_i, r_f: Initial and final radius for circular sections.
                - h_i, h_f: Initial and final height for rectangular sections.
                - w_i, w_f: Initial and final width for rectangular sections.
                - a_i, a_f: Initial and final semi-major axis for elliptical sections.
                - b_i, b_f: Initial and final semi-minor axis for elliptical sections.
        joints_list : List[JointAttributes]
            List of joint property objects (one per segment) describing the connection
            between segments :
            - jointtype: Type of joint ('Revolute', 'Prismatic', 'Helical', 'Cylindrical',
                'Planar', 'Spherical', 'Free', 'Fixed').
            - axis: Axis of motion ('x', 'y', 'z') for revolute/prismatic joints.
            - plane: Plane of motion ('xy', 'yz', 'xz') for cylindrical/planar joints.
            - pitch: Pitch for helical joints.
            - K_joint: Joint stiffness matrix (optional, defaults to zeros). If provided,
              must have shape (dof_joint, dof_joint); otherwise it is ignored and a
              zero matrix is used. Units match the active DOFs: rotational terms
              [N·m/rad], translational terms [N/m]; off-diagonal couplings accordingly.
        basis_list : List[BasisAttributes]
            List of strain basis attributes (one per segment) defining the parametrization
            of the variable strain field along the link:
            - basistype: Type of basis function ('Monomial', 'Legendre', 'Chebychev', 'Fourier', 'Gaussian', 'IMQ').
            - Bdof: Array indicating which types of deformation are selected (1) or not (0).
                e.g., [1, 0, 1, 0, 0, 0] means kappa_x and sigma_y are selected.
            - Bodr: Array indicating the orders of the basis functions for each type of deformation.
            - xi_ref: Reference strain values for each type of deformation.
        n_gauss_list : List[int]
            Number of Gauss-Legendre quadrature points for integration in each segment.
        gravity_vector : List[float]
            3-element gravity vector [gx, gy, gz] in m/s².
        max_dof : int, optional
            Maximum number of DOFs for a link or joint. If None, computed as minimal from the inputs.
        max_nGauss : int, optional
            Maximum number of Gauss points across all segments. If None, computed as minimal from `n_gauss_list`.
        p0: (optional) List/Array of shape (6,)
                Initial orientation angle and position in the inertial frame [rad, m]
                [ψ, θ, φ, x0, y0, z0]
        **kwargs: Additional keyword arguments for SoftRobot.__init__.

        Raises
        ------
        ValueError
            If the input lists differ in length or if `max_dof` / `max_nGauss`
            are smaller than required.
        TypeError
            If input types do not match the expected formats.

        Notes
        -----
        - This constructor precomputes and stores link/joint basis matrices,
        Gauss quadrature data, geometric properties, and reference strains
        for each segment.
        - Internal arrays are padded to `max_dof` and `max_nip` to allow vectorized
        batched computations across all segments.
        """
        super().__init__(**kwargs)

        warnings.warn(
            "GVS is not fully validated yet and might not match the behavior of PlanarPCS and PCS."
        )

        if (
            len(links_list) != len(joints_list)
            or len(links_list) != len(basis_list)
            or len(links_list) != len(n_gauss_list)
        ):
            raise ValueError(
                "The lists of links, joints, basis attributes and n_gauss must have the same length."
            )
        self.num_segments = len(links_list)

        if max_nGauss is None:
            max_nGauss = max(n_gauss_list)
        elif max_nGauss < max(n_gauss_list):
            raise ValueError(
                f"max_nGauss must be greater than or equal to the maximum n_gauss in n_gauss_list ({max(n_gauss_list)}), but got {max_nGauss}."
            )

        if max_nGauss < 5:
            raise ValueError(
                f"max_nGauss must be greater than 5, but got {max_nGauss}."
            )
        self.max_nGauss = max_nGauss
        self.max_nip = max_nGauss + 2  # +2 for the boundary points

        dofs_joint = [Joint.DICT_JOINT_TYPE_DOF[j.jointtype] for j in joints_list]
        dofs_link = [
            int(
                Basis.DOF_BRANCHES[Basis.BASISTYPE_MAP[b.basistype]](
                    (jnp.asarray(b.Bdof), jnp.asarray(b.Bodr))
                )
            )
            for b in basis_list
        ]
        real_max_dof = max(dofs_joint + dofs_link)

        if max_dof is None:
            max_dof = real_max_dof
        elif max_dof < real_max_dof:
            raise ValueError(
                f"max_dof must be greater than or equal to the maximum DOF in links and joints ({real_max_dof}), but got {max_dof}."
            )
        self.max_dof = max_dof

        V_L = jnp.empty((self.num_segments,), dtype=float)
        V_nip = jnp.empty((self.num_segments,), dtype=int)
        V_dof = jnp.empty((self.num_segments, 2), dtype=int)
        V_Xs = jnp.empty(
            (
                self.num_segments,
                self.max_nip,
            ),
            dtype=float,
        )
        V_Ws = jnp.empty(
            (
                self.num_segments,
                self.max_nip,
            ),
            dtype=float,
        )

        V_Ms = jnp.empty((self.num_segments, self.max_nip, 6, 6), dtype=float)
        V_Es = jnp.empty((self.num_segments, self.max_nip, 6, 6), dtype=float)
        V_Gs = jnp.empty((self.num_segments, self.max_nip, 6, 6), dtype=float)

        V_B_joint = jnp.empty((self.num_segments, 6, self.max_dof), dtype=float)
        V_B_Xs = jnp.empty(
            (self.num_segments, self.max_nip, 6, self.max_dof), dtype=float
        )
        V_B_Z1 = jnp.empty(
            (self.num_segments, self.max_nip - 1, 6, self.max_dof), dtype=float
        )
        V_B_Z2 = jnp.empty(
            (self.num_segments, self.max_nip - 1, 6, self.max_dof), dtype=float
        )

        V_xi_ref_joint = jnp.empty(
            (
                self.num_segments,
                6,
            ),
            dtype=float,
        )
        V_xi_ref_Xs = jnp.empty((self.num_segments, self.max_nip, 6), dtype=float)
        V_xi_ref_Z1 = jnp.empty((self.num_segments, self.max_nip - 1, 6), dtype=float)
        V_xi_ref_Z2 = jnp.empty((self.num_segments, self.max_nip - 1, 6), dtype=float)

        V_K_joint = jnp.empty(
            (self.num_segments, self.max_dof, self.max_dof), dtype=float
        )

        V_strain_selector = jnp.zeros(
            (self.num_segments, 2 * self.max_dof), dtype=jnp.bool_
        )

        # Addition to compute forward kinematics at s
        V_basistype_idx = jnp.empty((self.num_segments,), dtype=int)
        V_Bdof_params = jnp.empty((self.num_segments, 6))
        V_Bodr_params = jnp.empty((self.num_segments, 6))

        for i_segment in range(self.num_segments):
            # Use the provided attributes from the links_list
            joint_attrs = cast(JointAttributes, joints_list[i_segment])
            link_attrs = cast(LinkAttributes, links_list[i_segment])
            basis_attrs = cast(BasisAttributes, basis_list[i_segment])
            n_gauss_i = n_gauss_list[i_segment]

            segment_attributes = self._build_segment_i(
                max_dof=self.max_dof,
                max_nip=self.max_nip,
                link_attrs=link_attrs,
                joint_attrs=joint_attrs,
                basis_attrs=basis_attrs,
                n_gauss=n_gauss_i,
            )

            # Update the total vectors ==========================================================================
            V_L = V_L.at[i_segment].set(segment_attributes.L)
            V_nip = V_nip.at[i_segment].set(segment_attributes.nip)
            V_dof = V_dof.at[i_segment].set(segment_attributes.dofs_joint_link)

            V_Xs = V_Xs.at[i_segment].set(segment_attributes.Xs)
            V_Ws = V_Ws.at[i_segment].set(segment_attributes.Ws)

            V_Ms = V_Ms.at[i_segment].set(segment_attributes.Ms)
            V_Es = V_Es.at[i_segment].set(segment_attributes.Es)
            V_Gs = V_Gs.at[i_segment].set(segment_attributes.Gs)

            V_B_joint = V_B_joint.at[i_segment].set(segment_attributes.B_joint)
            V_B_Xs = V_B_Xs.at[i_segment].set(segment_attributes.B_Xs)
            V_B_Z1 = V_B_Z1.at[i_segment].set(segment_attributes.B_Z1)
            V_B_Z2 = V_B_Z2.at[i_segment].set(segment_attributes.B_Z2)

            V_xi_ref_joint = V_xi_ref_joint.at[i_segment].set(
                segment_attributes.xi_ref_joint
            )
            V_xi_ref_Xs = V_xi_ref_Xs.at[i_segment].set(segment_attributes.xi_ref_Xs)
            V_xi_ref_Z1 = V_xi_ref_Z1.at[i_segment].set(segment_attributes.xi_ref_Z1)
            V_xi_ref_Z2 = V_xi_ref_Z2.at[i_segment].set(segment_attributes.xi_ref_Z2)

            V_K_joint = V_K_joint.at[i_segment].set(segment_attributes.K_joint)

            V_strain_selector = V_strain_selector.at[i_segment].set(
                segment_attributes.strain_selector
            )

            # Addition to compute forward kinematics at s
            V_basistype_idx = V_basistype_idx.at[i_segment].set(
                Basis.BASISTYPE_MAP[basis_attrs.basistype]
            )
            V_Bdof_params = V_Bdof_params.at[i_segment].set(basis_attrs.Bdof)
            V_Bodr_params = V_Bodr_params.at[i_segment].set(basis_attrs.Bodr)

        self.V_L = V_L
        self.V_nip = V_nip
        self.V_dof = V_dof
        self.V_Xs = V_Xs
        self.V_Ws = V_Ws
        self.V_Ms = V_Ms
        self.V_Es = V_Es
        self.V_Gs = V_Gs
        self.V_B_joint = V_B_joint
        self.V_B_Xs = V_B_Xs
        self.V_B_Z1 = V_B_Z1
        self.V_B_Z2 = V_B_Z2
        self.V_xi_ref_joint = V_xi_ref_joint
        self.V_xi_ref_Xs = V_xi_ref_Xs
        self.V_xi_ref_Z1 = V_xi_ref_Z1
        self.V_xi_ref_Z2 = V_xi_ref_Z2
        self.V_K_joint = V_K_joint

        # Addition to compute forward kinematics at s
        self.V_basistype_idx = V_basistype_idx
        self.V_Bdof_params = V_Bdof_params
        self.V_Bodr_params = V_Bodr_params

        # Strain selector ========================================================
        strain_selector_full = jnp.concatenate(V_strain_selector, axis=0)
        self.B_select = compute_strain_basis(strain_selector_full)

        # Degrees of freedom =========================================================
        self.dof_tot_system = int(
            jnp.sum(V_dof, axis=(0, 1), dtype=int)
        )  # Total DOFs for the robot
        self.num_dofs = self.dof_tot_system
        self.dof_tot_max = int(
            jnp.array(self.num_segments * 2 * self.max_dof, dtype=int)
        )

        # Gravity vector ======================================
        self.g = jnp.concatenate([jnp.zeros(3), jnp.array(gravity_vector)])

        # Cumulative lengths =========================================================
        self.V_L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), V_L]))

        # Number of actuators
        self.num_actuators = self.dof_tot_system

        # Base pose g0: derive from p0 if provided, otherwise keep previous default base rotation
        if p0 is None:
            p0_arr = jnp.array(
                [jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64
            )
        else:
            p0_arr = jnp.asarray(p0, dtype=jnp.float64)

        if p0_arr.size != 6:
            raise ValueError("p0 must have shape (6,) when provided")
        self.p0 = p0_arr
        self.g0 = lie.exp_SE3(p0_arr)

    def _build_segment_i(
        self,
        link_attrs: LinkAttributes,
        joint_attrs: JointAttributes,
        basis_attrs: BasisAttributes,
        n_gauss: int,
        max_dof: int,
        max_nip: int,
    ) -> SegmentData:
        """
        Construct the discretized model data for a single segment.

        Args
        ----
        link_attrs : LinkAttributes
            Geometric and material properties for the link.
        joint_attrs : JointAttributes
            Joint type and kinematic properties between this link and the previous.
        basis_attrs : BasisAttributes
            Strain basis definition for variable strain representation in the link.
        n_gauss : int
            Number of Gauss-Legendre quadrature points for integration.
        max_dof : int
            Maximum number of DOFs to pad/truncate joint/link basis matrices.
        max_nip : int
            Maximum number of integration points for padding.

        Returns
        -------
        SegmentData
            Structured object containing:
            - Length, number of integration points, DOFs
            - Strain selection mask
            - Padded Gauss points/weights
            - Mass, stiffness, and damping matrices
            - Joint and link basis matrices at key points (Xs, Z1, Z2)
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
        jointtype = joint_attrs.jointtype
        jointtype_idx = Joint.JOINTTYPE_MAP[jointtype]

        dof_joint = Joint.DICT_JOINT_TYPE_DOF[jointtype]
        K_joint = (
            jnp.asarray(joint_attrs.K_joint)
            if jnp.asarray(joint_attrs.K_joint).shape == (dof_joint, dof_joint)
            else jnp.zeros((dof_joint, dof_joint))
        )

        joint_operand = JointOperand(
            axis_idx=Joint.AXIS_MAP[joint_attrs.axis],
            plane_idx=Joint.PLANE_MAP[joint_attrs.plane],
            pitch=joint_attrs.pitch,
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
        section = link_attrs.section
        section_idx = Link.SECTION_MAP[section]

        E = jnp.asarray(link_attrs.E)
        nu = jnp.asarray(link_attrs.nu)
        rho = jnp.asarray(link_attrs.rho)
        eta = jnp.asarray(link_attrs.eta)
        L = jnp.asarray(link_attrs.L)

        r_i = jnp.asarray(link_attrs.r_i)
        r_f = jnp.asarray(link_attrs.r_f)
        h_i = jnp.asarray(link_attrs.h_i)
        h_f = jnp.asarray(link_attrs.h_f)
        w_i = jnp.asarray(link_attrs.w_i)
        w_f = jnp.asarray(link_attrs.w_f)
        a_i = jnp.asarray(link_attrs.a_i)
        a_f = jnp.asarray(link_attrs.a_f)
        b_i = jnp.asarray(link_attrs.b_i)
        b_f = jnp.asarray(link_attrs.b_f)

        G = E / (2 * (1 + nu))  # Shear modulus

        r_params = (r_i, r_f)
        h_params = (h_i, h_f)
        w_params = (w_i, w_f)
        a_params = (a_i, a_f)
        b_params = (b_i, b_f)

        # === Basis attributes
        basetype = basis_attrs.basistype
        basistype_idx = Basis.BASISTYPE_MAP[basetype]
        Bdof = jnp.asarray(basis_attrs.Bdof).flatten()
        Bodr = jnp.asarray(basis_attrs.Bodr).flatten()
        xi_ref = jnp.asarray(basis_attrs.xi_ref).reshape(6, 1)

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
        Xs, Ws, nip = gauss_quadrature(N_GQ=n_gauss)

        deltas = Xs[1:] - Xs[:-1]  # Array of shape (nip - 1, )

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
            index=basistype_idx, branches=B_branches, operand=(Xs, Bdof, Bodr)
        )
        B_Z1 = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(Xs[:-1] + self.Z1 * deltas, Bdof, Bodr),
        )
        B_Z2 = lax.switch(
            index=basistype_idx,
            branches=B_branches,
            operand=(Xs[:-1] + self.Z2 * deltas, Bdof, Bodr),
        )

        # Compute the initial strain vectors at the integration points
        xi_reffn = lambda x: xi_ref  # TODO: allow to have an expression for xi_ref
        xi_ref_Xs = vmap(xi_reffn)(Xs).squeeze()
        xi_ref_Z1 = vmap(xi_reffn)(Xs[:-1] + self.Z1 * deltas).squeeze()
        xi_ref_Z2 = vmap(xi_reffn)(Xs[:-1] + self.Z2 * deltas).squeeze()

        # Compute the mass, stiffness, and damping matrices at the integration points
        geometric_operand = GeometricOperand(
            Xs=Xs,
            r_params=r_params,
            h_params=h_params,
            w_params=w_params,
            a_params=a_params,
            b_params=b_params,
        )
        Ix_p, Iy_p, Iz_p, A_p = lax.switch(
            index=section_idx,
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
        Xs_full = jnp.pad(Xs, (0, max_nip - nip), mode="constant")
        Ws_full = jnp.pad(Ws, (0, max_nip - nip), mode="constant")

        Ms_full = jnp.pad(Ms, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")
        Es_full = jnp.pad(Es, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")
        Gs_full = jnp.pad(Gs, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")

        B_Xs_full = jnp.pad(B_Xs, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")
        B_Z1_full = jnp.pad(B_Z1, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")
        B_Z2_full = jnp.pad(B_Z2, ((0, max_nip - nip), (0, 0), (0, 0)), mode="constant")

        xi_ref_Xs_full = jnp.pad(
            xi_ref_Xs, ((0, max_nip - nip), (0, 0)), mode="constant"
        )
        xi_ref_Z1_full = jnp.pad(
            xi_ref_Z1, ((0, max_nip - nip), (0, 0)), mode="constant"
        )
        xi_ref_Z2_full = jnp.pad(
            xi_ref_Z2, ((0, max_nip - nip), (0, 0)), mode="constant"
        )

        dofs_joint_link = jnp.stack([dof_joint, dof_link])

        strain_selector_segment = jnp.concatenate(
            [strain_selector_joint, strain_selector_link], axis=0
        )

        return SegmentData(
            L=L,
            nip=nip,
            dofs_joint_link=dofs_joint_link,
            strain_selector=strain_selector_segment,
            Xs=Xs_full,
            Ws=Ws_full,
            Ms=Ms_full,
            Es=Es_full,
            Gs=Gs_full,
            B_joint=B_joint_full,
            B_Xs=B_Xs_full,
            B_Z1=B_Z1_full,
            B_Z2=B_Z2_full,
            xi_ref_joint=xi_ref_joint,
            xi_ref_Xs=xi_ref_Xs_full,
            xi_ref_Z1=xi_ref_Z1_full,
            xi_ref_Z2=xi_ref_Z2_full,
            K_joint=K_joint_full,
        )

    # Gathering functions =========================================================
    @eqx.filter_jit
    def _min_size_gathered(self, vec_min_size_flat: Array) -> Array:
        """
        Function to split the joint coordinates into segments and pad them to the maximum DOF.

        Args:
            vec_min_size_flat (Array): shape (dof_tot, 1) or (dof_tot,) JAX array
                Joint coordinates or joint velocities.
        Returns:
            vec_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Padded joint coordinates.
        """
        vec = jnp.asarray(vec_min_size_flat).reshape(-1)  # Ensure q is a 1D array
        m, n_groups = self.V_dof.shape  # n_groups = 2

        # Start index calculation
        V_dof_flat = self.V_dof.reshape(-1)
        start_indices_flat = jnp.cumsum(jnp.pad(V_dof_flat, (1, 0)))[:-1]
        start_indices = start_indices_flat.reshape(m, n_groups)

        # Function for obtaining indexes
        def get_indices(start: Array) -> Array:
            """Return absolute indices for a block.

            Args:
                start (Array): Starting index in the flattened vector (scalar int/Array).

            Returns:
                indices (Array): Indices of shape (max_dof,) into the flattened vector.
            """
            indices = start + jnp.arange(self.max_dof)
            return indices

        # Vectorization: apply to (m, 2)
        get_indices_vmap = vmap(vmap(get_indices, in_axes=(0,)), in_axes=(0,))
        all_indices = get_indices_vmap(start_indices)

        # Creating masks
        def get_mask(length) -> Array:
            """Boolean mask of valid entries for a given `length`.

            Args:
                length: Actual number of valid entries (scalar int/Array).

            Returns:
                mask (Array): Boolean mask of shape (max_dof,), True where index < length.
            """
            mask = jnp.arange(self.max_dof) < length
            return mask

        get_mask_vmap = vmap(vmap(get_mask, in_axes=(0,)), in_axes=(0,))
        mask = get_mask_vmap(self.V_dof)

        # Secure padding
        vec_padded = jnp.concatenate([vec, jnp.zeros(self.max_dof)])

        # We retrieve the values
        vec_gathered = jnp.take(vec_padded, all_indices, mode="clip") * mask

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
        basistype_idx = self.V_basistype_idx[i_segment]
        Bdof = self.V_Bdof_params[i_segment]
        Bodr = self.V_Bodr_params[i_segment]
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

    # =========================================================================
    @eqx.filter_jit
    def _forward_kinematics_gauss(self, q_gathered: Array) -> Array:
        """
        Compute the forward kinematics transformation matrices at all significant points

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            g_list (Array): shape (num_segments, max_nip, 4, 4) JAX array
                Forward kinematics transformation matrices at all significant points
        """

        def body_segment_i(carry: Array, i_segment: Array) -> tuple[Array, Array]:
            """Propagate transforms for a segment and collect frames.

            Args:
                carry (Array): Current tip transform `g_tip`, shape (4, 4).
                i_segment (Array): Segment index.

            Returns:
                g_tip_link (Array): Updated tip transform after this segment, shape (4, 4).
                g_link (Array): Per-point transforms for this segment, shape (max_nip, 4, 4).
            """
            g_tip = carry

            # Joint =======================
            B_joint_i = self.V_B_joint[i_segment]  # shape (6, max_dof)
            xi_ref_joint_i = self.V_xi_ref_joint[i_segment]  # shape (6,)
            q_joint_i = q_gathered[i_segment, 0]  # shape (max_dof,)

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i  # shape (6,)

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # shape (4, 4)

            g_j = g_tip @ g_joint_i  # Start with the last transformation matrix

            # Link ========================
            Xs_i = self.V_Xs[i_segment]  # shape (max_nip,)
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]  # shape (max_nip - 1, 6)
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]  # shape (max_nip - 1, 6)
            length_i = self.V_L[i_segment]  # shape (1,)
            B_Z1_i = self.V_B_Z1[i_segment]  # shape (max_nip - 1, 6, max_dof)
            B_Z2_i = self.V_B_Z2[i_segment]  # shape (max_nip - 1, 6, max_dof)

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

                xi_ref_Z1_j = xi_ref_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_ref_Z2_j = xi_ref_Z2_i[j_eval].at[:3].multiply(length_i)

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j = B_Z1_j @ q_i + xi_ref_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_ref_Z2_j

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_j = (H / 2) * (xi_Z1_j + xi_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2_j
                )

                Magnus_j = Magnus_j.at[3:6].multiply(length_i)
                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)

                g_j = g_j_prev @ g_step

                return g_j, g_j

            indices_eval_points = jnp.arange(self.max_nip - 1)

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
        segment_idx = jnp.clip(jnp.sum(s > self.V_L_cum) - 1, 0, self.num_segments - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.V_L_cum[segment_idx]

        return segment_idx, s_local

    @eqx.filter_jit
    def forward_kinematics(self, q: Array, s: Array) -> Array:
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
            B_joint = self.V_B_joint[i_segment]
            xi_ref_joint = self.V_xi_ref_joint[i_segment]
            q_joint_i = q_gathered[i_segment, 0]

            g_joint_i = lie.exp_gn_SE3(
                B_joint @ q_joint_i + xi_ref_joint, self.global_eps
            )

            g_j = g_tip @ g_joint_i

            # Link ========================
            Xs_i = self.V_Xs[i_segment]
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]
            length_i = self.V_L[i_segment]
            B_Z1_i = self.V_B_Z1[i_segment]
            B_Z2_i = self.V_B_Z2[i_segment]

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

                xi_Z1_j = (B_Z1_i[j] @ q_i) + xi_ref_Z1_i[j].at[:3].multiply(length_i)
                xi_Z2_j = (B_Z2_i[j] @ q_i) + xi_ref_Z2_i[j].at[:3].multiply(length_i)

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_j = (H / 2) * (xi_Z1_j + xi_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2_j
                )

                Magnus_j = Magnus_j.at[3:6].multiply(length_i)
                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)

                return g_prev @ g_step, None

            # if segment before s: make entire link and return
            def do_full_link() -> Array:
                """Integrate the entire link (segment before the target `s`).

                Returns:
                    g_tip_link (Array): Tip transform after this link, shape (4, 4).
                """
                g_tip_link, _ = lax.scan(full_cell, g_j, jnp.arange(self.max_nip - 1))
                return g_tip_link

            # if segment is the one where s is located, we need to do a partial link
            def do_partial_link() -> Array:
                """Integrate up to location `s` within this segment.

                Consumes complete cells up to j-1, then a partial cell j.

                Returns:
                    g_out (Array): Tip transform at `s` within this segment, shape (4, 4).
                """
                x = s_local / length_i
                # idx of cell j such that Xs[j] <= x <= Xs[j+1]
                j = jnp.clip(jnp.searchsorted(Xs_i, x) - 1, 0, self.max_nip - 2)

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

                g_in, _ = lax.scan(full_cell_masked, g_j, jnp.arange(self.max_nip - 1))

                # enter the (partial) cell j
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                Xp = jnp.array(
                    [Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp]
                )  # length-2, static

                Bp = self._eval_B_segment(i_segment, Xp)  # (2, 6, max_dof)

                xi_Z1_j = (Bp[0] @ q_i) + self.V_xi_ref_Z1[i_segment][j].at[
                    :3
                ].multiply(length_i)
                xi_Z2_j = (Bp[1] @ q_i) + self.V_xi_ref_Z2[i_segment][j].at[
                    :3
                ].multiply(length_i)

                # Magnus expansion
                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                Magnus_p = (Hp / 2) * (xi_Z1_j + xi_Z2_j) + (
                    jnp.sqrt(3) * Hp**2 / 12
                ) * (ad_xi_Z1_j @ xi_Z2_j)
                Magnus_p = Magnus_p.at[3:6].multiply(length_i)

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
    def _jacobian_gauss(self, q_gathered: Array) -> Array:
        """
        Compute the Jacobian matrices at all significant points of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

        Returns:
            J_list (Array): shape (num_segments, max_nip, num_segments, 2, 6, max_dof) JAX array
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
                    - (g_tip_link_scaled, J_tip_link_scaled): scaled tip state with
                      shapes (4, 4) and (num_segments, 2, 6, max_dof).
                    - J_link: per-point Jacobians for this segment, shape
                      (max_nip, num_segments, 2, 6, max_dof).
            """
            g_tip, J_tip = carry

            # Joint ============================
            B_joint_i = self.V_B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.V_xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # shape (4, 4)
            T_g_joint = lie.Tangent_gi_se3(
                xi_joint_i, 1, eps=self.tangent_eps
            )  # shape (6, 6)

            T_g_joint_i_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_g_joint @ B_joint_i)
            )

            Ad_g_joint_inv = lie.Adjoint_g_inv_SE3(g_joint_i)  # shape (6, 6)

            g_j_scaled = g_tip @ g_joint_i
            J_j_scaled = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, (J_tip + T_g_joint_i_B_joint_i)
            )

            # Link ========================
            Xs_i = self.V_Xs[i_segment]  # shape (max_nip,)
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]  # shape (max_nip - 1, 6)
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]  # shape (max_nip - 1, 6)
            length_i = self.V_L[i_segment]  # shape (1,)
            B_Z1_i = self.V_B_Z1[i_segment]  # shape (max_nip - 1, 6, max_dof)
            B_Z2_i = self.V_B_Z2[i_segment]  # shape (max_nip - 1, 6, max_dof)

            g_j = g_j_scaled.at[0:3, 3].divide(length_i)  # shape (4, 4)
            J_j = J_j_scaled.at[:, :, 3:6, :].divide(
                length_i
            )  # shape (num_segments, 6, max_dof)

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
                        - J_next_scaled: scaled Jacobian, shape
                          (num_segments, 2, 6, max_dof).
                """
                g_prev, J_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                xi_ref_Z1_j = xi_ref_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_ref_Z2_j = xi_ref_Z2_i[j_eval].at[:3].multiply(length_i)

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j = B_Z1_j @ q_i + xi_ref_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_ref_Z2_j

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2_j)

                Magnus_j = (H / 2) * (xi_Z1_j + xi_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2_j
                )

                B_Magnus_j = (H / 2) * (B_Z1_j + B_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ B_Z2_j - ad_xi_Z2_j @ B_Z1_j
                )

                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)  # shape (4, 4)
                T_step = lie.Tangent_gi_se3(
                    Magnus_j, 1, eps=self.tangent_eps
                )  # shape (6, 6)

                T_step_B_step = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )

                Ad_g_step_inv = lie.Adjoint_g_inv_SE3(g_step)

                g_next = g_prev @ g_step
                J_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_g_step_inv, (J_prev + T_step_B_step)
                )  # shape (num_segments, 6, max_dof)

                J_next_scaled = J_next.at[:, :, 3:6, :].multiply(length_i)

                return (g_next, J_next), J_next_scaled

            indices_eval_points = jnp.arange(self.max_nip - 1)

            (g_tip_link, J_tip_link), J_link = lax.scan(
                f=body_eval_points, init=(g_j, J_j), xs=indices_eval_points
            )

            # # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = (g_j, J_j)
            # J_link = []
            # for x in indices_eval_points:
            #     (g_j, J_j), J_j_scaled_here = body_eval_points(carry, x)
            #     carry = (g_j, J_j)
            #     J_link.append(J_j_scaled_here)
            # (g_tip_link, J_tip_link) = carry
            # J_link = jnp.array(J_link)  # shape (max_nip - 1, num_segments, 6, max_dof)

            J_link = jnp.concatenate(
                (jnp.expand_dims(J_j_scaled, axis=0), J_link), axis=0
            )

            g_tip_link_scaled = g_tip_link.at[0:3, 3].multiply(length_i)
            J_tip_link_scaled = J_tip_link.at[:, :, 3:6, :].multiply(length_i)

            return (g_tip_link_scaled, J_tip_link_scaled), J_link

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

        # (num_segments, max_nip, num_segments, 2, 6, max_dof) => (num_segments, max_nip, 6, num_segments*2*max_dof)
        J_reordered = jnp.transpose(J_list, (0, 1, 4, 2, 3, 5))
        J = J_reordered.reshape(
            self.num_segments, self.max_nip, 6, self.num_segments * 2 * self.max_dof
        )

        return J

    @eqx.filter_jit
    def jacobian_bodyframe(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
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
                    - (g_pass, J_pass): scaled tip state to pass forward.
                    - (g_pass, J_pass): same as scan output.
            """
            g_tip, J_tip = carry

            # Joint ============================
            B_joint_i = self.V_B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.V_xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # (4,4)
            T_g_joint = lie.Tangent_gi_se3(xi_joint_i, 1, eps=self.tangent_eps)  # (6,6)

            # contribution of this joint to its own block
            T_g_joint_i_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(T_g_joint @ B_joint_i)
            )

            # propagate Jacobian through this joint (left-trivialized, body-frame)
            Ad_g_joint_inv = lie.Adjoint_g_inv_SE3(g_joint_i)  # shape (6, 6)

            g_j_scaled = g_tip @ g_joint_i
            J_j_scaled = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, (J_tip + T_g_joint_i_B_joint_i)
            )

            # Link ========================
            Xs_i = self.V_Xs[i_segment]  # (max_nip,)
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]  # (max_nip-1, 6)
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]  # (max_nip-1, 6)
            length_i = self.V_L[i_segment]
            B_Z1_i = self.V_B_Z1[i_segment]  # (max_nip-1, 6, max_dof)
            B_Z2_i = self.V_B_Z2[i_segment]  # (max_nip-1, 6, max_dof)

            # scale like in _jacobian_gauss (work in normalized coordinate, then rescale)
            g_j = g_j_scaled.at[0:3, 3].divide(length_i)
            J_j = J_j_scaled.at[:, :, 3:6, :].divide(length_i)

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

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j = (B_Z1_j @ q_i) + xi_ref_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_Z2_j = (B_Z2_j @ q_i) + xi_ref_Z2_i[j_eval].at[:3].multiply(length_i)

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2_j)

                Magnus_j = (H / 2) * (xi_Z1_j + xi_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2_j
                )

                B_Magnus_j = (H / 2) * (B_Z1_j + B_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ B_Z2_j - ad_xi_Z2_j @ B_Z1_j
                )

                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)
                T_step = lie.Tangent_gi_se3(Magnus_j, 1, eps=self.tangent_eps)

                # add contribution for this link block
                T_step_B_step = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_j)
                )

                Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)

                g_next = g_prev @ g_step
                J_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (J_prev + T_step_B_step)
                )

                return (g_next, J_next), None

            # If this segment is before the target, consume all cells.
            def do_full_link() -> tuple[Array, Array]:
                """Segment before target `s`: consume all its cells.

                Returns:
                    Tuple[Array, Array]: (g_end, J_end), shapes (4, 4) and
                    (num_segments, 2, 6, max_dof).
                """
                (g_end, J_end), _ = lax.scan(
                    full_cell, (g_j, J_j), jnp.arange(self.max_nip - 1)
                )
                return g_end, J_end

            # If this segment is the one containing s, step up to the cell j-1 (masked), then do a partial cell of size Hp.
            def do_partial_link() -> tuple[Array, Array]:
                """Segment containing `s`: consume up to j-1, then partial cell.

                Returns:
                    Tuple[Array, Array]: (g_out, J_out) at location `s`.
                """
                x = s_local / length_i
                j = jnp.clip(jnp.searchsorted(Xs_i, x) - 1, 0, self.max_nip - 2)

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
                    full_cell_masked, (g_j, J_j), jnp.arange(self.max_nip - 1)
                )

                # partial cell j with Hp
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                Xp = jnp.array([Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp])
                Bp = self._eval_B_segment(i_segment, Xp)  # (2,6,max_dof)
                xi_Z1_j = (Bp[0] @ q_i) + xi_ref_Z1_i[j].at[:3].multiply(length_i)
                xi_Z2_j = (Bp[1] @ q_i) + xi_ref_Z2_i[j].at[:3].multiply(length_i)

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2_j)

                Magnus_p = (Hp / 2) * (xi_Z1_j + xi_Z2_j) + (
                    jnp.sqrt(3) * Hp * Hp / 12
                ) * (ad_xi_Z1_j @ xi_Z2_j)
                B_Magnus_p = (Hp / 2) * (Bp[0] + Bp[1]) + (
                    jnp.sqrt(3) * Hp * Hp / 12
                ) * (ad_xi_Z1_j @ Bp[1] - ad_xi_Z2_j @ Bp[0])

                g_step = lie.exp_gn_SE3(Magnus_p, self.global_eps)
                T_step = lie.Tangent_gi_se3(Magnus_p, 1, eps=self.tangent_eps)

                T_block = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(T_step @ B_Magnus_p)
                )
                Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)

                g_out = g_in @ g_step
                J_out = jnp.einsum("ij,nmjk->nmik", Ad_step_inv, (J_in + T_block))
                return g_out, J_out

            g_out, J_out = lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

            # rescale back like in _jacobian_gauss for the state we pass forward
            g_pass = g_out.at[0:3, 3].multiply(length_i)
            J_pass = J_out.at[:, :, 3:6, :].multiply(length_i)
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
        J_flat = jnp.transpose(J_full, (2, 0, 1, 3)).reshape(
            6, self.num_segments * 2 * self.max_dof
        )
        J_local = J_flat @ self.B_select  # -> (6, num_active_strains)

        return J_local

    @eqx.filter_jit
    def jacobian(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian and its time derivative at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J (Array): Jacobian matrix of shape (6, num_dofs).
        """
        # TODO: Properly implement jacobian_and_derivative
        # This should compute the Jacobian and its time derivative in the inertial frame
        J = jnp.zeros((6, self.num_dofs))
        return J

    @eqx.filter_jit
    def _jacobian_derivative_gauss(
        self, q_gathered: Array, qd_gathered: Array
    ) -> Array:
        """
        Compute the Jacobian derivative matrices at all significant points of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.

            qd_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint velocities.

        Returns:
            Jd_list (Array): shape (num_segments, max_nip, 6, num_segments*2*max_dof) JAX array
                Jacobian derivative matrices at all significant points
        """

        def body_segment_i(
            carry: Array, i_segment: Array
        ) -> tuple[tuple[Array, Array, Array], Array]:
            """Accumulate transforms/Jacobian derivatives for a segment.

            Args:
                carry: Tuple (g_tip, Jd_tip, eta_tip) with shapes (4, 4),
                    (num_segments, 2, 6, max_dof), and (6,).
                i_segment: Segment index (int).

            Returns:
                Tuple[Tuple[Array, Array, Array], Array]:
                    - (g_tip_link_scaled, Jd_tip_link_scaled, eta_tip_link_scaled)
                      with shapes (4, 4), (num_segments, 2, 6, max_dof), (6,).
                    - Jd_link: per-point Jdot blocks, shape
                      (max_nip, num_segments, 2, 6, max_dof).
            """
            g_tip, Jd_tip, eta_tip = carry

            # Joint ============================
            B_joint_i = self.V_B_joint[i_segment]  # shape (6, max_dof)
            xi_ref_joint_i = self.V_xi_ref_joint[i_segment]  # shape (6,)
            q_joint_i = q_gathered[i_segment, 0]  # shape (max_dof,)
            qd_joint_i = qd_gathered[i_segment, 0]  # shape (max_dof,)

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i  # shape (6,)
            xid_joint_i = B_joint_i @ qd_joint_i  # shape (6,)

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # shape (4, 4)
            T_g_joint = lie.Tangent_gi_se3(
                xi_joint_i, 1, eps=self.tangent_eps
            )  # shape (6, 6)
            Td_g_joint = lie.Tangent_derivative_gi_se3(
                xi_joint_i, xid_joint_i, 1, eps=self.tangent_eps
            )  # shape (6, 6)

            Td_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(
                    lie.adjoint_se3(eta_tip) @ (T_g_joint @ B_joint_i)
                    + Td_g_joint @ B_joint_i
                )
            )  # shape (num_segments, 2, 6, max_dof)

            Ad_g_joint_inv = lie.Adjoint_g_inv_SE3(g_joint_i)

            g_j_scaled = g_tip @ g_joint_i
            Jd_j_scaled = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, Jd_tip + Td_g_joint_B_joint_i
            )
            eta_j_scaled = Ad_g_joint_inv @ (
                eta_tip + T_g_joint @ B_joint_i @ qd_joint_i
            )

            # Link ========================
            Xs_i = self.V_Xs[i_segment]  # shape (max_nip,)
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]  # shape (max_nip - 1, 6)
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]  # shape (max_nip - 1, 6)
            length_i = self.V_L[i_segment]  # shape (1,)
            B_Z1_i = self.V_B_Z1[i_segment]  # shape (max_nip - 1, 6, max_dof)
            B_Z2_i = self.V_B_Z2[i_segment]  # shape (max_nip - 1, 6, max_dof)

            g_j = g_j_scaled.at[0:3, 3].divide(length_i)
            Jd_j = Jd_j_scaled.at[:, :, 3:6, :].divide(length_i)
            eta_j = eta_j_scaled.at[3:6].divide(length_i)

            q_i = q_gathered[i_segment, 1]
            qd_i = qd_gathered[i_segment, 1]

            def body_eval_points(
                carry: Array, j_eval: Array
            ) -> tuple[tuple[Array, Array, Array], Array]:
                """Advance one cell; update Jdot using Magnus/Tangent terms.

                Args:
                    carry: Tuple (g_prev, Jd_prev, eta_prev).
                    j_eval: Cell index (int).

                Returns:
                    Tuple[Tuple[Array, Array, Array], Array]: Updated carry and
                    scaled Jdot block.
                """
                # g_prev, Jd_prev, eta_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                xi_ref_Z1_j = xi_ref_Z1_i[j_eval].at[:3].multiply(length_i)
                xi_ref_Z2_j = xi_ref_Z2_i[j_eval].at[:3].multiply(length_i)

                B_Z1_j = B_Z1_i[j_eval]
                B_Z2_j = B_Z2_i[j_eval]

                xi_Z1_j = B_Z1_j @ q_i + xi_ref_Z1_j
                xi_Z2_j = B_Z2_j @ q_i + xi_ref_Z2_j
                xid_Z1_j = B_Z1_j @ qd_i

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1_j)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2_j)

                Magnus_j = (H / 2) * (xi_Z1_j + xi_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2_j
                )

                B_Magnus_j = (H / 2) * (B_Z1_j + B_Z2_j) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ B_Z2_j - ad_xi_Z2_j @ B_Z1_j
                )

                Magnusd_j = B_Magnus_j @ qd_i

                Magnusdd_dq_j = (
                    ((jnp.sqrt(3) * H**2) / 6) * lie.adjoint_se3(xid_Z1_j) @ B_Z2_j
                )

                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)  # shape (4, 4)
                T_Magnus_j = lie.Tangent_gi_se3(
                    Magnus_j, 1, eps=self.tangent_eps
                )  # shape (6, 6)
                Td_Magnus_j = lie.Tangent_derivative_gi_se3(
                    Magnus_j, Magnusd_j, 1, eps=self.tangent_eps
                )  # shape (6, 6)

                T_B_Magnus_step = T_Magnus_j @ B_Magnus_j  # shape (6, max_dof)
                T_B_Magnusd_step = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(
                        lie.adjoint_se3(eta_j) @ T_Magnus_j @ B_Magnus_j
                        + Td_Magnus_j @ B_Magnus_j
                        + T_Magnus_j @ Magnusdd_dq_j
                    )
                )  # shape (num_segments, 2, 6, max_dof)

                Ad_g_step_inv = lie.Adjoint_g_inv_SE3(
                    g_step
                )  # shape (num_segments, 6, 6)
                _g_j = g_j @ g_step
                _Jd_j = jnp.einsum(
                    "ij,nmjk->nmik", Ad_g_step_inv, Jd_j + T_B_Magnusd_step
                )
                _eta_j = Ad_g_step_inv @ (eta_j + T_B_Magnus_step @ qd_i)

                Jd_j_scaled = _Jd_j.at[:, :, 3:6, :].multiply(length_i)

                return (_g_j, _Jd_j, _eta_j), Jd_j_scaled

            indices_eval_points = jnp.arange(self.max_nip - 1)

            (g_tip_link, Jd_tip_link, eta_tip_link), Jd_link = lax.scan(
                f=body_eval_points, init=(g_j, Jd_j, eta_j), xs=indices_eval_points
            )

            # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
            # carry = (g_j, Jd_j, eta_j)
            # Jd_link = []
            # for x in indices_eval_points:
            #     (g_j, Jd_j, eta_j), Jd_j_scaled_here = body_eval_points(carry, x)
            #     carry = (g_j, Jd_j, eta_j)
            #     Jd_link.append(Jd_j_scaled_here)
            # (g_tip_link, Jd_tip_link, eta_tip_link) = carry
            # Jd_link = jnp.array(Jd_link)

            Jd_link = jnp.concatenate(
                (jnp.expand_dims(Jd_j_scaled, axis=0), Jd_link), axis=0
            )

            g_tip_link_scaled = g_tip_link.at[0:3, 3].multiply(length_i)
            Jd_tip_link_scaled = Jd_tip_link.at[:, :, 3:6, :].multiply(length_i)
            eta_tip_link_scaled = eta_tip_link.at[3:6].multiply(length_i)

            return (g_tip_link_scaled, Jd_tip_link_scaled, eta_tip_link_scaled), Jd_link

        indices_link = jnp.arange(0, self.num_segments)

        g_init = self.g0

        Jd_init = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        eta_init = jnp.zeros((6,))

        _, Jd_list = lax.scan(
            f=body_segment_i, init=(g_init, Jd_init, eta_init), xs=indices_link
        )

        # For debugging purposes, you can uncomment the following lines to see the step-by-step computation
        # carry = (g_init, Jd_init, eta_init)
        # Jd_list = []
        # for x in indices_link:
        #     (g_tip_link, Jd_tip_link, eta_tip_link), Jd_j = body_segment_i(carry, x)
        #     carry = (g_tip_link, Jd_tip_link, eta_tip_link)
        #     Jd_list.append(Jd_j)
        # Jd_list = jnp.array(Jd_list)

        # (num_segments, max_nip, num_segments, 2, 6, max_dof) => (num_segments, max_nip, 6, num_segments*2*max_dof)
        Jd_reordered = jnp.transpose(Jd_list, (0, 1, 4, 2, 3, 5))
        Jd = Jd_reordered.reshape(
            self.num_segments, self.max_nip, 6, self.num_segments * 2 * self.max_dof
        )

        return Jd

    @eqx.filter_jit
    def jacobian_derivative_bodyframe(self, q: Array, qd: Array, s: Array) -> Array:
        """
        Compute the Jacobian derivative of the forward kinematics at a point s along the robot in the body frame.
        Args:
            q (Array): generalized coordinates of shape (dof_tot,).
            qd (Array): generalized velocities of shape (dof_tot,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            Jd_local (Array): Jacobian derivative of the forward kinematics at point s in the body frame, shape (6, num_active_strains)
        """
        q_gathered = self._min_size_gathered(q)
        qd_gathered = self._min_size_gathered(qd)

        # Segment contenant s et abscisse locale
        segment_idx, s_local = self.classify_segment(s)

        def body_segment_i(
            carry: Array, i_segment: Array
        ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
            """Propagate body-frame Jdot across a segment up to `s`.

            Args:
                carry (Array): Tuple (g_tip, Jd_tip, eta_tip).
                i_segment (Array): Segment index.

            Returns:
                Tuple[Tuple[Array, Array, Array], Tuple[Array, Array, Array]]:
                    - (g_pass, Jd_pass, eta_pass): scaled tip state to pass forward.
                    - (g_pass, Jd_pass, eta_pass): same as scan output.
            """
            g_tip, Jd_tip, eta_tip = carry

            # Joint ============================
            B_joint_i = self.V_B_joint[i_segment]  # (6, max_dof)
            xi_ref_joint_i = self.V_xi_ref_joint[i_segment]  # (6,)
            q_joint_i = q_gathered[i_segment, 0]  # (max_dof,)
            qd_joint_i = qd_gathered[i_segment, 0]  # (max_dof,)

            xi_joint_i = B_joint_i @ q_joint_i + xi_ref_joint_i  # (6,)
            xid_joint_i = B_joint_i @ qd_joint_i  # (6,)

            g_joint_i = lie.exp_gn_SE3(xi_joint_i, self.global_eps)  # (4,4)
            T_g_joint = lie.Tangent_gi_se3(xi_joint_i, 1, eps=self.tangent_eps)  # (6,6)
            Td_g_joint = lie.Tangent_derivative_gi_se3(
                xi_joint_i, xid_joint_i, 1, eps=self.tangent_eps
            )  # (6,6)

            # contribution dans le bloc "joint" de ce segment
            Td_g_joint_B_joint_i = (
                jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                .at[i_segment, 0]
                .set(
                    lie.adjoint_se3(eta_tip) @ T_g_joint @ B_joint_i
                    + Td_g_joint @ B_joint_i
                )
            )

            Ad_g_joint_inv = lie.Adjoint_g_inv_SE3(g_joint_i)  # (6,6)

            g_j_scaled = g_tip @ g_joint_i
            Jd_j_scaled = jnp.einsum(
                "ij,nmjk->nmik", Ad_g_joint_inv, Jd_tip + Td_g_joint_B_joint_i
            )
            eta_j_scaled = Ad_g_joint_inv @ (
                eta_tip + T_g_joint @ B_joint_i @ qd_joint_i
            )

            # Link ========================
            Xs_i = self.V_Xs[i_segment]  # (max_nip,)
            xi_ref_Z1_i = self.V_xi_ref_Z1[i_segment]  # (max_nip-1,6)
            xi_ref_Z2_i = self.V_xi_ref_Z2[i_segment]  # (max_nip-1,6)
            length_i = self.V_L[i_segment]
            B_Z1_i = self.V_B_Z1[i_segment]  # (max_nip-1,6,max_dof)
            B_Z2_i = self.V_B_Z2[i_segment]  # (max_nip-1,6,max_dof)

            g_j = g_j_scaled.at[0:3, 3].divide(length_i)
            Jd_j = Jd_j_scaled.at[:, :, 3:6, :].divide(length_i)
            eta_j = eta_j_scaled.at[3:6].divide(length_i)

            q_i = q_gathered[i_segment, 1]
            qd_i = qd_gathered[i_segment, 1]

            def full_cell(
                carry: Array, j_eval: Array
            ) -> tuple[tuple[Array, Array, Array], None]:
                """Consume a full cell; update g, Jdot and convective term.

                Args:
                    carry (Array): Tuple (g_prev, Jd_prev, eta_prev).
                    j_eval (Array): Cell index (int).

                Returns:
                    Tuple[Tuple[Array, Array, Array], None]: Updated carry and dummy output.
                """
                g_prev, Jd_prev, eta_prev = carry

                H = Xs_i[j_eval + 1] - Xs_i[j_eval]

                xi_Z1 = (B_Z1_i[j_eval] @ q_i) + xi_ref_Z1_i[j_eval].at[:3].multiply(
                    length_i
                )
                xi_Z2 = (B_Z2_i[j_eval] @ q_i) + xi_ref_Z2_i[j_eval].at[:3].multiply(
                    length_i
                )
                xid_Z1 = B_Z1_i[j_eval] @ qd_i

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2)

                Magnus_j = (H / 2) * (xi_Z1 + xi_Z2) + (jnp.sqrt(3) * H**2 / 12) * (
                    ad_xi_Z1_j @ xi_Z2
                )

                B_Magnus_j = (H / 2) * (B_Z1_i[j_eval] + B_Z2_i[j_eval]) + (
                    jnp.sqrt(3) * H**2 / 12
                ) * (ad_xi_Z1_j @ B_Z2_i[j_eval] - ad_xi_Z2_j @ B_Z1_i[j_eval])

                Magnusd_j = B_Magnus_j @ qd_i

                Magnusdd_dq = (
                    ((jnp.sqrt(3) * H**2) / 6)
                    * lie.adjoint_se3(xid_Z1)
                    @ B_Z2_i[j_eval]
                )

                g_step = lie.exp_gn_SE3(Magnus_j, self.global_eps)
                T_step = lie.Tangent_gi_se3(Magnus_j, 1, eps=self.tangent_eps)
                Td_step = lie.Tangent_derivative_gi_se3(
                    Magnus_j, Magnusd_j, 1, eps=self.tangent_eps
                )

                Td_block_link = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(
                        lie.adjoint_se3(eta_prev) @ T_step @ B_Magnus_j
                        + Td_step @ B_Magnus_j
                        + T_step @ Magnusdd_dq
                    )
                )

                Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)

                g_next = g_prev @ g_step
                Jd_next = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (Jd_prev + Td_block_link)
                )
                eta_next = Ad_step_inv @ (eta_prev + T_step @ B_Magnus_j @ qd_i)

                return (g_next, Jd_next, eta_next), None

            # Case 1: segment entirely before s → consume every cell
            def do_full_link() -> tuple[Array, Array, Array]:
                """Segment before target `s`: integrate all its cells (Jdot).

                Returns:
                    Tuple[Array, Array, Array]: (g_end, Jd_end, eta_end).
                """
                (g_end, Jd_end, eta_end), _ = lax.scan(
                    full_cell, (g_j, Jd_j, eta_j), jnp.arange(self.max_nip - 1)
                )
                return g_end, Jd_end, eta_end

            # Case 2: segment containing s → full cells up to j-1 then partial cell Hp
            def do_partial_link() -> tuple[Array, Array, Array]:
                """Segment containing `s`: consume up to j-1, then partial cell.

                Returns:
                    Tuple[Array, Array, Array]: (g_out, Jd_out, eta_out) at `s`.
                """
                x = s_local / length_i
                j = jnp.clip(jnp.searchsorted(Xs_i, x) - 1, 0, self.max_nip - 2)

                # consume up to j-1 (masked)
                def full_cell_masked(
                    carry: Array, idx: Array
                ) -> tuple[tuple[Array, Array, Array], None]:
                    """Advance only for idx < j; keep state otherwise.

                    Args:
                        carry (Array): Tuple (g_p, Jd_p, eta_p).
                        idx (Array): Cell index (int).

                    Returns:
                        Tuple[Tuple[Array, Array, Array], None]: Updated or kept carry; dummy output.
                    """
                    (g_p, Jd_p, eta_p), _ = full_cell(carry, idx)
                    g_keep, Jd_keep, eta_keep = carry
                    return (
                        lax.select(idx < j, g_p, g_keep),
                        lax.select(idx < j, Jd_p, Jd_keep),
                        lax.select(idx < j, eta_p, eta_keep),
                    ), None

                (g_in, Jd_in, eta_in), _ = lax.scan(
                    full_cell_masked, (g_j, Jd_j, eta_j), jnp.arange(self.max_nip - 1)
                )

                # partial cell j of size Hp
                H = Xs_i[j + 1] - Xs_i[j]
                Hp = jnp.clip(x - Xs_i[j], 0.0, H)

                Xp = jnp.array(
                    [Xs_i[j] + self.Z1 * Hp, Xs_i[j] + self.Z2 * Hp]
                )  # two partial Gauss points
                Bp = self._eval_B_segment(i_segment, Xp)  # (2,6,max_dof)

                xi_Z1 = (Bp[0] @ q_i) + xi_ref_Z1_i[j].at[:3].multiply(length_i)
                xi_Z2 = (Bp[1] @ q_i) + xi_ref_Z2_i[j].at[:3].multiply(length_i)
                xid_Z1 = Bp[0] @ qd_i

                ad_xi_Z1_j = lie.adjoint_se3(xi_Z1)
                ad_xi_Z2_j = lie.adjoint_se3(xi_Z2)

                Magnus_p = (Hp / 2) * (xi_Z1 + xi_Z2) + (jnp.sqrt(3) * Hp * Hp / 12) * (
                    ad_xi_Z1_j @ xi_Z2
                )
                B_Magnus_p = (Hp / 2) * (Bp[0] + Bp[1]) + (
                    jnp.sqrt(3) * Hp * Hp / 12
                ) * (ad_xi_Z1_j @ Bp[1] - ad_xi_Z2_j @ Bp[0])
                Magnusd_p = B_Magnus_p @ qd_i
                Magnusdd_dq_p = (
                    ((jnp.sqrt(3) * Hp * Hp) / 6) * lie.adjoint_se3(xid_Z1) @ Bp[1]
                )

                g_step = lie.exp_gn_SE3(Magnus_p, self.global_eps)
                T_step = lie.Tangent_gi_se3(Magnus_p, 1, eps=self.tangent_eps)
                Td_step = lie.Tangent_derivative_gi_se3(
                    Magnus_p, Magnusd_p, 1, eps=self.tangent_eps
                )

                Td_block_link = (
                    jnp.zeros((self.num_segments, 2, 6, self.max_dof))
                    .at[i_segment, 1]
                    .set(
                        lie.adjoint_se3(eta_in) @ T_step @ B_Magnus_p
                        + Td_step @ B_Magnus_p
                        + T_step @ Magnusdd_dq_p
                    )
                )

                Ad_step_inv = lie.Adjoint_g_inv_SE3(g_step)

                g_out = g_in @ g_step
                Jd_out = jnp.einsum(
                    "ij,nmjk->nmik", Ad_step_inv, (Jd_in + Td_block_link)
                )
                eta_out = Ad_step_inv @ (eta_in + T_step @ B_Magnus_p @ qd_i)

                return g_out, Jd_out, eta_out

            g_out, Jd_out, eta_out = lax.cond(
                i_segment < segment_idx,
                lambda _: do_full_link(),
                lambda _: do_partial_link(),
                operand=None,
            )

            # rescale the state passed to the next segment (same as the other functions)
            g_pass = g_out.at[0:3, 3].multiply(length_i)
            Jd_pass = Jd_out.at[:, :, 3:6, :].multiply(length_i)
            eta_pass = eta_out.at[3:6].multiply(length_i)

            return (g_pass, Jd_pass, eta_pass), (g_pass, Jd_pass, eta_pass)

        # Traverse the chain; freeze the state after the segment containing s
        def step(carry: Array, i: Array) -> tuple[tuple[Array, Array, Array], None]:
            """Walk segments; freeze state after the segment containing `s`.

            Args:
                carry (Array): Tuple (g_curr, Jd_curr, eta_curr).
                i (Array): Segment index (int).

            Returns:
                Tuple[Tuple[Array, Array, Array], None]: Next carry and dummy output.
            """
            (g_curr, Jd_curr, eta_curr), _ = body_segment_i(carry, i)
            g_next = jnp.where(i <= segment_idx, g_curr, carry[0])
            Jd_next = jnp.where(i <= segment_idx, Jd_curr, carry[1])
            eta_next = jnp.where(i <= segment_idx, eta_curr, carry[2])
            return (g_next, Jd_next, eta_next), None

        g0 = self.g0

        Jd0 = jnp.zeros((self.num_segments, 2, 6, self.max_dof))
        eta0 = jnp.zeros((6,))
        (g_s, Jd_full, _), _ = lax.scan(
            step, (g0, Jd0, eta0), jnp.arange(self.num_segments)
        )

        # (num_segments, 2, 6, max_dof) => (6, num_segments * 2 * max_dof)
        Jd_flat = jnp.transpose(Jd_full, (2, 0, 1, 3)).reshape(
            6, self.num_segments * 2 * self.max_dof
        )
        Jd_local = Jd_flat @ self.B_select  # (6, num_active_strains)

        return Jd_local

    @eqx.filter_jit
    def jacobian_and_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        """
        Compute the Jacobian and its time derivative at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).
            qd (Array): generalized velocities of shape (dof_tot,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J (Array): Jacobian matrix of shape (6, num_dofs).
            Jd (Array): Time derivative of the Jacobian, shape (6, num_dofs).
        """
        # TODO: Properly implement jacobian_and_derivative
        # This should compute the Jacobian and its time derivative in the inertial frame
        J = jnp.zeros((6, self.num_dofs))
        Jd = jnp.zeros((6, self.num_dofs))
        return J, Jd

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
        )  # (num_segments, max_nip, 6, num_segments * 2 * max_dof)

        # Define function for each quadrature point
        def B_segment_i(i_segment: Array) -> Array:
            """Assemble inertia contributions for one segment by quadrature.

            Args:
                i_segment (Array): Segment index (int).

            Returns:
                Array: Blocks over quadrature points, shape
                    (max_nip, num_segments*2*max_dof, num_segments*2*max_dof).
            """
            length_i = self.V_L[i_segment]
            Ws_i = self.V_Ws[i_segment]  # (max_nip, 1, )
            J_i = V_J[i_segment]  # (max_nip, 6, num_segments * 2 * max_dof)
            Ms_i = self.V_Ms[i_segment]  # (max_nip, 6, 6)

            def B_eval_points(i_eval: Array) -> Array:
                """Inertia block at a single quadrature point.

                Args:
                    i_eval (Array): Quadrature point index (int).

                Returns:
                    Array: Block of shape (num_segments*2*max_dof, num_segments*2*max_dof).
                """
                Ws_j = Ws_i[i_eval]
                J_j = J_i[i_eval]  # (6, num_segments * 2 * max_dof)
                Ms_j = Ms_i[i_eval]  # (6, 6)

                return Ws_j * (
                    J_j.T @ Ms_j @ J_j
                )  # (num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_segment_i = vmap(B_eval_points)(
                jnp.arange(1, self.max_nip - 1)
            )  # (max_nip - 2, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            # # For debugging purposes, we can use a list comprehension
            # B_blocks_segment_i = jnp.stack([B_eval_points(i_eval) for i_eval in range(self.max_nip)])  # (max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            return B_blocks_segment_i * length_i

        B_blocks_tot = vmap(
            B_segment_i
        )(
            jnp.arange(self.num_segments)
        )  # (num_segments, max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        # # For debugging purposes, we can use a list comprehension
        # B_blocks_tot = jnp.stack([B_segment_i(i_segment) for i_segment in range(self.num_segments)])  # (num_segments, max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))

        return B_full

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            B (Array): Inertia matrix, shape (dof_tot, dof_tot)
        """
        q_gathered = self._min_size_gathered(q)

        B_full = self._inertia_full_matrix(q_gathered)

        B = self.B_select.T @ B_full @ self.B_select

        return B

    @eqx.filter_jit
    def _coriolis_full_matrix(self, q_gathered: Array, qd_gathered: Array) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint coordinates.
            qd_gathered (Array): shape (num_segments, 2, max_dof) JAX array
                Joint velocities.

        Returns:
            C (Array): Full Coriolis matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """
        V_J = self._jacobian_gauss(
            q_gathered
        )  # (num_segments, max_nip, 6, num_segments * 2 * max_dof)
        V_Jd = self._jacobian_derivative_gauss(
            q_gathered, qd_gathered
        )  # (num_segments, max_nip, 6, num_segments * 2 * max_dof)

        qd_flat = qd_gathered.reshape(-1)

        # Define function for each quadrature point
        def C_segment_i(i_segment: Array) -> Array:
            """Assemble Coriolis contributions for one segment by quadrature.

            Args:
                i_segment (Array): Segment index (int).

            Returns:
                Array: Blocks over quadrature points, shape
                    (max_nip, num_segments*2*max_dof, num_segments*2*max_dof).
            """
            length_i = self.V_L[i_segment]
            Ws_i = self.V_Ws[i_segment]  # (max_nip, 1, )
            J_i = V_J[i_segment]  # (max_nip, 6, num_segments * 2 * max_dof)
            Jd_i = V_Jd[i_segment]  # (max_nip, 6, num_segments * 2 * max_dof)
            Ms_i = self.V_Ms[i_segment]  # (max_nip, 6, 6)

            def C_eval_points(i_eval: Array) -> Array:
                """Coriolis block at a single quadrature point.

                Args:
                    i_eval (Array): Quadrature point index (int).

                Returns:
                    Array: Block of shape (num_segments*2*max_dof, num_segments*2*max_dof).
                """
                Ws_j = Ws_i[i_eval]
                J_j = J_i[i_eval]  # (6, num_segments * 2 * max_dof)
                Jd_j = Jd_i[i_eval]  # (6, num_segments * 2 * max_dof)
                Ms_j = Ms_i[i_eval]  # (6, 6)

                return Ws_j * (
                    J_j.T
                    @ (Ms_j @ Jd_j + lie.coadjoint_se3(J_j @ qd_flat) @ Ms_j @ J_j)
                )  # (num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            # we can skip the first and last quadrature points since their weight is zero
            C_blocks_segment_i = vmap(C_eval_points)(
                jnp.arange(1, self.max_nip - 1)
            )  # (max_nip - 2, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            # For debugging purposes, we can use a list comprehension
            # C_blocks_segment_i = jnp.stack([C_eval_points(i_eval) for i_eval in range(self.max_nip)])  # (max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

            return C_blocks_segment_i * length_i

        C_blocks_tot = vmap(
            C_segment_i
        )(
            jnp.arange(self.num_segments)
        )  # (num_segments, max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        # For debugging purposes, we can use a list comprehension
        # C_blocks_tot = jnp.stack([C_segment_i(i_segment) for i_segment in range(self.num_segments)])  # (num_segments, max_nip, num_segments * 2 * max_dof, num_segments * 2 * max_dof)

        C = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C

    @eqx.filter_jit
    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).
            qd (Array): generalized velocities of shape (dof_tot,).

        Returns:
            C (Array): Coriolis matrix, shape (dof_tot, dof_tot)
        """
        q_gathered = self._min_size_gathered(q)
        qd_gathered = self._min_size_gathered(qd)

        C_full = self._coriolis_full_matrix(q_gathered, qd_gathered)

        C = self.B_select.T @ C_full @ self.B_select

        return C

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
        V_J = self._jacobian_gauss(
            q_gathered
        )  # (num_segments, max_nip, 6, num_segments * 2 * max_dof)
        V_g = self._forward_kinematics_gauss(
            q_gathered
        )  # (num_segments, max_nip, 4, 4)

        def G_segment_i(i_segment: Array) -> Array:
            """Assemble gravitational force contributions for one segment.

            Args:
                i_segment (Array): Segment index (int).

            Returns:
                Array: Blocks over quadrature points, shape (max_nip, num_segments*2*max_dof, 1).
            """
            length_i = self.V_L[i_segment]
            Ws_i = self.V_Ws[i_segment]  # (max_nip, 1, )
            g_i = V_g[i_segment]  # (max_nip, 4, 4)
            J_i = V_J[i_segment]  # (max_nip, 6, num_segments * 2 * max_dof)
            M_i = self.V_Ms[i_segment]  # (max_nip, 6, 6)

            def G_eval_points(i_eval: Array) -> Array:
                """Gravitational block at a single quadrature point.

                Args:
                    i_eval (Array): Quadrature point index (int).

                Returns:
                    Array: Block of shape (num_segments*2*max_dof, 1).
                """
                Ws_j = Ws_i[i_eval]  # ()
                g_j = g_i[i_eval]  # (4, 4)
                Ad_g_j_inv = lie.Adjoint_g_inv_SE3(g_j)  # (6, 6)
                J_j = J_i[i_eval]  # (6, num_segments * 2 * max_dof)
                M_j = M_i[i_eval]  # (6, 6)

                G_j = (
                    -Ws_j * J_j.T @ M_j @ Ad_g_j_inv @ self.g
                )  # (num_segments * 2 * max_dof, 1)
                return G_j

            # we can skip the first and last quadrature points since their weight is zero
            G_blocks_segment_i = vmap(G_eval_points)(
                jnp.arange(1, self.max_nip - 1)
            )  # (max_nip - 2, num_segments * 2 * max_dof, 1)

            # For debugging purposes, we can use a list comprehension
            # G_blocks_segment_i = jnp.stack([G_eval_points(i_eval) for i_eval in range(self.max_nip)])

            return G_blocks_segment_i * length_i

        G_blocks_tot = vmap(G_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, max_nip, num_segments * 2 * max_dof, 1)

        # For debugging purposes, we can use a list comprehension
        # G_blocks_tot = jnp.stack([G_segment_i(i_segment) for i_segment in range(self.num_segments)])  # (num_segments, max_nip, num_segments * 2 * max_dof, 1)

        G_full = jnp.sum(
            G_blocks_tot, axis=(0, 1)
        )  # Sum over links and quadrature points

        return G_full

    @eqx.filter_jit
    def gravitational_force(self, q: Array) -> Array:
        """
        Compute the gravitational force vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            G (Array): Gravitational force vector, shape (dof_tot, 1)
        """
        q_gathered = self._min_size_gathered(q)

        G_full = self._gravitational_full_force(q_gathered)

        G = self.B_select.T @ G_full

        return G

    @eqx.filter_jit
    def gravitational_energy(self, q: Array) -> Array:
        """
        Compute the gravitational potential energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            U_g (Array): Gravitational potential energy (scalar).
        """
        # TODO: Properly implement gravitational energy computation
        # This should integrate the gravitational potential energy over the robot's mass distribution
        return jnp.array(0.0)

    @eqx.filter_jit
    def _stiffness_full_matrix(self) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """

        def K_segment_i(i_segment: Array) -> Array:
            """Assemble stiffness contributions for one segment by quadrature.

            Args:
                i_segment (Array): Segment index (int).

            Returns:
                Array: Two blocks (joint/link) of shape (2, max_dof, max_dof).
            """
            # Joint ==============================
            K_joint_i = jnp.zeros(
                (self.max_dof, self.max_dof)
            )  # self.V_K_joint[i_segment]  # (max_dof, max_dof) TODO

            # Link ===============================
            length_i = self.V_L[i_segment]
            Ws_i = self.V_Ws[i_segment]  # (max_nip, 1, )
            Es_i = self.V_Es[i_segment]  # (max_nip, 6, 6)
            B_Xs_i = self.V_B_Xs[i_segment]  # (max_nip, 6, max_dof)

            def K_eval_points(i_eval: Array) -> Array:
                """Stiffness block at a single quadrature point.

                Args:
                    i_eval (Array): Quadrature point index (int).

                Returns:
                    Array: Block of shape (max_dof, max_dof).
                """
                Ws_j = Ws_i[i_eval]
                Es_j = Es_i[i_eval]  # (6, 6)
                B_Xs_j = B_Xs_i[i_eval]  # (6, max_dof)

                Es_j_scaled = (
                    Es_j.at[:3, :].divide(length_i**3).at[3:, :].divide(length_i)
                )

                return Ws_j * (B_Xs_j.T @ Es_j_scaled @ B_Xs_j)

            # we can skip the first and last quadrature points since their weight is zero
            K_link_i = (
                jnp.sum(vmap(K_eval_points)(jnp.arange(1, self.max_nip - 1)), axis=0)
                * length_i**2
            )  # (max_nip - 2, max_dof, max_dof)

            # Create a (2, max_dof, max_dof) array with K_joint_i and K_segment_i
            K_blocks_segment_i = jnp.stack([K_joint_i, K_link_i], axis=0)
            return K_blocks_segment_i

        K_blocks_tot = vmap(K_segment_i)(
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
            K (Array): Stiffness matrix, shape (dof_tot, dof_tot)
        """
        K_full = self._stiffness_full_matrix()

        K = self.B_select.T @ K_full @ self.B_select

        return K

    @eqx.filter_jit
    def elastic_force(self, q: Array) -> Array:
        """
        Compute the elastic forces of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            tau_el (Array): Elastic force of shape (dof_tot,).
        """
        K = self.stiffness_matrix()
        tau_el = K @ q

        return tau_el

    @eqx.filter_jit
    def _damping_full_matrix(self) -> Array:
        """
        Compute the full damping matrix of the robot.

        Returns:
            D_full (Array): Full damping matrix, shape (num_segments * 2 * max_dof, num_segments * 2 * max_dof)
        """

        def D_segment_i(i_segment: Array) -> Array:
            """Assemble damping contributions for one segment by quadrature.

            Args:
                i_segment (Array): Segment index (int).

            Returns:
                Array: Two blocks (joint/link) of shape (2, max_dof, max_dof).
            """
            # Joint ============================== TODO
            D_joint_i = jnp.zeros(
                (self.max_dof, self.max_dof)
            )  # Initialize joint stiffness matrix

            # Link ===============================
            length_i = self.V_L[i_segment]
            Ws_i = self.V_Ws[i_segment]  # (max_nip, 1, )
            Gs_i = self.V_Gs[i_segment]  # (max_nip, 6, 6)
            B_Xs_i = self.V_B_Xs[i_segment]  # (max_nip, 6, max_dof)

            def D_eval_points(i_eval: Array) -> Array:
                """Damping block at a single quadrature point.

                Args:
                    i_eval (Array): Quadrature point index (int).

                Returns:
                    Array: Block of shape (max_dof, max_dof).
                """
                Ws_j = Ws_i[i_eval]
                Gs_j = Gs_i[i_eval]  # (6, 6)
                B_Xs_j = B_Xs_i[i_eval]  # (6, max_dof)

                Gs_j_scaled = (
                    Gs_j.at[:3, :].divide(length_i**3).at[3:, :].divide(length_i)
                )

                return Ws_j * (B_Xs_j.T @ Gs_j_scaled @ B_Xs_j)

            # we can skip the first and last quadrature points since their weight is zero
            D_link_i = (
                jnp.sum(vmap(D_eval_points)(jnp.arange(1, self.max_nip - 1)), axis=0)
                * length_i**2
            )  # (max_nip - 2, max_dof, max_dof)

            # Create a (2, max_dof, max_dof) array with D_joint_i and D_segment_i
            D_blocks_segment_i = jnp.stack([D_joint_i, D_link_i], axis=0)
            return D_blocks_segment_i

        D_blocks_tot = vmap(D_segment_i)(
            jnp.arange(self.num_segments)
        )  # (num_segments, 2, max_dof, max_dof)

        # Supposons que D_blocks est de forme (num_segments, 2, max_dof, max_dof)
        D_blocks_flat = D_blocks_tot.reshape(
            -1, self.max_dof, self.max_dof
        )  # (num_segments * 2, max_dof, max_dof)

        # Convertir en liste de matrices
        D_blocks_list = [D_blocks_flat[i] for i in range(D_blocks_flat.shape[0])]

        # Construire la matrice diagonale par blocs
        D_full = jax.scipy.linalg.block_diag(*D_blocks_list)

        return D_full

    @eqx.filter_jit
    def damping_matrix(self, q: Array) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            D (Array): Damping matrix, shape (dof_tot, dof_tot)
        """
        D_full = self._damping_full_matrix()

        D = self.B_select.T @ D_full @ self.B_select

        return D

    @eqx.filter_jit
    def actuation_matrix(self, q: Array) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (dof_tot,).

        Returns:
            A (Array): Actuation matrix of shape (dof_tot, dof_tot)
        """
        A = jnp.identity(self.dof_tot_system)
        return A

    @eqx.filter_jit
    def actuation_force(self, q: Array, u: Array) -> Array:
        """
        Compute the actuation force acting on the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            u (Array): actuation/control input of shape (num_actuators,).

        Returns:
            tau_u (Array): Actuation force of shape (num_active_strains, ).
        """
        # evaluate the actuation matrix
        A = self.actuation_matrix(q)

        # compute the actuation force
        tau_u = A @ u

        return tau_u

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
            yd: Time derivative of the state vector.
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

        # evaluate the dynamical matrices
        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravitational_force(q)
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )  # Compute the acceleration

        yd = jnp.concatenate([qd, qdd])

        return yd
