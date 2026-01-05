__all__ = ["LinkAttributes", "JointAttributes", "BasisAttributes"]
from dataclasses import dataclass, field
from typing import Literal

from jax import Array

from soromox.systems.soft_robot import CrossSectionGeometry


@dataclass
class LinkAttributes:
    """
    Geometric and material attributes of one soft link/segment.

    The cross-section may vary linearly from an initial value at the link base
    (index i) to a final value at the link tip (index f). Only the parameters
    relevant to the selected `cross_section_geometry` are used; the others are ignored.

    Attributes
    ----------
    cross_section_geometry : CrossSectionGeometry
        Cross-section family used for geometric and inertial properties.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-]; typical range (-1, 0.5).
    rho : float
        Mass density [kg/m^3].
    eta : float
        Material damping coefficient used to scale link damping matrices [N·s/m].
    L : float
        Link length [m].

    r_i, r_f : float, optional
        Initial and final radius for circular section [m]. Used only if
        `cross_section_geometry == CrossSectionGeometry.CIRCULAR`.
    h_i, h_f : float, optional
        Initial and final height for rectangular section [m]. Used only if
        `cross_section_geometry == CrossSectionGeometry.RECTANGULAR`.
    w_i, w_f : float, optional
        Initial and final width for rectangular section [m]. Used only if
        `cross_section_geometry == CrossSectionGeometry.RECTANGULAR`.
    a_i, a_f : float, optional
        Initial and final semi-major axis for elliptical section [m]. Used only if
        `cross_section_geometry == CrossSectionGeometry.ELLIPTICAL`.
    b_i, b_f : float, optional
        Initial and final semi-minor axis for elliptical section [m]. Used only if
        `cross_section_geometry == CrossSectionGeometry.ELLIPTICAL`.

    Notes
    -----
    - Geometric parameters vary linearly along the link from the initial to the
      final value and are used to compute A, I_x, I_y, I_z for each quadrature
      point.
    - Shear modulus is internally computed as G = E / (2 (1 + nu)).
    """

    cross_section_geometry: CrossSectionGeometry
    E: float
    nu: float
    rho: float
    eta: float
    L: float

    r_i: float | None = 0.0
    r_f: float | None = 0.0
    h_i: float | None = 0.0
    h_f: float | None = 0.0
    w_i: float | None = 0.0
    w_f: float | None = 0.0
    a_i: float | None = 0.0
    a_f: float | None = 0.0
    b_i: float | None = 0.0
    b_f: float | None = 0.0


@dataclass
class JointAttributes:
    """
    Kinematic and stiffness attributes of the joint preceding a link.

    Attributes
    ----------
    jointtype : {"Revolute", "Prismatic", "Helical", "Cylindrical", "Planar", "Spherical", "Free", "Fixed"}
        Joint family which determines the number and nature of DOFs.
    axis : {"x", "y", "z"}, default "x"
        Axis for joints that use a single axis: revolute, prismatic, helical,
        cylindrical. Ignored for planar, spherical, free, fixed.
    plane : {"xy", "yz", "xz"}, default "xy"
        Motion plane for planar joints. Ignored for other joint types.
    pitch : float, default 0.0
        Helical joint pitch [m/rad]. Only used when `jointtype == "Helical"`.
    K_joint : Array or list, shape (dof_joint, dof_joint), optional
        Joint stiffness matrix in generalized joint coordinates. Units are
        consistent with the active DOFs: rotational terms [N·m/rad], translational
        terms [N/m]; off-diagonal couplings accordingly. If omitted or with
        mismatching shape it defaults to zeros.
    """

    jointtype: Literal[
        "Revolute",
        "Prismatic",
        "Helical",
        "Cylindrical",
        "Planar",
        "Spherical",
        "Free",
        "Fixed",
    ]

    axis: Literal["x", "y", "z"] = "x"
    plane: Literal["xy", "yz", "xz"] = "xy"
    pitch: float = 0.0

    K_joint: Array | list = field(default_factory=list)


@dataclass
class BasisAttributes:
    """
    Strain-basis parametrization for the link's variable strain field.

    The 6 strain components are ordered as
    [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z], where curvature
    components `kappa_*` have units [rad/m] and shear/axial strain components
    `sigma_*` have units [1/m]. These are the components of the Cosserat strain
    twist per unit length used in the SE(3) integration.

    Attributes
    ----------
    basistype : {"Monomial", "Legendre", "Chebychev", "Fourier", "Gaussian", "IMQ"}
        Basis family used to expand each selected strain component along the link.
    Bdof : ArrayLike of shape (6,) or (6, 1)
        Selection mask for each strain component; 1 enables that family of
        deformation, 0 disables it. Order is
        [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]. Dimensionless.
    Bodr : ArrayLike of shape (6,) or (6, 1)
        Basis order for each strain component (non-negative integers). The exact
        interpretation of the order depends on the chosen `basistype`.
    xi_ref : ArrayLike of shape (6,) or (6, 1)
        Reference (prestrain) vector for the 6 strain components. Units
        [rad/m, rad/m, rad/m, 1/m, 1/m, 1/m]. Default is the prestrain
        ([0, 0, 0, 1, 0, 0]) such that the soft robot backbone points along the local x-axis.

    Notes
    -----
    - The number of link DOFs is determined from (`Bdof`, `Bodr`) following the
      rules of the selected basis; internal code pads/truncates to the global
      `max_dof` for batched computations.
    - `xi_ref` can be used to impose prestrain.
    """

    basistype: Literal[
        "Monomial", "Legendre", "Chebychev", "Fourier", "Gaussian", "IMQ"
    ]
    Bdof: (
        Array | list
    )  # shape (6,1) indicating whether each type of deformation is selected (1) or not (0)
    Bodr: (
        Array | list
    )  # shape (6,1) indicating the orders of the basis functions for each type of deformation
    xi_ref: Array | list = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )  # shape (6,1) indicating the reference strain values for each type of deformation
