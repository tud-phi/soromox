from dataclasses import dataclass, field
from typing import Literal

from jax import Array

from soromox.systems.soft_robot import CrossSectionGeometry

__all__ = ["GVSSegment", "LinkSpec", "JointSpec", "StrainBasisSpec"]

StrainComponent = Literal[
    "kappa_x",
    "kappa_y",
    "kappa_z",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]

BasisType = Literal[
    "monomial",
    "legendre",
    "chebyshev",
    "fourier",
    "gaussian",
    "imq",
]

JointType = Literal[
    "revolute",
    "prismatic",
    "helical",
    "cylindrical",
    "planar",
    "spherical",
    "free",
    "fixed",
]


_COMPONENT_INDEX = {
    "kappa_x": 0,
    "kappa_y": 1,
    "kappa_z": 2,
    "sigma_x": 3,
    "sigma_y": 4,
    "sigma_z": 5,
}


def _component_mask(active: Array | list | tuple[str, ...]) -> Array | list:
    if isinstance(active, (list, tuple)) and all(
        isinstance(item, str) for item in active
    ):
        mask = [0, 0, 0, 0, 0, 0]
        for component in active:
            try:
                mask[_COMPONENT_INDEX[component]] = 1
            except KeyError as exc:
                raise ValueError(f"Unknown strain component {component!r}") from exc
        return mask
    return active


def _orders_vector(orders: int | Array | list | tuple) -> Array | list:
    if isinstance(orders, int):
        return [orders] * 6
    return orders


@dataclass
class LinkSpec:
    """Geometric and material specification of one GVS link.

    This is a single-link input object. When a GVS model is constructed, these
    scalar values are packed into per-segment arrays in ``GVSParams.link``.
    Initial and final cross-section dimensions are interpreted at normalized
    arclength ``x = 0`` and ``x = 1`` and are linearly interpolated along the
    link.

    Attributes:
        cross_section_geometry: Cross-section family used to compute area and
            second moments of area. Unit: dimensionless enum value. Shape:
            scalar ``()``.
        E: Young's modulus of the link material. Unit: pascal ``Pa``. Shape:
            scalar ``()``.
        nu: Poisson's ratio of the link material. Unit: dimensionless. Shape:
            scalar ``()``.
        rho: Mass density of the link material. Unit: ``kg/m^3``. Shape:
            scalar ``()``.
        eta: Material damping coefficient used to form the local damping
            matrix. Unit: ``Pa*s`` (``N*s/m^2``). Shape: scalar ``()``.
        L: Link arclength. Unit: meter ``m``. Shape: scalar ``()``.
        r_i: Initial radius for circular cross-sections at ``x = 0``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-circular links.
        r_f: Final radius for circular cross-sections at ``x = 1``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-circular links.
        h_i: Initial height for rectangular cross-sections at ``x = 0``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-rectangular
            links.
        h_f: Final height for rectangular cross-sections at ``x = 1``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-rectangular
            links.
        w_i: Initial width for rectangular cross-sections at ``x = 0``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-rectangular
            links.
        w_f: Final width for rectangular cross-sections at ``x = 1``. Unit:
            meter ``m``. Shape: scalar ``()``. Ignored for non-rectangular
            links.
        a_i: Initial semi-major axis for elliptical cross-sections at
            ``x = 0``. Unit: meter ``m``. Shape: scalar ``()``. Ignored for
            non-elliptical links.
        a_f: Final semi-major axis for elliptical cross-sections at
            ``x = 1``. Unit: meter ``m``. Shape: scalar ``()``. Ignored for
            non-elliptical links.
        b_i: Initial semi-minor axis for elliptical cross-sections at
            ``x = 0``. Unit: meter ``m``. Shape: scalar ``()``. Ignored for
            non-elliptical links.
        b_f: Final semi-minor axis for elliptical cross-sections at
            ``x = 1``. Unit: meter ``m``. Shape: scalar ``()``. Ignored for
            non-elliptical links.
    """

    cross_section_geometry: CrossSectionGeometry
    E: float
    nu: float
    rho: float
    eta: float
    L: float
    r_i: float = 0.0
    r_f: float = 0.0
    h_i: float = 0.0
    h_f: float = 0.0
    w_i: float = 0.0
    w_f: float = 0.0
    a_i: float = 0.0
    a_f: float = 0.0
    b_i: float = 0.0
    b_f: float = 0.0

    @classmethod
    def circular(
        cls,
        *,
        E: float,
        nu: float,
        rho: float,
        eta: float,
        L: float,
        r: float | None = None,
        r_i: float | None = None,
        r_f: float | None = None,
    ) -> "LinkSpec":
        """Create a circular-cross-section link specification.

        Use ``r`` for a constant radius, or ``r_i`` and ``r_f`` for a radius
        that varies linearly from the base to the tip.

        Args:
            E: Young's modulus of the link material. Unit: pascal ``Pa``.
                Shape: scalar ``()``.
            nu: Poisson's ratio of the link material. Unit: dimensionless.
                Shape: scalar ``()``.
            rho: Mass density of the link material. Unit: ``kg/m^3``. Shape:
                scalar ``()``.
            eta: Material damping coefficient used to form the local damping
                matrix. Unit: ``Pa*s`` (``N*s/m^2``). Shape: scalar ``()``.
            L: Link arclength. Unit: meter ``m``. Shape: scalar ``()``.
            r: Constant circular radius applied at both link ends. Unit: meter
                ``m``. Shape: scalar ``()``. Optional if both ``r_i`` and
                ``r_f`` are provided.
            r_i: Initial circular radius at normalized arclength ``x = 0``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``r`` is
                provided.
            r_f: Final circular radius at normalized arclength ``x = 1``. Unit:
                meter ``m``. Shape: scalar ``()``. Optional if ``r`` is
                provided.

        Returns:
            LinkSpec: Single-link spec with ``cross_section_geometry`` set to
            ``CrossSectionGeometry.CIRCULAR``.

        Raises:
            ValueError: If neither ``r`` nor both endpoint radii are provided.
        """
        if r is not None:
            r_i = r if r_i is None else r_i
            r_f = r if r_f is None else r_f
        if r_i is None or r_f is None:
            raise ValueError("Circular LinkSpec requires r or both r_i and r_f.")
        return cls(
            cross_section_geometry=CrossSectionGeometry.CIRCULAR,
            E=E,
            nu=nu,
            rho=rho,
            eta=eta,
            L=L,
            r_i=r_i,
            r_f=r_f,
        )

    @classmethod
    def rectangular(
        cls,
        *,
        E: float,
        nu: float,
        rho: float,
        eta: float,
        L: float,
        h: float | None = None,
        w: float | None = None,
        h_i: float | None = None,
        h_f: float | None = None,
        w_i: float | None = None,
        w_f: float | None = None,
    ) -> "LinkSpec":
        """Create a rectangular-cross-section link specification.

        Use ``h`` and ``w`` for constant height and width, or use endpoint
        values for dimensions that vary linearly from the base to the tip.

        Args:
            E: Young's modulus of the link material. Unit: pascal ``Pa``.
                Shape: scalar ``()``.
            nu: Poisson's ratio of the link material. Unit: dimensionless.
                Shape: scalar ``()``.
            rho: Mass density of the link material. Unit: ``kg/m^3``. Shape:
                scalar ``()``.
            eta: Material damping coefficient used to form the local damping
                matrix. Unit: ``Pa*s`` (``N*s/m^2``). Shape: scalar ``()``.
            L: Link arclength. Unit: meter ``m``. Shape: scalar ``()``.
            h: Constant rectangular height applied at both link ends. Unit:
                meter ``m``. Shape: scalar ``()``. Optional if ``h_i`` and
                ``h_f`` are provided.
            w: Constant rectangular width applied at both link ends. Unit:
                meter ``m``. Shape: scalar ``()``. Optional if ``w_i`` and
                ``w_f`` are provided.
            h_i: Initial rectangular height at normalized arclength ``x = 0``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``h`` is
                provided.
            h_f: Final rectangular height at normalized arclength ``x = 1``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``h`` is
                provided.
            w_i: Initial rectangular width at normalized arclength ``x = 0``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``w`` is
                provided.
            w_f: Final rectangular width at normalized arclength ``x = 1``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``w`` is
                provided.

        Returns:
            LinkSpec: Single-link spec with ``cross_section_geometry`` set to
            ``CrossSectionGeometry.RECTANGULAR``.

        Raises:
            ValueError: If neither constant dimensions nor all endpoint
                dimensions are provided.
        """
        if h is not None:
            h_i = h if h_i is None else h_i
            h_f = h if h_f is None else h_f
        if w is not None:
            w_i = w if w_i is None else w_i
            w_f = w if w_f is None else w_f
        if h_i is None or h_f is None or w_i is None or w_f is None:
            raise ValueError(
                "Rectangular LinkSpec requires h/w or h_i, h_f, w_i, and w_f."
            )
        return cls(
            cross_section_geometry=CrossSectionGeometry.RECTANGULAR,
            E=E,
            nu=nu,
            rho=rho,
            eta=eta,
            L=L,
            h_i=h_i,
            h_f=h_f,
            w_i=w_i,
            w_f=w_f,
        )

    @classmethod
    def elliptical(
        cls,
        *,
        E: float,
        nu: float,
        rho: float,
        eta: float,
        L: float,
        a: float | None = None,
        b: float | None = None,
        a_i: float | None = None,
        a_f: float | None = None,
        b_i: float | None = None,
        b_f: float | None = None,
    ) -> "LinkSpec":
        """Create an elliptical-cross-section link specification.

        Use ``a`` and ``b`` for constant semi-major and semi-minor axes, or use
        endpoint values for axes that vary linearly from the base to the tip.

        Args:
            E: Young's modulus of the link material. Unit: pascal ``Pa``.
                Shape: scalar ``()``.
            nu: Poisson's ratio of the link material. Unit: dimensionless.
                Shape: scalar ``()``.
            rho: Mass density of the link material. Unit: ``kg/m^3``. Shape:
                scalar ``()``.
            eta: Material damping coefficient used to form the local damping
                matrix. Unit: ``Pa*s`` (``N*s/m^2``). Shape: scalar ``()``.
            L: Link arclength. Unit: meter ``m``. Shape: scalar ``()``.
            a: Constant semi-major axis applied at both link ends. Unit: meter
                ``m``. Shape: scalar ``()``. Optional if ``a_i`` and ``a_f``
                are provided.
            b: Constant semi-minor axis applied at both link ends. Unit: meter
                ``m``. Shape: scalar ``()``. Optional if ``b_i`` and ``b_f``
                are provided.
            a_i: Initial semi-major axis at normalized arclength ``x = 0``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``a`` is
                provided.
            a_f: Final semi-major axis at normalized arclength ``x = 1``. Unit:
                meter ``m``. Shape: scalar ``()``. Optional if ``a`` is
                provided.
            b_i: Initial semi-minor axis at normalized arclength ``x = 0``.
                Unit: meter ``m``. Shape: scalar ``()``. Optional if ``b`` is
                provided.
            b_f: Final semi-minor axis at normalized arclength ``x = 1``. Unit:
                meter ``m``. Shape: scalar ``()``. Optional if ``b`` is
                provided.

        Returns:
            LinkSpec: Single-link spec with ``cross_section_geometry`` set to
            ``CrossSectionGeometry.ELLIPTICAL``.

        Raises:
            ValueError: If neither constant axes nor all endpoint axes are
                provided.
        """
        if a is not None:
            a_i = a if a_i is None else a_i
            a_f = a if a_f is None else a_f
        if b is not None:
            b_i = b if b_i is None else b_i
            b_f = b if b_f is None else b_f
        if a_i is None or a_f is None or b_i is None or b_f is None:
            raise ValueError(
                "Elliptical LinkSpec requires a/b or a_i, a_f, b_i, and b_f."
            )
        return cls(
            cross_section_geometry=CrossSectionGeometry.ELLIPTICAL,
            E=E,
            nu=nu,
            rho=rho,
            eta=eta,
            L=L,
            a_i=a_i,
            a_f=a_f,
            b_i=b_i,
            b_f=b_f,
        )


@dataclass
class JointSpec:
    """Kinematic and stiffness specification of the joint preceding a GVS link.

    The joint contributes the rigid joint coordinates before the soft link in a
    segment. Joint strain bases use the spatial twist ordering
    ``[omega_x, omega_y, omega_z, v_x, v_y, v_z]``.

    Attributes:
        type: Joint family name. Unit: dimensionless categorical value. Shape:
            scalar ``()``. Supported values are ``"revolute"``,
            ``"prismatic"``, ``"helical"``, ``"cylindrical"``, ``"planar"``,
            ``"spherical"``, ``"free"``, and ``"fixed"``.
        axis: Axis used by revolute, prismatic, helical, and cylindrical
            joints. Unit: dimensionless categorical value. Shape: scalar ``()``.
            Valid values are ``"x"``, ``"y"``, and ``"z"``.
        plane: Motion plane used by planar joints. Unit: dimensionless
            categorical value. Shape: scalar ``()``. Valid values are ``"xy"``,
            ``"yz"``, and ``"xz"``.
        pitch: Translation per unit angular coordinate for helical joints. Unit:
            meter per radian ``m/rad``. Shape: scalar ``()``. Ignored by
            non-helical joints.
        stiffness: Joint stiffness matrix for the active joint coordinates.
            Unit: generalized joint effort per generalized joint coordinate
            (for example ``N*m/rad`` for rotational coordinates and ``N/m`` for
            translational coordinates). Shape: ``(joint_dof, joint_dof)``,
            where ``joint_dof`` is 0 for fixed, 1 for revolute/prismatic/
            helical, 2 for cylindrical, 3 for planar/spherical, and 6 for free
            joints. An empty list is accepted and interpreted as zero stiffness.
    """

    type: JointType
    axis: Literal["x", "y", "z"] = "x"
    plane: Literal["xy", "yz", "xz"] = "xy"
    pitch: float = 0.0
    stiffness: Array | list = field(default_factory=list)

    @classmethod
    def fixed(cls) -> "JointSpec":
        """Create a fixed joint specification.

        The joint has no generalized coordinates and contributes no joint
        motion before the link.

        Returns:
            JointSpec: Fixed-joint spec with zero joint DOFs. The implied
            stiffness has unit not applicable and shape ``(0, 0)``.
        """
        return cls(type="fixed")

    @classmethod
    def revolute(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
    ) -> "JointSpec":
        """Create a revolute joint specification.

        Args:
            axis: Axis of rotation. Unit: dimensionless categorical value.
                Shape: scalar ``()``. Valid values are ``"x"``, ``"y"``, and
                ``"z"``.
            stiffness: Rotational stiffness matrix for the revolute coordinate.
                Unit: ``N*m/rad``. Shape: ``(1, 1)``. If omitted or ``None``,
                zero stiffness is used.

        Returns:
            JointSpec: Revolute-joint spec with one angular coordinate in
            radians about ``axis``.
        """
        return cls(
            type="revolute", axis=axis, stiffness=[] if stiffness is None else stiffness
        )

    @classmethod
    def prismatic(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
    ) -> "JointSpec":
        """Create a prismatic joint specification.

        Args:
            axis: Axis of translation. Unit: dimensionless categorical value.
                Shape: scalar ``()``. Valid values are ``"x"``, ``"y"``, and
                ``"z"``.
            stiffness: Translational stiffness matrix for the prismatic
                coordinate. Unit: ``N/m``. Shape: ``(1, 1)``. If omitted or
                ``None``, zero stiffness is used.

        Returns:
            JointSpec: Prismatic-joint spec with one translational coordinate
            in meters along ``axis``.
        """
        return cls(
            type="prismatic",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def helical(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        pitch: float = 0.0,
        stiffness: Array | list | None = None,
    ) -> "JointSpec":
        """Create a helical joint specification.

        Args:
            axis: Screw axis of the helical joint. Unit: dimensionless
                categorical value. Shape: scalar ``()``. Valid values are
                ``"x"``, ``"y"``, and ``"z"``.
            pitch: Translation per unit angular coordinate along ``axis``.
                Unit: meter per radian ``m/rad``. Shape: scalar ``()``.
            stiffness: Generalized stiffness matrix for the helical coordinate.
                Unit: generalized screw effort per radian. Shape: ``(1, 1)``.
                If omitted or ``None``, zero stiffness is used.

        Returns:
            JointSpec: Helical-joint spec with one screw coordinate in radians.
        """
        return cls(
            type="helical",
            axis=axis,
            pitch=pitch,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def cylindrical(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
    ) -> "JointSpec":
        """Create a cylindrical joint specification.

        Args:
            axis: Shared rotation and translation axis. Unit: dimensionless
                categorical value. Shape: scalar ``()``. Valid values are
                ``"x"``, ``"y"``, and ``"z"``.
            stiffness: Generalized stiffness matrix for coordinates
                ``[rotation_about_axis, translation_along_axis]``. Units are
                mixed by coordinate, with diagonal examples ``N*m/rad`` and
                ``N/m``. Shape: ``(2, 2)``. If omitted or ``None``, zero
                stiffness is used.

        Returns:
            JointSpec: Cylindrical-joint spec with one angular coordinate in
            radians and one translational coordinate in meters.
        """
        return cls(
            type="cylindrical",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def planar(
        cls,
        plane: Literal["xy", "yz", "xz"] = "xy",
        stiffness: Array | list | None = None,
    ) -> "JointSpec":
        """Create a planar joint specification.

        Args:
            plane: Plane in which the joint translates and about whose normal it
                rotates. Unit: dimensionless categorical value. Shape: scalar
                ``()``. Valid values are ``"xy"``, ``"yz"``, and ``"xz"``.
            stiffness: Generalized stiffness matrix for coordinates
                ``[rotation_normal_to_plane, translation_1, translation_2]``.
                Units are mixed by coordinate, with diagonal examples
                ``N*m/rad`` and ``N/m``. Shape: ``(3, 3)``. If omitted or
                ``None``, zero stiffness is used.

        Returns:
            JointSpec: Planar-joint spec with one angular coordinate in radians
            and two translational coordinates in meters.
        """
        return cls(
            type="planar",
            plane=plane,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def spherical(cls, stiffness: Array | list | None = None) -> "JointSpec":
        """Create a spherical joint specification.

        Args:
            stiffness: Rotational stiffness matrix for the three angular
                coordinates ``[rotation_x, rotation_y, rotation_z]``. Unit:
                ``N*m/rad``. Shape: ``(3, 3)``. If omitted or ``None``, zero
                stiffness is used.

        Returns:
            JointSpec: Spherical-joint spec with three angular coordinates in
            radians.
        """
        return cls(type="spherical", stiffness=[] if stiffness is None else stiffness)

    @classmethod
    def free(cls, stiffness: Array | list | None = None) -> "JointSpec":
        """Create a free joint specification.

        Args:
            stiffness: Generalized stiffness matrix for coordinates
                ``[rotation_x, rotation_y, rotation_z, translation_x,
                translation_y, translation_z]``. Units are mixed by coordinate,
                with diagonal examples ``N*m/rad`` and ``N/m``. Shape:
                ``(6, 6)``. If omitted or ``None``, zero stiffness is used.

        Returns:
            JointSpec: Free-joint spec with three angular coordinates in
            radians and three translational coordinates in meters.
        """
        return cls(type="free", stiffness=[] if stiffness is None else stiffness)

    def __post_init__(self) -> None:
        self.type = self.type.lower()


@dataclass
class StrainBasisSpec:
    """Strain-basis parametrization for one GVS link.

    The basis maps link generalized coordinates to the six spatial strain
    components ordered as ``[kappa_x, kappa_y, kappa_z, sigma_x, sigma_y,
    sigma_z]``.

    Attributes:
        type: Strain basis family name. Unit: dimensionless categorical value.
            Shape: scalar ``()``. Supported values are ``"monomial"``,
            ``"legendre"``, ``"chebyshev"``, ``"fourier"``, ``"gaussian"``,
            and ``"imq"``.
        active: Active strain-component selector. Unit: dimensionless mask.
            Shape: ``(6,)`` after initialization. It may be passed either as a
            six-entry mask in the strain order above or as a tuple/list of
            component names such as ``("kappa_y", "sigma_x")``.
        orders: Basis order for each strain component. Unit: dimensionless
            integer order. Shape: scalar ``()`` when passed as an ``int`` (then
            broadcast to all six components) or ``(6,)`` when passed as an
            array/list/tuple. Stored shape after initialization is ``(6,)``.
        xi_ref: Reference strain vector in the local link frame, ordered as
            ``[kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]``. Units:
            angular strain entries ``kappa_*`` are ``rad/m``; translational
            strain entries ``sigma_*`` are dimensionless stretch/shear values,
            with ``sigma_x = 1`` representing a straight unstretched link along
            the local x-axis. Shape: ``(6,)``.
    """

    type: BasisType
    active: Array | list | tuple[StrainComponent, ...]
    orders: int | Array | list | tuple
    xi_ref: Array | list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        self.type = self.type.lower()
        self.active = _component_mask(self.active)
        self.orders = _orders_vector(self.orders)


@dataclass
class GVSSegment:
    """Complete user-facing specification of one GVS segment.

    A segment consists of one joint followed by one soft GVS link. The segment
    spec is the public construction unit consumed by ``GVS.from_segments`` and
    ``GVS.params_from_segments``.

    Attributes:
        link: Link geometry, material, and length specification. Unit: not
            applicable; nested ``LinkSpec`` object. Shape: scalar object ``()``.
        joint: Specification of the joint preceding the link. Unit: not
            applicable; nested ``JointSpec`` object. Shape: scalar object ``()``.
        basis: Link strain-basis and reference-strain specification. Unit: not
            applicable; nested ``StrainBasisSpec`` object. Shape: scalar object
            ``()``.
        num_gauss_points: Number of interior Gauss-Legendre quadrature points
            requested for this link. Unit: points. Shape: scalar ``()``. The
            runtime integration arrays include two additional zero-weight
            boundary nodes, so their unpadded length is
            ``num_gauss_points + 2``. Expected value is at least 5.
    """

    link: LinkSpec
    joint: JointSpec
    basis: StrainBasisSpec
    num_gauss_points: int
