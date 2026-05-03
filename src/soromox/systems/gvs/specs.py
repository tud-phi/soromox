__all__ = ["GVSSegment", "LinkSpec", "JointSpec", "StrainBasisSpec"]

from dataclasses import dataclass, field
from typing import Literal
from jax import Array

from soromox.systems.soft_robot import CrossSectionGeometry


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
    """Geometric and material specification of one GVS link."""

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
    """Kinematic and stiffness specification of the joint preceding a GVS link."""

    type: JointType
    axis: Literal["x", "y", "z"] = "x"
    plane: Literal["xy", "yz", "xz"] = "xy"
    pitch: float = 0.0
    stiffness: Array | list = field(default_factory=list)

    @classmethod
    def fixed(cls) -> "JointSpec":
        return cls(type="fixed")

    @classmethod
    def revolute(
        cls, axis: Literal["x", "y", "z"] = "x", stiffness=None
    ) -> "JointSpec":
        return cls(
            type="revolute", axis=axis, stiffness=[] if stiffness is None else stiffness
        )

    @classmethod
    def prismatic(
        cls, axis: Literal["x", "y", "z"] = "x", stiffness=None
    ) -> "JointSpec":
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
        stiffness=None,
    ) -> "JointSpec":
        return cls(
            type="helical",
            axis=axis,
            pitch=pitch,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def cylindrical(
        cls, axis: Literal["x", "y", "z"] = "x", stiffness=None
    ) -> "JointSpec":
        return cls(
            type="cylindrical",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def planar(
        cls, plane: Literal["xy", "yz", "xz"] = "xy", stiffness=None
    ) -> "JointSpec":
        return cls(
            type="planar",
            plane=plane,
            stiffness=[] if stiffness is None else stiffness,
        )

    @classmethod
    def spherical(cls, stiffness=None) -> "JointSpec":
        return cls(type="spherical", stiffness=[] if stiffness is None else stiffness)

    @classmethod
    def free(cls, stiffness=None) -> "JointSpec":
        return cls(type="free", stiffness=[] if stiffness is None else stiffness)

    def __post_init__(self) -> None:
        self.type = self.type.lower()


@dataclass
class StrainBasisSpec:
    """Strain-basis parametrization for one GVS link."""

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
    """Complete user-facing specification of one GVS segment."""

    link: LinkSpec
    joint: JointSpec
    basis: StrainBasisSpec
    num_gauss_points: int
