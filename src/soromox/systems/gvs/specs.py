"""GVS-specific segment and strain-basis construction specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jax import Array

from soromox.systems.components import JointSpec, LinkSpec

__all__ = ["GVSSegment", "StrainBasisSpec"]

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

_COMPONENT_INDEX = {
    "kappa_x": 0,
    "kappa_y": 1,
    "kappa_z": 2,
    "sigma_x": 3,
    "sigma_y": 4,
    "sigma_z": 5,
}


def _component_mask(
    selector: Array | list | tuple[StrainComponent, ...],
) -> Array | list:
    if isinstance(selector, (list, tuple)) and all(
        isinstance(item, str) for item in selector
    ):
        mask = [0, 0, 0, 0, 0, 0]
        for component in selector:
            try:
                mask[_COMPONENT_INDEX[component]] = 1
            except KeyError as exc:
                raise ValueError(f"Unknown strain component {component!r}.") from exc
        return mask
    return selector


def _orders_vector(order: int | Array | list | tuple) -> Array | list:
    return [order] * 6 if isinstance(order, int) else order


@dataclass
class StrainBasisSpec:
    """Describe the variable-strain basis used by one GVS link.

    String-valued strain selectors are converted to the six-component mask used
    internally by GVS. A scalar basis order is replicated for all six strain
    components, including inactive components.

    Attributes:
        type: Basis family. Supported values are ``"monomial"``, ``"legendre"``,
            ``"chebyshev"``, ``"fourier"``, ``"gaussian"``, and ``"imq"``.
        strain_selector: Active strain components, supplied either as a
            six-element numeric mask or as a sequence of component names. The
            supported names are ``kappa_x``, ``kappa_y``, ``kappa_z``,
            ``sigma_x``, ``sigma_y``, and ``sigma_z``.
        basis_order: Basis order supplied as one integer for every strain
            component or as a six-element array-like value.
    """

    type: BasisType
    strain_selector: Array | list | tuple[StrainComponent, ...]
    basis_order: int | Array | list | tuple

    def __post_init__(self) -> None:
        self.type = self.type.lower()  # type: ignore[assignment]
        self.strain_selector = _component_mask(self.strain_selector)
        self.basis_order = _orders_vector(self.basis_order)


@dataclass
class GVSSegment:
    """Bundle all construction specifications for one GVS segment.

    A segment consists of a discrete joint followed by a variable-strain
    continuum link. The object is a construction specification only; calling
    :meth:`GVS.from_segments <soromox.systems.gvs.GVS.from_segments>` converts
    its values to canonical runtime parameters and static GVS structure.

    Attributes:
        link: Continuum-link geometry, reference strain, and either isotropic
            material properties or explicit generalized link matrices.
        joint: Joint type, kinematics, and generalized joint stiffness and
            damping in active joint coordinates.
        basis: Variable-strain basis used to parameterize the link strain.
        num_gauss_points: Number of Gauss points used for link quadrature. GVS
            requires at least five points per segment.
    """

    link: LinkSpec
    joint: JointSpec
    basis: StrainBasisSpec
    num_gauss_points: int
