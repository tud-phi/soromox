__all__ = [
    "PCSStructure",
    "PlanarPCSStructure",
    "GVSLinkStructure",
    "GVSJointStructure",
    "GVSStrainBasisStructure",
    "GVSSegmentStructure",
    "GVSStructure",
    "PlanarHSAStructure",
]

from typing import Any

import equinox as eqx
from jax import Array

from soromox.systems.soft_robot import CrossSectionGeometry


class PCSStructure(eqx.Module):
    """Static PCS layout that determines JAX compilation structure."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarPCSStructure(eqx.Module):
    """Static planar PCS layout."""

    num_gauss_points: int = eqx.field(static=True, default=5)
    strain_selector: Array | None = None
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class GVSLinkStructure(eqx.Module):
    """Static link choices for one GVS segment.

    This stores only choices that affect compilation or dispatch. Numeric link
    values such as length, material constants, and cross-section dimensions live
    in ``GVSParams.link``.
    """

    cross_section_geometry: CrossSectionGeometry = eqx.field(static=True)


class GVSJointStructure(eqx.Module):
    """Static joint choices for one GVS segment.

    Joint stiffness is dynamic and lives in ``GVSParams.joint_stiffness``.
    """

    type: str = eqx.field(static=True)
    axis: str = eqx.field(static=True, default="x")
    plane: str = eqx.field(static=True, default="xy")
    pitch: float = eqx.field(static=True, default=0.0)


class GVSStrainBasisStructure(eqx.Module):
    """Static strain-basis choices for one GVS segment.

    Reference strain is dynamic and lives in ``GVSParams.reference_strain``.
    """

    type: str = eqx.field(static=True)
    active: tuple[Any, ...] = eqx.field(static=True)
    orders: tuple[Any, ...] = eqx.field(static=True)


class GVSSegmentStructure(eqx.Module):
    """Static structure for one GVS segment."""

    link: GVSLinkStructure = eqx.field(static=True)
    joint: GVSJointStructure = eqx.field(static=True)
    basis: GVSStrainBasisStructure = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)


class GVSStructure(eqx.Module):
    """Static GVS segment structure and padded layout choices.

    ``segments`` stores stripped static segment structures: joint families, basis
    families/orders/active masks, quadrature counts, and cross-section families.
    Dynamic numeric values live in ``GVSParams``.
    """

    segments: tuple[GVSSegmentStructure, ...] = eqx.field(static=True)
    max_dof: int | None = eqx.field(static=True, default=None)
    max_num_gauss_points: int | None = eqx.field(static=True, default=None)
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)


class PlanarHSAStructure(eqx.Module):
    """Static symbolic and layout choices for planar HSA."""

    symbolic_expression_path: str = eqx.field(static=True)
    strain_selector: Array | None = None
    consider_underactuation: bool = eqx.field(static=True, default=True)
    consider_hysteresis: bool = eqx.field(static=True, default=False)
    eps: float = eqx.field(static=True, default=1e-6)
