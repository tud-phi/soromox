__all__ = [
    "GVSLinkStructure",
    "GVSJointStructure",
    "GVSStrainBasisStructure",
    "GVSSegmentStructure",
    "GVSStructure",
]

from typing import Any

import equinox as eqx

from soromox.systems.components import CrossSectionGeometry, ProfileType


class GVSLinkStructure(eqx.Module):
    """Static link choices for one GVS segment.

    This stores only choices that affect compilation or dispatch. Numeric link
    values such as length, material constants, and cross-section dimensions live
    in ``GVSParams.link``.

    Attributes:
        cross_section_geometry: Static solid cross-section family.
        cross_section_profile_types: Profile type for each geometry dimension.
        cross_section_profile_parameter_counts: Number of packed coefficients
            consumed by each geometry dimension.
    """

    cross_section_geometry: CrossSectionGeometry = eqx.field(static=True)
    cross_section_profile_types: tuple[ProfileType, ...] = eqx.field(static=True)
    cross_section_profile_parameter_counts: tuple[int, ...] = eqx.field(static=True)


class GVSJointStructure(eqx.Module):
    """Static joint choices for one GVS segment.

    Joint stiffness and damping are dynamic and live in ``GVSParams.joint``.

    Attributes:
        type: Joint family name.
        axis: Material-frame axis for axis-based joints.
        plane: Motion plane for planar joints.
        pitch: Translation per radian for a helical joint.
    """

    type: str = eqx.field(static=True)
    axis: str = eqx.field(static=True, default="x")
    plane: str = eqx.field(static=True, default="xy")
    pitch: float = eqx.field(static=True, default=0.0)


class GVSStrainBasisStructure(eqx.Module):
    """Static strain-basis choices for one GVS segment.

    Reference strain is dynamic and lives in ``GVSParams.link.reference_strain``.

    Attributes:
        type: Variable-strain basis family.
        strain_selector: Six-element active-strain mask.
        basis_order: Six-element basis-order tuple.
    """

    type: str = eqx.field(static=True)
    strain_selector: tuple[Any, ...] = eqx.field(static=True)
    basis_order: tuple[Any, ...] = eqx.field(static=True)


class GVSSegmentStructure(eqx.Module):
    """Static structure for one GVS segment.

    Attributes:
        link: Static link geometry and profile choices.
        joint: Static joint kinematic choices.
        basis: Static link strain-basis choices.
        num_gauss_points: Number of Gauss points used for segment quadrature.
    """

    link: GVSLinkStructure = eqx.field(static=True)
    joint: GVSJointStructure = eqx.field(static=True)
    basis: GVSStrainBasisStructure = eqx.field(static=True)
    num_gauss_points: int = eqx.field(static=True)


class GVSStructure(eqx.Module):
    """Static GVS segment structure and padded layout choices.

    ``segments`` stores stripped static segment structures: joint families, basis
    families/orders/active masks, quadrature counts, and cross-section families.
    Dynamic numeric values live in ``GVSParams``.

    Attributes:
        segments: Non-empty tuple of static segment structures.
        max_dof: Common padded link and joint generalized-coordinate dimension,
            or ``None`` before construction resolves it.
        max_num_gauss_points: Common quadrature padding size, or ``None`` before
            construction resolves it.
        scale_rotational_basis_by_length: Whether rotational strain-basis rows
            are divided by link length.
    """

    segments: tuple[GVSSegmentStructure, ...] = eqx.field(static=True)
    max_dof: int | None = eqx.field(static=True, default=None)
    max_num_gauss_points: int | None = eqx.field(static=True, default=None)
    scale_rotational_basis_by_length: bool = eqx.field(static=True, default=False)
