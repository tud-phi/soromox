"""Shear and torsion assumptions for isotropic GVS rod constitutive laws."""

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import _contains_tracer

from .cross_sections import (
    CrossSectionGeometry,
    section_properties,
    section_torsion_constant,
)

__all__ = ["RodConstitutiveLaw", "TransverseShearModel", "RectangularTorsionModel"]


class TransverseShearModel(Enum):
    """How transverse shear resultants relate to the rod's shear strains."""

    UNIFORM = "uniform"
    EFFECTIVE_AREA = "effective_area"


class RectangularTorsionModel(Enum):
    """Approximation to free-warping torsion of rectangular sections."""

    SAINT_VENANT_SERIES = "saint_venant_series"
    MARSHALL = "marshall"


def _rectangular_shear_areas(outer: Array, inner: Array | None) -> Array:
    """Integrate Jourawski shear energy exactly on each constant-breadth strip."""
    # Solve both directions together: depth along the applied shear, breadth
    # perpendicular to it. The last axis orders shear along material y and z.
    depth, breadth = outer, outer[..., ::-1]
    inner_depth = jnp.zeros_like(depth) if inner is None else inner
    inner_breadth = inner_depth[..., ::-1]
    half, inner_half = depth / 2, inner_depth / 2
    # Three-point Gauss integration is exact for the quartic Q(u)^2 on each
    # strip. Factored differences avoid cancellation in thin outer strips.
    nodes = jnp.array([-jnp.sqrt(3 / 5), 0.0, jnp.sqrt(3 / 5)])
    weights = jnp.array([5 / 9, 8 / 9, 5 / 9])
    central_u = inner_half[..., None] * (1 + nodes) / 2
    central_breadth = breadth - inner_breadth
    central_q = (
        central_breadth[..., None]
        * (inner_half[..., None] - central_u)
        * (inner_half[..., None] + central_u)
        + (breadth * (half - inner_half) * (half + inner_half))[..., None]
    ) / 2
    central_energy = (
        inner_half * jnp.sum(weights * central_q**2, axis=-1) / central_breadth
    )
    outer_u = inner_half[..., None] + (half - inner_half)[..., None] * (1 + nodes) / 2
    outer_q = (
        breadth[..., None]
        * (half[..., None] - outer_u)
        * (half[..., None] + outer_u)
        / 2
    )
    outer_energy = (
        (half - inner_half) * jnp.sum(weights * outer_q**2, axis=-1) / breadth
    )
    moment = (breadth * depth**3 - inner_breadth * inner_depth**3) / 12
    # Integration covers both halves: 2 cancels the affine quadrature's 1/2.
    return moment**2 / (central_energy + outer_energy)


@dataclass(frozen=True)
class RodConstitutiveLaw:
    """Independent sectional assumptions for an isotropic GVS rod.

    Attributes:
        transverse_shear: Uniform shear (solid sections only), or effective
            areas from Cowper's circular formula and rectangular Jourawski
            shear energy. Effective areas do not currently support ellipses.
        rectangular_torsion: Saint-Venant series for solid rectangles, or
            Marshall's approximation for both solid and hollow rectangles.
            Circular torsion is exact for either choice; solid ellipses use
            their exact Saint-Venant constant.

    Axial and bending stiffness always use EA, EIy and EIz. Rotational mass
    inertia always uses area moments, independently of these choices. The
    defaults preserve the existing solid-link stiffness and damping.
    """

    transverse_shear: TransverseShearModel = TransverseShearModel.UNIFORM
    rectangular_torsion: RectangularTorsionModel = (
        RectangularTorsionModel.SAINT_VENANT_SERIES
    )

    def __post_init__(self) -> None:
        if not isinstance(self.transverse_shear, TransverseShearModel):
            raise TypeError("transverse_shear must be a TransverseShearModel.")
        if not isinstance(self.rectangular_torsion, RectangularTorsionModel):
            raise TypeError("rectangular_torsion must be a RectangularTorsionModel.")

    def validate_geometry(self, geometry: CrossSectionGeometry, hollow: bool) -> None:
        """Reject unsupported geometry/law combinations at construction and update.

        Args:
            geometry: Cross-section family.
            hollow: Whether the section has a cavity.

        Returns:
            None after successful validation.

        Raises:
            ValueError: If the selected models do not support this geometry.
        """
        if hollow and self.transverse_shear == TransverseShearModel.UNIFORM:
            raise ValueError(
                "UNIFORM transverse shear requires solid geometry; choose EFFECTIVE_AREA for hollow links."
            )
        if (
            self.transverse_shear == TransverseShearModel.EFFECTIVE_AREA
            and geometry == CrossSectionGeometry.ELLIPTICAL
        ):
            raise ValueError(
                "EFFECTIVE_AREA currently supports circular and rectangular sections."
            )
        if (
            hollow
            and geometry == CrossSectionGeometry.RECTANGULAR
            and self.rectangular_torsion != RectangularTorsionModel.MARSHALL
        ):
            raise ValueError("Hollow rectangular links require MARSHALL torsion.")

    def requires_poisson_ratio(self, geometry: CrossSectionGeometry) -> bool:
        """Check whether shear areas depend on the elastic Poisson ratio.

        Args:
            geometry: Cross-section family.

        Returns:
            True for circular sections using effective shear areas.
        """
        return (
            geometry == CrossSectionGeometry.CIRCULAR
            and self.transverse_shear == TransverseShearModel.EFFECTIVE_AREA
        )

    def shear_areas(
        self,
        geometry: CrossSectionGeometry,
        outer: Array,
        inner: Array | None = None,
        *,
        poisson_ratio: Array | float = 0.0,
    ) -> Array:
        """Return effective areas along material y/z, in square metres.

        Args:
            geometry: Section family.
            outer: Outer dimensions in the last axis, in metres.
            inner: Optional centred cavity dimensions, in metres.
            poisson_ratio: Elastic Poisson ratio; only used for Cowper shear.

        Returns:
            Two effective shear areas per section. Rectangular areas integrate
            Q²/b over each strip, assuming Jourawski shear stress VQ/(Ib).
            This approximation neglects transverse stress redistribution.
        """
        self.validate_geometry(geometry, inner is not None)
        outer = jnp.asarray(outer)
        area = section_properties(geometry, outer, inner)[3]
        if self.transverse_shear == TransverseShearModel.UNIFORM:
            return jnp.stack((area, area), axis=-1)
        if geometry == CrossSectionGeometry.RECTANGULAR:
            return _rectangular_shear_areas(
                outer, None if inner is None else jnp.asarray(inner)
            )
        nu = jnp.asarray(poisson_ratio)
        if not _contains_tracer(nu) and not bool(
            jnp.all(jnp.isfinite(nu) & (nu > -1) & (nu <= 0.5))
        ):
            raise ValueError(
                "Cowper shear requires -1 < poisson_ratio <= 0.5, including when derived from E and G."
            )
        ratio2 = (
            0.0 if inner is None else (jnp.asarray(inner)[..., 0] / outer[..., 0]) ** 2
        )
        factor = (
            6
            * (1 + nu)
            * (1 + ratio2) ** 2
            / ((7 + 6 * nu) * (1 + ratio2) ** 2 + (20 + 12 * nu) * ratio2)
        )
        return jnp.stack((area * factor, area * factor), axis=-1)

    def torsion_constant(
        self, geometry: CrossSectionGeometry, outer: Array, inner: Array | None = None
    ) -> Array:
        """Return the torsion constant J in metres to the fourth power.

        Args:
            geometry: Section family.
            outer: Outer dimensions in the last axis, in metres.
            inner: Optional centred uniform-wall cavity dimensions, in metres.

        Returns:
            J for use in GJ. Marshall adds Pm*t³/3 to Bredt's 4*Am²*t/Pm.
            Its solid extension evaluates the same expression at
            t=min(height,width)/2, ensuring continuity as the cavity closes.
        """
        self.validate_geometry(geometry, inner is not None)
        outer = jnp.asarray(outer)
        if (
            geometry != CrossSectionGeometry.RECTANGULAR
            or self.rectangular_torsion == RectangularTorsionModel.SAINT_VENANT_SERIES
        ):
            return section_torsion_constant(geometry, outer, inner)
        thickness = (
            jnp.min(outer, axis=-1) / 2
            if inner is None
            else (outer[..., 0] - jnp.asarray(inner)[..., 0]) / 2
        )
        mid = outer - thickness[..., None]
        perimeter = 2 * jnp.sum(mid, axis=-1)
        mid_area = jnp.prod(mid, axis=-1)
        return 4 * thickness * mid_area**2 / perimeter + perimeter * thickness**3 / 3
