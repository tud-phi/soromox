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

__all__ = [
    "RodConstitutiveLaw",
    "TransverseShearRodModel",
    "TorsionRodModel",
]


class TransverseShearRodModel(Enum):
    r"""Transverse-shear assumptions in an isotropic rod constitutive law.

    ``UNIFORM`` uses the material area, $A_{s,y}=A_{s,z}=A$, and supports
    solid sections only. ``EFFECTIVE_AREA`` uses Cowper's expression for
    circles and annuli, and Jourawski shear-energy matching for rectangles.
    Elliptical sections use ``UNIFORM`` shear.

    For circles, with $m=r/R$ and elastic Poisson ratio $\nu$,

    $$
    A_s=kA,\qquad
    k=\frac{6(1+\nu)(1+m^2)^2}
            {(7+6\nu)(1+m^2)^2+(20+12\nu)m^2}.
    $$

    This expression applies across wall thicknesses. The solid limit is
    $k=6(1+\nu)/(7+6\nu)$; the thin-wall limit is
    $k=2(1+\nu)/(4+3\nu)$. The material supplies $\nu$ directly or through
    $E/(2G)-1$, and the same factor enters the scalar damping model.

    For rectangles, $A_s=I^2/\int Q(u)^2/b(u)\,du$ matches the energy of
    the assumed stress $\tau=VQ/(Ib)$. Here $b$ is the material breadth at
    a cut, $Q$ its first area moment, and $I$ the bending moment of area
    for the chosen shear direction. This recovers $5A/6$ for solids and
    continuously as a cavity closes. The assumed shear field uses the mean
    stress over the material breadth at each cut. These effective areas give
    solid shear stiffnesses of $kGA$ for circles and $5GA/6$ for rectangles.

    References:
        Cowper, G. R. (1966). The Shear Coefficient in Timoshenko's Beam
        Theory. *Journal of Applied Mechanics*, **33**(2), 335–340.
        [doi:10.1115/1.3625046](https://doi.org/10.1115/1.3625046).
        Source of the circular and annular shear factors.
    """

    UNIFORM = "uniform"
    EFFECTIVE_AREA = "effective_area"


class TorsionRodModel(Enum):
    r"""Torsion assumptions in an isotropic rod constitutive law.

    ``POLAR_MOMENT`` uses $J=I_y+I_z$ for solid sections. It assumes zero
    cross-sectional warping and is exact for circular solids. Hollow sections
    require ``SAINT_VENANT``.

    ``SAINT_VENANT`` accounts for free warping through geometry-specific
    torsion constants: exact circular/annular and solid elliptical formulas,
    and Marshall's approximation for rectangles. Marshall supports centred,
    uniform-wall tubes with sharp corners and a continuous solid extension:

    $$
    J=\frac{4tA_m^2}{P_m}+\frac{P_mt^3}{3}.
    $$

    $A_m$ is the area enclosed by the wall midline and $P_m$ its perimeter.
    The first term is Bredt's thin-wall expression. The relative correction,
    $P_m^2t^2/(12A_m^2)$, vanishes quadratically for thin walls. At square-tube
    wall/outer-side ratios of 1%, 5%, and 10%, the correction is 0.0136%,
    0.3693%, and 1.6461%, respectively.

    For a solid, Marshall is evaluated at $t=\min(h,w)/2$. This engineering
    extension is continuous with the tube formula. Its relative errors against
    the exact solid solution are +3.739% for a square, and approximately -2.065%,
    -1.882%, and -1.078% for side ratios 2, 5, and 10. A nonsquare uniform-wall
    cavity closes to a line; Marshall's solid endpoint is an engineering
    approximation along this path. The model assumes free warping and stable
    cross-sectional walls.

    For ``SAINT_VENANT``, circular torsion is exact for solids and annuli,
    $J=\pi(R^4-r^4)/2$. Solid ellipses use $J=\pi a^3b^3/(a^2+b^2)$.

    References:
        Ridley-Ellis, D. J., Owen, J. S., & Davies, G. (2003). Torsional
        behaviour of rectangular hollow sections. *Journal of Constructional
        Steel Research*, **59**(5), 641–663.
        [doi:10.1016/S0143-974X(02)00060-3](https://doi.org/10.1016/S0143-974X(02)00060-3).
        Equation (1) gives Marshall's approximation used here for tubes.
    """

    POLAR_MOMENT = "polar_moment"
    SAINT_VENANT = "saint_venant"


def _rectangular_shear_areas(outer: Array, inner: Array | None) -> Array:
    r"""Compute rectangular shear areas by matching Jourawski stress energy.

    Args:
        outer: Outer height and width in the last axis, spanning material y
            and z, in metres. Leading dimensions may batch several sections.
        inner: Centred cavity height and width with the same shape as
            ``outer``, or None for solid sections. Callers validate the cavity.

    Returns:
        Effective shear areas along y and z with shape ``(..., 2)``, in m².

    For each direction, the approximation $\tau=VQ/(Ib)$ gives
    $A_s=I^2/\int Q^2/b\,du$. The cavity divides the section into strips
    of constant material breadth $b$. Within each strip, $Q$ is quadratic,
    so three-point Gauss quadrature integrates its squared contribution
    exactly. The integration includes both symmetric halves and solves both
    shear directions together. The result matches the energy of this assumed
    shear field, using the mean stress over the material breadth at each cut.
    """
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
            shear energy. Elliptical sections use uniform shear.
        torsion: Polar area moment for solid sections, or free-warping
            Saint-Venant torsion using Marshall for rectangles and exact
            constants for circular/annular and solid elliptical sections.

    Axial and bending stiffness always use EA, EIy and EIz. Rotational mass
    inertia always uses area moments, independently of these choices. The
    default law uses uniform shear and polar-moment torsion for stiffness and
    damping. Hollow links require EFFECTIVE_AREA shear and SAINT_VENANT torsion.
    """

    transverse_shear: TransverseShearRodModel = TransverseShearRodModel.UNIFORM
    torsion: TorsionRodModel = TorsionRodModel.POLAR_MOMENT

    def __post_init__(self) -> None:
        if not isinstance(self.transverse_shear, TransverseShearRodModel):
            raise TypeError("transverse_shear must be a TransverseShearRodModel.")
        if not isinstance(self.torsion, TorsionRodModel):
            raise TypeError("torsion must be a TorsionRodModel.")

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
        if hollow and self.transverse_shear == TransverseShearRodModel.UNIFORM:
            raise ValueError(
                "UNIFORM transverse shear requires solid geometry; choose EFFECTIVE_AREA for hollow links."
            )
        if (
            self.transverse_shear == TransverseShearRodModel.EFFECTIVE_AREA
            and geometry == CrossSectionGeometry.ELLIPTICAL
        ):
            raise ValueError(
                "EFFECTIVE_AREA currently supports circular and rectangular sections."
            )
        if hollow and self.torsion == TorsionRodModel.POLAR_MOMENT:
            raise ValueError(
                "POLAR_MOMENT torsion requires solid geometry; choose SAINT_VENANT for hollow links."
            )

    def requires_poisson_ratio(self, geometry: CrossSectionGeometry) -> bool:
        """Check whether shear areas depend on the elastic Poisson ratio.

        Args:
            geometry: Cross-section family.

        Returns:
            True for circular sections using effective shear areas.
        """
        return (
            geometry == CrossSectionGeometry.CIRCULAR
            and self.transverse_shear == TransverseShearRodModel.EFFECTIVE_AREA
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
            The assumed stress is averaged over the material breadth at each cut.
        """
        self.validate_geometry(geometry, inner is not None)
        outer = jnp.asarray(outer)
        area = section_properties(geometry, outer, inner)[3]
        if self.transverse_shear == TransverseShearRodModel.UNIFORM:
            return jnp.stack((area, area), axis=-1)
        if geometry == CrossSectionGeometry.RECTANGULAR:
            return _rectangular_shear_areas(
                outer, None if inner is None else jnp.asarray(inner)
            )
        nu = jnp.asarray(poisson_ratio)
        valid_nu = jnp.all(jnp.isfinite(nu) & (nu > -1) & (nu <= 0.5))
        # Operations on a concrete closed-over array can still trace under JIT.
        if not _contains_tracer(valid_nu) and not bool(valid_nu):
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
            J for use in GJ: the polar moment for POLAR_MOMENT, or a
            geometry-specific constant for SAINT_VENANT. For rectangles,
            Marshall adds Pm*t³/3 to Bredt's 4*Am²*t/Pm. Its solid extension
            evaluates the same expression at
            t=min(height,width)/2, ensuring continuity as the cavity closes.
        """
        self.validate_geometry(geometry, inner is not None)
        outer = jnp.asarray(outer)
        if self.torsion == TorsionRodModel.POLAR_MOMENT:
            return section_properties(geometry, outer)[0]
        if geometry != CrossSectionGeometry.RECTANGULAR:
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
