"""Shared cross-section parameterization and geometric properties."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import BaseSystemParams

__all__ = [
    "CrossSectionGeometry",
    "CrossSectionParams",
    "LinearProfile",
    "ProfileType",
    "evaluate_profile",
    "section_properties",
    "section_inner_dimensions",
    "section_torsion_constant",
    "section_shear_factors",
]


class CrossSectionGeometry(IntEnum):
    """Supported cross-section boundary families."""

    CIRCULAR = 0
    RECTANGULAR = 1
    ELLIPTICAL = 2


ProfileType = Literal["constant", "linear"]


@dataclass(frozen=True)
class LinearProfile:
    """A scalar cross-section dimension varying linearly along a link.

    Attributes:
        base: Dimension at normalized arc length zero.
        tip: Dimension at normalized arc length one.
    """

    base: float
    tip: float

    def __post_init__(self) -> None:
        """Validate the endpoint dimensions.

        Raises:
            ValueError: If either endpoint is non-finite or not strictly
                positive.
        """
        values = jnp.asarray([self.base, self.tip])
        valid = bool(jnp.all(jnp.isfinite(values) & (values > 0.0)))
        if not valid:
            raise ValueError(
                "LinearProfile base and tip must be finite and strictly positive."
            )


class CrossSectionParams(BaseSystemParams):
    """Batched dynamic cross-section coefficients.

    Static geometry, profile kinds, and coefficient names live in each system's
    structure. Rows are zero-padded to the largest coefficient count in the
    model.

    Attributes:
        coefficients: Cross-section coefficients with shape
            ``(num_links, max_num_coefficients)``. Each row contains the
            coefficients required by that link's static geometry and profile
            declarations, followed by zero padding when necessary.
    """

    coefficients: Array

    def __check_init__(self) -> None:
        object.__setattr__(self, "coefficients", jnp.asarray(self.coefficients))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        """Normalize replacement coefficient values to JAX arrays."""
        if name == "coefficients":
            return jnp.asarray(value)
        return value

    def validate_structure(self) -> None:
        """Validate the batched cross-section coefficient array shape.

        Returns:
            None.

        Raises:
            ValueError: If ``coefficients`` is not a nonempty two-dimensional
                array.
        """
        coefficients = jnp.asarray(self.coefficients)
        if coefficients.ndim != 2:
            raise ValueError(
                "cross-section coefficients must have shape "
                "(num_links, max_num_coefficients)."
            )
        if coefficients.shape[0] < 1 or coefficients.shape[1] < 1:
            raise ValueError("cross-section coefficients must be non-empty.")

    def validate_values(self) -> None:
        """Validate eager cross-section coefficient values."""
        coefficients = jnp.asarray(self.coefficients)
        finite_and_nonnegative = bool(
            jnp.all(jnp.isfinite(coefficients) & (coefficients >= 0.0))
        )
        positive_per_link = bool(jnp.all(jnp.any(coefficients > 0.0, axis=1)))
        if not finite_and_nonnegative:
            raise ValueError(
                "cross-section coefficients must be finite and nonnegative."
            )
        if not positive_per_link:
            raise ValueError(
                "cross-section coefficients must contain a positive dimension "
                "for every link."
            )


def evaluate_profile(
    coefficients: Array,
    profile_type: ProfileType,
    normalized_arclength: Array,
) -> Array:
    """Evaluate a cross-section profile at normalized arc-length positions.

    Args:
        coefficients: Coefficients for one profile. A constant profile uses
            the first entry; a linear profile uses ``[base, tip]``.
        profile_type: Either ``"constant"`` or ``"linear"``.
        normalized_arclength: Scalar or array of positions measured from zero
            at the link base to one at the link tip.

    Returns:
        Profile values with the same shape as ``normalized_arclength``.

    Raises:
        ValueError: If ``profile_type`` is not supported.
    """
    coefficients = jnp.asarray(coefficients)
    x = jnp.asarray(normalized_arclength)
    if profile_type == "constant":
        return jnp.broadcast_to(coefficients[0], x.shape)
    if profile_type == "linear":
        return coefficients[0] + x * (coefficients[1] - coefficients[0])
    raise ValueError(f"Unknown cross-section profile type {profile_type!r}.")


def section_properties(
    geometry: CrossSectionGeometry | int,
    dimensions: Array,
    inner_dimensions: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(I_x, I_y, I_z, area)`` for a centred solid or hollow section.

    The material-frame x-axis is longitudinal. Rectangular dimensions are
    ordered ``[height, width]``, spanning local y and z respectively. Elliptical
    dimensions are ordered ``[semi_major, semi_minor]``, also along y and z.

    Args:
        geometry: Cross-section family, supplied as a
            :class:`CrossSectionGeometry` value or its integer value.
        dimensions: Geometry dimensions. Circular sections use ``[radius]``;
            rectangular sections use ``[height, width]``; elliptical sections
            use ``[semi_major, semi_minor]``.
        inner_dimensions: Optional cavity dimensions in the same order. The
            caller validates that the cavity lies inside the outer boundary.

    Returns:
        A tuple ``(I_x, I_y, I_z, area)`` containing the polar second moment
        about the longitudinal material x-axis, the transverse second moments,
        and the section area.

    Raises:
        ValueError: If ``geometry`` does not identify a supported section.
    """
    if inner_dimensions is not None:
        outer = section_properties(geometry, dimensions)
        inner = section_properties(geometry, inner_dimensions)
        return tuple(a - b for a, b in zip(outer, inner, strict=True))
    dimensions = jnp.asarray(dimensions)
    geometry = CrossSectionGeometry(int(geometry))
    if geometry == CrossSectionGeometry.CIRCULAR:
        radius = dimensions[..., 0]
        area = jnp.pi * radius**2
        transverse = jnp.pi * radius**4 / 4.0
        polar = 2.0 * transverse
        return polar, transverse, transverse, area
    if geometry == CrossSectionGeometry.RECTANGULAR:
        height, width = dimensions[..., 0], dimensions[..., 1]
        area = height * width
        i_y = height * width**3 / 12.0
        i_z = width * height**3 / 12.0
        return i_y + i_z, i_y, i_z, area
    semi_major, semi_minor = dimensions[..., 0], dimensions[..., 1]
    area = jnp.pi * semi_major * semi_minor
    i_y = jnp.pi * semi_major * semi_minor**3 / 4.0
    i_z = jnp.pi * semi_major**3 * semi_minor / 4.0
    return i_y + i_z, i_y, i_z, area


def section_inner_dimensions(
    geometry: CrossSectionGeometry | int,
    outer_dimensions: Array,
    wall_thickness: Array,
) -> Array:
    """Derive a uniform-wall circular or rectangular cavity.

    Args:
        geometry: Circular or rectangular boundary family.
        outer_dimensions: Outer dimensions in the last axis [m].
        wall_thickness: Uniform wall at each section [m].

    Returns:
        Inner dimensions with the same shape as the outer dimensions. Callers
        must validate positive wall thickness and positive resulting dimensions.
    """
    if geometry not in (
        CrossSectionGeometry.CIRCULAR,
        CrossSectionGeometry.RECTANGULAR,
    ):
        raise ValueError(
            "Uniform-wall cavities require circular or rectangular geometry."
        )
    factor = 1 if geometry == CrossSectionGeometry.CIRCULAR else 2
    return (
        jnp.asarray(outer_dimensions) - factor * jnp.asarray(wall_thickness)[..., None]
    )


def section_torsion_constant(
    geometry: CrossSectionGeometry | int,
    outer_dimensions: Array,
    inner_dimensions: Array | None = None,
) -> Array:
    """Return the Saint-Venant torsion constant, independently of polar inertia.

    Args:
        geometry: Boundary family; hollow ellipses are unsupported.
        outer_dimensions: Dimensions in the last axis [m].
        inner_dimensions: Optional centred, uniform-wall cavity dimensions [m].

    Returns:
        Torsion constant [m^4]. Circles and solid ellipses use exact formulas;
        solid rectangles use 64 odd series terms; rectangular tubes use Bredt's
        thin-wall approximation. The latter is an estimate for thick walls.
    """
    dimensions = jnp.asarray(outer_dimensions)
    if geometry == CrossSectionGeometry.CIRCULAR:
        return section_properties(geometry, dimensions, inner_dimensions)[0]
    a, b = dimensions[..., 0], dimensions[..., 1]
    if geometry == CrossSectionGeometry.ELLIPTICAL:
        if inner_dimensions is not None:
            raise ValueError("Hollow elliptical torsion is not supported.")
        return jnp.pi * a**3 * b**3 / (a**2 + b**2)
    if geometry != CrossSectionGeometry.RECTANGULAR:
        raise ValueError("Unsupported cross-section geometry.")
    if inner_dimensions is not None:
        inner = jnp.asarray(inner_dimensions)
        thickness = (a - inner[..., 0]) / 2
        mid_a, mid_b = (a + inner[..., 0]) / 2, (b + inner[..., 1]) / 2
        return 2 * thickness * mid_a**2 * mid_b**2 / (mid_a + mid_b)
    long, short = jnp.maximum(a, b), jnp.minimum(a, b)
    odd = jnp.arange(1, 128, 2, dtype=dimensions.dtype)
    correction = jnp.sum(
        jnp.tanh(odd * jnp.pi * long[..., None] / (2 * short[..., None])) / odd**5,
        axis=-1,
    )
    return long * short**3 / 3 * (1 - 192 * short / (jnp.pi**5 * long) * correction)


def section_shear_factors(
    geometry: CrossSectionGeometry | int,
    outer_dimensions: Array,
    inner_dimensions: Array | None = None,
) -> Array:
    """Return approximate effective shear-area fractions along material y/z.

    Args:
        geometry: Circular or rectangular boundary family.
        outer_dimensions: Dimensions in the last axis [m].
        inner_dimensions: Optional centred, uniform-wall cavity dimensions [m].

    Returns:
        Two dimensionless factors per section. Solid circles use 0.9 and solid
        rectangles 5/6. Circular tubes use Cowper's annular expression at zero
        Poisson ratio; rectangular tubes use the areas of the two webs aligned
        with each shear direction. Tube estimates should be calibrated when
        their assumptions do not suit the material or wall thickness.
    """
    dimensions = jnp.asarray(outer_dimensions)
    if geometry == CrossSectionGeometry.CIRCULAR:
        if inner_dimensions is None:
            factor = jnp.full(dimensions.shape[:-1], 0.9)
        else:
            ratio2 = (jnp.asarray(inner_dimensions)[..., 0] / dimensions[..., 0]) ** 2
            factor = 6 * (1 + ratio2) ** 2 / (7 * (1 + ratio2) ** 2 + 20 * ratio2)
        return jnp.stack((factor, factor), axis=-1)
    if geometry != CrossSectionGeometry.RECTANGULAR:
        raise ValueError("Shear factors require circular or rectangular geometry.")
    if inner_dimensions is None:
        return jnp.full((*dimensions.shape[:-1], 2), 5 / 6)
    thickness = (dimensions[..., 0] - jnp.asarray(inner_dimensions)[..., 0]) / 2
    area = section_properties(geometry, dimensions, inner_dimensions)[3]
    return 2 * thickness[..., None] * dimensions / area[..., None]
