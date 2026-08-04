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
]


class CrossSectionGeometry(IntEnum):
    """Supported solid cross-section families."""

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
        self.validate()

    def validate(self) -> None:
        """Validate the batched cross-section coefficient array.

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
) -> tuple[Array, Array, Array, Array]:
    """Return ``(I_x, I_y, I_z, area)`` for a solid section.

    The material-frame x-axis is longitudinal. Rectangular dimensions are
    ordered ``[height, width]`` and elliptical dimensions are ordered
    ``[semi_major, semi_minor]``.

    Args:
        geometry: Cross-section family, supplied as a
            :class:`CrossSectionGeometry` value or its integer value.
        dimensions: Geometry dimensions. Circular sections use ``[radius]``;
            rectangular sections use ``[height, width]``; elliptical sections
            use ``[semi_major, semi_minor]``.

    Returns:
        A tuple ``(I_x, I_y, I_z, area)`` containing the polar second moment
        about the longitudinal material x-axis, the transverse second moments,
        and the section area.

    Raises:
        ValueError: If ``geometry`` does not identify a supported section.
    """
    dimensions = jnp.asarray(dimensions)
    geometry = CrossSectionGeometry(int(geometry))
    if geometry == CrossSectionGeometry.CIRCULAR:
        radius = dimensions[0]
        area = jnp.pi * radius**2
        transverse = jnp.pi * radius**4 / 4.0
        polar = 2.0 * transverse
        return polar, transverse, transverse, area
    if geometry == CrossSectionGeometry.RECTANGULAR:
        height, width = dimensions[0], dimensions[1]
        area = height * width
        i_y = height * width**3 / 12.0
        i_z = width * height**3 / 12.0
        return i_y + i_z, i_y, i_z, area
    semi_major, semi_minor = dimensions[0], dimensions[1]
    area = jnp.pi * semi_major * semi_minor
    i_y = jnp.pi * semi_major * semi_minor**3 / 4.0
    i_z = jnp.pi * semi_major**3 * semi_minor / 4.0
    return i_y + i_z, i_y, i_z, area
