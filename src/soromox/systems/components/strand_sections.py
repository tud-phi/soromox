"""Strand geometry shared by helix mechanics and rendering."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.systems.components.cross_sections import (
    CrossSectionGeometry,
    section_inner_dimensions,
)
from soromox.systems.params import BaseSystemParams

__all__ = ["StrandSection"]


class StrandSection(BaseSystemParams):
    """Constant circular or rectangular section normal to a strand's x axis.

    Attributes:
        geometry: Static cross-section family.
        outer_dimensions: Dynamic dimensions [m]: ``[radius]`` for circles or
            ``[height, width]`` for rectangles. Height spans local y, width z.
        wall_thickness: Uniform wall [m] for a hollow section; None for solid.
    """

    geometry: CrossSectionGeometry = eqx.field(static=True)
    outer_dimensions: Array
    wall_thickness: Array | None = None

    def __check_init__(self) -> None:
        object.__setattr__(
            self, "outer_dimensions", jnp.asarray(self.outer_dimensions, dtype=float)
        )
        if self.wall_thickness is not None:
            object.__setattr__(
                self, "wall_thickness", jnp.asarray(self.wall_thickness, dtype=float)
            )
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        if name in {"outer_dimensions", "wall_thickness"} and value is not None:
            return jnp.asarray(value, dtype=float)
        return value

    @classmethod
    def circular(cls, radius: float) -> Self:
        """Construct a solid circular section with radius in metres."""
        return cls(CrossSectionGeometry.CIRCULAR, jnp.asarray([radius]))

    @classmethod
    def circular_hollow(cls, *, radius: float, wall_thickness: float) -> Self:
        """Construct a circular tube from its outer radius and wall [m]."""
        return cls(CrossSectionGeometry.CIRCULAR, jnp.asarray([radius]), wall_thickness)

    @classmethod
    def rectangular(cls, *, width: float, height: float) -> Self:
        """Construct a solid rectangle with height along y and width along z [m]."""
        return cls(CrossSectionGeometry.RECTANGULAR, jnp.asarray([height, width]))

    @classmethod
    def rectangular_hollow(
        cls, *, width: float, height: float, wall_thickness: float
    ) -> Self:
        """Construct a rectangular tube from outer dimensions and wall [m]."""
        return cls(
            CrossSectionGeometry.RECTANGULAR,
            jnp.asarray([height, width]),
            wall_thickness,
        )

    def validate_structure(self) -> None:
        if self.geometry not in (
            CrossSectionGeometry.CIRCULAR,
            CrossSectionGeometry.RECTANGULAR,
        ):
            raise ValueError("Strands require circular or rectangular geometry.")
        count = 1 if self.geometry == CrossSectionGeometry.CIRCULAR else 2
        if self.outer_dimensions.shape != (count,):
            raise ValueError(f"Strand outer_dimensions must have shape ({count},).")
        if self.wall_thickness is not None and self.wall_thickness.shape != ():
            raise ValueError("wall_thickness must be scalar.")

    def validate_values(self) -> None:
        if not bool(
            jnp.all(jnp.isfinite(self.outer_dimensions) & (self.outer_dimensions > 0))
        ):
            raise ValueError("Strand outer dimensions must be finite and positive.")
        if self.wall_thickness is not None:
            if not bool(jnp.isfinite(self.wall_thickness) & (self.wall_thickness > 0)):
                raise ValueError("wall_thickness must be finite and positive.")
            if not bool(jnp.all(self.inner_dimensions > 0)):
                raise ValueError(
                    "wall_thickness must leave a positive inner radius or positive inner height and width."
                )

    @property
    def inner_dimensions(self) -> Array | None:
        """Dimensions of the centred cavity, or None for a solid section."""
        if self.wall_thickness is None:
            return None
        return section_inner_dimensions(
            self.geometry, self.outer_dimensions, self.wall_thickness
        )

    @property
    def radial_extent(self) -> Array:
        """Half-extent along the strand's local y (radial) axis [m]."""
        if self.geometry == CrossSectionGeometry.CIRCULAR:
            return self.outer_dimensions[0]
        return self.outer_dimensions[0] / 2
