"""Shared continuum-link runtime parameters and construction specifications."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import BaseSystemParams

from .cross_sections import (
    CrossSectionGeometry,
    CrossSectionParams,
    LinearProfile,
    ProfileType,
)
from .materials import shear_modulus_from_poisson_ratio

__all__ = ["ContinuumLinkParams", "LinkSpec"]


def _validate_symmetric(name: str, value: Array) -> None:
    finite = bool(jnp.all(jnp.isfinite(value)))
    symmetric = bool(jnp.allclose(value, jnp.swapaxes(value, -1, -2)))
    if not finite:
        raise ValueError(f"{name} must contain only finite values.")
    if not symmetric:
        raise ValueError(f"{name} must be symmetric in its trailing dimensions.")


def _validate_finite_domain(
    name: str, value: Array | float, *, strictly_positive: bool = False
) -> None:
    """Validate eager numeric values."""
    array = jnp.asarray(value)
    finite = bool(jnp.all(jnp.isfinite(array)))
    positive = bool(jnp.all(array > 0.0)) if strictly_positive else True
    if not finite:
        raise ValueError(f"{name} must contain only finite values.")
    if not positive:
        raise ValueError(f"{name} must be strictly positive.")


class ContinuumLinkParams(BaseSystemParams):
    """Canonical batched dynamic parameters for continuum links.

    Attributes:
        length: Link lengths with shape ``(num_links,)``.
        density: Volumetric mass densities with shape ``(num_links,)``.
        reference_strain: Reference strain rows with shape
            ``(num_links, strain_dimension)``.
        cross_section: Batched cross-section coefficients shared by all
            continuum-system families.
        stiffness: Canonical generalized link stiffness matrices with shape
            ``(num_links, generalized_dimension, generalized_dimension)``.
        damping: Canonical generalized link damping matrices with the same
            shape as ``stiffness``.
    """

    length: Array
    density: Array
    reference_strain: Array
    cross_section: CrossSectionParams
    stiffness: Array
    damping: Array

    def __check_init__(self) -> None:
        for name in ("length", "density", "reference_strain", "stiffness", "damping"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))
        self.validate_for_update()

    def _normalize_replacement(self, name: str, value: object) -> object:
        """Normalize replacement link-array values to JAX arrays."""
        if name in ("length", "density", "reference_strain", "stiffness", "damping"):
            return jnp.asarray(value)
        return value

    def validate_structure(self) -> None:
        """Validate link-array and canonical matrix shapes.

        Returns:
            None.

        Raises:
            ValueError: If link fields disagree on ``num_links``, reference
                strain or coefficient arrays are not two-dimensional, or
                canonical matrices are not equally shaped square batches.
        """
        length = jnp.asarray(self.length)
        if length.ndim != 1 or length.shape[0] < 1:
            raise ValueError("length must have shape (num_links,) with num_links >= 1.")
        num_links = length.shape[0]
        density = jnp.asarray(self.density)
        if density.shape != (num_links,):
            raise ValueError(
                f"density must have shape ({num_links},), got {density.shape}."
            )
        reference_strain = jnp.asarray(self.reference_strain)
        if (
            reference_strain.ndim != 2
            or reference_strain.shape[0] != num_links
            or reference_strain.shape[1] < 1
        ):
            raise ValueError(
                "reference_strain must have shape (num_links, strain_dimension)."
            )
        self.cross_section.validate_structure()
        if self.cross_section.coefficients.shape[0] != num_links:
            raise ValueError("cross-section coefficients must have one row per link.")
        stiffness = jnp.asarray(self.stiffness)
        damping = jnp.asarray(self.damping)
        if stiffness.ndim != 3 or stiffness.shape[0] != num_links:
            raise ValueError(
                "stiffness must have shape "
                "(num_links, generalized_dimension, generalized_dimension)."
            )
        if stiffness.shape[1] != stiffness.shape[2]:
            raise ValueError("stiffness trailing dimensions must be square.")
        if damping.shape != stiffness.shape:
            raise ValueError(
                f"damping must have shape {stiffness.shape}, got {damping.shape}."
            )

    def validate_values(self) -> None:
        """Validate eager link values and canonical matrix properties."""
        _validate_finite_domain("length", self.length, strictly_positive=True)
        _validate_finite_domain("density", self.density, strictly_positive=True)
        _validate_finite_domain("reference_strain", self.reference_strain)
        self.cross_section.validate_values()
        _validate_symmetric("stiffness", self.stiffness)
        _validate_symmetric("damping", self.damping)


def _profile(
    value: float | LinearProfile, name: str
) -> tuple[tuple[float, ...], ProfileType, tuple[str, ...]]:
    if isinstance(value, LinearProfile):
        return (value.base, value.tip), "linear", (f"{name}_base", f"{name}_tip")
    return (float(value),), "constant", (name,)


@dataclass(frozen=True)
class LinkSpec:
    """Construction specification for one continuum link.

    Prefer the geometry-specific :meth:`circular`, :meth:`rectangular`, and
    :meth:`elliptical` factories. Stiffness must be described by an explicit
    ``stiffness`` matrix or by ``young_modulus`` together with exactly one of
    ``shear_modulus`` and ``poisson_ratio``. Damping may use
    ``material_damping_coefficient`` or an explicit ``damping`` matrix; omitting
    both disables damping.

    Attributes:
        cross_section_geometry: Static cross-section family.
        length: Link length.
        density: Volumetric mass density.
        reference_strain: Reference strain vector for this link.
        cross_section_coefficients: Packed constant or linear profile
            coefficients.
        cross_section_profile_types: Profile kind for each geometric dimension.
        cross_section_profile_parameter_counts: Number of packed coefficients
            occupied by each geometric dimension.
        cross_section_coefficient_names: Human-readable names corresponding to
            the packed coefficients.
        young_modulus: Optional isotropic Young's modulus.
        shear_modulus: Optional isotropic shear modulus.
        poisson_ratio: Optional isotropic Poisson's ratio.
        material_damping_coefficient: Optional isotropic material damping value.
            Omitting it together with ``damping`` disables damping.
        stiffness: Optional explicit generalized stiffness matrix.
        damping: Optional explicit generalized damping matrix.
    """

    cross_section_geometry: CrossSectionGeometry
    length: float
    density: float
    reference_strain: Array | list[float]
    cross_section_coefficients: tuple[float, ...]
    cross_section_profile_types: tuple[ProfileType, ...]
    cross_section_profile_parameter_counts: tuple[int, ...]
    cross_section_coefficient_names: tuple[str, ...]
    young_modulus: float | None = None
    shear_modulus: float | None = None
    poisson_ratio: float | None = None
    material_damping_coefficient: float | None = None
    stiffness: Array | None = None
    damping: Array | None = None

    def __post_init__(self) -> None:
        _validate_finite_domain("length", self.length, strictly_positive=True)
        _validate_finite_domain("density", self.density, strictly_positive=True)
        _validate_finite_domain("reference_strain", self.reference_strain)
        reference_strain = jnp.asarray(self.reference_strain)
        if reference_strain.ndim != 1 or reference_strain.size < 1:
            raise ValueError(
                "reference_strain must be a nonempty one-dimensional array."
            )
        _validate_finite_domain(
            "cross_section_coefficients",
            jnp.asarray(self.cross_section_coefficients),
            strictly_positive=True,
        )
        material_stiffness = any(
            value is not None
            for value in (
                self.young_modulus,
                self.shear_modulus,
                self.poisson_ratio,
            )
        )
        if material_stiffness and self.stiffness is not None:
            raise ValueError(
                "Provide either isotropic material properties or stiffness, not both."
            )
        if not material_stiffness and self.stiffness is None:
            raise ValueError(
                "Provide isotropic material properties or an explicit stiffness."
            )
        if material_stiffness and self.young_modulus is None:
            raise ValueError("Material stiffness construction requires young_modulus.")
        if material_stiffness and (
            (self.shear_modulus is None) == (self.poisson_ratio is None)
        ):
            raise ValueError(
                "Material stiffness construction requires exactly one of "
                "shear_modulus and poisson_ratio."
            )
        if self.material_damping_coefficient is not None and self.damping is not None:
            raise ValueError(
                "Provide either material_damping_coefficient or damping, not both."
            )
        if self.young_modulus is not None:
            _validate_finite_domain(
                "young_modulus", self.young_modulus, strictly_positive=True
            )
        if self.shear_modulus is not None:
            _validate_finite_domain(
                "shear_modulus", self.shear_modulus, strictly_positive=True
            )
        if self.poisson_ratio is not None:
            _validate_finite_domain("poisson_ratio", self.poisson_ratio)
            valid_poisson_ratio = bool(
                jnp.all(
                    (jnp.asarray(self.poisson_ratio) > -1.0)
                    & (jnp.asarray(self.poisson_ratio) < 0.5)
                )
            )
            if not valid_poisson_ratio:
                raise ValueError(
                    "poisson_ratio must be greater than -1 and less than 0.5."
                )
        if self.material_damping_coefficient is not None:
            _validate_finite_domain(
                "material_damping_coefficient", self.material_damping_coefficient
            )
            nonnegative_damping = bool(
                jnp.all(jnp.asarray(self.material_damping_coefficient) >= 0.0)
            )
            if not nonnegative_damping:
                raise ValueError("material_damping_coefficient must be nonnegative.")
        for name in ("stiffness", "damping"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = jnp.asarray(value)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Explicit {name} must be a square matrix.")
            _validate_symmetric(name, matrix)

    def _resolved_shear_modulus(self) -> Array | float:
        """Return the supplied or Poisson-ratio-derived shear modulus."""
        if self.shear_modulus is not None:
            return self.shear_modulus
        if self.young_modulus is None or self.poisson_ratio is None:
            raise ValueError("The link does not define isotropic material stiffness.")
        return shear_modulus_from_poisson_ratio(self.young_modulus, self.poisson_ratio)

    @classmethod
    def _make(
        cls,
        *,
        geometry: CrossSectionGeometry,
        profiles: tuple[tuple[str, float | LinearProfile], ...],
        length: float,
        density: float,
        reference_strain: Array | list[float],
        young_modulus: float | None,
        shear_modulus: float | None,
        poisson_ratio: float | None,
        material_damping_coefficient: float | None,
        stiffness: Array | None,
        damping: Array | None,
    ) -> LinkSpec:
        packed = [_profile(value, name) for name, value in profiles]
        return cls(
            cross_section_geometry=geometry,
            length=length,
            density=density,
            reference_strain=reference_strain,
            cross_section_coefficients=tuple(
                coefficient for values, _, _ in packed for coefficient in values
            ),
            cross_section_profile_types=tuple(kind for _, kind, _ in packed),
            cross_section_profile_parameter_counts=tuple(
                len(values) for values, _, _ in packed
            ),
            cross_section_coefficient_names=tuple(
                name for _, _, names in packed for name in names
            ),
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
            poisson_ratio=poisson_ratio,
            material_damping_coefficient=material_damping_coefficient,
            stiffness=stiffness,
            damping=damping,
        )

    @classmethod
    def circular(
        cls,
        *,
        length: float,
        radius: float | LinearProfile,
        density: float,
        reference_strain: Array | list[float],
        young_modulus: float | None = None,
        shear_modulus: float | None = None,
        poisson_ratio: float | None = None,
        material_damping_coefficient: float | None = None,
        stiffness: Array | None = None,
        damping: Array | None = None,
    ) -> LinkSpec:
        """Create a solid circular-link specification.

        Args:
            length: Link length.
            radius: Constant radius or a base-to-tip linear radius profile.
            density: Volumetric mass density.
            reference_strain: Reference strain vector. Spatial systems use six
                entries and planar PCS uses three entries.
            young_modulus: Young's modulus used with ``shear_modulus`` to build
                generalized stiffness. Mutually exclusive with ``stiffness``.
            shear_modulus: Shear modulus used with ``young_modulus``.
                Mutually exclusive with ``poisson_ratio``.
            poisson_ratio: Poisson's ratio used with ``young_modulus`` to derive
                the shear modulus. Mutually exclusive with ``shear_modulus``.
            material_damping_coefficient: Isotropic damping value used to build
                generalized damping. Mutually exclusive with ``damping``. If
                both are omitted, damping is disabled.
            stiffness: Explicit generalized stiffness matrix.
            damping: Explicit generalized damping matrix.

        Returns:
            A circular :class:`LinkSpec` ready for PCS or GVS construction.

        Raises:
            ValueError: If material and explicit sources are missing,
                incomplete, or supplied together, or if an explicit matrix is
                not finite, square, and symmetric, or if a physical dimension,
                density, length, or material scalar is outside its valid
                domain.
        """
        return cls._make(
            geometry=CrossSectionGeometry.CIRCULAR,
            profiles=(("radius", radius),),
            length=length,
            density=density,
            reference_strain=reference_strain,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
            poisson_ratio=poisson_ratio,
            material_damping_coefficient=material_damping_coefficient,
            stiffness=stiffness,
            damping=damping,
        )

    @classmethod
    def rectangular(
        cls,
        *,
        length: float,
        height: float | LinearProfile,
        width: float | LinearProfile,
        density: float,
        reference_strain: Array | list[float],
        young_modulus: float | None = None,
        shear_modulus: float | None = None,
        poisson_ratio: float | None = None,
        material_damping_coefficient: float | None = None,
        stiffness: Array | None = None,
        damping: Array | None = None,
    ) -> LinkSpec:
        """Create a solid rectangular-link specification.

        Args:
            length: Link length.
            height: Constant height or base-to-tip linear height profile.
            width: Constant width or base-to-tip linear width profile.
            density: Volumetric mass density.
            reference_strain: Reference strain vector for the link.
            young_modulus: Young's modulus used with ``shear_modulus`` to build
                generalized stiffness. Mutually exclusive with ``stiffness``.
            shear_modulus: Shear modulus used with ``young_modulus``.
                Mutually exclusive with ``poisson_ratio``.
            poisson_ratio: Poisson's ratio used with ``young_modulus`` to derive
                the shear modulus. Mutually exclusive with ``shear_modulus``.
            material_damping_coefficient: Isotropic damping value used to build
                generalized damping. Mutually exclusive with ``damping``. If
                both are omitted, damping is disabled.
            stiffness: Explicit generalized stiffness matrix.
            damping: Explicit generalized damping matrix.

        Returns:
            A rectangular :class:`LinkSpec` ready for GVS construction.

        Raises:
            ValueError: If material and explicit sources are missing,
                incomplete, or supplied together, or if an explicit matrix is
                not finite, square, and symmetric, or if a physical dimension,
                density, length, or material scalar is outside its valid
                domain.
        """
        return cls._make(
            geometry=CrossSectionGeometry.RECTANGULAR,
            profiles=(("height", height), ("width", width)),
            length=length,
            density=density,
            reference_strain=reference_strain,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
            poisson_ratio=poisson_ratio,
            material_damping_coefficient=material_damping_coefficient,
            stiffness=stiffness,
            damping=damping,
        )

    @classmethod
    def elliptical(
        cls,
        *,
        length: float,
        semi_major: float | LinearProfile,
        semi_minor: float | LinearProfile,
        density: float,
        reference_strain: Array | list[float],
        young_modulus: float | None = None,
        shear_modulus: float | None = None,
        poisson_ratio: float | None = None,
        material_damping_coefficient: float | None = None,
        stiffness: Array | None = None,
        damping: Array | None = None,
    ) -> LinkSpec:
        """Create a solid elliptical-link specification.

        Args:
            length: Link length.
            semi_major: Constant semi-major axis or base-to-tip linear profile.
            semi_minor: Constant semi-minor axis or base-to-tip linear profile.
            density: Volumetric mass density.
            reference_strain: Reference strain vector for the link.
            young_modulus: Young's modulus used with ``shear_modulus`` to build
                generalized stiffness. Mutually exclusive with ``stiffness``.
            shear_modulus: Shear modulus used with ``young_modulus``.
                Mutually exclusive with ``poisson_ratio``.
            poisson_ratio: Poisson's ratio used with ``young_modulus`` to derive
                the shear modulus. Mutually exclusive with ``shear_modulus``.
            material_damping_coefficient: Isotropic damping value used to build
                generalized damping. Mutually exclusive with ``damping``. If
                both are omitted, damping is disabled.
            stiffness: Explicit generalized stiffness matrix.
            damping: Explicit generalized damping matrix.

        Returns:
            An elliptical :class:`LinkSpec` ready for GVS construction.

        Raises:
            ValueError: If material and explicit sources are missing,
                incomplete, or supplied together, or if an explicit matrix is
                not finite, square, and symmetric, or if a physical dimension,
                density, length, or material scalar is outside its valid
                domain.
        """
        return cls._make(
            geometry=CrossSectionGeometry.ELLIPTICAL,
            profiles=(("semi_major", semi_major), ("semi_minor", semi_minor)),
            length=length,
            density=density,
            reference_strain=reference_strain,
            young_modulus=young_modulus,
            shear_modulus=shear_modulus,
            poisson_ratio=poisson_ratio,
            material_damping_coefficient=material_damping_coefficient,
            stiffness=stiffness,
            damping=damping,
        )
