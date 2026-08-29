__all__ = [
    "PCSParams",
    "PlanarPCSParams",
    "PlanarHSAParams",
    "ISupportParams",
]

from dataclasses import MISSING, fields
from pathlib import Path
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.systems.components import ContinuumLinkParams
from soromox.systems.params import (
    BaseContinuumSoftRobotParams,
    BaseSoftRobotParams,
)


def _require_shape(name: str, value: Array, expected_shape: tuple[int, ...]) -> None:
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")


def _validate_continuum_structure(
    params: BaseContinuumSoftRobotParams,
    *,
    strain_dim: int,
    gravity_dim: int,
) -> int:
    params.link.validate_structure()
    n_segments = params.link.length.shape[0]
    _require_shape("gravity", params.gravity, (gravity_dim,))
    expected_reference_shape = (n_segments, strain_dim)
    if params.link.reference_strain.shape != expected_reference_shape:
        raise ValueError(
            f"reference_strain must have shape {expected_reference_shape}, "
            f"got {params.link.reference_strain.shape}."
        )
    expected_matrix_shape = (n_segments, strain_dim, strain_dim)
    _require_shape("stiffness", params.link.stiffness, expected_matrix_shape)
    _require_shape("damping", params.link.damping, expected_matrix_shape)
    _require_shape(
        "cross_section.coefficients",
        params.link.cross_section.coefficients,
        (n_segments, 1),
    )
    return n_segments


class PCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the spatial PCS model.

    Per-link runtime values live in the shared ``link`` component. Its
    ``stiffness`` and ``damping`` fields are canonical generalized matrices with
    shape ``(num_links, 6, 6)``. Isotropic Young, shear, and material-damping
    values are separate construction or optimization variables and can be mapped
    into these matrices explicitly. Fixed mounting belongs to the constructed
    model and floating pose belongs to runtime configuration state.
    """

    is_planar: ClassVar[bool] = False

    link: ContinuumLinkParams

    def validate_structure(self) -> None:
        """Validate spatial PCS parameter structure.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If link fields, reference strain, canonical matrices,
                cross-section coefficients, or gravity have invalid shapes.
        """
        _validate_continuum_structure(self, strain_dim=6, gravity_dim=3)

    def validate_values(self) -> None:
        """Validate eager spatial PCS parameter values."""
        self.link.validate_values()


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of fields in ``link`` indexes planar constant-strain
    segments. Its canonical generalized stiffness and damping matrices have
    shape ``(num_links, 3, 3)``. Isotropic material parameters are mapped
    explicitly into canonical link matrices rather than duplicated in this
    runtime parameter tree. Fixed mounting belongs to the constructed model and
    floating pose belongs to runtime configuration state.
    """

    is_planar: ClassVar[bool] = True

    link: ContinuumLinkParams

    def validate_structure(self) -> None:
        """Validate planar PCS parameter structure.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If link fields, reference strain, canonical matrices,
                cross-section coefficients, or gravity have invalid shapes.
        """
        _validate_continuum_structure(self, strain_dim=3, gravity_dim=2)

    def validate_values(self) -> None:
        """Validate eager planar PCS parameter values."""
        self.link.validate_values()


class ISupportParams(PCSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic PCS model.

    The leading axis of the shared PCS ``link`` fields indexes every physical
    rigid or pneumatic segment in robot order. ``ISupport`` expands these fields
    into the internal PCS layout using ``ISupportStructure``. Chamber fields
    index only pneumatic segments, in their order of appearance. Chamber
    azimuth is right-handed about local +X, measured from +Y toward +Z; array
    index ``j`` is pressure channel ``j``. If ``chamber_azimuth_angles`` is
    omitted, ``ISupport`` uses 0, 120, and 240 degrees for every pneumatic
    segment. ``chamber_effective_pressure_area`` scales each routed chamber-path
    length into its pressure-conjugate equivalent-volume coordinate. It contains
    one value per pneumatic segment and is shared by all chambers in that
    segment. When omitted, it is derived as
    ``pi * (chamber_outer_radius**2 - chamber_inner_radius**2)``. Stiffness and
    damping are supplied as canonical per-link generalized matrices.

    ``pcs_segment_lengths`` optionally stores flattened PCS segment lengths in
    the order defined by ``ISupportStructure.pcs_segment_counts``. If omitted,
    each pneumatic segment is split into equal-length PCS segments. The entries
    assigned to each pneumatic segment must sum to the corresponding
    ``length`` entry when the model is constructed.

    Segment types are defined by ``ISupportStructure.rigid_segment_selector``.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_distance: Array
    chamber_azimuth_angles: Array | None = None
    pcs_segment_lengths: Array | None = None
    chamber_effective_pressure_area: Array | None = None

    def validate_structure(self) -> None:
        """Validate array shapes independently of runtime chamber values."""
        super().validate_structure()
        chamber_inner_radius = jnp.asarray(self.chamber_inner_radius)
        if chamber_inner_radius.ndim != 1:
            raise ValueError(
                "chamber_inner_radius must be one-dimensional with shape "
                "(num_pneumatic_segments,)."
            )
        n_pneumatic_segments = chamber_inner_radius.shape[0]
        if n_pneumatic_segments < 1:
            raise ValueError(
                "ISupportParams must describe at least one pneumatic segment"
            )
        for name in (
            "chamber_inner_radius",
            "chamber_outer_radius",
            "chamber_distance",
        ):
            _require_shape(name, getattr(self, name), (n_pneumatic_segments,))

        if self.chamber_effective_pressure_area is not None:
            effective_area = jnp.asarray(self.chamber_effective_pressure_area)
            _require_shape(
                "chamber_effective_pressure_area",
                effective_area,
                (n_pneumatic_segments,),
            )

        if self.chamber_azimuth_angles is not None:
            angles = jnp.asarray(self.chamber_azimuth_angles)
            if (
                angles.ndim != 2
                or angles.shape[0] != n_pneumatic_segments
                or angles.shape[1] < 1
            ):
                raise ValueError(
                    "chamber_azimuth_angles must have shape "
                    "(num_pneumatic_segments, num_chambers_per_segment) with at "
                    "least one chamber per segment"
                )

        if self.pcs_segment_lengths is not None:
            pcs_segment_lengths = jnp.asarray(self.pcs_segment_lengths)
            if pcs_segment_lengths.ndim != 1:
                raise ValueError(
                    "pcs_segment_lengths must be one-dimensional with shape "
                    "(num_pcs_segments,)."
                )

    def validate_values(self) -> None:
        """Validate eager PCS and pneumatic chamber values.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If inherited PCS values, chamber radii, effective
                areas, segment lengths, or chamber azimuths are invalid.
        """
        super().validate_values()
        chamber_inner_radius = jnp.asarray(self.chamber_inner_radius)
        chamber_outer_radius = jnp.asarray(self.chamber_outer_radius)
        if not bool(
            jnp.all(
                jnp.isfinite(chamber_inner_radius) & jnp.isfinite(chamber_outer_radius)
            )
        ):
            raise ValueError("chamber radii entries must be finite")
        if not bool(
            jnp.all(
                (chamber_inner_radius >= 0.0)
                & (chamber_outer_radius > chamber_inner_radius)
            )
        ):
            raise ValueError(
                "chamber radii must satisfy 0 <= chamber_inner_radius < "
                "chamber_outer_radius"
            )

        if self.chamber_effective_pressure_area is not None:
            effective_area = jnp.asarray(self.chamber_effective_pressure_area)
            if not bool(jnp.all(jnp.isfinite(effective_area) & (effective_area > 0.0))):
                raise ValueError(
                    "chamber_effective_pressure_area entries must be finite and positive"
                )

        if self.chamber_azimuth_angles is not None:
            angles = jnp.asarray(self.chamber_azimuth_angles)
            if not bool(jnp.all(jnp.isfinite(angles))):
                raise ValueError("chamber_azimuth_angles entries must be finite")
            wrapped = jnp.mod(angles, 2.0 * jnp.pi)
            ordered = jnp.sort(wrapped, axis=-1)
            closed = jnp.concatenate(
                [ordered, ordered[..., :1] + 2.0 * jnp.pi], axis=-1
            )
            gaps = jnp.diff(closed, axis=-1)
            expected_gap = 2.0 * jnp.pi / angles.shape[-1]
            if not bool(jnp.allclose(gaps, expected_gap, rtol=1e-6, atol=1e-8)):
                raise ValueError(
                    "chamber_azimuth_angles must be uniformly distributed around "
                    "the full circle for every pneumatic segment"
                )


class PlanarHSAParams(BaseSoftRobotParams):
    """Dynamic parameters for planar HSA systems.

    Field names denote one segment/platform quantity; leading axes store the
    batched values. Arrays store length, rod geometry, reference strain
    components, platform/cap dimensions, end-effector offset, and optional
    hysteresis coefficients used by the numerical HSA model. Fixed mounting
    belongs to the constructed model and floating pose belongs to runtime
    configuration state.
    """

    is_planar: ClassVar[bool] = True

    length: Array
    proximal_cap_length: Array
    distal_cap_length: Array
    rod_height: Array
    rod_outer_radius: Array
    rod_inner_radius: Array
    rod_offset: Array
    reference_strain: Array
    strain_coupling: Array
    platform_dimension: Array
    rod_density: Array
    platform_density: Array
    end_cap_density: Array
    nominal_bending_stiffness: Array
    nominal_shear_stiffness: Array
    nominal_axial_stiffness: Array
    bending_shear_stiffness: Array
    bending_stiffness_correction: Array
    shear_stiffness_correction: Array
    axial_stiffness_correction: Array
    bending_damping: Array
    shear_damping: Array
    axial_damping: Array
    phi_max: Array
    platform_mass: Array
    platform_center_of_gravity: Array
    end_effector_offset: Array
    hysteresis_basis: Array
    hysteresis_alpha: Array
    hysteresis_A: Array
    hysteresis_n: Array
    hysteresis_beta: Array
    hysteresis_gamma: Array

    @classmethod
    def from_npz(cls, path: str | Path) -> "PlanarHSAParams":
        """Load planar HSA parameters from an explicit ``.npz`` file path."""
        with np.load(path) as data:
            field_names = [field.name for field in fields(cls)]
            required_names = {
                field.name
                for field in fields(cls)
                if field.default is MISSING and field.default_factory is MISSING
            }
            missing = sorted(name for name in required_names if name not in data)
            if missing:
                missing_names = ", ".join(missing)
                raise KeyError(
                    f"Planar HSA parameter file is missing field(s): {missing_names}."
                )

            return cls(
                **{
                    field_name: jnp.asarray(data[field_name])
                    for field_name in field_names
                    if field_name in data
                }
            )

    def validate_structure(self) -> None:
        length = jnp.asarray(self.length)
        if len(length.shape) != 1:
            raise ValueError(
                "length must be one-dimensional with shape (num_segments,)."
            )
        n_segments = length.shape[0]
        if n_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {n_segments}.")
        rod_outer_radius = jnp.asarray(self.rod_outer_radius)
        if rod_outer_radius.ndim != 2 or rod_outer_radius.shape[0] != n_segments:
            raise ValueError(
                "rod_outer_radius must have shape "
                f"({n_segments}, num_rods_per_segment), got {rod_outer_radius.shape}."
            )
        rod_shape = rod_outer_radius.shape
        for name in (
            "rod_height",
            "rod_inner_radius",
            "rod_offset",
            "strain_coupling",
            "rod_density",
            "nominal_bending_stiffness",
            "nominal_shear_stiffness",
            "nominal_axial_stiffness",
            "bending_shear_stiffness",
            "bending_stiffness_correction",
            "shear_stiffness_correction",
            "axial_stiffness_correction",
            "bending_damping",
            "shear_damping",
            "axial_damping",
            "phi_max",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != rod_shape:
                raise ValueError(
                    f"{name} must have shape {rod_shape}, got {value.shape}."
                )

        reference_strain = jnp.asarray(self.reference_strain)
        if reference_strain.shape != rod_shape + (3,):
            raise ValueError(
                "reference_strain must have shape "
                f"{rod_shape + (3,)}, got {reference_strain.shape}."
            )

        per_segment_shapes = {
            "proximal_cap_length": (n_segments,),
            "distal_cap_length": (n_segments,),
            "platform_dimension": (n_segments, 3),
            "platform_density": (n_segments,),
            "end_cap_density": (n_segments,),
        }
        for name, expected_shape in per_segment_shapes.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

        scalar_shapes = {
            "platform_mass": (),
            "platform_center_of_gravity": (2,),
            "gravity": (2,),
            "end_effector_offset": (3,),
        }
        for name, expected_shape in scalar_shapes.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

        hysteresis_basis = jnp.asarray(self.hysteresis_basis)
        if hysteresis_basis.size == 0:
            return
        if hysteresis_basis.ndim != 2 or hysteresis_basis.shape[0] != 3 * n_segments:
            raise ValueError(
                "hysteresis_basis must have shape "
                f"({3 * n_segments}, num_hysteresis), got {hysteresis_basis.shape}."
            )
        num_hysteresis = hysteresis_basis.shape[1]
        for name in (
            "hysteresis_beta",
            "hysteresis_gamma",
            "hysteresis_A",
            "hysteresis_n",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (num_hysteresis,):
                raise ValueError(
                    f"{name} must have shape ({num_hysteresis},), got {value.shape}."
                )
        hysteresis_alpha = jnp.asarray(self.hysteresis_alpha)
        if hysteresis_alpha.shape != (3 * n_segments,):
            raise ValueError(
                "hysteresis_alpha must have shape "
                f"({3 * n_segments},), got {hysteresis_alpha.shape}."
            )

    def validate_values(self) -> None:
        """Validate eager planar HSA parameter values."""
