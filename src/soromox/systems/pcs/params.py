__all__ = [
    "PCSParams",
    "PlanarPCSParams",
    "ISupportParams",
]

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from soromox.systems.params import (
    BaseContinuumSoftRobotParams,
    validate_planar_base_pose,
    validate_quaternion_base_pose,
)


def _require_shape(name: str, value: Array, expected_shape: tuple[int, ...]) -> None:
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")


def _validate_damping_input(
    params: "PCSParams | PlanarPCSParams",
    *,
    strain_dim: int,
    n_segments: int,
) -> None:
    has_damping_matrix = params.damping_matrix is not None
    has_material_damping = params.material_damping_coefficient is not None
    if has_damping_matrix == has_material_damping:
        raise ValueError(
            "Exactly one of damping_matrix or material_damping_coefficient "
            "must be provided."
        )

    if has_damping_matrix:
        damping_matrix = jnp.asarray(params.damping_matrix)
        _require_shape(
            "damping_matrix",
            damping_matrix,
            (strain_dim * n_segments, strain_dim * n_segments),
        )
        return

    material_damping_coefficient = jnp.asarray(params.material_damping_coefficient)
    if material_damping_coefficient.ndim == 0:
        return
    _require_shape(
        "material_damping_coefficient",
        material_damping_coefficient,
        (n_segments,),
    )


def _validate_continuum_base(
    params: BaseContinuumSoftRobotParams,
    *,
    strain_dim: int,
    gravity_dim: int,
) -> int:
    if len(params.length.shape) != 1:
        raise ValueError("length must be one-dimensional with shape (num_segments,).")
    n_segments = params.length.shape[0]
    if n_segments < 1:
        raise ValueError(f"num_segments must be at least 1, got {n_segments}.")
    _require_shape("density", params.density, (n_segments,))
    _require_shape("gravity", params.gravity, (gravity_dim,))
    expected_reference_size = strain_dim * n_segments
    if params.reference_strain.size != expected_reference_size:
        raise ValueError(
            "reference_strain must contain "
            f"{expected_reference_size} entries, got {params.reference_strain.size}."
        )
    return n_segments


class PCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the spatial PCS model.

    ``length``, ``radius``, material parameters, and density use a leading
    segment axis. Damping can be supplied either as the preferred
    ``material_damping_coefficient`` or as a full flattened strain
    ``damping_matrix``. ``material_damping_coefficient`` is a viscosity-like
    modulus in Pa*s (N*s/m^2); it may be scalar or have one value per segment.
    The assembled matrix includes geometry and length factors, so its entries
    have generalized-coordinate-dependent units rather than a single Pa*s unit.
    ``base_pose`` is the scalar-first quaternion SE(3) base pose vector
    ``[qw, qx, qy, qz, x, y, z]`` used to initialize the base transform. The
    quaternion is normalized before use and must have nonzero finite norm.
    Omitting ``base_pose`` and ``gravity`` selects upright spatial mounting and
    negative-z Earth gravity.
    """

    is_planar: ClassVar[bool] = False

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    material_damping_coefficient: Array | None = eqx.field(default=None, kw_only=True)
    damping_matrix: Array | None = eqx.field(default=None, kw_only=True)

    def validate(self) -> None:
        n_segments = _validate_continuum_base(self, strain_dim=6, gravity_dim=3)
        _require_shape("radius", self.radius, (n_segments,))
        _require_shape("young_modulus", self.young_modulus, (n_segments,))
        _require_shape("shear_modulus", self.shear_modulus, (n_segments,))
        _validate_damping_input(self, strain_dim=6, n_segments=n_segments)
        validate_quaternion_base_pose("base_pose", self.base_pose, (7,))


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of per-segment fields indexes planar constant-strain
    segments. ``base_pose`` stores the planar pose ``[theta, x, y]`` with shape
    ``(3,)``. ``theta`` is a right-handed angle in radians about the
    out-of-plane z-axis, and ``x``/``y`` are direct translations in the parent
    frame. ``material_damping_coefficient`` is a viscosity-like modulus in Pa*s
    (N*s/m^2); it may be scalar or have one value per segment. The assembled
    matrix includes geometry and length factors and therefore has
    generalized-coordinate-dependent entry units.
    Omitting ``base_pose`` and ``gravity`` selects upright planar
    mounting and negative-y Earth gravity.
    """

    is_planar: ClassVar[bool] = True

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    material_damping_coefficient: Array | None = eqx.field(default=None, kw_only=True)
    damping_matrix: Array | None = eqx.field(default=None, kw_only=True)

    def validate(self) -> None:
        n_segments = _validate_continuum_base(self, strain_dim=3, gravity_dim=2)
        _require_shape("radius", self.radius, (n_segments,))
        _require_shape("young_modulus", self.young_modulus, (n_segments,))
        _require_shape("shear_modulus", self.shear_modulus, (n_segments,))
        _validate_damping_input(self, strain_dim=3, n_segments=n_segments)
        validate_planar_base_pose("base_pose", self.base_pose)


class ISupportParams(PCSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic PCS model.

    The leading axis of the standard PCS body fields indexes every physical
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
    ``pi * (chamber_outer_radius**2 - chamber_inner_radius**2)``. Damping can be
    supplied as ``material_damping_coefficient`` or as a full ``damping_matrix``.
    A custom ``damping_matrix`` is expressed in flattened pneumatic-segment
    strain coordinates and must be block diagonal by pneumatic segment when the
    model is constructed.

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

    def validate(self) -> None:
        super().validate()
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
            _require_shape(
                "chamber_effective_pressure_area",
                effective_area,
                (n_pneumatic_segments,),
            )
            if not bool(jnp.all(jnp.isfinite(effective_area) & (effective_area > 0.0))):
                raise ValueError(
                    "chamber_effective_pressure_area entries must be finite and positive"
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

        if self.pcs_segment_lengths is not None:
            pcs_segment_lengths = jnp.asarray(self.pcs_segment_lengths)
            if pcs_segment_lengths.ndim != 1:
                raise ValueError(
                    "pcs_segment_lengths must be one-dimensional with shape "
                    "(num_pcs_segments,)."
                )
