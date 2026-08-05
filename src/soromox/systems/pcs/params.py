__all__ = [
    "PCSParams",
    "PlanarPCSParams",
    "ISupportParams",
]

from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from soromox.systems.components import ContinuumLinkParams
from soromox.systems.params import (
    BaseContinuumSoftRobotParams,
    validate_planar_base_pose,
    validate_quaternion_base_pose,
)


def _require_shape(name: str, value: Array, expected_shape: tuple[int, ...]) -> None:
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")


def _validate_continuum_base(
    params: BaseContinuumSoftRobotParams,
    *,
    strain_dim: int,
    gravity_dim: int,
) -> int:
    params.link.validate()
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
    into these matrices explicitly.
    ``base_pose`` is the scalar-first quaternion SE(3) base pose vector
    ``[qw, qx, qy, qz, x, y, z]`` used to initialize the base transform. The
    quaternion is normalized before use and must have nonzero finite norm.
    Omitting ``base_pose`` and ``gravity`` selects upright spatial mounting and
    negative-z Earth gravity.
    """

    is_planar: ClassVar[bool] = False

    link: ContinuumLinkParams

    def validate(self) -> None:
        """Validate spatial PCS parameter shapes and values.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If link fields are invalid; reference strain, canonical
                matrices, cross-section coefficients, or gravity have the wrong
                shape; or the spatial base pose is invalid.
        """
        _validate_continuum_base(self, strain_dim=6, gravity_dim=3)
        validate_quaternion_base_pose("base_pose", self.base_pose, (7,))


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of fields in ``link`` indexes planar constant-strain
    segments. Its canonical generalized stiffness and damping matrices have
    shape ``(num_links, 3, 3)``. ``base_pose`` stores the planar pose
    ``[theta, x, y]`` with shape
    ``(3,)``. ``theta`` is a right-handed angle in radians about the
    out-of-plane z-axis, and ``x``/``y`` are direct translations in the parent
    frame. Isotropic material parameters are mapped explicitly into canonical
    link matrices rather than duplicated in this runtime parameter tree.
    Omitting ``base_pose`` and ``gravity`` selects upright planar
    mounting and negative-y Earth gravity.
    """

    is_planar: ClassVar[bool] = True

    link: ContinuumLinkParams

    def validate(self) -> None:
        """Validate planar PCS parameter shapes and values.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If link fields are invalid; reference strain, canonical
                matrices, cross-section coefficients, gravity, or the planar
                base pose has the wrong shape or contains invalid values.
        """
        _validate_continuum_base(self, strain_dim=3, gravity_dim=2)
        validate_planar_base_pose("base_pose", self.base_pose)


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

    def validate(self) -> None:
        """Validate PCS and pneumatic chamber parameters.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If inherited PCS fields are invalid; chamber arrays
                disagree in shape; radii, effective areas, or segment lengths
                are invalid; or chamber azimuths are non-finite or not uniformly
                distributed.
        """
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
