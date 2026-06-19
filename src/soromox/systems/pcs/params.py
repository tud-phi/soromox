__all__ = [
    "PCSParams",
    "PlanarPCSParams",
    "TendonActuatedPCSParams",
    "TendonActuatedPlanarPCSParams",
    "PressureActuatedPlanarPCSParams",
    "ISupportParams",
]

import equinox as eqx
from jax import Array

from soromox.systems.params import (
    BaseContinuumSoftRobotParams,
    BaseSystemParams,
    BaseTendonRoutingParams,
    LinearTendonRoutingParams,
    PassiveTendonParams,
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
    segment axis. ``damping_matrix`` is the full flattened strain damping matrix.
    ``base_pose`` is the scalar-first quaternion SE(3) base pose vector
    ``[qw, qx, qy, qz, x, y, z]`` used to initialize the base transform. The
    quaternion is normalized before use and must have nonzero finite norm.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array

    def validate(self) -> None:
        n_segments = _validate_continuum_base(self, strain_dim=6, gravity_dim=3)
        _require_shape("radius", self.radius, (n_segments,))
        _require_shape("young_modulus", self.young_modulus, (n_segments,))
        _require_shape("shear_modulus", self.shear_modulus, (n_segments,))
        _require_shape(
            "damping_matrix", self.damping_matrix, (6 * n_segments, 6 * n_segments)
        )
        validate_quaternion_base_pose("base_pose", self.base_pose, (7,))


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of per-segment fields indexes planar constant-strain
    segments. ``base_pose`` stores the planar pose ``[theta, x, y]`` with shape
    ``(3,)``. ``theta`` is a right-handed angle in radians about the
    out-of-plane z-axis, and ``x``/``y`` are direct translations in the parent
    frame.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array

    def validate(self) -> None:
        n_segments = _validate_continuum_base(self, strain_dim=3, gravity_dim=2)
        _require_shape("radius", self.radius, (n_segments,))
        _require_shape("young_modulus", self.young_modulus, (n_segments,))
        _require_shape("shear_modulus", self.shear_modulus, (n_segments,))
        _require_shape(
            "damping_matrix", self.damping_matrix, (3 * n_segments, 3 * n_segments)
        )
        validate_planar_base_pose("base_pose", self.base_pose)


class TendonActuatedPCSParams(BaseSystemParams):
    """Dynamic parameters for spatial tendon-actuated PCS.

    ``body`` contains the underlying PCS dynamic parameters. Active and passive
    tendon routing fields are batched by tendon; passive impedance fields are
    batched by passive tendon and must have the same leading length as
    ``passive_tendon_routing``.
    """

    body: PCSParams
    active_tendon_routing: BaseTendonRoutingParams
    passive_tendon_routing: BaseTendonRoutingParams = eqx.field(
        default_factory=LinearTendonRoutingParams.empty
    )
    passive_tendon: PassiveTendonParams = eqx.field(
        default_factory=PassiveTendonParams.empty
    )

    def validate(self) -> None:
        self.body.validate()
        self.active_tendon_routing.validate()
        self.passive_tendon_routing.validate()
        self.passive_tendon.validate()
        if self.passive_tendon.num_tendons != self.passive_tendon_routing.num_tendons:
            raise ValueError(
                "passive_tendon must have one impedance entry per "
                "passive_tendon_routing entry."
            )


class TendonActuatedPlanarPCSParams(PlanarPCSParams):
    """Dynamic parameters for parallel-routed planar tendon PCS.

    ``tendon_distance`` has shape ``(num_segments, num_tendons_per_segment)``.
    Each row defines the offsets of the parallel tendons routed through that
    segment.
    """

    tendon_distance: Array

    def validate(self) -> None:
        super().validate()
        n_segments = self.length.shape[0]
        if (
            self.tendon_distance.ndim != 2
            or self.tendon_distance.shape[0] != n_segments
        ):
            raise ValueError(
                "tendon_distance must have shape "
                f"({n_segments}, num_tendons_per_segment), got {self.tendon_distance.shape}."
            )


class PressureActuatedPlanarPCSParams(PlanarPCSParams):
    """Dynamic parameters for pressure-actuated planar PCS.

    Chamber geometry arrays describe the per-segment chamber layout and can be
    updated without changing the strain layout as long as their shapes are
    unchanged.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_angle: Array
    chamber_distance: Array

    def validate(self) -> None:
        super().validate()
        n_segments = self.length.shape[0]
        for name in (
            "chamber_inner_radius",
            "chamber_outer_radius",
            "chamber_angle",
            "chamber_distance",
        ):
            _require_shape(name, getattr(self, name), (n_segments,))


class ISupportParams(PCSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic PCS model.

    The PCS body parameters are inherited. Chamber fields describe per-segment
    pneumatic chamber geometry and angular offsets.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_distance: Array
    chamber_angle_offset: Array

    def validate(self) -> None:
        super().validate()
        n_segments = self.length.shape[0]
        for name in (
            "chamber_inner_radius",
            "chamber_outer_radius",
            "chamber_distance",
            "chamber_angle_offset",
        ):
            _require_shape(name, getattr(self, name), (n_segments,))
