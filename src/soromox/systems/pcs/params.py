__all__ = [
    "PCSParams",
    "PlanarPCSParams",
    "TendonActuatedPCSParams",
    "TendonActuatedPlanarPCSParams",
    "PneumaticActuatedPlanarPCSParams",
    "ISupportParams",
]

import equinox as eqx
from jax import Array

from soromox.systems.params import (
    BaseContinuumSoftRobotParams,
    BaseSystemParams,
    LinearTendonRoutingParams,
    PassiveTendonParams,
)


class PCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the spatial PCS model.

    ``length``, ``radius``, material parameters, and density use a leading
    segment axis. ``damping_matrix`` is the full flattened strain damping matrix.
    ``base_pose`` is the SE(3) base pose vector used to initialize the base
    transform.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array
    base_pose: Array


class PlanarPCSParams(BaseContinuumSoftRobotParams):
    """Dynamic parameters for the planar PCS model.

    The leading axis of per-segment fields indexes planar constant-strain
    segments. ``base_angle`` stores the planar base orientation.
    """

    radius: Array
    young_modulus: Array
    shear_modulus: Array
    damping_matrix: Array
    base_angle: Array


class TendonActuatedPCSParams(BaseSystemParams):
    """Dynamic parameters for spatial tendon-actuated PCS.

    ``body`` contains the underlying PCS dynamic parameters. Active and passive
    tendon routing fields are batched by tendon; passive impedance fields are
    batched by passive tendon and must have the same leading length as
    ``passive_tendon_routing``.
    """

    body: PCSParams
    active_tendon_routing: LinearTendonRoutingParams
    passive_tendon_routing: LinearTendonRoutingParams = eqx.field(
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


class PneumaticActuatedPlanarPCSParams(PlanarPCSParams):
    """Dynamic parameters for pneumatic planar PCS.

    Chamber geometry arrays describe the per-segment chamber layout and can be
    updated without changing the strain layout as long as their shapes are
    unchanged.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_angle: Array
    chamber_distance: Array


class ISupportParams(PCSParams):
    """Dynamic parameters for the I-SUPPORT spatial pneumatic PCS model.

    The PCS body parameters are inherited. Chamber fields describe per-segment
    pneumatic chamber geometry and angular offsets.
    """

    chamber_inner_radius: Array
    chamber_outer_radius: Array
    chamber_distance: Array
    chamber_angle_offset: Array
