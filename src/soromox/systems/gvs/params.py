__all__ = ["GVSLinkParams", "GVSParams", "TendonActuatedGVSParams"]

import equinox as eqx
from jax import Array

from soromox.systems.params import (
    BaseSoftRobotParams,
    BaseSystemParams,
    LinearTendonRoutingParams,
    PassiveTendonParams,
)


class GVSLinkParams(BaseSystemParams):
    """Dynamic per-link arrays for all GVS segments.

    Every field has leading shape ``(num_segments,)``; this is not a single-link
    object. It stores the numeric link values without duplicating the static
    cross-section family stored in ``GVSStructure.segments``. ``length`` follows
    the singular per-link naming convention used by the other fields. Reference
    strain is intentionally stored on ``GVSParams`` because it belongs to the
    strain basis state, not the link cross-section/material data.
    """

    length: Array
    young_modulus: Array
    poisson_ratio: Array
    density: Array
    damping_coefficient: Array
    radius_initial: Array
    radius_final: Array
    height_initial: Array
    height_final: Array
    width_initial: Array
    width_final: Array
    semi_major_initial: Array
    semi_major_final: Array
    semi_minor_initial: Array
    semi_minor_final: Array

    def validate(self) -> None:
        n_segments = self.length.shape[0]
        expected_shape = (n_segments,)
        for name in (
            "young_modulus",
            "poisson_ratio",
            "density",
            "damping_coefficient",
            "radius_initial",
            "radius_final",
            "height_initial",
            "height_final",
            "width_initial",
            "width_final",
            "semi_major_initial",
            "semi_major_final",
            "semi_minor_initial",
            "semi_minor_final",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )


class GVSParams(BaseSoftRobotParams):
    """Dynamic parameters for a GVS model.

    ``link`` contains per-segment link arrays. ``reference_strain`` has shape
    ``(num_segments, 6)`` and is kept here, rather than in ``GVSLinkParams``,
    because it parameterizes the strain basis/reference configuration rather
    than link geometry or material properties. ``joint_stiffness`` has shape
    ``(num_segments, max_dof, max_dof)`` and is padded to the static GVS layout.
    """

    link: GVSLinkParams
    base_pose: Array
    reference_strain: Array
    joint_stiffness: Array

    def validate(self) -> None:
        self.link.validate()
        n_segments = self.link.length.shape[0]
        if self.reference_strain.shape[0] != n_segments:
            raise ValueError(
                "reference_strain must have one leading entry per GVS segment."
            )
        if self.joint_stiffness.shape[0] != n_segments:
            raise ValueError(
                "joint_stiffness must have one leading entry per GVS segment."
            )


class TendonActuatedGVSParams(BaseSystemParams):
    """Dynamic parameters for tendon-actuated GVS.

    ``body`` stores the underlying GVS dynamic params. Active and passive routing
    objects are batched by tendon, so each tendon can use distinct linear routing
    parameters and attachment segment. ``passive_tendon`` stores per-passive-
    tendon impedance and rest-length offsets.
    """

    body: GVSParams
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
