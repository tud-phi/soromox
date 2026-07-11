__all__ = ["GVSLinkParams", "GVSParams", "TendonActuatedGVSParams"]

from typing import ClassVar

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import (
    BaseSoftRobotParams,
    BaseSystemParams,
    BaseTendonRoutingParams,
    LinearTendonRoutingParams,
    PassiveTendonParams,
    validate_quaternion_base_pose,
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
        length = jnp.asarray(self.length)
        if len(length.shape) != 1:
            raise ValueError(
                "length must be one-dimensional with shape (num_segments,)."
            )
        n_segments = length.shape[0]
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
            value = jnp.asarray(getattr(self, name))
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
    ``base_pose`` uses scalar-first quaternion SE(3) coordinates
    ``[qw, qx, qy, qz, x, y, z]`` with nonzero finite quaternion norm. Omitting
    ``base_pose`` and ``gravity`` selects upright spatial mounting and
    negative-z Earth gravity.
    """

    is_planar: ClassVar[bool] = False

    link: GVSLinkParams
    reference_strain: Array
    joint_stiffness: Array

    def validate(self) -> None:
        self.link.validate()
        n_segments = self.link.length.shape[0]
        gravity = jnp.asarray(self.gravity)
        if gravity.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")
        validate_quaternion_base_pose("base_pose", self.base_pose, (7,))
        reference_strain = jnp.asarray(self.reference_strain)
        if reference_strain.shape != (n_segments, 6):
            raise ValueError(
                f"reference_strain must have shape ({n_segments}, 6), "
                f"got {reference_strain.shape}."
            )
        joint_stiffness = jnp.asarray(self.joint_stiffness)
        if (
            joint_stiffness.ndim != 3
            or joint_stiffness.shape[0] != n_segments
            or joint_stiffness.shape[1] != joint_stiffness.shape[2]
        ):
            raise ValueError(
                "joint_stiffness must have shape "
                f"({n_segments}, max_dof, max_dof), got {joint_stiffness.shape}."
            )

    def validate_against_structure(self, structure) -> None:
        """Validate dynamic GVS arrays against the static padded layout."""
        from soromox.systems.gvs.primitives import Basis, Joint
        from soromox.systems.gvs.structures import GVSStructure

        if not isinstance(structure, GVSStructure):
            raise TypeError("structure must be a GVSStructure instance.")
        if not structure.segments:
            raise ValueError("GVS requires at least one segment.")

        self.validate()
        n_segments = len(structure.segments)
        if self.link.length.shape != (n_segments,):
            raise ValueError(
                f"link.length must have shape ({n_segments},), "
                f"got {self.link.length.shape}."
            )
        if self.reference_strain.shape != (n_segments, 6):
            raise ValueError(
                "reference_strain must have shape "
                f"({n_segments}, 6), got {self.reference_strain.shape}."
            )

        max_num_gauss_points = structure.max_num_gauss_points
        if max_num_gauss_points is not None and max_num_gauss_points < 5:
            raise ValueError(
                f"max_num_gauss_points must be at least 5, got {max_num_gauss_points}."
            )

        dofs_joint = [
            Joint.DICT_JOINT_TYPE_DOF[segment.joint.type]
            for segment in structure.segments
        ]
        dofs_link = [
            int(
                Basis.DOF_BRANCHES[Basis.BASISTYPE_MAP[segment.basis.type]](
                    (
                        jnp.asarray(segment.basis.active),
                        jnp.asarray(segment.basis.orders),
                    )
                )
            )
            for segment in structure.segments
        ]
        real_max_dof = max(dofs_joint + dofs_link)
        max_dof = real_max_dof if structure.max_dof is None else structure.max_dof
        if max_dof < real_max_dof:
            raise ValueError(
                "max_dof must be greater than or equal to the maximum DOF in links "
                f"and joints ({real_max_dof}), but got {max_dof}."
            )

        expected_joint_stiffness_shape = (n_segments, max_dof, max_dof)
        if self.joint_stiffness.shape != expected_joint_stiffness_shape:
            raise ValueError(
                "joint_stiffness must have shape "
                f"{expected_joint_stiffness_shape}, got {self.joint_stiffness.shape}."
            )

        for idx, segment in enumerate(structure.segments):
            if segment.num_gauss_points < 5:
                raise ValueError(
                    "Each GVS segment requires at least 5 Gauss points; "
                    f"segment {idx} has {segment.num_gauss_points}."
                )
            if (
                max_num_gauss_points is not None
                and max_num_gauss_points < segment.num_gauss_points
            ):
                raise ValueError(
                    "max_num_gauss_points must be greater than or equal to the "
                    f"maximum segment num_gauss_points; segment {idx} has "
                    f"{segment.num_gauss_points}, got {max_num_gauss_points}."
                )


class TendonActuatedGVSParams(BaseSystemParams):
    """Dynamic parameters for tendon-actuated GVS.

    ``body`` stores the underlying GVS dynamic params. Active and passive routing
    objects are batched by tendon, so each tendon can use distinct linear routing
    parameters and attachment segment. ``passive_tendon`` stores per-passive-
    tendon impedance and rest-length offsets.
    """

    body: GVSParams
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

    def validate_against_structure(self, structure) -> None:
        """Validate tendon-actuated params against the GVS structure."""
        self.validate()
        self.body.validate_against_structure(structure)
        num_segments = len(structure.segments)
        self.active_tendon_routing.validate_attachment_segments(
            num_segments, "active_tendon_routing"
        )
        self.passive_tendon_routing.validate_attachment_segments(
            num_segments, "passive_tendon_routing"
        )
