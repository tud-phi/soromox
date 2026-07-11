__all__ = ["PendulumParams", "TendonActuatedPendulumParams"]

from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import (
    BaseArticulatedSoftRobotParams,
    BaseSystemParams,
    PassiveTendonParams,
    validate_planar_base_pose,
)


class PendulumParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for planar pendulum chains.

    The leading axis indexes links/joints. ``moment_inertia`` and
    ``center_of_mass_length`` are per link, while stiffness and damping are
    generalized-coordinate matrices inherited from the articulated base class.
    ``base_pose`` stores the planar pose ``[theta, x, y]`` with shape ``(3,)``.
    ``theta`` is a right-handed angle in radians about the out-of-plane z-axis,
    and ``x``/``y`` are direct translations in the parent frame. Omitting
    ``base_pose`` and ``gravity`` selects upright mounting and negative-y Earth
    gravity.
    """

    is_planar: ClassVar[bool] = True

    moment_inertia: Array
    length: Array
    center_of_mass_length: Array
    radius: Array

    def validate(self) -> None:
        mass = jnp.asarray(self.mass)
        if len(mass.shape) != 1:
            raise ValueError("mass must be one-dimensional with shape (num_links,).")
        n_links = mass.shape[0]
        if n_links < 1:
            raise ValueError(f"num_links must be at least 1, got {n_links}.")
        for name in (
            "moment_inertia",
            "length",
            "center_of_mass_length",
            "joint_rest_configuration",
            "radius",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (n_links,):
                raise ValueError(
                    f"{name} must have shape ({n_links},), got {value.shape}."
                )
        for name in ("joint_stiffness", "joint_damping"):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (n_links, n_links):
                raise ValueError(
                    f"{name} must have shape ({n_links}, {n_links}), got {value.shape}."
                )
        gravity = jnp.asarray(self.gravity)
        if gravity.shape != (2,):
            raise ValueError(f"gravity must have shape (2,), got {gravity.shape}.")
        validate_planar_base_pose("base_pose", self.base_pose)


class TendonActuatedPendulumParams(BaseSystemParams):
    """Dynamic parameters for tendon-actuated pendulum chains.

    ``body`` stores pendulum dynamics. Active/passive routing matrices map joint
    coordinates to tendon displacements. ``active_tendon_reference_configuration``
    and ``passive_tendon_reference_configuration`` are joint-space configurations
    with shape ``(num_links,)`` at which the respective tendon displacement
    contribution ``R @ (q - q_ref)`` is zero. Passive tendon impedance fields are
    batched by passive tendon.
    """

    body: PendulumParams
    active_routing_matrix: Array
    passive_routing_matrix: Array
    active_tendon_reference_configuration: Array
    passive_tendon_reference_configuration: Array
    passive_tendon: PassiveTendonParams

    def validate(self) -> None:
        self.body.validate()
        self.passive_tendon.validate()
        active_routing_matrix = jnp.asarray(self.active_routing_matrix)
        if active_routing_matrix.ndim != 2:
            raise ValueError("active_routing_matrix must be two-dimensional.")
        passive_routing_matrix = jnp.asarray(self.passive_routing_matrix)
        if passive_routing_matrix.ndim != 2:
            raise ValueError("passive_routing_matrix must be two-dimensional.")
        if self.passive_tendon.num_tendons != passive_routing_matrix.shape[0]:
            raise ValueError(
                "passive_tendon must have one impedance entry per passive routing row."
            )
        n_links = self.body.mass.shape[0]
        if active_routing_matrix.shape[1] != n_links:
            raise ValueError("active_routing_matrix must have one column per link.")
        if passive_routing_matrix.shape[1] != n_links:
            raise ValueError("passive_routing_matrix must have one column per link.")
        if jnp.asarray(self.active_tendon_reference_configuration).shape != (n_links,):
            raise ValueError(
                "active_tendon_reference_configuration must have shape (num_links,)."
            )
        if jnp.asarray(self.passive_tendon_reference_configuration).shape != (n_links,):
            raise ValueError(
                "passive_tendon_reference_configuration must have shape (num_links,)."
            )
