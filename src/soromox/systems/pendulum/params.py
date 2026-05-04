__all__ = ["PendulumParams", "TendonActuatedPendulumParams"]

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import (
    BaseArticulatedSoftRobotParams,
    BaseSystemParams,
    PassiveTendonParams,
)


class PendulumParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for planar pendulum chains.

    The leading axis indexes links/joints. ``moment_inertia`` and
    ``center_of_mass_length`` are per link, while stiffness and damping are
    generalized-coordinate matrices inherited from the articulated base class.
    """

    moment_inertia: Array
    length: Array
    center_of_mass_length: Array
    radius: Array


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
        passive_routing_matrix = jnp.asarray(self.passive_routing_matrix)
        if passive_routing_matrix.ndim != 2:
            raise ValueError("passive_routing_matrix must be two-dimensional.")
        if self.passive_tendon.num_tendons != passive_routing_matrix.shape[0]:
            raise ValueError(
                "passive_tendon must have one impedance entry per passive routing row."
            )
        n_links = self.body.mass.shape[0]
        if self.active_tendon_reference_configuration.shape != (n_links,):
            raise ValueError(
                "active_tendon_reference_configuration must have shape (num_links,)."
            )
        if self.passive_tendon_reference_configuration.shape != (n_links,):
            raise ValueError(
                "passive_tendon_reference_configuration must have shape (num_links,)."
            )
