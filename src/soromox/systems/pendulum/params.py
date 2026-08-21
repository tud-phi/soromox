__all__ = ["PendulumParams"]

from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import (
    BaseArticulatedSoftRobotParams,
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

    def validate_structure(self) -> None:
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
        if jnp.asarray(self.base_pose).shape != (3,):
            raise ValueError(
                f"base_pose must have shape (3,), got {self.base_pose.shape}."
            )

    def validate_values(self) -> None:
        """Validate eager pendulum parameter values."""
        validate_planar_base_pose("base_pose", self.base_pose)
