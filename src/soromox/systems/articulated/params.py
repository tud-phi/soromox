__all__ = ["ArticulatedSoftRobotParams"]

from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseArticulatedSoftRobotParams


class ArticulatedSoftRobotParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for spatial articulated soft robots.

    Per-joint screw axes and transforms define the dynamic link geometry used by
    the articulated model. The number of joints is fixed by array shapes.
    """

    joint_screw: Array
    parent_to_joint_transform: Array
    tip_position: Array
    center_of_mass_position: Array
    center_of_mass_inertia: Array
    radius: Array

    def validate(self) -> None:
        joint_screw = jnp.asarray(self.joint_screw)
        if joint_screw.ndim != 2 or joint_screw.shape[1] != 6:
            raise ValueError(
                f"joint_screw must have shape (num_links, 6), got {joint_screw.shape}."
            )
        n_links = joint_screw.shape[0]
        if n_links < 1:
            raise ValueError(f"num_links must be at least 1, got {n_links}.")
        expected_shapes = {
            "parent_to_joint_transform": (n_links, 4, 4),
            "tip_position": (n_links, 3),
            "center_of_mass_position": (n_links, 3),
            "mass": (n_links,),
            "center_of_mass_inertia": (n_links, 3, 3),
            "gravity": (3,),
            "joint_stiffness": (n_links, n_links),
            "joint_damping": (n_links, n_links),
            "joint_rest_configuration": (n_links,),
            "radius": (n_links,),
        }
        for name, expected_shape in expected_shapes.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )
