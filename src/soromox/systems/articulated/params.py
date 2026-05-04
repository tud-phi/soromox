__all__ = ["ArticulatedSoftRobotParams"]

from jax import Array

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
