from __future__ import annotations

__all__ = ["ArticulatedSoftRobotParams", "McKibbenActuatedUMArmParams"]

from pathlib import Path
from typing import ClassVar

import numpy as np
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import (
    BaseArticulatedSoftRobotParams,
    validate_quaternion_base_pose,
)


class ArticulatedSoftRobotParams(BaseArticulatedSoftRobotParams):
    """Dynamic parameters for spatial articulated soft robots.

    Per-joint screw axes and transforms define the dynamic link geometry used by
    the articulated model. The number of joints is fixed by array shapes.
    ``base_pose`` uses scalar-first quaternion SE(3) coordinates
    ``[qw, qx, qy, qz, x, y, z]`` with nonzero finite quaternion norm. Omitting
    ``base_pose`` and ``gravity`` selects upright mounting and negative-z Earth
    gravity.
    """

    is_planar: ClassVar[bool] = False

    joint_screw: Array
    parent_to_joint_transform: Array
    tip_position: Array
    center_of_mass_position: Array
    center_of_mass_inertia: Array
    radius: Array

    def __init__(
        self,
        *,
        joint_screw: Array,
        parent_to_joint_transform: Array,
        tip_position: Array,
        center_of_mass_position: Array,
        mass: Array,
        center_of_mass_inertia: Array,
        gravity: Array | None = None,
        joint_stiffness: Array,
        joint_damping: Array,
        joint_rest_configuration: Array,
        radius: Array,
        base_pose: Array | None = None,
    ) -> None:
        self.base_pose = self._resolve_base_pose(base_pose)
        self.gravity = self._resolve_gravity(gravity)
        self.mass = jnp.asarray(mass)
        self.joint_stiffness = jnp.asarray(joint_stiffness)
        self.joint_damping = jnp.asarray(joint_damping)
        self.joint_rest_configuration = jnp.asarray(joint_rest_configuration)
        self.joint_screw = jnp.asarray(joint_screw)
        self.parent_to_joint_transform = jnp.asarray(parent_to_joint_transform)
        self.tip_position = jnp.asarray(tip_position)
        self.center_of_mass_position = jnp.asarray(center_of_mass_position)
        self.center_of_mass_inertia = jnp.asarray(center_of_mass_inertia)
        self.radius = jnp.asarray(radius)

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
        validate_quaternion_base_pose("base_pose", self.base_pose, (7,))


class McKibbenActuatedUMArmParams(ArticulatedSoftRobotParams):
    """Dynamic parameters for the articulated UMArm body.

    These parameters contain the rigid-link dynamics, compliant-joint mechanics,
    visualization radii, base pose, gravity, and per-joint rotor armature. The
    McKibben attachment geometry and pneumatic constitutive parameters belong to
    :class:`soromox.actuation.ArticulatedMcKibbenActuatorParams`, so body and
    actuator parameters can be updated independently.
    """

    joint_armature: Array

    def __init__(
        self,
        *,
        joint_screw: Array,
        parent_to_joint_transform: Array,
        tip_position: Array,
        center_of_mass_position: Array,
        mass: Array,
        center_of_mass_inertia: Array,
        gravity: Array | None = None,
        joint_stiffness: Array,
        joint_damping: Array,
        joint_rest_configuration: Array,
        radius: Array,
        joint_armature: Array,
        base_pose: Array | None = None,
    ) -> None:
        super().__init__(
            base_pose=base_pose,
            joint_screw=joint_screw,
            parent_to_joint_transform=parent_to_joint_transform,
            tip_position=tip_position,
            center_of_mass_position=center_of_mass_position,
            mass=mass,
            center_of_mass_inertia=center_of_mass_inertia,
            gravity=gravity,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            joint_rest_configuration=joint_rest_configuration,
            radius=radius,
        )
        self.joint_armature = jnp.asarray(joint_armature)

    @classmethod
    def from_cached_npz(cls, path: str | Path) -> McKibbenActuatedUMArmParams:
        """Load the articulated UMArm body from the existing cache format."""
        with np.load(path) as data:
            parent_to_joint_transform = np.asarray(
                data["parent_to_joint_transform"]
            ).copy()
            tip_position = np.asarray(data["tip_position"])
            cls._normalize_cached_base_frame(parent_to_joint_transform, tip_position)
            armature_key = "joint_armature" if "joint_armature" in data else "armature"
            return cls(
                base_pose=jnp.array(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64
                ),
                joint_screw=jnp.asarray(data["joint_screw"]),
                parent_to_joint_transform=jnp.asarray(parent_to_joint_transform),
                tip_position=jnp.asarray(tip_position),
                center_of_mass_position=jnp.asarray(data["center_of_mass_position"]),
                center_of_mass_inertia=jnp.asarray(data["center_of_mass_inertia"]),
                radius=jnp.asarray(data["radius"]),
                mass=jnp.asarray(data["mass"]),
                gravity=jnp.asarray(data["gravity"]),
                joint_stiffness=jnp.asarray(data["joint_stiffness"]),
                joint_damping=jnp.asarray(data["joint_damping"]),
                joint_rest_configuration=jnp.asarray(data["joint_rest_configuration"]),
                joint_armature=jnp.asarray(data[armature_key]),
            )

    @classmethod
    def _normalize_cached_base_frame(
        cls,
        parent_to_joint_transform: np.ndarray,
        tip_position: np.ndarray,
    ) -> None:
        """Normalize legacy cached UMArm placement to the SoRoMoX base frame."""
        nonzero_tip_indices = np.flatnonzero(np.linalg.norm(tip_position, axis=1) > 0.0)
        if nonzero_tip_indices.size == 0:
            return
        first_tip = tip_position[int(nonzero_tip_indices[0])]
        proximal_direction = parent_to_joint_transform[0, :3, :3] @ first_tip
        proximal_direction = proximal_direction / np.linalg.norm(proximal_direction)
        base_translation = parent_to_joint_transform[0, :3, 3]
        if np.allclose(base_translation, 0.0) and np.allclose(
            proximal_direction, np.array([1.0, 0.0, 0.0])
        ):
            return
        rotation = cls._rotation_to_soromox_base_axis(proximal_direction)
        parent_to_joint_transform[0, :3, :3] = (
            rotation @ parent_to_joint_transform[0, :3, :3]
        )
        parent_to_joint_transform[0, :3, 3] = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _rotation_to_soromox_base_axis(source_direction: np.ndarray) -> np.ndarray:
        source = np.asarray(source_direction, dtype=np.float64)
        source = source / np.linalg.norm(source)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.allclose(source, target):
            return np.eye(3, dtype=np.float64)
        if np.allclose(source, -target):
            return np.diag(np.array([-1.0, 1.0, -1.0], dtype=np.float64))
        cross = np.cross(source, target)
        sin_angle = np.linalg.norm(cross)
        cos_angle = float(np.dot(source, target))
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return (
            np.eye(3, dtype=np.float64)
            + skew
            + (skew @ skew) * ((1.0 - cos_angle) / (sin_angle**2))
        )

    def validate(self) -> None:
        super().validate()
        num_links = int(jnp.asarray(self.joint_screw).shape[0])
        if jnp.asarray(self.joint_armature).shape != (num_links,):
            raise ValueError(
                f"joint_armature must have shape ({num_links},), got "
                f"{jnp.asarray(self.joint_armature).shape}."
            )
