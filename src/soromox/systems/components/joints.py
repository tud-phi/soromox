"""Shared joint construction specifications and runtime parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jax import Array

from soromox.systems.params import BaseSystemParams

from .links import _validate_symmetric

__all__ = ["JointParams", "JointSpec", "JointType", "JOINT_DOF", "joint_dof"]

JointType = Literal[
    "revolute",
    "prismatic",
    "helical",
    "cylindrical",
    "planar",
    "spherical",
    "free",
    "fixed",
]

JOINT_DOF: dict[str, int] = {
    "revolute": 1,
    "prismatic": 1,
    "helical": 1,
    "cylindrical": 2,
    "planar": 3,
    "spherical": 3,
    "free": 6,
    "fixed": 0,
}


def joint_dof(joint_type: str) -> int:
    """Return the active-coordinate dimension of a supported joint type.

    Args:
        joint_type: Lower-case joint family name.

    Returns:
        Number of active joint coordinates.

    Raises:
        ValueError: If ``joint_type`` is unknown.
    """
    try:
        return JOINT_DOF[joint_type]
    except KeyError as exc:
        raise ValueError(f"Unknown joint type {joint_type!r}.") from exc


class JointParams(BaseSystemParams):
    """Canonical batched generalized joint matrices.

    Attributes:
        stiffness: Padded joint stiffness matrices with shape
            ``(num_joints, generalized_dimension, generalized_dimension)``.
        damping: Padded joint damping matrices with the same shape as
            ``stiffness``.
    """

    stiffness: Array
    damping: Array

    def __check_init__(self) -> None:
        object.__setattr__(self, "stiffness", jnp.asarray(self.stiffness))
        object.__setattr__(self, "damping", jnp.asarray(self.damping))
        self.validate()

    def _normalize_replacement(self, name: str, value: object) -> object:
        """Normalize replacement joint-matrix values to JAX arrays."""
        if name in ("stiffness", "damping"):
            return jnp.asarray(value)
        return value

    def validate(self) -> None:
        """Validate joint matrix shapes, finiteness, and symmetry.

        Returns:
            None.

        Raises:
            ValueError: If stiffness is not a batch of square matrices, damping
                has a different shape, or either matrix batch is non-finite or
                asymmetric.
        """
        stiffness = jnp.asarray(self.stiffness)
        damping = jnp.asarray(self.damping)
        if stiffness.ndim != 3 or stiffness.shape[1] != stiffness.shape[2]:
            raise ValueError(
                "joint stiffness must have shape "
                "(num_joints, generalized_dimension, generalized_dimension)."
            )
        if damping.shape != stiffness.shape:
            raise ValueError(
                f"joint damping must have shape {stiffness.shape}, got {damping.shape}."
            )
        _validate_symmetric("joint stiffness", stiffness)
        _validate_symmetric("joint damping", damping)


@dataclass
class JointSpec:
    """Kinematic and generalized viscoelastic specification of one joint.

    Attributes:
        type: Joint family name.
        axis: Axis used by revolute, prismatic, helical, and cylindrical joints.
        plane: Motion plane used by planar joints.
        pitch: Translation per radian for a helical joint.
        stiffness: Stiffness matrix in active joint coordinates. An empty value
            requests a zero matrix.
        damping: Damping matrix in active joint coordinates. An empty value
            requests a zero matrix.
    """

    type: JointType
    axis: Literal["x", "y", "z"] = "x"
    plane: Literal["xy", "yz", "xz"] = "xy"
    pitch: float = 0.0
    stiffness: Array | list = field(default_factory=list)
    damping: Array | list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.type = self.type.lower()  # type: ignore[assignment]
        dof = joint_dof(self.type)
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of 'x', 'y', or 'z'.")
        if self.plane not in ("xy", "yz", "xz"):
            raise ValueError("plane must be one of 'xy', 'yz', or 'xz'.")
        for name in ("stiffness", "damping"):
            value = jnp.asarray(getattr(self, name))
            if value.size == 0:
                continue
            if value.shape != (dof, dof):
                raise ValueError(
                    f"joint {name} must have shape ({dof}, {dof}), got {value.shape}."
                )
            _validate_symmetric(f"joint {name}", value)

    @classmethod
    def fixed(cls) -> JointSpec:
        """Create a zero-degree-of-freedom fixed joint.

        Returns:
            A fixed-joint specification with zero stiffness and damping.
        """
        return cls(type="fixed")

    @classmethod
    def revolute(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a one-degree-of-freedom revolute joint.

        Args:
            axis: Rotation axis in the joint material frame.
            stiffness: Optional ``(1, 1)`` stiffness matrix in active
                coordinates. ``None`` produces zero stiffness.
            damping: Optional ``(1, 1)`` damping matrix in active coordinates.
                ``None`` produces zero damping.

        Returns:
            A revolute-joint specification.

        Raises:
            ValueError: If the axis is invalid or a matrix has an invalid shape,
                non-finite entries, or asymmetry.
        """
        return cls(
            type="revolute",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def prismatic(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a one-degree-of-freedom prismatic joint.

        Args:
            axis: Translation axis in the joint material frame.
            stiffness: Optional ``(1, 1)`` active-coordinate stiffness matrix.
            damping: Optional ``(1, 1)`` active-coordinate damping matrix.

        Returns:
            A prismatic-joint specification.

        Raises:
            ValueError: If the axis or either matrix is invalid.
        """
        return cls(
            type="prismatic",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def helical(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        pitch: float = 0.0,
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a one-degree-of-freedom helical joint.

        Args:
            axis: Coupled rotation and translation axis.
            pitch: Translation per radian of rotation.
            stiffness: Optional ``(1, 1)`` active-coordinate stiffness matrix.
            damping: Optional ``(1, 1)`` active-coordinate damping matrix.

        Returns:
            A helical-joint specification.

        Raises:
            ValueError: If the axis or either matrix is invalid.
        """
        return cls(
            type="helical",
            axis=axis,
            pitch=pitch,
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def cylindrical(
        cls,
        axis: Literal["x", "y", "z"] = "x",
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a cylindrical joint with rotation and translation.

        Args:
            axis: Shared rotation and translation axis.
            stiffness: Optional ``(2, 2)`` active-coordinate stiffness matrix.
            damping: Optional ``(2, 2)`` active-coordinate damping matrix.

        Returns:
            A two-degree-of-freedom cylindrical-joint specification.

        Raises:
            ValueError: If the axis or either matrix is invalid.
        """
        return cls(
            type="cylindrical",
            axis=axis,
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def planar(
        cls,
        plane: Literal["xy", "yz", "xz"] = "xy",
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a three-degree-of-freedom planar joint.

        Args:
            plane: Plane in which the joint translates and rotates.
            stiffness: Optional ``(3, 3)`` active-coordinate stiffness matrix.
            damping: Optional ``(3, 3)`` active-coordinate damping matrix.

        Returns:
            A planar-joint specification.

        Raises:
            ValueError: If the plane or either matrix is invalid.
        """
        return cls(
            type="planar",
            plane=plane,
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def spherical(
        cls,
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a three-degree-of-freedom spherical joint.

        Args:
            stiffness: Optional ``(3, 3)`` active-coordinate stiffness matrix.
            damping: Optional ``(3, 3)`` active-coordinate damping matrix.

        Returns:
            A spherical-joint specification.

        Raises:
            ValueError: If either matrix is invalid.
        """
        return cls(
            type="spherical",
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )

    @classmethod
    def free(
        cls,
        stiffness: Array | list | None = None,
        damping: Array | list | None = None,
    ) -> JointSpec:
        """Create a free six-degree-of-freedom spatial joint.

        Args:
            stiffness: Optional ``(6, 6)`` active-coordinate stiffness matrix.
            damping: Optional ``(6, 6)`` active-coordinate damping matrix.

        Returns:
            A free-joint specification.

        Raises:
            ValueError: If either matrix is invalid.
        """
        return cls(
            type="free",
            stiffness=[] if stiffness is None else stiffness,
            damping=[] if damping is None else damping,
        )
