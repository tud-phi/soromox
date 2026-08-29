"""McKibben-actuated UMArm articulated system.

This module implements the UMArm body as a SoRoMoX articulated system composed
with :class:`soromox.actuation.ArticulatedMcKibbenActuator`. The UMArm is a
pneumatically driven rigid-soft hybrid arm introduced by Zuo, Han, Li, Jamal,
and Bruder in "UMArm: Untethered, Modular, Portable, Soft Pneumatic Arm",
arXiv:2505.11476, https://doi.org/10.48550/arXiv.2505.11476.

The body specialization retains UMArm-specific rotor armature and cached-model
construction. McKibben geometry, actuator coordinates, moment matrices, axial
forces, and rendering geometry live on the installed actuator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.actuation.core import PassiveElement
from soromox.actuation.mckibben import ArticulatedMcKibbenActuator
from soromox.systems.articulated.articulated_soft_robot import ArticulatedSoftRobot
from soromox.systems.articulated.params import McKibbenActuatedUMArmParams
from soromox.utils.geometry import poses


class McKibbenActuatedUMArm(ArticulatedSoftRobot):
    """Spatial UMArm with rigid links and composable McKibben actuation.

    The arm is represented as a 12-DOF serial articulated chain, where each
    universal joint contributes two one-DOF revolute joints. The required
    :class:`~soromox.actuation.ArticulatedMcKibbenActuator` maps pressure inputs
    to generalized joint torques through work-conjugate volume coordinates.
    This keeps the UMArm body dynamics separate from the pneumatic transmission
    while preserving the original analytic model and channel ordering.

    References:
        Zuo, R., Han, D. H., Li, R., Jamal, S., & Bruder, D. (2025). UMArm:
        Untethered, Modular, Portable, Soft Pneumatic Arm. arXiv:2505.11476.
        https://doi.org/10.48550/arXiv.2505.11476

    Attributes:
        params: Typed articulated-body and armature parameters.
        joint_armature: Per-joint rotor armature added to dense inertia and the
            articulated-body recursion, shape ``(num_dofs,)``.
        actuators: One-element tuple containing the installed McKibben actuator.
            Its parameters own moving/fixed attachment points, effective-length
            offsets, thread lengths, fiber turns, and joint-pair indices.
        num_segments: Number of physical UMArm segments.
        num_mckibben_groups: Number of universal-joint actuator groups.
    """

    params: McKibbenActuatedUMArmParams
    joint_armature: Array
    num_segments: int = eqx.field(static=True)
    num_mckibben_groups: int = eqx.field(static=True)

    def __init__(
        self,
        params: McKibbenActuatedUMArmParams,
        *,
        actuator: ArticulatedMcKibbenActuator,
        passive_elements: PassiveElement | tuple[PassiveElement, ...] | None = (),
        **kwargs: Any,
    ) -> None:
        """Initialize a UMArm body with one McKibben actuator.

        Args:
            params: UMArm body dynamics and joint-armature parameters.
            actuator: McKibben pressure actuator installed on the body.
            passive_elements: Optional independent passive mechanics.
            **kwargs: Additional arguments forwarded to
                :class:`ArticulatedSoftRobot`.

        Raises:
            TypeError: If the body or actuator parameter types are invalid.
        """
        if not isinstance(params, McKibbenActuatedUMArmParams):
            raise TypeError("params must be a McKibbenActuatedUMArmParams instance.")
        if not isinstance(actuator, ArticulatedMcKibbenActuator):
            raise TypeError("actuator must be an ArticulatedMcKibbenActuator instance.")
        super().__init__(
            params,
            actuators=(actuator,),
            passive_elements=passive_elements,
            **kwargs,
        )
        self.joint_armature = jnp.asarray(params.joint_armature)
        self.num_mckibben_groups = actuator.params.transmission.num_groups
        self.num_segments = self.num_mckibben_groups // 2

    @classmethod
    def from_cached_parameters(
        cls, path: str | Path, **kwargs: Any
    ) -> McKibbenActuatedUMArm:
        """Build the complete UMArm from an existing cached parameter file.

        The cache schema remains unchanged. Body and actuator fields are split
        into their respective immutable parameter objects during construction.

        Args:
            path: Path to a UMArm ``.npz`` parameter file.
            **kwargs: Additional arguments forwarded to the constructor.

        Returns:
            A UMArm with the cached body and McKibben actuator installed.
        """
        if not kwargs.get("floating_base", False):
            kwargs.setdefault("base_pose", poses.spatial_mounting_pose("horizontal"))
        return cls(
            McKibbenActuatedUMArmParams.from_cached_npz(path),
            actuator=ArticulatedMcKibbenActuator.from_cached_npz(path),
            **kwargs,
        )

    @eqx.filter_jit
    def inertia_matrix(self, q: Array) -> Array:
        """Return dense generalized inertia including rotor armature.

        Args:
            q: Joint coordinates with shape ``(num_dofs,)``.

        Returns:
            Symmetric inertia matrix with shape ``(num_dofs, num_dofs)``.
        """
        return super().inertia_matrix(q) + jnp.diag(self.joint_armature)

    def _joint_armature(self, q: Array) -> Array:
        """Return rotor armature for the inherited ABA forward dynamics."""
        del q
        return self.joint_armature

    def with_params(self, params: McKibbenActuatedUMArmParams) -> McKibbenActuatedUMArm:
        """Replace body and armature parameters while preserving the actuator."""
        if not isinstance(params, McKibbenActuatedUMArmParams):
            raise TypeError("params must be a McKibbenActuatedUMArmParams instance.")
        updated = super().with_params(params)
        return eqx.tree_at(
            lambda robot: robot.joint_armature,
            updated,
            jnp.asarray(params.joint_armature),
        )


__all__ = ["McKibbenActuatedUMArm"]
