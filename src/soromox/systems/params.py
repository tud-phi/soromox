__all__ = [
    "BaseSystemParams",
    "BaseSoftRobotParams",
    "BaseContinuumSoftRobotParams",
    "BaseArticulatedSoftRobotParams",
    "BaseTendonRoutingParams",
    "LinearTendonRoutingParams",
    "PassiveTendonParams",
]

from dataclasses import fields
from typing import Any

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp


class BaseSystemParams(eqx.Module):
    """Base class for dynamic system parameters stored as JAX PyTrees."""

    def replace(self, **updates: Any) -> "BaseSystemParams":
        """Return a copy with selected fields replaced."""
        valid_names = {field.name for field in fields(self)}
        unknown_names = set(updates) - valid_names
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise KeyError(f"Unknown parameter field(s): {unknown}")

        updated = self
        for name, value in updates.items():
            updated = eqx.tree_at(
                lambda params, field_name=name: getattr(params, field_name),
                updated,
                value,
            )
        updated.validate()
        return updated

    def validate(self) -> None:
        """Validate parameter consistency."""
        return None


class BaseSoftRobotParams(BaseSystemParams):
    """Common dynamic parameters for soft robot systems.

    ``gravity`` is a JAX array in the dimensional convention of the concrete
    model, for example 2D planar gravity or 3D spatial gravity.
    """

    gravity: Array


class BaseContinuumSoftRobotParams(BaseSoftRobotParams):
    """Shared dynamic parameters for continuum soft robots.

    Field names denote one segment's physical quantity; the leading axis stores
    the segment batch. ``reference_strain`` contains the flattened per-segment
    reference strain used by PCS-style continuum models.
    """

    length: Array
    density: Array
    reference_strain: Array


class BaseArticulatedSoftRobotParams(BaseSoftRobotParams):
    """Shared dynamic parameters for articulated systems.

    The leading axis indexes joints/links. Stiffness, damping, and reference
    coordinates are dynamic arrays so optimization can update them without
    changing the system structure. ``joint_rest_configuration`` is the joint
    coordinate at which the elastic joint force is zero.
    """

    mass: Array
    joint_stiffness: Array
    joint_damping: Array
    joint_rest_configuration: Array


class BaseTendonRoutingParams(BaseSystemParams):
    """Base class for batched tendon routing parameter sets.

    A tendon routing params object represents all tendons of one routing family.
    The leading axis of every array indexes tendon number; changing this axis
    changes the actuator layout and requires model reconstruction.
    """

    @property
    def num_tendons(self) -> int:
        raise NotImplementedError


class LinearTendonRoutingParams(BaseTendonRoutingParams):
    """Batched linear routing parameters for active or passive tendons.

    Each field has shape ``(num_tendons,)``. Tendon ``k`` has cross-section
    offsets ``y_intercept[k] + y_slope[k] * s`` and
    ``z_intercept[k] + z_slope[k] * s`` and attaches to
    ``attachment_segment_index[k]``. Systems use ``jax.vmap`` over the leading
    tendon axis, so every tendon can have distinct routing parameters.
    """

    y_intercept: Array
    y_slope: Array
    z_intercept: Array
    z_slope: Array
    attachment_segment_index: Array

    @classmethod
    def empty(cls) -> "LinearTendonRoutingParams":
        return cls(
            y_intercept=jnp.array([], dtype=jnp.float64),
            y_slope=jnp.array([], dtype=jnp.float64),
            z_intercept=jnp.array([], dtype=jnp.float64),
            z_slope=jnp.array([], dtype=jnp.float64),
            attachment_segment_index=jnp.array([], dtype=jnp.int32),
        )

    @property
    def num_tendons(self) -> int:
        return int(self.y_intercept.shape[0])

    def validate(self) -> None:
        if len(self.y_intercept.shape) != 1:
            raise ValueError(
                "LinearTendonRoutingParams fields must be one-dimensional with "
                "shape (num_tendons,)."
            )
        n_tendons = self.y_intercept.shape[0]
        expected_shape = (n_tendons,)
        for name in (
            "y_slope",
            "z_intercept",
            "z_slope",
            "attachment_segment_index",
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

    def routing_for_tendon(self, index: Array) -> "LinearTendonRoutingParams":
        """Return the scalar-field PyTree for one tendon index."""
        return jax.tree.map(lambda value: value[index], self)


class PassiveTendonParams(BaseSystemParams):
    """Batched passive tendon impedance and rest-length parameters.

    Each field has shape ``(num_passive_tendons,)``. Passive tendon ``k`` uses
    ``stiffness[k]``, ``damping[k]``, and ``rest_length_offset[k]``. The length
    must match the paired passive routing params.
    """

    stiffness: Array
    damping: Array
    rest_length_offset: Array

    @classmethod
    def empty(cls) -> "PassiveTendonParams":
        return cls(
            stiffness=jnp.array([], dtype=jnp.float64),
            damping=jnp.array([], dtype=jnp.float64),
            rest_length_offset=jnp.array([], dtype=jnp.float64),
        )

    @property
    def num_tendons(self) -> int:
        return int(self.stiffness.shape[0])

    def validate(self) -> None:
        if len(self.stiffness.shape) != 1:
            raise ValueError(
                "PassiveTendonParams fields must be one-dimensional with shape "
                "(num_passive_tendons,)."
            )
        n_tendons = self.stiffness.shape[0]
        expected_shape = (n_tendons,)
        for name in ("damping", "rest_length_offset"):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )
