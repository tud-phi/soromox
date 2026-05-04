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
from jax import Array
from jax import numpy as jnp


def _attachment_segment_index(value: Any) -> int | tuple[int, ...]:
    """Convert concrete tendon attachment indices to static Python metadata."""
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tuple):
        return tuple(int(idx) for idx in value)
    if isinstance(value, list):
        return tuple(int(idx) for idx in value)

    array = jnp.asarray(value)
    if array.ndim == 0:
        return int(array.item())
    if array.ndim != 1:
        raise ValueError(
            "attachment_segment_index must be a scalar or one-dimensional sequence."
        )
    return tuple(int(idx) for idx in array.tolist())


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

    def validate_against_structure(self, structure: Any) -> None:
        """Validate params against static construction choices."""
        self.validate()


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

    @property
    def attachment_segment_indices(self) -> tuple[int, ...]:
        """Attachment segment indices as concrete topology metadata."""
        raise NotImplementedError

    @property
    def attachment_segment_index_array(self) -> Array:
        """Attachment segment indices as a JAX array for vectorized computations."""
        return jnp.asarray(self.attachment_segment_indices, dtype=jnp.int32)

    def validate_attachment_segments(self, num_segments: int, name: str) -> None:
        """Validate static attachment indices against a concrete robot layout."""
        for idx in self.attachment_segment_indices:
            if idx >= num_segments:
                raise ValueError(
                    f"{name}.attachment_segment_index values must be strictly lower "
                    f"than the number of segments of the robot. Got {idx}; "
                    f"num_segments = {num_segments}."
                )

    def assert_same_attachment_segments(
        self, other: "BaseTendonRoutingParams", name: str
    ) -> None:
        """Reject runtime topology changes hidden inside same-shape params."""
        if self.attachment_segment_indices != other.attachment_segment_indices:
            raise ValueError(
                f"Changing {name}.attachment_segment_index changes tendon topology; "
                "construct a new system."
            )


class LinearTendonRoutingParams(BaseTendonRoutingParams):
    """Batched linear routing parameters for active or passive tendons.

    The continuous routing fields have shape ``(num_tendons,)``. Tendon ``k``
    has cross-section offsets ``y_intercept[k] + y_slope[k] * s`` and
    ``z_intercept[k] + z_slope[k] * s``. ``attachment_segment_index`` is static
    topology metadata: changing it changes which body segments the tendon spans
    and requires model reconstruction.
    """

    y_intercept: Array
    y_slope: Array
    z_intercept: Array
    z_slope: Array
    attachment_segment_index: int | tuple[int, ...] = eqx.field(
        static=True, converter=_attachment_segment_index
    )

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

    @property
    def attachment_segment_indices(self) -> tuple[int, ...]:
        """Attachment segment indices as a tuple, including single-tendon params."""
        if isinstance(self.attachment_segment_index, tuple):
            return self.attachment_segment_index
        return (int(self.attachment_segment_index),)

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
        ):
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )
        indices = self.attachment_segment_indices
        if len(indices) != n_tendons:
            raise ValueError(
                "attachment_segment_index must contain one entry per tendon; "
                f"expected {n_tendons}, got {len(indices)}."
            )
        for idx in indices:
            if idx < 0:
                raise ValueError(
                    f"attachment_segment_index values must be non-negative; got {idx}."
                )

    def replace(self, **updates: Any) -> "LinearTendonRoutingParams":
        """Return a copy with selected dynamic routing fields replaced."""
        if "attachment_segment_index" in updates:
            new_indices = _attachment_segment_index(
                updates.pop("attachment_segment_index")
            )
            if new_indices != self.attachment_segment_index:
                raise ValueError(
                    "Changing attachment_segment_index changes tendon topology; "
                    "construct a new system."
                )
        return super().replace(**updates)

    def routing_for_tendon(self, index: Array) -> "LinearTendonRoutingParams":
        """Return the scalar-field PyTree for one tendon index."""
        index = int(index)
        return LinearTendonRoutingParams(
            y_intercept=self.y_intercept[index],
            y_slope=self.y_slope[index],
            z_intercept=self.z_intercept[index],
            z_slope=self.z_slope[index],
            attachment_segment_index=self.attachment_segment_indices[index],
        )


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
