__all__ = [
    "DEFAULT_GRAVITY_MAGNITUDE",
    "BaseSystemParams",
    "BaseSoftRobotParams",
    "BaseContinuumSoftRobotParams",
    "BaseArticulatedSoftRobotParams",
    "validate_planar_base_pose",
    "validate_quaternion_base_pose",
]

from dataclasses import fields
from typing import Any, ClassVar, Literal

import equinox as eqx
from jax import Array
from jax import numpy as jnp
from jax.errors import ConcretizationTypeError, TracerBoolConversionError

DEFAULT_GRAVITY_MAGNITUDE = 9.81


def _validate_finite_array(
    name: str, value: Array, expected_shape: tuple[int, ...]
) -> Array:
    """Validate array shape and concrete finite values when available."""
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    try:
        all_finite = bool(jnp.isfinite(array).all())
    except (ConcretizationTypeError, TracerBoolConversionError):
        return array
    if not all_finite:
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_planar_base_pose(name: str, value: Array) -> None:
    """Validate a planar base pose.

    Args:
        name: Parameter name used in error messages.
        value: Planar pose array with shape ``(3,)`` in ``[theta, x, y]``
            order. ``theta`` is a right-handed rotation angle in radians about
            the out-of-plane z-axis. ``x`` and ``y`` are direct translation
            coordinates in the parent frame.

    Returns:
        None.

    Raises:
        ValueError: If the shape is not ``(3,)`` or any concrete entry is
            non-finite. During JAX tracing, only the static shape check is
            performed.
    """
    _validate_finite_array(name, value, (3,))


def validate_quaternion_base_pose(
    name: str,
    value: Array,
    expected_shape: tuple[int, ...],
    *,
    min_norm: float = 1e-12,
) -> None:
    """Validate a scalar-first quaternion base pose.

    Args:
        name: Parameter name used in error messages.
        value: Base pose array whose first four entries are a scalar-first
            Hamilton quaternion in ``[qw, qx, qy, qz]`` order. Spatial systems
            use shape ``(7,)`` with ``[qw, qx, qy, qz, x, y, z]``.
        expected_shape: Required full pose shape, typically ``(7,)``.
        min_norm: Minimum allowed Euclidean norm for the quaternion component.

    Concrete parameter objects are checked for finite entries and nonzero
    quaternion norm. During JAX tracing, only the static shape check is
    performed; transform helpers still avoid zero-norm division to keep traced
    code finite.

    Raises:
        ValueError: If the shape is wrong, any entry is non-finite, or the
            quaternion component is zero or numerically too small to normalize
            safely.
    """
    pose = _validate_finite_array(name, value, expected_shape)
    try:
        quaternion_norm = float(jnp.linalg.norm(pose[:4]))
    except (ConcretizationTypeError, TracerBoolConversionError):
        return
    if quaternion_norm <= min_norm:
        raise ValueError(
            f"{name} quaternion must have norm greater than {min_norm}, "
            f"got {quaternion_norm}."
        )


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
            value = self._normalize_replacement(name, value)
            updated = eqx.tree_at(
                lambda params, field_name=name: getattr(params, field_name),
                updated,
                value,
                is_leaf=lambda leaf: leaf is None,
            )
        updated.validate()
        return updated

    def _normalize_replacement(self, name: str, value: Any) -> Any:
        """Normalize a field replacement before rebuilding the PyTree."""
        return value

    def validate(self) -> None:
        """Validate parameter consistency."""
        return None

    def validate_against_structure(self, structure: Any) -> None:
        """Validate params against static construction choices."""
        self.validate()


class BaseSoftRobotParams(BaseSystemParams):
    """Common dynamic parameters for soft robot systems.

    ``base_pose`` and ``gravity`` are optional keyword-only constructor inputs.
    When omitted, the robot points upright and standard Earth gravity acts in
    the negative vertical world direction. Planar robots use
    ``base_pose = [theta, x, y]`` with 2D gravity, where ``theta`` is a
    right-handed angle in radians about the out-of-plane z-axis. Spatial robots use
    ``base_pose = [qw, qx, qy, qz, x, y, z]`` with 3D gravity. Spatial
    quaternions are scalar-first Hamilton quaternions, normalized before
    transform construction, and must have nonzero finite norm. Use
    :meth:`horizontal`, :meth:`upright`, or :meth:`hanging` for explicit common
    mounting configurations.
    """

    is_planar: ClassVar[bool | None] = None

    base_pose: Array | None = eqx.field(default=None, kw_only=True)
    gravity: Array | None = eqx.field(default=None, kw_only=True)

    def __check_init__(self) -> None:
        object.__setattr__(self, "base_pose", self._resolve_base_pose(self.base_pose))
        object.__setattr__(self, "gravity", self._resolve_gravity(self.gravity))

    @classmethod
    def _require_planarity(cls) -> bool:
        if cls.is_planar is None:
            raise TypeError(f"{cls.__name__} must declare is_planar as True or False.")
        return cls.is_planar

    @classmethod
    def _mounting_base_pose(
        cls,
        mounting: Literal["horizontal", "upright", "hanging"],
        base_position: Array | None = None,
    ) -> Array:
        is_planar = cls._require_planarity()
        position_dimension = 2 if is_planar else 3
        if base_position is None:
            position = jnp.zeros(position_dimension)
        else:
            position = _validate_finite_array(
                "base_position", base_position, (position_dimension,)
            )
            position = jnp.asarray(position)

        if is_planar:
            angles = {
                "horizontal": 0.0,
                "upright": jnp.pi / 2,
                "hanging": -jnp.pi / 2,
            }
            return jnp.concatenate([jnp.asarray([angles[mounting]]), position])

        sqrt_half = jnp.sqrt(jnp.asarray(0.5))
        quaternions = {
            "horizontal": jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            "upright": jnp.asarray([sqrt_half, 0.0, -sqrt_half, 0.0]),
            "hanging": jnp.asarray([sqrt_half, 0.0, sqrt_half, 0.0]),
        }
        return jnp.concatenate([quaternions[mounting], position])

    @classmethod
    def _default_gravity(cls) -> Array:
        if cls._require_planarity():
            return jnp.asarray([0.0, -DEFAULT_GRAVITY_MAGNITUDE])
        return jnp.asarray([0.0, 0.0, -DEFAULT_GRAVITY_MAGNITUDE])

    @classmethod
    def _resolve_base_pose(cls, base_pose: Array | None) -> Array:
        if base_pose is None:
            return cls._mounting_base_pose("upright")
        return jnp.asarray(base_pose)

    @classmethod
    def _resolve_gravity(cls, gravity: Array | None) -> Array:
        if gravity is None:
            return cls._default_gravity()
        return jnp.asarray(gravity)

    @classmethod
    def _from_mounting(
        cls,
        mounting: Literal["horizontal", "upright", "hanging"],
        *,
        base_position: Array | None = None,
        **kwargs: Any,
    ) -> "BaseSoftRobotParams":
        if "base_pose" in kwargs:
            raise TypeError(
                f"{cls.__name__}.{mounting}() does not accept base_pose; "
                "use base_position or the ordinary constructor instead."
            )
        return cls(base_pose=cls._mounting_base_pose(mounting, base_position), **kwargs)

    @classmethod
    def horizontal(
        cls, *, base_position: Array | None = None, **kwargs: Any
    ) -> "BaseSoftRobotParams":
        """Construct parameters with the backbone pointing along world +x."""
        return cls._from_mounting("horizontal", base_position=base_position, **kwargs)

    @classmethod
    def upright(
        cls, *, base_position: Array | None = None, **kwargs: Any
    ) -> "BaseSoftRobotParams":
        """Construct parameters pointing along world +y (planar) or +z (spatial)."""
        return cls._from_mounting("upright", base_position=base_position, **kwargs)

    @classmethod
    def hanging(
        cls, *, base_position: Array | None = None, **kwargs: Any
    ) -> "BaseSoftRobotParams":
        """Construct parameters pointing along world -y (planar) or -z (spatial)."""
        return cls._from_mounting("hanging", base_position=base_position, **kwargs)

    def _normalize_replacement(self, name: str, value: Any) -> Any:
        if name == "base_pose":
            return type(self)._resolve_base_pose(value)
        if name == "gravity":
            return type(self)._resolve_gravity(value)
        return value


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
