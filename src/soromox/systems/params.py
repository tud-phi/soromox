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
from typing import Any, ClassVar, final

import equinox as eqx
from jax import Array, core, ensure_compile_time_eval, tree_util
from jax import numpy as jnp

DEFAULT_GRAVITY_MAGNITUDE = 9.81


def _contains_tracer(tree: Any) -> bool:
    """Return whether a parameter PyTree is currently under a JAX transform."""
    return any(isinstance(leaf, core.Tracer) for leaf in tree_util.tree_leaves(tree))


def _validate_finite_array(
    name: str, value: Array, expected_shape: tuple[int, ...]
) -> Array:
    """Validate an eager array's shape and finite values."""
    array = jnp.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}.")
    if not bool(jnp.isfinite(array).all()):
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
        ValueError: If the shape is not ``(3,)`` or any entry is non-finite.
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

    Parameter objects are checked for finite entries and nonzero quaternion
    norm. Transform helpers still avoid zero-norm division to keep runtime code
    finite.

    Returns:
        ``None`` after successful validation.

    Raises:
        ValueError: If the shape is wrong, any entry is non-finite, or the
            quaternion component is zero or numerically too small to normalize
            safely.
    """
    pose = _validate_finite_array(name, value, expected_shape)
    quaternion_norm = float(jnp.linalg.norm(pose[:4]))
    if quaternion_norm <= min_norm:
        raise ValueError(
            f"{name} quaternion must have norm greater than {min_norm}, "
            f"got {quaternion_norm}."
        )


class BaseSystemParams(eqx.Module):
    """Base class for dynamic system parameters stored as JAX PyTrees."""

    def replace(self, **updates: Any) -> "BaseSystemParams":
        """Return an immutable copy with selected fields replaced.

        Args:
            **updates: Parameter field names and their replacement values.
                Nested parameter components should first be replaced with their
                own ``replace`` method, then supplied as a top-level field.

        Returns:
            A validated parameter PyTree of the same concrete class.

        Raises:
            KeyError: If any update names an unknown parameter field.
            TypeError: If a replacement cannot be inserted into the PyTree.
            ValueError: If validation of the resulting parameters fails.
        """
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
        updated.validate_for_update()
        return updated

    def _normalize_replacement(self, name: str, value: Any) -> Any:
        """Normalize a field replacement before rebuilding the PyTree."""
        return value

    def validate_structure(self) -> None:
        """Validate statically observable parameter structure.

        Returns:
            ``None``. Subclasses override this hook to raise on invalid types,
            shapes, PyTree layout, or static topology. Implementations must be
            safe when array leaves are JAX tracers.
        """
        return None

    def validate_values(self) -> None:
        """Validate eager value-dependent parameter invariants.

        Returns:
            ``None``. Subclasses override this hook for finiteness, physical
            domains, symmetry, and other checks requiring concrete values.
        """
        return None

    @final
    def validate(self) -> None:
        """Validate both parameter structure and eager numeric values."""
        self.validate_structure()
        self.validate_values()

    @final
    def validate_for_update(self) -> None:
        """Validate values when concrete and structure when dynamically traced."""
        self.validate_structure()
        if _contains_tracer(self):
            return
        with ensure_compile_time_eval():
            self.validate_values()

    def validate_structure_compatibility(self, structure: Any) -> None:
        """Validate statically observable compatibility with a model structure."""
        del structure

    def validate_value_compatibility(self, structure: Any) -> None:
        """Validate eager value compatibility with a model structure."""
        del structure

    @final
    def validate_against_structure(self, structure: Any) -> None:
        """Validate parameters fully against static construction choices.

        Args:
            structure: System-specific static structure to validate against.

        Returns:
            ``None`` after successful validation.

        Raises:
            ValueError: If intrinsic or structure-dependent validation fails.
        """
        self.validate()
        self.validate_structure_compatibility(structure)
        self.validate_value_compatibility(structure)

    @final
    def validate_for_update_against_structure(self, structure: Any) -> None:
        """Validate concrete values and traced structure against a model."""
        self.validate_structure()
        if _contains_tracer(self):
            self.validate_structure_compatibility(structure)
            return
        with ensure_compile_time_eval():
            self.validate_values()
        self.validate_structure_compatibility(structure)
        with ensure_compile_time_eval():
            self.validate_value_compatibility(structure)


class BaseSoftRobotParams(BaseSystemParams):
    """Common dynamic parameters for soft robot systems.

    Physical parameter trees contain gravity but deliberately do not contain a
    mounting or initial floating pose. Fixed mounting is model state and
    floating pose is runtime configuration state.
    """

    is_planar: ClassVar[bool | None] = None

    gravity: Array | None = eqx.field(default=None, kw_only=True)

    def __check_init__(self) -> None:
        object.__setattr__(self, "gravity", self._resolve_gravity(self.gravity))

    @classmethod
    def _require_planarity(cls) -> bool:
        if cls.is_planar is None:
            raise TypeError(f"{cls.__name__} must declare is_planar as True or False.")
        return cls.is_planar

    @classmethod
    def _default_gravity(cls) -> Array:
        if cls._require_planarity():
            return jnp.asarray([0.0, -DEFAULT_GRAVITY_MAGNITUDE])
        return jnp.asarray([0.0, 0.0, -DEFAULT_GRAVITY_MAGNITUDE])

    @classmethod
    def _resolve_gravity(cls, gravity: Array | None) -> Array:
        if gravity is None:
            return cls._default_gravity()
        return jnp.asarray(gravity)

    def _normalize_replacement(self, name: str, value: Any) -> Any:
        if name == "gravity":
            return type(self)._resolve_gravity(value)
        return value


class BaseContinuumSoftRobotParams(BaseSoftRobotParams):
    """Shared dynamic parameters for continuum soft robots.

    Concrete continuum systems expose their batched physical values through a
    shared ``ContinuumLinkParams`` field named ``link``. The field is declared
    by concrete subclasses to keep this base module independent from the
    reusable component package.
    """


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
