"""Shared isotropic material construction and optimization parameters."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.errors import ConcretizationTypeError, TracerBoolConversionError

from soromox.systems.params import BaseSystemParams

__all__ = ["IsotropicMaterialParams", "shear_modulus_from_poisson_ratio"]


def shear_modulus_from_poisson_ratio(
    young_modulus: Array | float,
    poisson_ratio: Array | float,
) -> Array:
    """Compute the shear modulus of an isotropic material.

    Args:
        young_modulus: Young's modulus ``E`` as a scalar or array.
        poisson_ratio: Poisson's ratio ``nu`` as a scalar or array broadcastable
            with ``young_modulus``.

    Returns:
        The shear modulus ``G = E / (2 (1 + nu))`` with the broadcasted input
        shape.
    """
    young = jnp.asarray(young_modulus)
    poisson = jnp.asarray(poisson_ratio)
    return young / (2.0 * (1.0 + poisson))


class IsotropicMaterialParams(BaseSystemParams):
    """Caller-owned isotropic material variables.

    These values are construction and optimization variables rather than
    canonical robot runtime parameters. Pass an instance to
    ``link_matrices_from_material`` or ``with_isotropic_material`` to map it to
    generalized link matrices.

    Attributes:
        young_modulus: Scalar or per-link Young's modulus values.
        shear_modulus: Optional scalar or per-link shear modulus values.
            Exactly one of ``shear_modulus`` and ``poisson_ratio`` must be
            provided.
        poisson_ratio: Optional scalar or per-link Poisson-ratio values.
        material_damping_coefficient: Optional scalar or per-link isotropic
            material damping values. ``None`` disables material damping.

    The presence of ``shear_modulus``, ``poisson_ratio``, and
    ``material_damping_coefficient`` is part of the PyTree structure. JIT-compiled
    functions therefore specialize on the chosen stiffness parameterization and
    on whether material damping is active, while the numeric arrays remain
    differentiable leaves.
    """

    young_modulus: Array
    shear_modulus: Array | None = None
    poisson_ratio: Array | None = None
    material_damping_coefficient: Array | None = None

    def __check_init__(self) -> None:
        for name in (
            "young_modulus",
            "shear_modulus",
            "poisson_ratio",
            "material_damping_coefficient",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, jnp.asarray(value))
        self.validate()

    def _normalize_replacement(self, name: str, value: object) -> object:
        """Normalize replacement material values to JAX arrays."""
        if (
            name
            in (
                "young_modulus",
                "shear_modulus",
                "poisson_ratio",
                "material_damping_coefficient",
            )
            and value is not None
        ):
            return jnp.asarray(value)
        return value

    def validate(self) -> None:
        """Validate material field dimensionality and physical values.

        Returns:
            None.

        Raises:
            ValueError: If both or neither of ``shear_modulus`` and
                ``poisson_ratio`` are provided; any active material field has
                more than one dimension; either modulus is non-finite or not
                strictly positive; Poisson's ratio is outside ``(-1, 0.5)``;
                or the material damping coefficient is non-finite or negative.
        """
        if (self.shear_modulus is None) == (self.poisson_ratio is None):
            raise ValueError("Provide exactly one of shear_modulus and poisson_ratio.")

        for name in (
            "young_modulus",
            "shear_modulus",
            "poisson_ratio",
            "material_damping_coefficient",
        ):
            field_value = getattr(self, name)
            if field_value is None:
                continue
            value = jnp.asarray(field_value)
            if value.ndim > 1:
                raise ValueError(f"{name} must be scalar or one-dimensional.")
            if value.ndim == 1 and value.shape[0] < 1:
                raise ValueError(f"{name} must not be empty.")
            try:
                finite = bool(jnp.all(jnp.isfinite(value)))
                if name == "material_damping_coefficient":
                    valid_domain = bool(jnp.all(value >= 0.0))
                elif name == "poisson_ratio":
                    valid_domain = bool(jnp.all((value > -1.0) & (value < 0.5)))
                else:
                    valid_domain = bool(jnp.all(value > 0.0))
            except (ConcretizationTypeError, TracerBoolConversionError):
                continue
            if not finite:
                raise ValueError(f"{name} must contain only finite values.")
            if not valid_domain:
                if name == "material_damping_coefficient":
                    qualifier = "nonnegative"
                elif name == "poisson_ratio":
                    qualifier = "greater than -1 and less than 0.5"
                else:
                    qualifier = "strictly positive"
                raise ValueError(f"{name} must be {qualifier}.")

    def _resolved_shear_modulus(self) -> Array:
        """Return the supplied or Poisson-ratio-derived shear modulus."""
        if self.shear_modulus is not None:
            return self.shear_modulus
        assert self.poisson_ratio is not None
        return shear_modulus_from_poisson_ratio(self.young_modulus, self.poisson_ratio)

    def broadcast(self, num_links: int) -> IsotropicMaterialParams:
        """Broadcast scalar material fields to one value per link.

        Args:
            num_links: Number of links in the target robot.

        Returns:
            A new material PyTree whose active fields all have shape
            ``(num_links,)``. Inactive optional fields remain ``None``.

        Raises:
            ValueError: If a non-scalar field does not have shape
                ``(num_links,)``.
        """

        def broadcast_one(name: str) -> Array | None:
            field_value = getattr(self, name)
            if field_value is None:
                return None
            value = jnp.asarray(field_value)
            if value.ndim == 0:
                return jnp.full((num_links,), value)
            if value.shape != (num_links,):
                raise ValueError(
                    f"{name} must be scalar or have shape ({num_links},), "
                    f"got {value.shape}."
                )
            return value

        return IsotropicMaterialParams(
            young_modulus=broadcast_one("young_modulus"),
            shear_modulus=broadcast_one("shear_modulus"),
            poisson_ratio=broadcast_one("poisson_ratio"),
            material_damping_coefficient=broadcast_one("material_damping_coefficient"),
        )
