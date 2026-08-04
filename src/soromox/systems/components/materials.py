"""Shared isotropic material construction and optimization parameters."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

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
        shear_modulus: Scalar or per-link shear modulus values.
        material_damping_coefficient: Scalar or per-link isotropic material
            damping values.
    """

    young_modulus: Array
    shear_modulus: Array
    material_damping_coefficient: Array

    def __check_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate material field dimensionality.

        Returns:
            None.

        Raises:
            ValueError: If any material field has more than one dimension.
        """
        for name in (
            "young_modulus",
            "shear_modulus",
            "material_damping_coefficient",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.ndim > 1:
                raise ValueError(f"{name} must be scalar or one-dimensional.")

    def broadcast(self, num_links: int) -> IsotropicMaterialParams:
        """Broadcast scalar material fields to one value per link.

        Args:
            num_links: Number of links in the target robot.

        Returns:
            A new material PyTree whose three fields all have shape
            ``(num_links,)``.

        Raises:
            ValueError: If a non-scalar field does not have shape
                ``(num_links,)``.
        """

        def broadcast_one(name: str) -> Array:
            value = jnp.asarray(getattr(self, name))
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
            material_damping_coefficient=broadcast_one("material_damping_coefficient"),
        )
