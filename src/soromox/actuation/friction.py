"""Friction models along threadlike routed paths that attenuate the effort."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams


def validate_friction_parameter_shape(value: Any, count: int, name: str) -> None:
    """Validate a per-path friction parameter shape.

    Args:
        value: Candidate friction parameter.
        count: Number of routed paths.
        name: Field name used in the error message.

    Raises:
        ValueError: If the value is neither a scalar nor shaped ``(count,)``.
    """
    shape = jnp.asarray(value).shape
    if shape not in ((), (count,)):
        raise ValueError(
            f"{name} must be a scalar or have shape ({count},), got {shape}."
        )


def validate_friction_parameter_values(value: Any, name: str) -> None:
    """Validate that a friction parameter is finite and non-negative.

    A negative or non-finite parameter pushes the transmission ratio outside
    ``(0, 1]``, causing the transmission to generate effort.

    Args:
        value: Concrete friction parameter.
        name: Field name used in the error message.

    Raises:
        ValueError: If the value is non-finite or negative.
    """
    array = jnp.asarray(value)
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must be finite, got {array}.")
    if bool(jnp.any(array < 0.0)):
        raise ValueError(f"{name} must be non-negative, got {array}.")


class ThreadlikeQuadratureContext(eqx.Module):
    """Routed-path geometry at one abscissa quadrature point.

    Attributes:
        abscissa: Backbone coordinate of the node from the robot base, in
        metres.
        path_abscissa: Arc-length covered along the span of each path from
        the path anchor, in metres.
        segment_index: Index of the segment containing the node.
        offset: Material-frame path offset of shape ``(num_paths, 3)``, ordered
            ``[x, y, z]``.
        offset_derivative: Arc-length derivative of ``offset``.
        tangent_norm: Length of the routed-path tangent, ``(num_paths,)``.
        active: True if the node lies inside the segment span of each path,
            shape ``(num_paths,)``.
        unit_tangent: Normalized routed-path tangent. Host-shaped:
            ``(num_paths, 2)`` in the plane and ``(num_paths, 3)`` in space.
        strain: Segment strain at the node. Host-shaped: ``(num_paths, 3)``
            ordered ``[kappa_z, sigma_x, sigma_y]`` in the plane, and
            ``(num_paths, 6)`` ordered ``[omega, v]`` in space.
        wrap_angle: Routed-path curvature integrated from the path anchor to
            the node, in radians, shape ``(num_paths,)``. ``None`` unless the
            installed model sets ``requires_wrap_angle``.
    """

    abscissa: Array
    path_abscissa: Array
    segment_index: Array
    offset: Array
    offset_derivative: Array
    tangent_norm: Array
    active: Array
    unit_tangent: Array
    strain: Array
    wrap_angle: Array | None = None

    def with_wrap_angle(self, wrap_angle: Array) -> ThreadlikeQuadratureContext:
        """Return a copy carrying the accumulated wrap angle.

        Args:
            wrap_angle: Wrap angle of shape ``(num_paths,)``, in radians.

        Returns:
            context: A copy whose ``wrap_angle`` is populated.
        """
        return eqx.tree_at(
            lambda context: context.wrap_angle,
            self,
            wrap_angle,
            is_leaf=lambda leaf: leaf is None,
        )


class BaseThreadlikeFrictionParams(BaseSystemParams):
    """Base PyTree contract for threadlike friction parameters."""

    def validate_structure_compatibility(self, num_paths: int) -> None:
        """Validate per-path friction parameter shapes against the path count.

        Args:
            num_paths: Number of paths in the routing family.
        """
        del num_paths


class FrictionlessParams(BaseThreadlikeFrictionParams):
    """Parameters of the ideal transmission. None."""


class CapstanFrictionParams(BaseThreadlikeFrictionParams):
    """Parameters of the Capstan (Euler) belt model.

    Attributes:
        coefficient: Coulomb coefficient between a routing and the robot, either
            a scalar shared by the family or shaped ``(num_paths,)``.
        eps: Curvature floor in radians per metre.
    """

    coefficient: Array = eqx.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float64)
    )
    eps: Array = eqx.field(default_factory=lambda: jnp.zeros((), dtype=jnp.float64))

    def __check_init__(self) -> None:
        self.validate_for_update()

    def validate_structure_compatibility(self, num_paths: int) -> None:
        """Validate the coefficient shape against the routed-path count."""
        validate_friction_parameter_shape(self.coefficient, num_paths, "coefficient")

    def validate_values(self) -> None:
        """Validate eager Capstan parameter values."""
        validate_friction_parameter_values(self.coefficient, "coefficient")
        validate_friction_parameter_values(self.eps, "eps")


class ExponentialLengthFrictionParams(BaseThreadlikeFrictionParams):
    """Parameters of a configuration-independent friction proportional
    to the routed-path length.

    Attributes:
        rate: Attenuation per unit arc length, in inverse metres, either a scalar
            shared by the family or shaped ``(num_paths,)``.
    """

    rate: Array = eqx.field(default_factory=lambda: jnp.zeros((), dtype=jnp.float64))

    def __check_init__(self) -> None:
        self.validate_for_update()

    def validate_structure_compatibility(self, num_paths: int) -> None:
        """Validate the rate shape against the routed-path count."""
        validate_friction_parameter_shape(self.rate, num_paths, "rate")

    def validate_values(self) -> None:
        """Validate eager exponential-length parameter values."""
        validate_friction_parameter_values(self.rate, "rate")


def frictionless_transmission_ratio(
    params: FrictionlessParams, context: ThreadlikeQuadratureContext
) -> Array:
    """Return ones, the identity of the weighted quadrature."""
    del params
    return jnp.ones_like(context.tangent_norm)


def capstan_transmission_ratio(
    params: CapstanFrictionParams, context: ThreadlikeQuadratureContext
) -> Array:
    """Return ``exp(-coefficient * wrap_angle)`` at the node."""
    if context.wrap_angle is None:
        raise ValueError(
            "the Capstan law reads context.wrap_angle, which this host did not "
            "supply. Declare requires_wrap_angle on the installed law."
        )
    return jnp.exp(-jnp.asarray(params.coefficient) * context.wrap_angle)


def exponential_length_transmission_ratio(
    params: ExponentialLengthFrictionParams, context: ThreadlikeQuadratureContext
) -> Array:
    """Return ``exp(-rate * path_abscissa)`` at the node."""
    return jnp.exp(-jnp.asarray(params.rate) * context.path_abscissa)


class ThreadlikeFriction(eqx.Module):
    """Runtime friction model for a routed-path family.

    Attributes:
        params: Friction parameters.
        ratio_fn: Model mapping ``(params, context)`` to the surviving fraction
            of shape ``(num_paths,)``.
        requires_wrap_angle: True if ``ratio_fn`` reads ``context.wrap_angle``.
        is_frictionless: True if the law returns exactly one everywhere.
    """

    params: BaseThreadlikeFrictionParams = eqx.field(default_factory=FrictionlessParams)
    ratio_fn: Callable = eqx.field(static=True, default=frictionless_transmission_ratio)
    requires_wrap_angle: bool = eqx.field(static=True, default=False)
    is_frictionless: bool = eqx.field(static=True, default=True)

    def __check_init__(self) -> None:
        if (
            self.is_frictionless
            and self.ratio_fn is not frictionless_transmission_ratio
        ):
            raise ValueError(
                "is_frictionless=True requires frictionless_transmission_ratio. "
                "A friction model requires is_frictionless=False."
            )

    @classmethod
    def frictionless(cls) -> ThreadlikeFriction:
        """Build the ideal transmission."""
        return cls()

    @classmethod
    def capstan(cls, *, coefficient: Any = 0.0, eps: Any = 0.0) -> ThreadlikeFriction:
        """Build the Capstan belt model.

        Args:
            coefficient: Coulomb coefficient, scalar or shaped ``(num_paths,)``.
            eps: Curvature floor in radians per metre.

        Returns:
            friction: Capstan model.
        """
        return cls(
            params=CapstanFrictionParams(
                coefficient=jnp.asarray(coefficient), eps=jnp.asarray(eps)
            ),
            ratio_fn=capstan_transmission_ratio,
            requires_wrap_angle=True,
            is_frictionless=False,
        )

    @classmethod
    def exponential_length(cls, *, rate: Any = 0.0) -> ThreadlikeFriction:
        """Build the length-decay model.

        Args:
            rate: Attenuation per unit arc length, scalar or ``(num_paths,)``.

        Returns:
            friction: Length-decay law.
        """
        return cls(
            params=ExponentialLengthFrictionParams(rate=jnp.asarray(rate)),
            ratio_fn=exponential_length_transmission_ratio,
            requires_wrap_angle=False,
            is_frictionless=False,
        )

    def transmission_ratio(self, context: ThreadlikeQuadratureContext) -> Array:
        """Return the transmission ratio at one quadrature node.

        Args:
            context: Routed-path geometry at the node, batched over paths.

        Returns:
            transmission_ratio: shape ``(num_paths,)``.
        """
        return self.ratio_fn(self.params, context)

    def with_params(self, params: BaseThreadlikeFrictionParams) -> ThreadlikeFriction:
        """Return a copy carrying replacement friction parameters."""
        if not isinstance(params, BaseThreadlikeFrictionParams):
            raise TypeError("params must be BaseThreadlikeFrictionParams.")
        if type(params) is not type(self.params):
            raise TypeError(
                "friction params must keep their type; got "
                f"{type(params).__name__} for an installed "
                f"{type(self.params).__name__} model."
            )
        params.validate_for_update()
        return ThreadlikeFriction(
            params=params,
            ratio_fn=self.ratio_fn,
            requires_wrap_angle=self.requires_wrap_angle,
            is_frictionless=self.is_frictionless,
        )

    def update_params(self, **updates: Any) -> ThreadlikeFriction:
        """Return a copy with replaced friction parameter leaves."""
        return self.with_params(
            cast(BaseThreadlikeFrictionParams, self.params.replace(**updates))
        )


def validate_friction_for_robot(friction: ThreadlikeFriction, robot) -> None:
    if friction.is_frictionless or not friction.requires_wrap_angle:
        return
    if not callable(getattr(robot, "_threadlike_path_curvature", None)):
        raise TypeError(
            f"{type(robot).__name__} does not support the threadlike wrap-angle model."
        )


__all__ = [
    "BaseThreadlikeFrictionParams",
    "CapstanFrictionParams",
    "ExponentialLengthFrictionParams",
    "FrictionlessParams",
    "ThreadlikeFriction",
    "ThreadlikeQuadratureContext",
    "capstan_transmission_ratio",
    "exponential_length_transmission_ratio",
    "frictionless_transmission_ratio",
    "validate_friction_for_robot",
    "validate_friction_parameter_shape",
    "validate_friction_parameter_values",
]
