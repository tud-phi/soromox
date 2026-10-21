"""Friction models and geometry helpers for threadlike actuation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
from jax import Array, vmap
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams
from soromox.utils.numerics import safe_divide, safe_norm


def _validate_parameter_shape(value: Any, count: int, name: str) -> None:
    """Validate a scalar or per-path parameter shape.

    Args:
        value: Candidate parameter value.
        count: Number of routed paths.
        name: Field name used in the error message.

    Raises:
        ValueError: If ``value`` is neither scalar nor shaped ``(count,)``.
    """
    shape = jnp.asarray(value).shape
    if shape not in ((), (count,)):
        raise ValueError(
            f"{name} must be a scalar or have shape ({count},), got {shape}."
        )


def _validate_nonnegative_values(value: Any, name: str) -> None:
    """Validate that a concrete friction parameter is finite and nonnegative.

    Args:
        value: Candidate parameter value.
        name: Field name used in the error message.

    Raises:
        ValueError: If any value is negative or non-finite.
    """
    array = jnp.asarray(value)
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must be finite, got {array}.")
    if bool(jnp.any(array < 0.0)):
        raise ValueError(f"{name} must be non-negative, got {array}.")


def threadlike_tangent(
    angular_strain: Array,
    linear_strain: Array,
    offset: Array,
    offset_derivative: Array,
) -> Array:
    r"""Return the material-frame tangent of a routed path.

    For backbone pose ``g(s) = (R(s), p(s))`` and material-frame routing
    offset ``r(s)``, the spatial path is ``x(s) = p(s) + R(s) r(s)``. The
    Cosserat relations ``R' = R hat(omega)`` and ``p' = R v`` give

    .. math::

        x' = R a, \qquad a = v + \omega \times r + r'.

    Args:
        angular_strain: Material angular strain ``omega`` with shape ``(3,)``.
        linear_strain: Material linear strain ``v`` with shape ``(3,)``.
        offset: Material-frame routing offset ``r`` with shape ``(3,)``.
        offset_derivative: Backbone-abscissa derivative ``r'`` with shape
            ``(3,)``.

    Returns:
        Material-frame path tangent ``a`` with shape ``(3,)``.
    """
    return linear_strain + jnp.cross(angular_strain, offset) + offset_derivative


def _threadlike_turn_rate_from_tangent(
    angular_strain: Array,
    tangent: Array,
    tangent_derivative: Array,
    *,
    eps: Array | float,
) -> Array:
    r"""Return spatial turn rate from a material tangent and its derivative.

    With ``u = a / ||a||``, the material unit-tangent derivative is

    .. math::

        u' = \frac{(I-u u^T)a'}{\lVert a\rVert}.

    The spatial unit tangent turns at
    ``rho = ||omega cross u + u'||``. A collapsed path has no defined tangent;
    the package's finite zero convention is used for both ``u`` and ``u'``.

    Args:
        angular_strain: Material angular strain ``omega`` with shape ``(3,)``.
        tangent: Material-frame path tangent ``a`` with shape ``(3,)``.
        tangent_derivative: Backbone-abscissa derivative ``a'`` with shape
            ``(3,)``.
        eps: Threshold below which ``||a||`` is treated as zero.

    Returns:
        Nonnegative spatial turn rate ``rho`` as a scalar.
    """
    tangent_norm = safe_norm(tangent)
    unit_tangent = safe_divide(tangent, tangent_norm, eps)
    projected = tangent_derivative - unit_tangent * jnp.dot(
        unit_tangent, tangent_derivative
    )
    unit_tangent_derivative = safe_divide(projected, tangent_norm, eps)
    return safe_norm(jnp.cross(angular_strain, unit_tangent) + unit_tangent_derivative)


def threadlike_turn_rate(
    angular_strain: Array,
    linear_strain: Array,
    angular_strain_derivative: Array,
    linear_strain_derivative: Array,
    offset: Array,
    offset_derivative: Array,
    offset_second_derivative: Array,
    *,
    eps: Array | float,
) -> Array:
    r"""Return how quickly the routed path direction turns in space.

    The spatial unit tangent is ``t_hat = R u``, where ``u`` is the normalized
    material tangent. Since ``R' = R hat(omega)``,

    .. math::

        \frac{d\hat t}{ds}
        = R(\omega \times u + u'), \qquad
        \rho = \left\|\omega \times u + u'\right\|.

    Rotation preserves the Euclidean norm, so ``rho`` is evaluated entirely
    in the material frame. It has units of radians per metre of backbone
    abscissa and includes both backbone strain variation and routing-offset
    variation.

    Args:
        angular_strain: Material angular strain ``omega`` with shape ``(3,)``.
        linear_strain: Material linear strain ``v`` with shape ``(3,)``.
        angular_strain_derivative: Backbone-abscissa derivative ``omega'`` with
            shape ``(3,)``.
        linear_strain_derivative: Backbone-abscissa derivative ``v'`` with
            shape ``(3,)``.
        offset: Material-frame routing offset ``r`` with shape ``(3,)``.
        offset_derivative: Backbone-abscissa derivative ``r'`` with shape
            ``(3,)``.
        offset_second_derivative: Backbone-abscissa derivative ``r''`` with
            shape ``(3,)``.
        eps: Threshold used for a collapsed tangent.

    Returns:
        Nonnegative turn rate ``rho`` as a scalar.
    """
    tangent = threadlike_tangent(
        angular_strain, linear_strain, offset, offset_derivative
    )
    tangent_derivative = (
        linear_strain_derivative
        + jnp.cross(angular_strain_derivative, offset)
        + jnp.cross(angular_strain, offset_derivative)
        + offset_second_derivative
    )
    return _threadlike_turn_rate_from_tangent(
        angular_strain,
        tangent,
        tangent_derivative,
        eps=eps,
    )


def threadlike_constant_strain_turn_rate(
    angular_strain: Array,
    linear_strain: Array,
    offset: Array,
    offset_derivative: Array,
    offset_second_derivative: Array,
    *,
    eps: Array | float,
) -> Array:
    r"""Return routed-path turn rate for a constant Cosserat strain.

    For PCS, ``omega' = v' = 0`` within every segment, so the material tangent
    derivative simplifies analytically to

    .. math::

        a' = \omega \times r' + r''.

    Args:
        angular_strain: Constant material angular strain ``omega`` with shape
            ``(3,)``.
        linear_strain: Constant material linear strain ``v`` with shape
            ``(3,)``.
        offset: Material-frame routing offset ``r`` with shape ``(3,)``.
        offset_derivative: Backbone-abscissa derivative ``r'`` with shape
            ``(3,)``.
        offset_second_derivative: Backbone-abscissa derivative ``r''`` with
            shape ``(3,)``.
        eps: Threshold below which a collapsed tangent uses the finite zero
            convention.

    Returns:
        Nonnegative turn rate ``rho`` as a scalar.
    """
    tangent = threadlike_tangent(
        angular_strain, linear_strain, offset, offset_derivative
    )
    tangent_derivative = (
        jnp.cross(angular_strain, offset_derivative) + offset_second_derivative
    )
    return _threadlike_turn_rate_from_tangent(
        angular_strain,
        tangent,
        tangent_derivative,
        eps=eps,
    )


def threadlike_turning_angle(
    left_unit_tangent: Array,
    right_unit_tangent: Array,
) -> Array:
    """Return the unsigned turning angle between two unit tangents.

    Args:
        left_unit_tangent: Unit tangent immediately before a segment boundary.
        right_unit_tangent: Unit tangent immediately after the boundary.

    Returns:
        Angle in radians in the closed interval ``[0, pi]``.
    """
    sine = safe_norm(jnp.cross(left_unit_tangent, right_unit_tangent))
    cosine = jnp.clip(jnp.dot(left_unit_tangent, right_unit_tangent), -1.0, 1.0)
    has_direction = (sine > 0.0) | (jnp.abs(cosine) > 0.0)
    safe_cosine = jnp.where(has_direction, cosine, jnp.ones_like(cosine))
    angle = jnp.arctan2(sine, safe_cosine)
    return jnp.where(has_direction, angle, jnp.zeros_like(angle))


def integrate_accumulated_turn(
    turn_rate_fn: Callable[[Array, Array], Array],
    boundary_turn_fn: Callable[[Array], Array],
    s: Array,
    segment_starts: Array,
    segment_lengths: Array,
    normalized_s_ps: Array,
    normalized_weights: Array,
    start_segment_index: Array,
    end_segment_index: Array,
) -> Array:
    """Integrate path turn from each anchor through backbone abscissa ``s``.

    The integral uses the host quadrature rule on every complete segment and a
    rescaled copy of that rule on the final partial segment. One-sided tangent
    jumps caused by the host discretization are added at crossed segment
    boundaries. These are continuum-model interfaces, not discrete friction
    guides.

    Args:
        turn_rate_fn: Function accepting ``(segment_index, s_ps)``
            and returning turn rates shaped ``(num_points, num_paths)``.
        boundary_turn_fn: Function accepting a left segment index and returning
            one-sided tangent angles shaped ``(num_paths,)``.
        s: Backbone abscissa through which to integrate.
        segment_starts: Backbone abscissa at the start of every segment.
        segment_lengths: Physical length of every segment.
        normalized_s_ps: Normalized quadrature abscissae in ``[0, 1]`` shaped
            ``(num_segments, num_points)``.
        normalized_weights: Corresponding quadrature weights with the same
            shape as ``normalized_s_ps``.
        start_segment_index: First segment of every routed path.
        end_segment_index: Last segment of every routed path.

    Returns:
        Accumulated unsigned turn, in radians, shaped ``(num_paths,)``.
    """
    segment_indices = jnp.arange(segment_lengths.shape[0])

    def integrate_segment(segment_index: Array) -> Array:
        """Integrate turn over the covered part of one segment."""
        covered_length = jnp.clip(
            jnp.asarray(s) - segment_starts[segment_index],
            0.0,
            segment_lengths[segment_index],
        )
        s_ps = (
            segment_starts[segment_index]
            + covered_length * normalized_s_ps[segment_index]
        )
        rates = turn_rate_fn(segment_index, s_ps)
        return covered_length * jnp.sum(
            normalized_weights[segment_index, :, None] * rates, axis=0
        )

    distributed = vmap(integrate_segment)(segment_indices)
    active_segments = (segment_indices[:, None] >= start_segment_index[None, :]) & (
        segment_indices[:, None] <= end_segment_index[None, :]
    )
    accumulated = jnp.sum(distributed * active_segments, axis=0)

    if segment_lengths.shape[0] <= 1:
        return accumulated

    boundary_indices = jnp.arange(segment_lengths.shape[0] - 1)
    boundary_s = segment_starts[1:]
    boundary_angles = vmap(boundary_turn_fn)(boundary_indices)
    crossed = boundary_s[:, None] <= jnp.asarray(s)
    spans_boundary = (boundary_indices[:, None] >= start_segment_index[None, :]) & (
        boundary_indices[:, None] + 1 <= end_segment_index[None, :]
    )
    return accumulated + jnp.sum(boundary_angles * crossed * spans_boundary, axis=0)


class BaseThreadlikeFrictionParams(BaseSystemParams):
    """Base PyTree contract for threadlike friction parameters."""

    def validate_structure_compatibility(self, num_paths: int) -> None:
        """Validate parameter shapes against the number of routed paths.

        Args:
            num_paths: Number of paths in the routing family.
        """
        del num_paths


class FrictionlessParams(BaseThreadlikeFrictionParams):
    """Parameter-free ideal threadlike transmission."""


class CapstanFrictionParams(BaseThreadlikeFrictionParams):
    """Coulomb coefficient of the continuum Capstan model.

    Attributes:
        coefficient: Nonnegative friction coefficient, either scalar or one
            value per routed path.
    """

    coefficient: Array = eqx.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float64)
    )

    def __check_init__(self) -> None:
        """Validate newly constructed Capstan parameters."""
        self.validate_for_update()

    def validate_structure_compatibility(self, num_paths: int) -> None:
        """Validate the coefficient shape against the path count.

        Args:
            num_paths: Number of paths in the routing family.
        """
        _validate_parameter_shape(self.coefficient, num_paths, "coefficient")

    def validate_values(self) -> None:
        """Validate concrete Capstan coefficient values."""
        _validate_nonnegative_values(self.coefficient, "coefficient")


def frictionless_effort_ratio(
    params: FrictionlessParams, accumulated_turn: Array
) -> Array:
    """Return the unit effort ratio for an ideal transmission.

    Args:
        params: Parameter-free frictionless model parameters.
        accumulated_turn: Accumulated routed-path turn in radians.

    Returns:
        Ones with the same shape as ``accumulated_turn``.
    """
    del params
    return jnp.ones_like(accumulated_turn)


def capstan_effort_ratio(
    params: CapstanFrictionParams, accumulated_turn: Array
) -> Array:
    r"""Return the surviving effort fraction ``exp(-mu * Theta)``.

    Args:
        params: Capstan parameters containing ``mu``.
        accumulated_turn: Unsigned turn ``Theta`` from each path anchor to the
            evaluation point, in radians.

    Returns:
        Effort ratio in ``(0, 1]`` with one value per routed path.
    """
    return jnp.exp(-jnp.asarray(params.coefficient) * accumulated_turn)


class ThreadlikeFriction(eqx.Module):
    """Runtime effort-loss law for a routed-path family.

    The effort supplied by the actuator is defined at each path anchor. The
    model returns the fraction that remains after the path has accumulated a
    given unsigned turn while moving along increasing backbone abscissa.

    Attributes:
        params: Dynamic friction parameters.
        effort_ratio_fn: Function of ``(params, accumulated_turn)``.
        is_frictionless: Whether the ratio is identically one.
    """

    params: BaseThreadlikeFrictionParams = eqx.field(default_factory=FrictionlessParams)
    effort_ratio_fn: Callable = eqx.field(
        static=True, default=frictionless_effort_ratio
    )
    is_frictionless: bool = eqx.field(static=True, default=True)

    def __check_init__(self) -> None:
        """Validate consistency between the law and frictionless marker."""
        if (
            self.is_frictionless
            and self.effort_ratio_fn is not frictionless_effort_ratio
        ):
            raise ValueError("is_frictionless=True requires frictionless_effort_ratio.")

    @classmethod
    def frictionless(cls) -> ThreadlikeFriction:
        """Construct the ideal threadlike transmission."""
        return cls()

    @classmethod
    def capstan(cls, *, coefficient: Any = 0.0) -> ThreadlikeFriction:
        """Construct the continuum Capstan effort-loss model.

        Args:
            coefficient: Nonnegative scalar or one value per routed path.

        Returns:
            A Capstan friction model.
        """
        return cls(
            params=CapstanFrictionParams(coefficient=jnp.asarray(coefficient)),
            effort_ratio_fn=capstan_effort_ratio,
            is_frictionless=False,
        )

    def effort_ratio(self, accumulated_turn: Array) -> Array:
        """Return the surviving effort fraction after accumulated turn.

        Args:
            accumulated_turn: Unsigned turn from each path anchor in radians.

        Returns:
            One effort ratio per routed path.
        """
        return self.effort_ratio_fn(self.params, accumulated_turn)

    def with_params(self, params: BaseThreadlikeFrictionParams) -> ThreadlikeFriction:
        """Return a copy carrying replacement friction parameters.

        Args:
            params: Replacement parameters of the installed model type.

        Returns:
            A friction model with updated dynamic leaves.

        Raises:
            TypeError: If the parameter type changes.
        """
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
            effort_ratio_fn=self.effort_ratio_fn,
            is_frictionless=self.is_frictionless,
        )

    def update_params(self, **updates: Any) -> ThreadlikeFriction:
        """Return a copy with replaced friction-parameter leaves.

        Args:
            **updates: Named parameter-leaf replacements.

        Returns:
            A friction model with updated parameter leaves.
        """
        return self.with_params(
            cast(BaseThreadlikeFrictionParams, self.params.replace(**updates))
        )


def _validate_friction_model(friction: ThreadlikeFriction, routing: Any) -> None:
    """Validate a friction model against a routing family.

    Args:
        friction: Installed threadlike friction model.
        routing: Installed threadlike routing family.

    Raises:
        ValueError: If nonzero friction is paired with a custom routing that
            does not provide an analytical second derivative.
    """
    if friction.is_frictionless:
        return
    if not routing.has_analytical_second_derivative:
        raise ValueError(
            "threadlike friction requires an analytical routing second_derivative_fn."
        )


__all__ = [
    "BaseThreadlikeFrictionParams",
    "CapstanFrictionParams",
    "FrictionlessParams",
    "ThreadlikeFriction",
    "_validate_friction_model",
    "capstan_effort_ratio",
    "frictionless_effort_ratio",
    "integrate_accumulated_turn",
    "threadlike_constant_strain_turn_rate",
    "threadlike_tangent",
    "threadlike_turn_rate",
    "threadlike_turning_angle",
]
