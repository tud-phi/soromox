"""Transmission-loss laws along threadlike routed paths.

A threadlike transmission carries effort from the base of a routed path toward
its tip. Guides, seals, and the surrounding material remove some of that effort
before it reaches the backbone. A :class:`ThreadlikeFriction` model expresses
how much survives, as a fraction ``eta(q, s)`` of the base effort, and the host
folds that fraction into the arc-length quadrature that assembles the moment
matrix.

Because the fraction weights the integrand rather than scaling the result, the
loss accumulates along the path: distal material contributes less than proximal
material. The fraction depends on the configuration only, never on the effort,
which keeps the generalized actuation force linear in the effort and every
controller structurally unchanged.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSystemParams


def validate_loss_parameter_shape(value: Any, count: int, name: str) -> None:
    """Validate a per-path loss parameter shape without concretizing it.

    Args:
        value: Candidate loss parameter.
        count: Number of routed paths in the family.
        name: Field name used in the error message.

    Raises:
        ValueError: If the value is neither a scalar nor shaped ``(count,)``.
    """
    shape = jnp.asarray(value).shape
    if shape not in ((), (count,)):
        raise ValueError(
            f"{name} must be a scalar or have shape ({count},), got {shape}."
        )


class ThreadlikeQuadratureContext(eqx.Module):
    """Routed-path geometry at one arc-length quadrature node.

    Instances are produced by the host inside its per-path ``vmap``, so every
    array field carries a leading ``(num_paths,)`` axis.

    Some fields mean the same thing on every host and some do not. A law that
    reads only the host-agnostic fields runs unchanged on ``PlanarPCS``,
    ``PCS``, and ``GVS``; a law that reads a host-shaped field must document
    which host it targets.

    Attributes:
        arc_length: Backbone coordinate ``s`` of the node, in metres.
            Host-agnostic.
        segment_index: Index of the segment containing the node. Host-agnostic.
        offset: Material-frame path offset of shape ``(num_paths, 3)``, ordered
            ``[x, y, z]`` with local x along the backbone. Host-agnostic.
        offset_derivative: Arc-length derivative of ``offset``, same shape.
            Host-agnostic.
        tangent_norm: Length of the routed-path tangent, ``(num_paths,)``. This
            is the local stretch of the path relative to the backbone, so it is
            the quantity a length-based loss integrates. Host-agnostic.
        active: Whether the node lies inside each path's segment span, shape
            ``(num_paths,)``. Host-agnostic.
        unit_tangent: Normalized routed-path tangent. Host-shaped:
            ``(num_paths, 2)`` in the plane and ``(num_paths, 3)`` in space.
        strain: Segment strain at the node. Host-shaped: ``(num_paths, 3)``
            ordered ``[kappa_z, sigma_x, sigma_y]`` in the plane, and
            ``(num_paths, 6)`` ordered ``[omega, v]`` in space.
        wrap_angle: Accumulated turn ``Theta`` of the routed path from the base
            to this node, in radians, shape ``(num_paths,)``. ``None`` unless
            the installed law sets ``requires_wrap_angle``, because computing
            it costs the host extra work.

    Note:
        There is no path-count field. The host assembles the context inside a
        ``vmap`` over paths, where a static count could not be set safely, so a
        law that needs the count reads it from a batched field, for example
        ``context.tangent_norm.shape[0]``.
    """

    arc_length: Array
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

        The host builds the geometric fields inside its per-path ``vmap`` and
        attaches the wrap angle afterwards, because the wrap density is
        assembled once per segment rather than once per path.

        Args:
            wrap_angle: Accumulated turn of shape ``(num_paths,)``.

        Returns:
            context: A copy whose ``wrap_angle`` is populated.
        """
        return eqx.tree_at(
            lambda context: context.wrap_angle,
            self,
            wrap_angle,
            is_leaf=lambda leaf: leaf is None,
        )


class ThreadlikeFriction(BaseSystemParams):
    """Fraction of routed-path effort that survives to a quadrature node.

    Subclasses declare what they need from the host through the class
    variables below, so a host computes only what the installed law reads and
    can refuse a law it cannot serve.

    Attributes:
        is_lossless: Whether the law returns exactly one everywhere. Hosts skip
            the friction path entirely when it is set, which keeps the default
            bit-identical to the frictionless model at no cost.
        requires_wrap_angle: Whether :meth:`transmission_ratio` reads
            ``context.wrap_angle``. Hosts that cannot supply an exact
            accumulated turn refuse such a law at construction.
    """

    is_lossless: ClassVar[bool] = False
    requires_wrap_angle: ClassVar[bool] = True

    @abstractmethod
    def transmission_ratio(self, context: ThreadlikeQuadratureContext) -> Array:
        """Return the surviving effort fraction at one quadrature node.

        Args:
            context: Routed-path geometry at the node, batched over paths.

        Returns:
            eta: Surviving fraction of shape ``(num_paths,)``. Implementations
                must stay within ``(0, 1]`` so a transmission can never
                generate effort, and must not depend on the applied effort,
                which would make the actuation force nonlinear in the control.
        """
        ...

    def validate_for_paths(self, num_paths: int) -> None:
        """Validate loss parameters against the routed-path count.

        Args:
            num_paths: Number of paths in the routing family.

        Returns:
            ``None``. Subclasses override this hook to check per-path shapes.
            Implementations must be safe when leaves are JAX tracers.
        """
        del num_paths


class Frictionless(ThreadlikeFriction):
    """Ideal transmission that delivers the full effort to every node."""

    is_lossless: ClassVar[bool] = True
    requires_wrap_angle: ClassVar[bool] = False

    def transmission_ratio(self, context: ThreadlikeQuadratureContext) -> Array:
        """Return ones, the identity of the weighted quadrature."""
        return jnp.ones_like(context.tangent_norm)


class CapstanFriction(ThreadlikeFriction):
    """Coulomb guide friction following the Capstan (Euler) belt law.

    A band drawn over a guide with total wrap angle ``phi`` transmits
    ``exp(-mu * phi)`` of its input tension, independently of the guide radius.
    Accumulating that loss over a continuum of guides gives

    ``eta(q, s) = exp(-mu * Theta(q, s))``

    with ``Theta`` the accumulated turn of the routed path. The law follows the
    tendon force model of Feliu-Talegon, Alkayas, Adamu, Mathew and Renda,
    "Actuation Reading Insights: Estimating Shape and Forces in Tendon-Driven
    Slender Soft Robots", IEEE/ASME Transactions on Mechatronics 30(6), 2025.

    Attributes:
        coefficient: Coulomb coefficient between a path and its guides, either
            a scalar shared by the family or shaped ``(num_paths,)``. Zero
            reproduces the frictionless transmission exactly.
    """

    coefficient: Array = eqx.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float64)
    )

    requires_wrap_angle: ClassVar[bool] = True

    def transmission_ratio(self, context: ThreadlikeQuadratureContext) -> Array:
        """Return the Capstan ratio ``exp(-mu * Theta)`` at the node."""
        return jnp.exp(-jnp.asarray(self.coefficient) * context.wrap_angle)

    def validate_for_paths(self, num_paths: int) -> None:
        """Validate the coefficient shape against the routed-path count."""
        validate_loss_parameter_shape(self.coefficient, num_paths, "coefficient")


class ExponentialLengthFriction(ThreadlikeFriction):
    """Loss accumulating with routed-path length rather than with turning.

    Both shipped laws solve the same tension ODE ``dT/ds = -lambda(s) T``,
    whose solution is ``eta = exp(-integral lambda ds)``. They differ only in
    where the normal load pressing the path against its guide comes from.
    :class:`CapstanFriction` takes it from curvature, ``N = T w``, giving
    ``lambda = mu w``. This law takes it from a snug sheath that grips in
    proportion to tension, ``N = c T``, giving a constant ``lambda = mu c``
    and

    ``eta(q, s) = exp(-rate * s)``

    so loss accrues with distance travelled rather than with turning. It suits
    a tendon in a close-fitting sheath or lumen. It reads no host-shaped field
    and needs no wrap angle, so unlike :class:`CapstanFriction` it runs on
    every threadlike host, ``GVS`` included.

    Attributes:
        rate: Loss per unit arc length ``mu c``, in inverse metres, either a
            scalar shared by the family or shaped ``(num_paths,)``. Unlike a
            Coulomb coefficient it is phenomenological: it lumps the friction
            coefficient together with the radial grip and has to be fitted
            rather than looked up. Zero reproduces the frictionless
            transmission exactly.
    """

    rate: Array = eqx.field(default_factory=lambda: jnp.zeros((), dtype=jnp.float64))

    requires_wrap_angle: ClassVar[bool] = False

    def transmission_ratio(self, context: ThreadlikeQuadratureContext) -> Array:
        """Return the length-decay ratio ``exp(-rate * s)`` at the node."""
        return jnp.exp(-jnp.asarray(self.rate) * context.arc_length)

    def validate_for_paths(self, num_paths: int) -> None:
        """Validate the rate shape against the routed-path count."""
        validate_loss_parameter_shape(self.rate, num_paths, "rate")


__all__ = [
    "CapstanFriction",
    "ExponentialLengthFriction",
    "Frictionless",
    "ThreadlikeFriction",
    "ThreadlikeQuadratureContext",
    "validate_loss_parameter_shape",
]
