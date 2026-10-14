"""Spatial inertia preparation shared by continuum system families."""

import jax
import jax.numpy as jnp
from jax import Array

from soromox.utils.lie_algebra import se2, se3


def spatial_inertia(
    mass: Array, center_of_mass: Array, rotational_inertia: Array
) -> Array:
    """Construct angular-first inertia about a material-frame origin.

    Args:
        mass: Mass [kg], or mass per unit length [kg/m].
        center_of_mass: Center of mass relative to the origin [m], shape
            ``(..., 2)`` for planar or ``(..., 3)`` for spatial mechanics.
        rotational_inertia: Inertia about the center of mass, a scalar for planar
            mechanics or shape ``(..., 3, 3)`` for spatial mechanics
            [kg m²], or [kg m] for a distributed section.

    Returns:
        Symmetric angular-first inertia, shape ``(..., 3, 3)`` (planar) or
        ``(..., 6, 6)`` (spatial). Inputs support
        broadcasting over leading dimensions. All quantities use the same frame.
    """
    mass = jnp.asarray(mass)
    center = jnp.asarray(center_of_mass)
    inertia = jnp.asarray(rotational_inertia)
    if center.shape[-1] == 2:
        shape = jnp.broadcast_shapes(mass.shape, center.shape[:-1], inertia.shape)
        mass = jnp.broadcast_to(mass, shape)
        center = jnp.broadcast_to(center, (*shape, 2))
        inertia = jnp.broadcast_to(inertia, shape)
        x, y = center[..., 0], center[..., 1]
        zero = jnp.zeros_like(mass)
        return jnp.stack(
            (
                inertia + mass * (x * x + y * y),
                -mass * y,
                mass * x,
                -mass * y,
                mass,
                zero,
                mass * x,
                zero,
                mass,
            ),
            axis=-1,
        ).reshape((*shape, 3, 3))
    if center.shape[-1] != 3 or inertia.shape[-2:] != (3, 3):
        raise ValueError(
            "Use a two-component COM and scalar planar inertia, or a three-component COM and 3x3 rotational inertia."
        )
    shape = jnp.broadcast_shapes(mass.shape, center.shape[:-1], inertia.shape[:-2])
    mass = jnp.broadcast_to(mass, shape)
    center = jnp.broadcast_to(center, (*shape, 3))
    inertia = jnp.broadcast_to(inertia, (*shape, 3, 3))
    # Explicit construction supports arbitrary leading batch dimensions.
    x, y, z = jnp.moveaxis(center, -1, 0)
    zero = jnp.zeros_like(x)
    skew = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(
        (*center.shape[:-1], 3, 3)
    )
    mass_skew = mass[..., None, None] * skew
    return jnp.concatenate(
        (
            jnp.concatenate((inertia - mass_skew @ skew, mass_skew), axis=-1),
            jnp.concatenate((-mass_skew, mass[..., None, None] * jnp.eye(3)), axis=-1),
        ),
        axis=-2,
    )


def has_custom_inertia(link) -> bool:
    """Whether a link parameter tree selects full spatial inertia assembly."""
    return link.sectional_inertia is not None or link.tip_inertia is not None


def sectional_inertias(link, geometric: Array) -> Array:
    """Resolve constant or quadrature-sampled overrides against geometric inertia."""
    if link.sectional_inertia is None:
        return geometric
    value = link.sectional_inertia
    if value.ndim == 3:
        value = value[:, None]
    if value.shape[1] not in (1, geometric.shape[1]):
        raise ValueError(
            f"sectional_inertia requires {geometric.shape[1]} quadrature samples per link; got {value.shape[1]}."
        )
    return jnp.broadcast_to(value, geometric.shape)


def weighted_inertias(link, geometric: Array, weights: Array) -> Array:
    """Pack diagonal inertia or full spatial inertia plus a link-tip slot.

    Full matrices use row-major packing in the final dimension. Tip inertia
    carries unit weight and uses the tip state computed by the chain traversal.
    """
    if not has_custom_inertia(link):
        return weights[..., None] * jnp.diagonal(geometric, axis1=-2, axis2=-1)
    matrices = weights[..., None, None] * sectional_inertias(link, geometric)
    tip = (
        jnp.zeros_like(matrices[:, 0]) if link.tip_inertia is None else link.tip_inertia
    )
    matrices = jnp.concatenate((matrices, tip[:, None]), axis=1)
    return matrices.reshape((*matrices.shape[:2], -1))


def spatial_mass_action(packed: Array, value: Array) -> Array:
    """Apply packed quadrature-weighted spatial inertia to vectors or matrices."""
    if packed.shape[-1] in (3, 6):
        return (
            packed.reshape((*packed.shape, *((1,) * (value.ndim - packed.ndim))))
            * value
        )
    dimension = 3 if packed.shape[-1] == 9 else 6
    matrices = packed.reshape((*packed.shape[:-1], dimension, dimension))
    if value.ndim == packed.ndim:
        return (matrices @ value[..., None])[..., 0]
    return matrices @ value


def mass_first_moment(matrices: Array) -> tuple[Array, Array]:
    """Extract mass and mass-times-COM from angular-first physical inertias."""
    if matrices.shape[-1] == 3:
        mass = (matrices[..., 1, 1] + matrices[..., 2, 2]) / 2
        return mass, jnp.stack((matrices[..., 0, 2], -matrices[..., 0, 1]), axis=-1)
    mass = jnp.trace(matrices[..., 3:, 3:], axis1=-2, axis2=-1) / 3
    first_moment = jnp.stack(
        (matrices[..., 2, 4], matrices[..., 0, 5], matrices[..., 1, 3]), axis=-1
    )
    return mass, first_moment


def assemble_inertia_terms(
    packed: Array,
    jacobians: Array,
    derivatives: Array,
    velocity: Array,
    gravity_local: Array,
    *,
    return_matrix: bool = False,
) -> tuple[Array, Array, Array]:
    """Reduce weighted body-frame inertia contributions in SE(2) or SE(3).

    Leading dimensions index arbitrary mass samples. Derivatives contain either
    body Jacobian derivatives or their product with generalized velocity.
    ``return_matrix`` requires complete Jacobian derivatives and returns C;
    otherwise the second result is the velocity bias C @ velocity.
    """
    dim, dofs = jacobians.shape[-2:]
    algebra = se2 if dim == 3 else se3
    J = jacobians.reshape(-1, dofs)
    MJ = spatial_mass_action(packed, jacobians).reshape(-1, dofs)
    M = J.T @ MJ
    eta = jacobians @ velocity
    if return_matrix:
        coadjoint = jax.vmap(algebra.coadjoint)(eta.reshape(-1, dim)).reshape(
            (*eta.shape[:-1], dim, dim)
        )
        wrench = spatial_mass_action(
            packed, derivatives
        ) + coadjoint @ spatial_mass_action(packed, jacobians)
        bias = J.T @ wrench.reshape(-1, dofs)
    else:
        derivative_velocity = (
            derivatives @ velocity
            if derivatives.ndim == jacobians.ndim
            else derivatives
        )
        momentum = spatial_mass_action(packed, eta)
        coadjoint = jax.vmap(algebra.coadjoint_action)(
            eta.reshape(-1, dim), momentum.reshape(-1, dim)
        ).reshape(eta.shape)
        wrench = spatial_mass_action(packed, derivative_velocity) + coadjoint
        bias = J.T @ wrench.reshape(-1)
    gravity = -J.T @ spatial_mass_action(packed, gravity_local).reshape(-1)
    return M, bias, gravity
