"""Degenerate-geometry autodiff tests for the McKibben transmission.

The existing McKibben tests all build well-formed attachment geometry from the
cached UMArm parameters, so they never reach the configuration where a moving
attachment point coincides with its fixed counterpart. At that point the
displacement vector is zero, and taking ``jnp.linalg.norm`` of it -- or dividing
by that norm to obtain a direction -- differentiates to ``NaN``.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_allclose

from soromox.actuation.mckibben import (
    ArticulatedMcKibbenTransmission,
    ArticulatedMcKibbenTransmissionParams,
)

jax.config.update("jax_enable_x64", True)

NUM_GROUPS = 1
ACTUATORS_PER_GROUP = 2


def make_transmission(coincident: bool) -> ArticulatedMcKibbenTransmission:
    """Build a single-group transmission with optional coincident endpoints.

    Args:
        coincident: If ``True``, the first actuator's moving point sits exactly
            on its fixed point at ``q = 0``, making the displacement vector zero.

    Returns:
        Transmission whose geometry is well formed apart from the requested
        degeneracy.
    """
    shape = (NUM_GROUPS, ACTUATORS_PER_GROUP)
    moving_points = jnp.array([[[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]]])
    if coincident:
        fixed_points = jnp.array([[[0.0, 0.02, 0.0], [0.0, -0.02, -0.1]]])
    else:
        fixed_points = jnp.array([[[0.0, 0.02, -0.1], [0.0, -0.02, -0.1]]])

    params = ArticulatedMcKibbenTransmissionParams(
        moving_base_transform=jnp.eye(4)[None, ...],
        moving_points=moving_points,
        fixed_points=fixed_points,
        reference_length=jnp.full(shape, 0.1),
        length_offset=jnp.zeros(shape),
        thread_length=jnp.full(shape, 0.5),
        num_turns=jnp.full(shape, 10.0),
        joint_pair_indices=jnp.array([[0, 1]]),
    )
    return ArticulatedMcKibbenTransmission(params)


@pytest.mark.parametrize("coincident", [False, True])
def test_effective_lengths_gradient_is_finite(coincident: bool):
    transmission = make_transmission(coincident)

    grad = jax.jacrev(transmission.effective_lengths)(jnp.zeros(2))

    assert jnp.isfinite(grad).all(), "d(effective_lengths)/dq is not finite" + (
        " for coincident attachment points" if coincident else ""
    )


@pytest.mark.parametrize("coincident", [False, True])
def test_length_jacobian_gradient_is_finite(coincident: bool):
    """Cover the direction vector, which divides by the displacement norm.

    This exercises the private group-local hook directly because the public
    moment matrix additionally needs an articulated host, which is irrelevant to
    the degeneracy under test.
    """
    transmission = make_transmission(coincident)
    group_index = jnp.array(0)

    def total(q: Array) -> Array:
        lengths, jacobian = transmission._length_jacobian_local_group(q, group_index)
        return jnp.sum(lengths) + jnp.sum(jacobian)

    value = total(jnp.zeros(2))
    grad = jax.grad(total)(jnp.zeros(2))

    assert jnp.isfinite(value), "length Jacobian is not finite"
    assert jnp.isfinite(grad).all(), "d(length Jacobian)/dq is not finite" + (
        " for coincident attachment points" if coincident else ""
    )


def test_effective_lengths_match_the_analytic_values():
    # Well-formed: both actuators span exactly the 0.1 m reference length at q = 0.
    well_formed = make_transmission(coincident=False)
    assert_allclose(
        well_formed.effective_lengths(jnp.zeros(2)), jnp.zeros(2), atol=1e-12
    )

    # Collapsed: first actuator has raw length 0, so effective length is -reference.
    degenerate = make_transmission(coincident=True)
    lengths = degenerate.effective_lengths(jnp.zeros(2))
    assert_allclose(float(lengths[0]), -0.1, atol=1e-12)
    assert_allclose(float(lengths[1]), 0.0, atol=1e-12)


@pytest.mark.parametrize("coincident", [False, True])
def test_coordinate_gradient_is_finite(coincident: bool):
    """Exercise the public work-coordinate path built on effective lengths."""
    transmission = make_transmission(coincident)

    jacobian = jax.jacrev(lambda q: transmission.coordinates(None, q))(jnp.zeros(2))

    assert jnp.isfinite(jacobian).all()


@pytest.mark.parametrize("coincident", [False, True])
def test_moment_matrix_gradient_is_finite(coincident: bool):
    """Exercise the public moment-matrix path through normalized directions."""
    transmission = make_transmission(coincident)
    robot = SimpleNamespace(num_velocities=2)

    jacobian = jax.jacrev(lambda q: transmission.moment_matrix(robot, q))(jnp.zeros(2))

    assert jnp.isfinite(jacobian).all()
