import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.lie_algebra import so2
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)
J = jnp.array([[0.0, -1.0], [1.0, 0.0]])


def test_skew_so2_returns_generator_scaled_by_angle():
    theta = jnp.array(0.5)

    result = so2.skew(theta)

    assert_allclose(result, theta * J, rtol=RTOL, atol=ATOL)


def test_exp_so2_returns_planar_rotation_matrix():
    theta = jnp.pi / 2.0

    result = so2.exp(theta)
    expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_log_so2_inverts_exp_so2():
    theta = jnp.array(-0.7)

    recovered = so2.log(so2.exp(theta), eps=EPS)

    assert_allclose(recovered, theta, rtol=RTOL, atol=ATOL)


def test_log_so2_regularizes_identity_to_zero():
    recovered = so2.log(jnp.eye(2), eps=EPS)

    assert_allclose(recovered, 0.0, rtol=RTOL, atol=ATOL)
