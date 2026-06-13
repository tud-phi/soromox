import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.lie_algebra import so2, so3
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)


def test_skew_so3_matches_z_axis_planar_generator():
    theta = jnp.array(0.5)
    omega = jnp.array([0.0, 0.0, theta])

    result = so3.skew(omega)

    assert_allclose(result[:2, :2], so2.skew(theta), rtol=RTOL, atol=ATOL)
    assert_allclose(result[:, 2], jnp.array([0.0, 0.0, 0.0]), rtol=RTOL, atol=ATOL)
    assert_allclose(result[2, :], jnp.array([0.0, 0.0, 0.0]), rtol=RTOL, atol=ATOL)


def test_exp_so3_matches_z_axis_planar_rotation():
    theta = jnp.pi / 4.0
    omega = jnp.array([0.0, 0.0, theta])

    result = so3.exp(omega, eps=EPS)

    expected = jnp.eye(3).at[:2, :2].set(so2.exp(theta))
    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_log_so3_inverts_exp_so3():
    omega = jnp.array([0.2, -0.4, 0.3])

    recovered = so3.log(so3.exp(omega, eps=EPS), eps=EPS)

    assert_allclose(recovered, omega, rtol=RTOL, atol=ATOL)


def test_log_so3_handles_near_pi_rotation():
    omega = jnp.array([0.0, 0.0, jnp.deg2rad(179.99)])

    recovered = so3.log(so3.exp(omega, eps=EPS), eps=EPS)

    assert_allclose(recovered, omega, rtol=1e-5, atol=1e-5)


def test_log_so3_small_angle_branch_recovers_tiny_rotation():
    omega = jnp.array([1e-10, -2e-10, 3e-10])

    recovered = so3.log(so3.exp(omega, eps=EPS), eps=EPS)

    assert jnp.isfinite(recovered).all()
    assert_allclose(recovered, omega, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "axis",
    [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0),
    ],
)
def test_log_so3_exact_pi_branch_is_finite_and_reconstructs(axis):
    R = so3.exp(jnp.pi * axis, eps=EPS)

    recovered = so3.log(R, eps=EPS)

    assert jnp.isfinite(recovered).all()
    assert_allclose(jnp.linalg.norm(recovered), jnp.pi, rtol=RTOL, atol=ATOL)
    assert_allclose(so3.exp(recovered, eps=EPS), R, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "omega",
    [
        jnp.zeros(3),
        jnp.array([1e-10, -2e-10, 3e-10]),
        jnp.array([0.2, -0.4, 0.3]),
        (jnp.pi - 1e-7) * jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0),
    ],
)
def test_log_so3_is_forward_and_reverse_mode_autodiff_finite(omega):
    R = so3.exp(omega, eps=EPS)

    def log_flat(R_flat):
        return so3.log(R_flat.reshape((3, 3)), eps=EPS)

    jac_fwd = jax.jacfwd(log_flat)(R.reshape(-1))
    jac_rev = jax.jacrev(log_flat)(R.reshape(-1))

    assert jnp.isfinite(jac_fwd).all()
    assert jnp.isfinite(jac_rev).all()
