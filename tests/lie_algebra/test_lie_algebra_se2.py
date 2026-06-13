import jax
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose

from soromox.utils.geometry import poses
from soromox.utils.lie_algebra import constant_strain, se2
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)
J = jnp.array([[0.0, -1.0], [1.0, 0.0]])


def test_log_se2_pure_translation():
    translation = jnp.array([0.35, -0.27])
    g = jnp.array(
        [
            [1.0, 0.0, translation[0]],
            [0.0, 1.0, translation[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    recovered = se2.log(g, eps=EPS)
    expected = jnp.array([0.0, translation[0], translation[1]])

    assert_allclose(recovered, expected, rtol=RTOL, atol=ATOL)


def test_log_se2_pure_rotation():
    theta = jnp.pi / 7.0
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    g = jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    recovered = se2.log(g, eps=EPS)
    expected = jnp.array([theta, 0.0, 0.0])

    assert_allclose(recovered, expected, rtol=RTOL, atol=ATOL)


def test_log_se2_inverts_exp_se2():
    vec = jnp.array([0.4, -0.7, 1.2])

    g = se2.exp(vec, eps=EPS)
    recovered = se2.log(g, eps=EPS)

    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "vec",
    [
        jnp.array([0.3, -0.2, 0.5]),
        jnp.array([1e-8, 0.7, -0.4]),
    ],
)
def test_exp_gn_se2_matches_matrix_exponential(vec):
    result = se2.exp(vec, EPS)
    expected = expm(se2.hat(vec))

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_hat_se2_returns_expected_matrix():
    vec = jnp.array([0.5, 2.0, -1.0])
    expected = jnp.array(
        [
            [0.0, -0.5, 2.0],
            [0.5, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = se2.hat(vec)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_small_adjoint_se2_matches_closed_form():
    vec = jnp.array([0.5, 2.0, -1.0])
    v1, v2 = vec[1], vec[2]
    ang = vec[0]
    expected = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [v2, 0.0, -ang],
            [-v1, ang, 0.0],
        ]
    )

    result = se2.small_adjoint(vec)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_coadjoint_se2_matches_closed_form():
    vec = jnp.array([0.5, 2.0, -1.0])
    v1, v2 = vec[1], vec[2]
    ang = vec[0]
    expected = jnp.array(
        [
            [0.0, v2, -v1],
            [0.0, 0.0, -ang],
            [0.0, ang, 0.0],
        ]
    )

    result = se2.coadjoint(vec)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_se2_matches_manual_construction():
    vec = jnp.array([jnp.pi / 3.0, 0.5, -0.2])
    g = poses.planar_pose_to_transform(vec)
    R = g[:2, :2]
    t = g[:2, 2].reshape((2, 1))
    expected = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones((1, 1)), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-J @ t, R], axis=1),
        ],
        axis=0,
    )

    result = se2.adjoint(g)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_inv_se2_is_matrix_inverse():
    vec = jnp.array([jnp.pi / 4.0, 0.2, -0.7])
    g = poses.planar_pose_to_transform(vec)

    adj = se2.adjoint(g)
    adj_inv = se2.adjoint_inverse(g)

    assert_allclose(adj @ adj_inv, jnp.eye(3), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_inv @ adj, jnp.eye(3), rtol=RTOL, atol=ATOL)


def test_adjoint_gi_se2_zero_theta_matches_first_order_series():
    xi = jnp.array([0.0, 1.0, -2.0])
    s = jnp.array(0.3)

    expected = jnp.eye(3) + s * se2.small_adjoint(xi)

    result = constant_strain.adjoint_se2(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_gi_se2_inverse_matches_identity():
    xi = jnp.array([0.7, 0.25, -0.1])
    s = jnp.array(0.5)

    adj = constant_strain.adjoint_se2(xi, s, eps=EPS)
    adj_inv = constant_strain.adjoint_inverse_se2(xi, s, eps=EPS)

    assert_allclose(adj @ adj_inv, jnp.eye(3), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_inv @ adj, jnp.eye(3), rtol=RTOL, atol=ATOL)


def test_tangent_gi_se2_zero_theta_matches_truncated_series():
    xi = jnp.array([0.0, 1.0, -0.5])
    s = jnp.array(0.4)

    expected = s * jnp.eye(3) + 0.5 * s**2 * se2.small_adjoint(xi)

    result = constant_strain.tangent_se2(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_gi_se2_derivative_zero_without_motion():
    xi = jnp.array([0.7, 0.2, -0.3])
    xid = jnp.zeros((3,))
    s = jnp.array(0.6)

    result = constant_strain.tangent_derivative_se2(xi, xid, s, eps=EPS)
    expected = jnp.zeros((3, 3))

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_gi_se2_zero_theta_matches_truncated_series():
    xi = jnp.array([0.0, 0.5, -0.1])
    xid = jnp.array([0.1, -0.4, 0.2])
    s = jnp.array(0.2)

    expected = 0.5 * s**2 * se2.small_adjoint(xid)

    result = constant_strain.tangent_derivative_se2(xi, xid, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_matches_autodiff(N: int = 10):
    key = jax.random.PRNGKey(0)
    for _i in range(N):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)

        # xi = jnp.array([0.7, -0.3, 0.25])
        # xid = jnp.array([-0.2, 0.4, -0.35])
        # s = jnp.array(0.6)
        # randomly sample xi, xid, s
        xi = jax.random.uniform(subkey1, shape=(3,), minval=-1.0, maxval=1.0)
        xid = jax.random.uniform(subkey2, shape=(3,), minval=-1.0, maxval=1.0)
        s = jax.random.uniform(subkey3, shape=(), minval=0.1, maxval=1.0)

        def tangent_map(xi_):
            return constant_strain.tangent_se2(xi_, s, eps=EPS)

        _, autodiff = jax.jvp(tangent_map, (xi,), (xid,))
        closed_form = constant_strain.tangent_derivative_se2(xi, xid, s, eps=EPS)

        assert_allclose(autodiff, closed_form, rtol=RTOL, atol=ATOL)


def test_se2_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((3,))
    xid_zero = jnp.zeros((3,))
    s = jnp.array(0.4)

    def assert_autodiff_finite(fn, arg, fn_name=None):
        jac_rev = jax.jacrev(fn)(arg)
        jac_fwd = jax.jacfwd(fn)(arg)
        assert jnp.isfinite(jac_rev).all(), (
            f"Jacobian (reverse) is not finite of fn {fn_name}"
        )
        assert jnp.isfinite(jac_fwd).all(), (
            f"Jacobian (forward) is not finite of fn {fn_name}"
        )

    def tangent_fn(xi):
        return constant_strain.tangent_se2(xi, s, eps=EPS).reshape(-1)

    assert_autodiff_finite(tangent_fn, xi_zero, fn_name="constant_strain.tangent_se2")

    def tangent_dot_wrt_xi(xi):
        return constant_strain.tangent_derivative_se2(xi, xid_zero, s, eps=EPS).reshape(
            -1
        )

    def tangent_dot_wrt_xid(xid):
        return constant_strain.tangent_derivative_se2(xi_zero, xid, s, eps=EPS).reshape(
            -1
        )

    for fn, arg in ((tangent_dot_wrt_xi, xi_zero), (tangent_dot_wrt_xid, xid_zero)):
        assert_autodiff_finite(fn, arg, fn_name=fn.__name__)

    def exp_gn_fn(xi):
        return se2.exp(xi, eps=EPS).reshape(-1)

    assert_autodiff_finite(exp_gn_fn, xi_zero, fn_name="se2.exp")

    def log_fn(g_flat):
        g = g_flat.reshape((3, 3))
        return se2.log(g, eps=EPS)

    g_identity = jnp.eye(3).reshape(-1)
    assert_autodiff_finite(log_fn, g_identity, fn_name="se2.log")


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
