import jax
from jax import numpy as jnp
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose
import pytest

from soromox.utils.lie_algebra.se2 import adjoint_se2, exp_SE2, tilde_SE2
from soromox.utils.lie_algebra.se3 import (
    Adjoint_g_SE3,
    Adjoint_g_inv_SE3,
    Adjoint_gi_se3,
    Adjoint_gi_se3_inv,
    Tangent_derivative_gi_se3,
    Tangent_gi_se3,
    adjoint_se3,
    coadjoint_se3,
    exp_SE3,
    exp_gn_SE3,
    hat_SE3,
    log_SE3,
    tilde_SE3,
)
from soromox.utils.tolerance import Tolerance


jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)


def _embed_se2_twist(vec3):
    theta, vx, vy = vec3
    return jnp.array([0.0, 0.0, theta, vx, vy, 0.0])


def _embed_se2_transform(mat3):
    g = jnp.eye(4)
    g = g.at[:2, :2].set(mat3[:2, :2])
    g = g.at[:2, 3].set(mat3[:2, 2])
    return g


def test_tilde_se3_matches_z_axis_rotation():
    theta = jnp.array(0.5)
    omega = jnp.array([0.0, 0.0, theta])

    result = tilde_SE3(omega)

    assert_allclose(result[:2, :2], tilde_SE2(theta), rtol=RTOL, atol=ATOL)
    assert_allclose(result[:, 2], jnp.array([0.0, 0.0, 0.0]), rtol=RTOL, atol=ATOL)
    assert_allclose(result[2, :], jnp.array([0.0, 0.0, 0.0]), rtol=RTOL, atol=ATOL)


def test_hat_se3_embeds_planar_hat():
    vec2 = jnp.array([0.4, -0.7, 1.2])
    vec6 = _embed_se2_twist(vec2)

    result = hat_SE3(vec6)

    expected = jnp.block(
        [
            [tilde_SE3(vec6[:3]), vec6[3:].reshape((3, 1))],
            [jnp.zeros((1, 3)), jnp.zeros((1, 1))],
        ]
    )

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_exp_se3_matches_planar_embedding():
    vec2 = jnp.array([jnp.pi / 4.0, 0.3, -0.5])
    vec6 = _embed_se2_twist(vec2)

    g_se3 = exp_SE3(vec6)
    g_expected = _embed_se2_transform(exp_SE2(vec2))

    assert_allclose(g_se3, g_expected, rtol=RTOL, atol=ATOL)


def test_log_se3_recovers_planar_rotation_without_translation():
    vec2 = jnp.array([0.6, 0.0, 0.0])
    vec6 = _embed_se2_twist(vec2)

    g = exp_SE3(vec6)
    recovered = log_SE3(g, eps=EPS)

    assert_allclose(recovered, vec6, rtol=RTOL, atol=ATOL)


def test_log_se3_pure_translation():
    translation = jnp.array([0.2, -0.15, 0.05])
    g = jnp.eye(4).at[:3, 3].set(translation)

    recovered = log_SE3(g, eps=EPS)
    expected = jnp.concatenate([jnp.zeros(3), translation])

    assert_allclose(recovered, expected, rtol=RTOL, atol=ATOL)


def test_log_se3_pure_rotation_about_x_axis():
    theta = jnp.pi / 8.0
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    R = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )
    g = jnp.eye(4).at[:3, :3].set(R)

    recovered = log_SE3(g, eps=EPS)
    expected = jnp.array([theta, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert_allclose(recovered, expected, rtol=RTOL, atol=ATOL)


def test_log_se3_handles_near_identity_transform():
    vec = jnp.array([1e-9, -2e-9, 3e-9, 5e-4, -4e-4, 3e-4])
    g = exp_SE3(vec)

    recovered = log_SE3(g, eps=EPS)

    assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_log_se3_handles_identity_transform():
    vec = jnp.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
    g = exp_SE3(vec)

    recovered = log_SE3(g, eps=EPS)

    assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_log_se3_round_trip_random_vectors():
    key = jax.random.PRNGKey(321)

    for _ in range(5):
        key, subkey = jax.random.split(key)
        vec = jax.random.uniform(subkey, shape=(6,), minval=-0.5, maxval=0.5)
        g = exp_gn_SE3(vec, EPS)
        recovered = log_SE3(g, eps=EPS)

        assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
        assert_allclose(recovered, vec, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "vec2",
    [
        jnp.array([0.2, -0.3, 0.4]),
        jnp.array([1e-8, 0.5, -0.1]),
    ],
)
def test_exp_gn_se3_matches_matrix_exponential(vec2):
    vec6 = _embed_se2_twist(vec2)

    result = exp_gn_SE3(vec6, EPS)
    expected = expm(hat_SE3(vec6))

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_se3_matches_block_structure():
    vec2 = jnp.array([0.5, 0.2, -0.3])
    vec6 = _embed_se2_twist(vec2)

    result = adjoint_se3(vec6)

    omega_tilde = tilde_SE3(vec6[:3])
    v_tilde = tilde_SE3(vec6[3:])
    expected = jnp.block([[omega_tilde, jnp.zeros((3, 3))], [v_tilde, omega_tilde]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_coadjoint_se3_matches_block_structure():
    vec2 = jnp.array([0.3, -0.4, 0.1])
    vec6 = _embed_se2_twist(vec2)

    result = coadjoint_se3(vec6)

    omega_tilde = tilde_SE3(vec6[:3])
    v_tilde = tilde_SE3(vec6[3:])
    expected = jnp.block([[omega_tilde, v_tilde], [jnp.zeros((3, 3)), omega_tilde]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_se3_matches_planar_embedding():
    vec2 = jnp.array([jnp.pi / 6.0, 0.4, -0.7])
    g2 = exp_SE2(vec2)
    g3 = _embed_se2_transform(g2)

    result = Adjoint_g_SE3(g3)

    R = g3[:3, :3]
    ttilde = tilde_SE3(g3[:3, 3])
    expected = jnp.block([[R, jnp.zeros((3, 3))], [ttilde @ R, R]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_inv_se3_is_inverse():
    vec2 = jnp.array([jnp.pi / 5.0, -0.2, 0.6])
    g3 = _embed_se2_transform(exp_SE2(vec2))

    adj = Adjoint_g_SE3(g3)
    adj_inv = Adjoint_g_inv_SE3(g3)

    assert_allclose(adj @ adj_inv, jnp.eye(6), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_inv @ adj, jnp.eye(6), rtol=RTOL, atol=ATOL)


def test_planar_adjoint_blocks_match_se2():
    vec2 = jnp.array([0.1, -0.2, 0.3])
    vec6 = _embed_se2_twist(vec2)

    adj_3 = adjoint_se3(vec6)
    adj_2 = adjoint_se2(vec2)

    assert_allclose(adj_3[:3, :3], tilde_SE3(vec6[:3]), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_3[3:, :3], tilde_SE3(vec6[3:]), rtol=RTOL, atol=ATOL)

    embedding = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    lhs = adj_3 @ embedding
    rhs = embedding @ adj_2

    assert_allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


def test_adjoint_gi_se3_inverse_matches_identity():
    xi = jnp.array([0.2, -0.1, 0.4, 0.3, -0.5, 0.1])
    s = jnp.array(0.45)

    adj = Adjoint_gi_se3(xi, s, eps=EPS)
    adj_inv = Adjoint_gi_se3_inv(xi, s, eps=EPS)

    assert_allclose(adj @ adj_inv, jnp.eye(6), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_inv @ adj, jnp.eye(6), rtol=RTOL, atol=ATOL)


def test_tangent_gi_se3_zero_theta_matches_truncated_series():
    xi = jnp.array([0.0, 0.0, 0.0, 0.4, -0.3, 0.2])
    s = jnp.array(0.35)

    expected = s * jnp.eye(6) + 0.5 * s**2 * adjoint_se3(xi)

    result = Tangent_gi_se3(xi, s, eps=EPS)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_gi_se3_zero_without_motion():
    xi = jnp.array([0.3, -0.2, 0.4, 0.5, -0.1, 0.2])
    xid = jnp.zeros((6,))
    s = jnp.array(0.6)

    result = Tangent_derivative_gi_se3(xi, xid, s, eps=EPS)
    expected = jnp.zeros((6, 6))

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_tangent_derivative_matches_autodiff_random(N: int = 10):
    key = jax.random.PRNGKey(2)
    samples = 0

    while samples < N:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)

        xi = jax.random.uniform(subkey1, shape=(6,), minval=-1.0, maxval=1.0)
        xid = jax.random.uniform(subkey2, shape=(6,), minval=-1.0, maxval=1.0)
        s = jax.random.uniform(subkey3, shape=(), minval=0.1, maxval=1.0)

        # Skip near-singular angular velocities to avoid the eps fallback dominating.
        if jnp.linalg.norm(xi[:3]) < 1e-3:
            continue

        def tangent_map(xi_):
            return Tangent_gi_se3(xi_, s, eps=EPS)

        _, autodiff = jax.jvp(tangent_map, (xi,), (xid,))
        closed_form = Tangent_derivative_gi_se3(xi, xid, s, eps=EPS)

        assert_allclose(autodiff, closed_form, rtol=RTOL, atol=ATOL)
        samples += 1


def test_se3_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((6,))
    xid_zero = jnp.zeros((6,))
    s = jnp.array(0.35)

    def assert_autodiff_finite(fn, arg, fn_name=None):
        jac_rev = jax.jacrev(fn)(arg)
        jac_fwd = jax.jacfwd(fn)(arg)
        assert jnp.isfinite(jac_rev).all(), (
            "Jacobian (reverse) is not finite of fn {}".format(fn_name)
        )
        assert jnp.isfinite(jac_fwd).all(), (
            "Jacobian (forward) is not finite of fn {}".format(fn_name)
        )

    def tangent_fn(xi):
        return Tangent_gi_se3(xi, s, eps=EPS).reshape(-1)

    assert_autodiff_finite(tangent_fn, xi_zero, fn_name="Tangent_gi_se3")

    def tangent_dot_wrt_xi(xi):
        return Tangent_derivative_gi_se3(xi, xid_zero, s, eps=EPS).reshape(-1)

    def tangent_dot_wrt_xid(xid):
        return Tangent_derivative_gi_se3(xi_zero, xid, s, eps=EPS).reshape(-1)

    for fn, arg in ((tangent_dot_wrt_xi, xi_zero), (tangent_dot_wrt_xid, xid_zero)):
        assert_autodiff_finite(fn, arg, fn_name=fn.__name__)

    def exp_gn_fn(xi):
        return exp_gn_SE3(xi, eps=EPS).reshape(-1)

    assert_autodiff_finite(exp_gn_fn, xi_zero, fn_name="exp_gn_SE3")

    def log_fn(g_flat):
        g = g_flat.reshape((4, 4))
        return log_SE3(g, eps=EPS)

    g_identity = jnp.eye(4).reshape(-1)
    log_at_identity = log_fn(g_identity)
    assert not jnp.isnan(log_at_identity).any()
    assert_allclose(log_at_identity, jnp.zeros((6,)), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__])
