import jax
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose

from soromox.utils.geometry import poses
from soromox.utils.lie_algebra import se2
from soromox.utils.lie_algebra.jacobian_coefficients import (
    inverse_left_jacobian_series_threshold,
)
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


@pytest.mark.parametrize("theta", [1e-10, 1e-8])
def test_log_se2_recovers_translation_at_tiny_nonzero_rotation(theta):
    """Avoid cancellation in ``1 - cos(theta)`` just above the zero branch."""
    expected = jnp.array([theta, 0.7, -0.4], dtype=jnp.float64)
    transform = expm(se2.hat(expected))

    recovered = se2.log(transform, eps=EPS)
    jacobian = jax.jacrev(lambda g: se2.log(g, eps=EPS))(transform)

    assert_allclose(recovered, expected, rtol=1e-12, atol=1e-13)
    assert jnp.isfinite(jacobian).all()


@pytest.mark.parametrize(
    ("dtype", "theta", "rtol", "atol"),
    [
        (jnp.float64, 1e-12, 1e-10, 1e-12),
        (jnp.float32, 1e-5, 1e-5, 2e-7),
    ],
)
def test_log_se2_matches_analytic_reverse_mode_jacobian_near_identity(
    dtype, theta, rtol, atol
):
    """Compare autodiff with the closed-form small-angle log Jacobian."""
    translation = jnp.array([0.7, -0.4], dtype=dtype)
    theta = jnp.asarray(theta, dtype=dtype)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    identity = jnp.eye(2, dtype=dtype)
    skew_basis = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    rotation = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ],
        dtype=dtype,
    )
    transform = jnp.block(
        [
            [rotation, translation.reshape((2, 1))],
            [jnp.zeros((1, 2), dtype=dtype), jnp.ones((1, 1), dtype=dtype)],
        ]
    )

    actual = jax.jacrev(lambda g_flat: se2.log(g_flat.reshape((3, 3)), eps=eps))(
        transform.reshape(-1)
    )

    recovered_theta = jnp.arctan2(rotation[1, 0], rotation[0, 0])
    theta_sq = recovered_theta**2
    coefficient = 1.0 - theta_sq / 12.0 - theta_sq**2 / 720.0
    coefficient_derivative = -recovered_theta / 6.0 - recovered_theta**3 / 180.0
    inverse_left_jacobian = coefficient * identity - 0.5 * recovered_theta * skew_basis
    translation_wrt_theta = (
        coefficient_derivative * identity - 0.5 * skew_basis
    ) @ translation

    rotation_norm_sq = rotation[0, 0] ** 2 + rotation[1, 0] ** 2
    theta_wrt_r00 = -rotation[1, 0] / rotation_norm_sq
    theta_wrt_r10 = rotation[0, 0] / rotation_norm_sq

    expected = jnp.zeros((3, 9), dtype=dtype)
    expected = expected.at[0, 0].set(theta_wrt_r00)
    expected = expected.at[0, 3].set(theta_wrt_r10)
    expected = expected.at[1:, 0].set(translation_wrt_theta * theta_wrt_r00)
    expected = expected.at[1:, 3].set(translation_wrt_theta * theta_wrt_r10)
    expected = expected.at[1:, 2].set(inverse_left_jacobian[:, 0])
    expected = expected.at[1:, 5].set(inverse_left_jacobian[:, 1])

    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("branch_scale", [1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float64, 1e-8, 1e-10),
        (jnp.float32, 2e-4, 2e-6),
    ],
)
def test_log_se2_reverse_mode_hessian_at_series_boundary(
    dtype, rtol, atol, branch_scale
):
    """Check first and second derivatives at and above the series cutoff."""
    translation = jnp.array([0.7, -0.4], dtype=dtype)
    skew_basis = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    theta = branch_scale * inverse_left_jacobian_series_threshold(dtype)
    eps = jnp.zeros((), dtype=dtype)

    def log_from_angle(angle):
        rotation = jnp.array(
            [
                [jnp.cos(angle), -jnp.sin(angle)],
                [jnp.sin(angle), jnp.cos(angle)],
            ],
            dtype=dtype,
        )
        transform = jnp.block(
            [
                [rotation, translation.reshape((2, 1))],
                [jnp.zeros((1, 2), dtype=dtype), jnp.ones((1, 1), dtype=dtype)],
            ]
        )
        return se2.log(transform, eps=eps)

    first_actual = jax.jacrev(log_from_angle)(theta)
    second_actual = jax.jacrev(jax.jacrev(log_from_angle))(theta)

    coefficient_first = (
        -theta / 6.0 - theta**3 / 180.0 - theta**5 / 5040.0 - theta**7 / 151200.0
    )
    coefficient_second = (
        -1.0 / 6.0 - theta**2 / 60.0 - theta**4 / 1008.0 - theta**6 / 21600.0
    )
    first_expected = jnp.concatenate(
        [
            jnp.ones((1,), dtype=dtype),
            (coefficient_first * jnp.eye(2, dtype=dtype) - 0.5 * skew_basis)
            @ translation,
        ]
    )
    second_expected = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=dtype),
            coefficient_second * translation,
        ]
    )

    assert_allclose(first_actual, first_expected, rtol=rtol, atol=atol)
    assert_allclose(second_actual, second_expected, rtol=rtol, atol=atol)


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


def test_se2_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((3,))

    def assert_autodiff_finite(fn, arg, fn_name=None):
        jac_rev = jax.jacrev(fn)(arg)
        jac_fwd = jax.jacfwd(fn)(arg)
        assert jnp.isfinite(jac_rev).all(), (
            f"Jacobian (reverse) is not finite of fn {fn_name}"
        )
        assert jnp.isfinite(jac_fwd).all(), (
            f"Jacobian (forward) is not finite of fn {fn_name}"
        )

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
