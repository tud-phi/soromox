import jax
import pytest
from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose

from soromox.autodiff import strict_singularities_mode
from soromox.utils.geometry import poses
from soromox.utils.lie_algebra import se2, se3, so3
from soromox.utils.lie_algebra.jacobian_coefficients import (
    inverse_left_jacobian_series_threshold,
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


def _transform_from_planar_pose(vec2):
    return poses.planar_pose_to_transform(vec2)


def _assert_forward_and_reverse_mode_finite(fn, arg):
    jac_fwd = jax.jacfwd(fn)(arg)
    jac_rev = jax.jacrev(fn)(arg)

    assert jnp.isfinite(jac_fwd).all()
    assert jnp.isfinite(jac_rev).all()


def _nested_grad_of_vmap(fn, singular, regular, tangent):
    values = jnp.stack([singular, regular])
    return jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(
                lambda value: jnp.sum(jax.jvp(fn, (value,), (tangent,))[1])
            )(batch)
        )
    )(values)


def test_hat_se3_embeds_planar_hat():
    vec2 = jnp.array([0.4, -0.7, 1.2])
    vec6 = _embed_se2_twist(vec2)

    result = se3.hat(vec6)

    expected = jnp.block(
        [
            [so3.skew(vec6[:3]), vec6[3:].reshape((3, 1))],
            [jnp.zeros((1, 3)), jnp.zeros((1, 1))],
        ]
    )

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_exp_gn_se3_matches_planar_embedding():
    vec2 = jnp.array([jnp.pi / 4.0, 0.3, -0.5])
    vec6 = _embed_se2_twist(vec2)

    g_se3 = se3.exp(vec6, EPS)
    g_expected = _embed_se2_transform(se2.exp(vec2, EPS))

    assert_allclose(g_se3, g_expected, rtol=RTOL, atol=ATOL)


def test_log_se3_recovers_planar_rotation_without_translation():
    vec2 = jnp.array([0.6, 0.0, 0.0])
    vec6 = _embed_se2_twist(vec2)

    g = se3.exp(vec6, EPS)
    recovered = se3.log(g, eps=EPS)

    assert_allclose(recovered, vec6, rtol=RTOL, atol=ATOL)


def test_log_se3_pure_translation():
    translation = jnp.array([0.2, -0.15, 0.05])
    g = jnp.eye(4).at[:3, 3].set(translation)

    recovered = se3.log(g, eps=EPS)
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

    recovered = se3.log(g, eps=EPS)
    expected = jnp.array([theta, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert_allclose(recovered, expected, rtol=RTOL, atol=ATOL)


def test_log_se3_handles_near_identity_transform():
    vec = jnp.array([1e-9, -2e-9, 3e-9, 5e-4, -4e-4, 3e-4])
    g = se3.exp(vec, EPS)

    recovered = se3.log(g, eps=EPS)

    assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_log_se3_handles_identity_transform():
    vec = jnp.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
    g = se3.exp(vec, EPS)

    recovered = se3.log(g, eps=EPS)

    assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_log_se3_round_trip_random_vectors():
    key = jax.random.PRNGKey(321)

    for _ in range(5):
        key, subkey = jax.random.split(key)
        vec = jax.random.uniform(subkey, shape=(6,), minval=-0.5, maxval=0.5)
        g = se3.exp(vec, EPS)
        recovered = se3.log(g, eps=EPS)

        assert not jnp.isnan(recovered).any(), "Logarithm returned NaN values"
        assert_allclose(recovered, vec, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "xi",
    [
        jnp.zeros(6),
        jnp.array([1e-10, -2e-10, 3e-10, 1e-4, -2e-4, 3e-4]),
        jnp.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3]),
        jnp.array([0.0, 0.0, jnp.pi - 1e-7, 0.1, -0.2, 0.3]),
    ],
)
def test_log_se3_forward_and_reverse_mode_autodiff_is_finite(xi):
    """Test autodiff through identity, small, regular, and near-pi log branches."""
    g = se3.exp(xi, EPS)

    _assert_forward_and_reverse_mode_finite(
        lambda g_flat: se3.log(g_flat.reshape((4, 4)), eps=EPS),
        g.reshape(-1),
    )


@pytest.mark.parametrize(
    ("dtype", "theta", "rtol", "atol"),
    [
        (jnp.float64, 3e-8, 1e-9, 1e-10),
        (jnp.float32, 1e-4, 1e-5, 2e-7),
    ],
)
def test_log_se3_matches_analytic_reverse_mode_derivative_near_identity(
    dtype, theta, rtol, atol
):
    """Compare autodiff with the planar closed-form SE(3) log derivative."""
    translation = jnp.array([0.7, -0.4, 0.2], dtype=dtype)
    theta = jnp.asarray(theta, dtype=dtype)
    eps = jnp.asarray(EPS, dtype=dtype)
    skew_basis = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)

    def transform_from_angle(angle):
        rotation = jnp.array(
            [
                [jnp.cos(angle), -jnp.sin(angle), 0.0],
                [jnp.sin(angle), jnp.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        transform = jnp.eye(4, dtype=dtype).at[:3, :3].set(rotation)
        return transform.at[:3, 3].set(translation)

    actual = jax.jacrev(lambda angle: se3.log(transform_from_angle(angle), eps=eps))(
        theta
    )

    coefficient_derivative = -theta / 6.0 - theta**3 / 180.0 - theta**5 / 5040.0
    translation_wrt_theta = (
        coefficient_derivative * jnp.eye(2, dtype=dtype) - 0.5 * skew_basis
    ) @ translation[:2]
    expected = jnp.concatenate(
        [
            jnp.array([0.0, 0.0, 1.0], dtype=dtype),
            translation_wrt_theta,
            jnp.zeros((1,), dtype=dtype),
        ]
    )

    assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("branch_scale", [1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float64, 2e-8, 2e-10),
        (jnp.float32, 3e-4, 3e-6),
    ],
)
def test_log_se3_arbitrary_axis_hessian_at_series_boundary(
    dtype, rtol, atol, branch_scale
):
    """Check analytic first and second derivatives for an arbitrary axis."""
    axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.sqrt(jnp.dot(axis, axis))
    axis_hat = so3.skew(axis)
    axis_hat_sq = axis_hat @ axis_hat
    translation = jnp.array([0.7, -0.4, 0.2], dtype=dtype)
    theta = branch_scale * inverse_left_jacobian_series_threshold(dtype)
    eps = jnp.asarray(EPS, dtype=dtype)

    def log_from_angle(angle):
        rotation = (
            jnp.eye(3, dtype=dtype)
            + jnp.sin(angle) * axis_hat
            + (1.0 - jnp.cos(angle)) * axis_hat_sq
        )
        transform = jnp.eye(4, dtype=dtype).at[:3, :3].set(rotation)
        transform = transform.at[:3, 3].set(translation)
        return se3.log(transform, eps=eps)

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
            axis,
            (-0.5 * axis_hat - coefficient_first * axis_hat_sq) @ translation,
        ]
    )
    second_expected = jnp.concatenate(
        [
            jnp.zeros((3,), dtype=dtype),
            -coefficient_second * axis_hat_sq @ translation,
        ]
    )

    assert_allclose(first_actual, first_expected, rtol=rtol, atol=atol)
    assert_allclose(second_actual, second_expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float64, 1e-8, 1e-10),
        (jnp.float32, 3e-4, 3e-6),
    ],
)
def test_log_se3_arbitrary_axis_round_trip_jacobian(dtype, rtol, atol):
    """The full log-after-exp Jacobian is identity for an arbitrary twist."""
    axis = jnp.array([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.sqrt(jnp.dot(axis, axis))
    theta = 0.5 * inverse_left_jacobian_series_threshold(dtype)
    xi = jnp.concatenate(
        [
            theta * axis,
            jnp.array([0.7, -0.4, 0.2], dtype=dtype),
        ]
    )
    eps = jnp.asarray(EPS, dtype=dtype)

    actual = jax.jacrev(lambda twist: se3.log(expm(se3.hat(twist)), eps=eps))(xi)

    assert_allclose(actual, jnp.eye(6, dtype=dtype), rtol=rtol, atol=atol)


def test_se2_and_se3_log_share_strict_singularity_policy():
    """Both logarithms expose their removable identity quotient in strict mode."""
    with strict_singularities_mode():
        planar = se2.log(jnp.eye(3), eps=EPS)
        spatial = se3.log(jnp.eye(4), eps=EPS)

    assert not jnp.isfinite(planar).all()
    assert not jnp.isfinite(spatial).all()


def test_log_se3_exact_pi_rotation_is_finite_value_only():
    """Check exact pi, where the logarithm value is finite but not unique."""
    xi = jnp.array([jnp.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    g = se3.exp(xi, EPS)

    recovered = se3.log(g, eps=EPS)

    assert jnp.isfinite(recovered).all()
    assert_allclose(se3.exp(recovered, EPS), g, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "vec2",
    [
        jnp.array([0.2, -0.3, 0.4]),
        jnp.array([1e-8, 0.5, -0.1]),
    ],
)
def test_exp_gn_se3_matches_matrix_exponential(vec2):
    vec6 = _embed_se2_twist(vec2)

    result = se3.exp(vec6, EPS)
    expected = expm(se3.hat(vec6))

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_small_adjoint_se3_matches_block_structure():
    vec2 = jnp.array([0.5, 0.2, -0.3])
    vec6 = _embed_se2_twist(vec2)

    result = se3.small_adjoint(vec6)

    omega_tilde = so3.skew(vec6[:3])
    v_tilde = so3.skew(vec6[3:])
    expected = jnp.block([[omega_tilde, jnp.zeros((3, 3))], [v_tilde, omega_tilde]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_left_jacobian_se3_matches_exponential_series():
    xi = jnp.array([0.2, -0.3, 0.4, -0.7, 0.1, 0.3])
    ad_xi = se3.small_adjoint(xi)
    expected = jnp.eye(6)
    power = expected
    factorial = 1.0
    for order in range(1, 30):
        power = power @ ad_xi
        factorial *= order + 1
        expected = expected + power / factorial

    actual = se3.left_jacobian(xi, EPS)

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_left_jacobian_se3_directional_derivative_matches_autodiff():
    xi = jnp.array([0.2, -0.3, 0.4, -0.7, 0.1, 0.3])
    xid = jnp.array([-0.1, 0.5, 0.2, 0.4, -0.2, 0.6])
    expected = jax.jvp(lambda value: se3.left_jacobian(value, EPS), (xi,), (xid,))[1]

    jacobian, actual = se3.left_jacobian_and_directional_derivative(xi, xid, EPS)

    assert_allclose(jacobian, se3.left_jacobian(xi, EPS), rtol=RTOL, atol=ATOL)
    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(
        actual,
        se3.left_jacobian_directional_derivative(xi, xid, EPS),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "xi",
    [
        jnp.zeros(6),
        jnp.array([1e-8, -2e-8, 3e-8, -0.7, 0.1, 0.3]),
        jnp.array([0.2, -0.3, 0.4, -0.7, 0.1, 0.3]),
    ],
)
def test_fused_exp_left_jacobian_kernel_matches_public_operators(xi):
    xid = jnp.array([-0.1, 0.5, 0.2, 0.4, -0.2, 0.6])

    transform, jacobian, jacobian_derivative = (
        se3._exp_with_left_jacobian_and_directional_derivative(xi, xid, EPS, EPS)
    )
    expected_jacobian, expected_derivative = (
        se3.left_jacobian_and_directional_derivative(xi, xid, EPS)
    )

    assert_allclose(transform, se3.exp(xi, EPS), rtol=RTOL, atol=ATOL)
    assert_allclose(jacobian, expected_jacobian, rtol=RTOL, atol=ATOL)
    assert_allclose(jacobian_derivative, expected_derivative, rtol=RTOL, atol=ATOL)


def _reduced_coefficient_references(angle_sq):
    adjoint_linear = 1.0 + angle_sq**2 * (
        -1.0 / 120.0
        + angle_sq
        * (1.0 / 2520.0 + angle_sq * (-1.0 / 120960.0 + angle_sq / 9979200.0))
    )
    adjoint_quadratic = 0.5 + angle_sq**2 * (
        -1.0 / 720.0
        + angle_sq
        * (1.0 / 20160.0 + angle_sq * (-1.0 / 1209600.0 + angle_sq / 119750400.0))
    )
    adjoint_cubic = 1.0 / 6.0 + angle_sq * (
        -1.0 / 60.0
        + angle_sq
        * (
            1.0 / 1680.0
            + angle_sq
            * (-1.0 / 90720.0 + angle_sq * (1.0 / 7983360.0 - angle_sq / 1037836800.0))
        )
    )
    adjoint_quartic = 1.0 / 24.0 + angle_sq * (
        -1.0 / 360.0
        + angle_sq
        * (
            1.0 / 13440.0
            + angle_sq
            * (
                -1.0 / 907200.0
                + angle_sq * (1.0 / 95800320.0 - angle_sq / 14529715200.0)
            )
        )
    )
    jacobian_quadratic = 1.0 / 6.0 + angle_sq**2 * (
        -1.0 / 5040.0
        + angle_sq
        * (1.0 / 181440.0 + angle_sq * (-1.0 / 13305600.0 + angle_sq / 1556755200.0))
    )
    jacobian_quartic = 1.0 / 120.0 + angle_sq * (
        -1.0 / 2520.0
        + angle_sq
        * (
            1.0 / 120960.0
            + angle_sq
            * (
                -1.0 / 9979200.0
                + angle_sq * (1.0 / 1245404160.0 - angle_sq / 217945728000.0)
            )
        )
    )
    return jnp.stack(
        [
            adjoint_linear,
            adjoint_quadratic,
            adjoint_cubic,
            adjoint_quartic,
            jacobian_quadratic,
            jacobian_quartic,
        ]
    )


@pytest.mark.parametrize("scale", [0.0, 1.0, 1.25])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol", "derivative_atol"),
    [
        (jnp.float64, 3e-9, 3e-11, 5e-10),
        (jnp.float32, 8e-4, 8e-5, 8e-5),
    ],
)
def test_reduced_coefficients_and_derivatives(
    scale, dtype, rtol, atol, derivative_atol
):
    angle = scale * se3._left_jacobian_series_threshold(dtype)
    angle_sq = angle**2
    adjoint_actual = jnp.stack(se3._adjoint_exponential_coefficients(angle_sq, 0.0))
    jacobian_actual, derivative_actual = (
        se3._left_jacobian_coefficients_and_x_derivatives(angle_sq, 0.0)
    )
    reference = _reduced_coefficient_references(angle_sq)
    jacobian_reference = reference[jnp.array([1, 4, 3, 5])]
    derivative_reference = jax.jacrev(
        lambda value: _reduced_coefficient_references(value)[jnp.array([1, 4, 3, 5])]
    )(angle_sq)

    assert_allclose(adjoint_actual, reference[:4], rtol=rtol, atol=atol)
    assert_allclose(
        jnp.stack(jacobian_actual), jacobian_reference, rtol=rtol, atol=atol
    )
    assert_allclose(
        jnp.stack(derivative_actual),
        derivative_reference,
        rtol=rtol,
        atol=derivative_atol,
    )


def test_strict_mode_exposes_exp_and_reduced_coefficient_quotients():
    with strict_singularities_mode():
        transform = se3.exp(jnp.array([0.0, 0.0, 0.0, 0.7, -0.4, 0.2]), 0.0)
        adjoint_coefficients = se3._adjoint_exponential_coefficients(
            jnp.array(0.0), 0.0
        )
        jacobian_coefficients, _ = se3._left_jacobian_coefficients_and_x_derivatives(
            jnp.array(0.0), 0.0
        )

    results = (transform, *adjoint_coefficients, *jacobian_coefficients)
    assert all(not jnp.isfinite(result).all() for result in results)


def test_coadjoint_se3_matches_block_structure():
    vec2 = jnp.array([0.3, -0.4, 0.1])
    vec6 = _embed_se2_twist(vec2)

    result = se3.coadjoint(vec6)

    omega_tilde = so3.skew(vec6[:3])
    v_tilde = so3.skew(vec6[3:])
    expected = jnp.block([[omega_tilde, v_tilde], [jnp.zeros((3, 3)), omega_tilde]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_se3_matches_planar_embedding():
    vec2 = jnp.array([jnp.pi / 6.0, 0.4, -0.7])
    g2 = _transform_from_planar_pose(vec2)
    g3 = _embed_se2_transform(g2)

    result = se3.adjoint(g3)

    R = g3[:3, :3]
    ttilde = so3.skew(g3[:3, 3])
    expected = jnp.block([[R, jnp.zeros((3, 3))], [ttilde @ R, R]])

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_adjoint_g_inv_se3_is_inverse():
    vec2 = jnp.array([jnp.pi / 5.0, -0.2, 0.6])
    g3 = _embed_se2_transform(_transform_from_planar_pose(vec2))

    adj = se3.adjoint(g3)
    adj_inv = se3.adjoint_inverse(g3)

    assert_allclose(adj @ adj_inv, jnp.eye(6), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_inv @ adj, jnp.eye(6), rtol=RTOL, atol=ATOL)


def test_planar_adjoint_blocks_match_se2():
    vec2 = jnp.array([0.1, -0.2, 0.3])
    vec6 = _embed_se2_twist(vec2)

    adj_3 = se3.small_adjoint(vec6)
    adj_2 = se2.small_adjoint(vec2)

    assert_allclose(adj_3[:3, :3], so3.skew(vec6[:3]), rtol=RTOL, atol=ATOL)
    assert_allclose(adj_3[3:, :3], so3.skew(vec6[3:]), rtol=RTOL, atol=ATOL)

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


def test_se3_helpers_are_autodiff_finite_at_zero():
    xi_zero = jnp.zeros((6,))

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
        return se3.exp(xi, eps=EPS).reshape(-1)

    assert_autodiff_finite(exp_gn_fn, xi_zero, fn_name="se3.exp")
    assert_autodiff_finite(
        lambda xi: se3.left_jacobian(xi, EPS),
        xi_zero,
        fn_name="se3.left_jacobian",
    )

    def log_fn(g_flat):
        g = g_flat.reshape((4, 4))
        return se3.log(g, eps=EPS)

    g_identity = jnp.eye(4).reshape(-1)
    log_at_identity = log_fn(g_identity)
    assert not jnp.isnan(log_at_identity).any()
    assert_allclose(log_at_identity, jnp.zeros((6,)), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("name", "fn", "singular", "regular", "tangent"),
    [
        (
            "se3.exp",
            lambda xi: se3.exp(xi, EPS),
            jnp.zeros(6),
            jnp.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.4]),
            jnp.linspace(0.1, 0.6, 6),
        ),
        (
            "se3.log",
            lambda g: se3.log(g, EPS),
            jnp.eye(4),
            se3.exp(jnp.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.4]), EPS),
            jnp.ones((4, 4)),
        ),
    ],
    ids=["se3.exp", "se3.log"],
)
def test_nested_grad_of_vmap_is_finite_at_se3_branch_points(
    name, fn, singular, regular, tangent
):
    gradient = _nested_grad_of_vmap(fn, singular, regular, tangent)

    assert jnp.isfinite(gradient).all(), (
        f"{name} produced a non-finite nested batched gradient"
    )


BRANCH_EPS = 1e1 * float(jnp.finfo(jnp.float64).eps)
TANGENT_BRANCH_EPS = float(jnp.sqrt(jnp.asarray(BRANCH_EPS)))
XI_STRAIGHT = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
ARC_LENGTH = jnp.array(0.1)


def _batched_grad(fn, x: Array, batch: int = 3) -> Array:
    """Differentiate ``sum(vmap(fn)(x))`` to exercise batched branch lowering."""
    stacked = jnp.broadcast_to(x, (batch,) + x.shape)
    return jax.grad(lambda v: jnp.sum(jax.vmap(fn)(v)))(stacked)


BRANCH_SAFETY_CASES = {
    "se3.exp": (lambda xi: se3.exp(xi, BRANCH_EPS), XI_STRAIGHT),
    "so3.exp": (lambda xi: so3.exp(xi[:3], BRANCH_EPS), XI_STRAIGHT),
    "se3.log": (lambda g: se3.log(g, BRANCH_EPS), jnp.eye(4)),
    "so3.log": (lambda R: so3.log(R, BRANCH_EPS), jnp.eye(3)),
}


@pytest.mark.parametrize("name", sorted(BRANCH_SAFETY_CASES))
def test_grad_of_vmap_is_finite_at_zero_rotation(name: str):
    """The batched lowering of every small-angle branch stays finite."""
    fn, arg = BRANCH_SAFETY_CASES[name]

    grad = _batched_grad(lambda x: jnp.sum(fn(x)), arg)

    assert jnp.isfinite(grad).all(), f"{name} produced a non-finite batched gradient"


if __name__ == "__main__":
    pytest.main([__file__])
