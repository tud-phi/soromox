import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.geometry import poses
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)


def _embed_se2_transform(mat3):
    g = jnp.eye(4)
    g = g.at[:2, :2].set(mat3[:2, :2])
    g = g.at[:2, 3].set(mat3[:2, 2])
    return g


def _quaternion_z(theta):
    half = 0.5 * theta
    return jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)])


def _assert_forward_and_reverse_mode_finite(fn, arg):
    jac_fwd = jax.jacfwd(fn)(arg)
    jac_rev = jax.jacrev(fn)(arg)

    assert jnp.isfinite(jac_fwd).all()
    assert jnp.isfinite(jac_rev).all()


def test_planar_pose_to_transform_produces_rotation_and_translation():
    vec = jnp.array([jnp.pi / 2.0, 1.0, -2.0])
    expected = jnp.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = poses.planar_pose_to_transform(vec)

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_planar_pose_from_transform_inverts_planar_pose_to_transform():
    vec = jnp.array([0.4, -0.7, 1.2])

    g = poses.planar_pose_to_transform(vec)
    recovered = poses.planar_pose_from_transform(g, eps=EPS)

    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_planar_pose_from_transform_inverts_zero_angle_pose():
    vec = jnp.array([0.0, 0.3, -0.2])

    g = poses.planar_pose_to_transform(vec)
    recovered = poses.planar_pose_from_transform(g, eps=EPS)

    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_planar_pose_from_transform_handles_near_identity_transform():
    vec = jnp.array([1e-10, 2e-4, -3e-4])
    g = poses.planar_pose_to_transform(vec)

    recovered = poses.planar_pose_from_transform(g, eps=EPS)

    assert not jnp.isnan(recovered).any()
    assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_planar_pose_from_transform_round_trip_random_vectors():
    key = jax.random.PRNGKey(123)

    for _ in range(5):
        key, subkey = jax.random.split(key)
        vec = jax.random.uniform(subkey, shape=(3,), minval=-0.9, maxval=0.9)
        g = poses.planar_pose_to_transform(vec)
        recovered = poses.planar_pose_from_transform(g, eps=EPS)

        assert_allclose(recovered, vec, rtol=RTOL, atol=ATOL)


def test_quaternion_pose_to_transform_matches_planar_embedding():
    theta = jnp.pi / 4.0
    translation = jnp.array([0.3, -0.5, 0.2])
    pose = jnp.concatenate([_quaternion_z(theta), translation])

    result = poses.quaternion_pose_to_transform(pose)
    expected = (
        _embed_se2_transform(
            poses.planar_pose_to_transform(
                jnp.array([theta, translation[0], translation[1]])
            )
        )
        .at[2, 3]
        .set(translation[2])
    )

    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_quaternion_pose_to_transform_zero_quaternion_is_finite_identity():
    pose = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 3.0])

    result = poses.quaternion_pose_to_transform(pose)

    expected = jnp.eye(4).at[:3, 3].set(jnp.array([1.0, -2.0, 3.0]))
    assert jnp.isfinite(result).all()
    assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


def test_quaternion_pose_to_transform_zero_quaternion_autodiff_is_finite():
    pose = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 3.0])

    _assert_forward_and_reverse_mode_finite(
        lambda pose_: poses.quaternion_pose_to_transform(pose_).reshape(-1),
        pose,
    )
