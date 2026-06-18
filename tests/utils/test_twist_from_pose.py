"""Tests for twist_callable_from_pose_callable_factory utility function."""

# ruff: noqa: E402

import jax

jax.config.update("jax_enable_x64", True)  # double precision

import jax.numpy as jnp
import pytest

from soromox.utils.geometry.rotations import RotationRepresentation
from soromox.utils.twist import twist_callable_from_pose_callable_factory


class TestCreateTwistFromPoseFnRotationVector:
    """Tests for twist derivation with ROTATION_VECTOR representation."""

    def test_constant_pose_gives_zero_velocity(self):
        """A constant pose should give zero velocity."""

        def pose_fn(t):
            return jnp.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        twist = twist_fn(0.5)

        assert twist.shape == (6,)
        assert jnp.allclose(twist, jnp.zeros(6), atol=1e-10)

    def test_linear_motion_gives_constant_velocity(self):
        """Linear motion in pose space should give constant velocity."""

        def pose_fn(t):
            # Orientation changes at rate [0.1, 0.0, 0.0]
            # Position changes at rate [0.0, 0.0, 0.05]
            return jnp.array([t * 0.1, 0.0, 0.0, 0.0, 0.0, 0.2 + t * 0.05])

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        twist = twist_fn(0.5)
        expected_twist = jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.05])

        assert twist.shape == (6,)
        assert jnp.allclose(twist, expected_twist, atol=1e-6)

    def test_sinusoidal_motion(self):
        """Sinusoidal motion should give cosine velocity."""
        omega = 2.0
        amplitude = 0.1

        def pose_fn(t):
            return jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,  # No rotation
                    amplitude * jnp.sin(omega * t),
                    0.0,
                    0.0,  # Position
                ]
            )

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        t_test = 0.5
        twist = twist_fn(t_test)
        expected_vx = omega * amplitude * jnp.cos(omega * t_test)

        assert jnp.allclose(twist[3], expected_vx, atol=1e-6)
        assert jnp.allclose(twist[:3], jnp.zeros(3), atol=1e-10)

    def test_multiple_points(self):
        """Test with multiple points."""

        def pose_fn(t):
            return jnp.array(
                [
                    0.0,
                    0.0,
                    t * 0.1,
                    0.0,
                    0.0,
                    0.1,  # Point 1
                    0.0,
                    0.0,
                    t * 0.2,
                    0.0,
                    0.0,
                    0.2,  # Point 2
                ]
            )

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=2,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        twist = twist_fn(0.5)
        expected_twist = jnp.array(
            [
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,  # Point 1
                0.0,
                0.0,
                0.2,
                0.0,
                0.0,
                0.0,  # Point 2
            ]
        )

        assert twist.shape == (12,)
        assert jnp.allclose(twist, expected_twist, atol=1e-6)


class TestCreateTwistFromPoseFnPlanar:
    """Tests for twist derivation with planar poses."""

    def test_planar_constant_pose(self):
        """Constant planar pose should give zero velocity."""

        def pose_fn(t):
            return jnp.array([0.5, 1.0, 2.0])  # [theta, x, y]

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=3,
            n_orientation_dim=1,
            rotation_representation=None,
            is_planar=True,
        )

        twist = twist_fn(0.5)

        assert twist.shape == (3,)
        assert jnp.allclose(twist, jnp.zeros(3), atol=1e-10)

    def test_planar_linear_motion(self):
        """Linear planar motion should give constant velocity."""

        def pose_fn(t):
            return jnp.array([t * 0.1, t * 1.0, t * 2.0])  # [theta, x, y]

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=3,
            n_orientation_dim=1,
            rotation_representation=None,
            is_planar=True,
        )

        twist = twist_fn(0.5)
        expected_twist = jnp.array([0.1, 1.0, 2.0])  # [omega_z, vx, vy]

        assert twist.shape == (3,)
        assert jnp.allclose(twist, expected_twist, atol=1e-6)


class TestCreateTwistFromPoseFnQuaternion:
    """Tests for twist derivation with QUATERNION representation."""

    def test_quaternion_constant_pose(self):
        """Constant quaternion pose should give zero angular velocity."""

        # Unit quaternion for identity rotation: [1, 0, 0, 0]
        def pose_fn(t):
            return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [quat, pos]

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=7,
            n_orientation_dim=4,
            rotation_representation=RotationRepresentation.QUATERNION,
            is_planar=False,
        )

        twist = twist_fn(0.5)

        assert twist.shape == (6,)  # Velocity is always 6D
        assert jnp.allclose(twist, jnp.zeros(6), atol=1e-10)

    def test_quaternion_linear_position_motion(self):
        """Linear position motion with constant quaternion."""

        def pose_fn(t):
            # Identity quaternion (constant), position moving
            return jnp.array([1.0, 0.0, 0.0, 0.0, t * 0.1, 0.0, 0.0])

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=7,
            n_orientation_dim=4,
            rotation_representation=RotationRepresentation.QUATERNION,
            is_planar=False,
        )

        twist = twist_fn(0.5)

        # Angular velocity should be zero, linear velocity [0.1, 0, 0]
        assert twist.shape == (6,)
        assert jnp.allclose(twist[:3], jnp.zeros(3), atol=1e-6)
        assert jnp.allclose(twist[3:], jnp.array([0.1, 0.0, 0.0]), atol=1e-6)


class TestCreateTwistFromPoseFnJITCompatibility:
    """Tests for JIT compatibility."""

    def test_jit_compatible(self):
        """The twist function should be JIT-compatible."""

        def pose_fn(t):
            return jnp.array([0.0, 0.0, t * 0.1, 0.0, 0.0, t * 0.05])

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        twist_fn_jit = jax.jit(twist_fn)

        twist = twist_fn_jit(0.5)
        expected_twist = jnp.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.05])

        assert jnp.allclose(twist, expected_twist, atol=1e-6)

    def test_vmap_over_time(self):
        """The twist function should work with vmap over time."""

        def pose_fn(t):
            return jnp.array([0.0, 0.0, t * 0.1, 0.0, 0.0, t * 0.05])

        twist_fn = twist_callable_from_pose_callable_factory(
            pose_fn=pose_fn,
            n_points=1,
            n_pose_dim=6,
            n_orientation_dim=3,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
            is_planar=False,
        )

        ts = jnp.linspace(0, 1, 10)
        twists = jax.vmap(twist_fn)(ts)

        assert twists.shape == (10, 6)
        # All should have the same velocity (linear motion)
        for i in range(10):
            expected = jnp.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.05])
            assert jnp.allclose(twists[i], expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
