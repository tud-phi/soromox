"""Tests for rotation utility functions.

These tests verify that the quaternion-based rotation conversions correctly
handle all rotation angles, including edge cases:
- Small rotations (θ ≈ 0)
- Near-180-degree rotations (θ ≈ π), which caused issues in previous implementations

The near-180-degree case is particularly important because sin(θ) ≈ 0 for both
θ ≈ 0 and θ ≈ π, but the correct behavior is different in each case.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.utils.rotations import (
    quaternion_to_rotation_matrix,
    quaternion_to_rotation_vector,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_quaternion,
    rotation_vector_to_rotation_matrix,
)


def rotation_matrix_from_axis_angle(axis, angle):
    """Create rotation matrix using Rodrigues formula.

    Args:
        axis: Rotation axis (will be normalized).
        angle: Rotation angle in radians.

    Returns:
        R: 3x3 rotation matrix.
    """
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return jnp.array(R)


class TestRotationMatrixToQuaternion:
    """Tests for rotation_matrix_to_quaternion."""

    def test_identity_rotation(self):
        """Test that identity matrix gives identity quaternion."""
        R = jnp.eye(3)
        q = rotation_matrix_to_quaternion(R)

        # Identity quaternion is [0, 0, 0, 1]
        expected = jnp.array([0.0, 0.0, 0.0, 1.0])
        assert_allclose(q, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "axis",
        [
            [1, 0, 0],  # X-axis
            [0, 1, 0],  # Y-axis
            [0, 0, 1],  # Z-axis
        ],
    )
    def test_90_degree_rotation(self, axis):
        """Test 90-degree rotations around principal axes."""
        angle = np.pi / 2
        R = rotation_matrix_from_axis_angle(axis, angle)
        q = rotation_matrix_to_quaternion(R)

        # Unit quaternion should have norm 1
        assert_allclose(jnp.linalg.norm(q), 1.0, atol=1e-10)

        # w component should be cos(45°) = sqrt(2)/2
        assert_allclose(q[3], np.cos(angle / 2), atol=1e-10)

        # Vector part norm should be sin(45°) = sqrt(2)/2
        assert_allclose(jnp.linalg.norm(q[:3]), np.sin(angle / 2), atol=1e-10)

    def test_quaternion_is_normalized(self):
        """Test that output quaternion is always normalized."""
        # Random rotation matrix
        R = rotation_matrix_from_axis_angle([1, 2, 3], 1.5)
        q = rotation_matrix_to_quaternion(R)

        assert_allclose(jnp.linalg.norm(q), 1.0, atol=1e-10)

    def test_positive_w_convention(self):
        """Test that w component is non-negative (canonical form)."""
        for angle in [0.5, 1.5, 2.5, 3.0]:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]:
                R = rotation_matrix_from_axis_angle(axis, angle)
                q = rotation_matrix_to_quaternion(R)
                assert q[3] >= 0, f"w should be non-negative, got {q[3]}"

    def test_custom_eps(self):
        """Test that custom eps parameter works."""
        R = rotation_matrix_from_axis_angle([1, 0, 0], 0.5)
        q1 = rotation_matrix_to_quaternion(R, eps=1e-10)
        q2 = rotation_matrix_to_quaternion(R, eps=1e-6)
        # Results should be very close
        assert_allclose(q1, q2, atol=1e-8)


class TestQuaternionToRotationMatrix:
    """Tests for quaternion_to_rotation_matrix."""

    def test_identity_quaternion(self):
        """Test that identity quaternion gives identity rotation matrix."""
        q = jnp.array([0.0, 0.0, 0.0, 1.0])
        R = quaternion_to_rotation_matrix(q)

        assert_allclose(R, jnp.eye(3), atol=1e-10)

    @pytest.mark.parametrize(
        "axis",
        [
            [1, 0, 0],  # X-axis
            [0, 1, 0],  # Y-axis
            [0, 0, 1],  # Z-axis
        ],
    )
    def test_90_degree_rotation(self, axis):
        """Test 90-degree rotations around principal axes."""
        angle = np.pi / 2
        axis_normalized = np.array(axis) / np.linalg.norm(axis)

        # Create quaternion for 90-degree rotation
        q = jnp.array(
            [
                axis_normalized[0] * np.sin(angle / 2),
                axis_normalized[1] * np.sin(angle / 2),
                axis_normalized[2] * np.sin(angle / 2),
                np.cos(angle / 2),
            ]
        )

        R = quaternion_to_rotation_matrix(q)
        R_expected = rotation_matrix_from_axis_angle(axis, angle)

        assert_allclose(R, R_expected, atol=1e-10)

    def test_rotation_matrix_is_orthonormal(self):
        """Test that output rotation matrix is orthonormal."""
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q = q / jnp.linalg.norm(q)  # Normalize

        R = quaternion_to_rotation_matrix(q)

        # R @ R.T should be identity
        assert_allclose(R @ R.T, jnp.eye(3), atol=1e-10)
        # det(R) should be 1
        assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-10)

    def test_roundtrip_with_rotation_matrix_to_quaternion(self):
        """Test that R -> q -> R is consistent."""
        for _ in range(10):
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(0.1, 3.0)

            R_original = rotation_matrix_from_axis_angle(axis, angle)
            q = rotation_matrix_to_quaternion(R_original)
            R_recovered = quaternion_to_rotation_matrix(q)

            assert_allclose(R_recovered, R_original, atol=1e-10)


class TestQuaternionToRotationVector:
    """Tests for quaternion_to_rotation_vector."""

    def test_identity_quaternion(self):
        """Test that identity quaternion gives zero rotation vector."""
        q = jnp.array([0.0, 0.0, 0.0, 1.0])
        omega = quaternion_to_rotation_vector(q)

        assert_allclose(omega, jnp.zeros(3), atol=1e-10)

    @pytest.mark.parametrize(
        "axis,angle",
        [
            ([1, 0, 0], np.pi / 4),  # X-axis, 45°
            ([0, 1, 0], np.pi / 2),  # Y-axis, 90°
            ([0, 0, 1], np.pi),  # Z-axis, 180°
            ([1, 1, 1], 1.5),  # Arbitrary axis
        ],
    )
    def test_rotation_vector_magnitude(self, axis, angle):
        """Test that rotation vector magnitude equals rotation angle."""
        axis = np.array(axis) / np.linalg.norm(axis)
        q = jnp.array(
            [
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
                np.cos(angle / 2),
            ]
        )
        omega = quaternion_to_rotation_vector(q)

        assert_allclose(jnp.linalg.norm(omega), angle, atol=1e-6)

    def test_custom_eps(self):
        """Test that custom eps parameter works."""
        q = jnp.array([0.0, 0.0, 0.1, 0.995])
        q = q / jnp.linalg.norm(q)
        omega1 = quaternion_to_rotation_vector(q, eps=1e-10)
        omega2 = quaternion_to_rotation_vector(q, eps=1e-6)
        assert_allclose(omega1, omega2, atol=1e-8)


class TestRotationVectorToQuaternion:
    """Tests for rotation_vector_to_quaternion."""

    def test_zero_rotation(self):
        """Test that zero rotation vector gives identity quaternion."""
        omega = jnp.zeros(3)
        q = rotation_vector_to_quaternion(omega)

        expected = jnp.array([0.0, 0.0, 0.0, 1.0])
        assert_allclose(q, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "axis,angle",
        [
            ([1, 0, 0], np.pi / 4),  # X-axis, 45°
            ([0, 1, 0], np.pi / 2),  # Y-axis, 90°
            ([0, 0, 1], np.pi),  # Z-axis, 180°
            ([1, 1, 1], 1.5),  # Arbitrary axis
        ],
    )
    def test_rotation_vector_to_quaternion(self, axis, angle):
        """Test rotation vector to quaternion conversion."""
        axis = np.array(axis) / np.linalg.norm(axis)
        omega = jnp.array(axis * angle)

        q = rotation_vector_to_quaternion(omega)

        # Expected quaternion
        q_expected = jnp.array(
            [
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
                np.cos(angle / 2),
            ]
        )

        # Quaternions may differ by sign (q and -q represent same rotation)
        # Compare with canonical form (positive w)
        if q_expected[3] < 0:
            q_expected = -q_expected

        assert_allclose(q, q_expected, atol=1e-6)

    def test_quaternion_is_normalized(self):
        """Test that output quaternion is always normalized."""
        omega = jnp.array([0.5, 1.0, 0.3])
        q = rotation_vector_to_quaternion(omega)

        assert_allclose(jnp.linalg.norm(q), 1.0, atol=1e-10)

    def test_roundtrip_with_quaternion_to_rotation_vector(self):
        """Test that omega -> q -> omega is consistent."""
        for _ in range(10):
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(0.01, 3.1)
            omega_original = jnp.array(axis * angle)

            q = rotation_vector_to_quaternion(omega_original)
            omega_recovered = quaternion_to_rotation_vector(q)

            assert_allclose(omega_recovered, omega_original, atol=1e-6)

    def test_small_angles(self):
        """Test small angle rotations."""
        for angle in [1e-8, 1e-6, 1e-4, 1e-2]:
            omega = jnp.array([0.0, 0.0, angle])
            q = rotation_vector_to_quaternion(omega)

            # Should be close to identity with small z component
            assert_allclose(q[3], 1.0, atol=1e-4)
            assert_allclose(jnp.linalg.norm(q[:3]), angle / 2, atol=angle)

    def test_custom_eps(self):
        """Test that custom eps parameter works."""
        omega = jnp.array([0.0, 0.0, 0.5])
        q1 = rotation_vector_to_quaternion(omega, eps=1e-10)
        q2 = rotation_vector_to_quaternion(omega, eps=1e-6)
        assert_allclose(q1, q2, atol=1e-8)


class TestRotationMatrixToRotationVector:
    """Tests for the combined rotation_matrix_to_rotation_vector function."""

    @pytest.mark.parametrize(
        "axis,angle_deg",
        [
            ([0, 0, 1], 0.001),  # Z-axis, tiny angle
            ([0, 0, 1], 1.0),  # Z-axis, small angle
            ([1, 0, 0], 45.0),  # X-axis, medium angle
            ([0, 1, 0], 90.0),  # Y-axis, 90 degrees
            ([0, 0, 1], 135.0),  # Z-axis, large angle
        ],
    )
    def test_general_rotations(self, axis, angle_deg):
        """Test rotation vector extraction for general rotation angles."""
        angle = np.radians(angle_deg)
        R = rotation_matrix_from_axis_angle(axis, angle)

        omega = rotation_matrix_to_rotation_vector(R)

        axis_normalized = np.array(axis) / np.linalg.norm(axis)
        omega_expected = jnp.array(axis_normalized * angle)

        assert_allclose(omega, omega_expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "axis,angle_deg",
        [
            ([0, 0, 1], 179.0),  # Z-axis, near 180
            ([0, 0, 1], 179.9),  # Z-axis, very near 180
            ([0, 0, 1], 179.99),  # Z-axis, extremely near 180
            ([1, 0, 0], 179.99),  # X-axis, near 180
            ([0, 1, 0], 179.99),  # Y-axis, near 180
            ([1, 1, 0], 179.99),  # XY diagonal, near 180
            ([1, 1, 1], 179.99),  # XYZ diagonal, near 180
            ([0.3, 0.7, 0.2], 179.99),  # Arbitrary axis, near 180
        ],
    )
    def test_near_180_degree_rotations(self, axis, angle_deg):
        """Test rotation vector extraction for near-180-degree rotations.

        This tests the fix for the singularity that occurred when
        sin(θ) ≈ 0 but θ ≈ π (not θ ≈ 0). The previous implementation
        incorrectly used scale=1.0 for both cases, leading to incorrect
        rotation vectors for near-180-degree rotations.
        """
        angle = np.radians(angle_deg)
        R = rotation_matrix_from_axis_angle(axis, angle)

        omega = rotation_matrix_to_rotation_vector(R)

        axis_normalized = np.array(axis) / np.linalg.norm(axis)
        omega_expected = jnp.array(axis_normalized * angle)

        # For near-180-degree rotations, the rotation vector magnitude
        # should be approximately π (≈3.14), not nearly zero
        assert jnp.linalg.norm(omega) > 3.0, (
            f"Rotation vector magnitude should be ~π for {angle_deg}° rotation, "
            f"got {float(jnp.linalg.norm(omega)):.4f}"
        )
        assert_allclose(omega, omega_expected, rtol=1e-5, atol=1e-5)

    def test_identity_rotation(self):
        """Test that identity rotation matrix gives zero rotation vector."""
        R = jnp.eye(3)
        omega = rotation_matrix_to_rotation_vector(R)

        assert_allclose(omega, jnp.zeros(3), atol=1e-10)

    def test_roundtrip_consistency(self):
        """Test that rotation matrix -> rotation vector is consistent."""
        # Test multiple random rotations
        for _ in range(10):
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(0.01, 3.1)  # Avoid exactly 0 and π

            R = rotation_matrix_from_axis_angle(axis, angle)
            omega = rotation_matrix_to_rotation_vector(R)

            # Check magnitude
            assert_allclose(jnp.linalg.norm(omega), angle, rtol=1e-5)

            # Check direction (for non-zero angles)
            if angle > 0.01:
                omega_direction = omega / jnp.linalg.norm(omega)
                assert_allclose(np.abs(omega_direction), np.abs(axis), atol=1e-5)

    def test_custom_eps(self):
        """Test that custom eps parameter works."""
        R = rotation_matrix_from_axis_angle([1, 0, 0], 0.5)
        omega1 = rotation_matrix_to_rotation_vector(R, eps=1e-10)
        omega2 = rotation_matrix_to_rotation_vector(R, eps=1e-6)
        assert_allclose(omega1, omega2, atol=1e-8)


class TestRotationVectorToRotationMatrix:
    """Tests for rotation_vector_to_rotation_matrix."""

    def test_zero_rotation(self):
        """Test that zero rotation vector gives identity matrix."""
        omega = jnp.zeros(3)
        R = rotation_vector_to_rotation_matrix(omega)

        assert_allclose(R, jnp.eye(3), atol=1e-10)

    @pytest.mark.parametrize(
        "axis,angle_deg",
        [
            ([1, 0, 0], 45.0),  # X-axis
            ([0, 1, 0], 90.0),  # Y-axis
            ([0, 0, 1], 135.0),  # Z-axis
            ([1, 1, 1], 60.0),  # Arbitrary axis
        ],
    )
    def test_general_rotations(self, axis, angle_deg):
        """Test rotation vector to rotation matrix conversion."""
        angle = np.radians(angle_deg)
        axis_normalized = np.array(axis) / np.linalg.norm(axis)
        omega = jnp.array(axis_normalized * angle)

        R = rotation_vector_to_rotation_matrix(omega)
        R_expected = rotation_matrix_from_axis_angle(axis, angle)

        assert_allclose(R, R_expected, atol=1e-10)

    def test_rotation_matrix_is_orthonormal(self):
        """Test that output rotation matrix is orthonormal."""
        omega = jnp.array([0.5, 1.0, 0.3])
        R = rotation_vector_to_rotation_matrix(omega)

        # R @ R.T should be identity
        assert_allclose(R @ R.T, jnp.eye(3), atol=1e-10)
        # det(R) should be 1
        assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-10)

    def test_roundtrip_with_rotation_matrix_to_rotation_vector(self):
        """Test that omega -> R -> omega is consistent."""
        for _ in range(10):
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(0.01, 3.1)
            omega_original = jnp.array(axis * angle)

            R = rotation_vector_to_rotation_matrix(omega_original)
            omega_recovered = rotation_matrix_to_rotation_vector(R)

            assert_allclose(omega_recovered, omega_original, atol=1e-6)

    def test_small_angles(self):
        """Test small angle rotations."""
        for angle in [1e-8, 1e-6, 1e-4, 1e-2]:
            omega = jnp.array([0.0, 0.0, angle])
            R = rotation_vector_to_rotation_matrix(omega)

            # For very small angles, R should be close to identity
            assert_allclose(R, jnp.eye(3), atol=10 * angle)

    def test_custom_eps(self):
        """Test that custom eps parameter works."""
        omega = jnp.array([0.0, 0.0, 0.5])
        R1 = rotation_vector_to_rotation_matrix(omega, eps=1e-10)
        R2 = rotation_vector_to_rotation_matrix(omega, eps=1e-6)
        assert_allclose(R1, R2, atol=1e-8)


class TestJAXCompatibility:
    """Tests for JAX compatibility (jit, vmap, grad)."""

    def test_jit_compatibility(self):
        """Test that functions work with jit."""
        R = rotation_matrix_from_axis_angle([1, 2, 3], 1.0)

        # These should compile and run without errors
        jit_r2q = jax.jit(rotation_matrix_to_quaternion)
        jit_q2r = jax.jit(quaternion_to_rotation_matrix)
        jit_q2rv = jax.jit(quaternion_to_rotation_vector)
        jit_rv2q = jax.jit(rotation_vector_to_quaternion)
        jit_r2rv = jax.jit(rotation_matrix_to_rotation_vector)
        jit_rv2r = jax.jit(rotation_vector_to_rotation_matrix)

        q = jit_r2q(R)
        R_recovered = jit_q2r(q)
        omega1 = jit_q2rv(q)
        q_recovered = jit_rv2q(omega1)
        omega2 = jit_r2rv(R)
        R_from_rv = jit_rv2r(omega2)

        assert q.shape == (4,)
        assert R_recovered.shape == (3, 3)
        assert omega1.shape == (3,)
        assert q_recovered.shape == (4,)
        assert omega2.shape == (3,)
        assert R_from_rv.shape == (3, 3)
        assert_allclose(omega1, omega2, atol=1e-10)

    def test_vmap_compatibility(self):
        """Test that functions work with vmap."""
        # Create batch of rotation matrices
        batch_size = 5
        Rs = jnp.stack(
            [
                rotation_matrix_from_axis_angle([1, 0, 0], 0.5 * i)
                for i in range(batch_size)
            ]
        )

        # Vectorized conversions
        vmapped_r2rv = jax.vmap(rotation_matrix_to_rotation_vector)
        vmapped_rv2r = jax.vmap(rotation_vector_to_rotation_matrix)
        vmapped_r2q = jax.vmap(rotation_matrix_to_quaternion)
        vmapped_q2r = jax.vmap(quaternion_to_rotation_matrix)

        omegas = vmapped_r2rv(Rs)
        Rs_recovered = vmapped_rv2r(omegas)
        qs = vmapped_r2q(Rs)
        Rs_from_q = vmapped_q2r(qs)

        assert omegas.shape == (batch_size, 3)
        assert Rs_recovered.shape == (batch_size, 3, 3)
        assert qs.shape == (batch_size, 4)
        assert Rs_from_q.shape == (batch_size, 3, 3)

    def test_grad_compatibility(self):
        """Test that quaternion_to_rotation_vector is differentiable.

        Note: rotation_matrix_to_quaternion uses jnp.where branches that
        can produce NaN gradients at certain configurations. However,
        quaternion_to_rotation_vector should be fully differentiable.
        """

        def loss_fn(q_vec_angle):
            # Create a quaternion from angle (rotation around z-axis)
            half_angle = q_vec_angle / 2
            quat = jnp.array([0.0, 0.0, jnp.sin(half_angle), jnp.cos(half_angle)])
            omega = quaternion_to_rotation_vector(quat)
            return jnp.sum(omega**2)

        angle = 0.5
        grad_fn = jax.grad(loss_fn)

        # Should compute gradient without errors
        grad_angle = grad_fn(angle)
        assert jnp.isfinite(grad_angle)
        # Gradient of ||omega||^2 = angle^2 w.r.t. angle is 2*angle
        assert_allclose(grad_angle, 2 * angle, rtol=1e-5)

    def test_grad_rotation_vector_to_quaternion(self):
        """Test that rotation_vector_to_quaternion is differentiable."""

        def loss_fn(omega):
            q = rotation_vector_to_quaternion(omega)
            return jnp.sum(q**2)

        omega = jnp.array([0.1, 0.2, 0.3])
        grad_fn = jax.grad(loss_fn)

        grad_omega = grad_fn(omega)
        assert grad_omega.shape == (3,)
        assert jnp.all(jnp.isfinite(grad_omega))

    def test_grad_quaternion_to_rotation_matrix(self):
        """Test that quaternion_to_rotation_matrix is differentiable."""

        def loss_fn(q):
            q = q / jnp.linalg.norm(q)  # Normalize
            R = quaternion_to_rotation_matrix(q)
            return jnp.sum(R**2)

        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        grad_fn = jax.grad(loss_fn)

        grad_q = grad_fn(q)
        assert grad_q.shape == (4,)
        assert jnp.all(jnp.isfinite(grad_q))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
