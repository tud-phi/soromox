"""Tests for rotation utility functions.

These tests verify that the quaternion-based rotation conversions correctly
handle all rotation angles, including edge cases:
- Small rotations (θ ≈ 0)
- Near-180-degree rotations (θ ≈ π), which caused issues in previous implementations

The near-180-degree case is particularly important because sin(θ) ≈ 0 for both
θ ≈ 0 and θ ≈ π, but the correct behavior is different in each case.
"""

# ruff: noqa: E402

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from soromox.utils.geometry.errors import (
    angle_geodesic_error,
    quaternion_geodesic_error,
    rotation_matrix_geodesic_error,
    wrap_angle,
)
from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    normalize_quaternion,
    principal_axis_rotation_matrix,
    principal_axis_rotation_matrix_derivative,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    quaternion_to_rotation_vector,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_to_6d,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_quaternion,
    rotation_vector_to_rotation_matrix,
)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("angle", [-0.7, 0.0, 0.45])
def test_principal_axis_rotation_derivative_matches_autodiff(axis, angle):
    angle = jnp.asarray(angle)
    expected = jax.jacrev(lambda value: principal_axis_rotation_matrix(value, axis))(
        angle
    )
    assert_allclose(
        principal_axis_rotation_matrix_derivative(angle, axis),
        expected,
        rtol=1e-12,
        atol=1e-12,
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


def assert_forward_and_reverse_mode_finite(fn, arg):
    """Assert that a vector-valued JAX function has finite forward/reverse Jacobians."""
    jac_fwd = jax.jacfwd(fn)(arg)
    jac_rev = jax.jacrev(fn)(arg)

    assert jnp.isfinite(jac_fwd).all()
    assert jnp.isfinite(jac_rev).all()


def nested_grad_of_vmap(fn, singular, regular, tangent):
    """Differentiate a vmapped JVP at a value that takes a branch."""
    values = jnp.stack([singular, regular])
    return jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(
                lambda value: jnp.sum(jax.jvp(fn, (value,), (tangent,))[1])
            )(batch)
        )
    )(values)


def test_normalize_quaternion_uses_configurable_eps():
    q = jnp.array([1e-8, 0.0, 0.0, 1e-8])

    assert_allclose(
        normalize_quaternion(q, eps=1e-6),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-12,
    )
    assert_allclose(
        normalize_quaternion(q, eps=1e-10),
        jnp.array([jnp.sqrt(0.5), 0.0, 0.0, jnp.sqrt(0.5)]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "fn,singular,regular,tangent",
    [
        (
            lambda q: normalize_quaternion(q, eps=1e-10),
            jnp.zeros(4),
            jnp.array([0.1, 0.2, 0.3, 0.9]),
            jnp.ones(4),
        ),
        (
            lambda q: quaternion_to_rotation_vector(q, eps=1e-10),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            jnp.array([0.0, 0.0, jnp.sin(0.1), jnp.cos(0.1)]),
            jnp.ones(4),
        ),
        (
            lambda omega: rotation_vector_to_quaternion(omega, eps=1e-10),
            jnp.zeros(3),
            jnp.array([0.1, -0.2, 0.3]),
            jnp.ones(3),
        ),
        (
            quaternion_to_rotation_matrix,
            jnp.zeros(4),
            jnp.array([0.1, 0.2, 0.3, 0.9]),
            jnp.ones(4),
        ),
    ],
    ids=[
        "normalize_quaternion",
        "quaternion_to_rotation_vector",
        "rotation_vector_to_quaternion",
        "quaternion_to_rotation_matrix",
    ],
)
def test_nested_grad_of_vmap_is_finite_at_quaternion_branch_points(
    fn, singular, regular, tangent
):
    gradient = nested_grad_of_vmap(fn, singular, regular, tangent)

    assert jnp.isfinite(gradient).all()


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

        # scalar component should be cos(45°) = sqrt(2)/2
        assert_allclose(q[3], np.cos(angle / 2), atol=1e-10)

        # Vector part norm should be sin(45°) = sqrt(2)/2
        assert_allclose(jnp.linalg.norm(q[:3]), np.sin(angle / 2), atol=1e-10)

    def test_quaternion_is_normalized(self):
        """Test that output quaternion is always normalized."""
        # Random rotation matrix
        R = rotation_matrix_from_axis_angle([1, 2, 3], 1.5)
        q = rotation_matrix_to_quaternion(R)

        assert_allclose(jnp.linalg.norm(q), 1.0, atol=1e-10)

    def test_matches_scipy_scalar_last_order(self):
        """Match SciPy's default XYZW representation without reordering."""
        rotation_vector = np.array([0.3, -0.4, 0.2])
        scipy_rotation = Rotation.from_rotvec(rotation_vector)

        q = rotation_matrix_to_quaternion(jnp.asarray(scipy_rotation.as_matrix()))

        assert_allclose(q, scipy_rotation.as_quat(), atol=1e-10)

    def test_positive_w_convention(self):
        """Test that scalar component is non-negative (canonical form)."""
        for angle in [0.5, 1.5, 2.5, 3.0]:
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]:
                R = rotation_matrix_from_axis_angle(axis, angle)
                q = rotation_matrix_to_quaternion(R)
                assert q[3] >= 0, f"scalar component should be non-negative, got {q[3]}"

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
        q = jnp.array([0.0, 0.1, 0.995, 0.0])
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
        # Compare with canonical form (positive scalar component)
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
        """Test that quaternion_to_rotation_vector is differentiable."""

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

    @pytest.mark.parametrize(
        "omega",
        [
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([1e-9, -2e-9, 3e-9]),
            jnp.array([0.2, -0.1, 0.3]),
            jnp.array([0.0, 0.0, jnp.pi - 1e-7]),
        ],
    )
    def test_rotation_matrix_to_quaternion_forward_and_reverse_mode_finite(self, omega):
        """Test finite autodiff through trace, regular, and near-pi branches."""
        R = rotation_vector_to_rotation_matrix(omega)

        assert_forward_and_reverse_mode_finite(
            lambda R_flat: rotation_matrix_to_quaternion(R_flat.reshape((3, 3))),
            R.reshape(-1),
        )

    @pytest.mark.parametrize(
        "omega",
        [
            jnp.array([jnp.pi, 0.0, 0.0]),
            jnp.array([0.0, jnp.pi, 0.0]),
            jnp.array([0.0, 0.0, jnp.pi]),
        ],
    )
    def test_rotation_matrix_to_quaternion_exact_pi_value_only(self, omega):
        """Check exact pi cases, where branch ties are not uniquely differentiable."""
        R = rotation_vector_to_rotation_matrix(omega)

        quat = rotation_matrix_to_quaternion(R)

        assert jnp.isfinite(quat).all()
        assert_allclose(quaternion_to_rotation_matrix(quat), R, rtol=1e-6, atol=1e-8)

    def test_zero_quaternion_branch_is_forward_and_reverse_mode_finite(self):
        """Test finite autodiff through the zero-quaternion identity fallback."""
        zero_quat = jnp.zeros(4)

        assert_forward_and_reverse_mode_finite(
            lambda q: normalize_quaternion(q, eps=1e-6),
            zero_quat,
        )
        assert_forward_and_reverse_mode_finite(
            lambda q: quaternion_to_rotation_matrix(q).reshape(-1),
            zero_quat,
        )

    def test_quaternion_identity_log_branch_is_forward_and_reverse_mode_finite(self):
        """Test finite autodiff through the zero-vector branch of quaternion log."""
        identity_quat = jnp.array([0.0, 0.0, 0.0, 1.0])

        assert_forward_and_reverse_mode_finite(
            lambda q: quaternion_to_rotation_vector(q, eps=1e-10),
            identity_quat,
        )

    def test_rotation_vector_zero_branch_is_forward_and_reverse_mode_finite(self):
        """Test finite autodiff through the zero-angle quaternion exponential."""
        zero_omega = jnp.zeros(3)

        assert_forward_and_reverse_mode_finite(
            lambda omega: rotation_vector_to_quaternion(omega, eps=1e-10),
            zero_omega,
        )


class TestRotationRepresentationEnum:
    """Tests for the RotationRepresentation enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert RotationRepresentation.ROTATION_VECTOR.value == "rotation_vector"
        assert RotationRepresentation.QUATERNION.value == "quaternion"
        assert RotationRepresentation.ROTATION_MATRIX_6D.value == "rotation_matrix_6d"

    def test_enum_members(self):
        """Test that we have exactly three representations."""
        members = list(RotationRepresentation)
        assert len(members) == 3


class TestRotationMatrix6D:
    """Tests for 6D continuous rotation representation (Zhou et al. 2019)."""

    def test_identity_rotation(self):
        """Test that identity rotation matrix gives expected 6D representation."""
        R = jnp.eye(3)
        r6d = rotation_matrix_to_6d(R)

        # First two columns of identity: [1,0,0] and [0,1,0]
        expected = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert_allclose(r6d, expected, atol=1e-10)

    def test_roundtrip_identity(self):
        """Test identity rotation roundtrip."""
        R = jnp.eye(3)
        r6d = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_rotation_matrix(r6d)
        assert_allclose(R_recovered, R, atol=1e-10)

    @pytest.mark.parametrize(
        "axis,angle_deg",
        [
            ([1, 0, 0], 45.0),
            ([0, 1, 0], 90.0),
            ([0, 0, 1], 135.0),
            ([1, 1, 1], 60.0),
            ([0.3, 0.7, 0.2], 170.0),
        ],
    )
    def test_roundtrip_general_rotations(self, axis, angle_deg):
        """Test roundtrip for general rotations."""
        angle = np.radians(angle_deg)
        R = rotation_matrix_from_axis_angle(axis, angle)

        r6d = rotation_matrix_to_6d(R)
        R_recovered = rotation_6d_to_rotation_matrix(r6d)

        assert_allclose(R_recovered, R, atol=1e-10)

    def test_6d_representation_shape(self):
        """Test that 6D representation has correct shape."""
        R = rotation_matrix_from_axis_angle([1, 2, 3], 1.0)
        r6d = rotation_matrix_to_6d(R)
        assert r6d.shape == (6,)

    def test_orthonormalization(self):
        """Test that Gram-Schmidt orthonormalization works for non-orthonormal input."""
        # Create a slightly perturbed 6D representation
        r6d = jnp.array([1.0, 0.1, 0.0, 0.1, 1.0, 0.0])  # Not perfectly orthonormal
        R = rotation_6d_to_rotation_matrix(r6d)

        # Output should be a valid rotation matrix
        assert_allclose(R @ R.T, jnp.eye(3), atol=1e-10)
        assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-10)

    def test_near_collinear_input_is_forward_and_reverse_mode_finite(self):
        """Test finite autodiff for ill-conditioned but non-degenerate 6D inputs."""
        r6d = jnp.array([1.0, 0.0, 0.0, 1.0, 1e-5, 0.0])

        R = rotation_6d_to_rotation_matrix(r6d)

        assert jnp.isfinite(R).all()
        assert_forward_and_reverse_mode_finite(
            lambda x: rotation_6d_to_rotation_matrix(x).reshape(-1),
            r6d,
        )

    def test_degenerate_input_has_no_unique_rotation_matrix(self):
        """Document that exact zero 6D inputs are outside the valid domain."""
        R = rotation_6d_to_rotation_matrix(jnp.zeros(6))

        assert jnp.isnan(R).any()

    def test_vmap_compatibility(self):
        """Test that 6D conversion works with vmap."""
        batch_size = 5
        Rs = jnp.stack(
            [
                rotation_matrix_from_axis_angle([1, 0, 0], 0.5 * i)
                for i in range(batch_size)
            ]
        )

        vmapped_to_6d = jax.vmap(rotation_matrix_to_6d)
        vmapped_from_6d = jax.vmap(rotation_6d_to_rotation_matrix)

        r6ds = vmapped_to_6d(Rs)
        Rs_recovered = vmapped_from_6d(r6ds)

        assert r6ds.shape == (batch_size, 6)
        assert Rs_recovered.shape == (batch_size, 3, 3)
        assert_allclose(Rs_recovered, Rs, atol=1e-10)


class TestQuaternionOperations:
    """Tests for quaternion multiplication and conjugate."""

    def test_quaternion_identity_multiply(self):
        """Test that multiplying by identity gives the same quaternion."""
        q_identity = jnp.array([0.0, 0.0, 0.0, 1.0])
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q = q / jnp.linalg.norm(q)

        result = quaternion_multiply(q_identity, q)
        assert_allclose(result, q, atol=1e-10)

        result2 = quaternion_multiply(q, q_identity)
        assert_allclose(result2, q, atol=1e-10)

    def test_quaternion_multiply_inverse(self):
        """Test that q * q^{-1} = identity."""
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q = q / jnp.linalg.norm(q)

        q_inv = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_inv)

        # Should be identity quaternion
        identity = jnp.array([0.0, 0.0, 0.0, 1.0])
        assert_allclose(result, identity, atol=1e-10)

    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q_conj = quaternion_conjugate(q)

        expected = jnp.array([-0.1, -0.2, -0.3, 0.9])
        assert_allclose(q_conj, expected, atol=1e-10)

    def test_quaternion_multiply_composition(self):
        """Test that quaternion multiplication correctly composes rotations."""
        # Two 90-degree rotations around z-axis should give 180-degree rotation
        angle = np.pi / 2
        q_90z = jnp.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])

        q_180z = quaternion_multiply(q_90z, q_90z)

        # Should represent 180-degree rotation around z
        # q = [0, 0, sin(90°), cos(90°)] = [0, 0, 1, 0]
        expected = jnp.array([0.0, 0.0, 1.0, 0.0])
        assert_allclose(jnp.abs(q_180z), jnp.abs(expected), atol=1e-10)


class TestWrapAngle:
    """Tests for angle wrapping utility."""

    def test_wrap_zero(self):
        """Test that zero remains zero."""
        assert_allclose(wrap_angle(jnp.array(0.0)), 0.0, atol=1e-10)

    def test_wrap_within_range(self):
        """Test that angles within [-π, π] are unchanged."""
        angles = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        wrapped = jax.vmap(wrap_angle)(angles)
        assert_allclose(wrapped, angles, atol=1e-10)

    def test_wrap_positive_overflow(self):
        """Test wrapping angles > π."""
        assert_allclose(wrap_angle(jnp.array(np.pi + 0.1)), -np.pi + 0.1, atol=1e-10)
        assert_allclose(wrap_angle(jnp.array(2 * np.pi)), 0.0, atol=1e-10)
        assert_allclose(wrap_angle(jnp.array(3 * np.pi)), -np.pi, atol=1e-10)

    def test_wrap_negative_overflow(self):
        """Test wrapping angles < -π."""
        assert_allclose(wrap_angle(jnp.array(-np.pi - 0.1)), np.pi - 0.1, atol=1e-10)
        assert_allclose(wrap_angle(jnp.array(-2 * np.pi)), 0.0, atol=1e-10)
        # -3π wraps to -π (which is equivalent to π, but wrap_angle returns [-π, π))
        assert_allclose(wrap_angle(jnp.array(-3 * np.pi)), -np.pi, atol=1e-10)

    def test_wrap_vectorized(self):
        """Test that wrapping works with arrays."""
        angles = jnp.array([0.0, 2 * np.pi, -2 * np.pi, 5 * np.pi, -5 * np.pi])
        wrapped = jax.vmap(wrap_angle)(angles)
        # 5π wraps to -π (same as π but in [-π, π) convention)
        expected = jnp.array([0.0, 0.0, 0.0, -np.pi, -np.pi])
        assert_allclose(wrapped, expected, atol=1e-10)


class TestAngleError:
    """Tests for planar angle error computation."""

    def test_zero_error(self):
        """Test that same angle gives zero error."""
        theta = jnp.array(1.0)
        error = angle_geodesic_error(theta, theta)
        assert_allclose(error, 0.0, atol=1e-10)

    def test_small_positive_error(self):
        """Test small positive error."""
        current = jnp.array(0.0)
        desired = jnp.array(0.5)
        error = angle_geodesic_error(current, desired)
        assert_allclose(error, 0.5, atol=1e-10)

    def test_small_negative_error(self):
        """Test small negative error."""
        current = jnp.array(0.5)
        desired = jnp.array(0.0)
        error = angle_geodesic_error(current, desired)
        assert_allclose(error, -0.5, atol=1e-10)

    def test_shortest_path_positive_wrap(self):
        """Test that error takes shortest path across 2π boundary."""
        # Current: 10°, Desired: 350° → Should be -20°, not +340°
        current = jnp.deg2rad(jnp.array(10.0))
        desired = jnp.deg2rad(jnp.array(350.0))
        error = angle_geodesic_error(current, desired)
        expected = jnp.deg2rad(jnp.array(-20.0))
        assert_allclose(error, expected, atol=1e-6)

    def test_shortest_path_negative_wrap(self):
        """Test that error takes shortest path the other way."""
        # Current: 350°, Desired: 10° → Should be +20°, not -340°
        current = jnp.deg2rad(jnp.array(350.0))
        desired = jnp.deg2rad(jnp.array(10.0))
        error = angle_geodesic_error(current, desired)
        expected = jnp.deg2rad(jnp.array(20.0))
        assert_allclose(error, expected, atol=1e-6)

    def test_half_rotation(self):
        """Test 180-degree error."""
        current = jnp.array(0.0)
        desired = jnp.array(np.pi)
        error = angle_geodesic_error(current, desired)
        assert_allclose(jnp.abs(error), np.pi, atol=1e-10)

    def test_vectorized(self):
        """Test vectorized angle error computation."""
        currents = jnp.array([0.0, np.pi / 2, np.pi])
        desireds = jnp.array([0.1, np.pi / 2, -np.pi])
        errors = jax.vmap(angle_geodesic_error)(currents, desireds)
        expected = jnp.array([0.1, 0.0, 0.0])  # π and -π are the same
        assert_allclose(errors, expected, atol=1e-10)


class TestRotationMatrixError:
    """Tests for geometric 3D rotation error computation."""

    def test_zero_error(self):
        """Test that same rotation gives zero error."""
        R = rotation_matrix_from_axis_angle([1, 2, 3], 1.0)
        error = rotation_matrix_geodesic_error(R, R)
        assert_allclose(error, jnp.zeros(3), atol=1e-10)

    def test_small_rotation_error(self):
        """Test small rotation error."""
        R_current = jnp.eye(3)
        omega_error = jnp.array([0.0, 0.0, 0.1])
        R_desired = rotation_vector_to_rotation_matrix(omega_error)

        error = rotation_matrix_geodesic_error(R_current, R_desired)
        assert_allclose(error, omega_error, atol=1e-6)

    def test_error_direction(self):
        """Test that error direction is correct."""
        # Current: 10° around z, Desired: 30° around z → Error: 20° around z
        R_current = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(10))
        R_desired = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(30))

        error = rotation_matrix_geodesic_error(R_current, R_desired)

        # Error should be approximately 20° around z-axis
        expected = jnp.array([0.0, 0.0, np.radians(20)])
        assert_allclose(error, expected, atol=1e-6)

    def test_shortest_path(self):
        """Test that error takes shortest path for large rotations."""
        # Current: 10° around z, Desired: 350° around z
        # Naive difference: 340°. Shortest path: -20°
        R_current = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(10))
        R_desired = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(350))

        error = rotation_matrix_geodesic_error(R_current, R_desired)

        # Error magnitude should be 20° (not 340°)
        error_magnitude = jnp.linalg.norm(error)
        assert error_magnitude < np.radians(30), (
            f"Expected ~20°, got {np.degrees(error_magnitude)}°"
        )
        assert_allclose(error_magnitude, np.radians(20), atol=1e-5)

        # Direction should be negative z (clockwise)
        assert error[2] < 0, "Error should be negative (clockwise) rotation"

    def test_near_180_degree_error(self):
        """Test error computation near 180 degrees."""
        R_current = jnp.eye(3)
        R_desired = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(179))

        error = rotation_matrix_geodesic_error(R_current, R_desired)
        error_magnitude = jnp.linalg.norm(error)

        assert_allclose(error_magnitude, np.radians(179), atol=1e-4)

    def test_exactly_180_degree_error(self):
        """Test error computation at exactly 180 degrees."""
        R_current = jnp.eye(3)
        R_desired = rotation_matrix_from_axis_angle([0, 0, 1], np.pi)

        error = rotation_matrix_geodesic_error(R_current, R_desired)
        error_magnitude = jnp.linalg.norm(error)

        assert_allclose(error_magnitude, np.pi, atol=1e-5)

    def test_arbitrary_axis_error(self):
        """Test error for rotation around arbitrary axis."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        R_current = rotation_matrix_from_axis_angle(axis, 0.5)
        R_desired = rotation_matrix_from_axis_angle(axis, 0.8)

        error = rotation_matrix_geodesic_error(R_current, R_desired)

        # Error should be 0.3 radians around the same axis
        error_magnitude = jnp.linalg.norm(error)
        assert_allclose(error_magnitude, 0.3, atol=1e-5)

        # Direction should be along the axis
        error_direction = error / error_magnitude
        assert_allclose(jnp.abs(error_direction), jnp.abs(axis), atol=1e-5)

    @pytest.mark.parametrize(
        "omega_error",
        [
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([1e-9, -2e-9, 3e-9]),
            jnp.array([0.2, -0.1, 0.3]),
            jnp.array([0.0, 0.0, jnp.pi - 1e-7]),
        ],
    )
    def test_forward_and_reverse_mode_autodiff_is_finite(self, omega_error):
        """Test autodiff through identity, small, regular, and near-pi errors."""
        R_current = jnp.eye(3)
        R_desired = rotation_vector_to_rotation_matrix(omega_error)

        assert_forward_and_reverse_mode_finite(
            lambda R_flat: rotation_matrix_geodesic_error(
                R_current,
                R_flat.reshape((3, 3)),
            ),
            R_desired.reshape(-1),
        )

    def test_exact_pi_error_is_finite_value_only(self):
        """Check exact pi, where the log value is finite but not uniquely differentiable."""
        R_current = jnp.eye(3)
        R_desired = rotation_vector_to_rotation_matrix(jnp.array([jnp.pi, 0.0, 0.0]))

        error = rotation_matrix_geodesic_error(R_current, R_desired)

        assert jnp.isfinite(error).all()
        assert_allclose(jnp.linalg.norm(error), jnp.pi, atol=1e-6)


class TestRotationQuatError:
    """Tests for geometric rotation error computation using quaternions."""

    def test_zero_error(self):
        """Test that same quaternion gives zero error."""
        q = rotation_vector_to_quaternion(jnp.array([0.5, 0.3, 0.1]))
        error = quaternion_geodesic_error(q, q)
        assert_allclose(error, jnp.zeros(3), atol=1e-10)

    def test_small_rotation_error(self):
        """Test small rotation error."""
        q_current = jnp.array([0.0, 0.0, 0.0, 1.0])  # Identity
        omega_error = jnp.array([0.0, 0.0, 0.1])
        q_desired = rotation_vector_to_quaternion(omega_error)

        error = quaternion_geodesic_error(q_current, q_desired)
        assert_allclose(error, omega_error, atol=1e-6)

    def test_consistency_with_rotation_matrix_version(self):
        """Test that quaternion and rotation matrix versions give same result."""
        # Random rotations
        for _ in range(10):
            axis1 = np.random.randn(3)
            axis1 = axis1 / np.linalg.norm(axis1)
            angle1 = np.random.uniform(0.1, 2.5)

            axis2 = np.random.randn(3)
            axis2 = axis2 / np.linalg.norm(axis2)
            angle2 = np.random.uniform(0.1, 2.5)

            R_current = rotation_matrix_from_axis_angle(axis1, angle1)
            R_desired = rotation_matrix_from_axis_angle(axis2, angle2)

            q_current = rotation_matrix_to_quaternion(R_current)
            q_desired = rotation_matrix_to_quaternion(R_desired)

            error_R = rotation_matrix_geodesic_error(R_current, R_desired)
            error_q = quaternion_geodesic_error(q_current, q_desired)

            assert_allclose(error_R, error_q, atol=1e-5)

    def test_antipodal_handling(self):
        """Test that antipodal quaternions give zero error."""
        q = jnp.array([0.1, 0.2, 0.3, 0.9])
        q = q / jnp.linalg.norm(q)

        # Antipodal quaternion represents the same rotation
        q_antipodal = -q

        error = quaternion_geodesic_error(q, q_antipodal)
        assert_allclose(error, jnp.zeros(3), atol=1e-10)

    @pytest.mark.parametrize(
        "omega_error",
        [
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([1e-9, -2e-9, 3e-9]),
            jnp.array([0.2, -0.1, 0.3]),
            jnp.array([0.0, 0.0, jnp.pi - 1e-7]),
        ],
    )
    def test_forward_and_reverse_mode_autodiff_is_finite(self, omega_error):
        """Test autodiff through identity, small, regular, and near-pi errors."""
        q_current = jnp.array([0.0, 0.0, 0.0, 1.0])
        q_desired = rotation_vector_to_quaternion(omega_error)

        assert_forward_and_reverse_mode_finite(
            lambda q: quaternion_geodesic_error(q_current, q),
            q_desired,
        )

    def test_exact_pi_error_is_finite_value_only(self):
        """Check exact pi, where the axis sign is finite but not unique."""
        q_current = jnp.array([0.0, 0.0, 0.0, 1.0])
        q_desired = rotation_vector_to_quaternion(jnp.array([0.0, jnp.pi, 0.0]))

        error = quaternion_geodesic_error(q_current, q_desired)

        assert jnp.isfinite(error).all()
        assert_allclose(jnp.linalg.norm(error), jnp.pi, atol=1e-6)


class TestGeometricErrorForControl:
    """Integration tests for geometric errors in control applications."""

    def test_planar_control_scenario(self):
        """Test typical planar control scenario with angle wrapping."""
        # Robot at 10°, wants to go to 350°
        # A naive controller would command +340° (the long way)
        # A proper controller should command -20° (shortest path)

        theta_current = jnp.deg2rad(jnp.array(10.0))
        theta_desired = jnp.deg2rad(jnp.array(350.0))

        # Wrong (naive) approach:
        naive_error = theta_desired - theta_current
        assert jnp.abs(naive_error) > np.pi, "Naive error should be > π"

        # Correct (geometric) approach:
        geometric_error = angle_geodesic_error(theta_current, theta_desired)
        assert jnp.abs(geometric_error) < np.pi, "Geometric error should be < π"
        assert geometric_error < 0, "Should rotate clockwise (negative)"

    def test_3d_orientation_control_scenario(self):
        """Test 3D orientation control with shortest path."""
        # Scenario: current at 10°, desired at 350° around z
        # The rotation vector for 350° wraps to -10° (or stays at 350° ≈ 6.1 rad)
        # depending on quaternion sign convention

        R_current = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(10))
        R_desired = rotation_matrix_from_axis_angle([0, 0, 1], np.radians(350))

        # Geometric error: should be the shortest path
        geometric_error = rotation_matrix_geodesic_error(R_current, R_desired)
        geometric_error_magnitude = jnp.linalg.norm(geometric_error)

        # The geometric error should be 20° (or -20°), not 340°
        assert geometric_error_magnitude < np.radians(30), (
            f"Expected ~20°, got {np.degrees(geometric_error_magnitude)}°"
        )
        assert_allclose(geometric_error_magnitude, np.radians(20), atol=1e-4)

        # Direction should be negative z (clockwise to go from 10° to 350°)
        assert geometric_error[2] < 0, "Error should be clockwise (negative)"

    def test_jit_compatibility_of_errors(self):
        """Test that all error functions work with jit."""
        jit_angle_geodesic_error = jax.jit(angle_geodesic_error)
        jit_rotation_error = jax.jit(rotation_matrix_geodesic_error)
        jit_rotation_error_quat = jax.jit(quaternion_geodesic_error)

        theta_c = jnp.array(0.1)
        theta_d = jnp.array(0.2)
        assert jnp.isfinite(jit_angle_geodesic_error(theta_c, theta_d))

        R_c = rotation_matrix_from_axis_angle([1, 0, 0], 0.5)
        R_d = rotation_matrix_from_axis_angle([1, 0, 0], 0.8)
        error_R = jit_rotation_error(R_c, R_d)
        assert jnp.all(jnp.isfinite(error_R))

        q_c = rotation_matrix_to_quaternion(R_c)
        q_d = rotation_matrix_to_quaternion(R_d)
        error_q = jit_rotation_error_quat(q_c, q_d)
        assert jnp.all(jnp.isfinite(error_q))

    def test_grad_compatibility_of_errors(self):
        """Test that error functions are differentiable."""

        def loss_rotation_error(omega_d):
            R_c = jnp.eye(3)
            R_d = rotation_vector_to_rotation_matrix(omega_d)
            error = rotation_matrix_geodesic_error(R_c, R_d)
            return jnp.sum(error**2)

        omega_d = jnp.array([0.1, 0.2, 0.3])
        grad_fn = jax.grad(loss_rotation_error)
        grad = grad_fn(omega_d)

        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
