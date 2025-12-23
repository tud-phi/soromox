"""Tests for ActuationSpaceDynamics class.

References:
    Pustina, P., Della Santina, C., Boyer, F., De Luca, A., & Renda, F. (2024).
    Input decoupling of lagrangian systems via coordinate transformation:
    General characterization and its application to soft robotics.
    IEEE Transactions on Robotics, 40, 2098-2110.

    Pustina, P. (2024). Analysis and control of the underactuation in
    cable-driven parallel robots and continuum soft robots.
    PhD Thesis, Sapienza University of Rome.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from soromox.coordinate_transformations import ActuationSpaceDynamics
from soromox.systems.tendon_actuated_pendulum import TendonActuatedPendulum

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def fully_actuated_pendulum():
    """Create a 3-link fully actuated tendon pendulum (identity routing)."""
    params = {
        "m": jnp.array([1.0, 1.0, 1.0]),
        "I": jnp.array([0.1, 0.1, 0.1]),
        "L": jnp.array([0.5, 0.5, 0.5]),
        "Lc": jnp.array([0.25, 0.25, 0.25]),
        "g": jnp.array([0.0, -9.81]),
        "K": jnp.eye(3) * 10.0,
        "D": jnp.eye(3) * 0.5,
    }
    return TendonActuatedPendulum(params)


@pytest.fixture
def underactuated_pendulum():
    """Create a 3-link underactuated tendon pendulum (2 tendons on 3 joints)."""
    params = {
        "m": jnp.array([1.0, 1.0, 1.0]),
        "I": jnp.array([0.1, 0.1, 0.1]),
        "L": jnp.array([0.5, 0.5, 0.5]),
        "Lc": jnp.array([0.25, 0.25, 0.25]),
        "g": jnp.array([0.0, -9.81]),
        "K": jnp.eye(3) * 10.0,
        "D": jnp.eye(3) * 0.5,
    }
    # Valid underactuated tendon routing (tendons can't skip joints)
    # Tendon 1: routes through joints 0 and 1 only (attaches at joint 1)
    # Tendon 2: routes through all 3 joints (attaches at joint 2)
    tendon_params = {
        "R_at": jnp.array(
            [
                [1.0, 1.0, 0.0],  # Tendon 1: routes through joints 0,1
                [1.0, 1.0, 1.0],  # Tendon 2: routes through all 3 joints
            ]
        )
    }
    return TendonActuatedPendulum(params, tendon_params)


# -----------------------
# Initialization tests
# -----------------------


class TestActuationSpaceDynamicsInit:
    """Tests for ActuationSpaceDynamics initialization."""

    def test_init_fully_actuated(self, fully_actuated_pendulum):
        """Test initialization with fully actuated pendulum."""
        robot = fully_actuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        assert asd.n_actuated == robot.num_actuators
        assert asd.n_unactuated == robot.num_dofs - robot.num_actuators
        assert asd.n_actuated == 3
        assert asd.n_unactuated == 0

    def test_init_underactuated(self, underactuated_pendulum):
        """Test initialization with underactuated pendulum."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        assert asd.n_actuated == robot.num_actuators
        assert asd.n_unactuated == robot.num_dofs - robot.num_actuators
        assert asd.n_actuated == 2
        assert asd.n_unactuated == 1

    def test_init_with_custom_h_unactuated(self, underactuated_pendulum):
        """Test initialization with custom H_unactuated matrix."""
        robot = underactuated_pendulum
        # Custom unactuated mapping: just pick the first coordinate
        H_custom = jnp.array([[1.0, 0.0, 0.0]])

        asd = ActuationSpaceDynamics(robot, H_unactuated=H_custom)
        assert_allclose(asd.H_unactuated, H_custom)

    def test_init_invalid_h_unactuated_shape(self, underactuated_pendulum):
        """Test that invalid H_unactuated shape raises error."""
        robot = underactuated_pendulum
        # Wrong shape: should be (1, 3) but we give (2, 3)
        H_wrong = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        with pytest.raises(ValueError):
            ActuationSpaceDynamics(robot, H_unactuated=H_wrong)


# -----------------------
# Coordinate transformation tests
# -----------------------


class TestCoordinateTransformations:
    """Tests for coordinate transformation methods."""

    def test_actuated_coordinates_fully_actuated(self, fully_actuated_pendulum):
        """Test actuated_coordinates for fully actuated system."""
        robot = fully_actuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        y_a = asd.actuated_coordinates(q)

        # For identity routing, actuated coords equal tendon lengths
        expected = robot.actuated_coordinates(q)
        assert_allclose(y_a, expected, rtol=1e-10)

    def test_actuated_coordinates_underactuated(self, underactuated_pendulum):
        """Test actuated_coordinates for underactuated system."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        y_a = asd.actuated_coordinates(q)

        # Actuated coords should equal tendon lengths
        expected = robot.actuated_coordinates(q)
        assert_allclose(y_a, expected, rtol=1e-10)

    def test_unactuated_coordinates(self, underactuated_pendulum):
        """Test unactuated_coordinates for underactuated system."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        y_u = asd.unactuated_coordinates(q)

        # Should be a linear mapping
        expected = asd.H_unactuated @ q
        assert_allclose(y_u, expected, rtol=1e-10)
        assert y_u.shape == (asd.n_unactuated,)

    def test_actuated_unactuated_coordinates(self, underactuated_pendulum):
        """Test actuated_unactuated_coordinates concatenates correctly."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        y = asd.actuated_unactuated_coordinates(q)

        y_a = asd.actuated_coordinates(q)
        y_u = asd.unactuated_coordinates(q)

        assert y.shape == (robot.num_dofs,)
        assert_allclose(y[: asd.n_actuated], y_a, rtol=1e-10)
        assert_allclose(y[asd.n_actuated :], y_u, rtol=1e-10)


# -----------------------
# Jacobian tests
# -----------------------


class TestJacobians:
    """Tests for Jacobian computations."""

    def test_jacobian_actuated_equals_actuation_matrix_transpose(
        self, underactuated_pendulum
    ):
        """Test that jacobian_actuated equals A_at^T."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_a = asd.jacobian_actuated(q)
        A_at = robot.actuation_matrix(q)

        assert_allclose(J_a, A_at.T, rtol=1e-10)

    def test_jacobian_unactuated_equals_h_unactuated(self, underactuated_pendulum):
        """Test that jacobian_unactuated equals H_unactuated."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_u = asd.jacobian_unactuated(q)

        assert_allclose(J_u, asd.H_unactuated, rtol=1e-10)

    def test_jacobian_is_square(self, underactuated_pendulum):
        """Test that full Jacobian is square."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J = asd.jacobian(q)

        assert J.shape == (robot.num_dofs, robot.num_dofs)

    def test_jacobian_is_invertible(self, underactuated_pendulum):
        """Test that full Jacobian is invertible."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J = asd.jacobian(q)
        J_inv = asd.jacobian_inverse(q)

        # J @ J_inv should be identity
        assert_allclose(J @ J_inv, jnp.eye(robot.num_dofs), atol=1e-10)

    def test_jacobian_stacks_correctly(self, underactuated_pendulum):
        """Test that jacobian stacks actuated and unactuated Jacobians."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J = asd.jacobian(q)
        J_a = asd.jacobian_actuated(q)
        J_u = asd.jacobian_unactuated(q)

        assert_allclose(J[: asd.n_actuated], J_a, rtol=1e-10)
        assert_allclose(J[asd.n_actuated :], J_u, rtol=1e-10)


# -----------------------
# Dynamics matrix tests
# -----------------------


class TestDynamicsMatrices:
    """Tests for actuation space dynamics matrices."""

    def test_inertia_matrix_is_symmetric_positive_definite(
        self, underactuated_pendulum
    ):
        """Test that M_y is symmetric positive definite."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        M_y = asd.inertia_matrix(q)

        # Check symmetry
        assert_allclose(M_y, M_y.T, atol=1e-10)

        # Check positive definiteness
        eigvals = jnp.linalg.eigvalsh(M_y)
        assert jnp.all(eigvals > 0)

    def test_inertia_matrix_transformation(self, underactuated_pendulum):
        """Test that M_y = J^{-T} @ M @ J^{-1}."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_inv = asd.jacobian_inverse(q)
        M = robot.inertia_matrix(q)
        M_y = asd.inertia_matrix(q)

        expected = J_inv.T @ M @ J_inv
        assert_allclose(M_y, expected, rtol=1e-10)

    def test_coriolis_matrix_shape(self, underactuated_pendulum):
        """Test that Coriolis matrix has correct shape."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        qd = jnp.array([0.01, 0.02, 0.03])
        eta_y = asd.coriolis_matrix(q, qd)

        assert eta_y.shape == (robot.num_dofs, robot.num_dofs)

    def test_damping_matrix_transformation(self, underactuated_pendulum):
        """Test that D_y = J^{-T} @ D @ J^{-1}."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_inv = asd.jacobian_inverse(q)
        D = robot.damping_matrix(q)
        D_y = asd.damping_matrix(q)

        expected = J_inv.T @ D @ J_inv
        assert_allclose(D_y, expected, rtol=1e-10)


# -----------------------
# Force tests
# -----------------------


class TestForces:
    """Tests for actuation space forces."""

    def test_elastic_force_transformation(self, underactuated_pendulum):
        """Test that tau_el_y = J^{-T} @ tau_el."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_inv = asd.jacobian_inverse(q)
        tau_el = robot.elastic_force(q)
        tau_el_y = asd.elastic_force(q)

        expected = J_inv.T @ tau_el
        assert_allclose(tau_el_y, expected, rtol=1e-10)

    def test_gravitational_force_transformation(self, underactuated_pendulum):
        """Test that G_y = J^{-T} @ G."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        J_inv = asd.jacobian_inverse(q)
        G = robot.gravitational_force(q)
        G_y = asd.gravitational_force(q)

        expected = J_inv.T @ G
        assert_allclose(G_y, expected, rtol=1e-10)

    def test_damping_force_transformation(self, underactuated_pendulum):
        """Test that tau_d_y = J^{-T} @ D @ qd."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        qd = jnp.array([0.01, 0.02, 0.03])
        J_inv = asd.jacobian_inverse(q)
        D = robot.damping_matrix(q)
        tau_d_y = asd.damping_force(q, qd)

        expected = J_inv.T @ D @ qd
        assert_allclose(tau_d_y, expected, rtol=1e-10)


# -----------------------
# Actuation matrix tests
# -----------------------


class TestActuationMatrix:
    """Tests for actuation matrix in actuation space."""

    def test_actuation_matrix_structure_fully_actuated(self, fully_actuated_pendulum):
        """Test actuation matrix is identity for fully actuated system."""
        robot = fully_actuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        A_y = asd.actuation_matrix(q)

        # For fully actuated: A_y = I
        assert_allclose(A_y, jnp.eye(robot.num_dofs), rtol=1e-10)

    def test_actuation_matrix_structure_underactuated(self, underactuated_pendulum):
        """Test actuation matrix structure for underactuated system."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        A_y = asd.actuation_matrix(q)

        # Structure: [[I_m], [0_{n-m, m}]]
        n_a = asd.n_actuated
        n_u = asd.n_unactuated

        # Top block should be identity
        assert_allclose(A_y[:n_a, :n_a], jnp.eye(n_a), rtol=1e-10)

        # Bottom block should be zeros
        assert_allclose(A_y[n_a:, :], jnp.zeros((n_u, n_a)), rtol=1e-10)

    def test_actuation_matrix_shape(self, underactuated_pendulum):
        """Test actuation matrix has correct shape."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        A_y = asd.actuation_matrix(q)

        assert A_y.shape == (robot.num_dofs, robot.num_actuators)


# -----------------------
# Integration tests
# -----------------------


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_actuated_coordinates_alias(self, fully_actuated_pendulum):
        """Test that actuated_coordinates is an alias of active_tendon_length for the TendonActuatedPendulum."""
        robot = fully_actuated_pendulum
        q = jnp.array([0.1, 0.2, 0.3])

        tendon_lengths = robot.active_tendon_length(q)
        actuation_coords = robot.actuated_coordinates(q)

        assert_allclose(tendon_lengths, actuation_coords, rtol=1e-10)

    def test_dynamics_consistency(self, underactuated_pendulum):
        """Test that dynamics are consistent in actuation space."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        qd = jnp.array([0.01, 0.02, 0.03])

        # Compute all dynamics matrices
        M_y = asd.inertia_matrix(q)
        eta_y = asd.coriolis_matrix(q, qd)
        D_y = asd.damping_matrix(q)
        G_y = asd.gravitational_force(q)
        tau_el_y = asd.elastic_force(q)

        # All should have correct shapes
        assert M_y.shape == (3, 3)
        assert eta_y.shape == (3, 3)
        assert D_y.shape == (3, 3)
        assert G_y.shape == (3,)
        assert tau_el_y.shape == (3,)

    def test_jit_compilation(self, underactuated_pendulum):
        """Test that all methods work correctly (they are already JIT-compiled via @eqx.filter_jit)."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        qd = jnp.array([0.01, 0.02, 0.03])

        # These should all work (methods are decorated with @eqx.filter_jit)
        _ = asd.actuated_coordinates(q)
        _ = asd.unactuated_coordinates(q)
        _ = asd.actuated_unactuated_coordinates(q)
        _ = asd.jacobian(q)
        _ = asd.jacobian_inverse(q)
        _ = asd.inertia_matrix(q)
        _ = asd.coriolis_matrix(q, qd)
        _ = asd.damping_matrix(q)
        _ = asd.elastic_force(q)
        _ = asd.gravitational_force(q)
        _ = asd.damping_force(q, qd)
        _ = asd.coriolis_force(q, qd)
        _ = asd.actuation_matrix(q)
