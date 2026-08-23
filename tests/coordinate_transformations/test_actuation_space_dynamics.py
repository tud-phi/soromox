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

# ruff: noqa: E402

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from system_param_builders import (
    pcs_params,
    pendulum_params,
)

from soromox.actuation import (
    ArticulatedTendonActuator,
    ThreadlikeActuator,
    ThreadlikeRouting,
)
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.coordinate_transformations import ActuationSpaceDynamics
from soromox.systems import (
    PCS,
    PCSStructure,
    Pendulum,
)

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def fully_actuated_pendulum():
    """Create a 3-link fully actuated tendon pendulum."""
    body = pendulum_params(
        mass=jnp.array([1.0, 1.0, 1.0]),
        moment_inertia=jnp.array([0.1, 0.1, 0.1]),
        length=jnp.array([0.5, 0.5, 0.5]),
        center_of_mass_length=jnp.array([0.25, 0.25, 0.25]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.eye(3) * 10.0,
        joint_damping=jnp.eye(3) * 0.5,
    )
    return Pendulum(
        body,
        actuators=ArticulatedTendonActuator.from_routing(jnp.tril(jnp.ones((3, 3)))),
    )


@pytest.fixture
def underactuated_pendulum():
    """Create a 3-link underactuated tendon pendulum (2 tendons on 3 joints)."""
    body = pendulum_params(
        mass=jnp.array([1.0, 1.0, 1.0]),
        moment_inertia=jnp.array([0.1, 0.1, 0.1]),
        length=jnp.array([0.5, 0.5, 0.5]),
        center_of_mass_length=jnp.array([0.25, 0.25, 0.25]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.eye(3) * 10.0,
        joint_damping=jnp.eye(3) * 0.5,
    )
    # Valid underactuated tendon routing (tendons can't skip joints)
    # Tendon 1: routes through joints 0 and 1 only (attaches at joint 1)
    # Tendon 2: routes through all 3 joints (attaches at joint 2)
    active_routing_matrix = jnp.array(
        [
            [1.0, 1.0, 0.0],  # Tendon 1: routes through joints 0,1
            [1.0, 1.0, 1.0],  # Tendon 2: routes through all 3 joints
        ]
    )
    return Pendulum(
        body,
        actuators=ArticulatedTendonActuator.from_routing(active_routing_matrix),
    )


@pytest.fixture
def fully_actuated_pcs():
    """Create a 1-segment fully actuated threadlike PCS.

    For a truly fully-actuated system, we use a single segment with 2 bending DOFs
    (kappa_y, kappa_z) and 2 tendons placed at different offsets.
    """
    # Note: D must be provided in full strain space (num_strains x num_strains = 6x6 for 1 segment)
    # The PCS.damping_matrix() method projects it to active strain space using B_xi
    body = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1e6]),
        shear_modulus=jnp.array([1e5]),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        damping_matrix=jnp.eye(6) * 0.01,  # Full strain space (6 strains x 1 segment)
    )
    # Select only bending strains (kappa_y, kappa_z = 2 DOFs total)
    # Strain components per segment: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
    strain_selector = jnp.array(
        [
            False,
            True,
            True,
            False,
            False,
            False,  # Segment 0: select kappa_y, kappa_z
        ]
    )
    # 2 tendons for 2 DOFs (fully actuated bending)
    # Tendon at +y creates moment around z (affects kappa_z)
    # Tendon at +z creates moment around y (affects kappa_y)
    tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.array(
            [[0.0, 0.005, 0.0], [0.0, 0.0, 0.005]]
        ),  # first tendon at +y, second at +z
        end_segment_index=(0, 0),
    )
    return PCS(
        body,
        structure=PCSStructure(strain_selector=strain_selector),
        actuators=ThreadlikeActuator.tendons(tendon_routing),
    )


@pytest.fixture
def underactuated_pcs():
    """Create a 2-segment underactuated threadlike PCS."""
    # Note: D must be provided in full strain space (num_strains x num_strains = 12x12)
    # The PCS.damping_matrix() method projects it to active strain space using B_xi
    body = pcs_params(
        length=jnp.array([0.1, 0.1]),
        radius=jnp.array([0.01, 0.01]),
        density=jnp.array([1000.0, 1000.0]),
        young_modulus=jnp.array([1e6, 1e6]),
        shear_modulus=jnp.array([1e5, 1e5]),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        damping_matrix=jnp.eye(12) * 0.01,  # Full strain space (6 strains x 2 segments)
    )
    # Select only bending strains (kappa_y, kappa_z per segment = 4 DOFs total)
    strain_selector = jnp.array(
        [
            False,
            True,
            True,
            False,
            False,
            False,  # Segment 0
            False,
            True,
            True,
            False,
            False,
            False,  # Segment 1
        ]
    )
    # 2 tendons for 4 DOFs (underactuated)
    # Use tendons at different offsets to ensure non-singular actuation matrix
    tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.array(
            [[0.0, 0.005, 0.0], [0.0, 0.0, 0.005]]
        ),  # first tendon at +y, second at +z
        end_segment_index=(1, 1),
    )
    return PCS(
        body,
        structure=PCSStructure(strain_selector=strain_selector),
        actuators=ThreadlikeActuator.tendons(tendon_routing),
    )


# -----------------------
# Helper function
# -----------------------


def random_configuration(robot, key=None):
    """Generate a small random configuration for a robot."""
    if key is None:
        key = jax.random.PRNGKey(0)
    return 0.1 * jax.random.normal(key, shape=(robot.num_dofs,))


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
        expected = robot.actuator_coordinates(q)
        assert_allclose(y_a, expected, rtol=1e-10)

    def test_actuated_coordinates_underactuated(self, underactuated_pendulum):
        """Test actuated_coordinates for underactuated system."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        y_a = asd.actuated_coordinates(q)

        # Actuated coords should equal tendon lengths
        expected = robot.actuator_coordinates(q)
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

    def test_potential_force_sums_conservative_forces(self, underactuated_pendulum):
        """Test that potential_force sums gravitational and elastic forces."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        tau_pot_y = asd.potential_force(q)
        expected = asd.gravitational_force(q) + asd.elastic_force(q)

        assert tau_pot_y.shape == (robot.num_dofs,)
        assert_allclose(tau_pot_y, expected, rtol=1e-10)


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

    def test_pendulum_actuator_coordinates(self, fully_actuated_pendulum):
        """Test the pendulum's common actuator-coordinate contract."""
        robot = fully_actuated_pendulum
        q = jnp.array([0.1, 0.2, 0.3])

        tendon_lengths = robot.actuators[0].coordinates(robot, q)
        actuation_coords = robot.actuator_coordinates(q)

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

    def test_dynamics_terms_match_individual_methods(self, underactuated_pendulum):
        """Test dynamics_terms returns the individual actuation-space terms."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        q = jnp.array([0.1, 0.2, 0.3])
        qd = jnp.array([0.01, 0.02, 0.03])
        M_y, Cyd, G_y = asd.dynamics_terms(q, qd)

        assert M_y.shape == (3, 3)
        assert Cyd.shape == (3,)
        assert G_y.shape == (3,)
        assert_allclose(M_y, asd.inertia_matrix(q), rtol=1e-10)
        assert_allclose(Cyd, asd.coriolis_force(q, qd), rtol=1e-10)
        assert_allclose(G_y, asd.gravitational_force(q), rtol=1e-10)

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
        _ = asd.potential_force(q)
        _ = asd.damping_force(q, qd)
        _ = asd.coriolis_force(q, qd)
        _ = asd.dynamics_terms(q, qd)
        _ = asd.actuation_matrix(q)


# -----------------------
# Reference trajectory conversion tests
# -----------------------


class TestReferenceTrajectoryConversion:
    """Tests for convert_reference_trajectory method."""

    def test_convert_trajectory_position_only(self, underactuated_pendulum):
        """Test conversion with position-only reference trajectory."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        # Create a simple configuration-space trajectory
        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, jnp.array([0.1, 0.2, 0.3]))  # (10, 3)

        ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check that output has correct shape
        assert ref_traj_actuation.x_des_ts is not None
        assert ref_traj_actuation.x_des_ts.shape == (10, 3)

        # Check that transformation is correct at each time step
        for i in range(len(ts)):
            q = x_des_ts[i]
            y_expected = asd.actuated_unactuated_coordinates(q)
            assert_allclose(ref_traj_actuation.x_des_ts[i], y_expected, rtol=1e-10)

    def test_convert_trajectory_with_velocities(self, underactuated_pendulum):
        """Test conversion with position and velocity reference trajectory."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        # Create a configuration-space trajectory with velocities
        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, jnp.array([0.1, 0.2, 0.3]))
        xd_des_ts = jnp.tile(jnp.array([0.1, 0.2, 0.3]), (10, 1))

        ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts, xd_des_ts=xd_des_ts)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check shapes
        assert ref_traj_actuation.x_des_ts is not None
        assert ref_traj_actuation.xd_des_ts is not None
        assert ref_traj_actuation.x_des_ts.shape == (10, 3)
        assert ref_traj_actuation.xd_des_ts.shape == (10, 3)

        # Check velocity transformation: yd = J(q) @ qd
        for i in range(len(ts)):
            q = x_des_ts[i]
            qd = xd_des_ts[i]
            J = asd.jacobian(q)
            yd_expected = J @ qd
            assert_allclose(ref_traj_actuation.xd_des_ts[i], yd_expected, rtol=1e-10)

    def test_convert_trajectory_with_accelerations(self, underactuated_pendulum):
        """Test conversion with position, velocity, and acceleration."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        # Create a configuration-space trajectory
        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, jnp.array([0.1, 0.2, 0.3]))
        xd_des_ts = jnp.tile(jnp.array([0.1, 0.2, 0.3]), (10, 1))
        xdd_des_ts = jnp.zeros((10, 3))  # Zero acceleration for simplicity

        ref_traj = ReferenceTrajectory(
            ts=ts, x_des_ts=x_des_ts, xd_des_ts=xd_des_ts, xdd_des_ts=xdd_des_ts
        )
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check shapes
        assert ref_traj_actuation.xdd_des_ts is not None
        assert ref_traj_actuation.xdd_des_ts.shape == (10, 3)

    def test_convert_trajectory_function_based(self, underactuated_pendulum):
        """Test conversion with function-based reference trajectory."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        # Create a function-based trajectory
        def q_des_fn(t):
            return t * jnp.array([0.1, 0.2, 0.3])

        ts = jnp.linspace(0.0, 1.0, 10)
        ref_traj = ReferenceTrajectory(ts=ts, x_des_fn=q_des_fn)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check that x_des_fn is present and works
        assert ref_traj_actuation.x_des_fn is not None

        # Check at a specific time
        t_test = jnp.array(0.5)
        q_test = q_des_fn(t_test)
        y_expected = asd.actuated_unactuated_coordinates(q_test)
        y_actual = ref_traj_actuation.x_des_fn(t_test)
        assert_allclose(y_actual, y_expected, rtol=1e-10)

    def test_convert_trajectory_preserves_metadata(self, underactuated_pendulum):
        """Test that trajectory metadata is preserved."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, jnp.array([0.1, 0.2, 0.3]))

        ref_traj = ReferenceTrajectory(
            ts=ts,
            x_des_ts=x_des_ts,
            is_planar=True,
            n_points=2,
        )
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        assert ref_traj_actuation.is_planar == ref_traj.is_planar
        assert ref_traj_actuation.n_points == ref_traj.n_points

    def test_convert_trajectory_fully_actuated(self, fully_actuated_pendulum):
        """Test conversion for fully actuated system."""
        robot = fully_actuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, jnp.array([0.1, 0.2, 0.3]))

        ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # For fully actuated system with identity routing, y = tendon_lengths
        assert ref_traj_actuation.x_des_ts is not None
        assert ref_traj_actuation.x_des_ts.shape == x_des_ts.shape

    def test_convert_trajectory_velocity_consistency(self, underactuated_pendulum):
        """Test that velocity transformation is consistent with position derivative."""
        robot = underactuated_pendulum
        asd = ActuationSpaceDynamics(robot)

        # Create a smooth trajectory
        def q_des_fn(t):
            return jnp.sin(t) * jnp.array([0.1, 0.2, 0.3])

        def qd_des_fn(t):
            return jnp.cos(t) * jnp.array([0.1, 0.2, 0.3])

        ts = jnp.linspace(0.0, 1.0, 10)
        ref_traj = ReferenceTrajectory(ts=ts, x_des_fn=q_des_fn, xd_des_fn=qd_des_fn)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check velocity at a specific time
        t_test = jnp.array(0.5)
        q = q_des_fn(t_test)
        qd = qd_des_fn(t_test)
        J = asd.jacobian(q)
        yd_expected = J @ qd

        assert ref_traj_actuation.xd_des_fn is not None
        yd_actual = ref_traj_actuation.xd_des_fn(t_test)
        assert_allclose(yd_actual, yd_expected, rtol=1e-10)


# -----------------------
# System-independent parameterized tests
# -----------------------


class TestActuationSpaceDynamicsSystemIndependent:
    """
    System-independent tests for ActuationSpaceDynamics.

    These tests are parameterized to run on multiple robot types (pendulum, PCS, etc.)
    without hard-coding system-specific values.
    """

    @pytest.fixture(
        params=[
            "fully_actuated_pendulum",
            "underactuated_pendulum",
            "fully_actuated_pcs",
            "underactuated_pcs",
        ]
    )
    def robot(self, request):
        """Parameterized fixture that yields different robot instances."""
        return request.getfixturevalue(request.param)

    @pytest.fixture(params=["underactuated_pendulum", "underactuated_pcs"])
    def underactuated_robot(self, request):
        """Parameterized fixture for underactuated robots only."""
        return request.getfixturevalue(request.param)

    @pytest.fixture(params=["fully_actuated_pendulum", "fully_actuated_pcs"])
    def fully_actuated_robot(self, request):
        """Parameterized fixture for fully actuated robots only."""
        return request.getfixturevalue(request.param)

    # -----------------------
    # Initialization tests
    # -----------------------

    def test_init_dimensions(self, robot):
        """Test that ActuationSpaceDynamics initializes with correct dimensions."""
        asd = ActuationSpaceDynamics(robot)

        assert asd.n_actuated == robot.num_actuators
        assert asd.n_unactuated == robot.num_dofs - robot.num_actuators
        assert asd.n_actuated + asd.n_unactuated == robot.num_dofs

    def test_dynamics_terms_match_individual_methods(self, robot):
        """Fused dynamics agree for articulated and nonlinear PCS mappings."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)
        qd = random_configuration(robot, key=jax.random.PRNGKey(1))

        M_y, Cyd, G_y = asd.dynamics_terms(q, qd)

        assert_allclose(M_y, asd.inertia_matrix(q), rtol=1e-9, atol=1e-12)
        assert_allclose(Cyd, asd.coriolis_force(q, qd), rtol=1e-9, atol=1e-12)
        assert_allclose(G_y, asd.gravitational_force(q), rtol=1e-9, atol=1e-12)

    def test_h_unactuated_shape(self, robot):
        """Test that H_unactuated has correct shape."""
        asd = ActuationSpaceDynamics(robot)

        expected_shape = (asd.n_unactuated, robot.num_dofs)
        assert asd.H_unactuated.shape == expected_shape

    # -----------------------
    # Coordinate transformation tests
    # -----------------------

    def test_actuated_coordinates_shape(self, robot):
        """Test actuated_coordinates returns correct shape."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        y_a = asd.actuated_coordinates(q)

        assert y_a.shape == (asd.n_actuated,)

    def test_actuated_coordinates_matches_robot(self, robot):
        """Test actuated_coordinates matches the robot actuator coordinates."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        y_a = asd.actuated_coordinates(q)
        expected = robot.actuator_coordinates(q)

        assert_allclose(y_a, expected, rtol=1e-10)

    def test_unactuated_coordinates_shape(self, robot):
        """Test unactuated_coordinates returns correct shape."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        y_u = asd.unactuated_coordinates(q)

        assert y_u.shape == (asd.n_unactuated,)

    def test_unactuated_coordinates_linear_mapping(self, robot):
        """Test unactuated_coordinates is a linear mapping."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        y_u = asd.unactuated_coordinates(q)
        expected = asd.H_unactuated @ q

        assert_allclose(y_u, expected, rtol=1e-10)

    def test_actuated_unactuated_coordinates_concatenation(self, robot):
        """Test actuated_unactuated_coordinates concatenates correctly."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        y = asd.actuated_unactuated_coordinates(q)
        y_a = asd.actuated_coordinates(q)
        y_u = asd.unactuated_coordinates(q)

        assert y.shape == (robot.num_dofs,)
        assert_allclose(y[: asd.n_actuated], y_a, rtol=1e-10)
        assert_allclose(y[asd.n_actuated :], y_u, rtol=1e-10)

    # -----------------------
    # Jacobian tests
    # -----------------------

    def test_jacobian_is_square(self, robot):
        """Test that full Jacobian is square."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J = asd.jacobian(q)

        assert J.shape == (robot.num_dofs, robot.num_dofs)

    def test_jacobian_is_invertible(self, robot):
        """Test that full Jacobian is invertible."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J = asd.jacobian(q)
        J_inv = asd.jacobian_inverse(q)

        # J @ J_inv should be identity
        assert_allclose(J @ J_inv, jnp.eye(robot.num_dofs), atol=1e-8)

    def test_jacobian_actuated_shape(self, robot):
        """Test jacobian_actuated has correct shape."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_a = asd.jacobian_actuated(q)

        assert J_a.shape == (asd.n_actuated, robot.num_dofs)

    def test_jacobian_actuated_equals_actuation_matrix_transpose(self, robot):
        """Test that jacobian_actuated equals A_at^T."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_a = asd.jacobian_actuated(q)
        A_at = robot.actuation_matrix(q)

        assert_allclose(J_a, A_at.T, rtol=1e-10)

    def test_jacobian_stacks_correctly(self, robot):
        """Test that jacobian stacks actuated and unactuated Jacobians."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J = asd.jacobian(q)
        J_a = asd.jacobian_actuated(q)
        J_u = asd.jacobian_unactuated(q)

        assert_allclose(J[: asd.n_actuated], J_a, rtol=1e-10)
        assert_allclose(J[asd.n_actuated :], J_u, rtol=1e-10)

    # -----------------------
    # Dynamics matrix tests
    # -----------------------

    def test_inertia_matrix_is_symmetric(self, robot):
        """Test that M_y is symmetric."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        M_y = asd.inertia_matrix(q)

        assert_allclose(M_y, M_y.T, atol=1e-10)

    def test_inertia_matrix_is_positive_definite(self, robot):
        """Test that M_y is positive definite."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        M_y = asd.inertia_matrix(q)
        eigvals = jnp.linalg.eigvalsh(M_y)

        assert jnp.all(eigvals > 0)

    def test_inertia_matrix_transformation_formula(self, robot):
        """Test that M_y = J^{-T} @ M @ J^{-1}."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_inv = asd.jacobian_inverse(q)
        M = robot.inertia_matrix(q)
        M_y = asd.inertia_matrix(q)

        expected = J_inv.T @ M @ J_inv
        assert_allclose(M_y, expected, rtol=1e-8)

    def test_coriolis_matrix_shape(self, robot):
        """Test that Coriolis matrix has correct shape."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)
        qd = random_configuration(robot, key=jax.random.PRNGKey(1))

        eta_y = asd.coriolis_matrix(q, qd)

        assert eta_y.shape == (robot.num_dofs, robot.num_dofs)

    def test_damping_matrix_transformation_formula(self, robot):
        """Test that D_y = J^{-T} @ D @ J^{-1}."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_inv = asd.jacobian_inverse(q)
        D = robot.damping_matrix(q)
        D_y = asd.damping_matrix(q)

        expected = J_inv.T @ D @ J_inv
        assert_allclose(D_y, expected, rtol=1e-8)

    # -----------------------
    # Force tests
    # -----------------------

    def test_elastic_force_transformation_formula(self, robot):
        """Test that tau_el_y = J^{-T} @ tau_el."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_inv = asd.jacobian_inverse(q)
        tau_el = robot.elastic_force(q)
        tau_el_y = asd.elastic_force(q)

        expected = J_inv.T @ tau_el
        assert_allclose(tau_el_y, expected, rtol=1e-8)

    def test_gravitational_force_transformation_formula(self, robot):
        """Test that G_y = J^{-T} @ G."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        J_inv = asd.jacobian_inverse(q)
        G = robot.gravitational_force(q)
        G_y = asd.gravitational_force(q)

        expected = J_inv.T @ G
        assert_allclose(G_y, expected, rtol=1e-8)

    def test_damping_force_transformation_formula(self, robot):
        """Test that tau_d_y = J^{-T} @ D @ qd."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)
        qd = random_configuration(robot, key=jax.random.PRNGKey(1))

        J_inv = asd.jacobian_inverse(q)
        D = robot.damping_matrix(q)
        tau_d_y = asd.damping_force(q, qd)

        expected = J_inv.T @ D @ qd
        assert_allclose(tau_d_y, expected, rtol=1e-8)

    # -----------------------
    # Actuation matrix tests
    # -----------------------

    def test_actuation_matrix_shape(self, robot):
        """Test actuation matrix has correct shape."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        A_y = asd.actuation_matrix(q)

        assert A_y.shape == (robot.num_dofs, robot.num_actuators)

    def test_actuation_matrix_structure(self, robot):
        """Test actuation matrix structure [[I], [0]]."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        A_y = asd.actuation_matrix(q)
        n_a = asd.n_actuated
        n_u = asd.n_unactuated

        # Top block should be identity
        assert_allclose(A_y[:n_a, :n_a], jnp.eye(n_a), rtol=1e-10)

        # Bottom block should be zeros (if underactuated)
        if n_u > 0:
            assert_allclose(A_y[n_a:, :], jnp.zeros((n_u, n_a)), rtol=1e-10)

    def test_actuation_matrix_fully_actuated_is_identity(self, fully_actuated_robot):
        """Test actuation matrix is identity for fully actuated system."""
        robot = fully_actuated_robot
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)

        A_y = asd.actuation_matrix(q)

        assert_allclose(A_y, jnp.eye(robot.num_dofs), rtol=1e-10)

    # -----------------------
    # Reference trajectory conversion tests
    # -----------------------

    def test_convert_trajectory_position(self, robot):
        """Test position conversion for reference trajectory."""
        asd = ActuationSpaceDynamics(robot)
        n_dof = robot.num_dofs

        # Create a simple configuration-space trajectory
        ts = jnp.linspace(0.0, 1.0, 10)
        # Create trajectory with small values
        x_des_ts = jnp.outer(ts, 0.1 * jnp.ones(n_dof))

        ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check that output has correct shape
        assert ref_traj_actuation.x_des_ts is not None
        assert ref_traj_actuation.x_des_ts.shape == (10, n_dof)

        # Check transformation at each time step
        for i in range(len(ts)):
            q = x_des_ts[i]
            y_expected = asd.actuated_unactuated_coordinates(q)
            # Use atol for near-zero values
            assert_allclose(
                ref_traj_actuation.x_des_ts[i], y_expected, rtol=1e-10, atol=1e-15
            )

    def test_convert_trajectory_velocity(self, robot):
        """Test velocity conversion for reference trajectory."""
        asd = ActuationSpaceDynamics(robot)
        n_dof = robot.num_dofs

        ts = jnp.linspace(0.0, 1.0, 10)
        x_des_ts = jnp.outer(ts, 0.1 * jnp.ones(n_dof))
        xd_des_ts = jnp.tile(0.1 * jnp.ones(n_dof), (10, 1))

        ref_traj = ReferenceTrajectory(ts=ts, x_des_ts=x_des_ts, xd_des_ts=xd_des_ts)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check velocity transformation: yd = J(q) @ qd
        assert ref_traj_actuation.xd_des_ts is not None
        for i in range(len(ts)):
            q = x_des_ts[i]
            qd = xd_des_ts[i]
            J = asd.jacobian(q)
            yd_expected = J @ qd
            # Use atol for near-zero values
            assert_allclose(
                ref_traj_actuation.xd_des_ts[i], yd_expected, rtol=1e-10, atol=1e-15
            )

    def test_convert_trajectory_acceleration(self, robot):
        """Test acceleration conversion for reference trajectory.

        This test verifies the acceleration transformation formula:
            ydd = J(q) @ qdd + Jd(q, qd) @ qd

        where Jd is the time derivative of the Jacobian. This is particularly
        important for systems with configuration-dependent actuation matrices
        (like PCS with threadlike actuation) where Jd ≠ 0.
        """
        asd = ActuationSpaceDynamics(robot)
        n_dof = robot.num_dofs

        # Create trajectory with non-zero velocity and acceleration
        # to test the Jd @ qd term
        ts = jnp.linspace(0.0, 1.0, 10)

        # Use a sinusoidal trajectory for smooth, non-zero derivatives
        # q(t) = A * sin(ω * t), qd(t) = A * ω * cos(ω * t), qdd(t) = -A * ω² * sin(ω * t)
        omega = 2.0
        amplitude = 0.1 * jnp.ones(n_dof)

        x_des_ts = jnp.outer(jnp.sin(omega * ts), amplitude)
        xd_des_ts = jnp.outer(omega * jnp.cos(omega * ts), amplitude)
        xdd_des_ts = jnp.outer(-(omega**2) * jnp.sin(omega * ts), amplitude)

        ref_traj = ReferenceTrajectory(
            ts=ts, x_des_ts=x_des_ts, xd_des_ts=xd_des_ts, xdd_des_ts=xdd_des_ts
        )
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        # Check acceleration transformation: ydd = J @ qdd + Jd @ qd
        assert ref_traj_actuation.xdd_des_ts is not None
        assert ref_traj_actuation.xdd_des_ts.shape == (10, n_dof)

        for i in range(len(ts)):
            q = x_des_ts[i]
            qd = xd_des_ts[i]
            qdd = xdd_des_ts[i]

            # Compute J @ qdd
            J = asd.jacobian(q)
            J_qdd = J @ qdd

            # Compute Jd @ qd using JVP (same method as the implementation)
            def J_times_qd(q_inner, qd_current=qd):
                return asd.jacobian(q_inner) @ qd_current

            Jd_qd = jax.jvp(J_times_qd, (q,), (qd,))[1]

            # Expected acceleration in actuation space
            ydd_expected = J_qdd + Jd_qd

            # Use atol for near-zero values
            assert_allclose(
                ref_traj_actuation.xdd_des_ts[i], ydd_expected, rtol=1e-10, atol=1e-15
            )

    def test_convert_trajectory_function_based(self, robot):
        """Test function-based reference trajectory conversion."""
        asd = ActuationSpaceDynamics(robot)
        n_dof = robot.num_dofs

        def q_des_fn(t):
            return t * 0.1 * jnp.ones(n_dof)

        ts = jnp.linspace(0.0, 1.0, 10)
        ref_traj = ReferenceTrajectory(ts=ts, x_des_fn=q_des_fn)
        ref_traj_actuation = asd.convert_reference_trajectory(ref_traj)

        assert ref_traj_actuation.x_des_fn is not None

        # Check at a specific time
        t_test = jnp.array(0.5)
        q_test = q_des_fn(t_test)
        y_expected = asd.actuated_unactuated_coordinates(q_test)
        y_actual = ref_traj_actuation.x_des_fn(t_test)
        assert_allclose(y_actual, y_expected, rtol=1e-10)

    # -----------------------
    # Integration tests
    # -----------------------

    def test_all_methods_jittable(self, robot):
        """Test that all methods can be JIT compiled and executed."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)
        qd = random_configuration(robot, key=jax.random.PRNGKey(1))

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
        _ = asd.potential_force(q)
        _ = asd.damping_force(q, qd)
        _ = asd.coriolis_force(q, qd)
        _ = asd.dynamics_terms(q, qd)
        _ = asd.actuation_matrix(q)

    def test_dynamics_shapes_consistent(self, robot):
        """Test that all dynamics matrices and forces have consistent shapes."""
        asd = ActuationSpaceDynamics(robot)
        q = random_configuration(robot)
        qd = random_configuration(robot, key=jax.random.PRNGKey(1))
        n_dof = robot.num_dofs

        M_y = asd.inertia_matrix(q)
        eta_y = asd.coriolis_matrix(q, qd)
        D_y = asd.damping_matrix(q)
        G_y = asd.gravitational_force(q)
        tau_el_y = asd.elastic_force(q)
        tau_pot_y = asd.potential_force(q)
        M_terms, Cyd, G_terms = asd.dynamics_terms(q, qd)

        assert M_y.shape == (n_dof, n_dof)
        assert eta_y.shape == (n_dof, n_dof)
        assert D_y.shape == (n_dof, n_dof)
        assert G_y.shape == (n_dof,)
        assert tau_el_y.shape == (n_dof,)
        assert tau_pot_y.shape == (n_dof,)
        assert M_terms.shape == (n_dof, n_dof)
        assert Cyd.shape == (n_dof,)
        assert G_terms.shape == (n_dof,)
