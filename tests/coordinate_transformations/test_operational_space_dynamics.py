"""Tests for OperationalSpaceDynamics class."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import pytest
from numpy.testing import assert_allclose

from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.systems.pcs import PCS
from soromox.systems.planar_pcs import PlanarPCS
from soromox.systems.pendulum import Pendulum
from soromox.utils.tolerance import Tolerance


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def pcs_robot():
    """Create a 2-segment PCS robot."""
    params = {
        "L": [0.1, 0.1],
        "r": [0.01, 0.01],
        "rho": [1000.0, 1000.0],
        "E": [1e6, 1e6],
        "G": [0.5e6, 0.5e6],
        "D": jnp.eye(12) * 0.1,
        "g": [0.0, 0.0, -9.81],
    }
    return PCS(num_segments=2, params=params)


@pytest.fixture
def planar_pcs_robot():
    """Create a 2-segment PlanarPCS robot."""
    params = {
        "L": [0.1, 0.1],
        "r": [0.01, 0.01],
        "rho": [1000.0, 1000.0],
        "E": [1e6, 1e6],
        "G": [0.5e6, 0.5e6],
        "D": jnp.eye(6) * 0.1,
        "g": [0.0, -9.81],
    }
    return PlanarPCS(num_segments=2, params=params)


@pytest.fixture
def pendulum_robot():
    """Create a 3-link pendulum robot."""
    params = {
        "m": jnp.array([1.0, 2.0, 0.8]),
        "I": jnp.array([0.1, 0.2, 0.05]),
        "L": jnp.array([1.0, 1.5, 0.8]),
        "Lc": jnp.array([0.5, 0.75, 0.4]),
        "g": jnp.array([0.0, -9.81]),
        "K": jnp.eye(3) * 10.0,
        "D": jnp.eye(3) * 0.5,
    }
    return Pendulum(params)


# -----------------------
# Initialization tests
# -----------------------


class TestOperationalSpaceDynamicsInit:
    """Tests for OperationalSpaceDynamics initialization."""

    def test_init_with_pcs(self, pcs_robot):
        """Test initialization with PCS robot."""
        s_ps = jnp.array([0.2])  # tip
        op_space = OperationalSpaceDynamics(robot=pcs_robot, s_ps=s_ps)

        assert op_space.n_points == 1
        assert op_space.n_pose_dim == 6  # 3D robot
        assert op_space.n_operational_space == 6

    def test_init_with_planar_pcs(self, planar_pcs_robot):
        """Test initialization with PlanarPCS robot."""
        s_ps = jnp.array([0.1, 0.2])  # midpoint and tip
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        assert op_space.n_points == 2
        assert op_space.n_pose_dim == 3  # planar robot
        assert op_space.n_operational_space == 6  # 2 points x 3 dims

    def test_init_with_pendulum(self, pendulum_robot):
        """Test initialization with Pendulum robot."""
        total_len = float(pendulum_robot.total_length)
        s_ps = jnp.array([total_len])  # tip
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        assert op_space.n_points == 1
        assert op_space.n_pose_dim == 3  # planar robot
        assert op_space.n_operational_space == 3

    def test_init_with_task_selector_uniform(self, planar_pcs_robot):
        """Test initialization with uniform task selector."""
        s_ps = jnp.array([0.1, 0.2])
        # Only select positions (deactivate orientation)
        task_selector = jnp.array([False, True, True])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        assert op_space.n_operational_space == 4  # 2 points x 2 dims (x, y only)

    def test_init_with_task_selector_per_point(self, planar_pcs_robot):
        """Test initialization with per-point task selector."""
        s_ps = jnp.array([0.1, 0.2])
        # Different selection per point
        task_selector = jnp.array(
            [[False, True, True], [True, True, True]]  # point 0: x, y only
        )  # point 1: all
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        assert op_space.n_operational_space == 5  # 2 + 3

    def test_init_empty_s_ps_raises(self, planar_pcs_robot):
        """Test that empty s_ps raises ValueError."""
        with pytest.raises(ValueError, match="at least one element"):
            OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=jnp.array([]))

    def test_init_empty_task_selector_raises(self, planar_pcs_robot):
        """Test that all-False task selector raises ValueError."""
        s_ps = jnp.array([0.1])
        task_selector = jnp.array([False, False, False])
        with pytest.raises(ValueError, match="at least one True"):
            OperationalSpaceDynamics(
                robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
            )


# -----------------------
# Jacobian tests
# -----------------------


class TestOperationalSpaceJacobian:
    """Tests for operational space Jacobian computation."""

    def test_jacobian_shape_pcs(self, pcs_robot):
        """Test Jacobian shape for PCS robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=pcs_robot, s_ps=s_ps)

        q = jnp.zeros(pcs_robot.num_dofs)
        J = op_space.jacobian(q)

        assert J.shape == (6, pcs_robot.num_dofs)

    def test_jacobian_shape_planar_pcs(self, planar_pcs_robot):
        """Test Jacobian shape for PlanarPCS robot with multiple points."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.zeros(planar_pcs_robot.num_dofs)
        J = op_space.jacobian(q)

        assert J.shape == (6, planar_pcs_robot.num_dofs)

    def test_jacobian_task_selection(self, planar_pcs_robot):
        """Test that task selection correctly reduces Jacobian."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # positions only
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.zeros(planar_pcs_robot.num_dofs)
        J = op_space.jacobian(q)

        assert J.shape == (2, planar_pcs_robot.num_dofs)

    def test_jacobian_and_derivative_shapes(self, pendulum_robot):
        """Test shapes of Jacobian and its derivative."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.linspace(0.6, -0.5, pendulum_robot.num_links)

        J, Jd = op_space.jacobian_and_derivative(q, qd)

        assert J.shape == (3, pendulum_robot.num_dofs)
        assert Jd.shape == (3, pendulum_robot.num_dofs)


# -----------------------
# Dynamically-consistent pseudo-inverse tests
# -----------------------


class TestDynamicallyConsistentPseudoinverse:
    """Tests for dynamically-consistent pseudo-inverse."""

    def test_j_jbar_identity(self, planar_pcs_robot):
        """Test that J @ J_bar = I in operational space."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        J, J_bar = op_space.jacobian_and_dynamically_consistent_pseudoinverse(q)

        J_Jbar = J @ J_bar
        I = jnp.eye(op_space.n_operational_space)

        assert_allclose(J_Jbar, I, rtol=1e-4, atol=1e-4)

    def test_j_jbar_identity_pendulum(self, pendulum_robot):
        """Test J @ J_bar = I for Pendulum robot."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        J, J_bar = op_space.jacobian_and_dynamically_consistent_pseudoinverse(q)

        J_Jbar = J @ J_bar
        I = jnp.eye(op_space.n_operational_space)

        assert_allclose(J_Jbar, I, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    def test_pseudoinverse_shape(self, pcs_robot):
        """Test shape of dynamically-consistent pseudo-inverse."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, False, False, True, True, True])  # pos only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        J_bar = op_space.dynamically_consistent_pseudoinverse(q)

        assert J_bar.shape == (pcs_robot.num_dofs, 3)


# -----------------------
# Inertia matrix tests
# -----------------------


class TestOperationalSpaceInertia:
    """Tests for operational space inertia matrix."""

    def test_inertia_matrix_symmetric(self, planar_pcs_robot):
        """Test that operational space inertia is symmetric."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        Lambda = op_space.inertia_matrix(q)

        assert_allclose(Lambda, Lambda.T, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    def test_inertia_matrix_positive_definite(self, pendulum_robot):
        """Test that operational space inertia is positive definite."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        Lambda = op_space.inertia_matrix(q)

        eigvals = jnp.linalg.eigvalsh(Lambda)
        assert jnp.all(eigvals > 0), f"Eigenvalues not all positive: {eigvals}"

    def test_inertia_matrix_shape(self, planar_pcs_robot):
        """Test shape of operational space inertia matrix."""
        s_ps = jnp.array([0.1])
        task_selector = jnp.array([False, True, True])  # 2D selection
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.zeros(planar_pcs_robot.num_dofs)
        Lambda = op_space.inertia_matrix(q)

        assert Lambda.shape == (2, 2)


# -----------------------
# Coriolis matrix tests
# -----------------------


class TestOperationalSpaceCoriolis:
    """Tests for operational space Coriolis matrix."""

    def test_coriolis_matrix_shape(self, planar_pcs_robot):
        """Test shape of operational space Coriolis matrix."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        qd = jnp.linspace(0.1, -0.1, planar_pcs_robot.num_dofs)
        mu = op_space.coriolis_matrix(q, qd)

        assert mu.shape == (3, 3)

    def test_coriolis_zero_at_zero_velocity(self, pendulum_robot):
        """Test that Coriolis terms are zero when velocity is zero."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.zeros(pendulum_robot.num_links)

        f_c = op_space.coriolis_force(q, qd)

        assert_allclose(
            f_c, jnp.zeros_like(f_c), rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


# -----------------------
# Force computation tests
# -----------------------


class TestOperationalSpaceForces:
    """Tests for operational space force computations."""

    def test_elastic_force_shape(self, pendulum_robot):
        """Test shape of operational space elastic force."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        f_el = op_space.elastic_force(q)

        assert f_el.shape == (3,)

    def test_gravitational_force_shape(self, planar_pcs_robot):
        """Test shape of operational space gravitational force."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        G_x = op_space.gravitational_force(q)

        assert G_x.shape == (3,)

    def test_damping_force_shape(self, pendulum_robot):
        """Test shape of operational space damping force."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.linspace(0.6, -0.5, pendulum_robot.num_links)
        f_d = op_space.damping_force(q, qd)

        assert f_d.shape == (3,)

    def test_damping_force_zero_at_zero_velocity(self, pendulum_robot):
        """Test that damping force is zero when velocity is zero."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.zeros(pendulum_robot.num_links)
        f_d = op_space.damping_force(q, qd)

        assert_allclose(
            f_d, jnp.zeros_like(f_d), rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


# -----------------------
# Null space projector tests
# -----------------------


class TestNullSpaceProjector:
    """Tests for null space projector."""

    def test_null_space_projector_shape(self, planar_pcs_robot):
        """Test shape of null space projector."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # 2D task
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.zeros(planar_pcs_robot.num_dofs)
        N = op_space.null_space_projector(q)

        assert N.shape == (planar_pcs_robot.num_dofs, planar_pcs_robot.num_dofs)

    def test_null_space_projector_idempotent(self, pendulum_robot):
        """Test that null space projector is idempotent (N @ N = N)."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        # Use reduced task to have non-trivial null space
        task_selector = jnp.array([False, True, True])  # 2D task, 3 DOF robot
        op_space = OperationalSpaceDynamics(
            robot=pendulum_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        N = op_space.null_space_projector(q)

        N_squared = N @ N
        assert_allclose(N_squared, N, rtol=1e-4, atol=1e-4)


# -----------------------
# JAX compatibility tests
# -----------------------


class TestJAXCompatibility:
    """Tests for JAX compatibility (JIT, vmap, grad)."""

    def test_jit_compatibility(self, pendulum_robot):
        """Test that methods are JIT-compatible (via eqx.filter_jit decorators)."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.linspace(0.6, -0.5, pendulum_robot.num_links)

        # Methods are already decorated with @eqx.filter_jit
        # Call them multiple times to verify caching works
        Lambda1 = op_space.inertia_matrix(q)
        Lambda2 = op_space.inertia_matrix(q)
        mu1 = op_space.coriolis_matrix(q, qd)
        mu2 = op_space.coriolis_matrix(q, qd)
        f_el1 = op_space.elastic_force(q)
        f_el2 = op_space.elastic_force(q)

        assert Lambda1.shape == (3, 3)
        assert mu1.shape == (3, 3)
        assert f_el1.shape == (3,)
        
        # Results should be identical (cached)
        assert_allclose(Lambda1, Lambda2, rtol=0, atol=0)
        assert_allclose(mu1, mu2, rtol=0, atol=0)
        assert_allclose(f_el1, f_el2, rtol=0, atol=0)

    def test_vmap_over_configurations(self, pendulum_robot):
        """Test vmapping over multiple configurations."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        # Batch of configurations
        batch_size = 5
        qs = jnp.tile(
            jnp.linspace(-0.3, 0.4, pendulum_robot.num_links), (batch_size, 1)
        )

        vmapped_inertia = jax.vmap(op_space.inertia_matrix)
        Lambdas = vmapped_inertia(qs)

        assert Lambdas.shape == (batch_size, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

