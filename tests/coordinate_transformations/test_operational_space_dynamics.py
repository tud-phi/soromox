"""Tests for OperationalSpaceDynamics class."""

# ruff: noqa: E402

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, pendulum_params, planar_pcs_params

from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.systems import PCS, Pendulum, PlanarPCS
from soromox.utils.geometry.rotations import RotationRepresentation
from soromox.utils.tolerance import Tolerance

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def pcs_robot():
    """Create a 2-segment PCS robot."""
    params = pcs_params(
        length=jnp.array([0.1, 0.1]),
        radius=jnp.array([0.01, 0.01]),
        density=jnp.array([1000.0, 1000.0]),
        young_modulus=jnp.array([1e6, 1e6]),
        shear_modulus=jnp.array([0.5e6, 0.5e6]),
        damping_matrix=jnp.eye(12) * 0.1,
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return PCS(params=params)


@pytest.fixture
def planar_pcs_robot():
    """Create a 2-segment PlanarPCS robot."""
    params = planar_pcs_params(
        length=jnp.array([0.1, 0.1]),
        radius=jnp.array([0.01, 0.01]),
        density=jnp.array([1000.0, 1000.0]),
        young_modulus=jnp.array([1e6, 1e6]),
        shear_modulus=jnp.array([0.5e6, 0.5e6]),
        damping_matrix=jnp.eye(6) * 0.1,
        gravity=jnp.array([0.0, -9.81]),
    )
    return PlanarPCS(params=params)


@pytest.fixture
def pendulum_robot():
    """Create a 3-link pendulum robot."""
    params = pendulum_params(
        mass=jnp.array([1.0, 2.0, 0.8]),
        moment_inertia=jnp.array([0.1, 0.2, 0.05]),
        length=jnp.array([1.0, 1.5, 0.8]),
        center_of_mass_length=jnp.array([0.5, 0.75, 0.4]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.eye(3) * 10.0,
        joint_damping=jnp.eye(3) * 0.5,
    )
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

    def test_jacobian_and_time_derivative_shapes(self, pendulum_robot):
        """Test shapes of Jacobian and its derivative."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.linspace(0.6, -0.5, pendulum_robot.num_links)

        J, Jd = op_space.jacobian_and_time_derivative(q, qd)

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
        identity = jnp.eye(op_space.n_operational_space)

        assert_allclose(J_Jbar, identity, rtol=1e-4, atol=1e-4)

    def test_j_jbar_identity_pendulum(self, pendulum_robot):
        """Test J @ J_bar = I for Pendulum robot."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        J, J_bar = op_space.jacobian_and_dynamically_consistent_pseudoinverse(q)

        J_Jbar = J @ J_bar
        identity = jnp.eye(op_space.n_operational_space)

        assert_allclose(J_Jbar, identity, rtol=Tolerance.rtol(), atol=Tolerance.atol())

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

    def test_potential_force_sums_conservative_forces(self, pendulum_robot):
        """Test that potential_force sums gravitational and elastic forces."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        f_pot = op_space.potential_force(q)
        expected = op_space.gravitational_force(q) + op_space.elastic_force(q)

        assert f_pot.shape == (3,)
        assert_allclose(f_pot, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    def test_dynamics_terms_match_individual_methods(self, pendulum_robot):
        """Test dynamics_terms returns the individual operational-space terms."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        qd = jnp.linspace(0.6, -0.5, pendulum_robot.num_links)
        Lambda, Cxd, G_x = op_space.dynamics_terms(q, qd)

        assert Lambda.shape == (3, 3)
        assert Cxd.shape == (3,)
        assert G_x.shape == (3,)
        assert_allclose(
            Lambda,
            op_space.inertia_matrix(q),
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        assert_allclose(
            Cxd,
            op_space.coriolis_force(q, qd),
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        assert_allclose(
            G_x,
            op_space.gravitational_force(q),
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
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


# -----------------------
# Rotation representation tests
# -----------------------


class TestRotationRepresentation:
    """Tests for different rotation representations in operational space."""

    def test_rotation_vector_representation_pcs(self, pcs_robot):
        """Test rotation vector representation for 3D robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        assert op_space.n_pose_dim == 6  # 3 orientation + 3 position
        assert op_space.n_orientation_dim == 3
        assert not op_space.is_planar

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)
        assert x.shape == (6,)

    def test_quaternion_representation_pcs(self, pcs_robot):
        """Test quaternion representation for 3D robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        assert op_space.n_pose_dim == 7  # 4 quaternion + 3 position
        assert op_space.n_orientation_dim == 4
        assert op_space.n_operational_space == 6  # Velocity space is always 6 for 3D
        assert op_space.n_pose_operational_space == 7  # Pose space with quaternion

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)
        assert x.shape == (7,)  # Returns pose-space coordinates

        # Check that quaternion part is unit quaternion
        quat = x[:4]
        assert_allclose(jnp.linalg.norm(quat), 1.0, atol=1e-6)

    def test_6d_representation_pcs(self, pcs_robot):
        """Test 6D continuous representation for 3D robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        assert op_space.n_pose_dim == 9  # 6 rotation + 3 position
        assert op_space.n_orientation_dim == 6
        assert op_space.n_operational_space == 6  # Velocity space is always 6 for 3D
        assert op_space.n_pose_operational_space == 9  # Pose space with 6D rotation

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)
        assert x.shape == (9,)  # Returns pose-space coordinates

    def test_planar_ignores_rotation_representation(self, planar_pcs_robot):
        """Test that planar robots ignore rotation representation setting."""
        s_ps = jnp.array([0.2])

        # All representations should give same result for planar robot
        op_space_rv = OperationalSpaceDynamics(
            robot=planar_pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )
        op_space_quat = OperationalSpaceDynamics(
            robot=planar_pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )
        op_space_6d = OperationalSpaceDynamics(
            robot=planar_pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        # All should be planar
        assert op_space_rv.is_planar
        assert op_space_quat.is_planar
        assert op_space_6d.is_planar

        # All should have same pose dimension (3 for planar)
        assert op_space_rv.n_pose_dim == 3
        assert op_space_quat.n_pose_dim == 3
        assert op_space_6d.n_pose_dim == 3

        # Coordinates should be identical
        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x_rv = op_space_rv.operational_space_coordinates(q)
        x_quat = op_space_quat.operational_space_coordinates(q)
        x_6d = op_space_6d.operational_space_coordinates(q)

        assert_allclose(x_rv, x_quat, atol=1e-10)
        assert_allclose(x_rv, x_6d, atol=1e-10)


# -----------------------
# Geometric pose error tests
# -----------------------


class TestComputePoseError:
    """Tests for geometric pose error computation."""

    def test_zero_error_planar(self, planar_pcs_robot):
        """Test that same pose gives zero error for planar robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)

        error = op_space.compute_pose_error(x, x)
        assert_allclose(error, jnp.zeros_like(error), atol=1e-10)

    def test_zero_error_pendulum(self, pendulum_robot):
        """Test that same pose gives zero error for pendulum."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        x = op_space.operational_space_coordinates(q)

        error = op_space.compute_pose_error(x, x)
        assert_allclose(error, jnp.zeros_like(error), atol=1e-10)

    def test_angle_wrapping_planar(self, planar_pcs_robot):
        """Test that angle wrapping works correctly for planar robot."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        # Create poses where angle is near wrap-around point
        # Current: [10°, 0, 0], Desired: [350°, 0, 0]
        # Naive error: 340°, Geometric error: -20°
        x_current = jnp.array([jnp.deg2rad(10), 0.0, 0.0])
        x_desired = jnp.array([jnp.deg2rad(350), 0.0, 0.0])

        error = op_space.compute_pose_error(x_current, x_desired)

        # Orientation error should be ~-20°, not 340°
        assert jnp.abs(error[0]) < jnp.pi / 2, (
            f"Angle error should be small, got {jnp.degrees(error[0])}°"
        )
        assert_allclose(error[0], jnp.deg2rad(-20), atol=1e-5)

        # Position errors should be zero
        assert_allclose(error[1:], jnp.zeros(2), atol=1e-10)

    def test_angle_wrapping_pendulum(self, pendulum_robot):
        """Test angle wrapping for pendulum at end-effector."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        # Similar test but checking that pose error respects angle wrapping
        # Note: For pendulum, the orientation at tip is sum of all joint angles
        q1 = jnp.array([0.0, 0.05, 0.05])  # Small positive angles
        q2 = jnp.array([jnp.pi - 0.1, jnp.pi - 0.05, jnp.pi - 0.05])  # Near wrap

        x1 = op_space.operational_space_coordinates(q1)
        x2 = op_space.operational_space_coordinates(q2)

        error = op_space.compute_pose_error(x1, x2)

        # Check that orientation error is reasonable (not huge due to wrapping issues)
        # The exact value depends on the pendulum FK, but it shouldn't be > π
        assert jnp.abs(error[0]) <= jnp.pi + 0.1, (
            f"Angle error exceeds π: {jnp.degrees(error[0])}°"
        )

    def test_position_error_correct(self, planar_pcs_robot):
        """Test that position error is computed correctly."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        x_current = jnp.array([0.0, 0.1, 0.2])
        x_desired = jnp.array([0.0, 0.3, 0.5])

        error = op_space.compute_pose_error(x_current, x_desired)

        # Position error should be desired - current
        assert_allclose(error[1:], jnp.array([0.2, 0.3]), atol=1e-10)

    def test_pose_error_jit_compatible(self, pendulum_robot):
        """Test that pose error computation works with JIT."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        x = op_space.operational_space_coordinates(q)
        x_des = x + 0.1

        # compute_pose_error is already JIT-decorated via @eqx.filter_jit
        # Just call it directly to test JIT compatibility
        error = op_space.compute_pose_error(x, x_des)

        assert error.shape == x.shape
        assert jnp.all(jnp.isfinite(error))


class TestPoseErrorFor3DRobots:
    """Tests for pose error with 3D robots and different representations."""

    def test_rotation_vector_error_zero(self, pcs_robot):
        """Test zero error for rotation vector representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)

        error = op_space.compute_pose_error(x, x)
        assert_allclose(error, jnp.zeros_like(error), atol=1e-10)

    def test_quaternion_error_zero(self, pcs_robot):
        """Test zero error for quaternion representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)

        error = op_space.compute_pose_error(x, x)
        # Error dimension is 6 (3 orient + 3 pos) even though x is 7D
        assert error.shape == (6,) or jnp.allclose(error, 0, atol=1e-8)

    def test_6d_error_zero(self, pcs_robot):
        """Test zero error for 6D representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_coordinates(q)

        error = op_space.compute_pose_error(x, x)
        # Error dimension is 6 (3 orient + 3 pos) even though x is 9D
        assert error.shape == (6,) or jnp.allclose(error, 0, atol=1e-8)


# -----------------------
# Error vs naive subtraction tests
# -----------------------


class TestGeometricVsNaiveError:
    """Tests comparing geometric error to naive subtraction."""

    def test_geometric_error_shorter_path_planar(self, planar_pcs_robot):
        """Test that geometric error takes shorter path for planar case."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        # Current at 5°, desired at 355° (5° the other way from 0)
        x_current = jnp.array([jnp.deg2rad(5), 0.0, 0.0])
        x_desired = jnp.array([jnp.deg2rad(355), 0.0, 0.0])

        # Naive error
        naive_error = x_desired - x_current
        naive_theta_delta = naive_error[0]

        # Geometric error
        geometric_error = op_space.compute_pose_error(x_current, x_desired)
        geometric_theta_delta = geometric_error[0]

        # Naive error is ~350°, geometric should be ~-10°
        assert jnp.abs(naive_theta_delta) > jnp.pi  # Long way around
        assert jnp.abs(geometric_theta_delta) < jnp.pi  # Short way around
        assert jnp.abs(geometric_theta_delta) < jnp.abs(naive_theta_delta)

    def test_orientation_error_always_less_than_pi(self, planar_pcs_robot):
        """Test that geometric orientation error magnitude is always ≤ π."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        # Test many random angle pairs
        key = jax.random.PRNGKey(42)
        for _ in range(50):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            theta_current = jax.random.uniform(
                subkey1, minval=-2 * jnp.pi, maxval=2 * jnp.pi
            )
            theta_desired = jax.random.uniform(
                subkey2, minval=-2 * jnp.pi, maxval=2 * jnp.pi
            )

            x_current = jnp.array([theta_current, 0.0, 0.0])
            x_desired = jnp.array([theta_desired, 0.0, 0.0])

            error = op_space.compute_pose_error(x_current, x_desired)

            assert jnp.abs(error[0]) <= jnp.pi + 1e-6, (
                f"Orientation error > π: {error[0]} for θ_c={theta_current}, θ_d={theta_desired}"
            )


# -----------------------
# Task selection basis matrix tests
# -----------------------


class TestTaskSelectionBasisMatrices:
    """Tests for B_task (velocity) and B_task_pose (pose) basis matrices."""

    def test_btask_shape_planar_full_selection(self, planar_pcs_robot):
        """Test B_task shape for planar robot with full selection."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        # B_task: (total_velocity_dim, n_operational_space)
        assert op_space.B_task.shape == (3, 3)
        assert op_space.B_task_pose.shape == (3, 3)

        # For full selection, B_task should be identity
        assert_allclose(op_space.B_task, jnp.eye(3), atol=1e-10)
        assert_allclose(op_space.B_task_pose, jnp.eye(3), atol=1e-10)

    def test_btask_shape_planar_position_only(self, planar_pcs_robot):
        """Test B_task shape for planar robot with position-only selection."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # [omega_z, v_x, v_y]
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        # B_task: (3, 2), B_task_pose: (3, 2)
        assert op_space.B_task.shape == (3, 2)
        assert op_space.B_task_pose.shape == (3, 2)
        assert op_space.n_operational_space == 2
        assert op_space.n_pose_operational_space == 2

    def test_btask_shape_3d_full_selection(self, pcs_robot):
        """Test B_task shape for 3D robot with full selection."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        # B_task: (6, 6), B_task_pose: (6, 6)
        assert op_space.B_task.shape == (6, 6)
        assert op_space.B_task_pose.shape == (6, 6)
        assert op_space.n_operational_space == 6
        assert op_space.n_pose_operational_space == 6

    def test_btask_shape_3d_position_only(self, pcs_robot):
        """Test B_task shape for 3D robot with position-only selection."""
        s_ps = jnp.array([0.2])
        # Select only position (v_x, v_y, v_z at indices 3,4,5)
        task_selector = jnp.array([False, False, False, True, True, True])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        # B_task: (6, 3) velocity space
        assert op_space.B_task.shape == (6, 3)
        # B_task_pose: (6, 3) pose space (same for ROTATION_VECTOR)
        assert op_space.B_task_pose.shape == (6, 3)
        assert op_space.n_operational_space == 3
        assert op_space.n_pose_operational_space == 3

    def test_btask_pose_differs_for_quaternion(self, pcs_robot):
        """Test that B_task_pose differs from B_task for QUATERNION representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        # Velocity space: 6D (3 angular + 3 linear)
        assert op_space.B_task.shape == (6, 6)
        # Pose space: 7D (4 quaternion + 3 position)
        assert op_space.B_task_pose.shape == (7, 7)

        assert op_space.n_operational_space == 6
        assert op_space.n_pose_operational_space == 7

    def test_btask_pose_differs_for_6d_rotation(self, pcs_robot):
        """Test that B_task_pose differs from B_task for ROTATION_MATRIX_6D representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        # Velocity space: 6D (3 angular + 3 linear)
        assert op_space.B_task.shape == (6, 6)
        # Pose space: 9D (6 rotation + 3 position)
        assert op_space.B_task_pose.shape == (9, 9)

        assert op_space.n_operational_space == 6
        assert op_space.n_pose_operational_space == 9

    def test_btask_pose_position_only_quaternion(self, pcs_robot):
        """Test B_task_pose for position-only task with QUATERNION."""
        s_ps = jnp.array([0.2])
        # Select only position
        task_selector = jnp.array([False, False, False, True, True, True])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        # Velocity space: (6, 3) - position only
        assert op_space.B_task.shape == (6, 3)
        # Pose space: (7, 3) - position only (no quaternion)
        assert op_space.B_task_pose.shape == (7, 3)

        assert op_space.n_operational_space == 3
        assert op_space.n_pose_operational_space == 3

    def test_btask_multiple_points(self, planar_pcs_robot):
        """Test B_task shapes for multiple points."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        # 2 points × 3 dims = 6
        assert op_space.B_task.shape == (6, 6)
        assert op_space.B_task_pose.shape == (6, 6)

    def test_btask_multiple_points_mixed_selection(self, planar_pcs_robot):
        """Test B_task for multiple points with different selection per point."""
        s_ps = jnp.array([0.1, 0.2])
        # Point 0: positions only, Point 1: all
        task_selector = jnp.array([[False, True, True], [True, True, True]])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        # 2 + 3 = 5 selected dimensions
        assert op_space.B_task.shape == (6, 5)
        assert op_space.B_task_pose.shape == (6, 5)
        assert op_space.n_operational_space == 5
        assert op_space.n_pose_operational_space == 5


# -----------------------
# Partial rotation selection validation tests
# -----------------------


class TestRotationSelectionValidation:
    """Tests for validation of rotation selection (all-or-nothing rule for 3D)."""

    def test_partial_rotation_selection_raises_3d(self, pcs_robot):
        """Test that partial rotation selection raises ValueError for 3D robot."""
        s_ps = jnp.array([0.2])

        # Select only omega_x and omega_y, not omega_z - should fail
        task_selector = jnp.array([True, True, False, True, True, True])
        with pytest.raises(
            ValueError, match="rotation components must be either ALL selected"
        ):
            OperationalSpaceDynamics(
                robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
            )

    def test_partial_rotation_selection_raises_3d_single_axis(self, pcs_robot):
        """Test that selecting single rotation axis raises ValueError for 3D robot."""
        s_ps = jnp.array([0.2])

        # Select only omega_z
        task_selector = jnp.array([False, False, True, True, True, True])
        with pytest.raises(
            ValueError, match="rotation components must be either ALL selected"
        ):
            OperationalSpaceDynamics(
                robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
            )

    def test_all_rotation_selected_ok(self, pcs_robot):
        """Test that all rotation components selected is valid."""
        s_ps = jnp.array([0.2])

        # Select all rotation and all position
        task_selector = jnp.array([True, True, True, True, True, True])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
        )
        assert op_space.n_operational_space == 6
        assert op_space.rotation_selected_per_point[0]

    def test_no_rotation_selected_ok(self, pcs_robot):
        """Test that no rotation components selected is valid."""
        s_ps = jnp.array([0.2])

        # Select no rotation, all position
        task_selector = jnp.array([False, False, False, True, True, True])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
        )
        assert op_space.n_operational_space == 3
        assert not op_space.rotation_selected_per_point[0]

    def test_planar_any_rotation_selection_ok(self, planar_pcs_robot):
        """Test that planar robots can select rotation freely."""
        s_ps = jnp.array([0.2])

        # Planar has only 1 rotation component, so there's no partial selection issue
        task_selector = jnp.array([True, True, True])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )
        assert op_space.n_operational_space == 3

        task_selector = jnp.array([False, True, True])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )
        assert op_space.n_operational_space == 2

    def test_per_point_rotation_selection_3d(self, pcs_robot):
        """Test per-point rotation selection for 3D robot."""
        s_ps = jnp.array([0.1, 0.2])

        # Point 0: no rotation, Point 1: all rotation
        task_selector = jnp.array(
            [
                [False, False, False, True, True, True],  # no rot, all pos
                [True, True, True, True, True, True],  # all rot, all pos
            ]
        )
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        assert op_space.n_operational_space == 9  # 3 + 6
        assert not op_space.rotation_selected_per_point[0]
        assert op_space.rotation_selected_per_point[1]

    def test_per_point_partial_rotation_raises_3d(self, pcs_robot):
        """Test that partial rotation in one point raises error for 3D robot."""
        s_ps = jnp.array([0.1, 0.2])

        # Point 0: all selected, Point 1: partial rotation
        task_selector = jnp.array(
            [
                [True, True, True, True, True, True],  # all
                [True, False, True, True, True, True],  # partial rotation - should fail
            ]
        )
        with pytest.raises(ValueError, match="For point 1"):
            OperationalSpaceDynamics(
                robot=pcs_robot, s_ps=s_ps, task_selector=task_selector
            )


# -----------------------
# operational_space_poses vs operational_space_coordinates tests
# -----------------------


class TestPosesVsCoordinates:
    """Tests comparing operational_space_poses and operational_space_coordinates."""

    def test_poses_and_coords_equal_full_selection_planar(self, planar_pcs_robot):
        """Test that poses and coordinates are equal for full selection (planar)."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (3,)
        assert coords.shape == (3,)
        assert_allclose(poses, coords, atol=1e-10)

    def test_poses_full_coords_selected_planar(self, planar_pcs_robot):
        """Test that poses are full and coordinates are selected (planar)."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # positions only
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (3,)  # Full: [theta, x, y]
        assert coords.shape == (2,)  # Selected: [x, y]

        # coords should be the position part of poses
        assert_allclose(coords, poses[1:], atol=1e-10)

    def test_poses_full_coords_selected_3d(self, pcs_robot):
        """Test that poses are full and coordinates are selected (3D)."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [False, False, False, True, True, True]
        )  # position only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (6,)  # Full: [rot(3), pos(3)]
        assert coords.shape == (3,)  # Selected: [pos(3)]

        # coords should be the position part of poses (last 3 elements for ROTATION_VECTOR)
        n_orient = op_space.n_orientation_dim  # 3 for ROTATION_VECTOR
        assert_allclose(coords, poses[n_orient:], atol=1e-10)

    def test_poses_full_coords_selected_quaternion(self, pcs_robot):
        """Test poses vs coordinates for QUATERNION representation."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [False, False, False, True, True, True]
        )  # position only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (7,)  # Full: [quat(4), pos(3)]
        assert coords.shape == (3,)  # Selected: [pos(3)]

        # coords should be the position part of poses (last 3 elements for QUATERNION)
        n_orient = op_space.n_orientation_dim  # 4 for QUATERNION
        assert_allclose(coords, poses[n_orient:], atol=1e-10)

    def test_poses_full_coords_selected_6d_rotation(self, pcs_robot):
        """Test poses vs coordinates for ROTATION_MATRIX_6D representation."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [False, False, False, True, True, True]
        )  # position only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (9,)  # Full: [rot6d(6), pos(3)]
        assert coords.shape == (3,)  # Selected: [pos(3)]

        # coords should be the position part of poses (last 3 elements for 6D rotation)
        n_orient = op_space.n_orientation_dim  # 6 for ROTATION_MATRIX_6D
        assert_allclose(coords, poses[n_orient:], atol=1e-10)

    def test_poses_coords_multiple_points(self, planar_pcs_robot):
        """Test poses vs coordinates for multiple points with mixed selection."""
        s_ps = jnp.array([0.1, 0.2])
        # Point 0: positions only, Point 1: all
        task_selector = jnp.array([[False, True, True], [True, True, True]])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        # Poses: 2 points × 3D = 6
        assert poses.shape == (6,)
        # Coords: 2 (point 0 pos) + 3 (point 1 all) = 5
        assert coords.shape == (5,)

        # Check that coords are the selected parts of poses
        # Point 0 coords should be poses[1:3] (x, y)
        # Point 1 coords should be poses[3:6] (theta, x, y)
        assert_allclose(coords[:2], poses[1:3], atol=1e-10)
        assert_allclose(coords[2:], poses[3:6], atol=1e-10)


# -----------------------
# compute_pose_error vs compute_task_pose_error tests
# -----------------------


class TestPoseErrorVsTaskPoseError:
    """Tests comparing compute_pose_error and compute_task_pose_error."""

    def test_equal_for_full_selection_planar(self, planar_pcs_robot):
        """Test that errors are equal for full selection (planar)."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.03])

        full_error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        assert full_error.shape == (3,)
        assert task_error.shape == (3,)
        assert_allclose(full_error, task_error, atol=1e-10)

    def test_task_error_subset_of_full_planar(self, planar_pcs_robot):
        """Test that task error is a subset of full error (planar)."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # positions only
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.03])

        full_error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        assert full_error.shape == (3,)
        assert task_error.shape == (2,)

        # Task error should be the position part of full error
        assert_allclose(task_error, full_error[1:], atol=1e-10)

    def test_task_error_subset_of_full_3d(self, pcs_robot):
        """Test that task error is a subset of full error (3D)."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [False, False, False, True, True, True]
        )  # position only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.02, 0.03, 0.01, -0.01])

        full_error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        # Full error is in velocity space (6D)
        assert full_error.shape == (6,)
        # Task error is selected (3D)
        assert task_error.shape == (3,)

        # Task error should be the position part of full error
        assert_allclose(task_error, full_error[3:], atol=1e-10)

    def test_task_error_rotation_only_3d(self, pcs_robot):
        """Test task error for rotation-only selection (3D)."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [True, True, True, False, False, False]
        )  # rotation only
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        q = jnp.zeros(pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.02, 0.03, 0.01, -0.01])

        full_error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        # Full error is in velocity space (6D)
        assert full_error.shape == (6,)
        # Task error is selected (3D rotation)
        assert task_error.shape == (3,)

        # Task error should be the rotation part of full error
        assert_allclose(task_error, full_error[:3], atol=1e-10)

    def test_task_error_multiple_points(self, planar_pcs_robot):
        """Test task error for multiple points with mixed selection."""
        s_ps = jnp.array([0.1, 0.2])
        # Point 0: positions only, Point 1: all
        task_selector = jnp.array([[False, True, True], [True, True, True]])
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.03, 0.2, 0.04, 0.01])

        full_error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        # Full error: 2 points × 3D = 6
        assert full_error.shape == (6,)
        # Task error: 2 (point 0 pos) + 3 (point 1 all) = 5
        assert task_error.shape == (5,)

        # Point 0 task error should be full_error[1:3]
        # Point 1 task error should be full_error[3:6]
        assert_allclose(task_error[:2], full_error[1:3], atol=1e-10)
        assert_allclose(task_error[2:], full_error[3:6], atol=1e-10)

    def test_pose_error_input_validation_wrong_shape(self, planar_pcs_robot):
        """Test that compute_pose_error raises for wrong input shape."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array([False, True, True])  # positions only
        op_space = OperationalSpaceDynamics(
            robot=planar_pcs_robot, s_ps=s_ps, task_selector=task_selector
        )

        # Try to pass task-selected coordinates instead of full poses
        wrong_shape = jnp.array([0.1, 0.2])  # 2D instead of 3D
        x = op_space.operational_space_poses(jnp.zeros(planar_pcs_robot.num_dofs))

        with pytest.raises(ValueError, match="must have shape"):
            op_space.compute_pose_error(wrong_shape, x)

        with pytest.raises(ValueError, match="must have shape"):
            op_space.compute_pose_error(x, wrong_shape)


# -----------------------
# 3D rotation representation comprehensive tests
# -----------------------


class TestRotationRepresentationComprehensive:
    """Comprehensive tests for different rotation representations."""

    def test_rotation_vector_dimensions(self, pcs_robot):
        """Test dimensions for ROTATION_VECTOR representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        assert op_space.n_pose_dim == 6
        assert op_space.n_orientation_dim == 3
        assert op_space.n_velocity_dim == 6
        assert op_space.n_angular_velocity_dim == 3

        q = jnp.zeros(pcs_robot.num_dofs)
        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (6,)
        assert coords.shape == (6,)

    def test_quaternion_dimensions(self, pcs_robot):
        """Test dimensions for QUATERNION representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        assert op_space.n_pose_dim == 7  # 4 quat + 3 pos
        assert op_space.n_orientation_dim == 4
        assert op_space.n_velocity_dim == 6
        assert op_space.n_angular_velocity_dim == 3

        q = jnp.zeros(pcs_robot.num_dofs)
        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (7,)
        assert coords.shape == (7,)

        # Quaternion should be unit
        quat = poses[:4]
        assert_allclose(jnp.linalg.norm(quat), 1.0, atol=1e-6)

    def test_rotation_matrix_6d_dimensions(self, pcs_robot):
        """Test dimensions for ROTATION_MATRIX_6D representation."""
        s_ps = jnp.array([0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        assert op_space.n_pose_dim == 9  # 6 rot + 3 pos
        assert op_space.n_orientation_dim == 6
        assert op_space.n_velocity_dim == 6
        assert op_space.n_angular_velocity_dim == 3

        q = jnp.zeros(pcs_robot.num_dofs)
        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (9,)
        assert coords.shape == (9,)

    def test_jacobian_same_for_all_representations(self, pcs_robot):
        """Test that Jacobian is the same regardless of rotation representation."""
        s_ps = jnp.array([0.2])

        op_rv = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )
        op_quat = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )
        op_6d = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        q = jnp.linspace(-0.05, 0.05, pcs_robot.num_dofs)

        J_rv = op_rv.jacobian(q)
        J_quat = op_quat.jacobian(q)
        J_6d = op_6d.jacobian(q)

        # All should have the same shape and values (velocity space is independent of pose repr)
        assert J_rv.shape == J_quat.shape == J_6d.shape == (6, pcs_robot.num_dofs)
        assert_allclose(J_rv, J_quat, atol=1e-10)
        assert_allclose(J_rv, J_6d, atol=1e-10)

    def test_dynamics_same_for_all_representations(self, pcs_robot):
        """Test that inertia matrix is the same regardless of rotation representation."""
        s_ps = jnp.array([0.2])

        op_rv = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )
        op_quat = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )
        op_6d = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        q = jnp.linspace(-0.05, 0.05, pcs_robot.num_dofs)

        Lambda_rv = op_rv.inertia_matrix(q)
        Lambda_quat = op_quat.inertia_matrix(q)
        Lambda_6d = op_6d.inertia_matrix(q)

        # All should have the same shape and values
        assert Lambda_rv.shape == Lambda_quat.shape == Lambda_6d.shape == (6, 6)
        assert_allclose(Lambda_rv, Lambda_quat, atol=1e-10)
        assert_allclose(Lambda_rv, Lambda_6d, atol=1e-10)

    def test_pose_error_zero_for_same_pose_all_representations(self, pcs_robot):
        """Test that same pose gives zero error for all representations."""
        s_ps = jnp.array([0.2])

        for rep in [
            RotationRepresentation.ROTATION_VECTOR,
            RotationRepresentation.QUATERNION,
            RotationRepresentation.ROTATION_MATRIX_6D,
        ]:
            op_space = OperationalSpaceDynamics(
                robot=pcs_robot, s_ps=s_ps, rotation_representation=rep
            )

            q = jnp.linspace(-0.05, 0.05, pcs_robot.num_dofs)
            x = op_space.operational_space_poses(q)

            error = op_space.compute_pose_error(x, x)
            assert error.shape == (6,)  # Always 6D velocity-space error
            assert_allclose(error, jnp.zeros(6), atol=1e-10)

    def test_task_pose_error_zero_for_same_pose_all_representations(self, pcs_robot):
        """Test that same pose gives zero task error for all representations."""
        s_ps = jnp.array([0.2])
        task_selector = jnp.array(
            [False, False, False, True, True, True]
        )  # position only

        for rep in [
            RotationRepresentation.ROTATION_VECTOR,
            RotationRepresentation.QUATERNION,
            RotationRepresentation.ROTATION_MATRIX_6D,
        ]:
            op_space = OperationalSpaceDynamics(
                robot=pcs_robot,
                s_ps=s_ps,
                task_selector=task_selector,
                rotation_representation=rep,
            )

            q = jnp.linspace(-0.05, 0.05, pcs_robot.num_dofs)
            x = op_space.operational_space_poses(q)

            error = op_space.compute_task_pose_error(x, x)
            assert error.shape == (3,)  # 3D position error
            assert_allclose(error, jnp.zeros(3), atol=1e-10)

    def test_position_same_for_all_representations(self, pcs_robot):
        """Test that position part is the same for all representations."""
        s_ps = jnp.array([0.2])

        op_rv = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )
        op_quat = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )
        op_6d = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_MATRIX_6D,
        )

        q = jnp.linspace(-0.05, 0.05, pcs_robot.num_dofs)

        poses_rv = op_rv.operational_space_poses(q)
        poses_quat = op_quat.operational_space_poses(q)
        poses_6d = op_6d.operational_space_poses(q)

        # Extract position (last 3 elements)
        pos_rv = poses_rv[3:]
        pos_quat = poses_quat[4:]
        pos_6d = poses_6d[6:]

        assert_allclose(pos_rv, pos_quat, atol=1e-10)
        assert_allclose(pos_rv, pos_6d, atol=1e-10)


# -----------------------
# Multiple points tests
# -----------------------


class TestMultiplePoints:
    """Tests for operational space with multiple points."""

    def test_two_points_planar_full_selection(self, planar_pcs_robot):
        """Test two points for planar robot with full selection."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        assert op_space.n_points == 2
        assert op_space.n_operational_space == 6  # 2 × 3

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)
        J = op_space.jacobian(q)

        assert poses.shape == (6,)
        assert coords.shape == (6,)
        assert J.shape == (6, planar_pcs_robot.num_dofs)

    def test_two_points_3d_full_selection(self, pcs_robot):
        """Test two points for 3D robot with full selection."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        assert op_space.n_points == 2
        assert op_space.n_operational_space == 12  # 2 × 6

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)
        J = op_space.jacobian(q)

        assert poses.shape == (12,)
        assert coords.shape == (12,)
        assert J.shape == (12, pcs_robot.num_dofs)

    def test_two_points_3d_quaternion(self, pcs_robot):
        """Test two points for 3D robot with quaternion representation."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            rotation_representation=RotationRepresentation.QUATERNION,
        )

        assert op_space.n_points == 2
        assert op_space.n_operational_space == 12  # 2 × 6 (velocity space)
        assert op_space.n_pose_operational_space == 14  # 2 × 7 (pose space)

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)
        J = op_space.jacobian(q)

        assert poses.shape == (14,)
        assert coords.shape == (14,)
        assert J.shape == (12, pcs_robot.num_dofs)

        # Check both quaternions are unit
        quat1 = poses[:4]
        quat2 = poses[7:11]
        assert_allclose(jnp.linalg.norm(quat1), 1.0, atol=1e-6)
        assert_allclose(jnp.linalg.norm(quat2), 1.0, atol=1e-6)

    def test_two_points_mixed_selection_3d(self, pcs_robot):
        """Test two points with mixed selection for 3D robot."""
        s_ps = jnp.array([0.1, 0.2])
        # Point 0: position only, Point 1: all
        task_selector = jnp.array(
            [
                [False, False, False, True, True, True],
                [True, True, True, True, True, True],
            ]
        )
        op_space = OperationalSpaceDynamics(
            robot=pcs_robot,
            s_ps=s_ps,
            task_selector=task_selector,
            rotation_representation=RotationRepresentation.ROTATION_VECTOR,
        )

        assert op_space.n_operational_space == 9  # 3 + 6

        q = jnp.zeros(pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (12,)  # Full: 2 × 6
        assert coords.shape == (9,)  # Selected: 3 + 6

    def test_three_points_planar(self, planar_pcs_robot):
        """Test three points for planar robot."""
        s_ps = jnp.array([0.05, 0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        assert op_space.n_points == 3
        assert op_space.n_operational_space == 9  # 3 × 3

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)

        poses = op_space.operational_space_poses(q)
        coords = op_space.operational_space_coordinates(q)

        assert poses.shape == (9,)
        assert coords.shape == (9,)

    def test_pose_error_multiple_points(self, planar_pcs_robot):
        """Test pose error computation for multiple points."""
        s_ps = jnp.array([0.1, 0.2])
        op_space = OperationalSpaceDynamics(robot=planar_pcs_robot, s_ps=s_ps)

        q = jnp.linspace(-0.1, 0.1, planar_pcs_robot.num_dofs)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, -0.03, 0.2, 0.04, 0.01])

        error = op_space.compute_pose_error(x, x_des)
        task_error = op_space.compute_task_pose_error(x, x_des)

        assert error.shape == (6,)
        assert task_error.shape == (6,)
        assert_allclose(error, task_error, atol=1e-10)  # Full selection


# -----------------------
# JIT and vmap tests for new methods
# -----------------------


class TestJITCompatibilityExtended:
    """Extended JIT compatibility tests for new methods."""

    def test_jit_operational_space_poses(self, pendulum_robot):
        """Test JIT compatibility of operational_space_poses."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)

        # Methods are already JIT-decorated
        poses1 = op_space.operational_space_poses(q)
        poses2 = op_space.operational_space_poses(q)

        assert poses1.shape == (3,)
        assert_allclose(poses1, poses2, rtol=0, atol=0)

    def test_jit_compute_task_pose_error(self, pendulum_robot):
        """Test JIT compatibility of compute_task_pose_error."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        task_selector = jnp.array([False, True, True])  # position only
        op_space = OperationalSpaceDynamics(
            robot=pendulum_robot, s_ps=s_ps, task_selector=task_selector
        )

        q = jnp.linspace(-0.3, 0.4, pendulum_robot.num_links)
        x = op_space.operational_space_poses(q)
        x_des = x + jnp.array([0.1, 0.05, 0.02])

        error1 = op_space.compute_task_pose_error(x, x_des)
        error2 = op_space.compute_task_pose_error(x, x_des)

        assert error1.shape == (2,)
        assert_allclose(error1, error2, rtol=0, atol=0)

    def test_vmap_operational_space_poses(self, pendulum_robot):
        """Test vmapping over operational_space_poses."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        batch_size = 5
        qs = jnp.tile(
            jnp.linspace(-0.3, 0.4, pendulum_robot.num_links), (batch_size, 1)
        )

        vmapped_poses = jax.vmap(op_space.operational_space_poses)
        poses = vmapped_poses(qs)

        assert poses.shape == (batch_size, 3)

    def test_vmap_compute_pose_error(self, pendulum_robot):
        """Test vmapping over compute_pose_error."""
        s_ps = jnp.array([float(pendulum_robot.total_length)])
        op_space = OperationalSpaceDynamics(robot=pendulum_robot, s_ps=s_ps)

        batch_size = 5
        qs = jnp.tile(
            jnp.linspace(-0.3, 0.4, pendulum_robot.num_links), (batch_size, 1)
        )
        xs = jax.vmap(op_space.operational_space_poses)(qs)
        xs_des = xs + 0.1

        vmapped_error = jax.vmap(op_space.compute_pose_error)
        errors = vmapped_error(xs, xs_des)

        assert errors.shape == (batch_size, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
