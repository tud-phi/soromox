# ruff: noqa: E402
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # double precision

from jax import grad, jacfwd, random
from jax import numpy as jnp
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose

from soromox.systems import PlanarHSA, PlanarHSAParams, PlanarHSAStructure
from soromox.utils.geometry import poses
from soromox.utils.lie_algebra import se2

HSA_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robot_parameters"
    / "planar_hsa"
    / "fpu_control.npz"
)
typed_params = PlanarHSAParams.from_npz(HSA_PARAMS_PATH)
num_segments = 1
num_rods_per_segment = 2


def _create_robot():
    """Helper to create a robot instance."""
    return PlanarHSA(
        params=typed_params,
        structure=PlanarHSAStructure(),
    )


def _sample_configuration(rng, num_segments):
    """Sample a random configuration."""
    rng, subrng1, subrng2, subrng3 = random.split(rng, 4)
    kappa_b = random.uniform(
        subrng1,
        (num_segments,),
        minval=-jnp.pi / jnp.mean(typed_params.length),
        maxval=jnp.pi / jnp.mean(typed_params.length),
    )
    sigma_sh = random.uniform(subrng2, (num_segments,), minval=-0.2, maxval=0.2)
    sigma_a = random.uniform(subrng3, (num_segments,), minval=0.0, maxval=0.5)
    q = jnp.concatenate((kappa_b, sigma_sh, sigma_a), axis=0)
    return rng, q


def test_end_effector_kinematics(seed: int = 0):
    print("Testing end effector kinematics...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(10):
        rng, q = _sample_configuration(rng, num_segments)

        print("q = ", q)

        # forward kinematics
        chiee = robot.forward_kinematics_end_effector(q)
        # inverse kinematics
        q_rec = robot.inverse_kinematics(chiee)

        if not jnp.allclose(q, q_rec, atol=1e-6):
            print("q = ", q)
            print("q_rec = ", q_rec)
            raise ValueError("q != q_rec")


def test_inverse_kinematics_is_reverse_mode_finite_when_straight():
    """Cover the analytic straight branch where ``cos(theta) - 1`` is zero."""
    robot = _create_robot()
    q = jnp.zeros(robot.num_dofs)
    chiee = robot.forward_kinematics_end_effector(q)

    recovered = robot.inverse_kinematics(chiee)
    jacobian = jax.jacrev(robot.inverse_kinematics)(chiee)

    assert jnp.isfinite(recovered).all()
    assert jnp.isfinite(jacobian).all()


def test_inverse_kinematics_retains_translation_at_tiny_nonzero_rotation():
    """Tiny bends must not lose translation through ``1 - cos(theta)``."""
    robot = _create_robot()
    theta = jnp.asarray(2.0 * robot.global_eps, dtype=jnp.float64)
    expected_vxi = jnp.array([theta / robot.length, 0.2, 1.1], dtype=jnp.float64)
    T_pe_to_de = expm(se2.hat(expected_vxi * robot.length))

    hp = robot.platform_dimension[0, 1]
    T_b_to_pe = poses.planar_pose_to_transform(
        jnp.array([0.0, 0.0, robot.proximal_cap_length[0]])
    )
    T_de_to_ee = poses.planar_pose_to_transform(
        jnp.array(
            [
                robot.end_effector_offset[0],
                robot.end_effector_offset[1],
                robot.distal_cap_length[0] + hp + robot.end_effector_offset[2],
            ]
        )
    )
    chiee = poses.planar_pose_from_transform(
        T_b_to_pe @ T_pe_to_de @ T_de_to_ee, eps=robot.global_eps
    )

    recovered_q = robot.inverse_kinematics(chiee)
    recovered_vxi = robot.B_xi @ recovered_q + robot.xi_ref

    assert_allclose(recovered_vxi, expected_vxi, rtol=1e-10, atol=1e-11)


def test_planar_hsa_archive_uses_descriptive_parameter_names():
    robot = _create_robot()

    with np.load(HSA_PARAMS_PATH) as data:
        assert str(data["schema_version"]) == "planar_hsa_params_v2"
        assert str(data["robot_name"]) == "PlanarHSA"
        assert "xi_ref" not in data
        assert jnp.allclose(typed_params.reference_strain, data["reference_strain"])
        assert jnp.allclose(typed_params.phi_max, data["phi_max"])
    assert jnp.allclose(robot.phi_max, typed_params.phi_max)


def test_planar_hsa_npz_uses_environment_defaults_when_fields_are_absent(tmp_path):
    with np.load(HSA_PARAMS_PATH) as data:
        values = {
            name: data[name]
            for name in data.files
            if name not in {"base_pose", "gravity"}
        }
    path = tmp_path / "hsa_without_environment.npz"
    np.savez(path, **values)

    params = PlanarHSAParams.from_npz(path)

    assert jnp.allclose(params.base_pose, jnp.array([jnp.pi / 2, 0.0, 0.0]))
    assert jnp.allclose(params.gravity, jnp.array([0.0, -9.81]))


def test_jacobian_matches_forward_kinematics_derivative(seed: int = 0):
    """
    Test that the numerical Jacobian matches the autograd Jacobian of forward kinematics.
    """
    print("Testing analytic Jacobian...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)

        # Test at multiple points along the robot
        rng, subrng = random.split(rng)
        s_values = random.uniform(subrng, (5,), minval=0.0, maxval=robot.length)

        for s in s_values:
            # Numerical Jacobian
            J_analytic = robot.jacobian(q, s)

            # Autograd Jacobian (computed from forward_kinematics)
            J_autograd = jacfwd(robot.forward_kinematics, argnums=0)(q, s)

            if not jnp.allclose(J_analytic, J_autograd, atol=1e-6):
                print(f"s = {s}")
                print(f"J_analytic =\n{J_analytic}")
                print(f"J_autograd =\n{J_autograd}")
                print(f"diff =\n{J_analytic - J_autograd}")
                raise ValueError(
                    "Analytic Jacobian does not match forward-kinematics derivative"
                )


def test_jacobian_matches_base_system_api(seed: int = 0):
    """
    Test the standard SoftRobot Jacobian API.
    """
    print("Testing jacobian wrapper...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    rng, q = _sample_configuration(rng, num_segments)
    rng, subrng = random.split(rng)
    s = random.uniform(subrng, (), minval=0.0, maxval=robot.length)

    J1 = robot.jacobian(q, s)
    J2 = robot._jacobian(q, s)

    assert jnp.allclose(J1, J2, atol=1e-12), "jacobian should use the analytic hook"


def test_jacobian_tips(seed: int = 0):
    """
    Test that jacobian_tips matches virtual-backbone Jacobians at segment tips.
    """
    print("Testing jacobian_tips...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    rng, q = _sample_configuration(rng, num_segments)

    s_tips = robot.L_cum[1:]
    J_tips = robot.jacobian_tips(q)
    J_expected = robot.jacobian_batched(q, s_tips)

    assert jnp.allclose(J_tips, J_expected, atol=1e-12), (
        "jacobian_tips should match jacobian_batched at segment tips"
    )


def test_jacobian_and_time_derivative(seed: int = 0):
    """
    Test the Jacobian and its time derivative.
    """
    print("Testing analytic Jacobian time derivative...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)
        rng, subrng = random.split(rng)
        qd = random.normal(subrng, shape=q.shape) * 0.1

        rng, subrng = random.split(rng)
        s = random.uniform(
            subrng, (), minval=0.1 * robot.length, maxval=0.9 * robot.length
        )

        J, Jd = robot.jacobian_and_time_derivative(q, qd, s)

        # Verify J matches the Jacobian method
        J_direct = robot.jacobian(q, s)
        assert jnp.allclose(J, J_direct, atol=1e-10), (
            "J from jacobian_and_time_derivative should match jacobian"
        )

        # Verify Jd by numerical differentiation
        eps_t = 1e-6
        q_plus = q + eps_t * qd
        J_plus = robot.jacobian(q_plus, s)
        Jd_numerical = (J_plus - J) / eps_t

        if not jnp.allclose(Jd, Jd_numerical, atol=1e-4):
            print(f"Jd =\n{Jd}")
            print(f"Jd_numerical =\n{Jd_numerical}")
            print(f"diff =\n{jnp.abs(Jd - Jd_numerical)}")
            raise ValueError("Jd does not match numerical derivative")


def test_jacobian_and_time_derivative_matches_analytic_hook(seed: int = 0):
    """
    Test that the standard derivative API uses the analytic hook.
    """
    print("Testing jacobian_and_time_derivative wrapper...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    rng, q = _sample_configuration(rng, num_segments)
    rng, subrng = random.split(rng)
    qd = random.normal(subrng, shape=q.shape) * 0.1
    rng, subrng = random.split(rng)
    s = random.uniform(subrng, (), minval=0.0, maxval=robot.length)

    J1, Jd1 = robot.jacobian_and_time_derivative(q, qd, s)
    J2, Jd2 = robot._jacobian_and_time_derivative(q, qd, s)

    assert jnp.allclose(J1, J2, atol=1e-12), (
        "jacobian_and_time_derivative J should match"
    )
    assert jnp.allclose(Jd1, Jd2, atol=1e-12), (
        "jacobian_and_time_derivative Jd should match"
    )


def test_gravitational_energy(seed: int = 0):
    """
    Test that the gravitational energy is consistent with the gravitational force.
    The gravitational force G should be the gradient of U_g with respect to q.
    """
    print("Testing gravitational_energy...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)

        # Compute gravitational force
        G = robot.gravitational_force(q)

        # The gravitational force should be the gradient of U_g
        G_from_energy = grad(robot.gravitational_energy)(q)

        if not jnp.allclose(G, G_from_energy, atol=1e-6):
            print(f"q = {q}")
            print(f"G = {G}")
            print(f"G_from_energy = {G_from_energy}")
            print(f"diff = {jnp.abs(G - G_from_energy)}")
            raise ValueError(
                "Gravitational force should be gradient of gravitational energy"
            )


def test_kinetic_energy(seed: int = 0):
    """
    Test the kinetic energy computation.
    """
    print("Testing kinetic_energy...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)
        rng, subrng = random.split(rng)
        qd = random.normal(subrng, shape=q.shape) * 0.1

        # Compute kinetic energy
        T = robot.kinetic_energy(q, qd)

        # Verify T = 0.5 * qd^T @ B @ qd
        B = robot.inertia_matrix(q)
        T_expected = 0.5 * qd.T @ B @ qd

        assert jnp.allclose(T, T_expected, atol=1e-10), "Kinetic energy mismatch"

        # Kinetic energy should be non-negative
        assert T >= 0.0, "Kinetic energy should be non-negative"

        # Zero velocity should give zero kinetic energy
        T_zero = robot.kinetic_energy(q, jnp.zeros_like(qd))
        assert jnp.abs(T_zero) < 1e-15, (
            "Kinetic energy with zero velocity should be zero"
        )


def test_elastic_energy(seed: int = 0):
    """
    Test the elastic energy computation.
    """
    print("Testing elastic_energy...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)

        # Compute elastic energy
        U_K = robot.elastic_energy(q)

        # Verify U_K = 0.5 * q^T @ K @ q
        K = robot.stiffness_matrix()
        U_K_expected = 0.5 * q.T @ K @ q

        assert jnp.allclose(U_K, U_K_expected, atol=1e-10), "Elastic energy mismatch"

        # Elastic energy should be non-negative (assuming positive semi-definite stiffness)
        assert U_K >= 0.0, "Elastic energy should be non-negative"

        # Zero configuration should give zero elastic energy
        U_K_zero = robot.elastic_energy(jnp.zeros_like(q))
        assert jnp.abs(U_K_zero) < 1e-15, "Elastic energy at zero config should be zero"


def test_elastic_force_matches_elastic_energy_gradient(seed: int = 0):
    """
    Test that elastic_force is the conservative gradient of elastic_energy.
    """
    print("Testing elastic_force gradient...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)

        force = robot.elastic_force(q)
        force_from_primal = grad(lambda q_: robot._elastic_energy(q_))(q)
        force_from_public = grad(lambda q_: robot.elastic_energy(q_))(q)

        assert jnp.allclose(force, force_from_primal, atol=1e-10), (
            "elastic_force should match the primal elastic-energy gradient"
        )
        assert jnp.allclose(force, force_from_public, atol=1e-10), (
            "elastic_energy custom JVP should use elastic_force"
        )


def test_stiffness_matrix(seed: int = 0):
    """
    Test the stiffness matrix computation.
    """
    print("Testing stiffness_matrix...")
    robot = _create_robot()

    K = robot.stiffness_matrix()

    # Stiffness matrix should be symmetric
    assert jnp.allclose(K, K.T, atol=1e-10), "Stiffness matrix should be symmetric"

    # Stiffness matrix should be positive semi-definite
    eigenvalues = jnp.linalg.eigvalsh(K)
    assert jnp.all(eigenvalues >= -1e-10), (
        "Stiffness matrix should be positive semi-definite"
    )


def test_potential_energy(seed: int = 0):
    """
    Test that potential energy is the sum of elastic and gravitational energy.
    """
    print("Testing potential_energy...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)

        U = robot.potential_energy(q)
        U_K = robot.elastic_energy(q)
        U_G = robot.gravitational_energy(q)

        assert jnp.allclose(U, U_K + U_G, atol=1e-10), (
            "Potential energy should be U_K + U_G"
        )


def test_total_energy(seed: int = 0):
    """
    Test that total energy is the sum of kinetic and potential energy.
    """
    print("Testing total_energy...")
    robot = _create_robot()

    rng = random.PRNGKey(seed)
    for _ in range(5):
        rng, q = _sample_configuration(rng, num_segments)
        rng, subrng = random.split(rng)
        qd = random.normal(subrng, shape=q.shape) * 0.1

        E = robot.total_energy(q, qd)
        T = robot.kinetic_energy(q, qd)
        U = robot.potential_energy(q)

        assert jnp.allclose(E, T + U, atol=1e-10), "Total energy should be T + U"


if __name__ == "__main__":
    test_end_effector_kinematics()
    test_jacobian_matches_forward_kinematics_derivative()
    test_jacobian_matches_base_system_api()
    test_jacobian_and_time_derivative()
    test_jacobian_and_time_derivative_matches_analytic_hook()
    test_gravitational_energy()
    test_kinetic_energy()
    test_elastic_energy()
    test_stiffness_matrix()
    test_potential_energy()
    test_total_energy()
    print("\nAll tests passed!")
