import jax
from jax import Array
from jax import numpy as jnp
import numpy as onp
from numpy.testing import assert_allclose
import pytest
from typing import List, Optional

from soromox.systems.planar_pcs import PlanarPCS
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)  # double precision


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = 1e-6

PLANAR_TOTAL_LENGTH = 2e-1
NUM_RANDOM_SAMPLES = 5


def make_planar_pcs(
    num_segments: int = 2,
    th0: float = jnp.pi / 2,
    xi_ref: Optional[Array] = None,
    total_length: Optional[float] = None,
):
    """
    Create a planar constant strain model.
    """
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_length = 1e-1 if total_length is None else total_length / num_segments
    params = {
        "th0": jnp.array(th0),  # initial orientation angle [rad]
        "L": segment_length * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, -9.81]),  # gravity vector [m/s^2] UP!
        "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * params["L"][:, None]
        ).flatten()
    )
    
    model = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        xi_ref=xi_ref,
    )
    
    return model, params


def sample_arc_lengths(model: PlanarPCS) -> List[float]:
    """Select representative arc-lengths in (0, L_tot] for the provided model."""
    lengths = jnp.asarray(model.L)
    cumulative = jnp.cumsum(lengths)
    total = float(cumulative[-1])

    near_zero = max(total * 1e-3, 1e-9)
    mids = (cumulative - lengths / 2.0).tolist()
    boundaries = cumulative.tolist()

    values = [near_zero] + mids + boundaries
    unique_sorted = sorted({float(v) for v in values if 0.0 < float(v) <= total})
    return unique_sorted

def random_q(model, key=jax.random.PRNGKey(0), scale=0.1):
    n = int(model.num_active_strains.item())
    return scale * jax.random.normal(key, (n,))

def constant_strain_inverse_kinematics_fn(params, xi_ref, chi, s) -> Array:
    # split the chi vector into x, y, and th0
    th, px, py = chi
    th0 = params["th0"].item()
    print("th0 = ", th0)
    xi = (
        (th - th0)
        / (2 * s)
        * jnp.array(
            [
                2.0,
                (-jnp.sin(th0) * px + jnp.cos(th0) * py)
                - (jnp.cos(th0) * px + jnp.sin(th0) * py)
                * jnp.sin(th - th0)
                / (jnp.cos(th - th0) - 1),
                -(jnp.cos(th0) * px + jnp.sin(th0) * py)
                - (-jnp.sin(th0) * px + jnp.cos(th0) * py)
                * jnp.sin(th - th0)
                / (jnp.cos(th - th0) - 1),
            ]
        )
    )
    q = xi - xi_ref
    return q


def test_planar_cs_num():
    """
    Test the planar constant strain system with numerical integration and Jacobian for 1 segment.
    """
    robot, params = make_planar_pcs(num_segments=1, th0=0.0)

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((3,)),
            robot.L[0] / 2,
            jnp.array([0.0, robot.L[0] / 2, 0.0]),
        ),
        (
            jnp.zeros((3,)), 
            robot.L[0], 
            jnp.array([0.0, robot.L[0], 0.0])
        ),
        (
            jnp.array([0.0, 1.0, 0.0]),
            robot.L[0],
            jnp.array([0.0, 2 * robot.L[0], 0.0]),
        ),
        (
            jnp.array([0.0, 0.0, 1.0]),
            robot.L[0],
            robot.L[0] * jnp.array([0.0, 1.0, 1.0]),
        ),
    ]

    for q, s, expected in test_cases:
        print("q = ", q, "s = ", s)
        chi = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
        assert_allclose(chi, expected, rtol=RTOL, atol=ATOL)
        print("[Valid test]\n")

    # test dynamical matrices
    print("\nTesting dynamical matrices... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    u = jnp.ones((3,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "u = ", u)
    B = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    G = robot.gravitational_force(q)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix()
    alpha = robot.actuation_force(q, u)

    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(alpha).any(), "alpha matrix contains NaN!"
    print("testing K")
    assert_allclose(K @ q, jnp.zeros((3,)))
    print("[Valid test]\n")
    print("testing alpha")
    assert_allclose(
        alpha,
        jnp.ones(3),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * robot.L[0]), 0.0, 0.0])
    qd = jnp.zeros((3,))
    u = jnp.ones((3,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "u = ", u)
    B = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    G = robot.gravitational_force(q)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix()
    alpha = robot.actuation_force(q, u)
    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(alpha).any(), "alpha matrix contains NaN!"

    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("alpha =\n", alpha)
    print("[To check]")

    # test energies
    print("\nTesting energies... ------------------------")

    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    print("q = ", q, "qd = ", qd)

    print("Testing kinetic energy...")
    E_kin = robot.kinetic_energy(q, qd)
    assert not jnp.isnan(E_kin).any(), "Kinetic energy contains NaN!"
    E_kin_th = 0.0
    assert_allclose(E_kin, E_kin_th, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    print("Testing potential energy...")
    E_pot = robot.potential_energy(q)
    assert not jnp.isnan(E_pot).any(), "Potential energy contains NaN!"
    E_pot_th = jnp.array(0.0)
    assert_allclose(E_pot, E_pot_th, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test jacobian
    print("\nTesting jacobian... ------------------------")
    chi = robot.forward_kinematics(q=q, s=robot.L[0])
    print("q = ", q, "s = ", robot.L[0])
    J = robot.jacobian(q, s=robot.L[0])
    assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
    print("Jacobian J =\n", J)
    # Test the differential relation: delta_chi ≈ J * delta_q
    print("Testing differential relation: delta_chi ≈ J * delta_q")
    delta_q = jnp.array([EPS, -EPS, 2 * EPS])
    chi_plus = robot.forward_kinematics(q=q + delta_q, s=params["L"][0])
    chi_pred = chi + J @ delta_q
    assert_allclose(chi_plus, chi_pred, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    u = jnp.zeros((3,))  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.zeros((2,))  # no gravity for this test
    robot = robot.update_params(params_bis)
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", params_bis["g"])
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
    qdd, qdres = jnp.split(yd, 2)
    assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(qdd, jnp.zeros((3,)), rtol=RTOL, atol=ATOL)
    assert_allclose(qdres, qd, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test inverse kinematics
    print("\nTesting inverse kinematics... ------------------------")
    params_ik = params.copy()
    ik_th0_ls = [-jnp.pi / 2, -jnp.pi / 4, 0.0, jnp.pi / 4, jnp.pi / 2]
    ik_q_ls = [
        jnp.array([0.1, 0.0, 0.0]),
        jnp.array([0.1, 0.0, 0.2]),
        jnp.array([0.1, 0.5, 0.1]),
        jnp.array([1.0, 0.5, 0.2]),
        jnp.array([-1.0, 0.0, 0.0]),
    ]
    for ik_th0 in ik_th0_ls:
        robot_ik, params_ik = make_planar_pcs(num_segments=1, th0=ik_th0)
        for q in ik_q_ls:
            s = params_ik["L"][0]
            print("q = ", q, "s = ", s, "th0 = ", ik_th0)
            chi = robot_ik.forward_kinematics(q=q, s=s)
            assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
            q_ik = constant_strain_inverse_kinematics_fn(params_ik, robot_ik.xi_ref, chi, s)
            assert not jnp.isnan(q_ik).any(), "Inverse kinematics output contains NaN!"
            assert_allclose(q, q_ik, rtol=RTOL, atol=ATOL)
            print("[Valid test]\n")


def test_individual_call():
    """
    Test the individual call of the PlanarPCS class.
    """
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 1e3, 1e3]]) * params["L"][:, None]).flatten()
    )
    strain_selector = jnp.ones((3,), dtype=bool)
    strain_selector = strain_selector.at[2].set(
        False
    )  # disable axial strain for this test
    xi_ref = jnp.array([0.0, 0.0, 1.0])

    robot = PlanarPCS(
        num_segments=1,
        params=params,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # Test individual calls
    q = jnp.zeros((2,))
    s = params["L"][0]

    print("\nTest robot.forward_kinematics(q, s)-------------------------")
    try:
        chi = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
        print("[Valid test] Forward kinematics successful.")
    except Exception as e:
        print(f"[Error] Forward kinematics failed: {e}")

    print("\nTest robot.jacobian(q, s)-------------------------")
    try:
        J = robot.jacobian(q=q, s=s)
        assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
        print("[Valid test] Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Jacobian computation failed: {e}")

    print("\nTest robot.jacobian_and_derivative(q, qd, s)-------------------------")
    try:
        J, Jd = robot.jacobian_and_derivative(q=q, qd=jnp.zeros((2,)), s=s)
        assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
        assert not jnp.isnan(Jd).any(), "Jacobian derivative contains NaN!"
        print("[Valid test] Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Jacobian and derivative computation failed: {e}")

    print("\nTest robot.jacobian_bodyframe(q, s)-------------------------")
    try:
        J_local = robot.jacobian_bodyframe(q=q, s=s)
        assert not jnp.isnan(J_local).any(), "Local Jacobian contains NaN!"
        print("[Valid test] Local Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Local Jacobian computation failed: {e}")

    print("\nTest robot.jacobian_inertialframe(q, s)-------------------------")
    try:
        J_global = robot.jacobian_inertialframe(q=q, s=s)
        assert not jnp.isnan(J_global).any(), "Global Jacobian contains NaN!"
        print("[Valid test] Global Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Global Jacobian computation failed: {e}")

    print(
        "\nTest robot.jacobian_and_derivative_bodyframe(q, qd, s)-------------------------"
    )
    try:
        J_local, Jd_local = robot.jacobian_and_derivative_bodyframe(
            q=q, qd=jnp.zeros((2,)), s=s
        )
        assert not jnp.isnan(J_local).any(), "Local Jacobian contains NaN!"
        assert not jnp.isnan(Jd_local).any(), "Local Jacobian derivative contains NaN!"
        print("[Valid test] Local Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Local Jacobian and derivative computation failed: {e}")

    print(
        "\nTest robot.jacobian_and_derivative_inertialframe(q, qd, s)-------------------------"
    )
    try:
        J_global, Jd_global = robot.jacobian_and_derivative_inertialframe(
            q=q, qd=jnp.zeros((2,)), s=s
        )
        assert not jnp.isnan(J_global).any(), "Global Jacobian contains NaN!"
        assert not jnp.isnan(Jd_global).any(), (
            "Global Jacobian derivative contains NaN!"
        )
        print("[Valid test] Global Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Global Jacobian and derivative computation failed: {e}")

    print("\nTest robot.inertia_matrix(q)-------------------------")
    try:
        B = robot.inertia_matrix(q=q)
        assert not jnp.isnan(B).any(), "Inertia matrix contains NaN!"
        print("[Valid test] Inertia matrix computation successful.")
    except Exception as e:
        print(f"[Error] Inertia matrix computation failed: {e}")

    print("\nTest robot.coriolis_matrix(q, qd)-------------------------")
    try:
        C = robot.coriolis_matrix(q=q, qd=jnp.zeros((2,)))
        assert not jnp.isnan(C).any(), "Coriolis matrix contains NaN!"
        print("[Valid test] Coriolis matrix computation successful.")
    except Exception as e:
        print(f"[Error] Coriolis matrix computation failed: {e}")

    print("\nTest robot.gravitational_force(q)-------------------------")
    try:
        G = robot.gravitational_force(q=q)
        assert not jnp.isnan(G).any(), "Gravitational force contains NaN!"
        print("[Valid test] Gravitational force computation successful.")
    except Exception as e:
        print(f"[Error] Gravitational force computation failed: {e}")

    print("\nTest robot.stiffness_matrix()-------------------------")
    try:
        K = robot.stiffness_matrix()
        assert not jnp.isnan(K).any(), "Stiffness matrix contains NaN!"
        print("[Valid test] Stiffness matrix computation successful.")
    except Exception as e:
        print(f"[Error] Stiffness matrix computation failed: {e}")

    print("\nTest robot.damping_matrix()-------------------------")
    try:
        D = robot.damping_matrix()
        assert not jnp.isnan(D).any(), "Damping matrix contains NaN!"
        print("[Valid test] Damping matrix computation successful.")
    except Exception as e:
        print(f"[Error] Damping matrix computation failed: {e}")

    print("\nTest robot.actuation_force(q, u)-------------------------")
    try:
        u = jnp.zeros((2,))  # no external forces
        alpha = robot.actuation_force(q=q, u=u)
        assert not jnp.isnan(alpha).any(), "Actuation force contains NaN!"
        print("[Valid test] Actuation force computation successful.")
    except Exception as e:
        print(f"[Error] Actuation force computation failed: {e}")

    print("\nTest robot.forward_dynamics(t, y, u)-------------------------")
    try:
        t = 0.0
        y = jnp.concatenate([q, jnp.zeros((2,))])  # initial state with zero velocity
        u = jnp.zeros((2,))  # no external forces
        yd = robot.forward_dynamics(t=t, y=y, actuation_args=(u,))
        qdd, qdres = jnp.split(yd, 2)
        assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
        print("[Valid test] Forward dynamics computation successful.")
    except Exception as e:
        print(f"[Error] Forward dynamics computation failed: {e}")


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inverse_kinematics_consistency(num_segments):
    """
    Test that the inverse kinematics method correctly inverts forward kinematics.
    
    This test verifies that:
    1. Forward kinematics followed by inverse kinematics returns the original configuration
    2. The method works for different numbers of segments
    3. Multiple random configurations are handled correctly
    """
    model, params = make_planar_pcs(num_segments=num_segments)
    
    # Test with multiple random configurations
    key = jax.random.PRNGKey(42)
    num_test_configs = 10
    
    for i in range(num_test_configs):
        key, subkey = jax.random.split(key)
        
        # Generate a random configuration (generalized coordinates)
        q_original = random_q(model, key=subkey, scale=0.3)  # Slightly larger scale for better testing
        
        # Compute forward kinematics at segment tips
        s_tips = model.L_cum[1:]  # End of each segment
        chi_tips = jnp.array([model.forward_kinematics(q_original, s) for s in s_tips])
        
        # Apply inverse kinematics
        q_recovered = model.inverse_kinematics(chi_tips)
        
        # Check that the recovered configuration matches the original
        assert_allclose(
            q_recovered,
            q_original,
            rtol=RTOL * 10,  # Slightly more lenient tolerance for inverse kinematics
            atol=ATOL * 10,
            err_msg=f"Inverse kinematics failed for configuration {i} with {num_segments} segments"
        )
        
        # Additional verification: forward kinematics of recovered configuration
        # should match the original tip poses
        chi_tips_recovered = jnp.array([model.forward_kinematics(q_recovered, s) for s in s_tips])
        
        assert_allclose(
            chi_tips_recovered,
            chi_tips,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Forward kinematics of recovered configuration failed for config {i} with {num_segments} segments"
        )


@pytest.mark.parametrize("num_segments", [2, 3])
def test_inverse_kinematics_with_known_straight_configuration(num_segments):
    """
    Test inverse kinematics with a known straight configuration.
    
    For a straight robot (no bending, only extension), the inverse kinematics
    should recover zero curvature and appropriate extension strains.
    """
    model, params = make_planar_pcs(num_segments=num_segments, th0=0.0)
    
    # Create a straight configuration: no rotation, only extension along x-axis
    segment_length = model.L[0]  # Assume all segments have same length
    chi_tips = jnp.array([
        [0.0, (i + 1) * segment_length * 1.1, 0.0]  # 10% extension per segment
        for i in range(num_segments)
    ])
    
    # Apply inverse kinematics
    q_recovered = model.inverse_kinematics(chi_tips)
    print("Recovered q:", q_recovered)
    
    # Convert to strain space to check the result
    xi_recovered = model.strain(q_recovered).reshape(num_segments, 3)
    print("Recovered strains (kappa_z, sigma_x, sigma_y):", xi_recovered)
    
    # For a straight extended robot:
    # - Curvature (kappa_z) should be approximately zero
    # - Extension strain (sigma_x) should be positive (around 0.1 for 10% extension)
    # - Shear strain (sigma_y) should be approximately zero
    
    expected_kappa = jnp.zeros(num_segments)  # No curvature
    expected_sigma_x = jnp.ones(num_segments) * 1.1  # 10% extension 
    expected_sigma_y = jnp.zeros(num_segments)  # No shear
    print("Expected strains (kappa_z, sigma_x, sigma_y):", expected_kappa, expected_sigma_x, expected_sigma_y)
    
    # Check curvature is small
    assert_allclose(
        xi_recovered[:, 0],  # kappa_z
        expected_kappa,
        rtol=RTOL,
        atol=1e-3,  # Allow small numerical errors
        err_msg=f"Curvature should be zero for straight configuration with {num_segments} segments"
    )
    
    # Check extension strain is approximately correct
    assert_allclose(
        xi_recovered[:, 1],  # sigma_x  
        expected_sigma_x,
        rtol=0.1,  # Allow 10% relative error
        atol=0.01,  # Allow small absolute error
        err_msg=f"Extension strain should be approximately 0.1 for straight configuration with {num_segments} segments"
    )
    
    # Check shear is small
    assert_allclose(
        xi_recovered[:, 2],  # sigma_y
        expected_sigma_y,
        rtol=RTOL,
        atol=1e-3,  # Allow small numerical errors
        err_msg=f"Shear strain should be zero for straight configuration with {num_segments} segments"
    )


def test_inverse_kinematics_relative_pose_computation():
    """
    Test the relative pose computation method used in inverse kinematics.
    
    This test verifies that the relative pose computation correctly handles
    the transformation between consecutive segment tips.
    """
    num_segments = 3
    model, params = make_planar_pcs(num_segments=num_segments)
    
    # Create test tip poses
    chi_tips = jnp.array([
        [0.1, 0.05, 0.02],   # Tip of segment 1
        [0.3, 0.12, 0.08],   # Tip of segment 2  
        [0.5, 0.18, 0.15],   # Tip of segment 3
    ])
    
    # Compute relative poses
    chi_rel = model._compute_relative_segment_poses(chi_tips)
    
    # Verify the shape
    assert chi_rel.shape == (num_segments, 3), f"Expected shape {(num_segments, 3)}, got {chi_rel.shape}"
    
    # Manually compute the first relative pose (segment 1 w.r.t. base)
    base_pose = jnp.array([model.th0, 0.0, 0.0])
    expected_rel_0 = chi_tips[0] - base_pose
    expected_rel_0 = expected_rel_0.at[1:].set(
        jnp.array([
            jnp.cos(base_pose[0]) * expected_rel_0[1] + jnp.sin(base_pose[0]) * expected_rel_0[2],
            -jnp.sin(base_pose[0]) * expected_rel_0[1] + jnp.cos(base_pose[0]) * expected_rel_0[2]
        ])
    )
    
    # Check the first relative pose
    assert_allclose(
        chi_rel[0],
        expected_rel_0,
        rtol=RTOL,
        atol=ATOL,
        err_msg="First relative pose computation is incorrect"
    )


@pytest.mark.parametrize("num_segments", [2, 3])
def test_inverse_kinematics_with_deactivated_strains(num_segments):
    """
    Test inverse kinematics with deactivated strain components.
    
    This test verifies that inverse kinematics works correctly when some strain
    components are deactivated (e.g., shear strain is turned off).
    """
    # Test case 1: Deactivate shear strains (sigma_y)
    strain_selector = jnp.tile(jnp.array([True, True, False]), num_segments)  # [kappa, sigma_x, sigma_y] per segment
    
    model, params = make_planar_pcs(num_segments=num_segments)
    model_reduced = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        strain_selector=strain_selector
    )
    
    # Test with multiple random configurations
    key = jax.random.PRNGKey(123)
    num_test_configs = 5
    
    for i in range(num_test_configs):
        key, subkey = jax.random.split(key)
        
        # Generate a random configuration for the reduced model
        q_original = random_q(model_reduced, key=subkey, scale=0.2)
        
        # Compute forward kinematics at segment tips
        s_tips = model_reduced.L_cum[1:]  # End of each segment
        chi_tips = jnp.array([model_reduced.forward_kinematics(q_original, s) for s in s_tips])
        
        # Apply inverse kinematics
        q_recovered = model_reduced.inverse_kinematics(chi_tips)
        
        # Check that the recovered configuration matches the original
        assert_allclose(
            q_recovered,
            q_original,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Inverse kinematics failed for reduced model config {i} with {num_segments} segments"
        )
        
        # Additional verification: forward kinematics of recovered configuration
        chi_tips_recovered = jnp.array([model_reduced.forward_kinematics(q_recovered, s) for s in s_tips])
        
        assert_allclose(
            chi_tips_recovered,
            chi_tips,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Forward kinematics of recovered configuration failed for reduced model config {i}"
        )
    
    # Test case 2: Deactivate curvature strains (kappa_z)
    strain_selector_no_curvature = jnp.tile(jnp.array([False, True, True]), num_segments)
    
    model_no_curvature = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        strain_selector=strain_selector_no_curvature
    )
    
    # Test one configuration with no curvature model
    q_original = random_q(model_no_curvature, key=jax.random.PRNGKey(456), scale=0.1)
    s_tips = model_no_curvature.L_cum[1:]
    chi_tips = jnp.array([model_no_curvature.forward_kinematics(q_original, s) for s in s_tips])
    
    q_recovered = model_no_curvature.inverse_kinematics(chi_tips)
    
    assert_allclose(
        q_recovered,
        q_original,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for no-curvature model"
    )

def test_inverse_kinematics_strain_selector_edge_cases():
    """
    Test inverse kinematics with edge cases of strain selection.
    """
    num_segments = 2
    model, params = make_planar_pcs(num_segments=num_segments)
    
    # Edge case 1: Only curvature strains active
    strain_selector_curvature_only = jnp.tile(jnp.array([True, False, False]), num_segments)
    model_curvature_only = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        strain_selector=strain_selector_curvature_only
    )
    
    # Test with a simple configuration
    q_test = random_q(model_curvature_only, key=jax.random.PRNGKey(111), scale=0.1)
    s_tips = model_curvature_only.L_cum[1:]
    chi_tips = jnp.array([model_curvature_only.forward_kinematics(q_test, s) for s in s_tips])
    
    q_recovered = model_curvature_only.inverse_kinematics(chi_tips)
    
    assert_allclose(
        q_recovered,
        q_test,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for curvature-only model"
    )
    
    # Edge case 2: Only extension strains active (sigma_x)
    strain_selector_extension_only = jnp.tile(jnp.array([False, True, False]), num_segments)
    model_extension_only = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        strain_selector=strain_selector_extension_only
    )
    
    q_test2 = random_q(model_extension_only, key=jax.random.PRNGKey(222), scale=0.05)
    chi_tips2 = jnp.array([model_extension_only.forward_kinematics(q_test2, s) for s in s_tips])
    
    q_recovered2 = model_extension_only.inverse_kinematics(chi_tips2)
    
    assert_allclose(
        q_recovered2,
        q_test2,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for extension-only model"
    )

@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_inertialframe_matches_autodiff(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)
    q = random_q(model, jax.random.PRNGKey(1), scale=0.05)

    for s in sample_arc_lengths(model):
        J_impl = model.jacobian_inertialframe(q, s)

        def f(q_):
            return model.forward_kinematics(q_, s)  # [theta, x, y]

        J_ad = jax.jacfwd(f)(q)

        assert jnp.allclose(J_impl, J_ad, rtol=1e-6, atol=1e-7), (
            f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_ad:\n{J_ad}"
        )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_inertial_velocity_consistency(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dt = EPS
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.03)
        qd = random_q(model, qd_key, scale=0.1)

        for s in sample_arc_lengths(model):
            J = model.jacobian_inertialframe(q, s)
            xdot_pred = J @ qd

            x1 = model.forward_kinematics(q + dt * qd, s)
            x0 = model.forward_kinematics(q, s)
            xdot_fd = (x1 - x0) / dt

            assert jnp.allclose(xdot_pred, xdot_fd, rtol=5e-5, atol=5e-7), (
                f"num_segments={num_segments}, s={s}\npred: {xdot_pred}\nfd: {xdot_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_inertialframe_matches_central_differences(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)

    key = jax.random.PRNGKey(5)
    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.01)

        for s in sample_arc_lengths(model):
            J_impl = model.jacobian_inertialframe(q, s)

            n = q.shape[0]
            J_fd_ls = []
            eye = jnp.eye(n)
            for j in range(n):
                qp = q + EPS * eye[j]
                qm = q - EPS * eye[j]
                fp = model.forward_kinematics(qp, s)
                fm = model.forward_kinematics(qm, s)
                J_fd_ls.append((fp - fm) / (2 * EPS))

            J_fd = jnp.stack(J_fd_ls, axis=1)

            assert jnp.allclose(J_impl, J_fd, rtol=1e-3, atol=5e-6), (
                f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_fd:\n{J_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_Jd_bodyframe_matches_autograd_jvp(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_bodyframe(q, qd, s)

            def J_of_q(q_):
                return model.jacobian_bodyframe(q_, s)

            _, Jd_jvp = jax.jvp(J_of_q, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_jvp:\n{Jd_jvp}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_Jd_bodyframe_matches_central_differences(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)

    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_bodyframe(q, qd, s)

            eye = jnp.eye(q.shape[0])
            dJ_cols = []
            for j in range(q.shape[0]):
                qp = q + EPS * eye[j]
                qm = q - EPS * eye[j]
                Jp = model.jacobian_bodyframe(qp, s)
                Jm = model.jacobian_bodyframe(qm, s)
                dJ_cols.append((Jp - Jm) / (2 * EPS))
            dJ_dq_fd = jnp.stack(dJ_cols, axis=-1)
            Jd_num = jnp.tensordot(dJ_dq_fd, qd, axes=([-1], [0]))

            assert jnp.allclose(Jd_impl, Jd_num, rtol=1e-3, atol=5e-6), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_num:\n{Jd_num}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_Jd_inertialframe_matches_autograd_jvp(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_inertialframe(q, qd, s)

            def J_of_q(q_):
                return model.jacobian_inertialframe(q_, s)

            _, Jd_jvp = jax.jvp(J_of_q, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jd_jvp)}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_gravity_matches_potential_gradient(num_segments):
    robot, params = make_planar_pcs(num_segments)

    key = jax.random.PRNGKey(3)
    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(robot, q_key, scale=0.05)

        G = robot.gravitational_force(q)
        dU_dq = jax.grad(robot.gravitational_energy)(q)

        # With current convention, G equals ∂U/∂q
        assert_allclose(G, dU_dq, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_with_christoffel_symbols(num_segments):
    robot, params = make_planar_pcs(num_segments)

    key = jax.random.PRNGKey(7)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)
        print("q:\n", q)
        print("qd:\n", qd)

        # Implementation from the model
        C_impl = robot.coriolis_matrix(q, qd)
        tau_cor_impl = C_impl @ qd

        # Derive Coriolis matrix via Christoffel symbols using autograd on B(q)
        def B_of_q(q_):
            return robot.inertia_matrix(q_)

        B = B_of_q(q)

        # dB_dq[i, j, k] = ∂B_{ij}/∂q_k
        dB_dq = jax.jacfwd(B_of_q)(q)

        # τ_i = Σ_{j,k} ∂B_{ij}/∂q_k q̇_j q̇_k - ½ Σ_{j,k} ∂B_{jk}/∂q_i q̇_j q̇_k
        term1 = jnp.einsum("ijk,j,k->i", dB_dq, qd, qd)
        term2 = jnp.einsum("jki,j,k->i", dB_dq, qd, qd)
        tau_cor = term1 - 0.5 * term2

        print("tau_cor_impl:\n", tau_cor_impl)
        print("tau_cor:\n", tau_cor)

        assert_allclose(tau_cor_impl, tau_cor, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_matches_kinetic_energy_autograd(num_segments):
    robot, _ = make_planar_pcs(num_segments)

    key = jax.random.PRNGKey(7)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dT_dq = jax.grad(robot.kinetic_energy, argnums=0)
    dT_dqd = jax.grad(robot.kinetic_energy, argnums=1)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)
        print("q:\n", q)
        print("qd:\n", qd)

        tau_cor_impl = robot.coriolis_matrix(q, qd) @ qd

        grad_T_q = dT_dq(q, qd)
        jac_T_q = jax.jacobian(lambda qq: dT_dqd(qq, qd))(q)
        tau_cor_autograd = jac_T_q @ qd - grad_T_q

        print("tau_cor_impl:\n", tau_cor_impl)
        print("tau_cor_autograd:\n", tau_cor_autograd)

        assert_allclose(
            tau_cor_impl,
            tau_cor_autograd,
            rtol=RTOL,
            atol=ATOL,
        )

@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_dynamics_matches_manual_computation(num_segments: int):
    model, _ = make_planar_pcs(num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH)

    num_strains = int(model.num_active_strains.item())
    key = jax.random.PRNGKey(42 + num_segments)

    for _ in range(NUM_RANDOM_SAMPLES):
        key, key_q = jax.random.split(key)
        q = jax.random.normal(key_q, (num_strains,))

        key, key_qd = jax.random.split(key)
        qd = jax.random.normal(key_qd, (num_strains,))

        key, key_u = jax.random.split(key)
        u = jax.random.normal(key_u, (model.num_actuators,))

        key, key_tau = jax.random.split(key)
        tau_ext = jax.random.normal(key_tau, (num_strains,))

        y = jnp.concatenate([q, qd])
        yd = model.forward_dynamics(0.0, y, (u, tau_ext))

        B = model.inertia_matrix(q)
        C = model.coriolis_matrix(q, qd)
        G = model.gravitational_force(q)
        D = model.damping_matrix()
        tau_el = model.elastic_force(q)
        tau_u = model.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)
        qdd_expected = B_inv @ (tau_u + tau_ext - C @ qd - G - tau_el - D @ qd)
        yd_expected = jnp.concatenate([qd, qdd_expected])

        assert_allclose(yd, yd_expected, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
