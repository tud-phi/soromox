import jax
from jax import Array, jacfwd, jacrev
from jax import numpy as jnp
import numpy as onp
from numpy.testing import assert_allclose
import pytest
from typing import List, Optional

from soromox.systems.planar_pcs import PlanarPCS
from soromox.utils.lie_algebra.se2 import Adjoint_g_SE2, exp_SE2
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)  # double precision


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)

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
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
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


def segment_tip_poses(model: PlanarPCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]

    def fk_at_s(s: Array) -> Array:
        return model.forward_kinematics(q, s)

    return jax.vmap(fk_at_s)(s_vals)


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


def test_planar_constant_strain_call():
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
        (jnp.zeros((3,)), robot.L[0], jnp.array([0.0, robot.L[0], 0.0])),
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
            q_ik = constant_strain_inverse_kinematics_fn(
                params_ik, robot_ik.xi_ref, chi, s
            )
            assert not jnp.isnan(q_ik).any(), "Inverse kinematics output contains NaN!"
            assert_allclose(q, q_ik, rtol=RTOL, atol=ATOL)
            print("[Valid test]\n")


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_tips_matches_pointwise_evaluation(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    rng = jax.random.PRNGKey(777)
    random_cfg = random_q(model, rng, scale=0.05)

    for q in (zero_cfg, random_cfg):
        chi_expected = segment_tip_poses(model, q)
        chi_tips = model.forward_kinematics_tips(q)

        assert_allclose(chi_tips, chi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_batched_matches_pointwise_evaluation(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    rng = jax.random.PRNGKey(888)
    random_cfg = random_q(model, rng, scale=0.05)

    s_values = [0.0] + sample_arc_lengths(model)
    s_ps = jnp.asarray(s_values, dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        chi_batched = model.forward_kinematics_batched(q, s_ps)
        chi_expected = jax.vmap(lambda s: model.forward_kinematics(q, s))(s_ps)

        assert_allclose(chi_batched, chi_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_inverse_kinematics_consistency(num_segments):
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
        q_original = random_q(
            model, key=subkey, scale=0.3
        )  # Slightly larger scale for better testing

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
            err_msg=f"Inverse kinematics failed for configuration {i} with {num_segments} segments",
        )

        # Additional verification: forward kinematics of recovered configuration
        # should match the original tip poses
        chi_tips_recovered = jnp.array(
            [model.forward_kinematics(q_recovered, s) for s in s_tips]
        )

        assert_allclose(
            chi_tips_recovered,
            chi_tips,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Forward kinematics of recovered configuration failed for config {i} with {num_segments} segments",
        )


@pytest.mark.parametrize("num_segments", [2, 3])
def test_inverse_kinematics_straight_configuration(num_segments):
    """
    Test inverse kinematics with a known straight configuration.

    For a straight robot (no bending, only extension), the inverse kinematics
    should recover zero curvature and appropriate extension strains.
    """
    model, params = make_planar_pcs(num_segments=num_segments, th0=0.0)

    # Create a straight configuration: no rotation, only extension along x-axis
    segment_length = model.L[0]  # Assume all segments have same length
    chi_tips = jnp.array(
        [
            [0.0, (i + 1) * segment_length * 1.1, 0.0]  # 10% extension per segment
            for i in range(num_segments)
        ]
    )

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
    print(
        "Expected strains (kappa_z, sigma_x, sigma_y):",
        expected_kappa,
        expected_sigma_x,
        expected_sigma_y,
    )

    # Check curvature is small
    assert_allclose(
        xi_recovered[:, 0],  # kappa_z
        expected_kappa,
        rtol=RTOL,
        atol=1e-3,  # Allow small numerical errors
        err_msg=f"Curvature should be zero for straight configuration with {num_segments} segments",
    )

    # Check extension strain is approximately correct
    assert_allclose(
        xi_recovered[:, 1],  # sigma_x
        expected_sigma_x,
        rtol=0.1,  # Allow 10% relative error
        atol=0.01,  # Allow small absolute error
        err_msg=f"Extension strain should be approximately 0.1 for straight configuration with {num_segments} segments",
    )

    # Check shear is small
    assert_allclose(
        xi_recovered[:, 2],  # sigma_y
        expected_sigma_y,
        rtol=RTOL,
        atol=1e-3,  # Allow small numerical errors
        err_msg=f"Shear strain should be zero for straight configuration with {num_segments} segments",
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
    chi_tips = jnp.array(
        [
            [0.1, 0.05, 0.02],  # Tip of segment 1
            [0.3, 0.12, 0.08],  # Tip of segment 2
            [0.5, 0.18, 0.15],  # Tip of segment 3
        ]
    )

    # Compute relative poses
    chi_rel = model._compute_relative_segment_poses(chi_tips)

    # Verify the shape
    assert chi_rel.shape == (num_segments, 3), (
        f"Expected shape {(num_segments, 3)}, got {chi_rel.shape}"
    )

    # Manually compute the first relative pose (segment 1 w.r.t. base)
    base_pose = jnp.array([model.th0, 0.0, 0.0])
    expected_rel_0 = chi_tips[0] - base_pose
    expected_rel_0 = expected_rel_0.at[1:].set(
        jnp.array(
            [
                jnp.cos(base_pose[0]) * expected_rel_0[1]
                + jnp.sin(base_pose[0]) * expected_rel_0[2],
                -jnp.sin(base_pose[0]) * expected_rel_0[1]
                + jnp.cos(base_pose[0]) * expected_rel_0[2],
            ]
        )
    )

    # Check the first relative pose
    assert_allclose(
        chi_rel[0],
        expected_rel_0,
        rtol=RTOL,
        atol=ATOL,
        err_msg="First relative pose computation is incorrect",
    )


@pytest.mark.parametrize("num_segments", [2, 3])
def test_inverse_kinematics_with_deactivated_strains(num_segments):
    """
    Test inverse kinematics with deactivated strain components.

    This test verifies that inverse kinematics works correctly when some strain
    components are deactivated (e.g., shear strain is turned off).
    """
    # Test case 1: Deactivate shear strains (sigma_y)
    strain_selector = jnp.tile(
        jnp.array([True, True, False]), num_segments
    )  # [kappa, sigma_x, sigma_y] per segment

    model, params = make_planar_pcs(num_segments=num_segments)
    model_reduced = PlanarPCS(
        num_segments=num_segments, params=params, strain_selector=strain_selector
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
        chi_tips = jnp.array(
            [model_reduced.forward_kinematics(q_original, s) for s in s_tips]
        )

        # Apply inverse kinematics
        q_recovered = model_reduced.inverse_kinematics(chi_tips)

        # Check that the recovered configuration matches the original
        assert_allclose(
            q_recovered,
            q_original,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Inverse kinematics failed for reduced model config {i} with {num_segments} segments",
        )

        # Additional verification: forward kinematics of recovered configuration
        chi_tips_recovered = jnp.array(
            [model_reduced.forward_kinematics(q_recovered, s) for s in s_tips]
        )

        assert_allclose(
            chi_tips_recovered,
            chi_tips,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Forward kinematics of recovered configuration failed for reduced model config {i}",
        )

    # Test case 2: Deactivate curvature strains (kappa_z)
    strain_selector_no_curvature = jnp.tile(
        jnp.array([False, True, True]), num_segments
    )

    model_no_curvature = PlanarPCS(
        num_segments=num_segments,
        params=params,
        strain_selector=strain_selector_no_curvature,
    )

    # Test one configuration with no curvature model
    q_original = random_q(model_no_curvature, key=jax.random.PRNGKey(456), scale=0.1)
    s_tips = model_no_curvature.L_cum[1:]
    chi_tips = jnp.array(
        [model_no_curvature.forward_kinematics(q_original, s) for s in s_tips]
    )

    q_recovered = model_no_curvature.inverse_kinematics(chi_tips)

    assert_allclose(
        q_recovered,
        q_original,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for no-curvature model",
    )


def test_inverse_kinematics_strain_selector_edge_cases():
    """
    Test inverse kinematics with edge cases of strain selection.
    """
    num_segments = 2
    model, params = make_planar_pcs(num_segments=num_segments)

    # Edge case 1: Only curvature strains active
    strain_selector_curvature_only = jnp.tile(
        jnp.array([True, False, False]), num_segments
    )
    model_curvature_only = PlanarPCS(
        num_segments=num_segments,
        params=params,
        strain_selector=strain_selector_curvature_only,
    )

    # Test with a simple configuration
    q_test = random_q(model_curvature_only, key=jax.random.PRNGKey(111), scale=0.1)
    s_tips = model_curvature_only.L_cum[1:]
    chi_tips = jnp.array(
        [model_curvature_only.forward_kinematics(q_test, s) for s in s_tips]
    )

    q_recovered = model_curvature_only.inverse_kinematics(chi_tips)

    assert_allclose(
        q_recovered,
        q_test,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for curvature-only model",
    )

    # Edge case 2: Only extension strains active (sigma_x)
    strain_selector_extension_only = jnp.tile(
        jnp.array([False, True, False]), num_segments
    )
    model_extension_only = PlanarPCS(
        num_segments=num_segments,
        params=params,
        strain_selector=strain_selector_extension_only,
    )

    q_test2 = random_q(model_extension_only, key=jax.random.PRNGKey(222), scale=0.05)
    chi_tips2 = jnp.array(
        [model_extension_only.forward_kinematics(q_test2, s) for s in s_tips]
    )

    q_recovered2 = model_extension_only.inverse_kinematics(chi_tips2)

    assert_allclose(
        q_recovered2,
        q_test2,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Inverse kinematics failed for extension-only model",
    )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_local_tips_matches_pointwise_evaluation(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments)
    q = random_q(model, jax.random.PRNGKey(5), scale=0.03)

    J_tips = model._J_local_tips(q)
    s_tips = model.L_cum[1:]

    for idx, s in enumerate(s_tips):
        if s < 1e-3:
            continue

        J_tip_batch = J_tips[idx]
        J_tip_single = model._J_local(q, s)

        assert_allclose(
            J_tip_batch,
            J_tip_single,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"num_segments={num_segments}, s={s}\n"
                f"J_tip_batch:\n{J_tip_batch}\nJ_tip_single:\n{J_tip_single}"
            ),
        )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_local_batched_matches_pointwise_evaluation(num_segments):
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    rng = jax.random.PRNGKey(321)
    random_cfg = random_q(model, rng, scale=0.05)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        J_batch = model._J_local_batched(q, s_points)

        for idx, s_val in enumerate(s_points):
            J_single = model._J_local(q, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_bodyframe_inertialframe_coherence(num_segments: int):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
    key = jax.random.PRNGKey(1)
    q = random_q(model, key, scale=0.05)

    for s in sample_arc_lengths(model):
        J_impl = model.jacobian_inertialframe(q, s)
        J_body = model.jacobian_bodyframe(q, s)
        chi = model.forward_kinematics(q, s)
        # only rotation
        g = exp_SE2(jnp.array([chi[0], 0.0, 0.0]))
        J_expected = Adjoint_g_SE2(g) @ J_body

        assert jnp.allclose(J_impl, J_expected, rtol=1e-6, atol=1e-7), (
            f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_expected:\n{J_expected}"
        )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_inertialframe_matches_autodiff(num_segments):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
    key = jax.random.PRNGKey(1)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.05)

        for s in sample_arc_lengths(model):
            J_impl = model.jacobian_inertialframe(q, s)

            def fk(q_):
                return model.forward_kinematics(q_, s)  # [theta, x, y]

            J_ad = jax.jacfwd(fk)(q)

            assert_allclose(
                J_impl,
                J_ad,
                rtol=1e-6,
                atol=1e-7,
                err_msg=(
                    f"num_segments={num_segments}, s={s}\n"
                    f"J_impl:\n{J_impl}\nJ_ad:\n{J_ad}"
                ),
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_inertial_velocity_matches_central_differences(num_segments):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dt = 1e-6
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
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )

    key = jax.random.PRNGKey(5)
    delta = 1e-6
    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.01)

        for s in sample_arc_lengths(model):
            J_impl = model.jacobian_inertialframe(q, s)

            n = q.shape[0]
            J_fd_ls = []
            eye = jnp.eye(n)
            for j in range(n):
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                fp = model.forward_kinematics(qp, s)
                fm = model.forward_kinematics(qm, s)
                J_fd_ls.append((fp - fm) / (2 * delta))

            J_fd = jnp.stack(J_fd_ls, axis=1)

            assert jnp.allclose(J_impl, J_fd, rtol=1e-3, atol=5e-6), (
                f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_fd:\n{J_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_derivative_bodyframe_matches_autograd_jvp(num_segments):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
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
def test_jacobian_derivative_bodyframe_matches_central_differences(num_segments):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
    key = jax.random.PRNGKey(3)

    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    delta = 1e-6
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_bodyframe(q, qd, s)

            eye = jnp.eye(q.shape[0])
            dJ_cols = []
            for j in range(q.shape[0]):
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                Jp = model.jacobian_bodyframe(qp, s)
                Jm = model.jacobian_bodyframe(qm, s)
                dJ_cols.append((Jp - Jm) / (2 * delta))
            dJ_dq_fd = jnp.stack(dJ_cols, axis=-1)
            Jd_num = jnp.tensordot(dJ_dq_fd, qd, axes=([-1], [0]))

            assert jnp.allclose(Jd_impl, Jd_num, rtol=1e-3, atol=5e-6), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_num:\n{Jd_num}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_derivative_inertialframe_matches_autograd_jvp(num_segments):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
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
def test_J_Jd_local_tips_matches_pointwise_evaluation(num_segments: int):
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(987)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(654), scale=0.1)

    s_tips = model.L_cum[1:]

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_local_tips, Jd_local_tips = model._J_Jd_local_tips(q, qd)

        for idx, s_tip in enumerate(s_tips):
            J_local, Jd_local = model._J_Jd_local(q, qd, s_tip)

            assert_allclose(J_local, J_local_tips[idx], rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_local, Jd_local_tips[idx], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_Jd_local_batched_matches_pointwise_evaluation(num_segments: int):
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(321)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(4321), scale=0.1)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = model._J_Jd_local_batched(q, qd, s_points)

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model._J_Jd_local(q, qd, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


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
def test_forward_dynamics_matches_manual_computation(num_segments: int):
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )

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


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_mode_automatic_differentiability_at_zero_configuration(
    num_segments: int,
) -> None:
    model, _ = make_planar_pcs(
        num_segments=num_segments, total_length=PLANAR_TOTAL_LENGTH
    )
    dof = int(model.num_active_strains.item())
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((model.num_actuators,), dtype=jnp.float64)
    s = model.L_cum[-1]

    dg_dq = jacfwd(model.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacfwd(model.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacfwd(model.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe = jacfwd(model.jacobian_and_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacfwd(
        model.jacobian_and_derivative_inertialframe, argnums=0
    )(q, qd, s)

    assert not jnp.isnan(dg_dq).any()
    assert not jnp.isnan(dJ_bodyframe_dq).any()
    assert not jnp.isnan(dJ_inertialframe_dq).any()
    assert not jnp.isnan(dJd_bodyframe).any()
    assert not jnp.isnan(dJd_inertialframe).any()

    dB_dq = jacfwd(model.inertia_matrix)(q)
    dC_dq = jacfwd(model.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacfwd(model.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacfwd(model.gravitational_force)(q)
    dtau_el_dq = jacfwd(model.elastic_force)(q)
    dtau_u_dq = jacfwd(model.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacfwd(model.actuation_force, argnums=1)(q, u)
    dy_dy = jacfwd(model.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacfwd(model.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dB_dq).any()
    assert not jnp.isnan(dC_dq).any()
    assert not jnp.isnan(dC_dqd).any()
    assert not jnp.isnan(dG_dq).any()
    assert not jnp.isnan(dtau_el_dq).any()
    assert not jnp.isnan(dtau_u_dq).any()
    assert not jnp.isnan(dtau_u_du).any()
    assert not jnp.isnan(dy_dy).any()
    assert not jnp.isnan(dy_du).any()

    dT_dq = jacfwd(model.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacfwd(model.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacfwd(model.gravitational_energy)(q)
    dU_dq = jacfwd(model.potential_energy)(q)
    dE_dq = jacfwd(model.total_energy, argnums=0)(q, qd)
    dE_dqd = jacfwd(model.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any()
    assert not jnp.isnan(dT_dqd).any()
    assert not jnp.isnan(dU_G_dq).any()
    assert not jnp.isnan(dU_dq).any()
    assert not jnp.isnan(dE_dq).any()
    assert not jnp.isnan(dE_dqd).any()


def test_reverse_mode_automatic_differentiability_at_zero_configuration() -> None:
    model, _ = make_planar_pcs(num_segments=2, total_length=PLANAR_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((model.num_actuators,), dtype=jnp.float64)
    s = model.L_cum[-1]

    dg_dq = jacrev(model.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacrev(model.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacrev(model.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe = jacrev(model.jacobian_and_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacrev(
        model.jacobian_and_derivative_inertialframe, argnums=0
    )(q, qd, s)

    assert not jnp.isnan(dg_dq).any()
    assert not jnp.isnan(dJ_bodyframe_dq).any()
    assert not jnp.isnan(dJ_inertialframe_dq).any()
    assert not jnp.isnan(dJd_bodyframe).any()
    assert not jnp.isnan(dJd_inertialframe).any()

    dB_dq = jacrev(model.inertia_matrix)(q)
    dC_dq = jacrev(model.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacrev(model.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacrev(model.gravitational_force)(q)
    dtau_el_dq = jacrev(model.elastic_force)(q)
    dtau_u_dq = jacrev(model.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacrev(model.actuation_force, argnums=1)(q, u)
    dy_dy = jacrev(model.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacrev(model.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dB_dq).any()
    assert not jnp.isnan(dC_dq).any()
    assert not jnp.isnan(dC_dqd).any()
    assert not jnp.isnan(dG_dq).any()
    assert not jnp.isnan(dtau_el_dq).any()
    assert not jnp.isnan(dtau_u_dq).any()
    assert not jnp.isnan(dtau_u_du).any()
    assert not jnp.isnan(dy_dy).any()
    assert not jnp.isnan(dy_du).any()

    dT_dq = jacrev(model.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacrev(model.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacrev(model.gravitational_energy)(q)
    dU_dq = jacrev(model.potential_energy)(q)
    dE_dq = jacrev(model.total_energy, argnums=0)(q, qd)
    dE_dqd = jacrev(model.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any()
    assert not jnp.isnan(dT_dqd).any()
    assert not jnp.isnan(dU_G_dq).any()
    assert not jnp.isnan(dU_dq).any()
    assert not jnp.isnan(dE_dq).any()
    assert not jnp.isnan(dE_dqd).any()


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
