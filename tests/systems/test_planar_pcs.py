import jax
import numpy as onp
import pytest
from jax import Array, jacfwd, jacrev, jvp
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import planar_base_pose, planar_pcs_params

from soromox.systems import CrossSectionGeometry, PlanarPCS, PlanarPCSStructure
from soromox.utils.geometry import poses
from soromox.utils.integration import scale_interior_gaussian_quadrature
from soromox.utils.lie_algebra import se2
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
    xi_ref: Array | None = None,
    total_length: float | None = None,
    num_gauss_points: int = 5,
    strain_selector: Array | None = None,
    scale_rotational_basis_by_length: bool = False,
):
    """
    Create a planar constant strain model.
    """
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    segment_length = 1e-1 if total_length is None else total_length / num_segments
    segment_lengths = segment_length * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * segment_lengths[:, None]
        ).flatten()
    )
    params = planar_pcs_params(
        base_pose=planar_base_pose(th0),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, -9.81]),  # gravity vector [m/s^2] UP!
        young_modulus=2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        shear_modulus=1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        damping_matrix=damping_matrix,
        reference_strain=xi_ref,
    )

    model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(
            num_gauss_points=num_gauss_points,
            strain_selector=strain_selector,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        ),
    )

    return model, params


def sample_arc_lengths(model: PlanarPCS) -> list[float]:
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


def strict_interior_arc_lengths(model: PlanarPCS) -> jnp.ndarray:
    fractions = jnp.asarray([0.37, 0.73], dtype=jnp.float64)
    return (model.L_cum[:-1, None] + model.L[:, None] * fractions).reshape(-1)


def random_q(model, key=jax.random.PRNGKey(0), scale=0.1):
    n = int(model.num_active_strains.item())
    return scale * jax.random.normal(key, (n,))


def expected_selection_basis(num_rows: int, active_indices: tuple[int, ...]) -> Array:
    basis = jnp.zeros((num_rows, len(active_indices)), dtype=jnp.float64)
    for col, row in enumerate(active_indices):
        basis = basis.at[row, col].set(1.0)
    return basis


def segment_tip_poses(model: PlanarPCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]

    def fk_at_s(s: Array) -> Array:
        return model.forward_kinematics(q, s)

    return jax.vmap(fk_at_s)(s_vals)


def constant_strain_inverse_kinematics_fn(params, xi_ref, chi, s) -> Array:
    # split the chi vector into x, y, and th0
    th, px, py = chi
    th0 = params.base_pose[0].item()
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
    D = robot.damping_matrix(q)
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
    D = robot.damping_matrix(q)
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
    chi_plus = robot.forward_kinematics(q=q + delta_q, s=params.length[0])
    chi_pred = chi + J @ delta_q
    assert_allclose(chi_plus, chi_pred, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    u = jnp.zeros((3,))  # no external forces
    robot = robot.update_params(gravity=jnp.zeros((2,)))  # no gravity for this test
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", robot.params.gravity)
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
    qdd, qdres = jnp.split(yd, 2)
    assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(qdd, jnp.zeros((3,)), rtol=RTOL, atol=ATOL)
    assert_allclose(qdres, qd, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")


def test_public_planar_pcs_accessors_geometry_and_chi() -> None:
    model, params = make_planar_pcs(num_segments=2)
    q = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)

    assert model.is_planar is True
    assert_allclose(model.length, jnp.sum(params.length), rtol=RTOL, atol=ATOL)
    assert_allclose(model.segment_length, params.length, rtol=RTOL, atol=ATOL)

    s_second = params.length[0] + 0.25 * params.length[1]
    segment_idx, s_local = model.classify_segment(s_second)
    assert int(segment_idx) == 1
    assert_allclose(s_local, 0.25 * params.length[1], rtol=RTOL, atol=ATOL)

    tag, geom = model.cross_section_geometry(q, s_second)
    assert int(tag) == CrossSectionGeometry.CIRCULAR
    assert_allclose(geom, jnp.array([params.radius[1]]), rtol=RTOL, atol=ATOL)

    xi = model.strain(q)
    assert_allclose(model.chi(xi, s_second), model.forward_kinematics(q, s_second))


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
        chi_expected = jax.vmap(lambda s, q=q: model.forward_kinematics(q, s))(s_ps)

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
    base_pose = model._base_planar_pose(chi_tips.dtype)
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
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
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
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector_no_curvature),
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
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector_curvature_only),
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
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector_extension_only),
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


@pytest.mark.parametrize(
    "strain_selector",
    [
        jnp.array(
            [
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
            ],
            dtype=bool,
        ),
        jnp.zeros((6,), dtype=bool),
    ],
    ids=["segment-specific-inactive-strains", "all-strains-inactive"],
)
def test_inverse_kinematics_with_inactive_strain_segments(strain_selector: Array):
    num_segments = int(strain_selector.size // 3)
    model, _ = make_planar_pcs(
        num_segments=num_segments, th0=0.0, strain_selector=strain_selector
    )

    q = random_q(model, key=jax.random.PRNGKey(333), scale=0.08)
    chi_tips = segment_tip_poses(model, q)
    q_recovered = model.inverse_kinematics(chi_tips)

    assert q_recovered.shape == q.shape
    assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)
    assert_allclose(
        segment_tip_poses(model, q_recovered), chi_tips, rtol=RTOL, atol=ATOL
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

        J_tip_batch = J_tips[idx] @ model.B_xi
        J_tip_single = model.jacobian_bodyframe(q, s)

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
def test_jacobian_bodyframe_batched_matches_pointwise_evaluation(
    num_segments,
) -> None:
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    rng = jax.random.PRNGKey(321)
    random_cfg = random_q(model, rng, scale=0.05)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        J_batch = model.jacobian_bodyframe_batched(q, s_points)

        for idx, s_val in enumerate(s_points):
            J_single = model.jacobian_bodyframe(q, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_bodyframe_with_pose_matches_forward_kinematics(
    num_segments: int,
) -> None:
    model, _ = make_planar_pcs(num_segments=num_segments)
    q = random_q(model, jax.random.PRNGKey(987), scale=0.05)

    for s in sample_arc_lengths(model):
        chi, J_body = model._jacobian_bodyframe_with_pose(q, s)

        assert_allclose(chi, model.forward_kinematics(q, s), rtol=RTOL, atol=ATOL)
        assert_allclose(J_body, model.jacobian_bodyframe(q, s), rtol=RTOL, atol=ATOL)


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
        g = poses.planar_pose_to_transform(jnp.array([chi[0], 0.0, 0.0]))
        J_expected = se2.adjoint(g) @ J_body

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

            def fk(q_, s=s):
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


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    rng = jax.random.PRNGKey(7890)
    random_cfg = random_q(model, rng, scale=0.05)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        J_batch = model.jacobian_inertialframe_batched(q, s_points)

        for idx, s_val in enumerate(s_points):
            J_single = model.jacobian_inertialframe(q, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_tips_matches_pointwise_inertialframe_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_planar_pcs(num_segments=num_segments)
    q = random_q(model, jax.random.PRNGKey(7891), scale=0.05)

    s_tips = model.L_cum[1:]
    J_tips = model.jacobian_tips(q)
    J_expected = jax.vmap(lambda s: model.jacobian_inertialframe(q, s))(s_tips)

    assert J_tips.shape == (num_segments, 3, int(model.num_active_strains.item()))
    assert_allclose(J_tips, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_time_derivative_bodyframe_matches_autograd_jvp(num_segments):
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
            J_impl, Jd_impl = model.jacobian_and_time_derivative_bodyframe(q, qd, s)

            def J_of_q(q_, s=s):
                return model.jacobian_bodyframe(q_, s)

            _, Jd_jvp = jax.jvp(J_of_q, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_jvp:\n{Jd_jvp}"
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3, 5])
def test_jacobian_time_derivative_bodyframe_matches_central_differences(num_segments):
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
            J_impl, Jd_impl = model.jacobian_and_time_derivative_bodyframe(q, qd, s)

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
def test_jacobian_time_derivative_inertialframe_matches_autograd_jvp(num_segments):
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
            J_impl, Jd_impl = model.jacobian_and_time_derivative_inertialframe(q, qd, s)

            def J_of_q(q_, s=s):
                return model.jacobian_inertialframe(q_, s)

            _, Jd_jvp = jax.jvp(J_of_q, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jd_jvp)}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_and_time_derivative_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_planar_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(7890)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(9876), scale=0.1)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = model.jacobian_and_time_derivative_inertialframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_time_derivative_inertialframe(
                q, qd, s_val
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


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
            J_local, Jd_local = model.jacobian_and_time_derivative_bodyframe(
                q, qd, s_tip
            )

            assert_allclose(
                J_local,
                J_local_tips[idx] @ model.B_xi,
                rtol=RTOL,
                atol=ATOL,
            )
            assert_allclose(
                Jd_local,
                Jd_local_tips[idx] @ model.B_xi,
                rtol=RTOL,
                atol=ATOL,
            )


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
        J_batch, Jd_batch = model.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_time_derivative_bodyframe(
                q, qd, s_val
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


def test_public_planar_pcs_jacobian_wrappers_match_inertialframe_methods() -> None:
    model, _ = make_planar_pcs(num_segments=2)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(2028))
    q = random_q(model, key_q, scale=0.03)
    qd = random_q(model, key_qd, scale=0.04)
    s = 0.6 * model.length
    s_ps = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    assert_allclose(
        model.jacobian(q, s),
        model.jacobian_inertialframe(q, s),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        model.jacobian_batched(q, s_ps),
        model.jacobian_inertialframe_batched(q, s_ps),
        rtol=RTOL,
        atol=ATOL,
    )

    J, Jd = model.jacobian_and_time_derivative(q, qd, s)
    J_expected, Jd_expected = model.jacobian_and_time_derivative_inertialframe(q, qd, s)
    assert_allclose(J, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd, Jd_expected, rtol=RTOL, atol=ATOL)

    J_batch, Jd_batch = model.jacobian_and_time_derivative_batched(q, qd, s_ps)
    J_batch_expected, Jd_batch_expected = (
        model.jacobian_and_time_derivative_inertialframe_batched(q, qd, s_ps)
    )
    assert_allclose(J_batch, J_batch_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_batch, Jd_batch_expected, rtol=RTOL, atol=ATOL)


def test_planar_pcs_arc_length_derivatives_match_autodiff() -> None:
    model, _ = make_planar_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(2029), scale=0.03)

    for s in strict_interior_arc_lengths(model):
        _, fk_s_autodiff = jvp(
            lambda s_: model._forward_kinematics(q, s_),
            (s,),
            (jnp.ones_like(s),),
        )
        assert_allclose(
            model.forward_kinematics_arc_length_derivative(q, s),
            fk_s_autodiff,
            rtol=RTOL,
            atol=ATOL,
        )

        _, Js_autodiff = jvp(
            lambda s_: model._jacobian(q, s_),
            (s,),
            (jnp.ones_like(s),),
        )
        assert_allclose(
            model.jacobian_arc_length_derivative(q, s),
            Js_autodiff,
            rtol=1e-6,
            atol=1e-7,
        )


def test_planar_pcs_custom_jvps_include_arc_length_derivative() -> None:
    model, _ = make_planar_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(2030), scale=0.03)
    qd = random_q(model, jax.random.PRNGKey(2031), scale=0.02)
    s = strict_interior_arc_lengths(model)[1]
    sd = jnp.array(0.37, dtype=jnp.float64)

    _, posed = jvp(
        lambda q_, s_: model.forward_kinematics(q_, s_),
        (q, s),
        (qd, sd),
    )
    _, expected_posed = jvp(
        lambda q_, s_: model._forward_kinematics(q_, s_),
        (q, s),
        (qd, sd),
    )
    assert_allclose(posed, expected_posed, rtol=RTOL, atol=ATOL)

    _, Jd = jvp(
        lambda q_, s_: model.jacobian(q_, s_),
        (q, s),
        (qd, sd),
    )
    _, expected_Jd = jvp(
        lambda q_, s_: model._jacobian(q_, s_),
        (q, s),
        (qd, sd),
    )
    assert_allclose(Jd, expected_Jd, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inertia_matrix_matches_kinetic_energy_autodiff(num_segments: int):
    robot, _ = make_planar_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(2)
    q_keys = jax.random.split(key, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key + 1, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        B_impl = robot.inertia_matrix(q)

        def T_of_qd(qd_, q=q):
            return robot.kinetic_energy(q, qd_)

        dT_dqdsq = jacfwd(jax.grad(T_of_qd))(qd)
        B_expected = dT_dqdsq

        assert_allclose(B_impl, B_expected, rtol=RTOL, atol=ATOL)


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
        jac_T_q = jax.jacobian(lambda qq, qd=qd: dT_dqd(qq, qd))(q)
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


def test_planar_pcs_gravitational_energy_gradient_matches_force() -> None:
    model, _ = make_planar_pcs(num_segments=1)
    q = random_q(model, jax.random.PRNGKey(315), scale=0.02)

    dU_dq = jax.grad(lambda q_: model.gravitational_energy(q_))(q)

    assert_allclose(dU_dq, model.gravitational_force(q), rtol=RTOL, atol=ATOL)


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
        D = model.damping_matrix(q)
        tau_el = model.elastic_force(q)
        tau_u = model.actuation_force(q, u)

        qdd_expected = jnp.linalg.solve(
            B, tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )
        yd_expected = jnp.concatenate([qd, qdd_expected])

        assert_allclose(yd, yd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 3])
@pytest.mark.parametrize(
    "selector_per_segment",
    [
        None,
        (True, True, False),
        (False, True, False),
    ],
)
def test_integration_kinematics_matches_existing_batched_path_planar(
    num_segments: int, selector_per_segment: tuple[bool, ...] | None
):
    base_model, params = make_planar_pcs(num_segments=num_segments)
    strain_selector = (
        None
        if selector_per_segment is None
        else jnp.tile(jnp.asarray(selector_per_segment, dtype=bool), num_segments)
    )
    model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
    )
    dof = int(model.num_active_strains.item())

    key_q, key_qd = jax.random.split(jax.random.PRNGKey(7123))
    q = random_q(model, key_q, scale=0.05)
    qd = random_q(model, key_qd, scale=0.1)

    g_quads, J_quads, Jd_quads = model.integration_kinematics(q, qd)
    Xs_scaled, _ = jax.vmap(
        scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
    )(
        model.integration_points,
        model.integration_weights,
        model.L_cum[:-1],
        model.L_cum[1:],
    )
    s_points = Xs_scaled.reshape(-1)
    num_inner = model.num_gauss_points

    chi_expected = model.forward_kinematics_batched(q, s_points)
    g_expected = jax.vmap(poses.planar_pose_to_transform)(chi_expected).reshape(
        num_segments, num_inner, 3, 3
    )
    J_full, Jd_full = model._J_Jd_local_batched(q, qd, s_points)
    J_expected = (J_full @ model.B_xi).reshape(num_segments, num_inner, 3, dof)
    Jd_expected = (Jd_full @ model.B_xi).reshape(num_segments, num_inner, 3, dof)

    assert_allclose(g_quads, g_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(J_quads, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_quads, Jd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 3])
@pytest.mark.parametrize(
    "selector_per_segment",
    [
        None,
        (True, True, False),
        (False, True, False),
    ],
)
def test_dynamics_terms_match_public_matrices_planar(
    num_segments: int, selector_per_segment: tuple[bool, ...] | None
):
    base_model, params = make_planar_pcs(num_segments=num_segments)
    strain_selector = (
        None
        if selector_per_segment is None
        else jnp.tile(jnp.asarray(selector_per_segment, dtype=bool), num_segments)
    )
    model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
    )
    dof = int(model.num_active_strains.item())

    zero_q = jnp.zeros((dof,), dtype=jnp.float64)
    zero_qd = jnp.zeros((dof,), dtype=jnp.float64)
    key_q, key_qd, key_u, key_tau = jax.random.split(jax.random.PRNGKey(7124), 4)
    random_q_ = random_q(model, key_q, scale=0.05)
    random_qd = random_q(model, key_qd, scale=0.1)
    u = random_q(model, key_u, scale=0.2)
    tau_ext = random_q(model, key_tau, scale=0.03)

    for q, qd in ((zero_q, zero_qd), (random_q_, random_qd)):
        B, Cqd, G = model.dynamics_terms(q, qd)
        C = model.coriolis_matrix(q, qd)

        assert_allclose(B, model.inertia_matrix(q), rtol=RTOL, atol=ATOL)
        assert_allclose(Cqd, C @ qd, rtol=RTOL, atol=ATOL)
        assert_allclose(G, model.gravitational_force(q), rtol=RTOL, atol=ATOL)

        y = jnp.concatenate([q, qd])
        yd = model.forward_dynamics(0.0, y, (u, tau_ext))
        tau_el = model.elastic_force(q)
        tau_u = model.actuation_force(q, u)
        D = model.damping_matrix(q)
        qdd_expected = jnp.linalg.solve(
            B, tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )
        assert_allclose(yd, jnp.concatenate([qd, qdd_expected]), rtol=RTOL, atol=ATOL)


def test_cached_constant_matrices_refresh_after_update_params_planar():
    selector_per_segment = jnp.array([True, True, False], dtype=bool)
    base_model, params = make_planar_pcs(num_segments=2)
    model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=jnp.tile(selector_per_segment, 2)),
    )

    updated = model.update_params(
        radius=1.1 * model.r,
        density=0.9 * model.rho,
        young_modulus=1.25 * model.E,
        shear_modulus=0.75 * model.G,
        damping_matrix=2.0 * model.D_full,
    )
    segment_ids = jnp.arange(updated.num_segments)
    expected_M = jax.vmap(updated._compute_local_mass_matrix)(segment_ids)
    expected_K_full = updated._compute_stiffness_full_matrix()
    expected_K = updated.B_xi.T @ expected_K_full @ updated.B_xi
    expected_D_full = updated.D_full
    expected_D = updated.B_xi.T @ expected_D_full @ updated.B_xi

    assert_allclose(updated.M_segments, expected_M, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.K_full, expected_K_full, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.K_active, expected_K, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.D_full, expected_D_full, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.D_active, expected_D, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.stiffness_matrix(), expected_K, rtol=RTOL, atol=ATOL)
    assert_allclose(
        updated.damping_matrix(jnp.zeros(updated.num_dofs)),
        expected_D,
        rtol=RTOL,
        atol=ATOL,
    )


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
    _, dJd_bodyframe = jacfwd(model.jacobian_and_time_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacfwd(
        model.jacobian_and_time_derivative_inertialframe, argnums=0
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
    _, dJd_bodyframe = jacrev(model.jacobian_and_time_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacrev(
        model.jacobian_and_time_derivative_inertialframe, argnums=0
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


# ======================================================================================
# Strain-basis consistency tests (selection basis applied correctly across APIs)
# ======================================================================================


def test_strain_basis_creation_matches_selector_order_planar():
    base_model, params = make_planar_pcs(num_segments=3)
    strain_selector = jnp.array(
        [True, False, True, False, True, False, True, True, False], dtype=bool
    )
    model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
    )

    expected_B = expected_selection_basis(9, (0, 2, 4, 6, 7))

    assert int(model.num_active_strains.item()) == 5
    assert model.B_xi.shape == (9, 5)
    assert_allclose(model.B_xi, expected_B, rtol=0.0, atol=0.0)
    assert_allclose(model.B_xi.T @ model.B_xi, jnp.eye(5), rtol=0.0, atol=0.0)

    q = jnp.arange(1.0, 6.0)
    expected_xi = expected_B @ q + model.xi_ref
    assert_allclose(model.strain(q), expected_xi, rtol=RTOL, atol=ATOL)


def test_rotational_strain_basis_length_scaling_matches_unscaled_coordinates_planar():
    num_segments = 2
    total_length = 0.5
    segment_length = total_length / num_segments
    unscaled, _ = make_planar_pcs(
        num_segments=num_segments,
        total_length=total_length,
        num_gauss_points=5,
        scale_rotational_basis_by_length=False,
    )
    scaled, _ = make_planar_pcs(
        num_segments=num_segments,
        total_length=total_length,
        num_gauss_points=5,
        scale_rotational_basis_by_length=True,
    )

    per_segment_scale = jnp.array([1 / segment_length, 1.0, 1.0], dtype=jnp.float64)
    coordinate_scale = jnp.tile(per_segment_scale, num_segments)
    coordinate_map = jnp.diag(coordinate_scale)

    assert scaled.scale_rotational_basis_by_length
    assert_allclose(
        scaled.B_xi,
        coordinate_scale[:, None] * unscaled.B_xi,
        rtol=RTOL,
        atol=ATOL,
    )

    q_scaled = jnp.linspace(-0.04, 0.05, int(scaled.num_dofs), dtype=jnp.float64)
    qd_scaled = jnp.linspace(0.02, -0.03, int(scaled.num_dofs), dtype=jnp.float64)
    q_unscaled = coordinate_map @ q_scaled
    qd_unscaled = coordinate_map @ qd_scaled

    assert_allclose(
        scaled.strain(q_scaled), unscaled.strain(q_unscaled), rtol=RTOL, atol=ATOL
    )

    for s in sample_arc_lengths(scaled):
        chi_scaled = scaled.forward_kinematics(q_scaled, s)
        chi_unscaled = unscaled.forward_kinematics(q_unscaled, s)
        assert_allclose(chi_scaled, chi_unscaled, rtol=RTOL, atol=ATOL)

        J_scaled = scaled.jacobian_bodyframe(q_scaled, s)
        J_unscaled = unscaled.jacobian_bodyframe(q_unscaled, s)
        assert_allclose(J_scaled, J_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)

        J_scaled, Jd_scaled = scaled.jacobian_and_time_derivative_bodyframe(
            q_scaled, qd_scaled, s
        )
        J_unscaled, Jd_unscaled = unscaled.jacobian_and_time_derivative_bodyframe(
            q_unscaled, qd_unscaled, s
        )
        assert_allclose(J_scaled, J_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_scaled, Jd_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)

    assert_allclose(
        scaled.inertia_matrix(q_scaled),
        coordinate_map.T @ unscaled.inertia_matrix(q_unscaled) @ coordinate_map,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        scaled.gravitational_force(q_scaled),
        coordinate_map.T @ unscaled.gravitational_force(q_unscaled),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        scaled.stiffness_matrix(),
        coordinate_map.T @ unscaled.stiffness_matrix() @ coordinate_map,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        scaled.damping_matrix(q_scaled),
        coordinate_map.T @ unscaled.damping_matrix(q_unscaled) @ coordinate_map,
        rtol=RTOL,
        atol=ATOL,
    )

    updated = scaled.update_params(length=jnp.array([0.2, 0.3]))
    updated_scale = jnp.array([5.0, 1.0, 1.0, 10 / 3, 1.0, 1.0])
    assert_allclose(
        updated.B_xi,
        updated_scale[:, None] * updated.B_xi_unscaled,
        rtol=RTOL,
        atol=ATOL,
    )


def _make_full_and_reduced_planar(num_segments: int, selector_per_segment: Array):
    strain_selector = jnp.tile(selector_per_segment, num_segments)

    # Build a base model and copy its params for both full and reduced variants
    base_model, params = make_planar_pcs(num_segments=num_segments)
    full_model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=None),
    )
    reduced_model = PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(strain_selector=strain_selector),
    )
    return full_model, reduced_model, reduced_model.B_xi


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_strain_and_kinematics_planar(num_segments: int):
    # Select [kappa_z, sigma_x] (activate bending + axial)
    selector_per_segment = jnp.array([True, True, False], dtype=bool)
    full, reduced, B = _make_full_and_reduced_planar(num_segments, selector_per_segment)

    key_q = jax.random.PRNGKey(1101)
    q_small = random_q(reduced, key_q, scale=0.05)
    q_full = B @ q_small

    # strain must match
    xi_small = reduced.strain(q_small)
    xi_full = full.strain(q_full)
    n_full_strains = int(full.num_strains)
    assert xi_small.shape == (n_full_strains,)
    assert xi_full.shape == (n_full_strains,)
    assert_allclose(xi_full, xi_small, rtol=RTOL, atol=ATOL)

    # forward kinematics at several s values and batched
    s_values = jnp.asarray([0.0] + sample_arc_lengths(full), dtype=jnp.float64)
    for s in s_values:
        chi_small = reduced.forward_kinematics(q_small, s)
        chi_full = full.forward_kinematics(q_full, s)
        assert chi_small.shape == (3,)
        assert chi_full.shape == (3,)
        assert_allclose(chi_full, chi_small, rtol=RTOL, atol=ATOL)

    chi_tips_small = reduced.forward_kinematics_tips(q_small)
    chi_tips_full = full.forward_kinematics_tips(q_full)
    assert chi_tips_small.shape == (num_segments, 3)
    assert chi_tips_full.shape == (num_segments, 3)
    assert_allclose(chi_tips_full, chi_tips_small, rtol=RTOL, atol=ATOL)

    chi_batched_small = reduced.forward_kinematics_batched(q_small, s_values)
    chi_batched_full = full.forward_kinematics_batched(q_full, s_values)
    N = s_values.shape[0]
    assert chi_batched_small.shape == (N, 3)
    assert chi_batched_full.shape == (N, 3)
    assert_allclose(chi_batched_full, chi_batched_small, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_jacobians_and_time_derivatives_planar(
    num_segments: int,
):
    selector_per_segment = jnp.array([True, True, False], dtype=bool)
    full, reduced, B = _make_full_and_reduced_planar(num_segments, selector_per_segment)

    key = jax.random.PRNGKey(1202)
    key_q, key_qd = jax.random.split(key)
    q_small = random_q(reduced, key_q, scale=0.05)
    qd_small = random_q(reduced, key_qd, scale=0.1)
    q_full, qd_full = B @ q_small, B @ qd_small

    s_points = jnp.asarray(sample_arc_lengths(full), dtype=jnp.float64)
    n_full_strains = int(full.num_strains)

    # Body-frame Jacobian
    n_small_act = int(reduced.num_active_strains.item())
    for s in s_points:
        Jb_small = reduced.jacobian_bodyframe(q_small, s)
        Jb_full = full.jacobian_bodyframe(q_full, s)
        assert Jb_small.shape == (3, n_small_act)
        assert Jb_full.shape == (3, int(full.num_active_strains.item()))
        assert (Jb_full @ B).shape == (3, n_small_act)
        assert_allclose(Jb_full @ B, Jb_small, rtol=RTOL, atol=ATOL)

    # Inertial-frame Jacobian
    for s in s_points:
        Ji_small = reduced.jacobian_inertialframe(q_small, s)
        Ji_full = full.jacobian_inertialframe(q_full, s)
        assert Ji_small.shape == (3, n_small_act)
        assert Ji_full.shape == (3, int(full.num_active_strains.item()))
        assert (Ji_full @ B).shape == (3, n_small_act)
        assert_allclose(Ji_full @ B, Ji_small, rtol=RTOL, atol=ATOL)

    # Body-frame (J, Jd)
    for s in s_points:
        J_small, Jd_small = reduced.jacobian_and_time_derivative_bodyframe(
            q_small, qd_small, s
        )
        J_full, Jd_full = full.jacobian_and_time_derivative_bodyframe(
            q_full, qd_full, s
        )
        assert J_small.shape == (3, n_small_act)
        assert Jd_small.shape == (3, n_small_act)
        assert J_full.shape == (3, int(full.num_active_strains.item()))
        assert Jd_full.shape == (3, int(full.num_active_strains.item()))
        assert_allclose(J_full @ B, J_small, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_full @ B, Jd_small, rtol=RTOL, atol=ATOL)

    # Inertial-frame (J, Jd)
    for s in s_points:
        J_small, Jd_small = reduced.jacobian_and_time_derivative_inertialframe(
            q_small, qd_small, s
        )
        J_full, Jd_full = full.jacobian_and_time_derivative_inertialframe(
            q_full, qd_full, s
        )
        assert J_small.shape == (3, n_small_act)
        assert Jd_small.shape == (3, n_small_act)
        assert J_full.shape == (3, int(full.num_active_strains.item()))
        assert Jd_full.shape == (3, int(full.num_active_strains.item()))
        assert_allclose(J_full @ B, J_small, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_full @ B, Jd_small, rtol=RTOL, atol=ATOL)

    # Tips inertial-frame Jacobian
    Ji_tips_small = reduced.jacobian_tips(q_small)
    Ji_tips_full = full.jacobian_tips(q_full)
    assert Ji_tips_small.shape == (num_segments, 3, n_small_act)
    assert Ji_tips_full.shape == (
        num_segments,
        3,
        int(full.num_active_strains.item()),
    )
    assert_allclose(
        jnp.einsum("ijk,kl->ijl", Ji_tips_full, B),
        Ji_tips_small,
        rtol=RTOL,
        atol=ATOL,
    )

    # Batched body-frame Jacobian (internal helper, returns full-strain size)
    Jb_batch_small = reduced._J_local_batched(q_small, s_points)
    Jb_batch_full = full._J_local_batched(q_full, s_points)
    assert Jb_batch_small.shape == (s_points.shape[0], 3, n_full_strains)
    assert Jb_batch_full.shape == (s_points.shape[0], 3, n_full_strains)
    assert_allclose(Jb_batch_full, Jb_batch_small, rtol=RTOL, atol=ATOL)

    # Tips body-frame Jacobian (internal helper, returns full-strain size)
    Jb_tips_small = reduced._J_local_tips(q_small)
    Jb_tips_full = full._J_local_tips(q_full)
    assert Jb_tips_small.shape == (num_segments, 3, n_full_strains)
    assert Jb_tips_full.shape == (num_segments, 3, n_full_strains)
    assert_allclose(Jb_tips_full, Jb_tips_small, rtol=RTOL, atol=ATOL)

    # Batched body-frame (J, Jd) internal helper (full-strain size)
    Jb_batch_small, Jbd_batch_small = reduced._J_Jd_local_batched(
        q_small, qd_small, s_points
    )
    Jb_batch_full, Jbd_batch_full = full._J_Jd_local_batched(q_full, qd_full, s_points)
    assert Jb_batch_small.shape == (s_points.shape[0], 3, n_full_strains)
    assert Jbd_batch_small.shape == (s_points.shape[0], 3, n_full_strains)
    assert Jb_batch_full.shape == (s_points.shape[0], 3, n_full_strains)
    assert Jbd_batch_full.shape == (s_points.shape[0], 3, n_full_strains)
    assert_allclose(Jb_batch_full, Jb_batch_small, rtol=RTOL, atol=ATOL)
    assert_allclose(Jbd_batch_full, Jbd_batch_small, rtol=RTOL, atol=ATOL)

    # Tips body-frame (J, Jd) internal helper (full-strain size)
    Jb_tips_small, Jbd_tips_small = reduced._J_Jd_local_tips(q_small, qd_small)
    Jb_tips_full, Jbd_tips_full = full._J_Jd_local_tips(q_full, qd_full)
    assert Jb_tips_small.shape == (num_segments, 3, n_full_strains)
    assert Jbd_tips_small.shape == (num_segments, 3, n_full_strains)
    assert Jb_tips_full.shape == (num_segments, 3, n_full_strains)
    assert Jbd_tips_full.shape == (num_segments, 3, n_full_strains)
    assert_allclose(Jb_tips_full, Jb_tips_small, rtol=RTOL, atol=ATOL)
    assert_allclose(Jbd_tips_full, Jbd_tips_small, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_dynamics_and_forces_planar(num_segments: int):
    selector_per_segment = jnp.array([True, True, False], dtype=bool)
    full, reduced, B = _make_full_and_reduced_planar(num_segments, selector_per_segment)

    key = jax.random.PRNGKey(1303)
    key_q, key_qd, key_u = jax.random.split(key, 3)
    q_small = random_q(reduced, key_q, scale=0.05)
    qd_small = random_q(reduced, key_qd, scale=0.1)
    u_small = random_q(reduced, key_u, scale=0.2)

    q_full, qd_full, u_full = B @ q_small, B @ qd_small, B @ u_small

    n_full_strains = int(full.num_strains)
    n_small_act = int(reduced.num_active_strains.item())
    # Inertia
    B_full_full = full.inertia_matrix(q_full)
    B_small_expected = B.T @ B_full_full @ B
    B_small = reduced.inertia_matrix(q_small)
    assert B_full_full.shape == (n_full_strains, n_full_strains)
    assert B_small.shape == (n_small_act, n_small_act)
    assert_allclose(B_small, B_small_expected, rtol=RTOL, atol=ATOL)

    # Coriolis
    C_full_full = full.coriolis_matrix(q_full, qd_full)
    C_small_expected = B.T @ C_full_full @ B
    C_small = reduced.coriolis_matrix(q_small, qd_small)
    assert C_full_full.shape == (n_full_strains, n_full_strains)
    assert C_small.shape == (n_small_act, n_small_act)
    assert_allclose(C_small, C_small_expected, rtol=RTOL, atol=ATOL)

    # Gravity
    G_full_full = full.gravitational_force(q_full)
    G_small_expected = B.T @ G_full_full
    G_small = reduced.gravitational_force(q_small)
    assert G_full_full.shape == (n_full_strains,)
    assert G_small.shape == (n_small_act,)
    assert_allclose(G_small, G_small_expected, rtol=RTOL, atol=ATOL)

    # Stiffness
    K_full_full = full.stiffness_matrix()
    K_small_expected = B.T @ K_full_full @ B
    K_small = reduced.stiffness_matrix()
    assert K_full_full.shape == (n_full_strains, n_full_strains)
    assert K_small.shape == (n_small_act, n_small_act)
    assert_allclose(K_small, K_small_expected, rtol=RTOL, atol=ATOL)

    # Damping
    D_full_full = full.damping_matrix(q_full)
    D_small_expected = B.T @ D_full_full @ B
    D_small = reduced.damping_matrix(q_small)
    assert D_full_full.shape == (n_full_strains, n_full_strains)
    assert D_small.shape == (n_small_act, n_small_act)
    assert_allclose(D_small, D_small_expected, rtol=RTOL, atol=ATOL)

    # Actuation
    A_full = full.actuation_matrix(q_full)
    A_small = reduced.actuation_matrix(q_small)
    assert A_full.shape == (
        int(full.num_active_strains.item()),
        int(full.num_actuators),
    )
    assert A_small.shape == (n_small_act, int(reduced.num_actuators))
    assert_allclose(A_small, B.T @ A_full @ B, rtol=RTOL, atol=ATOL)

    tau_u_full = full.actuation_force(q_full, u_full)
    tau_u_small = reduced.actuation_force(q_small, u_small)
    assert tau_u_full.shape == (int(full.num_active_strains.item()),)
    assert tau_u_small.shape == (n_small_act,)
    assert_allclose(B.T @ tau_u_full, tau_u_small, rtol=RTOL, atol=ATOL)

    # Energies
    U_full = full.potential_energy(q_full)
    U_small = reduced.potential_energy(q_small)
    assert jnp.ndim(U_full) == 0
    assert jnp.ndim(U_small) == 0
    assert_allclose(U_full, U_small, rtol=RTOL, atol=ATOL)

    # Forward dynamics consistency: ydot_full == [B @ qd_small, B @ qdd_small]
    y_small = jnp.concatenate([q_small, qd_small])
    y_full = jnp.concatenate([q_full, qd_full])

    yd_small = reduced.forward_dynamics(0.0, y_small, (u_small,))
    yd_full = full.forward_dynamics(0.0, y_full, (u_full,))

    assert yd_small.shape == (2 * n_small_act,)
    assert yd_full.shape == (2 * n_full_strains,)

    qdot_small, qdd_small = jnp.split(yd_small, 2)
    qdot_full, qdd_full_out = jnp.split(yd_full, 2)

    assert_allclose(qdot_small, qd_small, rtol=RTOL, atol=ATOL)
    assert_allclose(qdot_full, qd_full, rtol=RTOL, atol=ATOL)

    tau_el_small = reduced.elastic_force(q_small)
    qdd_small_expected = jnp.linalg.solve(
        B_small,
        tau_u_small - C_small @ qd_small - G_small - tau_el_small - D_small @ qd_small,
    )
    assert_allclose(qdd_small, qdd_small_expected, rtol=RTOL, atol=ATOL)

    tau_el_full = full.elastic_force(q_full)
    qdd_full_expected = jnp.linalg.solve(
        B_full_full,
        tau_u_full
        - C_full_full @ qd_full
        - G_full_full
        - tau_el_full
        - D_full_full @ qd_full,
    )
    assert_allclose(qdd_full_out, qdd_full_expected, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
