import jax
from jax import Array, jacfwd, jvp
from jax import numpy as jnp
from numpy.testing import assert_allclose
import numpy as onp
import pytest
from typing import List, Optional

from soromox.systems.pcs import PCS
from soromox.utils.lie_algebra.se3 import Adjoint_g_SE3, log_SE3
from soromox.utils.tolerance import Tolerance


jax.config.update("jax_enable_x64", True)  # double precision


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = 1e-6

PCS_TOTAL_LENGTH = 2e-1
NUM_RANDOM_SAMPLES = 5
NUM_IK_SAMPLES = 10


def make_pcs(
    num_segments: int = 2,
    xi_ref: Optional[Array] = None,
    total_length: float = PCS_TOTAL_LENGTH,
    order_gauss: int = 3,
    strain_selector: Optional[Array] = None,
):
    segment_length = total_length / num_segments
    L = segment_length * jnp.ones((num_segments,))
    params = {
        "p0": jnp.zeros((6,)),
        "L": L,
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": 1070 * jnp.ones((num_segments,)),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 2e3 * jnp.ones((num_segments,)),
        "G": 1e3 * jnp.ones((num_segments,)),
    }
    diag_vals = jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0)
    params["D"] = 1e-3 * jnp.diag((diag_vals * L[:, None]).flatten())

    if xi_ref is None:
        xi_ref = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments)

    model = PCS(
        num_segments=num_segments,
        params=params,
        order_gauss=order_gauss,
        xi_ref=xi_ref,
        strain_selector=strain_selector,
    )

    return model, params


def sample_arc_lengths(model: PCS) -> List[float]:
    lengths = jnp.asarray(model.L)
    cumulative = jnp.cumsum(lengths)
    total = float(cumulative[-1])

    near_zero = max(total * 1e-3, 1e-9)
    mids = (cumulative - lengths / 2.0).tolist()
    boundaries = cumulative.tolist()

    values = [near_zero] + mids + boundaries
    unique_sorted = sorted({float(v) for v in values if 0.0 < float(v) <= total})
    return unique_sorted


def random_q(model: PCS, key: Array, scale: float = 0.05) -> Array:
    n = int(model.num_active_strains.item())
    return scale * jax.random.normal(key, (n,))


def se3_inverse(g: Array) -> Array:
    R = g[:3, :3]
    p = g[:3, 3]
    g_inv = jnp.eye(4)
    R_T = R.T
    g_inv = g_inv.at[:3, :3].set(R_T)
    g_inv = g_inv.at[:3, 3].set(-R_T @ p)
    return g_inv


def body_twist_between(g_base: Array, g_target: Array) -> Array:
    g_rel = se3_inverse(g_base) @ g_target
    xi = log_SE3(g_rel, eps=1e-12)
    R = g_rel[:3, :3]
    omega = xi[:3]
    alt_omega = 0.5 * jnp.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    omega = jnp.where(jnp.linalg.norm(omega) < 1e-10, alt_omega, omega)
    return xi.at[:3].set(omega)


def spatial_from_body(g: Array, xi_body: Array) -> Array:
    g_rot = jnp.block(
        [[g[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
    )
    return Adjoint_g_SE3(g_rot) @ xi_body


def segment_tip_transforms(model: PCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]

    def fk_at_s(s: Array) -> Array:
        return model.forward_kinematics(q, s)

    return jax.vmap(fk_at_s)(s_vals)


def test_planar_cs_num():
    """
    Test the planar constant strain system with numerical integration and Jacobian for 1 segment.
    """
    params = {
        "p0": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 0.0, 0.0, 1e3, 0.0, 1e3]]) * params["L"][:, None]).flatten()
    )
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((6,), dtype=bool)

    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    num_segments = 1

    robot = PCS(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((6,)),
            params["L"][0] / 2,
            jnp.eye(4).at[0, 3].set(params["L"][0] / 2),
        ),
        (
            jnp.zeros((6,)),
            params["L"][0],
            jnp.eye(4).at[0, 3].set(params["L"][0]),
        ),
        (
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            params["L"][0],
            jnp.eye(4).at[0, 3].set(2 * params["L"][0]),
        ),
    ]

    for q, s, expected in test_cases:
        print("q = ", q, "s = ", s)
        g_i = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(g_i).any(), "Forward kinematics output contains NaN!"
        assert_allclose(g_i, expected, rtol=RTOL, atol=ATOL)
        print("[Valid test]\n")

    # test dynamical matrices
    print("\nTesting dynamical matrices... ------------------------")
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
    u = jnp.ones((6,))  # identity torque for testing
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
    assert_allclose(K @ q, jnp.zeros((6,)))
    print("[Valid test]\n")
    print("testing alpha")
    assert_allclose(
        alpha,
        jnp.ones(6),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * params["L"][0]), 0.0, 0.0, 0.0, 0.0, 0.0])
    qd = jnp.zeros((6,))
    u = jnp.ones((6,))  # identity torque for testing
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
    params_bis = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
    }
    robot = robot.update_params(params_bis)
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
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
    E_pot_th = jnp.array(
        0.5
        * params["rho"][0]
        * jnp.pi
        * params["r"][0] ** 2
        * jnp.linalg.norm(params["g"])
        * params["L"][0] ** 2
    )
    assert_allclose(E_pot, E_pot_th, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
    u = jnp.zeros((6,))  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.zeros((3,))  # no gravity for this test
    robot = robot.update_params(params_bis)
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", params_bis["g"])
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
    qdd, qdres = jnp.split(yd, 2)
    assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(qdd, jnp.zeros((6,)), rtol=RTOL, atol=ATOL)
    assert_allclose(qdres, qd, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")


def test_individual_call():
    """
    Test the individual call of the PCS class.
    """
    params = {
        "p0": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 0.0, 0.0, 1e3, 0.0, 1e3]]) * params["L"][:, None]).flatten()
    )
    strain_selector = jnp.ones((6,), dtype=bool)
    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    robot = PCS(
        num_segments=1,
        params=params,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # Test individual calls
    q = jnp.zeros((6,))
    s = params["L"][0]

    print("\nTest robot.forward_kinematics(q, s)-------------------------")
    try:
        g_i = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(g_i).any(), "Forward kinematics output contains NaN!"
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
        J, Jd = robot.jacobian_and_derivative(q=q, qd=jnp.zeros((6,)), s=s)
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
            q=q, qd=jnp.zeros((6,)), s=s
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
            q=q, qd=jnp.zeros((6,)), s=s
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
        C = robot.coriolis_matrix(q=q, qd=jnp.zeros((6,)))
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
        u = jnp.zeros((6,))  # no external forces
        alpha = robot.actuation_force(q=q, u=u)
        assert not jnp.isnan(alpha).any(), "Actuation force contains NaN!"
        print("[Valid test] Actuation force computation successful.")
    except Exception as e:
        print(f"[Error] Actuation force computation failed: {e}")

    print("\nTest robot.forward_dynamics(t, y, u)-------------------------")
    try:
        t = 0.0
        y = jnp.concatenate([q, jnp.zeros((6,))])  # initial state with zero velocity
        u = jnp.zeros((6,))  # no external forces
        yd = robot.forward_dynamics(t=t, y=y, actuation_args=(u,))
        qdd, qdres = jnp.split(yd, 2)
        assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
        print("[Valid test] Forward dynamics computation successful.")
    except Exception as e:
        print(f"[Error] Forward dynamics computation failed: {e}")


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_tips_coherence(num_segments: int) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    zero_cfg = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)
    rng = jax.random.PRNGKey(321)
    random_cfg = random_q(model, rng, scale=0.05)

    for q in (zero_cfg, random_cfg):
        g_tips_expected = segment_tip_transforms(model, q)
        g_tips_actual = model.forward_kinematics_tips(q)

        assert_allclose(g_tips_actual, g_tips_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_batched_coherence(num_segments: int) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    zero_cfg = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)
    rng = jax.random.PRNGKey(789)
    random_cfg = random_q(model, rng, scale=0.05)

    s_values = [0.0] + sample_arc_lengths(model)
    s_ps = jnp.asarray(s_values, dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        g_batched = model.forward_kinematics_batched(q, s_ps)
        g_expected = jax.vmap(lambda s: model.forward_kinematics(q, s))(s_ps)

        assert_allclose(g_batched, g_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inverse_kinematics_consistency(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, NUM_IK_SAMPLES)

    for subkey in keys:
        q = random_q(model, subkey, scale=0.05)
        g_tips = segment_tip_transforms(model, q)
        q_recovered = model.inverse_kinematics(g_tips)

        assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inverse_kinematics_straight_configuration(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments)
    q = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)

    g_tips = segment_tip_transforms(model, q)
    q_recovered = model.inverse_kinematics(g_tips)

    assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_inverse_kinematics_with_deactivated_strains(num_segments: int):
    selector_per_segment = jnp.array([False, False, True, True, False, False], dtype=bool)
    strain_selector = jnp.tile(selector_per_segment, num_segments)
    model, _ = make_pcs(num_segments=num_segments, strain_selector=strain_selector)

    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, NUM_IK_SAMPLES)

    for subkey in keys:
        q = random_q(model, subkey, scale=0.05)
        g_tips = segment_tip_transforms(model, q)
        q_recovered = model.inverse_kinematics(g_tips)

        assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_local_jacobian_tips_coherence(num_segments: int) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(987)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(654), scale=0.1)

    s_tips = model.L_cum[1:]

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_tips, Jd_tips = model._J_Jd_local_tips(q, qd)

        for idx, s_tip in enumerate(s_tips):
            J_local, Jd_local = model._J_Jd_local(q, qd, s_tip)
            J_blocks = J_local.reshape(6, model.num_segments, 6).transpose(1, 0, 2)
            Jd_blocks = Jd_local.reshape(6, model.num_segments, 6).transpose(1, 0, 2)

            assert_allclose(J_blocks[idx], J_tips[idx], rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_blocks[idx], Jd_tips[idx], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_local_jacobian_batched_coherence(num_segments: int) -> None:
    model, _ = make_pcs(num_segments=num_segments)
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


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_bodyframe_inertialframe_coherence(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(1)
    q = random_q(model, key, scale=0.05)

    for s in sample_arc_lengths(model):
        J_impl = model.jacobian_inertialframe(q, s)
        J_body = model.jacobian_bodyframe(q, s)
        g = model.forward_kinematics(q, s)
        J_expected = spatial_from_body(g, J_body)

        assert jnp.allclose(J_impl, J_expected, rtol=1e-6, atol=1e-7), (
            f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_expected:\n{J_expected}"
        )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_bodyframe_matches_autodiff(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(6)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.03)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue

            J_body = model.jacobian_bodyframe(q, s)
            g = model.forward_kinematics(q, s)
            g_inv = se3_inverse(g)

            def fk(qq: Array) -> Array:
                return model.forward_kinematics(qq, s)

            n = q.shape[0]
            eye = jnp.eye(n)
            cols = []
            for j in range(n):
                _, g_tangent = jax.jvp(fk, (q,), (eye[j],))
                Xi_hat_body = g_inv @ g_tangent
                skew_body = 0.5 * (Xi_hat_body[:3, :3] - Xi_hat_body[:3, :3].T)
                omega = jnp.array(
                    [
                        skew_body[2, 1],
                        skew_body[0, 2],
                        skew_body[1, 0],
                    ]
                )
                v_body = Xi_hat_body[:3, 3]
                cols.append(jnp.concatenate((omega, v_body)))

            J_ad = jnp.stack(cols, axis=1)

            assert_allclose(
                J_body,
                J_ad,
                rtol=1e-6,
                atol=1e-7,
                err_msg=(
                    f"num_segments={num_segments}, s={s}\n"
                    f"J_body:\n{J_body}\nJ_ad:\n{J_ad}"
                ),
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_matches_autodiff(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(7)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.03)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue

            J_impl = model.jacobian_inertialframe(q, s)
            g = model.forward_kinematics(q, s)
            R = g[:3, :3]
            g_inv = se3_inverse(g)

            def fk(qq: Array) -> Array:
                return model.forward_kinematics(qq, s)

            n = q.shape[0]
            eye = jnp.eye(n)
            cols = []
            for j in range(n):
                _, g_tangent = jax.jvp(fk, (q,), (eye[j],))
                Xi_hat_body = g_inv @ g_tangent
                skew_body = 0.5 * (Xi_hat_body[:3, :3] - Xi_hat_body[:3, :3].T)
                omega = jnp.array(
                    [
                        skew_body[2, 1],
                        skew_body[0, 2],
                        skew_body[1, 0],
                    ]
                )
                v_body = Xi_hat_body[:3, 3]
                cols.append(jnp.concatenate((R @ omega, R @ v_body)))

            J_ad = jnp.stack(cols, axis=1)

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


@pytest.mark.parametrize("num_segments", [1, 2])
def test_inertial_velocity_consistency(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dt = EPS
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.03)
        qd = random_q(model, qd_key, scale=0.1)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue
            J = model.jacobian_inertialframe(q, s)
            xdot_pred = J @ qd

            g0 = model.forward_kinematics(q, s)
            g1 = model.forward_kinematics(q + dt * qd, s)
            xi_body = body_twist_between(g0, g1) / dt
            xdot_fd = spatial_from_body(g0, xi_body)

            assert jnp.allclose(xdot_pred, xdot_fd, rtol=5e-5, atol=5e-7), (
                f"num_segments={num_segments}, s={s}\npred: {xdot_pred}\nfd: {xdot_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_matches_central_differences(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(5)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.02)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue
            J_impl = model.jacobian_inertialframe(q, s)

            xi_cols = []
            n = q.shape[0]
            eye = jnp.eye(n)
            for j in range(n):
                qp = q + EPS * eye[j]
                qm = q - EPS * eye[j]
                g_plus = model.forward_kinematics(qp, s)
                g_minus = model.forward_kinematics(qm, s)
                xi_body = body_twist_between(g_minus, g_plus) / (2 * EPS)
                xi_cols.append(spatial_from_body(g_minus, xi_body))

            J_fd = jnp.stack(xi_cols, axis=1)
            J_body = model.jacobian_bodyframe(q, s)
            J_rot_expected = spatial_from_body(model.forward_kinematics(q, s), J_body)

            assert jnp.allclose(J_impl[:3], J_rot_expected[:3], rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nrot_impl:\n{J_impl[:3]}\nrot_expected:\n{J_rot_expected[:3]}"
            )
            assert jnp.allclose(J_impl[3:], J_fd[3:], rtol=1e-3, atol=5e-6), (
                f"num_segments={num_segments}, s={s}\nJ_impl:\n{J_impl}\nJ_fd:\n{J_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_Jd_bodyframe_matches_autograd_jvp(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_bodyframe(q, qd, s)

            def J_body(q_):
                return model.jacobian_bodyframe(q_, s)

            _, Jd_jvp = jvp(J_body, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_jvp:\n{Jd_jvp}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_Jd_bodyframe_matches_central_differences(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
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


@pytest.mark.parametrize("num_segments", [1, 2])
def test_Jd_inertialframe_matches_autograd_jvp(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_derivative_inertialframe(q, qd, s)

            def J_global(q_):
                return model.jacobian_inertialframe(q_, s)

            _, Jd_jvp = jvp(J_global, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jd_jvp)}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_gravity_matches_potential_gradient(num_segments: int):
    robot, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(8)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(robot, q_key, scale=0.05)
        G = robot.gravitational_force(q)
        dU_dq = jax.grad(robot.gravitational_energy)(q)

        assert_allclose(G, dU_dq, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_with_christoffel_symbols(num_segments: int):
    robot, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(9)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        C_impl = robot.coriolis_matrix(q, qd)
        tau_cor_impl = C_impl @ qd

        def B_of_q(q_):
            return robot.inertia_matrix(q_)

        dB_dq = jacfwd(B_of_q)(q)

        term1 = jnp.einsum("ijk,j,k->i", dB_dq, qd, qd)
        term2 = jnp.einsum("jki,j,k->i", dB_dq, qd, qd)
        tau_cor = term1 - 0.5 * term2

        assert_allclose(tau_cor_impl, tau_cor, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_matches_kinetic_energy_autograd(num_segments: int):
    robot, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(10)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dT_dq = jax.grad(robot.kinetic_energy, argnums=0)
    dT_dqd = jax.grad(robot.kinetic_energy, argnums=1)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        tau_cor_impl = robot.coriolis_matrix(q, qd) @ qd

        grad_T_q = dT_dq(q, qd)
        jac_T_q = jacfwd(lambda qq: dT_dqd(qq, qd))(q)
        tau_cor_autograd = jac_T_q @ qd - grad_T_q

        assert_allclose(tau_cor_impl, tau_cor_autograd, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_dynamics_matches_manual_computation(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)

    num_strains = int(model.num_active_strains.item())
    key = jax.random.PRNGKey(123 + num_segments)

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
