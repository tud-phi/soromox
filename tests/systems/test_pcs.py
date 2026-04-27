import jax
import numpy as onp
import pytest
from jax import Array, jacfwd, jacrev, jvp
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import PCS, CrossSectionGeometry
from soromox.utils.integration import scale_interior_gaussian_quadrature
from soromox.utils.lie_algebra.se3 import Adjoint_g_SE3, log_SE3
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)  # double precision


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
EPS = float(jnp.finfo(jnp.float64).eps)

PCS_TOTAL_LENGTH = 2e-1
NUM_RANDOM_SAMPLES = 5
NUM_IK_SAMPLES = 10


def make_pcs(
    num_segments: int = 2,
    xi_ref: Array | None = None,
    total_length: float = PCS_TOTAL_LENGTH,
    order_gauss: int = 3,
    strain_selector: Array | None = None,
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
    diag_vals = jnp.repeat(
        jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
    )
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


def sample_arc_lengths(model: PCS) -> list[float]:
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


def test_constant_strain_call():
    """
    Test the constant strain system with numerical integration and Jacobian for 1 segment.
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
    D = robot.damping_matrix(q)
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


def test_public_pcs_accessors_and_geometry_helpers() -> None:
    model, params = make_pcs(num_segments=2)
    q = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)

    assert model.is_planar is False
    assert_allclose(model.length, jnp.sum(params["L"]), rtol=RTOL, atol=ATOL)
    assert_allclose(model.segment_length, params["L"], rtol=RTOL, atol=ATOL)

    s_second = params["L"][0] + 0.25 * params["L"][1]
    segment_idx, s_local = model.classify_segment(s_second)
    assert int(segment_idx) == 1
    assert_allclose(s_local, 0.25 * params["L"][1], rtol=RTOL, atol=ATOL)

    tag, geom = model.cross_section_geometry(q, s_second)
    assert int(tag) == CrossSectionGeometry.CIRCULAR
    assert_allclose(geom, jnp.array([params["r"][1]]), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_tips_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    zero_cfg = jnp.zeros((int(model.num_active_strains.item()),), dtype=jnp.float64)
    rng = jax.random.PRNGKey(321)
    random_cfg = random_q(model, rng, scale=0.05)

    for q in (zero_cfg, random_cfg):
        g_tips_expected = segment_tip_transforms(model, q)
        g_tips_actual = model.forward_kinematics_tips(q)

        assert_allclose(g_tips_actual, g_tips_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
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
def test_forward_inverse_kinematics_consistency(num_segments: int):
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
    print("g_tips:\n", g_tips)
    q_recovered = model.inverse_kinematics(g_tips)
    print("q_recovered:\n", q_recovered)

    assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_inverse_kinematics_with_deactivated_strains(num_segments: int):
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    strain_selector = jnp.tile(selector_per_segment, num_segments)
    model, _ = make_pcs(num_segments=num_segments, strain_selector=strain_selector)

    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, NUM_IK_SAMPLES)

    for subkey in keys:
        q = random_q(model, subkey, scale=0.05)
        g_tips = segment_tip_transforms(model, q)
        q_recovered = model.inverse_kinematics(g_tips)

        assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_J_local_tips_matches_pointwise_evaluation(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    q = random_q(model, jax.random.PRNGKey(5), scale=0.03)

    J_tips_ = model._J_local_tips(q)
    s_tips = model.L_cum[1:]

    for idx, s in enumerate(s_tips):
        if s < 1e-3:
            continue

        J_tip_batch = J_tips_[idx] @ model.B_xi
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
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(9876)
    q_random = random_q(model, rng, scale=0.05)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q in (zero_cfg, q_random):
        J_batch = model.jacobian_bodyframe_batched(q, s_points)

        for idx, s_val in enumerate(s_points):
            J_single = model.jacobian_bodyframe(q, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)


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
def test_inertial_velocity_matches_central_differences(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dt = 1e-6
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.03)
        qd = random_q(model, qd_key, scale=0.1)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue
            J = model.jacobian_inertialframe(q, s)
            xd_pred = J @ qd

            g0 = model.forward_kinematics(q, s)
            g1 = model.forward_kinematics(q + dt * qd, s)
            xi_body = body_twist_between(g0, g1) / dt
            xd_fd = spatial_from_body(g0, xi_body)

            assert jnp.allclose(xd_pred, xd_fd, rtol=5e-5, atol=5e-7), (
                f"num_segments={num_segments}, s={s}\npred: {xd_pred}\nfd: {xd_fd}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_matches_central_differences(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(5)

    # delta for finite differences
    delta = 1e-6
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
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                g_plus = model.forward_kinematics(qp, s)
                g_minus = model.forward_kinematics(qm, s)
                xi_body = body_twist_between(g_minus, g_plus) / (2 * delta)
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
def test_jacobian_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments)
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
    model, _ = make_pcs(num_segments=num_segments)
    q = random_q(model, jax.random.PRNGKey(7891), scale=0.05)

    s_tips = model.L_cum[1:]
    J_tips = model.jacobian_tips(q)
    J_expected = jax.vmap(lambda s: model.jacobian_inertialframe(q, s))(s_tips)

    assert J_tips.shape == (num_segments, 6, int(model.num_active_strains.item()))
    assert_allclose(J_tips, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_derivative_bodyframe_matches_autograd_jvp(num_segments: int):
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
def test_jacobian_derivative_bodyframe_matches_central_differences(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
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


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_Jd_local_tips_matches_pointwise_evaluation(num_segments: int) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(987)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(654), scale=0.1)

    s_tips = model.L_cum[1:]

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_local_tips_, Jd_local_tips_ = model._J_Jd_local_tips(q, qd)

        for idx, s_tip in enumerate(s_tips):
            J_local, Jd_local = model.jacobian_and_derivative_bodyframe(q, qd, s_tip)

            assert_allclose(
                J_local, J_local_tips_[idx] @ model.B_xi, rtol=RTOL, atol=ATOL
            )
            assert_allclose(
                Jd_local, Jd_local_tips_[idx] @ model.B_xi, rtol=RTOL, atol=ATOL
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_and_derivative_bodyframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(321)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(4321), scale=0.1)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = model.jacobian_and_derivative_bodyframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_derivative_bodyframe(q, qd, s_val)
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_derivative_inertialframe_matches_autograd_jvp(num_segments: int):
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
def test_jacobian_and_derivative_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments)
    dof = int(model.num_active_strains.item())

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    rng = jax.random.PRNGKey(7890)
    q_random = random_q(model, rng, scale=0.05)
    qd_random = random_q(model, jax.random.PRNGKey(9876), scale=0.1)

    s_points = jnp.asarray(sample_arc_lengths(model), dtype=jnp.float64)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = model.jacobian_and_derivative_inertialframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_derivative_inertialframe(
                q, qd, s_val
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


def test_public_pcs_jacobian_aliases_match_inertialframe_methods() -> None:
    model, _ = make_pcs(num_segments=2)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(2027))
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

    J, Jd = model.jacobian_and_derivative(q, qd, s)
    J_expected, Jd_expected = model.jacobian_and_derivative_inertialframe(q, qd, s)
    assert_allclose(J, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd, Jd_expected, rtol=RTOL, atol=ATOL)

    J_batch, Jd_batch = model.jacobian_and_derivative_batched(q, qd, s_ps)
    J_batch_expected, Jd_batch_expected = (
        model.jacobian_and_derivative_inertialframe_batched(q, qd, s_ps)
    )
    assert_allclose(J_batch, J_batch_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_batch, Jd_batch_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inertia_matrix_matches_kinetic_energy_autodiff(num_segments: int):
    robot, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(2)
    q_keys = jax.random.split(key, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key + 1, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        B_impl = robot.inertia_matrix(q)

        def T_of_qd(qd_):
            return robot.kinetic_energy(q, qd_)

        dT_dqdsq = jacfwd(jax.grad(T_of_qd))(qd)
        B_expected = dT_dqdsq

        assert_allclose(B_impl, B_expected, rtol=RTOL, atol=ATOL)


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


@pytest.mark.parametrize("num_segments", [1, 2, 3, 4])
def test_gravity_matches_potential_gradient(num_segments: int):
    robot, _ = make_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(8)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(robot, q_key, scale=0.05)
        G = robot.gravitational_force(q)
        dU_G_dq = jacfwd(robot.gravitational_energy)(q)
        print("q:\n", q)
        print("G:\n", G)
        print("dU_G_dq:\n", dU_G_dq)

        assert_allclose(G, dU_G_dq, rtol=RTOL, atol=ATOL)


def test_pcs_gravitational_energy_gradient_matches_force() -> None:
    model, _ = make_pcs(num_segments=1)
    q = random_q(model, jax.random.PRNGKey(314), scale=0.02)

    dU_dq = jax.grad(lambda q_: model.gravitational_energy(q_))(q)

    assert_allclose(dU_dq, model.gravitational_force(q), rtol=RTOL, atol=ATOL)


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
        (False, False, True, True, False, False),
        (False, False, False, True, False, False),
    ],
)
def test_active_quadrature_kinematics_matches_existing_batched_path(
    num_segments: int, selector_per_segment: tuple[bool, ...] | None
):
    strain_selector = (
        None
        if selector_per_segment is None
        else jnp.tile(jnp.asarray(selector_per_segment, dtype=bool), num_segments)
    )
    model, _ = make_pcs(num_segments=num_segments, strain_selector=strain_selector)
    dof = int(model.num_active_strains.item())

    key_q, key_qd = jax.random.split(jax.random.PRNGKey(6123))
    q = random_q(model, key_q, scale=0.05)
    qd = random_q(model, key_qd, scale=0.1)

    weights, g_quads, J_quads, Jd_quads = model._active_quadrature_kinematics(q, qd)
    Xs_scaled, weights_expected = jax.vmap(
        scale_interior_gaussian_quadrature, in_axes=(None, None, 0, 0)
    )(model.Xs, model.Ws, model.L_cum[:-1], model.L_cum[1:])
    s_points = Xs_scaled.reshape(-1)
    num_inner = model.num_gauss_points - 2

    g_expected = model.forward_kinematics_batched(q, s_points).reshape(
        num_segments, num_inner, 4, 4
    )
    J_full, Jd_full = model._J_Jd_local_batched(q, qd, s_points)
    J_expected = (J_full @ model.B_xi).reshape(num_segments, num_inner, 6, dof)
    Jd_expected = (Jd_full @ model.B_xi).reshape(num_segments, num_inner, 6, dof)

    assert_allclose(weights, weights_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(g_quads, g_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(J_quads, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_quads, Jd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 3])
@pytest.mark.parametrize(
    "selector_per_segment",
    [
        None,
        (False, False, True, True, False, False),
        (False, False, False, True, False, False),
    ],
)
def test_active_quadrature_forward_dynamics_terms_match_public_matrices(
    num_segments: int, selector_per_segment: tuple[bool, ...] | None
):
    strain_selector = (
        None
        if selector_per_segment is None
        else jnp.tile(jnp.asarray(selector_per_segment, dtype=bool), num_segments)
    )
    model, _ = make_pcs(num_segments=num_segments, strain_selector=strain_selector)
    dof = int(model.num_active_strains.item())

    zero_q = jnp.zeros((dof,), dtype=jnp.float64)
    zero_qd = jnp.zeros((dof,), dtype=jnp.float64)
    key_q, key_qd, key_u, key_tau = jax.random.split(jax.random.PRNGKey(6124), 4)
    random_q_ = random_q(model, key_q, scale=0.05)
    random_qd = random_q(model, key_qd, scale=0.1)
    u = random_q(model, key_u, scale=0.2)
    tau_ext = random_q(model, key_tau, scale=0.03)

    for q, qd in ((zero_q, zero_qd), (random_q_, random_qd)):
        B, Cqd, G = model._active_quadrature_forward_dynamics_terms(q, qd)
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


def test_cached_constant_matrices_refresh_after_update_params():
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    model, _ = make_pcs(
        num_segments=2,
        strain_selector=jnp.tile(selector_per_segment, 2),
    )

    updated = model.update_params(
        {
            "r": 1.1 * model.r,
            "rho": 0.9 * model.rho,
            "E": 1.25 * model.E,
            "G": 0.75 * model.G,
            "D": 2.0 * model.D,
        }
    )
    segment_ids = jnp.arange(updated.num_segments)
    expected_M = jax.vmap(updated._compute_local_mass_matrix)(segment_ids)
    expected_K_full = updated._compute_stiffness_full_matrix()
    expected_K = updated.B_xi.T @ expected_K_full @ updated.B_xi
    expected_D_full = updated.D
    expected_D = updated.B_xi.T @ expected_D_full @ updated.B_xi

    assert_allclose(updated.M_segments, expected_M, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.K_full, expected_K_full, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.K, expected_K, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.D_full, expected_D_full, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.D_active, expected_D, rtol=RTOL, atol=ATOL)
    assert_allclose(updated.stiffness_matrix(), expected_K, rtol=RTOL, atol=ATOL)
    assert_allclose(
        updated.damping_matrix(jnp.zeros(updated.num_dofs)),
        expected_D,
        rtol=RTOL,
        atol=ATOL,
    )


# ======================================================================================
# Strain-basis consistency tests (selection basis applied correctly across APIs)
# ======================================================================================


def _make_full_and_reduced_pcs(num_segments: int, selector_per_segment: Array):
    """Utility to build a full-DOF PCS and a reduced-DOF PCS sharing parameters.

    Returns (full_model, reduced_model, B) where B is the reduced model's basis.
    """
    strain_selector = jnp.tile(selector_per_segment, num_segments)
    full_model, params = make_pcs(num_segments=num_segments, strain_selector=None)
    reduced_model, _ = make_pcs(
        num_segments=num_segments, strain_selector=strain_selector
    )
    return full_model, reduced_model, reduced_model.B_xi


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_strain_and_kinematics(num_segments: int):
    # Select [kappa_z, sigma_x] per segment for 3D PCS
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    full, reduced, B = _make_full_and_reduced_pcs(num_segments, selector_per_segment)

    key_q = jax.random.PRNGKey(101)
    q_small = random_q(reduced, key_q, scale=0.05)
    q_full = B @ q_small

    # strain must match under mapping
    xi_small = reduced.strain(q_small)
    xi_full = full.strain(q_full)
    n_full_strains = int(full.num_strains)
    assert xi_small.shape == (n_full_strains,)
    assert xi_full.shape == (n_full_strains,)
    assert_allclose(xi_full, xi_small, rtol=RTOL, atol=ATOL)

    # forward kinematics at several s values (incl. tips) and batched
    s_values = jnp.asarray([0.0] + sample_arc_lengths(full), dtype=jnp.float64)
    for s in s_values:
        g_small = reduced.forward_kinematics(q_small, s)
        g_full = full.forward_kinematics(q_full, s)
        assert g_small.shape == (4, 4)
        assert g_full.shape == (4, 4)
        assert_allclose(g_full, g_small, rtol=RTOL, atol=ATOL)

    g_tips_small = reduced.forward_kinematics_tips(q_small)
    g_tips_full = full.forward_kinematics_tips(q_full)
    assert g_tips_small.shape == (num_segments, 4, 4)
    assert g_tips_full.shape == (num_segments, 4, 4)
    assert_allclose(g_tips_full, g_tips_small, rtol=RTOL, atol=ATOL)

    g_batched_small = reduced.forward_kinematics_batched(q_small, s_values)
    g_batched_full = full.forward_kinematics_batched(q_full, s_values)
    N = s_values.shape[0]
    assert g_batched_small.shape == (N, 4, 4)
    assert g_batched_full.shape == (N, 4, 4)
    assert_allclose(g_batched_full, g_batched_small, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_jacobians_and_derivatives(num_segments: int):
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    full, reduced, B = _make_full_and_reduced_pcs(num_segments, selector_per_segment)

    key = jax.random.PRNGKey(202)
    key_q, key_qd = jax.random.split(key)
    q_small = random_q(reduced, key_q, scale=0.05)
    qd_small = random_q(reduced, key_qd, scale=0.1)
    q_full, qd_full = B @ q_small, B @ qd_small

    # points to test (midpoints + tips)
    s_points = jnp.asarray(sample_arc_lengths(full), dtype=jnp.float64)
    n_full_strains = int(full.num_strains)

    # Body-frame Jacobian: J_small == J_full @ B
    n_small_act = int(reduced.num_active_strains.item())
    for s in s_points:
        Jb_small = reduced.jacobian_bodyframe(q_small, s)
        Jb_full = full.jacobian_bodyframe(q_full, s)
        assert Jb_small.shape == (6, n_small_act)
        assert Jb_full.shape == (6, int(full.num_active_strains.item()))
        assert (Jb_full @ B).shape == (6, n_small_act)
        assert_allclose(Jb_full @ B, Jb_small, rtol=RTOL, atol=ATOL)

    # Inertial-frame Jacobian
    for s in s_points:
        Ji_small = reduced.jacobian_inertialframe(q_small, s)
        Ji_full = full.jacobian_inertialframe(q_full, s)
        assert Ji_small.shape == (6, n_small_act)
        assert Ji_full.shape == (6, int(full.num_active_strains.item()))
        assert (Ji_full @ B).shape == (6, n_small_act)
        assert_allclose(Ji_full @ B, Ji_small, rtol=RTOL, atol=ATOL)

    # Body-frame (J, Jd)
    for s in s_points:
        J_small, Jd_small = reduced.jacobian_and_derivative_bodyframe(
            q_small, qd_small, s
        )
        J_full, Jd_full = full.jacobian_and_derivative_bodyframe(q_full, qd_full, s)
        assert J_small.shape == (6, n_small_act)
        assert Jd_small.shape == (6, n_small_act)
        assert J_full.shape == (6, int(full.num_active_strains.item()))
        assert Jd_full.shape == (6, int(full.num_active_strains.item()))
        assert_allclose(J_full @ B, J_small, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_full @ B, Jd_small, rtol=RTOL, atol=ATOL)

    # Inertial-frame (J, Jd)
    for s in s_points:
        J_small, Jd_small = reduced.jacobian_and_derivative_inertialframe(
            q_small, qd_small, s
        )
        J_full, Jd_full = full.jacobian_and_derivative_inertialframe(q_full, qd_full, s)
        assert J_small.shape == (6, n_small_act)
        assert Jd_small.shape == (6, n_small_act)
        assert J_full.shape == (6, int(full.num_active_strains.item()))
        assert Jd_full.shape == (6, int(full.num_active_strains.item()))
        assert_allclose(J_full @ B, J_small, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_full @ B, Jd_small, rtol=RTOL, atol=ATOL)

    # Tips inertial-frame Jacobian
    Ji_tips_small = reduced.jacobian_tips(q_small)
    Ji_tips_full = full.jacobian_tips(q_full)
    assert Ji_tips_small.shape == (num_segments, 6, n_small_act)
    assert Ji_tips_full.shape == (
        num_segments,
        6,
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
    assert Jb_batch_small.shape == (s_points.shape[0], 6, n_full_strains)
    assert Jb_batch_full.shape == (s_points.shape[0], 6, n_full_strains)
    assert_allclose(Jb_batch_full, Jb_batch_small, rtol=RTOL, atol=ATOL)

    # Tips body-frame Jacobian (internal helper, returns full-strain size)
    Jb_tips_small = reduced._J_local_tips(q_small)
    Jb_tips_full = full._J_local_tips(q_full)
    assert Jb_tips_small.shape == (num_segments, 6, n_full_strains)
    assert Jb_tips_full.shape == (num_segments, 6, n_full_strains)
    assert_allclose(Jb_tips_full, Jb_tips_small, rtol=RTOL, atol=ATOL)

    # Batched body-frame (J, Jd) internal helper (full-strain size)
    Jb_batch_small, Jbd_batch_small = reduced._J_Jd_local_batched(
        q_small, qd_small, s_points
    )
    Jb_batch_full, Jbd_batch_full = full._J_Jd_local_batched(q_full, qd_full, s_points)
    assert Jb_batch_small.shape == (s_points.shape[0], 6, n_full_strains)
    assert Jbd_batch_small.shape == (s_points.shape[0], 6, n_full_strains)
    assert Jb_batch_full.shape == (s_points.shape[0], 6, n_full_strains)
    assert Jbd_batch_full.shape == (s_points.shape[0], 6, n_full_strains)
    assert_allclose(Jb_batch_full, Jb_batch_small, rtol=RTOL, atol=ATOL)
    assert_allclose(Jbd_batch_full, Jbd_batch_small, rtol=RTOL, atol=ATOL)

    # Tips body-frame (J, Jd) internal helper (full-strain size)
    Jb_tips_small, Jbd_tips_small = reduced._J_Jd_local_tips(q_small, qd_small)
    Jb_tips_full, Jbd_tips_full = full._J_Jd_local_tips(q_full, qd_full)
    assert Jb_tips_small.shape == (num_segments, 6, n_full_strains)
    assert Jbd_tips_small.shape == (num_segments, 6, n_full_strains)
    assert Jb_tips_full.shape == (num_segments, 6, n_full_strains)
    assert Jbd_tips_full.shape == (num_segments, 6, n_full_strains)
    assert_allclose(Jb_tips_full, Jb_tips_small, rtol=RTOL, atol=ATOL)
    assert_allclose(Jbd_tips_full, Jbd_tips_small, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_dynamics_and_forces(num_segments: int):
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    full, reduced, B = _make_full_and_reduced_pcs(num_segments, selector_per_segment)

    key = jax.random.PRNGKey(303)
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
    # Expect A_small = B^T A_full B = I
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

    # qdot should match the current velocity in each model
    assert_allclose(qdot_small, qd_small, rtol=RTOL, atol=ATOL)
    assert_allclose(qdot_full, qd_full, rtol=RTOL, atol=ATOL)

    # Explicit dynamics check: qdd = B^{-1}(tau_u - C qd - G - tau_el - D qd)
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


@pytest.mark.parametrize("num_segments", [1, 2, 3, 4])
def test_forward_mode_automatic_differentiability_at_zero_configuration(
    num_segments: int,
):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    # initialize zero state
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    # initialize zero actuation
    u = jnp.zeros((model.num_actuators,), dtype=jnp.float64)
    # initialize point for forward kinematics
    s = model.L_cum[-1]

    # test differentiability of kinematics at zero configuration
    dg_dq = jacfwd(model.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacfwd(model.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacfwd(model.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe = jacfwd(model.jacobian_and_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacfwd(
        model.jacobian_and_derivative_inertialframe, argnums=0
    )(q, qd, s)

    assert not jnp.isnan(dg_dq).any(), "dg/dq contains NaN!"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), "dJ_bodyframe/dq contains NaN!"
    assert not jnp.isnan(dJ_inertialframe_dq).any(), "dJ_inertialframe/dq contains NaN!"
    assert not jnp.isnan(dJd_bodyframe).any(), "dJd_bodyframe/dq contains NaN!"
    assert not jnp.isnan(dJd_inertialframe).any(), "dJd_inertialframe/dq contains NaN!"

    # test differentiability of dynamics at zero configuration
    dB_dq = jacfwd(model.inertia_matrix)(q)
    dC_dq = jacfwd(model.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacfwd(model.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacfwd(model.gravitational_force)(q)
    dtau_el_dq = jacfwd(model.elastic_force)(q)
    dtau_u_dq = jacfwd(model.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacfwd(model.actuation_force, argnums=1)(q, u)
    dy_dy = jacfwd(model.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacfwd(model.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dB_dq).any(), "dB/dq contains NaN!"
    assert not jnp.isnan(dC_dq).any(), "dC/dq contains NaN!"
    assert not jnp.isnan(dC_dqd).any(), "dC/dqd contains NaN!"
    assert not jnp.isnan(dG_dq).any(), "dG/dq contains NaN!"
    assert not jnp.isnan(dtau_el_dq).any(), "dtau_el/dq contains NaN!"
    assert not jnp.isnan(dtau_u_dq).any(), "dtau_u/dq contains NaN!"
    assert not jnp.isnan(dtau_u_du).any(), "dtau_u/du contains NaN!"
    assert not jnp.isnan(dy_dy).any(), "dy/dy contains NaN!"
    assert not jnp.isnan(dy_du).any(), "dy/du contains NaN!"

    # test differentiability of the energy functions at zero configuration
    dT_dq = jacfwd(model.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacfwd(model.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacfwd(model.gravitational_energy)(q)
    dU_dq = jacfwd(model.potential_energy)(q)
    dE_dq = jacfwd(model.total_energy, argnums=0)(q, qd)
    dE_dqd = jacfwd(model.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any(), "dT/dq contains NaN!"
    assert not jnp.isnan(dT_dqd).any(), "dT/dqd contains NaN!"
    assert not jnp.isnan(dU_G_dq).any(), "dU_G/dq contains NaN!"
    assert not jnp.isnan(dU_dq).any(), "dU/dq contains NaN!"
    assert not jnp.isnan(dE_dq).any(), "dE/dq contains NaN!"
    assert not jnp.isnan(dE_dqd).any(), "dE/dqd contains NaN!"


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_reverse_mode_automatic_differentiability_at_zero_configuration(
    num_segments: int,
) -> None:
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    # initialize zero state
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    # initialize zero actuation
    u = jnp.zeros((model.num_actuators,), dtype=jnp.float64)
    # initialize point for forward kinematics
    s = model.L_cum[-1]

    # test differentiability of kinematics at zero configuration
    dg_dq = jacrev(model.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacrev(model.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacrev(model.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe = jacrev(model.jacobian_and_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacrev(
        model.jacobian_and_derivative_inertialframe, argnums=0
    )(q, qd, s)

    assert not jnp.isnan(dg_dq).any(), "dg/dq contains NaN!"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), "dJ_bodyframe/dq contains NaN!"
    assert not jnp.isnan(dJ_inertialframe_dq).any(), "dJ_inertialframe/dq contains NaN!"
    assert not jnp.isnan(dJd_bodyframe).any(), "dJd_bodyframe/dq contains NaN!"
    assert not jnp.isnan(dJd_inertialframe).any(), "dJd_inertialframe/dq contains NaN!"

    # test differentiability of dynamics at zero configuration
    dB_dq = jacrev(model.inertia_matrix)(q)
    dC_dq = jacrev(model.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacrev(model.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacrev(model.gravitational_force)(q)
    dtau_el_dq = jacrev(model.elastic_force)(q)
    dtau_u_dq = jacrev(model.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacrev(model.actuation_force, argnums=1)(q, u)
    dy_dy = jacrev(model.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacrev(model.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dB_dq).any(), "dB/dq contains NaN!"
    assert not jnp.isnan(dC_dq).any(), "dC/dq contains NaN!"
    assert not jnp.isnan(dC_dqd).any(), "dC/dqd contains NaN!"
    assert not jnp.isnan(dG_dq).any(), "dG/dq contains NaN!"
    assert not jnp.isnan(dtau_el_dq).any(), "dtau_el/dq contains NaN!"
    assert not jnp.isnan(dtau_u_dq).any(), "dtau_u/dq contains NaN!"
    assert not jnp.isnan(dtau_u_du).any(), "dtau_u/du contains NaN!"
    assert not jnp.isnan(dy_dy).any(), "dy/dy contains NaN!"
    assert not jnp.isnan(dy_du).any(), "dy/du contains NaN!"

    # test differentiability of the energy functions at zero configuration
    dT_dq = jacrev(model.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacrev(model.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacrev(model.gravitational_energy)(q)
    dU_dq = jacrev(model.potential_energy)(q)
    dE_dq = jacrev(model.total_energy, argnums=0)(q, qd)
    dE_dqd = jacrev(model.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any(), "dT/dq contains NaN!"
    assert not jnp.isnan(dT_dqd).any(), "dT/dqd contains NaN!"
    assert not jnp.isnan(dU_G_dq).any(), "dU_G/dq contains NaN!"
    assert not jnp.isnan(dU_dq).any(), "dU/dq contains NaN!"
    assert not jnp.isnan(dE_dq).any(), "dE/dq contains NaN!"
    assert not jnp.isnan(dE_dqd).any(), "dE/dqd contains NaN!"


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
