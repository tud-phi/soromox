import jax

jax.config.update("jax_enable_x64", True)  # use double precision in tests

from jax import jacfwd, jacrev, jvp
import jax.numpy as jnp
import numpy as onp
import pytest

from numpy.testing import assert_allclose

from soromox.systems.gvs import GVS, LinkAttributes, JointAttributes, BasisAttributes
from soromox.systems.pcs import PCS
import soromox.utils.lie_algebra as lie
from soromox.utils.tolerance import Tolerance


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
NUM_RANDOM_SAMPLES = 5


def build_matched_gvs_pcs(num_segments: int = 1, n_gauss: int = 5) -> tuple[GVS, PCS]:
    """
    Build a GVS model with constant-strain basis (monomial order 0) and fixed joints
    and a PCS model with the same physical parameters, so their predictions match.

    Note on gravity/signs:
    - PCS uses G = -∫ J^T M Ad_g^{-1} g ds
    - GVS uses G =  ∫ J^T M Ad_g^{-1} g ds
    For equivalent physical gravity (downwards), pass g=[0,0,-9.81] to PCS and
    g=[0,0, 9.81] to GVS so the resulting generalized gravity forces match.
    """
    # Physical parameters (kept simple, identical for each segment)
    Ls = 0.2 * jnp.ones((num_segments,))
    rs = 0.02 * jnp.ones((num_segments,))
    rhos = 1000.0 * jnp.ones((num_segments,))
    E = 1e6 * jnp.ones((num_segments,))
    nu = 0.5 * jnp.ones((num_segments,))
    # Match PCS shear modulus to GVS relation: G = E / (2(1+nu))
    Gpcs = (E / (2 * (1.0 + nu))).reshape(num_segments)

    # GVS definition: constant strain along each link, all 6 strain components enabled
    links = [
        LinkAttributes(
            section="Circular",
            E=float(E[i]),
            nu=float(nu[i]),
            rho=float(rhos[i]),
            eta=0.0,  # no damping in this cross-model comparison
            L=float(Ls[i]),
            r_i=float(rs[i]),
            r_f=float(rs[i]),
        )
        for i in range(num_segments)
    ]
    joints = [JointAttributes(jointtype="Fixed") for _ in range(num_segments)]
    bases = [
        BasisAttributes(
            basistype="Monomial",
            Bdof=[1, 1, 1, 1, 1, 1],
            Bodr=[0, 0, 0, 0, 0, 0],  # order 0 -> spatially constant strain
            xi_ref=[0, 0, 0, 1, 0, 0],
        )
        for _ in range(num_segments)
    ]
    n_gauss_list = [n_gauss for _ in range(num_segments)]

    # Gravity: pick physically consistent vectors as per note
    robot_gvs = GVS(
        links_list=links,
        joints_list=joints,
        basis_list=bases,
        n_gauss_list=n_gauss_list,
        gravity_vector=[0.0, 0.0, 9.81],
    )

    # PCS definition with identical geometry and material params
    params = {
        "p0": jnp.zeros(6),
        "L": Ls,
        "r": rs,
        "rho": rhos,
        "g": jnp.array([0.0, 0.0, -9.81]),  # sign chosen to match GVS convention
        "E": E,
        "G": Gpcs,
    }
    # zero damping for fair comparison
    params["D"] = jnp.zeros((6 * num_segments, 6 * num_segments))
    robot_pcs = PCS(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
        strain_selector=jnp.ones((6 * num_segments,), dtype=bool),
        xi_ref=jnp.tile(jnp.array([0, 0, 0, 1, 0, 0]), (num_segments, 1)).reshape(
            6 * num_segments
        ),
    )

    return robot_gvs, robot_pcs


def sample_arc_lengths(robot: GVS) -> jnp.ndarray:
    lengths = jnp.asarray(robot.V_L)
    cumulative = jnp.cumsum(lengths)
    total = float(cumulative[-1])

    near_zero = max(total * 1e-3, 1e-9)
    mids = (cumulative - lengths / 2.0).tolist()
    boundaries = cumulative.tolist()

    values = [near_zero] + mids + boundaries
    unique_sorted = sorted({float(v) for v in values if 0.0 < float(v) <= total})
    return jnp.asarray(unique_sorted, dtype=jnp.float64)


def tip_arc_lengths(robot: GVS) -> jnp.ndarray:
    return jnp.asarray(robot.V_L_cum[1:], dtype=jnp.float64)


def random_q(robot: GVS, key, scale: float = 0.05) -> jnp.ndarray:
    dof = int(robot.dof_tot_system)
    return scale * jax.random.normal(key, (dof,), dtype=jnp.float64)


def se3_inverse(g: jnp.ndarray) -> jnp.ndarray:
    R = g[:3, :3]
    p = g[:3, 3]
    R_T = R.T
    g_inv = jnp.eye(4, dtype=jnp.float64)
    g_inv = g_inv.at[:3, :3].set(R_T)
    g_inv = g_inv.at[:3, 3].set(-R_T @ p)
    return g_inv


def se3_tangent_to_body_twist(g_inv: jnp.ndarray, g_tangent: jnp.ndarray) -> jnp.ndarray:
    xi_hat_body = g_inv @ g_tangent
    skew_body = 0.5 * (xi_hat_body[:3, :3] - xi_hat_body[:3, :3].T)
    omega = jnp.array(
        [
            skew_body[2, 1],
            skew_body[0, 2],
            skew_body[1, 0],
        ],
        dtype=jnp.float64,
    )
    v_body = xi_hat_body[:3, 3]
    return jnp.concatenate((omega, v_body))


def stack_forward_kinematics(robot: GVS, q: jnp.ndarray, s_points: jnp.ndarray) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [robot.forward_kinematics(q, float(s)) for s in s_array],
        axis=0,
    )


def stack_jacobians(robot: GVS, q: jnp.ndarray, s_points: jnp.ndarray) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [robot.jacobian_bodyframe(q, float(s)) for s in s_array],
        axis=0,
    )


def stack_jacobian_derivatives(
    robot: GVS, q: jnp.ndarray, qd: jnp.ndarray, s_points: jnp.ndarray
) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [robot.jacobian_derivative_bodyframe(q, qd, float(s)) for s in s_array],
        axis=0,
    )


def gvs_jacobian_inertialframe_from_body(
    robot_gvs: GVS, q: jnp.ndarray, s: jnp.ndarray
) -> jnp.ndarray:
    """Compute inertial-frame Jacobian for GVS from its body-frame Jacobian.

    This mirrors PCS.jacobian_inertialframe: J_global = Ad_{[R,0;0,1]} @ J_body.
    """
    J_local = robot_gvs.jacobian_bodyframe(q, s)
    g_s = robot_gvs.forward_kinematics(q, s)
    g_s_wo_rot = jnp.block(
        [[g_s[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
    )
    Adj_g = lie.Adjoint_g_SE3(g_s_wo_rot)
    return Adj_g @ J_local


@pytest.mark.parametrize("num_segments", [1, 2])
def test_gvs_pcs_coherence(num_segments: int) -> None:
    print("\nTesting GVS-PCS coherence for", num_segments, "segments")
    robot_gvs, robot_pcs = build_matched_gvs_pcs(num_segments)
    n = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((n,), dtype=jnp.float64)
    zero_vel = jnp.zeros((n,), dtype=jnp.float64)

    linear_cfg = jnp.linspace(0.01, 0.01 * n, n, dtype=jnp.float64)
    linear_vel = jnp.linspace(-0.02, -0.02 * n, n, dtype=jnp.float64)

    random_cfg = random_q(robot_gvs, jax.random.PRNGKey(1234), scale=0.04)
    random_vel = random_q(robot_gvs, jax.random.PRNGKey(5678), scale=0.07)

    config_velocity_cases = (
        (zero_cfg, zero_vel),
        (linear_cfg, linear_vel),
        (random_cfg, random_vel),
    )

    L_total = float(jnp.sum(robot_gvs.V_L))
    s_candidates = jnp.concatenate(
        [
            jnp.asarray([0.0], dtype=jnp.float64),
            sample_arc_lengths(robot_gvs),
            tip_arc_lengths(robot_gvs),
            jnp.asarray([L_total], dtype=jnp.float64),
        ]
    )
    s_points = jnp.unique(s_candidates)
    s_loop = onp.asarray(s_points, dtype=float)

    for q, qd in config_velocity_cases:
        for s in s_loop:
            print("q =\n", q)
            print("qd =\n", qd)
            print("s =", s)

            # check the forward kinematics
            g_gvs = robot_gvs.forward_kinematics(q, s)
            g_pcs = robot_pcs.forward_kinematics(q, s)
            assert_allclose(g_gvs, g_pcs, rtol=RTOL, atol=ATOL), "FK mismatch at s={}".format(s)

            # check the Jacobians in body frames
            Jb_gvs = robot_gvs.jacobian_bodyframe(q, s)
            Jb_pcs = robot_pcs.jacobian_bodyframe(q, s)
            assert_allclose(Jb_gvs, Jb_pcs, rtol=RTOL, atol=ATOL), "Jacobian bodyframe mismatch at s={}".format(s)

            # check the Jacobians in inertial frames
            Ji_gvs = robot_gvs.jacobian_inertialframe(q, s)
            Ji_pcs = robot_pcs.jacobian_inertialframe(q, s)
            assert_allclose(Ji_gvs, Ji_pcs, rtol=RTOL, atol=ATOL), "Jacobian inertialframe mismatch at s={}".format(s)

            # check the Jacobian derivatives in body frames
            _, Jdb_gvs = robot_gvs.jacobian_derivative_bodyframe(q, qd, s)
            _, Jdb_pcs = robot_pcs.jacobian_derivative_bodyframe(q, qd, s)
            assert_allclose(Jdb_gvs, Jdb_pcs, rtol=RTOL, atol=ATOL), "Jacobian derivative bodyframe mismatch at s={}".format(s)

            # check the Jacobian derivatives in inertial frames
            _, Jdi_gvs = robot_gvs.jacobian_derivative_inertialframe(q, qd, s)
            _, Jdi_pcs = robot_pcs.jacobian_derivative_inertialframe(q, qd, s)
            assert_allclose(Jdi_gvs, Jdi_pcs, rtol=RTOL, atol=ATOL), "Jacobian derivative inertialframe mismatch at s={}".format(s)

        g_joints_gvs = robot_gvs.forward_kinematics_joints(q)
        print("g_joints_gvs.shape =", g_joints_gvs.shape)
        g_tips_pcs = robot_pcs.forward_kinematics_tips(q)
        print("g_tips_pcs.shape =", g_tips_pcs.shape)
        assert_allclose(g_joints_gvs[1:], g_tips_pcs[:-1], rtol=RTOL, atol=ATOL)
        
        J_joints_gvs = robot_gvs.jacobian_bodyframe_joints(q)
        print("J_joints_gvs.shape =", J_joints_gvs.shape)
        J_tips_pcs = robot_pcs._J_local_tips(q)
        print("J_tips_pcs.shape =", J_tips_pcs.shape)
        assert_allclose(J_joints_gvs[1:], J_tips_pcs[:-1], rtol=RTOL, atol=ATOL)

        J_joints_gvs, Jd_joints_gvs = robot_gvs.jacobian_derivative_joints(q, qd)
        J_tips_pcs, Jd_tips_pcs = robot_pcs._J_Jd_local_tips(q, qd)
        assert_allclose(J_joints_gvs, J_tips_pcs, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_joints_gvs, Jd_tips_pcs, rtol=RTOL, atol=ATOL)

        B_gvs = robot_gvs.inertia_matrix(q)
        B_pcs = robot_pcs.inertia_matrix(q)
        assert_allclose(B_gvs, B_pcs, rtol=RTOL, atol=ATOL)

        C_gvs = robot_gvs.coriolis_matrix(q, qd)
        C_pcs = robot_pcs.coriolis_matrix(q, qd)
        assert_allclose(C_gvs, C_pcs, rtol=RTOL, atol=ATOL)

        K_gvs = robot_gvs.stiffness_matrix()
        K_pcs = robot_pcs.stiffness_matrix()
        assert_allclose(K_gvs, K_pcs, rtol=RTOL, atol=ATOL)

        G_gvs = robot_gvs.gravitational_force(q).reshape(-1)
        G_pcs = robot_pcs.gravitational_force(q).reshape(-1)
        assert_allclose(G_gvs, G_pcs, rtol=RTOL, atol=ATOL)

        kinetic_gvs = robot_gvs.kinetic_energy(q, qd)
        kinetic_pcs = robot_pcs.kinetic_energy(q, qd)
        assert_allclose(kinetic_gvs, kinetic_pcs, rtol=RTOL, atol=ATOL)

        potential_gvs = robot_gvs.potential_energy(q)
        potential_pcs = robot_pcs.potential_energy(q)
        assert_allclose(potential_gvs, potential_pcs, rtol=RTOL, atol=ATOL)

        gravitational_gvs = robot_gvs.gravitational_energy(q)
        gravitational_pcs = robot_pcs.gravitational_energy(q)
        assert_allclose(gravitational_gvs, gravitational_pcs, rtol=RTOL, atol=ATOL)

        total_gvs = robot_gvs.total_energy(q, qd)
        total_pcs = robot_pcs.total_energy(q, qd)
        assert_allclose(total_gvs, total_pcs, rtol=RTOL, atol=ATOL)

        g_tips_gvs = robot_gvs.forward_kinematics_tips(q)
        g_tips_pcs = robot_pcs.forward_kinematics_tips(q)
        assert_allclose(g_tips_gvs, g_tips_pcs, rtol=RTOL, atol=ATOL)

        J_tips_gvs = robot_gvs._J_local_tips(q)
        J_tips_pcs = robot_pcs._J_local_tips(q)
        assert_allclose(J_tips_gvs, J_tips_pcs, rtol=RTOL, atol=ATOL)

        J_batched_gvs = robot_gvs._J_local_batched(q, s_points)
        J_batched_pcs = robot_pcs._J_local_batched(q, s_points)
        assert_allclose(J_batched_gvs, J_batched_pcs, rtol=RTOL, atol=ATOL)

        J_tips_gvs, Jd_tips_gvs = robot_gvs._J_Jd_local_tips(q, qd)
        J_tips_pcs, Jd_tips_pcs = robot_pcs._J_Jd_local_tips(q, qd)
        assert_allclose(J_tips_gvs, J_tips_pcs, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_tips_gvs, Jd_tips_pcs, rtol=RTOL, atol=ATOL)

        J_batched_gvs, Jd_batched_gvs = robot_gvs._J_Jd_local_batched(q, qd, s_points)
        J_batched_pcs, Jd_batched_pcs = robot_pcs._J_Jd_local_batched(q, qd, s_points)
        assert_allclose(J_batched_gvs, J_batched_pcs, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_batched_gvs, Jd_batched_pcs, rtol=RTOL, atol=ATOL)

        y = jnp.concatenate([q, qd])
        u = jnp.zeros(n, dtype=jnp.float64)
        yd_gvs = robot_gvs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
        yd_pcs = robot_pcs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
        assert_allclose(yd_gvs, yd_pcs, rtol=RTOL, atol=ATOL)



@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_tips_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    random_cfg = random_q(robot_gvs, jax.random.PRNGKey(777), scale=0.05)

    s_tips = tip_arc_lengths(robot_gvs)

    for q in (zero_cfg, random_cfg):
        g_expected = stack_forward_kinematics(robot_gvs, q, s_tips)
        g_tips = robot_gvs.forward_kinematics_tips(q)

        assert_allclose(g_tips, g_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_batched_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    random_cfg = random_q(robot_gvs, jax.random.PRNGKey(888), scale=0.05)

    s_sampled = sample_arc_lengths(robot_gvs)
    s_values = [0.0] + onp.asarray(s_sampled, dtype=float).tolist()
    s_points = jnp.asarray(s_values, dtype=jnp.float64)

    for q in (zero_cfg, random_cfg):
        g_batched = robot_gvs.forward_kinematics_batched(q, s_points)
        g_expected = stack_forward_kinematics(robot_gvs, q, s_points)

        assert_allclose(g_batched, g_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_local_tips_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    q = random_q(robot_gvs, jax.random.PRNGKey(5), scale=0.03)

    s_tips = tip_arc_lengths(robot_gvs)
    J_tips = robot_gvs._J_local_tips(q)
    J_expected = stack_jacobians(robot_gvs, q, s_tips)

    assert_allclose(J_tips, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_local_batched_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    random_cfg = random_q(robot_gvs, jax.random.PRNGKey(9876), scale=0.05)

    s_points = sample_arc_lengths(robot_gvs)

    for q in (zero_cfg, random_cfg):
        J_batch = robot_gvs._J_local_batched(q, s_points)
        J_expected = stack_jacobians(robot_gvs, q, s_points)

        assert_allclose(J_batch, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_bodyframe_matches_autodiff(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(6)
    q_keys = jax.random.split(key, NUM_RANDOM_SAMPLES)

    for q_key in q_keys:
        q = random_q(robot_gvs, q_key, scale=0.03)

        for s in sample_arc_lengths(robot_gvs):
            if s < 1e-3:
                continue

            J_body = robot_gvs.jacobian_bodyframe(q, float(s))
            g = robot_gvs.forward_kinematics(q, float(s))
            g_inv = se3_inverse(g)

            def fk(qq: jnp.ndarray) -> jnp.ndarray:
                return robot_gvs.forward_kinematics(qq, float(s))

            cols = []
            eye = jnp.eye(q.shape[0], dtype=jnp.float64)
            for j in range(q.shape[0]):
                _, g_tangent = jvp(fk, (q,), (eye[j],))
                cols.append(se3_tangent_to_body_twist(g_inv, g_tangent))

            J_ad = jnp.stack(cols, axis=1)
            assert_allclose(J_body, J_ad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_derivative_bodyframe_matches_autograd_jvp(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot_gvs, q_key, scale=0.05)
        qd = random_q(robot_gvs, qd_key, scale=0.2)

        for s in sample_arc_lengths(robot_gvs):
            if s < 1e-3:
                continue

            def J_body(q_):
                return robot_gvs.jacobian_bodyframe(q_, float(s))

            _, Jd_jvp = jvp(J_body, (q,), (qd,))
            Jd_impl = robot_gvs.jacobian_derivative_bodyframe(q, qd, float(s))

            assert_allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_derivative_bodyframe_matches_central_differences(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    delta = 1e-6
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot_gvs, q_key, scale=0.05)
        qd = random_q(robot_gvs, qd_key, scale=0.2)

        for s in sample_arc_lengths(robot_gvs):
            if s < 1e-3:
                continue

            Jd_impl = robot_gvs.jacobian_derivative_bodyframe(q, qd, float(s))

            eye = jnp.eye(q.shape[0], dtype=jnp.float64)
            dJ_cols = []
            for j in range(q.shape[0]):
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                Jp = robot_gvs.jacobian_bodyframe(qp, float(s))
                Jm = robot_gvs.jacobian_bodyframe(qm, float(s))
                dJ_cols.append((Jp - Jm) / (2 * delta))

            dJ_dq_fd = jnp.stack(dJ_cols, axis=-1)
            Jd_num = jnp.tensordot(dJ_dq_fd, qd, axes=([-1], [0]))

            assert_allclose(Jd_impl, Jd_num, rtol=1e-3, atol=5e-6)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_Jd_local_tips_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    q_random = random_q(robot_gvs, jax.random.PRNGKey(987), scale=0.05)
    qd_random = random_q(robot_gvs, jax.random.PRNGKey(654), scale=0.1)

    s_tips = tip_arc_lengths(robot_gvs)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_tips, Jd_tips = robot_gvs._J_Jd_local_tips(q, qd)
        J_expected = stack_jacobians(robot_gvs, q, s_tips)
        Jd_expected = stack_jacobian_derivatives(robot_gvs, q, qd, s_tips)

        assert_allclose(J_tips, J_expected, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_tips, Jd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_J_Jd_local_batched_matches_pointwise_evaluation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    q_random = random_q(robot_gvs, jax.random.PRNGKey(321), scale=0.05)
    qd_random = random_q(robot_gvs, jax.random.PRNGKey(4321), scale=0.1)

    s_points = sample_arc_lengths(robot_gvs)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = robot_gvs._J_Jd_local_batched(q, qd, s_points)
        J_expected = stack_jacobians(robot_gvs, q, s_points)
        Jd_expected = stack_jacobian_derivatives(robot_gvs, q, qd, s_points)

        assert_allclose(J_batch, J_expected, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_batch, Jd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_dynamics_matches_manual_computation(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    key = jax.random.PRNGKey(123 + num_segments)

    for _ in range(NUM_RANDOM_SAMPLES):
        key, key_q = jax.random.split(key)
        q = random_q(robot_gvs, key_q, scale=0.05)

        key, key_qd = jax.random.split(key)
        qd = random_q(robot_gvs, key_qd, scale=0.05)

        key, key_u = jax.random.split(key)
        u = random_q(robot_gvs, key_u, scale=0.02)

        key, key_tau = jax.random.split(key)
        tau_ext = random_q(robot_gvs, key_tau, scale=0.03)

        y = jnp.concatenate([q, qd])
        yd = robot_gvs.forward_dynamics(0.0, y, (u, tau_ext))

        B = robot_gvs.inertia_matrix(q)
        C = robot_gvs.coriolis_matrix(q, qd)
        G = robot_gvs.gravitational_force(q)
        D = robot_gvs.damping_matrix()
        tau_el = robot_gvs.elastic_force(q)
        tau_u = robot_gvs.actuation_force(q, u)

        qdd_expected = jnp.linalg.solve(
            B, tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )
        yd_expected = jnp.concatenate([qd, qdd_expected])

        assert_allclose(yd, yd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3, 4])
def test_forward_mode_automatic_differentiability_at_zero_configuration(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=num_segments)
    dof = int(robot_gvs.dof_tot_system)
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((robot_gvs.num_actuators,), dtype=jnp.float64)
    s = float(robot_gvs.V_L_cum[-1])

    dg_dq = jacfwd(robot_gvs.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacfwd(robot_gvs.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacfwd(robot_gvs.jacobian_inertialframe, argnums=0)(q, s)
    dJd_bodyframe_dq = jacfwd(robot_gvs.jacobian_derivative_bodyframe, argnums=0)(q, qd, s)
    dJd_inertialframe_dq = jacfwd(robot_gvs.jacobian_derivative_inertialframe, argnums=0)(
        q, qd, s
    )
    dB_dq = jacfwd(robot_gvs.inertia_matrix)(q)
    dC_dq = jacfwd(robot_gvs.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacfwd(robot_gvs.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacfwd(robot_gvs.gravitational_force)(q)
    dtau_el_dq = jacfwd(robot_gvs.elastic_force)(q)
    dtau_u_dq = jacfwd(robot_gvs.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacfwd(robot_gvs.actuation_force, argnums=1)(q, u)
    dy_dy = jacfwd(robot_gvs.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacfwd(robot_gvs.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dg_dq).any(), f"Found NaN in forward-mode of forward kinematics"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), f"Found NaN in forward-mode of bodyframe Jacobian"
    assert not jnp.isnan(dJ_inertialframe_dq).any(), f"Found NaN in forward-mode of inertialframe Jacobian"
    assert not jnp.isnan(dJd_bodyframe_dq).any(), f"Found NaN in forward-mode of bodyframe Jacobian derivative"
    assert not jnp.isnan(dJd_inertialframe_dq).any(), f"Found NaN in forward-mode of inertialframe Jacobian derivative"
    assert not jnp.isnan(dB_dq).any(), f"Found NaN in forward-mode of inertia matrix"
    assert not jnp.isnan(dC_dq).any(), f"Found NaN in forward-mode of coriolis matrix (dC/dq)"
    assert not jnp.isnan(dC_dqd).any(), f"Found NaN in forward-mode of coriolis matrix (dC/dqd)"
    assert not jnp.isnan(dG_dq).any(), f"Found NaN in forward-mode of gravitational force"
    assert not jnp.isnan(dtau_el_dq).any(), f"Found NaN in forward-mode of elastic force"
    assert not jnp.isnan(dtau_u_dq).any(), f"Found NaN in forward-mode of actuation force (dTau/du)"
    assert not jnp.isnan(dtau_u_du).any(), f"Found NaN in forward-mode of actuation force (dTau/du)"
    assert not jnp.isnan(dy_dy).any(), f"Found NaN in forward-mode of forward dynamics (dy/dy)"
    assert not jnp.isnan(dy_du).any(), f"Found NaN in forward-mode of forward dynamics (dy/du)"


def test_reverse_mode_automatic_differentiability_at_zero_configuration() -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments=2)
    dof = int(robot_gvs.dof_tot_system)
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((robot_gvs.num_actuators,), dtype=jnp.float64)
    s = float(robot_gvs.V_L_cum[-1])

    dg_dq = jacrev(robot_gvs.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacrev(robot_gvs.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacrev(robot_gvs.jacobian_inertialframe, argnums=0)(q, s)
    dJd_bodyframe_dq = jacrev(robot_gvs.jacobian_derivative_bodyframe, argnums=0)(q, qd, s)
    dJd_inertialframe_dq = jacrev(robot_gvs.jacobian_derivative_inertialframe, argnums=0)(
        q, qd, s
    )
    dB_dq = jacrev(robot_gvs.inertia_matrix)(q)
    dC_dq = jacrev(robot_gvs.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacrev(robot_gvs.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacrev(robot_gvs.gravitational_force)(q)
    dtau_el_dq = jacrev(robot_gvs.elastic_force)(q)
    dtau_u_dq = jacrev(robot_gvs.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacrev(robot_gvs.actuation_force, argnums=1)(q, u)
    dy_dy = jacrev(robot_gvs.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacrev(robot_gvs.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dg_dq).any(), f"Found NaN in reverse-mode of forward kinematics"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), f"Found NaN in reverse-mode of bodyframe Jacobian"
    assert not jnp.isnan(dJ_inertialframe_dq).any(), f"Found NaN in reverse-mode of inertialframe Jacobian"
    assert not jnp.isnan(dJd_bodyframe_dq).any(), f"Found NaN in reverse-mode of bodyframe Jacobian derivative"
    assert not jnp.isnan(dJd_inertialframe_dq).any(), f"Found NaN in reverse-mode of inertialframe Jacobian derivative"
    assert not jnp.isnan(dB_dq).any(), f"Found NaN in reverse-mode of inertia matrix"
    assert not jnp.isnan(dC_dq).any(), f"Found NaN in reverse-mode of coriolis matrix (dC/dq)"
    assert not jnp.isnan(dC_dqd).any(), f"Found NaN in reverse-mode of coriolis matrix (dC/dqd)"
    assert not jnp.isnan(dG_dq).any(), f"Found NaN in reverse-mode of gravitational force"
    assert not jnp.isnan(dtau_el_dq).any(), f"Found NaN in reverse-mode of elastic force"
    assert not jnp.isnan(dtau_u_dq).any(), f"Found NaN in reverse-mode of actuation force (dTau/du)"
    assert not jnp.isnan(dtau_u_du).any(), f"Found NaN in reverse-mode of actuation force (dTau/du)"
    assert not jnp.isnan(dy_dy).any(), f"Found NaN in reverse-mode of forward dynamics (dy/dy)"
    assert not jnp.isnan(dy_du).any(), f"Found NaN in reverse-mode of forward dynamics (dy/du)"


@pytest.mark.parametrize("num_segments", [1])
def test_gvs_autodiff_checks(num_segments: int) -> None:
    robot_gvs, _ = build_matched_gvs_pcs(num_segments)

    n = int(robot_gvs.dof_tot_system)
    q = jnp.linspace(0.01, 0.01 * n, n, dtype=jnp.float64)
    qd = jnp.linspace(0.02, 0.02 * n, n, dtype=jnp.float64)
    s = jnp.sum(robot_gvs.V_L, dtype=jnp.float64) * 0.7

    def J_body(q_):
        return robot_gvs.jacobian_bodyframe(q_, s)

    _, Jd_dir = jax.jvp(J_body, (q,), (qd,))
    Jd_impl = robot_gvs.jacobian_derivative_bodyframe(q, qd, s)
    assert_allclose(Jd_impl, Jd_dir, rtol=RTOL, atol=ATOL)

    # 2) Translational block of inertial Jacobian matches jacobian of position
    def pos_fn(q_):
        return robot_gvs.forward_kinematics(q_, s)[:3, 3]

    Jpos_ad = jax.jacfwd(pos_fn)(q)  # shape (3, n)
    Ji_gvs = gvs_jacobian_inertialframe_from_body(robot_gvs, q, s)
    Jpos_impl = Ji_gvs[3:6, :]
    assert_allclose(Jpos_impl, Jpos_ad, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
