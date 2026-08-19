import equinox as eqx
import jax
import numpy as onp
import pytest
from jax import Array, jacfwd, jacrev, jvp
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, spatial_base_pose

from soromox.systems import PCS, CrossSectionGeometry, PCSStructure
from soromox.utils.integration import scale_interior_gaussian_quadrature
from soromox.utils.lie_algebra import se3
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
    num_gauss_points: int = 3,
    strain_selector: Array | None = None,
    scale_rotational_basis_by_length: bool = False,
):
    segment_length = total_length / num_segments
    L = segment_length * jnp.ones((num_segments,))
    diag_vals = jnp.repeat(
        jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
    )
    if xi_ref is None:
        xi_ref = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments)
    params = pcs_params(
        base_pose=spatial_base_pose(),
        length=L,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=1070 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=1e-3 * jnp.diag((diag_vals * L[:, None]).flatten()),
        reference_strain=xi_ref,
    )

    model = PCS(
        params=params,
        structure=PCSStructure(
            num_gauss_points=num_gauss_points,
            strain_selector=strain_selector,
            scale_rotational_basis_by_length=scale_rotational_basis_by_length,
        ),
    )

    return model, params


def _updated_pcs_params(params):
    return params.replace(
        base_pose=params.base_pose.at[4].set(0.15),
        gravity=params.gravity.at[2].set(-9.7),
        link=params.link.replace(
            length=params.link.length + 0.01,
            density=params.link.density + 10.0,
            stiffness=1.02 * params.link.stiffness,
            damping=1.03 * params.link.damping,
        ),
    )


def _pcs_parameter_summary(model):
    return jnp.concatenate(
        [
            model.base_pose,
            model.g,
            model.L,
            model.rho,
            model.K_full.reshape(-1),
            model.D_full.reshape(-1),
        ]
    )


def test_parameter_updates_refresh_eager_runtime_arrays_immutably():
    model, params = make_pcs()
    updated_params = _updated_pcs_params(params)
    before = _pcs_parameter_summary(model)

    updated = model.with_params(updated_params)

    assert_allclose(_pcs_parameter_summary(model), before)
    assert not jnp.allclose(_pcs_parameter_summary(updated), before)


def test_same_structure_parameter_updates_compile_once():
    model, params = make_pcs()
    updated_params = _updated_pcs_params(params)
    trace_count = {"value": 0}

    @eqx.filter_jit
    def compiled_summary(current_params):
        trace_count["value"] += 1
        return _pcs_parameter_summary(model.with_params(current_params))

    baseline = compiled_summary(params)
    actual = compiled_summary(updated_params)

    assert trace_count["value"] == 1
    assert not jnp.allclose(actual, baseline)
    assert_allclose(actual, _pcs_parameter_summary(model.with_params(updated_params)))


def test_parameter_gradient_passes_through_compiled_update():
    model, params = make_pcs()

    @eqx.filter_jit
    def scalar_summary(current_params):
        return jnp.sum(_pcs_parameter_summary(model.with_params(current_params)))

    gradient = eqx.filter_grad(scalar_summary)(params)

    assert gradient.link.density.shape == params.link.density.shape
    assert jnp.all(jnp.isfinite(gradient.link.density))
    assert jnp.any(gradient.link.density != 0.0)


def test_parameter_structure_changes_require_reconstruction():
    model, params = make_pcs()
    wrong_shape = eqx.tree_at(
        lambda current: current.link.length,
        params,
        jnp.ones((3,), dtype=jnp.float64),
    )

    with pytest.raises(ValueError, match="shape|structure|construct"):
        model.with_params(wrong_shape)


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


def expected_selection_basis(num_rows: int, active_indices: tuple[int, ...]) -> Array:
    basis = jnp.zeros((num_rows, len(active_indices)), dtype=jnp.float64)
    for col, row in enumerate(active_indices):
        basis = basis.at[row, col].set(1.0)
    return basis


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
    xi = se3.log(g_rel, eps=1e-12)
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
    return se3.adjoint(g_rot) @ xi_body


def segment_tip_transforms(model: PCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]

    def fk_at_s(s: Array) -> Array:
        return model.forward_kinematics(q, s)

    return jax.vmap(fk_at_s)(s_vals)


def test_constant_strain_call():
    """
    Test the constant strain system with numerical integration and Jacobian for 1 segment.
    """
    L = jnp.array([1e-1])
    D = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 0.0, 0.0, 1e3, 0.0, 1e3]]) * L[:, None]).flatten()
    )
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((6,), dtype=bool)

    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    params = pcs_params(
        base_pose=spatial_base_pose(),
        length=L,
        radius=jnp.array([2e-2]),
        density=1000 * jnp.ones((1,)),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=1e8 * jnp.ones((1,)),
        shear_modulus=1e7 * jnp.ones((1,)),
        damping_matrix=D,
        reference_strain=xi_ref,
    )

    robot = PCS(
        params=params,
        structure=PCSStructure(num_gauss_points=5, strain_selector=strain_selector),
    )

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((6,)),
            params.link.length[0] / 2,
            jnp.eye(4).at[0, 3].set(params.link.length[0] / 2),
        ),
        (
            jnp.zeros((6,)),
            params.link.length[0],
            jnp.eye(4).at[0, 3].set(params.link.length[0]),
        ),
        (
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            params.link.length[0],
            jnp.eye(4).at[0, 3].set(2 * params.link.length[0]),
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

    q = jnp.array([jnp.pi / (2 * params.link.length[0]), 0.0, 0.0, 0.0, 0.0, 0.0])
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
    robot = robot.update_params(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
    )
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
        * params.link.density[0]
        * jnp.pi
        * params.link.cross_section.coefficients[0, 0] ** 2
        * jnp.linalg.norm(params.gravity)
        * params.link.length[0] ** 2
    )
    assert_allclose(E_pot, E_pot_th, rtol=RTOL, atol=ATOL)
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
    u = jnp.zeros((6,))  # no external forces
    robot = robot.update_params(gravity=jnp.zeros((3,)))  # no gravity for this test
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", robot.params.gravity)
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
    assert_allclose(model.length, jnp.sum(params.link.length), rtol=RTOL, atol=ATOL)
    assert_allclose(model.segment_length, params.link.length, rtol=RTOL, atol=ATOL)

    s_second = params.link.length[0] + 0.25 * params.link.length[1]
    segment_idx, s_local = model.classify_segment(s_second)
    assert int(segment_idx) == 1
    assert_allclose(s_local, 0.25 * params.link.length[1], rtol=RTOL, atol=ATOL)

    tag, geom = model.cross_section_geometry(q, s_second)
    assert int(tag) == CrossSectionGeometry.CIRCULAR
    assert_allclose(
        geom,
        jnp.array([params.link.cross_section.coefficients[1, 0]]),
        rtol=RTOL,
        atol=ATOL,
    )


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
        g_expected = jax.vmap(lambda s, q=q: model.forward_kinematics(q, s))(s_ps)

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


@pytest.mark.parametrize(
    "strain_selector",
    [
        jnp.array(
            [
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
            ],
            dtype=bool,
        ),
        jnp.zeros((12,), dtype=bool),
    ],
    ids=["segment-specific-inactive-strains", "all-strains-inactive"],
)
def test_inverse_kinematics_with_inactive_strain_segments(strain_selector: Array):
    num_segments = int(strain_selector.size // 6)
    model, _ = make_pcs(num_segments=num_segments, strain_selector=strain_selector)

    q = random_q(model, jax.random.PRNGKey(789), scale=0.05)
    g_tips = segment_tip_transforms(model, q)
    q_recovered = model.inverse_kinematics(g_tips)

    assert q_recovered.shape == q.shape
    assert_allclose(q_recovered, q, rtol=RTOL, atol=ATOL)
    assert_allclose(
        segment_tip_transforms(model, q_recovered), g_tips, rtol=RTOL, atol=ATOL
    )


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

            def fk(qq: Array, s=s) -> Array:
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

            def fk(qq: Array, s=s) -> Array:
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
def test_jacobian_time_derivative_bodyframe_matches_autograd_jvp(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_time_derivative_bodyframe(q, qd, s)

            def J_body(q_, s=s):
                return model.jacobian_bodyframe(q_, s)

            _, Jd_jvp = jvp(J_body, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{Jd_impl}\nJd_jvp:\n{Jd_jvp}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_time_derivative_bodyframe_matches_central_differences(
    num_segments: int,
):
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
            J_local, Jd_local = model.jacobian_and_time_derivative_bodyframe(
                q, qd, s_tip
            )

            assert_allclose(
                J_local, J_local_tips_[idx] @ model.B_xi, rtol=RTOL, atol=ATOL
            )
            assert_allclose(
                Jd_local, Jd_local_tips_[idx] @ model.B_xi, rtol=RTOL, atol=ATOL
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_and_time_derivative_bodyframe_batched_matches_pointwise_evaluation(
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
        J_batch, Jd_batch = model.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_time_derivative_bodyframe(
                q, qd, s_val
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_time_derivative_inertialframe_matches_autograd_jvp(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(model, q_key, scale=0.05)
        qd = random_q(model, qd_key, scale=0.2)

        for s in sample_arc_lengths(model):
            J_impl, Jd_impl = model.jacobian_and_time_derivative_inertialframe(q, qd, s)

            def J_global(q_, s=s):
                return model.jacobian_inertialframe(q_, s)

            _, Jd_jvp = jvp(J_global, (q,), (qd,))

            assert jnp.allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7), (
                f"num_segments={num_segments}, s={s}\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jd_jvp)}"
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_and_time_derivative_inertialframe_batched_matches_pointwise_evaluation(
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
        J_batch, Jd_batch = model.jacobian_and_time_derivative_inertialframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = model.jacobian_and_time_derivative_inertialframe(
                q, qd, s_val
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


def test_public_pcs_jacobian_wrappers_match_inertialframe_methods() -> None:
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


class _SentinelJacobianOnlyPCS(PCS):
    def _jacobian(self, q: Array, s: Array) -> Array:
        entries = jnp.arange(6 * q.shape[0], dtype=q.dtype).reshape(6, q.shape[0])
        return 1e-2 * (entries + 1.0)


class _SentinelForwardKinematicsArcLengthJVPPCS(PCS):
    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        pose = self._forward_kinematics(q, s)
        poses = jnp.full_like(pose, 0.375)
        return pose, poses


class _SentinelJacobianJVPPCS(PCS):
    def _jacobian_and_time_derivative(
        self, q: Array, qd: Array, s: Array
    ) -> tuple[Array, Array]:
        J = self._jacobian(q, s)
        Jd = jnp.full_like(J, 0.125)
        return J, Jd


class _SentinelJacobianArcLengthJVPPCS(PCS):
    def _jacobian_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        J = self._jacobian(q, s)
        Js = jnp.full_like(J, 0.25)
        return J, Js


class _SentinelGravitationalEnergyJVPPCS(PCS):
    def _gravitational_force(self, q: Array) -> Array:
        return jnp.linspace(0.25, 1.25, q.shape[0], dtype=q.dtype)


def _sentinel_pcs_model(cls: type[PCS]) -> PCS:
    base, params = make_pcs(num_segments=1)
    return cls(
        params=params,
        structure=PCSStructure(num_gauss_points=base.num_gauss_points),
    )


def _strict_interior_arc_lengths(model: PCS) -> jnp.ndarray:
    fractions = jnp.asarray([0.37, 0.73], dtype=jnp.float64)
    return (model.L_cum[:-1, None] + model.L[:, None] * fractions).reshape(-1)


def test_pcs_custom_jvp_primal_methods_match_protected_hooks() -> None:
    model, _ = make_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(1), scale=0.03)
    s = model.L_cum[-1]

    assert_allclose(model.forward_kinematics(q, s), model._forward_kinematics(q, s))
    assert_allclose(model.jacobian(q, s), model.jacobian_inertialframe(q, s))
    assert_allclose(model.gravitational_energy(q), model._gravitational_energy(q))


def test_pcs_forward_kinematics_jvp_uses_protected_primal_autodiff() -> None:
    model = _sentinel_pcs_model(_SentinelJacobianOnlyPCS)
    q = random_q(model, jax.random.PRNGKey(2), scale=0.02)
    qd = random_q(model, jax.random.PRNGKey(3), scale=0.01)
    s = model.L_cum[-1]

    _, posed = jax.jvp(lambda q_: model.forward_kinematics(q_, s), (q,), (qd,))
    _, expected_posed = jax.jvp(
        lambda q_: model._forward_kinematics(q_, s), (q,), (qd,)
    )

    assert_allclose(posed, expected_posed, rtol=RTOL, atol=ATOL)


def test_pcs_forward_kinematics_jvp_uses_arc_length_pair_hook() -> None:
    model = _sentinel_pcs_model(_SentinelForwardKinematicsArcLengthJVPPCS)
    q = random_q(model, jax.random.PRNGKey(2034), scale=0.02)
    s = model.L_cum[-1]
    sd = jnp.array(0.37, dtype=jnp.float64)

    _, posed = jax.jvp(lambda s_: model.forward_kinematics(q, s_), (s,), (sd,))

    assert_allclose(posed, jnp.full_like(posed, 0.375) * sd, rtol=RTOL, atol=ATOL)


def test_pcs_jacobian_jvp_uses_analytical_time_derivative_hook() -> None:
    model = _sentinel_pcs_model(_SentinelJacobianJVPPCS)
    q = random_q(model, jax.random.PRNGKey(4), scale=0.02)
    qd = random_q(model, jax.random.PRNGKey(5), scale=0.01)
    s = model.L_cum[-1]

    _, Jd = jax.jvp(lambda q_: model.jacobian(q_, s), (q,), (qd,))
    _, expected_Jd = model._jacobian_and_time_derivative(q, qd, s)

    assert_allclose(Jd, expected_Jd, rtol=RTOL, atol=ATOL)


def test_pcs_jacobian_jvp_uses_arc_length_pair_hook() -> None:
    model = _sentinel_pcs_model(_SentinelJacobianArcLengthJVPPCS)
    q = random_q(model, jax.random.PRNGKey(2035), scale=0.02)
    s = model.L_cum[-1]
    sd = jnp.array(0.37, dtype=jnp.float64)

    _, Jd = jax.jvp(lambda s_: model.jacobian(q, s_), (s,), (sd,))

    assert_allclose(Jd, jnp.full_like(Jd, 0.25) * sd, rtol=RTOL, atol=ATOL)


def test_pcs_jacobian_mixed_jvp_uses_protected_primal_autodiff() -> None:
    model = _sentinel_pcs_model(_SentinelJacobianOnlyPCS)
    q = random_q(model, jax.random.PRNGKey(2036), scale=0.02)
    qd = random_q(model, jax.random.PRNGKey(2037), scale=0.01)
    s = model.L_cum[-1]
    sd = jnp.array(0.37, dtype=jnp.float64)

    _, Jd = jax.jvp(lambda q_, s_: model.jacobian(q_, s_), (q, s), (qd, sd))
    _, expected_Jd = jax.jvp(
        lambda q_, s_: model._jacobian(q_, s_),
        (q, s),
        (qd, sd),
    )

    assert_allclose(Jd, expected_Jd, rtol=RTOL, atol=ATOL)


def test_pcs_gravitational_energy_grad_uses_analytical_force_hook() -> None:
    model = _sentinel_pcs_model(_SentinelGravitationalEnergyJVPPCS)
    q = random_q(model, jax.random.PRNGKey(6), scale=0.02)

    dU_dq = jax.grad(lambda q_: model.gravitational_energy(q_))(q)

    assert_allclose(dU_dq, model._gravitational_force(q), rtol=RTOL, atol=ATOL)


def test_pcs_custom_jvps_match_real_analytical_methods() -> None:
    model, _ = make_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(7), scale=0.03)
    qd = random_q(model, jax.random.PRNGKey(8), scale=0.02)
    s = model.L_cum[-1]

    pose, posed = jax.jvp(lambda q_: model.forward_kinematics(q_, s), (q,), (qd,))
    eta = model.jacobian(q, s) @ qd
    expected_posed = model._pose_tangent_from_inertial_velocity(pose, eta)
    assert_allclose(posed, expected_posed, rtol=RTOL, atol=ATOL)

    J, Jd = jax.jvp(lambda q_: model.jacobian(q_, s), (q,), (qd,))
    expected_J, expected_Jd = model.jacobian_and_time_derivative(q, qd, s)
    assert_allclose(J, expected_J, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd, expected_Jd, rtol=RTOL, atol=ATOL)

    dU_dq = jax.grad(lambda q_: model.gravitational_energy(q_))(q)
    assert_allclose(dU_dq, model.gravitational_force(q), rtol=RTOL, atol=ATOL)


def test_pcs_arc_length_derivatives_match_autodiff() -> None:
    model, _ = make_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(2029), scale=0.03)

    for s in _strict_interior_arc_lengths(model):
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


def test_pcs_custom_jvps_include_arc_length_derivative() -> None:
    model, _ = make_pcs(num_segments=2)
    q = random_q(model, jax.random.PRNGKey(2030), scale=0.03)
    qd = random_q(model, jax.random.PRNGKey(2031), scale=0.02)
    s = _strict_interior_arc_lengths(model)[1]
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
    robot, _ = make_pcs(num_segments=num_segments)
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
        jac_T_q = jacfwd(lambda qq, qd=qd: dT_dqd(qq, qd))(q)
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
def test_integration_kinematics_matches_existing_batched_path(
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

    g_expected = model.forward_kinematics_batched(q, s_points).reshape(
        num_segments, num_inner, 4, 4
    )
    J_full, Jd_full = model._J_Jd_local_batched(q, qd, s_points)
    J_expected = (J_full @ model.B_xi).reshape(num_segments, num_inner, 6, dof)
    Jd_expected = (Jd_full @ model.B_xi).reshape(num_segments, num_inner, 6, dof)

    assert_allclose(g_quads, g_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(J_quads, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_quads, Jd_expected, rtol=RTOL, atol=ATOL)

    _, g_convective, J_convective, Jd_qd = model._integration_kinematics(
        q, qd, convective_only_jd=True
    )
    assert_allclose(g_convective, g_quads, rtol=RTOL, atol=ATOL)
    assert_allclose(J_convective, J_quads, rtol=RTOL, atol=ATOL)
    assert_allclose(
        Jd_qd,
        jnp.einsum("...ij,j->...i", Jd_quads, qd),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("num_segments", [1, 3])
@pytest.mark.parametrize(
    "selector_per_segment",
    [
        None,
        (False, False, True, True, False, False),
        (False, False, False, True, False, False),
    ],
)
def test_dynamics_terms_match_public_matrices(
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


def test_dynamics_terms_support_leading_inactive_segment():
    strain_selector = jnp.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )
    model, _ = make_pcs(num_segments=2, strain_selector=strain_selector)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(6125))
    q = random_q(model, key_q, scale=0.05)
    qd = random_q(model, key_qd, scale=0.1)

    inertia, coriolis_qd, gravity = model.dynamics_terms(q, qd)

    assert_allclose(inertia, model.inertia_matrix(q), rtol=RTOL, atol=ATOL)
    assert_allclose(
        coriolis_qd,
        model.coriolis_matrix(q, qd) @ qd,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(gravity, model.gravitational_force(q), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 4, 8, 16])
def test_platform_inertia_solve_matches_generic_solve(num_segments: int):
    model, _ = make_pcs(num_segments=num_segments, total_length=PCS_TOTAL_LENGTH)
    key_q, key_qd, key_rhs = jax.random.split(jax.random.PRNGKey(6126), 3)
    q = random_q(model, key_q, scale=0.05)
    qd = random_q(model, key_qd, scale=0.1)
    rhs = random_q(model, key_rhs, scale=0.2)
    inertia, _, _ = model.dynamics_terms(q, qd)

    actual = model._solve_inertia(inertia, rhs)
    expected = jnp.linalg.solve(inertia, rhs)

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    assert_allclose(inertia @ actual, rhs, rtol=RTOL, atol=ATOL)


def test_cached_constant_matrices_refresh_after_update_params():
    selector_per_segment = jnp.array(
        [False, False, True, True, False, False], dtype=bool
    )
    model, _ = make_pcs(
        num_segments=2,
        strain_selector=jnp.tile(selector_per_segment, 2),
    )

    updated = model.update_link_params(
        cross_section=model.params.link.cross_section.replace(
            coefficients=1.1 * model.params.link.cross_section.coefficients
        ),
        density=0.9 * model.rho,
        stiffness=1.25 * model.params.link.stiffness,
        damping=2.0 * model.params.link.damping,
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


# ======================================================================================
# Strain-basis consistency tests (selection basis applied correctly across APIs)
# ======================================================================================


def test_strain_basis_creation_matches_selector_order():
    strain_selector = jnp.array(
        [
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            False,
        ],
        dtype=bool,
    )
    model, _ = make_pcs(num_segments=2, strain_selector=strain_selector)

    expected_B = expected_selection_basis(12, (0, 2, 5, 7, 9))

    assert int(model.num_active_strains.item()) == 5
    assert model.B_xi.shape == (12, 5)
    assert_allclose(model.B_xi, expected_B, rtol=0.0, atol=0.0)
    assert_allclose(model.B_xi.T @ model.B_xi, jnp.eye(5), rtol=0.0, atol=0.0)

    q = jnp.arange(1.0, 6.0)
    expected_xi = expected_B @ q + model.xi_ref
    assert_allclose(model.strain(q), expected_xi, rtol=RTOL, atol=ATOL)


def test_rotational_strain_basis_length_scaling_matches_unscaled_coordinates():
    num_segments = 2
    total_length = 0.5
    segment_length = total_length / num_segments
    unscaled, _ = make_pcs(
        num_segments=num_segments,
        total_length=total_length,
        num_gauss_points=5,
        scale_rotational_basis_by_length=False,
    )
    scaled, _ = make_pcs(
        num_segments=num_segments,
        total_length=total_length,
        num_gauss_points=5,
        scale_rotational_basis_by_length=True,
    )

    per_segment_scale = jnp.array(
        [1 / segment_length, 1 / segment_length, 1 / segment_length, 1, 1, 1],
        dtype=jnp.float64,
    )
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
        g_scaled = scaled.forward_kinematics(q_scaled, s)
        g_unscaled = unscaled.forward_kinematics(q_unscaled, s)
        assert_allclose(g_scaled, g_unscaled, rtol=RTOL, atol=ATOL)

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

    updated = scaled.update_link_params(length=jnp.array([0.2, 0.3]))
    updated_scale = jnp.array(
        [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 10 / 3, 10 / 3, 10 / 3, 1.0, 1.0, 1.0]
    )
    assert_allclose(
        updated.B_xi,
        updated_scale[:, None] * updated.B_xi_unscaled,
        rtol=RTOL,
        atol=ATOL,
    )


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
def test_strain_basis_consistency_jacobians_and_time_derivatives(num_segments: int):
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
        J_small, Jd_small = reduced.jacobian_and_time_derivative_bodyframe(
            q_small, qd_small, s
        )
        J_full, Jd_full = full.jacobian_and_time_derivative_bodyframe(
            q_full, qd_full, s
        )
        assert J_small.shape == (6, n_small_act)
        assert Jd_small.shape == (6, n_small_act)
        assert J_full.shape == (6, int(full.num_active_strains.item()))
        assert Jd_full.shape == (6, int(full.num_active_strains.item()))
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
    _, dJd_bodyframe = jacfwd(model.jacobian_and_time_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacfwd(
        model.jacobian_and_time_derivative_inertialframe, argnums=0
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
    _, dJd_bodyframe = jacrev(model.jacobian_and_time_derivative_bodyframe, argnums=0)(
        q, qd, s
    )
    _, dJd_inertialframe = jacrev(
        model.jacobian_and_time_derivative_inertialframe, argnums=0
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


def test_reverse_mode_of_vmap_at_zero_configuration() -> None:
    """Differentiate a batch that contains the exactly-straight configuration.

    ``vmap`` rewrites the small-angle ``lax.cond`` guards into ``select_n``, a
    lowering that bare ``jacrev`` never exercises, so this covers a path the
    other zero-configuration tests cannot reach.
    """
    model, _ = make_pcs(num_segments=1, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    s_ps = jnp.array([0.5, 1.0]) * PCS_TOTAL_LENGTH
    q_batch = jnp.stack([jnp.zeros((dof,)), jnp.zeros((dof,)).at[0].set(0.3)])
    qd_batch = jnp.zeros_like(q_batch)

    for name, fn in (
        (
            "forward_kinematics_batched",
            lambda q: jnp.sum(model.forward_kinematics_batched(q, s_ps)),
        ),
        ("inertia_matrix", lambda q: jnp.sum(model.inertia_matrix(q))),
        ("gravitational_force", lambda q: jnp.sum(model.gravitational_force(q))),
        ("elastic_force", lambda q: jnp.sum(model.elastic_force(q))),
        ("actuation_matrix", lambda q: jnp.sum(model.actuation_matrix(q))),
    ):
        grad = jax.grad(lambda batch, fn=fn: jnp.sum(jax.vmap(fn)(batch)))(q_batch)
        assert not jnp.isnan(grad).any(), f"d(vmap({name}))/dq contains NaN!"

    coriolis = jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(lambda q, qd: jnp.sum(model.coriolis_matrix(q, qd)))(
                batch, qd_batch
            )
        )
    )(q_batch)
    assert not jnp.isnan(coriolis).any(), "d(vmap(coriolis_matrix))/dq contains NaN!"


def test_reverse_mode_of_vmap_over_jacobian_mixed_jvp_at_zero_configuration() -> None:
    """The Jacobian custom-JVP mixed fallback stays finite in a batched reverse pass."""
    model, _ = make_pcs(num_segments=1, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    s = model.L_cum[-1]
    qd = jnp.linspace(0.01, 0.06, dof, dtype=jnp.float64)
    sd = jnp.asarray(0.07, dtype=jnp.float64)
    q_batch = jnp.stack([jnp.zeros((dof,)), jnp.zeros((dof,)).at[0].set(0.3)])

    grad = jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(
                lambda q: jnp.sum(
                    jax.jvp(
                        lambda q_, s_: model.jacobian(q_, s_),
                        (q, s),
                        (qd, sd),
                    )[1]
                )
            )(batch)
        )
    )(q_batch)

    assert jnp.isfinite(grad).all(), "d(vmap(jvp(jacobian, q, s)))/dq contains NaN!"


def test_reverse_mode_of_vmap_over_energy_custom_jvps_at_zero_configuration() -> None:
    """Kinetic and total-energy custom JVPs stay finite in a batched reverse pass."""
    model, _ = make_pcs(num_segments=1, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    qd = jnp.linspace(0.01, 0.06, dof, dtype=jnp.float64)
    q_batch = jnp.stack([jnp.zeros((dof,)), jnp.zeros((dof,)).at[0].set(0.3)])

    for name, fn in (
        ("kinetic_energy", lambda q: model.kinetic_energy(q, qd)),
        ("gravitational_energy", lambda q: model.gravitational_energy(q)),
        ("elastic_energy", lambda q: model.elastic_energy(q)),
        ("potential_energy", lambda q: model.potential_energy(q)),
        ("total_energy", lambda q: model.total_energy(q, qd)),
    ):
        grad = jax.grad(lambda batch, fn=fn: jnp.sum(jax.vmap(fn)(batch)))(q_batch)
        assert jnp.isfinite(grad).all(), f"d(vmap({name}))/dq contains NaN!"

    q = jnp.zeros((dof,))
    qd_batch = jnp.stack([jnp.zeros_like(qd), qd])
    for name, fn in (
        ("kinetic_energy", lambda qd_: model.kinetic_energy(q, qd_)),
        ("total_energy", lambda qd_: model.total_energy(q, qd_)),
    ):
        grad = jax.grad(lambda batch, fn=fn: jnp.sum(jax.vmap(fn)(batch)))(qd_batch)
        assert jnp.isfinite(grad).all(), f"d(vmap({name}))/dqd contains NaN!"


def test_reverse_mode_of_vmap_over_forward_kinematics_at_zero_configuration() -> None:
    model, _ = make_pcs(num_segments=1, total_length=PCS_TOTAL_LENGTH)
    dof = int(model.num_active_strains.item())
    s = model.L_cum[-1]
    q_batch = jnp.stack([jnp.zeros((dof,)), jnp.zeros((dof,)).at[0].set(0.3)])

    grad = jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(lambda q: jnp.sum(model.forward_kinematics(q, s)))(batch)
        )
    )(q_batch)

    assert not jnp.isnan(grad).any(), "d(vmap(forward_kinematics))/dq contains NaN!"


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
