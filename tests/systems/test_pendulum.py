import jax

jax.config.update("jax_enable_x64", True)  # double precision
import jax.numpy as jnp
import numpy as onp
import pytest
from numpy.testing import assert_allclose

from soromox.systems import Pendulum, TendonActuatedPendulum
from soromox.utils.tolerance import Tolerance
from system_param_builders import pendulum_params, tendon_actuated_pendulum_params


def make_pendulum(N: int = 2):
    """Make a simple N-link pendulum with distinct parameters."""
    assert N in (2, 3)
    if N == 2:
        m = jnp.array([1.0, 2.0])
        I = jnp.array([0.1, 0.2])
        L = jnp.array([1.0, 1.5])
        Lc = jnp.array([0.5, 0.75])
    else:  # N == 3
        m = jnp.array([1.0, 2.0, 0.8])
        I = jnp.array([0.1, 0.2, 0.05])
        L = jnp.array([1.0, 1.5, 0.8])
        Lc = jnp.array([0.5, 0.75, 0.4])

    params = pendulum_params(
        mass=m,
        moment_inertia=I,
        length=L,
        center_of_mass_length=Lc,
        gravity=jnp.array([0.0, -9.81]),
    )
    return Pendulum(params)


def test_pendulum_update_rejects_all_fixed_size_shape_changes():
    robot = make_pendulum(2)

    invalid_updates = {
        "length": jnp.ones((3,)),
        "moment_inertia": jnp.ones((3,)),
        "center_of_mass_length": jnp.ones((3,)),
        "joint_stiffness": jnp.eye(3),
        "joint_damping": jnp.eye(3),
        "joint_rest_configuration": jnp.ones((3,)),
        "radius": jnp.ones((3,)),
        "gravity": jnp.ones((3,)),
    }

    for name, value in invalid_updates.items():
        with pytest.raises(ValueError, match=name):
            robot.update_params(**{name: value})


def test_tendon_pendulum_update_reapplies_routing_invariants():
    body = pendulum_params(
        mass=jnp.ones((3,)),
        moment_inertia=0.1 * jnp.ones((3,)),
        length=jnp.ones((3,)),
        center_of_mass_length=0.5 * jnp.ones((3,)),
        gravity=jnp.array([0.0, -9.81]),
    )
    params = tendon_actuated_pendulum_params(
        body=body,
        active_routing_matrix=jnp.tril(jnp.ones((3, 3))),
    )
    robot = TendonActuatedPendulum(params)

    rank_deficient = params.replace(
        active_routing_matrix=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    with pytest.raises(ValueError, match="full row rank"):
        robot.with_params(rank_deficient)

    joint_skipping = params.replace(
        active_routing_matrix=jnp.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    with pytest.raises(ValueError, match="skip joints"):
        robot.with_params(joint_skipping)


@pytest.mark.parametrize("N", [2, 3])
def test_forward_kinematics_simple_configs(N):
    robot = make_pendulum(N)
    q = jnp.zeros((robot.num_links,))

    # Expected at q=0: all along +x, theta_i = 0
    L = onp.array(robot.L, dtype=float)
    Lc = onp.array(robot.Lc, dtype=float)
    thetas = onp.zeros((N,))
    # joint i at sum L[:i]
    p_joint_x = onp.concatenate([[0.0], onp.cumsum(L)[:-1]])
    p_tip_x = onp.cumsum(L)
    p_com_x = onp.concatenate([[0.0], onp.cumsum(L)[:-1]]) + Lc

    exp_joints = onp.stack([thetas, p_joint_x, onp.zeros_like(p_joint_x)], axis=-1)
    exp_tips = onp.stack([thetas, p_tip_x, onp.zeros_like(p_tip_x)], axis=-1)
    exp_coms = onp.stack([thetas, p_com_x, onp.zeros_like(p_com_x)], axis=-1)

    chi_j = onp.array(robot.forward_kinematics_joints(q))
    chi_t = onp.array(robot.forward_kinematics_tips(q))
    chi_c = onp.array(robot.forward_kinematics_coms(q))

    assert_allclose(chi_j, exp_joints, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(chi_t, exp_tips, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(chi_c, exp_coms, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("N", [2, 3])
def test_jacobian_match_autodiff(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # Tips
    J_tips_impl = robot.jacobian_tips(q)  # (N,3,N)
    # Autodiff jacobian for each link's [theta, px, py]
    for i in range(robot.num_links):
        f_tip_i = lambda q_: robot.forward_kinematics_tips(q_)[i]
        J_ad_i = jax.jacfwd(f_tip_i)(q)
        assert_allclose(
            J_tips_impl[i],
            J_ad_i,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )

    # COMs
    J_coms_impl = robot.jacobian_coms(q)
    for i in range(robot.num_links):
        f_com_i = lambda q_: robot.forward_kinematics_coms(q_)[i]
        J_ad_i = jax.jacfwd(f_com_i)(q)
        assert_allclose(
            J_coms_impl[i],
            J_ad_i,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )

    # Joints
    J_joints_impl = robot.jacobian_joints(q)
    for i in range(robot.num_links):
        f_joint_i = lambda q_: robot.forward_kinematics_joints(q_)[i]
        J_ad_i = jax.jacfwd(f_joint_i)(q)
        assert_allclose(
            J_joints_impl[i],
            J_ad_i,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("N", [2, 3])
def test_tip_jacobian_time_derivative_matches_jvp(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    # Closed-form J and Jdot at tips
    J_closed, Jd_closed = robot.jacobian_and_time_derivatives_tips(q, qd)

    # Jdot via directional derivative with JAX JVP
    def J_of_q(q_):
        return robot.jacobian_tips(q_)

    _, Jdot_dir = jax.jvp(J_of_q, (q,), (qd,))

    assert_allclose(
        Jd_closed,
        Jdot_dir,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize("N", [2, 3])
def test_tip_jacobian_time_derivative_autodiff_contraction(N):
    """Verify Jdot at tips using dJ/dq contracted with qd (autodiff)."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.2, 0.3, N)
    qd = jnp.linspace(0.7, -0.4, N)

    J_tips = robot.jacobian_tips(q)
    J_tips_closed, Jd_tips_closed = robot.jacobian_and_time_derivatives_tips(q, qd)

    # dJ/dq via autodiff, then Jdot = (dJ/dq) · qd
    dJdq_tips = jax.jacrev(robot.jacobian_tips)(q)  # (N,3,N,N)
    Jd_tips_auto = jnp.einsum("ijkl,l->ijk", dJdq_tips, qd)

    # J returned by jacobian_and_time_derivatives_tips matches jacobian_tips
    assert_allclose(
        J_tips,
        J_tips_closed,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    # And Jdot matches the autodiff contraction
    assert_allclose(
        Jd_tips_closed,
        Jd_tips_auto,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize("N", [2, 3])
def test_inertia_matrix_at_zero_config_matches_closed_form(N):
    robot = make_pendulum(N)
    q = jnp.zeros((robot.num_links,))
    B = onp.array(robot.inertia_matrix(q))

    # Closed-form B at q=0
    L = onp.array(robot.L, dtype=float)
    Lc = onp.array(robot.Lc, dtype=float)
    m = onp.array(robot.m, dtype=float)
    I = onp.array(robot.I, dtype=float)

    # x-positions of joints and COMs
    p_joint_x = onp.concatenate([[0.0], onp.cumsum(L)[:-1]])  # (N,)
    p_com_x = onp.concatenate([[0.0], onp.cumsum(L)[:-1]]) + Lc  # (N,)

    # Linear part: M_lin = sum_i m_i v_i v_i^T, with v_i[j] = (p_com_x[i] - p_joint_x[j]) if j<=i else 0
    M_lin = onp.zeros((N, N))
    for i in range(N):
        v = onp.zeros((N,))
        for j in range(i + 1):
            v[j] = p_com_x[i] - p_joint_x[j]
        M_lin += m[i] * onp.outer(v, v)

    # Angular part via planar Jw (lower-triangular ones per row)
    idx = onp.arange(N)
    Jw = (idx[None, :] <= idx[:, None]).astype(float)  # (N,N)
    W_ang = onp.einsum("i,ia,ib->ab", I, Jw, Jw)

    B_expected = M_lin + W_ang
    assert_allclose(B, B_expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("N", [2, 3])
def test_coriolis_christoffel_identity(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    C = robot.coriolis_matrix(q, qd)

    # Bdot via directional derivative B'(q)·qd
    def B_of_q(q_):
        return robot.inertia_matrix(q_)

    _, Bdot = jax.jvp(B_of_q, (q,), (qd,))

    # Identity: Bdot - (C + C^T) == 0 (equivalently, Bdot - 2C is skew-symmetric)
    sym_resid = Bdot - (C + C.T)
    assert_allclose(
        sym_resid,
        jnp.zeros_like(sym_resid),
        rtol=Tolerance.rtol(),
        atol=1e-9,
    )


@pytest.mark.parametrize("N", [2, 3])
def test_gravity_matches_potential_gradient(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.25, 0.5, N)

    G = robot.gravitational_force(q)
    dU_dq = jax.grad(robot.gravitational_energy)(q)

    # With current convention, G equals ∂U/∂q
    assert_allclose(G, dU_dq, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("N", [2, 3])
def test_forward_dynamics_rest_with_zero_forces(N):
    robot = make_pendulum(N)

    # Zero gravity, no stiffness or damping
    robot = robot.update_params(
        gravity=jnp.array([0.0, 0.0]),
        joint_stiffness=jnp.zeros((N, N)),
        joint_damping=jnp.zeros((N, N)),
    )

    q = jnp.zeros((N,))
    qd = jnp.zeros((N,))
    u = jnp.zeros((N,))
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
    qd_out, qdd_out = jnp.split(yd, 2)

    assert_allclose(qd_out, qd, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(
        qdd_out, jnp.zeros_like(qd), rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )


# -----------------------
# Arc-length parameterized kinematics tests
# -----------------------


@pytest.mark.parametrize("N", [2, 3])
def test_forward_kinematics_at_link_tips(N):
    """Test that forward_kinematics(q, s) at link tips matches forward_kinematics_tips."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # Get tips using both methods
    chi_tips_direct = robot.forward_kinematics_tips(q)

    # Get tips using arc-length method
    L_cum = onp.array(robot.L_cum)
    for i in range(N):
        s_tip = L_cum[i + 1]  # tip of link i
        chi_s = robot.forward_kinematics(q, jnp.array(s_tip))
        assert_allclose(
            chi_s, chi_tips_direct[i], rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


@pytest.mark.parametrize("N", [2, 3])
def test_forward_kinematics_at_base(N):
    """Test that forward_kinematics(q, s=0) returns correct base pose."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # At s=0, we're in link 0, so theta should be cumsum[0] = q[0]
    # Position should be at origin
    chi_s = robot.forward_kinematics(q, jnp.array(0.0))

    # Expected: theta = q[0], position = (0, 0)
    theta_cumsum = onp.cumsum(onp.array(q))
    expected_theta = theta_cumsum[0]
    expected_pos = onp.array([0.0, 0.0])

    assert_allclose(
        chi_s[0], expected_theta, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )
    assert_allclose(
        chi_s[1:], expected_pos, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )


@pytest.mark.parametrize("N", [2, 3])
def test_forward_kinematics_interpolates_along_link(N):
    """Test that forward_kinematics interpolates linearly within a link."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    L_cum = onp.array(robot.L_cum)
    L = onp.array(robot.L)
    theta_cumsum = onp.cumsum(onp.array(q))

    for i in range(N):
        # Test at start, middle, and end of link i (but not at boundaries with next link)
        s_start = L_cum[i] + 1e-10  # slightly after start
        s_mid = L_cum[i] + L[i] / 2
        s_end = L_cum[i + 1] - 1e-10  # slightly before end

        chi_start = robot.forward_kinematics(q, jnp.array(s_start))
        chi_mid = robot.forward_kinematics(q, jnp.array(s_mid))
        chi_end = robot.forward_kinematics(q, jnp.array(s_end))

        # Orientation should be constant within a link (theta_cumsum[i])
        expected_theta = theta_cumsum[i]
        assert_allclose(
            chi_start[0], expected_theta, rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )
        assert_allclose(
            chi_mid[0], expected_theta, rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )
        assert_allclose(
            chi_end[0], expected_theta, rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )

        # Position should be linearly interpolated along link direction
        # midpoint position should be average of start and end
        expected_mid_pos = (chi_start[1:] + chi_end[1:]) / 2
        assert_allclose(
            chi_mid[1:], expected_mid_pos, rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


@pytest.mark.parametrize("N", [2, 3])
def test_jacobian_at_tips_matches_jacobian_tips(N):
    """Test that jacobian(q, s) at link tips matches jacobian_tips."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # Get Jacobians using both methods
    J_tips_direct = robot.jacobian_tips(q)

    # Get Jacobians using arc-length method
    L_cum = onp.array(robot.L_cum)
    for i in range(N):
        s_tip = L_cum[i + 1]  # tip of link i
        J_s = robot.jacobian(q, jnp.array(s_tip))
        assert_allclose(
            J_s, J_tips_direct[i], rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


@pytest.mark.parametrize("N", [2, 3])
def test_jacobian_matches_autodiff(N):
    """Test that jacobian(q, s) matches autodiff of forward_kinematics."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # Test at several points along the robot
    L_cum = onp.array(robot.L_cum)
    total_len = L_cum[-1]
    s_values = jnp.linspace(0.0, total_len, 10)

    for s in s_values:
        s = float(s)
        # Analytical Jacobian
        J_analytical = robot.jacobian(q, jnp.array(s))

        # Autodiff Jacobian
        def fk_at_s(q_):
            return robot.forward_kinematics(q_, jnp.array(s))

        J_autodiff = jax.jacfwd(fk_at_s)(q)

        assert_allclose(
            J_analytical, J_autodiff, rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


@pytest.mark.parametrize("N", [2, 3])
def test_jacobian_and_time_derivative_at_tips(N):
    """Test jacobian_and_time_derivative at tips matches jacobian_and_time_derivatives_tips."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    # Get from tips method
    J_tips_direct, Jd_tips_direct = robot.jacobian_and_time_derivatives_tips(q, qd)

    # Get using arc-length method
    L_cum = onp.array(robot.L_cum)
    for i in range(N):
        s_tip = L_cum[i + 1]
        J_s, Jd_s = robot.jacobian_and_time_derivative(q, qd, jnp.array(s_tip))
        assert_allclose(
            J_s, J_tips_direct[i], rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )
        assert_allclose(
            Jd_s, Jd_tips_direct[i], rtol=Tolerance.rtol(), atol=Tolerance.atol()
        )


@pytest.mark.parametrize("N", [2, 3])
def test_arc_length_jacobian_time_derivative_matches_jvp(N):
    """Test that Jacobian time derivative matches JVP."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    L_cum = onp.array(robot.L_cum)
    total_len = L_cum[-1]
    s_values = [0.5 * total_len, total_len]  # midpoint and tip

    for s in s_values:
        s_arr = jnp.array(s)

        # Closed-form
        J_closed, Jd_closed = robot.jacobian_and_time_derivative(q, qd, s_arr)

        # JVP
        def J_of_q(q_):
            return robot.jacobian(q_, s_arr)

        _, Jd_jvp = jax.jvp(J_of_q, (q,), (qd,))

        assert_allclose(Jd_closed, Jd_jvp, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("N", [2, 3])
def test_batched_methods(N):
    """Test that batched methods work correctly."""
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    L_cum = onp.array(robot.L_cum)
    s_ps = jnp.array(L_cum[1:])  # all tip positions

    # Forward kinematics batched
    chi_batched = robot.forward_kinematics_batched(q, s_ps)
    chi_tips = robot.forward_kinematics_tips(q)
    assert_allclose(chi_batched, chi_tips, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # Jacobian batched
    J_batched = robot.jacobian_batched(q, s_ps)
    J_tips = robot.jacobian_tips(q)
    assert_allclose(J_batched, J_tips, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # Jacobian and derivative batched
    J_batched, Jd_batched = robot.jacobian_and_time_derivative_batched(q, qd, s_ps)
    J_tips, Jd_tips = robot.jacobian_and_time_derivatives_tips(q, qd)
    assert_allclose(J_batched, J_tips, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(Jd_batched, Jd_tips, rtol=Tolerance.rtol(), atol=Tolerance.atol())
