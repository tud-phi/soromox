import jax

jax.config.update("jax_enable_x64", True)  # double precision
import jax.numpy as jnp
import numpy as onp
import pytest

from numpy.testing import assert_allclose

from soromox.systems import Pendulum
from soromox.utils.tolerance import Tolerance


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

    params = {
        "m": m,
        "I": I,
        "L": L,
        "Lc": Lc,
        "g": jnp.array([0.0, -9.81]),
    }
    return Pendulum(params)


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
def test_jacobians_match_autodiff(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)

    # Tips
    J_tips_impl = robot.jacobians_tips(q)  # (N,3,N)
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
    J_coms_impl = robot.jacobians_coms(q)
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
    J_joints_impl = robot.jacobians_joints(q)
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
def test_jacobian_time_derivative_matches_jvp(N):
    robot = make_pendulum(N)
    q = jnp.linspace(-0.3, 0.4, N)
    qd = jnp.linspace(0.6, -0.5, N)

    # Closed-form J and Jdot at tips
    J_closed, Jd_closed = robot.jacobians_and_derivatives_tips(q, qd)

    # Jdot via directional derivative with JAX JVP
    def J_of_q(q_):
        return robot.jacobians_tips(q_)

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

    J_tips = robot.jacobians_tips(q)
    J_tips_closed, Jd_tips_closed = robot.jacobians_and_derivatives_tips(q, qd)

    # dJ/dq via autodiff, then Jdot = (dJ/dq) · qd
    dJdq_tips = jax.jacrev(robot.jacobians_tips)(q)  # (N,3,N,N)
    Jd_tips_auto = jnp.einsum("ijkl,l->ijk", dJdq_tips, qd)

    # J returned by jacobians_and_derivatives_tips matches jacobians_tips
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
        {"g": jnp.array([0.0, 0.0]), "K": jnp.zeros((N, N)), "D": jnp.zeros((N, N))}
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


if __name__ == "__main__":
    # Allow running this file directly for ad-hoc checks
    for N in (2, 3):
        test_forward_kinematics_simple_configs(N)
        test_jacobians_match_autodiff(N)
        test_jacobian_time_derivative_matches_jvp(N)
        test_inertia_matrix_at_zero_config_matches_closed_form(N)
        test_coriolis_christoffel_identity(N)
        test_gravity_matches_negative_potential_gradient(N)
        test_forward_dynamics_rest_with_zero_forces(N)
