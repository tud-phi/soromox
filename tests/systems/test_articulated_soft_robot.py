# ruff: noqa: E402, I001
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import pytest

from numpy.testing import assert_allclose

from soromox.systems import ArticulatedSoftRobot, Pendulum, SoftRobot
from soromox.utils.tolerance import Tolerance
from system_param_builders import articulated_params, pendulum_params


def make_articulated_robot(num_links: int = 2) -> ArticulatedSoftRobot:
    assert num_links in (2, 3)
    joint_screws = jnp.zeros((num_links, 6))
    joint_screws = joint_screws.at[:, 2].set(1.0)
    p_tip = jnp.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.8, 0.0, 0.0]])[:num_links]
    p_com = 0.5 * p_tip
    m = jnp.array([1.0, 2.0, 0.8])[:num_links]
    I_z = jnp.array([0.1, 0.2, 0.05])[:num_links]
    I_com = jnp.stack(
        [
            jnp.diag(jnp.array([0.01 + 0.01 * i, 0.02 + 0.01 * i, I_z[i]]))
            for i in range(num_links)
        ]
    )
    params = articulated_params(
        joint_screw=joint_screws,
        tip_position=p_tip,
        center_of_mass_position=p_com,
        mass=m,
        center_of_mass_inertia=I_com,
        gravity=jnp.array([0.0, -9.81, 0.0]),
    )
    return ArticulatedSoftRobot(params)


def make_matching_pendulum(num_links: int = 2) -> Pendulum:
    assert num_links in (2, 3)
    L = jnp.array([1.0, 1.5, 0.8])[:num_links]
    params = pendulum_params(
        mass=jnp.array([1.0, 2.0, 0.8])[:num_links],
        moment_inertia=jnp.array([0.1, 0.2, 0.05])[:num_links],
        length=L,
        center_of_mass_length=0.5 * L,
        gravity=jnp.array([0.0, -9.81]),
    )
    return Pendulum(params)


def make_spatial_robot() -> ArticulatedSoftRobot:
    joint_screws = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    params = articulated_params(
        joint_screw=joint_screws,
        tip_position=jnp.array(
            [[0.8, 0.0, 0.1], [0.6, 0.0, 0.2], [0.4, 0.1, 0.0]]
        ),
        center_of_mass_position=jnp.array(
            [[0.4, 0.0, 0.05], [0.3, 0.0, 0.1], [0.2, 0.05, 0.0]]
        ),
        mass=jnp.array([1.0, 0.8, 0.5]),
        center_of_mass_inertia=jnp.array(
            [
                jnp.diag(jnp.array([0.02, 0.03, 0.04])),
                jnp.diag(jnp.array([0.015, 0.025, 0.035])),
                jnp.diag(jnp.array([0.01, 0.02, 0.03])),
            ]
        ),
        gravity=jnp.array([0.0, 0.0, -9.81]),
        joint_stiffness=jnp.diag(jnp.array([0.2, 0.3, 0.4])),
        joint_damping=jnp.diag(jnp.array([0.01, 0.02, 0.03])),
    )
    return ArticulatedSoftRobot(params)


def assert_finite(value, name: str) -> None:
    assert jnp.isfinite(value).all(), f"{name} contains non-finite values."


def test_import_and_inheritance():
    robot = make_articulated_robot()

    assert isinstance(robot, SoftRobot)
    assert robot.num_dofs == 2
    assert robot.num_actuators == 2
    assert not robot.is_planar


def test_constructor_shape_validation():
    params = articulated_params(
        joint_screw=jnp.zeros((2, 6)),
        tip_position=jnp.zeros((2, 3)),
        center_of_mass_position=jnp.zeros((2, 3)),
        mass=jnp.ones((2,)),
        center_of_mass_inertia=jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        gravity=jnp.zeros((3,)),
    )

    with pytest.raises(ValueError, match="tip_position"):
        ArticulatedSoftRobot(params.replace(tip_position=jnp.zeros((3, 3))))

    with pytest.raises(ValueError, match="center_of_mass_inertia"):
        ArticulatedSoftRobot(
            params.replace(
                center_of_mass_inertia=jnp.broadcast_to(jnp.eye(3), (3, 3, 3))
            )
        )


@pytest.mark.parametrize("num_links", [2, 3])
def test_tip_linear_jacobian_match_autodiff(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.3, 0.4, num_links)
    J_tips = robot.jacobian_tips(q)

    for i in range(num_links):

        def tip_position(q_, link_idx=i):
            return robot.forward_kinematics_tips(q_)[link_idx, :3, 3]

        J_pos = jax.jacfwd(tip_position)(q)
        assert_allclose(
            J_tips[i, 3:, :],
            J_pos,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("num_links", [2, 3])
def test_jacobian_time_derivative_matches_jvp(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.3, 0.4, num_links)
    qd = jnp.linspace(0.6, -0.5, num_links)

    J, Jd = robot.jacobian_and_time_derivatives_tips(q, qd)
    J_direct = robot.jacobian_tips(q)
    _, Jd_jvp = jax.jvp(robot.jacobian_tips, (q,), (qd,))

    assert_allclose(J, J_direct, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(Jd, Jd_jvp, rtol=Tolerance.rtol(), atol=Tolerance.atol())


def test_spatial_jacobian_time_derivative_matches_jvp():
    robot = make_spatial_robot()
    q = jnp.array([0.2, -0.3, 0.4])
    qd = jnp.array([0.5, -0.2, 0.3])

    J_tips, Jd_tips = robot.jacobian_and_time_derivatives_tips(q, qd)
    J_tips_direct = robot.jacobian_tips(q)
    _, Jd_tips_jvp = jax.jvp(robot.jacobian_tips, (q,), (qd,))

    assert_allclose(J_tips, J_tips_direct, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(Jd_tips, Jd_tips_jvp, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    s_points = jnp.stack(
        [
            jnp.array(0.2, dtype=q.dtype),
            robot.L_cum[1] + jnp.array(0.2, dtype=q.dtype),
            robot.total_length - jnp.array(0.05, dtype=q.dtype),
        ]
    )
    for s in s_points:
        J, Jd = robot._jacobian_and_time_derivative(q, qd, s)
        J_direct = robot._jacobian(q, s)
        _, Jd_jvp = jax.jvp(lambda q_, s=s: robot._jacobian(q_, s), (q,), (qd,))

        assert_allclose(J, J_direct, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        assert_allclose(Jd, Jd_jvp, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("num_links", [2, 3])
def test_inertia_matrix_is_symmetric_positive_definite(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.2, 0.3, num_links)
    M = robot.inertia_matrix(q)

    assert_allclose(M, M.T, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    eigvals = onp.linalg.eigvalsh(onp.array(M))
    assert onp.all(eigvals > 0.0)


@pytest.mark.parametrize("num_links", [2, 3])
def test_gravity_matches_potential_gradient(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.25, 0.5, num_links)

    G = robot.gravitational_force(q)
    dU_dq = jax.grad(robot.gravitational_energy)(q)

    assert_allclose(G, dU_dq, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("num_links", [2, 3])
def test_coriolis_christoffel_identity(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.3, 0.4, num_links)
    qd = jnp.linspace(0.6, -0.5, num_links)

    C = robot.coriolis_matrix(q, qd)
    _, Mdot = jax.jvp(robot.inertia_matrix, (q,), (qd,))

    sym_resid = Mdot - (C + C.T)
    assert_allclose(
        sym_resid, jnp.zeros_like(sym_resid), rtol=Tolerance.rtol(), atol=1e-9
    )


@pytest.mark.parametrize("num_links", [2, 3])
def test_forward_dynamics_matches_dense_matrix_equation(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.2, 0.3, num_links)
    qd = jnp.linspace(0.5, -0.4, num_links)
    u = jnp.linspace(0.2, -0.1, num_links)
    tau_ext = jnp.linspace(-0.05, 0.07, num_links)

    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.array(0.0), y, (u, tau_ext))
    _, qdd = jnp.split(yd, 2)

    M = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    G = robot.gravitational_force(q)
    D = robot.damping_matrix(q)
    rhs = (
        robot.actuation_force(q, u)
        + tau_ext
        - C @ qd
        - G
        - robot.elastic_force(q)
        - D @ qd
    )
    qdd_dense = jnp.linalg.solve(M, rhs)

    assert_allclose(qdd, qdd_dense, rtol=Tolerance.rtol(), atol=Tolerance.atol())


def test_spatial_mixed_axis_chain_matches_dense_matrix_equation():
    robot = make_spatial_robot()
    q = jnp.array([0.2, -0.3, 0.4])
    qd = jnp.array([0.5, -0.2, 0.3])
    u = jnp.array([0.1, -0.05, 0.2])
    tau_ext = jnp.array([0.02, 0.01, -0.03])

    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.array(0.0), y, (u, tau_ext))
    _, qdd = jnp.split(yd, 2)

    rhs = (
        robot.actuation_force(q, u)
        + tau_ext
        - robot.coriolis_matrix(q, qd) @ qd
        - robot.gravitational_force(q)
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ qd
    )
    qdd_dense = jnp.linalg.solve(robot.inertia_matrix(q), rhs)

    assert robot.forward_kinematics_tips(q).shape == (3, 4, 4)
    assert robot.jacobian_tips(q).shape == (3, 6, 3)
    assert_allclose(qdd, qdd_dense, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("num_links", [2, 3])
def test_batched_kinematics_and_jacobian_match_pointwise_evaluation(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.linspace(-0.3, 0.4, num_links)
    qd = jnp.linspace(0.6, -0.5, num_links)
    s_ps = robot.L_cum[1:]

    g_batched = robot.forward_kinematics_batched(q, s_ps)
    g_expected = jax.vmap(lambda s: robot.forward_kinematics(q, s))(s_ps)
    assert_allclose(g_batched, g_expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    J_batched = robot.jacobian_batched(q, s_ps)
    J_expected = robot.jacobian_tips(q)
    assert_allclose(J_batched, J_expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    J_batched, Jd_batched = robot.jacobian_and_time_derivative_batched(q, qd, s_ps)
    J_expected, Jd_expected = robot.jacobian_and_time_derivatives_tips(q, qd)
    assert_allclose(J_batched, J_expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(
        Jd_batched, Jd_expected, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )


@pytest.mark.parametrize("num_links", [2, 3])
def test_forward_mode_automatic_differentiability_at_zero_configuration(num_links):
    robot = make_articulated_robot(num_links)
    q = jnp.zeros((robot.num_dofs,), dtype=jnp.float64)
    qd = jnp.zeros((robot.num_dofs,), dtype=jnp.float64)
    u = jnp.zeros((robot.num_actuators,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    s = robot.total_length

    dg_dq = jax.jacfwd(robot.forward_kinematics, argnums=0)(q, s)
    dJ_dq = jax.jacfwd(robot.jacobian, argnums=0)(q, s)
    _, dJd_dq = jax.jacfwd(robot.jacobian_and_time_derivative, argnums=0)(q, qd, s)

    assert_finite(dg_dq, "dg/dq")
    assert_finite(dJ_dq, "dJ/dq")
    assert_finite(dJd_dq, "dJd/dq")

    dM_dq = jax.jacfwd(robot.inertia_matrix)(q)
    dC_dq = jax.jacfwd(robot.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jax.jacfwd(robot.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jax.jacfwd(robot.gravitational_force)(q)
    dtau_el_dq = jax.jacfwd(robot.elastic_force)(q)
    dtau_u_dq = jax.jacfwd(robot.actuation_force, argnums=0)(q, u)
    dtau_u_du = jax.jacfwd(robot.actuation_force, argnums=1)(q, u)
    dy_dy = jax.jacfwd(robot.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jax.jacfwd(robot.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert_finite(dM_dq, "dM/dq")
    assert_finite(dC_dq, "dC/dq")
    assert_finite(dC_dqd, "dC/dqd")
    assert_finite(dG_dq, "dG/dq")
    assert_finite(dtau_el_dq, "dtau_el/dq")
    assert_finite(dtau_u_dq, "dtau_u/dq")
    assert_finite(dtau_u_du, "dtau_u/du")
    assert_finite(dy_dy, "dy/dy")
    assert_finite(dy_du, "dy/du")

    dT_dq = jax.jacfwd(robot.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jax.jacfwd(robot.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jax.jacfwd(robot.gravitational_energy)(q)
    dU_dq = jax.jacfwd(robot.potential_energy)(q)
    dE_dq = jax.jacfwd(robot.total_energy, argnums=0)(q, qd)
    dE_dqd = jax.jacfwd(robot.total_energy, argnums=1)(q, qd)

    assert_finite(dT_dq, "dT/dq")
    assert_finite(dT_dqd, "dT/dqd")
    assert_finite(dU_G_dq, "dU_G/dq")
    assert_finite(dU_dq, "dU/dq")
    assert_finite(dE_dq, "dE/dq")
    assert_finite(dE_dqd, "dE/dqd")


def test_reverse_mode_automatic_differentiability_at_zero_configuration():
    robot = make_spatial_robot()
    q = jnp.zeros((robot.num_dofs,), dtype=jnp.float64)
    qd = jnp.zeros((robot.num_dofs,), dtype=jnp.float64)
    u = jnp.zeros((robot.num_actuators,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    s = robot.total_length

    dg_dq = jax.jacrev(robot.forward_kinematics, argnums=0)(q, s)
    dJ_dq = jax.jacrev(robot.jacobian, argnums=0)(q, s)
    _, dJd_dq = jax.jacrev(robot.jacobian_and_time_derivative, argnums=0)(q, qd, s)

    assert_finite(dg_dq, "dg/dq")
    assert_finite(dJ_dq, "dJ/dq")
    assert_finite(dJd_dq, "dJd/dq")

    dM_dq = jax.jacrev(robot.inertia_matrix)(q)
    dC_dq = jax.jacrev(robot.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jax.jacrev(robot.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jax.jacrev(robot.gravitational_force)(q)
    dtau_el_dq = jax.jacrev(robot.elastic_force)(q)
    dtau_u_dq = jax.jacrev(robot.actuation_force, argnums=0)(q, u)
    dtau_u_du = jax.jacrev(robot.actuation_force, argnums=1)(q, u)
    dy_dy = jax.jacrev(robot.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jax.jacrev(robot.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert_finite(dM_dq, "dM/dq")
    assert_finite(dC_dq, "dC/dq")
    assert_finite(dC_dqd, "dC/dqd")
    assert_finite(dG_dq, "dG/dq")
    assert_finite(dtau_el_dq, "dtau_el/dq")
    assert_finite(dtau_u_dq, "dtau_u/dq")
    assert_finite(dtau_u_du, "dtau_u/du")
    assert_finite(dy_dy, "dy/dy")
    assert_finite(dy_du, "dy/du")

    dT_dq = jax.jacrev(robot.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jax.jacrev(robot.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jax.jacrev(robot.gravitational_energy)(q)
    dU_dq = jax.jacrev(robot.potential_energy)(q)
    dE_dq = jax.jacrev(robot.total_energy, argnums=0)(q, qd)
    dE_dqd = jax.jacrev(robot.total_energy, argnums=1)(q, qd)

    assert_finite(dT_dq, "dT/dq")
    assert_finite(dT_dqd, "dT/dqd")
    assert_finite(dU_G_dq, "dU_G/dq")
    assert_finite(dU_dq, "dU/dq")
    assert_finite(dE_dq, "dE/dq")
    assert_finite(dE_dqd, "dE/dqd")


@pytest.mark.parametrize("num_links", [2, 3])
def test_planar_case_matches_pendulum(num_links):
    articulated = make_articulated_robot(num_links)
    pendulum = make_matching_pendulum(num_links)
    q = jnp.linspace(-0.3, 0.4, num_links)
    qd = jnp.linspace(0.6, -0.5, num_links)
    u = jnp.linspace(0.2, -0.1, num_links)

    g_tips = articulated.forward_kinematics_tips(q)
    articulated_tip_xy = g_tips[:, :2, 3]
    pendulum_tip_xy = pendulum.forward_kinematics_tips(q)[:, 1:]
    assert_allclose(
        articulated_tip_xy,
        pendulum_tip_xy,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    assert_allclose(
        articulated.inertia_matrix(q),
        pendulum.inertia_matrix(q),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        articulated.gravitational_force(q),
        pendulum.gravitational_force(q),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    y = jnp.concatenate([q, qd])
    yd_articulated = articulated.forward_dynamics(jnp.array(0.0), y, (u,))
    yd_pendulum = pendulum.forward_dynamics(jnp.array(0.0), y, (u,))
    assert_allclose(
        yd_articulated,
        yd_pendulum,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
