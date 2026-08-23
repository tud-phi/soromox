import jax
from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.autodiff import (
    custom_jvp_enabled,
    custom_jvp_mode,
    set_custom_jvp_enabled,
)
from soromox.systems.components import CrossSectionGeometry
from soromox.systems.soft_robot import SoftRobot
from soromox.utils.geometry import poses
from soromox.utils.lie_algebra import se3

jax.config.update("jax_enable_x64", True)


class _PlanarDefaultRobot(SoftRobot):
    L: Array

    def __init__(self):
        super().__init__()
        self.num_dofs = 2
        self.num_actuators = 2
        self.L = jnp.array([1.0, 2.0], dtype=jnp.float64)
        self.num_gauss_points = 1
        self.num_integration_points = 3
        self.integration_points = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float64)
        self.integration_weights = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)

    @property
    def segment_length(self) -> Array:
        return self.L

    @property
    def is_planar(self) -> bool:
        return True

    def cross_section_geometry(self, q: Array, s: Array) -> tuple[Array, Array]:
        return jnp.asarray(CrossSectionGeometry.CIRCULAR), jnp.array([1.0])

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        theta = q[0] * s
        return jnp.array([theta, s * jnp.cos(theta) + q[1], s * jnp.sin(theta)])

    def inertia_matrix(self, q: Array) -> Array:
        return jnp.array(
            [
                [2.0 + 0.3 * q[0] ** 2, 0.1 * jnp.sin(q[0] + q[1])],
                [0.1 * jnp.sin(q[0] + q[1]), 4.0 + 0.2 * q[1] ** 2],
            ],
            dtype=q.dtype,
        )

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        dM = jax.jacfwd(self.inertia_matrix)(q)
        christoffel = dM + jnp.swapaxes(dM, 1, 2) - jnp.transpose(dM, (2, 0, 1))
        return 0.5 * jnp.einsum("ijk,k->ij", christoffel, qd)

    def damping_matrix(self, q: Array) -> Array:
        return jnp.diag(jnp.array([0.1, 0.2], dtype=q.dtype))

    def stiffness_matrix(self) -> Array:
        return jnp.eye(2, dtype=jnp.float64)

    def elastic_force(self, q: Array) -> Array:
        return q

    def _gravitational_energy(self, q: Array) -> Array:
        return q[0] ** 2 + 3.0 * q[1]

    def actuation_matrix(self, q: Array) -> Array:
        return jnp.eye(2, dtype=q.dtype)


class _SpatialDefaultRobot(_PlanarDefaultRobot):
    @property
    def is_planar(self) -> bool:
        return False

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        theta = q[0] * s
        c = jnp.cos(theta)
        sn = jnp.sin(theta)
        R = jnp.array(
            [
                [c, -sn, 0.0],
                [sn, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=q.dtype,
        )
        p = jnp.array([q[1] * s, q[0] * q[1], s], dtype=q.dtype)
        g = jnp.eye(4, dtype=q.dtype)
        g = g.at[:3, :3].set(R)
        return g.at[:3, 3].set(p)


class _SpatialLieExpDefaultRobot(_SpatialDefaultRobot):
    """Spatial default-hook fixture whose protected pose uses the SE(3) branch."""

    def __init__(self):
        super().__init__()
        self.num_dofs = 6

    def _forward_kinematics(self, q: Array, s: Array) -> Array:
        return se3.exp(s * q, eps=jnp.asarray(1e-10, dtype=q.dtype))


class _DynamicsTermsEnergyJVPDefaultRobot(_PlanarDefaultRobot):
    def _gravitational_force(self, q: Array) -> Array:
        return jnp.array([1.5, -0.25], dtype=q.dtype)

    def elastic_force(self, q: Array) -> Array:
        return jnp.array([-0.75, 0.5], dtype=q.dtype)

    def dynamics_terms(self, q: Array, qd: Array) -> tuple[Array, Array, Array]:
        return (
            self.inertia_matrix(q),
            self.coriolis_matrix(q, qd) @ qd,
            self._gravitational_force(q),
        )


class _ToggleSentinelDefaultRobot(_PlanarDefaultRobot):
    def _jacobian(self, q: Array, s: Array) -> Array:
        del s
        return jnp.array(
            [
                [2.0, -1.0],
                [0.5, 1.25],
                [-0.75, 0.25],
            ],
            dtype=q.dtype,
        )

    def _forward_kinematics_and_arc_length_derivative(
        self, q: Array, s: Array
    ) -> tuple[Array, Array]:
        pose = self._forward_kinematics(q, s)
        return pose, jnp.full_like(pose, 0.375)

    def _gravitational_force(self, q: Array) -> Array:
        return jnp.array([4.0, -2.0], dtype=q.dtype)


def test_custom_jvp_global_toggle_context_manager_restores_state() -> None:
    set_custom_jvp_enabled(True)
    assert custom_jvp_enabled()

    with custom_jvp_mode(False):
        assert not custom_jvp_enabled()

    assert custom_jvp_enabled()

    set_custom_jvp_enabled(False)
    with custom_jvp_mode(True):
        assert custom_jvp_enabled()

    assert not custom_jvp_enabled()
    set_custom_jvp_enabled(True)


def test_default_length_sums_segment_lengths() -> None:
    robot = _PlanarDefaultRobot()

    assert_allclose(robot.length, jnp.sum(robot.segment_length), rtol=1e-12, atol=1e-12)


def test_shared_inertia_solver_matches_generic_dense_solve() -> None:
    robot = _PlanarDefaultRobot()
    inertia = jnp.array([[3.0, 0.4], [0.4, 1.5]], dtype=jnp.float64)
    rhs = jnp.array([0.25, -0.75], dtype=jnp.float64)

    actual = robot._solve_inertia(inertia, rhs)
    expected = jnp.linalg.solve(inertia, rhs)

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(inertia @ actual, rhs, rtol=1e-12, atol=1e-12)


def test_custom_jvp_toggle_controls_public_forward_kinematics_arc_length_jvp() -> None:
    robot = _ToggleSentinelDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    s = jnp.array(0.8, dtype=jnp.float64)
    sd = jnp.array(0.37, dtype=jnp.float64)

    with custom_jvp_mode(True):
        _, posed = jax.jvp(lambda s_: robot.forward_kinematics(q, s_), (s,), (sd,))

    assert_allclose(posed, jnp.full_like(posed, 0.375) * sd, rtol=1e-12, atol=1e-12)

    with custom_jvp_mode(False):
        _, posed = jax.jvp(lambda s_: robot.forward_kinematics(q, s_), (s,), (sd,))

    _, expected = jax.jvp(lambda s_: robot._forward_kinematics(q, s_), (s,), (sd,))
    assert_allclose(posed, expected, rtol=1e-12, atol=1e-12)


def test_custom_jvp_toggle_controls_public_gravity_energy_grad() -> None:
    robot = _ToggleSentinelDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)

    with custom_jvp_mode(True):
        G = jax.grad(lambda q_: robot.gravitational_energy(q_))(q)
    assert_allclose(G, robot._gravitational_force(q), rtol=1e-12, atol=1e-12)

    with custom_jvp_mode(False):
        G = jax.grad(lambda q_: robot.gravitational_energy(q_))(q)
    expected = jax.grad(lambda q_: robot._gravitational_energy(q_))(q)
    assert_allclose(G, expected, rtol=1e-12, atol=1e-12)


def test_default_planar_jacobians_and_tip_methods() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    s = jnp.array(0.8, dtype=jnp.float64)

    J_expected = jax.jacfwd(lambda q_: robot.forward_kinematics(q_, s))(q)
    assert_allclose(robot.jacobian(q, s), J_expected, rtol=1e-12, atol=1e-12)

    J, Jd = robot.jacobian_and_time_derivative(q, qd, s)
    J_expected, Jd_expected = jax.jvp(lambda q_: robot.jacobian(q_, s), (q,), (qd,))
    assert_allclose(J, J_expected, rtol=1e-12, atol=1e-12)
    assert_allclose(Jd, Jd_expected, rtol=1e-12, atol=1e-12)

    s_tips = jnp.cumsum(robot.segment_length)
    fk_tips_expected = jax.vmap(lambda s_: robot.forward_kinematics(q, s_))(s_tips)
    J_tips_expected = jax.vmap(lambda s_: robot.jacobian(q, s_))(s_tips)

    assert_allclose(robot.forward_kinematics_tips(q), fk_tips_expected)
    assert_allclose(robot.jacobian_tips(q), J_tips_expected)


def test_default_spatial_jacobian_converts_se3_pose_derivative() -> None:
    robot = _SpatialDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    s = jnp.array(0.8, dtype=jnp.float64)

    J = robot.jacobian(q, s)
    g, gd = jax.jvp(lambda q_: robot.forward_kinematics(q_, s), (q,), (qd,))
    omega_hat = gd[:3, :3] @ g[:3, :3].T
    omega = 0.5 * jnp.array(
        [
            omega_hat[2, 1] - omega_hat[1, 2],
            omega_hat[0, 2] - omega_hat[2, 0],
            omega_hat[1, 0] - omega_hat[0, 1],
        ]
    )
    twist_expected = jnp.concatenate([omega, gd[:3, 3]])

    assert J.shape == (6, 2)
    assert_allclose(J @ qd, twist_expected, rtol=1e-12, atol=1e-12)

    J_body = robot.jacobian_bodyframe(q, s)
    R = g[:3, :3]
    body_twist_expected = jnp.concatenate([R.T @ omega, R.T @ gd[:3, 3]])
    assert_allclose(J_body @ qd, body_twist_expected, rtol=1e-12, atol=1e-12)

    assert robot.forward_kinematics_tips(q).shape == (2, 4, 4)
    assert robot.jacobian_tips(q).shape == (2, 6, 2)


def test_default_custom_jvp_paths_are_finite_under_grad_of_vmap() -> None:
    """Inherited nested-AD fallback paths remain finite under batched reverse mode."""
    robot = _PlanarDefaultRobot()
    q = jnp.zeros((2,), dtype=jnp.float64)
    q_batch = jnp.stack([q, jnp.array([0.3, -0.2], dtype=jnp.float64)])
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    s = jnp.asarray(0.8, dtype=jnp.float64)
    sd = jnp.asarray(0.13, dtype=jnp.float64)

    q_paths = (
        (
            "forward_kinematics",
            lambda q_: jnp.sum(robot.forward_kinematics(q_, s)),
        ),
        ("jacobian", lambda q_: jnp.sum(robot.jacobian(q_, s))),
        ("kinetic_energy", lambda q_: robot.kinetic_energy(q_, qd)),
        ("total_energy", lambda q_: robot.total_energy(q_, qd)),
        (
            "mixed_forward_kinematics_jvp",
            lambda q_: jnp.sum(
                jax.jvp(
                    lambda q__, s__: robot.forward_kinematics(q__, s__),
                    (q_, s),
                    (qd, sd),
                )[1]
            ),
        ),
        (
            "mixed_jacobian_jvp",
            lambda q_: jnp.sum(
                jax.jvp(
                    lambda q__, s__: robot.jacobian(q__, s__),
                    (q_, s),
                    (qd, sd),
                )[1]
            ),
        ),
    )

    for name, fn in q_paths:
        grad = jax.grad(lambda batch, fn=fn: jnp.sum(jax.vmap(fn)(batch)))(q_batch)
        assert jnp.isfinite(grad).all(), f"default {name} path contains NaN!"

    s_batch = jnp.array([0.8, 1.2], dtype=jnp.float64)
    for name, fn in (
        (
            "forward_kinematics_arc_length",
            lambda s_: jnp.sum(robot.forward_kinematics(q, s_)),
        ),
        ("jacobian_arc_length", lambda s_: jnp.sum(robot.jacobian(q, s_))),
    ):
        grad = jax.grad(lambda values, fn=fn: jnp.sum(jax.vmap(fn)(values)))(s_batch)
        assert jnp.isfinite(grad).all(), f"default {name} path contains NaN!"


def test_spatial_default_jacobian_fallback_is_finite_without_custom_jvp() -> None:
    """The spatial Jacobian fallback is finite when its public wrapper is bypassed."""
    robot = _SpatialLieExpDefaultRobot()
    q = jnp.zeros((6,), dtype=jnp.float64)
    q_batch = jnp.stack([q, q.at[0].set(0.3)])
    s = jnp.asarray(0.8, dtype=jnp.float64)

    def batch_grad(fn):
        return jax.grad(
            lambda batch: jnp.sum(jax.vmap(lambda q_: jnp.sum(fn(q_)))(batch))
        )(q_batch)

    protected_grad = batch_grad(lambda q_: robot._jacobian(q_, s))
    with custom_jvp_mode(False):
        wrapper_disabled_grad = batch_grad(lambda q_: robot.jacobian(q_, s))

    assert jnp.isfinite(protected_grad).all()
    assert jnp.isfinite(wrapper_disabled_grad).all()
    assert_allclose(wrapper_disabled_grad, protected_grad, rtol=1e-12, atol=1e-12)


def test_spatial_default_jacobian_custom_jvp_is_finite_under_grad_of_vmap() -> None:
    robot = _SpatialLieExpDefaultRobot()
    q = jnp.zeros((6,), dtype=jnp.float64)
    q_batch = jnp.stack([q, q.at[0].set(0.3)])
    s = jnp.asarray(0.8, dtype=jnp.float64)

    grad = jax.grad(
        lambda batch: jnp.sum(
            jax.vmap(lambda q_: jnp.sum(robot.jacobian(q_, s)))(batch)
        )
    )(q_batch)

    assert jnp.isfinite(grad).all(), "d(vmap(jacobian))/dq contains NaN!"


def test_default_integration_kinematics_uses_bodyframe_samples() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)

    g_ps, J_ps, Jd_ps = robot.integration_kinematics(q, qd)

    s_ps = jnp.array([[0.5], [2.0]], dtype=jnp.float64)
    s_flat = s_ps.reshape(-1)
    g_expected = jax.vmap(
        lambda s: poses.planar_pose_to_transform(robot.forward_kinematics(q, s))
    )(s_flat).reshape(2, 1, 3, 3)
    J_expected, Jd_expected = robot.jacobian_and_time_derivative_bodyframe_batched(
        q, qd, s_flat
    )

    assert g_ps.shape == (2, 1, 3, 3)
    assert J_ps.shape == (2, 1, 3, 2)
    assert Jd_ps.shape == (2, 1, 3, 2)
    assert_allclose(g_ps, g_expected, rtol=1e-12, atol=1e-12)
    assert_allclose(J_ps, J_expected.reshape(2, 1, 3, 2), rtol=1e-12, atol=1e-12)
    assert_allclose(Jd_ps, Jd_expected.reshape(2, 1, 3, 2), rtol=1e-12, atol=1e-12)

    pose = robot.forward_kinematics(q, s_flat[0])
    J_inertial = robot.jacobian(q, s_flat[0])
    theta = pose[0]
    R = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ],
        dtype=pose.dtype,
    )
    J_body_expected = jnp.concatenate([J_inertial[:1], R.T @ J_inertial[1:]], axis=0)
    assert_allclose(J_ps[0, 0], J_body_expected, rtol=1e-12, atol=1e-12)


def test_default_gravity_force_and_forward_dynamics() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    u = jnp.array([1.5, -2.0], dtype=jnp.float64)
    tau_ext = jnp.array([0.25, -0.75], dtype=jnp.float64)
    y = jnp.concatenate([q, qd])

    G = robot.gravitational_force(q)
    assert_allclose(G, jnp.array([0.4, 3.0], dtype=jnp.float64))
    M, Cqd, G_terms = robot.dynamics_terms(q, qd)
    assert_allclose(M, robot.inertia_matrix(q), rtol=1e-12, atol=1e-12)
    assert_allclose(Cqd, robot.coriolis_matrix(q, qd) @ qd, rtol=1e-12, atol=1e-12)
    assert_allclose(G_terms, G, rtol=1e-12, atol=1e-12)

    rhs = (
        robot.actuation_matrix(q) @ u
        + tau_ext
        - Cqd
        - G
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ qd
    )
    qdd = jnp.linalg.solve(robot.inertia_matrix(q), rhs)

    assert_allclose(robot.forward_dynamics(0.0, y, (u, tau_ext)), jnp.r_[qd, qdd])


def test_default_potential_force_sums_conservative_forces() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)

    expected = robot.gravitational_force(q) + robot.elastic_force(q)

    assert_allclose(robot.potential_force(q), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(
        robot._potential_energy_gradient(q),
        robot.potential_force(q),
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_coriolis_matrix_is_energy_consistent() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)

    _, M_dot = jax.jvp(robot.inertia_matrix, (q,), (qd,))
    C = robot.coriolis_matrix(q, qd)
    skew = M_dot - 2.0 * C

    assert_allclose(skew + skew.T, jnp.zeros_like(skew), rtol=1e-12, atol=1e-12)


def test_default_energy_custom_jvps_use_force_and_dynamics_terms() -> None:
    robot = _DynamicsTermsEnergyJVPDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    qd_tangent = jnp.array([-0.6, 0.25], dtype=jnp.float64)
    qdd = jnp.array([0.15, -0.35], dtype=jnp.float64)

    _, Ueld = jax.jvp(lambda q_: robot.elastic_energy(q_), (q,), (qd_tangent,))
    elastic_expected = robot.elastic_force(q) @ qd_tangent
    assert_allclose(Ueld, elastic_expected, rtol=1e-12, atol=1e-12)

    _, Ud = jax.jvp(lambda q_: robot.potential_energy(q_), (q,), (qd_tangent,))
    potential_expected = (
        robot.gravitational_force(q) + robot.elastic_force(q)
    ) @ qd_tangent
    assert_allclose(Ud, potential_expected, rtol=1e-12, atol=1e-12)

    _, Td = jax.jvp(
        lambda q_, qd_: robot.kinetic_energy(q_, qd_),
        (q, qd),
        (qd_tangent, qdd),
    )
    _, kinetic_expected = jax.jvp(
        lambda q_, qd_: robot._kinetic_energy(q_, qd_),
        (q, qd),
        (qd_tangent, qdd),
    )
    assert_allclose(Td, kinetic_expected, rtol=1e-12, atol=1e-12)

    _, Ed = jax.jvp(
        lambda q_, qd_: robot.total_energy(q_, qd_),
        (q, qd),
        (qd_tangent, qdd),
    )
    assert_allclose(
        Ed,
        kinetic_expected + potential_expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_default_kinetic_energy_custom_jvp_uses_default_dynamics_terms() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    qd_tangent = jnp.array([-0.6, 0.25], dtype=jnp.float64)
    qdd = jnp.array([0.15, -0.35], dtype=jnp.float64)

    _, Td = jax.jvp(
        lambda q_, qd_: robot.kinetic_energy(q_, qd_),
        (q, qd),
        (qd_tangent, qdd),
    )

    _, Cqdd = jax.jvp(
        lambda velocity: robot.dynamics_terms(q, velocity)[1],
        (qd,),
        (qd_tangent,),
    )
    expected_from_dynamics_terms = qd @ robot.inertia_matrix(q) @ qdd
    expected_from_dynamics_terms = expected_from_dynamics_terms + 0.5 * qd @ Cqdd
    _, expected = jax.jvp(
        lambda q_, qd_: robot._kinetic_energy(q_, qd_),
        (q, qd),
        (qd_tangent, qdd),
    )
    assert_allclose(expected_from_dynamics_terms, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(Td, expected, rtol=1e-12, atol=1e-12)
