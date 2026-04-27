import jax
from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems.soft_robot import CrossSectionGeometry, SoftRobot

jax.config.update("jax_enable_x64", True)


class _PlanarDefaultRobot(SoftRobot):
    L: Array

    def __init__(self):
        super().__init__()
        self.num_dofs = 2
        self.num_actuators = 2
        self.L = jnp.array([1.0, 2.0], dtype=jnp.float64)

    @property
    def length(self) -> Array:
        return jnp.sum(self.L)

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
        return jnp.diag(jnp.array([2.0, 4.0], dtype=q.dtype))

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        return jnp.diag(jnp.array([0.5, 1.0], dtype=q.dtype))

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


class _ActiveEnergyJVPDefaultRobot(_PlanarDefaultRobot):
    def _gravitational_force(self, q: Array) -> Array:
        return jnp.array([1.5, -0.25], dtype=q.dtype)

    def elastic_force(self, q: Array) -> Array:
        return jnp.array([-0.75, 0.5], dtype=q.dtype)

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        del q, qd
        return 1e3 * jnp.eye(2, dtype=jnp.float64)

    def _active_coriolis_force(self, qd: Array) -> Array:
        return jnp.array(
            [
                qd[0] * qd[0] + 2.0 * qd[0] * qd[1],
                qd[0] * qd[1] - 0.5 * qd[1] * qd[1],
            ],
            dtype=qd.dtype,
        )

    def _active_quadrature_forward_dynamics_terms(
        self, q: Array, qd: Array
    ) -> tuple[Array, Array, Array]:
        return (
            self.inertia_matrix(q),
            self._active_coriolis_force(qd),
            self._gravitational_force(q),
        )


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
    assert robot.forward_kinematics_tips(q).shape == (2, 4, 4)
    assert robot.jacobian_tips(q).shape == (2, 6, 2)


def test_default_gravity_force_and_forward_dynamics() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    u = jnp.array([1.5, -2.0], dtype=jnp.float64)
    tau_ext = jnp.array([0.25, -0.75], dtype=jnp.float64)
    y = jnp.concatenate([q, qd])

    G = robot.gravitational_force(q)
    assert_allclose(G, jnp.array([0.4, 3.0], dtype=jnp.float64))

    rhs = (
        robot.actuation_matrix(q) @ u
        + tau_ext
        - robot.coriolis_matrix(q, qd) @ qd
        - G
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ qd
    )
    qdd = jnp.linalg.solve(robot.inertia_matrix(q), rhs)

    assert_allclose(robot.forward_dynamics(0.0, y, (u, tau_ext)), jnp.r_[qd, qdd])


def test_default_energy_custom_jvps_use_force_and_active_dynamics_hooks() -> None:
    robot = _ActiveEnergyJVPDefaultRobot()
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
    _, active_coriolisd = jax.jvp(
        robot._active_coriolis_force,
        (qd,),
        (qd_tangent,),
    )
    kinetic_expected = qd @ robot.inertia_matrix(q) @ qdd
    kinetic_expected = kinetic_expected + 0.5 * qd @ active_coriolisd
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


def test_default_kinetic_energy_custom_jvp_falls_back_to_public_matrices() -> None:
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

    expected = qd @ robot.inertia_matrix(q) @ qdd
    expected = expected + qd @ robot.coriolis_matrix(q, qd_tangent) @ qd
    assert_allclose(Td, expected, rtol=1e-12, atol=1e-12)
