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

    def forward_kinematics(self, q: Array, s: Array) -> Array:
        theta = q[0] * s
        return jnp.array([theta, s * jnp.cos(theta) + q[1], s * jnp.sin(theta)])

    def inertia_matrix(self, q: Array) -> Array:
        return jnp.diag(jnp.array([2.0, 4.0], dtype=q.dtype))

    def coriolis_matrix(self, q: Array, qd: Array) -> Array:
        return jnp.diag(jnp.array([0.5, 1.0], dtype=q.dtype))

    def damping_matrix(self, q: Array) -> Array:
        return jnp.diag(jnp.array([0.1, 0.2], dtype=q.dtype))

    def elastic_force(self, q: Array) -> Array:
        return q

    def gravitational_energy(self, q: Array) -> Array:
        return q[0] ** 2 + 3.0 * q[1]

    def actuation_matrix(self, q: Array) -> Array:
        return jnp.eye(2, dtype=q.dtype)


class _SpatialDefaultRobot(_PlanarDefaultRobot):
    @property
    def is_planar(self) -> bool:
        return False

    def forward_kinematics(self, q: Array, s: Array) -> Array:
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


def test_default_planar_jacobians_and_tip_methods() -> None:
    robot = _PlanarDefaultRobot()
    q = jnp.array([0.2, -0.3], dtype=jnp.float64)
    qd = jnp.array([0.7, -0.4], dtype=jnp.float64)
    s = jnp.array(0.8, dtype=jnp.float64)

    J_expected = jax.jacfwd(lambda q_: robot.forward_kinematics(q_, s))(q)
    assert_allclose(robot.jacobian(q, s), J_expected, rtol=1e-12, atol=1e-12)

    J, Jd = robot.jacobian_and_derivative(q, qd, s)
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
    g, gdot = jax.jvp(lambda q_: robot.forward_kinematics(q_, s), (q,), (qd,))
    omega_hat = gdot[:3, :3] @ g[:3, :3].T
    omega = 0.5 * jnp.array(
        [
            omega_hat[2, 1] - omega_hat[1, 2],
            omega_hat[0, 2] - omega_hat[2, 0],
            omega_hat[1, 0] - omega_hat[0, 1],
        ]
    )
    twist_expected = jnp.concatenate([omega, gdot[:3, 3]])

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
