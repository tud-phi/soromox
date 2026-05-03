# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from soromox.control import ReferenceTrajectory


class FakeRobot:
    def __init__(self, actuation_matrix):
        self._actuation_matrix = jnp.asarray(actuation_matrix, dtype=jnp.float64)
        self.num_dofs = self._actuation_matrix.shape[0]
        self.num_actuators = self._actuation_matrix.shape[1]

    def actuation_matrix(self, q):
        return self._actuation_matrix


class BrokenActuationRobot:
    num_dofs = 2
    num_actuators = 2

    def actuation_matrix(self, q):
        raise RuntimeError("cannot evaluate actuation")


class FakeDynamicsRobot(FakeRobot):
    def __init__(self, actuation_matrix):
        super().__init__(actuation_matrix)
        self._M = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        self._C = jnp.array([[0.0, 0.25], [-0.1, 0.0]])
        self._G = jnp.array([0.3, -0.2])
        self._elastic = jnp.array([0.4, 0.1])
        self._D = jnp.array([[0.2, 0.0], [0.0, 0.4]])

    def inertia_matrix(self, q):
        return self._M

    def coriolis_matrix(self, q, qd):
        return self._C

    def gravitational_force(self, q):
        return self._G

    def elastic_force(self, q):
        return self._elastic

    def damping_matrix(self, q):
        return self._D


def make_reference(x_des, xd_des, xdd_des=None):
    x_des = jnp.asarray(x_des, dtype=jnp.float64)
    xd_des = jnp.asarray(xd_des, dtype=jnp.float64)
    if xdd_des is None:
        xdd_des = jnp.zeros_like(x_des)
    xdd_des = jnp.asarray(xdd_des, dtype=jnp.float64)

    return ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_fn=lambda t: x_des,
        xd_des_fn=lambda t: xd_des,
        xdd_des_fn=lambda t: xdd_des,
    )


@pytest.fixture
def fake_robot_factory():
    return FakeRobot


@pytest.fixture
def fake_dynamics_robot_factory():
    return FakeDynamicsRobot


@pytest.fixture
def broken_actuation_robot():
    return BrokenActuationRobot()


@pytest.fixture
def make_reference_factory():
    return make_reference
