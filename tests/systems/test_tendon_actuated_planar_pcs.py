import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import TendonActuatedPlanarPCS
from soromox.utils.tolerance import Tolerance


def _build_planar_robot(num_segments: int = 3) -> TendonActuatedPlanarPCS:
    rho = 1070.0 * jnp.ones((num_segments,))
    params = {
        "th0": jnp.array(jnp.pi / 2),
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 9.81]),
        "E": 5e3 * jnp.ones((num_segments,)),
        "G": 1e3 * jnp.ones((num_segments,)),
        "d": 2e-2 * jnp.array([[1.0, -1.0]]).repeat(num_segments, axis=0),
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * params["L"][:, None]
        ).flatten()
    )

    return TendonActuatedPlanarPCS(
        num_segments=num_segments,
        params=params,
    )


def test_tendon_length_gradient_matches_actuation_matrix():
    robot = _build_planar_robot(num_segments=3)
    q = jnp.linspace(-0.05, 0.05, robot.num_active_strains, dtype=jnp.float64)

    lengths = robot.tendon_length(q)
    assert lengths.shape == (robot.num_actuators,)

    jac_lengths = jax.jacrev(robot.tendon_length)(q)
    A = robot.actuation_matrix(q)

    assert_allclose(
        jac_lengths,
        A.T,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
