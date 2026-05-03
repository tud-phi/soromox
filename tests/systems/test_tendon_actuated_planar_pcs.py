import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import TendonActuatedPlanarPCS
from soromox.utils.tolerance import Tolerance
from system_param_builders import planar_pcs_params, tendon_actuated_planar_pcs_params


def _build_planar_robot(num_segments: int = 3) -> TendonActuatedPlanarPCS:
    rho = 1070.0 * jnp.ones((num_segments,))
    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * segment_lengths[:, None]
        ).flatten()
    )
    body = planar_pcs_params(
        base_angle=jnp.array(jnp.pi / 2),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 9.81]),
        young_modulus=5e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
    )
    params = tendon_actuated_planar_pcs_params(
        body=body,
        tendon_distance=2e-2 * jnp.array([[1.0, -1.0]]).repeat(num_segments, axis=0),
    )

    return TendonActuatedPlanarPCS(params=params)


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
