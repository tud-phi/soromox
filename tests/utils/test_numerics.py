import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils._numerics import eps_for_dtype

jax.config.update("jax_enable_x64", True)


def test_eps_for_dtype_uses_dtype_machine_epsilon_when_none():
    eps = eps_for_dtype(None, jnp.float64)

    assert eps.shape == ()
    assert eps.dtype == jnp.float64
    assert_allclose(eps, jnp.finfo(jnp.float64).eps, rtol=0.0, atol=0.0)


def test_eps_for_dtype_casts_explicit_eps_to_requested_dtype():
    eps = eps_for_dtype(1e-6, jnp.float32)

    assert eps.shape == ()
    assert eps.dtype == jnp.float32
    assert_allclose(eps, jnp.asarray(1e-6, dtype=jnp.float32), rtol=0.0, atol=0.0)
