# ruff: noqa: E402
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, random
from jax import numpy as jnp

from soromox.utils.numerical_jacobian import approx_derivative


def test_finite_differences(method="2-point"):
    def fun(x: Array):
        return jnp.stack([x[0] * jnp.sin(x[1]), x[0] * jnp.cos(x[1])])

    jac_autodiff_fn = jax.jacfwd(fun)
    jac_numdiff_fn = partial(approx_derivative, fun, method=method)

    rng = random.PRNGKey(0)
    for _i in range(100):
        rng, subrng = random.split(rng)
        x = random.uniform(subrng, (2,), minval=-1.0, maxval=1.0)

        jac_autodiff = jac_autodiff_fn(x)
        jac_numdiff = jac_numdiff_fn(x)
        print(
            "x = ",
            x,
            "\njac_autodiff = \n",
            jac_autodiff,
            "\njac_numdiff = \n",
            jac_numdiff,
        )

        error_jac = jnp.linalg.norm(jac_autodiff - jac_numdiff)
        print("error_jac = ", error_jac)

        if not jnp.allclose(jac_autodiff, jac_numdiff, atol=1e-6):
            raise ValueError("Jacobian mismatch!")


if __name__ == "__main__":
    test_finite_differences(method="2-point")
    test_finite_differences(method="3-point")
