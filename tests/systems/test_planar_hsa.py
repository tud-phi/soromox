import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import random
from jax import numpy as jnp
import soromox
from pathlib import Path

from soromox.parameters.hsa_params import PARAMS_FPU_CONTROL as params
from soromox.systems.planar_hsa import PlanarHSA

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(soromox.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)


def test_end_effector_kinematics(seed: int = 0):
    print("Testing end effector kinematics...")
    robot = PlanarHSA(
        sym_exp_filepath=sym_exp_filepath,
        params=params,
    )

    rng = random.PRNGKey(seed)
    for _ in range(10):
        rng, subrng1, subrng2, subrng3, subrng4, subrng5 = random.split(rng, 6)
        kappa_b = random.uniform(
            subrng1,
            (num_segments,),
            minval=-jnp.pi / jnp.mean(params["L"]),
            maxval=jnp.pi / jnp.mean(params["L"]),
        )
        sigma_sh = random.uniform(subrng2, (num_segments,), minval=-0.2, maxval=0.2)
        sigma_a = random.uniform(subrng3, (num_segments,), minval=0.0, maxval=0.5)
        q = jnp.concatenate((kappa_b, sigma_sh, sigma_a), axis=0)

        print("q = ", q)

        # forward kinematics
        chiee = robot.forward_kinematics_end_effector_fn(q)
        # inverse kinematics
        q_rec = robot.inverse_kinematics_end_effector_fn(chiee)

        if not jnp.allclose(q, q_rec, atol=1e-6):
            print("q = ", q)
            print("q_rec = ", q_rec)
            raise ValueError("q != q_rec")


if __name__ == "__main__":
    test_end_effector_kinematics()
