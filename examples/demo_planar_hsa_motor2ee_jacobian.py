import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, jacfwd, jacrev, jit, random, vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path
import scipy as sp
from typing import Callable, Dict, Tuple

import soromox
from soromox.parameters.hsa_params import PARAMS_FPU_CONTROL
from soromox.systems import planar_hsa
from soromox.utils.numerical_jacobian import approx_derivative

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(soromox.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)


def factory_fn(
    params: Dict[str, Array], verbose: bool = False
) -> Tuple[Callable, Callable]:
    """
    Factory function for the planar HSA.
    Args:
        params: dictionary with robot parameters
        verbose: flag to print additional information
    Returns:
        phi2chi_static_model_fn: function that maps motor angles to the end-effector pose
        jac_phi2chi_static_model_fn: function that computes the Jacobian between the actuation space and the task-space
    """
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath, strain_selector)
    dynamical_matrices_fn = partial(dynamical_matrices_fn, params)
    forward_kinematics_end_effector_fn = jit(
        partial(forward_kinematics_end_effector_fn, params)
    )
    jacobian_end_effector_fn = jit(partial(jacobian_end_effector_fn, params))

    def residual_fn(q: Array, phi: Array) -> Array:
        qd = jnp.zeros_like(q)
        _, _, G, K, _, alpha = dynamical_matrices_fn(q, qd, phi=phi)
        res = alpha - G - K
        return jnp.square(res).mean()

    # jit the residual function
    residual_fn = jit(residual_fn)
    print("Compiling residual_fn...")
    print(residual_fn(jnp.zeros((3,)), jnp.zeros((2,))))

    # define the Jacobian of the residual function
    jac_residual_fn = jit(jacrev(residual_fn, argnums=0))
    print("Compiling jac_residual_fn...")
    print(jac_residual_fn(jnp.zeros((3,)), jnp.zeros((2,))))

    def phi2q_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        A static model mapping the motor angles to the planar HSA configuration using scipy.optimize.minimize.
        Arguments:
            phi: motor angles
            q0: initial guess for the configuration
        Returns:
            q: planar HSA configuration consisting of (k_be, sigma_sh, sigma_ax)
            aux: dictionary with auxiliary data
        """
        # solve the nonlinear least squares problem
        sol = sp.optimize.minimize(
            fun=lambda q: residual_fn(q, phi).item(),
            x0=q0,
            jac=lambda q: jac_residual_fn(q, phi),
            options={"disp": True} if verbose else None,
        )
        if verbose:
            print(
                "Optimization converged after",
                sol.nit,
                "iterations with residual",
                sol.fun,
            )

        # configuration that minimizes the residual
        q = jnp.array(sol.x)

        aux = dict(
            phi=phi,
            q=q,
            residual=sol.fun,
        )

        return q, aux

    def phi2chi_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        A static model mapping the motor angles to the planar end-effector pose.
        Arguments:
            phi: motor angles
            q0: initial guess for the configuration
        Returns:
            chi: end-effector pose
            aux: dictionary with auxiliary data
        """
        q, aux = phi2q_static_model_fn(phi, q0=q0)
        chi = forward_kinematics_end_effector_fn(q)
        aux["chi"] = chi
        return chi, aux

    def jac_phi2chi_static_model_fn(
        phi: Array, q0: Array = jnp.zeros((3,))
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute the Jacobian between the actuation space and the task-space.
        Arguments:
            phi: motor angles
        """
        # evaluate the static model to convert motor angles into a configuration
        q, aux = phi2q_static_model_fn(phi, q0=q0)
        # approximate the Jacobian between the actuation and the task-space using finite differences
        J_phi2q = approx_derivative(
            fun=lambda _phi: phi2q_static_model_fn(_phi, q0=q0)[0],
            x0=phi,
            f0=q,
        )

        # evaluate the closed-form, analytical jacobian of the forward kinematics
        J_q2chi = jacobian_end_effector_fn(q)

        # evaluate the Jacobian between the actuation and the task-space
        J_phi2chi = J_q2chi @ J_phi2q

        return J_phi2chi, aux

    return phi2chi_static_model_fn, jac_phi2chi_static_model_fn


if __name__ == "__main__":
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
    params = PARAMS_FPU_CONTROL
    phi_max = params["phi_max"].flatten()

    # call the factory function
    phi2chi_static_model_fn, jac_phi2chi_static_model_fn = factory_fn(params)

    # define initial configuration
    q0 = jnp.array([0.0, 0.0, 0.0])

    rng = random.key(seed=0)
    for i in range(10):
        match i:
            case 0:
                phi = jnp.array([0.0, 0.0])
            case 1:
                phi = jnp.array([1.0, 1.0])
            case _:
                rng, subkey = random.split(rng)
                phi = random.uniform(subkey, phi_max.shape, minval=0.0, maxval=phi_max)

        print("i", i)

        chi, aux = phi2chi_static_model_fn(phi, q0=q0)
        print("phi", phi, "q", aux["q"], "chi", chi)

        J_phi2chi, aux = jac_phi2chi_static_model_fn(phi, q0=q0)
        print("J_phi2chi:\n", J_phi2chi)
