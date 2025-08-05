import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array
from jax import numpy as jnp
import soromox
from functools import partial
from numpy.testing import assert_allclose
from pathlib import Path

from soromox.systems import euler_lagrangian, planar_pcs_sym
from soromox.utils.tolerance import Tolerance


def constant_strain_inverse_kinematics_fn(params, xi_ref, chi, s) -> Array:
    # split the chi vector into x, y, and th0
    px, py, th = chi
    th0 = params["th0"].item()
    print("th0 = ", th0)
    xi = (
        (th - th0)
        / (2 * s)
        * jnp.array(
            [
                2.0,
                (-jnp.sin(th0) * px + jnp.cos(th0) * py)
                - (jnp.cos(th0) * px + jnp.sin(th0) * py)
                * jnp.sin(th - th0)
                / (jnp.cos(th - th0) - 1),
                -(jnp.cos(th0) * px + jnp.sin(th0) * py)
                - (-jnp.sin(th0) * px + jnp.cos(th0) * py)
                * jnp.sin(th - th0)
                / (jnp.cos(th - th0) - 1),
            ]
        )
    )
    q = xi - xi_ref
    return q


def test_planar_cs():
    sym_exp_filepath = (
        Path(soromox.__file__).parent / "symbolic_expressions" / "planar_pcs_ns-1.dill"
    )
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "l": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3,), dtype=bool)

    xi_ref = jnp.array([0.0, 0.0, 1.0])
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        planar_pcs_sym.factory(sym_exp_filepath, strain_selector, xi_ref)
    )
    forward_dynamics_fn = partial(
        euler_lagrangian.forward_dynamics, dynamical_matrices_fn
    )
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space, dynamical_matrices_fn
    )

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((3,)),
            params["l"][0] / 2,
            jnp.array([0.0, params["l"][0] / 2, 0.0]),
        ),
        (jnp.zeros((3,)), params["l"][0], jnp.array([0.0, params["l"][0], 0.0])),
        (
            jnp.array([0.0, 0.0, 1.0]),
            params["l"][0],
            jnp.array([0.0, 2 * params["l"][0], 0.0]),
        ),
        (
            jnp.array([0.0, 1.0, 0.0]),
            params["l"][0],
            params["l"][0] * jnp.array([1.0, 1.0, 0.0]),
        ),
    ]

    for q, s, expected in test_cases:
        print("q = ", q, "s = ", s)
        chi = forward_kinematics_fn(params, q=q, s=s)
        assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
        assert_allclose(chi, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        print("[Valid test]\n")

    # test inverse kinematics
    print("\nTesting inverse kinematics... ------------------------")
    params_ik = params.copy()
    ik_th0_ls = [-jnp.pi / 2, -jnp.pi / 4, 0.0, jnp.pi / 4, jnp.pi / 2]
    ik_q_ls = [
        jnp.array([0.1, 0.0, 0.0]),
        jnp.array([0.1, 0.0, 0.2]),
        jnp.array([0.1, 0.5, 0.1]),
        jnp.array([1.0, 0.5, 0.2]),
        jnp.array([-1.0, 0.0, 0.0]),
    ]
    for ik_th0 in ik_th0_ls:
        params_ik["th0"] = jnp.array(ik_th0)
        for q in ik_q_ls:
            s = params_ik["l"][0]
            print("q = ", q, "s = ", s, "th0 = ", ik_th0)
            chi = forward_kinematics_fn(params_ik, q=q, s=s)
            assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
            q_ik = constant_strain_inverse_kinematics_fn(params_ik, xi_ref, chi, s)
            assert not jnp.isnan(q_ik).any(), "Inverse kinematics output contains NaN!"
            assert_allclose(q, q_ik, rtol=Tolerance.rtol(), atol=Tolerance.atol())
            print("[Valid test]\n")

    # test dynamical matrices
    print("\nTesting dynamical matrices... ------------------------")
    q = jnp.zeros((3,))
    q_d = jnp.zeros((3,))
    print("q = ", q, "q_d = ", q_d)
    B, C, G, K, D, A = dynamical_matrices_fn(params, q, q_d)
    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(A).any(), "A matrix contains NaN!"
    print("testing K")
    assert_allclose(K, jnp.zeros((3,)))
    print("[Valid test]\n")
    print("testing A")
    assert_allclose(
        A,
        jnp.eye(3),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * params["l"][0]), 0.0, 0.0])
    q_d = jnp.zeros((3,))
    print("q = ", q, "q_d = ", q_d)
    B, C, G, K, D, A = dynamical_matrices_fn(params, q, q_d)
    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(A).any(), "A matrix contains NaN!"

    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("A =\n", A)
    print("[To check]")

    # test energies
    print("\nTesting energies... ------------------------")
    kinetic_energy_fn = auxiliary_fns["kinetic_energy_fn"]
    potential_energy_fn = auxiliary_fns["potential_energy_fn"]

    q = jnp.zeros((3,))
    q_d = jnp.zeros((3,))
    print("q = ", q, "q_d = ", q_d)

    print("Testing kinetic energy...")
    E_kin = kinetic_energy_fn(params, q, q_d)
    assert not jnp.isnan(E_kin).any(), "Kinetic energy contains NaN!"
    E_kin_th = 0.0
    assert_allclose(E_kin, E_kin_th, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    print("Testing potential energy...")
    E_pot = potential_energy_fn(params, q)
    assert not jnp.isnan(E_pot).any(), "Potential energy contains NaN!"
    E_pot_th = jnp.array(
        0.5
        * params["rho"][0]
        * jnp.pi
        * params["r"][0] ** 2
        * jnp.linalg.norm(params["g"])
        * params["l"][0] ** 2
    )
    assert_allclose(E_pot, E_pot_th, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test jacobian
    print("\nTesting jacobian... ------------------------")
    jacobian_fn = auxiliary_fns["jacobian_fn"]
    chi = forward_kinematics_fn(params, q=q, s=params["l"][0])
    print("q = ", q, "s = ", params["l"][0])
    J = jacobian_fn(params, q, s=params["l"][0])
    assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
    print("Jacobian J =\n", J)
    # Test the differential relation: delta_chi ≈ J * delta_q
    print("Testing differential relation: delta_chi ≈ J * delta_q")
    delta_q = jnp.array([1e-6, -1e-6, 2e-6])
    chi_plus = forward_kinematics_fn(params, q=q + delta_q, s=params["l"][0])
    chi_pred = chi + J @ delta_q
    assert_allclose(chi_plus, chi_pred, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((3,))
    q_d = jnp.zeros((3,))
    tau = jnp.array([0.0, 0.0, 0.0])  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.array([0.0, 0.0])  # no gravity for this test
    print("q = ", q, "q_d = ", q_d, "tau = ", tau, "g = ", params_bis["g"])
    q_dd = forward_dynamics_fn(params_bis, q, q_d, tau)
    assert not jnp.isnan(q_dd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(q_dd, jnp.zeros((3,)), rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test nonlinear state space
    print("\nTesting nonlinear state space... ------------------------")
    x = jnp.concatenate([q, q_d])
    print("x = ", x, "tau = ", tau)
    x_dot = nonlinear_state_space_fn(params, x, tau)
    assert not jnp.isnan(x_dot).any(), "Nonlinear state space output contains NaN!"
    print("x_dot = ", x_dot)
    print("[To check]")


if __name__ == "__main__":
    test_planar_cs()
