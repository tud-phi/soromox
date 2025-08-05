import jax

from jsrm.systems.planar_pcs import PlanarPCS

from jax import Array
from jax import numpy as jnp
import jsrm
from functools import partial
from numpy.testing import assert_allclose
from pathlib import Path

from jsrm.systems import euler_lagrangian
from jsrm.utils.tolerance import Tolerance

from typing import Optional, Literal

jax.config.update("jax_enable_x64", True)  # double precision


def constant_strain_inverse_kinematics_fn(params, xi_star, chi, s) -> Array:
    # split the chi vector into x, y, and th0
    th, px, py = chi
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
    q = xi - xi_star
    return q


def test_planar_cs_num(
):
    """
    Test the planar constant strain system with numerical integration and Jacobian for 1 segment.
    """
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "l": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.array([[1e0, 1e3, 1e3]])
            * params["l"][:, None]
        ).flatten()
    )
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3,), dtype=bool)

    xi_star = jnp.array([0.0, 0.0, 1.0])

    num_segments = 1
    
    robot = PlanarPCS(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_star=xi_star,
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
            jnp.array([0.0, 0.0, params["l"][0] / 2]),
        ),
        (
            jnp.zeros((3,)), 
            params["l"][0], 
            jnp.array([0.0, 0.0, params["l"][0]])
        ),
        (
            jnp.array([0.0, 0.0, 1.0]),
            params["l"][0],
            jnp.array([0.0, 0.0, 2 * params["l"][0]]),
        ),
        (
            jnp.array([0.0, 1.0, 0.0]),
            params["l"][0],
            params["l"][0] * jnp.array([0.0, 1.0, 1.0]),
        ),
    ]

    for q, s, expected in test_cases:
        print("q = ", q, "s = ", s)
        chi = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
        assert_allclose(chi, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        print("[Valid test]\n")

    # test dynamical matrices
    print("\nTesting dynamical matrices... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    tau = jnp.ones((3,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "tau = ", tau)
    B, C, G, K, D, alpha = robot.dynamical_matrices(q, qd, (tau,))
    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(alpha).any(), "alpha matrix contains NaN!"
    print("testing K")
    assert_allclose(K@q, jnp.zeros((3,)))
    print("[Valid test]\n")
    print("testing alpha")
    assert_allclose(
        alpha,
        jnp.ones(3),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * params["l"][0]), 0.0, 0.0])
    qd = jnp.zeros((3,))
    tau = jnp.ones((3,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "tau = ", tau)
    B, C, G, K, D, alpha = robot.dynamical_matrices(q, qd, (tau,))
    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(alpha).any(), "alpha matrix contains NaN!"

    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("alpha =\n", alpha)
    print("[To check]")

    # test energies
    print("\nTesting energies... ------------------------")

    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    print("q = ", q, "qd = ", qd)

    print("Testing kinetic energy...")
    E_kin = robot.kinetic_energy(q, qd)
    assert not jnp.isnan(E_kin).any(), "Kinetic energy contains NaN!"
    E_kin_th = 0.0
    assert_allclose(E_kin, E_kin_th, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    print("Testing potential energy...")
    E_pot = robot.potential_energy(q)
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
    chi = robot.forward_kinematics(q=q, s=params["l"][0])
    print("q = ", q, "s = ", params["l"][0])
    J = robot.jacobian(q, s=params["l"][0])
    assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
    print("Jacobian J =\n", J)
    # Test the differential relation: delta_chi ≈ J * delta_q
    print("Testing differential relation: delta_chi ≈ J * delta_q")
    delta_q = jnp.array([1e-6, -1e-6, 2e-6])
    chi_plus = robot.forward_kinematics(q=q + delta_q, s=params["l"][0])
    chi_pred = chi + J @ delta_q
    assert_allclose(chi_plus, chi_pred, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    tau = jnp.array([0.0, 0.0, 0.0])  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.array([0.0, 0.0])  # no gravity for this test
    robot = PlanarPCS(
        num_segments=num_segments,
        params=params_bis,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_star=xi_star,
    )
    print("q = ", q, "qd = ", qd, "tau = ", tau, "g = ", params_bis["g"])
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(0.0, y, (tau,))
    qdd, qdres = jnp.split(yd, 2)
    assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(qdd, jnp.zeros((3,)), rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(qdres, qd, rtol=Tolerance.rtol(), atol=Tolerance.atol())
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
        robot_ik = PlanarPCS(
                num_segments=1,
                params=params_ik,
                order_gauss=5,
                strain_selector=strain_selector,
                xi_star=xi_star,
            )
        for q in ik_q_ls:
            s = params_ik["l"][0]
            print("q = ", q, "s = ", s, "th0 = ", ik_th0)
            chi = robot_ik.forward_kinematics(q=q, s=s)
            assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
            q_ik = constant_strain_inverse_kinematics_fn(params_ik, xi_star, chi, s)
            assert not jnp.isnan(q_ik).any(), "Inverse kinematics output contains NaN!"
            assert_allclose(q, q_ik, rtol=Tolerance.rtol(), atol=Tolerance.atol())
            print("[Valid test]\n")


if __name__ == "__main__":
    test_planar_cs_num()
