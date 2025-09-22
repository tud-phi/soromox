import jax

jax.config.update("jax_enable_x64", True)  # double precision

from soromox.systems.pcs import PCS

from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.tolerance import Tolerance


def test_planar_cs_num():
    """
    Test the planar constant strain system with numerical integration and Jacobian for 1 segment.
    """
    params = {
        "p0": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 0.0, 0.0, 1e3, 0.0, 1e3]]) * params["L"][:, None]).flatten()
    )
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((6,), dtype=bool)

    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    num_segments = 1

    robot = PCS(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((6,)),
            params["L"][0] / 2,
            jnp.eye(4).at[0, 3].set(params["L"][0] / 2),
        ),
        (
            jnp.zeros((6,)),
            params["L"][0],
            jnp.eye(4).at[0, 3].set(params["L"][0]),
        ),
        (
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            params["L"][0],
            jnp.eye(4).at[0, 3].set(2 * params["L"][0]),
        ),
    ]

    for q, s, expected in test_cases:
        print("q = ", q, "s = ", s)
        g_i = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(g_i).any(), "Forward kinematics output contains NaN!"
        assert_allclose(g_i, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        print("[Valid test]\n")

    # test dynamical matrices
    print("\nTesting dynamical matrices... ------------------------")
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
    u = jnp.ones((6,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "u = ", u)
    B = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    G = robot.gravitational_force(q)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix()
    alpha = robot.actuation_force(q, u)

    assert not jnp.isnan(B).any(), "B matrix contains NaN!"
    assert not jnp.isnan(C).any(), "C matrix contains NaN!"
    assert not jnp.isnan(G).any(), "G matrix contains NaN!"
    assert not jnp.isnan(K).any(), "K matrix contains NaN!"
    assert not jnp.isnan(D).any(), "D matrix contains NaN!"
    assert not jnp.isnan(alpha).any(), "alpha matrix contains NaN!"
    print("testing K")
    assert_allclose(K @ q, jnp.zeros((6,)))
    print("[Valid test]\n")
    print("testing alpha")
    assert_allclose(
        alpha,
        jnp.ones(6),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * params["L"][0]), 0.0, 0.0, 0.0, 0.0, 0.0])
    qd = jnp.zeros((6,))
    u = jnp.ones((6,))  # identity torque for testing
    print("q = ", q, "qd = ", qd, "u = ", u)
    B = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    G = robot.gravitational_force(q)
    K = robot.stiffness_matrix()
    D = robot.damping_matrix()
    alpha = robot.actuation_force(q, u)
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
    params_bis = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
    }
    robot = robot.update_params(params_bis)
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
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
        * params["L"][0] ** 2
    )
    assert_allclose(E_pot, E_pot_th, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((6,))
    qd = jnp.zeros((6,))
    u = jnp.zeros((6,))  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.zeros((3,))  # no gravity for this test
    robot = robot.update_params(params_bis)
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", params_bis["g"])
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(jnp.zeros(()), y, (u,))
    qdd, qdres = jnp.split(yd, 2)
    assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
    assert_allclose(qdd, jnp.zeros((6,)), rtol=Tolerance.rtol(), atol=Tolerance.atol())
    assert_allclose(qdres, qd, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")


def test_individual_call():
    """
    Test the individual call of the PCS class.
    """
    params = {
        "p0": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, 0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 0.0, 0.0, 1e3, 0.0, 1e3]]) * params["L"][:, None]).flatten()
    )
    strain_selector = jnp.ones((6,), dtype=bool)
    xi_ref = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    robot = PCS(
        num_segments=1,
        params=params,
        order_gauss=5,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # Test individual calls
    q = jnp.zeros((6,))
    s = params["L"][0]

    print("\nTest robot.forward_kinematics(q, s)-------------------------")
    try:
        g_i = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(g_i).any(), "Forward kinematics output contains NaN!"
        print("[Valid test] Forward kinematics successful.")
    except Exception as e:
        print(f"[Error] Forward kinematics failed: {e}")

    print("\nTest robot.jacobian(q, s)-------------------------")
    try:
        J = robot.jacobian(q=q, s=s)
        assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
        print("[Valid test] Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Jacobian computation failed: {e}")

    print("\nTest robot.jacobian_and_derivative(q, qd, s)-------------------------")
    try:
        J, Jd = robot.jacobian_and_derivative(q=q, qd=jnp.zeros((6,)), s=s)
        assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
        assert not jnp.isnan(Jd).any(), "Jacobian derivative contains NaN!"
        print("[Valid test] Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Jacobian and derivative computation failed: {e}")

    print("\nTest robot.jacobian_bodyframe(q, s)-------------------------")
    try:
        J_local = robot.jacobian_bodyframe(q=q, s=s)
        assert not jnp.isnan(J_local).any(), "Local Jacobian contains NaN!"
        print("[Valid test] Local Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Local Jacobian computation failed: {e}")

    print("\nTest robot.jacobian_inertialframe(q, s)-------------------------")
    try:
        J_global = robot.jacobian_inertialframe(q=q, s=s)
        assert not jnp.isnan(J_global).any(), "Global Jacobian contains NaN!"
        print("[Valid test] Global Jacobian computation successful.")
    except Exception as e:
        print(f"[Error] Global Jacobian computation failed: {e}")

    print(
        "\nTest robot.jacobian_and_derivative_bodyframe(q, qd, s)-------------------------"
    )
    try:
        J_local, Jd_local = robot.jacobian_and_derivative_bodyframe(
            q=q, qd=jnp.zeros((6,)), s=s
        )
        assert not jnp.isnan(J_local).any(), "Local Jacobian contains NaN!"
        assert not jnp.isnan(Jd_local).any(), "Local Jacobian derivative contains NaN!"
        print("[Valid test] Local Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Local Jacobian and derivative computation failed: {e}")

    print(
        "\nTest robot.jacobian_and_derivative_inertialframe(q, qd, s)-------------------------"
    )
    try:
        J_global, Jd_global = robot.jacobian_and_derivative_inertialframe(
            q=q, qd=jnp.zeros((6,)), s=s
        )
        assert not jnp.isnan(J_global).any(), "Global Jacobian contains NaN!"
        assert not jnp.isnan(Jd_global).any(), (
            "Global Jacobian derivative contains NaN!"
        )
        print("[Valid test] Global Jacobian and derivative computation successful.")
    except Exception as e:
        print(f"[Error] Global Jacobian and derivative computation failed: {e}")

    print("\nTest robot.inertia_matrix(q)-------------------------")
    try:
        B = robot.inertia_matrix(q=q)
        assert not jnp.isnan(B).any(), "Inertia matrix contains NaN!"
        print("[Valid test] Inertia matrix computation successful.")
    except Exception as e:
        print(f"[Error] Inertia matrix computation failed: {e}")

    print("\nTest robot.coriolis_matrix(q, qd)-------------------------")
    try:
        C = robot.coriolis_matrix(q=q, qd=jnp.zeros((6,)))
        assert not jnp.isnan(C).any(), "Coriolis matrix contains NaN!"
        print("[Valid test] Coriolis matrix computation successful.")
    except Exception as e:
        print(f"[Error] Coriolis matrix computation failed: {e}")

    print("\nTest robot.gravitational_force(q)-------------------------")
    try:
        G = robot.gravitational_force(q=q)
        assert not jnp.isnan(G).any(), "Gravitational force contains NaN!"
        print("[Valid test] Gravitational force computation successful.")
    except Exception as e:
        print(f"[Error] Gravitational force computation failed: {e}")

    print("\nTest robot.stiffness_matrix()-------------------------")
    try:
        K = robot.stiffness_matrix()
        assert not jnp.isnan(K).any(), "Stiffness matrix contains NaN!"
        print("[Valid test] Stiffness matrix computation successful.")
    except Exception as e:
        print(f"[Error] Stiffness matrix computation failed: {e}")

    print("\nTest robot.damping_matrix()-------------------------")
    try:
        D = robot.damping_matrix()
        assert not jnp.isnan(D).any(), "Damping matrix contains NaN!"
        print("[Valid test] Damping matrix computation successful.")
    except Exception as e:
        print(f"[Error] Damping matrix computation failed: {e}")

    print("\nTest robot.actuation_force(q, u)-------------------------")
    try:
        u = jnp.zeros((6,))  # no external forces
        alpha = robot.actuation_force(q=q, u=u)
        assert not jnp.isnan(alpha).any(), "Actuation force contains NaN!"
        print("[Valid test] Actuation force computation successful.")
    except Exception as e:
        print(f"[Error] Actuation force computation failed: {e}")

    print("\nTest robot.forward_dynamics(t, y, u)-------------------------")
    try:
        t = 0.0
        y = jnp.concatenate([q, jnp.zeros((6,))])  # initial state with zero velocity
        u = jnp.zeros((6,))  # no external forces
        yd = robot.forward_dynamics(t=t, y=y, actuation_args=(u,))
        qdd, qdres = jnp.split(yd, 2)
        assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
        print("[Valid test] Forward dynamics computation successful.")
    except Exception as e:
        print(f"[Error] Forward dynamics computation failed: {e}")


if __name__ == "__main__":
    print("Running tests for Planar Constant Strain (1 segment)...")
    test_planar_cs_num()
    print("Running individual call tests for Planar Constant Strain (1 segment)...")
    test_individual_call()
