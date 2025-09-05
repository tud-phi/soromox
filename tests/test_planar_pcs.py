import jax
from jax import Array
from jax import numpy as jnp
import numpy as onp
from numpy.testing import assert_allclose
import pytest
from typing import Optional

from soromox.systems.planar_pcs import PlanarPCS
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)  # double precision


def make_planar_pcs(num_segments: int = 2, th0: float = jnp.pi / 2, xi_ref: Optional[Array] = None):
    """
    Create a planar constant strain model.
    """
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
    params = {
        "th0": jnp.array(th0),  # initial orientation angle [rad]
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, -9.81]),  # gravity vector [m/s^2] UP!
        "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
            * params["L"][:, None]
        ).flatten()
    )
    
    model = PlanarPCS(
        num_segments=num_segments, 
        params=params, 
        xi_ref=xi_ref,
    )
    
    return model, params

def random_q(model, key=jax.random.PRNGKey(0), scale=0.1):
    n = int(model.num_active_strains.item())
    return scale * jax.random.normal(key, (n,))

def constant_strain_inverse_kinematics_fn(params, xi_ref, chi, s) -> Array:
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
    q = xi - xi_ref
    return q


def test_planar_cs_num():
    """
    Test the planar constant strain system with numerical integration and Jacobian for 1 segment.
    """
    # params = {
    #     "th0": jnp.array(0.0),  # initial orientation angle [rad]
    #     "L": jnp.array([1e-1]),
    #     "r": jnp.array([2e-2]),
    #     "rho": 1000 * jnp.ones((1,)),
    #     "g": jnp.array([0.0, -9.81]),
    #     "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
    #     "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    # }
    # params["D"] = 1e-3 * jnp.diag(
    #     (jnp.array([[1e0, 1e3, 1e3]]) * params["L"][:, None]).flatten()
    # )
    # # activate all strains (i.e. bending, shear, and axial)
    # strain_selector = jnp.ones((3,), dtype=bool)

    # xi_ref = jnp.array([0.0, 0.0, 1.0])
    xi_ref = None

    # num_segments = 1

    # robot = PlanarPCS(
    #     num_segments=num_segments,
    #     params=params,
    #     strain_selector=strain_selector,
    #     xi_ref=xi_ref,
    # )
    robot, params = make_planar_pcs(num_segments=1, th0=0.0, xi_ref=xi_ref)

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics
    print("\nTesting forward kinematics... ------------------------")
    test_cases = [
        (
            jnp.zeros((3,)),
            robot.L[0] / 2,
            jnp.array([0.0, robot.L[0] / 2, 0.0]),
        ),
        (
            jnp.zeros((3,)), 
            robot.L[0], 
            jnp.array([0.0, robot.L[0], 0.0])
        ),
        (
            jnp.array([0.0, 1.0, 0.0]),
            robot.L[0],
            jnp.array([0.0, 2 * robot.L[0], 0.0]),
        ),
        (
            jnp.array([0.0, 0.0, 1.0]),
            robot.L[0],
            robot.L[0] * jnp.array([0.0, 1.0, 1.0]),
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
    u = jnp.ones((3,))  # identity torque for testing
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
    assert_allclose(K @ q, jnp.zeros((3,)))
    print("[Valid test]\n")
    print("testing alpha")
    assert_allclose(
        alpha,
        jnp.ones(3),
    )
    print("[Valid test]\n")

    q = jnp.array([jnp.pi / (2 * robot.L[0]), 0.0, 0.0])
    qd = jnp.zeros((3,))
    u = jnp.ones((3,))  # identity torque for testing
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
    E_pot_th = jnp.array(0.0)
    assert_allclose(E_pot, E_pot_th, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test jacobian
    print("\nTesting jacobian... ------------------------")
    chi = robot.forward_kinematics(q=q, s=robot.L[0])
    print("q = ", q, "s = ", robot.L[0])
    J = robot.jacobian(q, s=robot.L[0])
    assert not jnp.isnan(J).any(), "Jacobian contains NaN!"
    print("Jacobian J =\n", J)
    # Test the differential relation: delta_chi ≈ J * delta_q
    print("Testing differential relation: delta_chi ≈ J * delta_q")
    delta_q = jnp.array([1e-6, -1e-6, 2e-6])
    chi_plus = robot.forward_kinematics(q=q + delta_q, s=params["L"][0])
    chi_pred = chi + J @ delta_q
    assert_allclose(chi_plus, chi_pred, rtol=Tolerance.rtol(), atol=Tolerance.atol())
    print("[Valid test]\n")

    # test forward dynamics
    print("\nTesting forward dynamics... ------------------------")
    q = jnp.zeros((3,))
    qd = jnp.zeros((3,))
    u = jnp.zeros((3,))  # no external forces
    params_bis = params.copy()
    params_bis["g"] = jnp.zeros((2,))  # no gravity for this test
    robot = robot.update_params(params_bis)
    print("q = ", q, "qd = ", qd, "u = ", u, "g = ", params_bis["g"])
    y = jnp.concatenate([q, qd])
    yd = robot.forward_dynamics(0.0, y, (u,))
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
        robot_ik, params_ik = make_planar_pcs(num_segments=1, th0=ik_th0)
        for q in ik_q_ls:
            s = params_ik["L"][0]
            print("q = ", q, "s = ", s, "th0 = ", ik_th0)
            chi = robot_ik.forward_kinematics(q=q, s=s)
            assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
            q_ik = constant_strain_inverse_kinematics_fn(params_ik, robot_ik.xi_ref, chi, s)
            assert not jnp.isnan(q_ik).any(), "Inverse kinematics output contains NaN!"
            assert_allclose(q, q_ik, rtol=Tolerance.rtol(), atol=Tolerance.atol())
            print("[Valid test]\n")


def test_individual_call():
    """
    Test the individual call of the PlanarPCS class.
    """
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "L": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e8 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e7 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(
        (jnp.array([[1e0, 1e3, 1e3]]) * params["L"][:, None]).flatten()
    )
    strain_selector = jnp.ones((3,), dtype=bool)
    strain_selector = strain_selector.at[2].set(
        False
    )  # disable axial strain for this test
    xi_ref = jnp.array([0.0, 0.0, 1.0])

    robot = PlanarPCS(
        num_segments=1,
        params=params,
        strain_selector=strain_selector,
        xi_ref=xi_ref,
    )

    # Test individual calls
    q = jnp.zeros((2,))
    s = params["L"][0]

    print("\nTest robot.forward_kinematics(q, s)-------------------------")
    try:
        chi = robot.forward_kinematics(q=q, s=s)
        assert not jnp.isnan(chi).any(), "Forward kinematics output contains NaN!"
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
        J, Jd = robot.jacobian_and_derivative(q=q, qd=jnp.zeros((2,)), s=s)
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
            q=q, qd=jnp.zeros((2,)), s=s
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
            q=q, qd=jnp.zeros((2,)), s=s
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
        C = robot.coriolis_matrix(q=q, qd=jnp.zeros((2,)))
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
        u = jnp.zeros((2,))  # no external forces
        alpha = robot.actuation_force(q=q, u=u)
        assert not jnp.isnan(alpha).any(), "Actuation force contains NaN!"
        print("[Valid test] Actuation force computation successful.")
    except Exception as e:
        print(f"[Error] Actuation force computation failed: {e}")

    print("\nTest robot.forward_dynamics(t, y, u)-------------------------")
    try:
        t = 0.0
        y = jnp.concatenate([q, jnp.zeros((2,))])  # initial state with zero velocity
        u = jnp.zeros((2,))  # no external forces
        yd = robot.forward_dynamics(t=t, y=y, actuation_args=(u,))
        qdd, qdres = jnp.split(yd, 2)
        assert not jnp.isnan(qdd).any(), "Forward dynamics output contains NaN!"
        print("[Valid test] Forward dynamics computation successful.")
    except Exception as e:
        print(f"[Error] Forward dynamics computation failed: {e}")


@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_jacobian_inertialframe_matches_autodiff(s):
    model, params = make_planar_pcs(2)
    q = random_q(model, jax.random.PRNGKey(1), scale=0.05)

    # J via class
    J_impl = model.jacobian_inertialframe(q, s)

    # J via autodiff from forward_kinematics
    def f(q_):
        return model.forward_kinematics(q_, s)  # [theta, x, y] in inertial reference frame
    J_ad = jax.jacfwd(f)(q)

    assert jnp.allclose(J_impl, J_ad, rtol=1e-6, atol=1e-7), \
        f"\nJ_impl:\n{onp.array(J_impl)}\nJ_ad:\n{onp.array(J_ad)}"

@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_velocity_consistency(s):
    model, params = make_planar_pcs(2)
    key = jax.random.PRNGKey(4)
    q  = random_q(model, key, scale=0.03)
    qd = random_q(model, jax.random.split(key)[0], scale=0.1)

    J = model.jacobian_inertialframe(q, s)
    xdot_pred = J @ qd  # (3,)

    # Finite time difference on kinematics
    dt = 1e-6
    x1 = model.forward_kinematics(q + dt*qd, s)
    x0 = model.forward_kinematics(q, s)
    xdot_fd = (x1 - x0) / dt

    assert jnp.allclose(xdot_pred, xdot_fd, rtol=5e-5, atol=5e-7), \
        f"\npred: {onp.array(xdot_pred)}\nfd: {onp.array(xdot_fd)}"

@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_jacobian_inertialframe_matches_central_differences(s):
    model, params = make_planar_pcs(2)
    q = random_q(model, jax.random.PRNGKey(5), scale=0.02)
    eps = 1e-6

    # J by class implementation
    J_impl = model.jacobian_inertialframe(q, s)

    # J by central finite differences
    n = q.shape[0]
    J_fd = onp.zeros((3, n))
    eye = onp.eye(n)
    for j in range(n):
        qp = q + eps * eye[j]
        qm = q - eps * eye[j]
        fp = onp.array(model.forward_kinematics(qp, s))
        fm = onp.array(model.forward_kinematics(qm, s))
        J_fd[:, j] = (fp - fm) / (2 * eps)

    assert onp.allclose(onp.array(J_impl), J_fd, rtol=5e-5, atol=5e-7), \
        f"\nJ_impl:\n{onp.array(J_impl)}\nJ_fd:\n{J_fd}"
        
@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_Jd_bodyframeframe_matches_jvp_directional_derivative(s):
    model, params = make_planar_pcs(2)
    key = jax.random.PRNGKey(3)
    q = random_q(model, key, scale=0.05)
    qd = random_q(model, jax.random.split(key)[0], scale=0.2)

    J_impl, Jd_impl = model.jacobian_and_derivative_bodyframe(q, qd, s)

    # Directionnal derivative via jvp: d/dt J(q(t)) |_{t=0}, q(t)=q + t*qd
    def J_of_q(q_):
        return model.jacobian_bodyframe(q_, s)
    _, Jdot_dir = jax.jvp(J_of_q, (q,), (qd,))

    assert jnp.allclose(Jd_impl, Jdot_dir, rtol=1e-6, atol=1e-7), \
        f"\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jdot_dir)}"
    
@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_Jd_inertialframe_matches_jvp_directional_derivative(s):
    model, params = make_planar_pcs(2)
    key = jax.random.PRNGKey(3)
    q = random_q(model, key, scale=0.05)
    qd = random_q(model, jax.random.split(key)[0], scale=0.2)

    J_impl, Jd_impl = model.jacobian_and_derivative_inertialframe(q, qd, s)

    # Directionnal derivative via jvp: d/dt J(q(t)) |_{t=0}, q(t)=q + t*qd
    def J_of_q(q_):
        return model.jacobian_inertialframe(q_, s)
    _, Jdot_dir = jax.jvp(J_of_q, (q,), (qd,))

    assert jnp.allclose(Jd_impl, Jdot_dir, rtol=1e-6, atol=1e-7), \
        f"\nJd_impl:\n{onp.array(Jd_impl)}\nJd_jvp:\n{onp.array(Jdot_dir)}"


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_gravity_matches_potential_gradient(num_segments):
    robot, params = make_planar_pcs(num_segments)

    key = jax.random.PRNGKey(3)
    q = random_q(robot, key, scale=0.05)

    G = robot.gravitational_force(q)
    dU_dq = jax.grad(robot.gravitational_energy)(q)

    # With current convention, G equals ∂U/∂q
    assert_allclose(G, dU_dq, rtol=Tolerance.rtol(), atol=Tolerance.atol())


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
