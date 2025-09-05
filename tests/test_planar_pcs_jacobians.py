import jax
import jax.numpy as jnp
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)

from soromox.systems.planar_pcs import PlanarPCS

def make_model(num_segments=2):
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
    params = {
        "th0": jnp.array(jnp.pi / 2),  # initial orientation angle [rad]
        "L": 1e-1 * jnp.ones((num_segments,)),
        "r": 2e-2 * jnp.ones((num_segments,)),
        "rho": rho,
        "g": jnp.array([0.0, 9.81]),  # gravity vector [m/s^2] UP!
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
        order_gauss=5
    )
    
    return model

def random_q(model, key=jax.random.PRNGKey(0), scale=0.1):
    n = int(model.num_active_strains.item())
    return scale * jax.random.normal(key, (n,))

@pytest.mark.parametrize("s", [0.0, 0.05, 0.15, 0.19, 0.20])  # various points, at the edges & inside
def test_jacobian_inertialframe_matches_autodiff(s):
    model = make_model(2)
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
    model = make_model(2)
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
    model = make_model(2)
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
    model = make_model(2)
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
    model = make_model(2)
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

if __name__ == "__main__":
    pytest.main([__file__])