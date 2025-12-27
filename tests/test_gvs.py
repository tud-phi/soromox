import jax

jax.config.update("jax_enable_x64", True)  # use double precision in tests

import jax.numpy as jnp
import numpy as onp
import pytest

from numpy.testing import assert_allclose

from soromox.systems import GVS, PCS
from soromox.systems.gvs import LinkAttributes, JointAttributes, BasisAttributes
import soromox.utils.lie_algebra as lie
from soromox.utils.tolerance import Tolerance

pytestmark = pytest.mark.skip(reason="GVS testing temporarily deactivated")


def build_matched_gvs_pcs(num_segments: int = 1):
    """
    Build a GVS model with constant-strain basis (monomial order 0) and fixed joints
    and a PCS model with the same physical parameters, so their predictions match.

    Note on gravity/signs:
    - PCS uses G = -∫ J^T M Ad_g^{-1} g ds
    - GVS uses G =  ∫ J^T M Ad_g^{-1} g ds
    For equivalent physical gravity (downwards), pass g=[0,0,-9.81] to PCS and
    g=[0,0, 9.81] to GVS so the resulting generalized gravity forces match.
    """
    # Physical parameters (kept simple, identical for each segment)
    Ls = 0.2 * jnp.ones((num_segments,))
    rs = 0.02 * jnp.ones((num_segments,))
    rhos = 1000.0 * jnp.ones((num_segments,))
    E = 1e6 * jnp.ones((num_segments,))
    nu = 0.5 * jnp.ones((num_segments,))
    # Match PCS shear modulus to GVS relation: G = E / (2(1+nu))
    Gpcs = (E / (2 * (1.0 + nu))).reshape(num_segments)

    # GVS definition: constant strain along each link, all 6 strain components enabled
    links = [
        LinkAttributes(
            section="Circular",
            E=float(E[i]),
            nu=float(nu[i]),
            rho=float(rhos[i]),
            eta=0.0,  # no damping in this cross-model comparison
            L=float(Ls[i]),
            r_i=float(rs[i]),
            r_f=float(rs[i]),
        )
        for i in range(num_segments)
    ]
    joints = [JointAttributes(jointtype="Fixed") for _ in range(num_segments)]
    bases = [
        BasisAttributes(
            basistype="Monomial",
            Bdof=[1, 1, 1, 1, 1, 1],
            Bodr=[0, 0, 0, 0, 0, 0],  # order 0 -> spatially constant strain
            xi_ref=[0, 0, 0, 1, 0, 0],
        )
        for _ in range(num_segments)
    ]
    n_gauss_list = [5 for _ in range(num_segments)]

    # Gravity: pick physically consistent vectors as per note
    g = jnp.array([0.0, 0.0, -9.81])

    # initialize the GVS model
    robot_gvs = GVS(
        links_list=links,
        joints_list=joints,
        basis_list=bases,
        n_gauss_list=n_gauss_list,
        gravity_vector=g,
    )

    # PCS definition with identical geometry and material params
    params = {
        "p0": jnp.zeros(6),
        "L": Ls,
        "r": rs,
        "rho": rhos,
        "g": g,
        "E": E,
        "G": Gpcs,
    }
    # zero damping for fair comparison
    params["D"] = jnp.zeros((6 * num_segments, 6 * num_segments))
    robot_pcs = PCS(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
        strain_selector=jnp.ones((6 * num_segments,), dtype=bool),
        xi_ref=jnp.tile(jnp.array([0, 0, 0, 1, 0, 0]), (num_segments, 1)).reshape(
            6 * num_segments
        ),
    )

    return robot_gvs, robot_pcs


def gvs_jacobian_inertialframe_from_body(
    robot_gvs: GVS, q: jnp.ndarray, s: jnp.ndarray
) -> jnp.ndarray:
    """Compute inertial-frame Jacobian for GVS from its body-frame Jacobian.

    This mirrors PCS.jacobian_inertialframe: J_global = Ad_{[R,0;0,1]} @ J_body.
    """
    J_local = robot_gvs.jacobian_bodyframe(q, s)
    g_s = robot_gvs.forward_kinematics(q, s)
    g_s_wo_rot = jnp.block(
        [[g_s[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
    )
    Adj_g = lie.Adjoint_g_SE3(g_s_wo_rot)
    return Adj_g @ J_local


@pytest.mark.parametrize("num_segments", [1, 2])
def test_gvs_equals_pcs_constant_strain(num_segments: int):
    robot_gvs, robot_pcs = build_matched_gvs_pcs(num_segments)

    # pick a representative configuration/velocity
    n = robot_gvs.dof_tot_system
    # small nonzero strains to avoid degenerate cases
    q = jnp.linspace(0.01, 0.01 * n, n)
    qd = jnp.linspace(-0.02, -0.02 * n, n)

    # forward kinematics comparisons at a few points along the rod
    L_total = float(jnp.sum(robot_gvs.V_L))
    s_values = jnp.array([0.0, 0.33 * L_total, 0.75 * L_total, L_total])
    for s in onp.array(s_values):
        g_gvs = robot_gvs.forward_kinematics(q, s)
        g_pcs = robot_pcs.forward_kinematics(q, s)
        assert_allclose(g_gvs, g_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

        # body-frame Jacobian
        Jb_gvs = robot_gvs.jacobian_bodyframe(q, s)
        Jb_pcs = robot_pcs.jacobian_bodyframe(q, s)
        assert_allclose(Jb_gvs, Jb_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

        # inertial-frame Jacobian
        Ji_gvs = gvs_jacobian_inertialframe_from_body(robot_gvs, q, s)
        Ji_pcs = robot_pcs.jacobian_inertialframe(q, s)
        assert_allclose(Ji_gvs, Ji_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # dynamical matrices
    B_gvs = robot_gvs.inertia_matrix(q)
    B_pcs = robot_pcs.inertia_matrix(q)
    assert_allclose(B_gvs, B_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    C_gvs = robot_gvs.coriolis_matrix(q, qd)
    C_pcs = robot_pcs.coriolis_matrix(q, qd)
    assert_allclose(C_gvs, C_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    K_gvs = robot_gvs.stiffness_matrix()
    K_pcs = robot_pcs.stiffness_matrix()
    assert_allclose(K_gvs, K_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # gravitational generalized forces
    G_gvs = robot_gvs.gravitational_force(q).reshape(-1)
    G_pcs = robot_pcs.gravitational_force(q).reshape(-1)
    assert_allclose(G_gvs, G_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # forward dynamics consistency (no actuation, no external torques)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros(n)
    yd_gvs = robot_gvs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
    yd_pcs = robot_pcs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
    assert_allclose(yd_gvs, yd_pcs, rtol=Tolerance.rtol(), atol=Tolerance.atol())


@pytest.mark.parametrize("num_segments", [1])
def test_gvs_autodiff_checks(num_segments: int):
    robot_gvs, _ = build_matched_gvs_pcs(num_segments)

    n = robot_gvs.dof_tot_system
    q = jnp.linspace(0.01, 0.01 * n, n)
    qd = jnp.linspace(0.02, 0.02 * n, n)
    s = jnp.sum(robot_gvs.V_L) * 0.7

    # 1) Jdot_body matches directional derivative of J_body via jvp
    def J_body(q_):
        return robot_gvs.jacobian_bodyframe(q_, s)

    _, Jdot_dir = jax.jvp(J_body, (q,), (qd,))
    Jdot_impl = robot_gvs.jacobian_derivative_bodyframe(q, qd, s)
    assert_allclose(Jdot_impl, Jdot_dir, rtol=Tolerance.rtol(), atol=Tolerance.atol())

    # 2) Translational block of inertial Jacobian matches jacobian of position
    def pos_fn(q_):
        return robot_gvs.forward_kinematics(q_, s)[:3, 3]

    Jpos_ad = jax.jacfwd(pos_fn)(q)  # shape (3, n)
    Ji_gvs = gvs_jacobian_inertialframe_from_body(robot_gvs, q, s)
    Jpos_impl = Ji_gvs[3:6, :]  # translational rows
    assert_allclose(Jpos_impl, Jpos_ad, rtol=Tolerance.rtol(), atol=Tolerance.atol())


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
