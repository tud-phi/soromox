import jax

jax.config.update("jax_enable_x64", True)  # double precision

from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS

from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.utils.tolerance import Tolerance


def test_actuation_matrix_pcs():
    """
    Test ...
    """
    def createRobot(num_segments, spec_params, tendon_params):
        rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
        params = {
            "p0": jnp.zeros(6),  # Initial position and orientation
            "r": 1. * jnp.ones((num_segments,)),  # default: 2e-2
            "rho": rho,
            "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
            "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
            "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        }
        params.update(spec_params)#["L"] = 1e-1 * jnp.ones((num_segments,))
        params["D"] = 1e-3 * jnp.diag(
            (
                jnp.repeat(
                    jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
                )
                * params["L"][:, None]
            ).flatten()
        )

        robot = TendonActuatedPCS(
            num_segments=num_segments,
            params=params,
            order_gauss=5,
            # strain_selector=strain_selector,
            # xi_ref=xi_ref,
            tendon_routing_params=tendon_params,
        )
        
        return robot, jnp.zeros(6*num_segments)
    
    def A_special_case(spec_params, tendon_params):
        l = jnp.sum(spec_params["L"])
        ry, rz, my, mz = tendon_params["ry"][0], tendon_params["rz"][0], tendon_params["my"][0], tendon_params["mz"][0]
        A = jnp.array([
            l*(-my*rz + mz*ry),
            l**2*mz/2 + l*rz,
            -l**2*my/2 - l*ry,
            l,
            l*my,
            l*mz
        ])

        return A / jnp.sqrt(my**2 + mz**2 + 1)
    
    def actuation_basis(tendon_params, xi, s):
        ry, rz, my, mz = tendon_params["ry"][0], tendon_params["rz"][0], tendon_params["my"][0], tendon_params["mz"][0]
        xi_1, xi_2, xi_3, xi_4, xi_5, xi_6 = xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]
        Phi_a = jnp.array([
            (my*s + ry)*(mz + xi_1*(my*s + ry) + xi_6) + (-mz*s - rz)*(my - xi_1*(mz*s + rz) + xi_5),
            (mz*s + rz)*(xi_2*(mz*s + rz) - xi_3*(my*s + ry) + xi_4),
            (-my*s - ry)*(xi_2*(mz*s + rz) - xi_3*(my*s + ry) + xi_4),
            xi_2*(mz*s + rz) - xi_3*(my*s + ry) + xi_4,
            my - xi_1*(mz*s + rz) + xi_5,
            mz + xi_1*(my*s + ry) + xi_6
        ])
        norm = jnp.sqrt((my - xi_1*(mz*s + rz) + xi_5)**2 + (mz + xi_1*(my*s + ry) + xi_6)**2 + (xi_2*(mz*s + rz) - xi_3*(my*s + ry) + xi_4)**2)
        return Phi_a / norm


    # ========================================
    # Test of the functions
    # ========================================

    # test actuation matrix
    print("\nTesting actuation matrix... ------------------------")

    test_cases = [
        (
            1,
            {"L": jnp.array([0.5])},
            {"ry": jnp.array([0.1]), "rz": jnp.array([0.05]), "my": jnp.array([0.]), "mz": jnp.array([0.]), 'idx_seg_att': jnp.array([0])},
            A_special_case({"L": jnp.array([0.5])},
                           {"ry": jnp.array([0.1]), "rz": jnp.array([0.05]), "my": jnp.array([0.]), "mz": jnp.array([0.]), 'idx_seg_att': jnp.array([0])})
        ),
        (
            2,
            {"L": jnp.array([0.5, 0.6])},
            {"ry": jnp.array([0.15]), "rz": jnp.array([0.]), "my": jnp.array([0.01]), "mz": jnp.array([0.01]), 'idx_seg_att': jnp.array([1])},
            A_special_case({"L": jnp.array([0.5, 0.6])},
                           {"ry": jnp.array([0.15]), "rz": jnp.array([0.]), "my": jnp.array([0.01]), "mz": jnp.array([0.01]), 'idx_seg_att': jnp.array([1])})
        ),
        (
            3,
            {"L": jnp.array([0.5, 0.6, 0.7])},
            {"ry": jnp.array([0.1]), "rz": jnp.array([-0.05]), "my": jnp.array([0.01]), "mz": jnp.array([-0.05]), 'idx_seg_att': jnp.array([2])},
            A_special_case({"L": jnp.array([0.5, 0.6, 0.7])},
                           {"ry": jnp.array([0.1]), "rz": jnp.array([-0.05]), "my": jnp.array([0.01]), "mz": jnp.array([-0.05]), 'idx_seg_att': jnp.array([2])})
        ),
    ]

    for num_segments, spec_pars, tendon_params, expected in test_cases:
        robot, q0 = createRobot(num_segments, spec_pars, tendon_params)
        a = robot.actuation_matrix(q0)
        nq = a.shape[0] * a.shape[1]
        n1 = int(nq / 6)
        a = a.reshape((6, n1), order='F')
        a = jnp.sum(a, axis=1)
        assert not jnp.isnan(a).any(), "Actuation matrix contains NaN!"
        assert_allclose(a, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        print("[Valid test]\n")

    # test actuation basis
    print("\nTesting actuation basis... ------------------------")
    pars = {"ry": jnp.array([0.1]), "rz": jnp.array([-0.05]), "my": jnp.array([0.01]), "mz": jnp.array([-0.05]), 'idx_seg_att': jnp.array([0])}
    robot, _ = createRobot(1,
                           {"L": jnp.array([0.5])},
                           pars,
                          )
    xi_ref = jnp.array([0.,0.,0.,1.,0.,0.])
    
    test_cases = [
        (
            jnp.zeros_like(xi_ref),
            robot.L_cum[-1]*0.7,
            actuation_basis(pars, xi_ref, robot.L_cum[-1]*0.7)
        ),

        (
            jnp.array([0., jnp.pi*3, jnp.pi*2, 0.1, 0.2, 0.]),
            robot.L_cum[-1]*0.7,
            actuation_basis(pars, jnp.array([0., jnp.pi*3, jnp.pi*2, 0.1, 0.2, 0.]) + xi_ref, robot.L_cum[-1]*0.7)
        ),

        (
            jnp.zeros_like(xi_ref),
            robot.L_cum[-1]*0.25,
            actuation_basis(pars, xi_ref, robot.L_cum[-1]*0.25)
        ),

        (
            jnp.array([0., jnp.pi*3, jnp.pi*2, 0.1, 0.2, 0.]),
            robot.L_cum[-1]*0.25,
            actuation_basis(pars, jnp.array([0., jnp.pi*3, jnp.pi*2, 0.1, 0.2, 0.]) + xi_ref, robot.L_cum[-1]*0.25)
        ),


    ]

    for q, s, expected in test_cases:
        phi_a = robot._local_actuation_basis(q, s).squeeze()
        assert not jnp.isnan(phi_a).any(), "Actuation matrix contains NaN!"
        assert_allclose(phi_a, expected, rtol=Tolerance.rtol(), atol=Tolerance.atol())
        print("[Valid test]\n")



if __name__ == "__main__":
    print("Running tests for tendon actuated Planar Constant Strain...")
    test_actuation_matrix_pcs()
