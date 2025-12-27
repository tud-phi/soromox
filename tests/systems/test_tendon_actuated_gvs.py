import jax
import pytest

jax.config.update("jax_enable_x64", True)  # double precision
import numpy as onp
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import TendonActuatedGVS, TendonActuatedPCS
from soromox.systems.gvs import BasisAttributes, JointAttributes, LinkAttributes
from soromox.utils.tolerance import Tolerance


def test_actuation_matrix_gvs():
    """
    Test the methods responsible for the computation of the actuation matrix of the
    tendon actuated GVS class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test actuation matrix
    print("\nTesting actuation matrix... ------------------------")

    test_cases = [
        (
            # the tendon only affects the bending in z
            jnp.array([0.2]),
            {
                "ry": jnp.array([0.005]),
                "rz": jnp.array([0.0]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            jnp.array([0.2]),
            {
                "ry": jnp.array([0.002]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            # only first segment affected
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([-0.002]),
                "rz": jnp.array([-0.002]),
                "my": jnp.array([0.001]),
                "mz": jnp.array([-0.001]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
    ]

    for segment_lengths, tendon_params in test_cases:
        if segment_lengths.shape[0] == 1:
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )

            n_gauss_list = [10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1],
                joints_list=[joint1],
                basis_list=[basis1],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            joint2 = JointAttributes(jointtype="Fixed")

            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )
            basis2 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )

            n_gauss_list = [10, 10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1, link2],
                joints_list=[joint1, joint2],
                basis_list=[basis1, basis2],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.V_dof.reshape(-1))
        q0 = jnp.zeros((dof,))
        A = robotGVS.actuation_matrix(q0)
        print("Actuation matrix A:\n", A)
        if segment_lengths.shape[0] > 1:
            print("No contributions from the second segment.")
        elif tendon_params["rz"][0] == 0.0:
            print("No contributions along the y axis.")
        else:
            print("Contributions along both y and z axes.")
        assert not jnp.isnan(A).any(), "Actuation matrix contains NaN!"
        print("[Valid test]\n")


def test_tendon_length_gvs():
    """
    Test the methods responsible for the computation of the tendon length of the
    tendon actuated GVS class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test tendon length
    print("\nTesting tendon length... ------------------------")

    test_cases = [
        (
            # the tendon length must coincide with the hypotenuse of the triangle formed
            # by the segment length and 2*ry_distance
            jnp.array([0.2]),
            {
                "ry": jnp.array([0.005]),
                "rz": jnp.array([0.0]),
                "my": jnp.array([-0.05]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            # the tendon length must coincide with the sum of both segment lengths
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([0.0]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([1]),
            },
        ),
        (
            # the tendon length must coincide with the length of the first segment
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([0.0]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
    ]

    for segment_lengths, tendon_params in test_cases:
        if segment_lengths.shape[0] == 1:
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )

            n_gauss_list = [10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1],
                joints_list=[joint1],
                basis_list=[basis1],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            joint2 = JointAttributes(jointtype="Fixed")

            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )
            basis2 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )

            n_gauss_list = [10, 10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1, link2],
                joints_list=[joint1, joint2],
                basis_list=[basis1, basis2],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.V_dof.reshape(-1))
        q0 = jnp.zeros((dof,))
        l_tendons = robotGVS.tendon_length(q0)
        print("Length of the tendons:\n", l_tendons)
        if segment_lengths.shape[0] == 1:
            print(
                "The lenght coincide with the length of the hypotenuse of the triangle formed by the segment length and 2*ry_distance."
            )
            hypotenuse = onp.sqrt(
                (segment_lengths[0] ** 2) + ((2 * tendon_params["ry"][0]) ** 2)
            )
            assert_allclose(
                l_tendons,
                hypotenuse,
                rtol=Tolerance.rtol(),
                atol=Tolerance.atol(),
            )
        elif tendon_params["idx_seg_att"][0] == 1:
            print("The length coincide with the sum of the segments length.")
            assert_allclose(
                l_tendons,
                segment_lengths[0] + segment_lengths[1],
                rtol=Tolerance.rtol(),
                atol=Tolerance.atol(),
            )
        else:
            print("The length coincide with the length of the first segment.")
            assert_allclose(
                l_tendons,
                segment_lengths[0],
                rtol=Tolerance.rtol(),
                atol=Tolerance.atol(),
            )

        assert not jnp.isnan(l_tendons).any(), "Tendon lengths contains NaN!"

        print("[Valid test]\n")


def test_tendon_length_gradient_matches_actuation_matrix_random_configs():
    """Ensure tendon_length Jacobian equals actuation matrix transpose."""
    print(
        "\nTesting tendon_length Jacobian equals actuation matrix transpose... ------------------------"
    )

    test_cases = [
        (
            # the tendon length must coincide with the hypotenuse of the triangle formed
            # by the segment length and 2*ry_distance
            jnp.array([0.2]),
            {
                "ry": jnp.array([0.005]),
                "rz": jnp.array([0.0]),
                "my": jnp.array([-0.05]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            # the tendon length must coincide with the sum of both segment lengths
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([0.0]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([1]),
            },
        ),
        (
            # the tendon length must coincide with the length of the first segment
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([0.0]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
    ]
    for segment_lengths, tendon_params in test_cases:
        if segment_lengths.shape[0] == 1:
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )

            n_gauss_list = [10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1],
                joints_list=[joint1],
                basis_list=[basis1],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            joint2 = JointAttributes(jointtype="Fixed")

            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[1, 1, 1, 1, 0, 0]
            )
            basis2 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )

            n_gauss_list = [10, 10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1, link2],
                joints_list=[joint1, joint2],
                basis_list=[basis1, basis2],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.V_dof.reshape(-1))
        q0 = jnp.zeros((dof,))
        lengths = robotGVS.tendon_length(q0)
        assert lengths.shape == (robotGVS.num_actuators,)
        jac = jax.jacrev(robotGVS.tendon_length)(q0)
        A = robotGVS.actuation_matrix(q0)
        assert_allclose(
            jac,
            A.T,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )

        print("Actuation matrix A transpose:\n", A.T)
        print("tendon_length Jacobian:\n", jac)
        print("[Valid test]\n")


def test_tendon_actatuated_gvs_vs_pcs():
    """
    Compares the results of the tendon actuated GVS class with the tendon actuated
    Piecewise Constant Strain class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test actuation matrix GVS vs PCS
    print("\nTesting actuation matrix GVS vs PCS... ------------------------")

    test_cases = [
        (
            # the tendon only affects the bending in z
            jnp.array([0.2]),
            {
                "ry": jnp.array([0.005]),
                "rz": jnp.array([0.0]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
        (
            # the tendon only affects the bending in y
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([0.002]),
                "rz": jnp.array([0.002]),
                "my": jnp.array([0.0]),
                "mz": jnp.array([0.0]),
                "idx_seg_att": jnp.array([1]),
            },
        ),
        (
            # only first segment affected
            jnp.array([0.2, 0.05]),
            {
                "ry": jnp.array([-0.002]),
                "rz": jnp.array([-0.002]),
                "my": jnp.array([0.001]),
                "mz": jnp.array([-0.001]),
                "idx_seg_att": jnp.array([0]),
            },
        ),
    ]

    for segment_lengths, tendon_params in test_cases:
        if segment_lengths.shape[0] == 1:
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )

            n_gauss_list = [10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1],
                joints_list=[joint1],
                basis_list=[basis1],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )

        else:  # segment_lengths.shape[0] == 2
            link1 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkAttributes(
                section="Circular",
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointAttributes(jointtype="Fixed")
            joint2 = JointAttributes(jointtype="Fixed")

            basis1 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )
            basis2 = BasisAttributes(
                basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 0, 0, 0, 0, 0]
            )

            n_gauss_list = [10, 10]
            gravity_vector = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                links_list=[link1, link2],
                joints_list=[joint1, joint2],
                basis_list=[basis1, basis2],
                n_gauss_list=n_gauss_list,
                gravity_vector=gravity_vector,
                tendon_routing_params=tendon_params,
            )

        num_segments = int(segment_lengths.shape[0])
        E_val = 3e5
        nu = 0.45
        G_val = E_val / (2.0 * (1.0 + nu))
        params_pcs = {
            "p0": jnp.zeros((6,)),
            "L": jnp.asarray(segment_lengths),
            "r": jnp.full((num_segments,), 0.015),
            "rho": 1300.0 * jnp.ones((num_segments,)),
            "g": jnp.array([0.0, 0.0, 9.81]),
            "E": E_val * jnp.ones((num_segments,)),
            "G": G_val * jnp.ones((num_segments,)),
        }

        # params_pcs["D"] = 1e4 * jnp.diag(
        #    (jnp.repeat(jnp.array([[jnp.pi/2*(0.015 ** 4), 3*jnp.pi/4*(0.015 ** 4), 3*jnp.pi/4*(0.015 ** 4), 3*jnp.pi*(0.015 ** 2),jnp.pi*(0.015 ** 2),jnp.pi*(0.015 ** 2)]]), num_segments, axis=0) * params_pcs["L"][:, None]).flatten()
        # )
        params_pcs["D"] = 1e4 * jnp.diag(
            (
                jnp.repeat(
                    jnp.array(
                        [
                            [
                                jnp.pi / 2 * (0.015**4),
                                3 * jnp.pi / 4 * (0.015**4),
                                3 * jnp.pi / 4 * (0.015**4),
                                3 * jnp.pi * (0.015**2),
                                jnp.pi * (0.015**2),
                                jnp.pi * (0.015**2),
                            ]
                        ]
                    ),
                    num_segments,
                    axis=0,
                )
                * params_pcs["L"][:, None]
            ).flatten()
        )
        per_segment = jnp.array([1, 1, 1, 1, 0, 0], dtype=bool)
        strain_selector = jnp.tile(per_segment, num_segments)

        robotPCS = TendonActuatedPCS(
            num_segments=num_segments,
            params=params_pcs,
            active_tendon_routing_params=tendon_params,
            strain_selector=strain_selector,
        )

        dof = sum(robotGVS.V_dof.reshape(-1))
        q0 = jnp.zeros((dof,))
        A_GVS = robotGVS.actuation_matrix(q0)
        A_PCS = robotPCS.actuation_matrix(q0)

        print("GVS Actuation matrix A:\n", A_GVS)

        if segment_lengths.shape[0] == 1:
            A_PCS = A_PCS.at[:3, :].multiply(1.0 / segment_lengths[0])
            print("PCS Actuation matrix A (scaled for consistency):\n", A_PCS)
            print("No contributions along the y axis.")
        elif tendon_params["idx_seg_att"][0] == 1:
            A_PCS = A_PCS.at[:3, :].multiply(1.0 / segment_lengths[0])
            A_PCS = A_PCS.at[4:7, :].multiply(1.0 / segment_lengths[1])
            print("PCS Actuation matrix A (scaled for consistency):\n", A_PCS)

            print("Mixed contributions along y and z axes.")
        else:
            A_PCS = A_PCS.at[:3, :].multiply(1.0 / segment_lengths[0])
            A_PCS = A_PCS.at[4:7, :].multiply(1.0 / segment_lengths[1])
            print("PCS Actuation matrix A (scaled for consistency):\n", A_PCS)
            print("No contributions from the second segment.")
        assert not jnp.isnan(A_GVS).any(), "Actuation matrix contains NaN!"
        assert_allclose(
            A_GVS,
            A_PCS,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        print("[Valid test]\n")


if __name__ == "__main__":
    # run pytest with activated stdout
    # pytest.main([__file__])
    pytest.main(["-s", __file__])
