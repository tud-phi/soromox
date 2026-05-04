import jax
import pytest

jax.config.update("jax_enable_x64", True)  # double precision
import numpy as onp
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import CrossSectionGeometry, TendonActuatedGVS, TendonActuatedPCS
from soromox.systems.gvs import GVSSegment, JointSpec, LinkSpec, StrainBasisSpec
from soromox.utils.tolerance import Tolerance

import optimistix as optx


def _segments(
    links: list[LinkSpec],
    joints: list[JointSpec],
    bases: list[StrainBasisSpec],
    num_gauss_points: list[int],
) -> list[GVSSegment]:
    return [
        GVSSegment(link=link, joint=joint, basis=basis, num_gauss_points=n)
        for link, joint, basis, n in zip(links, joints, bases, num_gauss_points)
    ]


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
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                g=g,
                active_tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            joint2 = JointSpec(type="fixed")

            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )
            basis2 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10, 10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                g=g,
                active_tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.dofs_per_segment.reshape(-1))
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
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                g=g,
                active_tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            joint2 = JointSpec(type="fixed")

            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )
            basis2 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10, 10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                g=g,
                active_tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.dofs_per_segment.reshape(-1))
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
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                g=g,
                active_tendon_routing_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            joint2 = JointSpec(type="fixed")

            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )
            basis2 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10, 10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                g=g,
                active_tendon_routing_params=tendon_params,
            )

        dof = sum(robotGVS.dofs_per_segment.reshape(-1))
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


def test_tendon_actatuated_ActMatrix_gvs_vs_pcs():
    """
    Compares the actuation matrices of the tendon actuated GVS class with the tendon actuated
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
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                g=g,
                active_tendon_routing_params=tendon_params,
                scale_rotational_basis_by_length=False,
            )

        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[0],
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=segment_lengths[1],
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            joint2 = JointSpec(type="fixed")

            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )
            basis2 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10, 10]
            g = [0.0, 0.0, -9.81]

            robotGVS = TendonActuatedGVS(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                g=g,
                active_tendon_routing_params=tendon_params,
                scale_rotational_basis_by_length=False,
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

        dof = sum(robotGVS.dofs_per_segment.reshape(-1))
        q0 = jnp.zeros((dof,))
        A_GVS = robotGVS.actuation_matrix(q0)
        A_PCS = robotPCS.actuation_matrix(q0)

        print("GVS Actuation matrix A:\n", A_GVS)

        if segment_lengths.shape[0] == 1:
            print("PCS Actuation matrix A (scaled for consistency):\n", A_PCS)
            print("No contributions along the y axis.")
        elif tendon_params["idx_seg_att"][0] == 1:
            print("PCS Actuation matrix A (scaled for consistency):\n", A_PCS)

            print("Mixed contributions along y and z axes.")
        else:
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

def test_tendon_actatuated_gvs_vs_pcs():
    """
    Compares the results of the tendon actuated GVS class with the tendon actuated
    Piecewise Constant Strain class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test forward kinematics GVS vs PCS
    print("\nTesting Forward Kinematics GVS vs PCS... ------------------------")

    link1 = LinkSpec(
        cross_section_geometry=CrossSectionGeometry.CIRCULAR,
        E=3e5,
        nu=0.45,
        rho=1300.0,
        eta=1e4,
        L=0.2,
        r_i=0.015,
        r_f=0.015,
    )
    joint1 = JointSpec(type="fixed")
    basis1 = StrainBasisSpec(
        type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
    )

    num_gauss_points = [10]
    g = [0.0, 0.0, -9.81]
    tendon_params = {
        "ry": jnp.array([-0.002]),
        "rz": jnp.array([-0.002]),
        "my": jnp.array([0.001]),
        "mz": jnp.array([-0.001]),
        "idx_seg_att": jnp.array([0]),
    }

    robotGVS = TendonActuatedGVS(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        g=g,
        active_tendon_routing_params=tendon_params,
        p0=jnp.zeros((6,)),
        scale_rotational_basis_by_length=False,
    )

    segment_lengths = jnp.array([0.2])
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

    dof = sum(robotGVS.dofs_per_segment.reshape(-1))
    q0 = jnp.zeros((dof,))

    s_end_GVS = robotGVS.segment_end_positions[-1]
    g_in_L_GVS = robotGVS.forward_kinematics(q0, s_end_GVS)
    p_in_L_GVS = g_in_L_GVS[:3, 3]
    s_end_PCS = robotPCS.L
    g_in_L_PCS = robotPCS.forward_kinematics(q0, s_end_PCS)
    p_in_L_PCS = g_in_L_PCS[:3, 3]
    print("GVS End-effector initial position:\n", p_in_L_GVS)
    print("PCS End-effector initial position:\n", p_in_L_PCS)
    assert_allclose(
        p_in_L_GVS, p_in_L_PCS, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )

    u = jnp.asarray([-1], dtype=q0.dtype)

    def solve_equilibrium_GVS(
        robot: TendonActuatedGVS, u: jnp.ndarray, q0: jnp.ndarray
    ):
        def statics_eq(q, args):
            u = args
            K = robot.stiffness_matrix()
            B = robot.actuation_matrix(q)
            G = robot.gravitational_force(q)
            return K @ q + G - B @ u

        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        statics_eq_jit = jax.jit(statics_eq)
        return optx.root_find(statics_eq_jit, solver, q0, (u), max_steps=200)

    res_GVS = solve_equilibrium_GVS(robotGVS, u, q0)
    q_GVS = res_GVS.value  # equilibrium generalized coordinates (GVS)

    def solve_equilibrium_PCS(
        robot: TendonActuatedPCS, u: jnp.ndarray, q0: jnp.ndarray
    ):
        def statics_eq(q, args):
            u = args
            K = robot.stiffness_matrix()
            B = robot.actuation_matrix(q)
            G = robot.gravitational_force(q)
            return K @ q - G - B @ u

        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        statics_eq_jit = jax.jit(statics_eq)
        return optx.root_find(statics_eq_jit, solver, q0, (u), max_steps=200)

    res_PCS = solve_equilibrium_PCS(robotPCS, u, q0)
    q_PCS = res_PCS.value  # equilibrium generalized coordinates (PCS)

    g_end_L_GVS = robotGVS.forward_kinematics(q_GVS, s_end_GVS)
    p_end_L_GVS = g_end_L_GVS[:3, 3]
    g_end_L_PCS = robotPCS.forward_kinematics(q_PCS, s_end_PCS)
    p_end_L_PCS = g_end_L_PCS[:3, 3]
    print("GVS End-effector final position:\n", p_end_L_GVS)
    print("PCS End-effector final position:\n", p_end_L_PCS)
    assert_allclose(
        p_end_L_GVS, p_end_L_PCS, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )

    print("[Valid test]\n")


def test_angular_strain_basis_scaling_gvs():
    """
    Test the scaling procedure for the angular component of the strain basis in the
    (Tendon actuated) GVS class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test tendon length
    print(
        "\nTesting angular strain basis scaling procedure... ------------------------"
    )

    link1 = LinkSpec(
        cross_section_geometry=CrossSectionGeometry.CIRCULAR,
        E=3e5,
        nu=0.45,
        rho=1300.0,
        eta=1e4,
        L=0.2,
        r_i=0.015,
        r_f=0.015,
    )
    joint1 = JointSpec(type="fixed")
    basis1 = StrainBasisSpec(
        type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
    )

    num_gauss_points = [10]
    g = [0.0, 0.0, -9.81]
    tendon_params = {
        "ry": jnp.array([-0.002]),
        "rz": jnp.array([-0.002]),
        "my": jnp.array([0.001]),
        "mz": jnp.array([-0.001]),
        "idx_seg_att": jnp.array([0]),
    }
    robot_noScale = TendonActuatedGVS(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        g=g,
        active_tendon_routing_params=tendon_params,
        scale_rotational_basis_by_length=False,
    )
    robot_Scale = TendonActuatedGVS(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        g=g,
        active_tendon_routing_params=tendon_params,
        scale_rotational_basis_by_length=True,
    )

    # both robots have the same characteristics except for the scaling of the angular strain basis
    dof = sum(robot_noScale.dofs_per_segment.reshape(-1))
    L_cum = jax.device_get(robot_noScale.segment_end_positions)
    total_length = float(L_cum[-1])

    q0 = jnp.zeros((dof,))
    u = jnp.asarray([-1], dtype=q0.dtype)
    s_end = float(total_length)

    def solve_equilibrium(robot: TendonActuatedGVS, u: jnp.ndarray, q0: jnp.ndarray):
        def statics_eq(q, args):
            u = args
            K = robot.stiffness_matrix()
            B = robot.actuation_matrix(q)
            G = robot.gravitational_force(q)
            return K @ q + G - B @ u

        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        statics_eq_jit = jax.jit(statics_eq)
        return optx.root_find(statics_eq_jit, solver, q0, (u), max_steps=200)

    res_stat_noScale = solve_equilibrium(robot_noScale, u, q0)
    q_stat_noScale = (
        res_stat_noScale.value
    )  # equilibrium generalized coordinates (no Scaling)
    res_stat_Scale = solve_equilibrium(robot_Scale, u, q0)
    q_stat_Scale = (
        res_stat_Scale.value
    )  # equilibrium generalized coordinates (with Scaling)
    g_end_noScale = robot_noScale.forward_kinematics(q_stat_noScale, s_end)
    g_end_Scale = robot_Scale.forward_kinematics(q_stat_Scale, s_end)
    p_end_noScale = g_end_noScale[:3, 3]
    p_end_Scale = g_end_Scale[:3, 3]

    print("End-effector position without scaling:\n", p_end_noScale)
    print("End-effector position with scaling:\n", p_end_Scale)
    assert_allclose(
        p_end_noScale, p_end_Scale, rtol=Tolerance.rtol(), atol=Tolerance.atol()
    )
    print("[Valid test]\n")


if __name__ == "__main__":
    # run pytest with activated stdout
    # pytest.main([__file__])
    pytest.main(["-s", __file__])
