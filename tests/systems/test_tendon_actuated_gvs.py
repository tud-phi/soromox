import jax
import pytest

jax.config.update("jax_enable_x64", True)  # double precision
import numpy as onp
from jax import numpy as jnp
from numpy.testing import assert_allclose

from soromox.systems import (
    CrossSectionGeometry,
    PCSStructure,
    TendonActuatedGVS,
    TendonActuatedPCS,
)
from soromox.systems.gvs import GVSSegment, JointSpec, LinkSpec, StrainBasisSpec
from soromox.utils.tolerance import Tolerance
from system_param_builders import (
    linear_tendon_routing,
    passive_tendon_params,
    pcs_params,
    tendon_actuated_pcs_params,
)

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


def _make_tendon_gvs(
    *,
    segments: list[GVSSegment],
    gravity,
    tendon_params,
    passive_tendon_routing=None,
    passive_tendon=None,
    base_pose=None,
    max_dof=None,
    scale_rotational_basis_by_length: bool = False,
) -> TendonActuatedGVS:
    return TendonActuatedGVS.from_segments(
        segments,
        gravity=jnp.asarray(gravity),
        active_tendon_routing=tendon_params,
        passive_tendon_routing=passive_tendon_routing,
        passive_tendon=passive_tendon,
        base_pose=base_pose,
        max_dof=max_dof,
        scale_rotational_basis_by_length=scale_rotational_basis_by_length,
    )


def _make_tendon_pcs(
    *,
    length,
    gravity,
    tendon_params,
    strain_selector,
    base_pose=None,
) -> TendonActuatedPCS:
    segment_lengths = jnp.asarray(length)
    num_segments = int(segment_lengths.shape[0])
    E_val = 3e5
    nu = 0.45
    G_val = E_val / (2.0 * (1.0 + nu))
    damping_matrix = 1e4 * jnp.diag(
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
            * segment_lengths[:, None]
        ).flatten()
    )
    body = pcs_params(
        base_pose=jnp.zeros((6,)) if base_pose is None else base_pose,
        length=segment_lengths,
        radius=jnp.full((num_segments,), 0.015),
        density=1300.0 * jnp.ones((num_segments,)),
        gravity=jnp.asarray(gravity),
        young_modulus=E_val * jnp.ones((num_segments,)),
        shear_modulus=G_val * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
    )
    return TendonActuatedPCS(
        params=tendon_actuated_pcs_params(
            body=body,
            active_tendon_routing=tendon_params,
        ),
        structure=PCSStructure(strain_selector=strain_selector),
    )


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
            linear_tendon_routing(
                y_intercept=jnp.array([0.005]),
                z_intercept=jnp.array([0.0]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            jnp.array([0.2]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.002]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            # only first segment affected
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([-0.002]),
                z_intercept=jnp.array([-0.002]),
                y_slope=jnp.array([0.001]),
                z_slope=jnp.array([-0.001]),
                attachment_segment_index=jnp.array([0]),
            ),
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
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = _make_tendon_gvs(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                gravity=g,
                tendon_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[1]),
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

            robotGVS = _make_tendon_gvs(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                gravity=g,
                tendon_params=tendon_params,
            )

        dof = sum(robotGVS.dofs_per_segment.reshape(-1))
        q0 = jnp.zeros((dof,))
        A = robotGVS.actuation_matrix(q0)
        print("Actuation matrix A:\n", A)
        if segment_lengths.shape[0] > 1:
            print("No contributions from the second segment.")
        elif tendon_params.z_intercept[0] == 0.0:
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
            linear_tendon_routing(
                y_intercept=jnp.array([0.005]),
                z_intercept=jnp.array([0.0]),
                y_slope=jnp.array([-0.05]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            # the tendon length must coincide with the sum of both segment lengths
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.0]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([1]),
            ),
        ),
        (
            # the tendon length must coincide with the length of the first segment
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.0]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
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
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = _make_tendon_gvs(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                gravity=g,
                tendon_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[1]),
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

            robotGVS = _make_tendon_gvs(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                gravity=g,
                tendon_params=tendon_params,
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
                (segment_lengths[0] ** 2) + ((2 * tendon_params.y_intercept[0]) ** 2)
            )
            assert_allclose(
                l_tendons,
                hypotenuse,
                rtol=Tolerance.rtol(),
                atol=Tolerance.atol(),
            )
        elif tendon_params.attachment_segment_index[0] == 1:
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
            linear_tendon_routing(
                y_intercept=jnp.array([0.005]),
                z_intercept=jnp.array([0.0]),
                y_slope=jnp.array([-0.05]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            # the tendon length must coincide with the sum of both segment lengths
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.0]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([1]),
            ),
        ),
        (
            # the tendon length must coincide with the length of the first segment
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.0]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
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
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = _make_tendon_gvs(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                gravity=g,
                tendon_params=tendon_params,
            )
        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[1]),
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

            robotGVS = _make_tendon_gvs(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                gravity=g,
                tendon_params=tendon_params,
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
            linear_tendon_routing(
                y_intercept=jnp.array([0.005]),
                z_intercept=jnp.array([0.0]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            # the tendon only affects the bending in y
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.002]),
                z_intercept=jnp.array([0.002]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([1]),
            ),
        ),
        (
            # only first segment affected
            jnp.array([0.2, 0.05]),
            linear_tendon_routing(
                y_intercept=jnp.array([-0.002]),
                z_intercept=jnp.array([-0.002]),
                y_slope=jnp.array([0.001]),
                z_slope=jnp.array([-0.001]),
                attachment_segment_index=jnp.array([0]),
            ),
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
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            joint1 = JointSpec(type="fixed")
            basis1 = StrainBasisSpec(
                type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[0, 0, 0, 0, 0, 0]
            )

            num_gauss_points = [10]
            g = [0.0, 0.0, -9.81]

            robotGVS = _make_tendon_gvs(
                segments=_segments([link1], [joint1], [basis1], num_gauss_points),
                gravity=g,
                tendon_params=tendon_params,
                scale_rotational_basis_by_length=False,
            )

        else:  # segment_lengths.shape[0] == 2
            link1 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[0]),
                r_i=0.015,
                r_f=0.015,
            )
            link2 = LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=3e5,
                nu=0.45,
                rho=1300.0,
                eta=1e4,
                L=float(segment_lengths[1]),
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

            robotGVS = _make_tendon_gvs(
                segments=_segments(
                    [link1, link2], [joint1, joint2], [basis1, basis2], num_gauss_points
                ),
                gravity=g,
                tendon_params=tendon_params,
                scale_rotational_basis_by_length=False,
            )

        num_segments = int(segment_lengths.shape[0])
        per_segment = jnp.array([1, 1, 1, 1, 0, 0], dtype=bool)
        strain_selector = jnp.tile(per_segment, num_segments)

        robotPCS = _make_tendon_pcs(
            length=segment_lengths,
            gravity=jnp.array([0.0, 0.0, 9.81]),
            tendon_params=tendon_params,
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
        elif tendon_params.attachment_segment_index[0] == 1:
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
    tendon_params = linear_tendon_routing(
        y_intercept=jnp.array([-0.002]),
        z_intercept=jnp.array([-0.002]),
        y_slope=jnp.array([0.001]),
        z_slope=jnp.array([-0.001]),
        attachment_segment_index=jnp.array([0]),
    )

    robotGVS = _make_tendon_gvs(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        gravity=g,
        tendon_params=tendon_params,
        base_pose=jnp.zeros((6,)),
        scale_rotational_basis_by_length=False,
    )

    segment_lengths = jnp.array([0.2])
    num_segments = int(segment_lengths.shape[0])
    per_segment = jnp.array([1, 1, 1, 1, 0, 0], dtype=bool)
    strain_selector = jnp.tile(per_segment, num_segments)

    robotPCS = _make_tendon_pcs(
        length=segment_lengths,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        tendon_params=tendon_params,
        strain_selector=strain_selector,
        base_pose=jnp.zeros((6,)),
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
    tendon_params = linear_tendon_routing(
        y_intercept=jnp.array([-0.002]),
        z_intercept=jnp.array([-0.002]),
        y_slope=jnp.array([0.001]),
        z_slope=jnp.array([-0.001]),
        attachment_segment_index=jnp.array([0]),
    )
    robot_noScale = _make_tendon_gvs(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        gravity=g,
        tendon_params=tendon_params,
        scale_rotational_basis_by_length=False,
    )
    robot_Scale = _make_tendon_gvs(
        segments=_segments([link1], [joint1], [basis1], num_gauss_points),
        gravity=g,
        tendon_params=tendon_params,
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


def test_passive_tendon_params_are_batched_for_gvs():
    link = LinkSpec(
        cross_section_geometry=CrossSectionGeometry.CIRCULAR,
        E=3e5,
        nu=0.45,
        rho=1300.0,
        eta=1e4,
        L=0.2,
        r_i=0.015,
        r_f=0.015,
    )
    joint = JointSpec(type="fixed")
    basis = StrainBasisSpec(
        type="monomial", active=[1, 1, 1, 1, 0, 0], orders=[1, 1, 1, 1, 0, 0]
    )
    active_routing = linear_tendon_routing(
        y_intercept=jnp.array([0.004, -0.004]),
        z_intercept=jnp.array([0.003, 0.003]),
        y_slope=jnp.array([0.0, 0.001]),
        z_slope=jnp.array([0.0, -0.001]),
        attachment_segment_index=jnp.array([0, 0]),
    )
    passive_routing = linear_tendon_routing(
        y_intercept=jnp.array([0.002, -0.002]),
        z_intercept=jnp.array([-0.004, -0.004]),
        y_slope=jnp.array([0.001, 0.0]),
        z_slope=jnp.array([0.0, 0.001]),
        attachment_segment_index=jnp.array([0, 0]),
    )
    passive_tendon = passive_tendon_params(
        stiffness=jnp.array([10.0, 25.0]),
        damping=jnp.array([0.1, 0.3]),
        rest_length_offset=jnp.array([0.0, 0.01]),
    )
    robot = _make_tendon_gvs(
        segments=_segments([link], [joint], [basis], [8]),
        gravity=[0.0, 0.0, -9.81],
        tendon_params=active_routing,
        passive_tendon_routing=passive_routing,
        passive_tendon=passive_tendon,
    )
    q = jnp.zeros((robot.num_dofs,))

    assert robot.num_actuators == 2
    assert robot.n_p == 2
    assert robot.actuation_matrix(q).shape == (robot.num_dofs, 2)
    assert robot.jacobian_passive_tendon(q).shape == (2, robot.num_dofs)
    assert robot.forward_kinematics_active_tendons(q, 0.1).shape == (2, 3)
    assert robot.forward_kinematics_passive_tendons(q, 0.1).shape == (2, 3)
    assert robot.forward_kinematics_tendons(q, 0.1).shape == (4, 3)
    assert jnp.all(jnp.isfinite(robot.passive_tendon_length(q)))
    assert jnp.all(jnp.isfinite(robot.elastic_force(q)))
    assert jnp.all(jnp.isfinite(robot.damping_matrix(q)))
    assert jnp.isfinite(robot.elastic_energy(q))

    updated_passive = passive_tendon.replace(damping=passive_tendon.damping + 1.0)
    updated_robot = robot.update_params(passive_tendon=updated_passive)
    assert_allclose(jnp.diag(robot.D_pt), jnp.array([0.1, 0.3]))
    assert_allclose(jnp.diag(updated_robot.D_pt), jnp.array([1.1, 1.3]))

    with pytest.raises(ValueError, match="passive_tendon"):
        robot.update_params(
            passive_tendon_routing=linear_tendon_routing(
                y_intercept=jnp.array([0.002]),
                z_intercept=jnp.array([-0.004]),
                y_slope=jnp.array([0.001]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            )
        )


if __name__ == "__main__":
    # run pytest with activated stdout
    # pytest.main([__file__])
    pytest.main(["-s", __file__])
