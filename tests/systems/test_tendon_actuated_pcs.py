# ruff: noqa: E402
import jax
import pytest

jax.config.update("jax_enable_x64", True)  # double precision

from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    linear_tendon_routing,
    passive_tendon_params,
    pcs_params,
    spatial_base_pose,
    tendon_actuated_pcs_params,
)

from soromox.systems import (
    LinearTendonRoutingParams,
    PassiveTendonParams,
    PCSStructure,
    TendonActuatedPCS,
)
from soromox.utils.tolerance import Tolerance

XI_REF = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)


def _create_robot(
    segment_lengths: Array,
    tendon_params: LinearTendonRoutingParams,
    strain_selector=None,
    passive_tendon_routing: LinearTendonRoutingParams | None = None,
    passive_tendon: PassiveTendonParams | None = None,
) -> TendonActuatedPCS:
    """
    Helper that builds a TendonActuatedPCS instance with consistent material parameters.
    """
    num_segments = segment_lengths.shape[0]
    rho = 1070 * jnp.ones(
        (num_segments,)
    )  # Volumetric density of Dragon Skin 20 [kg/m^3]
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    body = pcs_params(
        base_pose=spatial_base_pose(),  # Initial position and orientation
        length=segment_lengths,
        radius=1.0 * jnp.ones((num_segments,)),  # default: 2e-2
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        young_modulus=2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        shear_modulus=1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
        damping_matrix=damping_matrix,
    )
    params = tendon_actuated_pcs_params(
        body=body,
        active_tendon_routing=tendon_params,
        passive_tendon_routing=passive_tendon_routing,
        passive_tendon=passive_tendon,
    )

    return TendonActuatedPCS(
        params=params,
        structure=PCSStructure(num_gauss_points=5, strain_selector=strain_selector),
    )


def _stacked_tendon_params(
    num_segments: int, num_tendons: int
) -> LinearTendonRoutingParams:
    """
    Build a set of tendon routing parameters covering multiple attachments and slopes.
    """
    if num_tendons < 1:
        raise ValueError("num_tendons must be at least 1.")
    idx = jnp.arange(num_tendons, dtype=jnp.int32)
    attachment = jnp.minimum(idx % num_segments, num_segments - 1)
    base = idx.astype(jnp.float64) + 1.0
    alternating = jnp.where(idx % 2 == 0, 1.0, -1.0)

    ry = 0.02 * base * alternating
    rz = -0.015 * (base + 0.5)
    my = 0.01 * (jnp.mod(idx, 3).astype(jnp.float64) - 1.0)
    mz = 0.012 * (jnp.mod(idx + 1, 3).astype(jnp.float64) - 1.0)

    return linear_tendon_routing(
        y_intercept=ry,
        z_intercept=rz,
        y_slope=my,
        z_slope=mz,
        attachment_segment_index=attachment,
    )


def reference_actuation_matrix(
    tendon_params: LinearTendonRoutingParams, l_tot: Array
) -> Array:
    """
    Compute the actuation matrix of a soft robot of length l_tot w.r.t. one
    linear tendon at the straight configuration corresponding to vector state
    q0 = zeros(6*num_segments), considered as reference.
    This configuration represents the rest (stress-free) shape of the robot
    and it is easily tractable analitically.

    Args:
        tendon_params (Dict[str, Array]): routing parameters of the given tendon
        l_tot (Array): total length of the robot

    Returns:
        A (Array): actuation matrix at straight configuration
    """
    ry, rz, my, mz = (
        tendon_params.y_intercept[0],
        tendon_params.z_intercept[0],
        tendon_params.y_slope[0],
        tendon_params.z_slope[0],
    )
    A = jnp.array(
        [
            l_tot * (-my * rz + mz * ry),
            l_tot**2 * mz / 2 + l_tot * rz,
            -(l_tot**2) * my / 2 - l_tot * ry,
            l_tot,
            l_tot * my,
            l_tot * mz,
        ]
    )

    return A / jnp.sqrt(my**2 + mz**2 + 1)


def reference_actuation_basis(
    tendon_params: LinearTendonRoutingParams, q: Array, s: Array
) -> Array:
    """
    Computes the vector representing the actuation basis of the given tendon
    at configuration q (assuming the strain basis to be the identity) and at
    abscissa point s for a single CS segment.

    Args:
        tendon_params (Dict[str, Array]): routing parameters of the given tendon
        q (Array): strains of the robot made by one single segment (6,)
        s (Array): abscissa point ()

    Returns:
        Phi_a (Array): actuation basis of the tendon at q, s (6,)
    """
    xi = jnp.eye(q.shape[0]) @ q + XI_REF
    ry, rz, my, mz = (
        tendon_params.y_intercept[0],
        tendon_params.z_intercept[0],
        tendon_params.y_slope[0],
        tendon_params.z_slope[0],
    )
    xi_1, xi_2, xi_3, xi_4, xi_5, xi_6 = xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]
    Phi_a = jnp.array(
        [
            (my * s + ry) * (mz + xi_1 * (my * s + ry) + xi_6)
            + (-mz * s - rz) * (my - xi_1 * (mz * s + rz) + xi_5),
            (mz * s + rz) * (xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4),
            (-my * s - ry) * (xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4),
            xi_2 * (mz * s + rz) - xi_3 * (my * s + ry) + xi_4,
            my - xi_1 * (mz * s + rz) + xi_5,
            mz + xi_1 * (my * s + ry) + xi_6,
        ]
    )
    norm = jnp.linalg.norm(Phi_a[3:])
    Phi_a = Phi_a / norm

    return Phi_a


def test_actuation_matrix_pcs():
    """
    Test the methods responsible for the computation of the actuation matrix of the
    tendon actuated Piecewise Constant Strain class.
    """

    # ========================================
    # Test of the functions
    # ========================================

    # test actuation matrix
    print("\nTesting actuation matrix... ------------------------")

    test_cases = [
        (
            jnp.array([0.5]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.1]),
                z_intercept=jnp.array([0.05]),
                y_slope=jnp.array([0.0]),
                z_slope=jnp.array([0.0]),
                attachment_segment_index=jnp.array([0]),
            ),
        ),
        (
            jnp.array([0.5, 0.6]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.15]),
                z_intercept=jnp.array([0.0]),
                y_slope=jnp.array([0.01]),
                z_slope=jnp.array([0.01]),
                attachment_segment_index=jnp.array([1]),
            ),
        ),
        (
            jnp.array([0.5, 0.6, 0.7]),
            linear_tendon_routing(
                y_intercept=jnp.array([0.1]),
                z_intercept=jnp.array([-0.05]),
                y_slope=jnp.array([0.01]),
                z_slope=jnp.array([-0.05]),
                attachment_segment_index=jnp.array([2]),
            ),
        ),
    ]

    for segment_lengths, tendon_params in test_cases:
        robot = _create_robot(segment_lengths, tendon_params)
        num_segments = segment_lengths.shape[0]
        q0 = jnp.zeros(6 * num_segments)
        target_actuation_matrix = robot.actuation_matrix(q0).reshape(
            (6, num_segments), order="F"
        )
        target_actuation_matrix = jnp.sum(target_actuation_matrix, axis=1)
        expected_actuation_matrix = reference_actuation_matrix(
            tendon_params, jnp.sum(segment_lengths)
        )
        assert not jnp.isnan(target_actuation_matrix).any(), (
            "Actuation matrix contains NaN!"
        )
        assert_allclose(
            target_actuation_matrix,
            expected_actuation_matrix,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        print("[Valid test]\n")

    # test actuation basis
    print("\nTesting actuation basis... ------------------------")
    tendon_params = linear_tendon_routing(
        y_intercept=jnp.array([0.1]),
        z_intercept=jnp.array([-0.05]),
        y_slope=jnp.array([0.01]),
        z_slope=jnp.array([-0.05]),
        attachment_segment_index=jnp.array([0]),
    )
    robot = _create_robot(jnp.array([0.5]), tendon_params)

    test_cases = [
        (
            jnp.zeros_like(XI_REF),
            robot.L_cum[-1] * 0.25,
        ),
        (
            jnp.array([0.0, jnp.pi * 3, jnp.pi * 2, 0.1, 0.2, 0.0]),
            robot.L_cum[-1] * 0.25,
        ),
        (
            jnp.zeros_like(XI_REF),
            robot.L_cum[-1] * 0.7,
        ),
        (
            jnp.array([0.0, jnp.pi * 3, jnp.pi * 2, 0.1, 0.2, 0.0]),
            robot.L_cum[-1] * 0.7,
        ),
    ]

    for q, s in test_cases:
        xi = robot.strain(q).reshape((robot.num_segments, 6))
        tendon_params_0 = tendon_params.routing_for_tendon(0)
        target_actuation_basis = robot._local_actuation_basis(
            0,
            xi[0],
            s,
            tendon_routing_params_k=tendon_params_0,
            d_s_fn=robot.active_d_s,
            dd_s_ds_fn=robot.active_dd_s_ds,
        ).squeeze()
        expected_actuation_basis = reference_actuation_basis(tendon_params, q, s)
        assert not jnp.isnan(target_actuation_basis).any(), (
            "Actuation basis contains NaN!"
        )
        assert_allclose(
            target_actuation_basis,
            expected_actuation_basis,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )
        print("[Valid test]\n")


def test_actuation_matrix_with_inactive_strains():
    """
    Ensure the actuation matrix projection works when some strains are deactivated.
    """

    segment_lengths = jnp.array([0.4, 0.6])
    tendon_params = linear_tendon_routing(
        y_intercept=jnp.array([0.12]),
        z_intercept=jnp.array([0.02]),
        y_slope=jnp.array([0.02]),
        z_slope=jnp.array([-0.03]),
        attachment_segment_index=jnp.array([1]),
    )
    strain_selector = jnp.array(
        [
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    robot_full = _create_robot(segment_lengths, tendon_params)
    robot_reduced = _create_robot(
        segment_lengths, tendon_params, strain_selector=strain_selector
    )

    num_full = int(robot_full.num_active_strains.item())
    num_active = int(robot_reduced.num_active_strains.item())
    assert num_active < num_full  # sanity check: some strains removed

    active_idx = jnp.nonzero(strain_selector, size=num_active, fill_value=0)[0]
    q_reduced = jnp.arange(num_active, dtype=jnp.float64) * 0.05
    q_full = jnp.zeros((num_full,), dtype=jnp.float64).at[active_idx].set(q_reduced)

    A_full = robot_full.actuation_matrix(q_full)
    A_reduced = robot_reduced.actuation_matrix(q_reduced)
    expected = A_full[active_idx, :]

    assert_allclose(
        A_reduced,
        expected,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_constant_curvature_segment_actuation_matches_closed_form_expression_from_pustina2024():
    """
    Validate Equation (15) and the following y expression from
    Pustina et al. (2024) for a single-segment constant-curvature rod.

    For a single segment with only bending (kappa_y, kappa_z) and axial strain
    active, Eq. (15) predicts a constant actuation matrix whose entries depend
    only on the tendon radius d and the segment length L. The subsequent
    expression for y states that tendon lengths equal the rest length plus
    A(q)^T q. We check both statements here.

    The PCC coordinates (kappa_i, phi_i, delta_L_i) used in the paper relate
    to SoRoMoX strains (kappa_y, kappa_z, sigma_x) as
        kappa_i = sqrt(kappa_y^2 + kappa_z^2),
        phi_i   = atan2(kappa_y, -kappa_z)   (bending angle measured from +z),
        delta_L_i = (sigma_x - 1) * L,
    where L is the segment length (also the tendon rest length in this test).
    """

    segment_lengths = jnp.array([1.0], dtype=jnp.float64)
    d = 1.0e-2  # tendon distance/radius from centerline
    tendon_angles = jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3], dtype=jnp.float64)
    tendon_params = linear_tendon_routing(
        y_intercept=d * jnp.cos(tendon_angles),
        z_intercept=d * jnp.sin(tendon_angles),
        y_slope=jnp.zeros_like(tendon_angles),
        z_slope=jnp.zeros_like(tendon_angles),
        attachment_segment_index=jnp.zeros((3,), dtype=jnp.int32),
    )
    # activate bending about y and z plus axial elongation
    strain_selector = jnp.array([False, True, True, True, False, False], dtype=bool)

    robot = _create_robot(
        segment_lengths, tendon_params, strain_selector=strain_selector
    )
    q = jnp.array([0.12, -0.07, 0.03], dtype=jnp.float64)

    A_numeric = robot.actuation_matrix(q)

    segment_length = segment_lengths[0]
    sqrt_three = jnp.sqrt(3.0)
    expected_A = jnp.array(
        [
            [
                0.0,
                (sqrt_three * d * segment_length) / 2.0,
                -(sqrt_three * d * segment_length) / 2.0,
            ],
            [
                -d * segment_length,
                0.5 * d * segment_length,
                0.5 * d * segment_length,
            ],
            [
                segment_length,
                segment_length,
                segment_length,
            ],
        ],
        dtype=jnp.float64,
    )

    assert_allclose(
        A_numeric,
        expected_A,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    tendon_lengths = robot.tendon_length(q)

    # Map SoRoMoX strains (kappa_y, kappa_z, sigma_x) to the PCC coordinates
    # used in Pustina et al. (2024), Example 5.
    kappa_y, kappa_z = q[0], q[1]
    sigma_x = q[2] + 1.0  # xi_ref adds 1.0 to axial strain
    kappa_mag = jnp.sqrt(kappa_y**2 + kappa_z**2)
    # Paper’s bending direction is measured from +z, so swapping arctan2
    # arguments guarantees cos(q2) = -kappa_z / kappa_mag and sin(q2) =
    # kappa_y / kappa_mag as in Eq. (15).
    phi = jnp.arctan2(kappa_y, -kappa_z)
    delta_L = (sigma_x - 1.0) * segment_length
    # q1 stores kappa_i * L to match the notation in the paper.
    q1, q2, q3 = segment_length * kappa_mag, phi, delta_L
    q4 = q5 = q6 = 0.0  # Only one segment is active in this scenario.
    cos_q2, sin_q2 = jnp.cos(q2), jnp.sin(q2)
    cos_q5, sin_q5 = jnp.cos(q5), jnp.sin(q5)

    common_cos = q1 * cos_q2 + q4 * cos_q5
    common_sin = q1 * sin_q2 + q4 * sin_q5
    y_expr = jnp.array(
        [
            q3 + q6 + d * common_cos,
            q3 + q6 - 0.5 * d * common_cos + (jnp.sqrt(3.0) / 2.0) * d * common_sin,
            q3 + q6 - 0.5 * d * common_cos - (jnp.sqrt(3.0) / 2.0) * d * common_sin,
        ],
        dtype=jnp.float64,
    )
    expected_lengths = segment_length + y_expr

    assert_allclose(
        tendon_lengths,
        expected_lengths,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize("num_tendons", [1, 2, 4])
@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_tendon_length_gradient_matches_actuation_matrix_random_configs(
    num_segments: int, num_tendons: int
):
    """Ensure tendon_length Jacobian equals actuation matrix transpose."""

    segment_lengths = jnp.linspace(
        0.2, 0.2 + 0.05 * (num_segments - 1), num_segments, dtype=jnp.float64
    )
    tendon_params = _stacked_tendon_params(num_segments, num_tendons)
    robot = _create_robot(segment_lengths, tendon_params)

    key = jax.random.PRNGKey(42 + num_segments * 10 + num_tendons)

    for _ in range(10):
        key, subkey = jax.random.split(key)
        q = 0.05 * jax.random.normal(
            subkey, (robot.num_active_strains,), dtype=jnp.float64
        )
        lengths = robot.tendon_length(q)
        assert lengths.shape == (robot.num_actuators,)
        jac = jax.jacrev(robot.tendon_length)(q)
        A = robot.actuation_matrix(q)
        assert_allclose(
            jac,
            A.T,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )


@pytest.mark.parametrize("num_passive_tendons", [1, 3])
@pytest.mark.parametrize("num_segments", [1, 2])
def test_passive_tendon_length_gradient_matches_jacobian(
    num_segments: int, num_passive_tendons: int
):
    """Ensure passive_tendon_length Jacobian equals jacobian_passive_tendon."""

    segment_lengths = jnp.linspace(
        0.25, 0.25 + 0.05 * (num_segments - 1), num_segments, dtype=jnp.float64
    )
    active_params = _stacked_tendon_params(num_segments, max(1, num_passive_tendons))
    passive_routing_params = _stacked_tendon_params(num_segments, num_passive_tendons)
    passive_tendon_phys = passive_tendon_params(
        stiffness=jnp.linspace(
            70.0,
            70.0 + 10.0 * (num_passive_tendons - 1),
            num_passive_tendons,
            dtype=jnp.float64,
        ),
        damping=jnp.linspace(
            0.6,
            0.6 + 0.2 * (num_passive_tendons - 1),
            num_passive_tendons,
            dtype=jnp.float64,
        ),
        rest_length_offset=-0.004 * jnp.arange(num_passive_tendons, dtype=jnp.float64),
    )
    robot = _create_robot(
        segment_lengths,
        active_params,
        passive_tendon_routing=passive_routing_params,
        passive_tendon=passive_tendon_phys,
    )

    key = jax.random.PRNGKey(11 + num_segments * 7 + num_passive_tendons)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        q = 0.04 * jax.random.normal(
            subkey, (robot.num_active_strains,), dtype=jnp.float64
        )
        lengths = robot.passive_tendon_length(q)
        assert lengths.shape == (num_passive_tendons,)
        jac = jax.jacrev(robot.passive_tendon_length)(q)
        J_pt = robot.jacobian_passive_tendon(q)
        assert_allclose(
            jac,
            J_pt,
            rtol=Tolerance.rtol(),
            atol=Tolerance.atol(),
        )


def test_passive_tendons_contribute_to_elastic_terms():
    """Verify passive tendons add stiffness, damping, and energy contributions."""

    segment_lengths = jnp.array([0.45, 0.55], dtype=jnp.float64)
    active_params = _stacked_tendon_params(2, 2)
    passive_routing_params = _stacked_tendon_params(2, 2)
    passive_tendon_phys = passive_tendon_params(
        stiffness=jnp.array([60.0, 85.0], dtype=jnp.float64),
        damping=jnp.array([0.7, 1.1], dtype=jnp.float64),
        rest_length_offset=jnp.array([-0.012, 0.008], dtype=jnp.float64),
    )
    robot_base = _create_robot(segment_lengths, active_params)
    robot_passive = _create_robot(
        segment_lengths,
        active_params,
        passive_tendon_routing=passive_routing_params,
        passive_tendon=passive_tendon_phys,
    )

    q = jnp.linspace(
        -0.03, 0.04, int(robot_passive.num_active_strains), dtype=jnp.float64
    )

    J_pt = robot_passive.jacobian_passive_tendon(q)
    l_pt = robot_passive.passive_tendon_length(q)

    tau_passive = J_pt.T @ robot_passive.K_pt @ (l_pt - robot_passive.l_pt0)
    assert_allclose(
        robot_passive.elastic_force(q),
        robot_base.elastic_force(q) + tau_passive,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    D_passive = J_pt.T @ robot_passive.D_pt @ J_pt
    assert_allclose(
        robot_passive.damping_matrix(q),
        robot_base.damping_matrix(q) + D_passive,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    energy_passive = (
        0.5
        * (l_pt - robot_passive.l_pt0).T
        @ robot_passive.K_pt
        @ (l_pt - robot_passive.l_pt0)
    )
    assert_allclose(
        robot_passive.elastic_energy(q),
        robot_base.elastic_energy(q) + energy_passive,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_empty_passive_tendons_preserve_pcs_dynamics_terms():
    active_params = _stacked_tendon_params(2, 2)
    robot = _create_robot(jnp.array([0.2, 0.25]), active_params)
    q = jnp.linspace(-0.02, 0.03, int(robot.num_active_strains), dtype=jnp.float64)

    assert robot.jacobian_passive_tendon(q).shape == (0, robot.num_active_strains)
    assert robot.passive_tendon_length(q).shape == (0,)
    assert jnp.all(jnp.isfinite(robot.elastic_force(q)))
    assert jnp.all(jnp.isfinite(robot.damping_matrix(q)))
    assert jnp.isfinite(robot.elastic_energy(q))


def test_tendon_actuated_pcs_exposes_actuator_visual_layers():
    active_params = _stacked_tendon_params(2, 3)
    passive_routing_params = _stacked_tendon_params(2, 2)
    passive_tendon_phys = passive_tendon_params(
        stiffness=jnp.array([40.0, 60.0]),
        damping=jnp.array([0.2, 0.4]),
        rest_length_offset=jnp.array([0.0, 0.01]),
    )
    robot = _create_robot(
        jnp.array([0.2, 0.25]),
        active_params,
        passive_tendon_routing=passive_routing_params,
        passive_tendon=passive_tendon_phys,
    )
    q = jnp.zeros((int(robot.num_active_strains),), dtype=jnp.float64)
    s_points = jnp.linspace(0.0, float(robot.length), 7)

    layers = robot.actuator_visual_layers(q, s_points)

    assert len(layers) == 2
    assert layers[0].name == "active_tendons"
    assert layers[0].kind == "tendon"
    assert layers[0].points.shape == (3, 7, 3)
    assert layers[1].name == "passive_tendons"
    assert layers[1].points.shape == (2, 7, 3)


def test_update_tendon_params():
    """Verify the update of active and passive tendon parameters."""

    segment_lengths = jnp.array([0.5, 0.5])
    num_segments = segment_lengths.shape[0]
    num_tendons = 2

    active_params = _stacked_tendon_params(num_segments, num_tendons)
    passive_routing = _stacked_tendon_params(num_segments, num_tendons)

    k_init = jnp.array([100.0, 100.0])
    d_init = jnp.array([1.0, 1.0])
    l0_init = jnp.array([0.0, 0.0])
    passive_phys = passive_tendon_params(
        stiffness=k_init,
        damping=d_init,
        rest_length_offset=l0_init,
    )

    robot = _create_robot(
        segment_lengths,
        active_params,
        passive_tendon_routing=passive_routing,
        passive_tendon=passive_phys,
    )

    # 1. active tendon routing update
    new_ry = active_params.y_intercept + 0.05
    updated_robot_active = robot.update_params(
        active_tendon_routing=active_params.replace(y_intercept=new_ry)
    )

    assert_allclose(updated_robot_active.active_tendon_routing.y_intercept, new_ry)
    assert_allclose(
        updated_robot_active.active_tendon_routing.z_intercept,
        active_params.z_intercept,
    )
    assert_allclose(robot.active_tendon_routing.y_intercept, active_params.y_intercept)

    with pytest.raises(KeyError, match="Unknown parameter field"):
        active_params.replace(invalid_param=jnp.array([1.0]))

    # 2. passive tendon routing update
    new_rz = passive_routing.z_intercept - 0.02
    updated_robot_passive_routing = robot.update_params(
        passive_tendon_routing=passive_routing.replace(z_intercept=new_rz)
    )

    assert_allclose(
        updated_robot_passive_routing.passive_tendon_routing.z_intercept, new_rz
    )
    assert_allclose(
        robot.passive_tendon_routing.z_intercept, passive_routing.z_intercept
    )

    with pytest.raises(KeyError, match="Unknown parameter field"):
        passive_routing.replace(invalid_param=jnp.array([1.0]))

    # 3. passive tendon impedance update
    new_k = jnp.array([250.0, 300.0])
    new_d = jnp.array([5.0, 6.0])
    new_l0 = jnp.array([0.01, 0.02])

    # Case A: partial update
    updated_robot_phys_k = robot.update_params(
        passive_tendon=passive_phys.replace(stiffness=new_k)
    )

    assert_allclose(updated_robot_phys_k.K_pt, jnp.diag(new_k))
    assert_allclose(updated_robot_phys_k.D_pt, jnp.diag(d_init))

    # Case B: full update
    updated_robot_phys_all = robot.update_params(
        passive_tendon=PassiveTendonParams(
            stiffness=new_k,
            damping=new_d,
            rest_length_offset=new_l0,
        )
    )

    assert_allclose(updated_robot_phys_all.K_pt, jnp.diag(new_k))
    assert_allclose(updated_robot_phys_all.D_pt, jnp.diag(new_d))
    assert_allclose(updated_robot_phys_all.l_pt0, new_l0)

    # Verify immutability of original object
    assert_allclose(robot.K_pt, jnp.diag(k_init))


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
