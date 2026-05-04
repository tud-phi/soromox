import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from jax import Array, jacfwd, jacrev, jvp
from numpy.testing import assert_allclose
from system_param_builders import gvs_params_from_segments, pcs_params

import soromox.utils.lie_algebra as lie
from soromox.systems import GVS, PCS, CrossSectionGeometry, PCSStructure
from soromox.systems.gvs import GVSSegment, JointSpec, LinkSpec, StrainBasisSpec
from soromox.utils.tolerance import Tolerance

jax.config.update("jax_enable_x64", True)

TIP_RTOL = 1e-6
TIP_ATOL = 1e-8
RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()
NUM_RANDOM_SAMPLES = 5


def build_matched_gvs_pcs(num_segments: int = 1, n_gauss: int = 5) -> tuple[GVS, PCS]:
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
    segments = [
        GVSSegment(
            link=LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=float(E[i]),
                nu=float(nu[i]),
                rho=float(rhos[i]),
                eta=0.0,  # no damping in this cross-model comparison
                L=float(Ls[i]),
                r_i=float(rs[i]),
                r_f=float(rs[i]),
            ),
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="monomial",
                active=[1, 1, 1, 1, 1, 1],
                orders=[0, 0, 0, 0, 0, 0],
                xi_ref=[0, 0, 0, 1, 0, 0],
            ),
            num_gauss_points=n_gauss,
        )
        for i in range(num_segments)
    ]

    # Gravity: pick physically consistent vectors as per note
    g = jnp.array([0.0, 0.0, -9.81])

    # initialize the GVS model
    gvs_params, gvs_structure = gvs_params_from_segments(
        segments, gravity=g, base_pose=jnp.zeros(6)
    )
    robot_gvs = GVS(params=gvs_params, structure=gvs_structure)

    # PCS definition with identical geometry and material params
    params = pcs_params(
        base_pose=jnp.zeros(6),
        length=Ls,
        radius=rs,
        density=rhos,
        gravity=g,
        young_modulus=E,
        shear_modulus=Gpcs,
        damping_matrix=jnp.zeros((6 * num_segments, 6 * num_segments)),
        reference_strain=jnp.tile(
            jnp.array([0, 0, 0, 1, 0, 0]), (num_segments, 1)
        ).reshape(6 * num_segments),
    )
    robot_pcs = PCS(
        params=params,
        structure=PCSStructure(
            num_gauss_points=5,
            strain_selector=jnp.ones((6 * num_segments,), dtype=bool),
        ),
    )

    return robot_gvs, robot_pcs


def test_params_from_segments_stores_resolved_max_dof():
    segment = GVSSegment(
        link=LinkSpec(
            cross_section_geometry=CrossSectionGeometry.CIRCULAR,
            E=1e6,
            nu=0.45,
            rho=1000.0,
            eta=0.0,
            L=0.2,
            r_i=0.02,
            r_f=0.02,
        ),
        joint=JointSpec(type="fixed"),
        basis=StrainBasisSpec(
            type="monomial",
            active=[1, 1, 1, 1, 1, 1],
            orders=[0, 0, 0, 0, 0, 0],
            xi_ref=[0, 0, 0, 1, 0, 0],
        ),
        num_gauss_points=5,
    )

    params, structure = GVS.params_from_segments(
        [segment], gravity=jnp.array([0.0, 0.0, -9.81])
    )

    assert structure.max_dof == params.joint_stiffness.shape[1] == 6

    oversized = params.replace(joint_stiffness=jnp.zeros((1, 7, 7)))
    with pytest.raises(ValueError, match="joint_stiffness"):
        oversized.validate_against_structure(structure)
    with pytest.raises(ValueError, match="joint_stiffness"):
        GVS(params=oversized, structure=structure)


def build_varied_basis_gvs(num_segments: int = 3) -> GVS:
    """
    Initialize a GVS robot with a configurable number of segments, mixing joint families and
    non-constant strain bases across the structure.
    """
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1")

    pattern_count = 3
    axes_cycle = ("x", "y", "z")
    planes_cycle = ("xy", "yz", "xz")

    def _circular_link(idx: int) -> LinkSpec:
        repeat = idx // pattern_count
        scale = 1.0 + 0.04 * repeat
        r_i = 0.015 + 0.0008 * repeat
        r_f = r_i + 0.003 + 0.0004 * repeat
        return LinkSpec(
            cross_section_geometry=CrossSectionGeometry.CIRCULAR,
            E=1.2e6,
            nu=0.45,
            rho=950.0,
            eta=5.0,
            L=float(0.25 * scale),
            r_i=float(r_i),
            r_f=float(r_f),
        )

    def _rectangular_link(idx: int) -> LinkSpec:
        repeat = idx // pattern_count
        scale = 1.0 + 0.03 * repeat
        h_i = 0.03 + 0.001 * repeat
        h_f = max(0.022, h_i * 0.9)
        w_i = 0.02 + 0.0008 * repeat
        w_f = max(0.016, w_i * 0.88)
        return LinkSpec(
            cross_section_geometry=CrossSectionGeometry.RECTANGULAR,
            E=9.5e5,
            nu=0.38,
            rho=1025.0,
            eta=4.0,
            L=float(0.18 * scale),
            h_i=float(h_i),
            h_f=float(h_f),
            w_i=float(w_i),
            w_f=float(w_f),
        )

    def _elliptical_link(idx: int) -> LinkSpec:
        repeat = idx // pattern_count
        scale = 1.0 + 0.025 * repeat
        a_i = max(0.016, 0.02 - 0.0006 * repeat)
        a_f = max(0.014, a_i * 0.92)
        b_i = 0.015 + 0.0007 * repeat
        b_f = b_i * 1.05
        return LinkSpec(
            cross_section_geometry=CrossSectionGeometry.ELLIPTICAL,
            E=8.0e5,
            nu=0.4,
            rho=980.0,
            eta=3.5,
            L=float(0.22 * scale),
            a_i=float(a_i),
            a_f=float(a_f),
            b_i=float(b_i),
            b_f=float(b_f),
        )

    def _revolute_joint(idx: int) -> JointSpec:
        axis = axes_cycle[idx % len(axes_cycle)]
        return JointSpec(type="revolute", axis=axis)

    def _planar_joint(idx: int) -> JointSpec:
        repeat = idx // pattern_count
        plane = planes_cycle[(idx + repeat) % len(planes_cycle)]
        return JointSpec(type="planar", plane=plane)

    def _helical_joint(idx: int) -> JointSpec:
        repeat = idx // pattern_count
        axis = axes_cycle[(idx + 1) % len(axes_cycle)]
        pitch = 0.02 + 0.003 * repeat
        return JointSpec(type="helical", axis=axis, pitch=float(pitch))

    def _monomial_basis(idx: int) -> StrainBasisSpec:
        repeat = idx // pattern_count
        extra = repeat % 2
        return StrainBasisSpec(
            type="monomial",
            active=[1, 1, 0, 1, 0, 0],
            orders=[2 + extra, 1 + (idx % 2), 0, 2 + ((idx + repeat) % 2), 0, 0],
            xi_ref=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )

    def _legendre_basis(idx: int) -> StrainBasisSpec:
        repeat = idx // pattern_count
        return StrainBasisSpec(
            type="legendre",
            active=[0, 1, 1, 0, 1, 0],
            orders=[0, 2 + (repeat % 2), 1 + ((idx + 1) % 2), 0, 1 + (repeat % 3), 0],
            xi_ref=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )

    def _fourier_basis(idx: int) -> StrainBasisSpec:
        repeat = idx // pattern_count
        xi_sigma = max(0.7, 0.9 - 0.05 * repeat)
        return StrainBasisSpec(
            type="fourier",
            active=[1, 0, 1, 1, 0, 1],
            orders=[
                1 + (idx % 2),
                0,
                2 + (repeat % 2),
                1 + ((idx + repeat) % 2),
                0,
                1 + ((repeat + 1) % 2),
            ],
            xi_ref=[0.2 + 0.02 * repeat, 0.0, 0.0, xi_sigma, 0.0, 0.0],
        )

    def _monomial_gauss(idx: int) -> int:
        repeat = idx // pattern_count
        return 6 + (repeat % 2)

    def _legendre_gauss(idx: int) -> int:
        repeat = idx // pattern_count
        return 7 + (repeat % 2)

    def _fourier_gauss(idx: int) -> int:
        repeat = idx // pattern_count
        return 6 + ((repeat + 1) % 2)

    pattern_builders = (
        {
            "link": _circular_link,
            "joint": _revolute_joint,
            "basis": _monomial_basis,
            "n_gauss": _monomial_gauss,
        },
        {
            "link": _rectangular_link,
            "joint": _planar_joint,
            "basis": _legendre_basis,
            "n_gauss": _legendre_gauss,
        },
        {
            "link": _elliptical_link,
            "joint": _helical_joint,
            "basis": _fourier_basis,
            "n_gauss": _fourier_gauss,
        },
    )

    segments: list[GVSSegment] = []

    for idx in range(num_segments):
        pattern = pattern_builders[idx % pattern_count]
        segments.append(
            GVSSegment(
                link=pattern["link"](idx),
                joint=pattern["joint"](idx),
                basis=pattern["basis"](idx),
                num_gauss_points=int(pattern["n_gauss"](idx)),
            )
        )

    return GVS.from_segments(segments, gravity=jnp.array([0.0, 0.0, -9.81]))


def build_constant_strain_gvs(
    num_segments: int = 1,
    selector_per_segment: tuple[bool, ...] | None = None,
    max_dof: int | None = None,
    segment_length: float = 0.2,
    scale_rotational_basis_by_length: bool = False,
) -> GVS:
    if selector_per_segment is None:
        selector_per_segment = (True, True, True, True, True, True)

    segments = [
        GVSSegment(
            link=LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=1e6,
                nu=0.5,
                rho=1000.0,
                eta=0.0,
                L=segment_length,
                r_i=0.02,
                r_f=0.02,
            ),
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="monomial",
                active=[int(active) for active in selector_per_segment],
                orders=[0, 0, 0, 0, 0, 0],
                xi_ref=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            num_gauss_points=5,
        )
        for _ in range(num_segments)
    ]

    params, structure = gvs_params_from_segments(
        segments,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        base_pose=jnp.zeros(6),
        max_dof=max_dof,
        scale_rotational_basis_by_length=scale_rotational_basis_by_length,
    )
    return GVS(params=params, structure=structure)


def test_gvs_segment_factories_match_explicit_constructor() -> None:
    segments = [
        GVSSegment(
            link=LinkSpec.circular(
                E=1.0e6,
                nu=0.45,
                rho=1000.0,
                eta=1.0,
                L=0.2,
                r=0.02,
            ),
            joint=JointSpec(type="fixed"),
            basis=StrainBasisSpec(
                type="monomial",
                active=[1, 1, 1, 1, 0, 0],
                orders=[0, 0, 0, 0, 0, 0],
            ),
            num_gauss_points=5,
        ),
        GVSSegment(
            link=LinkSpec.rectangular(
                E=9.0e5,
                nu=0.4,
                rho=950.0,
                eta=2.0,
                L=0.15,
                h=0.03,
                w=0.02,
            ),
            joint=JointSpec(type="revolute", axis="z", stiffness=jnp.array([[0.3]])),
            basis=StrainBasisSpec(
                type="legendre",
                active=[0, 1, 1, 0, 0, 0],
                orders=[0, 1, 1, 0, 0, 0],
            ),
            num_gauss_points=6,
        ),
    ]
    gravity = jnp.array([0.0, 0.0, -9.81])

    params, structure = GVS.params_from_segments(
        segments,
        gravity=gravity,
        base_pose=jnp.zeros(6),
        max_dof=6,
    )
    explicit = GVS(params=params, structure=structure)
    factory = GVS.from_segments(
        segments,
        gravity=gravity,
        base_pose=jnp.zeros(6),
        max_dof=6,
    )

    assert_allclose(factory.params.link.length, params.link.length)
    assert_allclose(factory.joint_stiffness, explicit.joint_stiffness)
    assert_allclose(
        factory.cross_section_geometry_index, explicit.cross_section_geometry_index
    )
    assert int(factory.cross_section_geometry_index[0]) == CrossSectionGeometry.CIRCULAR
    assert (
        int(factory.cross_section_geometry_index[1]) == CrossSectionGeometry.RECTANGULAR
    )


def test_gvs_structure_contains_only_static_segment_choices() -> None:
    segment = GVSSegment(
        link=LinkSpec.circular(
            E=1.0e6,
            nu=0.45,
            rho=1000.0,
            eta=1.0,
            L=0.2,
            r=0.02,
        ),
        joint=JointSpec(type="revolute", axis="z", stiffness=jnp.array([[0.3]])),
        basis=StrainBasisSpec(
            type="monomial",
            active=[1, 1, 0, 0, 0, 0],
            orders=[0, 1, 0, 0, 0, 0],
            xi_ref=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        num_gauss_points=5,
    )
    params, structure = GVS.params_from_segments(
        [segment],
        gravity=jnp.array([0.0, 0.0, -9.81]),
        max_dof=6,
    )
    stored_segment = structure.segments[0]

    assert stored_segment.link.cross_section_geometry == CrossSectionGeometry.CIRCULAR
    for dynamic_name in ("E", "L", "rho", "r_i"):
        assert not hasattr(stored_segment.link, dynamic_name)
    assert stored_segment.joint.type == "revolute"
    assert stored_segment.joint.axis == "z"
    assert not hasattr(stored_segment.joint, "stiffness")
    assert stored_segment.basis.active == (1, 1, 0, 0, 0, 0)
    assert stored_segment.basis.orders == (0, 1, 0, 0, 0, 0)
    assert not hasattr(stored_segment.basis, "xi_ref")
    assert_allclose(params.link.young_modulus, jnp.array([1.0e6]))
    assert_allclose(params.joint_stiffness[0, 0, 0], 0.3)


def sample_arc_lengths(robot: GVS) -> jnp.ndarray:
    lengths = jnp.asarray(robot.segment_length)
    cumulative = jnp.cumsum(lengths)
    total = float(cumulative[-1])

    near_zero = max(total * 1e-3, 1e-9)
    mids = (cumulative - lengths / 2.0).tolist()
    boundaries = cumulative.tolist()

    values = [near_zero] + mids + boundaries
    unique_sorted = sorted({float(v) for v in values if 0.0 < float(v) <= total})
    return jnp.asarray(unique_sorted, dtype=jnp.float64)


def strict_interior_arc_lengths(robot: GVS) -> jnp.ndarray:
    fractions = jnp.asarray([0.37], dtype=jnp.float64)
    starts = robot.segment_end_positions[:-1, None]
    lengths = robot.segment_length[:, None]
    return (starts + lengths * fractions).reshape(-1)


def tip_arc_lengths(robot: GVS) -> jnp.ndarray:
    return jnp.asarray(robot.segment_end_positions[1:], dtype=jnp.float64)


def random_q(robot: GVS, key, scale: float = 0.05) -> jnp.ndarray:
    dof = int(robot.num_dofs)
    return scale * jax.random.normal(key, (dof,), dtype=jnp.float64)


def expected_selection_basis(num_rows: int, active_indices: tuple[int, ...]) -> Array:
    basis = jnp.zeros((num_rows, len(active_indices)), dtype=jnp.float64)
    for col, row in enumerate(active_indices):
        basis = basis.at[row, col].set(1.0)
    return basis


def se3_inverse(g: jnp.ndarray) -> jnp.ndarray:
    R = g[:3, :3]
    p = g[:3, 3]
    R_T = R.T
    g_inv = jnp.eye(4, dtype=jnp.float64)
    g_inv = g_inv.at[:3, :3].set(R_T)
    g_inv = g_inv.at[:3, 3].set(-R_T @ p)
    return g_inv


def se3_tangent_to_body_twist(
    g_inv: jnp.ndarray, g_tangent: jnp.ndarray
) -> jnp.ndarray:
    xi_hat_body = g_inv @ g_tangent
    skew_body = 0.5 * (xi_hat_body[:3, :3] - xi_hat_body[:3, :3].T)
    omega = jnp.array(
        [
            skew_body[2, 1],
            skew_body[0, 2],
            skew_body[1, 0],
        ],
        dtype=jnp.float64,
    )
    v_body = xi_hat_body[:3, 3]
    return jnp.concatenate((omega, v_body))


def body_twist_between(g_base: jnp.ndarray, g_target: jnp.ndarray) -> jnp.ndarray:
    g_rel = se3_inverse(g_base) @ g_target
    xi = lie.log_SE3(g_rel, eps=1e-12)
    R = g_rel[:3, :3]
    omega = xi[:3]
    alt_omega = 0.5 * jnp.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=jnp.float64,
    )
    omega = jnp.where(jnp.linalg.norm(omega) < 1e-10, alt_omega, omega)
    return xi.at[:3].set(omega)


def spatial_from_body(g: jnp.ndarray, xi_body: jnp.ndarray) -> jnp.ndarray:
    g_rot = jnp.block(
        [[g[:3, :3], jnp.zeros((3, 1))], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
    )
    return lie.Adjoint_g_SE3(g_rot) @ xi_body


def segment_tip_transforms(robot: GVS, q: jnp.ndarray) -> jnp.ndarray:
    s_tips = tip_arc_lengths(robot)
    return stack_forward_kinematics(robot, q, s_tips)


def build_full_and_reduced_constant_strain_gvs(
    num_segments: int, selector_per_segment: tuple[bool, ...]
) -> tuple[GVS, GVS, jnp.ndarray]:
    full = build_constant_strain_gvs(num_segments=num_segments, max_dof=6)
    reduced = build_constant_strain_gvs(
        num_segments=num_segments,
        selector_per_segment=selector_per_segment,
        max_dof=6,
    )
    B = jnp.zeros((int(full.num_dofs), int(reduced.num_dofs)))
    col = 0
    for segment_idx in range(num_segments):
        for component_idx, active in enumerate(selector_per_segment):
            if active:
                B = B.at[6 * segment_idx + component_idx, col].set(1.0)
                col += 1
    return full, reduced, B


def stack_forward_kinematics(
    robot: GVS, q: jnp.ndarray, s_points: jnp.ndarray
) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [robot.forward_kinematics(q, float(s)) for s in s_array],
        axis=0,
    )


def stack_jacobians(robot: GVS, q: jnp.ndarray, s_points: jnp.ndarray) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [robot.jacobian_bodyframe(q, float(s)) for s in s_array],
        axis=0,
    )


def stack_jacobian_time_derivatives(
    robot: GVS, q: jnp.ndarray, qd: jnp.ndarray, s_points: jnp.ndarray
) -> jnp.ndarray:
    s_array = onp.asarray(s_points, dtype=float)
    return jnp.stack(
        [
            robot.jacobian_and_time_derivative_bodyframe(q, qd, float(s))[1]
            for s in s_array
        ],
        axis=0,
    )


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


def _assert_gvs_pcs_coherence(num_segments: int) -> None:
    print("\nTesting GVS-PCS coherence for", num_segments, "segments")
    robot_gvs, robot_pcs = build_matched_gvs_pcs(num_segments)
    n = int(robot_gvs.num_dofs)

    zero_cfg = jnp.zeros((n,), dtype=jnp.float64)
    zero_vel = jnp.zeros((n,), dtype=jnp.float64)

    linear_cfg = jnp.linspace(0.01, 0.01 * n, n, dtype=jnp.float64)
    linear_vel = jnp.linspace(-0.02, -0.02 * n, n, dtype=jnp.float64)

    config_velocity_cases = (
        (zero_cfg, zero_vel),
        (linear_cfg, linear_vel),
    )

    for i in range(NUM_RANDOM_SAMPLES):
        key = jax.random.PRNGKey(100 + i)
        key_q, key_qd = jax.random.split(key)
        q_random = random_q(robot_gvs, key_q, scale=0.05)
        qd_random = random_q(robot_gvs, key_qd, scale=0.1)
        config_velocity_cases += ((q_random, qd_random),)

    L_total = float(jnp.sum(robot_gvs.length))
    s_candidates = jnp.concatenate(
        [
            jnp.asarray([0.0], dtype=jnp.float64),
            sample_arc_lengths(robot_gvs),
            tip_arc_lengths(robot_gvs),
            jnp.asarray([L_total], dtype=jnp.float64),
        ]
    )
    s_points = jnp.unique(s_candidates)
    s_loop = onp.asarray(s_points, dtype=float)

    for q, qd in config_velocity_cases:
        for s in s_loop:
            print("q =\n", q)
            print("qd =\n", qd)
            print("s =", s)

            # check the forward kinematics
            g_gvs = robot_gvs.forward_kinematics(q, s)
            g_pcs = robot_pcs.forward_kinematics(q, s)
            assert_allclose(
                g_gvs, g_pcs, rtol=RTOL, atol=ATOL, err_msg=f"FK mismatch at s={s}"
            )

            # check the Jacobians in body frames
            Jb_gvs = robot_gvs.jacobian_bodyframe(q, s)
            Jb_pcs = robot_pcs.jacobian_bodyframe(q, s)
            assert_allclose(
                Jb_gvs,
                Jb_pcs,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian bodyframe mismatch at s={s}",
            )

            # check the Jacobians in inertial frames
            Ji_gvs = robot_gvs.jacobian_inertialframe(q, s)
            Ji_pcs = robot_pcs.jacobian_inertialframe(q, s)
            assert_allclose(
                Ji_gvs,
                Ji_pcs,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian inertialframe mismatch at s={s}",
            )

            # check the Jacobian time derivatives in body frames
            Jb_gvs2, Jdb_gvs = robot_gvs.jacobian_and_time_derivative_bodyframe(
                q, qd, s
            )
            _, Jdb_pcs = robot_pcs.jacobian_and_time_derivative_bodyframe(q, qd, s)
            assert_allclose(
                Jb_gvs,
                Jb_gvs2,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian bodyframe mismatch at s={s}",
            )
            assert_allclose(
                Jdb_gvs,
                Jdb_pcs,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian time derivative bodyframe mismatch at s={s}",
            )

            # check the Jacobian time derivatives in inertial frames
            Ji_gvs2, Jdi_gvs = robot_gvs.jacobian_and_time_derivative_inertialframe(
                q, qd, s
            )
            _, Jdi_pcs = robot_pcs.jacobian_and_time_derivative_inertialframe(q, qd, s)
            assert_allclose(
                Ji_gvs,
                Ji_gvs2,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian inertialframe mismatch at s={s}",
            )
            assert_allclose(
                Jdi_gvs,
                Jdi_pcs,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Jacobian time derivative inertialframe mismatch at s={s}",
            )

        B_gvs = robot_gvs.inertia_matrix(q)
        B_pcs = robot_pcs.inertia_matrix(q)
        assert_allclose(B_gvs, B_pcs, rtol=RTOL, atol=ATOL)

        C_gvs = robot_gvs.coriolis_matrix(q, qd)
        C_pcs = robot_pcs.coriolis_matrix(q, qd)
        assert_allclose(C_gvs, C_pcs, rtol=RTOL, atol=ATOL)
        # coriolis forces
        tau_cor_gvs = C_gvs @ qd
        tau_cor_pcs = C_pcs @ qd
        assert_allclose(tau_cor_gvs, tau_cor_pcs, rtol=RTOL, atol=ATOL)

        K_gvs = robot_gvs.stiffness_matrix()
        K_pcs = robot_pcs.stiffness_matrix()
        assert_allclose(K_gvs, K_pcs, rtol=RTOL, atol=ATOL)

        G_gvs = robot_gvs.gravitational_force(q).reshape(-1)
        G_pcs = robot_pcs.gravitational_force(q).reshape(-1)
        assert_allclose(G_gvs, G_pcs, rtol=RTOL, atol=ATOL)

        kinetic_gvs = robot_gvs.kinetic_energy(q, qd)
        kinetic_pcs = robot_pcs.kinetic_energy(q, qd)
        assert_allclose(kinetic_gvs, kinetic_pcs, rtol=RTOL, atol=ATOL)

        potential_gvs = robot_gvs.potential_energy(q)
        potential_pcs = robot_pcs.potential_energy(q)
        assert_allclose(potential_gvs, potential_pcs, rtol=RTOL, atol=ATOL)

        gravitational_gvs = robot_gvs.gravitational_energy(q)
        gravitational_pcs = robot_pcs.gravitational_energy(q)
        assert_allclose(gravitational_gvs, gravitational_pcs, rtol=RTOL, atol=ATOL)

        total_gvs = robot_gvs.total_energy(q, qd)
        total_pcs = robot_pcs.total_energy(q, qd)
        assert_allclose(total_gvs, total_pcs, rtol=RTOL, atol=ATOL)

        y = jnp.concatenate([q, qd])
        u = jnp.zeros(n, dtype=jnp.float64)
        yd_gvs = robot_gvs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
        yd_pcs = robot_pcs.forward_dynamics(t=jnp.zeros(()), y=y, actuation_args=(u,))
        assert_allclose(yd_gvs, yd_pcs, rtol=RTOL, atol=ATOL)


def test_public_gvs_accessors_geometry_and_actuation_matrix() -> None:
    robot = build_varied_basis_gvs(num_segments=3)
    q = jnp.zeros((int(robot.num_dofs),), dtype=jnp.float64)

    assert robot.is_planar is False
    assert_allclose(robot.length, jnp.sum(robot.segment_length), rtol=RTOL, atol=ATOL)
    assert_allclose(
        robot.segment_length, robot.params.link.length, rtol=RTOL, atol=ATOL
    )

    s_second = robot.segment_length[0] + 0.25 * robot.segment_length[1]
    segment_idx, s_local = robot.classify_segment(s_second)
    assert int(segment_idx) == 1
    assert_allclose(s_local, 0.25 * robot.segment_length[1], rtol=RTOL, atol=ATOL)

    tag, geom = robot.cross_section_geometry(q, 0.5 * robot.segment_length[0])
    expected_radius = robot.radius_params[0, 0] + 0.5 * (
        robot.radius_params[0, 1] - robot.radius_params[0, 0]
    )
    assert int(tag) == CrossSectionGeometry.CIRCULAR
    assert_allclose(geom, jnp.array([expected_radius]), rtol=RTOL, atol=ATOL)

    tag, geom = robot.cross_section_geometry(q, s_second)
    expected_width = robot.width_params[1, 0] + 0.25 * (
        robot.width_params[1, 1] - robot.width_params[1, 0]
    )
    expected_height = robot.height_params[1, 0] + 0.25 * (
        robot.height_params[1, 1] - robot.height_params[1, 0]
    )
    assert int(tag) == CrossSectionGeometry.RECTANGULAR
    assert_allclose(
        geom, jnp.array([expected_width, expected_height]), rtol=RTOL, atol=ATOL
    )

    s_third = (
        robot.segment_length[0]
        + robot.segment_length[1]
        + 0.75 * robot.segment_length[2]
    )
    tag, geom = robot.cross_section_geometry(q, s_third)
    expected_a = robot.semi_major_params[2, 0] + 0.75 * (
        robot.semi_major_params[2, 1] - robot.semi_major_params[2, 0]
    )
    expected_b = robot.semi_minor_params[2, 0] + 0.75 * (
        robot.semi_minor_params[2, 1] - robot.semi_minor_params[2, 0]
    )
    assert int(tag) == CrossSectionGeometry.ELLIPTICAL
    assert_allclose(geom, jnp.array([expected_a, expected_b]), rtol=RTOL, atol=ATOL)

    robot.precompute()
    assert_allclose(robot.K_full, robot._stiffness_full_matrix(), rtol=RTOL, atol=ATOL)
    assert_allclose(robot.D_full, robot._damping_full_matrix(), rtol=RTOL, atol=ATOL)

    A = robot.actuation_matrix(q)
    assert A.shape == (robot.num_dofs, robot.num_actuators)
    assert_allclose(A, jnp.eye(robot.num_actuators), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    total_length = float(robot.segment_end_positions[-1])

    q_keys = jax.random.split(jax.random.PRNGKey(888), NUM_RANDOM_SAMPLES)
    s_keys = jax.random.split(jax.random.PRNGKey(777), NUM_RANDOM_SAMPLES)

    num_points = max(3 * robot.num_segments, 3)

    for q_key, s_key in zip(q_keys, s_keys):
        q = random_q(robot, q_key, scale=0.05)
        random_points = jax.random.uniform(
            s_key,
            shape=(num_points,),
            minval=0.0,
            maxval=total_length,
            dtype=jnp.float64,
        )
        s_points = jnp.sort(
            jnp.concatenate(
                (
                    jnp.array([0.0, total_length], dtype=jnp.float64),
                    random_points,
                )
            )
        )

        g_batched = robot.forward_kinematics_batched(q, s_points)
        g_expected = stack_forward_kinematics(robot, q, s_points)

        assert_allclose(g_batched, g_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_tips_matches_pointwise_and_gauss(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    q = random_q(robot, jax.random.PRNGKey(1), scale=0.02)

    g_tips = robot.forward_kinematics_tips(q)
    s_tips = tip_arc_lengths(robot)
    g_expected = stack_forward_kinematics(robot, q, s_tips)

    q_gathered = robot._min_size_gathered(q)
    g_gauss_tips = robot._forward_kinematics_gauss(q_gathered)[:, -1]

    assert g_tips.shape == (num_segments, 4, 4)
    assert_allclose(g_tips, g_expected, rtol=TIP_RTOL, atol=TIP_ATOL)
    assert_allclose(g_tips, g_gauss_tips, rtol=TIP_RTOL, atol=TIP_ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_bodyframe_inertialframe_coherence(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    q = random_q(robot, jax.random.PRNGKey(1), scale=0.05)

    for s in sample_arc_lengths(robot):
        J_impl = robot.jacobian_inertialframe(q, float(s))
        J_body = robot.jacobian_bodyframe(q, float(s))
        J_expected = spatial_from_body(robot.forward_kinematics(q, float(s)), J_body)

        assert_allclose(J_impl, J_expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_bodyframe_matches_autodiff(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(6)
    q_keys = jax.random.split(key, NUM_RANDOM_SAMPLES)

    for q_key in q_keys:
        q = random_q(robot, q_key, scale=0.03)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            s_val = float(s)
            J_body = robot.jacobian_bodyframe(q, s_val)
            g = robot.forward_kinematics(q, s_val)
            g_inv = se3_inverse(g)

            def fk(qq: jnp.ndarray, s_current: float = s_val) -> jnp.ndarray:
                return robot.forward_kinematics(qq, s_current)

            cols = []
            eye = jnp.eye(q.shape[0], dtype=jnp.float64)
            for j in range(q.shape[0]):
                _, g_tangent = jvp(fk, (q,), (eye[j],))
                cols.append(se3_tangent_to_body_twist(g_inv, g_tangent))

            J_ad = jnp.stack(cols, axis=1)
            assert_allclose(J_body, J_ad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_bodyframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    total_length = float(robot.segment_end_positions[-1])

    q_keys = jax.random.split(jax.random.PRNGKey(4242), NUM_RANDOM_SAMPLES)
    s_keys = jax.random.split(jax.random.PRNGKey(1313), NUM_RANDOM_SAMPLES)

    num_points = max(3 * robot.num_segments, 3)

    for q_key, s_key in zip(q_keys, s_keys):
        q = random_q(robot, q_key, scale=0.05)
        random_points = jax.random.uniform(
            s_key,
            shape=(num_points,),
            minval=0.0,
            maxval=total_length,
            dtype=jnp.float64,
        )
        s_points = jnp.sort(
            jnp.concatenate(
                (
                    jnp.array([0.0, total_length], dtype=jnp.float64),
                    random_points,
                )
            )
        )

        J_batch = robot.jacobian_bodyframe_batched(q, s_points)
        J_expected = stack_jacobians(robot, q, s_points)

        assert_allclose(J_batch, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_matches_autodiff(num_segments: int):
    model = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(7)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(model, q_key, scale=0.03)

        for s in sample_arc_lengths(model):
            if s < 1e-3:
                continue

            s_val = float(s)
            J_impl = model.jacobian_inertialframe(q, s_val)
            g = model.forward_kinematics(q, s_val)
            R = g[:3, :3]
            g_inv = se3_inverse(g)

            def fk(qq: Array, s_current: float = s_val) -> Array:
                return model.forward_kinematics(qq, s_current)

            n = q.shape[0]
            eye = jnp.eye(n)
            cols = []
            for j in range(n):
                _, g_tangent = jax.jvp(fk, (q,), (eye[j],))
                Xi_hat_body = g_inv @ g_tangent
                skew_body = 0.5 * (Xi_hat_body[:3, :3] - Xi_hat_body[:3, :3].T)
                omega = jnp.array(
                    [
                        skew_body[2, 1],
                        skew_body[0, 2],
                        skew_body[1, 0],
                    ]
                )
                v_body = Xi_hat_body[:3, 3]
                cols.append(jnp.concatenate((R @ omega, R @ v_body)))

            J_ad = jnp.stack(cols, axis=1)

            assert_allclose(
                J_impl,
                J_ad,
                rtol=1e-6,
                atol=1e-7,
                err_msg=(
                    f"num_segments={num_segments}, s={s}\n"
                    f"J_impl:\n{J_impl}\nJ_ad:\n{J_ad}"
                ),
            )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_inertial_velocity_matches_central_differences(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(4))
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dt = 1e-6
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.03)
        qd = random_q(robot, qd_key, scale=0.1)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            J = robot.jacobian_inertialframe(q, float(s))
            xd_pred = J @ qd

            g0 = robot.forward_kinematics(q, float(s))
            g1 = robot.forward_kinematics(q + dt * qd, float(s))
            xi_body = body_twist_between(g0, g1) / dt
            xd_fd = spatial_from_body(g0, xi_body)

            assert_allclose(xd_pred, xd_fd, rtol=5e-5, atol=5e-7)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_matches_central_differences(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(5)

    delta = 1e-6
    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(robot, q_key, scale=0.02)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            J_impl = robot.jacobian_inertialframe(q, float(s))

            xi_cols = []
            eye = jnp.eye(q.shape[0], dtype=jnp.float64)
            for j in range(q.shape[0]):
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                g_plus = robot.forward_kinematics(qp, float(s))
                g_minus = robot.forward_kinematics(qm, float(s))
                xi_body = body_twist_between(g_minus, g_plus) / (2 * delta)
                xi_cols.append(spatial_from_body(g_minus, xi_body))

            J_fd = jnp.stack(xi_cols, axis=1)
            J_body = robot.jacobian_bodyframe(q, float(s))
            J_rot_expected = spatial_from_body(
                robot.forward_kinematics(q, float(s)), J_body
            )

            assert_allclose(J_impl[:3], J_rot_expected[:3], rtol=1e-6, atol=1e-7)
            assert_allclose(J_impl[3:], J_fd[3:], rtol=1e-3, atol=5e-6)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    total_length = float(robot.segment_end_positions[-1])

    q_keys = jax.random.split(jax.random.PRNGKey(4243), NUM_RANDOM_SAMPLES)
    s_keys = jax.random.split(jax.random.PRNGKey(1314), NUM_RANDOM_SAMPLES)

    num_points = max(3 * robot.num_segments, 3)

    for q_key, s_key in zip(q_keys, s_keys):
        q = random_q(robot, q_key, scale=0.05)
        random_points = jax.random.uniform(
            s_key,
            shape=(num_points,),
            minval=0.0,
            maxval=total_length,
            dtype=jnp.float64,
        )
        s_points = jnp.sort(
            jnp.concatenate(
                (
                    jnp.array([0.0, total_length], dtype=jnp.float64),
                    random_points,
                )
            )
        )

        J_batch = robot.jacobian_inertialframe_batched(q, s_points)
        J_expected = jnp.stack(
            [robot.jacobian_inertialframe(q, float(s)) for s in s_points],
            axis=0,
        )

        assert_allclose(J_batch, J_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_tips_matches_pointwise_and_gauss(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    q = random_q(robot, jax.random.PRNGKey(2), scale=0.02)

    J_tips = robot.jacobian_tips(q)
    s_tips = tip_arc_lengths(robot)
    J_expected = jnp.stack(
        [robot.jacobian_inertialframe(q, float(s)) for s in s_tips],
        axis=0,
    )

    q_gathered = robot._min_size_gathered(q)
    J_local_gauss_tips = robot._jacobian_gauss(q_gathered)[:, -1] @ robot.active_dof_map
    g_tips = robot.forward_kinematics_tips(q)

    def rotate_pair(g_i: jax.Array, J_i: jax.Array) -> jax.Array:
        R = g_i[:3, :3]
        g_rot = jnp.block(
            [
                [R, jnp.zeros((3, 1), dtype=R.dtype)],
                [jnp.zeros((1, 3), dtype=R.dtype), jnp.ones((1, 1), dtype=R.dtype)],
            ]
        )
        return lie.Adjoint_g_SE3(g_rot) @ J_i

    J_gauss_tips = jax.vmap(rotate_pair)(g_tips, J_local_gauss_tips)

    assert J_tips.shape == (num_segments, 6, robot.num_dofs)
    assert_allclose(J_tips, J_expected, rtol=TIP_RTOL, atol=TIP_ATOL)
    assert_allclose(J_tips, J_gauss_tips, rtol=TIP_RTOL, atol=TIP_ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_and_time_derivative_bodyframe_matches_autograd_jvp(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(3)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            s_val = float(s)

            def J_body(q_, s_current: float = s_val):
                return robot.jacobian_bodyframe(q_, s_current)

            _, Jd_jvp = jvp(J_body, (q,), (qd,))
            _, Jd_impl = robot.jacobian_and_time_derivative_bodyframe(q, qd, s_val)

            # product of Jd with qd
            Jd_jvp_product = Jd_jvp @ qd
            Jd_impl_product = Jd_impl @ qd

            # assert that the product of Jd with qd matches
            assert_allclose(Jd_impl_product, Jd_jvp_product, rtol=1e-6, atol=1e-7)
            # assert the full Jacobian time derivatives match
            assert_allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_time_derivative_bodyframe_matches_central_differences(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(4)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    delta = 1e-6
    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            _, Jd_impl = robot.jacobian_and_time_derivative_bodyframe(q, qd, float(s))

            eye = jnp.eye(q.shape[0], dtype=jnp.float64)
            dJ_cols = []
            for j in range(q.shape[0]):
                qp = q + delta * eye[j]
                qm = q - delta * eye[j]
                Jp = robot.jacobian_bodyframe(qp, float(s))
                Jm = robot.jacobian_bodyframe(qm, float(s))
                dJ_cols.append((Jp - Jm) / (2 * delta))

            dJ_dq_fd = jnp.stack(dJ_cols, axis=-1)
            Jd_num = jnp.tensordot(dJ_dq_fd, qd, axes=([-1], [0]))

            assert_allclose(Jd_impl, Jd_num, rtol=1e-3, atol=5e-6)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_time_derivative_inertialframe_matches_autograd_jvp(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(3))
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        for s in sample_arc_lengths(robot):
            if s < 1e-3:
                continue

            s_val = float(s)

            def J_global(q_, s_current: float = s_val):
                return robot.jacobian_inertialframe(q_, s_current)

            _, Jd_jvp = jvp(J_global, (q,), (qd,))
            _, Jd_impl = robot.jacobian_and_time_derivative_inertialframe(q, qd, s_val)

            assert_allclose(Jd_impl, Jd_jvp, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_and_time_derivative_bodyframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    dof = int(robot.num_dofs)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    q_random = random_q(robot, jax.random.PRNGKey(321), scale=0.05)
    qd_random = random_q(robot, jax.random.PRNGKey(4321), scale=0.1)

    s_points = sample_arc_lengths(robot)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = robot.jacobian_and_time_derivative_bodyframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = robot.jacobian_and_time_derivative_bodyframe(
                q, qd, float(s_val)
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_jacobian_and_time_derivative_inertialframe_batched_matches_pointwise_evaluation(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    dof = int(robot.num_dofs)

    zero_cfg = jnp.zeros((dof,), dtype=jnp.float64)
    zero_vel = jnp.zeros((dof,), dtype=jnp.float64)

    q_random = random_q(robot, jax.random.PRNGKey(7890), scale=0.05)
    qd_random = random_q(robot, jax.random.PRNGKey(9876), scale=0.1)

    s_points = sample_arc_lengths(robot)

    for q, qd in ((zero_cfg, zero_vel), (q_random, qd_random)):
        J_batch, Jd_batch = robot.jacobian_and_time_derivative_inertialframe_batched(
            q, qd, s_points
        )

        for idx, s_val in enumerate(s_points):
            J_single, Jd_single = robot.jacobian_and_time_derivative_inertialframe(
                q, qd, float(s_val)
            )
            assert_allclose(J_batch[idx], J_single, rtol=RTOL, atol=ATOL)
            assert_allclose(Jd_batch[idx], Jd_single, rtol=RTOL, atol=ATOL)


def test_public_gvs_jacobian_adapters_match_inertialframe_methods() -> None:
    robot = build_varied_basis_gvs(num_segments=2)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(2026))
    q = random_q(robot, key_q, scale=0.03)
    qd = random_q(robot, key_qd, scale=0.04)
    s = 0.6 * robot.length
    s_ps = sample_arc_lengths(robot)

    assert_allclose(
        robot.jacobian(q, s),
        robot.jacobian_inertialframe(q, s),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        robot.jacobian_batched(q, s_ps),
        robot.jacobian_inertialframe_batched(q, s_ps),
        rtol=RTOL,
        atol=ATOL,
    )

    J, Jd = robot.jacobian_and_time_derivative(q, qd, s)
    J_expected, Jd_expected = robot.jacobian_and_time_derivative_inertialframe(q, qd, s)
    assert_allclose(J, J_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd, Jd_expected, rtol=RTOL, atol=ATOL)

    J_batch, Jd_batch = robot.jacobian_and_time_derivative_batched(q, qd, s_ps)
    J_batch_expected, Jd_batch_expected = (
        robot.jacobian_and_time_derivative_inertialframe_batched(q, qd, s_ps)
    )
    assert_allclose(J_batch, J_batch_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(Jd_batch, Jd_batch_expected, rtol=RTOL, atol=ATOL)


def test_gvs_arc_length_derivatives_match_autodiff() -> None:
    robot = build_varied_basis_gvs(num_segments=3)
    q = random_q(robot, jax.random.PRNGKey(2029), scale=0.03)

    for s in strict_interior_arc_lengths(robot):
        _, fk_s_autodiff = jvp(
            lambda s_: robot._forward_kinematics(q, s_),
            (s,),
            (jnp.ones_like(s),),
        )
        assert_allclose(
            robot.forward_kinematics_arc_length_derivative(q, s),
            fk_s_autodiff,
            rtol=1e-6,
            atol=1e-7,
        )

        _, Js_autodiff = jvp(
            lambda s_: robot._jacobian(q, s_),
            (s,),
            (jnp.ones_like(s),),
        )
        assert_allclose(
            robot.jacobian_arc_length_derivative(q, s),
            Js_autodiff,
            rtol=1e-5,
            atol=1e-6,
        )


def test_gvs_custom_jvps_include_arc_length_derivative() -> None:
    robot = build_varied_basis_gvs(num_segments=3)
    q = random_q(robot, jax.random.PRNGKey(2030), scale=0.03)
    qd = random_q(robot, jax.random.PRNGKey(2031), scale=0.02)
    s = strict_interior_arc_lengths(robot)[1]
    sd = jnp.array(0.37, dtype=jnp.float64)

    _, posed = jvp(
        lambda q_, s_: robot.forward_kinematics(q_, s_),
        (q, s),
        (qd, sd),
    )
    _, expected_posed = jvp(
        lambda q_, s_: robot._forward_kinematics(q_, s_),
        (q, s),
        (qd, sd),
    )
    assert_allclose(posed, expected_posed, rtol=1e-6, atol=1e-7)

    _, Jd = jvp(
        lambda q_, s_: robot.jacobian(q_, s_),
        (q, s),
        (qd, sd),
    )
    _, expected_Jd = jvp(
        lambda q_, s_: robot._jacobian(q_, s_),
        (q, s),
        (qd, sd),
    )
    assert_allclose(Jd, expected_Jd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inertia_matrix_matches_kinetic_energy_autodiff(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(2)
    q_keys = jax.random.split(key, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key + 1, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        B_impl = robot.inertia_matrix(q)

        def T_of_qd(qd_, q_current=q):
            return robot.kinetic_energy(q_current, qd_)

        dT_dqdsq = jacfwd(jax.grad(T_of_qd))(qd)
        B_expected = dT_dqdsq

        assert_allclose(B_impl, B_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_with_christoffel_symbols(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(9)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        C_impl = robot.coriolis_matrix(q, qd)
        tau_cor_impl = C_impl @ qd

        def B_of_q(q_):
            return robot.inertia_matrix(q_)

        dB_dq = jacfwd(B_of_q)(q)

        term1 = jnp.einsum("ijk,j,k->i", dB_dq, qd, qd)
        term2 = jnp.einsum("jki,j,k->i", dB_dq, qd, qd)
        tau_cor = term1 - 0.5 * term2

        assert_allclose(tau_cor_impl, tau_cor, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_coriolis_force_matches_kinetic_energy_autograd(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(10)
    key_q, key_qd = jax.random.split(key)
    q_keys = jax.random.split(key_q, NUM_RANDOM_SAMPLES)
    qd_keys = jax.random.split(key_qd, NUM_RANDOM_SAMPLES)

    dT_dq = jax.grad(robot.kinetic_energy, argnums=0)
    dT_dqd = jax.grad(robot.kinetic_energy, argnums=1)

    for q_key, qd_key in zip(q_keys, qd_keys):
        q = random_q(robot, q_key, scale=0.05)
        qd = random_q(robot, qd_key, scale=0.2)

        tau_cor_impl = robot.coriolis_matrix(q, qd) @ qd

        grad_T_q = dT_dq(q, qd)
        jac_T_q = jacfwd(lambda qq, qd_current=qd: dT_dqd(qq, qd_current))(q)
        tau_cor_autograd = jac_T_q @ qd - grad_T_q

        assert_allclose(tau_cor_impl, tau_cor_autograd, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_gravity_matches_potential_gradient(num_segments: int):
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(8)

    for q_key in jax.random.split(key, NUM_RANDOM_SAMPLES):
        q = random_q(robot, q_key, scale=0.05)
        G = robot.gravitational_force(q)
        dU_G_dq = jacfwd(robot.gravitational_energy)(q)
        print("q:\n", q)
        print("G:\n", G)
        print("dU_G_dq:\n", dU_G_dq)

        assert_allclose(G, dU_G_dq, rtol=RTOL, atol=ATOL)


def test_gvs_gravitational_energy_gradient_matches_force() -> None:
    robot = build_varied_basis_gvs(num_segments=1)
    q = random_q(robot, jax.random.PRNGKey(313), scale=0.02)

    dU_dq = jax.grad(lambda q_: robot.gravitational_energy(q_))(q)

    assert_allclose(dU_dq, robot.gravitational_force(q), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_dynamics_matches_manual_computation(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key = jax.random.PRNGKey(123 + num_segments)

    for _ in range(NUM_RANDOM_SAMPLES):
        key, key_q = jax.random.split(key)
        q = random_q(robot, key_q, scale=0.05)

        key, key_qd = jax.random.split(key)
        qd = random_q(robot, key_qd, scale=0.05)

        key, key_u = jax.random.split(key)
        u = random_q(robot, key_u, scale=0.02)

        key, key_tau = jax.random.split(key)
        tau_ext = random_q(robot, key_tau, scale=0.03)

        y = jnp.concatenate([q, qd])
        yd = robot.forward_dynamics(0.0, y, (u, tau_ext))

        B = robot.inertia_matrix(q)
        C = robot.coriolis_matrix(q, qd)
        G = robot.gravitational_force(q)
        D = robot.damping_matrix(q)
        tau_el = robot.elastic_force(q)
        tau_u = robot.actuation_force(q, u)

        qdd_expected = jnp.linalg.solve(
            B, tau_u + tau_ext - C @ qd - G - tau_el - D @ qd
        )
        yd_expected = jnp.concatenate([qd, qdd_expected])

        assert_allclose(yd, yd_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 3])
def test_integration_kinematics_matches_existing_batched_path(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(6123))
    q = random_q(robot, key_q, scale=0.05)
    qd = random_q(robot, key_qd, scale=0.1)

    g_quads, J_quads, Jd_quads = robot.integration_kinematics(q, qd)
    weights = robot._inner_quadrature_weights()

    assert weights.shape[0] == num_segments
    assert g_quads.shape[:2] == weights.shape
    assert J_quads.shape[:2] == weights.shape
    assert Jd_quads.shape == J_quads.shape


@pytest.mark.parametrize("num_segments", [1, 3])
def test_dynamics_terms_match_public_matrices(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    dof = int(robot.num_dofs)

    zero_q = jnp.zeros((dof,), dtype=jnp.float64)
    zero_qd = jnp.zeros((dof,), dtype=jnp.float64)
    key_q, key_qd = jax.random.split(jax.random.PRNGKey(6124))
    q_random = random_q(robot, key_q, scale=0.05)
    qd_random = random_q(robot, key_qd, scale=0.1)

    for q, qd in ((zero_q, zero_qd), (q_random, qd_random)):
        B, Cqd, G = robot.dynamics_terms(q, qd)
        C = robot.coriolis_matrix(q, qd)

        assert_allclose(B, robot.inertia_matrix(q), rtol=RTOL, atol=ATOL)
        assert_allclose(Cqd, C @ qd, rtol=RTOL, atol=ATOL)
        assert_allclose(G, robot.gravitational_force(q), rtol=RTOL, atol=ATOL)


def test_cached_constant_matrices_refresh_after_update_params() -> None:
    selector_per_segment = (False, False, True, True, False, False)
    robot = build_constant_strain_gvs(
        num_segments=2,
        selector_per_segment=selector_per_segment,
        max_dof=6,
    )

    updated = robot.update_params(
        link=robot.params.link.replace(
            young_modulus=1.25e6 * jnp.ones_like(robot.segment_length),
            poisson_ratio=0.45 * jnp.ones_like(robot.segment_length),
            density=900.0 * jnp.ones_like(robot.segment_length),
            damping_coefficient=2.0 * jnp.ones_like(robot.segment_length),
            radius_initial=0.022 * jnp.ones_like(robot.segment_length),
            radius_final=0.022 * jnp.ones_like(robot.segment_length),
        )
    )

    assert (
        updated.structure.segments[0].link.cross_section_geometry
        == robot.structure.segments[0].link.cross_section_geometry
    )
    assert not hasattr(updated.structure.segments[0].link, "E")
    assert not hasattr(updated.structure.segments[0].link, "r_i")
    assert updated.K_full.shape == robot.K_full.shape
    assert updated.D_full.shape == robot.D_full.shape
    assert_allclose(
        updated.active_dof_map_blocks,
        updated.active_dof_map.reshape(
            updated.num_segments, 2, updated.max_dof, updated.num_dofs
        ),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        updated.inner_integration_weights,
        updated.integration_weights[:, 1 : updated.max_num_integration_points - 1]
        * updated.segment_length[:, None],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        updated.inner_mass_matrices,
        updated.mass_matrices[:, 1 : updated.max_num_integration_points - 1],
        rtol=RTOL,
        atol=ATOL,
    )
    assert not jnp.isnan(updated.K_full).any()
    assert not jnp.isnan(updated.D_full).any()


def test_active_dof_map_creation_matches_joint_and_link_dofs() -> None:
    robot = build_varied_basis_gvs(num_segments=3)

    active_indices: list[int] = []
    row = 0
    for dof_joint, dof_link in onp.asarray(robot.dofs_per_segment, dtype=int):
        active_indices.extend(range(row, row + dof_joint))
        row += robot.max_dof
        active_indices.extend(range(row, row + dof_link))
        row += robot.max_dof

    expected_map = expected_selection_basis(
        int(robot.num_padded_dofs), tuple(active_indices)
    )
    expected_blocks = expected_map.reshape(
        robot.num_segments, 2, robot.max_dof, robot.num_dofs
    )

    assert robot.active_dof_map.shape == (robot.num_padded_dofs, robot.num_dofs)
    assert int(robot.num_dofs) == int(jnp.sum(robot.dofs_per_segment))
    assert_allclose(robot.active_dof_map, expected_map, rtol=0.0, atol=0.0)
    assert_allclose(robot.active_dof_map_blocks, expected_blocks, rtol=0.0, atol=0.0)
    assert_allclose(
        robot.active_dof_map.T @ robot.active_dof_map,
        jnp.eye(robot.num_dofs),
        rtol=0.0,
        atol=0.0,
    )


def test_rotational_strain_basis_length_scaling_matches_unscaled_coordinates() -> None:
    length = 0.37
    unscaled = build_constant_strain_gvs(
        num_segments=1,
        max_dof=6,
        segment_length=length,
        scale_rotational_basis_by_length=False,
    )
    scaled = build_constant_strain_gvs(
        num_segments=1,
        max_dof=6,
        segment_length=length,
        scale_rotational_basis_by_length=True,
    )

    B_Z1 = jnp.arange(1.0, 37.0, dtype=jnp.float64).reshape(6, 6)
    B_Z2 = B_Z1 + 50.0
    B_Z1_scaled, B_Z2_scaled = scaled._scaled_link_basis_pair(length, B_Z1, B_Z2)
    B_Z1_unscaled, B_Z2_unscaled = unscaled._scaled_link_basis_pair(length, B_Z1, B_Z2)

    assert_allclose(B_Z1_unscaled, B_Z1, rtol=0.0, atol=0.0)
    assert_allclose(B_Z2_unscaled, B_Z2, rtol=0.0, atol=0.0)
    assert_allclose(B_Z1_scaled[:3], B_Z1[:3] / length, rtol=RTOL, atol=ATOL)
    assert_allclose(B_Z2_scaled[:3], B_Z2[:3] / length, rtol=RTOL, atol=ATOL)
    assert_allclose(B_Z1_scaled[3:], B_Z1[3:], rtol=0.0, atol=0.0)
    assert_allclose(B_Z2_scaled[3:], B_Z2[3:], rtol=0.0, atol=0.0)

    q_scaled = jnp.array([0.08, -0.04, 0.03, 0.015, -0.01, 0.02])
    qd_scaled = jnp.array([-0.03, 0.05, 0.02, 0.04, -0.02, 0.01])
    coordinate_map = jnp.diag(jnp.array([1 / length] * 3 + [1.0] * 3))
    q_unscaled = coordinate_map @ q_scaled
    qd_unscaled = coordinate_map @ qd_scaled

    for s in sample_arc_lengths(scaled):
        g_scaled = scaled.forward_kinematics(q_scaled, float(s))
        g_unscaled = unscaled.forward_kinematics(q_unscaled, float(s))
        assert_allclose(g_scaled, g_unscaled, rtol=RTOL, atol=ATOL)

        J_scaled = scaled.jacobian_bodyframe(q_scaled, float(s))
        J_unscaled = unscaled.jacobian_bodyframe(q_unscaled, float(s))
        assert_allclose(J_scaled, J_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)

        J_scaled, Jd_scaled = scaled.jacobian_and_time_derivative_bodyframe(
            q_scaled, qd_scaled, float(s)
        )
        J_unscaled, Jd_unscaled = unscaled.jacobian_and_time_derivative_bodyframe(
            q_unscaled, qd_unscaled, float(s)
        )
        assert_allclose(J_scaled, J_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_scaled, Jd_unscaled @ coordinate_map, rtol=RTOL, atol=ATOL)

    assert_allclose(
        scaled.stiffness_matrix(),
        coordinate_map.T @ unscaled.stiffness_matrix() @ coordinate_map,
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        scaled.damping_matrix(q_scaled),
        coordinate_map.T @ unscaled.damping_matrix(q_unscaled) @ coordinate_map,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("num_segments", [1, 2])
def test_strain_basis_consistency_kinematics_jacobians_and_dynamics(
    num_segments: int,
) -> None:
    selector_per_segment = (False, False, True, True, False, False)
    full, reduced, B = build_full_and_reduced_constant_strain_gvs(
        num_segments, selector_per_segment
    )

    key = jax.random.PRNGKey(202)
    key_q, key_qd, key_u = jax.random.split(key, 3)
    q_small = random_q(reduced, key_q, scale=0.05)
    qd_small = random_q(reduced, key_qd, scale=0.1)
    u_small = random_q(reduced, key_u, scale=0.2)
    q_full = B @ q_small
    qd_full = B @ qd_small
    u_full = B @ u_small

    s_points = sample_arc_lengths(full)
    for s in s_points:
        g_small = reduced.forward_kinematics(q_small, float(s))
        g_full = full.forward_kinematics(q_full, float(s))
        assert_allclose(g_full, g_small, rtol=RTOL, atol=ATOL)

        Jb_small = reduced.jacobian_bodyframe(q_small, float(s))
        Jb_full = full.jacobian_bodyframe(q_full, float(s))
        assert_allclose(Jb_full @ B, Jb_small, rtol=RTOL, atol=ATOL)

        Ji_small = reduced.jacobian_inertialframe(q_small, float(s))
        Ji_full = full.jacobian_inertialframe(q_full, float(s))
        assert_allclose(Ji_full @ B, Ji_small, rtol=RTOL, atol=ATOL)

        J_small, Jd_small = reduced.jacobian_and_time_derivative_bodyframe(
            q_small, qd_small, float(s)
        )
        J_full, Jd_full = full.jacobian_and_time_derivative_bodyframe(
            q_full, qd_full, float(s)
        )
        assert_allclose(J_full @ B, J_small, rtol=RTOL, atol=ATOL)
        assert_allclose(Jd_full @ B, Jd_small, rtol=RTOL, atol=ATOL)

    B_full = full.inertia_matrix(q_full)
    B_small = reduced.inertia_matrix(q_small)
    assert_allclose(B.T @ B_full @ B, B_small, rtol=RTOL, atol=ATOL)

    C_full = full.coriolis_matrix(q_full, qd_full)
    C_small = reduced.coriolis_matrix(q_small, qd_small)
    assert_allclose(B.T @ C_full @ B, C_small, rtol=RTOL, atol=ATOL)

    G_full = full.gravitational_force(q_full)
    G_small = reduced.gravitational_force(q_small)
    assert_allclose(B.T @ G_full, G_small, rtol=RTOL, atol=ATOL)

    K_full = full.stiffness_matrix()
    K_small = reduced.stiffness_matrix()
    assert_allclose(B.T @ K_full @ B, K_small, rtol=RTOL, atol=ATOL)

    D_full = full.damping_matrix(q_full)
    D_small = reduced.damping_matrix(q_small)
    assert_allclose(B.T @ D_full @ B, D_small, rtol=RTOL, atol=ATOL)

    tau_u_full = full.actuation_force(q_full, u_full)
    tau_u_small = reduced.actuation_force(q_small, u_small)
    assert_allclose(B.T @ tau_u_full, tau_u_small, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_gvs_pcs_coherence(num_segments: int) -> None:
    _assert_gvs_pcs_coherence(num_segments)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_mode_automatic_differentiability_at_zero_configuration(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    dof = int(robot.num_dofs)
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((robot.num_actuators,), dtype=jnp.float64)
    s = float(robot.segment_end_positions[-1])

    dg_dq = jacfwd(robot.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacfwd(robot.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacfwd(robot.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe_dq = jacfwd(
        robot.jacobian_and_time_derivative_bodyframe, argnums=0
    )(q, qd, s)
    _, dJd_bodyframe_dqd = jacfwd(
        robot.jacobian_and_time_derivative_bodyframe, argnums=1
    )(q, qd, s)
    _, dJd_inertialframe_dq = jacfwd(
        robot.jacobian_and_time_derivative_inertialframe, argnums=0
    )(q, qd, s)
    _, dJd_inertialframe_dqd = jacfwd(
        robot.jacobian_and_time_derivative_inertialframe, argnums=1
    )(q, qd, s)
    dB_dq = jacfwd(robot.inertia_matrix)(q)
    dC_dq = jacfwd(robot.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacfwd(robot.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacfwd(robot.gravitational_force)(q)
    dtau_el_dq = jacfwd(robot.elastic_force)(q)
    dtau_u_dq = jacfwd(robot.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacfwd(robot.actuation_force, argnums=1)(q, u)
    dy_dy = jacfwd(robot.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacfwd(robot.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dg_dq).any(), "Found NaN in forward-mode of forward kinematics"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), (
        "Found NaN in forward-mode of bodyframe Jacobian"
    )
    assert not jnp.isnan(dJ_inertialframe_dq).any(), (
        "Found NaN in forward-mode of inertialframe Jacobian"
    )
    assert not jnp.isnan(dJd_bodyframe_dq).any(), (
        "Found NaN in forward-mode of bodyframe Jacobian time derivative"
    )
    assert not jnp.isnan(dJd_bodyframe_dqd).any(), (
        "Found NaN in forward-mode of bodyframe Jacobian time derivative w.r.t. qd"
    )
    assert not jnp.isnan(dJd_inertialframe_dq).any(), (
        "Found NaN in forward-mode of inertialframe Jacobian time derivative w.r.t. q"
    )
    assert not jnp.isnan(dJd_inertialframe_dqd).any(), (
        "Found NaN in forward-mode of inertialframe Jacobian time derivative w.r.t. qd"
    )
    assert not jnp.isnan(dB_dq).any(), "Found NaN in forward-mode of inertia matrix"
    assert not jnp.isnan(dC_dq).any(), (
        "Found NaN in forward-mode of coriolis matrix (dC/dq)"
    )
    assert not jnp.isnan(dC_dqd).any(), (
        "Found NaN in forward-mode of coriolis matrix (dC/dqd)"
    )
    assert not jnp.isnan(dG_dq).any(), (
        "Found NaN in forward-mode of gravitational force"
    )
    assert not jnp.isnan(dtau_el_dq).any(), "Found NaN in forward-mode of elastic force"
    assert not jnp.isnan(dtau_u_dq).any(), (
        "Found NaN in forward-mode of actuation force (dTau/du)"
    )
    assert not jnp.isnan(dtau_u_du).any(), (
        "Found NaN in forward-mode of actuation force (dTau/du)"
    )
    assert not jnp.isnan(dy_dy).any(), (
        "Found NaN in forward-mode of forward dynamics (dy/dy)"
    )
    assert not jnp.isnan(dy_du).any(), (
        "Found NaN in forward-mode of forward dynamics (dy/du)"
    )

    # test differentiability of the energy functions at zero configuration
    dT_dq = jacfwd(robot.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacfwd(robot.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacfwd(robot.gravitational_energy)(q)
    dU_dq = jacfwd(robot.potential_energy)(q)
    dE_dq = jacfwd(robot.total_energy, argnums=0)(q, qd)
    dE_dqd = jacfwd(robot.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any(), "dT/dq contains NaN!"
    assert not jnp.isnan(dT_dqd).any(), "dT/dqd contains NaN!"
    assert not jnp.isnan(dU_G_dq).any(), "dU_G/dq contains NaN!"
    assert not jnp.isnan(dU_dq).any(), "dU/dq contains NaN!"
    assert not jnp.isnan(dE_dq).any(), "dE/dq contains NaN!"
    assert not jnp.isnan(dE_dqd).any(), "dE/dqd contains NaN!"


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_reverse_mode_automatic_differentiability_at_zero_configuration(
    num_segments: int,
) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)
    dof = int(robot.num_dofs)
    q = jnp.zeros((dof,), dtype=jnp.float64)
    qd = jnp.zeros((dof,), dtype=jnp.float64)
    y = jnp.concatenate([q, qd])
    u = jnp.zeros((robot.num_actuators,), dtype=jnp.float64)
    s = float(robot.segment_end_positions[-1])

    dg_dq = jacrev(robot.forward_kinematics, argnums=0)(q, s)
    dJ_bodyframe_dq = jacrev(robot.jacobian_bodyframe, argnums=0)(q, s)
    dJ_inertialframe_dq = jacrev(robot.jacobian_inertialframe, argnums=0)(q, s)
    _, dJd_bodyframe_dq = jacrev(
        robot.jacobian_and_time_derivative_bodyframe, argnums=0
    )(q, qd, s)
    _, dJd_bodyframe_dqd = jacrev(
        robot.jacobian_and_time_derivative_bodyframe, argnums=1
    )(q, qd, s)
    _, dJd_inertialframe_dq = jacrev(
        robot.jacobian_and_time_derivative_inertialframe, argnums=0
    )(q, qd, s)
    _, dJd_inertialframe_dqd = jacrev(
        robot.jacobian_and_time_derivative_inertialframe, argnums=1
    )(q, qd, s)
    dB_dq = jacrev(robot.inertia_matrix)(q)
    dC_dq = jacrev(robot.coriolis_matrix, argnums=0)(q, qd)
    dC_dqd = jacrev(robot.coriolis_matrix, argnums=1)(q, qd)
    dG_dq = jacrev(robot.gravitational_force)(q)
    dtau_el_dq = jacrev(robot.elastic_force)(q)
    dtau_u_dq = jacrev(robot.actuation_force, argnums=0)(q, u)
    dtau_u_du = jacrev(robot.actuation_force, argnums=1)(q, u)
    dy_dy = jacrev(robot.forward_dynamics, argnums=1)(0.0, y, (u,))
    (dy_du,) = jacrev(robot.forward_dynamics, argnums=2)(0.0, y, (u,))

    assert not jnp.isnan(dg_dq).any(), "Found NaN in reverse-mode of forward kinematics"
    assert not jnp.isnan(dJ_bodyframe_dq).any(), (
        "Found NaN in reverse-mode of bodyframe Jacobian"
    )
    assert not jnp.isnan(dJ_inertialframe_dq).any(), (
        "Found NaN in reverse-mode of inertialframe Jacobian"
    )
    assert not jnp.isnan(dJd_bodyframe_dq).any(), (
        "Found NaN in reverse-mode of bodyframe Jacobian time derivative"
    )
    assert not jnp.isnan(dJd_bodyframe_dqd).any(), (
        "Found NaN in reverse-mode of bodyframe Jacobian time derivative w.r.t. qd"
    )
    assert not jnp.isnan(dJd_inertialframe_dq).any(), (
        "Found NaN in reverse-mode of inertialframe Jacobian time derivative"
    )
    assert not jnp.isnan(dJd_inertialframe_dqd).any(), (
        "Found NaN in reverse-mode of inertialframe Jacobian time derivative w.r.t. qd"
    )
    assert not jnp.isnan(dB_dq).any(), "Found NaN in reverse-mode of inertia matrix"
    assert not jnp.isnan(dC_dq).any(), (
        "Found NaN in reverse-mode of coriolis matrix (dC/dq)"
    )
    assert not jnp.isnan(dC_dqd).any(), (
        "Found NaN in reverse-mode of coriolis matrix (dC/dqd)"
    )
    assert not jnp.isnan(dG_dq).any(), (
        "Found NaN in reverse-mode of gravitational force"
    )
    assert not jnp.isnan(dtau_el_dq).any(), "Found NaN in reverse-mode of elastic force"
    assert not jnp.isnan(dtau_u_dq).any(), (
        "Found NaN in reverse-mode of actuation force (dTau/du)"
    )
    assert not jnp.isnan(dtau_u_du).any(), (
        "Found NaN in reverse-mode of actuation force (dTau/du)"
    )
    assert not jnp.isnan(dy_dy).any(), (
        "Found NaN in reverse-mode of forward dynamics (dy/dy)"
    )
    assert not jnp.isnan(dy_du).any(), (
        "Found NaN in reverse-mode of forward dynamics (dy/du)"
    )

    # test differentiability of the energy functions at zero configuration
    dT_dq = jacrev(robot.kinetic_energy, argnums=0)(q, qd)
    dT_dqd = jacrev(robot.kinetic_energy, argnums=1)(q, qd)
    dU_G_dq = jacrev(robot.gravitational_energy)(q)
    dU_dq = jacrev(robot.potential_energy)(q)
    dE_dq = jacrev(robot.total_energy, argnums=0)(q, qd)
    dE_dqd = jacrev(robot.total_energy, argnums=1)(q, qd)

    assert not jnp.isnan(dT_dq).any(), "dT/dq contains NaN!"
    assert not jnp.isnan(dT_dqd).any(), "dT/dqd contains NaN!"
    assert not jnp.isnan(dU_G_dq).any(), "dU_G/dq contains NaN!"
    assert not jnp.isnan(dU_dq).any(), "dU/dq contains NaN!"
    assert not jnp.isnan(dE_dq).any(), "dE/dq contains NaN!"
    assert not jnp.isnan(dE_dqd).any(), "dE/dqd contains NaN!"


@pytest.mark.parametrize("num_segments", [1])
def test_gvs_autodiff_checks(num_segments: int) -> None:
    robot = build_varied_basis_gvs(num_segments=num_segments)

    n = int(robot.num_dofs)
    q = jnp.linspace(0.01, 0.01 * n, n, dtype=jnp.float64)
    qd = jnp.linspace(0.02, 0.02 * n, n, dtype=jnp.float64)
    s = jnp.sum(robot.length, dtype=jnp.float64) * 0.7

    def J_body(q_):
        return robot.jacobian_bodyframe(q_, s)

    _, Jd_dir = jax.jvp(J_body, (q,), (qd,))
    _, Jd_impl = robot.jacobian_and_time_derivative_bodyframe(q, qd, s)
    assert_allclose(Jd_impl, Jd_dir, rtol=RTOL, atol=ATOL)

    # 2) Translational block of inertial Jacobian matches jacobian of position
    def pos_fn(q_):
        return robot.forward_kinematics(q_, s)[:3, 3]

    Jpos_ad = jax.jacfwd(pos_fn)(q)  # shape (3, n)
    Ji_gvs = gvs_jacobian_inertialframe_from_body(robot, q, s)
    Jpos_impl = Ji_gvs[3:6, :]
    assert_allclose(Jpos_impl, Jpos_ad, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
