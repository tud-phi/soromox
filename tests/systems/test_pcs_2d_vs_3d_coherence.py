import jax
from jax import Array, numpy as jnp, random
from numpy.testing import assert_allclose
import pytest
from typing import List, Tuple

from soromox.systems import PCS, PCSStructure, PlanarPCS, PlanarPCSStructure
from soromox.utils.tolerance import Tolerance
from system_param_builders import (
    pcs_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
)


jax.config.update("jax_enable_x64", True)


RTOL = Tolerance.rtol()
ATOL = Tolerance.atol()

TOTAL_LENGTH = 2e-1
NUM_RANDOM_SAMPLES = 5


def make_planar_model(
    num_segments: int, total_length: float = TOTAL_LENGTH
) -> PlanarPCS:
    segment_length = total_length / num_segments
    L = segment_length * jnp.ones((num_segments,))
    diag_entries = (
        jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0)
        * L[:, None]
    ).flatten()
    params = planar_pcs_params(
        base_pose=planar_base_pose(),
        length=L,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=1070 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, -9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=1e-3 * jnp.diag(diag_entries),
    )

    return PlanarPCS(params=params, structure=PlanarPCSStructure(num_gauss_points=3))


def make_spatial_model(num_segments: int, total_length: float = TOTAL_LENGTH) -> PCS:
    segment_length = total_length / num_segments
    L = segment_length * jnp.ones((num_segments,))
    diag_entries = (
        jnp.repeat(jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0)
        * L[:, None]
    ).flatten()
    params = pcs_params(
        base_pose=spatial_base_pose(),
        length=L,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=1070 * jnp.ones((num_segments,)),
        gravity=jnp.array([0.0, -9.81, 0.0]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=1e-3 * jnp.diag(diag_entries),
    )

    return PCS(params=params, structure=PCSStructure(num_gauss_points=3))


def planar_arc_lengths(model: PlanarPCS) -> List[float]:
    lengths = jnp.asarray(model.L)
    cumulative = jnp.cumsum(lengths)
    total = float(cumulative[-1])

    near_zero = max(total * 1e-3, 1e-9)
    mids = (cumulative - lengths / 2.0).tolist()
    boundaries = cumulative.tolist()

    values = [near_zero] + mids + boundaries
    return sorted({float(v) for v in values if 0.0 < float(v) <= total})


def planar_to_spatial_index_pairs(num_segments: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for seg in range(num_segments):
        planar_offset = 3 * seg
        spatial_offset = 6 * seg
        pairs.extend(
            [
                (planar_offset + 0, spatial_offset + 2),  # kappa_z
                (planar_offset + 1, spatial_offset + 3),  # sigma_x
                (planar_offset + 2, spatial_offset + 4),  # sigma_y
            ]
        )
    return pairs


def lift_planar_configuration(planar: PlanarPCS, pcs: PCS, q_planar: Array) -> Array:
    num_segments = int(planar.num_segments)
    xi_planar = planar.B_xi @ q_planar + planar.xi_ref
    xi_planar_segments = xi_planar.reshape(num_segments, 3)

    xi_spatial_segments = []
    for seg in range(num_segments):
        kappa_z, sigma_x, sigma_y = xi_planar_segments[seg]
        xi_spatial_segments.append(
            jnp.array([0.0, 0.0, kappa_z, sigma_x, sigma_y, 0.0])
        )

    xi_spatial = jnp.concatenate(xi_spatial_segments)
    return xi_spatial - pcs.xi_ref


def lift_planar_vector(planar: PlanarPCS, pcs: PCS, vec_planar: Array) -> Array:
    dim_spatial = int(pcs.num_active_strains)
    vec_spatial = jnp.zeros((dim_spatial,))
    for planar_idx, spatial_idx in planar_to_spatial_index_pairs(
        int(planar.num_segments)
    ):
        vec_spatial = vec_spatial.at[spatial_idx].set(vec_planar[planar_idx])
    return vec_spatial


def lift_planar_velocity(planar: PlanarPCS, pcs: PCS, qd_planar: Array) -> Array:
    return lift_planar_vector(planar, pcs, qd_planar)


def spatial_planar_indices(num_segments: int) -> jnp.ndarray:
    return jnp.array(
        [pcs_idx for _, pcs_idx in planar_to_spatial_index_pairs(num_segments)]
    )


def se3_to_planar_pose(g: Array) -> Array:
    theta = jnp.arctan2(g[1, 0], g[0, 0])
    x = g[0, 3]
    y = g[1, 3]
    return jnp.array([theta, x, y])


def project_se3_matrix_to_planar(J: Array) -> Array:
    return jnp.vstack([J[2], J[3], J[4]])


def project_se3_derivative_to_planar(Jd: Array) -> Array:
    return jnp.vstack([Jd[2], Jd[3], Jd[4]])


def random_planar_state(model: PlanarPCS, key: Array, scale: float = 0.05) -> Array:
    size = int(model.num_active_strains)
    return scale * random.normal(key, (size,))


def planar_tip_poses(model: PlanarPCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]
    return jax.vmap(lambda s: model.forward_kinematics(q, s))(s_vals)


def spatial_tip_transforms(model: PCS, q: Array) -> Array:
    s_vals = model.L_cum[1:]
    return jax.vmap(lambda s: model.forward_kinematics(q, s))(s_vals)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_kinematics_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)
    arc_lengths = planar_arc_lengths(planar_model)

    key = random.PRNGKey(0)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey = random.split(key)
        q_planar = random_planar_state(planar_model, subkey)
        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)

        for s in arc_lengths:
            chi_planar = planar_model.forward_kinematics(q_planar, s)
            g_spatial = spatial_model.forward_kinematics(q_spatial, s)
            chi_spatial = se3_to_planar_pose(g_spatial)
            assert_allclose(chi_planar, chi_spatial, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_inverse_kinematics_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)

    key = random.PRNGKey(5)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey = random.split(key)
        q_planar = random_planar_state(planar_model, subkey)
        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)

        chi_tips = planar_tip_poses(planar_model, q_planar)
        g_tips = spatial_tip_transforms(spatial_model, q_spatial)

        q_planar_recovered = planar_model.inverse_kinematics(chi_tips)
        q_spatial_recovered = spatial_model.inverse_kinematics(g_tips)

        assert_allclose(q_planar_recovered, q_planar, rtol=RTOL, atol=ATOL)
        assert_allclose(q_spatial_recovered, q_spatial, rtol=RTOL, atol=ATOL)

        xi_spatial = spatial_model.B_xi @ q_spatial_recovered + spatial_model.xi_ref
        xi_planar = xi_spatial.reshape(num_segments, 6)[:, [2, 3, 4]].reshape(-1)
        xi_planar -= planar_model.xi_ref
        q_planar_from_spatial = jnp.linalg.pinv(planar_model.B_xi) @ xi_planar

        assert_allclose(q_planar_from_spatial, q_planar_recovered, rtol=RTOL, atol=ATOL)
        assert_allclose(q_planar_from_spatial, q_planar, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobians_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)
    arc_lengths = planar_arc_lengths(planar_model)
    idxs = spatial_planar_indices(num_segments)

    key = random.PRNGKey(1)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey = random.split(key)
        q_planar = random_planar_state(planar_model, subkey)
        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)

        for s in arc_lengths:
            J_planar_body = planar_model.jacobian_bodyframe(q_planar, s)
            J_planar_inertial = planar_model.jacobian_inertialframe(q_planar, s)

            J_spatial_body = project_se3_matrix_to_planar(
                spatial_model.jacobian_bodyframe(q_spatial, s)
            )[:, idxs]
            J_spatial_inertial = project_se3_matrix_to_planar(
                spatial_model.jacobian_inertialframe(q_spatial, s)
            )[:, idxs]

            assert_allclose(J_spatial_body, J_planar_body, rtol=RTOL, atol=ATOL)
            assert_allclose(J_spatial_inertial, J_planar_inertial, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_jacobian_time_derivatives_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)
    arc_lengths = planar_arc_lengths(planar_model)
    idxs = spatial_planar_indices(num_segments)

    key = random.PRNGKey(2)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey_q = random.split(key)
        subkey_q, subkey_qd = random.split(subkey_q)
        q_planar = random_planar_state(planar_model, subkey_q)
        qd_planar = random_planar_state(planar_model, subkey_qd, scale=0.1)

        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)
        qd_spatial = lift_planar_velocity(planar_model, spatial_model, qd_planar)

        for s in arc_lengths:
            Jb_planar, Jbd_planar = planar_model.jacobian_and_time_derivative_bodyframe(
                q_planar, qd_planar, s
            )
            Ji_planar, Jid_planar = planar_model.jacobian_and_time_derivative_inertialframe(
                q_planar, qd_planar, s
            )

            Jb_spatial, Jbd_spatial = spatial_model.jacobian_and_time_derivative_bodyframe(
                q_spatial, qd_spatial, s
            )
            Ji_spatial, Jid_spatial = (
                spatial_model.jacobian_and_time_derivative_inertialframe(
                    q_spatial, qd_spatial, s
                )
            )

            assert_allclose(
                project_se3_matrix_to_planar(Jb_spatial)[:, idxs],
                Jb_planar,
                rtol=RTOL,
                atol=ATOL,
            )
            assert_allclose(
                project_se3_derivative_to_planar(Jbd_spatial)[:, idxs],
                Jbd_planar,
                rtol=RTOL,
                atol=ATOL,
            )
            assert_allclose(
                project_se3_matrix_to_planar(Ji_spatial)[:, idxs],
                Ji_planar,
                rtol=RTOL,
                atol=ATOL,
            )
            assert_allclose(
                project_se3_derivative_to_planar(Jid_spatial)[:, idxs],
                Jid_planar,
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_dynamics_matrices_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)
    idxs = spatial_planar_indices(num_segments)

    key = random.PRNGKey(3)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey_q = random.split(key)
        subkey_q, subkey_qd = random.split(subkey_q)
        q_planar = random_planar_state(planar_model, subkey_q)
        qd_planar = random_planar_state(planar_model, subkey_qd, scale=0.1)

        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)
        qd_spatial = lift_planar_velocity(planar_model, spatial_model, qd_planar)

        M_planar = planar_model.inertia_matrix(q_planar)
        M_spatial = spatial_model.inertia_matrix(q_spatial)
        assert_allclose(
            M_spatial[jnp.ix_(idxs, idxs)],
            M_planar,
            rtol=RTOL,
            atol=ATOL,
        )

        C_planar = planar_model.coriolis_matrix(q_planar, qd_planar)
        C_spatial = spatial_model.coriolis_matrix(q_spatial, qd_spatial)
        assert_allclose(
            C_spatial[jnp.ix_(idxs, idxs)],
            C_planar,
            rtol=1e-3,
            atol=3e-4,
        )

        tau_cor_planar = C_planar @ qd_planar
        tau_cor_spatial = C_spatial @ qd_spatial
        assert_allclose(tau_cor_spatial[idxs], tau_cor_planar, rtol=RTOL, atol=ATOL)

        G_planar = planar_model.gravitational_force(q_planar)
        G_spatial = spatial_model.gravitational_force(q_spatial)
        assert_allclose(G_spatial[idxs], G_planar, rtol=RTOL, atol=ATOL)

        K_planar = planar_model.stiffness_matrix()
        K_spatial = spatial_model.stiffness_matrix()
        assert_allclose(
            K_spatial[jnp.ix_(idxs, idxs)],
            K_planar,
            rtol=RTOL,
            atol=ATOL,
        )

        tau_el_planar = planar_model.elastic_force(q_planar)
        tau_el_spatial = spatial_model.elastic_force(q_spatial)
        assert_allclose(tau_el_spatial[idxs], tau_el_planar, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2])
def test_energy_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)

    key = random.PRNGKey(4)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, subkey_q = random.split(key)
        subkey_q, subkey_qd = random.split(subkey_q)
        q_planar = random_planar_state(planar_model, subkey_q)
        qd_planar = random_planar_state(planar_model, subkey_qd, scale=0.1)

        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)
        qd_spatial = lift_planar_velocity(planar_model, spatial_model, qd_planar)

        T_planar = planar_model.kinetic_energy(q_planar, qd_planar)
        T_spatial = spatial_model.kinetic_energy(q_spatial, qd_spatial)
        assert_allclose(T_spatial, T_planar, rtol=RTOL, atol=ATOL)

        U_planar = planar_model.potential_energy(q_planar)
        U_spatial = spatial_model.potential_energy(q_spatial)
        assert_allclose(U_spatial, U_planar, rtol=RTOL, atol=ATOL)

        E_planar = planar_model.total_energy(q_planar, qd_planar)
        E_spatial = spatial_model.total_energy(q_spatial, qd_spatial)
        assert_allclose(E_spatial, E_planar, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_forward_dynamics_coherence(num_segments):
    planar_model = make_planar_model(num_segments)
    spatial_model = make_spatial_model(num_segments)

    indices = spatial_planar_indices(num_segments)
    dim_planar = int(planar_model.num_active_strains)

    key = random.PRNGKey(21)
    for _ in range(NUM_RANDOM_SAMPLES):
        key, key_q, key_qd, key_u, key_tau = random.split(key, 5)

        q_planar = random_planar_state(planar_model, key_q)
        qd_planar = random_planar_state(planar_model, key_qd)
        u_planar = random_planar_state(planar_model, key_u)
        tau_planar = random_planar_state(planar_model, key_tau)

        q_spatial = lift_planar_configuration(planar_model, spatial_model, q_planar)
        qd_spatial = lift_planar_velocity(planar_model, spatial_model, qd_planar)
        u_spatial = lift_planar_vector(planar_model, spatial_model, u_planar)
        tau_spatial = lift_planar_vector(planar_model, spatial_model, tau_planar)

        y_planar = jnp.concatenate([q_planar, qd_planar])
        y_spatial = jnp.concatenate([q_spatial, qd_spatial])

        yd_planar = planar_model.forward_dynamics(0.0, y_planar, (u_planar, tau_planar))
        yd_spatial = spatial_model.forward_dynamics(
            0.0, y_spatial, (u_spatial, tau_spatial)
        )

        qd_planar_next = yd_planar[:dim_planar]
        qdd_planar = yd_planar[dim_planar:]

        dim_spatial = int(spatial_model.num_active_strains)
        qd_spatial_next = yd_spatial[:dim_spatial]
        qdd_spatial = yd_spatial[dim_spatial:]

        assert_allclose(qd_spatial_next[indices], qd_planar_next, rtol=RTOL, atol=ATOL)
        assert_allclose(qdd_spatial[indices], qdd_planar, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    # run pytest with activated stdout
    pytest.main([__file__])
