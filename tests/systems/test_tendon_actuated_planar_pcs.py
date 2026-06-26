# ruff: noqa: E402
import equinox as eqx
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from jax import Array
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    linear_tendon_routing,
    passive_tendon_params,
    planar_base_pose,
    planar_pcs_params,
    tendon_actuated_planar_pcs_params,
)

from soromox.systems import (
    BaseTendonRoutingParams,
    PassiveTendonParams,
    PlanarPCSStructure,
    TendonActuatedPlanarPCS,
)
from soromox.utils.tolerance import Tolerance


class PlanarRoutingParams(BaseTendonRoutingParams):
    offset: Array
    slope: Array
    attachment_segment_index: tuple[int, ...] = eqx.field(static=True)

    @property
    def num_tendons(self) -> int:
        return int(self.offset.shape[0])

    @property
    def attachment_segment_indices(self) -> tuple[int, ...]:
        return self.attachment_segment_index

    def validate(self) -> None:
        if self.slope.shape != self.offset.shape:
            raise ValueError("slope must match offset shape.")
        if len(self.attachment_segment_index) != self.num_tendons:
            raise ValueError("attachment_segment_index must match num_tendons.")


def scalar_planar_routing(params: PlanarRoutingParams, s: Array) -> Array:
    return params.offset + params.slope * s


def scalar_planar_routing_derivative(params: PlanarRoutingParams, s: Array) -> Array:
    return params.slope + jnp.zeros_like(s)


def vector1_planar_routing(params: PlanarRoutingParams, s: Array) -> Array:
    return jnp.array([params.offset + params.slope * s])


def vector1_planar_routing_derivative(params: PlanarRoutingParams, s: Array) -> Array:
    return jnp.array([params.slope + jnp.zeros_like(s)])


def ambiguous_planar_routing(params: PlanarRoutingParams, s: Array) -> Array:
    d = params.offset + params.slope * s
    return jnp.array([0.0, d])


def ambiguous_planar_routing_derivative(params: PlanarRoutingParams, s: Array) -> Array:
    d = params.slope + jnp.zeros_like(s)
    return jnp.array([0.0, d])


def _body_params(num_segments: int):
    rho = 1070.0 * jnp.ones((num_segments,), dtype=jnp.float64)
    segment_lengths = 1e-1 * jnp.ones((num_segments,), dtype=jnp.float64)
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e3, 1e3]], dtype=jnp.float64), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    return planar_pcs_params(
        base_pose=planar_base_pose(jnp.pi / 2),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,), dtype=jnp.float64),
        density=rho,
        gravity=jnp.array([0.0, 9.81], dtype=jnp.float64),
        young_modulus=5e3 * jnp.ones((num_segments,), dtype=jnp.float64),
        shear_modulus=1e3 * jnp.ones((num_segments,), dtype=jnp.float64),
        damping_matrix=damping_matrix,
    )


def _constant_distance_matrix(num_segments: int) -> Array:
    return 2e-2 * jnp.array([[1.0, -1.0]], dtype=jnp.float64).repeat(
        num_segments, axis=0
    )


def _constant_distance_routing(distance_matrix: Array):
    num_segments, num_tendons_per_segment = distance_matrix.shape
    return linear_tendon_routing(
        y_intercept=distance_matrix.reshape(-1),
        y_slope=jnp.zeros((num_segments * num_tendons_per_segment,), dtype=jnp.float64),
        z_intercept=jnp.zeros(
            (num_segments * num_tendons_per_segment,), dtype=jnp.float64
        ),
        z_slope=jnp.zeros((num_segments * num_tendons_per_segment,), dtype=jnp.float64),
        attachment_segment_index=jnp.repeat(
            jnp.arange(num_segments, dtype=jnp.int32), num_tendons_per_segment
        ),
    )


def _build_planar_robot(
    num_segments: int = 3,
    *,
    active_tendon_routing=None,
    active_tendon_routing_basis=None,
    passive_tendon_routing=None,
    passive_tendon: PassiveTendonParams | None = None,
) -> TendonActuatedPlanarPCS:
    body = _body_params(num_segments)
    if active_tendon_routing is None:
        active_tendon_routing = _constant_distance_routing(
            _constant_distance_matrix(num_segments)
        )
    params = tendon_actuated_planar_pcs_params(
        body=body,
        active_tendon_routing=active_tendon_routing,
        passive_tendon_routing=passive_tendon_routing,
        passive_tendon=passive_tendon,
    )
    return TendonActuatedPlanarPCS(
        params=params,
        structure=PlanarPCSStructure(num_gauss_points=4),
        active_tendon_routing_basis=active_tendon_routing_basis,
    )


def _closed_form_constant_tendon_length(
    robot: TendonActuatedPlanarPCS, q: Array, distance_matrix: Array
) -> Array:
    xi = robot.strain(q).reshape(robot.num_segments, 3)
    kappa = xi[:, 0:1]
    axial = xi[:, 1:2]
    shear = xi[:, 2:3]
    local_extension = axial + distance_matrix * kappa
    segment_lengths = robot.L[:, None] * jnp.sqrt(shear**2 + local_extension**2)
    return jnp.cumsum(segment_lengths, axis=0).reshape(-1)


def _closed_form_constant_actuation_matrix(
    robot: TendonActuatedPlanarPCS, q: Array, distance_matrix: Array
) -> Array:
    xi = robot.strain(q).reshape(robot.num_segments, 3)
    columns = []
    for attachment_idx in range(robot.num_segments):
        for d in distance_matrix[attachment_idx]:
            segment_blocks = []
            for segment_idx in range(robot.num_segments):
                kappa, axial_strain, shear_strain = xi[segment_idx]
                axial = axial_strain + d * kappa
                norm = jnp.sqrt(axial**2 + shear_strain**2)
                local_basis = robot.L[segment_idx] * jnp.array(
                    [d * axial / norm, axial / norm, shear_strain / norm]
                )
                segment_blocks.append(
                    jnp.where(
                        segment_idx <= attachment_idx,
                        local_basis,
                        jnp.zeros_like(local_basis),
                    )
                )
            columns.append(jnp.concatenate(segment_blocks))
    return robot.B_xi.T @ jnp.stack(columns, axis=1)


def test_constant_distance_routing_matches_closed_form():
    num_segments = 3
    distance_matrix = _constant_distance_matrix(num_segments)
    robot = _build_planar_robot(num_segments=num_segments)
    q = jnp.linspace(-0.05, 0.05, robot.num_active_strains, dtype=jnp.float64)

    assert_allclose(
        robot.active_tendon_length(q),
        _closed_form_constant_tendon_length(robot, q, distance_matrix),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        robot.actuation_matrix(q),
        _closed_form_constant_actuation_matrix(robot, q, distance_matrix),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_tendon_length_gradient_matches_actuation_matrix():
    robot = _build_planar_robot(num_segments=3)
    q = jnp.linspace(-0.05, 0.05, robot.num_active_strains, dtype=jnp.float64)

    lengths = robot.tendon_length(q)
    assert lengths.shape == (robot.num_actuators,)

    jac_lengths = jax.jacrev(robot.tendon_length)(q)
    A = robot.actuation_matrix(q)

    assert_allclose(
        jac_lengths,
        A.T,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_variable_linear_routing_length_gradient_matches_actuation_matrix():
    num_segments = 2
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.004, -0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.01, -0.008], dtype=jnp.float64),
        z_intercept=jnp.zeros((2,), dtype=jnp.float64),
        z_slope=jnp.zeros((2,), dtype=jnp.float64),
        attachment_segment_index=jnp.array([1, 1], dtype=jnp.int32),
    )
    robot = _build_planar_robot(num_segments, active_tendon_routing=routing)
    q = jnp.linspace(-0.03, 0.04, robot.num_active_strains, dtype=jnp.float64)

    assert_allclose(
        jax.jacrev(robot.active_tendon_length)(q),
        robot.actuation_matrix(q).T,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


@pytest.mark.parametrize(
    ("basis", "expected_lengths_close"),
    [
        (
            {
                "d_s": scalar_planar_routing,
                "dd_s_ds": scalar_planar_routing_derivative,
            },
            True,
        ),
        (
            {
                "d_s": vector1_planar_routing,
                "dd_s_ds": vector1_planar_routing_derivative,
            },
            True,
        ),
    ],
)
def test_planar_custom_routing_accepts_scalar_and_single_entry_returns(
    basis, expected_lengths_close
):
    del expected_lengths_close
    routing = PlanarRoutingParams(
        offset=jnp.array([0.006, -0.004], dtype=jnp.float64),
        slope=jnp.array([0.004, -0.003], dtype=jnp.float64),
        attachment_segment_index=(1, 1),
    )
    robot = _build_planar_robot(
        2,
        active_tendon_routing=routing,
        active_tendon_routing_basis=basis,
    )
    q = jnp.linspace(-0.02, 0.03, robot.num_active_strains, dtype=jnp.float64)

    assert robot.active_tendon_length(q).shape == (2,)
    assert_allclose(
        jax.jacrev(robot.active_tendon_length)(q),
        robot.actuation_matrix(q).T,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_planar_routing_rejects_ambiguous_two_entry_returns():
    routing = PlanarRoutingParams(
        offset=jnp.array([0.006], dtype=jnp.float64),
        slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=(0,),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _build_planar_robot(
            1,
            active_tendon_routing=routing,
            active_tendon_routing_basis={
                "d_s": ambiguous_planar_routing,
                "dd_s_ds": ambiguous_planar_routing_derivative,
            },
        )


def test_planar_routing_rejects_nonzero_z_component():
    routing = linear_tendon_routing(
        y_intercept=jnp.array([0.006], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_intercept=jnp.array([0.001], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([0], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="nonzero z"):
        _build_planar_robot(1, active_tendon_routing=routing)


def test_empty_passive_tendons_preserve_shapes():
    robot = _build_planar_robot(num_segments=2)
    q = jnp.zeros((robot.num_active_strains,), dtype=jnp.float64)

    assert robot.passive_tendon_length(q).shape == (0,)
    assert robot.jacobian_passive_tendon(q).shape == (0, robot.num_active_strains)


def test_passive_tendon_length_gradient_matches_jacobian():
    passive_routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005, -0.006], dtype=jnp.float64),
        y_slope=jnp.array([0.002, -0.001], dtype=jnp.float64),
        z_intercept=jnp.zeros((2,), dtype=jnp.float64),
        z_slope=jnp.zeros((2,), dtype=jnp.float64),
        attachment_segment_index=jnp.array([1, 1], dtype=jnp.int32),
    )
    passive_phys = passive_tendon_params(
        stiffness=jnp.array([40.0, 60.0], dtype=jnp.float64),
        damping=jnp.array([0.4, 0.7], dtype=jnp.float64),
        rest_length_offset=jnp.array([0.001, -0.002], dtype=jnp.float64),
    )
    robot = _build_planar_robot(
        2,
        passive_tendon_routing=passive_routing,
        passive_tendon=passive_phys,
    )
    q = jnp.linspace(-0.02, 0.025, robot.num_active_strains, dtype=jnp.float64)

    assert_allclose(
        jax.jacrev(robot.passive_tendon_length)(q),
        robot.jacobian_passive_tendon(q),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


def test_passive_tendons_contribute_to_dynamics_terms():
    passive_routing = linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.001], dtype=jnp.float64),
        z_intercept=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([1], dtype=jnp.int32),
    )
    passive_phys = passive_tendon_params(
        stiffness=jnp.array([50.0], dtype=jnp.float64),
        damping=jnp.array([0.6], dtype=jnp.float64),
        rest_length_offset=jnp.array([-0.001], dtype=jnp.float64),
    )
    robot_passive = _build_planar_robot(
        2,
        passive_tendon_routing=passive_routing,
        passive_tendon=passive_phys,
    )
    robot_body = _build_planar_robot(2)
    q = jnp.linspace(-0.02, 0.025, robot_passive.num_active_strains, dtype=jnp.float64)

    J_pt = robot_passive.jacobian_passive_tendon(q)
    l_pt = robot_passive.passive_tendon_length(q)
    expected_elastic = J_pt.T @ robot_passive.K_pt @ (l_pt - robot_passive.l_pt0)
    expected_damping = J_pt.T @ robot_passive.D_pt @ J_pt
    delta_l = l_pt - robot_passive.l_pt0
    expected_energy = 0.5 * delta_l.T @ robot_passive.K_pt @ delta_l

    assert_allclose(
        robot_passive.elastic_force(q) - robot_body.elastic_force(q),
        expected_elastic,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        robot_passive.damping_matrix(q) - robot_body.damping_matrix(q),
        expected_damping,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        robot_passive.elastic_energy(q) - robot_body.elastic_energy(q),
        expected_energy,
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
