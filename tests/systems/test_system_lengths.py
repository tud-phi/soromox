from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from system_param_builders import (
    articulated_params,
    linear_tendon_routing,
    pcs_params,
    pendulum_params,
    planar_base_pose,
    planar_pcs_params,
    spatial_base_pose,
    tendon_actuated_pcs_params,
    tendon_actuated_pendulum_params,
    tendon_actuated_planar_pcs_params,
)

import soromox
from soromox.systems import (
    GVS,
    PCS,
    ArticulatedSoftRobot,
    CrossSectionGeometry,
    ISupport,
    ISupportParams,
    PCSStructure,
    Pendulum,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    PlanarPCS,
    PlanarPCSStructure,
    PneumaticActuatedPlanarPCS,
    PneumaticActuatedPlanarPCSParams,
    TendonActuatedGVS,
    TendonActuatedPCS,
    TendonActuatedPendulum,
    TendonActuatedPlanarPCS,
)
from soromox.systems.gvs import GVSSegment, JointSpec, LinkSpec, StrainBasisSpec

jax.config.update("jax_enable_x64", True)


def _pcs_params(length):
    num_segments = len(length)
    return pcs_params(
        base_pose=spatial_base_pose(),
        length=jnp.asarray(length, dtype=jnp.float64),
        radius=0.02 * jnp.ones((num_segments,), dtype=jnp.float64),
        density=1000.0 * jnp.ones((num_segments,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((num_segments,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((num_segments,), dtype=jnp.float64),
        damping_matrix=jnp.eye(6 * num_segments, dtype=jnp.float64),
        gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
    )


def _planar_pcs_params(length):
    num_segments = len(length)
    return planar_pcs_params(
        base_pose=planar_base_pose(),
        length=jnp.asarray(length, dtype=jnp.float64),
        radius=0.02 * jnp.ones((num_segments,), dtype=jnp.float64),
        density=1000.0 * jnp.ones((num_segments,), dtype=jnp.float64),
        young_modulus=1e6 * jnp.ones((num_segments,), dtype=jnp.float64),
        shear_modulus=1e5 * jnp.ones((num_segments,), dtype=jnp.float64),
        damping_matrix=jnp.eye(3 * num_segments, dtype=jnp.float64),
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
    )


def _pendulum_params(length):
    length = jnp.asarray(length, dtype=jnp.float64)
    num_links = len(length)
    return pendulum_params(
        mass=jnp.arange(1, num_links + 1, dtype=jnp.float64),
        moment_inertia=0.1 * jnp.arange(1, num_links + 1, dtype=jnp.float64),
        length=length,
        center_of_mass_length=0.5 * length,
        gravity=jnp.array([0.0, -9.81], dtype=jnp.float64),
    )


def _articulated_robot():
    p_tip = jnp.array(
        [[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.5]], dtype=jnp.float64
    )
    num_links = p_tip.shape[0]
    joint_screw = jnp.zeros((num_links, 6), dtype=jnp.float64).at[:, 2].set(1.0)
    inertia = jnp.stack(
        [jnp.eye(3, dtype=jnp.float64) * (0.01 + 0.01 * i) for i in range(num_links)]
    )
    return ArticulatedSoftRobot(
        articulated_params(
            joint_screw=joint_screw,
            tip_position=p_tip,
            center_of_mass_position=0.5 * p_tip,
            mass=jnp.ones((num_links,), dtype=jnp.float64),
            center_of_mass_inertia=inertia,
            gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64),
        )
    )


def _segments():
    lengths = [0.11, 0.13, 0.17]
    return [
        GVSSegment(
            link=LinkSpec(
                cross_section_geometry=CrossSectionGeometry.CIRCULAR,
                E=1e6,
                nu=0.45,
                rho=1000.0,
                eta=0.0,
                L=length,
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
        for length in lengths
    ]


def _tendon_routing(num_segments):
    return linear_tendon_routing(
        y_intercept=jnp.array([0.005], dtype=jnp.float64),
        z_intercept=jnp.array([0.005], dtype=jnp.float64),
        y_slope=jnp.array([0.0], dtype=jnp.float64),
        z_slope=jnp.array([0.0], dtype=jnp.float64),
        attachment_segment_index=jnp.array([num_segments - 1], dtype=jnp.int32),
    )


def _pneumatic_planar_robot():
    body = _planar_pcs_params([0.08, 0.12])
    params = PneumaticActuatedPlanarPCSParams(
        **body.__dict__,
        chamber_inner_radius=jnp.array([0.002, 0.002], dtype=jnp.float64),
        chamber_outer_radius=jnp.array([0.004, 0.004], dtype=jnp.float64),
        chamber_angle=jnp.array([jnp.pi / 3.0, jnp.pi / 3.0], dtype=jnp.float64),
        chamber_distance=jnp.array([0.01, 0.01], dtype=jnp.float64),
    )
    return PneumaticActuatedPlanarPCS(
        params=params, structure=PlanarPCSStructure(num_gauss_points=1)
    )


def _isupport_robot():
    body = _pcs_params([0.08, 0.12])
    params = ISupportParams(
        **body.__dict__,
        chamber_inner_radius=jnp.array([0.002, 0.002], dtype=jnp.float64),
        chamber_outer_radius=jnp.array([0.004, 0.004], dtype=jnp.float64),
        chamber_distance=jnp.array([0.01, 0.01], dtype=jnp.float64),
        chamber_angle_offset=jnp.array([0.0, 0.0], dtype=jnp.float64),
    )
    return ISupport(params=params, structure=PCSStructure(num_gauss_points=1))


def _planar_hsa_robot():
    params = PlanarHSAParams.from_npz(
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robot_parameters"
        / "planar_hsa"
        / "fpu_control.npz"
    )
    path = (
        Path(soromox.__file__).parent
        / "symbolic_expressions"
        / "planar_hsa_ns-1_nrs-2.dill"
    )
    return PlanarHSA(
        params=params,
        structure=PlanarHSAStructure(symbolic_expression_path=str(path)),
    )


@pytest.mark.parametrize(
    "robot",
    [
        PCS(params=_pcs_params([0.08, 0.12, 0.2])),
        PlanarPCS(params=_planar_pcs_params([0.08, 0.12, 0.2])),
        Pendulum(_pendulum_params([0.7, 0.9, 1.1])),
        TendonActuatedPendulum(
            tendon_actuated_pendulum_params(
                body=_pendulum_params([0.7, 0.9, 1.1]),
                active_routing_matrix=jnp.tril(jnp.ones((3, 3), dtype=jnp.float64)),
            )
        ),
        _articulated_robot(),
        GVS.from_segments(_segments(), gravity=jnp.array([0.0, 0.0, -9.81])),
        TendonActuatedGVS.from_segments(
            _segments(),
            gravity=jnp.array([0.0, 0.0, -9.81]),
            active_tendon_routing=_tendon_routing(3),
        ),
        TendonActuatedPCS(
            params=tendon_actuated_pcs_params(
                body=_pcs_params([0.08, 0.12, 0.2]),
                active_tendon_routing=_tendon_routing(3),
            ),
            structure=PCSStructure(num_gauss_points=3),
        ),
        TendonActuatedPlanarPCS(
            params=tendon_actuated_planar_pcs_params(
                body=_planar_pcs_params([0.08, 0.12, 0.2]),
                tendon_distance=0.02
                * jnp.ones((3, 2), dtype=jnp.float64).at[:, 1].set(-1.0),
            )
        ),
        _pneumatic_planar_robot(),
        _isupport_robot(),
        _planar_hsa_robot(),
    ],
)
def test_length_returns_total_robot_length(robot):
    assert_allclose(
        robot.length,
        jnp.sum(jnp.asarray(robot.segment_length)),
        rtol=1e-12,
        atol=1e-12,
    )
