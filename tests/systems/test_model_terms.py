# ruff: noqa: E402
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from system_param_builders import articulated_params, pendulum_params, planar_base_pose

from soromox.actuation import ThreadlikeActuator, ThreadlikeFriction, ThreadlikeRouting
from soromox.systems import (
    GVS,
    PCS,
    ArticulatedSoftRobot,
    GVSSegment,
    JointSpec,
    LinkSpec,
    McKibbenActuatedUMArm,
    Pendulum,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    PlanarPCS,
    StrainBasisSpec,
    spatial_inertia,
)

from .test_helicoidal import build as build_helicoid
from .test_helicoidal import segment as helicoid_segment

ROBOT_PARAMETERS = Path(__file__).resolve().parents[2] / "assets" / "robot_parameters"
SPATIAL_BASE_POSE = jnp.array([0.0, jnp.sin(0.2), 0.0, jnp.cos(0.2), 0.1, -0.2, 0.3])


def _assert_close(actual, desired, rtol=1e-10):
    """Compare with a tolerance relative to the largest entry of ``desired``."""
    assert_allclose(
        actual, desired, rtol=rtol, atol=rtol * jnp.max(jnp.abs(desired), initial=0.0)
    )


def _link(reference_strain: list[float]) -> LinkSpec:
    return LinkSpec.circular(
        length=0.1,
        radius=0.01,
        density=1000.0,
        young_modulus=1.0e5,
        shear_modulus=4.0e4,
        material_damping_coefficient=1.0e2,
        reference_strain=reference_strain,
    )


def _friction(enabled: bool) -> ThreadlikeFriction | None:
    # Capstan friction separates the actuation matrix from the transpose of the
    # actuator-coordinate Jacobian.
    return ThreadlikeFriction.capstan(coefficient=0.8) if enabled else None


def _planar_pcs(
    tendons: bool = False, friction: bool = False, custom_inertia: bool = False
) -> PlanarPCS:
    # The JAX backend: with Warp dynamics the terms hold no node kinematics.
    kwargs = {}
    if tendons:
        routing = ThreadlikeRouting.linear(
            intercept=jnp.array([[0.0, 0.005, 0.0], [0.0, -0.005, 0.0]]),
            start_segment_index=0,
            end_segment_index=(1, 1),
        )
        kwargs["actuators"] = ThreadlikeActuator.tendons(
            routing, friction=_friction(friction)
        )
    robot = PlanarPCS.from_links(
        [_link([0.0, 1.0, 0.0])] * 2,
        gravity=jnp.array([0.0, -9.81]),
        base_pose=planar_base_pose(0.3, 0.1, -0.2),
        backend="jax",
        **kwargs,
    )
    if custom_inertia:
        # Custom inertia appends each segment tip to the samples of the
        # dynamics pass.
        section = spatial_inertia(0.3, jnp.array([0.0, 0.002]), 1e-5)
        tip = spatial_inertia(0.02, jnp.array([0.005, -0.003]), 1e-6)
        robot = robot.update_link_params(
            sectional_inertia=jnp.stack([section] * 2),
            tip_inertia=jnp.stack([tip] * 2),
        )
    return robot


def _pcs() -> PCS:
    return PCS.from_links(
        [_link([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])] * 2,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        base_pose=SPATIAL_BASE_POSE,
    )


def _gvs(
    tendons: bool = False, friction: bool = False, custom_inertia: bool = False
) -> GVS:
    # Segments with different strain selections and Gauss-point counts use
    # different padded layouts. The JAX backend: with Warp dynamics the terms
    # hold no node kinematics.
    segments = [
        GVSSegment(
            link=_link([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            joint=JointSpec.revolute(
                "z", stiffness=jnp.array([[0.3]]), damping=jnp.array([[0.02]])
            ),
            basis=StrainBasisSpec(
                type="legendre", strain_selector=strain_selector, basis_order=1
            ),
            num_gauss_points=num_gauss_points,
        )
        for strain_selector, num_gauss_points in (
            (("kappa_y", "sigma_x"), 7),
            (("kappa_y", "kappa_z", "sigma_x"), 5),
        )
    ]
    kwargs = {}
    if tendons:
        angles = jnp.deg2rad(jnp.array([0.0, 120.0, 240.0]))
        routing = ThreadlikeRouting.linear(
            intercept=0.008
            * jnp.stack([jnp.zeros_like(angles), jnp.sin(angles), jnp.cos(angles)], -1),
            start_segment_index=0,
            end_segment_index=(1, 1, 1),
        )
        kwargs["actuators"] = ThreadlikeActuator.tendons(
            routing, friction=_friction(friction)
        )
    robot = GVS.from_segments(
        segments, base_pose=SPATIAL_BASE_POSE, backend="jax", **kwargs
    )
    if custom_inertia:
        section = spatial_inertia(
            0.3, jnp.array([0.0, 0.002, -0.001]), 1e-5 * jnp.eye(3)
        )
        tip = spatial_inertia(
            0.02, jnp.array([0.005, -0.003, 0.002]), 1e-6 * jnp.eye(3)
        )
        robot = robot.update_link_params(
            sectional_inertia=jnp.stack([section] * 2),
            tip_inertia=jnp.stack([tip] * 2),
        )
    return robot


def _helicoid(tendons: bool = False) -> GVS:
    # The helicoid always has custom inertia, and it inherits the GVS terms.
    kwargs = {}
    if tendons:
        routing = ThreadlikeRouting.linear(
            intercept=jnp.array([[0.0, 0.015, 0.0]]),
            start_segment_index=0,
            end_segment_index=(1,),
        )
        kwargs["actuators"] = ThreadlikeActuator.tendons(routing)
    segments = [helicoid_segment(active=(0, 1, 0, 1, 0, 0), gauss=5) for _ in range(2)]
    return build_helicoid(segments, backend="jax", **kwargs)


def _pendulum(floating_base: bool = False) -> Pendulum:
    params = pendulum_params(
        mass=jnp.array([1.0, 2.0]),
        moment_inertia=jnp.array([0.1, 0.2]),
        length=jnp.array([1.0, 1.5]),
        center_of_mass_length=jnp.array([0.5, 0.75]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.eye(2) * 10.0,
        joint_damping=jnp.eye(2) * 0.5,
    )
    return Pendulum(
        params, base_pose=planar_base_pose(0.3, 0.1, -0.2), floating_base=floating_base
    )


def _articulated_soft_robot() -> ArticulatedSoftRobot:
    tip_position = jnp.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    params = articulated_params(
        joint_screw=jnp.array(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        tip_position=tip_position,
        center_of_mass_position=0.5 * tip_position,
        mass=jnp.array([1.0, 2.0]),
        center_of_mass_inertia=jnp.stack(
            [
                jnp.diag(jnp.array([0.01, 0.02, 0.1])),
                jnp.diag(jnp.array([0.02, 0.03, 0.2])),
            ]
        ),
        gravity=jnp.array([0.0, -9.81, 0.0]),
        joint_stiffness=jnp.eye(2) * 10.0,
        joint_damping=jnp.eye(2) * 0.5,
    )
    return ArticulatedSoftRobot(params, base_pose=SPATIAL_BASE_POSE)


def _mckibben_umarm() -> McKibbenActuatedUMArm:
    return McKibbenActuatedUMArm.from_cached_parameters(
        ROBOT_PARAMETERS / "mckibben_umarm" / "reference_parameters.npz"
    )


def _planar_hsa() -> PlanarHSA:
    params = PlanarHSAParams.from_npz(
        ROBOT_PARAMETERS / "planar_hsa" / "fpu_control.npz"
    )
    num_strains = 3 * params.length.shape[0]
    params = params.replace(
        hysteresis_basis=jnp.ones((num_strains, 1)),
        hysteresis_alpha=0.5 * jnp.ones((num_strains,)),
        hysteresis_A=jnp.ones((1,)),
        hysteresis_n=jnp.ones((1,)),
        hysteresis_beta=0.5 * jnp.ones((1,)),
        hysteresis_gamma=0.5 * jnp.ones((1,)),
    )
    return PlanarHSA(
        params=params, structure=PlanarHSAStructure(consider_hysteresis=True)
    )


NODE_KINEMATICS_SYSTEMS = [
    pytest.param(_planar_pcs, id="planar-pcs"),
    pytest.param(_gvs, id="gvs"),
    pytest.param(
        lambda: _planar_pcs(custom_inertia=True), id="planar-pcs-custom-inertia"
    ),
    pytest.param(lambda: _gvs(custom_inertia=True), id="gvs-custom-inertia"),
    pytest.param(_helicoid, id="helicoid"),
    pytest.param(lambda: _helicoid(tendons=True), id="helicoid-tendons"),
]
SYSTEMS = [
    *NODE_KINEMATICS_SYSTEMS,
    pytest.param(
        lambda: _planar_pcs(tendons=True, friction=True), id="planar-pcs-capstan"
    ),
    pytest.param(lambda: _gvs(tendons=True, friction=True), id="gvs-capstan"),
    pytest.param(_pcs, id="pcs"),
    pytest.param(_pendulum, id="pendulum"),
    pytest.param(_articulated_soft_robot, id="articulated"),
    pytest.param(_mckibben_umarm, id="mckibben-umarm"),
    pytest.param(_planar_hsa, id="planar-hsa-hysteresis"),
]


def _state(robot) -> jax.Array:
    y = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (robot.state_size,))
    return robot.project_state(y)


@pytest.mark.parametrize("make_robot", SYSTEMS)
def test_model_terms_match_system_methods(make_robot):
    robot = make_robot()
    t = jnp.array(0.0)
    y = _state(robot)
    q, qd, _ = robot.split_state(y)

    terms = robot.model_terms(t, y)

    inertia, coriolis_force, gravitational_force = robot.dynamics_terms(q, qd)
    _assert_close(terms.inertia_matrix, inertia)
    _assert_close(terms.coriolis_force, coriolis_force)
    _assert_close(terms.gravitational_force, gravitational_force)
    _assert_close(terms.damping_matrix, robot.damping_matrix(q))
    _assert_close(terms.actuation_matrix, robot.actuation_matrix(q))
    rhs = jax.random.normal(jax.random.PRNGKey(2), (q.shape[0], 2))
    _assert_close(terms.solve_inertia(rhs[:, 0]), jnp.linalg.solve(inertia, rhs[:, 0]))
    _assert_close(terms.solve_inertia(rhs), jnp.linalg.solve(inertia, rhs))
    # The terms take no auxiliary state, so for the HSA their elastic force
    # excludes the hysteresis coordinates.
    _assert_close(terms.elastic_force, robot.elastic_force(q))


class _ArmaturePlanarPCS(PlanarPCS):
    """PlanarPCS whose inertia includes an armature on every coordinate."""

    def dynamics_terms(self, q, qd, *, backend=None):
        inertia, coriolis_force, gravitational_force = super().dynamics_terms(
            q, qd, backend=backend
        )
        armature = 0.05 * jnp.eye(inertia.shape[0], dtype=inertia.dtype)
        return inertia + armature, coriolis_force, gravitational_force


def test_model_terms_follow_a_subclass_that_overrides_the_dynamics():
    # The dynamics pass of PlanarPCS would bypass the override, so the terms
    # come from the subclass's dynamics_terms, without node kinematics.
    robot = _ArmaturePlanarPCS.from_links(
        [_link([0.0, 1.0, 0.0])] * 2, gravity=jnp.array([0.0, -9.81]), backend="jax"
    )
    t = jnp.array(0.0)
    y = _state(robot)
    q, qd, _ = robot.split_state(y)

    terms = robot.model_terms(t, y)

    _assert_close(terms.inertia_matrix, robot.dynamics_terms(q, qd)[0])
    assert terms.node_jacobians is None
    assert terms.node_poses is None


@pytest.mark.parametrize("make_robot", NODE_KINEMATICS_SYSTEMS)
def test_task_kinematics_from_model_terms(make_robot):
    robot = make_robot()
    y = _state(robot)
    q, _, _ = robot.split_state(y)
    # The base, inside the first segment, the segment end, inside the second
    # segment, and the tip.
    s_ps = jnp.array([0.0, 0.03, 0.1, 0.13, 0.2])

    terms = robot.model_terms(jnp.array(0.0), y)

    assert terms.node_jacobians is not None
    assert terms.node_poses is not None
    # One pose per node Jacobian. The reads below stop before the last node, so
    # they alone would miss a node counted twice.
    node_axes = terms.node_jacobians.ndim - 2
    assert terms.node_poses.shape[:node_axes] == terms.node_jacobians.shape[:node_axes]
    _assert_close(
        robot.forward_kinematics_abscissa_batched(q, s_ps, model_terms=terms),
        robot.forward_kinematics_abscissa_batched(q, s_ps),
    )
    _assert_close(
        robot.jacobian_bodyframe_abscissa_batched(q, s_ps, model_terms=terms),
        robot.jacobian_bodyframe_abscissa_batched(q, s_ps),
    )
    _assert_close(
        robot.jacobian_abscissa_batched(q, s_ps, model_terms=terms),
        robot.jacobian_abscissa_batched(q, s_ps),
    )


def test_floating_base_provides_no_model_terms():
    robot = _pendulum(floating_base=True)

    assert robot.model_terms(jnp.array(0.0), jnp.zeros((robot.state_size,))) is None
