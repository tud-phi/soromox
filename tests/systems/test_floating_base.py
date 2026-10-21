"""Shared floating-base contracts across continuum-system families."""

from pathlib import Path

import jax
import pytest
from jax import jvp
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import (
    articulated_params,
    pcs_params,
    pendulum_params,
    planar_pcs_params,
)

from soromox.control.configuration_space.pid_controller import PIDController
from soromox.control.pid_control import PIDControl
from soromox.control.reference_trajectory import ReferenceTrajectory
from soromox.systems import (
    GVS,
    PCS,
    ArticulatedSoftRobot,
    GVSSegment,
    JointSpec,
    LinkSpec,
    PCSStructure,
    Pendulum,
    PlanarHSA,
    PlanarHSAParams,
    PlanarHSAStructure,
    PlanarPCS,
    PlanarPCSStructure,
    StrainBasisSpec,
)
from soromox.systems.soft_robot import SoftRobot
from soromox.systems.system_state import SystemState

jax.config.update("jax_enable_x64", True)


def _spatial_pcs() -> PCS:
    length = jnp.array([0.1])
    params = pcs_params(
        length=length,
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(6),
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return PCS(
        params=params,
        structure=PCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


def _planar_pcs() -> PlanarPCS:
    length = jnp.array([0.1])
    params = planar_pcs_params(
        length=length,
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(3),
        gravity=jnp.array([0.0, -9.81]),
    )
    return PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(num_gauss_points=3),
        floating_base=True,
        backend="jax",
    )


def _gvs() -> GVS:
    segment = GVSSegment(
        link=LinkSpec.circular(
            young_modulus=1.0e5,
            shear_modulus=4.0e4,
            density=1000.0,
            length=0.1,
            radius=0.01,
            reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        joint=JointSpec.fixed(),
        basis=StrainBasisSpec(
            type="legendre",
            strain_selector=[1, 1, 1, 1, 1, 1],
            basis_order=[0, 0, 0, 0, 0, 0],
        ),
        num_gauss_points=5,
    )
    return GVS.from_segments([segment], floating_base=True, backend="jax")


def _planar_hsa() -> PlanarHSA:
    path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robot_parameters"
        / "planar_hsa"
        / "fpu_control.npz"
    )
    return PlanarHSA(
        params=PlanarHSAParams.from_npz(path),
        structure=PlanarHSAStructure(),
        floating_base=True,
    )


def _articulated() -> ArticulatedSoftRobot:
    num_links = 2
    joint_screw = jnp.zeros((num_links, 6)).at[:, 2].set(1.0)
    tip_position = jnp.array([[0.8, 0.0, 0.1], [0.6, 0.0, 0.2]])
    center_of_mass_position = 0.5 * tip_position
    params = articulated_params(
        joint_screw=joint_screw,
        tip_position=tip_position,
        center_of_mass_position=center_of_mass_position,
        mass=jnp.array([1.0, 0.8]),
        center_of_mass_inertia=jnp.array(
            [
                jnp.diag(jnp.array([0.02, 0.03, 0.04])),
                jnp.diag(jnp.array([0.015, 0.025, 0.035])),
            ]
        ),
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return ArticulatedSoftRobot(params, floating_base=True)


def _pendulum() -> Pendulum:
    params = pendulum_params(
        mass=jnp.array([1.0, 0.8]),
        moment_inertia=jnp.array([0.1, 0.08]),
        length=jnp.array([0.8, 0.6]),
        center_of_mass_length=jnp.array([0.4, 0.3]),
        gravity=jnp.array([0.0, -9.81]),
    )
    return Pendulum(params, floating_base=True)


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs, _gvs, _planar_hsa])
def test_continuum_floating_state_and_dynamics_contract(factory) -> None:
    robot = factory()
    spatial = not robot.is_planar
    base_pose = (
        jnp.array([0.1, -0.05, 0.12, 0.98, 0.2, -0.1, 0.3])
        if spatial
        else jnp.array([0.2, 0.3, -0.1])
    )
    q_internal = jnp.linspace(-0.01, 0.02, robot.num_internal_dofs)
    qd_internal = jnp.linspace(0.03, -0.02, robot.num_internal_dofs)
    base_velocity = jnp.linspace(-0.1, 0.2, robot.num_base_velocities)

    q = robot.pack_configuration(q_internal, base_pose=base_pose)
    v = robot.pack_velocity(qd_internal, base_velocity=base_velocity)
    y = robot.pack_state(q, v)

    split_base_pose, split_q_internal = robot.split_configuration(q)
    split_base_velocity, split_qd_internal = robot.split_velocity(v)
    split_q, split_v, auxiliary = robot.split_state(y)
    assert split_base_pose is not None
    assert split_base_velocity is not None
    assert_allclose(split_q_internal, q_internal)
    assert_allclose(split_qd_internal, qd_internal)
    assert_allclose(split_q, q)
    assert_allclose(split_v, v)
    assert auxiliary.shape == (robot.num_auxiliary_states,)
    assert q.shape == (robot.num_coordinates,)
    assert v.shape == (robot.num_velocities,)
    assert y.shape == (robot.state_size,)
    assert not hasattr(robot, "num_dofs")
    assert robot.fixed_base_pose is None

    inertia, coriolis_velocity, gravity = robot.dynamics_terms(q, v, backend="jax")
    assert inertia.shape == (robot.num_velocities, robot.num_velocities)
    assert coriolis_velocity.shape == (robot.num_velocities,)
    assert gravity.shape == (robot.num_velocities,)
    assert_allclose(inertia, inertia.T, rtol=1.0e-9, atol=1.0e-11)

    actuation_matrix = robot.actuation_matrix(q)
    actuation_force = robot.actuation_force(q, jnp.zeros((robot.num_actuators,)), qd=v)
    elastic_force = robot.elastic_force(q)
    damping_matrix = robot.damping_matrix(q)
    num_base = robot.num_base_velocities
    assert actuation_matrix.shape[0] == robot.num_velocities
    assert_allclose(actuation_matrix[:num_base], 0.0, atol=0.0)
    assert_allclose(actuation_force[:num_base], 0.0, atol=0.0)
    assert_allclose(elastic_force[:num_base], 0.0, atol=0.0)
    assert_allclose(damping_matrix[:num_base], 0.0, atol=0.0)
    assert_allclose(damping_matrix[:, :num_base], 0.0, atol=0.0)


@pytest.mark.parametrize("factory", [_articulated, _pendulum])
def test_articulated_floating_state_and_dynamics_contract(factory) -> None:
    robot = factory()
    spatial = not robot.is_planar
    base_pose = (
        jnp.array([0.1, -0.05, 0.12, 0.98, 0.2, -0.1, 0.3])
        if spatial
        else jnp.array([0.2, 0.3, -0.1])
    )
    q_internal = jnp.linspace(-0.1, 0.2, robot.num_internal_dofs)
    qd_internal = jnp.linspace(0.2, -0.1, robot.num_internal_dofs)
    q = robot.pack_configuration(q_internal, base_pose=base_pose)
    v = robot.pack_velocity(
        qd_internal,
        base_velocity=jnp.linspace(-0.1, 0.2, robot.num_base_velocities),
    )

    inertia, coriolis_velocity, gravity = robot.dynamics_terms(q, v)
    assert inertia.shape == (robot.num_velocities, robot.num_velocities)
    assert coriolis_velocity.shape == (robot.num_velocities,)
    assert gravity.shape == (robot.num_velocities,)
    assert_allclose(inertia, inertia.T, rtol=1.0e-9, atol=1.0e-11)

    actuation_matrix = robot.actuation_matrix(q)
    actuation_force = robot.actuation_force(q, jnp.zeros((robot.num_actuators,)), qd=v)
    elastic_force = robot.elastic_force(q)
    damping_matrix = robot.damping_matrix(q)
    num_base = robot.num_base_velocities
    assert_allclose(actuation_matrix[:num_base], 0.0, atol=0.0)
    assert_allclose(actuation_force[:num_base], 0.0, atol=0.0)
    assert_allclose(elastic_force[:num_base], 0.0, atol=0.0)
    assert_allclose(damping_matrix[:num_base], 0.0, atol=0.0)
    assert_allclose(damping_matrix[:, :num_base], 0.0, atol=0.0)


@pytest.mark.parametrize(
    "factory", [_spatial_pcs, _planar_pcs, _gvs, _planar_hsa, _articulated, _pendulum]
)
def test_floating_gravity_matches_raw_energy_tangent(factory) -> None:
    robot = factory()
    base_pose = (
        jnp.array([0.0, 2**-0.5, 0.0, 2**-0.5, 0.03, -0.02, 0.01])
        if not robot.is_planar
        else jnp.array([0.73, 0.03, -0.02])
    )
    q = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64),
        base_pose=base_pose,
    )
    zero_velocity = jnp.zeros((robot.num_velocities,), dtype=jnp.float64)
    configuration_velocity_map = jax.jacfwd(
        lambda velocity: robot.configuration_derivative(q, velocity)
    )(zero_velocity)
    expected = configuration_velocity_map.T @ jax.grad(robot._gravitational_energy)(q)
    actual = robot.gravitational_force(q)
    _, _, assembled = robot.dynamics_terms(q, zero_velocity)

    assert_allclose(robot.gravitational_energy(q), robot._gravitational_energy(q))
    assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-11)
    assert_allclose(actual, assembled, rtol=1.0e-9, atol=1.0e-11)
    _, gravity_tangent = jvp(
        robot.gravitational_force,
        (q,),
        (jnp.linspace(-0.02, 0.03, robot.num_coordinates),),
    )
    assert bool(jnp.all(jnp.isfinite(gravity_tangent)))


@pytest.mark.parametrize(
    "factory", [_spatial_pcs, _planar_pcs, _gvs, _planar_hsa, _articulated, _pendulum]
)
def test_floating_gravity_uses_analytical_force_path(
    factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = factory()
    base_pose = (
        jnp.array([0.0, 2**-0.5, 0.0, 2**-0.5, 0.03, -0.02, 0.01])
        if not robot.is_planar
        else jnp.array([0.73, 0.03, -0.02])
    )
    q = robot.pack_configuration(
        jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64),
        base_pose=base_pose,
    )

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("generic autodiff or complete dynamics path was used")

    monkeypatch.setattr(type(robot), "dynamics_terms", fail_if_called)
    monkeypatch.setattr(SoftRobot, "_gravitational_force", fail_if_called)

    gravity = robot.gravitational_force(q)

    assert gravity.shape == (robot.num_velocities,)
    assert bool(jnp.all(jnp.isfinite(gravity)))


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs])
@pytest.mark.parametrize("floating_base", [False, True])
def test_pcs_gravity_reuses_one_pose_pass_and_energy_skips_jacobians(
    factory,
    floating_base: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = factory()
    base_pose = (
        jnp.array([0.0, 2**-0.5, 0.0, 2**-0.5, 0.03, -0.02, 0.01])
        if not robot.is_planar
        else jnp.array([0.73, 0.03, -0.02])
    )
    if floating_base:
        q = robot.pack_configuration(
            jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64),
            base_pose=base_pose,
        )
    else:
        robot = robot._fixed_base_robot_at_pose(base_pose)
        q = jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64)

    robot_type = type(robot)
    original_forward = robot_type._forward_kinematics_abscissa_batched
    calls = 0

    def counted_forward(self, q_, points):
        nonlocal calls
        calls += 1
        return original_forward(self, q_, points)

    monkeypatch.setattr(
        robot_type, "_forward_kinematics_abscissa_batched", counted_forward
    )
    with jax.disable_jit():
        gravity = robot.gravitational_force(q)
    assert bool(jnp.all(jnp.isfinite(gravity)))
    assert calls == 1

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("gravitational energy assembled Jacobians")

    monkeypatch.setattr(robot_type, "_J_local_abscissa_batched", fail_if_called)
    with jax.disable_jit():
        energy = robot.gravitational_energy(q)
    assert bool(jnp.isfinite(energy))


@pytest.mark.parametrize("floating_base", [False, True])
def test_gvs_gravity_reuses_one_pose_pass_and_energy_skips_jacobians(
    floating_base: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _gvs()
    base_pose = jnp.array([0.0, 2**-0.5, 0.0, 2**-0.5, 0.03, -0.02, 0.01])
    if floating_base:
        q = robot.pack_configuration(
            jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64),
            base_pose=base_pose,
        )
    else:
        robot = robot._fixed_base_robot_at_pose(base_pose)
        q = jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64)

    robot_type = type(robot)
    original_kinematics = robot_type._integration_pose_jacobians
    calls = 0

    def counted_kinematics(self, q_):
        nonlocal calls
        calls += 1
        return original_kinematics(self, q_)

    monkeypatch.setattr(robot_type, "_integration_pose_jacobians", counted_kinematics)
    with jax.disable_jit():
        gravity = robot.gravitational_force(q)
    assert bool(jnp.all(jnp.isfinite(gravity)))
    assert calls == 1

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("gravitational energy assembled Jacobians")

    monkeypatch.setattr(robot_type, "_integration_pose_jacobians", fail_if_called)
    with jax.disable_jit():
        energy = robot.gravitational_energy(q)
    assert bool(jnp.isfinite(energy))


@pytest.mark.parametrize("floating_base", [False, True])
def test_planar_hsa_gravity_uses_one_geometry_pass(
    floating_base: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = _planar_hsa()
    base_pose = jnp.array([0.73, 0.03, -0.02])
    if floating_base:
        q = robot.pack_configuration(
            jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64),
            base_pose=base_pose,
        )
    else:
        robot = robot._fixed_base_robot_at_pose(base_pose)
        q = jnp.zeros((robot.num_internal_dofs,), dtype=jnp.float64)

    robot_type = type(robot)
    original_geometry = robot_type._integration_geometry_full
    calls = 0

    def counted_geometry(self, q_):
        nonlocal calls
        calls += 1
        return original_geometry(self, q_)

    monkeypatch.setattr(robot_type, "_integration_geometry_full", counted_geometry)
    with jax.disable_jit():
        gravity = robot.gravitational_force(q)
    assert bool(jnp.all(jnp.isfinite(gravity)))
    assert calls == 1


def test_spatial_quaternion_derivative_and_retraction_are_consistent() -> None:
    robot = _spatial_pcs()
    q = robot.pack_configuration(
        jnp.zeros(robot.num_internal_dofs),
        base_pose=jnp.array([0.2, -0.1, 0.3, 0.9, 0.1, -0.2, 0.4]),
    )
    v = robot.pack_velocity(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs),
        base_velocity=jnp.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1]),
    )
    epsilon = 1.0e-6
    finite_difference = (robot.retract_configuration(q, epsilon * v) - q) / epsilon
    assert_allclose(
        finite_difference,
        robot.configuration_derivative(q, v),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert_allclose(jnp.linalg.norm(q[:4]), 1.0)
    assert_allclose(robot.normalize_configuration(-q)[:4], -q[:4])


def test_floating_gvs_warp_matches_jax_for_batch_and_jvp(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "floating-gvs"))
    pytest.importorskip("warp")
    robot = _gvs()
    q = robot.pack_configuration(
        jnp.zeros(robot.num_internal_dofs),
        base_pose=jnp.array([0.1, -0.05, 0.12, 0.98, 0.2, -0.1, 0.3]),
    )
    v = robot.pack_velocity(
        jnp.linspace(-0.03, 0.04, robot.num_internal_dofs),
        base_velocity=jnp.linspace(-0.1, 0.2, 6),
    )

    def warp_terms(q_, v_):
        return robot.dynamics_terms(q_, v_, backend="warp")

    def jax_terms(q_, v_):
        return robot.dynamics_terms(q_, v_, backend="jax")

    for actual, expected in zip(warp_terms(q, v), jax_terms(q, v), strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    q_batch = jnp.stack([q, robot.retract_configuration(q, 0.01 * v)])
    v_batch = jnp.stack([v, 0.8 * v])
    for actual, expected in zip(
        jax.vmap(warp_terms)(q_batch, v_batch),
        jax.vmap(jax_terms)(q_batch, v_batch),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    tangent = (
        jnp.linspace(-0.01, 0.02, robot.num_coordinates),
        jnp.linspace(0.02, -0.01, robot.num_velocities),
    )
    warp_primal, warp_tangent = jvp(
        lambda state: warp_terms(*state), ((q, v),), (tangent,)
    )
    jax_primal, jax_tangent = jvp(
        lambda state: jax_terms(*state), ((q, v),), (tangent,)
    )
    for actual, expected in zip(warp_primal, jax_primal, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(warp_tangent, jax_tangent, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    s = jnp.array(0.07)
    warp_pose, warp_jacobian = robot.forward_kinematics_and_jacobian_inertialframe(
        q, s, backend="warp"
    )
    jax_pose, jax_jacobian = robot.forward_kinematics_and_jacobian_inertialframe(
        q, s, backend="jax"
    )
    assert_allclose(warp_pose, jax_pose, rtol=2.0e-10, atol=2.0e-11)
    assert_allclose(warp_jacobian, jax_jacobian, rtol=2.0e-10, atol=2.0e-11)

    samples = jnp.linspace(0.0, float(robot.length), 5)
    warp_poses, warp_jacobians = (
        robot.forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
            q, samples, backend="warp"
        )
    )
    jax_poses, jax_jacobians = (
        robot.forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
            q, samples, backend="jax"
        )
    )
    assert_allclose(warp_poses, jax_poses, rtol=2.0e-10, atol=2.0e-11)
    assert_allclose(warp_jacobians, jax_jacobians, rtol=2.0e-10, atol=2.0e-11)

    warp_primal, warp_tangent = jvp(
        lambda q_: robot.forward_kinematics_and_jacobian_inertialframe(
            q_, s, backend="warp"
        ),
        (q,),
        (tangent[0],),
    )
    jax_primal, jax_tangent = jvp(
        lambda q_: robot.forward_kinematics_and_jacobian_inertialframe(
            q_, s, backend="jax"
        ),
        (q,),
        (tangent[0],),
    )
    for actual, expected in zip(warp_primal, jax_primal, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(warp_tangent, jax_tangent, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_fixed_robot_rejects_runtime_base_state() -> None:
    floating = _planar_pcs()
    fixed = floating._fixed_base_robot_at_pose(jnp.array([0.1, 0.2, -0.3]))
    q_internal = jnp.zeros(fixed.num_internal_dofs)

    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_configuration(q_internal, base_pose=jnp.zeros(3))
    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_velocity(q_internal, base_velocity=jnp.zeros(3))
    with pytest.raises(ValueError, match="has no fixed base pose"):
        floating.with_fixed_base_pose(jnp.zeros(3))


def test_floating_controller_matches_current_pose_fixed_view() -> None:
    robot = _planar_pcs()
    q_internal = jnp.linspace(-0.02, 0.03, robot.num_internal_dofs)
    qd_internal = jnp.linspace(0.04, -0.01, robot.num_internal_dofs)
    base_pose = jnp.array([0.3, 0.2, -0.1])
    q = robot.pack_configuration(q_internal, base_pose=base_pose)
    v = robot.pack_velocity(qd_internal, base_velocity=jnp.array([0.2, -0.1, 0.05]))
    floating_state = SystemState(t=jnp.array(0.2), y=robot.pack_state(q, v))
    reference = ReferenceTrajectory(
        ts=jnp.array([0.0, 1.0]),
        x_des_ts=jnp.stack([q_internal + 0.01, q_internal + 0.01]),
        xd_des_ts=jnp.stack([jnp.zeros_like(qd_internal)] * 2),
        xdd_des_ts=jnp.stack([jnp.zeros_like(qd_internal)] * 2),
    )
    gains = PIDControl(Kp=2.0, Ki=0.0, Kd=0.5)
    floating_controller = PIDController(robot, reference, gains)

    fixed = robot._fixed_base_robot_at_pose(base_pose)
    fixed_controller = PIDController(fixed, reference, gains)
    fixed_state = SystemState(
        t=floating_state.t,
        y=fixed.pack_state(q_internal, qd_internal),
    )

    floating_u, _ = floating_controller.error_based_feedback_term(floating_state)
    fixed_u, _ = fixed_controller.error_based_feedback_term(fixed_state)
    assert_allclose(floating_u, fixed_u, rtol=1.0e-12, atol=1.0e-12)


def test_spatial_rollout_projection_normalizes_only_base_quaternion() -> None:
    robot = _spatial_pcs()
    q_internal = jnp.linspace(-0.02, 0.03, robot.num_internal_dofs)
    v = robot.pack_velocity(
        jnp.linspace(0.04, -0.01, robot.num_internal_dofs),
        base_velocity=jnp.linspace(-0.1, 0.2, robot.num_base_velocities),
    )
    q = jnp.concatenate([jnp.array([0.2, -0.1, 0.3, 1.8, 0.1, -0.2, 0.4]), q_internal])
    auxiliary = jnp.zeros((robot.num_auxiliary_states,))
    y = robot.pack_state(q, v, auxiliary)

    projected = robot.project_state(jnp.stack([y, y]))
    projected_q, projected_v, projected_auxiliary = robot.split_state(projected)
    assert_allclose(jnp.linalg.norm(projected_q[..., :4], axis=-1), 1.0)
    assert_allclose(projected_q[..., 4:], jnp.stack([q[4:], q[4:]]))
    assert_allclose(projected_v, jnp.stack([v, v]))
    assert_allclose(projected_auxiliary, jnp.stack([auxiliary, auxiliary]))


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs, _gvs])
def test_floating_continuum_warp_kinematics_matches_jax(
    factory, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "WARP_CACHE_PATH", str(tmp_path / f"floating-{factory.__name__}")
    )
    pytest.importorskip("warp")
    robot = factory()
    base_pose = (
        jnp.array([0.1, -0.2, 0.15, 0.96, 0.3, -0.2, 0.4])
        if not robot.is_planar
        else jnp.array([0.4, 0.3, -0.2])
    )
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs), base_pose=base_pose
    )
    samples = jnp.linspace(0.0, float(robot.length), 5)

    warp_result = robot.forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
        q, samples, backend="warp"
    )
    jax_result = robot.forward_kinematics_and_jacobian_inertialframe_abscissa_batched(
        q, samples, backend="jax"
    )
    for actual, expected in zip(warp_result, jax_result, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    direction = jnp.linspace(-0.01, 0.02, robot.num_coordinates)
    s = jnp.array(0.07)
    warp_primal, warp_tangent = jvp(
        lambda q_: robot.forward_kinematics_and_jacobian_inertialframe(
            q_, s, backend="warp"
        ),
        (q,),
        (direction,),
    )
    jax_primal, jax_tangent = jvp(
        lambda q_: robot.forward_kinematics_and_jacobian_inertialframe(
            q_, s, backend="jax"
        ),
        (q,),
        (direction,),
    )
    for actual, expected in zip(warp_primal, jax_primal, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(warp_tangent, jax_tangent, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("factory", [_spatial_pcs, _planar_pcs])
def test_floating_pcs_warp_dynamics_matches_jax_on_cuda(
    factory, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "WARP_CACHE_PATH", str(tmp_path / f"floating-dynamics-{factory.__name__}")
    )
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("Floating PCS Warp dynamics require CUDA.")

    robot = factory()
    base_pose = (
        jnp.array([0.1, -0.2, 0.15, 0.96, 0.3, -0.2, 0.4])
        if not robot.is_planar
        else jnp.array([0.4, 0.3, -0.2])
    )
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs), base_pose=base_pose
    )
    v = robot.pack_velocity(
        jnp.linspace(0.04, -0.03, robot.num_internal_dofs),
        base_velocity=jnp.linspace(-0.1, 0.2, robot.num_base_velocities),
    )

    q_batch = jnp.stack([q, robot.retract_configuration(q, 0.01 * v)])
    v_batch = jnp.stack([v, 0.8 * v])
    warp_result = jax.vmap(lambda q_, v_: robot.dynamics_terms(q_, v_, backend="warp"))(
        q_batch, v_batch
    )
    jax_result = jax.vmap(lambda q_, v_: robot.dynamics_terms(q_, v_, backend="jax"))(
        q_batch, v_batch
    )
    for actual, expected in zip(warp_result, jax_result, strict=True):
        assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
