# ruff: noqa: E402
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.articulated.simulate_mckibben_umarm import (
    UMARM_Z_DOWN_BASE_POSE,
)
from examples.simulation.articulated.simulate_mckibben_umarm import (
    build_robot as build_example_robot,
)
from soromox.rendering.viser_renderer import VISER_AVAILABLE
from soromox.systems import (
    ArticulatedSoftRobot,
    McKibbenActuatedUMArm,
    McKibbenActuatedUMArmParams,
)

ASSET_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robot_parameters"
    / "mckibben_umarm"
    / "reference_parameters.npz"
)


def make_robot() -> McKibbenActuatedUMArm:
    return McKibbenActuatedUMArm(
        McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    )


def test_cached_params_shapes() -> None:
    params = McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    robot = McKibbenActuatedUMArm(params)

    with np.load(ASSET_PARAMS_PATH) as data:
        assert str(data["schema_version"]) == "mckibben_umarm_params_v1"
        assert str(data["robot_name"]) == "McKibbenActuatedUMArm"
        assert "mckibben_reference_length" in data
        assert "mck_base_length" not in data

    assert robot.num_dofs == 12
    assert robot.num_actuators == 24
    assert robot.num_segments == 3
    assert params.joint_armature.shape == (12,)
    assert params.mckibben_moving_base_transform.shape == (6, 4, 4)
    assert params.mckibben_moving_points.shape == (6, 4, 3)
    assert params.mckibben_fixed_points.shape == (6, 4, 3)
    assert params.mckibben_reference_length.shape == (6, 4)
    assert params.mckibben_length_offset.shape == (6, 4)
    assert params.mckibben_thread_length.shape == (6, 4)
    assert params.mckibben_num_turns.shape == (6, 4)
    assert params.mckibben_joint_pair_indices.shape == (6, 2)


def test_cached_params_use_soromox_base_axis() -> None:
    robot = make_robot()
    q = jnp.zeros((robot.num_dofs,))
    joint_frames = np.asarray(robot.forward_kinematics_joints(q))
    tip_frames = np.asarray(robot.forward_kinematics_tips(q))

    base_position = joint_frames[0, :3, 3]
    first_physical_tip = tip_frames[1, :3, 3]
    proximal_direction = first_physical_tip - base_position
    proximal_direction = proximal_direction / np.linalg.norm(proximal_direction)

    assert_allclose(base_position, np.zeros(3), rtol=0.0, atol=1e-12)
    assert_allclose(proximal_direction, np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_example_build_robot_uses_z_down_base_pose() -> None:
    robot = build_example_robot()
    q = jnp.zeros((robot.num_dofs,))
    joint_frames = np.asarray(robot.forward_kinematics_joints(q))
    tip_frames = np.asarray(robot.forward_kinematics_tips(q))

    base_position = joint_frames[0, :3, 3]
    first_physical_tip = tip_frames[1, :3, 3]
    proximal_direction = first_physical_tip - base_position
    proximal_direction = proximal_direction / np.linalg.norm(proximal_direction)

    assert_allclose(robot.base_pose, UMARM_Z_DOWN_BASE_POSE, rtol=0.0, atol=1e-12)
    assert_allclose(base_position, np.array([0.0, 0.0, 1.2]), atol=1e-12)
    assert_allclose(proximal_direction, np.array([0.0, 0.0, -1.0]), atol=1e-12)


def test_mckibben_dynamics_terms_are_finite() -> None:
    robot = make_robot()
    q = jnp.linspace(-0.2, 0.2, robot.num_dofs)
    qd = jnp.linspace(0.1, -0.1, robot.num_dofs)
    pressures = jnp.linspace(0.0, 8.0e4, robot.num_actuators)
    y = jnp.concatenate([q, qd])

    segments = robot.mckibben_actuator_segments(q)
    lengths = robot.effective_lengths(q)
    forces = robot.actuator_forces(q, pressures)
    actuation = robot.actuation_matrix(q)
    inertia = robot.inertia_matrix(q)
    yd = robot.forward_dynamics(jnp.array(0.0), y, (pressures, None))

    assert segments.shape == (robot.num_actuators, 2, 3)
    assert lengths.shape == (robot.num_actuators,)
    assert forces.shape == (robot.num_actuators,)
    assert actuation.shape == (robot.num_dofs, robot.num_actuators)
    assert yd.shape == y.shape
    assert jnp.isfinite(segments).all()
    assert jnp.isfinite(lengths).all()
    assert jnp.isfinite(forces).all()
    assert jnp.isfinite(actuation).all()
    assert jnp.isfinite(yd).all()
    assert_allclose(inertia, inertia.T, rtol=1e-10, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(np.asarray(inertia)) > 0.0)


def test_mckibben_actuation_matrix_matches_potential_gradient() -> None:
    robot = make_robot()
    q_samples = jnp.stack(
        [
            jnp.linspace(-0.2, 0.2, robot.num_dofs),
            jnp.sin(jnp.linspace(0.0, 1.1, robot.num_dofs)) * 0.15,
            jnp.cos(jnp.linspace(-0.7, 0.4, robot.num_dofs)) * -0.12,
        ]
    )

    for q in q_samples:
        expected = jax.jacobian(robot._potential_per_pressure)(q).T
        assert_allclose(robot.actuation_matrix(q), expected, rtol=1e-10, atol=1e-10)


def test_mckibben_forward_dynamics_is_inherited_and_matches_dense_equation() -> None:
    assert McKibbenActuatedUMArm.forward_dynamics is ArticulatedSoftRobot.forward_dynamics

    robot = make_robot()
    q = jnp.linspace(-0.12, 0.15, robot.num_dofs)
    qd = jnp.linspace(0.05, -0.04, robot.num_dofs)
    pressures = jnp.linspace(0.0, 6.0e4, robot.num_actuators)
    tau_ext = jnp.linspace(-0.02, 0.03, robot.num_dofs)
    y = jnp.concatenate([q, qd])

    yd = robot.forward_dynamics(jnp.array(0.0), y, (pressures, tau_ext))
    _, qdd = jnp.split(yd, 2)

    rhs = (
        robot.actuation_force(q, pressures)
        + tau_ext
        - robot.coriolis_matrix(q, qd) @ qd
        - robot.gravitational_force(q)
        - robot.elastic_force(q)
        - robot.damping_matrix(q) @ qd
    )
    qdd_dense = jnp.linalg.solve(robot.inertia_matrix(q), rhs)

    assert_allclose(qdd, qdd_dense, rtol=1e-9, atol=1e-9)


def test_umarm_viser_renderer_smoke() -> None:
    if not VISER_AVAILABLE:
        pytest.skip("viser is not installed")

    from soromox.rendering import UMArmViserRenderer

    robot = make_robot()
    renderer = UMArmViserRenderer(robot, auto_start=False, open_browser=False)
    q = jnp.zeros((robot.num_dofs,))
    layers = renderer.compute_actuator_visual_layers_batched(
        q[None, :], jnp.zeros((1, 3))
    )
    assert len(layers) == 1
    assert layers[0].name == "mckibben_muscles"
    assert layers[0].kind == "muscle"
    assert layers[0].points.shape == (1, robot.num_actuators, 2, 3)
    backbone = np.asarray(
        renderer.compute_backbone_curves_batched(q[None, :], jnp.zeros((1, 3)))
    )
    base_plate_position, _ = renderer._base_plate_pose(backbone[0, 0])
    base_plate_direction = renderer._base_tangent_axis(dim=3)
    assert_allclose(
        base_plate_direction,
        np.array([1.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert_allclose(
        base_plate_position,
        backbone[0, 0] - 0.5 * renderer._base_plate_thickness * base_plate_direction,
        rtol=0.0,
        atol=1e-12,
    )


def test_umarm_viser_default_camera_is_tuned_for_z_down_mount() -> None:
    if not VISER_AVAILABLE:
        pytest.skip("viser is not installed")

    from soromox.rendering import UMArmViserRenderer

    robot = build_example_robot()
    renderer = UMArmViserRenderer(robot, auto_start=False, open_browser=False)
    config = renderer._default_umarm_camera_config()
    center = np.array([0.0, 0.0, 0.7])

    camera_pos, look_at = config.compute_auto_position(
        center,
        max_extent=1.0,
        reference_transform=np.asarray(robot.base_transform),
    )

    assert config.distance_factor == 1.05
    assert config.position_offset == (-0.5, -0.9, 0.45)
    assert_allclose(look_at, center, atol=1e-12)
    assert_allclose(
        camera_pos - look_at,
        np.array([0.4725, -0.945, 0.525]),
        atol=1e-12,
    )


def test_umarm_viser_live_mode_smoke() -> None:
    if not VISER_AVAILABLE:
        pytest.skip("viser is not installed")

    from soromox.rendering import UMArmViserRenderer

    robot = make_robot()
    renderer = UMArmViserRenderer(
        robot,
        auto_start=False,
        open_browser=False,
        port=0,
        actuator_color_mode="pressure",
    )
    controller = renderer.start_live_mode()
    try:
        q = np.zeros((robot.num_dofs,))
        pressures = np.zeros((robot.num_actuators,))
        pressures[0] = 1.0e5
        controller.push_state_with_pressures(q, pressures)
        assert renderer._current_geometry_q is not None
        assert renderer._current_geometry_q.shape == (1, robot.num_dofs)
        assert renderer._scene_handles is not None
        assert len(renderer._scene_handles.base_plates) == 1
        assert len(renderer._scene_handles.actuator_lines) == 1
        assert renderer._scene_handles.actuator_lines[0].points.shape == (
            robot.num_actuators,
            2,
            3,
        )
        base_handle = renderer._scene_handles.base_plates[0]
        body_handle = renderer._scene_handles.backbone_points[0][0]
        actuator_handle = renderer._scene_handles.actuator_lines[0]

        q_next = q.copy()
        q_next[0] = 0.1
        controller.push_state_with_pressures(q_next, pressures)

        assert renderer._scene_handles.base_plates[0] is base_handle
        assert renderer._scene_handles.backbone_points[0][0] is body_handle
        assert renderer._scene_handles.actuator_lines[0] is actuator_handle
    finally:
        controller.stop()
        renderer.stop()


def test_mckibben_actuator_segments_are_in_backbone_frame() -> None:
    robot = make_robot()
    q = jnp.zeros((robot.num_dofs,))
    actuator_segments = np.asarray(robot.mckibben_actuator_segments(q))
    joint_positions = np.asarray(robot.forward_kinematics_joints(q))[:, :3, 3]
    tip_positions = np.asarray(robot.forward_kinematics_tips(q))[:, :3, 3]
    backbone_positions = np.concatenate([joint_positions, tip_positions], axis=0)

    actuator_points = actuator_segments.reshape(-1, 3)
    lower = backbone_positions.min(axis=0) - np.array([0.15, 0.15, 0.15])
    upper = backbone_positions.max(axis=0) + np.array([0.15, 0.15, 0.15])
    assert np.all(actuator_points >= lower)
    assert np.all(actuator_points <= upper)
