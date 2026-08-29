# ruff: noqa: E402
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from system_param_builders import pcs_params

from examples.simulation.articulated.simulate_mckibben_umarm import (
    UMARM_Z_DOWN_BASE_POSE,
)
from examples.simulation.articulated.simulate_mckibben_umarm import (
    build_robot as build_example_robot,
)
from soromox.actuation import ArticulatedMcKibbenActuator
from soromox.rendering import actuator_visual_layers
from soromox.rendering.viser_renderer import VISER_AVAILABLE
from soromox.systems import (
    PCS,
    ArticulatedSoftRobot,
    McKibbenActuatedUMArm,
    McKibbenActuatedUMArmParams,
    PCSStructure,
)
from soromox.utils.geometry import poses

ASSET_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robot_parameters"
    / "mckibben_umarm"
    / "reference_parameters.npz"
)


def make_robot() -> McKibbenActuatedUMArm:
    return McKibbenActuatedUMArm.from_cached_parameters(ASSET_PARAMS_PATH)


def make_zero_actuator_robot() -> McKibbenActuatedUMArm:
    params = McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    return McKibbenActuatedUMArm(
        params,
        actuator=ArticulatedMcKibbenActuator.empty(),
        base_pose=poses.spatial_mounting_pose("horizontal"),
    )


def _legacy_local_segments_and_lengths(robot, q):
    """Independent transcription of the pre-composition UMArm geometry."""
    params = robot.actuators[0].params.transmission
    groups = []
    for group_index in range(params.num_groups):
        transform = params.moving_base_transform[group_index]
        joint_pair = params.joint_pair_indices[group_index]
        qx, qy = q[joint_pair[0]], q[joint_pair[1]]
        cx, sx = jnp.cos(qx), jnp.sin(qx)
        cy, sy = jnp.cos(qy), jnp.sin(qy)
        rotation_x = jnp.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rotation_y = jnp.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rotation = transform[:3, :3] @ rotation_x @ rotation_y
        moving = params.moving_points[group_index] @ rotation.T + transform[:3, 3]
        fixed = params.fixed_points[group_index]
        groups.append(jnp.stack([fixed, moving], axis=1))
    local_segments = jnp.stack(groups)
    geometric_length = jnp.linalg.norm(
        local_segments[:, :, 1] - local_segments[:, :, 0], axis=-1
    )
    effective_length = geometric_length - params.reference_length + params.length_offset
    return local_segments, effective_length


def _legacy_coordinates(robot, q):
    params = robot.actuators[0].params.transmission
    _, length = _legacy_local_segments_and_lengths(robot, q)
    return (
        (params.thread_length**2 - length**2)
        * length
        / (4.0 * jnp.pi * params.num_turns**2)
    ).reshape((-1,))


def _legacy_world_segments(robot, q):
    params = robot.actuators[0].params.transmission
    local_segments, _ = _legacy_local_segments_and_lengths(robot, q)
    joint_frames, link_frames, _, _ = robot._kinematic_frames(q)
    frames = []
    for group_index in range(params.num_groups):
        pair_start = params.joint_pair_indices[group_index, 0]
        if group_index % 2:
            frames.append(link_frames[jnp.maximum(pair_start - 1, 0)])
        else:
            frames.append(joint_frames[pair_start])
    frames = jnp.stack(frames)
    world = (
        jnp.einsum("gij,gapj->gapi", frames[:, :3, :3], local_segments)
        + frames[:, :3, 3][:, None, None, :]
    )
    return world.reshape((-1, 2, 3))


def test_cached_params_shapes() -> None:
    params = McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    robot = make_robot()
    actuator_params = robot.actuators[0].params.transmission

    with np.load(ASSET_PARAMS_PATH) as data:
        assert str(data["schema_version"]) == "mckibben_umarm_params_v1"
        assert str(data["robot_name"]) == "McKibbenActuatedUMArm"
        assert "mckibben_reference_length" in data
        assert "mck_base_length" not in data

    assert robot.num_dofs == 12
    assert robot.num_actuators == 24
    assert robot.num_segments == 3
    assert params.joint_armature.shape == (12,)
    assert actuator_params.moving_base_transform.shape == (6, 4, 4)
    assert actuator_params.moving_points.shape == (6, 4, 3)
    assert actuator_params.fixed_points.shape == (6, 4, 3)
    assert actuator_params.reference_length.shape == (6, 4)
    assert actuator_params.length_offset.shape == (6, 4)
    assert actuator_params.thread_length.shape == (6, 4)
    assert actuator_params.num_turns.shape == (6, 4)
    assert actuator_params.joint_pair_indices.shape == (6, 2)


def test_zero_actuator_umarm_exposes_empty_actuator_interface() -> None:
    robot = make_zero_actuator_robot()
    q = jnp.linspace(-0.2, 0.2, robot.num_dofs)
    qd = jnp.linspace(0.1, -0.1, robot.num_dofs)
    pressures = jnp.zeros((0,), dtype=q.dtype)
    y = jnp.concatenate([q, qd])

    assert robot.num_actuators == 0
    assert robot.num_mckibben_groups == 0
    actuator = robot.actuators[0]
    assert actuator.segments(robot, q).shape == (0, 2, 3)
    assert actuator.effective_lengths(q).shape == (0,)
    assert robot.actuator_coordinates(q).shape == (0,)
    assert actuator.axial_forces(q, pressures).shape == (0,)
    assert robot.actuation_matrix(q).shape == (robot.num_dofs, 0)
    assert_allclose(robot.actuation_force(q, pressures), jnp.zeros((robot.num_dofs,)))
    assert actuator_visual_layers(robot, q, jnp.linspace(0.0, 1.0, 3)) == ()
    assert robot.forward_dynamics(jnp.array(0.0), y, (pressures, None)).shape == y.shape


def test_articulated_mckibben_actuator_rejects_continuum_host() -> None:
    params = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([5.0e4]),
        damping_matrix=jnp.eye(6),
        gravity=jnp.zeros(3),
    )
    actuator = ArticulatedMcKibbenActuator.from_cached_npz(ASSET_PARAMS_PATH)
    with pytest.raises(TypeError, match="spatial articulated host"):
        PCS(params, PCSStructure(num_gauss_points=3), actuators=actuator)


def test_mckibben_transmission_rejects_duplicate_joint_pair_indices() -> None:
    actuator = ArticulatedMcKibbenActuator.from_cached_npz(ASSET_PARAMS_PATH)
    transmission = actuator.params.transmission
    duplicate_pair_indices = transmission.joint_pair_indices.at[0, 1].set(
        transmission.joint_pair_indices[0, 0]
    )

    with pytest.raises(ValueError, match="two distinct DOF indices"):
        transmission.replace(joint_pair_indices=duplicate_pair_indices)


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

    assert_allclose(
        robot.fixed_base_pose, UMARM_Z_DOWN_BASE_POSE, rtol=0.0, atol=1e-12
    )
    assert_allclose(base_position, np.array([0.0, 0.0, 1.2]), atol=1e-12)
    assert_allclose(proximal_direction, np.array([0.0, 0.0, -1.0]), atol=1e-12)


def test_mckibben_dynamics_terms_are_finite() -> None:
    robot = make_robot()
    q = jnp.linspace(-0.2, 0.2, robot.num_dofs)
    qd = jnp.linspace(0.1, -0.1, robot.num_dofs)
    pressures = jnp.linspace(0.0, 8.0e4, robot.num_actuators)
    y = jnp.concatenate([q, qd])

    actuator = robot.actuators[0]
    segments = actuator.segments(robot, q)
    lengths = actuator.effective_lengths(q)
    forces = actuator.axial_forces(q, pressures)
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
        expected = jax.jacobian(robot.actuator_coordinates)(q).T
        assert_allclose(robot.actuation_matrix(q), expected, rtol=1e-10, atol=1e-10)


def test_mckibben_component_matches_legacy_geometry_and_constitutive_oracle() -> None:
    robot = make_robot()
    actuator = robot.actuators[0]
    params = actuator.params.transmission
    q_samples = jnp.stack(
        [
            jnp.zeros((robot.num_dofs,)),
            jnp.linspace(-0.2, 0.2, robot.num_dofs),
            0.17 * jnp.sin(jnp.arange(robot.num_dofs)),
        ]
    )

    for q in q_samples:
        legacy_local, legacy_length = _legacy_local_segments_and_lengths(robot, q)
        legacy_coordinate = _legacy_coordinates(robot, q)
        pressure = jnp.linspace(0.0, 8.0e4, robot.num_actuators)
        legacy_force = (
            pressure.reshape(params.group_shape)
            * (params.thread_length**2 - 3.0 * legacy_length**2)
            / (4.0 * jnp.pi * params.num_turns**2)
        ).reshape((-1,))

        assert_allclose(
            actuator.local_segments(q), legacy_local, rtol=1e-10, atol=1e-12
        )
        assert_allclose(
            actuator.segments(robot, q),
            _legacy_world_segments(robot, q),
            rtol=1e-10,
            atol=1e-12,
        )
        assert_allclose(
            actuator.effective_lengths(q),
            legacy_length.reshape((-1,)),
            rtol=1e-10,
            atol=1e-12,
        )
        assert_allclose(
            robot.actuator_coordinates(q),
            legacy_coordinate,
            rtol=1e-10,
            atol=1e-12,
        )
        assert_allclose(
            actuator.axial_forces(q, pressure),
            legacy_force,
            rtol=1e-10,
            atol=1e-12,
        )
        assert_allclose(
            robot.actuation_matrix(q),
            jax.jacrev(lambda q_: _legacy_coordinates(robot, q_))(q).T,
            rtol=1e-10,
            atol=1e-12,
        )


def test_mckibben_optional_qd_power_jit_and_independent_updates() -> None:
    robot = make_robot()
    q = jnp.linspace(-0.13, 0.16, robot.num_dofs)
    qd = jnp.linspace(0.08, -0.05, robot.num_dofs)
    pressure = jnp.linspace(1.0e4, 7.0e4, robot.num_actuators)

    tau = robot.actuation_force(q, pressure)
    assert_allclose(
        tau,
        robot.actuation_force(q, pressure, qd=qd),
        rtol=1e-10,
        atol=1e-12,
    )
    assert_allclose(
        qd @ tau,
        robot.actuator_velocities(q, qd) @ pressure,
        rtol=1e-10,
        atol=1e-12,
    )
    assert_allclose(
        eqx.filter_jit(robot.actuation_force)(q, pressure, qd),
        tau,
        rtol=1e-10,
        atol=1e-12,
    )

    updated_body = robot.update_params(
        joint_armature=robot.params.joint_armature + 1.0e-5
    )
    assert_allclose(
        updated_body.actuators[0].params.transmission.length_offset,
        robot.actuators[0].params.transmission.length_offset,
    )

    actuator_params = robot.actuators[0].params
    transmission = actuator_params.transmission.replace(
        length_offset=actuator_params.transmission.length_offset + 1.0e-5
    )
    updated_actuator = robot.update_actuator_params(0, transmission=transmission)
    assert_allclose(updated_actuator.params.joint_armature, robot.params.joint_armature)
    assert_allclose(
        updated_actuator.actuators[0].params.transmission.length_offset,
        transmission.length_offset,
    )


def _updated_umarm_parameter_pair(robot):
    body = robot.params.replace(
        gravity=robot.params.gravity.at[2].set(-9.7),
        mass=1.01 * robot.params.mass,
        joint_armature=robot.params.joint_armature + 1.0e-5,
    )
    actuator = robot.actuators[0].params
    transmission = actuator.transmission.replace(
        moving_points=actuator.transmission.moving_points + 1.0e-5,
        length_offset=actuator.transmission.length_offset + 1.0e-5,
        thread_length=1.001 * actuator.transmission.thread_length,
        num_turns=1.001 * actuator.transmission.num_turns,
    )
    return body, actuator.replace(
        transmission=transmission,
        lower_bounds=actuator.lower_bounds + 1.0,
        upper_bounds=actuator.upper_bounds,
    )


def _umarm_parameter_summary(robot):
    transmission = robot.actuators[0].params.transmission
    return jnp.concatenate(
        [
            robot.fixed_base_pose,
            robot.g,
            robot.m,
            robot.joint_armature,
            transmission.moving_points.reshape(-1),
            transmission.length_offset.reshape(-1),
            transmission.thread_length.reshape(-1),
            transmission.num_turns.reshape(-1),
        ]
    )


def test_umarm_body_and_actuator_parameter_updates_refresh_eager_runtime_arrays():
    robot = make_robot()
    body_params, actuator_params = _updated_umarm_parameter_pair(robot)
    before = _umarm_parameter_summary(robot)

    updated = robot.with_params(body_params).with_actuator_params(0, actuator_params)

    assert_allclose(_umarm_parameter_summary(robot), before)
    assert not jnp.allclose(_umarm_parameter_summary(updated), before)


def test_umarm_body_and_actuator_parameter_updates_compile_once():
    robot = make_robot()
    body_params, actuator_params = _updated_umarm_parameter_pair(robot)
    trace_count = {"value": 0}

    @eqx.filter_jit
    def compiled_summary(current_body, current_actuator):
        trace_count["value"] += 1
        updated = robot.with_params(current_body).with_actuator_params(
            0, current_actuator
        )
        return _umarm_parameter_summary(updated)

    baseline = compiled_summary(robot.params, robot.actuators[0].params)
    actual = compiled_summary(body_params, actuator_params)

    assert trace_count["value"] == 1
    assert not jnp.allclose(actual, baseline)
    eager = robot.with_params(body_params).with_actuator_params(0, actuator_params)
    assert_allclose(actual, _umarm_parameter_summary(eager))


def test_umarm_compiled_partial_actuator_update_matches_eager_metadata():
    robot = make_robot()
    lower_bounds = robot.actuators[0].params.lower_bounds + 1.0
    trace_count = {"value": 0}

    @eqx.filter_jit
    def compiled_lower_bounds(current_lower_bounds):
        trace_count["value"] += 1
        updated = robot.update_actuator_params(0, lower_bounds=current_lower_bounds)
        return updated.actuators[0].metadata.lower_bounds

    baseline = compiled_lower_bounds(robot.actuators[0].params.lower_bounds)
    actual = compiled_lower_bounds(lower_bounds)
    eager = robot.update_actuator_params(0, lower_bounds=lower_bounds)

    assert trace_count["value"] == 1
    assert not jnp.allclose(actual, baseline)
    assert_allclose(actual, eager.actuators[0].metadata.lower_bounds)


def test_umarm_parameter_gradient_passes_through_compiled_updates():
    robot = make_robot()
    q = jnp.linspace(-0.1, 0.1, robot.num_dofs)
    actuator_params = robot.actuators[0].params

    @eqx.filter_jit
    def coordinate_sum(length_offset):
        transmission = actuator_params.transmission.replace(length_offset=length_offset)
        current = actuator_params.replace(transmission=transmission)
        updated = robot.with_actuator_params(0, current)
        return jnp.sum(updated.actuator_coordinates(q))

    gradient = jax.grad(coordinate_sum)(actuator_params.transmission.length_offset)

    assert gradient.shape == actuator_params.transmission.length_offset.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


def test_mckibben_forward_dynamics_is_inherited_and_matches_dense_equation() -> None:
    assert (
        McKibbenActuatedUMArm.forward_dynamics is ArticulatedSoftRobot.forward_dynamics
    )

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
    actuator_segments = np.asarray(robot.actuators[0].segments(robot, q))
    joint_positions = np.asarray(robot.forward_kinematics_joints(q))[:, :3, 3]
    tip_positions = np.asarray(robot.forward_kinematics_tips(q))[:, :3, 3]
    backbone_positions = np.concatenate([joint_positions, tip_positions], axis=0)

    actuator_points = actuator_segments.reshape(-1, 3)
    lower = backbone_positions.min(axis=0) - np.array([0.15, 0.15, 0.15])
    upper = backbone_positions.max(axis=0) + np.array([0.15, 0.15, 0.15])
    assert np.all(actuator_points >= lower)
    assert np.all(actuator_points <= upper)
