"""Per-world descriptor wiring, using only SoRoMoX operands and kernels."""

from __future__ import annotations

import inspect
import os
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.execution.warp.common import floating_kinematics as composition
from soromox.execution.warp.common import joints, joints_cooperative
from soromox.execution.warp.gvs import chain, floating_chain, kinematics, packed
from soromox.execution.warp.gvs import directional_kinematics as directional
from soromox.execution.warp.gvs.actuation import threadlike
from soromox.execution.warp.gvs.actuation.operands import GVSThreadlikeOperands
from soromox.execution.warp.gvs.executor import _local_dynamics_terms
from soromox.execution.warp.gvs.operands import (
    GVSFloatingOperands,
    GVSFloatingPipelineShapes,
    GVSKinematicsOperands,
    GVSKinematicsShapes,
    GVSOperands,
    GVSPipelineShapes,
)
from soromox.systems import GVS, GVSSegment, JointSpec, LinkSpec, StrainBasisSpec


def _array(value: Any, device: wp.Device) -> wp.array:
    """Copy a numerical fixture to the selected Warp device.

    Args:
        value: Host or JAX numerical array.
        device: Target test device.

    Returns:
        FP64 or INT32 Warp array owning its allocation.
    """
    value = np.asarray(value)
    dtype = wp.int32 if np.issubdtype(value.dtype, np.integer) else wp.float64
    return wp.array(value, dtype=dtype, device=device)


@pytest.fixture(scope="module")
def _stage_worlds() -> list[dict]:
    """Prepare two distinct structures with fixed and floating bases.

    Returns:
        Four sets of numerical operands, state and caller-owned stage buffers.
    """
    device = wp.get_device("cuda:0" if os.getenv("SOROMOX_TEST_CUDA") == "1" else "cpu")
    worlds = []
    for floating in (False, True):
        for segments in (1, 2):
            routing = ThreadlikeRouting.linear(
                intercept=jnp.array([[0.0, 0.012, 0.002], [0.0, -0.009, 0.006]]),
                slope=jnp.array([[0.0, 0.001, 0.0], [0.0, 0.0, 0.002]]),
                start_segment_index=(0, 0),
                end_segment_index=(segments - 1, segments - 1),
            )
            model = GVS.from_segments(
                tuple(
                    GVSSegment(
                        link=LinkSpec.circular(
                            length=0.1 + i * 0.025,
                            radius=0.015,
                            density=1040,
                            young_modulus=2e5,
                            shear_modulus=8e4,
                            reference_strain=[0, 0, 0, 1, 0, 0],
                        ),
                        joint=JointSpec.revolute(axis="y")
                        if i == 0
                        else JointSpec.prismatic(axis="x"),
                        basis=StrainBasisSpec(
                            type="legendre",
                            strain_selector=[True] * 6,
                            basis_order=[segments] * 6,
                        ),
                        num_gauss_points=5 + 2 * i,
                    )
                    for i in range(segments)
                ),
                floating_base=floating,
                actuators=ThreadlikeActuator.tendons(routing),
                backend="jax",
            )
            op = (GVSFloatingOperands if floating else GVSOperands).from_model(
                model, block_dim=1
            )
            count, width, cells = model.num_internal_dofs, op.max_dof, op.num_cells
            q = jnp.linspace(-0.004, 0.003, count)[None] * (1 + len(worlds) * 0.2)
            qd = q * -0.7
            local = _local_dynamics_terms(op, q, qd)
            names = (
                "joint_adjoint",
                "joint_adjoint_dot",
                "joint_tangent",
                "joint_tangent_dot_qd",
                "joint_velocity",
                "cell_adjoint",
                "cell_tangent_local",
                "cell_link_velocity",
                "cell_step_velocity",
                "cell_tangent_velocity_dot",
            )
            widths = (6, 6, count, 1, 1, 6, width, 1, 1, 1)
            values = {
                name: np.asarray(value).reshape(-1, columns)
                for name, value, columns in zip(names, local, widths, strict=True)
            }
            values.update(
                q=q,
                qd=qd,
                basis=np.asarray(op.joint_basis).reshape(-1, width),
                reference=op.joint_reference,
                local_to_global=op.joint_local_to_global,
                global_to_local=op.link_global_to_local,
                active_dofs=op.active_dofs_per_segment,
                inertia_upper_rows=op.inertia_upper_rows,
                inertia_upper_columns=op.inertia_upper_columns,
                weighted_masses=np.asarray(op.weighted_mass_diagonals).reshape(-1, 6),
                num_cells_array=np.array([cells], np.int32),
                num_quadrature_array=np.array([op.num_quadrature], np.int32),
                lanes_per_block=np.array([32 if device.is_cuda else 1], np.int32),
                gravity_base=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81]),
                gravity_world=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81]),
                base_pose=np.array(
                    [[0.1, -0.2, 0.05, np.sqrt(0.9475), 0.02, -0.01, 0.03]]
                ),
                velocity=np.concatenate(
                    [np.full((1, 6), 0.02), np.asarray(qd)], axis=1
                ),
            )
            shape_type = GVSFloatingPipelineShapes if floating else GVSPipelineShapes
            for name, shape in (
                shape_type.from_operands(op, batch_size=1).chain_outputs().items()
            ):
                values[name] = np.zeros(shape)
            kop = GVSKinematicsOperands.from_model(model, block_dim=1)
            for name in (
                "base_transform",
                "link_local_to_global",
                "link_global_to_local",
                "link_basis_rows",
                "segment_lengths",
                "segment_starts",
                "integration_points",
                "basis_type_index",
                "basis_active_params",
                "basis_order_params",
                "magnus_points",
            ):
                values[name] = getattr(kop, name)
            for name in ("link_reference_z1", "link_reference_z2"):
                values[name] = np.asarray(getattr(kop, name)).reshape(-1, 6)
            samples = 3 if segments == 1 else 35
            values.update(
                s=np.linspace(0, float(np.sum(model.segment_lengths)), samples)[None],
                scale_rotations_array=np.array(
                    [int(kop.scale_rotational_basis_by_length)], np.int32
                ),
                epsilons=np.array([kop.global_eps, kop.tangent_eps]),
            )
            shapes = GVSKinematicsShapes.from_operands(
                kop, batch_size=1, num_samples=samples
            )
            for allocation in (
                shapes.workspace(),
                shapes.jvp_workspace(),
                shapes.vjp_workspace(),
            ):
                for name, shape in allocation.items():
                    values[name] = np.zeros(shape)
            values.update(
                poses=np.zeros(shapes.pose_output()),
                jacobians=np.zeros(shapes.jacobian_output()),
                twists=np.zeros(shapes.twist_output()),
                generalized_force=np.zeros(shapes.generalized_force_output()),
                inertial_wrenches=np.random.default_rng(segments).normal(
                    size=shapes.twist_output()
                ),
            )
            aop = GVSThreadlikeOperands.from_model(model)
            for name in (
                "link_basis",
                "reference_strain",
                "physical_points",
                "integration_weights",
            ):
                values[name] = getattr(aop, name)
            for name, attr in (
                ("routing_intercepts", "intercepts"),
                ("routing_slopes", "slopes"),
                ("path_start_segments", "start_segments"),
                ("path_end_segments", "end_segments"),
                ("coordinate_scales", "coordinate_scales"),
            ):
                values[name] = getattr(aop.routing, attr)
            values.update(
                controls=np.array([[0.4, -0.2]]),
                strain_workspace=np.zeros((1, *aop.reference_strain.shape)),
            )
            data = {name: _array(value, device) for name, value in values.items()}
            data.update(
                segments=segments,
                samples=samples,
                dofs=count,
                floating=floating,
                device=device,
            )
            worlds.append(data)
    return worlds


@pytest.fixture
def stage_worlds(_stage_worlds: list[dict]) -> list[dict]:
    """Give each stage test independent state and workspace allocations.

    Args:
        _stage_worlds: Module-scoped model fixtures.

    Returns:
        Per-test copies of all Warp arrays and their dimensions.
    """
    return [
        {
            name: wp.clone(value) if isinstance(value, wp.array) else value
            for name, value in world.items()
        }
        for world in _stage_worlds
    ]


def _compare(
    worlds: list[dict],
    direct: Any,
    packed_kernel: Any,
    data_type: Any,
    counts: list[int],
    outputs: tuple[str, ...],
    *,
    tiled: int = 0,
    samples: bool = False,
    sample_group: int = 1,
) -> None:
    """Compare packed world views with independent direct launches.

    Args:
        worlds: Per-world arrays and fixture dimensions.
        direct: Homogeneous reference kernel.
        packed_kernel: Packed kernel under test.
        data_type: Native descriptor type consumed by the packed kernel.
        counts: Work items in each world.
        outputs: Array fields to compare after execution.
        tiled: Cooperative block width, or zero for scalar work.
        samples: Reference kernel uses a two-dimensional environment/sample grid.
        sample_group: Number of independent samples in a packed work item.

    Returns:
        None. Asserts numerical agreement and updates reference stage buffers.
    """
    device = worlds[0]["device"]
    descriptors, owners, expected = [], [], []
    for world, count in zip(worlds, counts, strict=True):
        descriptor = data_type()
        fields = {name: wp.clone(world[name]) for name in data_type.vars}
        owners.append(fields)
        for name, value in fields.items():
            setattr(descriptor, name, value)
        descriptors.append(descriptor)
        arguments = [world[name] for name in inspect.signature(direct.func).parameters]
        if tiled:
            wp.launch_tiled(
                direct, dim=count, inputs=arguments, block_dim=tiled, device=device
            )
        else:
            wp.launch(
                direct,
                dim=(1, count) if samples else count,
                inputs=arguments,
                device=device,
            )
        expected.append({name: world[name].numpy() for name in outputs})
    work = [
        (index, item, 0)
        for index, count in enumerate(counts)
        for item in range(0, count, sample_group)
    ]
    work.reverse()  # Exercise world IDs and local IDs independently of launch order.
    arguments = [
        wp.array(descriptors, dtype=data_type, device=device),
        wp.array(work, dtype=wp.vec3i, device=device),
    ]
    if tiled:
        wp.launch_tiled(
            packed_kernel,
            dim=len(work),
            inputs=arguments,
            block_dim=tiled,
            device=device,
        )
    else:
        wp.launch(
            packed_kernel,
            dim=(len(work), sample_group) if sample_group > 1 else len(work),
            inputs=arguments,
            block_dim=128,
            device=device,
        )
    for fields, reference in zip(owners, expected, strict=True):
        for name in outputs:
            assert np.isfinite(reference[name]).all()
            np.testing.assert_allclose(
                fields[name].numpy(),
                reference[name],
                rtol=3e-11,
                atol=3e-12,
                equal_nan=False,
            )


def test_packed_joint_and_chain_stages(stage_worlds: list[dict]) -> None:
    """Compare moving joints and all three chain descriptors across world sizes.

    Args:
        stage_worlds: Fixed and floating numerical fixtures.

    Returns:
        None. Checks per-world joint and chain outputs.
    """
    worlds = stage_worlds
    device = worlds[0]["device"]
    for world in worlds:
        rows, width = world["segments"] * 6, world["basis"].shape[1]
        world.update(
            adjoint=wp.zeros((rows, 6), dtype=wp.float64, device=device),
            adjoint_dot=wp.zeros((rows, 6), dtype=wp.float64, device=device),
            tangent_local=wp.zeros((rows, width), dtype=wp.float64, device=device),
            tangent_dot_qd=wp.zeros((rows, 1), dtype=wp.float64, device=device),
        )
    outputs = (
        "adjoint",
        "adjoint_dot",
        "tangent_local",
        "tangent_dot_qd",
        "joint_velocity",
    )
    _compare(
        worlds,
        joints.homogeneous_joint_terms_kernel,
        joints.packed_joint_terms_kernel,
        joints.JointTermsData,
        [w["segments"] for w in worlds],
        outputs,
    )
    if device.is_cuda:
        _compare(
            worlds,
            joints_cooperative.homogeneous_cooperative_joint_terms_kernel,
            joints_cooperative.packed_cooperative_joint_terms_kernel,
            joints.JointTermsData,
            [w["segments"] for w in worlds],
            outputs,
            tiled=32,
        )
    outputs = ("inertia", "coriolis_qd", "gravity_force")
    block = 32 if device.is_cuda else 1
    for floating, module, name, data_type in (
        (False, chain, "persistent_chain", chain.GVSChainData),
        (
            True,
            floating_chain,
            "floating_persistent_chain",
            floating_chain.GVSFloatingChainData,
        ),
    ):
        selected = [w for w in worlds if w["floating"] == floating]
        _compare(
            selected,
            getattr(module, "homogeneous_" + name + "_kernel"),
            getattr(module, "packed_" + name + "_kernel"),
            data_type,
            [1] * 2,
            outputs,
            tiled=block,
        )
    descriptors = []
    for world in worlds:
        data = packed.GVSMixedChainData()
        for name in packed.GVSMixedChainData.vars:
            value = world[name]
            setattr(data, name, wp.zeros_like(value) if name in outputs else value)
        descriptors.append(data)
    wp.launch_tiled(
        packed.packed_mixed_chain_kernel,
        dim=4,
        block_dim=block,
        inputs=[
            wp.array(descriptors, dtype=packed.GVSMixedChainData, device=device),
            wp.array([(i, 0, 0) for i in (3, 1, 2, 0)], dtype=wp.vec3i, device=device),
            wp.array(
                [int(w["floating"]) for w in worlds], dtype=wp.int32, device=device
            ),
        ],
        device=device,
    )
    for data, world in zip(descriptors, worlds, strict=True):
        for name in outputs:
            np.testing.assert_allclose(
                getattr(data, name).numpy(), world[name].numpy(), rtol=3e-11, atol=3e-12
            )


def test_packed_kinematics_stages(stage_worlds: list[dict]) -> None:
    """Compare pose, Jacobian, JVP and VJP descriptor wiring.

    Args:
        stage_worlds: Fixed and floating numerical fixtures.

    Returns:
        None. Checks independent stages and partial final sample groups.
    """
    worlds = stage_worlds
    stages = [
        (
            kinematics,
            "gvs_node_poses",
            kinematics.GVSNodePosesData,
            False,
            ("node_pose",),
        ),
        (
            kinematics,
            "gvs_node_states",
            kinematics.GVSNodeStatesData,
            False,
            ("node_pose", "node_jacobian"),
        ),
        (
            kinematics,
            "gvs_pose_samples",
            kinematics.GVSPoseSamplesData,
            True,
            ("poses",),
        ),
        (
            kinematics,
            "gvs_jacobian_samples",
            kinematics.GVSJacobianSamplesData,
            True,
            ("jacobians",),
        ),
        (
            kinematics,
            "gvs_samples",
            kinematics.GVSSamplesData,
            True,
            ("poses", "jacobians"),
        ),
        (
            directional,
            "gvs_node_pose_twist",
            directional.GVSNodePoseTwistData,
            False,
            ("node_pose", "node_twist"),
        ),
        (
            directional,
            "gvs_pose_twist_samples",
            directional.GVSPoseTwistSamplesData,
            True,
            ("poses", "twists"),
        ),
        (
            directional,
            "gvs_sample_vjp",
            directional.GVSSampleVJPData,
            True,
            ("node_wrench", "generalized_force"),
        ),
        (
            directional,
            "gvs_node_vjp",
            directional.GVSNodeVJPData,
            False,
            ("node_wrench", "generalized_force"),
        ),
    ]
    for module, name, data_type, samples, outputs in stages:
        counts = [w["samples"] if samples else 1 for w in worlds]
        _compare(
            worlds,
            getattr(module, "homogeneous_" + name + "_kernel"),
            getattr(module, "packed_" + name + "_kernel"),
            data_type,
            counts,
            outputs,
            samples=samples,
        )
        grouped = getattr(module, "packed_" + name + "_warp_kernel", None)
        if grouped is not None:
            _compare(
                worlds,
                getattr(module, "homogeneous_" + name + "_kernel"),
                grouped,
                data_type,
                counts,
                outputs,
                samples=samples,
                sample_group=32,
            )
    if worlds[0]["device"].is_cuda:
        for module, name, data_type, outputs in (
            (
                kinematics,
                "gvs_cooperative_node_states",
                kinematics.GVSNodeStatesData,
                ("node_pose", "node_jacobian"),
            ),
            (
                directional,
                "gvs_cooperative_node_vjp",
                directional.GVSCooperativeNodeVJPData,
                ("node_wrench", "generalized_force"),
            ),
        ):
            _compare(
                worlds,
                getattr(module, "homogeneous_" + name + "_kernel"),
                getattr(module, "packed_" + name + "_kernel"),
                data_type,
                [1] * 4,
                outputs,
                tiled=32,
            )


def test_packed_floating_composition_stages(stage_worlds: list[dict]) -> None:
    """Compare floating pose, Jacobian, twist and force composition.

    Args:
        stage_worlds: Fixtures with dense and directional kinematics buffers.

    Returns:
        None. Checks two differently sized floating worlds.
    """
    worlds = [dict(w) for w in stage_worlds if w["floating"]]
    for w in worlds:
        _compare(
            [w],
            kinematics.homogeneous_gvs_node_states_kernel,
            kinematics.packed_gvs_node_states_kernel,
            kinematics.GVSNodeStatesData,
            [1],
            ("node_pose", "node_jacobian"),
        )
        _compare(
            [w],
            kinematics.homogeneous_gvs_samples_kernel,
            kinematics.packed_gvs_samples_kernel,
            kinematics.GVSSamplesData,
            [w["samples"]],
            ("poses", "jacobians"),
            samples=True,
        )
        w.update(
            relative_pose=wp.clone(w["poses"]),
            jacobians_internal=w["jacobians"],
            twists_internal=_array(np.full(w["twists"].shape, 0.03), w["device"]),
            qd=w["velocity"],
            internal_generalized_force=_array(
                np.full((1, w["dofs"]), 0.04), w["device"]
            ),
            relative_wrenches=wp.zeros_like(w["inertial_wrenches"]),
            base_generalized_force=wp.zeros(
                (1, 6), dtype=wp.float64, device=w["device"]
            ),
            generalized_force=wp.zeros(
                (1, w["dofs"] + 6), dtype=wp.float64, device=w["device"]
            ),
            jacobians=wp.zeros(
                (1, w["samples"], 6, w["dofs"] + 6),
                dtype=wp.float64,
                device=w["device"],
            ),
        )
    for name, data_type, outputs in (
        ("compose_spatial_poses", composition.SpatialPoseCompositionData, ("poses",)),
        (
            "compose_spatial_jacobians",
            composition.SpatialJacobianCompositionData,
            ("jacobians",),
        ),
        (
            "compose_spatial_poses_and_twists",
            composition.SpatialPoseTwistCompositionData,
            ("poses", "twists"),
        ),
        (
            "compose_spatial_wrench_vjp",
            composition.SpatialWrenchProjectionData,
            ("relative_wrenches", "base_generalized_force"),
        ),
        (
            "assemble_floating_vjp",
            composition.FloatingBaseForceData,
            ("generalized_force",),
        ),
    ):
        counts = [
            w["dofs"] + 6 if name.startswith("assemble") else w["samples"]
            for w in worlds
        ]
        _compare(
            worlds,
            getattr(composition, "homogeneous_" + name + "_kernel"),
            getattr(composition, "packed_" + name + "_kernel"),
            data_type,
            counts,
            outputs,
            samples=True,
        )
    if worlds[0]["device"].is_cuda:
        _compare(
            worlds,
            composition.homogeneous_compose_spatial_jacobians_tiled_kernel,
            composition.packed_compose_spatial_jacobians_tiled_kernel,
            composition.SpatialJacobianCompositionData,
            [w["samples"] for w in worlds],
            ("jacobians",),
            tiled=32,
        )


def test_packed_threadlike_stages(stage_worlds: list[dict]) -> None:
    """Compare strain preparation, actuation matrices and generalized forces.

    Args:
        stage_worlds: Differently sized models with two routed tendons each.

    Returns:
        None. Checks scalar and cooperative threadlike descriptors.
    """
    worlds = [dict(w) for w in stage_worlds]
    _compare(
        worlds,
        threadlike.homogeneous_gvs_threadlike_strain_kernel,
        threadlike.packed_prepare_gvs_threadlike_strain_kernel,
        threadlike.GVSThreadlikeStrainData,
        [int(np.prod(w["strain_workspace"].shape[:-1])) for w in worlds],
        ("strain_workspace",),
    )
    for w in worlds:
        w["output"] = wp.zeros((1, w["dofs"], 2), dtype=wp.float64, device=w["device"])
    _compare(
        worlds,
        threadlike.homogeneous_gvs_threadlike_matrix_kernel,
        threadlike.packed_gvs_threadlike_matrix_kernel,
        threadlike.GVSThreadlikeMatrixData,
        [w["dofs"] * 2 for w in worlds],
        ("output",),
    )
    for w in worlds:
        w["output"] = wp.zeros((1, w["dofs"]), dtype=wp.float64, device=w["device"])
    _compare(
        worlds,
        threadlike.homogeneous_gvs_threadlike_force_kernel,
        threadlike.packed_gvs_threadlike_force_kernel,
        threadlike.GVSThreadlikeForceData,
        [w["dofs"] for w in worlds],
        ("output",),
    )
    if worlds[0]["device"].is_cuda:
        _compare(
            worlds,
            threadlike.homogeneous_gvs_threadlike_force_block_kernel,
            threadlike.packed_gvs_threadlike_force_block_kernel,
            threadlike.GVSThreadlikeForceBlockData,
            [w["segments"] for w in worlds],
            ("output",),
            tiled=128,
        )
