"""Native free-space actuation from configurations and filament tensions.

``StraightSpanPipeline`` owns reusable Warp buffers. Its steady-state methods
launch Warp kernels only and can be recorded in a CUDA graph. Numerical model
and routing operands are prepared once when the pipeline is constructed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from soromox.execution.warp.actuation.straight_span import (
    _transform,
    relative_intervals_kernel,
)
from soromox.execution.warp.actuation.straight_span_kinematics import (
    StraightKinematicsOperands,
    relative_kinematics_kernel,
    strain_samples_kernel,
)
from soromox.execution.warp.common.rotations import _quaternion_rotation_matrix


def interval_workspace_count(routing):
    """Reuse link buffers directly when all spans stay inside one segment."""
    bounds = tuple(
        zip(routing.spans.start_segment_index, routing.spans.end_segment_index)
    )
    return 0 if all(first == last for first, last in bounds) else len(set(bounds))


def routing_topology(routing, model):
    """Prepare shared intervals and predecessor spans, preserving input order."""
    bounds = tuple(
        zip(routing.spans.start_segment_index, routing.spans.end_segment_index)
    )
    local = all(first == last for first, last in bounds)
    intervals = (
        tuple((i, i) for i in range(max((end for _, end in bounds), default=-1) + 1))
        if local
        else tuple(dict.fromkeys(bounds))
    )
    previous, predecessors, coincident = {}, [], []
    for index, path in enumerate(routing.spans.span_path_index):
        predecessor = previous.get(path, -1)
        predecessors.append(predecessor)
        fixed = jnp.array(False)
        if predecessor >= 0:
            segment = routing.spans.start_segment_index[index]
            start = routing.spans.start_offset[index]
            end = routing.spans.end_offset[predecessor]
            generators = jnp.concatenate(
                (model.basis1[segment, 0], model.reference1[segment, 0, :, None]),
                axis=1,
            )
            fixed_point = jnp.cross(generators[:3].T, start).T + generators[3:]
            fixed = jnp.all(start == end) & jnp.all(fixed_point == 0)
        coincident.append(fixed)
        previous[path] = index
    return (
        jnp.asarray(intervals, dtype=jnp.int32).reshape((-1, 2)),
        jnp.asarray([intervals.index(bound) for bound in bounds], dtype=jnp.int32),
        jnp.asarray(routing.spans.span_path_index, dtype=jnp.int32),
        jnp.asarray(predecessors, dtype=jnp.int32),
        jnp.asarray(coincident, dtype=bool),
    )


@wp.func
def _chord(
    transform: wp.mat44d,
    upstream: wp.vec3d,
    downstream: wp.vec3d,
):
    """Return a displacement and rotation in the upstream frame."""
    rotation = wp.mat33d()
    displacement = wp.vec3d()
    for row in range(3):
        displacement[row] = transform[row, 3] - upstream[row]
        for column in range(3):
            rotation[row, column] = transform[row, column]
    return displacement + rotation @ downstream, rotation


@wp.func
def _offset(offsets: wp.array2d[wp.float64], index: int):
    """Read a material-frame guide offset."""
    return wp.vec3d(offsets[index, 0], offsets[index, 1], offsets[index, 2])


@wp.func
def _distance(displacement: wp.vec3d):
    """Match the reference distance at zero and nonfinite squared norms."""
    squared = wp.dot(displacement, displacement)
    length = wp.float64(0.0)
    if squared > wp.float64(0.0):
        length = wp.sqrt(squared)
    return length


@wp.func
def _length_gradient(
    displacement: wp.vec3d,
    rotation: wp.mat33d,
    downstream: wp.vec3d,
    jacobian: wp.array4d[wp.float64],
    environment: int,
    interval: int,
    coordinate: int,
):
    """Return chord length and one derivative; reject collapsed geometry."""
    length = _distance(displacement)
    gradient = wp.float64(wp.nan)
    if wp.isfinite(displacement) and length > wp.float64(1e-12):
        gradient = wp.float64(0.0)
        if coordinate >= 0:
            angular, linear = wp.vec3d(), wp.vec3d()
            for row in range(3):
                angular[row] = jacobian[environment, interval, row, coordinate]
                linear[row] = jacobian[environment, interval, row + 3, coordinate]
            gradient = (
                wp.dot(
                    displacement, rotation @ (linear + wp.cross(angular, downstream))
                )
                / length
            )
    return wp.vec2d(length, gradient)


@wp.struct
class StraightPathData:
    """Non-owning arrays for path and joint-guide evaluation."""

    transforms: wp.array4d[wp.float64]
    jacobians: wp.array4d[wp.float64]
    joints: wp.array4d[wp.float64]
    joint_jacobians: wp.array4d[wp.float64]
    coincident: wp.array[wp.bool]
    intervals: wp.array2d[wp.int32]
    span_intervals: wp.array[wp.int32]
    paths: wp.array[wp.int32]
    predecessors: wp.array[wp.int32]
    start_offsets: wp.array2d[wp.float64]
    end_offsets: wp.array2d[wp.float64]


@wp.func
def _span_column(data: StraightPathData, environment: int, span: int, coordinate: int):
    """Evaluate one span and its preceding connection without scanning other paths."""
    result = wp.vec2d()
    interval = data.span_intervals[span]
    start = _offset(data.start_offsets, span)
    end = _offset(data.end_offsets, span)
    displacement, rotation = _chord(
        _transform(data.transforms, environment, interval), start, end
    )
    result += _length_gradient(
        displacement,
        rotation,
        end,
        data.jacobians,
        environment,
        interval,
        coordinate,
    )
    previous = data.predecessors[span]
    if previous >= 0:
        segment = data.intervals[interval, 0]
        upstream = _offset(data.end_offsets, previous)
        displacement, rotation = _chord(
            _transform(data.joints, environment, segment), upstream, start
        )
        if not data.coincident[span]:
            result += _length_gradient(
                displacement,
                rotation,
                start,
                data.joint_jacobians,
                environment,
                segment,
                coordinate,
            )
        else:
            result += wp.vec2d(_distance(displacement), wp.float64(0.0))
    return result


@wp.func
def _path_column(data: StraightPathData, environment: int, path: int, coordinate: int):
    """Sum span and joint-connection lengths and one internal derivative."""
    result = wp.vec2d()
    for span in range(data.paths.shape[0]):
        if data.paths[span] == path:
            result += _span_column(data, environment, span, coordinate)
    return result


@wp.kernel(enable_backward=False)
def path_matrix_kernel(
    data: StraightPathData, velocity_coordinates: bool, matrix: wp.array3d[wp.float64]
):
    """Write minus the path derivative in configuration or velocity coordinates."""
    environment, coordinate, path = wp.tid()
    internal = coordinate - (matrix.shape[1] - data.jacobians.shape[3])
    if velocity_coordinates and internal < 0:
        matrix[environment, coordinate, path] = wp.float64(0.0)
    else:
        value = _path_column(data, environment, path, internal)
        matrix[environment, coordinate, path] = -value[1]


@wp.kernel(enable_backward=False)
def path_force_kernel(
    data: StraightPathData,
    tensions: wp.array2d[wp.float64],
    force: wp.array2d[wp.float64],
):
    """Accumulate generalized forces directly, without a path Jacobian buffer."""
    environment, span, coordinate = wp.tid()
    internal = coordinate - (force.shape[1] - data.jacobians.shape[3])
    path = data.paths[span]
    derivative = wp.float64(0.0)
    if internal >= 0:
        column = _span_column(data, environment, span, internal)
        derivative = column[1]
    wp.atomic_add(
        force, environment, coordinate, -derivative * tensions[environment, path]
    )


@wp.kernel(enable_backward=False)
def routing_boundaries_kernel(
    q: wp.array2d[wp.float64],
    data: StraightPathData,
    links: wp.array4d[wp.float64],
    fixed_base: wp.array2d[wp.float64],
    entries: wp.array4d[wp.float64],
    exits: wp.array4d[wp.float64],
):
    """Compose world endpoints only when a full routing solution is requested."""
    environment = wp.tid()
    base_count = q.shape[1] - data.jacobians.shape[3]
    pose = wp.identity(n=4, dtype=wp.float64)
    if base_count == 0:
        for row in range(4):
            for column in range(4):
                pose[row, column] = fixed_base[row, column]
    elif base_count == 3:
        angle = q[environment, 0]
        pose[0, 0] = wp.cos(angle)
        pose[0, 1] = -wp.sin(angle)
        pose[1, 0] = wp.sin(angle)
        pose[1, 1] = wp.cos(angle)
        pose[0, 3] = q[environment, 1]
        pose[1, 3] = q[environment, 2]
    else:
        rotation = _quaternion_rotation_matrix(q, environment)
        for row in range(3):
            pose[row, 3] = q[environment, row + 4]
            for column in range(3):
                pose[row, column] = rotation[row, column]
    for segment in range(links.shape[1]):
        entry = pose @ _transform(data.joints, environment, segment)
        pose = entry @ _transform(links, environment, segment)
        for row in range(4):
            for column in range(4):
                entries[environment, segment, row, column] = entry[row, column]
                exits[environment, segment, row, column] = pose[row, column]


@wp.kernel(enable_backward=False)
def guide_geometry_kernel(
    data: StraightPathData, geometry: wp.array3d[wp.float64], valid: wp.array2d[wp.bool]
):
    """Write joint connection displacements and lengths in downstream span order."""
    environment, span = wp.tid()
    previous = data.predecessors[span]
    displacement = wp.vec3d()
    length = wp.float64(0.0)
    accepted = True
    if previous >= 0:
        segment = data.intervals[data.span_intervals[span], 0]
        start, end = (
            _offset(data.end_offsets, previous),
            _offset(data.start_offsets, span),
        )
        displacement, rotation = _chord(
            _transform(data.joints, environment, segment), start, end
        )
        coincident = data.coincident[span]
        length = _distance(displacement)
        accepted = coincident or length > wp.float64(1e-12)
    for row in range(3):
        geometry[environment, span, row] = displacement[row]
    geometry[environment, span, 3] = length
    valid[environment, span] = accepted


@wp.kernel(enable_backward=False)
def path_lengths_kernel(data: StraightPathData, lengths: wp.array2d[wp.float64]):
    """Sum physical span and connecting guide lengths for each path."""
    environment, path = wp.tid()
    value = _path_column(data, environment, path, -1)
    lengths[environment, path] = value[0]


def launch_kinematics(
    q, operands, strains, links, link_jacobians, joints, joint_jacobians
):
    """Launch native relative kinematics with prepared Warp operand arrays."""
    wp.launch(
        strain_samples_kernel,
        dim=(q.shape[0], links.shape[1], operands[4].shape[1], 6),
        inputs=[q, *operands[:4]],
        outputs=[strains],
        device=q.device,
    )
    wp.launch(
        relative_kinematics_kernel,
        dim=(q.shape[0], links.shape[1], link_jacobians.shape[3] + 1),
        inputs=[strains, operands[0], operands[1], operands[4]],
        outputs=[links, link_jacobians, joints, joint_jacobians],
        device=q.device,
    )


def prepare_path_data(q, operands, topology, offsets, workspace):
    """Evaluate local and shared-interval kinematics in caller-owned buffers."""
    strains, links, link_jacobians, joints, joint_jacobians, transforms, jacobians = (
        workspace
    )
    launch_kinematics(
        q, operands, strains, links, link_jacobians, joints, joint_jacobians
    )
    intervals, span_intervals, paths, predecessors, coincident = topology
    if transforms.shape[1]:
        wp.launch(
            relative_intervals_kernel,
            dim=(q.shape[0], intervals.shape[0], link_jacobians.shape[3] + 1),
            inputs=[links, joints, link_jacobians, intervals],
            outputs=[transforms, jacobians],
            device=q.device,
        )
    data = StraightPathData()
    data.transforms = transforms if transforms.shape[1] else links
    data.jacobians = jacobians if transforms.shape[1] else link_jacobians
    data.joints, data.joint_jacobians = joints, joint_jacobians
    data.coincident = coincident
    data.intervals, data.span_intervals = intervals, span_intervals
    data.paths, data.predecessors = paths, predecessors
    data.start_offsets, data.end_offsets = offsets
    return data


def workspace_shapes(batch, segments, internal_dofs, intervals, cells):
    """Return reusable shared-strain, local, and interval workspace shapes."""
    return {
        "strains": (batch, segments * cells, 2, 6),
        "links": (batch, segments, 4, 4),
        "link_jacobians": (batch, segments, 6, internal_dofs),
        "joints": (batch, segments, 4, 4),
        "joint_jacobians": (batch, segments, 6, internal_dofs),
        "transforms": (batch, intervals, 4, 4),
        "jacobians": (batch, intervals, 6, internal_dofs),
    }


class StraightSpanPipeline:
    """Reusable native Warp actuation for a batch sharing model and guides.

    Construct outside capture, then call ``force(q, tensions)`` or ``matrix(q)``
    with contiguous FP64 Warp arrays. Outputs alias reusable buffers; copy them
    if they must survive subsequent calls. Calls on one instance must be ordered
    on the same stream. Rebuild after changing model or routing parameters.
    No JAX evaluation, device-buffer allocation, or numerical host transfer occurs
    during these calls.
    """

    def __init__(self, robot, routing, *, batch_size=1, device=None):
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not jax.config.jax_enable_x64:
            raise ValueError("Native Warp routing requires JAX 64-bit mode.")
        routing.validate_structure_for_robot(robot)
        routing.validate_values_for_robot(robot)
        model = StraightKinematicsOperands.from_model(robot)
        self.device = wp.get_device(device)
        self.batch_size = batch_size
        self.num_coordinates = robot.num_coordinates
        self.num_velocities = robot.num_velocities
        self.num_paths = routing.spans.num_paths

        def array(value, dtype=wp.float64):
            return wp.array(np.asarray(value), dtype=dtype, device=self.device)

        self.operands = tuple(
            array(value)
            for value in (
                model.basis1,
                model.basis2,
                model.reference1,
                model.reference2,
                model.distances,
            )
        )
        self.topology = tuple(
            array(value, wp.bool if index == 4 else wp.int32)
            for index, value in enumerate(routing_topology(routing, model))
        )
        self.offsets = (
            array(routing.spans.start_offset),
            array(routing.spans.end_offset),
        )
        self.workspace = tuple(
            wp.empty(shape, dtype=wp.float64, device=self.device)
            for shape in workspace_shapes(
                batch_size,
                robot.num_segments,
                robot.num_internal_dofs,
                interval_workspace_count(routing),
                model.distances.shape[1],
            ).values()
        )
        self.force_buffer = wp.empty(
            (batch_size, self.num_velocities), dtype=wp.float64, device=self.device
        )
        self.matrix_buffer = wp.empty(
            (batch_size, self.num_velocities, self.num_paths),
            dtype=wp.float64,
            device=self.device,
        )

    def _check(self, array, shape, name):
        """Validate inputs before unchecked kernel indexing, including capture."""
        if (
            array.shape != shape
            or array.dtype != wp.float64
            or array.device != self.device
            or not array.is_contiguous
        ):
            raise ValueError(
                f"{name} must be contiguous FP64 with shape {shape} on {self.device}."
            )

    def _prepare(self, q):
        self._check(q, (self.batch_size, self.num_coordinates), "q")
        return prepare_path_data(
            q, self.operands, self.topology, self.offsets, self.workspace
        )

    def force(self, q, tensions):
        """Write and return ``(E, V)`` generalized forces from ``(E, P)`` tensions."""
        self._check(tensions, (self.batch_size, self.num_paths), "tensions")
        data = self._prepare(q)
        self.force_buffer.zero_()
        if self.num_velocities and self.topology[2].shape[0]:
            wp.launch(
                path_force_kernel,
                dim=(self.batch_size, self.topology[2].shape[0], self.num_velocities),
                inputs=[data, tensions],
                outputs=[self.force_buffer],
                device=self.device,
            )
        return self.force_buffer

    def matrix(self, q):
        """Write and return ``(E, V, P)`` actuation matrices."""
        data = self._prepare(q)
        if self.num_velocities and self.num_paths:
            wp.launch(
                path_matrix_kernel,
                dim=self.matrix_buffer.shape,
                inputs=[data, True],
                outputs=[self.matrix_buffer],
                device=self.device,
            )
        return self.matrix_buffer
