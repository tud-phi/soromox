"""Native free-space actuation from configurations and filament tensions.

``StraightSpanActuation`` owns reusable Warp buffers. Its steady-state methods
launch Warp kernels only and can be recorded in a CUDA graph. Numerical model
and routing operands are prepared once when the executor is constructed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from soromox.execution.warp.common.relative_kinematics import (
    RelativeKinematicsOperands,
    launch_kinematics,
    read_transform,
    relative_intervals_kernel,
)


def interval_workspace_count(routing):
    """Count distinct intervals requiring composition beyond a single link.

    Args:
        routing (StraightSpanRouting): Guide topology, shared by all environments.

    Returns:
        int: Number of distinct interval workspace slots, or zero when every
            span can reuse its link transform and Jacobian directly.
    """
    bounds = tuple(
        zip(routing.spans.start_segment_index, routing.spans.end_segment_index)
    )
    return 0 if all(first == last for first, last in bounds) else len(set(bounds))


def routing_topology(routing, model):
    """Prepare shared intervals and predecessor spans in input order.

    Args:
        routing (StraightSpanRouting): Validated span topology and guide offsets.
        model (RelativeKinematicsOperands): Prepared joint generators used to
            identify guides that coincide for every joint configuration.

    Returns:
        tuple[Array, ...]: Inclusive interval bounds ``(I, 2)``, span-to-interval
            indices ``(S,)``, path indices ``(S,)``, predecessor indices ``(S,)``
            with ``-1`` for the first span, and invariant-coincidence flags
            ``(S,)``. Indices use int32 and flags use bool. Single-link spans
            index the link buffers directly; other intervals preserve first-use
            order. Here S denotes routing spans.
    """
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
    """Borrowed arrays for straight threadlike paths and joint connections.

    Keep each referenced allocation alive for launches and captured graphs.
    E denotes environments, I intervals, L links, S spans, and D internal DOFs.

    Attributes:
        transforms: Relative interval transforms ``(E, I, 4, 4)``.
        jacobians: Interval body Jacobians, angular-linear order, ``(E, I, 6, D)``.
        joints: Relative preceding-joint transforms ``(E, L, 4, 4)``.
        joint_jacobians: Joint body Jacobians ``(E, L, 6, D)``.
        coincident: Invariant coincidence of each preceding connection ``(S,)``.
        intervals: Inclusive first/last link indices ``(I, 2)``.
        span_intervals: Interval index for each span ``(S,)``.
        paths: Physical path index for each span ``(S,)``.
        predecessors: Previous span on the same path, or -1, ``(S,)``.
        start_offsets: Guides in link-entry frames ``(S, 3)``, in metres.
        end_offsets: Guides in link-exit frames ``(S, 3)``, in metres.
    """

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
        read_transform(data.transforms, environment, interval), start, end
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
            read_transform(data.joints, environment, segment), upstream, start
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
    """Write minus the path-length derivative in the requested coordinates.

    Args:
        data: Prepared interval and guide arrays; see ``StraightPathData``.
        velocity_coordinates: Whether matrix rows represent host velocities.
            Otherwise rows represent configuration coordinates, including the
            base quaternion when present.
        matrix: Caller-owned FP64 output ``(E, V, P)`` or ``(E, Q, P)`` for
            environments, velocities/configurations, and physical paths.

    Returns:
        None. Launch with ``dim=matrix.shape``. Every entry is overwritten.
        Base velocity rows are zero; invalid internal geometry yields NaNs.
    """
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
    """Accumulate span and joint-guide forces into a generalized-force buffer.

    Args:
        data: Prepared interval and guide arrays; see ``StraightPathData``.
        tensions: Signed axial loads ``(E, P)``; positive denotes tension.
        force: Caller-owned FP64 output ``(E, V)``. Clear it before launching.

    Returns:
        None. Launch with ``dim=(E, S, V)`` for environments, spans, and
        velocities. FP64 atomic additions accumulate the negative path-length
        derivatives times loads; invalid internal derivatives propagate NaNs.
    """
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
def guide_geometry_kernel(
    data: StraightPathData, geometry: wp.array3d[wp.float64], valid: wp.array2d[wp.bool]
):
    """Evaluate connections between consecutive spans of each threadlike path.

    Args:
        data: Prepared relative interval and joint-guide data.
        geometry: FP64 output ``(E, S, 4)``: upstream-frame displacement followed
            by length in metres. Spans without predecessors receive zeros.
        valid: Boolean output ``(E, S)`` indicating noncollapsed or invariant
            coincident connections.

    Returns:
        None. Launch with ``dim=(E, S)``; all output entries are overwritten.
    """
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
            read_transform(data.joints, environment, segment), start, end
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
    """Sum physical spans and joint connections for each threadlike path.

    Args:
        data: Prepared relative interval and joint-guide data.
        lengths: FP64 output ``(E, P)`` in metres, for environments and paths.

    Returns:
        None. Launch with ``dim=lengths.shape``; all entries are overwritten.
    """
    environment, path = wp.tid()
    value = _path_column(data, environment, path, -1)
    lengths[environment, path] = value[0]


def prepare_path_data(q, operands, topology, offsets, workspace):
    """Evaluate local and shared-interval kinematics in caller-owned buffers.

    Args:
        q (wp.array): Contiguous FP64 Warp configurations ``(E, Q)``.
        operands: Warp arrays for basis1, basis2, reference1, reference2, and
            distances, following ``RelativeKinematicsOperands``.
        topology: Warp arrays returned by ``routing_topology`` in the same order.
        offsets: Start and end guide arrays, each ``(S, 3)``.
        workspace: Warp buffers in ``workspace_shapes`` dictionary order.

    Returns:
        StraightPathData: Views into the supplied buffers and guide data.
            Single-link intervals reuse link buffers. Launches are enqueued
            on the current stream of q's device.
    """
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
    """Describe reusable strain, local-frame, and interval buffers.

    Args:
        batch (int): Number of environments.
        segments (int): Number of robot links with preceding joints.
        internal_dofs (int): Configuration coordinates excluding the base.
        intervals (int): Number of interval workspace slots; zero reuses links.
        cells (int): Padded integration-cell count, including the joint cell.

    Returns:
        dict[str, tuple[int, ...]]: FP64 buffer shapes in launch-argument order:
            strains, links, link_jacobians, joints, joint_jacobians, transforms,
            and jacobians. Preserve this order when allocating the workspace.
    """
    return {
        "strains": (batch, segments * cells, 2, 6),
        "links": (batch, segments, 4, 4),
        "link_jacobians": (batch, segments, 6, internal_dofs),
        "joints": (batch, segments, 4, 4),
        "joint_jacobians": (batch, segments, 6, internal_dofs),
        "transforms": (batch, intervals, 4, 4),
        "jacobians": (batch, intervals, 6, internal_dofs),
    }


class StraightSpanActuation:
    """Native Warp threadlike actuation for straight spans between guides.

    Construct before CUDA graph capture. Outputs alias reusable buffers; copy
    results that must survive subsequent calls. Calls on one instance must be
    ordered on the same stream. Rebuild after changing model or guide parameters.

    Args:
        robot (SoftRobot): PlanarPCS, PCS, or GVS host with FP64 model data.
        routing (StraightSpanRouting): Prescribed guides and span connectivity.
        batch_size (int): Positive number of environments sharing model and guides.
        device (str | wp.Device | None): Warp device or device name; None uses Warp's current device.

    Attributes:
        force_buffer: Reused generalized forces ``(batch_size, num_velocities)``.
        matrix_buffer: Reused actuation matrices
            ``(batch_size, num_velocities, num_paths)``.

    Raises:
        ValueError: Invalid batch size, guide geometry, topology, or disabled
            JAX 64-bit mode during model preparation.
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
        model = RelativeKinematicsOperands.from_model(robot)
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

    def actuation_force(self, q, tensions):
        """Evaluate generalized forces for a batch of configurations and loads.

        Args:
            q (wp.array): Contiguous FP64 Warp configurations ``(batch_size, num_coordinates)``.
            tensions (wp.array): Contiguous FP64 signed axial loads ``(batch_size, num_paths)``.
                Positive values denote tension; negative values denote ideal axial
                compression below buckling. Both inputs must use this executor's device.

        Returns:
            result (wp.array): Reused ``force_buffer``, shape ``(batch_size, num_velocities)``.
                No actuation matrix or world guide geometry is formed. Synchronize
                the stream before reading the result on the host.

        Raises:
            ValueError: An input has the wrong shape, dtype, device, or contiguity.
        """
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

    def actuation_matrix(self, q):
        """Evaluate the mapping from signed axial loads to generalized forces.

        Args:
            q (wp.array): Contiguous FP64 Warp configurations ``(batch_size, num_coordinates)``
                on this executor's device.

        Returns:
            result (wp.array): Reused ``matrix_buffer``, shape
                ``(batch_size, num_velocities, num_paths)``. Base rows are zero and
                invalid internal derivatives are NaN. Synchronize before host reads.

        Raises:
            ValueError: The input has the wrong shape, dtype, device, or contiguity.
        """
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


@wp.kernel(enable_backward=False)
def straight_spans_kernel(
    transforms: wp.array4d[wp.float64],
    body_jacobians: wp.array4d[wp.float64],
    entries: wp.array4d[wp.float64],
    exits: wp.array4d[wp.float64],
    intervals: wp.array2d[wp.int32],
    span_intervals: wp.array[wp.int32],
    start_offsets: wp.array2d[wp.float64],
    end_offsets: wp.array2d[wp.float64],
    geometry: wp.array3d[wp.float64],
    jacobians: wp.array3d[wp.float64],
    valid: wp.array2d[wp.bool],
):
    """Evaluate guide endpoints, exact chord lengths, and analytical derivatives.

    Args:
        transforms: Shared relative interval transforms, ``(E, I, 4, 4)``.
        body_jacobians: Interval body Jacobians, ``(E, I, 6, D)``.
        entries: World link entry poses, ``(E, S, 4, 4)``.
        exits: World link exit poses, ``(E, S, 4, 4)``.
        intervals: Inclusive segment bounds, ``(I, 2)``.
        span_intervals: Interval index for each span, ``(P,)``.
        start_offsets: Entry guide offsets, ``(P, 3)``.
        end_offsets: Exit guide offsets, ``(P, 3)``.
        geometry: Output world start/end positions and length, ``(E, P, 7)``.
        jacobians: Output length configuration derivatives, ``(E, P, D)``.
        valid: Output finite, noncollapsed displacement flags, ``(E, P)``.

    Returns:
        None. Launch with dimensions ``(E, P, D + 1)``. Lane zero writes geometry;
        the other lanes write Jacobian columns. Collapsed spans receive NaNs.
    """
    environment, span, lane = wp.tid()
    interval = span_intervals[span]
    transform = read_transform(transforms, environment, interval)
    start_offset = wp.vec3d()
    end_offset = wp.vec3d()
    displacement = wp.vec3d()
    rotation = wp.mat33d()
    for row in range(3):
        start_offset[row] = start_offsets[span, row]
        end_offset[row] = end_offsets[span, row]
        displacement[row] = transform[row, 3] - start_offset[row]
        for column in range(3):
            rotation[row, column] = transform[row, column]
    displacement += rotation @ end_offset
    squared = wp.dot(displacement, displacement)
    length = wp.float64(0.0)
    if squared > wp.float64(0.0):
        length = wp.sqrt(squared)
    accepted = wp.isfinite(displacement) and length > wp.float64(1e-12)
    if lane == 0:
        entry = read_transform(entries, environment, intervals[interval, 0])
        exit_pose = read_transform(exits, environment, intervals[interval, 1])
        valid[environment, span] = accepted
        geometry[environment, span, 6] = length
        for row in range(3):
            start = entry[row, 3]
            end = exit_pose[row, 3]
            for column in range(3):
                start += entry[row, column] * start_offset[column]
                end += exit_pose[row, column] * end_offset[column]
            geometry[environment, span, row] = start
            geometry[environment, span, row + 3] = end
    else:
        coordinate = lane - 1
        derivative = wp.float64(wp.nan)
        if accepted:
            angular = wp.vec3d()
            linear = wp.vec3d()
            for component in range(3):
                angular[component] = body_jacobians[
                    environment, interval, component, coordinate
                ]
                linear[component] = body_jacobians[
                    environment, interval, component + 3, coordinate
                ]
            variation = rotation @ (linear + wp.cross(angular, end_offset))
            derivative = wp.dot(displacement, variation) / length
        jacobians[environment, span, coordinate] = derivative
