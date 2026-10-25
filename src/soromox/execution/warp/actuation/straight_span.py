"""Batched straight-span geometry from relative segment kinematics.

Geometry and Jacobians use FP64 and an explicit environment dimension. Launchers
write to caller-owned buffers and can be used directly by Warp integrators. Spans
must terminate at moving joints; the native path kernels assemble joint guides.
The kernels supply analytical length Jacobians. Derivatives through the public
routing solve use its JAX implementation.
"""

import warp as wp


@wp.func
def _transform(values: wp.array4d[wp.float64], environment: int, index: int):
    """Read a homogeneous transform from an ``(E, S, 4, 4)`` array.

    Args:
        values: Transform array.
        environment: Environment index.
        index: Transform index within the environment.

    Returns:
        The selected FP64 matrix.
    """
    result = wp.mat44d()
    for row in range(4):
        for column in range(4):
            result[row, column] = values[environment, index, row, column]
    return result


@wp.kernel(enable_backward=False)
def relative_intervals_kernel(
    links: wp.array4d[wp.float64],
    joints: wp.array4d[wp.float64],
    body_jacobians: wp.array4d[wp.float64],
    intervals: wp.array2d[wp.int32],
    transforms: wp.array4d[wp.float64],
    jacobians: wp.array4d[wp.float64],
):
    """Compose each distinct span interval and its body Jacobian.

    Args:
        links: Link transforms, ``(E, S, 4, 4)``.
        joints: Preceding joint transforms, ``(E, S, 4, 4)``. Interior joints
            of each interval must be fixed.
        body_jacobians: Link body Jacobians, angular-linear order, ``(E, S, 6, D)``.
        intervals: Inclusive first and last segment indices, ``(I, 2)``.
        transforms: Output interval transforms, ``(E, I, 4, 4)``.
        jacobians: Output interval body Jacobians, ``(E, I, 6, D)``.

    Returns:
        None. Launch with dimensions ``(E, I, D + 1)``; lane zero writes poses.
    """
    environment, interval, lane = wp.tid()
    first = intervals[interval, 0]
    last = intervals[interval, 1]
    if lane == 0:
        pose = _transform(links, environment, first)
        for segment in range(first + 1, last + 1):
            pose = pose @ (
                _transform(joints, environment, segment)
                @ _transform(links, environment, segment)
            )
        for row in range(4):
            for column in range(4):
                transforms[environment, interval, row, column] = pose[row, column]
    else:
        coordinate = lane - 1
        angular = wp.vec3d()
        linear = wp.vec3d()
        for component in range(3):
            angular[component] = body_jacobians[
                environment, first, component, coordinate
            ]
            linear[component] = body_jacobians[
                environment, first, component + 3, coordinate
            ]
        for segment in range(first + 1, last + 1):
            step = _transform(joints, environment, segment) @ _transform(
                links, environment, segment
            )
            rotation = wp.mat33d()
            translation = wp.vec3d()
            for row in range(3):
                translation[row] = step[row, 3]
                for column in range(3):
                    rotation[row, column] = step[row, column]
            linear = wp.transpose(rotation) @ (linear - wp.cross(translation, angular))
            angular = wp.transpose(rotation) @ angular
            for component in range(3):
                angular[component] += body_jacobians[
                    environment, segment, component, coordinate
                ]
                linear[component] += body_jacobians[
                    environment, segment, component + 3, coordinate
                ]
        for component in range(3):
            jacobians[environment, interval, component, coordinate] = angular[component]
            jacobians[environment, interval, component + 3, coordinate] = linear[
                component
            ]


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
    transform = _transform(transforms, environment, interval)
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
        entry = _transform(entries, environment, intervals[interval, 0])
        exit_pose = _transform(exits, environment, intervals[interval, 1])
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
