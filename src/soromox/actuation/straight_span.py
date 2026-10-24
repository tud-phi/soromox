"""Closed-form routing between prescribed guides without wall contact."""

import equinox as eqx
import jax
import jax.numpy as jnp

from soromox.utils.geometry.kinematics import segment_boundary_poses
from soromox.utils.lie_algebra import se3

from ._routing_geometry import guide_kinematics, validate_span
from .implicit_threadlike import (
    ConfigurationDependentThreadlikeRouting,
    RoutingGuideSolution,
    RoutingSolution,
    RoutingSpanParams,
    StraightSpanSolution,
    _contains_tracer,
    _guide_length,
)


class StraightSpanRouting(ConfigurationDependentThreadlikeRouting):
    """Straight filaments between guides that move with the robot.

    A path is one physical filament. A span is the straight portion between
    two prescribed guides. Adjacent spans include their joint guide connection.
    The user must ensure that wall contact is irrelevant throughout operation.
    Guide kinematics determine the path directly. The model describes taut
    frictionless tendons and ideal straight axial members with negligible
    bending stiffness, operating below their buckling load.

    Attributes:
        params: Shared guide offsets and span connectivity.
    """

    params: RoutingSpanParams

    @property
    def spans(self):
        """Return the shared guide geometry and connectivity parameters.

        Returns:
            result (RoutingSpanParams): This strategy's guide geometry.
        """
        return self.params

    def with_params(self, params):
        """Replace numerical guide geometry while retaining topology.

        Args:
            params (RoutingSpanParams): Replacement with identical topology.

        Returns:
            result (StraightSpanRouting): Validated replacement parameters.
        """
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return eqx.tree_at(lambda routing: routing.params, self, params)

    def validate_structure_for_robot(self, robot):
        """Validate span connectivity and prescribed guides at moving joints.

        Args:
            robot (SoftRobot): Continuum host with segment topology.

        Returns:
            result (None): Invalid connectivity raises ValueError.
        """
        self.spans.validate_structure_for_robot(robot)
        for start, end in zip(
            self.spans.start_segment_index, self.spans.end_segment_index
        ):
            validate_span(robot, start, end)

    def validate_values_for_robot(self, robot):
        """Validate concrete guide offsets and segment lengths.

        Args:
            robot (SoftRobot): Continuum host with material frame conventions.

        Returns:
            result (None): Invalid geometry raises ValueError.
        """
        self.spans.validate_values_for_robot(robot)
        if not bool(
            jnp.all(jnp.isfinite(robot.segment_length) & (robot.segment_length > 0))
        ):
            raise ValueError("Routing requires positive finite segment lengths.")

    def solve_routing(self, robot, configuration, *, initial_guess=None):
        """Resolve straight spans and their analytical length derivatives.

        Args:
            robot (SoftRobot): Host supplying local segment and joint guide kinematics.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            initial_guess (RoutingSolution | None): Unused; accepted for the
                common routing interface.

        Returns:
            result (RoutingSolution): StraightSpanSolution entries. Nonfinite
                or collapsed spans report failed convergence and NaN derivatives.
        """
        del initial_guess
        configuration = jnp.asarray(configuration, dtype=float)
        self.validate_structure_for_robot(robot)
        if not _contains_tracer((self.params, robot)):
            with jax.ensure_compile_time_eval():
                self.validate_values_for_robot(robot)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        if self.spans.num_spans == 0:
            return RoutingSolution((), jnp.zeros(0, dtype=configuration.dtype), ())
        links, body_jacobians, joints = robot.relative_segment_kinematics(configuration)
        entries, exits = segment_boundary_poses(
            robot.base_transform_from_configuration(configuration), links, joints
        )
        relative = {}
        for first, last in zip(
            self.spans.start_segment_index, self.spans.end_segment_index
        ):
            if (first, last) in relative:
                continue
            transform, derivative = links[first], body_jacobians[first]
            for segment in range(first + 1, last + 1):
                step = joints[segment] @ links[segment]
                transform = transform @ step
                derivative = (
                    se3.adjoint_inverse(step) @ derivative + body_jacobians[segment]
                )
            relative[first, last] = transform, derivative

        def evaluate(
            transform, body_jacobian, start_offset, end_offset, entry, exit_pose
        ):
            """Return guide positions, length, derivative, and validity for one span."""
            displacement = (
                transform[:3, 3] + transform[:3, :3] @ end_offset - start_offset
            )
            local_jacobian = transform[:3, :3] @ (
                body_jacobian[3:] + jnp.cross(body_jacobian[:3].T, end_offset).T
            )
            start = entry[:3, 3] + entry[:3, :3] @ start_offset
            end = exit_pose[:3, 3] + exit_pose[:3, :3] @ end_offset
            length = _guide_length(displacement, jnp.array(False))
            valid = jnp.all(jnp.isfinite(displacement)) & (length > 1e-12)
            derivative = displacement @ local_jacobian / jnp.where(valid, length, 1.0)
            return StraightSpanSolution(
                start, end, length, jnp.where(valid, derivative, jnp.nan), valid
            )

        intervals = tuple(
            zip(self.spans.start_segment_index, self.spans.end_segment_index)
        )
        evaluated = jax.vmap(evaluate)(
            jnp.stack([relative[interval][0] for interval in intervals]),
            jnp.stack([relative[interval][1] for interval in intervals]),
            self.spans.start_offset,
            self.spans.end_offset,
            entries[jnp.asarray(self.spans.start_segment_index)],
            exits[jnp.asarray(self.spans.end_segment_index)],
        )
        spans, guides, previous = [], [], {}
        lengths = jnp.zeros(self.spans.num_paths, dtype=configuration.dtype)
        for index, path in enumerate(self.spans.span_path_index):
            spans.append(
                jax.tree.map(lambda values, index=index: values[index], evaluated)
            )
            lengths = lengths.at[path].add(evaluated.length[index])
            if path in previous:
                displacement, _, coincident = guide_kinematics(
                    robot,
                    configuration,
                    self.spans.start_segment_index[index],
                    self.spans.end_offset[previous[path]],
                    self.spans.start_offset[index],
                )
                distance = _guide_length(displacement, coincident)
                guides.append(
                    RoutingGuideSolution(
                        index, distance, displacement, coincident | (distance > 1e-12)
                    )
                )
                lengths = lengths.at[path].add(distance)
            previous[path] = index
        return RoutingSolution(tuple(spans), lengths, tuple(guides))

    def path_length_jacobian(self, robot, configuration, solution):
        """Return the analytical derivative of a resolved path family.

        Args:
            robot (SoftRobot): Host supplying joint guide kinematics.
            configuration (Array): Configuration matching the solution.
            solution (RoutingSolution): Result returned by this strategy.

        Returns:
            result (Array): Jacobian, shape ``(num_paths, num_coordinates)``, with
                zero base columns. Invalid spans or collapsed moving connections
                yield NaNs.
        """
        self.validate_structure_for_robot(robot)
        if len(solution.spans) != self.spans.num_spans or any(
            not isinstance(span, StraightSpanSolution) for span in solution.spans
        ):
            raise ValueError("solution must contain matching straight spans.")
        if jnp.shape(configuration) != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        jacobian = jnp.zeros((self.spans.num_paths, robot.num_coordinates))
        previous = {}
        for index, path in enumerate(self.spans.span_path_index):
            span = solution.spans[index]
            jacobian = jacobian.at[path].add(
                jnp.where(span.converged, span.length_jacobian, jnp.nan)
            )
            if path in previous:
                displacement, derivative, coincident = guide_kinematics(
                    robot,
                    configuration,
                    self.spans.start_segment_index[index],
                    self.spans.end_offset[previous[path]],
                    self.spans.start_offset[index],
                )
                distance = _guide_length(displacement, coincident)
                gradient = (
                    displacement
                    @ derivative
                    / jnp.where(distance > 1e-12, distance, 1.0)
                )
                gradient = jnp.where(coincident, 0.0, gradient)
                jacobian = jacobian.at[path].add(
                    jnp.where(coincident | (distance > 1e-12), gradient, jnp.nan)
                )
            previous[path] = index
        return jacobian

    def span_positions(self, robot, configuration, solution, span_index, fractions):
        """Interpolate a straight span in world coordinates.

        Args:
            robot (SoftRobot): Unused host for the common sampling interface.
            configuration (Array): Unused configuration associated with the solution.
            solution (RoutingSolution): Routing containing world endpoint positions.
            span_index (int): Static span index.
            fractions (Array): One-dimensional interpolation fractions in ``[0, 1]``.

        Returns:
            result (Array): World positions in metres, shape ``(num_samples, 3)``.
        """
        del robot, configuration
        span = solution.spans[span_index]
        return span.start_position + jnp.atleast_1d(fractions)[:, None] * (
            span.end_position - span.start_position
        )
