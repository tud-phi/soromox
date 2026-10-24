"""Closed-form routing between prescribed guides without wall contact."""

import equinox as eqx
import jax
import jax.numpy as jnp

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
    No lumen geometry, optimization, or wall collision checks are evaluated.
    Taut frictionless tendons and ideal straight axial members are supported;
    slack, rod bending, buckling, and finite pulley geometry require other models.

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
            robot.validate_routing_span(start, end)

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
            robot (SoftRobot): Host supplying endpoint and joint guide kinematics.
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
        endpoints = {}
        for at_end, segments, offsets in (
            (False, self.spans.start_segment_index, self.spans.start_offset),
            (True, self.spans.end_segment_index, self.spans.end_offset),
        ):
            for segment in dict.fromkeys(segments):
                indices = tuple(
                    index for index, value in enumerate(segments) if value == segment
                )
                positions, jacobians = jax.vmap(
                    lambda offset, segment=segment, at_end=at_end: (
                        robot.routing_endpoint_kinematics(
                            configuration, segment, at_end, offset
                        )
                    )
                )(offsets[jnp.asarray(indices)])
                for local, index in enumerate(indices):
                    endpoints[index, at_end] = positions[local], jacobians[local]
        spans, guides, previous = [], [], {}
        lengths = jnp.zeros(self.spans.num_paths, dtype=configuration.dtype)
        for index, path in enumerate(self.spans.span_path_index):
            start, start_jacobian = endpoints[index, False]
            end, end_jacobian = endpoints[index, True]
            displacement = end - start
            length = _guide_length(displacement, jnp.array(False))
            valid = jnp.all(jnp.isfinite(displacement)) & (length > 1e-12)
            derivative = (
                displacement
                @ (end_jacobian - start_jacobian)
                / jnp.where(valid, length, 1.0)
            )
            spans.append(
                StraightSpanSolution(
                    start, end, length, jnp.where(valid, derivative, jnp.nan), valid
                )
            )
            lengths = lengths.at[path].add(length)
            if path in previous:
                displacement, _, coincident = robot.routing_guide_kinematics(
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
                displacement, derivative, coincident = robot.routing_guide_kinematics(
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
