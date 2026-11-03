"""Closed-form routing between prescribed guides without wall contact."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.params import _contains_tracer
from soromox.utils.geometry.poses import segment_boundary_poses
from soromox.utils.lie_algebra import se3

from .routing import (
    ConfigurationDependentThreadlikeRouting,
    RoutingSolution,
    RoutingSpanParams,
    StraightSpanSolution,
    _guide_length,
)

if TYPE_CHECKING:
    from soromox.actuation.friction import ThreadlikeFriction
    from soromox.systems.soft_robot import SoftRobot


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

    def solve_routing(self, robot, configuration, *, initial_guess=None, backend=None):
        """Resolve straight spans and their analytical length derivatives.

        Args:
            robot (SoftRobot): Host supplying local segment and joint guide kinematics.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            initial_guess (RoutingSolution | None): Unused; accepted for the
                common routing interface.
            backend (str | None): Per-call override: ``"auto"``, ``"jax"``, or
                ``"warp"``. ``None`` inherits ``robot.backend``; hosts without a
                preference use JAX. Automatic selection uses Warp on GPU and JAX
                on CPU. Warp uses the optional ``warp-lang`` package for native
                segment kinematics, span geometry, joint connections, and path
                derivatives. Differentiation through the solve uses JAX.

        Returns:
            result (RoutingSolution): StraightSpanSolution entries. Nonfinite
                or collapsed spans report failed convergence and NaN derivatives.

        Raises:
            ValueError: If the backend, configuration shape, or routing parameters
                are invalid, or Warp is requested without JAX 64-bit mode.
            ImportError: If Warp is requested and ``warp-lang`` is unavailable.
        """
        del initial_guess
        configuration = jnp.asarray(configuration, dtype=float)
        backend = self._actuation_backend(robot, configuration, backend)
        if self.spans.num_spans == 0:
            return RoutingSolution((), jnp.zeros(0, dtype=configuration.dtype), ())
        if backend == "warp":
            from soromox.execution.warp.loader import load_straight_span_evaluator

            return load_straight_span_evaluator()(self, robot, configuration)
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
        spans = tuple(
            jax.tree.map(lambda values, index=index: values[index], evaluated)
            for index in range(self.spans.num_spans)
        )
        solution = self._assemble_solution(robot, configuration, spans)
        return RoutingSolution(
            spans,
            solution.path_lengths,
            solution.guides,
            self._assemble_jacobian(
                robot, configuration, [span.length_jacobian for span in spans]
            ),
        )

    def path_length_jacobian(self, robot, configuration, solution=None):
        """Return the analytical derivative of a resolved path family.

        Args:
            robot (SoftRobot): Host supplying joint guide kinematics.
            configuration (Array): Configuration matching the solution.
            solution (RoutingSolution | None): Result returned by this strategy.
                Omission solves the route once.

        Returns:
            result (Array): Jacobian, shape ``(num_paths, num_coordinates)``, with
                zero base columns. Invalid spans or collapsed moving connections
                yield NaNs.
        """
        if solution is None:
            solution = self.solve_routing(robot, configuration)
        self.validate_structure_for_robot(robot)
        if (
            not isinstance(solution, RoutingSolution)
            or len(solution.spans) != self.spans.num_spans
            or any(
                not isinstance(span, StraightSpanSolution) for span in solution.spans
            )
        ):
            raise ValueError("solution must contain matching straight spans.")
        configuration = jnp.asarray(configuration, dtype=float)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        if solution.length_jacobian is not None:
            accepted = jnp.ones(self.spans.num_paths, dtype=bool)
            for path, span in zip(self.spans.span_path_index, solution.spans):
                accepted = accepted.at[path].set(accepted[path] & span.converged)
            for guide in solution.guides:
                path = self.spans.span_path_index[guide.downstream_span]
                accepted = accepted.at[path].set(accepted[path] & guide.differentiable)
            return jnp.where(accepted[:, None], solution.length_jacobian, jnp.nan)
        gradients = [
            jnp.where(span.converged, span.length_jacobian, jnp.nan)
            for span in solution.spans
        ]
        return self._assemble_jacobian(robot, configuration, gradients)

    def _actuation_backend(self, robot, configuration, backend):
        """Validate native actuation inputs and resolve the execution preference."""
        from soromox.execution import select_execution_backend

        selected = select_execution_backend(
            getattr(robot, "backend", "jax") if backend is None else backend,
            family_name="straight routing",
            warp_cpu_supported=True,
            warp_supported=hasattr(robot, "B_xi") or hasattr(robot, "B_Z1"),
            num_velocities=robot.num_velocities,
        )
        if selected == "warp" and not jax.config.jax_enable_x64:
            raise ValueError("Warp straight routing requires JAX 64-bit mode.")
        self.validate_structure_for_robot(robot)
        if not _contains_tracer((self.params, robot)):
            with jax.ensure_compile_time_eval():
                self.validate_values_for_robot(robot)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        return selected

    def actuation_matrix(self, robot, configuration, *, backend=None):
        """Evaluate ``-J_velocity.T`` without reconstructing world guide positions.

        Args:
            robot (SoftRobot): PlanarPCS, PCS, or GVS host.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            backend (str | None): Execution override, following :meth:`solve_routing`.

        Returns:
            result (Array): Matrix of shape ``(num_velocities, num_paths)``.
                Invalid geometry produces NaN internal rows for the affected paths.
                Floating base rows are zero because routing is base-invariant.
        """
        configuration = jnp.asarray(configuration, dtype=float)
        selected = self._actuation_backend(robot, configuration, backend)
        if selected == "warp":
            from soromox.execution.warp.loader import load_straight_span_evaluator

            return load_straight_span_evaluator()(
                self, robot, configuration, operation="matrix"
            )
        solution = self.solve_routing(robot, configuration, backend="jax")
        return -self.path_velocity_jacobian(robot, configuration, solution).T

    def path_force_matrix(
        self, robot: SoftRobot, q: Array, *, friction: ThreadlikeFriction
    ) -> Array:
        """Return the length-conjugate effort map using the selected backend.

        Args:
            robot (SoftRobot): Continuum host receiving generalized forces.
            q (Array): Configuration with shape ``(robot.num_coordinates,)``.
            friction (ThreadlikeFriction): Frictionless runtime transmission model.

        Returns:
            result (Array): Force map with shape ``(robot.num_velocities, num_paths)``
                before applying actuator signs and scales.

        Raises:
            ValueError: The supplied friction model is not frictionless.
        """
        self.validate_friction(friction)
        return -self.actuation_matrix(robot, q)

    def actuation_force(self, robot, configuration, tensions, *, backend=None):
        """Evaluate tendon forces directly without reconstructing world geometry.

        Args:
            robot (SoftRobot): PlanarPCS, PCS, or GVS host.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            tensions (Array): Signed filament loads, shape ``(num_paths,)``; positive is
                tension and negative is ideal axial compression below buckling.
            backend (str | None): Execution override, following :meth:`solve_routing`.

        Returns:
            result (Array): Generalized force, shape ``(num_velocities,)``. Warp
                skips world geometry and does not materialize an actuation matrix.
        """
        configuration, tensions = (
            jnp.asarray(configuration, dtype=float),
            jnp.asarray(tensions, dtype=float),
        )
        selected = self._actuation_backend(robot, configuration, backend)
        if tensions.shape != (self.spans.num_paths,):
            raise ValueError("tensions must match the number of routing paths.")
        if selected == "warp":
            from soromox.execution.warp.loader import load_straight_span_evaluator

            return load_straight_span_evaluator()(
                self, robot, configuration, tensions, "force"
            )
        return self.actuation_matrix(robot, configuration, backend="jax") @ tensions

    def _selected_span_positions(
        self,
        robot: SoftRobot,
        q: Array,
        solution: RoutingSolution,
        span_indices: Array,
        fractions: Array,
    ) -> Array:
        """Interpolate the gathered endpoints of one selected span per path.

        Args:
            robot (SoftRobot): Unused host associated with the solution.
            q (Array): Unused configuration associated with the solution.
            solution (RoutingSolution): Resolved world endpoints for all spans.
            span_indices (Array): Selected span per path, shape ``(num_paths,)``.
            fractions (Array): Interpolation fractions with shape ``(num_paths,)``.

        Returns:
            result (Array): World positions in metres, shape ``(num_paths, 3)``.
        """
        del robot, q
        starts = jnp.stack([span.start_position for span in solution.spans])[
            span_indices
        ]
        ends = jnp.stack([span.end_position for span in solution.spans])[span_indices]
        return starts + fractions[:, None] * (ends - starts)

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
