"""Configuration-dependent shortest paths inside circular continuum lumens."""

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.systems.params import BaseSystemParams, _contains_tracer

from ._routing_solver import (
    RoutingProblem,
    RoutingSolverOptions,
    equalities,
    inequalities,
    inequality_jacobian,
    objective,
    solve_problem,
)


def _indices(values):
    values = tuple(values)
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in values
    ):
        raise TypeError("Routing topology must contain integers.")
    return tuple(int(value) for value in values)


def _offsets(values):
    array = jnp.asarray(values, dtype=float)
    return array[None, :] if array.shape == (3,) else array


class RoutingSpanParams(BaseSystemParams):
    """Shared endpoints and topology, with one row per guided span.

    Offsets have shape ``(num_spans, 3)`` in metres and local ``[0, y, z]``
    order. Segment indices are inclusive. Spans of each path must be ordered
    and contiguous; a segment boundary inside a span does not add a guide.
    """

    start_offset: Array = eqx.field(converter=_offsets)
    end_offset: Array = eqx.field(converter=_offsets)
    span_path_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)
    start_segment_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)
    end_segment_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)

    def __check_init__(self):
        self.validate_for_update()

    @property
    def num_spans(self) -> int:
        return len(self.span_path_index)

    @property
    def num_paths(self) -> int:
        return max(self.span_path_index, default=-1) + 1

    def validate_structure(self):
        """Validate shapes and immutable connectivity without reading tracers."""
        expected = (self.num_spans, 3)
        if self.start_offset.shape != expected or self.end_offset.shape != expected:
            raise ValueError(f"Routing offsets must have shape {expected}.")
        if (
            len(self.start_segment_index) != self.num_spans
            or len(self.end_segment_index) != self.num_spans
        ):
            raise ValueError("Every span requires start and end segment indices.")
        if sorted(set(self.span_path_index)) != list(range(self.num_paths)):
            raise ValueError("Path indices must be contiguous and start at zero.")
        previous = {}
        for path, start, end in zip(
            self.span_path_index, self.start_segment_index, self.end_segment_index
        ):
            if start < 0 or end < start:
                raise ValueError(
                    "Each span requires 0 <= start_segment_index <= end_segment_index."
                )
            if path in previous and start != previous[path] + 1:
                raise ValueError("Spans of a path must be ordered and contiguous.")
            previous[path] = end

    def validate_values(self):
        """Validate finite transverse offsets and shared guide continuity."""
        for offsets in (self.start_offset, self.end_offset):
            if not bool(jnp.all(jnp.isfinite(offsets))) or bool(
                jnp.any(offsets[:, 0] != 0)
            ):
                raise ValueError(
                    "Offsets must be finite with zero longitudinal components."
                )
        previous = {}
        for index, path in enumerate(self.span_path_index):
            if path in previous and not bool(
                jnp.allclose(
                    self.start_offset[index],
                    self.end_offset[previous[path]],
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError("Adjacent spans must share the same guide offset.")
            previous[path] = index

    def validate_structure_for_robot(self, robot):
        """Check span bounds against the host segment count."""
        self.validate_structure()
        if any(end >= robot.num_segments for end in self.end_segment_index):
            raise ValueError("Routing span exceeds the host segment count.")

    def validate_values_for_robot(self, robot):
        """Check transverse geometry in aligned continuum segment frames."""
        self.validate_values()
        if robot.is_planar and (
            bool(jnp.any(self.start_offset[:, 2] != 0))
            or bool(jnp.any(self.end_offset[:, 2] != 0))
        ):
            raise ValueError("Planar routing requires zero z offsets.")

    def assert_same_topology(self, other):
        """Reject changes in span connectivity or parameter type."""
        if type(other) is not type(self) or any(
            getattr(self, name) != getattr(other, name)
            for name in ("span_path_index", "start_segment_index", "end_segment_index")
        ):
            raise ValueError("Changing routing topology requires reconstruction.")

    def replace(self, **updates: Any):
        """Replace numeric geometry while retaining span topology."""
        for name in ("span_path_index", "start_segment_index", "end_segment_index"):
            if name in updates and _indices(updates.pop(name)) != getattr(self, name):
                raise ValueError("Changing routing topology requires reconstruction.")
        result = super().replace(**updates)
        self.assert_same_topology(result)
        return result

    def _normalize_replacement(self, name, value):
        return _offsets(value) if name in ("start_offset", "end_offset") else value


class CircularLumenParams(BaseSystemParams):
    """Positive inner radii, in metres, with one value per host segment."""

    inner_radius: Array = eqx.field(
        converter=lambda value: jnp.atleast_1d(jnp.asarray(value, dtype=float))
    )

    def __check_init__(self):
        self.validate_for_update()

    @classmethod
    def constant(cls, radius, num_segments: int):
        """Broadcast one radius to an explicitly sized segment array."""
        return cls(jnp.full((num_segments,), radius, dtype=float))

    def validate_structure(self):
        if self.inner_radius.ndim != 1:
            raise ValueError("inner_radius must be a one-dimensional array.")

    def validate_values(self):
        if not bool(jnp.all(jnp.isfinite(self.inner_radius) & (self.inner_radius > 0))):
            raise ValueError("Lumen radii must be finite and positive.")

    def _normalize_replacement(self, name, value):
        return jnp.atleast_1d(jnp.asarray(value, dtype=float))


class ShortestPathRoutingParams(BaseSystemParams):
    """Compose shared span geometry with a circular containment domain."""

    spans: RoutingSpanParams
    domain: CircularLumenParams

    def __check_init__(self):
        self.validate_for_update()

    def validate_structure(self):
        if not isinstance(self.spans, RoutingSpanParams) or not isinstance(
            self.domain, CircularLumenParams
        ):
            raise TypeError("Expected RoutingSpanParams and CircularLumenParams.")
        self.spans.validate_structure()
        self.domain.validate_structure()
        if any(
            end >= self.domain.inner_radius.size for end in self.spans.end_segment_index
        ):
            raise ValueError("The domain must cover every routed segment.")

    def validate_values(self):
        self.spans.validate_values()
        self.domain.validate_values()
        for offsets, indices in (
            (self.spans.start_offset, self.spans.start_segment_index),
            (self.spans.end_offset, self.spans.end_segment_index),
        ):
            radii = self.domain.inner_radius[jnp.asarray(indices, dtype=int)]
            if bool(jnp.any(jnp.linalg.norm(offsets, axis=-1) > radii)):
                raise ValueError("Routing endpoints must lie inside the lumen.")

    def assert_same_topology(self, other):
        if type(other) is not type(self) or type(other.domain) is not type(self.domain):
            raise ValueError(
                "Changing routing parameter types requires reconstruction."
            )
        self.spans.assert_same_topology(other.spans)
        if self.domain.inner_radius.shape != other.domain.inner_radius.shape:
            raise ValueError(
                "Changing the domain segment count requires reconstruction."
            )

    def replace(self, **updates):
        result = super().replace(**updates)
        self.assert_same_topology(result)
        return result


class RoutingDiscretization(eqx.Module):
    """Piecewise linear offset-derivative basis, independent of strain bases."""

    elements_per_segment: int = eqx.field(static=True, default=6)
    quadrature_order: int = eqx.field(static=True, default=2)
    constraint_samples_per_element: int = eqx.field(static=True, default=2)
    validation_samples_per_element: int = eqx.field(static=True, default=9)

    def __check_init__(self):
        for name in (
            "elements_per_segment",
            "quadrature_order",
            "constraint_samples_per_element",
            "validation_samples_per_element",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


class RoutingSpanSolution(eqx.Module):
    """One solved span and normalized KKT diagnostics.

    Coefficients are nodal transverse slopes ``d(offset)/ds``, shaped
    ``(nodes, 2)`` in space or ``(nodes, 1)`` in the plane. Multipliers use
    normalized objectives/constraints and the nonpositive wall convention.
    ``dense_wall_violation`` is measured in metres; sampled convergence alone
    does not certify containment between samples.
    """

    coefficients: Array
    knots: Array
    length: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    active_mask: Array
    converged: Array
    iterations: Array
    primal_residual: Array
    stationarity_residual: Array
    complementarity_residual: Array
    dense_wall_violation: Array


class RoutingSolution(eqx.Module):
    """Per-span solutions and lengths reduced to actuator path order."""

    spans: tuple[RoutingSpanSolution, ...]
    path_lengths: Array

    @property
    def converged(self):
        return jnp.all(jnp.asarray([span.converged for span in self.spans], dtype=bool))


class ImplicitThreadlikeRouting(eqx.Module):
    """Interface for configuration-dependent routing with explicit solve state."""

    @property
    @abstractmethod
    def spans(self) -> RoutingSpanParams: ...

    @abstractmethod
    def validate_structure_for_robot(self, robot) -> None: ...

    @abstractmethod
    def validate_values_for_robot(self, robot) -> None: ...

    @abstractmethod
    def solve_routing(
        self, robot, configuration, *, initial_guess=None
    ) -> RoutingSolution: ...


def _basis(knots, points, dimension):
    widths = jnp.diff(knots)
    fraction = jnp.clip((points[:, None] - knots[:-1]) / widths, 0, 1)
    integrated = jnp.pad(
        widths * (fraction - fraction**2 / 2), ((0, 0), (0, 1))
    ) + jnp.pad(widths * fraction**2 / 2, ((0, 0), (1, 0)))
    index = jnp.clip(
        jnp.searchsorted(knots, points, side="right") - 1, 0, widths.size - 1
    )
    local = (points - knots[index]) / widths[index]
    derivative = (1 - local[:, None]) * jax.nn.one_hot(index, knots.size) + local[
        :, None
    ] * jax.nn.one_hot(index + 1, knots.size)

    def expand(scalar):
        transverse = jnp.einsum("pn,ij->pinj", scalar, jnp.eye(dimension)).reshape(
            points.size, dimension, -1
        )
        return jnp.pad(transverse, ((0, 0), (1, 2 - dimension), (0, 0)))

    return expand(integrated), expand(derivative)


def _samples(knots, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    widths = jnp.diff(knots)
    points = knots[:-1, None] + widths[:, None] * jnp.asarray((nodes + 1) / 2)
    return points.ravel(), (widths[:, None] * jnp.asarray(weights / 2)).ravel()


def _host_strains(robot, configuration, segment_indices, points):
    if hasattr(robot, "_eval_B_segment"):
        gathered = robot._min_size_gathered(configuration)

        def evaluate(segment, position):
            local = (
                position - robot.segment_end_positions[segment]
            ) / robot.segment_lengths[segment]
            basis = robot._eval_B_segment(segment, local[None])[0]
            if robot.scale_rotational_basis_by_length:
                basis = basis.at[:3].divide(robot.segment_lengths[segment])
            return basis @ gathered[segment, 1] + robot.xi_ref_Xs[segment, 0]

        return jax.vmap(evaluate)(segment_indices, points)
    strains = robot.strain(configuration).reshape(robot.num_segments, -1)[
        segment_indices
    ]
    if robot.is_planar:
        spatial = jnp.zeros((points.size, 6), dtype=strains.dtype)
        return spatial.at[:, 2:5].set(strains)
    return strains


class ShortestPathRouting(ImplicitThreadlikeRouting):
    """Shortest feasible routing on fixed-base PCS, PlanarPCS, or fixed-joint GVS.

    The forward solve uses the optional ``implicit-routing`` dependencies.
    Derivatives solve the converged KKT equations for a fixed active set.
    Slack, friction, movable joints, and rod bending stiffness are excluded.

    References:
        Mathew, A. T., Alkayas, A. Y., Alshehhi, A. A. S., Adamu, Y. A.,
        Taha, T., & Renda, F. (2026). A Differentiable Framework for Hollow
        Tendon-Driven Continuum Robots With Implicit Internal Routing.
        *IEEE Transactions on Robotics*, 42, 3009-3027.
        [doi:10.1109/TRO.2026.3714660](https://doi.org/10.1109/TRO.2026.3714660).
    """

    params: ShortestPathRoutingParams
    discretization: RoutingDiscretization = eqx.field(
        default_factory=RoutingDiscretization
    )
    solver_options: RoutingSolverOptions = eqx.field(
        default_factory=RoutingSolverOptions
    )

    @property
    def spans(self):
        return self.params.spans

    def with_params(self, params: ShortestPathRoutingParams):
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return eqx.tree_at(lambda routing: routing.params, self, params)

    def validate_structure_for_robot(self, robot):
        from soromox.systems import GVS, PCS, PlanarPCS

        if not isinstance(robot, (GVS, PCS, PlanarPCS)):
            raise TypeError("ShortestPathRouting requires GVS, PCS, or PlanarPCS.")
        if robot.floating_base:
            raise ValueError("Implicit routing currently requires a fixed-base host.")
        if isinstance(robot, GVS) and any(
            segment.joint.type != "fixed" for segment in robot.structure.segments
        ):
            raise ValueError("Implicit GVS routing currently requires fixed joints.")
        self.params.validate_structure()
        self.spans.validate_structure_for_robot(robot)
        if self.params.domain.inner_radius.shape != (robot.num_segments,):
            raise ValueError("Lumen radii must contain one value per host segment.")

    def validate_values_for_robot(self, robot):
        self.params.validate_values()
        self.spans.validate_values_for_robot(robot)
        if not bool(
            jnp.all(jnp.isfinite(robot.segment_length) & (robot.segment_length > 0))
        ):
            raise ValueError("Routing requires positive finite segment lengths.")

    def solve_routing(self, robot, configuration, *, initial_guess=None):
        """Solve all spans and return lengths, coefficients, and diagnostics.

        Args:
            robot: Fixed-base continuum host with aligned segment interfaces.
            configuration: Host generalized coordinates.
            initial_guess: Previous ``RoutingSolution`` with identical topology.

        Returns:
            A ``RoutingSolution``. Inspect ``converged`` and dense wall violations
            before using a route. Invalid input geometry raises eagerly.
        """
        configuration = jnp.asarray(configuration, dtype=float)
        self.validate_structure_for_robot(robot)
        if not _contains_tracer((self.params, robot)):
            with jax.ensure_compile_time_eval():
                self.validate_values_for_robot(robot)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        if initial_guess is not None and (
            not isinstance(initial_guess, RoutingSolution)
            or len(initial_guess.spans) != self.spans.num_spans
        ):
            raise ValueError(
                "initial_guess must be a RoutingSolution with matching spans."
            )
        solutions = []
        lengths = jnp.zeros(self.spans.num_paths, dtype=configuration.dtype)
        for span_index, path_index in enumerate(self.spans.span_path_index):
            guess = (
                None
                if initial_guess is None
                else initial_guess.spans[span_index].coefficients
            )
            solution = self._solve_span(robot, configuration, span_index, guess)
            solutions.append(solution)
            lengths = lengths.at[path_index].add(solution.length)
        return RoutingSolution(tuple(solutions), lengths)

    def _solve_span(self, robot, configuration, span_index, initial_guess):
        start = self.spans.start_segment_index[span_index]
        end = self.spans.end_segment_index[span_index]
        lengths = robot.segment_length
        boundaries = jnp.concatenate(
            (jnp.zeros(1, dtype=lengths.dtype), jnp.cumsum(lengths))
        )
        count = self.discretization.elements_per_segment
        knots = jnp.concatenate(
            tuple(
                boundaries[segment] + lengths[segment] * jnp.arange(count) / count
                for segment in range(start, end + 1)
            )
            + (boundaries[end + 1 : end + 2],)
        )
        dimension = 1 if robot.is_planar else 2
        points, weights = _samples(knots, self.discretization.quadrature_order)
        wall_points, _ = _samples(
            knots, self.discretization.constraint_samples_per_element
        )
        segment_indices = jnp.repeat(
            jnp.arange(start, end + 1), count * self.discretization.quadrature_order
        )
        wall_segments = jnp.repeat(
            jnp.arange(start, end + 1),
            count * self.discretization.constraint_samples_per_element,
        )
        offsets, derivatives = _basis(knots, points, dimension)
        walls, _ = _basis(knots, wall_points, dimension)
        endpoint, _ = _basis(knots, knots[-1:], dimension)
        strains = _host_strains(robot, configuration, segment_indices, points)
        start_offset = self.spans.start_offset[span_index]
        end_offset = self.spans.end_offset[span_index]
        radius_scale = self.params.domain.inner_radius[start]
        length = knots[-1] - knots[0]
        variable_scale = radius_scale / length
        objective_scale = radius_scale**2 / length
        tangent_matrix = derivatives + jnp.cross(strains[:, :3, None], offsets, axis=1)
        problem = RoutingProblem(
            tangent_matrix * variable_scale,
            strains[:, 3:] + jnp.cross(strains[:, :3], start_offset),
            weights / objective_scale,
            endpoint[0, 1 : dimension + 1] * variable_scale / radius_scale,
            (end_offset - start_offset)[1 : dimension + 1] / radius_scale,
            walls[:, 1 : dimension + 1] * variable_scale / radius_scale,
            jnp.broadcast_to(
                start_offset[1 : dimension + 1] / radius_scale,
                (wall_points.size, dimension),
            ),
            (self.params.domain.inner_radius[wall_segments] / radius_scale) ** 2,
        )
        expected = (knots.size, dimension)
        if initial_guess is None:
            initial_guess = jnp.broadcast_to(
                (end_offset - start_offset)[1 : dimension + 1] / length, expected
            )
        if initial_guess.shape != expected:
            raise ValueError(f"Warm-start coefficients must have shape {expected}.")
        initial = initial_guess.ravel() / variable_scale
        state, diagnostics = solve_problem(problem, initial, self.solver_options)
        coefficients = state[: initial.size]
        equality_multipliers = state[initial.size : initial.size + dimension]
        wall_multipliers = state[initial.size + dimension :]
        wall_values = inequalities(coefficients, problem)
        primal = jnp.maximum(
            jnp.max(jnp.abs(equalities(coefficients, problem))),
            jnp.maximum(0, jnp.max(wall_values)),
        )
        stationarity = jnp.max(
            jnp.abs(
                objective(coefficients, problem)[1]
                + problem.equality_matrix.T @ equality_multipliers
                + inequality_jacobian(coefficients, problem).T @ wall_multipliers
            )
        )
        complementarity = jnp.max(jnp.abs(wall_multipliers * wall_values))
        tangent = problem.tangent_offset + jnp.einsum(
            "pin,n->pi", problem.tangent_matrix, coefficients
        )
        tolerance = self.solver_options.tolerance * 10
        converged = (
            (diagnostics[0] > 0)
            & (primal <= tolerance)
            & (stationarity <= tolerance)
            & (complementarity <= tolerance)
            & jnp.all(wall_multipliers >= -tolerance)
            & jnp.all(jnp.linalg.norm(tangent, axis=-1) > 1e-12)
        )
        physical_coefficients = coefficients * variable_scale
        dense_count = self.discretization.validation_samples_per_element
        dense_points = (
            knots[:-1, None]
            + jnp.diff(knots)[:, None] * jnp.linspace(0, 1, dense_count + 1)
        ).ravel()
        dense_basis, _ = _basis(knots, dense_points, dimension)
        dense_offsets = start_offset + jnp.einsum(
            "pin,n->pi", dense_basis, physical_coefficients
        )
        dense_segments = jnp.repeat(
            jnp.arange(start, end + 1), count * (dense_count + 1)
        )
        violation = jnp.maximum(
            0,
            jnp.max(
                jnp.linalg.norm(dense_offsets, axis=-1)
                - self.params.domain.inner_radius[dense_segments]
            ),
        )
        return RoutingSpanSolution(
            physical_coefficients.reshape(expected),
            knots,
            objective(coefficients, problem)[0] * objective_scale,
            equality_multipliers,
            wall_multipliers,
            wall_values >= -self.solver_options.active_tolerance,
            converged,
            diagnostics[1].astype(jnp.int32),
            primal,
            stationarity,
            complementarity,
            violation,
        )

    def span_offsets(self, solution: RoutingSolution, span_index: int, points: Array):
        """Sample material-frame offsets at global backbone coordinates."""
        span = solution.spans[span_index]
        basis, _ = _basis(
            span.knots, jnp.atleast_1d(points), span.coefficients.shape[1]
        )
        return self.spans.start_offset[span_index] + jnp.einsum(
            "pin,n->pi", basis, span.coefficients.ravel()
        )

    def path_length_jacobian(self, robot, configuration, solution: RoutingSolution):
        """Evaluate the optimal length gradient at an already solved route.

        The envelope identity applies because the material-frame endpoint and
        wall constraints are independent of configuration. The caller must
        supply a converged solution for this robot, configuration, and routing.
        Positive tendon tensions produce ``-jacobian.T @ tension``.

        This computes a partial derivative holding routing coefficients fixed.
        Differentiating this method again requires the supplied solution to
        retain its dependence on configuration through ``solve_routing``.
        Returns an array shaped ``(num_paths, robot.num_coordinates)`` and NaNs
        when the supplied solution has not converged.
        """
        self.validate_structure_for_robot(robot)
        if len(solution.spans) != self.spans.num_spans:
            raise ValueError("solution must have matching routing spans.")

        def lengths(current_configuration):
            values = jnp.zeros(self.spans.num_paths, dtype=current_configuration.dtype)
            for index, path in enumerate(self.spans.span_path_index):
                span = solution.spans[index]
                points, weights = _samples(
                    span.knots, self.discretization.quadrature_order
                )
                basis, derivative = _basis(
                    span.knots, points, span.coefficients.shape[1]
                )
                coefficients = span.coefficients.ravel()
                offset = self.spans.start_offset[index] + jnp.einsum(
                    "pin,n->pi", basis, coefficients
                )
                offset_derivative = jnp.einsum("pin,n->pi", derivative, coefficients)
                segments = jnp.repeat(
                    jnp.arange(
                        self.spans.start_segment_index[index],
                        self.spans.end_segment_index[index] + 1,
                    ),
                    self.discretization.elements_per_segment
                    * self.discretization.quadrature_order,
                )
                strains = _host_strains(robot, current_configuration, segments, points)
                tangent = (
                    offset_derivative
                    + jnp.cross(strains[:, :3], offset)
                    + strains[:, 3:]
                )
                values = values.at[path].add(
                    jnp.dot(weights, jnp.linalg.norm(tangent, axis=-1))
                )
            return values

        jacobian = jax.jacfwd(lengths)(jnp.asarray(configuration, dtype=float))
        return jnp.where(solution.converged, jacobian, jnp.nan)
