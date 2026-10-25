"""Configuration-dependent shortest paths inside circular continuum lumens."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.systems.params import BaseSystemParams, _contains_tracer
from soromox.utils.geometry.poses import transform_point

from ._geometry import span_entry_position
from ._solver import (
    RoutingProblem,
    RoutingSolverOptions,
    _norm_and_unit,
    equalities,
    inequalities,
    inequality_jacobian,
    objective,
    solve_problem,
)
from .routing import (
    ConfigurationDependentThreadlikeRouting,
    RoutingSolution,
    RoutingSpanParams,
    RoutingSpanSolution,
)


class CircularLumenParams(BaseSystemParams):
    """Circular hollow interiors available to the routed filament.

    The lumen is the empty passage inside a hollow section, bounded by its
    inner wall. It is centered on the backbone in every material cross section.

    Attributes:
        inner_radius: Positive inner radii in metres, shape ``(num_segments,)``.
            Radii are constant within each segment and exclude the wall thickness.
    """

    inner_radius: Array = eqx.field(
        converter=lambda value: jnp.atleast_1d(jnp.asarray(value, dtype=float))
    )

    def __check_init__(self):
        """Validate parameters after construction.

        Returns:
            result (None): ``None``. Invalid parameter values raise ValueError or
                TypeError.
        """
        self.validate_for_update()

    @classmethod
    def constant(cls, radius, num_segments: int):
        """Broadcast one radius to an explicitly sized segment array.

        Args:
            radius (float | Array): Positive inner radius in metres.
            num_segments (int): Number of host segments to receive the radius.

        Returns:
            result (CircularLumenParams): CircularLumenParams with radius array shaped
                ``(num_segments,)``.
        """
        return cls(jnp.full((num_segments,), radius, dtype=float))

    def validate_structure(self):
        """Validate parameter shapes and connectivity.

        Returns:
            result (None): ``None``. Invalid shapes or connectivity raise ValueError or
                TypeError.
        """
        if self.inner_radius.ndim != 1:
            raise ValueError("inner_radius must be a one-dimensional array.")

    def validate_values(self):
        """Validate concrete parameter values.

        Returns:
            result (None): ``None``. Invalid concrete geometry raises ValueError.
        """
        if not bool(jnp.all(jnp.isfinite(self.inner_radius) & (self.inner_radius > 0))):
            raise ValueError("Lumen radii must be finite and positive.")

    def _normalize_replacement(self, name, value):
        """Normalize a replacement parameter value.

        Args:
            name (str): Name of the parameter being replaced.
            value (Any): New value for that parameter.

        Returns:
            result (Any): Replacement value converted to the parameter array
                representation.
        """
        return jnp.atleast_1d(jnp.asarray(value, dtype=float))


class ShortestPathRoutingParams(BaseSystemParams):
    """Compose shared guide geometry with a circular containment domain.

    Attributes:
        spans: Guides and connectivity shared across routing strategies.
        domain: Circular lumen radii, with one value per host segment.
    """

    spans: RoutingSpanParams
    domain: CircularLumenParams

    def __check_init__(self):
        """Validate parameters after construction.

        Returns:
            result (None): ``None``. Invalid parameter values raise ValueError or
                TypeError.
        """
        self.validate_for_update()

    def validate_structure(self):
        """Validate parameter shapes and connectivity.

        Returns:
            result (None): ``None``. Invalid shapes or connectivity raise ValueError or
                TypeError.
        """
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
        """Validate concrete parameter values.

        Returns:
            result (None): ``None``. Invalid concrete geometry raises ValueError.
        """
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
        """Check that a replacement preserves topology.

        Args:
            other (BaseSystemParams): Replacement parameters whose static topology must
                match this object.

        Returns:
            result (None): ``None``. Changed parameter types, shapes, or connectivity
                raise ValueError.
        """
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
        """Replace numeric parameters while preserving topology.

        Args:
            updates (Any): Named parameter replacements; numeric geometry can change
                while
                topology is retained.

        Returns:
            result (ShortestPathRoutingParams): New parameter object with validated
                geometry and unchanged static topology.
        """
        result = super().replace(**updates)
        self.assert_same_topology(result)
        return result


class PiecewiseLinearRoutingDiscretization(eqx.Module):
    """Piecewise linear slope basis, independent of the host strain basis.

    Nodal values describe transverse offset derivatives along the backbone.
    Integrating this basis gives piecewise quadratic offsets.

    Attributes:
        elements_per_segment: Number of uniform slope elements in each segment.
        quadrature_order: Gauss points per element for length integration.
        constraint_samples_per_element: Gauss points per element for wall constraints.
        validation_samples_per_element: Uniform checking intervals per element;
            both endpoints are included, giving one more point than intervals.
    """

    elements_per_segment: int = eqx.field(static=True, default=6)
    quadrature_order: int = eqx.field(static=True, default=2)
    constraint_samples_per_element: int = eqx.field(static=True, default=2)
    validation_samples_per_element: int = eqx.field(static=True, default=9)

    def __check_init__(self):
        """Validate parameters after construction.

        Returns:
            result (None): ``None``. Invalid parameter values raise ValueError or
                TypeError.
        """
        for name in (
            "elements_per_segment",
            "quadrature_order",
            "constraint_samples_per_element",
            "validation_samples_per_element",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@jax.custom_jvp
def _validated_length(length, valid):
    """Retain a length value while invalidating derivatives of rejected geometry.

    Args:
        length: Computed scalar length in metres.
        valid: Scalar boolean indicating physical and numerical acceptance.

    Returns:
        The unchanged length. Its derivative is NaN when validation fails.
    """
    return length


@_validated_length.defjvp
def _validated_length_jvp(primals, tangents):
    """Propagate length derivatives only for accepted physical routes.

    Args:
        primals: Length and validation flag.
        tangents: Their directional perturbations.

    Returns:
        Length and its validated directional derivative.
    """
    length, valid = primals
    return length, tangents[0] * jnp.where(valid, 1.0, jnp.nan)


def _basis(knots, points, dimension):
    """Build the integrated piecewise linear slope basis.

    Args:
        knots (Array): Strictly increasing global reference backbone coordinates, shape
            ``(nodes,)``.
        points (Array): Global reference backbone sample coordinates in metres, shape
            ``(points,)``.
        dimension (int): Number of transverse components: one for planar routing or two
            for
            spatial routing.

    Returns:
        result (tuple[Array, Array]): Integrated offset and slope basis arrays, each
            shaped ``(points, 3, nodes * dimension)``.
    """
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
        """Embed scalar basis values in transverse spatial components.

        Args:
            scalar (Array): Scalar basis values, shape ``(points, nodes)``.

        Returns:
            result (Array): Spatial basis with shape ``(points, 3, nodes * dimension)``
                and zero longitudinal row.
        """
        transverse = jnp.einsum("pn,ij->pinj", scalar, jnp.eye(dimension)).reshape(
            points.size, dimension, -1
        )
        return jnp.pad(transverse, ((0, 0), (1, 2 - dimension), (0, 0)))

    return expand(integrated), expand(derivative)


def _samples(knots, order):
    """Construct Gauss quadrature on each knot interval.

    Args:
        knots (Array): Strictly increasing global reference backbone coordinates, shape
            ``(nodes,)``.
        order (int): Positive Gauss quadrature order per knot interval.

    Returns:
        result (tuple[Array, Array]): Flattened Gauss sample coordinates and integration
            weights in metres, each shaped ``((nodes - 1) * order,)``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    widths = jnp.diff(knots)
    points = knots[:-1, None] + widths[:, None] * jnp.asarray((nodes + 1) / 2)
    return points.ravel(), (widths[:, None] * jnp.asarray(weights / 2)).ravel()


class ShortestPathRouting(ConfigurationDependentThreadlikeRouting):
    """Shortest feasible paths through hollow sections and prescribed guides.

    A path represents one physical tendon or filament. A span is the portion
    between two prescribed guides, optimized inside one or more continuous
    hollow sections. Adjacent spans of a path are connected by a straight cable.
    A lumen is the hollow interior available to the filament.

    The host supplies material strains, their basis, and joint guide kinematics.
    Floating bases and moving joints are supported. Spans must end at moving
    joints so the upstream and downstream guide offsets can be prescribed.
    Derivatives solve the converged KKT equations for a fixed active set.
    Slack, friction, joint obstacle geometry, and rod bending stiffness are excluded.

    Attributes:
        params: Guide offsets, span topology, and circular lumen dimensions.
        discretization: Piecewise linear slope mesh and sampling counts.
        solver_options: Physical feasibility, solver, and sensitivity tolerances.

    References:
        Mathew, A. T., Alkayas, A. Y., Alshehhi, A. A. S., Adamu, Y. A.,
        Taha, T., & Renda, F. (2026). A Differentiable Framework for Hollow
        Tendon-Driven Continuum Robots With Implicit Internal Routing.
        *IEEE Transactions on Robotics*, 42, 3009-3027.
        [doi:10.1109/TRO.2026.3714660](https://doi.org/10.1109/TRO.2026.3714660).
    """

    params: ShortestPathRoutingParams
    discretization: PiecewiseLinearRoutingDiscretization = eqx.field(
        default_factory=PiecewiseLinearRoutingDiscretization
    )
    solver_options: RoutingSolverOptions = eqx.field(
        default_factory=RoutingSolverOptions
    )

    @property
    def spans(self):
        """Return the shared span parameters.

        Returns:
            result (RoutingSpanParams): RoutingSpanParams containing guide offsets and
                span-to-path connectivity.
        """
        return self.params.spans

    def validate_structure_for_robot(self, robot):
        """Check host topology and span compatibility.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid host topology or unsupported span crossings
                raise an exception.
        """
        self.params.validate_structure()
        super().validate_structure_for_robot(robot)
        if self.params.domain.inner_radius.shape != (robot.num_segments,):
            raise ValueError("Lumen radii must contain one value per host segment.")

    def validate_values_for_robot(self, robot):
        """Validate concrete host and guide geometry.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid concrete host or guide geometry raises
                ValueError.
        """
        self.params.validate_values()
        super().validate_values_for_robot(robot)

    def solve_routing(self, robot, configuration, *, initial_guess=None):
        """Solve all spans and return lengths, coefficients, and diagnostics.

        Args:
            robot (SoftRobot): Continuum host implementing the material routing
                interface.
            configuration (Array): Host generalized coordinates.
            initial_guess (RoutingSolution | None): Previous ``RoutingSolution`` with
                identical topology.

        Returns:
            result (RoutingSolution): A ``RoutingSolution``. Inspect ``converged`` and
                dense wall violations before using a route. Invalid input geometry
                raises eagerly.
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
            or any(
                not isinstance(span, RoutingSpanSolution)
                for span in initial_guess.spans
            )
        ):
            raise ValueError(
                "initial_guess must be a RoutingSolution with matching spans."
            )
        solutions = tuple(
            self._solve_span(
                robot,
                configuration,
                index,
                None
                if initial_guess is None
                else initial_guess.spans[index].coefficients,
            )
            for index in range(self.spans.num_spans)
        )
        return self._assemble_solution(robot, configuration, solutions)

    def _solve_span(self, robot, configuration, span_index, initial_guess):
        """Assemble, optimize, and validate one guided span.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.
            configuration (Array): Generalized configuration, shape
                ``(robot.num_coordinates,)``.
            span_index (int): Zero-based index of the requested span.
            initial_guess (Array | None): Previous physical slope coefficients, shape
                ``(nodes, 1 or
                2)``, or ``None``.

        Returns:
            result (RoutingSpanSolution): RoutingSpanSolution containing physical
                coefficients, length, multipliers, and independent residual checks.
        """
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
        strains, _ = robot.strain_and_basis(configuration, segment_indices, points)
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
            radius_scale,
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
        converged = diagnostics[0] > 0
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
        element_offsets = dense_offsets.reshape(-1, dense_count + 1, 3)
        element_radii = self.params.domain.inner_radius[dense_segments].reshape(
            -1, dense_count + 1
        )[:, 0]
        slopes = physical_coefficients.reshape(knots.size, dimension)
        curvature = jnp.linalg.norm(jnp.diff(slopes, axis=0), axis=-1) / jnp.diff(knots)
        interpolation_error = curvature * (jnp.diff(knots) / dense_count) ** 2 / 8
        radial_bound = (
            jnp.max(jnp.linalg.norm(element_offsets, axis=-1), axis=1)
            + interpolation_error
        )
        violation_bound = jnp.maximum(0.0, jnp.max(radial_bound - element_radii))
        converged = converged & (violation_bound <= self.solver_options.wall_tolerance)
        return RoutingSpanSolution(
            physical_coefficients.reshape(expected),
            knots,
            _validated_length(
                objective(coefficients, problem)[0] * objective_scale, converged
            ),
            equality_multipliers,
            wall_multipliers,
            wall_values >= -self.solver_options.active_tolerance,
            converged,
            diagnostics[1].astype(jnp.int32),
            primal,
            stationarity,
            complementarity,
            violation,
            violation_bound,
        )

    def span_offsets(self, solution: RoutingSolution, span_index: int, points: Array):
        """Sample material-frame offsets at global backbone coordinates.

        Args:
            solution (RoutingSolution): RoutingSolution containing resolved spans and
                solver diagnostics.
            span_index (int): Zero-based index of the requested span.
            points (Array): Global reference backbone sample coordinates in metres,
                shape
                ``(points,)``.

        Returns:
            result (Array): Material-frame offsets in metres with shape ``(points, 3)``.
        """
        span = solution.spans[span_index]
        basis, _ = _basis(
            span.knots, jnp.atleast_1d(points), span.coefficients.shape[1]
        )
        return self.spans.start_offset[span_index] + jnp.einsum(
            "pin,n->pi", basis, span.coefficients.ravel()
        )

    def span_positions(self, robot, configuration, solution, span_index, fractions):
        """Sample an optimized span in world coordinates.

        Args:
            robot (SoftRobot): Host supplying forward kinematics.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution): Resolved routing at this configuration.
            span_index (int): Static span index.
            fractions (Array): One-dimensional fractions of the backbone interval.

        Returns:
            result (Array): World positions in metres, shape ``(num_samples, 3)``.
        """
        span = solution.spans[span_index]
        fractions = jnp.atleast_1d(fractions)
        points = span.knots[0] + fractions * (span.knots[-1] - span.knots[0])
        offsets = self.span_offsets(solution, span_index, points)

        def sample(position, offset):
            """Transform one material offset at a backbone abscissa.

            Args:
                position: Scalar backbone abscissa in metres.
                offset: Material offset, shape ``(3,)``.

            Returns:
                World position, shape ``(3,)``.
            """
            pose = robot.forward_kinematics(configuration, position)
            return transform_point(pose, offset)

        positions = jax.vmap(sample)(points, offsets)
        entry = span_entry_position(
            robot,
            configuration,
            self.spans.start_segment_index[span_index],
            self.spans.start_offset[span_index],
        )
        return jnp.where((fractions == 0)[:, None], entry, positions)

    def path_length_jacobian(self, robot, configuration, solution: RoutingSolution):
        """Evaluate the optimal length gradient at an already solved route.

        The envelope identity applies because the material-frame endpoint and
        wall constraints are independent of configuration. The caller must
        supply a converged solution for this robot, configuration, and routing.
        On a fixed base host, positive tensions produce ``-jacobian.T @ tension``.
        Use the velocity Jacobian to compute forces on floating base hosts.

        This computes a partial derivative holding routing coefficients fixed.
        Differentiating this method again requires the supplied solution to
        retain its dependence on configuration through ``solve_routing``.
        This implements the paper's tangent derivative (B2) integrated against
        the unit tangent, equivalently the transpose of (B5), with the envelope
        identity (26). It does not invoke autodiff to evaluate this Jacobian.

        Args:
            robot (SoftRobot): Continuum host implementing the material routing
                interface.
            configuration (Array): Generalized coordinates, shape
                ``(num_coordinates,)``.
            solution (RoutingSolution): Converged routing for this geometry and
                configuration.

        Returns:
            result (Array): Length Jacobian, shape ``(num_paths, num_coordinates)``.
                Floating base columns vanish. Failed span solves and collapsed moving
                guide connections produce NaNs in the affected path row. Spatial
                floating systems use quaternion configuration coordinates; to obtain
                forces in velocity coordinates, use :meth:`path_velocity_jacobian`.
        """
        self.validate_structure_for_robot(robot)
        if (
            not isinstance(solution, RoutingSolution)
            or len(solution.spans) != self.spans.num_spans
            or any(not isinstance(span, RoutingSpanSolution) for span in solution.spans)
        ):
            raise ValueError("solution must contain matching optimized spans.")

        configuration = jnp.asarray(configuration, dtype=float)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        gradients = []
        for index, span in enumerate(solution.spans):
            points, weights = _samples(span.knots, self.discretization.quadrature_order)
            basis, derivative = _basis(span.knots, points, span.coefficients.shape[1])
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
            strains, strain_basis = robot.strain_and_basis(
                configuration, segments, points
            )
            tangent = (
                offset_derivative + jnp.cross(strains[:, :3], offset) + strains[:, 3:]
            )
            norms, unit = _norm_and_unit(tangent)
            density = jnp.concatenate((jnp.cross(offset, unit), unit), axis=-1)
            gradient = jnp.einsum("p,pi,pij->j", weights, density, strain_basis)
            gradient = jnp.where(
                span.converged & jnp.all(norms > 1e-12), gradient, jnp.nan
            )
            gradients.append(gradient)
        return self._assemble_jacobian(robot, configuration, gradients)
