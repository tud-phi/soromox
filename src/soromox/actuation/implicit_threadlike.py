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
from soromox.utils.geometry.kinematics import transform_point

from ._routing_geometry import guide_kinematics, span_entry_position, validate_span
from ._routing_solver import (
    RoutingProblem,
    RoutingSolverOptions,
    _norm_and_unit,
    equalities,
    inequalities,
    inequality_jacobian,
    objective,
    solve_problem,
)


def _indices(values):
    """Normalize routing topology to immutable integer indices.

    Args:
        values (Any): Input values to normalize into the required array or index
            representation.

    Returns:
        result (tuple[int, ...]): Immutable tuple of integer topology indices.
    """
    values = tuple(values)
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in values
    ):
        raise TypeError("Routing topology must contain integers.")
    return tuple(int(value) for value in values)


def _offsets(values):
    """Normalize one guide offset or a batch of offsets.

    Args:
        values (Any): Input values to normalize into the required array or index
            representation.

    Returns:
        result (Array): Floating array with shape ``(num_spans, 3)`` for valid inputs; a
            single three-vector gains a leading axis.
    """
    array = jnp.asarray(values, dtype=float)
    return array[None, :] if array.shape == (3,) else array


class RoutingSpanParams(BaseSystemParams):
    """Shared endpoints and topology, with one row per guided span.

    Offsets have shape ``(num_spans, 3)`` in metres and local ``[0, y, z]``
    order. A path is one physical filament, associated with one actuator
    channel. A span is the portion between prescribed guides inside one or
    more continuous sections. Segment indices are inclusive. Spans of each
    path must be ordered and contiguous; a numerical segment boundary inside
    a span does not add a guide. Adjacent spans are connected by a straight
    cable between their prescribed exit and entry guides.

    Parameters live alongside their routing strategy, following the existing
    threadlike actuation API. This shared class also serves other strategies.

    Attributes:
        start_offset: Entry guide offsets, shape ``(num_spans, 3)``, in metres
            in each entry segment's material frame. Longitudinal entries are zero.
        end_offset: Exit guide offsets, shape ``(num_spans, 3)``, in metres
            in each exit segment's material frame.
        span_path_index: Physical path assigned to each span. Indices are
            contiguous from zero; multiple spans can belong to the same path.
        start_segment_index: Inclusive entry segment for each span.
        end_segment_index: Inclusive exit segment for each span.
    """

    start_offset: Array = eqx.field(converter=_offsets)
    end_offset: Array = eqx.field(converter=_offsets)
    span_path_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)
    start_segment_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)
    end_segment_index: tuple[int, ...] = eqx.field(static=True, converter=_indices)

    def __check_init__(self):
        """Validate parameters after construction.

        Returns:
            result (None): ``None``. Invalid parameter values raise ValueError or
                TypeError.
        """
        self.validate_for_update()

    @property
    def num_spans(self) -> int:
        """Return the number of spans.

        Returns:
            result (int): Number of guided portions represented by the topology.
        """
        return len(self.span_path_index)

    @property
    def num_paths(self) -> int:
        """Return the number of physical paths.

        Returns:
            result (int): Number of physical filament paths represented by the topology.
        """
        return max(self.span_path_index, default=-1) + 1

    def validate_structure(self):
        """Validate shapes and immutable connectivity without reading tracers.

        Returns:
            result (None): ``None``. Invalid shapes or connectivity raise ValueError or
                TypeError.
        """
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
        """Validate finite transverse guide offsets.

        Returns:
            result (None): ``None``. Invalid concrete geometry raises ValueError.
        """
        for offsets in (self.start_offset, self.end_offset):
            if not bool(jnp.all(jnp.isfinite(offsets))) or bool(
                jnp.any(offsets[:, 0] != 0)
            ):
                raise ValueError(
                    "Offsets must be finite with zero longitudinal components."
                )

    def validate_structure_for_robot(self, robot):
        """Check span bounds against the host segment count.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid host topology or unsupported span crossings
                raise an exception.
        """
        self.validate_structure()
        if any(end >= robot.num_segments for end in self.end_segment_index):
            raise ValueError("Routing span exceeds the host segment count.")

    def validate_values_for_robot(self, robot):
        """Check transverse geometry in aligned continuum segment frames.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid concrete host or guide geometry raises
                ValueError.
        """
        self.validate_values()
        if robot.is_planar and (
            bool(jnp.any(self.start_offset[:, 2] != 0))
            or bool(jnp.any(self.end_offset[:, 2] != 0))
        ):
            raise ValueError("Planar routing requires zero z offsets.")

    def assert_same_topology(self, other):
        """Reject changes in span connectivity or parameter type.

        Args:
            other (BaseSystemParams): Replacement parameters whose static topology must
                match this object.

        Returns:
            result (None): ``None``. Changed parameter types, shapes, or connectivity
                raise ValueError.
        """
        if type(other) is not type(self) or any(
            getattr(self, name) != getattr(other, name)
            for name in ("span_path_index", "start_segment_index", "end_segment_index")
        ):
            raise ValueError("Changing routing topology requires reconstruction.")

    def replace(self, **updates: Any):
        """Replace numeric geometry while retaining span topology.

        Args:
            updates (Any): Named parameter replacements; numeric geometry can change
                while
                topology is retained.

        Returns:
            result (RoutingSpanParams): New parameter object with validated geometry and
                unchanged static topology.
        """
        for name in ("span_path_index", "start_segment_index", "end_segment_index"):
            if name in updates and _indices(updates.pop(name)) != getattr(self, name):
                raise ValueError("Changing routing topology requires reconstruction.")
        result = super().replace(**updates)
        self.assert_same_topology(result)
        return result

    def _normalize_replacement(self, name, value):
        """Normalize a replacement parameter value.

        Args:
            name (str): Name of the parameter being replaced.
            value (Any): New value for that parameter.

        Returns:
            result (Any): Replacement value converted to the parameter array
                representation.
        """
        return _offsets(value) if name in ("start_offset", "end_offset") else value


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


class RoutingSpanSolution(eqx.Module):
    """One solved span and normalized KKT diagnostics.

    Coefficients are nodal transverse slopes ``d(offset)/ds``, shaped
    ``(nodes, 2)`` in space or ``(nodes, 1)`` in the plane. Multipliers use
    normalized objectives/constraints and the nonpositive wall convention.
    ``dense_wall_violation`` is measured in metres. Convergence also requires
    the continuous interpolation bound to meet the physical wall tolerance.

    Attributes:
        coefficients: Physical transverse offset slopes, shape ``(nodes, 1 or 2)``.
        knots: Global reference backbone coordinates in metres, shape ``(nodes,)``.
        length: Integrated span length in metres.
        equality_multipliers: Endpoint constraint multipliers, shape ``(1 or 2,)``.
        inequality_multipliers: Wall multipliers, shape ``(wall_samples,)``.
        active_mask: Active sampled wall constraints, shape ``(wall_samples,)``.
        converged: Scalar boolean from independent optimality and feasibility checks.
        iterations: Scalar SLSQP iteration count.
        primal_residual: Maximum normalized endpoint or wall constraint violation.
        stationarity_residual: Maximum absolute normalized Lagrangian gradient.
        complementarity_residual: Maximum absolute wall residual times multiplier.
        dense_wall_violation: Maximum sampled penetration in metres on the checking
            grid.
        wall_violation_bound: Conservative upper bound on penetration between
            checking points, in metres, using the quadratic offset interpolation
            error on each element. Included in convergence acceptance.
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
    wall_violation_bound: Array


class StraightSpanSolution(eqx.Module):
    """A straight span between prescribed guides, with no optimization state.

    Attributes:
        start_position: World entry guide, shape ``(3,)``, in metres.
        end_position: World exit guide, shape ``(3,)``, in metres.
        length: Distance between guides in metres.
        length_jacobian: Analytical configuration derivative, shape
            ``(num_coordinates,)``, with zero base columns.
        converged: Whether endpoints are finite and the span has positive length.
    """

    start_position: Array
    end_position: Array
    length: Array
    length_jacobian: Array
    converged: Array


class RoutingSolution(eqx.Module):
    """Solved spans and connecting guides, grouped into physical filament paths.

    Attributes:
        spans: Resolved portions between prescribed guides, in input span order.
        path_lengths: Total lengths in metres, shape ``(num_paths,)``, including
            straight connections between adjacent spans of the same path.
        guides: Straight connections in downstream span order. An empty tuple
            means every physical path consists of a single span.
    """

    spans: tuple[RoutingSpanSolution | StraightSpanSolution, ...]
    path_lengths: Array
    guides: tuple[RoutingGuideSolution, ...] = ()

    @property
    def converged(self):
        """Check the independent convergence diagnostics.

        Returns:
            result (Array): Scalar boolean indicating that every span satisfies the
                solve and physical clearance checks and guide lengths are finite.
        """
        return jnp.all(
            jnp.asarray([span.converged for span in self.spans], dtype=bool)
        ) & jnp.all(
            jnp.asarray(
                [jnp.isfinite(guide.length) for guide in self.guides], dtype=bool
            )
        )


class RoutingGuideSolution(eqx.Module):
    """Straight connection between prescribed guides on adjacent spans.

    Attributes:
        downstream_span: Index of the span entered after this connection.
        length: Cable length in metres, including separation due to joint motion.
        displacement: Downstream guide minus upstream guide in the upstream
            material frame, shape ``(3,)``.
        differentiable: Whether the connection has a configuration derivative.
            Always coincident guides have zero length and derivative. A moving
            connection collapsing to zero length has an undefined derivative.
    """

    downstream_span: int = eqx.field(static=True)
    length: Array
    displacement: Array
    differentiable: Array


@jax.custom_jvp
def _guide_length(displacement, coincident):
    """Evaluate exact guide distance with explicit treatment of coincidence.

    Args:
        displacement (Array): Guide displacement, shape ``(3,)``, in metres.
        coincident (Array): Whether the prescribed guides coincide for every
            configuration, independently of the current joint coordinates.

    Returns:
        result (Array): Scalar distance. Configuration derivatives are zero for always
            coincident guides and NaN for a collapsed moving connection.
    """
    del coincident
    squared = jnp.sum(displacement**2)
    return jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1)), 0)


@_guide_length.defjvp
def _guide_length_jvp(primals, tangents):
    """Differentiate a guide distance away from a collapsing connection.

    Args:
        primals (tuple): Guide displacement and invariant coincidence flag.
        tangents (tuple): Displacement perturbation and unused boolean tangent.

    Returns:
        result (tuple[Array, Array]): Scalar distance and its directional derivative.
    """
    displacement, coincident = primals
    displacement_tangent, _ = tangents
    length = _guide_length(displacement, coincident)
    direction = displacement / jnp.where(length > 1e-12, length, 1)
    direction = jnp.where(coincident, 0, direction)
    direction = direction * jnp.where(coincident | (length > 1e-12), 1, jnp.nan)
    return length, jnp.dot(direction, displacement_tangent)


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


class ConfigurationDependentThreadlikeRouting(eqx.Module):
    """Interface for configuration-dependent filament routing.

    A path represents one physical filament; a span is a portion between
    prescribed guides. Strategies return resolved geometry; optimizers also
    return their state explicitly for warm starts.

    Attributes:
        spans: Shared guide geometry and mapping of spans to physical paths.
    """

    @property
    @abstractmethod
    def spans(self) -> RoutingSpanParams:
        """Return shared endpoints and connectivity.

        Returns:
            result (RoutingSpanParams): RoutingSpanParams containing guide offsets and
                span-to-path connectivity.
        """
        ...

    @abstractmethod
    def path_length_jacobian(self, robot, configuration, solution):
        """Evaluate the analytical length derivative at a resolved route.

        Args:
            robot (SoftRobot): Host supplying guide and material kinematics.
            configuration (Array): Host configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution): Solution for this routing and configuration.

        Returns:
            result (Array): Length Jacobian, shape ``(num_paths, num_coordinates)``.
        """
        ...

    def path_velocity_jacobian(self, robot, configuration, solution):
        """Map host velocities to path length rates.

        Args:
            robot (SoftRobot): Host supplying base coordinate and velocity dimensions.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution): Solution at this configuration.

        Returns:
            result (Array): Jacobian, shape ``(num_paths, num_velocities)``. Positive
                filament tensions produce generalized forces ``-jacobian.T @ tension``.
        """
        jacobian = self.path_length_jacobian(robot, configuration, solution)
        return jnp.pad(
            jacobian[:, robot.num_base_coordinates :],
            ((0, 0), (robot.num_base_velocities, 0)),
        )

    @abstractmethod
    def span_positions(self, robot, configuration, solution, span_index, fractions):
        """Sample a resolved span in world coordinates.

        Args:
            robot (SoftRobot): Host supplying forward kinematics.
            configuration (Array): Host configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution): Solution at this configuration.
            span_index (int): Static span index.
            fractions (Array): One-dimensional sample fractions in ``[0, 1]``.

        Returns:
            result (Array): World positions, shape ``(num_samples, 3)``, in metres.
                Fractions parameterize the backbone interval for optimized routes
                and the straight line for straight spans.
        """
        ...

    @abstractmethod
    def validate_structure_for_robot(self, robot) -> None:
        """Validate static host compatibility and span connectivity.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid host topology or unsupported span crossings
                raise an exception.
        """
        ...

    @abstractmethod
    def validate_values_for_robot(self, robot) -> None:
        """Validate concrete geometry for the host.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.

        Returns:
            result (None): ``None``. Invalid concrete host or guide geometry raises
                ValueError.
        """
        ...

    @abstractmethod
    def solve_routing(
        self, robot, configuration, *, initial_guess=None
    ) -> RoutingSolution:
        """Resolve all physical paths for a robot configuration.

        Args:
            robot (SoftRobot): Continuum host providing material strains and guide
                kinematics.
            configuration (Array): Generalized configuration, shape
                ``(robot.num_coordinates,)``.
            initial_guess (RoutingSolution | None): Previous RoutingSolution with
                matching span topology, or
                ``None``.

        Returns:
            result (RoutingSolution): RoutingSolution with resolved spans, guide
                connections, and total path lengths.
        """
        ...


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


def _host_strains(robot, configuration, segment_indices, points):
    """Evaluate material strains through the host's kinematics interface.

    Args:
        robot (SoftRobot): Continuum host implementing ``strain_and_basis``.
        configuration (Array): Generalized configuration, shape ``(num_coordinates,)``.
        segment_indices (Array): Segment indices with shape ``(num_points,)``.
        points (Array): Global backbone coordinates with shape ``(num_points,)``.

    Returns:
        result (Array): Strains in angular/linear spatial order, shape ``(num_points,
            6)``.
    """
    return robot.strain_and_basis(configuration, segment_indices, points)[0]


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

    def with_params(self, params: ShortestPathRoutingParams):
        """Create a routing instance with replacement parameters.

        Args:
            params (ShortestPathRoutingParams): Replacement ShortestPathRoutingParams
                with the same topology.

        Returns:
            result (ShortestPathRouting): New routing instance with updated geometry and
                the same solver and mesh settings.
        """
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return eqx.tree_at(lambda routing: routing.params, self, params)

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
        self.spans.validate_structure_for_robot(robot)
        for start, end in zip(
            self.spans.start_segment_index, self.spans.end_segment_index
        ):
            validate_span(robot, start, end)
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
        self.spans.validate_values_for_robot(robot)
        if not bool(
            jnp.all(jnp.isfinite(robot.segment_length) & (robot.segment_length > 0))
        ):
            raise ValueError("Routing requires positive finite segment lengths.")

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
        ):
            raise ValueError(
                "initial_guess must be a RoutingSolution with matching spans."
            )
        solutions = []
        guides = []
        previous = {}
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
            if path_index in previous:
                displacement, _, coincident = guide_kinematics(
                    robot,
                    configuration,
                    self.spans.start_segment_index[span_index],
                    self.spans.end_offset[previous[path_index]],
                    self.spans.start_offset[span_index],
                )
                distance = _guide_length(displacement, coincident)
                guides.append(
                    RoutingGuideSolution(
                        span_index,
                        distance,
                        displacement,
                        coincident | (distance > 1e-12),
                    )
                )
                lengths = lengths.at[path_index].add(distance)
            previous[path_index] = span_index
        return RoutingSolution(tuple(solutions), lengths, tuple(guides))

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
        if len(solution.spans) != self.spans.num_spans:
            raise ValueError("solution must have matching routing spans.")

        configuration = jnp.asarray(configuration, dtype=float)
        if configuration.shape != (robot.num_coordinates,):
            raise ValueError("configuration must match the host coordinate count.")
        jacobian = jnp.zeros(
            (self.spans.num_paths, robot.num_coordinates), dtype=configuration.dtype
        )
        previous = {}
        for index, path in enumerate(self.spans.span_path_index):
            span = solution.spans[index]
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
            jacobian = jacobian.at[path].add(gradient)
            if path in previous:
                displacement, guide_jacobian, coincident = guide_kinematics(
                    robot,
                    configuration,
                    self.spans.start_segment_index[index],
                    self.spans.end_offset[previous[path]],
                    self.spans.start_offset[index],
                )
                distance = _guide_length(displacement, coincident)
                unit = displacement / jnp.where(distance > 1e-12, distance, 1)
                gradient = jnp.where(coincident, 0, unit @ guide_jacobian)
                gradient = jnp.where(coincident | (distance > 1e-12), gradient, jnp.nan)
                jacobian = jacobian.at[path].add(gradient)
            previous[path] = index
        return jacobian
