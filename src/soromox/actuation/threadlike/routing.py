"""Shared geometry, solutions, and interfaces for configuration-dependent routing."""

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from soromox.systems.params import BaseSystemParams

from ._geometry import guide_kinematics
from .base import BaseThreadlikeRouting

if TYPE_CHECKING:
    from soromox.systems.soft_robot import SoftRobot


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
        if any(
            start < joint <= end
            for start, end in zip(self.start_segment_index, self.end_segment_index)
            for joint in robot.moving_joint_indices
        ):
            raise ValueError(
                "A span crossing a moving joint requires prescribed guides; split it at the joint."
            )

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
        length_jacobian: Optional cached path derivatives in configuration
            coordinates, including connecting guides. Straight routing supplies
            this during its shared geometry evaluation.
    """

    spans: tuple[RoutingSpanSolution | StraightSpanSolution, ...]
    path_lengths: Array
    guides: tuple[RoutingGuideSolution, ...] = ()
    length_jacobian: Array | None = None

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


class ConfigurationDependentThreadlikeRouting(BaseThreadlikeRouting):
    """Interface for configuration-dependent filament routing.

    A path represents one physical filament; a span is a portion between
    prescribed guides. Strategies return resolved geometry; optimizers also
    return their state explicitly for warm starts.

    Attributes:
        params: Strategy parameters containing numerical geometry and static topology.
        spans: Shared guide geometry and mapping of spans to physical paths.
    """

    params: BaseSystemParams

    @property
    def num_paths(self) -> int:
        """Return the number of physical paths rather than guided spans."""
        return self.spans.num_paths

    def path_lengths(
        self,
        robot: SoftRobot,
        configuration: Array,
        solution: RoutingSolution | None = None,
    ) -> Array:
        """Return physical lengths, solving once if no solution is supplied.

        Args:
            robot: Host defining the route geometry.
            configuration: Configuration with shape ``(num_coordinates,)``.
            solution: Optional resolved route for these exact inputs. Keep its
                configuration dependence when differentiating this call.

        Returns:
            result (Array): Lengths in metres with shape ``(num_paths,)``. Paths
                containing failed spans return NaN. Use ``solve_routing`` to
                inspect diagnostics and provide explicit warm starts.
        """
        if solution is None:
            solution = self.solve_routing(robot, configuration)
        accepted = jnp.ones(self.num_paths, dtype=bool)
        for path, span in zip(self.spans.span_path_index, solution.spans):
            accepted = accepted.at[path].set(accepted[path] & span.converged)
        return jnp.where(accepted, solution.path_lengths, jnp.nan)

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
    def path_length_jacobian(self, robot, configuration, solution=None):
        """Evaluate the analytical length derivative at a resolved route.

        Args:
            robot (SoftRobot): Host supplying guide and material kinematics.
            configuration (Array): Host configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution | None): Solution for this routing and configuration;
                omission resolves the route once.

        Returns:
            result (Array): Length Jacobian, shape ``(num_paths, num_coordinates)``.
        """
        ...

    def path_velocity_jacobian(self, robot, configuration, solution=None):
        """Map host velocities to path length rates.

        Args:
            robot (SoftRobot): Host supplying base coordinate and velocity dimensions.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            solution (RoutingSolution | None): Solution at this configuration;
                omitted solutions are computed once for this call.

        Returns:
            result (Array): Jacobian, shape ``(num_paths, num_velocities)``. Positive
                filament tensions produce generalized forces ``-jacobian.T @ tension``.
        """
        jacobian = self.path_length_jacobian(robot, configuration, solution)
        return jnp.pad(
            jacobian[:, robot.num_base_coordinates :],
            ((0, 0), (robot.num_base_velocities, 0)),
        )

    def path_positions(
        self,
        robot: SoftRobot,
        q: Array,
        s: Array,
        solution: RoutingSolution | None = None,
    ) -> Array:
        """Return one world position per path at a reference backbone station.

        Args:
            robot: Host supplying kinematics and segment lengths.
            q: Configuration with shape ``(num_coordinates,)``.
            s: Scalar reference backbone station in metres.
            solution: Optional solution for these exact inputs.

        Returns:
            result (Array): Positions with shape ``(num_paths, 2 or 3)`` in
                metres. Stations clamp to path endpoints. Within a straight
                span they interpolate along its chord; optimized spans use
                their material offsets. At shared boundaries the upstream span
                is selected. Explicit span sampling includes both sides of a
                moving joint's guide connection.
        """
        if solution is None:
            solution = self.solve_routing(robot, q)
        dimension = 2 if robot.is_planar else 3
        if not self.num_paths:
            return jnp.zeros((0, dimension), dtype=jnp.asarray(q).dtype)
        boundaries = jnp.concatenate((jnp.zeros(1), jnp.cumsum(robot.segment_length)))
        paths = []
        for path in range(self.num_paths):
            indices = tuple(
                index
                for index, item in enumerate(self.spans.span_path_index)
                if item == path
            )
            point = jnp.zeros(3)
            # Earlier spans overwrite later ones at their shared boundary.
            for index in reversed(indices):
                start = boundaries[self.spans.start_segment_index[index]]
                end = boundaries[self.spans.end_segment_index[index] + 1]
                fraction = jnp.clip((s - start) / (end - start), 0.0, 1.0)
                candidate = self.span_positions(robot, q, solution, index, fraction)[0]
                point = jnp.where((s <= end) | (index == indices[-1]), candidate, point)
            paths.append(point[:dimension])
        return jnp.stack(paths)

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

    def with_params(self, params):
        """Replace numerical geometry while retaining routing topology.

        Args:
            params (RoutingSpanParams | ShortestPathRoutingParams): Parameters of
                the same type and topology as this strategy's current parameters.

        Returns:
            result (ConfigurationDependentThreadlikeRouting): Strategy with
                validated replacement geometry and unchanged algorithm settings.
        """
        self.params.assert_same_topology(params)
        params.validate_for_update()
        return eqx.tree_at(lambda routing: routing.params, self, params)

    def validate_structure_for_robot(self, robot) -> None:
        """Validate segment bounds and prescribed guides at moving joints.

        Args:
            robot (SoftRobot): Continuum host with static segment topology.

        Raises:
            ValueError: Span connectivity is incompatible with the host.
        """
        self.spans.validate_structure_for_robot(robot)

    def validate_values_for_robot(self, robot) -> None:
        """Validate guide geometry and positive finite segment lengths.

        Args:
            robot (SoftRobot): Continuum host with material frame conventions.

        Raises:
            ValueError: Host lengths or guide offsets are invalid.
        """
        self.spans.validate_values_for_robot(robot)
        if not bool(
            jnp.all(jnp.isfinite(robot.segment_length) & (robot.segment_length > 0))
        ):
            raise ValueError("Routing requires positive finite segment lengths.")

    def _assemble_solution(self, robot, configuration, spans):
        """Combine resolved spans and joint connections into physical paths.

        Args:
            robot (SoftRobot): Host supplying relative joint kinematics.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            spans (tuple): Resolved span solutions in input order.

        Returns:
            result (RoutingSolution): Spans, guide connections, and total lengths.
        """
        lengths = jnp.zeros(self.spans.num_paths, dtype=configuration.dtype)
        guides, previous = [], {}
        for index, path in enumerate(self.spans.span_path_index):
            lengths = lengths.at[path].add(spans[index].length)
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
        return RoutingSolution(spans, lengths, tuple(guides))

    def _assemble_jacobian(self, robot, configuration, span_gradients):
        """Combine span and guide derivatives in physical path order.

        Args:
            robot (SoftRobot): Host supplying relative joint kinematics.
            configuration (Array): Configuration, shape ``(num_coordinates,)``.
            span_gradients (list[Array]): One configuration gradient per span,
                each shaped ``(num_coordinates,)``.

        Returns:
            result (Array): Length Jacobian, shape ``(num_paths, num_coordinates)``.
                Collapsed moving guide connections invalidate the affected row.
        """
        jacobian = jnp.zeros(
            (self.spans.num_paths, robot.num_coordinates), dtype=configuration.dtype
        )
        previous = {}
        for index, path in enumerate(self.spans.span_path_index):
            jacobian = jacobian.at[path].add(span_gradients[index])
            if path in previous:
                displacement, derivative, coincident = guide_kinematics(
                    robot,
                    configuration,
                    self.spans.start_segment_index[index],
                    self.spans.end_offset[previous[path]],
                    self.spans.start_offset[index],
                )
                distance = _guide_length(displacement, coincident)
                unit = displacement / jnp.where(distance > 1e-12, distance, 1)
                gradient = jnp.where(coincident, 0, unit @ derivative)
                jacobian = jacobian.at[path].add(
                    jnp.where(coincident | (distance > 1e-12), gradient, jnp.nan)
                )
            previous[path] = index
        return jacobian

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
