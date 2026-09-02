"""Renderer-neutral cross-section contours and swept-mesh construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from soromox.systems.components import CrossSectionGeometry
from soromox.systems.soft_robot import SoftRobot

__all__ = [
    "CrossSection",
    "DiscreteCrossSectionMarker",
    "CrossSectionSweepLayout",
    "cross_section_sweep_layout",
    "evaluate_cross_sections",
    "loft_cross_section_contours",
    "loft_cross_sections",
    "register_cross_section_contour",
]


ContourBuilder = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


@dataclass(frozen=True)
class CrossSection:
    """A cross-section family and its dimensions at one backbone position.

    The contour is represented in the robot material frame. Its first coordinate
    is longitudinal and therefore zero; the remaining coordinates lie in the
    material-frame yz plane. The contour is implicitly closed, so the first point
    is not repeated at the end.
    """

    geometry: int
    dimensions: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Normalize the descriptor without imposing geometry-specific dimensions."""
        geometry = int(np.asarray(self.geometry).item())
        dimensions = np.asarray(self.dimensions, dtype=np.float64)
        if dimensions.ndim != 1 or dimensions.size == 0:
            raise ValueError("cross-section dimensions must be a nonempty 1D array")
        if not np.all(np.isfinite(dimensions)):
            raise ValueError("cross-section dimensions must be finite")
        dimensions = dimensions.copy()
        dimensions.setflags(write=False)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "dimensions", dimensions)

    def contour(self, resolution: int) -> NDArray[np.float64]:
        """Return the ordered material-frame contour for this section."""
        return _cross_section_contour(self.geometry, self.dimensions, resolution)

    def discrete_marker(self) -> DiscreteCrossSectionMarker:
        """Return the shared 3D marker representation for discrete rendering."""
        if self.geometry == CrossSectionGeometry.CIRCULAR:
            radius = _positive_dimension(self.dimensions, 0, "radius")
            return DiscreteCrossSectionMarker(
                primitive="sphere",
                scale_xyz=np.full(3, radius),
                align_with_material_frame=False,
            )
        if self.geometry == CrossSectionGeometry.RECTANGULAR:
            height = _positive_dimension(self.dimensions, 0, "height")
            width = _positive_dimension(self.dimensions, 1, "width")
            return DiscreteCrossSectionMarker(
                primitive="box",
                scale_xyz=np.array([width, height, max(width, height)]),
                align_with_material_frame=True,
            )

        semi_major = _positive_dimension(self.dimensions, 0, "semi_major")
        semi_minor = _positive_dimension(self.dimensions, 1, "semi_minor")
        return DiscreteCrossSectionMarker(
            primitive="ellipsoid",
            scale_xyz=np.array([semi_major, semi_minor, max(semi_major, semi_minor)]),
            align_with_material_frame=True,
        )


@dataclass(frozen=True)
class DiscreteCrossSectionMarker:
    """Backend-neutral 3D proxy used for one discrete backbone station.

    The local primitive axes are ordered as material-frame ``(y, z, x)``. The
    third scale gives a discrete marker finite longitudinal depth; it is not a
    swept cross-section thickness.
    """

    primitive: Literal["sphere", "box", "ellipsoid"]
    scale_xyz: NDArray[np.float64]
    align_with_material_frame: bool

    def __post_init__(self) -> None:
        """Validate and freeze the marker scale."""
        scale = np.asarray(self.scale_xyz, dtype=np.float64)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)):
            raise ValueError("discrete marker scale_xyz must be a finite 3-vector")
        if np.any(scale <= 0.0):
            raise ValueError("discrete marker scales must be strictly positive")
        scale = scale.copy()
        scale.setflags(write=False)
        object.__setattr__(self, "scale_xyz", scale)


@dataclass(frozen=True)
class CrossSectionSweepLayout:
    """Backbone stations and link-local index ranges for a swept surface.

    Internal link boundaries occur twice in ``abscissae``: once just inside the
    outgoing link and once just inside the incoming link. This lets each side
    carry its own cross-section family and material frame without lofting one
    link directly into the next.
    """

    abscissae: NDArray[np.float64]
    segment_starts: NDArray[np.int64]
    segment_ends: NDArray[np.int64]

    def edge_caps(self) -> dict[int, tuple[bool, bool]]:
        """Return start- and end-cap flags for every link-local surface edge.

        Returns:
            Mapping from surface-edge indices to ``(cap_start, cap_end)``.
            Zero-length links have empty station ranges and therefore contribute
            no edges. The first positive-length link is open at its base; every
            later positive-length link is capped at both sides of the preceding
            interface.
        """
        caps: dict[int, tuple[bool, bool]] = {}
        has_previous_surface = False
        for start, end in zip(self.segment_starts, self.segment_ends):
            start_index = int(start)
            end_index = int(end)
            if end_index - start_index < 2:
                continue
            for edge_index in range(start_index, end_index - 1):
                caps[edge_index] = (
                    has_previous_surface and edge_index == start_index,
                    edge_index == end_index - 2,
                )
            has_previous_surface = True
        return caps


def cross_section_sweep_layout(
    segment_lengths: Iterable[float], num_points: int
) -> CrossSectionSweepLayout:
    """Distribute swept-surface stations across links with explicit interfaces.

    ``num_points`` remains the total number of stations. Each positive-length
    link receives at least a base and tip station; zero-length articulated-chain
    entries receive an empty station range. Internal boundary stations are moved
    by one float32 ULP into their owning link. The offset survives JAX's default
    float32 conversion while remaining negligible at rendering scale.
    """
    lengths = np.asarray(segment_lengths, dtype=np.float64).reshape(-1)
    if lengths.size == 0:
        raise ValueError("segment_lengths must contain at least one link")
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 0.0):
        raise ValueError("segment_lengths must be finite and nonnegative")

    positive_indices = np.flatnonzero(lengths > 0.0)
    if positive_indices.size == 0:
        raise ValueError("segment_lengths must contain at least one positive length")

    point_count = int(num_points)
    minimum_points = 2 * int(positive_indices.size)
    if point_count < minimum_points:
        raise ValueError(
            "swept rendering requires at least two backbone points per "
            "positive-length link; "
            f"got num_points={point_count} for {positive_indices.size} "
            "positive-length links "
            f"(minimum {minimum_points})"
        )

    counts = np.zeros(lengths.size, dtype=np.int64)
    counts[positive_indices] = 2
    remaining = point_count - minimum_points
    if remaining:
        positive_lengths = lengths[positive_indices]
        raw_extra = remaining * positive_lengths / positive_lengths.sum()
        extra = np.floor(raw_extra).astype(np.int64)
        remainder = remaining - int(extra.sum())
        if remainder:
            fractional = raw_extra - extra
            order = np.argsort(-fractional, kind="stable")
            extra[order[:remainder]] += 1
        counts[positive_indices] += extra

    boundaries = np.concatenate(([0.0], np.cumsum(lengths)))
    station_parts: list[NDArray[np.float64]] = []
    for segment_index, count in enumerate(counts):
        if count == 0:
            continue
        start = boundaries[segment_index]
        end = boundaries[segment_index + 1]
        stations = np.linspace(start, end, int(count), dtype=np.float64)
        if segment_index > 0:
            stations[0] = float(
                np.nextafter(np.float32(start), np.float32(end), dtype=np.float32)
            )
        if segment_index < positive_indices[-1]:
            stations[-1] = float(
                np.nextafter(np.float32(end), np.float32(start), dtype=np.float32)
            )
        station_parts.append(stations)

    starts = np.cumsum(np.concatenate(([0], counts[:-1]))).astype(np.int64)
    ends = np.cumsum(counts).astype(np.int64)
    abscissae = np.concatenate(station_parts)
    for array in (abscissae, starts, ends):
        array.setflags(write=False)
    return CrossSectionSweepLayout(
        abscissae=abscissae,
        segment_starts=starts,
        segment_ends=ends,
    )


def _positive_dimension(
    dimensions: NDArray[np.float64], index: int, name: str
) -> float:
    if dimensions.size <= index:
        raise ValueError(f"{name} is missing from the cross-section dimensions")
    value = float(dimensions[index])
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")
    return value


def _circular_contour(
    dimensions: NDArray[np.float64], resolution: int
) -> NDArray[np.float64]:
    radius = _positive_dimension(dimensions, 0, "radius")
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    return np.column_stack(
        (np.zeros(resolution), radius * np.cos(angles), radius * np.sin(angles))
    )


def _elliptical_contour(
    dimensions: NDArray[np.float64], resolution: int
) -> NDArray[np.float64]:
    semi_major = _positive_dimension(dimensions, 0, "semi_major")
    semi_minor = _positive_dimension(dimensions, 1, "semi_minor")
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    return np.column_stack(
        (
            np.zeros(resolution),
            semi_major * np.cos(angles),
            semi_minor * np.sin(angles),
        )
    )


def _rectangular_contour(
    dimensions: NDArray[np.float64], resolution: int
) -> NDArray[np.float64]:
    height = _positive_dimension(dimensions, 0, "height")
    width = _positive_dimension(dimensions, 1, "width")
    corners_yz = np.array(
        [
            [-0.5 * width, -0.5 * height],
            [0.5 * width, -0.5 * height],
            [0.5 * width, 0.5 * height],
            [-0.5 * width, 0.5 * height],
        ],
        dtype=np.float64,
    )

    points_per_edge, remainder = divmod(resolution, 4)
    edge_counts = [
        points_per_edge + (edge_index < remainder) for edge_index in range(4)
    ]
    contour_parts = []
    for edge_index, count in enumerate(edge_counts):
        start = corners_yz[edge_index]
        end = corners_yz[(edge_index + 1) % 4]
        weights = np.arange(count, dtype=np.float64)[:, None] / float(count)
        contour_parts.append(start + weights * (end - start))
    contour_yz = np.concatenate(contour_parts, axis=0)
    return np.column_stack((np.zeros(resolution), contour_yz))


_CONTOUR_BUILDERS: dict[int, ContourBuilder] = {
    int(CrossSectionGeometry.CIRCULAR): _circular_contour,
    int(CrossSectionGeometry.RECTANGULAR): _rectangular_contour,
    int(CrossSectionGeometry.ELLIPTICAL): _elliptical_contour,
}


def register_cross_section_contour(
    geometry: int,
    builder: ContourBuilder,
    *,
    replace: bool = False,
) -> None:
    """Register a contour builder for an additional cross-section family.

    A builder receives the geometry-specific dimension array and requested
    contour resolution. It must return exactly ``resolution`` ordered points
    with shape ``(resolution, 3)`` in the material-frame yz plane.
    """
    geometry_key = int(geometry)
    if not callable(builder):
        raise TypeError("builder must be callable")
    if geometry_key in _CONTOUR_BUILDERS and not replace:
        raise ValueError(
            f"cross-section geometry {geometry_key} already has a contour builder"
        )
    _CONTOUR_BUILDERS[geometry_key] = builder


def _cross_section_contour(
    geometry: int,
    dimensions: NDArray[np.float64],
    resolution: int,
) -> NDArray[np.float64]:
    if isinstance(resolution, bool) or int(resolution) != resolution:
        raise ValueError("cross-section resolution must be an integer")
    resolution = int(resolution)
    if resolution < 4:
        raise ValueError("cross-section resolution must be at least 4")
    try:
        builder = _CONTOUR_BUILDERS[int(geometry)]
    except KeyError as exc:
        raise ValueError(
            f"cross-section geometry {int(geometry)} has no registered contour builder"
        ) from exc

    contour = np.asarray(builder(dimensions, resolution), dtype=np.float64)
    if contour.shape != (resolution, 3):
        raise ValueError(
            "cross-section contour builders must return shape "
            f"({resolution}, 3), got {contour.shape}"
        )
    if not np.all(np.isfinite(contour)):
        raise ValueError("cross-section contour points must be finite")
    if not np.allclose(contour[:, 0], 0.0):
        raise ValueError(
            "cross-section contour points must lie in the material-frame yz plane"
        )
    return contour


def evaluate_cross_sections(
    robot: SoftRobot,
    q: Any,
    abscissae: Iterable[float],
) -> tuple[CrossSection, ...]:
    """Evaluate cross-section descriptors with one compiled vectorized call."""
    stations = np.asarray(tuple(abscissae), dtype=np.float64).reshape(-1)
    geometries, dimensions = _evaluate_cross_section_arrays(
        robot,
        jnp.asarray(q),
        jnp.asarray(stations),
    )
    return tuple(
        CrossSection(geometry, dimension)
        for geometry, dimension in zip(np.asarray(geometries), np.asarray(dimensions))
    )


@eqx.filter_jit
def _evaluate_cross_section_arrays(
    robot: SoftRobot,
    q: Array,
    abscissae: Array,
) -> tuple[Array, Array]:
    """Provide a persistent JIT boundary for vectorized array evaluation.

    This array-only function remains separate from :func:`evaluate_cross_sections`
    so JAX can reuse its compiled executable. The caller converts the results to
    NumPy-backed :class:`CrossSection` objects, which cannot happen while tracing.
    """
    return jax.vmap(robot.cross_section_geometry, in_axes=(None, 0))(q, abscissae)


def loft_cross_sections(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    frame0: NDArray[np.float64],
    frame1: NDArray[np.float64],
    section0: CrossSection,
    section1: CrossSection,
    resolution: int,
    *,
    cap_start: bool = False,
    cap_end: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Connect two material-frame contours into one swept surface segment."""
    return loft_cross_section_contours(
        p0,
        p1,
        frame0,
        frame1,
        section0.contour(resolution),
        section1.contour(resolution),
        cap_start=cap_start,
        cap_end=cap_end,
    )


def loft_cross_section_contours(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    frame0: NDArray[np.float64],
    frame1: NDArray[np.float64],
    contour0: NDArray[np.float64],
    contour1: NDArray[np.float64],
    *,
    cap_start: bool = False,
    cap_end: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Connect two prepared material-frame contours into a swept segment."""
    contour0 = np.asarray(contour0, dtype=np.float64)
    contour1 = np.asarray(contour1, dtype=np.float64)
    if contour0.ndim != 2 or contour0.shape[1] != 3:
        raise ValueError("cross-section contours must have shape (N, 3)")
    if contour1.shape != contour0.shape:
        raise ValueError(
            "swept-segment endpoint contours must have matching shapes, got "
            f"{contour0.shape} and {contour1.shape}"
        )
    resolution = contour0.shape[0]
    if resolution < 4:
        raise ValueError("cross-section contours must contain at least 4 points")

    world_contour0 = (
        np.asarray(p0, dtype=np.float64)
        + contour0 @ np.asarray(frame0, dtype=np.float64).T
    )
    world_contour1 = (
        np.asarray(p1, dtype=np.float64)
        + contour1 @ np.asarray(frame1, dtype=np.float64).T
    )

    vertex_groups = [world_contour0, world_contour1]
    start_center_index: int | None = None
    end_center_index: int | None = None
    if cap_start:
        start_center_index = sum(len(group) for group in vertex_groups)
        vertex_groups.append(np.asarray(p0, dtype=np.float64).reshape(1, 3))
    if cap_end:
        end_center_index = sum(len(group) for group in vertex_groups)
        vertex_groups.append(np.asarray(p1, dtype=np.float64).reshape(1, 3))
    vertices = np.concatenate(vertex_groups, axis=0)

    faces: list[tuple[int, int, int]] = []
    for index in range(resolution):
        next_index = (index + 1) % resolution
        faces.append((index, next_index, resolution + index))
        faces.append((next_index, resolution + next_index, resolution + index))
    if start_center_index is not None:
        for index in range(resolution):
            next_index = (index + 1) % resolution
            faces.append((start_center_index, next_index, index))
    if cap_end:
        assert end_center_index is not None
        for index in range(resolution):
            next_index = (index + 1) % resolution
            faces.append(
                (end_center_index, resolution + index, resolution + next_index)
            )
    return vertices, np.asarray(faces, dtype=np.int32)
