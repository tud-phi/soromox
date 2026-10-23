"""Renderer-neutral cross-section contours and swept-mesh construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from soromox.systems.components import CrossSectionGeometry, StrandCrossSection
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
    "sweep_cross_sections",
]


ContourBuilder = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


@dataclass(frozen=True)
class CrossSection:
    """A cross-section family and its dimensions at one backbone position.

    The contour is represented in the robot material frame. Its first coordinate
    is longitudinal and therefore zero; the remaining coordinates lie in the
    material-frame yz plane. The contour is implicitly closed, so the first point
    is not repeated at the end. Optional ``inner_dimensions`` describe a
    centred cavity of the same geometry family; use ``contours`` and the shared
    sweep/loft functions to retain its opening.

    Attributes:
        geometry: CrossSectionGeometry tag or a registered contour family.
        outer_dimensions: Dimensions of the exterior boundary [m]. Rectangular
            ``[height, width]`` spans material y and z, matching GVS mechanics.
        inner_dimensions: Optional dimensions of a centred cavity [m].
    """

    geometry: int
    outer_dimensions: NDArray[np.float64]
    inner_dimensions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Normalize the descriptor without imposing geometry-specific dimensions."""
        geometry = int(np.asarray(self.geometry).item())
        dimensions = np.asarray(self.outer_dimensions, dtype=np.float64)
        if dimensions.ndim != 1 or dimensions.size == 0:
            raise ValueError("cross-section dimensions must be a nonempty 1D array")
        if not np.all(np.isfinite(dimensions)):
            raise ValueError("cross-section dimensions must be finite")
        dimensions = dimensions.copy()
        dimensions.setflags(write=False)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "outer_dimensions", dimensions)
        if self.inner_dimensions is not None:
            inner = np.asarray(self.inner_dimensions, dtype=np.float64)
            if (
                inner.shape != dimensions.shape
                or not np.all(np.isfinite(inner))
                or np.any(inner <= 0)
                or np.any(inner >= dimensions)
            ):
                raise ValueError(
                    "inner dimensions must be positive and smaller than the outer dimensions"
                )
            inner = inner.copy()
            inner.setflags(write=False)
            object.__setattr__(self, "inner_dimensions", inner)

    @classmethod
    def from_strand_cross_section(cls, section: StrandCrossSection) -> CrossSection:
        """Extract a rendering descriptor from shared strand geometry eagerly."""
        section.validate_for_update()
        return cls(section.geometry, section.outer_dimensions, section.inner_dimensions)

    def contours(self, resolution: int) -> NDArray[np.float64]:
        """Return all boundary loops in the material-frame yz plane.

        Args:
            resolution: Number of vertices per loop, at least four.

        Returns:
            Array of shape ``(loops, resolution, 3)``: outer boundary first,
            then the optional centred cavity. Both loops wind counterclockwise
            about x with corresponding vertices; sweep construction reverses
            the cavity walls and builds annular end caps.
        """
        outer = _cross_section_contour(self.geometry, self.outer_dimensions, resolution)
        if self.inner_dimensions is None:
            return outer[None]
        return np.stack(
            (
                outer,
                _cross_section_contour(
                    self.geometry, self.inner_dimensions, resolution
                ),
            )
        )

    def discrete_marker(self) -> DiscreteCrossSectionMarker:
        """Return an outer-envelope proxy for discrete rendering.

        Discrete sphere/box/ellipsoid markers intentionally omit cavities; use
        swept rendering to display section walls and openings.
        """
        if self.geometry == CrossSectionGeometry.CIRCULAR:
            radius = _positive_dimension(self.outer_dimensions, 0, "radius")
            return DiscreteCrossSectionMarker(
                primitive="sphere",
                scale_xyz=np.full(3, radius),
                align_with_material_frame=False,
            )
        if self.geometry == CrossSectionGeometry.RECTANGULAR:
            height = _positive_dimension(self.outer_dimensions, 0, "height")
            width = _positive_dimension(self.outer_dimensions, 1, "width")
            return DiscreteCrossSectionMarker(
                primitive="box",
                scale_xyz=np.array([height, width, max(width, height)]),
                align_with_material_frame=True,
            )

        semi_major = _positive_dimension(self.outer_dimensions, 0, "semi_major")
        semi_minor = _positive_dimension(self.outer_dimensions, 1, "semi_minor")
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
            [-0.5 * height, -0.5 * width],
            [0.5 * height, -0.5 * width],
            [0.5 * height, 0.5 * width],
            [-0.5 * height, 0.5 * width],
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
    """Evaluate outer and optional inner dimensions in one vectorized call.

    Args:
        robot: Model whose ``cross_section_geometry`` returns the geometry tag,
            outer dimensions and optional inner dimensions together. Inner
            dimensions use None for an entirely solid model or all-zero rows
            for solid stations.
        q: Generalized coordinates used for geometry evaluation.
        abscissae: Backbone positions at which to sample the sections [m].

    Returns:
        One descriptor per station, including the optional cavity.
    """
    stations = np.asarray(tuple(abscissae), dtype=np.float64).reshape(-1)
    geometries, dimensions, inner_dimensions = _evaluate_cross_section_arrays(
        robot,
        jnp.asarray(q),
        jnp.asarray(stations),
    )
    inner_rows = (
        [None] * len(stations)
        if inner_dimensions is None
        else np.asarray(inner_dimensions)
    )
    return tuple(
        CrossSection(
            geometry, outer, None if inner is None or np.all(inner == 0) else inner
        )
        for geometry, outer, inner in zip(
            np.asarray(geometries), np.asarray(dimensions), inner_rows
        )
    )


@eqx.filter_jit
def _evaluate_cross_section_arrays(
    robot: SoftRobot,
    q: Array,
    abscissae: Array,
) -> tuple[Array, Array, Array | None]:
    """Provide a persistent JIT boundary for vectorized array evaluation.

    This array-only function remains separate from :func:`evaluate_cross_sections`
    so JAX can reuse its compiled executable. The caller converts the results to
    NumPy-backed :class:`CrossSection` objects, which cannot happen while tracing.
    """
    return jax.vmap(robot.cross_section_geometry, in_axes=(None, 0))(q, abscissae)


@lru_cache(maxsize=64)
def _sweep_faces(
    stations: int, resolution: int, hollow: bool, cap_start: bool, cap_end: bool
) -> NDArray[np.int32]:
    """Build cached triangle indices for a solid or hollow sweep.

    Args:
        stations: Number of longitudinal stations (at least two).
        resolution: Vertices per boundary loop (at least four).
        hollow: Whether an inner boundary follows the outer boundary vertices.
        cap_start: Include the proximal cap.
        cap_end: Include the distal cap.

    Returns:
        Read-only int32 array of outward-facing triangles, shape ``(F, 3)``.
        All outer walls precede inner walls, followed by the enabled caps.
    """
    station = np.arange(stations - 1, dtype=np.int32)[:, None] * resolution
    edge = np.arange(resolution, dtype=np.int32)
    a = (station + edge).ravel()
    b = (station + (edge + 1) % resolution).ravel()
    walls = np.stack(
        (
            np.stack((a, b, a + resolution), axis=-1),
            np.stack((b, b + resolution, a + resolution), axis=-1),
        ),
        axis=1,
    ).reshape(-1, 3)
    offset = stations * resolution
    parts = [walls]
    if hollow:
        parts.append(walls[:, [0, 2, 1]] + offset)
    center = offset
    for endpoint, enabled in ((0, cap_start), (stations - 1, cap_end)):
        if not enabled:
            continue
        a = endpoint * resolution + edge
        b = endpoint * resolution + (edge + 1) % resolution
        if hollow:
            cap = np.stack(
                (
                    np.stack((a, b, b + offset), axis=-1),
                    np.stack((a, b + offset, a + offset), axis=-1),
                ),
                axis=1,
            ).reshape(-1, 3)
        else:
            cap = np.stack((np.full_like(a, center), a, b), axis=-1)
            center += 1
        parts.append(cap[:, [0, 2, 1]] if endpoint == 0 else cap)
    result = np.concatenate(parts)
    result.setflags(write=False)
    return result


def _sweep_contours(
    centers: NDArray[np.float64],
    frames: NDArray[np.float64],
    loops: NDArray[np.float64],
    *,
    cap_start: bool,
    cap_end: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Transform boundary loops and attach cached outward-facing triangles.

    Args:
        centers: World positions, shape ``(N, 3)``.
        frames: Material-to-world rotations, shape ``(N, 3, 3)``.
        loops: Prepared boundaries, shape ``(N, 1 or 2, resolution, 3)``.
        cap_start: Close the proximal end.
        cap_end: Close the distal end.

    Returns:
        World vertices ``(V, 3)`` and cached int32 faces ``(F, 3)``.
    """
    centers, frames, loops = (
        np.asarray(x, dtype=np.float64) for x in (centers, frames, loops)
    )
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 2:
        raise ValueError("centers must have shape (N, 3) with N >= 2")
    if frames.shape != (len(centers), 3, 3):
        raise ValueError("frames must have shape (N, 3, 3)")
    if (
        loops.ndim != 4
        or loops.shape[0] != len(centers)
        or loops.shape[1] not in (1, 2)
        or loops.shape[2] < 4
        or loops.shape[3] != 3
    ):
        raise ValueError("contours must have shape (N, 1 or 2, resolution >= 4, 3)")
    if not all(np.isfinite(x).all() for x in (centers, frames, loops)):
        raise ValueError("sweep centers, frames and contours must be finite")
    if not np.allclose(loops[..., 0], 0):
        raise ValueError("contours must lie in the material-frame yz plane")
    world = centers[:, None, None, :] + np.einsum("nij,nlkj->nlki", frames, loops)
    vertices = world.transpose(1, 0, 2, 3).reshape(-1, 3)
    hollow = loops.shape[1] == 2
    if not hollow:
        caps = centers[[0, -1]][[cap_start, cap_end]]
        vertices = np.concatenate((vertices, caps))
    return vertices, _sweep_faces(
        len(centers), loops.shape[2], hollow, cap_start, cap_end
    )


def sweep_cross_sections(
    centers: NDArray[np.float64],
    frames: NDArray[np.float64],
    sections: CrossSection | Iterable[CrossSection],
    resolution: int,
    *,
    cap_start: bool = True,
    cap_end: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Sweep solid or hollow sections through material frames at many stations.

    Supply one constant section or one per station. Sections must share a
    geometry family and cavity topology. Hollow sections have one centred
    cavity with corresponding inner/outer vertices; end caps join the loops.
    Vertices and cached faces keep a fixed order during deformation.

    Args:
        centers: World-space station positions, shape ``(N, 3)``, N >= 2.
        frames: Material-to-world rotations, shape ``(N, 3, 3)``.
        sections: One constant section or N sections with matching topology.
        resolution: Number of vertices per boundary loop, at least four.
        cap_start: Close the proximal end, preserving any cavity opening.
        cap_end: Close the distal end, preserving any cavity opening.

    Returns:
        World vertices ``(V, 3)`` and cached int32 triangle indices ``(F, 3)``.
        All outer stations precede inner stations; solid cap centres come last.

    Raises:
        ValueError: If station shapes, section topology or contours are invalid.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if isinstance(sections, CrossSection):
        contours = np.asarray(sections.contours(resolution))
        loops = np.broadcast_to(contours, (len(centers), *contours.shape))
    else:
        sections = tuple(sections)
        if len(sections) != len(centers):
            raise ValueError("one cross section is required per sweep station")
        if len({(s.geometry, s.inner_dimensions is not None) for s in sections}) != 1:
            raise ValueError("sweep sections must share geometry and cavity topology")
        loops = np.asarray([s.contours(resolution) for s in sections])
    return _sweep_contours(centers, frames, loops, cap_start=cap_start, cap_end=cap_end)


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
    """Connect two cross sections, retaining their outer and inner boundaries.

    Args:
        p0: First world-space station, shape ``(3,)``.
        p1: Second world-space station, shape ``(3,)``.
        frame0: First material-to-world rotation, shape ``(3, 3)``.
        frame1: Second material-to-world rotation, shape ``(3, 3)``.
        section0: First cross-section descriptor.
        section1: Second descriptor with matching geometry and cavity topology.
        resolution: Vertices per boundary loop, at least four.
        cap_start: Close the proximal end.
        cap_end: Close the distal end.

    Returns:
        World vertices ``(V, 3)`` and cached int32 faces ``(F, 3)``.
    """
    if section0.geometry != section1.geometry:
        raise ValueError("loft sections must share geometry and cavity topology")
    return loft_cross_section_contours(
        p0,
        p1,
        frame0,
        frame1,
        section0.contours(resolution),
        section1.contours(resolution),
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
    """Loft two prepared sets of boundary loops, including a possible cavity.

    Args:
        p0: First world-space station, shape ``(3,)``.
        p1: Second world-space station, shape ``(3,)``.
        frame0: First material-to-world rotation, shape ``(3, 3)``.
        frame1: Second material-to-world rotation, shape ``(3, 3)``.
        contour0: First section loops, shape ``(1 or 2, resolution, 3)``.
        contour1: Second section loops, with the same shape and vertex order.
        cap_start: Include a proximal solid or annular cap.
        cap_end: Include a distal solid or annular cap.

    Returns:
        World vertices ``(V, 3)`` and int32 triangle indices ``(F, 3)``.

    Raises:
        ValueError: If loop shapes or station data are invalid.
    """
    contour0, contour1 = np.asarray(contour0), np.asarray(contour1)
    if contour0.shape != contour1.shape:
        raise ValueError(
            "swept-segment endpoint contours must have matching shapes and topology"
        )
    return _sweep_contours(
        np.asarray([p0, p1]),
        np.asarray([frame0, frame1]),
        np.stack((contour0, contour1)),
        cap_start=cap_start,
        cap_end=cap_end,
    )
