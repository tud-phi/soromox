"""Viser surfaces for helicoidal lattice arms."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from soromox.rendering.config import RendererConfig
from soromox.rendering.config.colors import validate_rgb
from soromox.rendering.viser_renderer import ViserRenderer
from soromox.systems.components import CrossSectionGeometry
from soromox.systems.gvs.helicoidal import Helicoidal

__all__ = [
    "RECTANGULAR_STRAND_HEIGHT",
    "RECTANGULAR_STRAND_WIDTH",
    "HelicoidalModuleVisual",
    "HelicoidalVisualConfig",
    "HelicoidalViserRenderer",
]


# Default solid rectangular strand section [m]: the N6 module's 12 mm radial
# width and its 2.9268 mm thickness evaluated at the strand mid-radius.
RECTANGULAR_STRAND_WIDTH = 0.012
RECTANGULAR_STRAND_HEIGHT = 0.0029268


@dataclass(frozen=True)
class HelicoidalModuleVisual:
    """Helix geometry for one link, expressed in its material frame.

    Attributes:
        segment_index: Index of the GVS link carrying the lattice.
        turns: Number of turns along the link's reference length.
        strands_per_family: Equally spaced helices in each winding direction.
        strand_radius: Radius of each solid circular helix tube [m]. Also the
            tube radius of the end rings for every strand section.
        phase: Starting azimuth of the negative winding family [rad].
        family_phase: Additional azimuth of the positive winding family [rad].
        strand_shape: ``"circular"`` (the default) for tubes of
            ``strand_radius``, or ``"rectangular_hollow"`` for a solid
            rectangular section (the identifier is kept for compatibility; the
            section is solid). Leave ``None`` to infer the shape: rectangular
            when ``strand_width`` and ``strand_height`` are supplied, circular
            otherwise.
        strand_width: Extent of the rectangular section along the strand's
            local y axis, the projected outward radial direction [m]. Defaults
            to ``RECTANGULAR_STRAND_WIDTH`` for rectangular strands.
        strand_height: Extent of the rectangular section along the strand's
            local z axis, the circumferential direction [m]. Defaults to
            ``RECTANGULAR_STRAND_HEIGHT`` for rectangular strands.

    The circular link cross section specifies the helix centerline radius.
    Geometry follows the rod's material frames, including bending and twist.
    Strand frames match ``HelicoidalHelixParams``: x along the strand tangent,
    y along the outward radial direction projected normal to x, and z
    completing a right-handed frame.
    """

    segment_index: int
    turns: float = 3.0
    strands_per_family: int = 3
    strand_radius: float = 0.002
    phase: float = 0.0
    family_phase: float = np.pi / 6
    strand_shape: str | None = None
    strand_width: float | None = None
    strand_height: float | None = None

    def __post_init__(self) -> None:
        for name, minimum in (("segment_index", 0), ("strands_per_family", 1)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("turns", "strand_radius"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("phase", "family_phase"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        rectangular = ("strand_width", "strand_height")
        given = [getattr(self, name) is not None for name in rectangular]
        shape = self.strand_shape
        if shape is None:
            if any(given) and not all(given):
                raise ValueError(
                    "strand_width and strand_height must be supplied together."
                )
            shape = "rectangular_hollow" if all(given) else "circular"
            object.__setattr__(self, "strand_shape", shape)
        if shape == "circular":
            if any(given):
                raise ValueError(
                    "strand_width and strand_height apply to rectangular strands."
                )
            return
        if shape != "rectangular_hollow":
            raise ValueError('strand_shape must be "circular" or "rectangular_hollow".')
        if self.strand_width is None:
            object.__setattr__(self, "strand_width", RECTANGULAR_STRAND_WIDTH)
        if self.strand_height is None:
            object.__setattr__(self, "strand_height", RECTANGULAR_STRAND_HEIGHT)
        for name in rectangular:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    @property
    def strand_radial_extent(self) -> float:
        """Half-extent of the strand section along the outward radial axis [m]."""
        if self.strand_width is None:
            return self.strand_radius
        return self.strand_width / 2


@dataclass(frozen=True)
class HelicoidalVisualConfig:
    """Sampling and appearance of helices, end rings, and connecting links.

    Attributes:
        samples_per_turn: Minimum longitudinal sampling density of each helix.
        strand_sections: Vertices around a strand cross section.
        negative_color: RGB color of the negative winding family, in [0, 1].
        positive_color: RGB color of the positive winding family, in [0, 1].
        interface_color: RGB color of end rings and connecting links, in [0, 1].
    """

    samples_per_turn: int = 32
    strand_sections: int = 10
    negative_color: tuple[float, float, float] = (0.16, 0.46, 0.76)
    positive_color: tuple[float, float, float] = (0.93, 0.65, 0.26)
    interface_color: tuple[float, float, float] = (0.35, 0.40, 0.47)

    def __post_init__(self) -> None:
        for name, minimum in (("samples_per_turn", 8), ("strand_sections", 6)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("negative_color", "positive_color", "interface_color"):
            validate_rgb(getattr(self, name), name=name)


def _side_faces(rings: int, sections: int) -> list[tuple[int, int, int]]:
    """Triangulate the walls between consecutive loops, facing outward.

    Vertex ``i * sections + j`` is loop ``i``, position ``j``. Loops must wind
    counterclockwise about the sweep tangent for outward normals.
    """
    faces = []
    for i in range(rings - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections
            faces.extend(((a, b, a + sections), (b, b + sections, a + sections)))
    return faces


@lru_cache(maxsize=64)
def _surface_faces(rings: int, sections: int) -> np.ndarray:
    """Triangulate a capped sweep with outward-facing triangles."""
    faces = _side_faces(rings, sections)
    for j in range(sections):
        k = (j + 1) % sections
        faces.extend(
            (
                (rings * sections, k, j),
                (
                    rings * sections + 1,
                    (rings - 1) * sections + j,
                    (rings - 1) * sections + k,
                ),
            )
        )
    return np.asarray(faces, dtype=np.uint32)


def _capped_vertices(rings: np.ndarray) -> np.ndarray:
    return np.concatenate((rings.reshape(-1, 3), rings[[0, -1]].mean(axis=1))).astype(
        np.float32
    )


def _strand_frames(
    centers: np.ndarray, radial: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strand section axes (local y, local z) at each station.

    Local x follows the deformed centerline tangent, y is the outward radial
    direction projected normal to x, and z completes a right-handed frame,
    matching the ``HelicoidalHelixParams`` strand frame convention.
    """
    tangent = np.gradient(centers, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    normal = radial - np.sum(radial * tangent, axis=1, keepdims=True) * tangent
    singular = np.linalg.norm(normal, axis=1) < 1e-8
    if np.any(singular):
        axis = np.eye(3)[np.argmin(np.abs(tangent[singular]), axis=1)]
        normal[singular] = np.cross(tangent[singular], axis)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    return normal, np.cross(tangent, normal)


def _strand_vertices(
    centers: np.ndarray, radial: np.ndarray, radius: float, sections: int
) -> np.ndarray:
    """Sweep circular sections normal to a deformed strand's tangent."""
    normal, binormal = _strand_frames(centers, radial)
    angle = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = centers[:, None] + radius * (
        np.cos(angle)[None, :, None] * normal[:, None]
        + np.sin(angle)[None, :, None] * binormal[:, None]
    )
    return _capped_vertices(rings)


def _rectangle_loop(width: float, height: float) -> np.ndarray:
    """Corners of a centred rectangle in the (y, z) plane, counterclockwise."""
    y, z = width / 2, height / 2
    return np.array([[y, -z], [y, z], [-y, z], [-y, -z]])


def _rectangular_vertices(
    centers: np.ndarray, radial: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Sweep solid rectangular sections normal to a deformed strand's tangent.

    Width lies along local y and height along local z. Each station carries the
    four corners of ``_rectangle_loop`` followed by the two cap centres from
    ``_capped_vertices``, so vertex counts and ordering stay fixed during
    animation.
    """
    normal, binormal = _strand_frames(centers, radial)
    loop = _rectangle_loop(width, height)
    rings = (
        centers[:, None]
        + loop[None, :, 0, None] * normal[:, None]
        + loop[None, :, 1, None] * binormal[:, None]
    )
    return _capped_vertices(rings)


def _strand_mesh(
    module: HelicoidalModuleVisual,
    cfg: HelicoidalVisualConfig,
    centers: np.ndarray,
    radial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the module's strand section along a centerline."""
    if module.strand_width is None:
        return (
            _strand_vertices(
                centers, radial, module.strand_radius, cfg.strand_sections
            ),
            _surface_faces(len(centers), cfg.strand_sections),
        )
    return (
        _rectangular_vertices(
            centers, radial, module.strand_width, module.strand_height
        ),
        _surface_faces(len(centers), 4),
    )


class HelicoidalViserRenderer(ViserRenderer):
    """Display helicoidal strands, end rings, and connecting link surfaces.

    Supply ``modules`` to select the lattice links and their helix geometry.
    By default every link is a lattice module. Links outside this selection
    are rendered with their cross-section surfaces. The inherited static,
    playback, export, and live APIs include contiguous tendon overlays.
    """

    def __init__(
        self,
        robot: Helicoidal,
        *,
        modules: tuple[HelicoidalModuleVisual, ...] | None = None,
        visual_config: HelicoidalVisualConfig | None = None,
        config: RendererConfig | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(robot, Helicoidal):
            raise TypeError("HelicoidalViserRenderer requires a Helicoidal robot.")
        self.visual_config = visual_config or HelicoidalVisualConfig()
        if not isinstance(self.visual_config, HelicoidalVisualConfig):
            raise TypeError("visual_config must be a HelicoidalVisualConfig instance.")
        self.modules = (
            tuple(HelicoidalModuleVisual(i) for i in range(robot.num_segments))
            if modules is None
            else tuple(modules)
        )
        if not self.modules or not all(
            isinstance(m, HelicoidalModuleVisual) for m in self.modules
        ):
            raise ValueError("modules must contain HelicoidalModuleVisual entries.")
        indices = [m.segment_index for m in self.modules]
        if len(set(indices)) != len(indices) or max(indices) >= robot.num_segments:
            raise ValueError(
                "Module segment indices must be unique and within the robot."
            )
        lengths = np.asarray(robot.segment_length)
        if np.any(lengths[indices] <= 0):
            raise ValueError("Lattice modules must have positive length.")
        cfg = config or RendererConfig()
        # The shared sweep layout allocates extra stations by reference length.
        density = max(
            m.turns * self.visual_config.samples_per_turn / lengths[m.segment_index]
            for m in self.modules
        )
        count = max(
            cfg.geometry.num_points,
            2 * len(lengths) + int(np.ceil(lengths.sum() * density)),
        )
        cfg = replace(
            cfg,
            geometry=replace(cfg.geometry, num_points=count, backbone_style="swept"),
        )
        # Validate geometry before opening a server.
        auto_start = kwargs.pop("auto_start", True)
        super().__init__(robot, config=cfg, auto_start=False, **kwargs)
        self._module_by_link = {m.segment_index: m for m in self.modules}
        for m in self.modules:
            start, end = self._segment_bounds(self.num_points)
            sections = self._cross_sections[
                start[m.segment_index] : end[m.segment_index]
            ]
            if any(s.geometry != CrossSectionGeometry.CIRCULAR for s in sections):
                raise ValueError(
                    "Lattice links require circular cross-section envelopes."
                )
            if any(s.dimensions[0] <= m.strand_radial_extent for s in sections):
                raise ValueError(
                    "The strand's radial extent (strand_radius or strand_width / 2) "
                    "must be smaller than the lattice radius."
                )
        if auto_start:
            self.start()

    def _meshes(self, curve: np.ndarray, frames: np.ndarray):
        """Yield named meshes from the shared backbone samples and frames."""
        cfg = self.visual_config
        starts, ends = self._segment_bounds(self.num_points)
        boundaries = np.r_[0.0, np.cumsum(np.asarray(self.robot.segment_length))]
        for index, (start, end) in enumerate(zip(starts, ends)):
            if end - start < 2:
                continue
            points, rotations = curve[start:end], frames[start:end]
            module = self._module_by_link.get(index)
            if module is None:
                contours = np.asarray(self._cross_section_contours[start:end])
                rings = points[:, None] + np.einsum("nij,nkj->nki", rotations, contours)
                yield (
                    f"link_{index}/surface",
                    _capped_vertices(rings),
                    _surface_faces(len(points), rings.shape[1]),
                    cfg.interface_color,
                )
                continue
            radii = np.asarray(
                [s.dimensions[0] for s in self._cross_sections[start:end]]
            )
            fraction = (
                np.asarray(self._backbone_abscissae[start:end]) - boundaries[index]
            ) / (boundaries[index + 1] - boundaries[index])
            for sign, shift, color in (
                (-1, 0, cfg.negative_color),
                (1, module.family_phase, cfg.positive_color),
            ):
                for strand in range(module.strands_per_family):
                    theta = (
                        sign * 2 * np.pi * module.turns * fraction
                        + module.phase
                        + shift
                        + 2 * np.pi * strand / module.strands_per_family
                    )
                    local = np.column_stack(
                        (np.zeros_like(theta), np.cos(theta), np.sin(theta))
                    )
                    radial = np.einsum("nij,nj->ni", rotations, local)
                    centers = points + radii[:, None] * radial
                    vertices, faces = _strand_mesh(module, cfg, centers, radial)
                    yield (
                        f"link_{index}/family_{sign}/strand_{strand}",
                        vertices,
                        faces,
                        color,
                    )
            # End rings are interface parts, not strands: they share each
            # module's end frame and keep a circular tube of strand_radius.
            angle = np.linspace(
                0, 2 * np.pi, max(32, self._cross_section_resolution) + 1
            )
            local = np.column_stack(
                (np.zeros_like(angle), np.cos(angle), np.sin(angle))
            )
            for endpoint in (0, -1):
                radial = local @ rotations[endpoint].T
                centers = points[endpoint] + radii[endpoint] * radial
                yield (
                    f"link_{index}/ring_{endpoint}",
                    _strand_vertices(
                        centers, radial, module.strand_radius, cfg.strand_sections
                    ),
                    _surface_faces(len(angle), cfg.strand_sections),
                    cfg.interface_color,
                )

    def _build_robot_geometry(
        self, curves, point_colors, *, material_frames, base_plate_color
    ) -> None:
        if self._server is None or self._scene_handles is None:
            return
        for handles in self._scene_handles.backbone_points:
            for handle in handles:
                handle.remove()
        for handle in self._scene_handles.base_plates:
            handle.remove()
        self._scene_handles.backbone_points = []
        self._scene_handles.base_plates = []
        if not hasattr(self, "_appearance_extent"):
            self._fit_scene_bounds(curves)
        self._ground_base_axes = material_frames[:, 0, :, 0]
        self._add_ground_plane(curves[:, 0])
        for robot_index, (curve, frames) in enumerate(zip(curves, material_frames)):
            handles = []
            for name, vertices, faces, color in self._meshes(curve, frames):
                handles.append(
                    self._add_mesh(
                        name=f"/robots/robot_{robot_index}/helicoidal/{name}",
                        vertices=vertices,
                        faces=faces,
                        color=tuple(int(round(255 * v)) for v in color),
                        material=self._material,
                        flat_shading=self._flat_shading,
                        wireframe=self._wireframe,
                        cast_shadow=self._backbone_cast_shadow,
                    )
                )
            self._scene_handles.backbone_points.append(handles)
        self._scene_handles.geometry_initialized = True
        self._scene_handles.num_robots = len(curves)
        self._scene_handles.num_backbone_points = curves.shape[1]

    def _update_robot_geometry(
        self, curves, material_frames, *, atomic: bool = True
    ) -> None:
        if self._server is None or self._scene_handles is None:
            return
        with self._server.atomic() if atomic else nullcontext():
            for handles, curve, frames in zip(
                self._scene_handles.backbone_points, curves, material_frames
            ):
                for handle, (_, vertices, _, _) in zip(
                    handles, self._meshes(curve, frames), strict=True
                ):
                    handle.vertices = vertices
