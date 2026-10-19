"""Viser surfaces for helicoidal lattice arms."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import numpy as np

from soromox.rendering.config import RendererConfig
from soromox.rendering.config.colors import validate_rgb
from soromox.rendering.cross_sections import CrossSection, sweep_cross_sections
from soromox.rendering.viser_renderer import ViserRenderer
from soromox.systems.components import CrossSectionGeometry, StrandSection
from soromox.systems.gvs.helicoidal import Helicoidal
from soromox.systems.gvs.params import HelicoidalHelixParams

__all__ = [
    "HelicoidalModuleVisual",
    "HelicoidalVisualConfig",
    "HelicoidalViserRenderer",
]


@dataclass(frozen=True)
class HelicoidalModuleVisual:
    """Helix geometry for one link, expressed in its material frame.

    Attributes:
        segment_index: Index of the GVS link carrying the lattice.
        turns: Number of turns along the link's reference length.
        strands_per_family: Equally spaced helices in each winding direction.
        strand_section: Shared strand geometry. Height follows the projected
            outward radial direction (local y); width follows local z. Defaults
            to a solid circular section with a 2 mm radius.
        ring_radius: Circular end-ring tube radius [m], independent of strand
            shape. Defaults to 2 mm.
        phase: Starting azimuth of the negative winding family [rad].
        family_phase: Additional azimuth of the positive winding family [rad].

    The circular link cross section specifies the helix centerline radius.
    Geometry follows the rod's material frames, including bending and twist.
    """

    segment_index: int
    turns: float = 3.0
    strands_per_family: int = 3
    phase: float = 0.0
    family_phase: float = np.pi / 6
    strand_section: StrandSection = field(
        default_factory=lambda: StrandSection.circular(0.002)
    )
    ring_radius: float = 0.002

    def __post_init__(self) -> None:
        if not isinstance(self.strand_section, StrandSection):
            raise TypeError("strand_section must be a StrandSection.")
        self.strand_section.validate_for_update()
        for name, minimum in (("segment_index", 0), ("strands_per_family", 1)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("turns", "ring_radius"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("phase", "family_phase"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")

    @classmethod
    def from_helix_params(
        cls,
        segment_index: int,
        params: HelicoidalHelixParams,
        *,
        ring_radius: float = 0.002,
        strand_section: StrandSection | None = None,
    ) -> HelicoidalModuleVisual:
        """Copy strand geometry and winding parameters from helix inputs.

        Supply ``strand_section`` explicitly for property-only helix inputs.
        The link's existing circular envelope supplies the centreline radius;
        it should match ``params.radius_base`` and ``params.radius_tip``.
        End-ring tube radius defaults to 2 mm, independently of strand geometry.
        """
        if not isinstance(params, HelicoidalHelixParams):
            raise TypeError("params must be HelicoidalHelixParams.")
        section = params.strand_section if strand_section is None else strand_section
        if section is None:
            raise ValueError(
                "Helix inputs have no strand geometry; supply strand_section explicitly."
            )
        if not isinstance(section, StrandSection):
            raise TypeError("strand_section must be a StrandSection.")
        return cls(
            segment_index=segment_index,
            turns=float(params.num_turns),
            strands_per_family=params.helices_per_direction,
            phase=float(params.phase),
            family_phase=float(params.winding_family_phase_offset),
            strand_section=section,
            ring_radius=ring_radius,
        )

    @property
    def strand_radial_extent(self) -> float:
        """Half-extent of the strand along its local radial direction [m]."""
        return float(self.strand_section.radial_extent)


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


def _strand_frames(centers: np.ndarray, radial: np.ndarray) -> np.ndarray:
    """Build strand frames: tangent x, projected radial y, right-handed z."""
    tangent = np.gradient(centers, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    normal = radial - np.sum(radial * tangent, axis=1, keepdims=True) * tangent
    singular = np.linalg.norm(normal, axis=1) < 1e-8
    if np.any(singular):
        axis = np.eye(3)[np.argmin(np.abs(tangent[singular]), axis=1)]
        normal[singular] = np.cross(tangent[singular], axis)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    return np.stack((tangent, normal, np.cross(tangent, normal)), axis=-1)


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
        self._strand_sections = {
            m.segment_index: CrossSection.from_strand_section(m.strand_section)
            for m in self.modules
        }
        start, end = self._segment_bounds(self.num_points)
        for m in self.modules:
            sections = self._cross_sections[
                start[m.segment_index] : end[m.segment_index]
            ]
            if any(s.geometry != CrossSectionGeometry.CIRCULAR for s in sections):
                raise ValueError(
                    "Lattice links require circular cross-section envelopes."
                )
            radial_extent = m.strand_radial_extent
            if any(s.outer_dimensions[0] <= radial_extent for s in sections):
                raise ValueError(
                    "Strand radial extent must be smaller than the lattice radius."
                )
            if any(
                s.outer_dimensions[0] <= m.ring_radius
                for s in (sections[0], sections[-1])
            ):
                raise ValueError(
                    "ring_radius must be smaller than the lattice radius at each end."
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
                vertices, faces = sweep_cross_sections(
                    points,
                    rotations,
                    self._cross_sections[start:end],
                    self._cross_section_resolution,
                )
                yield (
                    f"link_{index}/surface",
                    vertices.astype(np.float32),
                    faces,
                    cfg.interface_color,
                )
                continue
            radii = np.asarray(
                [s.outer_dimensions[0] for s in self._cross_sections[start:end]]
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
                    vertices, faces = sweep_cross_sections(
                        centers,
                        _strand_frames(centers, radial),
                        self._strand_sections[index],
                        cfg.strand_sections,
                    )
                    yield (
                        f"link_{index}/family_{sign}/strand_{strand}",
                        vertices.astype(np.float32),
                        faces,
                        color,
                    )
            # End rings have their own circular section, independent of strand shape.
            angle = np.linspace(
                0, 2 * np.pi, max(32, self._cross_section_resolution) + 1
            )
            local = np.column_stack(
                (np.zeros_like(angle), np.cos(angle), np.sin(angle))
            )
            for endpoint in (0, -1):
                radial = local @ rotations[endpoint].T
                centers = points[endpoint] + radii[endpoint] * radial
                vertices, faces = sweep_cross_sections(
                    centers,
                    _strand_frames(centers, radial),
                    CrossSection(
                        CrossSectionGeometry.CIRCULAR,
                        np.array([module.ring_radius]),
                    ),
                    cfg.strand_sections,
                )
                yield (
                    f"link_{index}/ring_{endpoint}",
                    vertices.astype(np.float32),
                    faces,
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
