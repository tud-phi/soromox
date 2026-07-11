"""Viser renderer specialized for the pneumatic I-SUPPORT robot."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter

from soromox.rendering.color_config import validate_rgb
from soromox.rendering.viser_renderer import (
    LiveModeController,
    ViserRenderer,
    _direction_to_quaternion,
)
from soromox.systems.pcs.isupport import ISupport

__all__ = ["ISupportVisualConfig", "ISupportViserRenderer"]


@dataclass(frozen=True)
class ISupportVisualConfig:
    """Physical and stylistic defaults for :class:`ISupportViserRenderer`."""

    interface_color: tuple[float, float, float] = (0.08, 0.20, 0.72)
    chamber_color: tuple[float, float, float] = (0.94, 0.91, 0.72)
    chamber_aspect_ratio: float = 21.72 / 13.03
    bellows_pitch: float = 0.010
    bellows_amplitude: float = 0.15
    chamber_sections: int = 20
    samples_per_fold: int = 4
    spacers_per_segment: int = 6
    spacer_thickness: float = 0.0015
    spacer_color: tuple[float, float, float] = (0.72, 0.82, 0.88)
    spacer_opacity: float = 0.35
    pressure_colormap: str = "Blues"
    pressure_range: tuple[float, float] = (0.0, 3.0e5)
    pressure_colormap_start: float = 0.15
    pressure_label_offset: float = 0.012

    def __post_init__(self) -> None:
        positive_fields = (
            "chamber_aspect_ratio",
            "bellows_pitch",
            "spacer_thickness",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.bellows_amplitude) or not (
            0.0 <= self.bellows_amplitude < 1.0
        ):
            raise ValueError("bellows_amplitude must be finite and in [0, 1).")
        if self.chamber_sections < 6:
            raise ValueError("chamber_sections must be at least 6.")
        if self.samples_per_fold < 2:
            raise ValueError("samples_per_fold must be at least 2.")
        if self.spacers_per_segment < 0:
            raise ValueError("spacers_per_segment must be nonnegative.")
        if not np.isfinite(self.spacer_opacity) or not (
            0.0 <= self.spacer_opacity <= 1.0
        ):
            raise ValueError("spacer_opacity must be finite and in [0, 1].")
        pressure_range = np.asarray(self.pressure_range, dtype=np.float64)
        if pressure_range.shape != (2,) or not np.all(np.isfinite(pressure_range)):
            raise ValueError("pressure_range must contain two finite values.")
        if pressure_range[1] <= pressure_range[0]:
            raise ValueError("pressure_range maximum must exceed its minimum.")
        if not np.isfinite(self.pressure_colormap_start) or not (
            0.0 <= self.pressure_colormap_start < 1.0
        ):
            raise ValueError("pressure_colormap_start must be finite and in [0, 1).")
        if (
            not np.isfinite(self.pressure_label_offset)
            or self.pressure_label_offset < 0
        ):
            raise ValueError("pressure_label_offset must be finite and nonnegative.")
        try:
            colormaps.get_cmap(self.pressure_colormap)
        except ValueError as exc:
            raise ValueError(
                f"Unknown Matplotlib colormap {self.pressure_colormap!r}."
            ) from exc
        validate_rgb(self.interface_color, name="interface_color")
        validate_rgb(self.chamber_color, name="chamber_color")
        validate_rgb(self.spacer_color, name="spacer_color")


@dataclass(frozen=True)
class _PneumaticVisualSpec:
    pneumatic_idx: int
    radius: float
    s_start: float
    s_end: float
    ring_slice: slice
    spacer_sample_indices: tuple[int, ...]
    num_folds: int
    faces: np.ndarray


@dataclass(frozen=True)
class _InterfaceVisualSpec:
    slot_idx: int
    modeled: bool
    sample_indices: tuple[int, ...]
    thickness: float
    radius: float


@dataclass
class _ISupportRobotHandles:
    chamber_handles: list[Any]
    interface_handles: list[Any]
    spacer_handles: list[Any]
    pressure_label_handles: list[Any]


def _tube_faces(num_rings: int, sections: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for ring_idx in range(num_rings - 1):
        ring_start = ring_idx * sections
        next_start = (ring_idx + 1) * sections
        for section_idx in range(sections):
            nxt = (section_idx + 1) % sections
            faces.append(
                (ring_start + section_idx, ring_start + nxt, next_start + section_idx)
            )
            faces.append((ring_start + nxt, next_start + nxt, next_start + section_idx))

    start_center = num_rings * sections
    end_center = start_center + 1
    last_ring = (num_rings - 1) * sections
    for section_idx in range(sections):
        nxt = (section_idx + 1) % sections
        faces.append((start_center, section_idx, nxt))
        faces.append((end_center, last_ring + nxt, last_ring + section_idx))
    return np.asarray(faces, dtype=np.uint32)


class ISupportViserRenderer(ViserRenderer):
    """Render the chambers, rigid interfaces, and spacers of an I-SUPPORT robot."""

    def __init__(
        self,
        robot: ISupport,
        *args,
        visual_config: ISupportVisualConfig | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(robot, ISupport):
            raise TypeError("ISupportViserRenderer requires an ISupport robot.")
        resolved_visual_config = visual_config or ISupportVisualConfig()
        if not isinstance(resolved_visual_config, ISupportVisualConfig):
            raise TypeError("visual_config must be an ISupportVisualConfig instance.")
        super().__init__(robot, *args, **kwargs)
        self.visual_config = resolved_visual_config
        self._current_geometry_q: Array | None = None
        self._current_pressures: np.ndarray | None = None
        self._pressure_ts: np.ndarray | None = None
        self._pressure_frame_idx = 0
        self._show_pressure_labels = False
        self._force_pressure_labels = False
        self._current_world_poses: np.ndarray | None = None
        self._robot_visual_handles: list[_ISupportRobotHandles] = []
        (
            self._pneumatic_specs,
            self._interface_specs,
            self._sample_s,
        ) = self._build_visual_layout()

    @staticmethod
    def _as_batch(q: Array) -> Array:
        q_array = jnp.asarray(q)
        return q_array[None, :] if q_array.ndim == 1 else q_array

    @staticmethod
    def _color_tuple(
        color: tuple[float, float, float] | np.ndarray,
    ) -> tuple[int, int, int]:
        return tuple(np.asarray(np.clip(color, 0.0, 1.0) * 255.0, dtype=np.uint8))

    def _pressure_color(self, pressure_pa: float) -> tuple[int, int, int]:
        if not np.isfinite(pressure_pa):
            return (145, 145, 145)
        pressure_min, pressure_max = self.visual_config.pressure_range
        normalized = np.clip(
            (pressure_pa - pressure_min) / (pressure_max - pressure_min),
            0.0,
            1.0,
        )
        sample = (
            self.visual_config.pressure_colormap_start
            + (1.0 - self.visual_config.pressure_colormap_start) * normalized
        )
        rgb = np.asarray(
            colormaps.get_cmap(self.visual_config.pressure_colormap)(float(sample))[:3]
        )
        return tuple(int(round(255.0 * component)) for component in rgb)

    def _static_pressures(
        self, pressures: Array | np.ndarray | None, *, num_robots: int
    ) -> np.ndarray | None:
        if pressures is None:
            return None
        values = np.asarray(pressures, dtype=np.float64)
        expected = self.robot.num_actuators
        if values.shape == (expected,):
            return np.broadcast_to(values, (num_robots, expected)).copy()
        if values.shape == (num_robots, expected):
            return values.copy()
        raise ValueError(
            "pressures must have shape (num_actuators,) or "
            f"(num_robots, num_actuators); got {values.shape}."
        )

    def _sequence_pressures(
        self,
        pressures: Array | np.ndarray | None,
        *,
        num_robots: int,
        num_frames: int,
    ) -> np.ndarray | None:
        if pressures is None:
            return None
        values = np.asarray(pressures, dtype=np.float64)
        expected = self.robot.num_actuators
        if values.shape == (expected,):
            return np.broadcast_to(values, (num_robots, num_frames, expected)).copy()
        if values.shape == (num_frames, expected):
            return np.broadcast_to(
                values[None, ...], (num_robots, num_frames, expected)
            ).copy()
        if values.shape == (num_robots, num_frames, expected):
            return values.copy()
        raise ValueError(
            "sequence pressures must have shape (num_actuators,), "
            "(T, num_actuators), or (N, T, num_actuators); "
            f"got {values.shape}."
        )

    def _build_visual_layout(
        self,
    ) -> tuple[
        tuple[_PneumaticVisualSpec, ...],
        tuple[_InterfaceVisualSpec, ...],
        np.ndarray,
    ]:
        lengths = np.asarray(self.robot.L, dtype=np.float64)
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(lengths)))
        segment_mapping = np.asarray(
            self.robot.pcs_segment_to_pneumatic_segment, dtype=np.int64
        )
        rigid_selector = np.asarray(self.robot.pcs_segment_is_rigid, dtype=bool)
        sample_s: list[float] = []
        pneumatic_specs: list[_PneumaticVisualSpec] = []

        for pneumatic_idx in range(self.robot.num_pneumatic_segments):
            pcs_indices = np.flatnonzero(
                (segment_mapping == pneumatic_idx) & ~rigid_selector
            )
            if pcs_indices.size == 0:
                raise ValueError(
                    f"Pneumatic segment {pneumatic_idx} has no PCS segments."
                )
            expected = np.arange(pcs_indices[0], pcs_indices[-1] + 1)
            if not np.array_equal(pcs_indices, expected):
                raise ValueError(
                    "PCS segments for each pneumatic segment must be contiguous."
                )

            s_start = float(cumulative_lengths[pcs_indices[0]])
            s_end = float(cumulative_lengths[pcs_indices[-1] + 1])
            segment_length = s_end - s_start
            num_folds = max(
                1, int(np.round(segment_length / self.visual_config.bellows_pitch))
            )
            num_rings = num_folds * self.visual_config.samples_per_fold + 1
            ring_start = len(sample_s)
            sample_s.extend(np.linspace(s_start, s_end, num_rings).tolist())
            ring_slice = slice(ring_start, len(sample_s))

            if self.visual_config.spacers_per_segment:
                fractions = np.arange(1, self.visual_config.spacers_per_segment + 1)
                fractions = fractions / (self.visual_config.spacers_per_segment + 1)
                spacer_s = s_start + fractions * segment_length
            else:
                spacer_s = np.empty((0,), dtype=np.float64)
            spacer_start = len(sample_s)
            sample_s.extend(spacer_s.tolist())
            spacer_indices = tuple(range(spacer_start, len(sample_s)))

            pneumatic_specs.append(
                _PneumaticVisualSpec(
                    pneumatic_idx=pneumatic_idx,
                    radius=float(self.robot.r[pcs_indices[0]]),
                    s_start=s_start,
                    s_end=s_end,
                    ring_slice=ring_slice,
                    spacer_sample_indices=spacer_indices,
                    num_folds=num_folds,
                    faces=_tube_faces(num_rings, self.visual_config.chamber_sections),
                )
            )

        rigid_indices = np.flatnonzero(rigid_selector).tolist()
        rigid_physical_indices = [
            i
            for i, is_rigid in enumerate(self.robot.structure.rigid_segment_selector)
            if is_rigid
        ]
        if len(rigid_indices) != len(rigid_physical_indices):
            raise ValueError("Rigid segment topology does not match the PCS layout.")
        interface_specs: list[_InterfaceVisualSpec] = []
        for physical_idx, pcs_idx in zip(rigid_physical_indices, rigid_indices):
            start_idx = len(sample_s)
            sample_s.extend(
                [
                    float(cumulative_lengths[pcs_idx]),
                    float(cumulative_lengths[pcs_idx + 1]),
                ]
            )
            interface_specs.append(
                _InterfaceVisualSpec(
                    slot_idx=physical_idx,
                    modeled=True,
                    sample_indices=(start_idx, start_idx + 1),
                    thickness=float(lengths[pcs_idx]),
                    radius=float(self.robot.r[pcs_idx]),
                )
            )

        return (
            tuple(pneumatic_specs),
            tuple(interface_specs),
            np.asarray(sample_s, dtype=np.float64),
        )

    def _sample_world_poses(self, q: Array, curves: np.ndarray) -> np.ndarray:
        s_points = jnp.asarray(self._sample_s)
        poses = np.array(
            jax.vmap(lambda q_i: self.robot.forward_kinematics_batched(q_i, s_points))(
                q
            ),
            copy=True,
        )
        base_origin = np.asarray(self.robot.g0, dtype=np.float64)[:3, 3]
        base_offsets = curves[:, 0, :] - base_origin
        poses[..., :3, 3] += base_offsets[:, None, :]
        return poses

    def _chamber_vertices(
        self,
        poses: np.ndarray,
        spec: _PneumaticVisualSpec,
        chamber_idx: int,
    ) -> np.ndarray:
        segment_poses = poses[spec.ring_slice]
        rotations = segment_poses[:, :3, :3]
        centerline = segment_poses[:, :3, 3]
        pneumatic_idx = spec.pneumatic_idx
        local_offset = np.asarray(
            self.robot.local_chamber_offsets(pneumatic_idx)[chamber_idx]
        )
        radial_local = local_offset / np.linalg.norm(local_offset)
        tangential_local = np.cross(np.array([1.0, 0.0, 0.0]), radial_local)
        radial = rotations @ radial_local
        tangential = rotations @ tangential_local
        chamber_centers = centerline + rotations @ local_offset

        equivalent_radius = float(self.robot.params.chamber_outer_radius[pneumatic_idx])
        aspect_scale = np.sqrt(self.visual_config.chamber_aspect_ratio)
        radial_radius = equivalent_radius * aspect_scale
        tangential_radius = equivalent_radius / aspect_scale
        num_rings = segment_poses.shape[0]
        phase = np.linspace(0.0, 2.0 * np.pi * spec.num_folds, num_rings)
        corrugation = 1.0 - self.visual_config.bellows_amplitude * np.cos(phase)
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.visual_config.chamber_sections,
            endpoint=False,
        )

        cross_sections = (
            radial_radius * np.cos(angles)[None, :, None] * radial[:, None, :]
            + tangential_radius * np.sin(angles)[None, :, None] * tangential[:, None, :]
        )
        rings = chamber_centers[:, None, :] + corrugation[:, None, None] * (
            cross_sections
        )
        cap_centers = np.stack([chamber_centers[0], chamber_centers[-1]])
        return np.concatenate([rings.reshape(-1, 3), cap_centers], axis=0).astype(
            np.float32
        )

    @staticmethod
    def _disk_pose(
        pose: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        return pose[:3, 3], _direction_to_quaternion(pose[:3, 0])

    def _interface_pose(
        self, poses: np.ndarray, spec: _InterfaceVisualSpec
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if not spec.modeled:
            return self._disk_pose(poses[spec.sample_indices[0]])
        p0 = poses[spec.sample_indices[0], :3, 3]
        p1 = poses[spec.sample_indices[1], :3, 3]
        direction = p1 - p0
        if np.linalg.norm(direction) < 1e-9:
            direction = poses[spec.sample_indices[0], :3, 0]
        return 0.5 * (p0 + p1), _direction_to_quaternion(direction)

    def _add_disk(
        self,
        name: str,
        *,
        radius: float,
        thickness: float,
        color: tuple[float, float, float],
        opacity: float | None,
        position: np.ndarray,
        wxyz: tuple[float, float, float, float],
    ) -> Any:
        mesh = self._make_cylinder_trimesh(
            length=thickness,
            radius=radius,
            color=color,
        )
        return self._server.scene.add_mesh_simple(
            name=name,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=self._color_tuple(color),
            opacity=opacity,
            position=tuple(position),
            wxyz=wxyz,
            material=self._material,
            flat_shading=self._flat_shading,
            wireframe=self._wireframe,
            cast_shadow=self._backbone_cast_shadow,
        )

    def _pressure_label_position(
        self,
        poses: np.ndarray,
        spec: _PneumaticVisualSpec,
        chamber_idx: int,
    ) -> np.ndarray:
        ring_poses = poses[spec.ring_slice]
        pose = ring_poses[len(ring_poses) // 2]
        local_offset = np.asarray(
            self.robot.local_chamber_offsets(spec.pneumatic_idx)[chamber_idx],
            dtype=np.float64,
        )
        offset_norm = float(np.linalg.norm(local_offset))
        radial_world = pose[:3, :3] @ (local_offset / offset_norm)
        outer_radius = float(self.robot.params.chamber_outer_radius[spec.pneumatic_idx])
        label_radius = (
            offset_norm + outer_radius + self.visual_config.pressure_label_offset
        )
        return pose[:3, 3] + label_radius * radial_world

    def _add_pressure_labels(
        self,
        robot_idx: int,
        poses: np.ndarray,
    ) -> list[Any]:
        if (
            self._server is None
            or self._current_pressures is None
            or not (self._force_pressure_labels or self._show_pressure_labels)
        ):
            return []
        labels: list[Any] = []
        for spec in self._pneumatic_specs:
            for chamber_idx in range(self.robot.num_chambers_per_segment):
                labels.append(
                    self._server.scene.add_label(
                        name=(
                            f"/robots/robot_{robot_idx}/isupport/pressure_labels/"
                            f"segment_{spec.pneumatic_idx}/chamber_{chamber_idx}"
                        ),
                        text="",
                        position=self._pressure_label_position(
                            poses, spec, chamber_idx
                        ),
                        anchor="center-center",
                        depth_test=False,
                        font_screen_scale=0.8,
                        visible=(
                            self._force_pressure_labels or self._show_pressure_labels
                        ),
                    )
                )
        return labels

    def _apply_pressure_visuals(self, world_poses: np.ndarray) -> None:
        cfg = self.visual_config
        for robot_idx, (robot_handles, poses) in enumerate(
            zip(self._robot_visual_handles, world_poses, strict=True)
        ):
            if self._current_pressures is None:
                for handle in robot_handles.chamber_handles:
                    handle.color = self._color_tuple(cfg.chamber_color)
                for handle in robot_handles.pressure_label_handles:
                    handle.visible = False
                continue

            labels_enabled = self._force_pressure_labels or self._show_pressure_labels
            if labels_enabled and not robot_handles.pressure_label_handles:
                robot_handles.pressure_label_handles = self._add_pressure_labels(
                    robot_idx, poses
                )

            pressures = self._current_pressures[robot_idx]
            flat_chamber_idx = 0
            for spec in self._pneumatic_specs:
                for chamber_idx in range(self.robot.num_chambers_per_segment):
                    pressure = float(pressures[flat_chamber_idx])
                    chamber_handle = robot_handles.chamber_handles[flat_chamber_idx]
                    chamber_handle.color = self._pressure_color(pressure)
                    if labels_enabled:
                        label_handle = robot_handles.pressure_label_handles[
                            flat_chamber_idx
                        ]
                        value_text = (
                            f"{pressure / 1.0e3:.1f} kPa"
                            if np.isfinite(pressure)
                            else "--"
                        )
                        label_handle.text = (
                            f"Segment {spec.pneumatic_idx + 1} · "
                            f"Chamber {chamber_idx + 1}\n{value_text}"
                        )
                        label_handle.position = self._pressure_label_position(
                            poses, spec, chamber_idx
                        )
                        label_handle.visible = True
                    flat_chamber_idx += 1

    def _remove_pressure_labels(self) -> None:
        for robot_handles in self._robot_visual_handles:
            for handle in robot_handles.pressure_label_handles:
                if hasattr(handle, "remove"):
                    handle.remove()
            robot_handles.pressure_label_handles = []

    def _set_pressure_labels_visible(self, visible: bool) -> None:
        self._show_pressure_labels = bool(visible)
        if not self._show_pressure_labels:
            self._remove_pressure_labels()
            return
        if self._current_pressures is None or self._current_world_poses is None:
            return
        if self._server is None:
            return
        with self._server.atomic():
            self._apply_pressure_visuals(self._current_world_poses)

    def _pressure_colorbar_image(self) -> np.ndarray:
        cfg = self.visual_config
        base_colormap = colormaps.get_cmap(cfg.pressure_colormap)
        samples = np.linspace(cfg.pressure_colormap_start, 1.0, 256)
        pressure_colormap = LinearSegmentedColormap.from_list(
            "isupport_pressure", base_colormap(samples)
        )

        figure = Figure(figsize=(3.0, 0.65), dpi=120)
        figure.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_axes((0.08, 0.48, 0.84, 0.28))
        colorbar = ColorbarBase(
            axes,
            cmap=pressure_colormap,
            norm=Normalize(*cfg.pressure_range),
            orientation="horizontal",
        )
        colorbar.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1.0e3:g}")
        )
        colorbar.ax.xaxis.set_major_locator(
            FixedLocator(np.linspace(*cfg.pressure_range, 5))
        )
        colorbar.ax.tick_params(labelsize=8, length=2, pad=1)
        colorbar.set_label("Pressure [kPa]", fontsize=8, labelpad=2)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()

    def _setup_pressure_gui(self) -> None:
        if self._server is None:
            return
        existing = self._gui_handles.get("show_pressure_labels")
        if existing is not None:
            existing.value = self._show_pressure_labels
            return
        with self._server.gui.add_folder("I-SUPPORT"):
            checkbox = self._server.gui.add_checkbox(
                "Show pressure labels",
                initial_value=self._show_pressure_labels,
            )
            self._gui_handles["show_pressure_labels"] = checkbox
            self._gui_handles["pressure_colorbar"] = self._server.gui.add_image(
                self._pressure_colorbar_image(),
                format="png",
            )

        @checkbox.on_update
        def _(event):
            self._set_pressure_labels_visible(event.target.value)

    def render_frame(
        self,
        q: Array,
        *,
        pressures: Array | np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Render one I-SUPPORT configuration, optionally with pressure labels."""
        self._current_geometry_q = self._as_batch(q)
        self._current_pressures = self._static_pressures(
            pressures, num_robots=int(self._current_geometry_q.shape[0])
        )
        self._pressure_ts = None
        self._force_pressure_labels = pressures is not None
        return super().render_frame(q, **kwargs)

    def show(
        self,
        q: Array,
        *,
        pressures: Array | np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Display one I-SUPPORT configuration with optional chamber pressures."""
        self._current_geometry_q = self._as_batch(q)
        self._current_pressures = self._static_pressures(
            pressures, num_robots=int(self._current_geometry_q.shape[0])
        )
        self._pressure_ts = None
        self._force_pressure_labels = False
        self._set_pressure_labels_visible(False)
        if self._server is None:
            self.start()
        self._setup_pressure_gui()
        super().show(q, **kwargs)

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        *,
        pressures: Array | np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Render an I-SUPPORT trajectory with optional chamber pressures."""
        q_array = jnp.asarray(q_ts)
        if q_array.ndim == 2:
            self._current_geometry_q = q_array[0][None, :]
            num_robots, num_frames = 1, int(q_array.shape[0])
        elif q_array.ndim == 3:
            self._current_geometry_q = q_array[:, 0, :]
            num_robots, num_frames = int(q_array.shape[0]), int(q_array.shape[1])
        else:
            raise ValueError(
                f"q_ts must have shape (T, DOF) or (N, T, DOF), got {q_array.shape}."
            )
        self._pressure_ts = self._sequence_pressures(
            pressures,
            num_robots=num_robots,
            num_frames=num_frames,
        )
        self._pressure_frame_idx = 0
        self._current_pressures = (
            None if self._pressure_ts is None else self._pressure_ts[:, 0, :]
        )
        self._force_pressure_labels = False
        self._set_pressure_labels_visible(False)
        super().render_sequence(ts, q_ts, **kwargs)

    def _update_frame(self, frame_idx: int, *args, **kwargs) -> None:
        self._pressure_frame_idx = int(frame_idx)
        self._current_pressures = (
            None
            if self._pressure_ts is None
            else self._pressure_ts[:, self._pressure_frame_idx, :]
        )
        if args:
            q_ts = jnp.asarray(args[0])
            self._current_geometry_q = q_ts[:, frame_idx, :]
        super()._update_frame(frame_idx, *args, **kwargs)

    def _setup_playback_gui(self, on_frame_seek=None) -> None:
        super()._setup_playback_gui(on_frame_seek=on_frame_seek)
        self._setup_pressure_gui()

    def _remove_robot_geometry(self) -> None:
        if self._scene_handles is None:
            return
        self._remove_pressure_labels()
        for robot_handles in self._scene_handles.backbone_points:
            for handle in robot_handles:
                if hasattr(handle, "remove"):
                    handle.remove()
        for handle in self._scene_handles.base_plates:
            if hasattr(handle, "remove"):
                handle.remove()
        self._scene_handles.backbone_points = []
        self._scene_handles.base_plates = []
        self._robot_visual_handles = []

    def _build_robot_geometry(
        self,
        curves: np.ndarray,
        point_colors: np.ndarray,
        *,
        material_frames: np.ndarray,
        base_plate_color: tuple[float, float, float],
    ) -> None:
        """Build I-SUPPORT geometry instead of the generic backbone."""
        if (
            self._server is None
            or self._scene_handles is None
            or self._current_geometry_q is None
        ):
            super()._build_robot_geometry(
                curves,
                point_colors,
                material_frames=material_frames,
                base_plate_color=base_plate_color,
            )
            return

        self._remove_robot_geometry()
        q = self._as_batch(self._current_geometry_q)
        world_poses = self._sample_world_poses(q, curves)
        self._current_world_poses = world_poses
        cfg = self.visual_config

        for robot_idx in range(q.shape[0]):
            poses = world_poses[robot_idx]
            all_handles: list[Any] = []
            chamber_handles: list[Any] = []
            interface_handles: list[Any] = []
            spacer_handles: list[Any] = []

            for spec in self._pneumatic_specs:
                for chamber_idx in range(self.robot.num_chambers_per_segment):
                    handle = self._server.scene.add_mesh_simple(
                        name=(
                            f"/robots/robot_{robot_idx}/isupport/chambers/"
                            f"segment_{spec.pneumatic_idx}/chamber_{chamber_idx}"
                        ),
                        vertices=self._chamber_vertices(poses, spec, chamber_idx),
                        faces=spec.faces,
                        color=self._color_tuple(cfg.chamber_color),
                        material=self._material,
                        flat_shading=self._flat_shading,
                        wireframe=self._wireframe,
                        cast_shadow=self._backbone_cast_shadow,
                    )
                    chamber_handles.append(handle)
                    all_handles.append(handle)

                for spacer_idx, sample_idx in enumerate(spec.spacer_sample_indices):
                    position, wxyz = self._disk_pose(poses[sample_idx])
                    handle = self._add_disk(
                        (
                            f"/robots/robot_{robot_idx}/isupport/spacers/"
                            f"segment_{spec.pneumatic_idx}/spacer_{spacer_idx}"
                        ),
                        radius=spec.radius,
                        thickness=cfg.spacer_thickness,
                        color=cfg.spacer_color,
                        opacity=cfg.spacer_opacity,
                        position=position,
                        wxyz=wxyz,
                    )
                    spacer_handles.append(handle)
                    all_handles.append(handle)

            for spec in self._interface_specs:
                position, wxyz = self._interface_pose(poses, spec)
                handle = self._add_disk(
                    (
                        f"/robots/robot_{robot_idx}/isupport/interfaces/"
                        f"interface_{spec.slot_idx}"
                    ),
                    radius=spec.radius,
                    thickness=spec.thickness,
                    color=cfg.interface_color,
                    opacity=None,
                    position=position,
                    wxyz=wxyz,
                )
                interface_handles.append(handle)
                all_handles.append(handle)

            self._scene_handles.backbone_points.append(all_handles)
            self._robot_visual_handles.append(
                _ISupportRobotHandles(
                    chamber_handles=chamber_handles,
                    interface_handles=interface_handles,
                    spacer_handles=spacer_handles,
                    pressure_label_handles=self._add_pressure_labels(robot_idx, poses),
                )
            )

        self._apply_pressure_visuals(world_poses)
        self._scene_handles.geometry_initialized = True
        self._scene_handles.num_robots = int(q.shape[0])
        self._scene_handles.num_backbone_points = curves.shape[1]

    def _update_robot_geometry(
        self, curves: np.ndarray, material_frames: np.ndarray
    ) -> None:
        """Update all custom geometry without replacing Viser handles."""
        if self._current_geometry_q is None:
            super()._update_robot_geometry(curves, material_frames)
            return
        if (
            self._server is None
            or self._scene_handles is None
            or len(self._robot_visual_handles) != curves.shape[0]
        ):
            return

        q = self._as_batch(self._current_geometry_q)
        world_poses = self._sample_world_poses(q, curves)
        self._current_world_poses = world_poses
        with self._server.atomic():
            for robot_idx, robot_handles in enumerate(self._robot_visual_handles):
                poses = world_poses[robot_idx]
                chamber_handle_idx = 0
                spacer_handle_idx = 0
                for spec in self._pneumatic_specs:
                    for chamber_idx in range(self.robot.num_chambers_per_segment):
                        robot_handles.chamber_handles[
                            chamber_handle_idx
                        ].vertices = self._chamber_vertices(poses, spec, chamber_idx)
                        chamber_handle_idx += 1
                    for sample_idx in spec.spacer_sample_indices:
                        position, wxyz = self._disk_pose(poses[sample_idx])
                        handle = robot_handles.spacer_handles[spacer_handle_idx]
                        handle.position = tuple(position)
                        handle.wxyz = wxyz
                        spacer_handle_idx += 1

                for handle, spec in zip(
                    robot_handles.interface_handles, self._interface_specs
                ):
                    position, wxyz = self._interface_pose(poses, spec)
                    handle.position = tuple(position)
                    handle.wxyz = wxyz
            self._apply_pressure_visuals(world_poses)

    def start_live_mode(
        self,
        callback: Callable[[float], np.ndarray | tuple[np.ndarray, np.ndarray]]
        | None = None,
        dt: float = 0.033,
        pressure_callback: Callable[[float], np.ndarray] | None = None,
    ) -> ISupportLiveModeController:
        """Start live mode with optional synchronized chamber pressures."""
        if self._server is None:
            self.start()
        self._pressure_ts = None
        self._current_pressures = None
        self._force_pressure_labels = False
        self._set_pressure_labels_visible(False)
        self._setup_pressure_gui()
        controller = ISupportLiveModeController(self, callback, dt, pressure_callback)
        controller.start()
        return controller


class ISupportLiveModeController(LiveModeController):
    """Live controller that exposes the current state to the custom renderer."""

    def __init__(
        self,
        renderer: ISupportViserRenderer,
        callback: Callable[[float], np.ndarray | tuple[np.ndarray, np.ndarray]]
        | None = None,
        dt: float = 0.033,
        pressure_callback: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        super().__init__(renderer, callback, dt)
        self._renderer: ISupportViserRenderer = renderer
        self._pressure_callback = pressure_callback

    def push_state_with_pressures(self, q: np.ndarray, pressures: np.ndarray) -> None:
        """Push a configuration and matching chamber pressures."""
        q_batch = self._renderer._as_batch(q)
        self._renderer._current_pressures = self._renderer._static_pressures(
            pressures, num_robots=int(q_batch.shape[0])
        )
        self.push_state(q)

    def _callback_loop(self) -> None:
        while self._running:
            t = time.time() - self._start_time
            if self._callback is not None:
                try:
                    callback_result = self._callback(t)
                    if isinstance(callback_result, tuple):
                        q, pressures = callback_result
                    else:
                        q = callback_result
                        pressures = (
                            None
                            if self._pressure_callback is None
                            else self._pressure_callback(t)
                        )
                    if pressures is not None:
                        q_batch = self._renderer._as_batch(q)
                        self._renderer._current_pressures = (
                            self._renderer._static_pressures(
                                pressures, num_robots=int(q_batch.shape[0])
                            )
                        )
                    self._process_state(q)
                except Exception as exc:
                    print(f"[ISupportViserRenderer] Callback error: {exc}")
            time.sleep(self._dt)

    def _process_state(self, q: np.ndarray) -> None:
        self._renderer._current_geometry_q = self._renderer._as_batch(q)
        super()._process_state(q)
