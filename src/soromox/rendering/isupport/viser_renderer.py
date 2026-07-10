"""Viser renderer specialized for the pneumatic I-SUPPORT robot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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

    interface_radius: float = 0.030
    interface_color: tuple[float, float, float] = (0.08, 0.20, 0.72)
    fallback_interface_thickness: float = 0.005
    chamber_color: tuple[float, float, float] = (0.94, 0.91, 0.72)
    chamber_aspect_ratio: float = 21.72 / 13.03
    bellows_pitch: float = 0.010
    bellows_amplitude: float = 0.15
    chamber_sections: int = 20
    samples_per_fold: int = 4
    spacers_per_segment: int = 6
    spacer_radius: float = 0.030
    spacer_thickness: float = 0.0015
    spacer_color: tuple[float, float, float] = (0.72, 0.82, 0.88)
    spacer_opacity: float = 0.35

    def __post_init__(self) -> None:
        positive_fields = (
            "interface_radius",
            "fallback_interface_thickness",
            "chamber_aspect_ratio",
            "bellows_pitch",
            "spacer_radius",
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
        validate_rgb(self.interface_color, name="interface_color")
        validate_rgb(self.chamber_color, name="chamber_color")
        validate_rgb(self.spacer_color, name="spacer_color")


@dataclass(frozen=True)
class _PneumaticVisualSpec:
    pneumatic_idx: int
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


@dataclass
class _ISupportRobotHandles:
    chamber_handles: list[Any]
    interface_handles: list[Any]
    spacer_handles: list[Any]


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
    def _color_tuple(color: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(np.asarray(np.clip(color, 0.0, 1.0) * 255.0, dtype=np.uint8))

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
        pneumatic_bounds: list[tuple[float, float]] = []

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
                    s_start=s_start,
                    s_end=s_end,
                    ring_slice=ring_slice,
                    spacer_sample_indices=spacer_indices,
                    num_folds=num_folds,
                    faces=_tube_faces(num_rings, self.visual_config.chamber_sections),
                )
            )
            pneumatic_bounds.append((s_start, s_end))

        rigid_indices = iter(np.flatnonzero(rigid_selector).tolist())
        interface_specs: list[_InterfaceVisualSpec] = []
        connector_selector = self.robot.structure.rigid_connector_selector
        if connector_selector is None:
            connector_selector = tuple(
                False for _ in range(self.robot.num_pneumatic_segments + 1)
            )

        for slot_idx, modeled in enumerate(connector_selector):
            if modeled:
                try:
                    pcs_idx = next(rigid_indices)
                except StopIteration as exc:
                    raise ValueError(
                        "Rigid connector topology does not match the PCS layout."
                    ) from exc
                start_idx = len(sample_s)
                sample_s.extend(
                    [
                        float(cumulative_lengths[pcs_idx]),
                        float(cumulative_lengths[pcs_idx + 1]),
                    ]
                )
                interface_specs.append(
                    _InterfaceVisualSpec(
                        slot_idx=slot_idx,
                        modeled=True,
                        sample_indices=(start_idx, start_idx + 1),
                        thickness=float(lengths[pcs_idx]),
                    )
                )
            else:
                if slot_idx == 0:
                    boundary = pneumatic_bounds[0][0]
                elif slot_idx == self.robot.num_pneumatic_segments:
                    boundary = pneumatic_bounds[-1][1]
                else:
                    boundary = 0.5 * (
                        pneumatic_bounds[slot_idx - 1][1]
                        + pneumatic_bounds[slot_idx][0]
                    )
                sample_idx = len(sample_s)
                sample_s.append(float(boundary))
                interface_specs.append(
                    _InterfaceVisualSpec(
                        slot_idx=slot_idx,
                        modeled=False,
                        sample_indices=(sample_idx,),
                        thickness=self.visual_config.fallback_interface_thickness,
                    )
                )

        try:
            next(rigid_indices)
        except StopIteration:
            pass
        else:
            raise ValueError("PCS layout contains an unmatched rigid connector.")

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
        material_y = rotations[:, :, 1]
        material_z = rotations[:, :, 2]

        pneumatic_idx = spec.pneumatic_idx
        angle = float(self.robot.params.chamber_angle_offset[pneumatic_idx]) + (
            2.0 * np.pi * chamber_idx / self.robot.num_chambers_per_segment
        )
        radial = np.sin(angle) * material_y + np.cos(angle) * material_z
        tangential = np.cos(angle) * material_y - np.sin(angle) * material_z
        chamber_centers = (
            centerline
            + float(self.robot.params.chamber_distance[pneumatic_idx]) * radial
        )

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

    def render_frame(self, q: Array, **kwargs) -> np.ndarray:
        """Render and capture one I-SUPPORT configuration."""
        self._current_geometry_q = self._as_batch(q)
        return super().render_frame(q, **kwargs)

    def show(self, q: Array, **kwargs) -> None:
        """Display one I-SUPPORT configuration interactively."""
        self._current_geometry_q = self._as_batch(q)
        super().show(q, **kwargs)

    def render_sequence(self, ts: Array, q_ts: Array, **kwargs) -> None:
        """Render an I-SUPPORT trajectory."""
        q_array = jnp.asarray(q_ts)
        if q_array.ndim == 2:
            self._current_geometry_q = q_array[0][None, :]
        elif q_array.ndim == 3:
            self._current_geometry_q = q_array[:, 0, :]
        else:
            raise ValueError(
                f"q_ts must have shape (T, DOF) or (N, T, DOF), got {q_array.shape}."
            )
        super().render_sequence(ts, q_ts, **kwargs)

    def _update_frame(self, frame_idx: int, *args, **kwargs) -> None:
        if args:
            q_ts = jnp.asarray(args[0])
            self._current_geometry_q = q_ts[:, frame_idx, :]
        super()._update_frame(frame_idx, *args, **kwargs)

    def _remove_robot_geometry(self) -> None:
        if self._scene_handles is None:
            return
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
                        radius=cfg.spacer_radius,
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
                    radius=cfg.interface_radius,
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
                )
            )

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

    def start_live_mode(
        self,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
    ) -> ISupportLiveModeController:
        """Start live mode while retaining the active I-SUPPORT configuration."""
        if self._server is None:
            self.start()
        controller = ISupportLiveModeController(self, callback, dt)
        controller.start()
        return controller


class ISupportLiveModeController(LiveModeController):
    """Live controller that exposes the current state to the custom renderer."""

    def __init__(
        self,
        renderer: ISupportViserRenderer,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
    ) -> None:
        super().__init__(renderer, callback, dt)
        self._renderer: ISupportViserRenderer = renderer

    def _process_state(self, q: np.ndarray) -> None:
        self._renderer._current_geometry_q = self._renderer._as_batch(q)
        super()._process_state(q)
