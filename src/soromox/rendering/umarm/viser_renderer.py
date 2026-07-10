"""Viser renderer specialized for the McKibben-actuated UMArm."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.viser_renderer import (
    LiveModeController,
    ViserRenderer,
    _direction_to_quaternion,
)


def _rgb255(color: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(color[:3], 0.0, 1.0) * 255.0, dtype=np.uint8)


@dataclass
class _UMArmRigidSpecs:
    cylinder_handles: list
    frame_sources: np.ndarray
    link_indices: np.ndarray
    local_endpoints: np.ndarray
    sphere_handle: Any
    sphere_link_idx: int


class UMArmViserRenderer(ViserRenderer):
    """Viser renderer that adds McKibben actuator segments to the UMArm backbone."""

    def __init__(
        self,
        *args,
        actuator_color_mode: Literal["uniform", "pressure", "force"] = "uniform",
        actuator_line_width: float | None = None,
        actuator_radius: float = 0.003,
        rod_core_radius_scale: float = 0.45,
        linkage_radius: float = 0.007,
        end_effector_radius: float = 0.010,
        end_effector_sphere_radius: float = 0.019,
        ujoint_disk_thickness: float = 0.010,
        actuator_disk_thickness: float = 0.010,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.robot, "mckibben_actuator_segments"):
            raise TypeError(
                "UMArmViserRenderer requires a robot with mckibben_actuator_segments(q)."
        )
        self._actuator_color_mode = actuator_color_mode
        self._actuator_radius = actuator_radius
        self._rod_core_radius_scale = rod_core_radius_scale
        self._linkage_radius = linkage_radius
        self._end_effector_radius = end_effector_radius
        self._end_effector_sphere_radius = end_effector_sphere_radius
        self._ujoint_disk_thickness = ujoint_disk_thickness
        self._actuator_disk_thickness = actuator_disk_thickness
        self._actuator_line_width = (
            actuator_line_width
            if actuator_line_width is not None
            else self._actuator_line_width
        )
        self._actuator_pressures: Array | None = None
        self._actuator_frame_idx = 0
        self._current_geometry_q: Array | None = None
        self._umarm_rigid_specs: list[_UMArmRigidSpecs] = []

    def show(self, q: Array, *, pressures: Array | None = None, **kwargs) -> None:
        """Display a UMArm frame, optionally coloring actuators by pressure or force."""
        q_arr = jnp.asarray(q)
        self._current_geometry_q = q_arr[None, :] if q_arr.ndim == 1 else q_arr
        self._actuator_pressures = None if pressures is None else jnp.asarray(pressures)
        self._actuator_frame_idx = 0
        kwargs.setdefault("camera_config", self._default_umarm_camera_config())
        kwargs.setdefault("render_actuators", True)
        if pressures is not None:
            kwargs.setdefault("actuator_inputs", pressures)
        super().show(q, **kwargs)

    def render_sequence(
        self,
        ts: Array,
        q_ts: Array,
        *,
        pressures: Array | None = None,
        actuator_color_mode: Literal["uniform", "pressure", "force"] | None = None,
        **kwargs,
    ) -> None:
        """Render a UMArm trajectory, including McKibben actuator segments."""
        q_arr = jnp.asarray(q_ts)
        self._current_geometry_q = q_arr[None, 0, :] if q_arr.ndim == 2 else q_arr[:, 0, :]
        self._actuator_pressures = None if pressures is None else jnp.asarray(pressures)
        self._actuator_frame_idx = 0
        old_mode = self._actuator_color_mode
        if actuator_color_mode is not None:
            self._actuator_color_mode = actuator_color_mode
        try:
            kwargs.setdefault("camera_config", self._default_umarm_camera_config())
            kwargs.setdefault("render_actuators", True)
            if pressures is not None:
                kwargs.setdefault("actuator_inputs", pressures)
            super().render_sequence(ts, q_ts, **kwargs)
        finally:
            self._actuator_color_mode = old_mode

    def _update_frame(self, frame_idx: int, *args, **kwargs) -> None:
        self._actuator_frame_idx = frame_idx
        if args:
            q_ts = jnp.asarray(args[0])
            self._current_geometry_q = q_ts[:, frame_idx, :]
        super()._update_frame(frame_idx, *args, **kwargs)

    @staticmethod
    def _color_tuple(color: tuple[float, float, float] | np.ndarray) -> tuple[int, int, int]:
        color_array = np.asarray(color, dtype=np.float64)
        color_array = np.clip(color_array[:3], 0.0, 1.0)
        return tuple(np.asarray(255.0 * color_array, dtype=np.uint8).tolist())

    def _default_umarm_camera_config(self) -> CameraConfig:
        return CameraConfig(
            fov=self._camera_fov,
            distance_factor=1.05,
            position_offset=(-0.5, -0.9, 0.45),
        )

    def _infer_umarm_radii(self) -> tuple[float, float]:
        fixed_radii = np.linalg.norm(
            np.asarray(self.robot.mckibben_fixed_points)[..., :2], axis=-1
        )
        moving_radii = np.linalg.norm(
            np.asarray(self.robot.mckibben_moving_points)[..., :2], axis=-1
        )
        all_radii = np.concatenate([fixed_radii.ravel(), moving_radii.ravel()])
        ujoint_radius = float(np.max(all_radii))
        rod_candidates = all_radii[all_radii < 0.95 * ujoint_radius]
        rod_radius = (
            float(np.max(rod_candidates))
            if rod_candidates.size
            else float(self._robot_radius)
        )
        return ujoint_radius, rod_radius

    @staticmethod
    def _transform_points(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        return points @ frame[:3, :3].T + frame[:3, 3]

    def _add_cylinder_between(
        self,
        name: str,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        radius: float,
        color: tuple[float, float, float] | np.ndarray,
        handles: list,
    ):
        if self._server is None:
            return None
        direction = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        center = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
        mesh = self._make_cylinder_trimesh(length=length, radius=radius, color=color)
        handle = self._server.scene.add_mesh_simple(
            name=name,
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=self._color_tuple(color),
            position=tuple(center),
            wxyz=_direction_to_quaternion(direction),
            material=self._material,
            flat_shading=self._flat_shading,
            wireframe=self._wireframe,
            cast_shadow=self._backbone_cast_shadow,
        )
        handles.append(handle)
        return handle

    @staticmethod
    def _local_z_endpoints(z_center: float, length: float) -> np.ndarray:
        return np.asarray(
            [
                [0.0, 0.0, z_center - 0.5 * length],
                [0.0, 0.0, z_center + 0.5 * length],
            ],
            dtype=np.float64,
        )

    def _add_local_z_cylinder(
        self,
        name: str,
        frame: np.ndarray,
        z_center: float,
        *,
        length: float,
        radius: float,
        color: tuple[float, float, float],
        handles: list,
    ):
        endpoints = self._transform_points(frame, self._local_z_endpoints(z_center, length))
        return self._add_cylinder_between(
            name,
            endpoints[0],
            endpoints[1],
            radius=radius,
            color=color,
            handles=handles,
        )

    def _build_robot_geometry(
        self,
        curves: np.ndarray,
        point_colors: np.ndarray,
        *,
        material_frames: np.ndarray,
        base_plate_color: tuple[float, float, float],
    ) -> None:
        """Build a UMArm-specific rigid visual model instead of a soft tube."""
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

        for robot_handles in self._scene_handles.backbone_points:
            for handle in robot_handles:
                if hasattr(handle, "remove"):
                    handle.remove()
        for handle in self._scene_handles.base_plates:
            if hasattr(handle, "remove"):
                handle.remove()
        self._scene_handles.backbone_points = []
        self._scene_handles.base_plates = []
        self._umarm_rigid_specs = []

        q = jnp.asarray(self._current_geometry_q)
        joint_frames, link_frames, _, tip_frames = jax.vmap(
            self.robot._kinematic_frames
        )(q)
        joint_frames_np = np.asarray(joint_frames)
        link_frames_np = np.asarray(link_frames)
        tip_frames_np = np.asarray(tip_frames)
        p_tip_np = np.asarray(self.robot.p_tip)

        num_robots = q.shape[0]
        base_origins = joint_frames_np[:, 0, :3, 3]
        base_offsets = curves[:, 0, :] - base_origins
        ujoint_radius, rod_radius = self._infer_umarm_radii()
        rod_core_radius = rod_radius * self._rod_core_radius_scale

        pair_starts = np.asarray(self.robot.mckibben_joint_pair_indices[:, 0])
        upper_group_indices = np.arange(0, self.robot.num_mckibben_groups, 2)
        rod_link_indices = pair_starts[upper_group_indices] + 1
        linkage_link_indices = pair_starts[1:-1:2] + 1
        ujoint_link_indices = pair_starts
        end_effector_link_idx = self.robot.num_links - 1

        rod_color = (0.20, 0.80, 0.20)
        disk_color = (0.20, 0.60, 0.95)
        ujoint_color = (0.10, 0.40, 0.90)
        linkage_color = (0.20, 0.50, 0.90)
        ee_rod_color = (0.95, 0.55, 0.20)
        ee_sphere_color = (0.95, 0.30, 0.18)

        for robot_idx in range(num_robots):
            offset = base_offsets[robot_idx]
            robot_handles = []
            cylinder_handles = []
            frame_sources = []
            link_indices = []
            local_endpoints = []

            base_handle = self._add_base_plate(
                robot_idx, curves[robot_idx, 0], base_plate_color
            )
            self._scene_handles.base_plates.append(base_handle)

            for joint_idx in ujoint_link_indices:
                frame = joint_frames_np[robot_idx, joint_idx].copy()
                frame[:3, 3] += offset
                handle = self._add_local_z_cylinder(
                    f"/robots/robot_{robot_idx}/umarm/ujoint_disk_{joint_idx}",
                    frame,
                    0.0,
                    length=self._ujoint_disk_thickness,
                    radius=ujoint_radius,
                    color=ujoint_color,
                    handles=robot_handles,
                )
                if handle is not None:
                    cylinder_handles.append(handle)
                    frame_sources.append(0)
                    link_indices.append(int(joint_idx))
                    local_endpoints.append(
                        self._local_z_endpoints(0.0, self._ujoint_disk_thickness)
                    )

            for group_idx, rod_link_idx in zip(upper_group_indices, rod_link_indices):
                link_frame = link_frames_np[robot_idx, rod_link_idx].copy()
                link_frame[:3, 3] += offset
                p_tip = p_tip_np[rod_link_idx]
                endpoints = self._transform_points(
                    link_frame, np.asarray([[0.0, 0.0, 0.0], p_tip])
                )
                handle = self._add_cylinder_between(
                    f"/robots/robot_{robot_idx}/umarm/rod_{rod_link_idx}",
                    endpoints[0],
                    endpoints[1],
                    radius=rod_core_radius,
                    color=rod_color,
                    handles=robot_handles,
                )
                if handle is not None:
                    cylinder_handles.append(handle)
                    frame_sources.append(1)
                    link_indices.append(int(rod_link_idx))
                    local_endpoints.append(
                        np.asarray([[0.0, 0.0, 0.0], p_tip], dtype=np.float64)
                    )

                upper_disk_z = float(self.robot.mckibben_moving_points[group_idx, 0, 2])
                lower_disk_z = float(
                    self.robot.mckibben_fixed_points[group_idx + 1, 0, 2]
                )
                for label, z_center in (
                    ("upper_actuator_disk", upper_disk_z),
                    ("lower_actuator_disk", lower_disk_z),
                ):
                    handle = self._add_local_z_cylinder(
                        f"/robots/robot_{robot_idx}/umarm/{label}_{rod_link_idx}",
                        link_frame,
                        z_center,
                        length=self._actuator_disk_thickness,
                        radius=rod_radius,
                        color=disk_color,
                        handles=robot_handles,
                    )
                    if handle is not None:
                        cylinder_handles.append(handle)
                        frame_sources.append(1)
                        link_indices.append(int(rod_link_idx))
                        local_endpoints.append(
                            self._local_z_endpoints(
                                float(z_center), self._actuator_disk_thickness
                            )
                        )

            for linkage_link_idx in linkage_link_indices:
                link_frame = link_frames_np[robot_idx, linkage_link_idx].copy()
                link_frame[:3, 3] += offset
                endpoints = self._transform_points(
                    link_frame,
                    np.asarray([[0.0, 0.0, 0.0], p_tip_np[linkage_link_idx]]),
                )
                handle = self._add_cylinder_between(
                    f"/robots/robot_{robot_idx}/umarm/linkage_{linkage_link_idx}",
                    endpoints[0],
                    endpoints[1],
                    radius=self._linkage_radius,
                    color=linkage_color,
                    handles=robot_handles,
                )
                if handle is not None:
                    cylinder_handles.append(handle)
                    frame_sources.append(1)
                    link_indices.append(int(linkage_link_idx))
                    local_endpoints.append(
                        np.asarray(
                            [[0.0, 0.0, 0.0], p_tip_np[linkage_link_idx]],
                            dtype=np.float64,
                        )
                    )

            ee_frame = link_frames_np[robot_idx, end_effector_link_idx].copy()
            ee_frame[:3, 3] += offset
            ee_endpoints = self._transform_points(
                ee_frame,
                np.asarray([[0.0, 0.0, 0.0], p_tip_np[end_effector_link_idx]]),
            )
            handle = self._add_cylinder_between(
                f"/robots/robot_{robot_idx}/umarm/end_effector_rod",
                ee_endpoints[0],
                ee_endpoints[1],
                radius=self._end_effector_radius,
                color=ee_rod_color,
                handles=robot_handles,
            )
            if handle is not None:
                cylinder_handles.append(handle)
                frame_sources.append(1)
                link_indices.append(int(end_effector_link_idx))
                local_endpoints.append(
                    np.asarray(
                        [[0.0, 0.0, 0.0], p_tip_np[end_effector_link_idx]],
                        dtype=np.float64,
                    )
                )
            sphere_handle = self._server.scene.add_icosphere(
                name=f"/robots/robot_{robot_idx}/umarm/end_effector_sphere",
                radius=self._end_effector_sphere_radius,
                color=self._color_tuple(ee_sphere_color),
                position=tuple(tip_frames_np[robot_idx, end_effector_link_idx, :3, 3] + offset),
                subdivisions=self._sphere_resolution,
                material=self._material,
                flat_shading=self._flat_shading,
                wireframe=self._wireframe,
                cast_shadow=self._sphere_cast_shadow,
            )
            robot_handles.append(sphere_handle)
            self._scene_handles.backbone_points.append(robot_handles)
            self._umarm_rigid_specs.append(
                _UMArmRigidSpecs(
                    cylinder_handles=cylinder_handles,
                    frame_sources=np.asarray(frame_sources, dtype=np.int8),
                    link_indices=np.asarray(link_indices, dtype=np.int64),
                    local_endpoints=np.asarray(local_endpoints, dtype=np.float64),
                    sphere_handle=sphere_handle,
                    sphere_link_idx=int(end_effector_link_idx),
                )
            )

        self._scene_handles.geometry_initialized = True
        self._scene_handles.num_robots = num_robots
        self._scene_handles.num_backbone_points = curves.shape[1]

    def _update_robot_geometry(
        self, curves: np.ndarray, material_frames: np.ndarray
    ) -> None:
        """Update the UMArm rigid visual model without rebuilding scene objects."""
        del material_frames
        if (
            self._scene_handles is None
            or self._current_geometry_q is None
            or len(self._umarm_rigid_specs) != len(self._scene_handles.backbone_points)
        ):
            return

        q = jnp.asarray(self._current_geometry_q)
        joint_frames, link_frames, _, tip_frames = jax.vmap(
            self.robot._kinematic_frames
        )(q)
        joint_frames_np = np.asarray(joint_frames)
        link_frames_np = np.asarray(link_frames)
        tip_frames_np = np.asarray(tip_frames)

        base_origins = joint_frames_np[:, 0, :3, 3]
        base_offsets = curves[:, 0, :] - base_origins

        for robot_idx, curve in enumerate(curves):
            if robot_idx < len(self._scene_handles.base_plates):
                base_pos, base_wxyz = self._base_plate_pose(curve[0])
                base_handle = self._scene_handles.base_plates[robot_idx]
                base_handle.position = tuple(base_pos)
                base_handle.wxyz = base_wxyz

            specs = self._umarm_rigid_specs[robot_idx]
            cylinder_handles = specs.cylinder_handles
            frame_sources = specs.frame_sources
            link_indices = specs.link_indices
            local_endpoints = specs.local_endpoints
            if len(cylinder_handles) != len(local_endpoints):
                continue

            offset = base_offsets[robot_idx]
            if len(cylinder_handles) > 0:
                joint_selected = joint_frames_np[robot_idx, link_indices]
                link_selected = link_frames_np[robot_idx, link_indices]
                frames = np.where(
                    frame_sources[:, None, None] == 0,
                    joint_selected,
                    link_selected,
                )
                endpoints = (
                    np.einsum("cij,cpj->cpi", frames[:, :3, :3], local_endpoints)
                    + (frames[:, :3, 3] + offset)[:, None, :]
                )
                centers = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
                directions = endpoints[:, 1] - endpoints[:, 0]
                norms = np.linalg.norm(directions, axis=1)
                for handle, center, direction, norm in zip(
                    cylinder_handles, centers, directions, norms
                ):
                    if norm < 1e-9:
                        continue
                    handle.position = tuple(center)
                    handle.wxyz = _direction_to_quaternion(direction)

            sphere_handle = specs.sphere_handle
            sphere_link_idx = specs.sphere_link_idx
            if sphere_handle is not None:
                sphere_handle.position = tuple(
                    tip_frames_np[robot_idx, sphere_link_idx, :3, 3] + offset
                )

    def start_live_mode(
        self,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
        pressure_callback: Callable[[float], np.ndarray] | None = None,
    ) -> UMArmLiveModeController:
        """Start live visualization mode with UMArm-specific rigid geometry.

        Args:
            callback: Optional state provider function ``time -> q``. The
                callback may also return ``(q, pressures)`` to update actuator
                colors together with the configuration.
            dt: Callback-mode update period in seconds.
            pressure_callback: Optional pressure provider ``time -> pressures``.
                Ignored when ``callback`` returns ``(q, pressures)``.

        Returns:
            Controller for streaming UMArm states into the Viser scene.
        """
        if self._server is None:
            self.start()

        controller = UMArmLiveModeController(self, callback, dt, pressure_callback)
        controller.start()
        return controller

    def _pressures_for_frame(
        self, q: Array, actuator_inputs: Array | None = None
    ) -> Array | None:
        if actuator_inputs is not None:
            pressures = jnp.asarray(actuator_inputs)
            if pressures.ndim == 1:
                return jnp.broadcast_to(pressures, (q.shape[0], pressures.shape[0]))
            if pressures.ndim == 2:
                return pressures
        if self._actuator_pressures is None:
            return None
        pressures = self._actuator_pressures
        if pressures.ndim == 1:
            return jnp.broadcast_to(pressures, (q.shape[0], pressures.shape[0]))
        if pressures.ndim == 2:
            if pressures.shape[0] == q.shape[0]:
                return pressures
            frame_idx = min(self._actuator_frame_idx, pressures.shape[0] - 1)
            return jnp.broadcast_to(pressures[frame_idx], (q.shape[0], pressures.shape[1]))
        if pressures.ndim == 3:
            frame_idx = min(self._actuator_frame_idx, pressures.shape[1] - 1)
            return pressures[:, frame_idx, :]
        raise ValueError(
            "pressures must have shape (num_actuators,), (T, num_actuators), "
            "or (N, T, num_actuators)."
        )

    @staticmethod
    def _heat_colors(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        max_abs = max(float(np.max(np.abs(values))), 1e-12)
        scaled = np.clip(np.abs(values) / max_abs, 0.0, 1.0)
        low = np.array([0.12, 0.38, 0.72], dtype=np.float64)
        high = np.array([0.90, 0.22, 0.12], dtype=np.float64)
        colors = low + (high - low) * scaled[..., None]
        return np.asarray(colors * 255.0, dtype=np.uint8)

    def _actuator_colors(
        self,
        q: Array,
        fallback_color: tuple[float, float, float] | None,
        actuator_inputs: Array | None = None,
    ) -> np.ndarray:
        fallback = (
            np.asarray(fallback_color, dtype=np.float64)
            if fallback_color is not None
            else np.asarray(self.color_config.actuators.default_color, dtype=np.float64)
        )
        uniform = np.tile(_rgb255(fallback), (q.shape[0], self.robot.num_actuators, 1))
        pressures = self._pressures_for_frame(q, actuator_inputs=actuator_inputs)
        if pressures is None or self._actuator_color_mode == "uniform":
            return uniform
        if self._actuator_color_mode == "pressure":
            return self._heat_colors(np.asarray(pressures))
        if self._actuator_color_mode == "force":
            forces = jax.vmap(self.robot.actuator_forces)(q, pressures)
            return self._heat_colors(np.asarray(forces))
        raise ValueError(
            "actuator_color_mode must be one of 'uniform', 'pressure', or 'force'."
        )

    def _build_actuator_geometry(
        self,
        q: Array,
        base_offsets: np.ndarray,
        num_robots: int,
        *,
        color_config=None,
        actuator_inputs: Array | None = None,
    ) -> None:
        if self._server is None or self._scene_handles is None:
            return

        q = jnp.asarray(q)
        actuator_segments = self.compute_actuator_visual_layers_batched(
            q,
            jnp.asarray(base_offsets),
            actuator_inputs=actuator_inputs,
        )[0].points
        actuator_segments_np = np.asarray(actuator_segments)
        cfg = color_config or self.color_config
        colors = self._actuator_colors(
            q,
            cfg.actuators.default_color,
            actuator_inputs=actuator_inputs,
        )

        if len(self._scene_handles.actuator_lines) > num_robots:
            for handle in self._scene_handles.actuator_lines[num_robots:]:
                if hasattr(handle, "remove"):
                    handle.remove()
            self._scene_handles.actuator_lines = self._scene_handles.actuator_lines[
                :num_robots
            ]

        for robot_idx in range(num_robots):
            points = np.asarray(actuator_segments_np[robot_idx], dtype=np.float32)
            segment_colors = np.repeat(
                np.asarray(colors[robot_idx], dtype=np.uint8)[:, None, :],
                2,
                axis=1,
            )
            if robot_idx < len(self._scene_handles.actuator_lines):
                handle = self._scene_handles.actuator_lines[robot_idx]
                handle.points = points
                handle.colors = segment_colors
                handle.line_width = self._actuator_line_width
            else:
                handle = self._server.scene.add_line_segments(
                    name=f"/robots/robot_{robot_idx}/mckibben_actuators/segments",
                    points=points,
                    colors=segment_colors,
                    line_width=self._actuator_line_width,
                )
                self._scene_handles.actuator_lines.append(handle)


class UMArmLiveModeController(LiveModeController):
    """Live-mode controller that updates UMArm rigid geometry from ``q``."""

    def __init__(
        self,
        renderer: UMArmViserRenderer,
        callback: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.033,
        pressure_callback: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        super().__init__(renderer, callback, dt)
        self._renderer: UMArmViserRenderer = renderer
        self._pressure_callback = pressure_callback

    def push_state_with_pressures(self, q: np.ndarray, pressures: np.ndarray) -> None:
        """Push a live UMArm state and actuator pressures."""
        self._renderer._actuator_pressures = jnp.asarray(pressures)
        self.push_state(q)

    def _callback_loop(self) -> None:
        while self._running:
            t = time.time() - self._start_time

            if self._callback is not None:
                try:
                    callback_result = self._callback(t)
                    if isinstance(callback_result, tuple):
                        q, pressures = callback_result
                        self._renderer._actuator_pressures = jnp.asarray(pressures)
                    else:
                        q = callback_result
                        if self._pressure_callback is not None:
                            self._renderer._actuator_pressures = jnp.asarray(
                                self._pressure_callback(t)
                            )
                    self._process_state(q)
                except Exception as exc:
                    print(f"[UMArmViserRenderer] Callback error: {exc}")

            time.sleep(self._dt)

    def _process_state(self, q: np.ndarray) -> None:
        q = jnp.asarray(q)
        if q.ndim == 1:
            q = q[None, :]

        num_robots = q.shape[0]
        self._renderer._current_geometry_q = q

        base_offsets = self._renderer._normalize_base_offsets(
            self._renderer._compute_grid_offsets(
                num_robots, self._renderer._grid_spacing
            ),
            num_robots=int(num_robots),
            target_dim=3,
        )

        curves, material_frames = (
            self._renderer.compute_backbone_curves_and_frames_batched(q, base_offsets)
        )
        curves = np.asarray(curves)
        material_frames = np.asarray(material_frames)

        if (
            self._resolved_colors is None
            or self._resolved_colors_num_robots != num_robots
        ):
            self._resolved_colors = self._renderer.resolve_backbone_colors(num_robots)
            self._resolved_colors_num_robots = num_robots

        scene_handles = self._renderer._scene_handles
        can_update = (
            scene_handles is not None
            and scene_handles.geometry_initialized
            and scene_handles.num_robots == num_robots
            and scene_handles.num_backbone_points == curves.shape[1]
        )
        if can_update:
            self._renderer._update_robot_geometry(curves, material_frames)
        else:
            self._renderer._build_robot_geometry(
                curves,
                self._resolved_colors.per_robot_point_rgba,
                material_frames=material_frames,
                base_plate_color=self._renderer.color_config.base_plate_color,
            )
        if self._renderer._has_actuator_visuals:
            self._renderer._build_actuator_geometry(
                q,
                np.asarray(base_offsets),
                num_robots,
                color_config=self._renderer.color_config,
                actuator_inputs=self._renderer._pressures_for_frame(q),
            )
