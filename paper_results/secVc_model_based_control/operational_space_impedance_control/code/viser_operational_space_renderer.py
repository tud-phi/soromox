"""Viser rendering helpers for operational-space control examples.

This module intentionally lives under ``examples/``. It is shared by example
scripts in this folder, but it is not part of the public ``soromox`` API.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from trajectory_primitives import (
    SurfaceGeometry,
    TaskSpaceTrajectoryConfig,
    make_surface_geometry_metadata,
    surface_normal_at,
)

from soromox.rendering import ViserRenderer
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import BackboneColorConfig, RendererColorConfig
from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_vector_to_rotation_matrix,
)


@dataclass(frozen=True)
class SurfaceMesh:
    """Triangular mesh for rendering an operational-space reference surface."""

    vertices: np.ndarray
    faces: np.ndarray


def make_transparent_robot_color_config(
    num_points: int,
    opacity: float,
) -> RendererColorConfig:
    """Return a transparent backbone color configuration."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    num_points = max(2, int(num_points))
    start = np.array([0.09, 0.22, 0.42, opacity], dtype=np.float64)
    end = np.array([0.00, 0.55, 0.78, opacity], dtype=np.float64)
    blend = np.linspace(0.0, 1.0, num_points)[:, None]
    point_colors = (1.0 - blend) * start + blend * end
    return RendererColorConfig(
        backbone=BackboneColorConfig(point_colors=point_colors),
        base_plate_color=(0.12, 0.12, 0.12),
    )


def make_polyline_segments(points: np.ndarray) -> np.ndarray:
    """Convert polyline points into Viser line-segment geometry."""
    points = _ensure_3d_positions(points).astype(np.float32)
    if len(points) < 2:
        if len(points) == 0:
            points = np.zeros((2, 3), dtype=np.float32)
        else:
            points = np.repeat(points, 2, axis=0)
    return np.stack([points[:-1], points[1:]], axis=1)


def make_fading_trail_colors(
    num_segments: int,
    *,
    current_color: tuple[float, float, float] = (0.90, 0.25, 0.10),
    faded_color: tuple[float, float, float] = (0.82, 0.84, 0.86),
) -> np.ndarray:
    """Return per-segment RGB colors from faded history to current state."""
    num_segments = max(1, int(num_segments))
    current = np.asarray(current_color, dtype=np.float64)
    faded = np.asarray(faded_color, dtype=np.float64)
    weights = np.linspace(0.0, 1.0, num_segments)[:, None] ** 1.4
    colors = (1.0 - weights) * faded + weights * current
    colors_u8 = _to_uint8(colors)
    return np.repeat(colors_u8[:, None, :], 2, axis=1)


def make_operational_space_camera_config(
    reference_positions: np.ndarray,
    actual_positions: np.ndarray | None = None,
) -> CameraConfig:
    """Return a close y-z aligned camera for operational-space rollouts."""
    points = [_ensure_3d_positions(reference_positions)]
    if actual_positions is not None:
        points.append(_ensure_3d_positions(actual_positions))
    points.append(np.zeros((1, 3), dtype=np.float64))
    all_points = np.concatenate(points, axis=0)

    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    look_at = 0.5 * (lower + upper)
    radius = max(
        float(np.linalg.norm(upper - lower)),
        float(np.max(np.linalg.norm(all_points - look_at, axis=1))),
        0.12,
    )
    look_at = look_at + np.array([0.0, 0.025 * radius, 0.045 * radius])
    view_direction = _normalize(np.array([0.0, -0.58, 1.0], dtype=np.float64))
    camera_position = look_at + 2.20 * radius * view_direction
    return CameraConfig(
        fov=36.0,
        position=tuple(camera_position),
        look_at=tuple(look_at),
        up=(0.0, 0.0, 1.0),
    )


def make_target_cross_segments(position: np.ndarray, radius: float) -> np.ndarray:
    """Return a 3D cross marker centered at ``position``."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}.")
    radius = float(radius)
    axes = np.eye(3, dtype=np.float64)
    return np.asarray(
        [[position - radius * axis, position + radius * axis] for axis in axes],
        dtype=np.float32,
    )


def make_target_ring_segments(
    position: np.ndarray,
    radius: float,
    *,
    num_points: int = 48,
) -> np.ndarray:
    """Return three orthogonal target rings centered at ``position``."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}.")
    radius = float(radius)
    num_points = max(8, int(num_points))
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    rings = [
        position + radius * np.column_stack([np.cos(theta), np.sin(theta), 0 * theta]),
        position + radius * np.column_stack([np.cos(theta), 0 * theta, np.sin(theta)]),
        position + radius * np.column_stack([0 * theta, np.cos(theta), np.sin(theta)]),
    ]
    return np.concatenate(
        [np.stack([ring, np.roll(ring, -1, axis=0)], axis=1) for ring in rings],
        axis=0,
    ).astype(np.float32)


def make_target_marker_segments(
    position: np.ndarray,
    normal: np.ndarray,
    radius: float,
    *,
    num_points: int = 40,
) -> np.ndarray:
    """Return a compact line-only target marker in the tangent plane."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}.")
    normal = _normalize(normal)
    radius = float(radius)
    num_points = max(8, int(num_points))
    y_axis, z_axis = _surface_tangent_axes(normal, np.array([0.0, 1.0, 0.0]))

    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    ring = position + radius * (
        np.cos(theta)[:, None] * y_axis + np.sin(theta)[:, None] * z_axis
    )
    ring_segments = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)
    cross_radius = 0.62 * radius
    cross_segments = np.asarray(
        [
            [position - cross_radius * y_axis, position + cross_radius * y_axis],
            [position - cross_radius * z_axis, position + cross_radius * z_axis],
        ],
        dtype=np.float64,
    )
    return np.concatenate([ring_segments, cross_segments], axis=0).astype(np.float32)


def make_surface_mesh(
    surface_geometry: SurfaceGeometry,
    reference_positions: np.ndarray,
    *,
    resolution: int = 48,
) -> SurfaceMesh:
    """Create a translucent render mesh for a plane, cylinder, or sphere."""
    reference_positions = _ensure_3d_positions(reference_positions)
    resolution = max(8, int(resolution))

    if surface_geometry.surface == "plane":
        return _make_plane_mesh(surface_geometry, reference_positions, resolution)
    if surface_geometry.surface == "cylinder":
        return _make_cylinder_mesh(surface_geometry, reference_positions, resolution)
    if surface_geometry.surface == "sphere":
        return _make_sphere_mesh(surface_geometry, resolution)
    raise ValueError(f"Unsupported surface: {surface_geometry.surface}")


def render_operational_space_tracking(
    *,
    robot: Any,
    osd: Any,
    q0: Array,
    trajectory_config: TaskSpaceTrajectoryConfig,
    t_traj: Array,
    q_traj: Array,
    x_traj_full: Array,
    x_des_traj_full: Array,
    viser_renderer_cls: type[ViserRenderer] | None,
    port: int = 8080,
    open_browser: bool = True,
    robot_opacity: float = 0.35,
    video_output: str | None = None,
    record_every_n: int = 1,
    record_client_timeout: float = 10.0,
    record_frame_timeout: float = 10.0,
    camera_config: CameraConfig | None = None,
    blocking: bool = True,
) -> None:
    """Render operational-space tracking with Viser overlays."""
    if viser_renderer_cls is None:
        raise ImportError(
            "Viser is required to render this example. Install the Viser extras "
            "or omit --render."
        )

    num_points = 80
    color_config = make_transparent_robot_color_config(num_points, robot_opacity)
    renderer = viser_renderer_cls(
        robot,
        num_points=num_points,
        port=port,
        open_browser=open_browser,
        color_config=color_config,
        backbone_style="swept",
        cylinder_sections=64,
        sphere_resolution=3,
        cast_shadows=False,
        backbone_cast_shadow=False,
        sphere_cast_shadow=False,
    )

    surface_geometry = None
    if (
        trajectory_config.tracking == "pose"
        and trajectory_config.orientation_primitive == "surface-following"
    ):
        surface_geometry = make_surface_geometry_metadata(
            osd=osd,
            q0=q0,
            config=trajectory_config,
        )

    x_traj_full_np = np.asarray(x_traj_full, dtype=np.float64)
    x_des_traj_full_np = np.asarray(x_des_traj_full, dtype=np.float64)
    actual_positions = _extract_positions(osd, x_traj_full_np)
    reference_positions = _extract_positions(osd, x_des_traj_full_np)
    if camera_config is None:
        camera_config = make_operational_space_camera_config(
            reference_positions,
            actual_positions,
        )

    overlay = _OperationalSpaceTrackingOverlay(
        osd=osd,
        trajectory_config=trajectory_config,
        t_traj=np.asarray(t_traj, dtype=np.float64),
        x_traj_full=x_traj_full_np,
        x_des_traj_full=x_des_traj_full_np,
        surface_geometry=surface_geometry,
        target_radius=0.50 * _mean_robot_radius(robot),
    )
    overlay.start(renderer)
    try:
        renderer.render_sequence(
            ts=np.asarray(t_traj),
            q_ts=jnp.asarray(q_traj),
            playback_speed=1.0,
            autoplay=True,
            loop=video_output is None,
            record_path=video_output,
            record_every_n=record_every_n,
            record_client_timeout=record_client_timeout,
            record_frame_timeout=record_frame_timeout,
            stop_when_recording_done=video_output is not None,
            camera_config=camera_config,
            color_config=color_config,
            robot_name="PCS Operational-Space Tracking",
            blocking=blocking,
        )
    finally:
        overlay.stop()


class _OperationalSpaceTrackingOverlay:
    """Background overlay updater synchronized to ViserRenderer animation state."""

    def __init__(
        self,
        *,
        osd: Any,
        trajectory_config: TaskSpaceTrajectoryConfig,
        t_traj: np.ndarray,
        x_traj_full: np.ndarray,
        x_des_traj_full: np.ndarray,
        surface_geometry: SurfaceGeometry | None,
        target_radius: float,
    ) -> None:
        self._osd = osd
        self._trajectory_config = trajectory_config
        self._t_traj = t_traj
        self._reference_positions = _extract_positions(osd, x_des_traj_full)
        self._actual_positions = _extract_positions(osd, x_traj_full)
        self._surface_geometry = surface_geometry
        self._target_radius = max(float(target_radius), 1e-3)
        self._reference_wxyz = (
            _poses_to_quaternions(osd, x_des_traj_full)
            if trajectory_config.tracking == "pose"
            else None
        )
        self._target_normals = _target_marker_normals(
            reference_positions=self._reference_positions,
            surface_geometry=surface_geometry,
            reference_wxyz=self._reference_wxyz,
        )

        dt = np.median(np.diff(t_traj)) if len(t_traj) > 1 else 0.01
        self._trail_max_segments = max(
            2,
            int(np.ceil(trajectory_config.period / max(float(dt), 1e-6))),
        )

        self._renderer: ViserRenderer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._goal_marker_handle: Any | None = None
        self._trail_handle: Any | None = None

    def start(self, renderer: ViserRenderer) -> None:
        self._renderer = renderer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        assert self._renderer is not None
        while (
            not self._stop.is_set()
            and getattr(self._renderer, "_animation_state", None) is None
        ):
            time.sleep(0.02)

        if self._stop.is_set():
            return

        self._build_scene()
        last_frame_idx = -1
        while not self._stop.is_set():
            state = getattr(self._renderer, "_animation_state", None)
            if state is None:
                break
            frame_idx = int(np.clip(state.frame_idx, 0, len(self._t_traj) - 1))
            if frame_idx != last_frame_idx:
                self._update(frame_idx)
                last_frame_idx = frame_idx
            time.sleep(0.02)

    def _build_scene(self) -> None:
        assert self._renderer is not None
        scene = self._renderer.server.scene

        if self._surface_geometry is not None:
            mesh = make_surface_mesh(
                self._surface_geometry,
                self._reference_positions,
            )
            scene.add_mesh_simple(
                name="/tracking/surface_wire",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(150, 165, 181),
                wireframe=True,
                opacity=0.055,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )

        reference_segments = make_polyline_segments(self._reference_positions)
        scene.add_line_segments(
            name="/tracking/reference_trajectory",
            points=reference_segments,
            colors=_constant_segment_colors(
                len(reference_segments),
                (0.02, 0.38, 0.78),
            ),
            line_width=3.2,
        )

        initial_trail = self._trail_segments(0)
        self._trail_handle = scene.add_line_segments(
            name="/tracking/actual_trail",
            points=initial_trail,
            colors=make_fading_trail_colors(len(initial_trail)),
            line_width=4.2,
        )

        target_marker = self._target_marker_segments(0)
        self._goal_marker_handle = scene.add_line_segments(
            name="/tracking/current_goal_marker",
            points=target_marker,
            colors=_target_marker_colors(len(target_marker)),
            line_width=2.8,
        )

        if self._reference_wxyz is not None:
            frame_stride = max(1, len(self._reference_positions) // 8)
            for frame_idx in range(0, len(self._reference_positions), frame_stride):
                scene.add_frame(
                    name=f"/tracking/reference_frame_{frame_idx}",
                    axes_length=0.32 * self._target_radius,
                    axes_radius=0.010 * self._target_radius,
                    origin_radius=0.025 * self._target_radius,
                    position=tuple(self._reference_positions[frame_idx]),
                    wxyz=tuple(self._reference_wxyz[frame_idx]),
                )

    def _update(self, frame_idx: int) -> None:
        if self._goal_marker_handle is not None:
            self._goal_marker_handle.points = self._target_marker_segments(frame_idx)

        if self._trail_handle is not None:
            trail_segments = self._trail_segments(frame_idx)
            self._trail_handle.points = trail_segments
            self._trail_handle.colors = make_fading_trail_colors(len(trail_segments))

    def _trail_segments(self, frame_idx: int) -> np.ndarray:
        end = min(frame_idx + 1, len(self._actual_positions))
        start = max(0, end - self._trail_max_segments - 1)
        return make_polyline_segments(self._actual_positions[start:end])

    def _target_marker_segments(self, frame_idx: int) -> np.ndarray:
        return make_target_marker_segments(
            self._reference_positions[frame_idx],
            self._target_normals[frame_idx],
            1.8 * self._target_radius,
        )


def _extract_positions(osd: Any, x_full: np.ndarray) -> np.ndarray:
    pos_start = osd.n_orientation_dim
    pos_dim = 2 if osd.is_planar else 3
    return _ensure_3d_positions(x_full[:, pos_start : pos_start + pos_dim])


def _poses_to_quaternions(osd: Any, x_full: np.ndarray) -> np.ndarray | None:
    if osd.is_planar:
        return None

    orientations = x_full[:, : osd.n_orientation_dim]
    if osd.rotation_representation == RotationRepresentation.QUATERNION:
        quats = orientations
    elif osd.rotation_representation == RotationRepresentation.ROTATION_VECTOR:
        quats = np.array(
            [
                np.asarray(
                    rotation_matrix_to_quaternion(
                        rotation_vector_to_rotation_matrix(jnp.asarray(orientation))
                    )
                )
                for orientation in orientations
            ]
        )
    elif osd.rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
        quats = np.array(
            [
                np.asarray(
                    rotation_matrix_to_quaternion(
                        rotation_6d_to_rotation_matrix(jnp.asarray(orientation))
                    )
                )
                for orientation in orientations
            ]
        )
    else:
        raise ValueError(
            f"Unsupported rotation representation: {osd.rotation_representation}"
        )

    return np.array([_normalize(quat) for quat in quats], dtype=np.float64)


def _target_marker_normals(
    *,
    reference_positions: np.ndarray,
    surface_geometry: SurfaceGeometry | None,
    reference_wxyz: np.ndarray | None,
) -> np.ndarray:
    if surface_geometry is not None:
        return np.asarray(
            [
                surface_normal_at(jnp.asarray(position), surface_geometry)
                for position in reference_positions
            ],
            dtype=np.float64,
        )
    if reference_wxyz is not None:
        return _quaternion_x_axes(reference_wxyz)
    return np.tile(np.array([0.0, 0.0, 1.0]), (len(reference_positions), 1))


def _quaternion_x_axes(wxyz: np.ndarray) -> np.ndarray:
    axes = []
    for quat in wxyz:
        w, x, y, z = _normalize(quat)
        axes.append(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
            ]
        )
    return np.asarray(axes, dtype=np.float64)


def _make_plane_mesh(
    surface_geometry: SurfaceGeometry,
    reference_positions: np.ndarray,
    resolution: int,
) -> SurfaceMesh:
    n_grid = max(8, resolution // 2)
    normal = _normalize(surface_geometry.plane_normal)
    y_axis, z_axis = _surface_tangent_axes(normal, surface_geometry.preferred_y)
    plane_point = np.asarray(surface_geometry.plane_point, dtype=np.float64)

    rel = reference_positions - plane_point
    u_values = rel @ y_axis
    v_values = rel @ z_axis
    half_u = max(
        0.5 * np.ptp(u_values) + 1.5 * surface_geometry.radius,
        2.0 * surface_geometry.radius,
    )
    half_v = max(
        0.5 * np.ptp(v_values) + 1.5 * surface_geometry.radius,
        2.0 * surface_geometry.radius,
    )
    center = (
        plane_point
        + 0.5 * (np.max(u_values) + np.min(u_values)) * y_axis
        + 0.5 * (np.max(v_values) + np.min(v_values)) * z_axis
    )

    us = np.linspace(-half_u, half_u, n_grid)
    vs = np.linspace(-half_v, half_v, n_grid)
    vertices = np.array(
        [center + u * y_axis + v * z_axis for u in us for v in vs],
        dtype=np.float32,
    )
    return SurfaceMesh(vertices=vertices, faces=_grid_faces(n_grid, n_grid))


def _make_cylinder_mesh(
    surface_geometry: SurfaceGeometry,
    reference_positions: np.ndarray,
    resolution: int,
) -> SurfaceMesh:
    axis_count = max(8, resolution // 4)
    theta_count = max(32, resolution)
    radius = float(surface_geometry.radius)
    axis = _normalize(surface_geometry.cylinder_axis)
    origin = np.asarray(surface_geometry.cylinder_origin, dtype=np.float64)
    radial = np.asarray(surface_geometry.plane_normal, dtype=np.float64)
    radial = radial - np.dot(radial, axis) * axis
    if np.linalg.norm(radial) <= 1e-12:
        radial = _fallback_axis(axis)
    radial = _normalize(radial)
    binormal = _normalize(np.cross(axis, radial))

    axis_values = (reference_positions - origin) @ axis
    half_axis = max(
        0.5 * np.ptp(axis_values) + 1.5 * radius,
        2.5 * radius,
    )
    axis_center = 0.5 * (np.max(axis_values) + np.min(axis_values))
    heights = np.linspace(axis_center - half_axis, axis_center + half_axis, axis_count)
    thetas = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False)

    vertices = np.array(
        [
            origin
            + height * axis
            + radius * (np.cos(theta) * radial + np.sin(theta) * binormal)
            for height in heights
            for theta in thetas
        ],
        dtype=np.float32,
    )
    return SurfaceMesh(
        vertices=vertices,
        faces=_grid_faces(axis_count, theta_count, wrap_cols=True),
    )


def _make_sphere_mesh(
    surface_geometry: SurfaceGeometry,
    resolution: int,
) -> SurfaceMesh:
    lon_count = max(32, resolution)
    lat_count = max(16, resolution // 2)
    radius = float(surface_geometry.radius)
    center = np.asarray(surface_geometry.sphere_center, dtype=np.float64)
    x_axis = _normalize(surface_geometry.plane_normal)
    y_axis, z_axis = _surface_tangent_axes(x_axis, surface_geometry.preferred_y)

    phis = np.linspace(0.0, np.pi, lat_count)
    thetas = np.linspace(0.0, 2.0 * np.pi, lon_count, endpoint=False)
    vertices = np.array(
        [
            center
            + radius
            * (
                np.cos(phi) * x_axis
                + np.sin(phi) * (np.cos(theta) * y_axis + np.sin(theta) * z_axis)
            )
            for phi in phis
            for theta in thetas
        ],
        dtype=np.float32,
    )
    return SurfaceMesh(
        vertices=vertices,
        faces=_grid_faces(lat_count, lon_count, wrap_cols=True),
    )


def _grid_faces(rows: int, cols: int, *, wrap_cols: bool = False) -> np.ndarray:
    faces: list[list[int]] = []
    for row in range(rows - 1):
        for col in range(cols):
            next_col = (col + 1) % cols if wrap_cols else col + 1
            if next_col >= cols:
                continue
            a = row * cols + col
            b = row * cols + next_col
            c = (row + 1) * cols + col
            d = (row + 1) * cols + next_col
            faces.append([a, c, b])
            faces.append([b, c, d])
    return np.asarray(faces, dtype=np.uint32)


def _surface_tangent_axes(
    normal: np.ndarray,
    preferred_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_axis = np.asarray(preferred_y, dtype=np.float64)
    y_axis = y_axis - np.dot(y_axis, normal) * normal
    if np.linalg.norm(y_axis) <= 1e-12:
        y_axis = _fallback_axis(normal)
    y_axis = _normalize(y_axis)
    z_axis = _normalize(np.cross(normal, y_axis))
    return y_axis, z_axis


def _ensure_3d_positions(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(
            f"points must have shape (N, 2) or (N, 3), got {points.shape}."
        )
    if points.shape[1] == 3:
        return points
    return np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)


def _mean_robot_radius(robot: Any) -> float:
    radius = getattr(robot, "r", None)
    if radius is None and hasattr(robot, "params"):
        radius = getattr(robot.params, "radius", None)
    if radius is None:
        return 0.02
    return float(np.mean(np.asarray(radius, dtype=np.float64)))


def _constant_segment_colors(
    num_segments: int,
    color: tuple[float, float, float],
) -> np.ndarray:
    colors = _to_uint8(np.tile(np.asarray(color, dtype=np.float64), (num_segments, 1)))
    return np.repeat(colors[:, None, :], 2, axis=1)


def _target_marker_colors(num_segments: int) -> np.ndarray:
    colors = np.tile(np.array([0.95, 0.30, 0.04]), (num_segments, 1))
    if num_segments >= 2:
        colors[-2:] = np.array([1.0, 0.08, 0.02])
    return np.repeat(_to_uint8(colors)[:, None, :], 2, axis=1)


def _fallback_axis(axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3)
    candidate = basis[int(np.argmin(np.abs(basis @ axis)))]
    return _normalize(candidate - np.dot(candidate, axis) * axis)


def _normalize(value: Array | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return arr / norm


def _to_uint8(colors: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(colors, 0.0, 1.0) * 255.0, dtype=np.uint8)


__all__ = [
    "SurfaceMesh",
    "make_fading_trail_colors",
    "make_operational_space_camera_config",
    "make_polyline_segments",
    "make_surface_mesh",
    "make_target_cross_segments",
    "make_target_marker_segments",
    "make_target_ring_segments",
    "make_transparent_robot_color_config",
    "render_operational_space_tracking",
]
