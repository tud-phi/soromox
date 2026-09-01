"""Publication renderer for the operational-space impedance-control rollout.

The visual language intentionally mirrors the trained Section Vf RL rendering:
a solid coral robot, a dark base, and a green target trajectory on white.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from PIL import Image
from trajectory_primitives import make_surface_geometry_metadata

from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import BackboneColorConfig, RendererColorConfig
from soromox.rendering.viser_renderer import ViserRenderer
from soromox.utils.geometry.rotations import (
    RotationRepresentation,
    rotation_6d_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_vector_to_rotation_matrix,
)

CORAL_RGB = np.array([0xF1, 0x55, 0x2E], dtype=np.float64) / 255.0
TARGET_GREEN_RGB = np.array([0x0E, 0xAD, 0x69], dtype=np.float64) / 255.0
TARGET_DARK_GREEN_RGB = np.array([0x08, 0x7A, 0x4A], dtype=np.float64) / 255.0
TARGET_TRAIL_RGB = 0.35 * np.ones(3) + 0.65 * TARGET_GREEN_RGB
BASE_RGB = np.array([0.2, 0.2, 0.2], dtype=np.float64)
SURFACE_FILL_RGB = np.array([0xB7, 0xC3, 0xCC], dtype=np.float64) / 255.0
SURFACE_MESH_RGB = np.array([0x82, 0x92, 0x9E], dtype=np.float64) / 255.0

DEFAULT_SNAPSHOT_TIMES = (4.5, 5.5, 6.5, 7.5)
DEFAULT_VIDEO_SIZE = (1920, 1080)
DEFAULT_FOV_DEGREES = 60.0
DEFAULT_CAMERA_AZIMUTH_DEGREES = 270.0
DEFAULT_CAMERA_ELEVATION_DEGREES = 55.0
DEFAULT_BASE_PLATE_RADIUS_SCALE = 2.0
DEFAULT_BASE_PLATE_THICKNESS = 0.06
SURFACE_FILL_OPACITY = 0.10
SURFACE_MESH_OPACITY = 0.16
SURFACE_LONGITUDE_COUNT = 24
SURFACE_LATITUDE_COUNT = 13
SNAPSHOT_STRIP_SIZE_CM = (15.2, 4.8)
SNAPSHOT_STRIP_WSPACE = -0.06
TIME_TEXT_COLOR = "0.45"
TIME_ARROW_COLOR = "0.60"


@dataclass(frozen=True)
class TargetDiskGeometry:
    """Local target-disk mesh and line geometry in the y-z tangent plane."""

    vertices: np.ndarray
    faces: np.ndarray
    rim_segments: np.ndarray
    cross_segments: np.ndarray


@dataclass(frozen=True)
class SphereSurfaceMesh:
    """World-space spherical surface mesh used by the paper renderer."""

    vertices: np.ndarray
    faces: np.ndarray


def make_paper_robot_color_config(num_points: int) -> RendererColorConfig:
    """Return the opaque coral robot style used by the trained RL rendering."""
    num_points = max(2, int(num_points))
    rgba = np.concatenate([CORAL_RGB, np.array([1.0])])
    point_colors = np.tile(rgba, (num_points, 1))
    return RendererColorConfig(
        backbone=BackboneColorConfig(point_colors=point_colors),
        base_plate_color=tuple(BASE_RGB),
    )


def make_target_disk_geometry(
    radius: float,
    *,
    num_points: int = 64,
) -> TargetDiskGeometry:
    """Return a double-sided disk mesh with a rim and tangent-frame crosshair."""
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Target disk radius must be finite and positive.")
    num_points = max(12, int(num_points))
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    ring = np.column_stack(
        [np.zeros_like(theta), radius * np.cos(theta), radius * np.sin(theta)]
    )
    vertices = np.concatenate([np.zeros((1, 3)), ring], axis=0).astype(np.float32)
    faces = np.array(
        [[0, 1 + index, 1 + ((index + 1) % num_points)] for index in range(num_points)],
        dtype=np.uint32,
    )
    rim_segments = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1).astype(
        np.float32
    )
    cross_radius = 0.82 * radius
    cross_segments = np.array(
        [
            [[0.0, -cross_radius, 0.0], [0.0, cross_radius, 0.0]],
            [[0.0, 0.0, -cross_radius], [0.0, 0.0, cross_radius]],
        ],
        dtype=np.float32,
    )
    return TargetDiskGeometry(
        vertices=vertices,
        faces=faces,
        rim_segments=rim_segments,
        cross_segments=cross_segments,
    )


def make_sphere_surface_mesh(
    center: np.ndarray,
    radius: float,
    *,
    longitude_count: int = SURFACE_LONGITUDE_COUNT,
    latitude_count: int = SURFACE_LATITUDE_COUNT,
) -> SphereSurfaceMesh:
    """Create a sparse, closed UV sphere with outward-facing triangles."""
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Sphere center must contain three finite coordinates.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Sphere radius must be finite and positive.")
    longitude_count = max(8, int(longitude_count))
    latitude_count = max(5, int(latitude_count))

    theta = np.linspace(0.0, 2.0 * np.pi, longitude_count, endpoint=False)
    phi = np.linspace(0.0, np.pi, latitude_count)
    vertices = [center + radius * np.array([0.0, 0.0, 1.0])]
    for latitude in phi[1:-1]:
        vertices.extend(
            center
            + radius
            * np.column_stack(
                [
                    np.sin(latitude) * np.cos(theta),
                    np.sin(latitude) * np.sin(theta),
                    np.full_like(theta, np.cos(latitude)),
                ]
            )
        )
    vertices.append(center + radius * np.array([0.0, 0.0, -1.0]))

    faces: list[list[int]] = []
    first_ring = 1
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append([0, first_ring + longitude, first_ring + next_longitude])

    ring_count = latitude_count - 2
    for ring in range(ring_count - 1):
        current_start = 1 + ring * longitude_count
        next_start = current_start + longitude_count
        for longitude in range(longitude_count):
            next_longitude = (longitude + 1) % longitude_count
            a = current_start + longitude
            b = current_start + next_longitude
            c = next_start + longitude
            d = next_start + next_longitude
            faces.extend([[a, c, b], [b, c, d]])

    bottom = len(vertices) - 1
    last_ring = bottom - longitude_count
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        faces.append([bottom, last_ring + next_longitude, last_ring + longitude])

    return SphereSurfaceMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.uint32),
    )


def transform_segments(
    segments: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Transform local line segments to world coordinates."""
    segments = np.asarray(segments, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments must have shape (N, 2, 3), got {segments.shape}.")
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("position and rotation must have shapes (3,) and (3, 3).")
    return (segments @ rotation.T + position).astype(np.float32)


def resample_periodic_path_by_arclength(
    t: np.ndarray,
    positions: np.ndarray,
    *,
    period: float,
    spacing: float,
) -> np.ndarray:
    """Resample one unique period of a trajectory at near-uniform arc length."""
    t = np.asarray(t, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    period = float(period)
    spacing = float(spacing)
    if t.ndim != 1 or positions.shape != (len(t), 3):
        raise ValueError("t and positions must have shapes (T,) and (T, 3).")
    if len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must contain at least two strictly increasing samples.")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive.")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive.")

    start_time = float(t[0])
    end_time = start_time + period
    if end_time > float(t[-1]) + 1e-9:
        raise ValueError("The trajectory does not contain one complete period.")

    mask = t < end_time
    period_t = t[mask]
    period_positions = positions[mask]
    end_position = np.array(
        [np.interp(end_time, t, positions[:, dim]) for dim in range(3)],
        dtype=np.float64,
    )
    if len(period_t) == 0 or not np.isclose(period_t[-1], end_time):
        period_t = np.concatenate([period_t, np.array([end_time])])
        period_positions = np.concatenate(
            [period_positions, end_position.reshape(1, 3)], axis=0
        )

    segment_lengths = np.linalg.norm(np.diff(period_positions, axis=0), axis=1)
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return period_positions[:1].copy()

    target_distances = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if len(target_distances) < 2:
        target_distances = np.array([0.0, 0.5 * total_length], dtype=np.float64)
    beads = np.column_stack(
        [
            np.interp(target_distances, cumulative, period_positions[:, dim])
            for dim in range(3)
        ]
    )
    return beads.astype(np.float64)


def make_operational_space_camera_config(
    reference_positions: np.ndarray,
    actual_positions: np.ndarray,
    *,
    robot_radius: float,
    sphere_surface: SphereSurfaceMesh | None = None,
) -> CameraConfig:
    """Return the elevated frontal-oblique camera used by the paper render."""
    reference_positions = _ensure_3d_positions(reference_positions)
    actual_positions = _ensure_3d_positions(actual_positions)
    robot_radius = float(robot_radius)
    base_radius = DEFAULT_BASE_PLATE_RADIUS_SCALE * robot_radius
    base_z = -DEFAULT_BASE_PLATE_THICKNESS
    base_bounds = np.array(
        [
            [x, y, z]
            for x in (-base_radius, base_radius)
            for y in (-base_radius, base_radius)
            for z in (base_z, 0.0)
        ],
        dtype=np.float64,
    )
    point_groups = [reference_positions, actual_positions, base_bounds]
    if sphere_surface is not None:
        point_groups.append(np.asarray(sphere_surface.vertices, dtype=np.float64))
    points = np.concatenate(point_groups, axis=0)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    look_at = 0.5 * (lower + upper)
    radius = max(float(np.max(np.linalg.norm(points - look_at, axis=1))), 0.1)

    azimuth = np.deg2rad(DEFAULT_CAMERA_AZIMUTH_DEGREES)
    elevation = np.deg2rad(DEFAULT_CAMERA_ELEVATION_DEGREES)
    view_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    distance = 1.15 * radius / np.sin(np.deg2rad(DEFAULT_FOV_DEGREES / 2.0))
    camera_position = look_at + distance * view_direction
    return CameraConfig(
        fov=DEFAULT_FOV_DEGREES,
        position=tuple(camera_position),
        look_at=tuple(look_at),
        up=(0.0, 0.0, 1.0),
    )


def snapshot_indices_for_times(
    t: np.ndarray,
    snapshot_times: tuple[float, ...] | list[float] | np.ndarray,
) -> tuple[int, int, int, int]:
    """Map exactly four valid, distinct times to distinct nearest frame indices."""
    t = np.asarray(t, dtype=np.float64)
    requested = np.asarray(snapshot_times, dtype=np.float64)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be a non-empty, strictly increasing vector.")
    if requested.shape != (4,) or not np.all(np.isfinite(requested)):
        raise ValueError("Exactly four finite snapshot times are required.")
    if len(np.unique(requested)) != 4:
        raise ValueError("Snapshot times must be distinct.")
    if np.any(requested < t[0]) or np.any(requested > t[-1]):
        raise ValueError(
            f"Snapshot times must lie within [{float(t[0])}, {float(t[-1])}] s."
        )
    indices = tuple(int(np.argmin(np.abs(t - value))) for value in requested)
    if len(set(indices)) != 4:
        raise ValueError("Snapshot times must map to four distinct saved frames.")
    return indices  # type: ignore[return-value]


def snapshot_filename(time_seconds: float) -> str:
    """Return a stable filename containing a non-negative snapshot time."""
    time_seconds = float(time_seconds)
    if not np.isfinite(time_seconds) or time_seconds < 0.0:
        raise ValueError("Snapshot time must be finite and non-negative.")
    encoded_time = f"{time_seconds:05.2f}".replace(".", "p")
    return f"snapshot_t{encoded_time}s.png"


def crop_snapshots_to_common_square(
    snapshot_paths: list[Path] | tuple[Path, ...],
    *,
    padding_fraction: float = 0.06,
    white_tolerance: int = 8,
) -> tuple[int, int, int, int]:
    """Apply one square crop containing the union of all non-white content."""
    paths = [Path(path) for path in snapshot_paths]
    if len(paths) != 4:
        raise ValueError("Exactly four snapshots are required for common cropping.")
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        sizes = {image.size for image in images}
        if len(sizes) != 1:
            raise ValueError("All snapshots must have identical source dimensions.")
        width, height = images[0].size
        union_mask = np.zeros((height, width), dtype=bool)
        for image in images:
            rgb = np.asarray(image, dtype=np.uint8)
            union_mask |= np.max(255 - rgb.astype(np.int16), axis=2) >= white_tolerance
        rows, cols = np.nonzero(union_mask)
        if len(rows) == 0:
            raise ValueError("Snapshots contain no non-white rendered content.")

        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        content_side = max(x1 - x0, y1 - y0)
        padding = int(np.ceil(padding_fraction * content_side))
        side = content_side + 2 * padding
        if side > min(width, height):
            raise ValueError("Rendered content is too large for a common square crop.")
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        left = int(round(center_x - 0.5 * side))
        top = int(round(center_y - 0.5 * side))
        left = int(np.clip(left, 0, width - side))
        top = int(np.clip(top, 0, height - side))
        crop_box = (left, top, left + side, top + side)
        for path, image in zip(paths, images, strict=True):
            image.crop(crop_box).save(path, format="PNG")
        return crop_box
    finally:
        for image in images:
            image.close()


def save_snapshot_strip(
    snapshot_paths: list[Path] | tuple[Path, ...],
    output: Path,
    *,
    snapshot_times: tuple[float, ...] | list[float] | np.ndarray = (
        DEFAULT_SNAPSHOT_TIMES
    ),
) -> Path:
    """Assemble four cropped PNGs into a compact, time-annotated PDF strip."""
    paths = [Path(path) for path in snapshot_paths]
    if len(paths) != 4:
        raise ValueError("Exactly four snapshots are required for the PDF strip.")
    times = np.asarray(snapshot_times, dtype=np.float64)
    if times.shape != (4,) or not np.all(np.isfinite(times)):
        raise ValueError("Exactly four finite snapshot times are required.")
    style_path = Path(__file__).resolve().parents[3] / "paper.mplstyle"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with plt.style.context(style_path), mpl.rc_context({"savefig.bbox": None}):
        # 254 dpi is exactly 100 pixels/cm, which prevents Matplotlib from
        # quantizing the physical PDF page size to coarse 0.01-inch steps.
        fig = plt.figure(
            figsize=tuple(value / 2.54 for value in SNAPSHOT_STRIP_SIZE_CM),
            dpi=254,
            facecolor="white",
        )
        grid = fig.add_gridspec(
            1,
            4,
            left=0.005,
            right=0.995,
            bottom=0.185,
            top=0.995,
            # The source panels remain square.  A small negative spacing only
            # overlaps their unused white margins, preserving image scale while
            # moving the rendered robots closer together.
            wspace=SNAPSHOT_STRIP_WSPACE,
        )
        panel_centers = []
        for column, path in enumerate(paths):
            axis = fig.add_subplot(grid[0, column])
            with Image.open(path) as image:
                axis.imshow(np.asarray(image.convert("RGB")))
            panel_box = axis.get_position()
            panel_centers.append(0.5 * (panel_box.x0 + panel_box.x1))
            axis.set_axis_off()

        for panel_center, time_seconds in zip(panel_centers, times, strict=True):
            fig.text(
                panel_center,
                0.12,
                rf"$t = {time_seconds:.2f}\,\mathrm{{s}}$",
                color=TIME_TEXT_COLOR,
                fontsize=7,
                ha="center",
                va="bottom",
            )

        arrow_axis = fig.add_axes((0.01, 0.01, 0.98, 0.095), frameon=False)
        arrow_axis.set_axis_off()
        arrow_axis.annotate(
            "",
            xy=(0.92, 0.5),
            xytext=(0.02, 0.5),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": TIME_ARROW_COLOR,
                "linewidth": 0.8,
            },
        )
        arrow_axis.text(
            0.93,
            0.5,
            "Time",
            color=TIME_TEXT_COLOR,
            fontsize=7,
            ha="left",
            va="center",
        )
        fig.savefig(output, facecolor="white", bbox_inches=None)
        plt.close(fig)
    return output


class OperationalSpacePaperRenderer(ViserRenderer):
    """Viser renderer with a deterministic moving target-pose disk overlay."""

    def __init__(
        self,
        *args: Any,
        reference_positions: np.ndarray,
        reference_wxyz: np.ndarray,
        target_disk_radius: float,
        sphere_surface: SphereSurfaceMesh | None = None,
        **kwargs: Any,
    ) -> None:
        self._reference_positions = _ensure_3d_positions(reference_positions)
        self._reference_wxyz = np.asarray(reference_wxyz, dtype=np.float64)
        if self._reference_wxyz.shape != (len(self._reference_positions), 4):
            raise ValueError("reference_wxyz must have shape (T, 4).")
        self._target_disk_geometry = make_target_disk_geometry(target_disk_radius)
        self._sphere_surface = sphere_surface
        self._sphere_fill_handle: Any | None = None
        self._sphere_mesh_handle: Any | None = None
        self._target_disk_handle: Any | None = None
        self._target_rim_handle: Any | None = None
        self._target_cross_handle: Any | None = None
        super().__init__(*args, **kwargs)

    def _after_sequence_scene_built(self) -> None:
        if self._server is None:
            return
        # Viser otherwise inherits its background from the browser theme.  A
        # tiny viewport background image makes paper captures deterministically
        # white without changing the generic renderer's interactive behavior.
        self._server.scene.set_background_image(
            np.full((2, 2, 3), 255, dtype=np.uint8),
            format="png",
        )
        if self._sphere_surface is not None:
            sphere = self._sphere_surface
            self._sphere_fill_handle = self._server.scene.add_mesh_simple(
                name="/tracking/reference_sphere_fill",
                vertices=sphere.vertices,
                faces=sphere.faces,
                color=tuple(np.rint(255.0 * SURFACE_FILL_RGB).astype(int)),
                wireframe=False,
                opacity=SURFACE_FILL_OPACITY,
                side="front",
                material="standard",
                flat_shading=False,
                cast_shadow=False,
                receive_shadow=False,
            )
            self._sphere_mesh_handle = self._server.scene.add_mesh_simple(
                name="/tracking/reference_sphere_mesh",
                vertices=sphere.vertices,
                faces=sphere.faces,
                color=tuple(np.rint(255.0 * SURFACE_MESH_RGB).astype(int)),
                wireframe=True,
                opacity=SURFACE_MESH_OPACITY,
                side="front",
                material="standard",
                flat_shading=False,
                cast_shadow=False,
                receive_shadow=False,
            )
        position = self._reference_positions[0]
        wxyz = self._reference_wxyz[0]
        rotation = _quaternion_to_rotation_matrix(wxyz)
        geometry = self._target_disk_geometry
        self._target_disk_handle = self._server.scene.add_mesh_simple(
            name="/tracking/current_target_disk",
            vertices=geometry.vertices,
            faces=geometry.faces,
            color=tuple(np.rint(255.0 * TARGET_GREEN_RGB).astype(int)),
            position=tuple(position),
            wxyz=tuple(wxyz),
            side="double",
            material="standard",
            flat_shading=False,
            opacity=1.0,
            cast_shadow=False,
            receive_shadow=False,
        )
        line_color = _constant_segment_colors(1, TARGET_DARK_GREEN_RGB)[0, 0]
        rim_colors = np.tile(
            line_color.reshape(1, 1, 3),
            (len(geometry.rim_segments), 2, 1),
        )
        cross_colors = np.tile(
            line_color.reshape(1, 1, 3),
            (len(geometry.cross_segments), 2, 1),
        )
        self._target_rim_handle = self._server.scene.add_line_segments(
            name="/tracking/current_target_disk_rim",
            points=transform_segments(geometry.rim_segments, position, rotation),
            colors=rim_colors,
            line_width=3.0,
        )
        self._target_cross_handle = self._server.scene.add_line_segments(
            name="/tracking/current_target_disk_cross",
            points=transform_segments(geometry.cross_segments, position, rotation),
            colors=cross_colors,
            line_width=3.2,
        )

    def _after_sequence_frame_updated(self, frame_idx: int) -> None:
        if self._target_disk_handle is None:
            return
        frame_idx = int(np.clip(frame_idx, 0, len(self._reference_positions) - 1))
        position = self._reference_positions[frame_idx]
        wxyz = self._reference_wxyz[frame_idx]
        rotation = _quaternion_to_rotation_matrix(wxyz)
        self._target_disk_handle.position = tuple(position)
        self._target_disk_handle.wxyz = tuple(wxyz)
        if self._target_rim_handle is not None:
            self._target_rim_handle.points = transform_segments(
                self._target_disk_geometry.rim_segments,
                position,
                rotation,
            )
        if self._target_cross_handle is not None:
            self._target_cross_handle.points = transform_segments(
                self._target_disk_geometry.cross_segments,
                position,
                rotation,
            )


def render_operational_space_tracking(
    *,
    robot: Any,
    osd: Any,
    trajectory_config: Any,
    t_traj: Array,
    q_traj: Array,
    x_traj_full: Array,
    x_des_traj_full: Array,
    video_output: str | Path,
    snapshot_paths: dict[int, Path],
    port: int = 8080,
    open_browser: bool = True,
    record_every_n: int = 1,
    record_client_timeout: float = 10.0,
    record_frame_timeout: float = 10.0,
    renderer_cls: type[OperationalSpacePaperRenderer] = OperationalSpacePaperRenderer,
) -> None:
    """Render the complete rollout and selected PNGs from one synchronized pass."""
    t = np.asarray(t_traj, dtype=np.float64)
    reference_positions = _extract_positions(osd, np.asarray(x_des_traj_full))
    actual_positions = _extract_positions(osd, np.asarray(x_traj_full))
    reference_wxyz = _poses_to_quaternions(osd, np.asarray(x_des_traj_full))
    robot_radius = _mean_robot_radius(robot)
    q_trajectory = jnp.asarray(q_traj)
    sphere_surface = None
    if (
        trajectory_config.tracking == "pose"
        and trajectory_config.orientation_primitive == "surface-following"
        and trajectory_config.surface == "sphere"
    ):
        surface_geometry = make_surface_geometry_metadata(
            osd,
            q_trajectory[0],
            trajectory_config,
        )
        sphere_surface = make_sphere_surface_mesh(
            np.asarray(surface_geometry.sphere_center, dtype=np.float64),
            float(surface_geometry.radius),
        )
    bead_positions = resample_periodic_path_by_arclength(
        t,
        reference_positions,
        period=float(trajectory_config.period),
        spacing=0.22 * robot_radius,
    )
    bead_radii = np.full(len(bead_positions), 0.10 * robot_radius)
    bead_colors = np.tile(TARGET_TRAIL_RGB, (len(bead_positions), 1))
    color_config = make_paper_robot_color_config(num_points=80)
    camera_config = make_operational_space_camera_config(
        reference_positions,
        actual_positions,
        robot_radius=robot_radius,
        sphere_surface=sphere_surface,
    )

    renderer = renderer_cls(
        robot,
        reference_positions=reference_positions,
        reference_wxyz=reference_wxyz,
        target_disk_radius=1.25 * robot_radius,
        sphere_surface=sphere_surface,
        width=DEFAULT_VIDEO_SIZE[0],
        height=DEFAULT_VIDEO_SIZE[1],
        num_points=80,
        port=port,
        open_browser=open_browser,
        color_config=color_config,
        backbone_style="swept",
        cylinder_sections=64,
        background_color=(1.0, 1.0, 1.0),
        material="standard",
        flat_shading=False,
        wireframe=False,
        cast_shadows=False,
        backbone_cast_shadow=False,
        sphere_cast_shadow=False,
        base_plate_radius_scale=DEFAULT_BASE_PLATE_RADIUS_SCALE,
        base_plate_thickness=DEFAULT_BASE_PLATE_THICKNESS,
    )
    try:
        renderer.render_sequence(
            ts=t,
            q_ts=q_trajectory,
            playback_speed=1.0,
            autoplay=True,
            loop=False,
            record_path=str(video_output),
            record_every_n=record_every_n,
            stop_when_recording_done=True,
            record_client_timeout=record_client_timeout,
            record_frame_timeout=record_frame_timeout,
            camera_config=camera_config,
            color_config=color_config,
            render_actuators=False,
            static_spheres_positions=bead_positions,
            static_spheres_radii=bead_radii,
            static_spheres_colors=bead_colors,
            snapshot_paths=snapshot_paths,
            robot_name="PCS Operational-Space Tracking",
            blocking=True,
        )
    finally:
        renderer.stop()


def _extract_positions(osd: Any, x_full: np.ndarray) -> np.ndarray:
    pos_start = osd.n_orientation_dim
    pos_dim = 2 if osd.is_planar else 3
    return _ensure_3d_positions(x_full[:, pos_start : pos_start + pos_dim])


def _poses_to_quaternions(osd: Any, x_full: np.ndarray) -> np.ndarray:
    if osd.is_planar:
        raise ValueError("The paper target disk requires a spatial pose trajectory.")
    orientations = np.asarray(x_full[:, : osd.n_orientation_dim], dtype=np.float64)
    quaternions = []
    for orientation in orientations:
        if osd.rotation_representation == RotationRepresentation.QUATERNION:
            quaternion = orientation
        elif osd.rotation_representation == RotationRepresentation.ROTATION_VECTOR:
            quaternion = rotation_matrix_to_quaternion(
                rotation_vector_to_rotation_matrix(jnp.asarray(orientation))
            )
        elif osd.rotation_representation == RotationRepresentation.ROTATION_MATRIX_6D:
            quaternion = rotation_matrix_to_quaternion(
                rotation_6d_to_rotation_matrix(jnp.asarray(orientation))
            )
        else:
            raise ValueError(
                f"Unsupported rotation representation: {osd.rotation_representation}"
            )
        xyzw = _normalize(np.asarray(quaternion, dtype=np.float64))
        quaternions.append(xyzw[[3, 0, 1, 2]])
    return np.asarray(quaternions, dtype=np.float64)


def _quaternion_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(np.asarray(wxyz, dtype=np.float64))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _constant_segment_colors(
    num_segments: int,
    color: np.ndarray,
) -> np.ndarray:
    color_u8 = np.rint(255.0 * np.clip(color, 0.0, 1.0)).astype(np.uint8)
    return np.tile(color_u8.reshape(1, 1, 3), (num_segments, 2, 1))


def _mean_robot_radius(robot: Any) -> float:
    radius = getattr(robot, "r", None)
    if radius is None and hasattr(robot, "params"):
        radius = getattr(robot.params, "radius", None)
    if radius is None:
        return 0.02
    return float(np.mean(np.asarray(radius, dtype=np.float64)))


def _ensure_3d_positions(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(
            f"points must have shape (N, 2) or (N, 3), got {points.shape}."
        )
    if points.shape[1] == 3:
        return points
    return np.concatenate([points, np.zeros((len(points), 1))], axis=1)


def _normalize(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return value / norm


__all__ = [
    "BASE_RGB",
    "CORAL_RGB",
    "DEFAULT_CAMERA_AZIMUTH_DEGREES",
    "DEFAULT_CAMERA_ELEVATION_DEGREES",
    "DEFAULT_SNAPSHOT_TIMES",
    "OperationalSpacePaperRenderer",
    "SURFACE_FILL_OPACITY",
    "SURFACE_FILL_RGB",
    "SURFACE_MESH_OPACITY",
    "SURFACE_MESH_RGB",
    "SphereSurfaceMesh",
    "TARGET_DARK_GREEN_RGB",
    "TARGET_GREEN_RGB",
    "TARGET_TRAIL_RGB",
    "TargetDiskGeometry",
    "crop_snapshots_to_common_square",
    "make_operational_space_camera_config",
    "make_paper_robot_color_config",
    "make_sphere_surface_mesh",
    "make_target_disk_geometry",
    "render_operational_space_tracking",
    "resample_periodic_path_by_arclength",
    "save_snapshot_strip",
    "snapshot_filename",
    "snapshot_indices_for_times",
    "transform_segments",
]
