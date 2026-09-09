"""Open3D-based renderer for any continuum soft robot.

Provides 3D visualization with:
- Backbone as spheres (per-segment colors & radii)
- Actuators as polyline LineSets (if robot exposes actuator visual layers)
- Recording frames to PNGs
- Interactive camera controls

Controls:
    Space: play/pause (press to start if autoplay is disabled)
    →/← : next/prev frame
    H   : go to frame 0
    S   : save snapshot
    R   : reset camera to initial view
    C   : capture/save current camera view
    L   : load/restore saved camera view
    V   : print current camera parameters
    Q/ESC: quit
"""

from __future__ import annotations

import copy
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from soromox.rendering.actuators import (
    TrajectoryActuatorVisualLayer,
    resolve_actuator_rgba,
)
from soromox.rendering.base import BaseSoftRobotRenderer
from soromox.rendering.config import DirectionalLightConfig, RendererConfig
from soromox.rendering.config.camera import CameraConfig
from soromox.rendering.config.colors import RendererColorConfig, ensure_rgba
from soromox.rendering.config.output import VideoEncodingConfig
from soromox.rendering.cross_sections import (
    CrossSection,
    cross_section_sweep_layout,
    evaluate_cross_sections,
    loft_cross_section_contours,
    loft_cross_sections,
)
from soromox.rendering.scenery import (
    backdrop_mesh,
    ground_grid,
    linear_to_srgb,
    srgb_to_linear,
)
from soromox.rendering.video_encoding import FFmpegVideoWriter
from soromox.systems.soft_robot import SoftRobot

# ======================================================================================
# Geometry helper functions (module-level, stateless)
# ======================================================================================


def _make_polyline_lineset(
    points_np: np.ndarray,
    color: tuple[float, float, float] = (0.9, 0.15, 0.15),
) -> o3d.geometry.LineSet:
    """Create a colored polyline LineSet from (N,3) points."""
    pts = np.array(points_np, dtype=np.float64, order="C", copy=True)
    N = pts.shape[0]
    if N < 2:
        pts = np.vstack([pts, pts[-1]])
        N = 2
    lines = np.stack([np.arange(0, N - 1), np.arange(1, N)], axis=1).astype(np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64)[None, :], (lines.shape[0], 1))
    )
    return ls


def _make_polylines_lineset(
    polylines: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.LineSet:
    """Create multiple independently colored polylines as one LineSet."""
    polylines = np.array(polylines, dtype=np.float64, order="C", copy=True)
    colors = np.array(colors, dtype=np.float64, order="C", copy=True)
    if polylines.ndim != 3 or polylines.shape[-1] not in (2, 3):
        raise ValueError(
            f"polylines must have shape (N, P, 2|3), got {polylines.shape}"
        )
    if polylines.shape[-1] == 2:
        polylines = np.pad(polylines, ((0, 0), (0, 0), (0, 1)))
    if polylines.shape[1] < 2:
        raise ValueError("polylines must contain at least two points")
    if colors.shape != (polylines.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({polylines.shape[0]}, 3), got {colors.shape}"
        )

    num_polylines, num_points, _ = polylines.shape
    point_indices = np.arange(num_polylines * num_points, dtype=np.int32).reshape(
        num_polylines, num_points
    )
    lines = np.stack((point_indices[:, :-1], point_indices[:, 1:]), axis=-1).reshape(
        -1, 2
    )
    line_colors = np.repeat(colors, num_points - 1, axis=0)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(polylines.reshape(-1, 3)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def _update_polylines_lineset(
    line_set: o3d.geometry.LineSet,
    polylines: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Update a batched LineSet without changing its fixed line topology."""
    polylines = np.array(polylines, dtype=np.float64, order="C", copy=True)
    colors = np.array(colors, dtype=np.float64, order="C", copy=True)
    if polylines.ndim != 3 or polylines.shape[-1] not in (2, 3):
        raise ValueError(
            f"polylines must have shape (N, P, 2|3), got {polylines.shape}"
        )
    if polylines.shape[-1] == 2:
        polylines = np.pad(polylines, ((0, 0), (0, 0), (0, 1)))
    if polylines.shape[1] < 2:
        raise ValueError("polylines must contain at least two points")
    if colors.shape != (polylines.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({polylines.shape[0]}, 3), got {colors.shape}"
        )
    expected_lines = polylines.shape[0] * (polylines.shape[1] - 1)
    if len(line_set.lines) != expected_lines:
        raise ValueError(
            "updated polylines must retain the original number of line segments; "
            f"expected {len(line_set.lines)}, got {expected_lines}"
        )
    line_set.points = o3d.utility.Vector3dVector(polylines.reshape(-1, 3))
    line_set.colors = o3d.utility.Vector3dVector(
        np.repeat(colors, polylines.shape[1] - 1, axis=0)
    )


def _make_base_plate(
    center_xyz: np.ndarray,
    radius: float,
    thickness: float = 0.005,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: int = 48,
    apply_color: bool = True,
    apply_translation: bool = True,
    normal_xyz: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    """Create a base plate cylinder mesh."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=float(thickness), resolution=resolution, split=1
    )
    if normal_xyz is not None:
        R = _axis_alignment_rotation(normal_xyz)
        transform = np.eye(4)
        transform[:3, :3] = R
        mesh.transform(transform)
    mesh.compute_vertex_normals()
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    if apply_translation:
        mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


def _make_ground_plane(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    size: float,
    plane_color: tuple[float, float, float],
    grid_color: tuple[float, float, float],
    grid_divisions: int = 10,
    ground_config=None,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
    """Create a finite, base-aligned ground plane and grid."""
    normal = np.asarray(normal_xyz, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    normal = (
        normal / normal_norm
        if normal_norm > 1e-9
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = np.cross(normal, reference)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    center = np.asarray(center_xyz, dtype=np.float64)
    half = 0.5 * float(size)

    corners = np.stack(
        [
            center - half * axis_u - half * axis_v,
            center + half * axis_u - half * axis_v,
            center + half * axis_u + half * axis_v,
            center - half * axis_u + half * axis_v,
        ]
    )
    plane = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corners),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]])),
    )
    plane.compute_vertex_normals()
    plane.paint_uniform_color(np.asarray(plane_color, dtype=np.float64))

    grid_center = center + 2e-4 * max(float(size), 1.0) * normal
    coordinates = np.linspace(-half, half, int(grid_divisions) + 1)[:, None]
    lines_along_u = np.stack(
        (
            grid_center - half * axis_u + coordinates * axis_v,
            grid_center + half * axis_u + coordinates * axis_v,
        ),
        axis=1,
    )
    lines_along_v = np.stack(
        (
            grid_center + coordinates * axis_u - half * axis_v,
            grid_center + coordinates * axis_u + half * axis_v,
        ),
        axis=1,
    )
    grid_segments = np.concatenate((lines_along_u, lines_along_v), axis=0)
    grid_points = grid_segments.reshape(-1, 3)
    grid_lines = np.arange(grid_points.shape[0], dtype=np.int32).reshape(-1, 2)
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_points),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    grid.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.asarray(grid_color, dtype=np.float64)[None, :],
            (grid_lines.shape[0], 1),
        )
    )
    if ground_config is not None:
        points, colors = ground_grid(ground_config, center, normal, size)
        grid.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        grid.lines = o3d.utility.Vector2iVector(
            np.arange(points.size // 3).reshape(-1, 2)
        )
        grid.colors = o3d.utility.Vector3dVector(colors)
    return plane, grid


def _make_sphere(
    center_xyz: np.ndarray,
    radius: float,
    color: tuple[float, float, float] = (0.1, 0.45, 1.0),
    resolution: int = 16,
    apply_color: bool = True,
    apply_translation: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere at center_xyz with given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(radius), resolution=resolution
    )
    mesh.compute_vertex_normals()
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    if apply_translation:
        mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


def _make_spheres_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    resolution: int = 16,
) -> o3d.geometry.TriangleMesh:
    """Create many independently colored spheres as one triangle mesh."""
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    colors = np.asarray(colors, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must have shape (N, 3), got {centers.shape}")
    if radii.shape != (centers.shape[0],):
        raise ValueError(
            f"radii must have shape ({centers.shape[0]},), got {radii.shape}"
        )
    if colors.shape != (centers.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({centers.shape[0]}, 3), got {colors.shape}"
        )
    if centers.shape[0] == 0:
        return o3d.geometry.TriangleMesh()

    unit = _make_unit_sphere_mesh(resolution)
    num_spheres = centers.shape[0]
    num_vertices = unit.vertices.shape[0]
    vertices = (
        unit.vertices[None, :, :] * radii[:, None, None] + centers[:, None, :]
    ).reshape(-1, 3)
    triangles = (
        unit.triangles[None, :, :]
        + (np.arange(num_spheres) * num_vertices)[:, None, None]
    ).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.tile(unit.normals, (num_spheres, 1))
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(colors, num_vertices, axis=0)
    )
    return mesh


_MODERN_GUI_CREATED = False


@dataclass(frozen=True)
class UnitMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray


@dataclass
class CachedMesh:
    mesh: o3d.geometry.TriangleMesh
    base_vertices: np.ndarray
    base_normals: np.ndarray | None
    scale_base: np.ndarray
    dynamic_length: bool
    rotate_to_axis: bool
    swept_contours: tuple[np.ndarray, np.ndarray] | None = None
    cap_start: bool = False
    cap_end: bool = False


def _merge_triangle_meshes(
    meshes: list[o3d.geometry.TriangleMesh],
) -> o3d.geometry.TriangleMesh:
    """Merge meshes while preserving triangles, normals, and vertex colors."""
    merged = o3d.geometry.TriangleMesh()
    if not meshes:
        return merged

    vertex_counts = np.asarray([len(mesh.vertices) for mesh in meshes], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(vertex_counts)))
    merged.vertices = o3d.utility.Vector3dVector(
        np.concatenate([np.asarray(mesh.vertices) for mesh in meshes], axis=0)
    )
    merged.triangles = o3d.utility.Vector3iVector(
        np.concatenate(
            [
                np.asarray(mesh.triangles, dtype=np.int64) + offsets[index]
                for index, mesh in enumerate(meshes)
            ],
            axis=0,
        )
    )
    if all(mesh.has_vertex_normals() for mesh in meshes):
        merged.vertex_normals = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(mesh.vertex_normals) for mesh in meshes], axis=0)
        )
    if all(mesh.has_vertex_colors() for mesh in meshes):
        merged.vertex_colors = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(mesh.vertex_colors) for mesh in meshes], axis=0)
        )
    return merged


def _refresh_merged_triangle_mesh(
    merged: o3d.geometry.TriangleMesh,
    meshes: list[o3d.geometry.TriangleMesh],
) -> None:
    """Refresh dynamic vertices and normals of a previously merged mesh."""
    merged.vertices = o3d.utility.Vector3dVector(
        np.concatenate([np.asarray(mesh.vertices) for mesh in meshes], axis=0)
    )
    if all(mesh.has_vertex_normals() for mesh in meshes):
        merged.vertex_normals = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(mesh.vertex_normals) for mesh in meshes], axis=0)
        )


def _unit_mesh_from_o3d(mesh: o3d.geometry.TriangleMesh) -> UnitMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return UnitMesh(vertices=vertices, triangles=triangles, normals=normals)


def _make_unit_sphere_mesh(resolution: int) -> UnitMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    mesh.compute_vertex_normals()
    return _unit_mesh_from_o3d(mesh)


def _make_unit_cylinder_mesh(resolution: int) -> UnitMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=1.0, height=1.0, resolution=resolution, split=1
    )
    mesh.translate(np.array([0.0, 0.0, -0.5], dtype=np.float64), relative=True)
    mesh.compute_vertex_normals()
    return _unit_mesh_from_o3d(mesh)


def _make_unit_box_mesh() -> UnitMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.translate(np.array([-0.5, -0.5, -0.5], dtype=np.float64), relative=True)
    mesh.compute_vertex_normals()
    return _unit_mesh_from_o3d(mesh)


def _axis_alignment_rotation(axis: np.ndarray) -> np.ndarray:
    """Rotation matrix aligning +Z with the given axis."""
    axis = np.asarray(axis, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / length

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z_axis, axis)
    s = np.linalg.norm(v)
    c = float(np.dot(z_axis, axis))
    if s < 1e-9 and c > 0.0:
        return np.eye(3)
    if s < 1e-9 and c < 0.0:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


def _make_cylinder_between(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    resolution: int = 20,
    apply_color: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create a cylinder mesh connecting p0->p1."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        length = 1e-9
    else:
        axis = axis / length

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=length, resolution=resolution, split=1
    )
    mesh.compute_vertex_normals()
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))

    # Align cylinder's +Z with desired axis
    R = _axis_alignment_rotation(axis)
    mesh.rotate(R, center=np.zeros(3))

    mid = (p0 + p1) / 2.0
    mesh.translate(mid, relative=False)
    return mesh


def _make_box_centered(
    center_xyz: np.ndarray,
    width: float,
    height: float,
    depth: float,
    color: tuple[float, float, float],
    apply_color: bool = True,
    apply_translation: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create a box mesh centered at center_xyz."""
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(width), height=float(height), depth=float(depth)
    )
    mesh.compute_vertex_normals()
    mesh.translate(
        np.array([-width / 2.0, -height / 2.0, -depth / 2.0], dtype=np.float64),
        relative=True,
    )
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    if apply_translation:
        mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


def _make_ellipsoid(
    center_xyz: np.ndarray,
    radii: tuple[float, float, float],
    color: tuple[float, float, float],
    resolution: int = 16,
    apply_color: bool = True,
    apply_translation: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create an ellipsoid mesh centered at center_xyz."""
    rx, ry, rz = radii
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    mesh.compute_vertex_normals()
    scale = np.diag([float(rx), float(ry), float(rz), 1.0])
    mesh.transform(scale)
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    if apply_translation:
        mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


def _make_box_between(
    p0: np.ndarray,
    p1: np.ndarray,
    width: float,
    height: float,
    color: tuple[float, float, float],
    apply_color: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create a box prism connecting p0->p1 with rectangular cross-section."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        length = 1e-9
    else:
        axis = axis / length
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(width), height=float(height), depth=length
    )
    mesh.compute_vertex_normals()
    mesh.translate(
        np.array([-width / 2.0, -height / 2.0, -length / 2.0], dtype=np.float64),
        relative=True,
    )
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    R = _axis_alignment_rotation(axis)
    mesh.rotate(R, center=np.zeros(3))
    mid = (p0 + p1) / 2.0
    mesh.translate(mid, relative=False)
    return mesh


def _make_elliptical_cylinder_between(
    p0: np.ndarray,
    p1: np.ndarray,
    a: float,
    b: float,
    color: tuple[float, float, float],
    resolution: int = 20,
    apply_color: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create an elliptical cylinder mesh connecting p0->p1."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        length = 1e-9
    else:
        axis = axis / length

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=1.0, height=length, resolution=resolution, split=1
    )
    mesh.compute_vertex_normals()
    scale = np.diag([float(a), float(b), 1.0, 1.0])
    mesh.transform(scale)
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    R = _axis_alignment_rotation(axis)
    mesh.rotate(R, center=np.zeros(3))
    mid = (p0 + p1) / 2.0
    mesh.translate(mid, relative=False)
    return mesh


def _primitive_rotation_from_material_frame(frame: np.ndarray) -> np.ndarray:
    """Map a +Z-axis primitive into the robot's +X backbone convention."""
    frame = np.asarray(frame, dtype=np.float64)
    return frame[:, [1, 2, 0]]


def _make_swept_cross_section_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    frame0: np.ndarray,
    frame1: np.ndarray,
    section0: CrossSection,
    section1: CrossSection,
    color: tuple[float, float, float],
    resolution: int,
    apply_color: bool = True,
    cap_start: bool = False,
    cap_end: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Create a body segment by lofting FK-oriented cross-section contours."""
    vertices, faces = loft_cross_sections(
        p0,
        p1,
        frame0,
        frame1,
        section0,
        section1,
        resolution,
        cap_start=cap_start,
        cap_end=cap_end,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    if apply_color:
        mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return mesh


def _make_target_sphere(
    center_xyz: np.ndarray,
    radius: float = 0.01,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    resolution: int = 16,
) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere marking a target point."""
    return _make_sphere(center_xyz, radius, color, resolution)


def _make_obstacle_sphere(
    center_xyz: np.ndarray,
    radius: float,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    resolution: int = 24,
) -> o3d.geometry.TriangleMesh:
    """Create a colored obstacle sphere."""
    return _make_sphere(center_xyz, radius, color, resolution)


def _mesh_center(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Approximate center: mean of vertices."""
    return np.asarray(mesh.vertices).mean(axis=0)


@dataclass
class SegmentLayout:
    """Precomputed segment metadata for backbone geometry."""

    starts: np.ndarray
    ends: np.ndarray
    radii: np.ndarray

    @property
    def segments(self) -> int:
        return int(self.starts.shape[0])


@dataclass
class SphereSet:
    centers: np.ndarray  # (N, 3)
    radii: np.ndarray  # (N,)
    colors: np.ndarray  # (N, 3)


@dataclass
class DynamicSpheres:
    trajectories: np.ndarray  # (N, T, 3)
    radii: np.ndarray  # (N,)
    colors: np.ndarray  # (N, 3)


@dataclass
class DynamicSphereBatchHandle:
    mesh: o3d.geometry.TriangleMesh
    trajectories: np.ndarray  # (N, T, 3)
    local_vertices: np.ndarray  # (N, V, 3)


@dataclass
class SceneData:
    curves: np.ndarray  # (N, T, P, 3)
    material_frames: np.ndarray  # (N, T, P, 3, 3)
    q_ts: np.ndarray  # (N, T, DOF)
    ts: np.ndarray  # (T,)
    layout: SegmentLayout
    segment_colors_rgba: np.ndarray  # (N, S, 4)
    actuator_layers: tuple[TrajectoryActuatorVisualLayer, ...] = ()
    static_spheres: SphereSet | None = None
    dynamic_spheres: DynamicSpheres | None = None

    @property
    def num_robots(self) -> int:
        return int(self.curves.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.curves.shape[1])


@dataclass
class RecordingConfig:
    path: str | None
    prefix: str = "frame_"
    every_n: int = 1
    video_config: VideoEncodingConfig | None = None
    close_when_done: bool = False


@dataclass
class SceneHandles:
    ground_meshes: list
    ground_lines: list
    base_meshes: list
    backbone_meshes: list[list[list[CachedMesh]]]
    merged_backbone_meshes: list[o3d.geometry.TriangleMesh | None]
    actuator_lines: list[o3d.geometry.LineSet]
    static_meshes: list
    dynamic_sphere_batch: DynamicSphereBatchHandle | None


# ======================================================================================
# Open3D Renderer Class
# ======================================================================================


class Open3DRenderer(BaseSoftRobotRenderer):
    """Open3D visualization for any continuum soft robot.

    Modern static views and image/video exports share materials, lighting and
    scene settings.
    Interactive playback uses the legacy visualizer for efficient mesh updates
    and warns about differences in shading.

    Example:
        ```python
        renderer = Open3DRenderer(robot)
        renderer.show(q)
        renderer.render_sequence(ts, q_ts, playback_speed=1.0)
        ```
    """

    def __init__(
        self,
        robot: SoftRobot,
        config: RendererConfig | None = None,
        recompute_normals: bool = True,
        sphere_resolution: int = 32,
        base_offsets: Array | None = None,
        camera_margin_ratio: float = 0.05,
        merge_backbone_meshes: bool | None = None,
    ):
        """Initialize Open3D renderer.

        Args:
            config: Shared scene, camera, color, geometry and output defaults.
            robot: Robot system with forward_kinematics method
            recompute_normals: Whether to recompute vertex normals per segment update
            sphere_resolution: Resolution for backbone spheres
            base_offsets: Explicit base offsets of shape (N, 2) or (N, 3) for batched rendering
            camera_margin_ratio: Margin ratio for camera bounding box
            merge_backbone_meshes: Whether to merge each robot's backbone
                primitives into one dynamic mesh. ``None`` selects merging
                automatically for multi-robot scenes while retaining the
                lower per-frame update cost of unmerged single-robot scenes."""
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install the rendering dependencies; see tools/open3d/README.md"
            )

        super().__init__(robot, config=config)
        backbone_style = self.config.geometry.backbone_style
        cross_section_resolution = self.config.geometry.cross_section_resolution
        base_plate_radius_scale = self.config.geometry.base_plate_radius_scale
        base_plate_thickness = self.config.geometry.base_plate_thickness
        grid_spacing = self.config.geometry.grid_spacing
        actuator_line_width = self.config.geometry.actuator_line_width

        self.scene_config = self.config.scene
        self.backbone_style = backbone_style
        self._backbone_mode = self._resolve_backbone_mode(backbone_style)
        self._sweep_layout = None
        self._swept_edge_caps: dict[int, tuple[bool, bool]] = {}
        if self._backbone_mode == "swept":
            self._sweep_layout = cross_section_sweep_layout(
                np.asarray(self.robot.segment_length, dtype=np.float64),
                self.num_points,
            )
            self._backbone_abscissae = jnp.asarray(self._sweep_layout.abscissae)
            self._segment_cache[self.num_points] = (
                self._sweep_layout.segment_starts,
                self._sweep_layout.segment_ends,
            )
            self._swept_edge_caps = self._sweep_layout.edge_caps()
        self.recompute_normals = bool(recompute_normals)

        self.sphere_resolution = sphere_resolution
        self.cross_section_resolution = int(cross_section_resolution)
        self.grid_spacing = grid_spacing
        self._base_offsets = base_offsets
        self.actuator_line_width = actuator_line_width
        self.camera_margin_ratio = camera_margin_ratio
        self.merge_backbone_meshes = (
            None if merge_backbone_meshes is None else bool(merge_backbone_meshes)
        )
        self.base_plate_radius_scale = float(base_plate_radius_scale)
        self.base_plate_thickness = float(base_plate_thickness)
        self._warned_dynamic_geometry = False

        self._unit_meshes = {
            "sphere": _make_unit_sphere_mesh(self.sphere_resolution),
            "ellipsoid": _make_unit_sphere_mesh(self.sphere_resolution),
            "box": _make_unit_box_mesh(),
            "cylinder": _make_unit_cylinder_mesh(self.cross_section_resolution),
            "elliptical_cylinder": _make_unit_cylinder_mesh(
                self.cross_section_resolution
            ),
        }

    @staticmethod
    def _resolve_backbone_mode(style: str) -> str:
        style_norm = str(style).strip().lower()
        style_map = {
            "discrete": "discrete",
            "swept": "swept",
        }
        if style_norm not in style_map:
            raise ValueError("backbone_style must be one of: discrete, swept")
        return style_map[style_norm]

    def _should_merge_backbone_meshes(self, num_robots: int) -> bool:
        """Resolve the explicit or scene-size-aware backbone batching mode."""
        if self.merge_backbone_meshes is None:
            return int(num_robots) > 1
        return self.merge_backbone_meshes

    @property
    def is_3d(self) -> bool:
        """Whether this renderer operates in 3D space.

        Returns:
            is_3d (bool): Always True for the Open3D renderer.
        """
        return True

    def _extract_positions(self, poses: Array) -> Array:
        """Extract 3D positions from SE(2) or SE(3) poses.

        Supports both planar (SE(2)) and 3D (SE(3)) robots by converting
        SE(2) poses [theta, x, y] to 3D coordinates [x, y, 0].

        Args:
            poses: Forward kinematics output - either:
                - (N, 3) for SE(2) poses [theta, x, y]
                - (N, 4, 4) for SE(3) transformation matrices

        Returns:
            Position array of shape (N, 3) with xyz coordinates.
        """
        return self._extract_positions_3d(poses)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _make_mesh_material(
        self,
        color_rgba: tuple[float, float, float] | tuple[float, float, float, float],
        *,
        ground: bool = False,
    ) -> o3d.visualization.rendering.MaterialRecord:
        """Create an Open3D material with sRGB adaptation and independent opacity.

        Args:
            color_rgba: Normalized public sRGB color with optional alpha.
            ground: Use ground opacity and a fully rough nonmetallic surface.

        Returns:
            MaterialRecord for the selected lit or unlit rendering path.
        """
        rgba = ensure_rgba(np.asarray(color_rgba, dtype=np.float64))[0]
        mat = o3d.visualization.rendering.MaterialRecord()
        rgba = rgba.copy()
        rgba[3] *= (
            self.scene_config.ground.opacity
            if ground
            else self.scene_config.material.opacity
        )
        lit = self.scene_config.material.shading != "unlit"
        if lit:
            rgba[:3] = srgb_to_linear(rgba[:3])
        else:
            # Unlit output bypasses grading. Compensate Open3D's automatic
            # sRGB-to-linear conversion and Filament's 0.5 unlit scaling
            # to preserve framebuffer colors with post-processing disabled.
            rgba[:3] = linear_to_srgb(2.0 * rgba[:3])
        mat.shader = (
            ("defaultLitTransparency" if lit else "defaultUnlitTransparency")
            if rgba[3] < 0.999
            else ("defaultLit" if lit else "defaultUnlit")
        )
        mat.base_color = (
            float(rgba[0]),
            float(rgba[1]),
            float(rgba[2]),
            float(rgba[3]),
        )
        mat.base_roughness = 1.0 if ground else self.scene_config.material.roughness
        mat.base_metallic = 0.0 if ground else self.scene_config.material.metallic
        mat.base_reflectance = self.scene_config.material.reflectance
        return mat

    def _blend_with_background(
        self,
        color: tuple[float, float, float]
        | tuple[float, float, float, float]
        | np.ndarray,
    ) -> tuple[float, float, float]:
        """Approximate transparency for the visualizer by blending with the background."""
        rgba = ensure_rgba(np.asarray(color, dtype=np.float64))[0]
        alpha = float(rgba[3])
        rgb = rgba[:3]
        if alpha >= 0.999:
            return tuple(rgb.tolist())
        bg = np.asarray(self.background_color, dtype=np.float64)
        blended = alpha * rgb + (1.0 - alpha) * bg
        return tuple(np.clip(blended, 0.0, 1.0).tolist())

    def _instantiate_mesh(self, unit: UnitMesh, color: tuple[float, float, float]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(unit.vertices.copy())
        mesh.triangles = o3d.utility.Vector3iVector(unit.triangles.copy())
        if unit.normals.size:
            mesh.vertex_normals = o3d.utility.Vector3dVector(unit.normals.copy())
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
        return mesh

    def _apply_cached_mesh(
        self,
        cached: CachedMesh,
        translation: np.ndarray,
        rotation: np.ndarray | None,
        length: float | None,
    ) -> None:
        if cached.dynamic_length:
            length_val = max(float(length or 0.0), 1e-9)
            scale = np.array(
                [cached.scale_base[0], cached.scale_base[1], length_val],
                dtype=np.float64,
            )
        else:
            scale = np.asarray(cached.scale_base, dtype=np.float64)

        if cached.rotate_to_axis and rotation is not None:
            R = rotation
        else:
            R = np.eye(3)

        linear = R @ np.diag(scale)
        verts = cached.base_vertices @ linear.T + translation
        cached.mesh.vertices = o3d.utility.Vector3dVector(verts)

        if self.recompute_normals:
            cached.mesh.compute_vertex_normals()
            return

        if cached.base_normals is not None and cached.base_normals.size:
            scale_safe = np.where(scale > 1e-9, scale, 1e-9)
            normals = cached.base_normals / scale_safe
            normals = normals @ R.T
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.where(norms > 1e-9, normals / norms, normals)
            cached.mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    @staticmethod
    def _apply_swept_cached_mesh(
        cached: CachedMesh,
        p0: np.ndarray,
        p1: np.ndarray,
        frame0: np.ndarray,
        frame1: np.ndarray,
    ) -> None:
        if cached.swept_contours is None:
            raise ValueError("Swept mesh is missing cross-section contours.")
        vertices, _ = loft_cross_section_contours(
            p0,
            p1,
            frame0,
            frame1,
            cached.swept_contours[0],
            cached.swept_contours[1],
            cap_start=cached.cap_start,
            cap_end=cached.cap_end,
        )
        cached.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        cached.mesh.compute_vertex_normals()

    def _frame_intervals_from_ts(self, ts: Array, playback_speed: float) -> np.ndarray:
        """Compute per-frame wall-clock intervals from timestamps, scaled by playback speed."""
        speed = float(playback_speed)
        if speed <= 0.0:
            raise ValueError("playback_speed must be positive")
        t_arr = np.asarray(ts, dtype=np.float64).reshape(-1)
        if t_arr.size < 2:
            base = np.array([1.0], dtype=np.float64)
        else:
            diffs = np.diff(t_arr)
            if not np.all(np.isfinite(diffs)):
                diffs = np.ones_like(diffs)
            diffs = np.where(diffs > 0.0, diffs, 1.0)
            base = diffs
        dt = base / speed
        return np.concatenate([dt, dt[-1:]])  # pad so indexing at last frame is safe

    def _compute_segment_layout(self, P: int) -> SegmentLayout:
        """Compute start/end indices, radii, and colors for backbone segments."""
        lengths = np.asarray(self.robot.segment_length, dtype=np.float64).reshape(-1)
        if hasattr(self.robot, "r"):
            r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        else:
            r_seg = np.zeros_like(lengths, dtype=np.float64)
        starts, ends = self._segment_bounds(P)
        return SegmentLayout(
            starts=starts,
            ends=ends,
            radii=r_seg,
        )

    @staticmethod
    def _normalize_color_array(
        colors: Array | None, count: int, default_color: tuple[float, float, float]
    ) -> np.ndarray:
        """Validate and normalize color arrays to shape (count, 3)."""
        if colors is None:
            return np.tile(np.asarray(default_color, dtype=np.float64), (count, 1))
        colors_np = np.asarray(colors, dtype=np.float64)
        if colors_np.ndim != 2 or colors_np.shape[0] != count:
            raise ValueError(
                f"Color array must have shape ({count}, 3) or ({count}, 4); "
                f"got {colors_np.shape}"
            )
        if colors_np.shape[1] not in (3, 4):
            raise ValueError(
                f"Color array must have 3 or 4 channels; got {colors_np.shape[1]}"
            )
        if colors_np.shape[1] == 4:
            colors_np = colors_np[:, :3]
        return colors_np

    def _prepare_static_spheres(
        self,
        static_spheres_positions: Array | None,
        static_spheres_radii: Array | None,
        static_spheres_colors: Array | None,
        default_color: tuple[float, float, float] = (0.8, 0.2, 0.2),
    ) -> SphereSet | None:
        """Validate static sphere inputs and return a SphereSet."""
        if static_spheres_positions is None:
            return None

        centers = np.asarray(static_spheres_positions, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(
                f"static_spheres_positions must have shape (N, 3); got shape {centers.shape}"
            )
        N = centers.shape[0]

        if static_spheres_radii is None:
            raise ValueError(
                "static_spheres_radii is required when static_spheres_positions is set"
            )
        radii = np.asarray(static_spheres_radii, dtype=np.float64).reshape(-1)
        if radii.shape[0] != N:
            raise ValueError(
                f"static_spheres_radii must have length {N}; got length {radii.shape[0]}"
            )

        colors = self._normalize_color_array(static_spheres_colors, N, default_color)
        return SphereSet(centers=centers, radii=radii, colors=colors)

    def _prepare_dynamic_spheres(
        self,
        dynamic_spheres_positions: Array | None,
        dynamic_spheres_radii: Array | None,
        dynamic_spheres_colors: Array | None,
        expected_T: int | None,
        default_color: tuple[float, float, float] = (0.2, 0.2, 0.8),
    ) -> DynamicSpheres | None:
        """Validate dynamic sphere inputs and return a DynamicSpheres struct."""
        if dynamic_spheres_positions is None:
            return None

        centers = np.asarray(dynamic_spheres_positions, dtype=np.float64)
        if centers.ndim != 3 or centers.shape[2] != 3:
            raise ValueError(
                f"dynamic_spheres_positions must have shape (N, T, 3); got shape {centers.shape}"
            )
        N, T_dyn, _ = centers.shape
        if expected_T is not None and T_dyn != expected_T:
            raise ValueError(
                f"dynamic_spheres_positions time dimension ({T_dyn}) must match trajectory length ({expected_T})"
            )

        if dynamic_spheres_radii is None:
            raise ValueError(
                "dynamic_spheres_radii is required when dynamic_spheres_positions is set"
            )
        radii = np.asarray(dynamic_spheres_radii, dtype=np.float64).reshape(-1)
        if radii.shape[0] != N:
            raise ValueError(
                f"dynamic_spheres_radii must have length {N}; got length {radii.shape[0]}"
            )

        colors = self._normalize_color_array(dynamic_spheres_colors, N, default_color)
        return DynamicSpheres(trajectories=centers, radii=radii, colors=colors)

    def _cross_sections_for_points(
        self, q: Array, s_ps: np.ndarray
    ) -> tuple[CrossSection, ...]:
        return evaluate_cross_sections(self.robot, jnp.asarray(q), s_ps)

    def _base_plate_radius_for_section(self, section: CrossSection) -> float:
        """Size the circular base plate from the actual base contour."""
        contour = section.contour(self.cross_section_resolution)
        transverse_distance = np.linalg.norm(contour[:, 1:], axis=1)
        return float(self.base_plate_radius_scale * np.max(transverse_distance))

    def _primitive_from_section(
        self, section: CrossSection
    ) -> tuple[str, np.ndarray, bool, bool]:
        marker = section.discrete_marker()
        return (
            marker.primitive,
            marker.scale_xyz,
            False,
            marker.align_with_material_frame,
        )

    def _prepare_scene_data(
        self,
        ts: Array,
        q_ts: Array,
        *,
        base_offsets: Array | None,
        color_config: RendererColorConfig | None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
        static_spheres_positions: Array | None,
        static_spheres_radii: Array | None,
        static_spheres_colors: Array | None,
        dynamic_spheres_positions: Array | None,
        dynamic_spheres_radii: Array | None,
        dynamics_spheres_colors: Array | None,
    ) -> SceneData:
        """Compute curves/actuators and validate auxiliary geometry."""
        ts_np = np.asarray(ts, dtype=np.float64).reshape(-1)
        q_ts_arr = jnp.asarray(q_ts)

        if q_ts_arr.ndim == 1:
            q_ts_arr = q_ts_arr[None, :]  # (1, DOF) -> (1, DOF)
        if q_ts_arr.ndim == 2:
            q_ts_arr = q_ts_arr[None, ...]  # (T, DOF) -> (1, T, DOF)
        if q_ts_arr.ndim != 3:
            raise ValueError(
                f"q_ts must have shape (T, DOF) or (N, T, DOF); got {q_ts_arr.shape}"
            )

        N, T, _ = q_ts_arr.shape
        q_ts_np = np.asarray(q_ts_arr, dtype=np.float64)
        if ts_np.shape[0] not in (1, T):
            raise ValueError(
                f"ts length ({ts_np.shape[0]}) must be 1 or match q_ts time dimension ({T})"
            )
        if ts_np.shape[0] == 1 and T > 1:
            # Auto-generate monotonic timestamps if only a single origin is provided
            ts_np = np.arange(T, dtype=np.float64)

        # Compute base offsets
        uses_configured_offsets = (
            base_offsets is None and self._base_offsets is not None
        )
        offsets = base_offsets if base_offsets is not None else self._base_offsets
        if offsets is None:
            offsets = self._compute_grid_offsets(int(N), self.grid_spacing)
        offsets = self._normalize_base_offsets(
            offsets,
            num_robots=int(N),
            target_dim=3,
            allow_extra_rows=uses_configured_offsets,
        )

        # Backbone curves and material frames, time-first then robot-first.
        q_ts_time_first = q_ts_arr.transpose(1, 0, 2)

        def _compute_geometry_for_timestep(
            q_batch: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return self.compute_backbone_curves_and_frames_batched(q_batch, offsets)

        all_curves_time_first, all_frames_time_first = jax.vmap(
            _compute_geometry_for_timestep
        )(q_ts_time_first)
        curves = np.array(all_curves_time_first.transpose(1, 0, 2, 3), dtype=np.float64)
        material_frames = np.array(
            all_frames_time_first.transpose(1, 0, 2, 3, 4), dtype=np.float64
        )

        layout = self._compute_segment_layout(curves.shape[2])
        resolved_colors = self.resolve_backbone_colors(
            int(N), color_config=color_config
        )

        actuator_layers = ()
        if render_actuators and self._has_actuator_visuals:
            actuator_layers = self.compute_actuator_visual_layers_trajectory(
                q_ts_arr,
                offsets,
                actuator_inputs=actuator_inputs,
            )

        static_spheres = self._prepare_static_spheres(
            static_spheres_positions, static_spheres_radii, static_spheres_colors
        )
        dynamic_spheres = self._prepare_dynamic_spheres(
            dynamic_spheres_positions,
            dynamic_spheres_radii,
            dynamics_spheres_colors,
            expected_T=T,
        )

        return SceneData(
            curves=curves,
            material_frames=material_frames,
            q_ts=q_ts_np,
            ts=ts_np,
            layout=layout,
            segment_colors_rgba=resolved_colors.per_robot_segment_rgba,
            actuator_layers=actuator_layers,
            static_spheres=static_spheres,
            dynamic_spheres=dynamic_spheres,
        )

    # -------------------------------------------------------------------------
    # Offscreen rendering / shared geometry for rendering backend
    # -------------------------------------------------------------------------

    def _populate_rendering_scene(
        self,
        scene,
        scene_data: SceneData,
        frame_idx: int,
        material_cache: dict[tuple[float, float, float], object] | None = None,
        color_config: RendererColorConfig | None = None,
    ) -> None:
        """Add all geometries for a specific frame to an Open3DScene."""
        if material_cache is None:
            material_cache = {}
        cfg = color_config or self.color_config

        def mat_for(color_rgba: np.ndarray | tuple[float, ...]):
            key = tuple(np.asarray(color_rgba, dtype=np.float64).reshape(-1))
            if key not in material_cache:
                material_cache[key] = self._make_mesh_material(key)
            return material_cache[key]

        geometry_names = []

        def add_geometry(name, geometry, material):
            """Register geometry and track its name for shadow configuration.

            Args:
                name: Unique geometry name.
                geometry: Open3D mesh or line set.
                material: Rendering material.

            Returns:
                None.
            """
            scene.add_geometry(name, geometry, material)
            geometry_names.append(name)

        scene.clear_geometry()
        layout = scene_data.layout
        s_ps = np.asarray(self._backbone_abscissae, dtype=np.float64)

        self._appearance_center, self._appearance_extent = self._scene_bounds(
            scene_data
        )
        if self.show_ground_plane and not self.scene_config.backdrop.enabled:
            ground_cfg = self.scene_config.ground
            for i, (center, normal, size) in enumerate(
                self._resolve_ground_planes(
                    scene_data.curves[:, frame_idx],
                    scene_data.material_frames[:, frame_idx, 0, :, 0],
                )
            ):
                plane, grid = _make_ground_plane(
                    center,
                    normal,
                    size,
                    ground_cfg.color,
                    ground_cfg.grid_color,
                    grid_divisions=max(1, round(size / ground_cfg.grid_spacing))
                    if ground_cfg.grid_spacing
                    else 10,
                    ground_config=ground_cfg,
                )
                if ground_cfg.surface:
                    name = f"ground_plane_{i}"
                    plane.paint_uniform_color((1.0, 1.0, 1.0))
                    add_geometry(
                        name,
                        plane,
                        self._make_mesh_material(ground_cfg.color, ground=True),
                    )
                    scene.scene.geometry_shadows(name, False, ground_cfg.receive_shadow)
                if ground_cfg.grid:
                    material = o3d.visualization.rendering.MaterialRecord()
                    material.shader = "unlitLine"
                    material.line_width = 1.0
                    grid.colors = o3d.utility.Vector3dVector(
                        srgb_to_linear(np.asarray(grid.colors))
                    )
                    add_geometry(f"ground_grid_{i}", grid, material)

        for robot_idx in range(scene_data.num_robots):
            curve = scene_data.curves[robot_idx, frame_idx]
            material_frames = scene_data.material_frames[robot_idx, frame_idx]
            q_frame = scene_data.q_ts[robot_idx, frame_idx]
            sections = self._cross_sections_for_points(q_frame, s_ps)
            base_color_rgba = ensure_rgba(np.asarray(cfg.base_plate_color))[0]
            base_axis = material_frames[0, :, 0]
            base_mesh = _make_base_plate(
                curve[0] - 0.5 * self.base_plate_thickness * base_axis,
                radius=self._base_plate_radius_for_section(sections[0]),
                thickness=self.base_plate_thickness,
                color=tuple(base_color_rgba[:3]),
                apply_color=False,
                apply_translation=True,
                normal_xyz=base_axis,
            )
            add_geometry(f"base_{robot_idx}", base_mesh, mat_for(base_color_rgba))

            for s in range(layout.segments):
                c0, c1 = int(layout.starts[s]), int(layout.ends[s])
                raw_color_rgba = scene_data.segment_colors_rgba[robot_idx, s]
                raw_color_rgb = tuple(np.asarray(raw_color_rgba).reshape(-1)[:3])
                if self._backbone_mode == "swept" and c1 - c0 >= 1:
                    for p in range(c0, c1 - 1):
                        body = _make_swept_cross_section_segment(
                            curve[p],
                            curve[p + 1],
                            material_frames[p],
                            material_frames[p + 1],
                            sections[p],
                            sections[p + 1],
                            raw_color_rgb,
                            self.cross_section_resolution,
                            apply_color=False,
                            cap_start=self._swept_edge_caps[p][0],
                            cap_end=self._swept_edge_caps[p][1],
                        )
                        add_geometry(
                            f"body_{robot_idx}_{s}_{p}",
                            body,
                            mat_for(raw_color_rgba),
                        )
                else:
                    for p in range(c0, c1):
                        section = sections[p]
                        marker = section.discrete_marker()
                        if marker.primitive == "sphere":
                            sp = _make_sphere(
                                curve[p],
                                radius=float(marker.scale_xyz[0]),
                                color=raw_color_rgb,
                                resolution=self.sphere_resolution,
                                apply_color=False,
                                apply_translation=True,
                            )
                            add_geometry(
                                f"sphere_{robot_idx}_{s}_{p}",
                                sp,
                                mat_for(raw_color_rgba),
                            )
                        elif marker.primitive == "box":
                            width, height, depth = marker.scale_xyz
                            box = _make_box_centered(
                                curve[p],
                                width=float(width),
                                height=float(height),
                                depth=float(depth),
                                color=raw_color_rgb,
                                apply_color=False,
                                apply_translation=True,
                            )
                            box.rotate(
                                _primitive_rotation_from_material_frame(
                                    material_frames[p]
                                ),
                                center=curve[p],
                            )
                            add_geometry(
                                f"box_{robot_idx}_{s}_{p}",
                                box,
                                mat_for(raw_color_rgba),
                            )
                        else:
                            ell = _make_ellipsoid(
                                curve[p],
                                radii=tuple(marker.scale_xyz),
                                color=raw_color_rgb,
                                resolution=self.sphere_resolution,
                                apply_color=False,
                                apply_translation=True,
                            )
                            ell.rotate(
                                _primitive_rotation_from_material_frame(
                                    material_frames[p]
                                ),
                                center=curve[p],
                            )
                            add_geometry(
                                f"ell_{robot_idx}_{s}_{p}",
                                ell,
                                mat_for(raw_color_rgba),
                            )

            for layer_idx, layer in enumerate(scene_data.actuator_layers):
                colors = resolve_actuator_rgba(
                    layer,
                    override_color=cfg.robot_override,
                    default_color=cfg.actuators.color_for_kind(layer.kind),
                    scalar_colormap=cfg.actuators.scalar_colormap,
                )
                robot_actuators = np.asarray(layer.points)[robot_idx, frame_idx]
                for actuator_idx in range(robot_actuators.shape[0]):
                    color = tuple(colors[robot_idx, frame_idx, actuator_idx, :3])
                    ls = _make_polyline_lineset(
                        robot_actuators[actuator_idx], color=color
                    )
                    mat_line = o3d.visualization.MaterialRecord()
                    mat_line.shader = "unlitLine"
                    mat_line.line_width = layer.line_width or self.actuator_line_width
                    add_geometry(
                        f"actuator_{robot_idx}_{layer_idx}_{actuator_idx}",
                        ls,
                        mat_line,
                    )

        if scene_data.static_spheres is not None:
            static_set = scene_data.static_spheres
            for idx, (ctr, rad, col) in enumerate(
                zip(static_set.centers, static_set.radii, static_set.colors)
            ):
                rgb = tuple(np.asarray(col, dtype=np.float64).reshape(-1)[:3])
                mesh = _make_sphere(
                    ctr,
                    float(rad),
                    rgb,
                    self.sphere_resolution,
                    apply_color=False,
                    apply_translation=True,
                )
                add_geometry(f"static_{idx}", mesh, mat_for(rgb))

        if scene_data.dynamic_spheres is not None:
            dyn_set = scene_data.dynamic_spheres
            for dyn_idx, (traj, rad, col) in enumerate(
                zip(dyn_set.trajectories, dyn_set.radii, dyn_set.colors)
            ):
                j_idx = min(frame_idx, traj.shape[0] - 1)
                rgb = tuple(np.asarray(col, dtype=np.float64).reshape(-1)[:3])
                mesh = _make_sphere(
                    traj[j_idx],
                    float(rad),
                    rgb,
                    max(12, self.sphere_resolution // 2),
                    apply_color=False,
                    apply_translation=True,
                )
                add_geometry(f"dynamic_{dyn_idx}", mesh, mat_for(rgb))

        for name in geometry_names:
            if name.startswith("ground"):
                continue
            cast = (
                self.scene_config.sphere_cast_shadow
                if name.startswith(("static_", "dynamic_"))
                else self.scene_config.backbone_cast_shadow
            )
            scene.scene.geometry_shadows(
                name, self.scene_config.shadows and cast, self.scene_config.shadows
            )

    # -------------------------------------------------------------------------
    def _scene_bounds(self, scene_data: SceneData) -> tuple[np.ndarray, float]:
        """Compute framing bounds from robot trajectories and helper spheres.

        Ground planes and studio scenery do not contribute to camera framing.
        Padding covers the base plates and the cross-sections evaluated at the
        first robot configuration. Sphere trajectories contribute their complete
        motion and radii.

        Args:
            scene_data: Prepared robot trajectories and optional helper spheres.

        Returns:
            Tuple containing the world-space bounding-box center, shape ``(3,)``,
            and its largest side length in metres, with a minimum of ``1e-3``.
        """
        points = scene_data.curves.reshape(-1, 3)
        sections = self._cross_sections_for_points(
            scene_data.q_ts[0, 0], np.asarray(self._backbone_abscissae)
        )
        padding = max(
            self.base_plate_thickness,
            max(float(np.max(section.dimensions)) for section in sections)
            * max(1.0, self.base_plate_radius_scale),
        )
        lower, upper = points.min(axis=0) - padding, points.max(axis=0) + padding
        for spheres in (scene_data.static_spheres, scene_data.dynamic_spheres):
            if spheres is None:
                continue
            centers = (
                spheres.centers
                if isinstance(spheres, SphereSet)
                else spheres.trajectories.reshape(-1, 3)
            )
            if centers.size:
                radius = float(np.max(spheres.radii))
                lower = np.minimum(lower, centers.min(axis=0) - radius)
                upper = np.maximum(upper, centers.max(axis=0) + radius)
        return (lower + upper) / 2, max(float(np.max(upper - lower)), 1e-3)

    def _studio_backdrop(self, scene_data: SceneData):
        """Build the configured curved studio floor and wall.

        Args:
            scene_data: Complete trajectory and helper geometry used for sizing.

        Returns:
            Open3D triangle mesh with smooth normals and the ground color.
        """
        center, extent = self._scene_bounds(scene_data)
        vertices, faces = backdrop_mesh(
            self.scene_config, center, extent, self._world_up()
        )
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.scene_config.ground.color)
        return mesh

    def _configure_modern_scene(
        self, scene, scene_data: SceneData, camera_config=None
    ) -> None:
        """Configure physical illumination and output processing for a modern scene.

        Args:
            scene: Open3DScene owned by a GUI widget or offscreen renderer.
            scene_data: Complete trajectories and helpers used to fit scenery.
            camera_config: Per-call camera and exposure override.

        Returns:
            None. Modifies scene appearance without moving its camera.
        """
        cfg = self.scene_config
        camera = camera_config or self.config.camera
        gain = 2.0 ** (15.0 - camera.exposure_ev100)
        grading = o3d.visualization.rendering.ColorGrading
        # This enum is added by the patch that restores mapper selection.
        tone_mapping_supported = hasattr(grading.ToneMapping, "PBR_NEUTRAL")
        features = []
        if cfg.material.shading != "unlit" and not tone_mapping_supported:
            features.append(
                "tone-mapper selection is not verified for this Open3D build; "
                "update the source build if neutral highlights appear warm"
            )
        if sum(isinstance(light, DirectionalLightConfig) for light in cfg.lights) > 1:
            features.append(
                "Filament uses only the dominant directional light; use point lights for fill"
            )
        if camera.exposure_ev100 != 15:
            features.append("exposure through illumination scaling")
        if cfg.ambient.color != (1.0, 1.0, 1.0):
            features.append("environment tint")
        if cfg.material.shading.startswith("toon"):
            features.append("toon material as standard shading")
        if cfg.material.wireframe or cfg.material.flat_shading:
            features.append("wireframe/face-normal material settings")
        self._warn_appearance("modern", features)
        self._appearance_center, self._appearance_extent = self._scene_bounds(
            scene_data
        )
        scene.set_background(np.array([*cfg.background, 1.0]))
        scene.show_skybox(False)
        scene.set_lighting(scene.SOFT_SHADOWS, (0, 0, -1))
        scene.scene.enable_sun_light(False)
        scene.scene.set_indirect_light_intensity(cfg.ambient.strength * 60000 * gain)
        scene.view.set_post_processing(cfg.material.shading != "unlit")
        scene.view.set_antialiasing(True)
        scene.view.set_ambient_occlusion(cfg.ambient_occlusion)
        scene.view.set_shadowing(
            cfg.shadows, o3d.visualization.rendering.View.ShadowType.VSM
        )
        # Filmic keeps greys neutral and preserves readable midtones, without
        # recoloring materials or disabling shadows and ambient occlusion.
        algorithm = {
            "backend-default": (
                grading.ToneMapping.FILMIC if tone_mapping_supported else None
            ),
            "linear": grading.ToneMapping.LINEAR,
            "aces": grading.ToneMapping.ACES,
        }[cfg.tone_mapping]
        if algorithm is not None:
            scene.view.set_color_grading(grading(grading.Quality.ULTRA, algorithm))
        for index, light in enumerate(cfg.lights):
            name = f"configured_light_{index}"
            color = srgb_to_linear(light.color)
            cast = cfg.shadows and light.cast_shadow
            if isinstance(light, DirectionalLightConfig):
                scene.scene.add_directional_light(
                    name, color, light.direction, light.illuminance_lux * gain, cast
                )
            else:
                scene.scene.add_point_light(
                    name,
                    color,
                    light.position,
                    light.intensity_candela * 4 * np.pi * gain,
                    light.range_m,
                    cast,
                )

    @staticmethod
    def _require_modern_capture_support() -> None:
        """Reject macOS builds known to abort during modern image capture.

        Returns:
            None when modern rendering and snapshot capture can be initialized.

        Raises:
            RuntimeError: The macOS Open3D build lacks the Metal readback fix.
                The local build suffix identifies the validated patched build.
        """
        if sys.platform == "darwin" and not o3d.__version__.endswith(
            (".soromox1", ".soromox2")
        ):
            raise RuntimeError(
                "Modern Open3D rendering on macOS requires the source build "
                "with the Metal readback fix. "
                "The unpatched development wheel can abort during capture. "
                "See tools/open3d/README.md and run uv sync --extra rendering."
            )

    @contextmanager
    def _modern_session(
        self, scene_data: SceneData, camera_config: CameraConfig | None
    ):
        """Create a modern offscreen context with a fixed export camera.

        Camera fitting considers the complete trajectory and helper spheres.
        Explicit camera coordinates override automatic placement. The context
        owns the native renderer for the duration of the export.

        Args:
            scene_data: Prepared geometry and trajectories for camera fitting.
            camera_config: Camera placement and field of view, or ``None`` to
                use ``CameraConfig`` defaults.

        Yields:
            Open3D ``OffscreenRenderer`` configured for ``width`` by ``height``
            RGB capture. Frame geometry is populated separately.

        Raises:
            RuntimeError: The macOS installation lacks the required readback fix.
        """
        self._require_modern_capture_support()
        render = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        try:
            self._configure_modern_scene(render.scene, scene_data, camera_config)
            center, extent = self._scene_bounds(scene_data)
            config = camera_config or self.config.camera
            position, target = config.compute_auto_position(
                center, extent, reference_transform=np.asarray(self.base_transform)
            )
            # The scenery must fit the clipping volume but must not control camera fitting.
            distance = float(np.linalg.norm(position - target))
            render.setup_camera(
                config.fov,
                target,
                position,
                config.compute_up(),
                max(1e-5, extent * 1e-3),
                distance + 20 * extent,
            )
            yield render
        finally:
            del render

    def _populate_modern_scene(self, scene, scene_data, frame_idx, color_config=None):
        """Populate one frame's modern geometry and studio backdrop.

        Args:
            scene: Modern Open3D scene to receive the geometry.
            scene_data: Prepared robot and helper trajectories.
            frame_idx: Zero-based trajectory frame to display.
            color_config: Color override, or ``None`` for renderer defaults.

        Returns:
            None. Replaces scene geometry while preserving lights and camera.
        """
        self._populate_rendering_scene(
            scene, scene_data, frame_idx, color_config=color_config
        )
        if self.scene_config.backdrop.enabled and self.scene_config.ground.visible:
            material = self._make_mesh_material(
                self.scene_config.ground.color, ground=True
            )
            material.base_roughness = 1.0
            scene.add_geometry(
                "studio_backdrop",
                self._studio_backdrop(scene_data).paint_uniform_color((1.0, 1.0, 1.0)),
                material,
            )
            scene.scene.geometry_shadows(
                "studio_backdrop", False, self.scene_config.ground.receive_shadow
            )

    def _render_modern_frame(self, render, scene_data, frame_idx, color_config=None):
        """Render one prepared frame and copy its RGB pixels.

        Args:
            render: Configured Open3D offscreen renderer.
            scene_data: Prepared robot and helper trajectories.
            frame_idx: Zero-based frame to render.
            color_config: Color override, or ``None`` for renderer defaults.

        Returns:
            Independent ``uint8`` RGB array of shape ``(height, width, 3)``.

        Raises:
            RuntimeError: Open3D returns an unexpected image shape.
        """
        self._populate_modern_scene(render.scene, scene_data, frame_idx, color_config)
        if self.scene_config.backdrop.enabled and self.scene_config.ground.visible:
            render.scene.scene.geometry_shadows(
                "studio_backdrop", False, self.scene_config.ground.receive_shadow
            )
        # Copy before the native image/context is released.
        frame = np.asarray(render.render_to_image())
        if (
            frame.ndim != 3
            or frame.shape[:2] != (self.height, self.width)
            or frame.shape[2] not in (3, 4)
        ):
            raise RuntimeError(f"Open3D returned an invalid image shape: {frame.shape}")
        frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        return frame.copy()

    def _export_sequence(
        self, scene_data, record_cfg, playback_speed, camera_config, color_config
    ):
        """Export selected trajectory frames as a video or PNG sequence.

        Frames are rendered in ascending index order using one graphics context
        and a fixed camera. Video FPS is ``playback_speed / (median_dt * every_n)``;
        a single-frame trajectory uses a nominal 30 Hz interval. Nonuniform
        timestamps produce a warning because constant-FPS video approximates
        their timing. Video output requires FFmpeg.

        Args:
            scene_data: Prepared trajectories with finite, increasing timestamps.
            record_cfg: Output path, image prefix, frame stride and encoder options.
                Video extensions select encoding; other paths are PNG directories.
            playback_speed: Positive multiplier for the output video's frame rate.
            camera_config: Export camera configuration, or ``None`` for defaults.
            color_config: Color override, or ``None`` for the renderer's colors.

        Returns:
            None. Creates the output directory or video and closes the encoder
            before returning, including when frame rendering fails.

        Raises:
            ValueError: Timestamps, playback speed or frame stride are invalid.
            RuntimeError: Native capture, PNG writing or video finalization fails.
            FileNotFoundError: The FFmpeg executable is unavailable for video export.
        """
        ts = np.asarray(scene_data.ts)
        if not len(ts) or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
            raise ValueError("Export timestamps must be finite and strictly increasing")
        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("playback_speed must be finite and positive")
        if (
            isinstance(record_cfg.every_n, bool)
            or not isinstance(record_cfg.every_n, (int, np.integer))
            or record_cfg.every_n < 1
        ):
            raise ValueError("record_every_n must be a positive integer")
        path = Path(record_cfg.path)
        indices = range(0, scene_data.num_frames, record_cfg.every_n)
        video = path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30
        if (
            video
            and len(ts) > 2
            and not np.allclose(np.diff(ts), dt, rtol=1e-4, atol=1e-9)
        ):
            warnings.warn(
                "Open3D video uses a constant frame rate derived from the median timestamp interval; resample nonuniform trajectories for exact timing.",
                UserWarning,
                stacklevel=3,
            )
        writer = None
        try:
            with self._modern_session(scene_data, camera_config) as render:
                if video:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = FFmpegVideoWriter(
                        str(path),
                        self.width,
                        self.height,
                        playback_speed / (dt * record_cfg.every_n),
                        input_pix_fmt="rgb24",
                        video_config=record_cfg.video_config,
                    )
                else:
                    path.mkdir(parents=True, exist_ok=True)
                for index in indices:
                    frame = self._render_modern_frame(
                        render, scene_data, index, color_config
                    )
                    if writer is not None:
                        writer.write(frame)
                    else:
                        filename = path / f"{record_cfg.prefix}{index:05d}.png"
                        if not o3d.io.write_image(
                            str(filename), o3d.geometry.Image(frame)
                        ):
                            raise RuntimeError(f"Could not save {filename}")
        finally:
            if writer is not None:
                writer.close()
        if writer is not None and writer.proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to finish {path}: {writer.stderr_log}")

    # Public API
    # -------------------------------------------------------------------------

    def render_frame(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        dynamic_spheres_positions: Array | None = None,
        dynamic_spheres_radii: Array | None = None,
        dynamics_spheres_colors: Array | None = None,
    ) -> np.ndarray:
        """Render a single configuration with the modern renderer and return RGB.

        Args:
            q: Robot configuration of shape (DOF,) or batched (N, DOF).
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            color_config: Optional shared renderer color configuration.
            camera_config: Camera configuration (fov, position, look_at, etc.)
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            static_spheres_positions: Optional static sphere centers, shape (M, 3).
            static_spheres_radii: Optional static sphere radii, length M.
            static_spheres_colors: Optional static sphere colors, shape (M, 3/4).
            dynamic_spheres_positions: Optional dynamic sphere trajectories, shape (K, T, 3).
            dynamic_spheres_radii: Optional dynamic sphere radii, length K.
            dynamics_spheres_colors: Optional dynamic sphere colors, shape (K, 3/4).

        Returns:
            img (np.ndarray): Rendered RGB image of shape (height, width, 3), dtype uint8.
        """
        scene_data = self._prepare_scene_data(
            ts=np.array([0.0], dtype=np.float64),
            q_ts=jnp.asarray(q)[:, None, :]
            if jnp.asarray(q).ndim == 2
            else jnp.asarray(q),
            base_offsets=base_offsets,
            color_config=color_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )

        with self._modern_session(scene_data, camera_config) as render:
            return self._render_modern_frame(render, scene_data, 0, color_config)

    def show(
        self,
        q: Array,
        *,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        camera_config: CameraConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        dynamic_spheres_positions: Array | None = None,
        dynamic_spheres_radii: Array | None = None,
        dynamics_spheres_colors: Array | None = None,
    ) -> None:
        """Display a fixed configuration with modern materials, lighting and shadows.

        Mouse controls orbit, pan and zoom. R resets the camera; C/L save and
        restore it; S saves a modern snapshot; V prints the view; Q/Esc closes.

        Args:
            q: Robot configuration of shape (DOF,) or batched (N, DOF).
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            color_config: Optional shared renderer color configuration.
            camera_config: Camera configuration (fov, position, look_at, etc.)
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            static_spheres_positions: Optional static sphere centers, shape (M, 3).
            static_spheres_radii: Optional static sphere radii, length M.
            static_spheres_colors: Optional static sphere colors, shape (M, 3/4).
            dynamic_spheres_positions: Optional dynamic sphere trajectories, shape (K, T, 3).
            dynamic_spheres_radii: Optional dynamic sphere radii, length K.
            dynamics_spheres_colors: Optional dynamic sphere colors, shape (K, 3/4).

        Returns:
            None
        """
        ts = np.array([0.0], dtype=np.float64)
        q_ts = np.asarray(q)
        if q_ts.ndim == 2:
            q_ts = q_ts[:, None, :]
        scene_data = self._prepare_scene_data(
            ts=ts,
            q_ts=q_ts,
            base_offsets=base_offsets,
            color_config=color_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )
        self._run_modern_viewer(scene_data, camera_config, color_config)

    def _create_modern_window(self, scene_data, camera_config, color_config):
        """Create a modern window for one fixed robot configuration.

        Geometry, materials, lights and backdrop are shared with image/video
        exports. Mouse interaction changes the view. Keyboard callbacks support
        camera reset/save/restore, RGB snapshots, camera reporting and closure.
        Resizing updates the camera aspect ratio without resetting its pose.

        Args:
            scene_data: Prepared scene; frame zero is displayed for every robot.
            camera_config: Initial camera configuration, or ``None`` for defaults.
            color_config: Color override, or ``None`` for the renderer's colors.

        Returns:
            Tuple of the GUI application, window and scene widget. The caller
            runs the event loop and owns window cleanup. Call on the main thread.
        """
        self._require_modern_capture_support()
        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()
        window = app.create_window("Robot (Open3D)", self.width, self.height)
        widget = gui.SceneWidget()
        widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        widget.frame = window.content_rect
        window.add_child(widget)
        self._configure_modern_scene(widget.scene, scene_data, camera_config)
        self._populate_modern_scene(widget.scene, scene_data, 0, color_config)
        center, extent = self._scene_bounds(scene_data)
        config = camera_config or self.config.camera
        eye, target = config.compute_auto_position(
            center, extent, reference_transform=np.asarray(self.base_transform)
        )
        widget.setup_camera(
            config.fov,
            o3d.geometry.AxisAlignedBoundingBox(
                center - extent / 2, center + extent / 2
            ),
            target,
        )
        widget.look_at(target, eye, config.compute_up())
        widget.scene.camera.set_projection(
            config.fov,
            self.width / self.height,
            max(1e-5, extent * 1e-3),
            float(np.linalg.norm(eye - target)) + 20 * extent,
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )

        def capture_camera():
            """Read the camera pose, orbit pivot and projection for restoration.

            Returns:
                Dictionary of copied world-space vectors and projection scalars.
            """
            camera = widget.scene.camera
            model = np.asarray(camera.get_model_matrix()).copy()
            return {
                "eye": model[:3, 3],
                "target": model[:3, 3] - model[:3, 2],
                "up": model[:3, 1],
                "pivot": np.asarray(widget.center_of_rotation).copy(),
                "fov": camera.get_field_of_view(),
                "near": camera.get_near(),
                "far": camera.get_far(),
            }

        def restore_camera(state):
            """Restore a captured view at the widget's current aspect ratio.

            Args:
                state: Camera dictionary returned by ``capture_camera``.

            Returns:
                None. Updates camera, orbit pivot and redraw state.
            """
            widget.look_at(state["target"], state["eye"], state["up"])
            widget.center_of_rotation = state["pivot"]
            widget.scene.camera.set_projection(
                state["fov"],
                max(widget.frame.width, 1) / max(widget.frame.height, 1),
                state["near"],
                state["far"],
                o3d.visualization.rendering.Camera.FovType.Vertical,
            )
            widget.force_redraw()

        initial = capture_camera()
        saved = [initial]

        def layout(_):
            """Fit the scene widget to the resized window's content area.

            Args:
                _: GUI layout context supplied by Open3D.

            Returns:
                None. Preserves the current camera pose during resizing.
            """
            state = capture_camera()
            widget.frame = window.content_rect
            restore_camera(state)

        def save_snapshot(image):
            """Save the modern capture callback's image to the current directory.

            Args:
                image: RGB Open3D image delivered by the rendering callback.

            Returns:
                None. Writes ``frame_00000.png`` or warns if writing fails.
            """
            filename = "frame_00000.png"
            if not o3d.io.write_image(filename, image):
                warnings.warn(f"Could not save {filename}", UserWarning, stacklevel=2)
            else:
                print(f"[Open3D] Saved {filename}")

        def on_key(event):
            """Handle camera, snapshot and close shortcuts for a static view.

            Args:
                event: Open3D key event; only key-down events trigger actions.

            Returns:
                ``True`` for a handled shortcut, otherwise ``False``.
            """
            if event.type != gui.KeyEvent.Type.DOWN:
                return False
            if event.key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
                window.close()
            elif event.key == gui.KeyName.R:
                restore_camera(initial)
            elif event.key == gui.KeyName.C:
                saved[0] = capture_camera()
            elif event.key == gui.KeyName.L:
                restore_camera(saved[0])
            elif event.key == gui.KeyName.S:
                widget.scene.scene.render_to_image(save_snapshot)
            elif event.key == gui.KeyName.V:
                print(f"[Open3D] Camera: {capture_camera()}")
            else:
                return False
            return True

        window.set_on_layout(layout)
        window.set_on_key(on_key)
        return app, window, widget

    def _run_modern_viewer(self, scene_data, camera_config, color_config):
        """Display a fixed modern scene until its window closes.

        Uses Open3D's main-thread event loop while preserving the shared graphics
        engine for subsequent ``show()`` or export calls in the same process.

        Args:
            scene_data: Prepared scene whose frame zero is displayed.
            camera_config: Initial camera settings, or ``None`` for defaults.
            color_config: Color override, or ``None`` for the renderer's colors.

        Returns:
            None. Blocks until closure; Open3D releases the window and its scene.
        """
        global _MODERN_GUI_CREATED
        app, window, widget = self._create_modern_window(
            scene_data, camera_config, color_config
        )
        _MODERN_GUI_CREATED = True
        closed = [False]

        def on_close():
            """Signal that the viewer's event loop should stop.

            Returns:
                ``True`` to allow Open3D to close the window.
            """
            closed[0] = True
            return True

        window.set_on_close(on_close)
        print(
            "Open3D: drag to orbit; R=ResetCam C=CaptureCam L=LoadCam S=Snapshot V=PrintCam Q/Esc=Quit"
        )
        try:
            # run_one_tick keeps the graphics engine usable for later exports or
            # another show() call in the same Python process.
            while not closed[0] and app.run_one_tick():
                pass
        finally:
            # Open3D destroys the window and its children when it closes. The
            # Python handles must not be accessed after that native destruction.
            if not closed[0]:
                window.close()

    def render_sequence(  # type: ignore[override]
        self,
        ts: Array,
        q_ts: Array,
        *,
        playback_speed: float = 1.0,
        autoplay: bool = True,
        loop: bool = False,
        record_path: str | None = None,
        record_every_n: int = 1,
        record_prefix: str = "frame_",
        video_config: VideoEncodingConfig | None = None,
        close_when_recording_done: bool = False,
        camera_config: CameraConfig | None = None,
        base_offsets: Array | None = None,
        color_config: RendererColorConfig | None = None,
        render_actuators: bool = True,
        actuator_inputs: Array | None = None,
        static_spheres_positions: Array | None = None,
        static_spheres_radii: Array | None = None,
        static_spheres_colors: Array | None = None,
        dynamic_spheres_positions: Array | None = None,
        dynamic_spheres_radii: Array | None = None,
        dynamics_spheres_colors: Array | None = None,
        window_name: str = "Robot Animation (Open3D)",
    ) -> None:
        """Preview a trajectory, or export it with the modern renderer.

        With ``record_path``, export all selected frames synchronously and return.
        Without it, open the legacy interactive preview. ``autoplay``, ``loop``,
        ``window_name`` and ``close_when_recording_done`` affect no exported frames;
        the latter is retained for existing callers, since export always returns.
        Video uses constant FPS derived from median timestamp spacing and
        ``playback_speed``. ``record_every_n`` subsamples images and video, reducing
        video FPS by the same factor to preserve playback speed.


        Args:
            ts: Time stamps of shape (T,).
            q_ts: Configurations of shape (T, DOF) or batched (N, T, DOF).
            playback_speed: Playback speed multiplier (>0).
            autoplay: Start playback immediately. If False, wait for space bar to play.
            loop: Whether to loop the animation when it reaches the end.
            record_path: Optional path to save frames or video (extension determines mode).
            record_every_n: Export every n-th frame for both images and video.
            record_prefix: Filename prefix for recorded frames.
            video_config: Optional ffmpeg encoding configuration for video output.
            close_when_recording_done: Retained for compatibility; export always returns.
            camera_config: Camera configuration (fov, position, look_at, etc.).
                Note: For interactive viewing, user can adjust camera with mouse.
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            color_config: Optional shared renderer color configuration.
            render_actuators: Whether to render actuator visual layers if available.
            actuator_inputs: Optional actuator inputs for scalar-colored layers.
            static_spheres_positions: Optional static sphere centers, shape (M, 3).
            static_spheres_radii: Optional static sphere radii, length M.
            static_spheres_colors: Optional static sphere colors, shape (M, 3/4).
            dynamic_spheres_positions: Optional dynamic sphere trajectories, shape (K, T, 3).
            dynamic_spheres_radii: Optional dynamic sphere radii, length K.
            dynamics_spheres_colors: Optional dynamic sphere colors, shape (K, 3/4).
            window_name: Title for the viewer window.

        Returns:
            None
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install the rendering dependencies; see tools/open3d/README.md"
            )

        scene_data = self._prepare_scene_data(
            ts=ts,
            q_ts=q_ts,
            base_offsets=base_offsets,
            color_config=color_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )
        record_cfg = RecordingConfig(
            path=record_path,
            prefix=record_prefix,
            every_n=record_every_n,
            video_config=video_config or self.config.output.video,
            close_when_done=close_when_recording_done,
        )
        if record_path is not None:
            self._export_sequence(
                scene_data, record_cfg, playback_speed, camera_config, color_config
            )
            return
        self._warn_appearance(
            "animated",
            [
                "legacy shading for efficient mesh updates: materials, lighting, transparency, shadows and ambient occlusion differ from modern image/video exports; snapshots capture this preview"
            ],
        )
        self._run_viewer(
            scene_data,
            playback_speed=playback_speed,
            autoplay=autoplay,
            loop=loop,
            record_cfg=record_cfg,
            window_name=window_name,
            camera_config=camera_config,
            color_config=color_config,
        )

    def _create_visualizer(self, window_name: str):
        """Create a visualizer with common options set."""
        if sys.platform == "darwin" and _MODERN_GUI_CREATED:
            raise RuntimeError(
                "This Open3D build cannot open a legacy OpenGL preview after a modern "
                "GUI window in the same macOS process. Start the animated preview in "
                "a fresh Python process; modern image/video exports remain available."
            )
        vis = o3d.visualization.VisualizerWithKeyCallback()
        window_created = vis.create_window(
            window_name=window_name,
            width=self.width,
            height=self.height,
        )
        if not window_created:
            raise RuntimeError(
                "Open3D failed to create the visualization window. Check that "
                "a usable X11/Wayland display and OpenGL context are available, "
                "or use a headless rendering path."
            )

        opt = vis.get_render_option()
        opt.background_color = np.array(
            self.scene_config.ground.color
            if self.scene_config.backdrop.enabled
            else self.background_color,
            dtype=np.float64,
        )
        opt.line_width = self.actuator_line_width
        opt.light_on = True
        # Python names the C++ SmoothShade enum member "Color".
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        return vis, vis.get_view_control()

    def _setup_interactive_camera(
        self,
        vis,
        ctrl,
        scene_data: SceneData,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Set up camera for interactive viewer using CameraConfig.

        Args:
            vis: Open3D visualizer
            ctrl: View control from visualizer
            scene_data: Scene data containing curves for bounding box computation
            camera_config: Camera configuration, or None to use defaults
        """
        # First, let Open3D compute its default view to initialize internals
        vis.reset_view_point(True)

        # Compute scene bounds from all curves (shape: N, T, P, 3)
        all_points = scene_data.curves.reshape(-1, 3)
        center = np.mean(all_points, axis=0)
        extent = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        max_extent = float(np.max(extent))

        # Use provided config or defaults
        config = camera_config or self.config.camera
        camera_pos, look_at = config.compute_auto_position(
            center,
            max_extent,
            reference_transform=np.asarray(self.base_transform),
        )
        up = config.compute_up()

        # Open3D ViewControl stores the front vector from the look-at target
        # toward the camera eye, opposite to the eye-to-target viewing ray.
        front = look_at - camera_pos
        front_norm = np.linalg.norm(front)
        if front_norm > 1e-9:
            front = front / front_norm

        # Use ViewControl API for orientation
        ctrl.set_front(-front)
        ctrl.set_lookat(look_at)
        ctrl.set_up(up)

        # Compute zoom empirically based on distance-to-extent ratio
        # Open3D's zoom doesn't scale linearly with distance, but this formula
        # provides consistent results: when camera is at ~10x scene extent, zoom≈1.0
        # The constant 0.1 was determined empirically to match typical viewing distances.
        desired_distance = float(np.linalg.norm(camera_pos - look_at))
        zoom = 0.1 * (desired_distance / max_extent) if max_extent > 1e-9 else 0.7
        ctrl.set_zoom(zoom)
        if self.scene_config.backdrop.enabled and self.scene_config.ground.visible:
            # An explicit pinhole camera avoids fitting the large studio wall.
            # ViewControl still owns interactive orbit/pan/zoom and snapshots.
            forward = (look_at - camera_pos) / desired_distance
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.stack((right, down, forward))
            parameters = o3d.camera.PinholeCameraParameters()
            focal = self.height / (2 * np.tan(np.deg2rad(config.fov) / 2))
            parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.width,
                self.height,
                focal,
                focal,
                (self.width - 1) / 2,
                (self.height - 1) / 2,
            )
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = -rotation @ camera_pos
            parameters.extrinsic = extrinsic
            ctrl.convert_from_pinhole_camera_parameters(
                parameters, allow_arbitrary=True
            )

        vis.poll_events()
        vis.update_renderer()

    def _register_key_callbacks(
        self,
        vis,
        ctrl,
        state: dict[str, int | bool | float],
        update_frame,
        save_frame_fn,
        initial_cam,
        saved_cam_holder: list,
        print_prefix: str,
    ) -> None:
        """Attach shared keyboard callbacks to a visualizer."""

        def cb_space(_):
            was_playing = bool(state["playing"])
            state["playing"] = not was_playing
            if state["playing"] and not was_playing:
                state["last_tick"] = time.time()
            return False

        def cb_next(_):
            update_frame(state["idx"] + 1)
            return False

        def cb_prev(_):
            update_frame(state["idx"] - 1)
            return False

        def cb_home(_):
            update_frame(0)
            return False

        def cb_save(_):
            save_frame_fn(state["idx"], force=True)
            return False

        def cb_reset_camera(_):
            ctrl.convert_from_pinhole_camera_parameters(initial_cam)
            vis.update_renderer()
            print(f"{print_prefix} Camera reset to initial view.")
            return False

        def cb_capture_camera(_):
            saved_cam_holder[0] = ctrl.convert_to_pinhole_camera_parameters()
            print(f"{print_prefix} Camera view captured.")
            return False

        def cb_load_camera(_):
            ctrl.convert_from_pinhole_camera_parameters(saved_cam_holder[0])
            vis.update_renderer()
            print(f"{print_prefix} Saved camera view loaded.")
            return False

        def cb_print_camera(_):
            params = ctrl.convert_to_pinhole_camera_parameters()
            K = np.asarray(params.intrinsic.intrinsic_matrix)
            E = np.asarray(params.extrinsic)
            print(f"\n{print_prefix} Camera parameters:")
            print("Intrinsic:\n", K)
            print("Extrinsic:\n", E)
            return False

        def cb_quit(_):
            state["playing"] = False
            vis.close()
            return False

        vis.register_key_callback(ord(" "), cb_space)
        vis.register_key_callback(262, cb_next)  # Right
        vis.register_key_callback(263, cb_prev)  # Left
        vis.register_key_callback(ord("H"), cb_home)
        vis.register_key_callback(ord("S"), cb_save)
        vis.register_key_callback(ord("R"), cb_reset_camera)
        vis.register_key_callback(ord("C"), cb_capture_camera)
        vis.register_key_callback(ord("L"), cb_load_camera)
        vis.register_key_callback(ord("V"), cb_print_camera)
        vis.register_key_callback(81, cb_quit)  # Q
        vis.register_key_callback(256, cb_quit)  # Esc

    def _build_robot_geometry(
        self,
        vis,
        curve0: np.ndarray,
        material_frames0: np.ndarray,
        sections: tuple[CrossSection, ...],
        layout: SegmentLayout,
        segment_colors: np.ndarray,
        base_plate_color: tuple[float, float, float],
        add_backbone_geometry: bool = True,
    ) -> tuple[object, list[list[CachedMesh]]]:
        """Add base and backbone meshes for one robot; return handles."""
        base_color = self._blend_with_background(base_plate_color)
        base_axis = material_frames0[0, :, 0]
        base_mesh = _make_base_plate(
            curve0[0] - 0.5 * self.base_plate_thickness * base_axis,
            radius=self._base_plate_radius_for_section(sections[0]),
            thickness=self.base_plate_thickness,
            color=base_color,
            normal_xyz=base_axis,
        )
        vis.add_geometry(base_mesh)

        meshes_groups: list[list[CachedMesh]] = []
        for s in range(layout.segments):
            seg_meshes: list[CachedMesh] = []
            c0, c1 = int(layout.starts[s]), int(layout.ends[s])
            raw_color_rgba = segment_colors[s]
            seg_color = self._blend_with_background(raw_color_rgba)
            if self._backbone_mode == "swept" and c1 - c0 >= 1:
                for p in range(c0, c1 - 1):
                    cap_start, cap_end = self._swept_edge_caps[p]
                    mesh = _make_swept_cross_section_segment(
                        curve0[p],
                        curve0[p + 1],
                        material_frames0[p],
                        material_frames0[p + 1],
                        sections[p],
                        sections[p + 1],
                        seg_color,
                        self.cross_section_resolution,
                        cap_start=cap_start,
                        cap_end=cap_end,
                    )
                    vertices = np.asarray(mesh.vertices).copy()
                    cached = CachedMesh(
                        mesh=mesh,
                        base_vertices=vertices,
                        base_normals=None,
                        scale_base=np.ones(3),
                        dynamic_length=False,
                        rotate_to_axis=False,
                        swept_contours=(
                            sections[p].contour(self.cross_section_resolution),
                            sections[p + 1].contour(self.cross_section_resolution),
                        ),
                        cap_start=cap_start,
                        cap_end=cap_end,
                    )
                    seg_meshes.append(cached)
                    if add_backbone_geometry:
                        vis.add_geometry(mesh)
            else:
                for p in range(c0, c1):
                    unit_key, scale_base, dynamic_length, rotate_to_axis = (
                        self._primitive_from_section(sections[p])
                    )
                    unit = self._unit_meshes[unit_key]
                    mesh = self._instantiate_mesh(unit, seg_color)
                    cached = CachedMesh(
                        mesh=mesh,
                        base_vertices=unit.vertices,
                        base_normals=unit.normals,
                        scale_base=scale_base,
                        dynamic_length=dynamic_length,
                        rotate_to_axis=rotate_to_axis,
                    )
                    rotation = _primitive_rotation_from_material_frame(
                        material_frames0[p]
                    )
                    self._apply_cached_mesh(cached, curve0[p], rotation, None)
                    seg_meshes.append(cached)
                    if add_backbone_geometry:
                        vis.add_geometry(mesh)
            meshes_groups.append(seg_meshes)
        return base_mesh, meshes_groups

    def _update_robot_geometry(
        self,
        vis,
        curve: np.ndarray,
        material_frames: np.ndarray,
        base_mesh,
        meshes_groups: list[list[CachedMesh]],
        layout: SegmentLayout,
        update_backbone_geometry: bool = True,
    ) -> None:
        """Translate base and backbone geometry to a new curve position."""
        base_axis = material_frames[0, :, 0]
        base_center = curve[0] - 0.5 * self.base_plate_thickness * base_axis
        delta = base_center - _mesh_center(base_mesh)
        base_mesh.translate(delta, relative=True)
        vis.update_geometry(base_mesh)

        for s in range(len(meshes_groups)):
            c0, c1 = int(layout.starts[s]), int(layout.ends[s])
            seg_meshes = meshes_groups[s]
            if self._backbone_mode == "swept" and c1 - c0 >= 1:
                for i_local, p in enumerate(range(c0, c1 - 1)):
                    if i_local >= len(seg_meshes):
                        break
                    cached = seg_meshes[i_local]
                    self._apply_swept_cached_mesh(
                        cached,
                        curve[p],
                        curve[p + 1],
                        material_frames[p],
                        material_frames[p + 1],
                    )
                    if update_backbone_geometry:
                        vis.update_geometry(cached.mesh)
            else:
                for i_local, p in enumerate(range(c0, min(c1, c0 + len(seg_meshes)))):
                    cached = seg_meshes[i_local]
                    rotation = _primitive_rotation_from_material_frame(
                        material_frames[p]
                    )
                    self._apply_cached_mesh(cached, curve[p], rotation, None)
                    if update_backbone_geometry:
                        vis.update_geometry(cached.mesh)

    def _build_scene(
        self,
        vis,
        scene_data: SceneData,
        frame_idx: int = 0,
        *,
        color_config: RendererColorConfig | None = None,
    ) -> SceneHandles:
        """Construct initial geometry for the viewer."""
        cfg = color_config or self.color_config
        layout = scene_data.layout
        s_ps = np.asarray(self._backbone_abscissae, dtype=np.float64)
        ground_meshes: list = []
        ground_lines: list = []
        if self.scene_config.backdrop.enabled and self.scene_config.ground.visible:
            backdrop = self._studio_backdrop(scene_data)
            vis.add_geometry(backdrop)
            ground_meshes.append(backdrop)
        base_meshes: list = []
        backbone_meshes: list[list[list[CachedMesh]]] = []
        merged_backbone_meshes: list[o3d.geometry.TriangleMesh | None] = []
        merge_backbone_meshes = self._should_merge_backbone_meshes(
            scene_data.num_robots
        )

        if not self._warned_dynamic_geometry:
            warnings.warn(
                "Open3D viewer assumes cross_section_geometry is configuration-independent; "
                "geometry is frozen to the first frame.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_dynamic_geometry = True

        self._appearance_center, self._appearance_extent = self._scene_bounds(
            scene_data
        )
        if self.show_ground_plane and not self.scene_config.backdrop.enabled:
            ground_cfg = self.scene_config.ground
            for center, normal, size in self._resolve_ground_planes(
                scene_data.curves[:, frame_idx],
                scene_data.material_frames[:, frame_idx, 0, :, 0],
            ):
                plane, grid = _make_ground_plane(
                    center,
                    normal,
                    size,
                    ground_cfg.color,
                    ground_cfg.grid_color,
                    grid_divisions=max(1, round(size / ground_cfg.grid_spacing))
                    if ground_cfg.grid_spacing
                    else 10,
                    ground_config=ground_cfg,
                )
                if ground_cfg.surface:
                    vis.add_geometry(plane)
                    ground_meshes.append(plane)
                if ground_cfg.grid:
                    vis.add_geometry(grid)
                    ground_lines.append(grid)

        for robot_idx in range(scene_data.num_robots):
            curve0 = scene_data.curves[robot_idx, frame_idx]
            material_frames0 = scene_data.material_frames[robot_idx, frame_idx]
            q0 = scene_data.q_ts[robot_idx, frame_idx]
            sections = self._cross_sections_for_points(q0, s_ps)
            base_mesh, groups = self._build_robot_geometry(
                vis,
                curve0,
                material_frames0,
                sections,
                layout,
                segment_colors=scene_data.segment_colors_rgba[robot_idx],
                base_plate_color=cfg.base_plate_color,
                add_backbone_geometry=not merge_backbone_meshes,
            )
            base_meshes.append(base_mesh)
            backbone_meshes.append(groups)
            if merge_backbone_meshes:
                merged = _merge_triangle_meshes(
                    [cached.mesh for group in groups for cached in group]
                )
                vis.add_geometry(merged)
                merged_backbone_meshes.append(merged)
            else:
                merged_backbone_meshes.append(None)

        actuator_lines: list[o3d.geometry.LineSet] = []
        for layer in scene_data.actuator_layers:
            colors = resolve_actuator_rgba(
                layer,
                override_color=cfg.robot_override,
                default_color=cfg.actuators.color_for_kind(layer.kind),
                scalar_colormap=cfg.actuators.scalar_colormap,
            )
            frame_points = np.asarray(layer.points)[:, frame_idx]
            num_points = frame_points.shape[-2]
            line_set = _make_polylines_lineset(
                frame_points.reshape(-1, num_points, frame_points.shape[-1]),
                colors[:, frame_idx, :, :3].reshape(-1, 3),
            )
            actuator_lines.append(line_set)
            vis.add_geometry(line_set)

        static_meshes: list = []
        if scene_data.static_spheres is not None:
            static_set = scene_data.static_spheres
            if static_set.centers.shape[0] > 0:
                mesh = _make_spheres_mesh(
                    static_set.centers,
                    static_set.radii,
                    static_set.colors,
                    self.sphere_resolution,
                )
                static_meshes.append(mesh)
                vis.add_geometry(mesh)

        dynamic_sphere_batch: DynamicSphereBatchHandle | None = None
        if scene_data.dynamic_spheres is not None:
            dyn_set = scene_data.dynamic_spheres
            trajectories = np.asarray(dyn_set.trajectories, dtype=np.float64)
            if trajectories.shape[0] > 0:
                resolution = max(12, self.sphere_resolution // 2)
                mesh = _make_spheres_mesh(
                    trajectories[:, 0],
                    dyn_set.radii,
                    dyn_set.colors,
                    resolution,
                )
                unit_vertices = _make_unit_sphere_mesh(resolution).vertices
                local_vertices = (
                    unit_vertices[None, :, :]
                    * np.asarray(dyn_set.radii, dtype=np.float64)[:, None, None]
                )
                dynamic_sphere_batch = DynamicSphereBatchHandle(
                    mesh=mesh,
                    trajectories=trajectories,
                    local_vertices=local_vertices,
                )
                vis.add_geometry(mesh)

        return SceneHandles(
            ground_meshes=ground_meshes,
            ground_lines=ground_lines,
            base_meshes=base_meshes,
            backbone_meshes=backbone_meshes,
            merged_backbone_meshes=merged_backbone_meshes,
            actuator_lines=actuator_lines,
            static_meshes=static_meshes,
            dynamic_sphere_batch=dynamic_sphere_batch,
        )

    def _update_scene(
        self,
        vis,
        scene_data: SceneData,
        handles: SceneHandles,
        frame_idx: int,
        *,
        color_config: RendererColorConfig | None = None,
    ) -> None:
        """Update geometry positions for a given frame."""
        cfg = color_config or self.color_config
        layout = scene_data.layout
        ground = self.scene_config.ground
        if (
            ground.visible
            and ground.alignment == "base"
            and not self.scene_config.backdrop.enabled
        ):
            for i, (center, normal, size) in enumerate(
                self._resolve_ground_planes(
                    scene_data.curves[:, frame_idx],
                    scene_data.material_frames[:, frame_idx, 0, :, 0],
                )
            ):
                plane, grid = _make_ground_plane(
                    center,
                    normal,
                    size,
                    ground.color,
                    ground.grid_color,
                    grid_divisions=max(1, round(size / ground.grid_spacing))
                    if ground.grid_spacing
                    else 10,
                    ground_config=ground,
                )
                if ground.surface:
                    handles.ground_meshes[i].vertices = plane.vertices
                    handles.ground_meshes[i].vertex_normals = plane.vertex_normals
                    vis.update_geometry(handles.ground_meshes[i])
                if ground.grid:
                    handles.ground_lines[i].points = grid.points
                    handles.ground_lines[i].lines = grid.lines
                    handles.ground_lines[i].colors = grid.colors
                    vis.update_geometry(handles.ground_lines[i])

        for robot_idx in range(scene_data.num_robots):
            curve = scene_data.curves[robot_idx, frame_idx]
            material_frames = scene_data.material_frames[robot_idx, frame_idx]
            merged_backbone = handles.merged_backbone_meshes[robot_idx]
            self._update_robot_geometry(
                vis,
                curve,
                material_frames,
                handles.base_meshes[robot_idx],
                handles.backbone_meshes[robot_idx],
                layout,
                update_backbone_geometry=merged_backbone is None,
            )
            if merged_backbone is not None:
                _refresh_merged_triangle_mesh(
                    merged_backbone,
                    [
                        cached.mesh
                        for group in handles.backbone_meshes[robot_idx]
                        for cached in group
                    ],
                )
                vis.update_geometry(merged_backbone)

        for layer_idx, layer in enumerate(scene_data.actuator_layers):
            colors = resolve_actuator_rgba(
                layer,
                override_color=cfg.robot_override,
                default_color=cfg.actuators.color_for_kind(layer.kind),
                scalar_colormap=cfg.actuators.scalar_colormap,
            )
            frame_points = np.asarray(layer.points)[:, frame_idx]
            line_set = handles.actuator_lines[layer_idx]
            _update_polylines_lineset(
                line_set,
                frame_points.reshape(
                    -1, frame_points.shape[-2], frame_points.shape[-1]
                ),
                colors[:, frame_idx, :, :3].reshape(-1, 3),
            )
            vis.update_geometry(line_set)

        dynamic_batch = handles.dynamic_sphere_batch
        if dynamic_batch is not None:
            j_idx = min(frame_idx, dynamic_batch.trajectories.shape[1] - 1)
            centers = dynamic_batch.trajectories[:, j_idx]
            dynamic_batch.mesh.vertices = o3d.utility.Vector3dVector(
                (dynamic_batch.local_vertices + centers[:, None, :]).reshape(-1, 3)
            )
            vis.update_geometry(dynamic_batch.mesh)

    def _run_viewer(
        self,
        scene_data: SceneData,
        *,
        playback_speed: float,
        autoplay: bool,
        loop: bool,
        record_cfg: RecordingConfig,
        window_name: str,
        camera_config: CameraConfig | None = None,
        color_config: RendererColorConfig | None = None,
    ) -> None:
        """Play a trajectory with efficient legacy geometry updates.

        Keyboard snapshots capture the preview's approximate materials and
        lighting. Video and PNG-sequence exports use ``_export_sequence``.

        Args:
            scene_data: Prepared robot trajectories and helper geometry.
            playback_speed: Positive multiplier for trajectory playback timing.
            autoplay: Whether playback starts immediately when the window opens.
            loop: Whether playback restarts after reaching the final frame.
            record_cfg: Snapshot filename prefix; automated export is handled by
                the modern renderer before this viewer is called.
            window_name: Title of the native preview window.
            camera_config: Initial camera settings, or ``None`` for defaults.
            color_config: Color override, or ``None`` for renderer defaults.

        Returns:
            None. Blocks until closure and destroys the native preview window.
        """
        vis, ctrl = self._create_visualizer(window_name)
        handles = self._build_scene(
            vis, scene_data, frame_idx=0, color_config=color_config
        )

        # Apply camera configuration
        self._setup_interactive_camera(vis, ctrl, scene_data, camera_config)

        dt_seq = self._frame_intervals_from_ts(scene_data.ts, playback_speed)

        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam_holder = [copy.deepcopy(initial_cam)]

        state = {
            "idx": 0,
            "playing": bool(autoplay),
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def _save_frame(i: int, force: bool = False) -> None:
            if force:
                vis.capture_screen_image(
                    f"{record_cfg.prefix}{i:05d}.png", do_render=True
                )

        def update_frame(i: int) -> None:
            i = max(0, min(scene_data.num_frames - 1, i))
            state["idx"] = i

            self._update_scene(
                vis,
                scene_data,
                handles,
                frame_idx=i,
                color_config=color_config,
            )

            vis.poll_events()
            vis.update_renderer()
            _save_frame(i)

        self._register_key_callbacks(
            vis=vis,
            ctrl=ctrl,
            state=state,
            update_frame=update_frame,
            save_frame_fn=_save_frame,
            initial_cam=initial_cam,
            saved_cam_holder=saved_cam_holder,
            print_prefix="[Open3D]",
        )

        try:
            update_frame(0)

            print(
                f"{window_name}: Space=Play/Pause  ←/→=Step  H=Home  "
                "S=Snapshot  R=ResetCam  C=CaptureCam  L=LoadCam  "
                "V=PrintCam  Q/Esc=Quit"
            )

            while vis.poll_events():
                now = time.time()
                dt_now = state["dt_seq"][min(state["idx"], len(state["dt_seq"]) - 1)]
                if state["playing"] and (now - state["last_tick"] >= dt_now):
                    nxt = state["idx"] + 1
                    if nxt >= scene_data.num_frames:
                        nxt = 0 if loop else scene_data.num_frames - 1
                        state["playing"] = state["playing"] and loop
                    update_frame(nxt)
                    state["last_tick"] = now
                vis.update_renderer()
                time.sleep(0.001)
        finally:
            vis.destroy_window()
