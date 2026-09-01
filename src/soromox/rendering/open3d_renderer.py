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
import os
import time
import warnings
from dataclasses import dataclass

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
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.color_config import RendererColorConfig, ensure_rgba
from soromox.rendering.cross_sections import (
    CrossSection,
    cross_section_sweep_layout,
    evaluate_cross_sections,
    loft_cross_section_contours,
    loft_cross_sections,
)
from soromox.rendering.video_encoding import FFmpegVideoWriter, VideoEncodingConfig
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

    Provides interactive 3D visualization with spheres for backbone,
    optional actuator rendering, and keyboard controls.

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
        width: int = 1920,
        height: int = 1200,
        num_points: int = 80,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_config: RendererColorConfig | None = None,
        backbone_style: str = "swept",
        recompute_normals: bool = True,
        cross_section_resolution: int = 20,
        sphere_resolution: int = 32,
        base_plate_radius_scale: float = 2.0,
        base_plate_thickness: float = 0.06,
        show_ground_plane: bool = True,
        ground_plane_size: float | None = None,
        grid_spacing: tuple[float, float] = (0.5, 0.5),
        base_offsets: Array | None = None,
        actuator_line_width: float = 2.0,
        camera_margin_ratio: float = 0.05,
        merge_backbone_meshes: bool | None = None,
    ):
        """Initialize Open3D renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Window width in pixels
            height: Window height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            color_config: Shared renderer color configuration
            backbone_style: "swept" (material-frame surface) or "discrete" (markers)
            recompute_normals: Whether to recompute vertex normals per segment update
            cross_section_resolution: Number of contour points used to construct
                swept cross-sections. Higher values produce smoother curved sections;
                ignored when ``backbone_style="discrete"``.
            sphere_resolution: Resolution for backbone spheres
            base_plate_radius_scale: Multiplier applied to the base-contour
                transverse extent to size the circular base plate
            base_plate_thickness: Absolute thickness of the base plate geometry
            show_ground_plane: Whether to render a base-aligned ground plane
            ground_plane_size: Optional side length of the ground plane in meters
            grid_spacing: (x, y) spacing between robot bases for batched rendering
            base_offsets: Explicit base offsets of shape (N, 2) or (N, 3) for batched rendering
            actuator_line_width: Width of actuator lines
            camera_margin_ratio: Margin ratio for camera bounding box
            merge_backbone_meshes: Whether to merge each robot's backbone
                primitives into one dynamic mesh. ``None`` selects merging
                automatically for multi-robot scenes while retaining the
                lower per-frame update cost of unmerged single-robot scenes.
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install with: pip install open3d"
            )

        super().__init__(
            robot,
            width,
            height,
            num_points,
            background_color,
            color_config=color_config,
            show_ground_plane=show_ground_plane,
            ground_plane_size=ground_plane_size,
        )

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
        self, color_rgba: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> o3d.visualization.rendering.MaterialRecord:
        """Create an Open3D material honoring per-color alpha."""
        rgba = ensure_rgba(np.asarray(color_rgba, dtype=np.float64))[0]
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency" if rgba[3] < 0.999 else "defaultLit"
        mat.base_color = (
            float(rgba[0]),
            float(rgba[1]),
            float(rgba[2]),
            float(rgba[3]),
        )
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

        scene.clear_geometry()
        layout = scene_data.layout
        s_ps = np.asarray(self._backbone_abscissae, dtype=np.float64)

        for robot_idx in range(scene_data.num_robots):
            curve = scene_data.curves[robot_idx, frame_idx]
            material_frames = scene_data.material_frames[robot_idx, frame_idx]
            q_frame = scene_data.q_ts[robot_idx, frame_idx]
            sections = self._cross_sections_for_points(q_frame, s_ps)
            base_color_rgba = ensure_rgba(np.asarray(cfg.base_plate_color))[0]
            base_axis = material_frames[0, :, 0]
            if self.show_ground_plane:
                ground_size = self._resolve_ground_plane_size()
                ground_plane, ground_grid = _make_ground_plane(
                    curve[0] - self.base_plate_thickness * base_axis,
                    base_axis,
                    ground_size,
                    cfg.ground_plane_color,
                    cfg.ground_plane_grid_color,
                )
                scene.add_geometry(
                    f"ground_plane_{robot_idx}",
                    ground_plane,
                    mat_for((*cfg.ground_plane_color, 0.78)),
                )
                grid_material = o3d.visualization.rendering.MaterialRecord()
                grid_material.shader = "unlitLine"
                grid_material.line_width = 1.0
                scene.add_geometry(
                    f"ground_grid_{robot_idx}", ground_grid, grid_material
                )
            base_mesh = _make_base_plate(
                curve[0] - 0.5 * self.base_plate_thickness * base_axis,
                radius=self._base_plate_radius_for_section(sections[0]),
                thickness=self.base_plate_thickness,
                color=tuple(base_color_rgba[:3]),
                apply_color=False,
                apply_translation=True,
                normal_xyz=base_axis,
            )
            scene.add_geometry(f"base_{robot_idx}", base_mesh, mat_for(base_color_rgba))

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
                        scene.add_geometry(
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
                            scene.add_geometry(
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
                            scene.add_geometry(
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
                            scene.add_geometry(
                                f"ell_{robot_idx}_{s}_{p}",
                                ell,
                                mat_for(raw_color_rgba),
                            )

            for layer_idx, layer in enumerate(scene_data.actuator_layers):
                colors = resolve_actuator_rgba(
                    layer,
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
                    scene.add_geometry(
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
                scene.add_geometry(f"static_{idx}", mesh, mat_for(rgb))

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
                scene.add_geometry(f"dynamic_{dyn_idx}", mesh, mat_for(rgb))

    # -------------------------------------------------------------------------
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
        """Render a single configuration headlessly and return an RGB array.

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
            q_ts=jnp.asarray(q),
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

        render = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        render.scene.set_background(np.array([*self.background_color, 1.0]))

        self._populate_rendering_scene(
            render.scene, scene_data, frame_idx=0, color_config=color_config
        )

        bbox = render.scene.bounding_box
        center = bbox.get_center()
        extent = bbox.get_max_extent()

        # Use camera config if provided, otherwise use defaults
        config = camera_config or CameraConfig()
        camera_pos, look_at = config.compute_auto_position(
            center,
            extent,
            reference_transform=np.asarray(self.base_transform),
        )
        up = config.compute_up()

        render.setup_camera(config.fov, look_at, camera_pos, list(up))

        img = render.render_to_image()
        frame = np.asarray(img)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        return frame

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
        """Display a single frame interactively with full geometry options.

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
        self.render_sequence(
            ts=ts,
            q_ts=q_ts,
            playback_speed=1.0,
            loop=False,
            record_path=None,
            base_offsets=base_offsets,
            color_config=color_config,
            camera_config=camera_config,
            render_actuators=render_actuators,
            actuator_inputs=actuator_inputs,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )

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
        """Render an animated trajectory interactively or headlessly.

        Args:
            ts: Time stamps of shape (T,).
            q_ts: Configurations of shape (T, DOF) or batched (N, T, DOF).
            playback_speed: Playback speed multiplier (>0).
            autoplay: Start playback immediately. If False, wait for space bar to play.
            loop: Whether to loop the animation when it reaches the end.
            record_path: Optional path to save frames or video (extension determines mode).
            record_every_n: Save every n-th frame when recording images.
            record_prefix: Filename prefix for recorded frames.
            video_config: Optional ffmpeg encoding configuration for video output.
            close_when_recording_done: Close the viewer after the final recorded frame.
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
                "Open3D is not installed. Install with: pip install open3d"
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
            video_config=video_config,
            close_when_done=close_when_recording_done,
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
        opt.background_color = np.array(self.background_color, dtype=np.float64)
        opt.line_width = self.actuator_line_width
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
        config = camera_config or CameraConfig()
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

        for robot_idx in range(scene_data.num_robots):
            curve0 = scene_data.curves[robot_idx, frame_idx]
            material_frames0 = scene_data.material_frames[robot_idx, frame_idx]
            q0 = scene_data.q_ts[robot_idx, frame_idx]
            sections = self._cross_sections_for_points(q0, s_ps)
            if self.show_ground_plane:
                base_axis = material_frames0[0, :, 0]
                ground_size = self._resolve_ground_plane_size()
                ground_plane, ground_grid = _make_ground_plane(
                    curve0[0] - self.base_plate_thickness * base_axis,
                    base_axis,
                    ground_size,
                    cfg.ground_plane_color,
                    cfg.ground_plane_grid_color,
                )
                vis.add_geometry(ground_plane)
                vis.add_geometry(ground_grid)
                ground_meshes.append(ground_plane)
                ground_lines.append(ground_grid)
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

    def _init_recorder(
        self,
        record_path: str | None,
        dt_seq: np.ndarray,
        video_config: VideoEncodingConfig | None,
    ) -> tuple[FFmpegVideoWriter | None, str | None, str | None]:
        """Initialize ffmpeg writer or frame directory based on path."""
        if record_path is None:
            return None, None, None

        frame_dir: str | None = None
        video_writer: FFmpegVideoWriter | None = None
        video_path: str | None = None

        ext = os.path.splitext(record_path)[1].lower()
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        if ext in video_exts:
            video_path = record_path
            fps_est = 1.0 / max(1e-6, float(np.median(dt_seq)))
            try:
                video_writer = FFmpegVideoWriter(
                    video_path,
                    int(self.width),
                    int(self.height),
                    fps_est,
                    input_pix_fmt="rgb24",
                    video_config=video_config,
                )
                print(f"[Open3D] Writing video to: {video_path} (fps≈{fps_est:.2f})")
            except FileNotFoundError:
                frame_dir = os.path.splitext(video_path)[0]
                os.makedirs(frame_dir, exist_ok=True)
                print(
                    "[Open3D] ffmpeg not found; falling back to frame directory "
                    f"{frame_dir}"
                )
            except Exception as exc:
                frame_dir = os.path.splitext(video_path)[0]
                os.makedirs(frame_dir, exist_ok=True)
                print(
                    f"[Open3D] ffmpeg failed to start ({exc}); "
                    f"falling back to frame directory {frame_dir}"
                )
        else:
            frame_dir = record_path
            os.makedirs(frame_dir, exist_ok=True)
            print(f"[Open3D] Recording frames to: {frame_dir}")

        return video_writer, frame_dir, video_path

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
        """Interactive viewer with shared geometry and recording."""
        vis, ctrl = self._create_visualizer(window_name)
        handles = self._build_scene(
            vis, scene_data, frame_idx=0, color_config=color_config
        )

        # Apply camera configuration
        self._setup_interactive_camera(vis, ctrl, scene_data, camera_config)

        dt_seq = self._frame_intervals_from_ts(scene_data.ts, playback_speed)
        video_writer, frame_dir, video_path = self._init_recorder(
            record_cfg.path, dt_seq, record_cfg.video_config
        )

        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam_holder = [copy.deepcopy(initial_cam)]

        state = {
            "idx": 0,
            "playing": bool(autoplay),
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def _fallback_to_frames(reason: str):
            nonlocal video_writer, frame_dir
            if video_writer is not None:
                try:
                    video_writer.close()
                    if video_writer.stderr_log:
                        print(f"[Open3D] ffmpeg stderr: {video_writer.stderr_log}")
                except Exception:
                    pass
            video_writer = None
            if frame_dir is None:
                fallback_dir = os.path.splitext(video_path or "frames")[0]
                frame_dir = fallback_dir
                os.makedirs(frame_dir, exist_ok=True)
                print(
                    f"[Open3D] ffmpeg stream failed ({reason}); "
                    f"falling back to frame directory {frame_dir}"
                )

        def _save_frame(i: int, force: bool = False) -> None:
            if video_writer is not None:
                try:
                    img = np.asarray(
                        vis.capture_screen_float_buffer(do_render=True),
                        dtype=np.float32,
                    )
                    frame = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
                    video_writer.write(frame)
                    return
                except BrokenPipeError:
                    _fallback_to_frames("broken pipe")
                except Exception as exc:
                    _fallback_to_frames(str(exc))
            if frame_dir is None:
                if force:
                    vis.capture_screen_image(
                        f"{record_cfg.prefix}{i:05d}.png", do_render=True
                    )
                return
            if force or (i % record_cfg.every_n) == 0:
                fname = os.path.join(frame_dir, f"{record_cfg.prefix}{i:05d}.png")
                vis.capture_screen_image(fname, do_render=True)

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
                    if (
                        record_cfg.close_when_done
                        and record_cfg.path is not None
                        and not loop
                        and nxt == scene_data.num_frames - 1
                    ):
                        break
                vis.update_renderer()
                time.sleep(0.001)
        finally:
            try:
                if video_writer is not None:
                    video_writer.close()
            finally:
                vis.destroy_window()
