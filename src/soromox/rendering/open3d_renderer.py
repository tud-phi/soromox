"""Open3D-based renderer for any continuum soft robot.

Provides 3D visualization with:
- Backbone as spheres (per-segment colors & radii)
- Tendons as polyline LineSets (if robot supports forward_kinematics_tendons)
- Recording frames to PNGs
- Interactive camera controls

Controls:
    Space: play/pause
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
import subprocess
import time
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

# Track initialization of the Open3D GUI app for older versions without is_running.
_GUI_APP_INITIALIZED = False

# Default colormap used for segment colors when none are provided
DEFAULT_SEGMENT_COLORMAP = "coolwarm"

from soromox.rendering.base import BaseSoftRobotRenderer
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


def _make_base_plate(
    center_xyz: np.ndarray,
    radius: float,
    thickness: float = 0.005,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: int = 48,
    apply_color: bool = True,
    apply_translation: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Create a base plate cylinder mesh."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=float(thickness), resolution=resolution, split=1
    )
    mesh.compute_vertex_normals()
    if apply_color:
        mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    if apply_translation:
        mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


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
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z_axis, axis)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, axis)
    if s < 1e-9 and c > 0.0:
        R = np.eye(3)
    elif s < 1e-9 and c < 0.0:
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
    else:
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    mesh.rotate(R, center=np.zeros(3))

    mid = (p0 + p1) / 2.0
    mesh.translate(mid, relative=False)
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


def _split_counts_by_lengths(num_points: int, L: np.ndarray) -> list[int]:
    """Split num_points across len(L) segments proportionally to lengths."""
    L = np.asarray(L, dtype=np.float64).reshape(-1)
    S = int(L.size)
    if S == 0:
        return [num_points]
    if num_points < S:
        base = [1] * S
        base[-1] += num_points - S
        return base

    Ltot = float(L.sum())
    if Ltot <= 0:
        base = [num_points // S] * S
        rem = num_points - sum(base)
        for i in range(rem):
            base[i] += 1
        return base

    raw = [num_points * (float(li) / Ltot) for li in L]
    base = [max(1, int(round(x))) for x in raw]
    diff = num_points - sum(base)
    if diff != 0:
        fracs = np.array([x - np.floor(x) for x in raw])
        order = np.argsort(fracs)[::-1] if diff > 0 else np.argsort(fracs)
        idxs = order.tolist()
        k = abs(diff)
        j = 0
        while k > 0 and idxs:
            i = idxs[j % len(idxs)]
            if diff > 0:
                base[i] += 1
            else:
                if base[i] > 1:
                    base[i] -= 1
                else:
                    j += 1
                    continue
            k -= 1
            j += 1
    return base


def _colormap_rgba(name: str, count: int) -> np.ndarray:
    """Return RGBA colors from a Matplotlib colormap name."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap(name)
    return np.asarray(
        [cmap(i / max(1, count - 1))[:4] for i in range(count)], dtype=np.float64
    )


def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
    """Ensure array has shape (N, 4) with alpha; accepts (N, 3/4)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] not in (3, 4):
            raise ValueError(f"Color must have 3 or 4 channels; got shape {arr.shape}")
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError(
            f"Color array must have shape (N, 3) or (N, 4); got {arr.shape}"
        )
    if arr.shape[1] == 3:
        alpha = np.ones((arr.shape[0], 1), dtype=np.float64)
        arr = np.concatenate([arr, alpha], axis=1)
    return np.clip(arr, 0.0, 1.0)


def _normalize_color_palette(
    color_spec: str | Array | np.ndarray | None,
    count: int,
    *,
    default_palette: np.ndarray,
) -> np.ndarray:
    """Resolve a color specification to an RGBA palette of length `count`.

    Supports:
        - None: use default_palette
        - str: Matplotlib colormap name
        - array/list/tuple: shape (N, 3) or (N, 4), cycled if N < count
    """
    if color_spec is None or color_spec == "same":
        palette = np.asarray(default_palette, dtype=np.float64)
    elif isinstance(color_spec, str):
        palette = _colormap_rgba(color_spec, count)
    else:
        palette = _ensure_rgba(np.asarray(color_spec, dtype=np.float64))

    if palette.shape[0] == 0:
        palette = np.asarray(default_palette, dtype=np.float64)

    if palette.shape[0] < count:
        reps = int(np.ceil(count / palette.shape[0]))
        palette = np.tile(palette, (reps, 1))

    palette = palette[:count]
    return np.clip(palette, 0.0, 1.0)


class _FFmpegVideoWriter:
    """Minimal ffmpeg pipe for RGB frames."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.path = path
        self._stderr_log: str | None = None
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            if self.proc.stderr:
                try:
                    err = self.proc.stderr.read()
                    if err:
                        self._stderr_log = err.decode("utf-8", errors="ignore")
                except Exception:
                    pass
        finally:
            self.proc.wait()

    @property
    def stderr_log(self) -> str | None:
        return self._stderr_log


def _mesh_center(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Approximate center: mean of vertices."""
    return np.asarray(mesh.vertices).mean(axis=0)


@dataclass
class SegmentLayout:
    """Precomputed segment metadata for backbone geometry."""

    starts: np.ndarray
    ends: np.ndarray
    radii: np.ndarray
    colors_rgba: np.ndarray  # (S, 4)

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
class SceneData:
    curves: np.ndarray  # (N, T, P, 3)
    ts: np.ndarray  # (T,)
    layout: SegmentLayout
    robot_colors_rgba: np.ndarray | None  # (N, 4) or None for per-segment colors
    tendon_curves: np.ndarray | None = None  # (N, T, n_tend, P, 3)
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


@dataclass
class LegacySceneHandles:
    base_meshes: list
    backbone_meshes: list[list[list]]
    tendon_lines: list[list]
    static_meshes: list
    dynamic_meshes: list
    dynamic_trajs: list[np.ndarray]


# ======================================================================================
# Open3D Renderer Class
# ======================================================================================


class Open3DRenderer(BaseSoftRobotRenderer):
    """Open3D visualization for any continuum soft robot.

    Provides interactive 3D visualization with spheres for backbone,
    optional tendon rendering, and keyboard controls.

    Example:
        >>> renderer = Open3DRenderer(robot)
        >>> renderer.show(q)  # Single interactive frame
        >>> renderer.render_sequence(ts, q_ts, playback_speed=1.0)  # Animated playback
        >>> img = renderer.render_frame(q)  # Headless capture
    """

    def __init__(
        self,
        robot: SoftRobot,
        width: int = 1280,
        height: int = 800,
        num_points: int = 80,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seg_colors: str | Array | np.ndarray | None = DEFAULT_SEGMENT_COLORMAP,
        robot_colors: str | Array | np.ndarray | None = None,
        backbone_style: str = "spheres",
        tube_resolution: int = 20,
        sphere_resolution: int = 32,
        base_plate_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_plate_radius_scale: float = 2.0,
        base_plate_thickness: float = 5e-2 * 1.3,
        grid_spacing: tuple[float, float] = (0.5, 0.5),
        base_offsets: Array | None = None,
        tendon_color: tuple[float, float, float] = (0.9, 0.15, 0.15),
        tendon_line_width: float = 2.0,
        camera_margin_ratio: float = 0.05,
        viewer_backend: str = "legacy",
    ):
        """Initialize Open3D renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Window width in pixels
            height: Window height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            seg_colors: Optional colors for each segment. Defaults to a colormap
                ("coolwarm"). Accepts a Matplotlib colormap name or an array of shape
                (N, 3) or (N, 4) with optional alpha channel.
            robot_colors: Color configuration when rendering multiple robots. Can be:
                - None: Use per-segment colors for all robots
                - "same": All robots use the first resolved segment color
                - Colormap name (e.g., "viridis", "plasma"): Distinct colors per robot
                - Array of RGB or RGBA colors of shape (N, 3/4); cycled if fewer than N
            backbone_style: "spheres" (default) or "tube" (swept cylinder segments)
            tube_resolution: Radial resolution for tube segments
            sphere_resolution: Resolution for backbone spheres
            base_plate_color: RGB color for the base plate geometry
            base_plate_radius_scale: Multiplier applied to the maximum segment radius to size the base plate
            base_plate_thickness: Absolute thickness of the base plate geometry
            grid_spacing: (x, y) spacing between robot bases for batched rendering
            base_offsets: Explicit base offsets of shape (N, 2) or (N, 3) for batched rendering
            tendon_color: RGB color for tendon lines
            tendon_line_width: Width of tendon lines
            camera_margin_ratio: Margin ratio for camera bounding box
            viewer_backend: "legacy" (default), "gui" (force SceneWidget), or "auto"
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install with: pip install open3d"
            )

        super().__init__(robot, width, height, num_points, background_color)

        self.seg_colors = seg_colors
        self._seg_colors_config = seg_colors
        self._robot_colors_config = robot_colors
        self.sphere_resolution = sphere_resolution
        self.grid_spacing = grid_spacing
        self._base_offsets = base_offsets
        self.tendon_color = tendon_color
        self.tendon_line_width = tendon_line_width
        self.camera_margin_ratio = camera_margin_ratio
        self.base_plate_color = tuple(base_plate_color)
        self.base_plate_radius_scale = float(base_plate_radius_scale)
        self.base_plate_thickness = float(base_plate_thickness)
        self.backbone_style = backbone_style
        self.tube_resolution = int(tube_resolution)
        vb = viewer_backend.lower()
        if vb not in ("auto", "gui", "legacy"):
            raise ValueError('viewer_backend must be "auto", "gui", or "legacy"')
        self.viewer_backend = vb

        # Cache tendon FK if available
        self._has_tendons = hasattr(robot, "forward_kinematics_tendons")
        if self._has_tendons:
            self._batched_fk_tendons = jax.vmap(
                robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1
            )

    @property
    def is_3d(self) -> bool:
        """Whether this renderer operates in 3D space.

        Returns:
            is_3d (bool): Always True for the Open3D renderer.
        """
        return True

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xyz from SE(3) matrices."""
        return poses[:, :3, 3]

    def compute_tendon_curves(self, q: Array) -> Array | None:
        """Compute tendon paths if the robot exposes tendon kinematics.

        Args:
            q: Robot configuration of shape (DOF,) for a single robot.

        Returns:
            curves (Optional[Array]): An array of shape (n_tendons, num_points, 3) with
            tendon curves expressed in world coordinates, or None if tendons are
            not supported by the robot.
        """
        if not self._has_tendons:
            return None
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self._batched_fk_tendons(q, s_ps)

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _make_mesh_material(
        self, color_rgba: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> o3d.visualization.rendering.MaterialRecord:
        """Create an Open3D material honoring per-color alpha."""
        rgba = _ensure_rgba(np.asarray(color_rgba, dtype=np.float64))[0]
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
        """Approximate transparency for the legacy visualizer by blending with the background."""
        rgba = _ensure_rgba(np.asarray(color, dtype=np.float64))[0]
        alpha = float(rgba[3])
        rgb = rgba[:3]
        if alpha >= 0.999:
            return tuple(rgb.tolist())
        bg = np.asarray(self.background_color, dtype=np.float64)
        blended = alpha * rgb + (1.0 - alpha) * bg
        return tuple(np.clip(blended, 0.0, 1.0).tolist())

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
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        counts = _split_counts_by_lengths(P, L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        colors_rgba = _normalize_color_palette(
            self._seg_colors_config,
            len(counts),
            default_palette=_colormap_rgba(DEFAULT_SEGMENT_COLORMAP, len(counts)),
        )
        return SegmentLayout(
            starts=starts,
            ends=ends,
            radii=r_seg,
            colors_rgba=colors_rgba,
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
        dynamics_spheres_colors: Array | None,
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

        colors = self._normalize_color_array(dynamics_spheres_colors, N, default_color)
        return DynamicSpheres(trajectories=centers, radii=radii, colors=colors)

    def _resolve_backend(self, backend: str | None) -> str:
        """Resolve requested backend to either 'legacy' or 'gui'."""
        mode = (backend or self.viewer_backend or "legacy").lower()
        if mode == "auto":
            try:
                import open3d.visualization.gui as _  # noqa: F401

                return "gui"
            except Exception:
                return "legacy"
        if mode in ("legacy", "gui"):
            return mode
        raise ValueError("backend must be 'legacy', 'gui', or 'auto'")

    def _prepare_scene_data(
        self,
        ts: Array,
        q_ts: Array,
        *,
        base_offsets: Array | None,
        robot_colors: str | Array | np.ndarray | None,
        static_spheres_positions: Array | None,
        static_spheres_radii: Array | None,
        static_spheres_colors: Array | None,
        dynamic_spheres_positions: Array | None,
        dynamic_spheres_radii: Array | None,
        dynamics_spheres_colors: Array | None,
    ) -> SceneData:
        """Compute curves/tendons and validate auxiliary geometry."""
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
        if ts_np.shape[0] not in (1, T):
            raise ValueError(
                f"ts length ({ts_np.shape[0]}) must be 1 or match q_ts time dimension ({T})"
            )
        if ts_np.shape[0] == 1 and T > 1:
            # Auto-generate monotonic timestamps if only a single origin is provided
            ts_np = np.arange(T, dtype=np.float64)

        # Compute base offsets
        offsets = base_offsets if base_offsets is not None else self._base_offsets
        if offsets is None:
            offsets = self._compute_grid_offsets(int(N), self.grid_spacing)
        offsets = jnp.asarray(offsets)
        if offsets.ndim != 2 or offsets.shape[0] != N:
            raise ValueError(
                f"base_offsets must have shape (N, 2/3); got {offsets.shape}"
            )

        # Backbone curves (T, N, P, 3) -> (N, T, P, 3)
        q_ts_time_first = q_ts_arr.transpose(1, 0, 2)

        def _compute_curves_for_timestep(q_batch: jax.Array) -> jax.Array:
            return self.compute_backbone_curves_batched(q_batch, offsets)

        all_curves_time_first = jax.vmap(_compute_curves_for_timestep)(q_ts_time_first)
        curves = np.array(all_curves_time_first.transpose(1, 0, 2, 3), dtype=np.float64)

        layout = self._compute_segment_layout(curves.shape[2])
        robot_colors_cfg = (
            self._robot_colors_config if robot_colors is None else robot_colors
        )
        robot_colors_rgba = None
        if robot_colors_cfg is not None:
            robot_colors_rgba = _get_robot_colors(
                N, robot_colors_cfg, layout.colors_rgba[0]
            )

        tendon_curves = None
        if self._has_tendons:

            def _compute_tendons_for_timestep(q_batch: jax.Array) -> jax.Array:
                return jax.vmap(self.compute_tendon_curves)(q_batch)

            all_tendons_time_first = jax.vmap(_compute_tendons_for_timestep)(
                q_ts_time_first
            )
            tendon_curves = np.array(
                all_tendons_time_first.transpose(1, 0, 2, 3, 4), dtype=np.float64
            )

            offsets_np = np.asarray(offsets)
            if offsets_np.shape[1] == 2:
                offsets_np = np.concatenate(
                    [offsets_np, np.zeros((offsets_np.shape[0], 1))], axis=1
                )
            tendon_curves = tendon_curves + offsets_np[:, None, None, None, :]

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
            ts=ts_np,
            layout=layout,
            robot_colors_rgba=robot_colors_rgba,
            tendon_curves=tendon_curves,
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
    ) -> None:
        """Add all geometries for a specific frame to an Open3DScene."""
        try:
            import open3d.visualization.rendering as rendering
        except (
            Exception
        ) as exc:  # pragma: no cover - only used when rendering is available
            raise RuntimeError("Open3D rendering backend unavailable") from exc

        if material_cache is None:
            material_cache = {}

        def mat_for(color_rgba: np.ndarray | tuple[float, ...]):
            key = tuple(np.asarray(color_rgba, dtype=np.float64).reshape(-1))
            if key not in material_cache:
                material_cache[key] = self._make_mesh_material(key)
            return material_cache[key]

        scene.clear_geometry()
        layout = scene_data.layout

        for robot_idx in range(scene_data.num_robots):
            curve = scene_data.curves[robot_idx, frame_idx]
            color_override = (
                scene_data.robot_colors_rgba[robot_idx]
                if scene_data.robot_colors_rgba is not None
                else None
            )
            base_color_rgba = _ensure_rgba(np.asarray(self.base_plate_color))[0]
            base_mesh = _make_base_plate(
                curve[0],
                radius=float(self.base_plate_radius_scale * layout.radii.max()),
                thickness=self.base_plate_thickness,
                color=tuple(base_color_rgba[:3]),
                apply_color=False,
                apply_translation=True,
            )
            scene.add_geometry(f"base_{robot_idx}", base_mesh, mat_for(base_color_rgba))

            for s in range(layout.segments):
                c0, c1 = int(layout.starts[s]), int(layout.ends[s])
                raw_color_rgba = (
                    color_override
                    if color_override is not None
                    else layout.colors_rgba[s]
                )
                raw_color_rgb = tuple(np.asarray(raw_color_rgba).reshape(-1)[:3])
                if self.backbone_style == "tube" and c1 - c0 >= 1:
                    for p in range(c0, c1 - 1):
                        cyl = _make_cylinder_between(
                            curve[p],
                            curve[p + 1],
                            radius=float(layout.radii[min(s, len(layout.radii) - 1)]),
                            color=raw_color_rgb,
                            resolution=self.tube_resolution,
                            apply_color=False,
                        )
                        scene.add_geometry(
                            f"tube_{robot_idx}_{s}_{p}", cyl, mat_for(raw_color_rgba)
                        )
                else:
                    for p in range(c0, c1):
                        sp = _make_sphere(
                            curve[p],
                            radius=float(layout.radii[min(s, len(layout.radii) - 1)]),
                            color=raw_color_rgb,
                            resolution=self.sphere_resolution,
                            apply_color=False,
                            apply_translation=True,
                        )
                        scene.add_geometry(
                            f"sphere_{robot_idx}_{s}_{p}", sp, mat_for(raw_color_rgba)
                        )

            if scene_data.tendon_curves is not None:
                robot_tendons = scene_data.tendon_curves[robot_idx, frame_idx]
                for k in range(robot_tendons.shape[0]):
                    ls = _make_polyline_lineset(
                        robot_tendons[k], color=self.tendon_color
                    )
                    mat_line = rendering.MaterialRecord()
                    mat_line.shader = "unlitLine"
                    mat_line.line_width = self.tendon_line_width
                    scene.add_geometry(f"tendon_{robot_idx}_{k}", ls, mat_line)

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
        robot_colors: str | Array | np.ndarray | None = None,
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
            robot_colors: Optional per-robot color configuration (colormap name or RGBA array).
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
            robot_colors=robot_colors,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )

        render = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        render.scene.set_background(np.array([*self.background_color, 1.0]))

        self._populate_rendering_scene(render.scene, scene_data, frame_idx=0)

        bbox = render.scene.bounding_box
        center = bbox.get_center()
        extent = bbox.get_max_extent()
        render.setup_camera(60.0, center, center + [0, 0, extent * 2], [0, 1, 0])

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
        backend: str | None = None,
        base_offsets: Array | None = None,
        robot_colors: str | Array | np.ndarray | None = None,
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
            backend: Viewer backend override ("legacy", "gui", or "auto").
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            robot_colors: Optional per-robot color configuration (colormap name or RGBA array).
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
            backend=backend,
            playback_speed=1.0,
            loop=False,
            record_path=None,
            base_offsets=base_offsets,
            robot_colors=robot_colors,
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
        backend: str | None = None,
        playback_speed: float = 1.0,
        loop: bool = False,
        record_path: str | None = None,
        record_every_n: int = 1,
        record_prefix: str = "frame_",
        base_offsets: Array | None = None,
        robot_colors: str | Array | np.ndarray | None = None,
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
            backend: Viewer backend override ("legacy", "gui", or "auto").
            playback_speed: Playback speed multiplier (>0).
            loop: Whether to loop the animation when it reaches the end.
            record_path: Optional path to save frames or video (extension determines mode).
            record_every_n: Save every n-th frame when recording images.
            record_prefix: Filename prefix for recorded frames.
            base_offsets: Optional base offsets of shape (N, 2/3) for batched layouts.
            robot_colors: Optional per-robot color configuration (colormap name or RGBA array).
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

        backend_resolved = self._resolve_backend(backend)
        scene_data = self._prepare_scene_data(
            ts=ts,
            q_ts=q_ts,
            base_offsets=base_offsets,
            robot_colors=robot_colors,
            static_spheres_positions=static_spheres_positions,
            static_spheres_radii=static_spheres_radii,
            static_spheres_colors=static_spheres_colors,
            dynamic_spheres_positions=dynamic_spheres_positions,
            dynamic_spheres_radii=dynamic_spheres_radii,
            dynamics_spheres_colors=dynamics_spheres_colors,
        )
        record_cfg = RecordingConfig(
            path=record_path, prefix=record_prefix, every_n=record_every_n
        )
        if backend_resolved == "legacy":
            self._run_legacy_viewer(
                scene_data,
                playback_speed=playback_speed,
                loop=loop,
                record_cfg=record_cfg,
                window_name=window_name,
            )
            return

        self._run_gui_viewer(
            scene_data,
            playback_speed=playback_speed,
            loop=loop,
            record_cfg=record_cfg,
            window_name=window_name,
        )

    # -------------------------------------------------------------------------
    # Legacy viewer backend
    # -------------------------------------------------------------------------

    def _create_visualizer(self, window_name: str):
        """Create a visualizer with common options set."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name=window_name,
            width=self.width,
            height=self.height,
        )

        opt = vis.get_render_option()
        opt.background_color = np.array(self.background_color, dtype=np.float64)
        opt.line_width = self.tendon_line_width
        return vis, vis.get_view_control()

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
            state["playing"] = not state["playing"]
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
            vis.destroy_window()
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
        layout: SegmentLayout,
        color_override: np.ndarray | None = None,
    ) -> tuple[object, list[list[object]]]:
        """Add base and backbone meshes for one robot; return handles."""
        base_color = self._blend_with_background(self.base_plate_color)
        base_mesh = _make_base_plate(
            curve0[0],
            radius=float(self.base_plate_radius_scale * layout.radii.max()),
            thickness=self.base_plate_thickness,
            color=base_color,
        )
        vis.add_geometry(base_mesh)

        spheres_groups: list[list] = []
        for s in range(layout.segments):
            seg_spheres: list = []
            c0, c1 = int(layout.starts[s]), int(layout.ends[s])
            raw_color_rgba = (
                color_override if color_override is not None else layout.colors_rgba[s]
            )
            seg_color = self._blend_with_background(raw_color_rgba)
            if self.backbone_style == "tube" and c1 - c0 >= 1:
                for p in range(c0, c1 - 1):
                    cyl = _make_cylinder_between(
                        curve0[p],
                        curve0[p + 1],
                        radius=float(layout.radii[min(s, len(layout.radii) - 1)]),
                        color=seg_color,
                        resolution=self.tube_resolution,
                    )
                    seg_spheres.append(cyl)
                    vis.add_geometry(cyl)
            else:
                for p in range(c0, c1):
                    sp = _make_sphere(
                        curve0[p],
                        radius=float(layout.radii[min(s, len(layout.radii) - 1)]),
                        color=seg_color,
                        resolution=self.sphere_resolution,
                    )
                    seg_spheres.append(sp)
                    vis.add_geometry(sp)
            spheres_groups.append(seg_spheres)
        return base_mesh, spheres_groups

    def _update_robot_geometry(
        self,
        vis,
        curve: np.ndarray,
        base_mesh,
        spheres_groups: list[list],
        layout: SegmentLayout,
    ) -> None:
        """Translate base and backbone geometry to a new curve position."""
        delta = curve[0] - _mesh_center(base_mesh)
        base_mesh.translate(delta, relative=True)
        vis.update_geometry(base_mesh)

        for s in range(len(spheres_groups)):
            c0, c1 = int(layout.starts[s]), int(layout.ends[s])
            seg_spheres = spheres_groups[s]
            if self.backbone_style == "tube" and c1 - c0 >= 1:
                # Cylinders need full re-orientation + centering, not just translation
                for i_local, p in enumerate(
                    range(c0, min(c1 - 1, c0 + len(seg_spheres)))
                ):
                    mesh = seg_spheres[i_local]
                    # Preserve existing color while regenerating geometry
                    color_arr = np.asarray(mesh.vertex_colors)
                    base_color = (
                        color_arr[0]
                        if color_arr.size >= 3
                        else np.array([1.0, 1.0, 1.0])
                    )
                    updated = _make_cylinder_between(
                        curve[p],
                        curve[p + 1],
                        radius=float(layout.radii[min(s, len(layout.radii) - 1)]),
                        color=tuple(np.asarray(base_color).reshape(-1)[:3]),
                        resolution=self.tube_resolution,
                    )
                    mesh.vertices = updated.vertices
                    mesh.triangles = updated.triangles
                    mesh.vertex_normals = updated.vertex_normals
                    mesh.triangle_normals = updated.triangle_normals
                    mesh.vertex_colors = updated.vertex_colors
                    vis.update_geometry(mesh)
            else:
                for i_local, p in enumerate(range(c0, min(c1, c0 + len(seg_spheres)))):
                    mesh = seg_spheres[i_local]
                    delta = curve[p] - _mesh_center(mesh)
                    mesh.translate(delta, relative=True)
                    vis.update_geometry(mesh)

    def _build_legacy_scene(
        self, vis, scene_data: SceneData, frame_idx: int = 0
    ) -> LegacySceneHandles:
        """Construct initial geometry for the legacy viewer."""
        layout = scene_data.layout
        base_meshes: list = []
        backbone_meshes: list[list[list]] = []

        for robot_idx in range(scene_data.num_robots):
            curve0 = scene_data.curves[robot_idx, frame_idx]
            color_override = (
                scene_data.robot_colors_rgba[robot_idx]
                if scene_data.robot_colors_rgba is not None
                else None
            )
            base_mesh, groups = self._build_robot_geometry(
                vis, curve0, layout, color_override=color_override
            )
            base_meshes.append(base_mesh)
            backbone_meshes.append(groups)

        tendon_lines: list[list] = []
        if scene_data.tendon_curves is not None:
            for robot_idx in range(scene_data.num_robots):
                robot_lines: list = []
                robot_tendons = scene_data.tendon_curves[robot_idx, frame_idx]
                for k in range(robot_tendons.shape[0]):
                    ls = _make_polyline_lineset(
                        robot_tendons[k], color=self.tendon_color
                    )
                    robot_lines.append(ls)
                    vis.add_geometry(ls)
                tendon_lines.append(robot_lines)

        static_meshes: list = []
        if scene_data.static_spheres is not None:
            static_set = scene_data.static_spheres
            for ctr, rad, col in zip(
                static_set.centers, static_set.radii, static_set.colors
            ):
                mesh = _make_sphere(
                    ctr,
                    float(rad),
                    tuple(col),
                    self.sphere_resolution,
                )
                static_meshes.append(mesh)
                vis.add_geometry(mesh)

        dynamic_meshes: list = []
        dynamic_trajs: list[np.ndarray] = []
        if scene_data.dynamic_spheres is not None:
            dyn_set = scene_data.dynamic_spheres
            for centers_T3, rad, col in zip(
                dyn_set.trajectories, dyn_set.radii, dyn_set.colors
            ):
                centers_np = np.asarray(centers_T3, dtype=np.float64)
                mesh0 = _make_sphere(
                    centers_np[0],
                    float(rad),
                    tuple(col),
                    max(12, self.sphere_resolution // 2),
                )
                dynamic_meshes.append(mesh0)
                dynamic_trajs.append(centers_np)
                vis.add_geometry(mesh0)

        return LegacySceneHandles(
            base_meshes=base_meshes,
            backbone_meshes=backbone_meshes,
            tendon_lines=tendon_lines,
            static_meshes=static_meshes,
            dynamic_meshes=dynamic_meshes,
            dynamic_trajs=dynamic_trajs,
        )

    def _update_legacy_scene(
        self,
        vis,
        scene_data: SceneData,
        handles: LegacySceneHandles,
        frame_idx: int,
    ) -> None:
        """Update geometry positions for a given frame."""
        layout = scene_data.layout

        for robot_idx in range(scene_data.num_robots):
            curve = scene_data.curves[robot_idx, frame_idx]
            self._update_robot_geometry(
                vis,
                curve,
                handles.base_meshes[robot_idx],
                handles.backbone_meshes[robot_idx],
                layout,
            )

        if scene_data.tendon_curves is not None:
            for robot_idx, robot_lines in enumerate(handles.tendon_lines):
                robot_tendons = scene_data.tendon_curves[robot_idx, frame_idx]
                for k, ls in enumerate(robot_lines):
                    t_pts = np.array(robot_tendons[k], dtype=np.float64, copy=True)
                    ls.points = o3d.utility.Vector3dVector(t_pts)
                    vis.update_geometry(ls)

        for mesh, traj in zip(handles.dynamic_meshes, handles.dynamic_trajs):
            j_idx = min(frame_idx, traj.shape[0] - 1)
            delta = traj[j_idx] - _mesh_center(mesh)
            mesh.translate(delta, relative=True)
            vis.update_geometry(mesh)

    def _init_recorder(
        self, record_path: str | None, dt_seq: np.ndarray
    ) -> tuple[_FFmpegVideoWriter | None, str | None, str | None]:
        """Initialize ffmpeg writer or frame directory based on path."""
        if record_path is None:
            return None, None, None

        frame_dir: str | None = None
        video_writer: _FFmpegVideoWriter | None = None
        video_path: str | None = None

        ext = os.path.splitext(record_path)[1].lower()
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        if ext in video_exts:
            video_path = record_path
            fps_est = 1.0 / max(1e-6, float(np.median(dt_seq)))
            try:
                video_writer = _FFmpegVideoWriter(
                    video_path, int(self.width), int(self.height), fps_est
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

    def _run_legacy_viewer(
        self,
        scene_data: SceneData,
        *,
        playback_speed: float,
        loop: bool,
        record_cfg: RecordingConfig,
        window_name: str,
    ) -> None:
        """Interactive legacy viewer with shared geometry and recording."""
        vis, ctrl = self._create_visualizer(window_name)
        handles = self._build_legacy_scene(vis, scene_data, frame_idx=0)
        dt_seq = self._frame_intervals_from_ts(scene_data.ts, playback_speed)
        video_writer, frame_dir, video_path = self._init_recorder(
            record_cfg.path, dt_seq
        )

        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam_holder = [copy.deepcopy(initial_cam)]

        state = {
            "idx": 0,
            "playing": True,
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

            self._update_legacy_scene(vis, scene_data, handles, frame_idx=i)

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

        update_frame(0)

        print(
            f"{window_name}: Space=Play/Pause  ←/→=Step  H=Home  "
            "S=Snapshot  R=ResetCam  C=CaptureCam  L=LoadCam  V=PrintCam  Q/Esc=Quit"
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
        if video_writer is not None:
            video_writer.close()

    # -------------------------------------------------------------------------
    # GUI viewer backend (SceneWidget)
    # -------------------------------------------------------------------------

    def _run_gui_viewer(
        self,
        scene_data: SceneData,
        *,
        playback_speed: float,
        loop: bool,
        record_cfg: RecordingConfig,
        window_name: str,
    ) -> None:
        """Interactive GUI viewer using SceneWidget."""
        raise NotImplementedError("Open3D GUI backend doesn't work well yet.")
        try:
            import open3d.visualization.gui as gui
            import open3d.visualization.rendering as rendering
        except Exception as exc:
            raise RuntimeError(
                f"[Open3D] GUI backend unavailable ({exc}). "
                "Use backend='legacy' or install Open3D with GUI support."
            ) from exc

        dt_seq = self._frame_intervals_from_ts(scene_data.ts, playback_speed)
        # Recorder is initialized after we confirm capture support
        video_writer: _FFmpegVideoWriter | None = None
        frame_dir: str | None = None
        video_path: str | None = None
        material_cache: dict[tuple[float, float, float], rendering.MaterialRecord] = {}
        missing_render_to_image_warned = False
        recording_enabled = {"ok": True}

        app = gui.Application.instance
        global _GUI_APP_INITIALIZED
        is_running = getattr(app, "is_running", lambda: _GUI_APP_INITIALIZED)()
        if not is_running:
            app.initialize()
            _GUI_APP_INITIALIZED = True
        window = app.create_window(window_name, self.width, self.height)

        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background(
            np.array([*self.background_color, 1.0], dtype=np.float32)
        )
        window.add_child(scene_widget)

        # Check capture support; disable recording pre-emptively if unsupported
        _supports_capture = hasattr(scene_widget.scene, "render_to_image") or hasattr(
            getattr(scene_widget.scene, "scene", None), "render_to_image"
        )
        if record_cfg.path and not _supports_capture:
            print(
                "[Open3D] GUI: capture not supported on this build; "
                "record_path will be ignored."
            )
        else:
            video_writer, frame_dir, video_path = self._init_recorder(
                record_cfg.path, dt_seq
            )

        def _grab_frame():
            """Best-effort frame capture across Open3D builds."""
            nonlocal \
                missing_render_to_image_warned, \
                recording_enabled, \
                video_writer, \
                frame_dir
            # Newer builds: method on Open3DScene
            if hasattr(scene_widget.scene, "render_to_image"):
                try:
                    return scene_widget.scene.render_to_image()
                except Exception:
                    pass
            # Older builds: RenderScene may be exposed via .scene
            inner_scene = getattr(scene_widget.scene, "scene", None)
            if inner_scene is not None and hasattr(inner_scene, "render_to_image"):
                try:
                    return inner_scene.render_to_image()
                except Exception:
                    pass
            if not missing_render_to_image_warned:
                print(
                    "[Open3D] GUI: render_to_image unavailable on this build; "
                    "recording disabled."
                )
                missing_render_to_image_warned = True
            if recording_enabled["ok"]:
                if video_writer is not None:
                    try:
                        video_writer.close()
                    except Exception:
                        pass
                    video_writer = None
                frame_dir = None
                recording_enabled["ok"] = False
            return None

        def populate(idx: int) -> None:
            self._populate_rendering_scene(
                scene_widget.scene,
                scene_data,
                frame_idx=idx,
                material_cache=material_cache,
            )

        populate(0)
        bbox = scene_widget.scene.bounding_box
        center = bbox.get_center()
        try:
            scene_widget.scene.setup_camera(60.0, bbox, center)
        except Exception:
            extent = bbox.get_max_extent()
            eye = center + np.array([0, 0, extent * 2])
            scene_widget.scene.camera.look_at(center, eye, [0, 1, 0])

        # Cache initial camera for reset/load
        cam = scene_widget.scene.camera
        bbox_init = scene_widget.scene.bounding_box
        center_init = bbox_init.get_center()
        extent_init = bbox_init.get_max_extent()
        eye_init = center_init + np.array([0, 0, extent_init * 2])
        up_init = np.array([0, 1, 0])

        def _capture_cam():
            """Capture current camera parameters in a version-tolerant way."""
            try:
                eye = np.asarray(cam.get_position())
                look_at = np.asarray(cam.get_look_at())
                up = np.asarray(cam.get_up_vector())
            except Exception:
                eye, look_at, up = eye_init, center_init, up_init
            fov = getattr(cam, "get_field_of_view", lambda: None)()
            aspect = getattr(cam, "get_aspect_ratio", lambda: None)()
            near = getattr(cam, "get_near", lambda: None)()
            far = getattr(cam, "get_far", lambda: None)()
            fov_type = getattr(cam, "get_field_of_view_type", lambda: None)()
            print(
                "[Open3D][gui] Capture cam:",
                f"eye={eye}, look_at={look_at}, up={up}, fov={fov}, aspect={aspect}, near={near}, far={far}, fov_type={fov_type}",
            )
            return {
                "eye": eye,
                "look_at": look_at,
                "up": up,
                "fov": fov,
                "aspect": aspect,
                "near": near,
                "far": far,
                "fov_type": fov_type,
            }

        camera_state = {
            "initial": {
                "eye": eye_init,
                "look_at": center_init,
                "up": up_init,
                "fov": getattr(cam, "get_field_of_view", lambda: None)(),
                "aspect": getattr(cam, "get_aspect_ratio", lambda: None)(),
                "near": getattr(cam, "get_near", lambda: None)(),
                "far": getattr(cam, "get_far", lambda: None)(),
                "fov_type": getattr(cam, "get_field_of_view_type", lambda: None)(),
            },
            "saved": _capture_cam(),
        }

        def _apply_projection(saved):
            """Apply saved projection if compatible; otherwise skip quietly."""
            fov = saved.get("fov", None)
            aspect = saved.get("aspect", None)
            near = saved.get("near", None)
            far = saved.get("far", None)
            fov_type = saved.get("fov_type", None)
            if fov is None or near is None or far is None:
                return
            aspect_eff = (
                aspect
                if aspect not in (None, 0)
                else float(self.width) / float(max(1, self.height))
            )
            try:
                cam.set_projection(
                    float(fov),
                    float(aspect_eff),
                    float(near),
                    float(far),
                    fov_type if fov_type is not None else cam.FovType.Vertical,
                )
            except Exception as exc:
                print(f"[Open3D][gui] Projection apply failed: {exc}")

        state = {
            "idx": 0,
            "playing": True,  # match legacy default: start playing
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def _record_frame(i: int, force: bool = False) -> None:
            nonlocal video_writer, frame_dir, video_path
            if video_writer is not None:
                if not (force or (i % record_cfg.every_n) == 0):
                    return
                try:
                    img = _grab_frame()
                    if img is None:
                        return
                    frame = np.asarray(img)
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    video_writer.write(frame)
                    return
                except BrokenPipeError:
                    if frame_dir is None:
                        frame_dir = os.path.splitext(video_path or "frames")[0]
                        os.makedirs(frame_dir, exist_ok=True)
                        print(
                            "[Open3D] GUI: ffmpeg stream failed (broken pipe); "
                            f"falling back to frame directory {frame_dir}"
                        )
                    if video_writer.stderr_log:
                        print(f"[Open3D] ffmpeg stderr: {video_writer.stderr_log}")
                    video_writer = None
                except Exception as exc:
                    if frame_dir is None:
                        frame_dir = os.path.splitext(video_path or "frames")[0]
                        os.makedirs(frame_dir, exist_ok=True)
                        print(
                            f"[Open3D] GUI: ffmpeg stream failed ({exc}); "
                            f"falling back to frame directory {frame_dir}"
                        )
                    if video_writer and video_writer.stderr_log:
                        print(f"[Open3D] ffmpeg stderr: {video_writer.stderr_log}")
                    video_writer = None

            if frame_dir is None:
                return

            if force or (i % record_cfg.every_n) == 0:
                img = _grab_frame()
                if img is None:
                    return
                frame_np = np.asarray(img)
                if frame_np.ndim == 3 and frame_np.shape[2] == 4:
                    frame_np = frame_np[:, :, :3]
                if frame_np.dtype != np.uint8:
                    frame_np = np.clip(frame_np * 255.0, 0.0, 255.0).astype(np.uint8)
                img_to_save = o3d.geometry.Image(frame_np)
                fname = os.path.join(frame_dir, f"{record_cfg.prefix}{i:05d}.png")
                o3d.io.write_image(fname, img_to_save)

        def update_frame(i: int) -> None:
            i = max(0, min(scene_data.num_frames - 1, i))
            state["idx"] = i
            populate(i)
            _record_frame(i)

        _record_frame(0, force=True)

        _ECR = getattr(gui, "EventCallbackResult", None)
        _HANDLED = _ECR.HANDLED if _ECR else True
        _IGNORED = _ECR.IGNORED if _ECR else False
        _KEYTYPE_DOWN = getattr(gui.KeyEvent, "DOWN", None)
        _KEYTYPE_UP = getattr(gui.KeyEvent, "UP", None)
        _TYPE_ENUM = getattr(gui.KeyEvent, "Type", None)
        _KEY_DOWN_ENUM = getattr(_TYPE_ENUM, "DOWN", None) if _TYPE_ENUM else None
        _KEY_UP_ENUM = getattr(_TYPE_ENUM, "UP", None) if _TYPE_ENUM else None
        _debug_count = {"n": 0}

        def _handle_key(event):
            evt_type = getattr(event, "type", None)
            key = getattr(event, "key", None)
            if _debug_count["n"] < 5:
                print(f"[Open3D][gui] key event: type={evt_type} key={key}")
                _debug_count["n"] += 1

            # Only act on key-down (ignore repeats/up to avoid double toggles)
            is_down = False
            if _KEYTYPE_DOWN is not None and evt_type == _KEYTYPE_DOWN:
                is_down = True
            if _KEY_DOWN_ENUM is not None and evt_type == _KEY_DOWN_ENUM:
                is_down = True
            if isinstance(evt_type, int):
                # Many builds use 0 for DOWN, 2 for UP; treat only 0 as down
                is_down = evt_type == 0
            if not is_down:
                return _IGNORED

            # Key mapping aligned with legacy viewer
            space = getattr(gui.KeyName, "SPACE", None)
            left = getattr(gui.KeyName, "LEFT", None)
            right = getattr(gui.KeyName, "RIGHT", None)
            home = getattr(gui.KeyName, "HOME", None)
            esc = getattr(gui.KeyName, "ESCAPE", None)
            reset_cam = getattr(gui.KeyName, "R", ord("R"))
            capture_cam = getattr(gui.KeyName, "C", ord("C"))
            load_cam = getattr(gui.KeyName, "L", ord("L"))
            print_cam = getattr(gui.KeyName, "V", ord("V"))
            snapshot_key = getattr(gui.KeyName, "S", ord("S"))

            # Build tolerant lookup sets (fallback to GLFW key codes where possible)
            left_keys = {left, 263, 260}  # LEFT, GLFW_LEFT
            right_keys = {right, 262, 261}  # RIGHT, GLFW_RIGHT
            home_keys = {home, ord("h"), ord("H")}
            quit_keys = {esc, ord("q"), ord("Q")}
            reset_keys = {reset_cam, ord("r"), ord("R")}
            capture_keys = {capture_cam, ord("c"), ord("C")}
            load_keys = {load_cam, ord("l"), ord("L")}
            print_keys = {print_cam, ord("v"), ord("V")}
            snapshot_keys = {snapshot_key, ord("s"), ord("S")}

            if key == space or key == ord(" "):
                state["playing"] = not state["playing"]
                state["last_tick"] = time.time()
                # print(f"[Open3D][gui] Playing={state['playing']}")
                return _HANDLED
            if key in left_keys:
                update_frame(state["idx"] - 1)
                return _HANDLED
            if key in right_keys:
                update_frame(state["idx"] + 1)
                return _HANDLED
            if key in home_keys:
                update_frame(0)
                return _HANDLED
            if key in snapshot_keys:
                _record_frame(state["idx"], force=True)
                print(f"[Open3D][gui] Snapshot at frame {state['idx']}")
                return _HANDLED
            if key in quit_keys:
                state["playing"] = False
                app.post_to_main_thread(window, window.close) if hasattr(
                    app, "post_to_main_thread"
                ) else window.close()
                return _HANDLED
            if key in reset_keys:
                try:
                    _apply_projection(camera_state["initial"])
                    cam.look_at(
                        camera_state["initial"]["look_at"],
                        camera_state["initial"]["eye"],
                        camera_state["initial"]["up"],
                    )
                    if hasattr(scene_widget, "force_redraw"):
                        scene_widget.force_redraw()
                    else:
                        scene_widget.scene.force_redraw()
                    try:
                        pos = cam.get_position()
                        la = cam.get_look_at()
                        up_vec = cam.get_up_vector()
                        print(
                            f"[Open3D][gui] Camera reset. pos={pos}, look_at={la}, up={up_vec}"
                        )
                    except Exception:
                        print("[Open3D][gui] Camera reset (camera query unavailable).")
                except Exception as exc:
                    print(f"[Open3D][gui] Camera reset failed: {exc}")
                return _HANDLED
            if key in capture_keys:
                camera_state["saved"] = _capture_cam()
                print("[Open3D][gui] Camera captured.")
                return _HANDLED
            if key in load_keys:
                try:
                    saved = camera_state["saved"]
                    _apply_projection(saved)
                    cam.look_at(saved["look_at"], saved["eye"], saved["up"])
                    if hasattr(scene_widget, "force_redraw"):
                        scene_widget.force_redraw()
                    else:
                        scene_widget.scene.force_redraw()
                    try:
                        pos = cam.get_position()
                        la = cam.get_look_at()
                        up_vec = cam.get_up_vector()
                        print(
                            f"[Open3D][gui] Camera loaded. pos={pos}, look_at={la}, up={up_vec}"
                        )
                    except Exception:
                        print("[Open3D][gui] Camera loaded (camera query unavailable).")
                except Exception as exc:
                    print(f"[Open3D][gui] Load failed: {exc}")
                return _HANDLED
            if key in print_keys:
                try:
                    print("[Open3D][gui] Camera matrices:")
                    if hasattr(cam, "get_view_matrix"):
                        print("View:\n", cam.get_view_matrix())
                    if hasattr(cam, "get_projection_matrix"):
                        print("Proj:\n", cam.get_projection_matrix())
                    try:
                        print("Eye:", cam.get_position())
                        print("Look-at:", cam.get_look_at())
                        print("Up:", cam.get_up_vector())
                    except Exception:
                        pass
                except Exception as exc:
                    print(f"[Open3D][gui] Print failed: {exc}")
                return _HANDLED
            return _IGNORED

        # Attach to both window and scene_widget to handle focus differences
        if hasattr(window, "set_on_key"):
            window.set_on_key(_handle_key)
        if hasattr(scene_widget, "set_on_key"):
            scene_widget.set_on_key(_handle_key)

        def on_tick(_dt: float) -> bool:
            now = time.time()
            dt_now = state["dt_seq"][min(state["idx"], len(state["dt_seq"]) - 1)]
            if state["playing"] and (now - state["last_tick"] >= dt_now):
                nxt = state["idx"] + 1
                if nxt >= scene_data.num_frames:
                    nxt = 0 if loop else scene_data.num_frames - 1
                    state["playing"] = state["playing"] and loop
                update_frame(nxt)
                state["last_tick"] = now
            return True

        if hasattr(app, "set_on_tick"):
            app.set_on_tick(on_tick)
            print(
                f"{window_name} (gui): Space=Play/Pause  ←/→=Step  H=Home  S=Snapshot  R/C/L/V camera controls  Q=close window"
            )
            try:
                app.run()
            finally:
                if video_writer is not None:
                    video_writer.close()
            return

        # Fallback: manual ticking
        if hasattr(app, "run_one_tick"):
            print(
                f"{window_name} (gui, manual loop): Space=Play/Pause  ←/→=Step  H=Home  S=Snapshot  R/C/L/V camera controls  Q=close window"
            )

            def on_close():
                state["playing"] = False
                return True

            if hasattr(window, "set_on_close"):
                window.set_on_close(on_close)

            vis_attr = getattr(window, "is_visible", None)

            def _window_visible():
                if callable(vis_attr):
                    try:
                        return bool(vis_attr())
                    except Exception:
                        return False
                return bool(vis_attr) if vis_attr is not None else True

            while _window_visible():
                now = time.time()
                dt_now = state["dt_seq"][min(state["idx"], len(state["dt_seq"]) - 1)]
                if state["playing"] and (now - state["last_tick"] >= dt_now):
                    nxt = state["idx"] + 1
                    if nxt >= scene_data.num_frames:
                        nxt = 0 if loop else scene_data.num_frames - 1
                        state["playing"] = state["playing"] and loop
                    update_frame(nxt)
                    state["last_tick"] = now
                app.run_one_tick()
                time.sleep(0.01)
            if video_writer is not None:
                video_writer.close()
            return

        raise RuntimeError(
            "[Open3D] GUI ticking not available in this build. "
            "Use backend='legacy' or upgrade Open3D."
        )


def _get_robot_colors(
    num_robots: int,
    robot_colors: str | Array | np.ndarray | None,
    default_color: tuple[float, float, float] | tuple[float, float, float, float] = (
        0.1,
        0.45,
        1.0,
    ),
) -> np.ndarray:
    """Get colors for each robot.

    Args:
        num_robots: Number of robots
        robot_colors: "same" (use default), colormap name, or explicit color array/list
        default_color: Default color if robot_colors is None or "same". Can include alpha.

    Returns:
        Array of RGBA colors, one per robot
    """
    if isinstance(robot_colors, (list, tuple)):
        raise TypeError(
            "robot_colors must be a colormap string or an array of shape (N, 3) or (N, 4)."
        )

    default_palette = _ensure_rgba(np.asarray(default_color, dtype=np.float64))
    if robot_colors is None or robot_colors == "same":
        palette = np.tile(default_palette[:1], (num_robots, 1))
        return palette[:num_robots]

    palette = _normalize_color_palette(
        robot_colors,
        num_robots,
        default_palette=default_palette,
    )
    return palette
