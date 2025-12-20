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
import time
from typing import Dict, List, Optional, Tuple, Union

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Track initialization of the Open3D GUI app for older versions without is_running.
_GUI_APP_INITIALIZED = False

from soromox.rendering.base import BaseContinuumSoftRobotRenderer


# ======================================================================================
# Geometry helper functions (module-level, stateless)
# ======================================================================================


def _make_polyline_lineset(
    points_np: np.ndarray,
    color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
) -> "o3d.geometry.LineSet":
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
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: int = 48,
    apply_color: bool = True,
    apply_translation: bool = True,
) -> "o3d.geometry.TriangleMesh":
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
    color: Tuple[float, float, float] = (0.1, 0.45, 1.0),
    resolution: int = 16,
    apply_color: bool = True,
    apply_translation: bool = True,
) -> "o3d.geometry.TriangleMesh":
    """Create a colored sphere at center_xyz with given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
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
    color: Tuple[float, float, float],
    resolution: int = 20,
    apply_color: bool = True,
) -> "o3d.geometry.TriangleMesh":
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
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    resolution: int = 16,
) -> "o3d.geometry.TriangleMesh":
    """Create a colored sphere marking a target point."""
    return _make_sphere(center_xyz, radius, color, resolution)


def _make_obstacle_sphere(
    center_xyz: np.ndarray,
    radius: float,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    resolution: int = 24,
) -> "o3d.geometry.TriangleMesh":
    """Create a colored obstacle sphere."""
    return _make_sphere(center_xyz, radius, color, resolution)


def _split_counts_by_lengths(num_points: int, L: np.ndarray) -> List[int]:
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


def _expand_colors(
    seg_colors: Tuple[Tuple[float, float, float], ...], S: int
) -> List[Tuple[float, float, float]]:
    """Ensure we have at least S colors; cycle if not."""
    palette = list(seg_colors)
    if len(palette) == 0:
        palette = [(0.1, 0.45, 1.0)]
    return [palette[i % len(palette)] for i in range(S)]


def _mesh_center(mesh: "o3d.geometry.TriangleMesh") -> np.ndarray:
    """Approximate center: mean of vertices."""
    return np.asarray(mesh.vertices).mean(axis=0)


# ======================================================================================
# Open3D Renderer Class
# ======================================================================================


class Open3DRenderer(BaseContinuumSoftRobotRenderer):
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
        robot,
        width: int = 1280,
        height: int = 800,
        num_points: int = 80,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seg_colors: Tuple[Tuple[float, float, float], ...] = (
            (0.1, 0.45, 1.0),
            (0.0, 0.6, 0.2),
        ),
        body_alpha: float = 1.0,
        backbone_style: str = "spheres",
        tube_resolution: int = 20,
        sphere_resolution: int = 32,
        tendon_color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
        tendon_line_width: float = 2.0,
        camera_margin_ratio: float = 0.05,
        grid_spacing: Tuple[float, float] = (0.5, 0.5),
        base_plate_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_plate_radius_scale: float = 2.0,
        base_plate_thickness: float = 5e-2 * 1.3,
        base_offsets: Optional[Array] = None,
        robot_colors: Optional[
            Union[str, Tuple[Tuple[float, float, float], ...], List[Tuple[float, float, float]]]
        ] = None,
        viewer_backend: str = "auto",
    ):
        """Initialize Open3D renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Window width in pixels
            height: Window height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            seg_colors: Tuple of RGB colors for each segment
            body_alpha: Opacity for robot meshes (0.0 transparent, 1.0 opaque)
            backbone_style: "spheres" (default) or "tube" (swept cylinder segments)
            tube_resolution: Radial resolution for tube segments
            sphere_resolution: Resolution for backbone spheres
            tendon_color: RGB color for tendon lines
            tendon_line_width: Width of tendon lines
            camera_margin_ratio: Margin ratio for camera bounding box
            grid_spacing: (x, y) spacing between robot bases for batched rendering
            base_plate_color: RGB color for the base plate geometry
            base_plate_radius_scale: Multiplier applied to the maximum segment radius to size the base plate
            base_plate_thickness: Absolute thickness of the base plate geometry
            base_offsets: Explicit base offsets of shape (N, 2) or (N, 3) for batched rendering
            robot_colors: Color configuration when rendering multiple robots. Can be:
                - None or "same": All robots use first segment color
                - Colormap name (e.g., "viridis", "plasma"): Distinct colors per robot
                - Tuple/list of RGB colors: Explicit colors (cycled if fewer than N)
            viewer_backend: "auto" (default), "gui" (force SceneWidget), or "legacy"
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install with: pip install open3d"
            )

        super().__init__(robot, width, height, num_points, background_color)

        self.seg_colors = seg_colors
        self.sphere_resolution = sphere_resolution
        self.tendon_color = tendon_color
        self.tendon_line_width = tendon_line_width
        self.camera_margin_ratio = camera_margin_ratio
        self.grid_spacing = grid_spacing
        self.base_plate_color = tuple(base_plate_color)
        self.base_plate_radius_scale = float(base_plate_radius_scale)
        self.base_plate_thickness = float(base_plate_thickness)
        self._base_offsets = base_offsets
        self._robot_colors_config = robot_colors
        self.body_alpha = float(np.clip(body_alpha, 0.0, 1.0))
        self.backbone_style = backbone_style
        self.tube_resolution = int(tube_resolution)
        self._warned_tube_legacy = False
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
        """Always True for Open3D renderer."""
        return True

    def _extract_positions(self, poses: Array) -> Array:
        """Extract xyz from SE(3) matrices."""
        return poses[:, :3, 3]

    def compute_tendon_curves(self, q: Array) -> Optional[Array]:
        """Compute tendon paths if robot supports tendons.

        Args:
            q: Robot configuration

        Returns:
            Array of shape (n_tendons, num_points, 3) or None
        """
        if not self._has_tendons:
            return None
        s_ps = jnp.linspace(0.0, self.L_max, self.num_points)
        return self._batched_fk_tendons(q, s_ps)

    def _make_mesh_material(
        self, color: Tuple[float, float, float]
    ) -> "o3d.visualization.rendering.MaterialRecord":
        """Create an Open3D material with optional transparency."""
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = (
            "defaultLitTransparency" if self.body_alpha < 0.999 else "defaultLit"
        )
        mat.base_color = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            self.body_alpha,
        )
        return mat

    def _blend_with_background(
        self, color: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Approximate transparency for the legacy visualizer by blending with the background."""
        if self.body_alpha >= 0.999:
            return color
        bg = np.asarray(self.background_color, dtype=np.float64)
        col = np.asarray(color, dtype=np.float64)
        blended = self.body_alpha * col + (1.0 - self.body_alpha) * bg
        return tuple(np.clip(blended, 0.0, 1.0).tolist())

    def _frame_intervals_from_ts(
        self, ts: Array, playback_speed: float
    ) -> np.ndarray:
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

    # -------------------------------------------------------------------------
    # Public rendering API
    # -------------------------------------------------------------------------

    def render_frame(self, q: Array) -> np.ndarray:
        """Render single configuration headlessly, return RGB array.

        Args:
            q: Robot configuration

        Returns:
            RGB image as numpy array of shape (height, width, 3), dtype uint8
        """
        curve = np.array(self.compute_backbone_curve(q), dtype=np.float64)
        tendon_curves = None
        if self._has_tendons:
            tendon_curves = np.array(self.compute_tendon_curves(q), dtype=np.float64)

        # Create offscreen renderer
        render = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        render.scene.set_background(np.array([*self.background_color, 1.0]))

        # Add geometries
        self._add_geometries_to_scene(render.scene, curve, tendon_curves)

        # Set up camera
        bounds = render.scene.bounding_box
        center = bounds.get_center()
        extent = bounds.get_max_extent()
        render.setup_camera(60.0, center, center + [0, 0, extent * 2], [0, 1, 0])

        # Render
        img = render.render_to_image()
        return np.asarray(img)

    def _add_geometries_to_scene(
        self,
        scene,
        curve: np.ndarray,
        tendon_curves: Optional[np.ndarray] = None,
    ) -> None:
        """Add robot geometries to Open3D scene."""
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        S = int(L_np.size)

        counts = _split_counts_by_lengths(curve.shape[0], L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        seg_cols = _expand_colors(self.seg_colors, S)

        material_cache: Dict[
            Tuple[float, float, float], o3d.visualization.rendering.MaterialRecord
        ] = {}

        def mat_for(color: Tuple[float, float, float]):
            key = tuple(color)
            if key not in material_cache:
                material_cache[key] = self._make_mesh_material(key)
            return material_cache[key]

        # Base plate
        base_color = self.base_plate_color
        base_mesh = _make_base_plate(
            curve[0],
            radius=float(self.base_plate_radius_scale * r_seg.max()),
            thickness=self.base_plate_thickness,
            color=base_color,
            apply_color=False,
        )
        scene.add_geometry("base", base_mesh, mat_for(base_color))

        # Backbone
        for s in range(S):
            c0, c1 = int(starts[s]), int(ends[s])
            if self.backbone_style == "tube" and c1 - c0 >= 1:
                for p in range(c0, c1 - 1):
                    cyl = _make_cylinder_between(
                        curve[p],
                        curve[p + 1],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
                        color=seg_cols[s],
                        resolution=self.tube_resolution,
                        apply_color=False,
                    )
                    scene.add_geometry(f"tube_{s}_{p}", cyl, mat_for(seg_cols[s]))
            else:
                for p in range(c0, c1):
                    sp = _make_sphere(
                        curve[p],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
                        color=seg_cols[s],
                        resolution=self.sphere_resolution,
                        apply_color=False,
                    )
                    scene.add_geometry(f"sphere_{s}_{p}", sp, mat_for(seg_cols[s]))

        # Tendons
        if tendon_curves is not None:
            for i in range(tendon_curves.shape[0]):
                ls = _make_polyline_lineset(tendon_curves[i], color=self.tendon_color)
                mat_line = o3d.visualization.rendering.MaterialRecord()
                mat_line.shader = "unlitLine"
                mat_line.line_width = self.tendon_line_width
                scene.add_geometry(f"tendon_{i}", ls, mat_line)

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

    def _compute_segment_splits(
        self, P: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Compute segment start/end indices and radii for geometry creation."""
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        S = int(L_np.size)
        counts = _split_counts_by_lengths(P, L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        return starts, ends, r_seg, S

    def _build_robot_geometry(
        self,
        vis,
        curve0: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        r_seg: np.ndarray,
        seg_cols: List[Tuple[float, float, float]],
        color_override: Optional[Tuple[float, float, float]] = None,
    ):
        """Add base and backbone meshes for one robot; return handles."""
        base_color = self._blend_with_background(self.base_plate_color)
        base_mesh = _make_base_plate(
            curve0[0],
            radius=float(self.base_plate_radius_scale * r_seg.max()),
            thickness=self.base_plate_thickness,
            color=base_color,
        )
        vis.add_geometry(base_mesh)

        spheres_groups: List[List] = []
        for s in range(len(starts)):
            seg_spheres = []
            c0, c1 = int(starts[s]), int(ends[s])
            raw_color = color_override if color_override is not None else seg_cols[s]
            seg_color = self._blend_with_background(raw_color)
            if self.backbone_style == "tube" and c1 - c0 >= 1:
                for p in range(c0, c1 - 1):
                    cyl = _make_cylinder_between(
                        curve0[p],
                        curve0[p + 1],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
                        color=seg_color,
                        resolution=self.tube_resolution,
                    )
                    seg_spheres.append(cyl)
                    vis.add_geometry(cyl)
            else:
                for p in range(c0, c1):
                    sp = _make_sphere(
                        curve0[p],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
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
        spheres_groups: List[List],
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> None:
        """Translate base and backbone spheres to a new curve position."""
        delta = curve[0] - _mesh_center(base_mesh)
        base_mesh.translate(delta, relative=True)
        vis.update_geometry(base_mesh)

        for s in range(len(spheres_groups)):
            c0, c1 = int(starts[s]), int(ends[s])
            seg_spheres = spheres_groups[s]
            for i_local, p in enumerate(range(c0, min(c1, c0 + len(seg_spheres)))):
                mesh = seg_spheres[i_local]
                delta = curve[p] - _mesh_center(mesh)
                mesh.translate(delta, relative=True)
                vis.update_geometry(mesh)

    def _register_key_callbacks(
        self,
        vis,
        ctrl,
        state: Dict[str, Union[int, bool, float]],
        update_frame,
        save_frame_fn,
        initial_cam,
        saved_cam_holder: List,
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

    def show(self, q: Array) -> None:
        """Display single frame interactively with Open3D viewer.

        Args:
            q: Robot configuration
        """
        curve = np.array(self.compute_backbone_curve(q), dtype=np.float64)
        tendon_curves = None
        if self._has_tendons:
            tendon_curves = np.array(self.compute_tendon_curves(q), dtype=np.float64)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Robot Visualization",
            width=self.width,
            height=self.height,
        )

        opt = vis.get_render_option()
        opt.background_color = np.array(self.background_color, dtype=np.float64)
        opt.line_width = self.tendon_line_width

        self._add_geometries_to_visualizer(vis, curve, tendon_curves)

        vis.run()
        vis.destroy_window()

    def _add_geometries_to_visualizer(
        self,
        vis,
        curve: np.ndarray,
        tendon_curves: Optional[np.ndarray] = None,
    ) -> None:
        """Add robot geometries to Open3D visualizer."""
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        S = int(L_np.size)

        counts = _split_counts_by_lengths(curve.shape[0], L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        seg_cols = _expand_colors(self.seg_colors, S)

        # Base plate
        base_color = self._blend_with_background(self.base_plate_color)
        base_mesh = _make_base_plate(
            curve[0],
            radius=float(self.base_plate_radius_scale * r_seg.max()),
            thickness=self.base_plate_thickness,
            color=base_color,
        )
        vis.add_geometry(base_mesh)

        # Backbone
        for s in range(S):
            c0, c1 = int(starts[s]), int(ends[s])
            if self.backbone_style == "tube" and c1 - c0 >= 1:
                for p in range(c0, c1 - 1):
                    cyl = _make_cylinder_between(
                        curve[p],
                        curve[p + 1],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
                        color=seg_cols[s],
                        resolution=self.tube_resolution,
                    )
                    vis.add_geometry(cyl)
            else:
                for p in range(c0, c1):
                    sp = _make_sphere(
                        curve[p],
                        radius=float(r_seg[min(s, len(r_seg) - 1)]),
                        color=seg_cols[s],
                        resolution=self.sphere_resolution,
                    )
                    vis.add_geometry(sp)

        # Tendons
        if tendon_curves is not None:
            for i in range(tendon_curves.shape[0]):
                ls = _make_polyline_lineset(tendon_curves[i], color=self.tendon_color)
                vis.add_geometry(ls)

    def _render_backbone_batch(
        self,
        all_curves: np.ndarray,
        ts: np.ndarray,
        color_overrides: Optional[List[Optional[Tuple[float, float, float]]]],
        record_dir: Optional[str],
        record_every_n: int,
        record_prefix: str,
        loop: bool,
        playback_speed: float,
        window_name: str,
        extra_init=None,
        extra_update=None,
    ) -> None:
        """Shared rendering routine for a batch of backbone curves.

        all_curves: (N, T, P, 3) array of backbone points.
        color_overrides: list of per-robot colors (or None to use segment colors).
        extra_init/extra_update: optional hooks for additional geometry (tendons, etc.).
        """
        N, T, P, _ = all_curves.shape

        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)
            print(f"[Open3D] Recording frames to: {record_dir}")

        vis, ctrl = self._create_visualizer(window_name)

        starts, ends, r_seg, S = self._compute_segment_splits(P)
        seg_cols = _expand_colors(self.seg_colors, S)

        # Create geometry for each robot
        all_robot_spheres: List[List[List]] = []
        all_base_plates: List = []
        for robot_idx in range(N):
            curve0 = all_curves[robot_idx, 0]
            color_override = None
            if color_overrides is not None and len(color_overrides) > robot_idx:
                color_override = color_overrides[robot_idx]

            base_mesh, robot_spheres = self._build_robot_geometry(
                vis,
                curve0,
                starts,
                ends,
                r_seg,
                seg_cols,
                color_override=color_override,
            )
            all_base_plates.append(base_mesh)
            all_robot_spheres.append(robot_spheres)

        extra_state = extra_init(vis) if extra_init else None

        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam_holder = [copy.deepcopy(initial_cam)]

        dt_seq = self._frame_intervals_from_ts(ts, playback_speed)
        state = {
            "idx": 0,
            "playing": True,
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def _clamp(i: int) -> int:
            return max(0, min(T - 1, i))

        def _save_frame(i: int, force: bool = False) -> None:
            if record_dir is None:
                if force:
                    vis.capture_screen_image(f"frame_{i:05d}.png", do_render=True)
                return
            if force or (i % record_every_n) == 0:
                fname = os.path.join(record_dir, f"{record_prefix}{i:05d}.png")
                vis.capture_screen_image(fname, do_render=True)

        def update_frame(i: int) -> None:
            i = _clamp(i)
            state["idx"] = i

            for robot_idx in range(N):
                curve = all_curves[robot_idx, i]
                base_mesh = all_base_plates[robot_idx]
                robot_spheres = all_robot_spheres[robot_idx]
                self._update_robot_geometry(
                    vis, curve, base_mesh, robot_spheres, starts, ends
                )

            if extra_update is not None:
                extra_update(vis, i, extra_state)

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
                if nxt >= T:
                    if loop:
                        nxt = 0
                    else:
                        state["playing"] = False
                        nxt = T - 1
                update_frame(nxt)
                state["last_tick"] = now
            vis.update_renderer()
            time.sleep(0.001)

    def render_sequence(  # type: ignore[override]
        self,
        ts: Array,
        q_ts: Array,
        playback_speed: float = 1.0,
        output_path: Optional[str] = None,
        loop: bool = False,
        record_every_n: int = 1,
        record_prefix: str = "frame_",
        target_point: Optional[Tuple[float, float, float]] = None,
        target_radius: float = 0.01,
        target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        obstacles: Optional[
            List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]
        ] = None,
        moving_spheres: Optional[
            List[Tuple[np.ndarray, float, Tuple[float, float, float]]]
        ] = None,
    ) -> None:
        """Dispatch to single-robot or batched rendering based on q_ts shape."""
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is not installed. Install with: pip install open3d"
            )

        q_ts_arr = jnp.asarray(q_ts)

        if q_ts_arr.ndim == 3:
            return self._render_sequence_batched(
                ts=ts,
                q_ts=q_ts_arr,
                playback_speed=playback_speed,
                output_path=output_path,
                loop=loop,
                record_every_n=record_every_n,
                record_prefix=record_prefix,
                target_point=target_point,
                target_radius=target_radius,
                target_color=target_color,
                obstacles=obstacles,
                moving_spheres=moving_spheres,
            )

        if q_ts_arr.ndim != 2:
            raise ValueError(
                f"q_ts must have shape (T, DOF) or (N, T, DOF); got shape {q_ts_arr.shape}"
            )

        return self._render_sequence_single(
            ts=ts,
            q_ts=q_ts_arr,
            playback_speed=playback_speed,
            output_path=output_path,
            loop=loop,
            record_every_n=record_every_n,
            record_prefix=record_prefix,
            target_point=target_point,
            target_radius=target_radius,
            target_color=target_color,
            obstacles=obstacles,
            moving_spheres=moving_spheres,
        )

    def _render_sequence_single(
        self,
        ts: Array,
        q_ts: Array,
        playback_speed: float = 1.0,
        output_path: Optional[str] = None,
        loop: bool = False,
        record_every_n: int = 1,
        record_prefix: str = "frame_",
        target_point: Optional[Tuple[float, float, float]] = None,
        target_radius: float = 0.01,
        target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        obstacles: Optional[
            List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]
        ] = None,
        moving_spheres: Optional[
            List[Tuple[np.ndarray, float, Tuple[float, float, float]]]
        ] = None,
    ) -> None:
        """Native Open3D animated playback with keyboard controls."""
        record_dir = output_path  # Use output_path as record directory
        print("[Open3D] Initializing...")

        q_ts = jnp.asarray(q_ts)
        ts = np.asarray(ts).reshape((-1,))
        T = q_ts.shape[0]

        # Precompute curves with vmap over time for efficiency, using batched FK when available
        base_offsets = jnp.zeros((1, 3), dtype=q_ts.dtype)
        all_robot_curves_time_first = jax.vmap(
            lambda q: self.compute_backbone_curves_batched(q[None, :], base_offsets)
        )(q_ts)  # (T, 1, P, 3)
        all_robot_curves = np.array(
            all_robot_curves_time_first[:, 0], dtype=np.float64
        )  # (T, P, 3)

        all_tendon_curves = None
        n_act = 0
        if self._has_tendons:
            all_tendon_curves = np.array(
                jax.vmap(self.compute_tendon_curves)(q_ts), dtype=np.float64
            )
            n_act = all_tendon_curves.shape[1]

        color_overrides = [None]  # Use per-segment colors for single robot
        if self.viewer_backend == "gui" and output_path is not None:
            raise NotImplementedError(
                "Recording to disk is not supported with viewer_backend='gui'. "
                "Use viewer_backend='legacy' for recording."
            )

        tendon_curves_gui = (
            all_tendon_curves[None, ...] if all_tendon_curves is not None else None
        )

        def extra_init(vis):
            # Target point
            if target_point is not None:
                tgt = np.array(target_point, dtype=np.float64).reshape(3)
                target_mesh = _make_target_sphere(tgt, target_radius, target_color)
                vis.add_geometry(target_mesh)

            # Static obstacles
            if obstacles:
                for ctr, rad, col in obstacles:
                    ctr_np = np.array(ctr, dtype=np.float64).reshape(3)
                    mesh = _make_obstacle_sphere(ctr_np, float(rad), col)
                    vis.add_geometry(mesh)

            dynamic_sphere_meshes = []
            dynamic_sphere_trajs = []
            if moving_spheres:
                for centers_T3, rad, col in moving_spheres:
                    centers_np = np.asarray(centers_T3, dtype=np.float64)
                    mesh0 = _make_sphere(
                        centers_np[0],
                        float(rad),
                        col,
                        max(12, self.sphere_resolution // 2),
                    )
                    dynamic_sphere_meshes.append(mesh0)
                    dynamic_sphere_trajs.append(centers_np)
                    vis.add_geometry(mesh0)

            tendon_ls = []
            if all_tendon_curves is not None:
                for i in range(n_act):
                    ls = _make_polyline_lineset(
                        all_tendon_curves[0, i], self.tendon_color
                    )
                    tendon_ls.append(ls)
                    vis.add_geometry(ls)

            return {
                "tendon_ls": tendon_ls,
                "moving_meshes": dynamic_sphere_meshes,
                "moving_trajs": dynamic_sphere_trajs,
            }

        def extra_update(vis, frame_idx: int, extra_state):
            # Update tendons
            tendon_ls = extra_state["tendon_ls"]
            if all_tendon_curves is not None:
                for k in range(n_act):
                    t_pts = np.array(
                        all_tendon_curves[frame_idx, k], dtype=np.float64, copy=True
                    )
                    tendon_ls[k].points = o3d.utility.Vector3dVector(t_pts)
                    vis.update_geometry(tendon_ls[k])

            # Update moving spheres
            for mesh, traj in zip(
                extra_state["moving_meshes"], extra_state["moving_trajs"]
            ):
                j_idx = min(frame_idx, traj.shape[0] - 1)
                delta = traj[j_idx] - _mesh_center(mesh)
                mesh.translate(delta, relative=True)
                vis.update_geometry(mesh)
        if self.viewer_backend == "legacy":
            return self._render_backbone_batch(
                all_curves=all_robot_curves[None, ...],  # (1, T, P, 3)
                ts=ts,
                color_overrides=color_overrides,
                record_dir=record_dir,
                record_every_n=record_every_n,
                record_prefix=record_prefix,
                loop=loop,
                playback_speed=playback_speed,
                window_name="Robot Animation (Open3D)",
                extra_init=extra_init,
                extra_update=extra_update,
            )

        if output_path is None:
            return self._render_backbone_batch_gui(
                all_curves=all_robot_curves[None, ...],  # (1, T, P, 3)
                ts=ts,
                color_overrides=color_overrides,
                loop=loop,
                playback_speed=playback_speed,
                window_name="Robot Animation (Open3D)",
                tendon_curves=tendon_curves_gui,
                target_point=target_point,
                target_radius=target_radius,
                target_color=target_color,
                obstacles=obstacles,
                moving_spheres=moving_spheres,
            )

        return self._render_backbone_batch(
            all_curves=all_robot_curves[None, ...],  # (1, T, P, 3)
            ts=ts,
            color_overrides=color_overrides,
            record_dir=record_dir,
            record_every_n=record_every_n,
            record_prefix=record_prefix,
            loop=loop,
            playback_speed=playback_speed,
            window_name="Robot Animation (Open3D)",
            extra_init=extra_init,
            extra_update=extra_update,
        )

    # ------------------------------------------------------------------
    # New rendering backend (Open3D SceneWidget) for interactive playback
    # ------------------------------------------------------------------
    def _render_backbone_batch_gui(
        self,
        all_curves: np.ndarray,
        ts: np.ndarray,
        color_overrides: Optional[List[Optional[Tuple[float, float, float]]]],
        loop: bool,
        playback_speed: float,
        window_name: str,
        tendon_curves: Optional[np.ndarray] = None,
        target_point: Optional[Tuple[float, float, float]] = None,
        target_radius: float = 0.01,
        target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        obstacles: Optional[
            List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]
        ] = None,
        moving_spheres: Optional[
            List[Tuple[np.ndarray, float, Tuple[float, float, float]]]
        ] = None,
    ) -> None:
        """Interactive playback using Open3D's new rendering GUI backend."""
        try:
            import open3d.visualization.gui as gui
            import open3d.visualization.rendering as rendering
        except Exception as exc:  # pragma: no cover - caller selects backend
            raise RuntimeError(
                f"[Open3D] GUI backend unavailable ({exc}). "
                "Set viewer_backend='legacy' or install Open3D with GUI support."
            ) from exc

        N, T, P, _ = all_curves.shape
        starts, ends, r_seg, S = self._compute_segment_splits(P)
        seg_cols = _expand_colors(self.seg_colors, S)

        material_cache: Dict[
            Tuple[float, float, float], rendering.MaterialRecord
        ] = {}

        def mat_for(color: Tuple[float, float, float]):
            key = tuple(color)
            if key not in material_cache:
                material_cache[key] = self._make_mesh_material(key)
            return material_cache[key]

        def populate_scene(scene: rendering.Open3DScene, frame_idx: int):
            scene.clear_geometry()

            for robot_idx in range(N):
                curve = all_curves[robot_idx, frame_idx]
                color_override = None
                if color_overrides is not None and len(color_overrides) > robot_idx:
                    color_override = color_overrides[robot_idx]

                # Base
                base_color = self.base_plate_color
                base_mesh = _make_base_plate(
                    curve[0],
                    radius=float(self.base_plate_radius_scale * r_seg.max()),
                    thickness=self.base_plate_thickness,
                    color=base_color,
                    apply_color=False,
                    apply_translation=True,
                )
                scene.add_geometry(
                    f"base_{robot_idx}", base_mesh, mat_for(base_color)
                )

                # Backbone
                for s in range(S):
                    c0, c1 = int(starts[s]), int(ends[s])
                    raw_color = (
                        color_override if color_override is not None else seg_cols[s]
                    )
                    if self.backbone_style == "tube" and c1 - c0 >= 1:
                        for p in range(c0, c1 - 1):
                            cyl = _make_cylinder_between(
                                curve[p],
                                curve[p + 1],
                                radius=float(r_seg[min(s, len(r_seg) - 1)]),
                                color=raw_color,
                                resolution=self.tube_resolution,
                                apply_color=False,
                            )
                            scene.add_geometry(
                                f"tube_{robot_idx}_{s}_{p}", cyl, mat_for(raw_color)
                            )
                    else:
                        for p in range(c0, c1):
                            sp = _make_sphere(
                                curve[p],
                                radius=float(r_seg[min(s, len(r_seg) - 1)]),
                                color=raw_color,
                                resolution=self.sphere_resolution,
                                apply_color=False,
                                apply_translation=True,
                            )
                            scene.add_geometry(
                                f"sphere_{robot_idx}_{s}_{p}", sp, mat_for(raw_color)
                            )

                # Tendons
                if tendon_curves is not None:
                    robot_tendons = tendon_curves[robot_idx, frame_idx]
                    for k in range(robot_tendons.shape[0]):
                        ls = _make_polyline_lineset(
                            robot_tendons[k], color=self.tendon_color
                        )
                        mat_line = rendering.MaterialRecord()
                        mat_line.shader = "unlitLine"
                        mat_line.line_width = self.tendon_line_width
                        scene.add_geometry(
                            f"tendon_{robot_idx}_{k}", ls, mat_line
                        )

            # Static extras
            if target_point is not None:
                tgt_mesh = _make_target_sphere(target_point, target_radius, target_color)
                scene.add_geometry("target", tgt_mesh, mat_for(target_color))

            if obstacles:
                for obs_idx, (ctr, rad, col) in enumerate(obstacles):
                    mesh = _make_obstacle_sphere(np.asarray(ctr), float(rad), col)
                    scene.add_geometry(f"obstacle_{obs_idx}", mesh, mat_for(col))

            if moving_spheres:
                for mv_idx, (traj, rad, col) in enumerate(moving_spheres):
                    centers = np.asarray(traj, dtype=np.float64)
                    j_idx = min(frame_idx, centers.shape[0] - 1)
                    mesh = _make_sphere(
                        centers[j_idx],
                        float(rad),
                        col,
                        max(12, self.sphere_resolution // 2),
                        apply_color=False,
                        apply_translation=True,
                    )
                    scene.add_geometry(f"moving_{mv_idx}", mesh, mat_for(col))

        app = gui.Application.instance
        # Older Open3D versions lack is_running; use a guard flag instead.
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

        # Event return codes (old API used bools)
        _ECR = getattr(gui, "EventCallbackResult", None)
        _HANDLED = _ECR.HANDLED if _ECR else True
        _IGNORED = _ECR.IGNORED if _ECR else False

        populate_scene(scene_widget.scene, 0)
        bbox = scene_widget.scene.bounding_box
        center = bbox.get_center()
        extent = bbox.get_max_extent()
        try:
            scene_widget.scene.setup_camera(60.0, bbox, center)
        except Exception as exc_setup:
            try:
                eye = center + np.array([0, 0, extent * 2])
                scene_widget.scene.camera.look_at(center, eye, [0, 1, 0])
            except Exception as exc_lookat:
                raise RuntimeError(
                    "[Open3D] SceneWidget camera setup failed; new GUI viewer is not "
                    "supported by this Open3D build. Use viewer_backend='legacy' or "
                    "upgrade Open3D."
                ) from exc_lookat

        dt_seq = self._frame_intervals_from_ts(ts, playback_speed)
        state = {
            "idx": 0,
            "playing": True,
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def update_frame(i: int):
            i = max(0, min(T - 1, i))
            state["idx"] = i
            populate_scene(scene_widget.scene, i)

        def on_key(event):
            if event.type == gui.KeyEvent.DOWN:
                if event.key == gui.KeyName.SPACE:
                    state["playing"] = not state["playing"]
                    return _HANDLED
                if event.key == gui.KeyName.LEFT:
                    update_frame(state["idx"] - 1)
                    return _HANDLED
                if event.key == gui.KeyName.RIGHT:
                    update_frame(state["idx"] + 1)
                    return _HANDLED
            return _IGNORED

        window.set_on_key(on_key)

        def on_tick(dt: float) -> bool:
            now = time.time()
            dt_now = state["dt_seq"][min(state["idx"], len(state["dt_seq"]) - 1)]
            if state["playing"] and (now - state["last_tick"] >= dt_now):
                nxt = state["idx"] + 1
                if nxt >= T:
                    nxt = 0 if loop else T - 1
                    state["playing"] = state["playing"] and loop
                update_frame(nxt)
                state["last_tick"] = now
            return True  # keep ticking

        if hasattr(app, "set_on_tick"):
            app.set_on_tick(on_tick)
            print(
                f"{window_name} (new backend): Space=Play/Pause  ←/→=Step  Q=close window"
            )
            app.run()
            return

        # Older GUI builds: emulate ticking manually via run_one_tick if available.
        if hasattr(app, "run_one_tick"):
            print(
                f"{window_name} (new backend, manual loop): Space=Play/Pause  ←/→=Step  Q=close window"
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
                    if nxt >= T:
                        nxt = 0 if loop else T - 1
                        state["playing"] = state["playing"] and loop
                    update_frame(nxt)
                    state["last_tick"] = now
                app.run_one_tick()
                time.sleep(0.01)
            return

        raise RuntimeError(
            "[Open3D] GUI ticking not available in this build. "
            "Use viewer_backend='legacy' or upgrade Open3D."
        )


    def _render_backbone_batch(
        self,
        all_curves: np.ndarray,
        ts: np.ndarray,
        color_overrides: Optional[List[Optional[Tuple[float, float, float]]]],
        record_dir: Optional[str],
        record_every_n: int,
        record_prefix: str,
        loop: bool,
        playback_speed: float,
        window_name: str,
        extra_init=None,
        extra_update=None,
    ) -> None:
        """Shared rendering routine for a batch of backbone curves.

        all_curves: (N, T, P, 3) array of backbone points.
        color_overrides: list of per-robot colors (or None to use segment colors).
        extra_init/extra_update: optional hooks for additional geometry (tendons, etc.).
        """
        N, T, P, _ = all_curves.shape

        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)
            print(f"[Open3D] Recording frames to: {record_dir}")

        vis, ctrl = self._create_visualizer(window_name)

        starts, ends, r_seg, S = self._compute_segment_splits(P)
        seg_cols = _expand_colors(self.seg_colors, S)

        # Create geometry for each robot
        all_robot_spheres: List[List[List]] = []
        all_base_plates: List = []
        for robot_idx in range(N):
            curve0 = all_curves[robot_idx, 0]
            color_override = None
            if color_overrides is not None and len(color_overrides) > robot_idx:
                color_override = color_overrides[robot_idx]

            base_mesh, robot_spheres = self._build_robot_geometry(
                vis,
                curve0,
                starts,
                ends,
                r_seg,
                seg_cols,
                color_override=color_override,
            )
            all_base_plates.append(base_mesh)
            all_robot_spheres.append(robot_spheres)

        extra_state = extra_init(vis) if extra_init else None

        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam_holder = [copy.deepcopy(initial_cam)]

        dt_seq = self._frame_intervals_from_ts(ts, playback_speed)
        state = {
            "idx": 0,
            "playing": True,
            "last_tick": time.time(),
            "dt_seq": dt_seq,
        }

        def _clamp(i: int) -> int:
            return max(0, min(T - 1, i))

        def _save_frame(i: int, force: bool = False) -> None:
            if record_dir is None:
                if force:
                    vis.capture_screen_image(f"frame_{i:05d}.png", do_render=True)
                return
            if force or (i % record_every_n) == 0:
                fname = os.path.join(record_dir, f"{record_prefix}{i:05d}.png")
                vis.capture_screen_image(fname, do_render=True)

        def update_frame(i: int) -> None:
            i = _clamp(i)
            state["idx"] = i

            for robot_idx in range(N):
                curve = all_curves[robot_idx, i]
                base_mesh = all_base_plates[robot_idx]
                robot_spheres = all_robot_spheres[robot_idx]
                self._update_robot_geometry(
                    vis, curve, base_mesh, robot_spheres, starts, ends
                )

            if extra_update is not None:
                extra_update(vis, i, extra_state)

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
                if nxt >= T:
                    if loop:
                        nxt = 0
                    else:
                        state["playing"] = False
                        nxt = T - 1
                update_frame(nxt)
                state["last_tick"] = now
            vis.update_renderer()
            time.sleep(0.001)


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


    def _compute_segment_splits(
        self, P: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Compute segment start/end indices and radii for geometry creation."""
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        S = int(L_np.size)
        counts = _split_counts_by_lengths(P, L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        return starts, ends, r_seg, S


    def _build_robot_geometry(
        self,
        vis,
        curve0: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        r_seg: np.ndarray,
        seg_cols: List[Tuple[float, float, float]],
        color_override: Optional[Tuple[float, float, float]] = None,
    ):
        """Add base and backbone spheres for one robot; return handles."""
        base_color = self._blend_with_background(self.base_plate_color)
        base_mesh = _make_base_plate(
            curve0[0],
            radius=float(self.base_plate_radius_scale * r_seg.max()),
            thickness=self.base_plate_thickness,
            color=base_color,
        )
        vis.add_geometry(base_mesh)

        spheres_groups: List[List] = []
        for s in range(len(starts)):
            seg_spheres = []
            c0, c1 = int(starts[s]), int(ends[s])
            seg_color = color_override if color_override is not None else seg_cols[s]
            for p in range(c0, c1):
                sp = _make_sphere(
                    curve0[p],
                    radius=float(r_seg[min(s, len(r_seg) - 1)]),
                    color=seg_color,
                    resolution=self.sphere_resolution,
                )
                seg_spheres.append(sp)
                vis.add_geometry(sp)
            spheres_groups.append(seg_spheres)
        return base_mesh, spheres_groups



    def _render_sequence_batched(
        self,
        ts: Array,
        q_ts: Array,
        playback_speed: float = 1.0,
        output_path: Optional[str] = None,
        loop: bool = False,
        record_every_n: int = 1,
        record_prefix: str = "frame_",
        target_point: Optional[Tuple[float, float, float]] = None,
        target_radius: float = 0.01,
        target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        obstacles: Optional[
            List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]
        ] = None,
        moving_spheres: Optional[
            List[Tuple[np.ndarray, float, Tuple[float, float, float]]]
        ] = None,
    ) -> None:
        """Render multiple robots in parallel using Open3D."""
        record_dir = output_path
        print("[Open3D] Initializing batched rendering...")

        q_ts = jnp.asarray(q_ts)
        ts = jnp.asarray(ts).reshape((-1,))

        if q_ts.ndim == 2:
            q_ts = q_ts[None, ...]
        if q_ts.ndim != 3:
            raise ValueError(
                f"q_ts must have shape (N, T, DOF); got ndim={q_ts.ndim}. "
                "For single-robot trajectories, pass shape (T, DOF)."
            )

        N, T, _ = q_ts.shape
        if ts.shape[0] != T:
            raise ValueError(
                f"ts length ({ts.shape[0]}) must match q_ts time dimension ({T})"
            )
        print(f"[Open3D] Rendering {N} robots, {T} timesteps")

        # Compute grid offsets if not provided
        base_offsets = self._base_offsets
        if base_offsets is None:
            base_offsets = self._compute_grid_offsets(int(N), self.grid_spacing)
        else:
            base_offsets = jnp.asarray(base_offsets)

        # Get colors for each robot
        robot_colors = _get_robot_colors(
            N, self._robot_colors_config, self.seg_colors[0]
        )

        # Precompute all curves for all robots at all timesteps
        print("[Open3D] Precomputing backbone curves...")

        def compute_all_curves_for_timestep(q_batch: jax.Array) -> jax.Array:
            return self.compute_backbone_curves_batched(q_batch, base_offsets)

        q_ts_time_first = q_ts.transpose(1, 0, 2)  # (T, N, DOF)
        all_curves_time_first = jax.vmap(compute_all_curves_for_timestep)(
            q_ts_time_first
        )  # (T, N, P, 3)
        all_curves = np.array(
            all_curves_time_first.transpose(1, 0, 2, 3), dtype=np.float64
        )  # (N, T, P, 3)

        print(f"[Open3D] Curves computed: shape {all_curves.shape}")

        all_tendon_curves = None
        n_act = 0
        if self._has_tendons:
            print("[Open3D] Computing tendon curves...")

            def compute_all_tendons_for_timestep(q_batch: jax.Array) -> jax.Array:
                return jax.vmap(self.compute_tendon_curves)(q_batch)

            all_tendons_time_first = jax.vmap(compute_all_tendons_for_timestep)(
                q_ts_time_first
            )  # (T, N, n_tend, P, 3)
            all_tendon_curves = np.array(
                all_tendons_time_first.transpose(1, 0, 2, 3, 4), dtype=np.float64
            )  # (N, T, n_tend, P, 3)

            offsets_np = np.asarray(base_offsets)
            if offsets_np.shape[1] == 2:
                offsets_np = np.concatenate(
                    [offsets_np, np.zeros((offsets_np.shape[0], 1))], axis=1
                )
            all_tendon_curves = all_tendon_curves + offsets_np[:, None, None, None, :]
            n_act = all_tendon_curves.shape[2]

        if self.viewer_backend == "gui" and output_path is not None:
            raise NotImplementedError(
                "Recording to disk is not supported with viewer_backend='gui'. "
                "Use viewer_backend='legacy' for recording."
            )

        def extra_init(vis):
            tendon_ls: List[List] = []
            if all_tendon_curves is not None and n_act > 0:
                for robot_idx in range(N):
                    robot_ls = []
                    for k in range(n_act):
                        ls = _make_polyline_lineset(
                            all_tendon_curves[robot_idx, 0, k], self.tendon_color
                        )
                        robot_ls.append(ls)
                        vis.add_geometry(ls)
                    tendon_ls.append(robot_ls)
            return {"tendon_ls": tendon_ls}

        def extra_update(vis, frame_idx: int, extra_state):
            tendon_ls = extra_state.get("tendon_ls", [])
            if all_tendon_curves is not None and n_act > 0:
                for robot_idx, robot_ls in enumerate(tendon_ls):
                    for k, ls in enumerate(robot_ls):
                        t_pts = np.array(
                            all_tendon_curves[robot_idx, frame_idx, k],
                            dtype=np.float64,
                            copy=True,
                        )
                        ls.points = o3d.utility.Vector3dVector(t_pts)
                        vis.update_geometry(ls)

        if self.viewer_backend == "legacy":
            return self._render_backbone_batch(
                all_curves=all_curves,
                ts=np.asarray(ts),
                color_overrides=list(robot_colors),
                record_dir=record_dir,
                record_every_n=record_every_n,
                record_prefix=record_prefix,
                loop=loop,
                playback_speed=playback_speed,
                window_name=f"Robot Animation ({N} robots)",
                extra_init=extra_init,
                extra_update=extra_update,
            )

        if output_path is None:
            return self._render_backbone_batch_gui(
                all_curves=all_curves,
                ts=np.asarray(ts),
                color_overrides=list(robot_colors),
                loop=loop,
                playback_speed=playback_speed,
                window_name=f"Robot Animation ({N} robots)",
                tendon_curves=all_tendon_curves,
                target_point=target_point,
                target_radius=target_radius,
                target_color=target_color,
                obstacles=obstacles,
                moving_spheres=moving_spheres,
            )

        return self._render_backbone_batch(
            all_curves=all_curves,
            ts=np.asarray(ts),
            color_overrides=list(robot_colors),
            record_dir=record_dir,
            record_every_n=record_every_n,
            record_prefix=record_prefix,
            loop=loop,
            playback_speed=playback_speed,
            window_name=f"Robot Animation ({N} robots)",
            extra_init=extra_init,
            extra_update=extra_update,
        )

def _get_robot_colors(
    num_robots: int,
    robot_colors: Optional[
        Union[str, Tuple[Tuple[float, float, float], ...], List[Tuple[float, float, float]]]
    ],
    default_color: Tuple[float, float, float] = (0.1, 0.45, 1.0),
) -> List[Tuple[float, float, float]]:
    """Get colors for each robot.

    Args:
        num_robots: Number of robots
        robot_colors: "same" (use default), colormap name, or explicit tuple of colors
        default_color: Default color if robot_colors is None or "same"

    Returns:
        List of RGB colors, one per robot
    """
    if robot_colors is None or robot_colors == "same":
        return [default_color] * num_robots
    elif isinstance(robot_colors, str):
        # Colormap name (e.g., "viridis", "plasma")
        import matplotlib.cm as cm

        cmap = cm.get_cmap(robot_colors)
        return [
            tuple(cmap(i / max(1, num_robots - 1))[:3]) for i in range(num_robots)
        ]
    else:
        # Explicit tuple/list - cycle if too short
        return [robot_colors[i % len(robot_colors)] for i in range(num_robots)]
