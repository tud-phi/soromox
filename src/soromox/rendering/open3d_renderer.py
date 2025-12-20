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
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

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
) -> "o3d.geometry.TriangleMesh":
    """Create a base plate cylinder mesh."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=float(thickness), resolution=resolution, split=1
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
    return mesh


def _make_sphere(
    center_xyz: np.ndarray,
    radius: float,
    color: Tuple[float, float, float] = (0.1, 0.45, 1.0),
    resolution: int = 16,
) -> "o3d.geometry.TriangleMesh":
    """Create a colored sphere at center_xyz with given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    mesh.translate(np.array(center_xyz, dtype=np.float64), relative=False)
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
        >>> renderer.render_sequence(ts, q_ts, fps=30)  # Animated playback
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
        sphere_resolution: int = 32,
        tendon_color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
        tendon_line_width: float = 2.0,
        camera_margin_ratio: float = 0.05,
    ):
        """Initialize Open3D renderer.

        Args:
            robot: Robot system with forward_kinematics method
            width: Window width in pixels
            height: Window height in pixels
            num_points: Number of points for backbone discretization
            background_color: RGB background color (0-1 range)
            seg_colors: Tuple of RGB colors for each segment
            sphere_resolution: Resolution for backbone spheres
            tendon_color: RGB color for tendon lines
            tendon_line_width: Width of tendon lines
            camera_margin_ratio: Margin ratio for camera bounding box
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

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # Base plate
        base_mesh = _make_base_plate(
            curve[0],
            radius=float(2.0 * r_seg.max()),
            thickness=float(5e-2 * 1.3),
            color=(0.0, 0.0, 0.0),
        )
        scene.add_geometry("base", base_mesh, mat)

        # Backbone spheres
        for s in range(S):
            c0, c1 = int(starts[s]), int(ends[s])
            for p in range(c0, c1):
                sp = _make_sphere(
                    curve[p],
                    radius=float(r_seg[min(s, len(r_seg) - 1)]),
                    color=seg_cols[s],
                    resolution=self.sphere_resolution,
                )
                scene.add_geometry(f"sphere_{s}_{p}", sp, mat)

        # Tendons
        if tendon_curves is not None:
            for i in range(tendon_curves.shape[0]):
                ls = _make_polyline_lineset(tendon_curves[i], color=self.tendon_color)
                mat_line = o3d.visualization.rendering.MaterialRecord()
                mat_line.shader = "unlitLine"
                mat_line.line_width = self.tendon_line_width
                scene.add_geometry(f"tendon_{i}", ls, mat_line)

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
        base_mesh = _make_base_plate(
            curve[0],
            radius=float(2.0 * r_seg.max()),
            thickness=float(5e-2 * 1.3),
            color=(0.0, 0.0, 0.0),
        )
        vis.add_geometry(base_mesh)

        # Backbone spheres
        for s in range(S):
            c0, c1 = int(starts[s]), int(ends[s])
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

    def render_sequence(  # type: ignore[override]
        self,
        ts: Array,
        q_ts: Array,
        fps: int = 25,
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
        """Native Open3D animated playback with keyboard controls.

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

        Args:
            ts: Time stamps (T,)
            q_ts: Configurations (T, DOF)
            fps: Frames per second
            output_path: Directory to save PNG frames (optional, alias for record_dir)
            loop: Whether to loop animation
            record_every_n: Save every n-th frame
            record_prefix: Filename prefix for saved frames
            target_point: Optional target point to visualize
            target_radius: Radius of target sphere
            target_color: Color of target sphere
            obstacles: List of (center, radius, color) tuples for static obstacles
            moving_spheres: List of (trajectory, radius, color) for moving objects
        """
        record_dir = output_path  # Use output_path as record directory
        print("[Open3D] Initializing...")

        q_ts = jnp.asarray(q_ts)
        ts = jnp.asarray(ts).reshape((-1,))
        T = q_ts.shape[0]

        # Precompute all curves
        batched_curve = jax.vmap(lambda q: self.compute_backbone_curve(q))
        all_robot_curves = np.array(batched_curve(q_ts), dtype=np.float64)

        all_tendon_curves = None
        n_act = 0
        if self._has_tendons:
            batched_tendon = jax.vmap(lambda q: self.compute_tendon_curves(q))
            all_tendon_curves = np.array(batched_tendon(q_ts), dtype=np.float64)
            n_act = all_tendon_curves.shape[1]

        P = all_robot_curves.shape[1]

        # Setup recording
        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)
            print(f"[Open3D] Recording frames to: {record_dir}")

        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name="Robot Animation (Open3D)",
            width=self.width,
            height=self.height,
        )

        opt = vis.get_render_option()
        opt.background_color = np.array(self.background_color, dtype=np.float64)
        opt.line_width = self.tendon_line_width

        # Setup geometries
        L_np = np.asarray(self.robot.L, dtype=np.float64).reshape(-1)
        r_seg = np.asarray(self.robot.r, dtype=np.float64).reshape(-1)
        S = int(L_np.size)

        counts = _split_counts_by_lengths(P, L_np)
        starts = np.cumsum([0] + counts[:-1]).astype(int)
        ends = np.cumsum(counts).astype(int)
        seg_cols = _expand_colors(self.seg_colors, S)

        curve0 = all_robot_curves[0]

        # Base
        base_mesh = _make_base_plate(
            curve0[0],
            radius=float(2.0 * r_seg.max()),
            thickness=float(5e-2 * 1.3),
        )
        vis.add_geometry(base_mesh)

        # Spheres per segment
        spheres_groups: List[List] = []
        for s in range(S):
            seg_spheres = []
            c0, c1 = int(starts[s]), int(ends[s])
            for p in range(c0, c1):
                sp = _make_sphere(
                    curve0[p],
                    radius=float(r_seg[min(s, len(r_seg) - 1)]),
                    color=seg_cols[s],
                    resolution=self.sphere_resolution,
                )
                seg_spheres.append(sp)
                vis.add_geometry(sp)
            spheres_groups.append(seg_spheres)

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

        # Moving spheres
        dynamic_sphere_meshes = []
        dynamic_sphere_trajs = []
        if moving_spheres:
            for centers_T3, rad, col in moving_spheres:
                centers_np = np.asarray(centers_T3, dtype=np.float64)
                mesh0 = _make_sphere(
                    centers_np[0], float(rad), col, max(12, self.sphere_resolution // 2)
                )
                dynamic_sphere_meshes.append(mesh0)
                dynamic_sphere_trajs.append(centers_np)
                vis.add_geometry(mesh0)

        # Tendons
        tendon_ls = []
        if all_tendon_curves is not None:
            for i in range(n_act):
                ls = _make_polyline_lineset(all_tendon_curves[0, i], self.tendon_color)
                tendon_ls.append(ls)
                vis.add_geometry(ls)

        # Camera setup
        ctrl = vis.get_view_control()
        initial_cam = ctrl.convert_to_pinhole_camera_parameters()
        saved_cam = copy.deepcopy(initial_cam)

        # Playback state
        state = {"idx": 0, "playing": True, "last_tick": time.time(), "dt": 1.0 / fps}

        def _clamp(i: int) -> int:
            return max(0, min(T - 1, i))

        def _save_frame(i: int) -> None:
            if record_dir is None or (i % record_every_n) != 0:
                return
            fname = os.path.join(record_dir, f"{record_prefix}{i:05d}.png")
            vis.capture_screen_image(fname, do_render=True)

        def update_frame(i: int) -> None:
            i = _clamp(i)
            state["idx"] = i
            curve = all_robot_curves[i]

            # Update spheres
            for s in range(len(spheres_groups)):
                c0, c1 = int(starts[s]), int(ends[s])
                seg_spheres = spheres_groups[s]
                for i_local, p in enumerate(range(c0, min(c1, c0 + len(seg_spheres)))):
                    mesh = seg_spheres[i_local]
                    delta = curve[p] - _mesh_center(mesh)
                    mesh.translate(delta, relative=True)
                    vis.update_geometry(mesh)

            # Update tendons
            if all_tendon_curves is not None:
                for k in range(n_act):
                    t_pts = np.array(all_tendon_curves[i, k], dtype=np.float64, copy=True)
                    tendon_ls[k].points = o3d.utility.Vector3dVector(t_pts)
                    vis.update_geometry(tendon_ls[k])

            # Update moving spheres
            for idx_s, mesh in enumerate(dynamic_sphere_meshes):
                traj = dynamic_sphere_trajs[idx_s]
                j_idx = min(i, traj.shape[0] - 1)
                delta = traj[j_idx] - _mesh_center(mesh)
                mesh.translate(delta, relative=True)
                vis.update_geometry(mesh)

            vis.poll_events()
            vis.update_renderer()
            _save_frame(i)

        # Key callbacks
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
            if record_dir is None:
                vis.capture_screen_image(f"frame_{state['idx']:05d}.png", do_render=True)
            else:
                _save_frame(state["idx"])
            return False

        def cb_reset_camera(_):
            ctrl.convert_from_pinhole_camera_parameters(initial_cam)
            vis.update_renderer()
            print("[Open3D] Camera reset to initial view.")
            return False

        def cb_capture_camera(_):
            nonlocal saved_cam
            saved_cam = ctrl.convert_to_pinhole_camera_parameters()
            print("[Open3D] Camera view captured.")
            return False

        def cb_load_camera(_):
            ctrl.convert_from_pinhole_camera_parameters(saved_cam)
            vis.update_renderer()
            print("[Open3D] Saved camera view loaded.")
            return False

        def cb_print_camera(_):
            params = ctrl.convert_to_pinhole_camera_parameters()
            K = np.asarray(params.intrinsic.intrinsic_matrix)
            E = np.asarray(params.extrinsic)
            print("\n[Open3D] Camera parameters:")
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

        update_frame(0)

        print(
            "[Open3D] Running. Keys: Space=Play/Pause  ←/→=Step  H=Home  "
            "S=Snapshot  R=ResetCam  C=CaptureCam  L=LoadCam  V=PrintCam  Q/Esc=Quit"
        )

        while vis.poll_events():
            now = time.time()
            if state["playing"] and (now - state["last_tick"] >= state["dt"]):
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
