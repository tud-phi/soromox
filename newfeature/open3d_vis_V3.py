# open3d_vis.py
# ======================================================================================
# Open3D visualization utility for tendon-actuated PCS robots.
# - Backbone as spheres (per-segment colors & radii)
# - Tendons as polyline LineSets
# - Recording frames to PNGs (use ffmpeg to make a video)
# - Unified public interface: visualize_robot_open3d(...)
# - Camera helpers (PinholeCameraParameters-based, compatible with older Open3D):
#     * R: reset camera to initial programmatic view (after bbox fit)
#     * C: capture/save current camera view (PinholeCameraParameters)
#     * L: load/restore the saved camera view
#     * V: print current camera intrinsic/extrinsic for copy-paste
# ======================================================================================

from __future__ import annotations

import os
import time
import copy
from typing import Callable, Optional, Dict, Tuple, List

import numpy as onp
import jax
import jax.numpy as jnp
import open3d as o3d


# ======================================================================================
# Public helpers (can be reused in your pipeline)
# ======================================================================================

def draw_robot_curve(
    batched_forward_kinematics: Callable,
    L_max: float,
    q: jnp.ndarray,
    num_points: int = 50,
) -> jnp.ndarray:
    """Sample backbone curve: returns (num_points, 3)."""
    s_ps = jnp.linspace(0.0, float(L_max), num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]
    return jnp.array(g_ps, dtype=jnp.float64)


def draw_tendon_curves(
    batched_forward_kinematics_tendons: Callable,
    L_max: float,
    q: jnp.ndarray,
    num_points: int = 50,
) -> jnp.ndarray:
    """Sample all tendons: returns (n_tendons, num_points, 3)."""
    s_ps = jnp.linspace(0.0, float(L_max), num_points)
    ps = batched_forward_kinematics_tendons(q, s_ps)
    return jnp.array(ps, dtype=jnp.float64)


# ======================================================================================
# Internal helpers (geometry creation / utilities)
# ======================================================================================

def _make_polyline_lineset(
    points_np: onp.ndarray,
    color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
) -> o3d.geometry.LineSet:
    """Create a colored polyline LineSet from (N,3) points."""
    pts = onp.array(points_np, dtype=onp.float64, order="C", copy=True)
    N = pts.shape[0]
    if N < 2:
        pts = onp.vstack([pts, pts[-1]])
        N = 2
    lines = onp.stack([onp.arange(0, N - 1), onp.arange(1, N)], axis=1).astype(onp.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(
        onp.tile(onp.array(color, dtype=onp.float64)[None, :], (lines.shape[0], 1))
    )
    return ls


def _make_base_plate(
    center_xyz: onp.ndarray,
    radius: float,
    thickness: float = 0.005,
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: int = 48,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius), height=float(thickness), resolution=resolution, split=1
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(onp.array(color, dtype=onp.float64))
    mesh.translate(onp.array(center_xyz, dtype=onp.float64), relative=False)
    return mesh


def _make_sphere(
    center_xyz: onp.ndarray,
    radius: float,
    color: Tuple[float, float, float] = (0.1, 0.45, 1.0),
    resolution: int = 16,
) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere at center_xyz with given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(onp.array(color, dtype=onp.float64))
    mesh.translate(onp.array(center_xyz, dtype=onp.float64), relative=False)
    return mesh


def _make_target_sphere(
    center_xyz: onp.ndarray,
    radius: float = 0.01,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    resolution: int = 16,
) -> o3d.geometry.TriangleMesh:
    """Create a colored sphere marking a target point."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(onp.array(color, dtype=onp.float64))
    mesh.translate(onp.array(center_xyz, dtype=onp.float64), relative=False)
    return mesh


def _make_obstacle_sphere(
    center_xyz: onp.ndarray,
    radius: float,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    resolution: int = 24,
) -> o3d.geometry.TriangleMesh:
    """Create a colored obstacle sphere at center_xyz with given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=resolution)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(onp.array(color, dtype=onp.float64))
    mesh.translate(onp.array(center_xyz, dtype=onp.float64), relative=False)
    return mesh


def _split_counts_by_length(num_points: int, L: onp.ndarray) -> int:
    """
    (legacy) Two-segment helper retained for backward compatibility.
    """
    L = onp.asarray(L, dtype=onp.float64)
    L_tot = float(L.sum()) if L.size else 0.0
    if L_tot <= 0:
        return num_points // 2
    n1 = int(round(num_points * (float(L[0]) / L_tot)))
    return max(1, min(num_points - 1, n1))


def _split_counts_by_lengths(num_points: int, L: onp.ndarray) -> List[int]:
    """
    Split 'num_points' across len(L) segments proportionally to lengths.
    - Each segment gets at least 1 point.
    - The sum equals num_points.
    - The last segment includes the end boundary.
    """
    L = onp.asarray(L, dtype=onp.float64).reshape(-1)
    S = int(L.size)
    if S == 0:
        return [num_points]
    if num_points < S:
        base = [1] * S
        base[-1] += (num_points - S)
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
        fracs = onp.array([x - onp.floor(x) for x in raw])
        order = onp.argsort(fracs)[::-1] if diff > 0 else onp.argsort(fracs)
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


def _expand_colors(seg_colors: Tuple[Tuple[float, float, float], ...], S: int) -> List[Tuple[float, float, float]]:
    """
    Ensure we have at least S colors; if not, cycle the given palette.
    Backward-compatible with a 2-color palette.
    """
    palette = list(seg_colors)
    if len(palette) == 0:
        palette = [(0.1, 0.45, 1.0)]
    out = [palette[i % len(palette)] for i in range(S)]
    return out


def _mesh_center(mesh: o3d.geometry.TriangleMesh) -> onp.ndarray:
    """Approx center: mean of vertices."""
    return onp.asarray(mesh.vertices).mean(axis=0)


# ======================================================================================
# Core: Open3D animation
# ======================================================================================

def animate_robot_tendons_open3d(
    robot,
    t_list,
    q_list,
    *,
    num_points: int = 80,
    fps: int = 25,
    seg_colors: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.1, 0.45, 1.0), (0.0, 0.6, 0.2)),
    sphere_resolution: int = 32,
    loop: bool = False,
    # rendering opts
    window_size: Tuple[int, int] = (1280, 800),
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tendon_color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
    tendon_line_width: float = 2.0,
    camera_margin_ratio: float = 0.05,
    target_point: Optional[Tuple[float, float, float]] = None,
    target_radius: float = 0.01,
    target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    # static spherical obstacles [(center_xyz, radius, color), ...]
    obstacles: Optional[List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]] = None,
    # moving spheres: [(centers_T3, radius, color), ...]
    moving_spheres: Optional[List[Tuple[onp.ndarray, float, Tuple[float, float, float]]]] = None,
    # recording
    record_dir: Optional[str] = None,
    record_every_n: int = 1,
    record_prefix: str = "frame_",
) -> None:
    """
    Open3D real-time animation with spheres (backbone) and polylines (tendons).

    Controls:
      Space: play/pause
      →/← : next/prev frame
      H   : go to frame 0
      S   : save snapshot of current frame
      R   : reset camera to initial programmatic view
      C   : capture/save current camera view
      L   : load/restore the saved camera view
      V   : print current camera intrinsic/extrinsic for copy-paste into scripts
      Q/ESC: quit

    Recording:
      If record_dir is not None, will save PNG frames via open3d.Visualizer.capture_screen_image.

    moving_spheres:
      Each element is (centers_T3, radius, color), where:
        - centers_T3: array-like, shape (T, 3), sphere center for each frame
        - radius: float
        - color: (r,g,b)
      T should be >= len(t_list); if longer, will be truncated.
    """
    print("[Open3D] Initializing...")
    print("Working dir:", os.getcwd())

    q_list = jnp.asarray(q_list)
    t_list = jnp.asarray(t_list).reshape((-1,))
    assert q_list.ndim == 2, f"q_list should be (T, DOF), got {q_list.shape}"
    assert t_list.shape[0] == q_list.shape[0], f"t_list len {t_list.shape[0]} != q_list T {q_list.shape[0]}"
    print("[Open3D] T, DOF =", q_list.shape)

    # ---------- precompute curves ----------
    batched_fk = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    batched_fk_tendons = jax.vmap(robot.forward_kinematics_tendons, in_axes=(None, 0), out_axes=1)
    L_max = float(jnp.sum(robot.L))

    def _draw_robot_curve_once(q):
        return draw_robot_curve(batched_fk, L_max, q, num_points).astype(jnp.float64)

    def _draw_tendon_curves_once(q):
        return draw_tendon_curves(batched_fk_tendons, L_max, q, num_points).astype(jnp.float64)

    batched_robot_curve = jax.vmap(_draw_robot_curve_once)
    batched_tendon_curve = jax.vmap(_draw_tendon_curves_once)

    all_robot_curves = onp.array(batched_robot_curve(q_list), dtype=onp.float64, order="C", copy=True)    # (T, P, 3)
    all_tendon_curves = onp.array(batched_tendon_curve(q_list), dtype=onp.float64, order="C", copy=True)  # (T, n_act, P, 3)
    T, P, _ = all_robot_curves.shape
    n_act = all_tendon_curves.shape[1]

    # ---------- recorder dir ----------
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
        print(f"[Open3D] Recording frames to: {record_dir} (every {record_every_n} frames)")

    # ---------- visualizer ----------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Tendon-Actuated PCS (Open3D)", width=int(window_size[0]), height=int(window_size[1]))
    opt = vis.get_render_option()
    opt.background_color = onp.array(background_color, dtype=onp.float64)
    opt.line_width = float(tendon_line_width)  # affects polylines (tendons)

    # ---------- backbone as spheres (segmented, arbitrary S) ----------
    L_np = onp.asarray(robot.L, dtype=onp.float64).reshape(-1)
    r_seg = onp.asarray(robot.r, dtype=onp.float64).reshape(-1)
    S = int(L_np.size)
    assert S >= 1, "robot.L should have at least one segment length."

    # Split P points across S segments
    counts = _split_counts_by_lengths(P, L_np)             # len S, sum = P
    starts = onp.cumsum([0] + counts[:-1]).astype(int)     # len S
    ends   = onp.cumsum(counts).astype(int)                # len S
    seg_cols = _expand_colors(seg_colors, S)

    curve0 = onp.asarray(all_robot_curves[0], dtype=onp.float64)

    # base (at s=0)
    base_center = curve0[0]
    base_radius = float(2.0 * float(r_seg.max()))  # enlarged for visibility
    base_mesh = _make_base_plate(
        base_center, radius=base_radius, thickness=float(5e-2*1.3), color=(0.0, 0.0, 0.0)
    )
    vis.add_geometry(base_mesh)

    # spheres per segment
    spheres_groups: List[List[o3d.geometry.TriangleMesh]] = []
    for s in range(S):
        seg_spheres: List[o3d.geometry.TriangleMesh] = []
        c0, c1 = int(starts[s]), int(ends[s])
        c0 = max(0, min(P - 1, c0))
        c1 = max(c0 + 1, min(P, c1))
        for p in range(c0, c1):
            sp = _make_sphere(
                curve0[p],
                radius=float(r_seg[min(s, len(r_seg) - 1)]),
                color=seg_cols[s],
                resolution=sphere_resolution,
            )
            seg_spheres.append(sp)
            vis.add_geometry(sp)
        spheres_groups.append(seg_spheres)

    # ---------- target point (optional) ----------
    target_pts: List[onp.ndarray] = []
    if target_point is not None:
        tgt = onp.array(target_point, dtype=onp.float64).reshape(3,)
        print("[Open3D] target_point =", tgt)
        target_mesh = _make_target_sphere(
            tgt, radius=float(target_radius), color=target_color, resolution=max(8, sphere_resolution // 2)
        )
        vis.add_geometry(target_mesh)
        target_pts.append(tgt)

    # ---------- static spherical obstacles (optional) ----------
    obstacle_meshes: List[o3d.geometry.TriangleMesh] = []
    obstacle_points: List[onp.ndarray] = []
    if obstacles:
        for (ctr, rad, col) in obstacles:
            ctr_np = onp.array(ctr, dtype=onp.float64).reshape(3,)
            mesh = _make_obstacle_sphere(
                ctr_np, float(rad), color=col, resolution=max(12, sphere_resolution // 2)
            )
            obstacle_meshes.append(mesh)
            vis.add_geometry(mesh)
            obstacle_points.append(ctr_np)  # for bbox expansion

    # ---------- moving spheres (with trajectories) ----------
    dynamic_sphere_meshes: List[o3d.geometry.TriangleMesh] = []
    dynamic_sphere_trajs: List[onp.ndarray] = []

    if moving_spheres:
        for centers_T3, rad, col in moving_spheres:
            centers_np = onp.asarray(centers_T3, dtype=onp.float64)
            if centers_np.ndim != 2 or centers_np.shape[1] != 3:
                raise ValueError("moving_spheres centers must have shape (T, 3)")
            # truncate or keep as is; we'll clamp index in update_frame
            mesh0 = _make_sphere(
                centers_np[0],
                radius=float(rad),
                color=col,
                resolution=max(12, sphere_resolution // 2),
            )
            dynamic_sphere_meshes.append(mesh0)
            dynamic_sphere_trajs.append(centers_np)
            vis.add_geometry(mesh0)

    # ---------- tendons as polylines ----------
    tendon_ls: List[o3d.geometry.LineSet] = []
    for i in range(n_act):
        ls = _make_polyline_lineset(all_tendon_curves[0, i], color=tendon_color)
        tendon_ls.append(ls)
        vis.add_geometry(ls)

    # ---------- view fit ----------
    all_pts = onp.concatenate(
        [all_robot_curves.reshape(-1, 3), all_tendon_curves.reshape(-1, 3)],
        axis=0
    ).astype(onp.float64, copy=True)

    if obstacle_points:
        obs_pts = onp.stack(obstacle_points, axis=0).astype(onp.float64, copy=True)
        all_pts = onp.concatenate([all_pts, obs_pts], axis=0)

    if target_pts:
        tgt_pts = onp.stack(target_pts, axis=0).astype(onp.float64, copy=True)
        all_pts = onp.concatenate([all_pts, tgt_pts], axis=0)

    # add moving spheres' whole trajectories to bbox
    if moving_spheres:
        dyn_pts_list = []
        for centers_T3, _, _ in moving_spheres:
            centers_np = onp.asarray(centers_T3, dtype=onp.float64).reshape(-1, 3)
            dyn_pts_list.append(centers_np)
        if dyn_pts_list:
            dyn_all = onp.concatenate(dyn_pts_list, axis=0)
            all_pts = onp.concatenate([all_pts, dyn_all], axis=0)

    min_b = all_pts.min(axis=0)
    max_b = all_pts.max(axis=0)
    diag = float(onp.linalg.norm(max_b - min_b) + 1e-12)
    margin = camera_margin_ratio * diag
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_b - margin, max_b + margin)

    ctrl = vis.get_view_control()
    try:
        ctrl.fit_area_to_geometry(bbox, allow_rotation=True)  # newer Open3D
    except Exception:
        try:
            ctrl.fit_to_geometry(bbox)  # older Open3D
        except Exception:
            vis.add_geometry(bbox)

    # ---------- save initial camera as PinholeCameraParameters ----------
    initial_cam = ctrl.convert_to_pinhole_camera_parameters()
    # saved_cam is mutable snapshot you can overwrite with C / reload with L
    saved_cam = copy.deepcopy(initial_cam)

    # ---------- playback state ----------
    state = {"idx": 0, "playing": True, "last_tick": time.time(), "dt": 1.0 / max(1, fps)}

    def _clamp(i: int) -> int:
        return max(0, min(T - 1, i))

    def _save_frame(i: int) -> None:
        if record_dir is None:
            return
        if (i % int(record_every_n)) != 0:
            return
        fname = os.path.join(record_dir, f"{record_prefix}{i:05d}.png")
        vis.capture_screen_image(fname, do_render=True)

    def update_frame(i: int) -> None:
        i = _clamp(i)
        state["idx"] = i
        curve = onp.asarray(all_robot_curves[i], dtype=onp.float64)

        # move spheres for each segment
        for s in range(len(spheres_groups)):
            c0, c1 = int(starts[s]), int(ends[s])
            seg_spheres = spheres_groups[s]
            n_this = min(len(seg_spheres), max(0, c1 - c0))
            for i_local in range(n_this):
                p = c0 + i_local
                mesh = seg_spheres[i_local]
                target = curve[p]
                delta = target - _mesh_center(mesh)
                mesh.translate(delta, relative=True)
                vis.update_geometry(mesh)

        # update tendons
        for k in range(n_act):
            t_pts = onp.array(all_tendon_curves[i, k], dtype=onp.float64, order="C", copy=True)
            tendon_ls[k].points = o3d.utility.Vector3dVector(t_pts)
            vis.update_geometry(tendon_ls[k])

        # update moving spheres
        if dynamic_sphere_meshes:
            for idx_s, mesh in enumerate(dynamic_sphere_meshes):
                traj = dynamic_sphere_trajs[idx_s]
                j_idx = min(i, traj.shape[0] - 1)
                target = traj[j_idx]
                delta = target - _mesh_center(mesh)
                mesh.translate(delta, relative=True)
                vis.update_geometry(mesh)

        vis.poll_events()
        vis.update_renderer()
        _save_frame(i)

    # ---------- key callbacks ----------
    def cb_space(_):  # play/pause
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
        # manual snapshot of current frame
        if record_dir is None:
            vis.capture_screen_image(f"frame_{state['idx']:05d}.png", do_render=True)
        else:
            _save_frame(state["idx"])
        return False

    def cb_reset_camera(_):
        """Reset camera to the initial programmatic view (pinhole parameters)."""
        ctrl.convert_from_pinhole_camera_parameters(initial_cam)
        vis.update_renderer()
        print("[Open3D] Camera reset to initial view.")
        return False

    def cb_capture_camera(_):
        """Capture/save current camera view into saved_cam."""
        nonlocal saved_cam
        saved_cam = ctrl.convert_to_pinhole_camera_parameters()
        print("[Open3D] Camera view captured and saved (use 'L' to load it).")
        return False

    def cb_load_camera(_):
        """Load/restore the saved camera view (captured via C)."""
        ctrl.convert_from_pinhole_camera_parameters(saved_cam)
        vis.update_renderer()
        print("[Open3D] Saved camera view loaded.")
        return False

    def cb_print_camera(_):
        """Print current camera pinhole parameters (intrinsic & extrinsic)."""
        params = ctrl.convert_to_pinhole_camera_parameters()
        K = onp.asarray(params.intrinsic.intrinsic_matrix)
        E = onp.asarray(params.extrinsic)

        print("\n[Open3D] Current camera pinhole parameters (paste into your script):")
        print("camera_params = {")
        print("    'intrinsic': [")
        for row in K:
            print(f"        {row.tolist()},")
        print("    ],")
        print("    'extrinsic': [")
        for row in E:
            print(f"        {row.tolist()},")
        print("    ],")
        print("}\n")
        return False

    def cb_quit(_):
        state["playing"] = False
        vis.destroy_window()
        return False

    vis.register_key_callback(ord(" "), cb_space)  # Space
    vis.register_key_callback(262, cb_next)        # Right
    vis.register_key_callback(263, cb_prev)        # Left
    vis.register_key_callback(ord("H"), cb_home)   # H
    vis.register_key_callback(ord("S"), cb_save)   # S
    vis.register_key_callback(ord("R"), cb_reset_camera)   # R: reset to initial view
    vis.register_key_callback(ord("C"), cb_capture_camera) # C: capture/save view
    vis.register_key_callback(ord("L"), cb_load_camera)    # L: load saved view
    vis.register_key_callback(ord("V"), cb_print_camera)   # V: print camera params
    vis.register_key_callback(81, cb_quit)         # Q
    vis.register_key_callback(256, cb_quit)        # Esc

    # initial frame
    update_frame(0)

    # main loop
    print(
        "[Open3D] Running. Keys: "
        "Space=Play/Pause  ←/→=Step  H=Home  "
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


# ======================================================================================
# Unified public interface
# ======================================================================================

def visualize_robot_open3d(
    robot,
    t_list,
    q_list,
    *,
    num_points: int = 80,
    fps: int = 25,
    seg_colors: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.1, 0.45, 1.0), (0.0, 0.6, 0.2)),
    sphere_resolution: int = 32,
    loop: bool = False,
    # rendering opts
    window_size: Tuple[int, int] = (1280, 800),
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tendon_color: Tuple[float, float, float] = (0.9, 0.15, 0.15),
    tendon_line_width: float = 2.0,
    camera_margin_ratio: float = 0.05,
    target_point: Optional[Tuple[float, float, float]] = None,
    target_radius: float = 0.01,
    target_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    obstacles: Optional[List[Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]]] = None,
    # moving spheres with trajectories
    moving_spheres: Optional[List[Tuple[onp.ndarray, float, Tuple[float, float, float]]]] = None,
    # recording
    record: Optional[Dict] = None,
) -> None:
    """
    Unified Open3D visualization entry point.

    record dict can include:
      - "dir": str, output directory
      - "every_n": int, save every n-th frame (default 1)
      - "prefix": str, filename prefix (default "frame_")

    moving_spheres:
      List of (centers_T3, radius, color), where centers_T3 is (T,3).
    """
    rec_dir = None
    rec_every = 1
    rec_prefix = "frame_"
    if record is not None:
        rec_dir = record.get("dir", None)
        rec_every = int(record.get("every_n", 1))
        rec_prefix = str(record.get("prefix", "frame_"))

    return animate_robot_tendons_open3d(
        robot,
        t_list=t_list, q_list=q_list,
        num_points=num_points, fps=fps,
        seg_colors=seg_colors, sphere_resolution=sphere_resolution, loop=loop,
        window_size=window_size, background_color=background_color,
        tendon_color=tendon_color, tendon_line_width=tendon_line_width,
        camera_margin_ratio=camera_margin_ratio,
        target_point=target_point,
        target_radius=target_radius,
        target_color=target_color,
        obstacles=obstacles,
        moving_spheres=moving_spheres,
        record_dir=rec_dir, record_every_n=rec_every, record_prefix=rec_prefix,
    )


# ======================================================================================
# (Optional) CLI / quick demo stub
# ======================================================================================
if __name__ == "__main__":
    print(
        "This module provides Open3D visualization for a tendon-actuated PCS robot.\n"
        "- Import visualize_robot_open3d(...) and call it with your 'robot', 't_list', 'q_list'.\n"
        "- Example:\n"
        "    from open3d_vis import visualize_robot_open3d\n"
        "    visualize_robot_open3d(robot, ts, q_ts, num_points=80, fps=60,\n"
        "                           record={'dir':'frames','every_n':1,'prefix':'frame_'})\n"
        "\n"
        "Note: you can also pass 'moving_spheres=[(centers_T3, radius, color), ...]' to visualize\n"
        "      objects (like a pushed ball) moving over time.\n"
        "\n"
        "Controls:\n"
        "  Space: play/pause\n"
        "  ←/→  : next/prev frame\n"
        "  H    : go to frame 0\n"
        "  S    : save snapshot of current frame\n"
        "  R    : reset camera to initial view (pinhole)\n"
        "  C    : capture/save current camera view\n"
        "  L    : load/restore the saved camera view\n"
        "  V    : print current camera intrinsic/extrinsic\n"
        "  Q/Esc: quit\n"
    )
