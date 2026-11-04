"""Render the summary figure's panel A: one RTX frame of an iCEM replanning step's sample cloud.

For a saved run's --timestep and CEM --iteration, draws every sample's predicted rod shape at
--horizon-step, each translucent by cost rank, the lowest-cost sample solid, the best
--max-object-samples predicted object poses, the real executed rod/object, and the goal marker.
Run once per iteration to show the elite population narrowing.

Runs in this section's environment (../pyproject.toml) and needs an RTX GPU. Needs ovrtx 0.4.x:
0.5.1 renders dense translucent sample clouds black.

Usage:
    uv run --project . python code/render_sample_cloud.py \\
        --npz data/square_icem_seed0.npz --timestep 0 --iteration 0 --horizon-step 10 \\
        --output outputs/frames/square_icem_seed0_t0_iter0.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import run_experiment  # noqa: E402
from rendering import (  # noqa: E402
    CAMERA_EYE,
    CAMERA_TARGET,
    ROD_COLLISION_BODIES,
    ROD_COLOR,
    TABLE_COLOR,
    ExampleViewerRTX,
    build_goal_overlay,
    look_at,
    paper_rgb,
    update_goal_overlay,
)
from four_tendon_soft_arm import FourTendonSoftArm  # noqa: E402
from newton_cosserat.gvs import gvs_state_views  # noqa: E402


SAMPLE_COLOR = paper_rgb("pre_opt_1")  # cloud samples, their objects, and the real executed rod/object
BEST_PREDICTION_COLOR = ROD_COLOR
CAP_LATITUDE_RINGS = 4
WORST_SAMPLE_OPACITY_FACTOR = 0.1  # worst-ranked sample's opacity relative to --sample-opacity
OBJECT_MASS = 0.05  # render-only scene, never stepped, so the mass is irrelevant


def _rollout_backbone_poses(
    robot, X_row: np.ndarray, ndof: int, num_base_coordinates: int, s_ps: np.ndarray
) -> np.ndarray:
    """Backbone SE(3) poses at one state-trajectory row.

    Args:
        robot: SoRoMoX GVS model.
        X_row: One state row, shape (state_dim,).
        ndof: Internal DOF count.
        num_base_coordinates: Floating-base coordinate count.
        s_ps: Arc-length sample points, shape (N,).

    Returns:
        World-frame poses, shape (N, 4, 4); column 0 is the backbone tangent.
    """
    q_internal = X_row[:ndof]
    base_pose = X_row[2 * ndof + 5 : 2 * ndof + 5 + num_base_coordinates]
    poses = robot.forward_kinematics_abscissa_batched(jnp.concatenate([base_pose, q_internal]), jnp.asarray(s_ps))
    return np.asarray(poses)


def cross_section_radii(robot, s_ps: np.ndarray, ndof: int, num_base_coordinates: int) -> np.ndarray:
    """Circular cross-section radius at each arc-length sample.

    The geometry is q-independent for this robot (fixed radius profile), so q = 0 is used.

    Raises:
        ValueError: If any sample's cross-section is not circular.
    """
    q_ref = jnp.zeros(ndof + num_base_coordinates, dtype=jnp.float64)
    radii = []
    for s in s_ps:
        geometry, outer_dimensions, _ = robot.cross_section_geometry(q_ref, jnp.asarray(s, dtype=jnp.float64))
        if int(geometry) != 0:
            raise ValueError(f"cross_section_radii assumes CIRCULAR cross-sections, got geometry={geometry} at s={s}")
        radii.append(float(np.asarray(outer_dimensions).reshape(-1)[0]))
    return np.asarray(radii, dtype=np.float64)


def _ring_band_tris(ring_sides: int, lo_base: int, hi_base: int, *, reverse: bool) -> np.ndarray:
    """Triangle indices of the quad strip between two rings starting at `lo_base`/`hi_base`; `reverse` flips winding."""
    j = np.arange(ring_sides)
    j_next = (j + 1) % ring_sides
    a, b = lo_base + j, lo_base + j_next
    c, d = hi_base + j, hi_base + j_next
    tris = np.stack([a, b, d, a, d, c] if not reverse else [a, d, b, a, c, d], axis=-1)
    return tris.reshape(-1)


def _pole_fan_tris(ring_sides: int, ring_base: int, pole_idx: int, *, reverse: bool) -> np.ndarray:
    """Triangle-fan indices connecting one ring to a single pole vertex."""
    j = np.arange(ring_sides)
    j_next = (j + 1) % ring_sides
    tris = np.stack([np.full(ring_sides, pole_idx), j_next, j] if reverse else [np.full(ring_sides, pole_idx), j, j_next], axis=-1)
    return (tris + np.array([0, ring_base, ring_base])).reshape(-1)


def tube_mesh_topology(
    num_strands: int, rings_per_strand: int, ring_sides: int, cap_rings: int = CAP_LATITUDE_RINGS
) -> np.ndarray:
    """Triangle indices for `num_strands` tubes with hemispherical end caps, in one mesh.

    Per-strand vertex layout (as `tube_mesh_points` emits it): body rings, start-cap rings
    (equator to pole), start pole, end-cap rings (equator to pole), end pole.

    Returns:
        Flat int32 triangle indices.
    """
    if ring_sides < 3:
        raise ValueError("ring_sides must be >= 3")

    start_cap_base = rings_per_strand * ring_sides
    start_pole = start_cap_base + cap_rings * ring_sides
    end_cap_base = start_pole + 1
    end_pole = end_cap_base + cap_rings * ring_sides
    vertices_per_strand = end_pole + 1

    tri_parts = [
        _ring_band_tris(ring_sides, i * ring_sides, (i + 1) * ring_sides, reverse=False)
        for i in range(rings_per_strand - 1)
    ]

    start_chain = [0] + [start_cap_base + k * ring_sides for k in range(cap_rings)]
    tri_parts += [
        _ring_band_tris(ring_sides, start_chain[k], start_chain[k + 1], reverse=True) for k in range(len(start_chain) - 1)
    ]
    tri_parts.append(_pole_fan_tris(ring_sides, start_chain[-1], start_pole, reverse=True))

    end_chain = [(rings_per_strand - 1) * ring_sides] + [end_cap_base + k * ring_sides for k in range(cap_rings)]
    tri_parts += [
        _ring_band_tris(ring_sides, end_chain[k], end_chain[k + 1], reverse=False) for k in range(len(end_chain) - 1)
    ]
    tri_parts.append(_pole_fan_tris(ring_sides, end_chain[-1], end_pole, reverse=False))

    strand_tris = np.concatenate(tri_parts)
    strand_offset = np.arange(num_strands) * vertices_per_strand
    all_tris = (strand_tris[None, :] + strand_offset[:, None]).reshape(-1)
    return all_tris.astype(np.int32)


def _hemisphere_cap_points_normals(
    center: np.ndarray,
    radius: float,
    axis_dir: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    cos_t: np.ndarray,
    sin_t: np.ndarray,
    cap_rings: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Latitude rings and pole of one hemispherical tube cap with its pole along `axis_dir`.

    The equator is the tube's own end ring and is not repeated here.

    Returns:
        (ring_points, ring_normals, pole_point, pole_normal); rings have shape
        (cap_rings, ring_sides, 3), ordered equator to pole.
    """
    phi = (np.arange(cap_rings) + 1) / (cap_rings + 1) * (np.pi / 2.0)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    radial_dirs = cos_t[:, None] * y_axis[None, :] + sin_t[:, None] * z_axis[None, :]

    ring_points = (
        center[None, None, :]
        + sin_phi[:, None, None] * radius * axis_dir[None, None, :]
        + cos_phi[:, None, None] * radius * radial_dirs[None, :, :]
    )
    ring_normals = cos_phi[:, None, None] * radial_dirs[None, :, :] + sin_phi[:, None, None] * axis_dir[None, None, :]
    return ring_points, ring_normals, center + radius * axis_dir, axis_dir


def tube_mesh_points(
    strand_poses: list[np.ndarray], radii: np.ndarray, ring_sides: int, cap_rings: int = CAP_LATITUDE_RINGS
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex positions and normals for a batch of backbone-pose strands, laid out as `tube_mesh_topology` expects.

    Args:
        strand_poses: One (N, 4, 4) pose array per strand, all on the same arc-length samples.
        radii: Cross-section radius per sample, shape (N,).
        ring_sides: Cross-section polygon resolution (>= 3).
        cap_rings: Latitude rings per hemispherical end cap.

    Returns:
        (points, normals), each float32 of shape (total_vertices, 3).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, ring_sides, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    all_points, all_normals = [], []
    for poses in strand_poses:
        positions = poses[:, :3, 3]
        x_axes = poses[:, :3, 0]
        y_axes = poses[:, :3, 1]
        z_axes = poses[:, :3, 2]
        radial = cos_t[None, :, None] * y_axes[:, None, :] + sin_t[None, :, None] * z_axes[:, None, :]
        ring_points = positions[:, None, :] + radii[:, None, None] * radial
        all_points.append(ring_points.reshape(-1, 3))
        all_normals.append(radial.reshape(-1, 3))

        start_rings, start_normals, start_pole_pt, start_pole_n = _hemisphere_cap_points_normals(
            positions[0], radii[0], -x_axes[0], y_axes[0], z_axes[0], cos_t, sin_t, cap_rings
        )
        end_rings, end_normals, end_pole_pt, end_pole_n = _hemisphere_cap_points_normals(
            positions[-1], radii[-1], x_axes[-1], y_axes[-1], z_axes[-1], cos_t, sin_t, cap_rings
        )
        all_points += [start_rings.reshape(-1, 3), start_pole_pt[None, :]]
        all_normals += [start_normals.reshape(-1, 3), start_pole_n[None, :]]
        all_points += [end_rings.reshape(-1, 3), end_pole_pt[None, :]]
        all_normals += [end_normals.reshape(-1, 3), end_pole_n[None, :]]

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return points, normals


def _solid_cylinder_mesh(center_x: float, center_y: float, center_z: float, radius: float, half_height: float, sides: int = 24):
    """Solid upright cylinder: side rings plus separately-normalled end caps.

    Returns:
        (points, indices, normals), the triple `viewer.log_mesh` takes.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rim_xy = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
    side_normals_xy = np.concatenate([rim_xy / radius, np.zeros((sides, 1), dtype=np.float32)], axis=1)

    bottom_rim = np.concatenate([rim_xy, np.full((sides, 1), -half_height, dtype=np.float32)], axis=1)
    top_rim = np.concatenate([rim_xy, np.full((sides, 1), half_height, dtype=np.float32)], axis=1)
    bottom_center = np.array([[0.0, 0.0, -half_height]], dtype=np.float32)
    top_center = np.array([[0.0, 0.0, half_height]], dtype=np.float32)

    # Rims appear twice: once with side normals, once with cap normals.
    points = np.concatenate([bottom_rim, top_rim, bottom_rim, top_rim, bottom_center, top_center], axis=0)
    normals = np.concatenate(
        [
            side_normals_xy, side_normals_xy,
            np.tile([[0.0, 0.0, -1.0]], (sides, 1)), np.tile([[0.0, 0.0, 1.0]], (sides, 1)),
            [[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]],
        ],
        axis=0,
    )
    cap_bottom_base, cap_top_base = 2 * sides, 3 * sides
    bottom_center_idx, top_center_idx = 4 * sides, 4 * sides + 1

    indices = []
    for i in range(sides):
        j = (i + 1) % sides
        indices.extend([i, j, sides + j, i, sides + j, sides + i])
        indices.extend([bottom_center_idx, cap_bottom_base + j, cap_bottom_base + i])
        indices.extend([top_center_idx, cap_top_base + i, cap_top_base + j])

    points = points + np.array([center_x, center_y, center_z], dtype=np.float32)
    return points.astype(np.float32), np.asarray(indices, dtype=np.int32), normals.astype(np.float32)


def _yawed_object_ghost_mesh(object_shape: str, center_x: float, center_y: float, center_z: float, half_extent: float, yaw: float):
    """Predicted-object geometry, turned by `yaw` about world z (ignored for the symmetric cylinder).

    Args:
        object_shape: "box" or "cylinder".
        center_x: Object-center world x [m].
        center_y: Object-center world y [m].
        center_z: Object-center world z [m].
        half_extent: Half side length (box) or radius and half height (cylinder) [m].
        yaw: Rotation about world z [rad].

    Returns:
        (points, indices, normals), the triple `viewer.log_mesh` takes.
    """
    if object_shape == "cylinder":
        return _solid_cylinder_mesh(center_x, center_y, center_z, half_extent, half_extent)

    h = half_extent
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    corners = np.array(
        [[sx * h, sy * h, sz * h] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], dtype=np.float32
    )
    points = corners @ rotation.T + np.array([center_x, center_y, center_z], dtype=np.float32)
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    face_normals = np.array(
        [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)], dtype=np.float32
    ) @ rotation.T
    tri_points, tri_normals, indices = [], [], []
    for (a, b, c, d), normal in zip(faces, face_normals):
        base = len(tri_points)
        tri_points.extend([points[a], points[b], points[c], points[d]])
        tri_normals.extend([normal] * 4)
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])
    return (
        np.asarray(tri_points, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(tri_normals, dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, required=True, help="Saved run with diagnostics_iter_X_samples/diagnostics_iter_cost.")
    parser.add_argument("--timestep", type=int, required=True, help="Real MPC step index (diagnostics_iter_X_samples' first axis).")
    parser.add_argument(
        "--iteration", type=int, required=True, choices=[0, 1, 2],
        help="CEM iteration within that MPC step (0=~200 samples, 1=~106, 2=~53).",
    )
    parser.add_argument(
        "--horizon-step", type=int, required=True,
        help="Which predicted horizon step (0 = current real state) to draw each sample's tentacle shape at.",
    )
    parser.add_argument("--object-shape", choices=("box", "cylinder"), default="box", help="Must match the run's --object-shape.")
    parser.add_argument("--device", default="cuda", help="ViewerRTX requires a real GPU; cpu will not work.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples-per-pixel", type=int, default=8)
    parser.add_argument("--backbone-points", type=int, default=25, help="Arc-length samples along each drawn tentacle shape.")
    parser.add_argument("--ring-sides", type=int, default=8, help="Cross-section polygon resolution (>=3) for the drawn tubes.")
    parser.add_argument(
        "--sample-opacity", type=float, default=0.5,
        help="Display opacity in [0, 1] of the best non-best cloud sample; the others scale down by "
             "cost rank to --sample-opacity * 0.1 at the worst.",
    )
    parser.add_argument(
        "--object-opacity", type=float, default=None,
        help="Opacity of the cloud samples' predicted-object boxes, scaled by cost rank like "
             "--sample-opacity. Defaults to --sample-opacity.",
    )
    parser.add_argument(
        "--max-object-samples", type=int, default=None,
        help="Draw predicted-object boxes for only this many of the best drawn samples (rods unaffected; "
             "default: all). The boxes' top faces are coplanar, so a ray crosses one per box, and "
             "ViewerRTX's bounded transmission-bounce budget turns deep stacks black. Lowering the "
             "opacity makes this worse; capping the box count fixes it.",
    )
    parser.add_argument(
        "--object-sink", type=float, default=0.001,
        help="Sink the translucent cloud boxes this far below the table top [m] so the solid best-sample "
             "box is no longer coplanar with them and reads unveiled. 0 disables it. Does not stop the "
             "cloud boxes blackening each other; --max-object-samples controls that.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of non-best sample tentacles drawn, evenly spaced across the cost ranking "
             "so the subset keeps the iteration's cost spread. Default (no cap) draws every sample.",
    )
    return parser.parse_args()


def _log_solid_mesh(viewer, name: str, mesh: tuple, color, opacity: float | None = None) -> None:
    """Log one (points, indices, normals) mesh with the matte material every cloud mesh uses."""
    points, indices, normals = mesh
    viewer.log_mesh(
        name,
        wp.array(points, dtype=wp.vec3),
        wp.array(indices, dtype=wp.int32),
        normals=wp.array(normals, dtype=wp.vec3),
        color=color,
        roughness=1.0,
        metallic=0.0,
        opacity=opacity,
    )


def main() -> None:
    args = parse_args()

    data = np.load(args.npz, allow_pickle=True)
    for required in ("X_final", "object_pose", "goal_xy", "reach_tolerance",
                     "diagnostics_iter_X_samples", "diagnostics_iter_cost"):
        if required not in data.files:
            raise ValueError(f"{args.npz} has no {required!r} array (files: {data.files})")

    X_final = data["X_final"]
    object_pose = data["object_pose"]
    goal_xy = data["goal_xy"]
    reach_tolerance = data["reach_tolerance"]
    iter_X_samples = data["diagnostics_iter_X_samples"]
    iter_cost = data["diagnostics_iter_cost"]

    num_timesteps = iter_X_samples.shape[0]
    if not (0 <= args.timestep < num_timesteps):
        raise ValueError(f"--timestep must be in [0, {num_timesteps}), got {args.timestep}")

    step_X = iter_X_samples[args.timestep, args.iteration]  # (N, horizon+1, state_dim)
    step_cost = iter_cost[args.timestep, args.iteration]  # (N,)
    if step_X.shape[0] != step_cost.shape[0]:
        raise ValueError(f"sample/cost count mismatch at (timestep={args.timestep}, iteration={args.iteration}): "
                         f"{step_X.shape[0]} samples vs {step_cost.shape[0]} costs")
    horizon = step_X.shape[1] - 1
    if not (0 <= args.horizon_step <= horizon):
        raise ValueError(f"--horizon-step must be in [0, {horizon}], got {args.horizon_step}")
    best_idx = int(np.argmin(step_cost))
    print(f"{step_X.shape[0]} samples at (timestep={args.timestep}, iteration={args.iteration}); "
          f"best sample index={best_idx}, cost={step_cost[best_idx]:.6g}")

    robot = FourTendonSoftArm().model
    ndof = robot.num_internal_dofs
    num_base_coordinates = robot.num_base_coordinates
    state_dim = 2 * ndof + 5 + num_base_coordinates
    if X_final.shape[1] != state_dim:
        raise ValueError(f"X_final has state_dim={X_final.shape[1]}, expected {state_dim} for this robot")

    # Never stepped: sync_kinematics() alone places the rod from the saved q.
    model, box_bodies, coupled_solver, _, _ = run_experiment.build_scene(
        robot, args.device, object_shape=args.object_shape, object_mass=OBJECT_MASS, world_count=1,
        num_collision_bodies=ROD_COLLISION_BODIES, tentacle_color=SAMPLE_COLOR, object_color=SAMPLE_COLOR,
        table_color=TABLE_COLOR,
    )
    gvs_solver = coupled_solver.solver("gvs")
    state = model.state()
    q_view, qd_view = gvs_state_views(
        state, world_count=1, coordinate_count=robot.num_coordinates, velocity_count=robot.num_velocities
    )
    box_body = int(box_bodies[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    viewer = ExampleViewerRTX(
        width=args.width, height=args.height, headless=True, fps=30, samples_per_pixel=args.samples_per_pixel,
    )
    viewer.set_model(model)
    look_at(viewer, CAMERA_EYE, CAMERA_TARGET)

    # New named geometry must be logged before the first begin_frame(): ViewerRTX bakes it into the build-phase scene.
    goal_corners, goal_meshes = build_goal_overlay(
        viewer, object_pose, goal_xy, reach_tolerance, run_experiment.OBJECT_FOOTPRINT, args.object_shape
    )

    s_ps = np.linspace(0.0, float(robot.length), args.backbone_points)
    radii = cross_section_radii(robot, s_ps, ndof, num_base_coordinates)
    tube_indices = tube_mesh_topology(1, len(s_ps), args.ring_sides)
    object_half_extent = 0.5 * run_experiment.OBJECT_FOOTPRINT
    object_opacity = args.sample_opacity if args.object_opacity is None else args.object_opacity

    def rod_mesh(k: int) -> tuple:
        poses = _rollout_backbone_poses(robot, step_X[k, args.horizon_step], ndof, num_base_coordinates, s_ps)
        points, normals = tube_mesh_points([poses], radii, args.ring_sides)
        return points, tube_indices, normals

    def object_mesh(k: int, center_z: float) -> tuple:
        row = step_X[k, args.horizon_step]
        return _yawed_object_ghost_mesh(
            args.object_shape, row[2 * ndof], row[2 * ndof + 2], center_z, object_half_extent, row[2 * ndof + 4]
        )

    cloud_idx = sorted((k for k in range(step_X.shape[0]) if k != best_idx), key=lambda k: step_cost[k])
    if args.max_samples is not None and args.max_samples < len(cloud_idx):
        pick_positions = np.unique(np.linspace(0, len(cloud_idx) - 1, args.max_samples).round().astype(int))
        cloud_idx = [cloud_idx[p] for p in pick_positions]
    n_cloud = len(cloud_idx)
    n_object_boxes = n_cloud if args.max_object_samples is None else min(n_cloud, args.max_object_samples)
    print(f"drawing {n_cloud} of {step_X.shape[0] - 1} cloud samples (evenly spaced by cost rank), "
          f"{n_object_boxes} of them with a predicted-object box")

    # Opacity falls linearly with cost rank; one log_mesh per sample since opacity is per call.
    sample_mesh_names = []
    for rank, k in enumerate(cloud_idx):
        factor = 1.0 if n_cloud <= 1 else 1.0 - (1.0 - WORST_SAMPLE_OPACITY_FACTOR) * (rank / (n_cloud - 1))
        name = f"sample_{k}"
        sample_mesh_names.append(name)
        _log_solid_mesh(viewer, name, rod_mesh(k), SAMPLE_COLOR, args.sample_opacity * factor)
        if rank < n_object_boxes:
            sample_mesh_names.append(f"{name}_object")
            _log_solid_mesh(
                viewer, f"{name}_object", object_mesh(k, object_half_extent - args.object_sink),
                SAMPLE_COLOR, object_opacity * factor,
            )

    _log_solid_mesh(viewer, "best_sample", rod_mesh(best_idx), BEST_PREDICTION_COLOR)
    _log_solid_mesh(viewer, "best_sample_object", object_mesh(best_idx, object_half_extent), BEST_PREDICTION_COLOR)
    # Exploratory, untested on a GPU: overlapping translucent tubes showed dark streaks that may be cast shadows.
    viewer.shadow_excluded_mesh_names = (*sample_mesh_names, "best_sample", "best_sample_object")

    # Real executed rod/object at --timestep, drawn underneath the cloud for context.
    row = X_final[args.timestep]
    q_internal = row[:ndof]
    qd_internal = row[ndof : 2 * ndof]
    cube_x, cube_vx, cube_y, cube_vy = row[2 * ndof : 2 * ndof + 4]
    # row[2 * ndof + 4] is object_yaw; the base pose starts one column later.
    base_pose = row[2 * ndof + 5 : 2 * ndof + 5 + num_base_coordinates]

    q_full = q_view.numpy()
    qd_full = qd_view.numpy()
    q_full[:, num_base_coordinates:] = q_internal[None, :]
    qd_full[:, robot.num_base_velocities:] = qd_internal[None, :]
    q_view.assign(q_full)
    qd_view.assign(qd_full)
    gvs_solver.set_kinematic_base_targets(
        wp.array(base_pose[None, :], dtype=wp.float64, device=gvs_solver.device),
        wp.zeros((1, 6), dtype=wp.float64, device=gvs_solver.device),
        wp.zeros((1, 6), dtype=wp.float64, device=gvs_solver.device),
    )

    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    body_q[box_body, 0] = cube_x
    body_q[box_body, 1] = cube_y
    body_q[box_body, 2:7] = object_pose[args.timestep, 2:7]
    body_qd[box_body, 0] = cube_vx
    body_qd[box_body, 1] = cube_vy
    state.body_q.assign(body_q)
    state.body_qd.assign(body_qd)
    gvs_solver.sync_kinematics(state)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    # Visibility toggles must land between begin_frame() and end_frame(); begin_frame() clears earlier ones.
    update_goal_overlay(viewer, goal_xy, goal_corners, goal_meshes, args.timestep)
    viewer.end_frame()
    viewer.save_screenshot(str(args.output))
    viewer.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
