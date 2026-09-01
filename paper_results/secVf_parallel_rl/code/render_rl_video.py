#!/usr/bin/env python3
"""Render saved Section Vf RL rollouts as MP4 and GIF videos."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/soromox_matplotlib")

import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.rendering.camera_config import CameraConfig
from soromox.rendering.open3d_renderer import Open3DRenderer
from soromox.rendering.video_encoding import FFmpegVideoWriter, VideoEncodingConfig
from soromox.systems import PCS, LinkSpec

if __package__:
    from .rl_render_style import (
        BACKBONE_NUM_POINTS,
        BACKGROUND_COLOR,
        RENDER_HEIGHT,
        RENDER_WIDTH,
        TARGET_COLOR,
        TARGET_SPHERE_RADIUS,
        make_rl_camera_config,
        make_rl_color_config,
        make_target_trail_spheres,
    )
else:
    from rl_render_style import (
        BACKBONE_NUM_POINTS,
        BACKGROUND_COLOR,
        RENDER_HEIGHT,
        RENDER_WIDTH,
        TARGET_COLOR,
        TARGET_SPHERE_RADIUS,
        make_rl_camera_config,
        make_rl_color_config,
        make_target_trail_spheres,
    )

CASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CASE_DIR / "data"
OUTPUT_DIR = CASE_DIR / "outputs"

DEFAULT_ARM_LENGTH = 0.25
DEFAULT_ARM_RADIUS = 0.025
DEFAULT_GRID_SPACING = 0.16
GRID_CAMERA_DISTANCE_FACTOR = 5.5
# Camera offsets are expressed in the arm's base frame. This transforms to the
# symmetric world-grid direction (0.8, -0.8, 0.9), retaining an elevated view.
GRID_CAMERA_POSITION_OFFSET = (0.9, -0.8, 0.8)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class RLRollout:
    """Normalized renderer input with environment-first trajectory arrays."""

    ts: np.ndarray
    q_ts: np.ndarray  # (N, T, D)
    ball_ts: np.ndarray  # (N, T, 3)
    arm_length: float
    arm_radius: float
    policy: str | None

    @property
    def num_envs(self) -> int:
        return int(self.q_ts.shape[0])


class Progress:
    """Small tqdm wrapper with a print-only fallback."""

    def __init__(self, total: int, desc: str):
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self._next_percent = 0
        try:
            from tqdm import tqdm
        except ImportError:
            self._bar = None
            print(f"{self.desc}: 0/{self.total}")
        else:
            self._bar = tqdm(total=self.total, desc=self.desc)

    def update(self, amount: int = 1) -> None:
        self.current += amount
        if self._bar is not None:
            self._bar.update(amount)
            return

        percent = int(self.current * 100 / self.total)
        if percent >= self._next_percent or self.current >= self.total:
            print(f"{self.desc}: {self.current}/{self.total} ({percent}%)")
            self._next_percent = percent + 10

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


class HeadlessRLVideoRenderer(Open3DRenderer):
    """Open3D renderer that records deterministically without an interactive loop."""

    def __init__(
        self,
        *args,
        visible: bool = False,
        progress: Progress | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.visible = visible
        self.progress = progress

    def _create_visualizer(self, window_name: str):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name=window_name,
            width=self.width,
            height=self.height,
            visible=self.visible,
        )
        opt = vis.get_render_option()
        opt.background_color = np.array(self.background_color, dtype=np.float64)
        opt.line_width = self.actuator_line_width
        return vis, vis.get_view_control()

    def _run_viewer(
        self,
        scene_data,
        *,
        playback_speed,
        autoplay,
        loop,
        record_cfg,
        window_name,
        camera_config=None,
        color_config=None,
    ) -> None:
        if record_cfg.path is None:
            raise ValueError("record_path is required for MP4 rendering.")

        video_path = Path(record_cfg.path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_indices = list(range(0, scene_data.num_frames, record_cfg.every_n))
        if not frame_indices or frame_indices[-1] != scene_data.num_frames - 1:
            frame_indices.append(scene_data.num_frames - 1)

        dt_seq = np.diff(np.asarray(scene_data.ts, dtype=np.float64))
        fps = 15.0 if dt_seq.size == 0 else 1.0 / max(1e-6, float(np.median(dt_seq)))
        fps /= max(1, record_cfg.every_n)

        video_writer = FFmpegVideoWriter(
            str(video_path),
            int(self.width),
            int(self.height),
            fps,
            input_pix_fmt="rgb24",
            video_config=record_cfg.video_config,
        )
        print(f"[Open3D] Writing MP4: {video_path} (fps={fps:.2f})")

        vis, ctrl = self._create_visualizer(window_name)
        try:
            print(
                f"[Open3D] Building initial scene for {scene_data.num_robots} arm(s)..."
            )
            build_started = time.perf_counter()
            handles = self._build_scene(
                vis,
                scene_data,
                frame_idx=0,
                color_config=color_config,
            )
            print(
                f"[Open3D] Initial scene ready in "
                f"{time.perf_counter() - build_started:.2f}s"
            )
            self._setup_interactive_camera(vis, ctrl, scene_data, camera_config)
            if self.progress is None:
                self.progress = Progress(len(frame_indices), "Render frames")

            for frame_idx in frame_indices:
                self._update_scene(
                    vis,
                    scene_data,
                    handles,
                    frame_idx=frame_idx,
                    color_config=color_config,
                )
                vis.poll_events()
                vis.update_renderer()
                image = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True),
                    dtype=np.float32,
                )
                video_writer.write(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))
                if self.progress is not None:
                    self.progress.update()
        finally:
            video_writer.close()
            if video_writer.stderr_log:
                stderr_tail = "\n".join(video_writer.stderr_log.splitlines()[-8:])
                if stderr_tail:
                    print(f"[ffmpeg] {stderr_tail}")
            vis.destroy_window()


def build_rl_robot(
    arm_length: float = DEFAULT_ARM_LENGTH,
    arm_radius: float = DEFAULT_ARM_RADIUS,
) -> PCS:
    """Reconstruct the static robot parameters to match the RL environment."""
    num_segments = 1
    rho = 1070 * jnp.ones((num_segments,))
    segment_length = jnp.array([arm_length])
    backbone_radius = jnp.array([arm_radius])
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]) * segment_length[:, None]
        ).flatten()
    )

    r0 = float(backbone_radius[0]) - 0.005
    theta0 = jnp.deg2rad(jnp.array([0, 90, 180, 270]))
    ry0, rz0 = r0 * jnp.cos(theta0), r0 * jnp.sin(theta0)
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=jnp.stack((jnp.zeros(4), ry0, rz0), axis=-1),
        slope=jnp.zeros((4, 3)),
        start_segment_index=0,
        end_segment_index=(0, 0, 0, 0),
    )

    body_params = PCS.params_from_links(
        [
            LinkSpec.circular(
                length=float(segment_length[index]),
                radius=float(backbone_radius[index]),
                density=float(rho[index]),
                young_modulus=20e3,
                shear_modulus=20e3,
                damping=damping_matrix[
                    6 * index : 6 * (index + 1),
                    6 * index : 6 * (index + 1),
                ],
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for index in range(num_segments)
        ],
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )
    return PCS(
        params=body_params,
        base_pose=jnp.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
    )


def _scalar_metadata(data: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in data.files:
        return default
    value = np.asarray(data[key], dtype=np.float64)
    if value.size != 1:
        raise ValueError(f"{key} metadata must be scalar; got shape {value.shape}")
    scalar = float(value.reshape(()))
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{key} metadata must be a positive finite value")
    return scalar


def _string_metadata(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} metadata must be scalar; got shape {value.shape}")
    return str(value.reshape(()))


def load_rollout(path: Path, *, max_envs: int | None = None) -> RLRollout:
    """Load either unbatched or time-major batched Section Vf trajectory data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find trajectory file: {path}")
    if max_envs is not None and max_envs <= 0:
        raise ValueError("--max-envs must be positive")

    with np.load(path) as data:
        missing = [key for key in ("ts", "q_ts", "ball_ts") if key not in data.files]
        if missing:
            raise ValueError(
                "Trajectory file is missing array(s): " + ", ".join(missing)
            )
        ts = np.asarray(data["ts"], dtype=np.float64)
        q_time_first = np.asarray(data["q_ts"], dtype=np.float64)
        ball_time_first = np.asarray(data["ball_ts"], dtype=np.float64)
        arm_length = _scalar_metadata(data, "arm_length", DEFAULT_ARM_LENGTH)
        arm_radius = _scalar_metadata(data, "arm_radius", DEFAULT_ARM_RADIUS)
        policy = _string_metadata(data, "policy")
        if policy not in (None, "trained", "random"):
            raise ValueError(
                f"policy metadata must be 'trained' or 'random'; got {policy!r}"
            )

    if ts.ndim != 1 or ts.size == 0:
        raise ValueError(f"ts must have shape (T,) with T > 0; got {ts.shape}")
    if q_time_first.ndim == 2:
        q_time_first = q_time_first[:, None, :]
    elif q_time_first.ndim != 3:
        raise ValueError(
            f"q_ts must have shape (T,D) or (T,N,D); got {q_time_first.shape}"
        )
    if ball_time_first.ndim == 2:
        ball_time_first = ball_time_first[:, None, :]
    elif ball_time_first.ndim != 3:
        raise ValueError(
            f"ball_ts must have shape (T,3) or (T,N,3); got {ball_time_first.shape}"
        )

    num_steps, num_envs = q_time_first.shape[:2]
    if num_steps != ts.size:
        raise ValueError(
            f"q_ts time dimension ({num_steps}) must match ts length ({ts.size})"
        )
    if ball_time_first.shape[:2] != (num_steps, num_envs):
        raise ValueError(
            "ball_ts time/environment dimensions must match q_ts; "
            f"got {ball_time_first.shape[:2]} and {(num_steps, num_envs)}"
        )
    if ball_time_first.shape[2] != 3:
        raise ValueError(
            f"ball_ts positions must have size 3; got {ball_time_first.shape}"
        )
    if num_envs <= 0:
        raise ValueError("Trajectory data must contain at least one environment")

    selected_envs = num_envs if max_envs is None else min(max_envs, num_envs)
    return RLRollout(
        ts=ts,
        q_ts=np.transpose(q_time_first[:, :selected_envs, :], (1, 0, 2)),
        ball_ts=np.transpose(ball_time_first[:, :selected_envs, :], (1, 0, 2)),
        arm_length=arm_length,
        arm_radius=arm_radius,
        policy=policy,
    )


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three comma-separated floats, got: {value!r}"
        )
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected three comma-separated floats, got: {value!r}"
        ) from exc


def resolve_grid_dimensions(
    num_envs: int,
    rows: int | None,
    cols: int | None,
) -> tuple[int, int]:
    """Resolve and validate a centered near-square or explicit grid."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if (rows is None) != (cols is None):
        raise ValueError("--rows and --cols must be provided together")
    if rows is None:
        resolved_cols = int(math.ceil(math.sqrt(num_envs)))
        resolved_rows = int(math.ceil(num_envs / resolved_cols))
        return resolved_rows, resolved_cols
    if rows <= 0 or cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if rows * cols < num_envs:
        raise ValueError(
            f"--rows * --cols must cover {num_envs} environment(s); got {rows}x{cols}"
        )
    return rows, cols


def make_grid_offsets(
    num_envs: int,
    rows: int,
    cols: int,
    spacing: float,
) -> np.ndarray:
    """Return centered XYZ base offsets for a row-major grid."""
    if spacing <= 0.0:
        raise ValueError("--grid-spacing must be positive")
    if rows <= 0 or cols <= 0 or rows * cols < num_envs:
        raise ValueError(f"rows * cols must cover num_envs; got {rows}x{cols}")
    offsets = np.zeros((num_envs, 3), dtype=np.float64)
    for env_idx in range(num_envs):
        row, col = divmod(env_idx, cols)
        offsets[env_idx] = (
            (col - (cols - 1) / 2.0) * spacing,
            (row - (rows - 1) / 2.0) * spacing,
            0.0,
        )
    return offsets


def make_render_camera_config(
    *,
    num_envs: int,
    arm_length: float,
    fov: float,
    up: tuple[float, float, float],
    distance_factor: float | None,
    position_offset: tuple[float, float, float] | None,
) -> CameraConfig:
    """Select the fixed paper camera or a scene-aware automatic camera."""
    manual_auto_camera = distance_factor is not None or position_offset is not None
    if num_envs == 1 and not manual_auto_camera:
        return make_rl_camera_config(arm_length, fov=fov, up=up)

    defaults = (
        CameraConfig()
        if num_envs == 1
        else CameraConfig(
            distance_factor=GRID_CAMERA_DISTANCE_FACTOR,
            position_offset=GRID_CAMERA_POSITION_OFFSET,
        )
    )
    return CameraConfig(
        fov=fov,
        distance_factor=(
            defaults.distance_factor if distance_factor is None else distance_factor
        ),
        position_offset=(
            defaults.position_offset if position_offset is None else position_offset
        ),
        up=up,
    )


def resolve_output_paths(
    output: Path,
    gif_output: Path | None,
) -> tuple[Path, Path]:
    """Resolve distinct MP4 and GIF output paths."""
    mp4_path = Path(output).resolve()
    gif_path = (
        mp4_path.with_suffix(".gif")
        if gif_output is None
        else Path(gif_output).resolve()
    )
    if mp4_path == gif_path:
        raise ValueError("MP4 and GIF output paths must be different")
    return mp4_path, gif_path


def default_video_output(data_path: Path) -> Path:
    """Derive a case-local MP4 filename from the trajectory input filename."""
    return OUTPUT_DIR / f"{Path(data_path).stem}.mp4"


def preflight_outputs(paths: tuple[Path, ...], *, force: bool) -> None:
    """Enforce overwrite safety and create output directories."""
    if len(set(paths)) != len(paths):
        raise ValueError("Output paths must be unique")
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = ", ".join(map(str, existing))
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {formatted}; pass --force"
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def encode_gif_from_mp4(
    *,
    mp4_path: Path,
    gif_path: Path,
    fps: int,
    width: int,
) -> None:
    """Encode a looping palette-optimized GIF from the rendered MP4."""
    filters = [f"fps={fps}"]
    if width > 0:
        filters.append(f"scale={width}:-1:flags=lanczos")
    filters.append(
        "split[s0][s1];[s0]palettegen=max_colors=128[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vf",
        ",".join(filters),
        "-loop",
        "0",
        str(gif_path),
    ]
    print(f"Encoding GIF: {gif_path}")
    subprocess.run(command, check=True)


def render_rollout_to_mp4(
    rollout: RLRollout,
    *,
    output: Path,
    color_label: str,
    args: argparse.Namespace,
) -> None:
    """Render a normalized single- or multi-environment rollout to MP4."""
    rows, cols = resolve_grid_dimensions(rollout.num_envs, args.rows, args.cols)
    offsets = make_grid_offsets(
        rollout.num_envs,
        rows,
        cols,
        args.grid_spacing,
    )
    shifted_ball_ts = rollout.ball_ts + offsets[:, None, :]
    robot = build_rl_robot(rollout.arm_length, rollout.arm_radius)
    camera_config = make_render_camera_config(
        num_envs=rollout.num_envs,
        arm_length=rollout.arm_length,
        fov=args.camera_fov,
        up=args.camera_up,
        distance_factor=args.camera_distance_factor,
        position_offset=args.camera_position_offset,
    )

    static_positions = static_radii = static_colors = None
    if args.show_trajectory:
        static_positions, static_radii, static_colors = make_target_trail_spheres(
            shifted_ball_ts
        )

    renderer = HeadlessRLVideoRenderer(
        robot,
        width=args.width,
        height=args.height,
        num_points=args.num_points,
        color_config=make_rl_color_config(color_label),
        backbone_style="discrete",
        recompute_normals=False,
        background_color=BACKGROUND_COLOR,
        sphere_resolution=args.sphere_resolution,
        actuator_line_width=args.tendon_line_width,
        grid_spacing=(args.grid_spacing, args.grid_spacing),
        base_offsets=offsets,
        ground_plane_size=(args.grid_spacing if rollout.num_envs > 1 else None),
        visible=args.visible,
    )
    print("Precomputing vectorized scene geometry...")
    try:
        renderer.render_sequence(
            ts=rollout.ts,
            q_ts=rollout.q_ts,
            playback_speed=1.0,
            autoplay=True,
            loop=False,
            record_path=str(output),
            record_every_n=args.record_every_n,
            video_config=VideoEncodingConfig(
                codec="libx264",
                pix_fmt="yuv420p",
                preset=args.preset,
                crf=args.crf,
                tune=None,
                extra_args=("-movflags", "+faststart"),
            ),
            base_offsets=offsets,
            camera_config=camera_config,
            render_actuators=True,
            static_spheres_positions=static_positions,
            static_spheres_radii=static_radii,
            static_spheres_colors=static_colors,
            dynamic_spheres_positions=shifted_ball_ts,
            dynamic_spheres_radii=np.full(
                (rollout.num_envs,), TARGET_SPHERE_RADIUS, dtype=np.float64
            ),
            dynamics_spheres_colors=np.tile(
                np.asarray(TARGET_COLOR, dtype=np.float64)[None, :],
                (rollout.num_envs, 1),
            ),
            window_name="SoRoMoX PPO Rollout",
        )
    finally:
        if renderer.progress is not None:
            renderer.progress.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for offline rollout rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "traj" / "rl_rollout_trained_1_env.npz",
        help="Saved single- or multi-environment rollout NPZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4. Defaults to outputs/<data filename>.mp4.",
    )
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to --output with a .gif suffix.",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=None,
        help="Render only the first N stored environments. Defaults to all.",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument(
        "--show-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show each target ball's dotted trajectory trail.",
    )
    parser.add_argument("--width", type=int, default=RENDER_WIDTH)
    parser.add_argument("--height", type=int, default=RENDER_HEIGHT)
    parser.add_argument("--num-points", type=int, default=BACKBONE_NUM_POINTS)
    parser.add_argument("--sphere-resolution", type=int, default=12)
    parser.add_argument("--tendon-line-width", type=float, default=1.0)
    parser.add_argument("--record-every-n", type=int, default=1)
    parser.add_argument("--camera-fov", type=float, default=60.0)
    parser.add_argument(
        "--camera-distance-factor",
        type=float,
        default=None,
        help="Use automatic camera placement with this distance factor.",
    )
    parser.add_argument(
        "--camera-position-offset",
        type=parse_vec3,
        default=None,
        help="Use automatic camera placement with this direction vector.",
    )
    parser.add_argument("--camera-up", type=parse_vec3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--visible", action="store_true")

    parser.add_argument("--crf", type=int, default=18, help="Lower is higher quality.")
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument(
        "--gif-width",
        type=int,
        default=720,
        help="GIF width in pixels. Use 0 to keep the MP4 width.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = default_video_output(args.data)
    if args.gif_output is None:
        args.gif_output = args.output.with_suffix(".gif")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.record_every_n <= 0:
        raise ValueError("--record-every-n must be positive")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive")
    if args.gif_width < 0:
        raise ValueError("--gif-width must be nonnegative")
    if args.width <= 0 or args.height <= 0 or args.num_points <= 0:
        raise ValueError("--width, --height, and --num-points must be positive")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    data_path = args.data.resolve()
    rollout = load_rollout(data_path, max_envs=args.max_envs)
    output, gif_output = resolve_output_paths(args.output, args.gif_output)
    preflight_outputs((output, gif_output), force=args.force)
    is_random_policy = rollout.policy == "random" or (
        rollout.policy is None
        and any(label in str(data_path) for label in ("random", "initialized"))
    )
    color_label = "pre_opt_1" if is_random_policy else "post_opt_1"

    print(f"Trajectory data: {data_path}")
    print(f"Rendering {rollout.num_envs} environment(s)")
    print(f"Output MP4: {output}")
    print(f"Output GIF: {gif_output}")
    render_rollout_to_mp4(
        rollout,
        output=output,
        color_label=color_label,
        args=args,
    )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not created correctly: {output}")
    encode_gif_from_mp4(
        mp4_path=output,
        gif_path=gif_output,
        fps=args.gif_fps,
        width=args.gif_width,
    )
    if not gif_output.exists() or gif_output.stat().st_size == 0:
        raise RuntimeError(f"GIF was not created correctly: {gif_output}")
    print(f"Saved MP4: {output}")
    print(f"Saved GIF: {gif_output}")


if __name__ == "__main__":
    main()
