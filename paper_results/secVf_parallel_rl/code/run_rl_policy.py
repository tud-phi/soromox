"""Generate a Section Vf PPO rollout and optionally render it directly.

The saved NPZ uses the schema consumed by ``render_rl_video.py``. Direct
multi-environment MP4/GIF rendering remains available through ``--render``.
"""

import argparse
import importlib.util
import math
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
DATA_DIR = CASE_DIR / "data"
OUTPUTS_DIR = CASE_DIR / "outputs"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/soromox_matplotlib")

import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

if __package__:
    from .parallel_soromox_env import ParallelSoromoxEnv
else:
    env_path = CODE_DIR / "parallel_soromox_env.py"
    spec = importlib.util.spec_from_file_location("parallel_soromox_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ParallelSoromoxEnv from {env_path}")
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    ParallelSoromoxEnv = env_module.ParallelSoromoxEnv

try:
    import open3d as o3d

    from soromox.rendering.camera_config import CameraConfig
    from soromox.rendering.color_config import (
        ActuatorStyleConfig,
        BackboneColorConfig,
        RendererColorConfig,
    )
    from soromox.rendering.open3d_renderer import Open3DRenderer
    from soromox.rendering.video_encoding import FFmpegVideoWriter, VideoEncodingConfig
except ImportError as exc:
    o3d = None
    CameraConfig = None
    ActuatorStyleConfig = None
    BackboneColorConfig = None
    RendererColorConfig = None
    FFmpegVideoWriter = None
    VideoEncodingConfig = None
    Open3DRenderer = object
    OPEN3D_IMPORT_ERROR = exc
else:
    OPEN3D_IMPORT_ERROR = None


DEFAULT_MODEL_PATH = DATA_DIR / "checkpoints" / "ppo_model.zip"
DEFAULT_VECNORMALIZE_PATH = DATA_DIR / "checkpoints" / "env_vecnormalize.pkl"
DEFAULT_TRAJECTORY_OUTPUT = DATA_DIR / "traj" / "open3d_rollout_data.npz"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "parallel_track_video.mp4"


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


def to_numpy(value, dtype=None) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    return array.astype(dtype) if dtype is not None else array


def extract_q_batch(raw_env: ParallelSoromoxEnv) -> np.ndarray:
    if raw_env._env_states is None:
        raise RuntimeError("Environment must be reset before extracting q.")
    y = raw_env._env_states.arm_state.y
    q, _ = jnp.split(y, 2, axis=1)
    return to_numpy(q, dtype=np.float64)


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


def make_grid_offsets(
    num_envs: int, rows: int, cols: int, spacing: float
) -> np.ndarray:
    if rows * cols < num_envs:
        raise ValueError(f"rows * cols must cover num_envs; got {rows}x{cols}")

    offsets = np.zeros((num_envs, 3), dtype=np.float64)
    for env_idx in range(num_envs):
        row, col = divmod(env_idx, cols)
        offsets[env_idx] = [
            (col - (cols - 1) / 2.0) * spacing,
            (row - (rows - 1) / 2.0) * spacing,
            0.0,
        ]
    return offsets


def auto_grid(num_envs: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    return rows, cols


def make_lineset(points: np.ndarray, color: tuple[float, float, float]):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None

    lines = np.stack(
        [np.arange(0, pts.shape[0] - 1), np.arange(1, pts.shape[0])],
        axis=1,
    ).astype(np.int32)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64)[None, :], (lines.shape[0], 1))
    )
    return line_set


class HeadlessMultiArmVideoRenderer(Open3DRenderer):
    def __init__(
        self,
        *args,
        ball_trajectories: np.ndarray | None = None,
        trajectory_color: tuple[float, float, float] = (224 / 255, 56 / 255, 62 / 255),
        visible: bool = False,
        progress: Progress | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ball_trajectories = ball_trajectories
        self.trajectory_color = trajectory_color
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

    def _build_scene(self, vis, scene_data, frame_idx=0, *, color_config=None):
        handles = super()._build_scene(
            vis,
            scene_data,
            frame_idx=frame_idx,
            color_config=color_config,
        )
        if self.ball_trajectories is not None:
            for traj in np.asarray(self.ball_trajectories, dtype=np.float64):
                line_set = make_lineset(traj, self.trajectory_color)
                if line_set is not None:
                    vis.add_geometry(line_set)
        return handles

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
        fps = fps / max(1, record_cfg.every_n)

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
            handles = self._build_scene(
                vis,
                scene_data,
                frame_idx=0,
                color_config=color_config,
            )
            self._setup_interactive_camera(vis, ctrl, scene_data, camera_config)

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
                img = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True),
                    dtype=np.float32,
                )
                frame = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
                video_writer.write(frame)
                if self.progress is not None:
                    self.progress.update()
        finally:
            video_writer.close()
            if video_writer.stderr_log:
                stderr_tail = "\n".join(video_writer.stderr_log.splitlines()[-8:])
                if stderr_tail:
                    print(f"[ffmpeg] {stderr_tail}")
            vis.destroy_window()


def run_policy_and_collect(
    args: argparse.Namespace,
) -> tuple[
    ParallelSoromoxEnv,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    raw_env = ParallelSoromoxEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        game_time=args.game_time,
        control_fps=args.control_fps,
        length=args.arm_length,
        radius=args.arm_radius,
        ball_radius=args.ball_radius,
        ball_surface_vmax=args.ball_surface_vmax,
        success_threshold=args.success_threshold,
    )
    vec_env = VecNormalize.load(str(args.vecnormalize_path), raw_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(str(args.model_path), env=vec_env)

    obs = raw_env.reset()
    obs = vec_env.normalize_obs(obs)

    q_list = [extract_q_batch(raw_env)]
    ball_list = [to_numpy(raw_env._env_states.ball_pos, dtype=np.float64)]
    action_list = []
    reward_list = []

    progress = Progress(args.n_steps, "Rollout")
    try:
        for _ in range(args.n_steps):
            actions, _ = model.predict(obs, deterministic=not args.stochastic)
            actions = np.asarray(actions, dtype=np.float32)
            (
                env_states,
                raw_obs,
                _rewards,
                _terminateds,
                _truncateds,
                _ee_pos,
                _invalid_obs,
            ) = raw_env._step_batch(raw_env._env_states, actions)
            raw_env._env_states = env_states
            obs = vec_env.normalize_obs(to_numpy(raw_obs, dtype=np.float32))
            action_list.append(actions)
            reward_list.append(to_numpy(_rewards, dtype=np.float64))
            q_list.append(extract_q_batch(raw_env))
            ball_list.append(to_numpy(raw_env._env_states.ball_pos, dtype=np.float64))
            progress.update()
    finally:
        progress.close()

    q_ts = np.asarray(q_list, dtype=np.float64)
    ball_ts = np.asarray(ball_list, dtype=np.float64)
    actions = np.asarray(action_list, dtype=np.float64)
    rewards = np.asarray(reward_list, dtype=np.float64)
    ts = np.arange(q_ts.shape[0], dtype=np.float64) / float(args.control_fps)
    vec_env.close()
    return raw_env, ts, q_ts, ball_ts, actions, rewards


def render_rollout_to_mp4(
    raw_env: ParallelSoromoxEnv,
    ts: np.ndarray,
    q_ts_time_first: np.ndarray,
    ball_ts_time_first: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if OPEN3D_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Open3D rendering dependencies are not available. Install the example "
            "dependencies with `pip install soromox[examples]` and ensure `open3d` "
            "is installed."
        ) from None

    num_envs = min(args.max_envs, q_ts_time_first.shape[1])
    q_ts_time_first = q_ts_time_first[:, :num_envs, :]
    ball_ts_time_first = ball_ts_time_first[:, :num_envs, :]

    rows, cols = (args.rows, args.cols)
    if rows is None or cols is None:
        rows, cols = auto_grid(num_envs)
    offsets = make_grid_offsets(num_envs, rows, cols, args.grid_spacing)

    q_ts_batched = np.transpose(q_ts_time_first, (1, 0, 2))
    ball_ts_batched = np.transpose(ball_ts_time_first, (1, 0, 2)) + offsets[:, None, :]

    color_config = RendererColorConfig(
        backbone=BackboneColorConfig(
            segment_colors=np.array([[0.0039, 0.6510, 0.8392, 1.0]], dtype=np.float64)
        ),
        base_plate_color=(0.2, 0.2, 0.2),
        actuators=ActuatorStyleConfig(default_color=(0.9, 0.15, 0.15)),
    )
    camera_config = CameraConfig(
        fov=args.camera_fov,
        distance_factor=args.camera_distance_factor,
        position_offset=args.camera_position_offset,
        up=args.camera_up,
    )

    frame_count = len(list(range(0, len(ts), args.record_every_n)))
    if (len(ts) - 1) % args.record_every_n != 0:
        frame_count += 1
    progress = Progress(frame_count, "Render/encode")
    renderer = HeadlessMultiArmVideoRenderer(
        raw_env.arm,
        width=args.width,
        height=args.height,
        num_points=args.num_points,
        backbone_style="discrete",
        background_color=(1.0, 1.0, 1.0),
        sphere_resolution=args.sphere_resolution,
        actuator_line_width=args.tendon_line_width,
        grid_spacing=(args.grid_spacing, args.grid_spacing),
        base_offsets=offsets,
        ball_trajectories=None if args.no_trajectories else ball_ts_batched,
        visible=args.visible,
        progress=progress,
    )

    try:
        renderer.render_sequence(
            ts=ts,
            q_ts=q_ts_batched,
            playback_speed=1.0,
            autoplay=True,
            loop=False,
            record_path=str(args.output),
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
            color_config=color_config,
            camera_config=camera_config,
            render_actuators=True,
            dynamic_spheres_positions=ball_ts_batched,
            dynamic_spheres_radii=np.full((num_envs,), 0.02, dtype=np.float64),
            dynamics_spheres_colors=np.tile(
                np.array([[1.0, 170 / 255, 0.0]], dtype=np.float64),
                (num_envs, 1),
            ),
            window_name="SoRoMoX PPO Rollout Multi-Arm",
        )
    finally:
        progress.close()


def encode_gif_from_mp4(
    *,
    mp4_path: Path,
    gif_path: Path,
    fps: int,
    width: int,
) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    filters = [f"fps={fps}"]
    if width > 0:
        filters.append(f"scale={width}:-1:flags=lanczos")
    filters.append(
        "split[s0][s1];[s0]palettegen=max_colors=128[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )

    cmd = [
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
    subprocess.run(cmd, check=True)


def existing_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--vecnormalize-path", type=Path, default=DEFAULT_VECNORMALIZE_PATH
    )
    parser.add_argument(
        "--trajectory-output", type=Path, default=DEFAULT_TRAJECTORY_OUTPUT
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the complete batched rollout directly to MP4 and GIF.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help="GIF output path. Defaults to --output with a .gif suffix.",
    )

    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=105)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--game-time", type=float, default=7.0)
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--arm-length", type=float, default=0.25)
    parser.add_argument("--arm-radius", type=float, default=0.025)
    parser.add_argument("--ball-radius", type=float, default=0.10)
    parser.add_argument("--ball-surface-vmax", type=float, default=0.015)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--stochastic", action="store_true")

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, default=0.16)
    parser.add_argument("--sphere-resolution", type=int, default=12)
    parser.add_argument("--tendon-line-width", type=float, default=1.0)
    parser.add_argument("--record-every-n", type=int, default=1)
    parser.add_argument("--camera-fov", type=float, default=75.0)
    parser.add_argument("--camera-distance-factor", type=float, default=10.0)
    parser.add_argument(
        "--camera-position-offset", type=parse_vec3, default=(0.8, -0.8, 0.5)
    )
    parser.add_argument("--camera-up", type=parse_vec3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--visible", action="store_true")

    parser.add_argument(
        "--crf", type=int, default=18, help="Lower means higher quality."
    )
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument(
        "--gif-width",
        type=int,
        default=720,
        help="GIF width in pixels. Use 0 to keep the MP4 width.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.vecnormalize_path = args.vecnormalize_path.resolve()
    args.trajectory_output = args.trajectory_output.resolve()
    args.output = args.output.resolve()
    args.gif_output = (
        args.output.with_suffix(".gif")
        if args.gif_output is None
        else args.gif_output.resolve()
    )

    if args.record_every_n <= 0:
        raise ValueError("--record-every-n must be positive.")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive.")
    if args.gif_width < 0:
        raise ValueError("--gif-width must be nonnegative.")
    if args.max_envs <= 0 or args.num_envs <= 0:
        raise ValueError("--num-envs and --max-envs must be positive.")
    if args.max_envs > args.num_envs:
        raise ValueError("--max-envs cannot be greater than --num-envs.")
    existing_paths((args.model_path, args.vecnormalize_path))
    requested_outputs = [args.trajectory_output]
    if args.render:
        requested_outputs.extend((args.output, args.gif_output))
    existing_outputs = [path for path in requested_outputs if path.exists()]
    if existing_outputs and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing_outputs)
            + "; pass --force"
        )
    if args.render and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH.")
    if args.render and OPEN3D_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Open3D rendering dependencies are not available. Install the example "
            "dependencies with `pip install soromox[examples]` and ensure `open3d` "
            "is installed."
        ) from None

    print(f"Model: {args.model_path}")
    print(f"VecNormalize: {args.vecnormalize_path}")
    print(f"Trajectory data: {args.trajectory_output}")

    raw_env, ts, q_ts, ball_ts, actions, rewards = run_policy_and_collect(args)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.trajectory_output,
        ts=ts,
        q_ts=q_ts[:, 0, :],
        ball_ts=ball_ts[:, 0, :],
        actions=actions[:, :1, :],
        rewards=rewards[:, :1],
    )
    print(f"Saved trajectory data: {args.trajectory_output}")
    if not args.render:
        return

    print(f"Output MP4: {args.output}")
    print(f"Output GIF: {args.gif_output}")
    render_rollout_to_mp4(raw_env, ts, q_ts, ball_ts, args)

    if not args.output.exists() or args.output.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not created correctly: {args.output}")
    encode_gif_from_mp4(
        mp4_path=args.output,
        gif_path=args.gif_output,
        fps=args.gif_fps,
        width=args.gif_width,
    )
    if not args.gif_output.exists() or args.gif_output.stat().st_size == 0:
        raise RuntimeError(f"GIF was not created correctly: {args.gif_output}")
    print(f"Saved MP4: {args.output}")
    print(f"Saved GIF: {args.gif_output}")


if __name__ == "__main__":
    main()
