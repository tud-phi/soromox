"""Render Section Vd optimized robot trajectories with Viser."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from secvd_case import (  # noqa: E402
    build_sec_vd_robot,
    end_effector_pose_trajectory,
)
from secvd_results import METHODS, load_results  # noqa: E402

from soromox.rendering import BackboneColorConfig, RendererColorConfig, ViserRenderer

DEFAULT_DATA_DIR = CASE_DIR / "data"
DEFAULT_OUTPUT_DIR = CASE_DIR / "outputs"
ROBOT_COLOR = np.array([0xF1, 0x55, 0x2E]) / 255.0
TRAIL_COLOR = np.array([0x63, 0x9F, 0xC5]) / 255.0
TARGET_COLOR = np.array([0x0E, 0xAD, 0x69]) / 255.0


def resample_trajectory(
    t: np.ndarray, q: np.ndarray, pose: np.ndarray, fps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample a dense rollout onto an exact video frame grid."""
    if fps < 1:
        raise ValueError("--fps must be at least 1")
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    pose = np.asarray(pose, dtype=float)
    if t.ndim != 1 or q.shape[0] != len(t) or pose.shape != (len(t), 6):
        raise ValueError("Time, configuration, and pose trajectories are misaligned")
    frame_count = max(2, int(np.ceil((t[-1] - t[0]) * fps)) + 1)
    frame_t = np.linspace(t[0], t[-1], frame_count)
    frame_q = np.column_stack(
        [np.interp(frame_t, t, q[:, idx]) for idx in range(q.shape[1])]
    )
    frame_pose = np.column_stack(
        [np.interp(frame_t, t, pose[:, idx]) for idx in range(pose.shape[1])]
    )
    return frame_t, frame_q, frame_pose


def validate_pose_consistency(
    robot, total_length: float, data: dict[str, np.ndarray]
) -> None:
    """Ensure stored poses match FK from the authoritative configurations."""
    stored = np.asarray(data["x_ts_best"])[0]
    reconstructed = np.asarray(
        end_effector_pose_trajectory(robot, data["q_ts_best"][0], total_length)
    )
    if not np.allclose(reconstructed, stored, rtol=1e-6, atol=1e-8):
        maximum = float(np.max(np.abs(reconstructed - stored)))
        raise ValueError(
            f"Stored pose does not agree with q_ts_best FK (max error {maximum:.3e})"
        )


def _color_config(num_points: int) -> RendererColorConfig:
    rgba = np.r_[ROBOT_COLOR, 1.0]
    return RendererColorConfig(
        backbone=BackboneColorConfig(point_colors=np.tile(rgba, (num_points, 1))),
        base_plate_color=(0.2, 0.2, 0.2),
    )


def _convert_gif(mp4_path: Path, gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --gif")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=12,scale=960:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif_path),
        ],
        check=True,
    )


def render_method(
    *,
    name: str,
    data_dir: Path,
    output_dir: Path,
    fps: int,
    make_gif: bool,
    force: bool,
    port: int,
    open_browser: bool,
    record_client_timeout: float,
    record_frame_timeout: float,
    renderer_factory: Callable = ViserRenderer,
) -> list[Path]:
    mp4_path = output_dir / f"{name}_tracking.mp4"
    gif_path = output_dir / f"{name}_tracking.gif"
    requested = [mp4_path] + ([gif_path] if make_gif else [])
    existing = [path for path in requested if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")

    data = load_results(data_dir / name, expected_method=name)
    robot, _routing, total_length = build_sec_vd_robot()
    validate_pose_consistency(robot, total_length, data)
    t, q, pose = resample_trajectory(
        data["t_ts"][0], data["q_ts_best"][0], data["x_ts_best"][0], fps
    )
    target = np.asarray(data["x_des_ts"])[0, -1, 3:6]
    trail_stride = max(1, len(pose) // 50)
    trail = pose[::trail_stride, 3:6]
    static_positions = np.vstack([trail, target])
    static_radii = np.r_[np.full(len(trail), 0.0015), 0.004]
    static_colors = np.vstack([np.tile(TRAIL_COLOR, (len(trail), 1)), TARGET_COLOR])

    renderer = renderer_factory(
        robot,
        width=1920,
        height=1080,
        num_points=80,
        port=port,
        open_browser=open_browser,
        color_config=_color_config(80),
        backbone_style="swept",
    )
    renderer.render_sequence(
        ts=t,
        q_ts=q,
        record_path=str(mp4_path),
        stop_when_recording_done=True,
        record_client_timeout=record_client_timeout,
        record_frame_timeout=record_frame_timeout,
        render_actuators=False,
        static_spheres_positions=static_positions,
        static_spheres_radii=static_radii,
        static_spheres_colors=static_colors,
        dynamic_spheres_positions=pose[None, :, 3:6],
        dynamic_spheres_radii=np.array([0.0025]),
        dynamic_spheres_colors=np.array([TRAIL_COLOR]),
        blocking=True,
        robot_name=f"Section Vd {name}",
    )
    outputs = [mp4_path]
    if make_gif:
        _convert_gif(mp4_path, gif_path)
        outputs.append(gif_path)
    return outputs


def render_saved_results(
    *,
    data_dir: Path,
    output_dir: Path,
    method: str = "both",
    fps: int = 24,
    make_gif: bool = False,
    force: bool = False,
    port: int = 8080,
    open_browser: bool = True,
    record_client_timeout: float = 30.0,
    record_frame_timeout: float = 10.0,
    renderer_factory: Callable = ViserRenderer,
) -> list[Path]:
    if renderer_factory is None:
        raise RuntimeError("Viser is unavailable; install the rendering dependencies")
    selected = METHODS if method == "both" else (method,)
    if any(name not in METHODS for name in selected):
        raise ValueError(f"Unknown method {method!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [output_dir / f"{name}_tracking.mp4" for name in selected]
    if make_gif:
        requested.extend(output_dir / f"{name}_tracking.gif" for name in selected)
    existing = [path for path in requested if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite {existing}; pass --force")
    outputs: list[Path] = []
    for offset, name in enumerate(selected):
        outputs.extend(
            render_method(
                name=name,
                data_dir=data_dir,
                output_dir=output_dir,
                fps=fps,
                make_gif=make_gif,
                force=force,
                port=port + offset,
                open_browser=open_browser,
                record_client_timeout=record_client_timeout,
                record_frame_timeout=record_frame_timeout,
                renderer_factory=renderer_factory,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=(*METHODS, "both"), default="both")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--open-browser", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--record-client-timeout", type=float, default=30.0)
    parser.add_argument("--record-frame-timeout", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_saved_results(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        fps=args.fps,
        make_gif=args.gif,
        force=args.force,
        port=args.port,
        open_browser=args.open_browser,
        record_client_timeout=args.record_client_timeout,
        record_frame_timeout=args.record_frame_timeout,
    )
    print("Generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
