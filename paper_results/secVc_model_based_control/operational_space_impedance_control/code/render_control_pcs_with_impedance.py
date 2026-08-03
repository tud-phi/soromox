"""Render the canonical operational-space rollout to video and paper stills."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from operational_space_impedance_common import (
    DEFAULT_TRAJECTORY_OUTPUT,
    OUTPUTS_DIR,
    build_problem,
    load_run_npz,
    print_problem_summary,
)
from viser_operational_space_renderer import (
    DEFAULT_SNAPSHOT_TIMES,
    crop_snapshots_to_common_square,
    render_operational_space_tracking,
    save_snapshot_strip,
    snapshot_filename,
    snapshot_indices_for_times,
)

jax.config.update("jax_enable_x64", True)

DEFAULT_VIDEO_OUTPUT = OUTPUTS_DIR / "control_pcs_with_impedance_trajectory-render.mp4"
DEFAULT_SNAPSHOT_DIR = OUTPUTS_DIR / "operational_space_impedance_snapshots"
DEFAULT_STRIP_OUTPUT = OUTPUTS_DIR / "operational_space_impedance_snapshots.pdf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help="Saved operational-space trajectory NPZ.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=DEFAULT_VIDEO_OUTPUT,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Directory for the four cropped lossless PNG snapshots.",
    )
    parser.add_argument(
        "--strip-output",
        type=Path,
        default=DEFAULT_STRIP_OUTPUT,
        help="Output path for the four-panel PDF strip.",
    )
    parser.add_argument(
        "--snapshot-times",
        nargs=4,
        type=float,
        metavar=("T1", "T2", "T3", "T4"),
        default=DEFAULT_SNAPSHOT_TIMES,
        help="Exactly four distinct snapshot times in seconds.",
    )
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open the Viser browser tab automatically.",
    )
    parser.add_argument(
        "--record-every-n",
        type=int,
        default=1,
        help="Record every Nth simulation frame in the MP4.",
    )
    parser.add_argument(
        "--record-client-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a connected browser client.",
    )
    parser.add_argument(
        "--record-frame-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each browser-rendered frame.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing video, PNG, and PDF output paths.",
    )
    return parser.parse_args(argv)


def snapshot_output_paths(
    snapshot_dir: Path,
    snapshot_times: tuple[float, ...] | list[float],
) -> tuple[Path, Path, Path, Path]:
    """Return four stable timestamped PNG paths."""
    if len(snapshot_times) != 4:
        raise ValueError("Exactly four snapshot times are required.")
    paths = tuple(
        Path(snapshot_dir) / snapshot_filename(time_seconds)
        for time_seconds in snapshot_times
    )
    if len(set(paths)) != 4:
        raise ValueError("Snapshot times must produce four unique output filenames.")
    return paths  # type: ignore[return-value]


def preflight_outputs(paths: list[Path] | tuple[Path, ...], *, force: bool) -> None:
    """Validate overwrite policy and create only the required parent directories."""
    paths = [Path(path) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Output paths must be unique.")
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Rendering outputs already exist. Pass --force to replace them:\n"
            f"{formatted}"
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run = load_run_npz(args.trajectory)
    problem = build_problem(run.trajectory_config)
    print_problem_summary(problem)

    snapshot_times = tuple(float(value) for value in args.snapshot_times)
    frame_indices = snapshot_indices_for_times(
        np.asarray(run.t_traj, dtype=np.float64),
        snapshot_times,
    )
    saved_times = np.asarray(run.t_traj, dtype=np.float64)
    actual_snapshot_times = tuple(float(saved_times[index]) for index in frame_indices)
    snapshot_paths = snapshot_output_paths(args.snapshot_dir, snapshot_times)
    all_outputs = (args.video_output, *snapshot_paths, args.strip_output)
    preflight_outputs(all_outputs, force=args.force)
    snapshot_path_map = dict(zip(frame_indices, snapshot_paths, strict=True))

    print(f"Video output: {args.video_output}")
    print("Snapshot frames:")
    for time_seconds, actual_time, frame_idx, path in zip(
        snapshot_times,
        actual_snapshot_times,
        frame_indices,
        snapshot_paths,
        strict=True,
    ):
        print(
            f"  - requested {time_seconds:.2f} s -> frame {frame_idx} "
            f"({actual_time:.2f} s): {path}"
        )

    render_operational_space_tracking(
        robot=problem.robot,
        osd=problem.osd,
        trajectory_config=run.trajectory_config,
        t_traj=run.t_traj,
        q_traj=run.q_traj,
        x_traj_full=run.x_traj_full,
        x_des_traj_full=run.x_des_traj_full,
        video_output=args.video_output,
        snapshot_paths=snapshot_path_map,
        port=args.viser_port,
        open_browser=not args.no_open_browser,
        record_every_n=args.record_every_n,
        record_client_timeout=args.record_client_timeout,
        record_frame_timeout=args.record_frame_timeout,
    )
    crop_box = crop_snapshots_to_common_square(snapshot_paths)
    strip_output = save_snapshot_strip(
        snapshot_paths,
        args.strip_output,
        snapshot_times=actual_snapshot_times,
    )
    print(f"Common snapshot crop: {crop_box}")
    print(f"Snapshot strip: {strip_output}")


if __name__ == "__main__":
    main()
