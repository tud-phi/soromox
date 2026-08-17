"""Render Section Vd tracking diagnostics and animations from saved MAT data.

The optimization generators normally create interactive Open3D previews, but
the committed Section Vd datasets contain the trajectory fields needed for an
offline, reproducible Matplotlib rendering.  This script supports both the
legacy batched MAT layout and the newer single-run layout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.animation import FFMpegWriter, FuncAnimation

CASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = CASE_DIR / "data"
DEFAULT_OUTPUT_DIR = CASE_DIR / "outputs"


@dataclass(frozen=True)
class MethodData:
    """Normalized optimization and trajectory data for one controller."""

    name: str
    loss: np.ndarray
    t: np.ndarray
    initial: np.ndarray
    best: np.ndarray
    target: np.ndarray
    labels: tuple[str, str, str]
    units: str

    @property
    def best_batch(self) -> int:
        return int(np.unravel_index(np.argmin(self.loss), self.loss.shape)[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Section Vd trajectory diagnostics and videos."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also convert each MP4 to a compact GIF preview.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        key: np.asarray(value)
        for key, value in sio.loadmat(path, squeeze_me=False).items()
        if not key.startswith("__")
    }


def _normalize_loss_and_batches(
    loss: np.ndarray,
    initial: np.ndarray,
    best: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize legacy batched and current single-run arrays."""
    initial = np.asarray(initial, dtype=float)
    best = np.asarray(best, dtype=float)
    if initial.ndim == 2:
        initial = initial[None, ...]
        best = best[None, ...]
        loss = np.asarray(loss, dtype=float).reshape(-1, 1)
    elif initial.ndim == 3:
        loss = np.asarray(loss, dtype=float)
        if loss.ndim == 1:
            loss = loss[:, None]
        if loss.shape[1] != initial.shape[0]:
            if loss.size % initial.shape[0] != 0:
                raise ValueError(
                    "Loss history cannot be aligned with trajectory batches: "
                    f"loss={loss.shape}, trajectories={initial.shape}."
                )
            loss = loss.reshape(-1, initial.shape[0])
    else:
        raise ValueError(f"Expected a 2D or 3D trajectory, got {initial.shape}.")

    if best.shape != initial.shape:
        raise ValueError(
            f"Initial and optimized trajectories differ: {initial.shape} vs {best.shape}."
        )
    return loss, initial, best


def _normalize_time(raw_t: np.ndarray, batches: int, samples: int) -> np.ndarray:
    raw_t = np.asarray(raw_t, dtype=float)
    if raw_t.ndim == 1:
        raw_t = raw_t[None, :]
    if raw_t.shape == (1, samples):
        raw_t = np.repeat(raw_t, batches, axis=0)
    if raw_t.shape != (batches, samples):
        raise ValueError(
            f"Time array {raw_t.shape} does not match {(batches, samples)}."
        )
    return raw_t


def _normalize_target(
    raw_target: np.ndarray, samples: int, channels: int
) -> np.ndarray:
    target = np.asarray(raw_target, dtype=float).squeeze()
    if target.ndim == 1:
        if target.size == channels:
            target = np.repeat(target[None, :], samples, axis=0)
        elif channels == 1 and target.size == samples:
            target = target[:, None]
    if target.shape == (1, channels):
        target = np.repeat(target, samples, axis=0)
    if target.shape != (samples, channels):
        raise ValueError(
            f"Target array {target.shape} does not match {(samples, channels)}."
        )
    return target


def load_method(data_dir: Path, name: str) -> MethodData:
    data = _load_mat(data_dir / name / "optimization_results.mat")
    if name == "collocated":
        initial_key, best_key = "q_ts_init", "q_ts_best"
        target_key = "q_des_ts" if "q_des_ts" in data else "x_des_ts"
        channel_indices = (0, 1, 2)
        labels = (r"$\kappa_x$", r"$\kappa_y$", r"$\kappa_z$")
        units = "rad/m"
    elif name == "synergistic":
        initial_key, best_key, target_key = "x_ts_init", "x_ts_best", "x_des_ts"
        channel_indices = (0, 1, 2)
        labels = (r"$p_x$", r"$p_y$", r"$p_z$")
        units = "m"
    else:
        raise ValueError(f"Unknown method: {name}")

    required = {"history_loss", "t_ts", initial_key, best_key, target_key}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"{name} MAT file is missing {sorted(missing)}")

    loss, initial, best = _normalize_loss_and_batches(
        data["history_loss"], data[initial_key], data[best_key]
    )
    initial = initial[..., channel_indices]
    best = best[..., channel_indices]
    t = _normalize_time(data["t_ts"], initial.shape[0], initial.shape[1])
    raw_target = np.asarray(data[target_key])
    if raw_target.ndim >= 2 and raw_target.shape[-1] > len(channel_indices):
        raw_target = raw_target[..., channel_indices]
    target = _normalize_target(raw_target, initial.shape[1], len(channel_indices))

    arrays = {
        "loss": loss,
        "time": t,
        "initial": initial,
        "best": best,
        "target": target,
    }
    invalid = [key for key, value in arrays.items() if not np.all(np.isfinite(value))]
    if invalid:
        raise ValueError(f"{name} contains non-finite arrays: {invalid}")

    return MethodData(name, loss, t, initial, best, target, labels, units)


def _check_outputs(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite outputs; pass --force:\n{joined}")


def render_method(
    data: MethodData, output_dir: Path, fps: int, make_gif: bool
) -> list[Path]:
    batch = data.best_batch
    t = data.t[batch]
    initial = data.initial[batch]
    best = data.best[batch]
    target = data.target

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    colors = ("#d81b60", "#2a9d8f", "#e9b949")
    cursors = []
    markers = []
    for channel, ax in enumerate(axes):
        color = colors[channel]
        ax.plot(
            t, target[:, channel], "--", color="black", linewidth=1.4, label="Target"
        )
        ax.plot(
            t,
            initial[:, channel],
            "-.",
            color=color,
            alpha=0.65,
            linewidth=1.2,
            label="Initial",
        )
        ax.plot(t, best[:, channel], "-", color=color, linewidth=1.8, label="Optimized")
        cursor = ax.axvline(t[0], color="#555555", linewidth=1.0, alpha=0.75)
        (marker,) = ax.plot([t[0]], [best[0, channel]], "o", color=color, markersize=6)
        cursors.append(cursor)
        markers.append(marker)
        ax.set_ylabel(f"{data.labels[channel]} [{data.units}]")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncols=3)
    axes[-1].set_xlabel("Time [s]")
    title = (
        "Collocated strain tracking"
        if data.name == "collocated"
        else "Synergistic end-effector tracking"
    )
    fig.suptitle(f"{title} — best batch {batch + 1}")

    pdf_path = output_dir / f"{data.name}_tracking.pdf"
    png_path = output_dir / f"{data.name}_tracking.png"
    mp4_path = output_dir / f"{data.name}_tracking.mp4"
    gif_path = output_dir / f"{data.name}_tracking.gif"
    fig.savefig(pdf_path, dpi=250)
    fig.savefig(png_path, dpi=180)

    frame_count = max(2, int(np.ceil((t[-1] - t[0]) * fps)) + 1)
    frame_times = np.linspace(t[0], t[-1], frame_count)
    frame_indices = np.searchsorted(t, frame_times, side="left")
    frame_indices = np.clip(frame_indices, 0, len(t) - 1)

    def update(frame: int):
        index = int(frame_indices[frame])
        current_t = float(t[index])
        artists = []
        for channel in range(3):
            cursors[channel].set_xdata([current_t, current_t])
            markers[channel].set_data([current_t], [best[index, channel]])
            artists.extend((cursors[channel], markers[channel]))
        fig.suptitle(f"{title} — best batch {batch + 1} — t = {current_t:.2f} s")
        return artists

    animation = FuncAnimation(
        fig, update, frames=frame_count, interval=1000 / fps, blit=False
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="h264",
        bitrate=2200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(mp4_path, writer=writer, dpi=140)
    plt.close(fig)

    outputs = [pdf_path, png_path, mp4_path]
    if make_gif:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to create GIF previews")
        filter_graph = (
            "fps=12,scale=960:-2:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        )
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(mp4_path),
                "-vf",
                filter_graph,
                str(gif_path),
            ],
            check=True,
        )
        outputs.append(gif_path)
    return outputs


def summarize(data: MethodData) -> dict[str, object]:
    best_iteration, best_batch = np.unravel_index(np.argmin(data.loss), data.loss.shape)
    best_loss_by_iteration = np.min(data.loss, axis=1)
    initial_best = float(np.min(data.loss[0]))
    final_best = float(np.min(data.loss[-1]))
    global_best = float(data.loss[best_iteration, best_batch])
    tail_count = min(10, len(best_loss_by_iteration))
    tail_x = np.arange(tail_count, dtype=float)
    tail_slope = float(np.polyfit(tail_x, best_loss_by_iteration[-tail_count:], 1)[0])
    target = data.target
    initial_rmse = np.sqrt(np.mean((data.initial[best_batch] - target) ** 2, axis=0))
    optimized_rmse = np.sqrt(np.mean((data.best[best_batch] - target) ** 2, axis=0))
    return {
        "iterations_per_batch": int(data.loss.shape[0]),
        "batches": int(data.loss.shape[1]),
        "recorded_evaluations": int(data.loss.size),
        "initial_best_loss": initial_best,
        "final_best_loss": final_best,
        "global_best_loss": global_best,
        "global_best_iteration_1_based": int(best_iteration + 1),
        "global_best_batch_1_based": int(best_batch + 1),
        "global_best_is_final_iteration": bool(
            best_iteration == data.loss.shape[0] - 1
        ),
        "loss_reduction_from_initial_best_percent": float(
            100.0 * (initial_best - global_best) / initial_best
        ),
        "last_10_iteration_best_loss_slope": tail_slope,
        "initial_rmse": initial_rmse.tolist(),
        "optimized_rmse": optimized_rmse.tolist(),
        "rmse_units": data.units,
    }


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        load_method(args.data_dir, name) for name in ("collocated", "synergistic")
    ]

    expected: list[Path] = []
    for method in methods:
        expected.extend(
            args.output_dir / f"{method.name}_tracking{suffix}"
            for suffix in (".pdf", ".png", ".mp4")
        )
        if args.gif:
            expected.append(args.output_dir / f"{method.name}_tracking.gif")
    metrics_path = args.output_dir / "intermediate_metrics.json"
    expected.append(metrics_path)
    _check_outputs(expected, args.force)

    outputs: list[Path] = []
    for method in methods:
        outputs.extend(render_method(method, args.output_dir, args.fps, args.gif))

    metrics = {
        "source": "recoverable canonical MAT histories",
        "note": (
            "The explicitly interrupted live 285/119-iteration processes did not "
            "checkpoint their in-memory histories; these artifacts use the finite "
            "100-iteration x 6-batch datasets already present on disk."
        ),
        "methods": {method.name: summarize(method) for method in methods},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs.append(metrics_path)

    print("Generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
