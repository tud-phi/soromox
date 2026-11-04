#!/usr/bin/env python3
"""SMPC push-task composite figure.

2x2 grid, lettered A-D: A) RTX renders of the sample cloud across iCEM iterations,
B) the MPPI-vs-iCEM planning-time/performance trade-off, C) RTX renders of the
Franka-mounted tentacle pushing along the trajectory, D) the top-down square-push
trajectories, iCEM vs MPPI K=800 over three seeds each, overlaid.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
PAPER_RESULTS_DIR = CASE_DIR.parent
for path in (CODE_DIR, PAPER_RESULTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE
from four_tendon_soft_arm import FourTendonSoftArm
from smpc import SMPCParams, push_cost

RESULTS_DIR = CASE_DIR / "data"
FRAMES_DIR = CASE_DIR / "outputs" / "frames"
OUTPUT_DIR = CASE_DIR / "outputs"
OUTPUT_FILENAME = "smpc_summary"
TRADEOFF_RESULTS_DIRS = [RESULTS_DIR / "cmp_seeds" / f"seed{i}" for i in range(3)]

# Copies of run_experiment.py's values: importing it would pull in newton_cosserat just to plot.
PUSH_PATH = [0.05, -0.1, 0.15, -0.1, 0.15, 0.0, 0.05, 0.0]
OBJECT_FOOTPRINT = 0.05
TABLE_TOP_Z = 0.0
CONTACT_SEGMENT_INDEX = 2

TRAJECTORY_RUNS = [
    (f"square_icem_seed{i}.npz", f"iCEM, seed {i}", c)
    for i, c in enumerate([COLORS["post_opt_1"], COLORS["post_opt_2"], COLORS["post_opt_3"]])
] + [
    (f"square_mppi_k800_white_seed{i}.npz", f"MPPI K=800, seed {i}", c)
    for i, c in enumerate([COLORS["pre_opt_1"], COLORS["pre_opt_2"], "#023E5C"])
]

ITERATION_IMAGES = [FRAMES_DIR / f"square_icem_seed0_t0_iter{i}.png" for i in range(3)]
ITERATION_LABELS = ["Iteration 0", "Iteration 1", "Iteration 2"]
SCENE_ZOOM = 0.9
PUSH_SCENE_ZOOM = 0.8
SATURATION_THRESHOLD = 0.15

# Frames rendered by rendering.py --output outputs/square_icem_seed0.mp4.
TRAJECTORY_FRAMES_DIR = OUTPUT_DIR / "square_icem_seed0_frames" / "frames"
TRAJECTORY_IMAGES = [TRAJECTORY_FRAMES_DIR / f"frame_{i:04d}.png" for i in (75, 189, 318, 453)]
CONTROL_DT = 0.05
RENDER_FRAMES_PER_STEP = 10

WIDTH_RATIOS = [1.0, 1.0]
HEIGHT_RATIOS = [0.78, 0.72]
PANEL_WSPACE = 0.26
PANEL_HSPACE = 0.36


@dataclasses.dataclass(frozen=True)
class RunSpec:
    filename: str
    family: str
    label: str


RUNS = [
    RunSpec("cmp_mppi_k200.npz", "mppi", "K=200"),
    RunSpec("cmp_mppi_k400.npz", "mppi", "K=400"),
    RunSpec("cmp_mppi_k800.npz", "mppi", "K=800"),
    RunSpec("cmp_mppi_k1600.npz", "mppi", "K=1600"),
    RunSpec("cmp_icem_it3_k50.npz", "icem_fixed_iters", "K=50"),
    RunSpec("cmp_icem_it3_k100.npz", "icem_fixed_iters", "K=100"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_iters", "K=200"),
    RunSpec("cmp_icem_it3_k400.npz", "icem_fixed_iters", "K=400"),
    RunSpec("cmp_icem_k200_it1.npz", "icem_fixed_batch", "it=1"),
    RunSpec("cmp_icem_k200_it2.npz", "icem_fixed_batch", "it=2"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_batch", "it=3"),
    RunSpec("cmp_icem_k200_it4.npz", "icem_fixed_batch", "it=4"),
]

FAMILY_STYLE = {
    "mppi": {
        "marker": "^", "color": COLORS["post_opt_1"], "name": "MPPI",
        "offset": (-8, 0), "ha": "right", "va": "center",
    },
    "icem_fixed_iters": {
        "marker": "x", "color": COLORS["pre_opt_1"], "name": "iCEM (fixed iters=3)",
        "offset": (0, 9), "ha": "center", "va": "bottom",
    },
    "icem_fixed_batch": {
        "marker": "o", "color": COLORS["post_opt_2"], "name": "iCEM (fixed K=200)",
        "offset": (0, -9), "ha": "center", "va": "top",
    },
}

LABEL_OVERRIDES = {
    ("cmp_mppi_k400.npz", "mppi"): {"offset": (8, 0), "ha": "left", "va": "center"},
    ("cmp_icem_k200_it3.npz", "icem_fixed_iters"): {"offset": (8, 7), "ha": "left", "va": "bottom"},
    ("cmp_mppi_k1600.npz", "mppi"): {"offset": (0, 9), "ha": "center", "va": "bottom"},
    ("cmp_icem_it3_k50.npz", "icem_fixed_iters"): {"offset": (-8, 0), "ha": "right", "va": "center"},
    ("cmp_icem_it3_k100.npz", "icem_fixed_iters"): {"offset": (0, -9), "ha": "center", "va": "top"},
}


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "savefig.pad_inches": 0.1,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def build_params(robot) -> SMPCParams:
    """SMPCParams for re-scoring the cmp_ runs; score_run fills in each run's goal, contact_z and yaw."""
    return SMPCParams(
        robot=robot,
        control_std=jnp.ones(robot.num_actuators + 3),
        K=1,
        contact_segment_index=CONTACT_SEGMENT_INDEX,
        object_footprint=OBJECT_FOOTPRINT,
        table_z=TABLE_TOP_Z,
    )


score_push_cost = eqx.filter_jit(push_cost)


def score_run(path: Path, params: SMPCParams) -> tuple[float, float]:
    """Return (mean planning time per step [s], -push_cost of the executed trajectory).

    Planning time sums diagnostics_iter_time over iterations (rollouts included) and averages over
    steps 1.., skipping step 0's JIT warm-up. The cost uses the run's goal, starting object height
    and starting yaw, as run_experiment.py sets them.
    """
    with np.load(path, allow_pickle=True) as data:
        iter_time = data["diagnostics_iter_time"].astype(float)
        X_final = data["X_final"]
        U_best = data["U_best"]
        goal_xy = data["goal_xy"][0]
        initial_object_z = data["object_pose"][0, 2]

    yaw_index = 2 * params.robot.num_internal_dofs + 4
    params = dataclasses.replace(
        params,
        push_goal_xy=jnp.asarray(goal_xy),
        contact_z=float(initial_object_z),
        push_goal_yaw=jnp.asarray(X_final[0, yaw_index]),
    )
    cost = float(score_push_cost(params, jnp.asarray(X_final), jnp.asarray(U_best)))
    return float(iter_time[1:].sum(axis=1).mean()), -cost


def load_seed_mean_points(results_dirs: list[Path], params: SMPCParams) -> list[tuple[RunSpec, float, float]]:
    """Score every RUNS configuration in each of results_dirs (one seed per directory)
    and return (spec, mean_time, mean_neg_cost) per configuration.
    """
    scores: dict[Path, tuple[float, float]] = {}  # cmp_icem_k200_it3 appears in two families
    points = []
    for spec in RUNS:
        paths = [results_dir / spec.filename for results_dir in results_dirs]
        for path in paths:
            if path not in scores:
                scores[path] = score_run(path, params)
        times, costs = zip(*(scores[path] for path in paths), strict=True)
        points.append((spec, float(np.mean(times)), float(np.mean(costs))))
    return points


def plot_planner_tradeoff(points: list[tuple[RunSpec, float, float]], ax: plt.Axes) -> None:
    """Scatter (mean planning time, mean -push_cost) per configuration, one marker style per family."""
    seen_families = []
    for spec, x, y in points:
        style = FAMILY_STYLE[spec.family]
        scatter_kwargs = {"edgecolor": "black", "linewidth": 0.5} if style["marker"] != "x" else {}
        ax.scatter(
            x,
            y,
            marker=style["marker"],
            color=style["color"],
            s=60,
            zorder=3,
            label=style["name"] if spec.family not in seen_families else None,
            **scatter_kwargs,
        )
        if spec.family not in seen_families:
            seen_families.append(spec.family)
        place = LABEL_OVERRIDES.get((spec.filename, spec.family), style)
        ax.annotate(
            spec.label,
            (x, y),
            textcoords="offset points",
            xytext=place["offset"],
            ha=place["ha"],
            va=place["va"],
            fontsize=5.5,
        )

    ax.set_xlabel(r"Mean planning wall-clock time per MPC step $\mathrm{[s]}$")
    ax.set_ylabel(r"Mean $-$push cost per MPC step")
    xs = [x for _, x, _ in points]
    ys = [y for _, _, y in points]
    x_span = (max(xs) - min(xs)) or 1.0
    y_span = (max(ys) - min(ys)) or 1.0
    ax.set_xlim(0.0, max(xs) + 0.08 * x_span)
    ax.set_ylim(min(ys) - 0.14 * y_span, max(ys) + 0.12 * y_span)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(loc="best", frameon=True)


def load_object_xy(input_path: Path) -> np.ndarray:
    """Load the object's (x, y) position trace from object_pose, shape (n_steps, 2)."""
    with np.load(input_path) as data:
        return data["object_pose"][:, :2]


def draw_goal_path(ax: plt.Axes, goal_xy: np.ndarray) -> None:
    """Draw the dashed goal-loop outline, goal-corner markers, and 1..N labels."""
    goal_loop = np.vstack([goal_xy, goal_xy[:1]])
    ax.plot(goal_loop[:, 0], goal_loop[:, 1], linestyle="--", color="0.55", linewidth=1.0, zorder=1)
    ax.scatter(
        goal_xy[:, 0],
        goal_xy[:, 1],
        marker="s",
        s=40,
        facecolor=COLORS["target"],
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
        label="Planner goals",
    )
    for i, (gx, gy) in enumerate(goal_xy, start=1):
        ax.annotate(str(i), (gx, gy), textcoords="offset points", xytext=(6, 6), fontsize=7, color="0.25")


def draw_trajectory(ax: plt.Axes, object_xy: np.ndarray, color: str, label: str | None = None) -> None:
    """Draw one object path with a hollow start marker and a solid 'X' end marker."""
    ax.plot(object_xy[:, 0], object_xy[:, 1], color=color, linewidth=1.5, zorder=2, label=label)
    ax.scatter(
        object_xy[0, 0], object_xy[0, 1], marker="o", s=30,
        facecolor="white", edgecolor=color, linewidth=1.2, zorder=4,
    )
    ax.scatter(
        object_xy[-1, 0], object_xy[-1, 1], marker="X", s=35,
        facecolor=color, edgecolor="black", linewidth=0.5, zorder=4,
    )


def plot_push_trajectory_xy(trajectories: list[tuple[str, np.ndarray, str]], goal_xy: np.ndarray, ax: plt.Axes) -> None:
    """Top-down (x, y) plot of the goal corners and every object path, overlaid in one axes."""
    ax.set_axisbelow(True)
    draw_goal_path(ax, goal_xy)

    all_y = [goal_xy[:, 1]]
    for label, object_xy, color in trajectories:
        draw_trajectory(ax, object_xy, color, label=label)
        all_y.append(object_xy[:, 1])

    center_y = goal_xy[:, 1].mean()
    half_extent_y = np.abs(np.concatenate(all_y) - center_y).max() * 1.15
    ax.set_ylim(center_y - half_extent_y, center_y + half_extent_y)

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.65)

    ax.set_xlabel(r"$x$ $\mathrm{[m]}$")
    ax.set_ylabel(r"$y$ $\mathrm{[m]}$")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(loc="center right", fontsize=5.5, handlelength=1.2, borderaxespad=0.3, frameon=True)


def scene_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Pixel bounding box (x0, x1, y0, y1) of the saturated scene content in an RTX render.

    The rod, object, goal patch and sample cloud are the only saturated colours on a grey
    table, sky and robot arm.
    """
    rgb = image[..., :3]
    ys, xs = np.nonzero(rgb.max(axis=-1) - rgb.min(axis=-1) > SATURATION_THRESHOLD)
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def crop_to_scene(image: np.ndarray, target_aspect: float, zoom: float) -> np.ndarray:
    """Crop an (H, W, C) render to `target_aspect`, keeping `zoom` x its height, anchored on the scene content."""
    height, width = image.shape[:2]
    crop_height = min(height, max(1, round(zoom * height)))
    crop_width = min(width, max(1, round(crop_height * target_aspect)))
    crop_height = min(height, max(1, round(crop_width / target_aspect)))

    bbox = scene_bbox(image)
    if bbox is None:
        x_center, y_bottom = 0.5 * width, 0.5 * (height + crop_height)
    else:
        x0, x1, _, y1 = bbox
        x_center = 0.5 * (x0 + x1)
        y_bottom = y1 + 0.04 * crop_height

    x_start = min(max(int(round(x_center - 0.5 * crop_width)), 0), width - crop_width)
    y_start = min(max(int(round(y_bottom - crop_height)), 0), height - crop_height)
    return image[y_start : y_start + crop_height, x_start : x_start + crop_width]


def draw_image(fig: plt.Figure, ax: plt.Axes, image_path: Path, zoom: float) -> None:
    """Draw one borderless image, cropped to the axes' aspect ratio so it fills the cell."""
    fig_width_in, fig_height_in = fig.get_size_inches()
    bbox = ax.get_position()
    axes_aspect = (bbox.width * fig_width_in) / (bbox.height * fig_height_in)
    ax.imshow(crop_to_scene(plt.imread(image_path), axes_aspect, zoom), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_image_grid(
    fig: plt.Figure,
    cell,
    image_paths: list[Path],
    titles: list[str] | None = None,
    zoom: float = 1.0,
    ncols: int | None = None,
) -> None:
    """Draw a borderless grid of images filling a nested subgridspec cell."""
    ncols = ncols or len(image_paths)
    nrows = -(-len(image_paths) // ncols)
    sub_gs = cell.subgridspec(nrows, ncols, wspace=0.05, hspace=0.3)
    for i, image_path in enumerate(image_paths):
        ax = fig.add_subplot(sub_gs[i // ncols, i % ncols])
        draw_image(fig, ax, image_path, zoom)
        if titles is not None:
            ax.set_xlabel(titles[i], fontsize=7)


def frame_time_labels(image_paths: list[Path]) -> list[str]:
    """Elapsed-push-time labels for frame_XXXX.png renders, from the frame index."""
    labels = []
    for image_path in image_paths:
        index = int(re.search(r"frame_(\d+)", image_path.stem).group(1))
        labels.append(f"$t={index / RENDER_FRAMES_PER_STEP * CONTROL_DT:.2f}$ s")
    return labels


def add_panel_label(fig: plt.Figure, cell, label: str, ax: plt.Axes | None = None) -> None:
    """Draw a bold panel letter at the top-left corner of a gridspec cell.

    Pass `ax` when its fixed aspect shrinks it inside the cell, to put the letter at its drawn top.
    """
    bbox = cell.get_position(fig)
    top = bbox.y1
    if ax is not None:
        ax.apply_aspect()
        top = ax.get_position().y1
    fig.text(bbox.x0 - 0.015, top + 0.002, label, fontsize=11, fontweight="bold", va="bottom", ha="left")


def plot_smpc_summary(
    trajectories: list,
    goal_xy: np.ndarray,
    tradeoff_points: list,
    iteration_images: list[Path],
    trajectory_images: list[Path],
    figsize_cm: tuple[float, float] = (18.0, 15.0),
) -> plt.Figure:
    """Assemble the 2x2 lettered composite figure."""
    fig = plt.figure(figsize=tuple(v / 2.54 for v in figsize_cm))
    grid = fig.add_gridspec(
        2, 2, width_ratios=WIDTH_RATIOS, height_ratios=HEIGHT_RATIOS, wspace=PANEL_WSPACE, hspace=PANEL_HSPACE
    )

    add_image_grid(fig, grid[0, 0], iteration_images, ITERATION_LABELS, zoom=SCENE_ZOOM)
    add_panel_label(fig, grid[0, 0], "A")

    plot_planner_tradeoff(tradeoff_points, fig.add_subplot(grid[0, 1]))
    add_panel_label(fig, grid[0, 1], "B")

    add_image_grid(
        fig, grid[1, 0], trajectory_images, frame_time_labels(trajectory_images),
        zoom=PUSH_SCENE_ZOOM, ncols=2,
    )
    add_panel_label(fig, grid[1, 0], "C")

    trajectory_ax = fig.add_subplot(grid[1, 1])
    plot_push_trajectory_xy(trajectories, goal_xy, trajectory_ax)
    add_panel_label(fig, grid[1, 1], "D", ax=trajectory_ax)

    return fig


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.pdf", dpi=300, pad_inches=0.05)
    plt.close(fig)


def load_trajectory_runs(results_dir: Path) -> list[tuple[str, np.ndarray, str]]:
    """Load TRAJECTORY_RUNS' object (x, y) paths as (label, xy, color), failing on a missing file."""
    return [(label, load_object_xy(results_dir / filename), color) for filename, label, color in TRAJECTORY_RUNS]


def main() -> int:
    configure_matplotlib()

    trajectories = load_trajectory_runs(RESULTS_DIR)
    goal_xy = np.reshape(PUSH_PATH, (-1, 2))
    robot = FourTendonSoftArm().model
    tradeoff_points = load_seed_mean_points(TRADEOFF_RESULTS_DIRS, build_params(robot))

    fig = plot_smpc_summary(trajectories, goal_xy, tradeoff_points, ITERATION_IMAGES, TRAJECTORY_IMAGES)
    save_figure(fig, OUTPUT_DIR, OUTPUT_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
