#!/usr/bin/env python3
"""Standalone visualization for HOCLF/HOCBF controller metrics from raw q_ts data."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Enable 64-bit precision to prevent truncation warnings during offline kinematics
jax.config.update("jax_enable_x64", True)

from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)

# --- Directory Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

# --- Configuration Constants ---
COLORS = {
    "pre_opt_1": "#006BA6",  # Blue (Goal Distance)
    "post_opt_1": "#D81159",  # Red (Force)
}


def configure_matplotlib() -> None:
    """Configure the standardized CMU Serif publication theme."""
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "savefig.pad_inches": 0.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
            "mathtext.fontset": "custom",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.color": "#B0B0B0",
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Serif", "DejaVu Serif"],
            "axes.formatter.use_mathtext": True,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "figure.dpi": 200,
            "lines.linewidth": 2.0,
        }
    )


def cm2inch(*tupl: float) -> tuple[float, ...]:
    """Convert centimeters to inches for Matplotlib figsize."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


# --- Kinematics Recomputation ---
def build_dummy_robot() -> TendonActuatedPCS:
    """Reconstruct basic kinematic properties to evaluate poses from strains."""
    num_segments = 2
    segment_length = jnp.array([0.15, 0.15])
    backbone_radius = jnp.array([0.036, 0.036])

    # Minimal routing required to initialize the object (irrelevant for pure forward kinematics)
    tendon_routing_params = {
        "ry": jnp.zeros(6),
        "rz": jnp.zeros(6),
        "my": jnp.zeros(6),
        "mz": jnp.zeros(6),
        "idx_seg_att": jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
    }

    body_params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_length,
        radius=backbone_radius,
        density=jnp.ones(num_segments),
        gravity=jnp.zeros(3),
        young_modulus=jnp.ones(num_segments),
        shear_modulus=jnp.ones(num_segments),
        damping_matrix=jnp.eye(12),
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    active_tendon_routing = LinearTendonRoutingParams(
        y_intercept=tendon_routing_params["ry"],
        z_intercept=tendon_routing_params["rz"],
        y_slope=tendon_routing_params["my"],
        z_slope=tendon_routing_params["mz"],
        attachment_segment_index=tendon_routing_params["idx_seg_att"],
    )

    return TendonActuatedPCS(
        params=TendonActuatedPCSParams(
            body=body_params, active_tendon_routing=active_tendon_routing
        )
    )


def recompute_metrics(
    q_ts: np.ndarray,
    obs_centers: np.ndarray,
    obs_radii: np.ndarray,
    target_center: np.ndarray,
):
    """Batched calculation of goal distance and collision force using JAX."""
    robot = build_dummy_robot()

    # 10 discretization points per segment matches the original s_ps spatial distribution
    s_ps = jnp.linspace(0.0, jnp.sum(robot.L), 20)
    robot_radius = 0.036
    total_length = jnp.sum(robot.L)

    # Convert Numpy arrays to JAX arrays for compilation
    obs_c = jnp.asarray(obs_centers)
    obs_r = jnp.asarray(obs_radii)
    tgt_c = jnp.asarray(target_center)

    @jax.jit
    def compute_step(q):
        # 1. Goal Distance
        g_ee = robot.forward_kinematics(q, total_length)
        p_ee = g_ee[:3, 3]
        distance = jnp.linalg.norm(p_ee - tgt_c)

        # 2. Pairwise Normal Force
        g_body = robot.forward_kinematics_batched(q, s_ps)
        p_body = g_body[:, :3, 3]  # Shape: (20, 3)

        # Broadcast distances: p_body (20, 1, 3) - obs_centers (1, 3, 3)
        diff = p_body[:, None, :] - obs_c[None, :, :]
        dist_to_obs = jnp.linalg.norm(diff, axis=-1)  # Shape: (20, 3)

        clearance = dist_to_obs - (obs_r[None, :] + robot_radius)
        min_clearance = jnp.min(clearance)

        # 1000 N/m penalty force model (computes > 0 only when intersecting)
        force = jnp.maximum(0.0, -1000.0 * min_clearance)

        return distance, force

    # Vectorize computation across the 8000 frames
    dists, forces = jax.vmap(compute_step)(jnp.asarray(q_ts))
    return np.asarray(dists), np.asarray(forces)


def plot_metrics(npz_paths: list[Path], output_path: Path, show: bool) -> None:
    if not npz_paths:
        print("[!] No .npz files provided.")
        return

    fig, ax1 = plt.subplots(figsize=cm2inch(18.0, 10.0), constrained_layout=True)
    ax2 = ax1.twinx()

    force_limit = 5.0
    global_force_max = force_limit
    ts_ref = None

    force_lines = []
    goal_lines = []

    for path in npz_paths:
        try:
            print(f"Loading and processing trajectory: {path.name}")
            data = np.load(path)
            ts = data["ts"]
            q_ts = data["q_ts"]
            obs_centers = data["obs_centers"]
            obs_radii = data["obs_radii"]
            target_center = data["target_center"]
        except KeyError as e:
            print(f"[!] File {path.name} is missing key {e}. Skipping.")
            continue

        # Recompute physical metrics dynamically
        distance, force = recompute_metrics(q_ts, obs_centers, obs_radii, target_center)

        if ts_ref is None:
            ts_ref = ts

        global_force_max = max(global_force_max, force.max())

        # Determine styling based on filename
        is_cbf = "without_cbf" not in path.name.lower()
        linestyle = "-" if is_cbf else "--"
        label_suffix = "+HOCBF" if is_cbf else ""

        (line_f,) = ax1.plot(
            ts,
            force,
            color=COLORS["post_opt_1"],
            linestyle=linestyle,
            label=f"Force for HOCLF{label_suffix} controller",
        )
        force_lines.append(line_f)

        (line_d,) = ax2.plot(
            ts,
            distance,
            color=COLORS["pre_opt_1"],
            linestyle=linestyle,
            label=f"Goal distance for HOCLF{label_suffix} controller",
        )
        goal_lines.append(line_d)

    if not force_lines:
        plt.close(fig)
        return

    # Render Force Limit Boundaries
    force_axis_max = 1.08 * global_force_max
    ax1.axhline(
        force_limit, color=COLORS["post_opt_1"], linestyle=":", linewidth=1.5, alpha=0.8
    )

    if ts_ref is not None:
        ax1.fill_between(
            ts_ref,
            force_limit,
            force_axis_max,
            color=COLORS["post_opt_1"],
            alpha=0.12,
            linewidth=0.0,
        )

    # Apply Standardized Formatting
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(
        "Max Pairwise Normal Force [N]", color=COLORS["post_opt_1"], fontweight="bold"
    )
    ax1.tick_params(axis="y", labelcolor=COLORS["post_opt_1"])
    ax1.set_ylim(0.0, force_axis_max)
    ax1.set_xlim(ts_ref.min(), ts_ref.max())

    ax2.set_ylabel("Goal Distance [m]", color=COLORS["pre_opt_1"], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLORS["pre_opt_1"])
    ax2.set_ylim(bottom=0.0)

    # Enforce Spine Colors
    ax1.spines["left"].set_color(COLORS["post_opt_1"])
    ax1.spines["left"].set_linewidth(1.2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["pre_opt_1"])
    ax2.spines["right"].set_linewidth(1.2)
    ax1.spines["right"].set_visible(False)

    # Construct Legend
    all_lines = force_lines + goal_lines
    if len(npz_paths) > 1:
        # Reorder for visual clarity if comparing two trajectories
        all_lines = [force_lines[0], force_lines[1], goal_lines[0], goal_lines[1]]

    ax1.legend(
        all_lines,
        [line.get_label() for line in all_lines],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved figure to {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HOCLF/HOCBF controller metrics from raw q_ts .npz files."
    )
    parser.add_argument(
        "--npz",
        nargs="+",
        type=Path,
        # required=True,
        default=[Path(DATA_DIR / "clf_cbf_rollout_with_cbf.npz")],
        help="Path(s) to the .npz rollout files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "force_goal_distance_plot.pdf",
        help="Path to save the PDF plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_matplotlib()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plot_metrics(args.npz, args.output, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
