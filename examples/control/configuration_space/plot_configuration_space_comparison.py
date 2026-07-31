"""Plot saved configuration-space control comparison rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from configuration_space_comparison_simulation import (
    ComparisonRun,
    PlotGroup,
    RmseSummary,
    load_comparison_npz,
    print_metrics_table,
)


def output_path(
    output_dir: Path,
    filename: str,
    output_prefix: str | None = None,
) -> Path:
    """Build an output path, optionally prefixing the filename."""
    if output_prefix:
        filename = f"{output_prefix}_{filename}"
    return output_dir / filename


def add_timing_markers(
    ax: plt.Axes,
    group: PlotGroup,
    *,
    t0: float,
    show_phase_label: bool,
) -> None:
    """Mark setpoint changes and the regulation-to-tracking boundary."""
    for event_time in group.event_times:
        if event_time > t0:
            ax.axvline(x=event_time, color="gray", linestyle=":", alpha=0.5)

    if group.phase_boundary_time is None:
        return

    ax.axvline(
        x=group.phase_boundary_time,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    if show_phase_label and group.phase_boundary_label:
        ax.text(
            group.phase_boundary_time,
            0.98,
            group.phase_boundary_label,
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=9,
        )


def plot_tracking_group(
    run: ComparisonRun,
    group: PlotGroup,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    show: bool = True,
) -> tuple[Path, Path]:
    """Plot desired/actual strains and tracking errors for one controller group."""
    all_results = {
        name: run.results[group.scenario_key][name] for name in group.controller_names
    }
    first_controller = group.controller_names[0]
    strain_indices = run.strain_indices
    strain_names = run.strain_names

    fig, axes = plt.subplots(len(strain_indices), 1, figsize=(12, 10), sharex=True)
    axes = np.atleast_1d(axes)

    for idx, (strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
        ax = axes[idx]
        t = all_results[first_controller]["t"]
        q_des = all_results[first_controller]["metrics"]["q_des"]
        ax.plot(t, q_des[:, strain_idx], "k--", linewidth=2, label="Desired", alpha=0.7)

        for name, results in all_results.items():
            ax.plot(
                results["t"],
                results["q"][:, strain_idx],
                color=group.colors[name],
                linewidth=1.5,
                label=name,
            )

        add_timing_markers(
            ax,
            group,
            t0=float(t[0]),
            show_phase_label=idx == 0,
        )

        ax.set_ylabel(f"{strain_name}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(group.title)
    plt.tight_layout()

    tracking_output = output_path(
        output_dir,
        f"{group.filename_prefix}_tracking.pdf",
        output_prefix,
    )
    fig.savefig(tracking_output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, axes = plt.subplots(len(strain_indices), 1, figsize=(12, 8), sharex=True)
    axes = np.atleast_1d(axes)

    for idx, (strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
        ax = axes[idx]
        for name, results in all_results.items():
            tracking_error = results["metrics"]["tracking_error"]
            ax.plot(
                results["t"],
                tracking_error[:, strain_idx],
                color=group.colors[name],
                linewidth=1.5,
                label=name,
            )

        add_timing_markers(
            ax,
            group,
            t0=float(all_results[first_controller]["t"][0]),
            show_phase_label=idx == 0,
        )

        ax.set_ylabel(f"Error {strain_name}")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"{group.title}: Tracking Errors")
    plt.tight_layout()

    error_output = output_path(
        output_dir,
        f"{group.filename_prefix}_errors.pdf",
        output_prefix,
    )
    fig.savefig(error_output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return tracking_output, error_output


def plot_rmse_summary(
    run: ComparisonRun,
    summary: RmseSummary,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    show: bool = True,
) -> Path:
    """Plot one RMSE summary figure."""
    fig, axes = plt.subplots(
        1, len(summary.panels), figsize=(7 * len(summary.panels), 5)
    )
    axes = np.atleast_1d(axes)
    width = 0.8 / len(run.strain_names)

    for ax, panel in zip(axes, summary.panels):
        controller_names = list(panel.controller_names)
        x = np.arange(len(controller_names))
        for strain_offset, strain_name in enumerate(run.strain_names):
            rmse_values = [
                run.results[panel.scenario_key][name]["metrics"]["rmse"][strain_offset]
                for name in controller_names
            ]
            ax.bar(x + strain_offset * width, rmse_values, width, label=strain_name)

        ax.set_xlabel("Controller")
        ax.set_ylabel("RMSE")
        ax.set_title(panel.title)
        ax.set_xticks(x + width * (len(run.strain_names) - 1) / 2)
        ax.set_xticklabels(controller_names, rotation=30, ha="right", fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output = output_path(output_dir, summary.filename, output_prefix)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def plot_run(
    run: ComparisonRun,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    group_keys: set[str] | None = None,
    show: bool = True,
) -> list[Path]:
    """Generate all selected plots for a saved comparison run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    selected_groups = [
        group
        for group in run.plot_groups
        if group_keys is None or group.key in group_keys
    ]
    selected_summaries = [
        summary
        for summary in run.rmse_summaries
        if group_keys is None or summary.key in group_keys
    ]
    selected_keys = {group.key for group in selected_groups} | {
        summary.key for summary in selected_summaries
    }
    if group_keys is not None and selected_keys != group_keys:
        unknown = ", ".join(sorted(group_keys - selected_keys))
        raise ValueError(f"Unknown plot group(s): {unknown}")

    for group in selected_groups:
        outputs.extend(
            plot_tracking_group(
                run,
                group,
                output_dir=output_dir,
                output_prefix=output_prefix,
                show=show,
            )
        )

    for summary in selected_summaries:
        outputs.append(
            plot_rmse_summary(
                run,
                summary,
                output_dir=output_dir,
                output_prefix=output_prefix,
                show=show,
            )
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="Saved comparison .npz file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated PDF plots.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional string prepended to each generated filename.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Plot only this group key. Can be passed multiple times.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_comparison_npz(args.trajectory)

    for scenario in run.scenarios:
        print_metrics_table(
            run.results[scenario.key],
            run.strain_names,
            f"{scenario.title}: RMSE",
        )

    outputs = plot_run(
        run,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        group_keys=set(args.group) if args.group else None,
        show=not args.no_show,
    )

    print("\nPlots saved to:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
