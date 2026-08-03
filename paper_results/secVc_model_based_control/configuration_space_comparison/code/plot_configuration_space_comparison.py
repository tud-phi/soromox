"""Plot saved configuration-space control comparison rollouts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from configuration_space_comparison_simulation import (
    OUTPUTS_DIR,
    ComparisonRun,
    PlotGroup,
    RmsePanel,
    RmseSummary,
    load_comparison_npz,
    print_metrics_table,
)

SCRIPT_PATH = Path(__file__)
CASE_DIR = SCRIPT_PATH.parent.parent
PAPER_STYLE = SCRIPT_PATH.parents[3] / "paper.mplstyle"
CM_TO_INCH = 1.0 / 2.54
CANONICAL_INPUT = CASE_DIR / "data" / "regulation_tracking_comparison.npz"
CANONICAL_OUTPUT = OUTPUTS_DIR / "configuration_space_comparison.pdf"

STRAIN_UNITS = (
    r"rad\,m^{-1}",
    r"rad\,m^{-1}",
    "-",
)
CONTROLLER_LINESTYLES = {
    "PD (model-free)": ":",
    "PID (model-free)": "-.",
    "Potential Compensation": "--",
    "Potential Cancellation": "--",
    "Gravity Cancellation": "-.",
    "Feedforward Compensation": "-",
    "Computed Torque": "-",
    "Mixed State Feedback": "-.",
}
PHYSICAL_STRAIN_OFFSETS = {
    r"$\sigma_x$": 1.0,
}


def configure_matplotlib() -> None:
    """Apply the shared publication style with dense multi-panel sizing."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": None,
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def cm_to_inches(size_cm: tuple[float, float]) -> tuple[float, float]:
    """Convert a figure size in centimeters to inches."""
    return tuple(value * CM_TO_INCH for value in size_cm)


def output_path(
    output_dir: Path,
    filename: str,
    output_prefix: str | None = None,
) -> Path:
    """Build an output path, optionally prefixing the filename."""
    if output_prefix:
        filename = f"{output_prefix}_{filename}"
    return output_dir / filename


def ensure_writable(path: Path, *, force: bool) -> None:
    """Reject accidental replacement of a canonical paper artifact."""
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def add_timing_markers(ax: plt.Axes, group: PlotGroup, *, t0: float) -> None:
    """Mark setpoint changes and the regulation-to-tracking boundary."""
    for event_time in group.event_times:
        if event_time > t0:
            ax.axvline(
                event_time,
                color="0.55",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
                zorder=0,
            )
    if group.phase_boundary_time is not None:
        ax.axvline(
            group.phase_boundary_time,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=0,
        )


def _group_results(
    run: ComparisonRun, group: PlotGroup
) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: run.results[group.scenario_key][name] for name in group.controller_names
    }


def physical_strain_values(values: np.ndarray, strain_name: str) -> np.ndarray:
    """Convert stored configuration displacements to plotted physical strains."""
    return np.asarray(values) + PHYSICAL_STRAIN_OFFSETS.get(strain_name, 0.0)


def plot_tracking_axes(
    run: ComparisonRun,
    group: PlotGroup,
    axes: np.ndarray,
    *,
    errors: bool,
) -> None:
    """Plot tracking signals or errors into caller-owned strain axes."""
    all_results = _group_results(run, group)
    first = all_results[group.controller_names[0]]
    t = np.asarray(first["t"])
    q_des = np.asarray(first["metrics"]["q_des"])

    for ax, strain_idx, strain_name, unit in zip(
        axes, run.strain_indices, run.strain_names, STRAIN_UNITS
    ):
        if not errors:
            ax.plot(
                t,
                physical_strain_values(q_des[:, strain_idx], strain_name),
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Desired",
                zorder=5,
            )

        for name in group.controller_names:
            results = all_results[name]
            values = (
                results["metrics"]["tracking_error"][:, strain_idx]
                if errors
                else physical_strain_values(
                    results["q"][:, strain_idx], strain_name
                )
            )
            ax.plot(
                results["t"],
                values,
                color=group.colors[name],
                linestyle=CONTROLLER_LINESTYLES.get(name, "-"),
                linewidth=1.2,
                label=name,
            )

        add_timing_markers(ax, group, t0=float(t[0]))
        if errors:
            ax.axhline(0.0, color="0.25", linewidth=0.7, alpha=0.7, zorder=0)
        symbol = rf"$e_{{{strain_name.strip('$')}}}$" if errors else strain_name
        ax.set_ylabel(rf"{symbol} $\mathrm{{[{unit}]}}$")
        ax.set_xlim(float(t[0]), float(t[-1]))


def _phase_labels(axes: np.ndarray, boundary_time: float, t1: float) -> None:
    split = boundary_time / t1
    for ax in axes:
        ax.text(
            split / 2.0,
            1.025,
            "Setpoint regulation",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax.text(
            split + (1.0 - split) / 2.0,
            1.025,
            "Trajectory tracking",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
        )


def build_configuration_space_paper_figure(run: ComparisonRun) -> plt.Figure:
    """Build the canonical regulation-to-tracking comparison figure."""
    group = next((item for item in run.plot_groups if item.key == "combined"), None)
    if group is None or group.phase_boundary_time is None:
        raise ValueError("The canonical figure requires the combined phase plot group.")

    configure_matplotlib()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=cm_to_inches((16.5, 9.5)),
        sharex=True,
        constrained_layout=True,
    )
    plot_tracking_axes(run, group, axes, errors=False)
    axes[-1].set_xlabel(r"Time $\mathrm{[s]}$")

    t1 = float(_group_results(run, group)[group.controller_names[0]]["t"][-1])
    _phase_labels(axes[:1], group.phase_boundary_time, t1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        columnspacing=1.2,
        handlelength=2.5,
    )
    return fig


def save_configuration_space_paper_figure(
    run: ComparisonRun,
    output: Path,
    *,
    show: bool,
    force: bool,
) -> Path:
    """Save the canonical configuration-space comparison PDF."""
    ensure_writable(output, force=force)
    fig = build_configuration_space_paper_figure(run)
    fig.savefig(output, bbox_inches=None)
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_tracking_group(
    run: ComparisonRun,
    group: PlotGroup,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    show: bool = True,
    force: bool = False,
) -> tuple[Path, Path]:
    """Plot desired/actual strains and tracking errors for one controller group."""
    configure_matplotlib()
    outputs: list[Path] = []
    for errors, suffix in ((False, "tracking"), (True, "errors")):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=cm_to_inches((16.5, 9.5)),
            sharex=True,
            constrained_layout=True,
        )
        plot_tracking_axes(run, group, axes, errors=errors)
        axes[-1].set_xlabel(r"Time $\mathrm{[s]}$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3)
        output = output_path(
            output_dir,
            f"{group.filename_prefix}_{suffix}.pdf",
            output_prefix,
        )
        ensure_writable(output, force=force)
        fig.savefig(output, bbox_inches=None)
        if show:
            plt.show()
        plt.close(fig)
        outputs.append(output)
    return outputs[0], outputs[1]


def metric_values(
    run: ComparisonRun, panel: RmsePanel, strain_offset: int
) -> np.ndarray:
    """Return controller metric values for one saved phase and strain."""
    strain_index = run.strain_indices[strain_offset]
    return np.asarray(
        [
            run.results[panel.scenario_key][name]["metrics"][panel.metric_name][
                strain_index
            ]
            for name in panel.controller_names
        ],
        dtype=float,
    )


def _controller_colors(run: ComparisonRun) -> dict[str, str]:
    colors: dict[str, str] = {}
    for group in run.plot_groups:
        colors.update(group.colors)
    return colors


def _format_bar_value(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.1e}"
    return f"{value:.2f}"


def build_rmse_summary_figure(run: ComparisonRun, summary: RmseSummary) -> plt.Figure:
    """Build a phase-by-strain metric matrix with unit-correct axes."""
    configure_matplotlib()
    n_columns = len(summary.panels)
    width_cm = 8.2 if n_columns == 1 else 16.5
    fig, axes = plt.subplots(
        len(run.strain_names),
        n_columns,
        figsize=cm_to_inches((width_cm, 10.5)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = _controller_colors(run)

    for row, (strain_name, unit) in enumerate(zip(run.strain_names, STRAIN_UNITS)):
        row_values = [metric_values(run, panel, row) for panel in summary.panels]
        metric_limits: dict[str, float] = {}
        for panel, values in zip(summary.panels, row_values):
            metric_limits[panel.metric_name] = max(
                metric_limits.get(panel.metric_name, 0.0),
                float(values.max(initial=0.0)),
            )
        metric_limits = {
            metric_name: max(limit * 1.22, 1e-12)
            for metric_name, limit in metric_limits.items()
        }

        for column, (panel, values) in enumerate(zip(summary.panels, row_values)):
            ax = axes[row, column]
            row_limit = metric_limits[panel.metric_name]
            names = list(panel.controller_names)
            y = np.arange(len(names))
            bars = ax.barh(
                y,
                values,
                color=[colors.get(name, "0.5") for name in names],
                height=0.7,
            )
            ax.set_xlim(0.0, row_limit)
            ax.invert_yaxis()
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.set_xlabel(
                rf"{panel.metric_label} {strain_name} $\mathrm{{[{unit}]}}$"
            )
            ax.set_yticks(y)
            ax.set_yticklabels(names if column == 0 else [])
            if row == 0:
                title = panel.title.removesuffix(f": {panel.metric_label}")
                ax.set_title(title)
            for bar, value in zip(bars, values):
                ax.text(
                    min(float(value) + 0.015 * row_limit, 0.98 * row_limit),
                    bar.get_y() + bar.get_height() / 2.0,
                    _format_bar_value(float(value)),
                    ha="left",
                    va="center",
                    fontsize=5.5,
                )
    return fig


def plot_rmse_summary(
    run: ComparisonRun,
    summary: RmseSummary,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    show: bool = True,
    force: bool = False,
) -> Path:
    """Save a unit-correct phase-by-strain metric summary figure."""
    output = output_path(output_dir, summary.filename, output_prefix)
    ensure_writable(output, force=force)
    fig = build_rmse_summary_figure(run, summary)
    fig.savefig(output, bbox_inches=None)
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_run(
    run: ComparisonRun,
    *,
    output_dir: Path,
    output_prefix: str | None = None,
    group_keys: set[str] | None = None,
    show: bool = True,
    force: bool = False,
) -> list[Path]:
    """Generate all selected plots for a saved comparison run."""
    output_dir.mkdir(parents=True, exist_ok=True)
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

    outputs: list[Path] = []
    for group in selected_groups:
        outputs.extend(
            plot_tracking_group(
                run,
                group,
                output_dir=output_dir,
                output_prefix=output_prefix,
                show=show,
                force=force,
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
                force=force,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="Saved comparison .npz file.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR, help="Generated PDF directory."
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_comparison_npz(args.trajectory)
    for scenario in run.scenarios:
        print_metrics_table(
            run.results[scenario.key],
            run.evaluation_strain_names,
            f"{scenario.title}: RMSE",
        )
    outputs = plot_run(
        run,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        group_keys=set(args.group) if args.group else None,
        show=not args.no_show,
        force=args.force,
    )
    print("\nPlots saved to:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
