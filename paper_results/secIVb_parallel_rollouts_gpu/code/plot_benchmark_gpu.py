#!/usr/bin/env python3
"""Visualise batch-scaling benchmark CSV results."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Directory Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
PAPER_RESULTS_DIR = SCRIPT_DIR.parent
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE

# --- Configuration Constants ---
MODEL_ORDER = ["articulated_soft_robot", "planar_pcs", "pcs", "gvs"]
MODEL_LABELS = {
    "articulated_soft_robot": "Articulated Soft Robot",
    "planar_pcs": "Planar PCS",
    "pcs": "Spatial PCS",
    "gvs": "Spatial GVS",
}
MODEL_COLORS = {
    "articulated_soft_robot": COLORS["pre_opt_1"],
    "planar_pcs": COLORS["pre_opt_2"],
    "pcs": COLORS["post_opt_1"],
    "gvs": COLORS["post_opt_2"],
}

SIZE_STYLES = {
    1: ("-", "o"),
    2: ("--", "s"),
    4: ("-.", "^"),
    8: (":", "D"),
    16: ((0, (5, 2)), "p"),
    32: ((0, (1, 1)), "x"),
}
FALLBACK_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1))]
FALLBACK_MARKERS = ["o", "s", "^", "D", "p", "x", "v", "*"]
SEGMENT_SIZE_LABELS = {"num_segments", "num_links"}
MAX_SYSTEM_COLUMNS = 2
SYSTEM_COLUMN_WIDTH_CM = 9.0


def configure_matplotlib() -> None:
    """Configure the standardized CMU Serif publication theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "figure.dpi": 300,
        }
    )


def cm2inch(*tupl: float) -> tuple[float, ...]:
    """Convert centimeters to inches for Matplotlib figsize."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


def _subplot_grid(num_systems: int) -> tuple[int, int]:
    """Return a paper-width grid for one panel per system."""
    if num_systems < 1:
        raise ValueError("At least one system is required.")
    columns = min(MAX_SYSTEM_COLUMNS, num_systems)
    rows = (num_systems + columns - 1) // columns
    return rows, columns


# --- Data Processing Helpers ---
def _system_key(system: Any) -> str:
    return str(system).strip().replace("-", "_").replace(" ", "_").lower()


def _compact_system_key(system: Any) -> str:
    return _system_key(system).replace("_", "")


def _canonical_system(system: Any) -> str:
    key = _system_key(system)
    if key in MODEL_LABELS:
        return key
    compact = key.replace("_", "")
    for canonical in MODEL_LABELS:
        if compact == canonical.replace("_", ""):
            return canonical
    return key


def _system_sort_key(system: Any) -> tuple[int, str]:
    canonical = _canonical_system(system)
    try:
        return MODEL_ORDER.index(canonical), canonical
    except ValueError:
        return len(MODEL_ORDER), canonical


def _model_label(system: Any) -> str:
    canonical = _canonical_system(system)
    return MODEL_LABELS.get(canonical, str(system))


def _model_color(system: Any, index: int) -> str:
    canonical = _canonical_system(system)
    if canonical in MODEL_COLORS:
        return MODEL_COLORS[canonical]
    palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]
    return palette[index % len(palette)]


def _size_value(row: Mapping[str, Any]) -> int:
    return int(row.get("size_value", row["segment_count"]))


def _size_label(size_label: str, *, compact: bool = False) -> str:
    if size_label in {"num_segments", "num_links"}:
        return r"$N$" if compact else r"Number of segments/links $N$"
    if size_label == "strain_basis_order":
        return r"$p$" if compact else r"Strain-basis order $p$"
    return size_label


def _speed_value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "") and key == "total_speed_ratio":
        value = row.get("throughput_ratio")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _style_for_size(size_value: int) -> tuple[Any, str]:
    if size_value in SIZE_STYLES:
        return SIZE_STYLES[size_value]
    idx = abs(int(size_value)) % len(FALLBACK_LINESTYLES)
    marker_idx = abs(int(size_value)) % len(FALLBACK_MARKERS)
    return FALLBACK_LINESTYLES[idx], FALLBACK_MARKERS[marker_idx]


def _coerce(value: str) -> Any:
    try:
        if value.lower() in {"", "none", "null"}:
            return None
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
    except AttributeError:
        return value

    for cast in (int, float):
        try:
            return cast(value)
        except (ValueError, TypeError):
            continue
    return value


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"none", "null"}


def _load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows.extend(
                {key: _coerce(val) for key, val in row.items()} for row in reader
            )
    if not rows:
        raise ValueError(f"No rows found in {', '.join(str(path) for path in paths)}")
    return rows


def _filter_rows(
    rows: Iterable[Mapping[str, Any]],
    systems: Sequence[str] | None,
    segment_counts: Sequence[int] | None,
    gauss_points: Sequence[int] | None,
) -> list[Mapping[str, Any]]:
    filtered: list[Mapping[str, Any]] = []
    system_keys = (
        {_compact_system_key(system) for system in systems} if systems else None
    )
    for row in rows:
        if system_keys and _compact_system_key(row["system"]) not in system_keys:
            continue
        if segment_counts and _size_value(row) not in segment_counts:
            continue
        if (
            gauss_points
            and not _is_missing(row.get("gauss_points"))
            and int(row["gauss_points"]) not in gauss_points
        ):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No rows remain after applying the requested filters.")
    return filtered


def _filter_to_shared_segment_range(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    sizes_by_system: dict[Any, list[int]] = {}
    for row in rows:
        sizes_by_system.setdefault(row["system"], []).append(_size_value(row))

    if len(sizes_by_system) < 2:
        return list(rows)

    common_min = max(min(sizes) for sizes in sizes_by_system.values())
    common_max = min(max(sizes) for sizes in sizes_by_system.values())
    filtered = [row for row in rows if common_min <= _size_value(row) <= common_max]

    if not filtered:
        raise ValueError(
            "No rows remain after limiting to the shared segment-count range."
        )
    return filtered


def _gauss_groups(rows: Sequence[Mapping[str, Any]]) -> Sequence[int | None]:
    gauss_values = sorted(
        {
            int(row["gauss_points"])
            for row in rows
            if not _is_missing(row.get("gauss_points"))
        }
    )
    return gauss_values if gauss_values else [None]


def _rows_for_group(
    rows: Sequence[Mapping[str, Any]],
    size_value: int,
    gauss_points: int | None,
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if _size_value(row) == size_value
            and (
                gauss_points is None
                or (
                    not _is_missing(row.get("gauss_points"))
                    and int(row["gauss_points"]) == gauss_points
                )
            )
        ),
        key=lambda row: row["batch_size"],
    )


def _group_label(
    size_label: str,
    size_value: int,
    gauss_points: int | None,
    show_gauss: bool,
) -> str:
    label = f"{_size_label(size_label, compact=True)}={size_value}"
    if show_gauss:
        label = f"{label}, gauss={gauss_points}"
    return label


def _curve_xy(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    values_by_batch: dict[float, list[float]] = {}
    for row in rows:
        try:
            x = float(row["batch_size"])
        except (TypeError, ValueError):
            continue
        y = _speed_value(row, "total_speed_ratio")
        if x <= 0 or y is None or y <= 0:
            continue
        values_by_batch.setdefault(x, []).append(y)

    x_values = np.array(sorted(values_by_batch), dtype=float)
    y_values = np.array(
        [np.mean(values_by_batch[x]) for x in x_values],
        dtype=float,
    )
    return x_values, y_values


# --- Plotting Functions ---
def _plot(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    system_rows, system_columns = _subplot_grid(len(systems))

    fig, axes = plt.subplots(
        2 * system_rows,
        system_columns,
        figsize=cm2inch(
            SYSTEM_COLUMN_WIDTH_CM * system_columns,
            12.0 * system_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for system_index, system in enumerate(systems):
        system_row, col = divmod(system_index, system_columns)
        ax_per_env = axes[2 * system_row, col]
        ax_total = axes[2 * system_row + 1, col]
        subset = [row for row in rows if row["system"] == system]
        size_label = subset[0]["size_label"]
        segment_values = sorted({_size_value(row) for row in subset})
        gauss_groups = _gauss_groups(subset)
        show_gauss = len(gauss_groups) > 1

        for size in segment_values:
            for gauss_points in gauss_groups:
                seg_rows = _rows_for_group(subset, size, gauss_points)
                if not seg_rows:
                    continue
                x = [row["batch_size"] for row in seg_rows]
                y_per_env = [
                    _speed_value(row, "per_env_speed_ratio")
                    if row.get("per_env_speed_ratio") not in (None, "")
                    else _speed_value(row, "speed_ratio")
                    for row in seg_rows
                ]
                y_total = [_speed_value(row, "total_speed_ratio") for row in seg_rows]
                label = _group_label(size_label, size, gauss_points, show_gauss)

                ax_per_env.plot(x, y_per_env, marker="o", label=label)
                ax_total.plot(x, y_total, marker="o", label=label)

        ax_per_env.set_title(f"{_model_label(system)}\nPer-Environment Speed")
        ax_total.set_title(f"{_model_label(system)}\nTotal Throughput")
        ax_total.set_xlabel(r"Number of environments $n_\mathrm{b}$")
        if col == 0:
            ax_per_env.set_ylabel("Per-environment speed ratio")
            ax_total.set_ylabel(r"Total throughput $\Gamma_\mathrm{sim}$")

        for axis in (ax_per_env, ax_total):
            if log_x:
                axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
        ax_per_env.legend()

    for system_index in range(len(systems), system_rows * system_columns):
        system_row, col = divmod(system_index, system_columns)
        axes[2 * system_row, col].set_visible(False)
        axes[2 * system_row + 1, col].set_visible(False)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Saved figure to {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_total_throughput(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
    log_x: bool,
    log_y: bool,
) -> None:
    if not rows:
        return

    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    system_rows, system_columns = _subplot_grid(len(systems))

    fig, axes = plt.subplots(
        system_rows,
        system_columns,
        figsize=cm2inch(
            SYSTEM_COLUMN_WIDTH_CM * system_columns,
            8.0 * system_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for system_index, system in enumerate(systems):
        system_row, col = divmod(system_index, system_columns)
        ax = axes[system_row, col]
        subset = [row for row in rows if row["system"] == system]
        size_label = subset[0]["size_label"]
        segment_values = sorted({_size_value(row) for row in subset})
        gauss_groups = _gauss_groups(subset)
        show_gauss = len(gauss_groups) > 1

        for size in segment_values:
            for gauss_points in gauss_groups:
                seg_rows = _rows_for_group(subset, size, gauss_points)
                if not seg_rows:
                    continue
                x = [row["batch_size"] for row in seg_rows]
                y_total = [_speed_value(row, "total_speed_ratio") for row in seg_rows]
                label = _group_label(size_label, size, gauss_points, show_gauss)
                ax.plot(x, y_total, marker="o", label=label)

        ax.set_title(_model_label(system))
        ax.set_xlabel(r"Number of environments $n_\mathrm{b}$")
        if col == 0:
            ax.set_ylabel(r"Total throughput $\Gamma_\mathrm{sim}$")
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    for system_index in range(len(systems), system_rows * system_columns):
        system_row, col = divmod(system_index, system_columns)
        axes[system_row, col].set_visible(False)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Saved total-throughput figure to {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_combined_scaling(
    rows: list[Mapping[str, Any]],
    output: Path | None,
    show: bool,
) -> None:
    if not rows:
        return

    systems = sorted({row["system"] for row in rows}, key=_system_sort_key)
    fig, ax = plt.subplots(figsize=cm2inch(18.0, 12.0), constrained_layout=True)
    all_x: list[float] = []
    all_y: list[float] = []

    for system_index, system in enumerate(systems):
        subset = [row for row in rows if row["system"] == system]
        if not subset:
            continue

        color = _model_color(system, system_index)
        size_values = sorted({_size_value(row) for row in subset})
        curves: list[tuple[int, np.ndarray, np.ndarray]] = []
        for size in size_values:
            curve_rows = [row for row in subset if _size_value(row) == size]
            x_values, y_values = _curve_xy(curve_rows)
            if x_values.size == 0:
                continue
            curves.append((size, x_values, y_values))
            all_x.extend(x_values.tolist())
            all_y.extend(y_values.tolist())

        if not curves:
            continue

        envelope_x = np.array(sorted({x for _, xs, _ in curves for x in xs}))
        lower: list[float] = []
        upper: list[float] = []
        fill_x: list[float] = []
        for x_val in envelope_x:
            y_at_x: list[float] = []
            for _, xs, ys in curves:
                matches = np.where(xs == x_val)[0]
                if matches.size:
                    y_at_x.append(float(ys[matches[0]]))
            if y_at_x:
                fill_x.append(float(x_val))
                lower.append(min(y_at_x))
                upper.append(max(y_at_x))

        if len(fill_x) >= 2:
            ax.fill_between(
                fill_x,
                lower,
                upper,
                color=color,
                alpha=0.12,
                linewidth=0.0,
                zorder=1,
            )

        outer_sizes = {
            min(size for size, _, _ in curves),
            max(size for size, _, _ in curves),
        }
        for size, x_values, y_values in curves:
            linestyle, marker = _style_for_size(size)
            is_outer = size in outer_sizes
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5.0 if is_outer else 4.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=1.9 if is_outer else 1.15,
                alpha=0.95 if is_outer else 0.52,
                zorder=3 if is_outer else 2,
            )

    if not all_x or not all_y:
        plt.close(fig)
        return

    x_min = min(all_x)
    x_min = min(all_x)
    y_min, y_max = min(all_y), max(all_y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of parallel environments $n_\mathrm{b}$")
    ax.set_ylabel(r"Simulation throughput $\Gamma_\mathrm{sim}$")
    ax.set_title("GPU Batch-Scaling Throughput")

    if y_min <= 1.0 <= y_max:
        ax.axhline(1.0, color="#717674", linestyle=":", linewidth=1.8, alpha=0.9)
        ax.text(
            x_min,
            1.0,
            "Real time",
            fontsize=14,
            color="#717674",
            ha="left",
            va="bottom",
        )

    # ---------------------------------------------------------
    # DYNAMIC LEGEND: Built only from the filtered rows
    # ---------------------------------------------------------
    model_handles = [
        Line2D(
            [0],
            [0],
            color=_model_color(system, idx),
            linewidth=2.2,
            label=_model_label(system),
        )
        for idx, system in enumerate(systems)
    ]

    # Read sizes directly from the user-filtered dataset
    size_values = sorted({_size_value(row) for row in rows})

    size_handles = []
    for size in size_values:
        linestyle, marker = _style_for_size(size)
        size_handles.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=1.3,
                label=rf"$N$={size}",
            )
        )

    first_legend = ax.legend(
        handles=model_handles,
        title="Robot model",
        loc="upper left",
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=size_handles,
        title="Number of segments",
        loc="lower right",
        ncol=2 if len(size_handles) > 4 else 1,
        columnspacing=1.2,
        handlelength=2.2,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Saved combined scaling figure to {output.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


# --- Interactive Menu ---
def _interactive_selection(
    args: argparse.Namespace, rows: list[Mapping[str, Any]]
) -> argparse.Namespace:
    """Prompt user for model and segment selections if not provided via CLI."""
    print("\n" + "=" * 40)
    print(" SoRoMoX Batch Scaling Plotter")
    print("=" * 40)

    # 1. Model Selection
    if not args.systems:
        available_systems = sorted(
            {_canonical_system(r["system"]) for r in rows},
            key=lambda s: MODEL_ORDER.index(s) if s in MODEL_ORDER else 99,
        )
        print("\nAvailable Robot Models:")
        print("  0: All Models (Default)")
        for i, sys_val in enumerate(available_systems, 1):
            print(f"  {i}: {_model_label(sys_val)}")

        sel = input(
            "\nSelect models (comma-separated numbers, e.g., 1,3) [0]: "
        ).strip()
        if sel and sel != "0":
            try:
                indices = [int(x.strip()) - 1 for x in sel.split(",")]
                args.systems = [
                    available_systems[i]
                    for i in indices
                    if 0 <= i < len(available_systems)
                ]
            except ValueError:
                print("[!] Invalid input, proceeding with all models.")

    # 2. Segment Selection
    if not args.segment_counts:
        temp_systems = (
            {_compact_system_key(s) for s in args.systems} if args.systems else None
        )
        # Assess available segment sizes for the chosen models
        temp_rows = [
            r
            for r in rows
            if r["size_label"] in SEGMENT_SIZE_LABELS
            and (not temp_systems or _compact_system_key(r["system"]) in temp_systems)
        ]
        available_sizes = sorted({_size_value(r) for r in temp_rows})

        if available_sizes:
            print("\nAvailable Segment Counts:")
            print(f"  0: All Available Segments {available_sizes} (Default)")
            sel = input(
                "\nSelect segment counts (comma-separated, e.g., 1,2,4) [0]: "
            ).strip()
            if sel and sel != "0":
                try:
                    args.segment_counts = [int(x.strip()) for x in sel.split(",")]
                except ValueError:
                    print("[!] Invalid input, proceeding with all available segments.")

    print("\nGenerating plots...\n")
    return args


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise batch-scaling benchmark CSV results."
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        type=Path,
        default=None,
        help="Specific CSV files to plot. If omitted, discovers all in data_dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to search for CSVs if --csv is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="batch_scaling.pdf",
        help="Filename for the main scaling plot.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        help="Optional subset of systems to plot (skips interactive menu if provided)",
    )
    parser.add_argument(
        "--segment-counts",
        nargs="*",
        type=int,
        help="Optional subset of segment counts to plot (skips interactive menu if provided)",
    )
    parser.add_argument(
        "--gauss-points",
        nargs="*",
        type=int,
        help="Optional subset of Gauss-Legendre point counts to plot",
    )
    parser.add_argument(
        "--no-combined-plot",
        action="store_true",
        help="Do not generate the consolidated all-model scaling plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic scale for the number-of-environments axis",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic scale for the speed ratio axes",
    )
    args = parser.parse_args(argv)

    if not args.csv:
        if args.data_dir.exists():
            args.csv = list(args.data_dir.glob("*.csv"))
        else:
            args.csv = []

    if not args.csv:
        print(f"[!] No CSV files found in {args.data_dir}. Exiting.")
        sys.exit(1)

    args.output = args.output_dir / args.output_name
    args.combined_output = None
    if not args.no_combined_plot:
        args.combined_output = args.output.with_name(
            f"{args.output.stem}_combined{args.output.suffix}"
        )

    return args


def main(argv: Sequence[str] | None = None) -> int:
    configure_matplotlib()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Load all rows
    rows = _load_rows(args.csv)

    # Isolate segment data strictly to prevent 'p' basis-order corruption in the prompt/legend
    rows = [r for r in rows if r["size_label"] in SEGMENT_SIZE_LABELS]

    # Launch interactive CLI
    args = _interactive_selection(args, rows)

    # Filter dataset based on CLI or interactive choices
    rows = _filter_rows(rows, args.systems, args.segment_counts, args.gauss_points)
    rows = _filter_to_shared_segment_range(rows)

    throughput_output = args.output.with_name(
        f"{args.output.stem}_total_throughput{args.output.suffix}"
    )

    _plot(rows, args.output, args.show, args.log_x, args.log_y)
    _plot_total_throughput(rows, throughput_output, args.show, args.log_x, args.log_y)

    if args.combined_output is not None or args.show:
        _plot_combined_scaling(rows, args.combined_output, args.show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
