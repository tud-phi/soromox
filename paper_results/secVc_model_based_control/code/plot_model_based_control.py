"""Compose the model-based-control application figure."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

SECTION_DIR = Path(__file__).resolve().parent.parent
PAPER_RESULTS_DIR = SECTION_DIR.parent
PAPER_STYLE = PAPER_RESULTS_DIR / "paper.mplstyle"
CONFIGURATION_CODE_DIR = (
    SECTION_DIR
    / "configuration_space_comparison"
    / "code"
)
OPERATIONAL_CODE_DIR = (
    SECTION_DIR
    / "operational_space_impedance_control"
    / "code"
)
for code_dir in (CONFIGURATION_CODE_DIR, OPERATIONAL_CODE_DIR):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import plot_configuration_space_comparison as configuration_plot  # noqa: E402
from operational_space_impedance_common import (  # noqa: E402
    COORDINATE_COLORS,
    DEFAULT_TRAJECTORY_OUTPUT,
    PCSImpedanceProblem,
    PCSImpedanceRun,
    build_problem,
    load_run_npz,
    operational_tracking_data,
)

FIGURE_SIZE_CM = (18.1, 15.1)
CM_TO_INCH = 1.0 / 2.54

FONT_SIZE = 8.0
SMALL_FONT_SIZE = 7.0
PANEL_LABEL_SIZE = 11.0
SPINE_WIDTH = 0.8
GRID_WIDTH = 0.6
REFERENCE_LINE_WIDTH = 1.5
PRIMARY_LINE_WIDTH = 1.25
SECONDARY_LINE_WIDTH = 1.0
MARKER_SIZE = 2.5

DEFAULT_CONFIGURATION_INPUT = configuration_plot.CANONICAL_INPUT
DEFAULT_OPERATIONAL_INPUT = DEFAULT_TRAJECTORY_OUTPUT
DEFAULT_SNAPSHOT_DIR = OPERATIONAL_CODE_DIR.parent / "outputs" / (
    "operational_space_impedance_snapshots"
)
DEFAULT_OUTPUT_BASE = (
    SECTION_DIR / "outputs" / "model_based_control"
)

SNAPSHOT_PATTERN = re.compile(
    r"^snapshot_t(?P<seconds>\d+)p(?P<fraction>\d+)s\.png$"
)

COMPOSITE_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
    "font.size": FONT_SIZE,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.linewidth": SPINE_WIDTH,
    "xtick.labelsize": SMALL_FONT_SIZE,
    "ytick.labelsize": SMALL_FONT_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": SMALL_FONT_SIZE,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.63",
    "legend.fancybox": False,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 2.4,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "grid.color": "0.69",
    "grid.alpha": 0.5,
    "grid.linestyle": "-",
    "grid.linewidth": GRID_WIDTH,
    "lines.linewidth": PRIMARY_LINE_WIDTH,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
}


@dataclass(frozen=True)
class CompositeAxes:
    """Named axes used by the composite figure and its regression tests."""

    configuration: tuple[Axes, Axes, Axes]
    operational: tuple[Axes, Axes]
    operational_errors: tuple[Axes, Axes]
    snapshots: tuple[Axes, Axes, Axes, Axes]

    @property
    def plot_axes(self) -> tuple[Axes, ...]:
        """Return all line-plot axes in reading order."""
        return (
            *self.configuration,
            *self.operational,
            *self.operational_errors,
        )


def cm_to_inches(size_cm: tuple[float, float]) -> tuple[float, float]:
    """Convert a two-dimensional size from centimetres to inches."""
    return tuple(value * CM_TO_INCH for value in size_cm)


def _combined_configuration_group(run):
    group = next((item for item in run.plot_groups if item.key == "combined"), None)
    if group is None or group.phase_boundary_time is None:
        raise ValueError("The configuration input requires the combined phase group.")
    return group


def _style_axis(ax: Axes) -> None:
    """Apply the shared physical typography and axes contract."""
    ax.grid(True, color="0.69", alpha=0.5, linewidth=GRID_WIDTH)
    ax.xaxis.label.set_fontsize(FONT_SIZE)
    ax.yaxis.label.set_fontsize(FONT_SIZE)
    ax.title.set_fontsize(FONT_SIZE)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=SMALL_FONT_SIZE,
        width=SPINE_WIDTH,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
        gridline.set_linewidth(GRID_WIDTH)
    for text in ax.texts:
        text.set_fontsize(FONT_SIZE)


def _set_line_role(line: Line2D, role: str) -> None:
    """Assign a semantic line role and its globally fixed physical style."""
    widths = {
        "reference": REFERENCE_LINE_WIDTH,
        "primary": PRIMARY_LINE_WIDTH,
        "secondary": SECONDARY_LINE_WIDTH,
    }
    line.set_linewidth(widths[role])
    line.set_gid(role)
    marker = line.get_marker()
    if marker not in (None, "None", "", " "):
        line.set_markersize(MARKER_SIZE)
        point_count = len(line.get_xdata())
        line.set_markevery(max(1, point_count // 50))


def _normalize_configuration_lines(axes: tuple[Axes, ...]) -> None:
    for ax in axes:
        for line in ax.lines:
            label = line.get_label()
            if label == "Desired":
                _set_line_role(line, "reference")
            elif label and not label.startswith("_"):
                _set_line_role(line, "primary")


def _legend(
    fig: Figure,
    handles: list,
    labels: list[str],
    subplot_spec,
    *,
    ncol: int,
    loc: str = "center",
    y_offset: float = 0.0,
) -> mpl.legend.Legend:
    """Place a consistently styled legend inside an otherwise empty grid band."""
    box = subplot_spec.get_position(fig)
    legend = fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(box.x0, box.y0 + y_offset, box.width, box.height),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        fontsize=SMALL_FONT_SIZE,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.7)
    return legend


def _panel_label(fig: Figure, label: str, subplot_spec) -> mpl.text.Text:
    """Add one external group label aligned to a group grid region."""
    box = subplot_spec.get_position(fig)
    return fig.text(
        box.x0 - 0.055,
        box.y1,
        label,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _align_y_label_column(axes: tuple[Axes, ...], x_coordinate: float) -> None:
    """Center y-label glyph boxes on one fixed column coordinate."""
    for axis in axes:
        axis.yaxis.set_label_coords(x_coordinate, 0.5)
        axis.yaxis.label.set_verticalalignment("center")


def _plot_configuration_group(
    fig: Figure,
    subplot_spec,
    run,
) -> tuple[Axes, Axes, Axes]:
    group = _combined_configuration_group(run)
    grid = subplot_spec.subgridspec(
        5,
        1,
        height_ratios=(0.62, 0.28, 1.0, 1.0, 1.0),
        hspace=0.24,
    )
    axes_list: list[Axes] = []
    for row in range(2, 5):
        axis = fig.add_subplot(
            grid[row, 0],
            sharex=axes_list[0] if axes_list else None,
        )
        axes_list.append(axis)
    axes = tuple(axes_list)
    configuration_plot.plot_tracking_axes(
        run,
        group,
        np.asarray(axes),
        errors=False,
    )
    axes[-1].set_xlabel(r"Time $\mathrm{[s]}$")
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)

    first_controller = group.controller_names[0]
    t1 = float(run.results[group.scenario_key][first_controller]["t"][-1])
    split = group.phase_boundary_time / t1
    phase_box = grid[1, 0].get_position(fig)
    fig.text(
        phase_box.x0 + 0.5 * split * phase_box.width,
        phase_box.y0 + 0.5 * phase_box.height,
        "Setpoint regulation",
        fontsize=FONT_SIZE,
        ha="center",
        va="center",
    )
    fig.text(
        phase_box.x0 + (split + 0.5 * (1.0 - split)) * phase_box.width,
        phase_box.y0 + 0.5 * phase_box.height,
        "Trajectory tracking",
        fontsize=FONT_SIZE,
        ha="center",
        va="center",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    _legend(fig, handles, labels, grid[0, 0], ncol=3)
    _normalize_configuration_lines(axes)
    for axis in axes:
        _style_axis(axis)
    return axes


def _plot_operational_group(
    tracking_axes: tuple[Axes, Axes],
    error_axes: tuple[Axes, Axes],
    run: PCSImpedanceRun,
    problem: PCSImpedanceProblem,
) -> None:
    position_axis, orientation_axis = tracking_axes
    position_error_axis, orientation_error_axis = error_axes
    x_pos, x_des_pos, position_error, pose_error = operational_tracking_data(
        run,
        problem,
    )
    t = np.asarray(run.t_traj)

    for component, label in enumerate(("x", "y", "z")[: x_pos.shape[1]]):
        color = COORDINATE_COLORS[component]
        actual = position_axis.plot(
            t,
            np.asarray(x_pos[:, component]),
            color=color,
            label=rf"$p_{label}$",
        )[0]
        desired = position_axis.plot(
            t,
            np.asarray(x_des_pos[:, component]),
            color=color,
            linestyle="--",
            label=rf"$p_{{{label},\mathrm{{des}}}}$",
            zorder=5,
        )[0]
        _set_line_role(actual, "primary")
        _set_line_role(desired, "reference")
        error = position_error_axis.plot(
            t,
            1e3 * np.asarray(position_error[:, component]),
            color=color,
            label=rf"$e_{{p_{label}}}$",
        )[0]
        _set_line_role(error, "primary")

    orientation_dim = min(problem.osd.n_orientation_dim, 3)
    if pose_error is None:
        raise ValueError("The composite requires pose-tracking orientation errors.")
    orientation_error = pose_error[:, : problem.osd.n_angular_velocity_dim]
    for component, label in enumerate(("x", "y", "z")[:orientation_dim]):
        color = COORDINATE_COLORS[component]
        actual = orientation_axis.plot(
            t,
            np.asarray(run.x_traj_full[:, component]),
            color=color,
            label=rf"$r_{label}$",
        )[0]
        desired = orientation_axis.plot(
            t,
            np.asarray(run.x_des_traj_full[:, component]),
            color=color,
            linestyle="--",
            label=rf"$r_{{{label},\mathrm{{des}}}}$",
            zorder=5,
        )[0]
        _set_line_role(actual, "primary")
        _set_line_role(desired, "reference")
        error = orientation_error_axis.plot(
            t,
            np.rad2deg(np.asarray(orientation_error[:, component])),
            color=color,
            label=rf"$e_{{r_{label}}}$",
        )[0]
        _set_line_role(error, "primary")

    for error_axis in error_axes:
        zero_line = error_axis.axhline(
            0.0,
            color="0.25",
            alpha=0.7,
            zorder=0,
        )
        _set_line_role(zero_line, "secondary")

    position_axis.set_ylabel(r"Pos. $\mathrm{[m]}$")
    orientation_axis.set_ylabel(r"Orient. $\mathrm{[rad]}$")
    position_error_axis.set_ylabel(r"$e_p$ $\mathrm{[mm]}$")
    orientation_error_axis.set_ylabel(r"$e_r$ $\mathrm{[deg]}$")
    orientation_axis.set_xlabel(r"Time $\mathrm{[s]}$")
    orientation_error_axis.set_xlabel(r"Time $\mathrm{[s]}$")
    position_axis.tick_params(labelbottom=False)
    position_error_axis.tick_params(labelbottom=False)
    for axis in (*tracking_axes, *error_axes):
        axis.set_xlim(float(t[0]), float(t[-1]))
        _style_axis(axis)


def load_snapshots(snapshot_dir: Path) -> list[tuple[float, Path]]:
    """Load exactly four timestamped snapshot paths in chronological order."""
    snapshots: list[tuple[float, Path]] = []
    for path in Path(snapshot_dir).glob("snapshot_t*s.png"):
        match = SNAPSHOT_PATTERN.match(path.name)
        if match is None:
            continue
        fraction = match.group("fraction")
        time_seconds = float(
            f"{int(match.group('seconds'))}.{fraction}"
        )
        snapshots.append((time_seconds, path))
    snapshots.sort(key=lambda item: item[0])
    if len(snapshots) != 4:
        raise ValueError(
            f"Expected exactly four timestamped snapshots in {snapshot_dir}, "
            f"found {len(snapshots)}."
        )
    return snapshots


def _plot_snapshots(
    axes: tuple[Axes, Axes, Axes, Axes],
    snapshots: list[tuple[float, Path]],
) -> None:
    images: list[np.ndarray] = []
    for _, path in snapshots:
        with Image.open(path) as snapshot:
            images.append(np.asarray(snapshot.convert("RGB")))

    for axis, (time_seconds, _), image in zip(
        axes,
        snapshots,
        images,
        strict=True,
    ):
        visible_mask = np.max(255 - image.astype(np.int16), axis=2) >= 8
        rows, columns = np.nonzero(visible_mask)
        if len(rows) == 0:
            raise ValueError("A snapshot contains no visible rendered content.")
        content_side = max(
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        )
        padding = int(np.ceil(0.01 * content_side))
        crop_side = min(content_side + 2 * padding, *image.shape[:2])
        center_x = 0.5 * (float(columns.min()) + float(columns.max()) + 1.0)
        center_y = 0.5 * (float(rows.min()) + float(rows.max()) + 1.0)
        left = int(
            np.clip(
                round(center_x - 0.5 * crop_side),
                0,
                image.shape[1] - crop_side,
            )
        )
        top = int(
            np.clip(
                round(center_y - 0.5 * crop_side),
                0,
                image.shape[0] - crop_side,
            )
        )
        cropped = image[top : top + crop_side, left : left + crop_side]
        axis.imshow(cropped, interpolation="lanczos")
        axis.set_axis_off()
        axis.text(
            0.5,
            -0.05,
            rf"$t = {time_seconds:.2f}\,\mathrm{{s}}$",
            transform=axis.transAxes,
            color="0.45",
            fontsize=SMALL_FONT_SIZE,
            ha="center",
            va="top",
            clip_on=False,
        )


def _operational_legend_handles() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="none",
            markersize=MARKER_SIZE,
        )
        for color in COORDINATE_COLORS
    ]
    handles.extend(
        (
            Line2D([], [], color="black", linewidth=PRIMARY_LINE_WIDTH),
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=REFERENCE_LINE_WIDTH,
            ),
        )
    )
    return handles, [r"$x$", r"$y$", r"$z$", "Actual", "Desired"]


def build_composite_figure(
    configuration_input: Path = DEFAULT_CONFIGURATION_INPUT,
    operational_input: Path = DEFAULT_OPERATIONAL_INPUT,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> tuple[Figure, CompositeAxes]:
    """Build the final-size hybrid composite from canonical saved data."""
    configuration_run = configuration_plot.load_comparison_npz(configuration_input)
    operational_run = load_run_npz(operational_input)
    operational_problem = build_problem(operational_run.trajectory_config)
    snapshots = load_snapshots(snapshot_dir)

    with plt.style.context(PAPER_STYLE), mpl.rc_context(COMPOSITE_RC_PARAMS):
        fig = plt.figure(figsize=cm_to_inches(FIGURE_SIZE_CM), facecolor="white")
        outer = fig.add_gridspec(
            3,
            1,
            left=0.10,
            right=0.975,
            bottom=0.019,
            top=0.985,
            height_ratios=(6.15, 1.05, 7.35),
            hspace=0.0,
        )

        configuration_axes = _plot_configuration_group(
            fig,
            outer[0, 0],
            configuration_run,
        )

        operational_grid = outer[2, 0].subgridspec(
            7,
            2,
            height_ratios=(0.42, 1.0, 0.12, 1.0, 0.82, 1.90, 0.18),
            hspace=0.05,
            wspace=0.24,
        )
        position_axis = fig.add_subplot(operational_grid[1, 0])
        position_error_axis = fig.add_subplot(
            operational_grid[1, 1],
            sharex=position_axis,
        )
        orientation_axis = fig.add_subplot(
            operational_grid[3, 0],
            sharex=position_axis,
        )
        orientation_error_axis = fig.add_subplot(
            operational_grid[3, 1],
            sharex=position_axis,
        )
        operational_axes = (position_axis, orientation_axis)
        operational_error_axes = (position_error_axis, orientation_error_axis)
        _plot_operational_group(
            operational_axes,
            operational_error_axes,
            operational_run,
            operational_problem,
        )
        operational_handles, operational_labels = _operational_legend_handles()
        _legend(
            fig,
            operational_handles,
            operational_labels,
            operational_grid[0, :],
            ncol=5,
        )

        snapshot_grid = operational_grid[5, :].subgridspec(
            1,
            4,
            wspace=0.06,
        )
        snapshot_axes = tuple(
            fig.add_subplot(snapshot_grid[0, column])
            for column in range(4)
        )
        for snapshot_axis in snapshot_axes:
            snapshot_axis.set_anchor("C")
        _plot_snapshots(snapshot_axes, snapshots)

        _panel_label(fig, "A", outer[0, 0])
        _panel_label(fig, "B", outer[2, 0])
        _align_y_label_column(configuration_axes, -0.069)
        _align_y_label_column(operational_axes, -0.161)
        _align_y_label_column(operational_error_axes, -0.129)

        axes = CompositeAxes(
            configuration=configuration_axes,
            operational=operational_axes,
            operational_errors=operational_error_axes,
            snapshots=snapshot_axes,
        )
        return fig, axes


def output_paths(output_base: Path) -> tuple[Path, Path]:
    """Return the PDF and SVG paths for one suffix-free output base."""
    output_base = Path(output_base)
    return output_base.with_suffix(".pdf"), output_base.with_suffix(".svg")


def save_composite_figure(
    fig: Figure,
    output_base: Path,
    *,
    force: bool,
) -> tuple[Path, Path]:
    """Save both vector outputs without changing the declared canvas size."""
    pdf_output, svg_output = output_paths(output_base)
    existing = [path for path in (pdf_output, svg_output) if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Composite outputs already exist. Pass --force to replace them:\n"
            f"{formatted}"
        )
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(PAPER_STYLE), mpl.rc_context(COMPOSITE_RC_PARAMS):
        fig.savefig(pdf_output, dpi=300, facecolor="white", bbox_inches=None)
        fig.savefig(svg_output, dpi=300, facecolor="white", bbox_inches=None)
    return pdf_output, svg_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration-input",
        type=Path,
        default=DEFAULT_CONFIGURATION_INPUT,
    )
    parser.add_argument(
        "--operational-input",
        type=Path,
        default=DEFAULT_OPERATIONAL_INPUT,
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the expected PDF and SVG outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = parse_args(argv)
    fig, _ = build_composite_figure(
        configuration_input=args.configuration_input,
        operational_input=args.operational_input,
        snapshot_dir=args.snapshot_dir,
    )
    try:
        outputs = save_composite_figure(fig, args.output_base, force=args.force)
    finally:
        plt.close(fig)
    for output in outputs:
        print(f"Figure written to {output}")
    return outputs


if __name__ == "__main__":
    main()
