#!/usr/bin/env python3
"""Plot sequential CPU rollout comparisons with matplotlib."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import colors
from matplotlib.lines import Line2D

Coordinate = Literal["x", "y", "z"]

SCRIPT_DIR = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "outputs"
PAPER_RESULTS_DIR = SCRIPT_DIR.parents[2]
if str(PAPER_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE

COORDINATES: tuple[Coordinate, ...] = ("x", "y", "z")
COORDINATE_COLORS: dict[Coordinate, str] = {
    "x": COLORS["x_t"],
    "y": COLORS["y_t"],
    "z": COLORS["z_t"],
}
cmap_x = colors.LinearSegmentedColormap.from_list("grad_x", ["#000000", COLORS["x_t"]])
cmap_y = colors.LinearSegmentedColormap.from_list("grad_y", ["#000000", COLORS["y_t"]])
cmap_z = colors.LinearSegmentedColormap.from_list("grad_z", ["#000000", COLORS["z_t"]])
grad_x = cmap_x(np.array([0.4, 0.7, 1.0]))[:, :3]
grad_y = cmap_y(np.array([0.4, 0.7, 1.0]))[:, :3]
grad_z = cmap_z(np.array([0.4, 0.7, 1.0]))[:, :3]
COLORS_CASES = {
    "SoRoSim_x": grad_x[2],
    "SoRoSim_y": grad_y[2],
    "SoRoSim_z": grad_z[2],
    "SoRoMoX_x": grad_x[1],
    "SoRoMoX_y": grad_y[1],
    "SoRoMoX_z": grad_z[1],
    "PyElastica_x": grad_x[0],
    "PyElastica_y": grad_y[0],
    "PyElastica_z": grad_z[0],
}
WIDTHS = {
    "SoRoSim": 2.5,
    "SoRoMoX": 1.5,
    "PyElastica": 0.75,
}
AXIS = {
    "x_axis_label": r"$x$ $\mathrm{[m]}$",
    "y_axis_label": r"$y$ $\mathrm{[m]}$",
    "z_axis_label": r"$z$ $\mathrm{[m]}$",
}
BACKEND_LINESTYLES = {
    "SoRoSim": "-",
    "PyElastica": "--",
    "SoRoMoX": ":",
}


@dataclass(frozen=True)
class CaseSpec:
    key: str
    output_stem: str
    sorosim_file: str
    pyelastica_file: str
    soromox_file: str


@dataclass(frozen=True)
class RolloutData:
    backend: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


CASES: dict[str, CaseSpec] = {
    "planar-pcs": CaseSpec(
        key="planar-pcs",
        output_stem="rollout_planar_pcs",
        sorosim_file="end_effector_position_planar_pcs_sorosim.mat",
        pyelastica_file="end_effector_position_planar_pyelastica.xlsx",
        soromox_file="end_effector_position_planar_pcs_soromox.csv",
    ),
    "spatial-pcs": CaseSpec(
        key="spatial-pcs",
        output_stem="rollout_spatial_pcs",
        sorosim_file="end_effector_position_spatial_pcs_sorosim.mat",
        pyelastica_file="end_effector_position_spatial_pyelastica.xlsx",
        soromox_file="end_effector_position_spatial_pcs_soromox.csv",
    ),
    "complex-gvs": CaseSpec(
        key="complex-gvs",
        output_stem="rollout_complex_gvs",
        sorosim_file="end_effector_position_complex_gvs_sorosim.mat",
        pyelastica_file="end_effector_position_spatial_pyelastica.xlsx",
        soromox_file="end_effector_position_complex_gvs_soromox.csv",
    ),
    "tendon-driven-gvs": CaseSpec(
        key="tendon-driven-gvs",
        output_stem="rollout_tendon_driven_gvs",
        sorosim_file="end_effector_position_tendon_driven_gvs_sorosim.mat",
        pyelastica_file="end_effector_position_tendon_driven_pyelastica.xlsx",
        soromox_file="end_effector_position_tendon_driven_gvs_soromox.csv",
    ),
}


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def _as_1d_array(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def load_sorosim(path: Path) -> RolloutData:
    mat = sio.loadmat(path)
    try:
        t = _as_1d_array(mat["t"])
        pos = np.asarray(mat["Pos_EE_all"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"{path} must contain 't' and 'Pos_EE_all'") from exc

    if pos.shape[0] != 3:
        raise ValueError(f"{path} has unexpected Pos_EE_all shape {pos.shape}")

    return RolloutData(
        backend="SoRoSim",
        t=t,
        x=_as_1d_array(pos[0, :]),
        y=_as_1d_array(pos[1, :]),
        z=_as_1d_array(pos[2, :]),
    )


def load_table(path: Path, *, backend: str, time_column: str) -> RolloutData:
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        table = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")

    missing = {time_column, "x", "z"} - set(table.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    t = _as_1d_array(table[time_column])
    x = _as_1d_array(table["x"])
    y = _as_1d_array(table["y"]) if "y" in table.columns else np.zeros_like(t)
    z = _as_1d_array(table["z"])
    return RolloutData(backend=backend, t=t, x=x, y=y, z=z)


def load_case(data_dir: Path, case: CaseSpec) -> list[RolloutData]:
    return [
        load_sorosim(data_dir / case.sorosim_file),
        load_table(
            data_dir / case.pyelastica_file,
            backend="PyElastica",
            time_column="time",
        ),
        load_table(data_dir / case.soromox_file, backend="SoRoMoX", time_column="t"),
    ]


def cm2inch(*tupl: float) -> tuple[float, ...]:
    """Convert centimeters to inches for Matplotlib figsize."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


def plot_case(
    # case: CaseSpec,
    rollouts: list[RolloutData],
    figsize_cm: tuple[float, float] = (15.0, 8.0),
    # Dictionaries to pass custom properties per plot curve
    custom_colors: dict[str, str] | None = None,
    custom_linewidths: dict[str, float] | None = None,
    # Controlled properties strictly for the legend box
    legend_line_color: str = "black",
    legend_lw: float = 1.5,
    legend_markersize: float = 5.0,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=cm2inch(figsize_cm), constrained_layout=True)

    # Fallback to defaults if no custom dictionaries are passed
    colors = custom_colors or {}
    linewidths = custom_linewidths or {}

    for rollout in rollouts:
        linestyle = BACKEND_LINESTYLES[rollout.backend]

        for coordinate in COORDINATES:
            # Generate a unique key for each unique line/plot combination
            plot_key = f"{rollout.backend}_{coordinate}"

            # Determine color: check custom_colors first, fallback to standard coordinate color
            line_color = colors.get(plot_key, COORDINATE_COLORS[coordinate])

            # Determine linewidth: check custom_linewidths first, fallback to a standard default
            line_lw = linewidths.get(plot_key, linewidths.get(rollout.backend, 1.8))

            # Draw the line using the highly customized parameters
            ax.plot(
                rollout.t,
                getattr(rollout, coordinate),
                color=line_color,
                linestyle=linestyle,
                linewidth=line_lw,
            )

    ax.set_xlim(round(min(rollout.t)), round(max(rollout.t)))
    ax.set_ylim(-0.6, 0.8)
    ax.set_xlabel(r"Time $\mathrm{[s]}$")
    ax.set_ylabel(r"Tip coordinates $\mathrm{[m]}$")
    # ax.margins(x=0.01)

    # --- THE LEGEND DECOUPLING ---
    # We construct the legend manually. No matter what wild colors or widths
    # are used on the plot above, the legend lines will always use `legend_line_color`
    legend_elements = [
        # Linestyles use one uniform color (e.g., black) and one uniform linewidth
        Line2D(
            [0],
            [0],
            color=legend_line_color,
            linestyle="-",
            lw=legend_lw,
            label="SoRoSim",
        ),
        Line2D(
            [0],
            [0],
            color=legend_line_color,
            linestyle="--",
            lw=legend_lw,
            label="PyElastica",
        ),
        Line2D(
            [0],
            [0],
            color=legend_line_color,
            # linestyle="-.",
            linestyle=(0, (1, 1.5)),
            lw=legend_lw,
            label="SoRoMoX",
        ),
        # Coordinate circles retain their fixed category reference colors
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=legend_markersize,
            markerfacecolor=COLORS["x_t"],
            markeredgecolor=COLORS["x_t"],
            label=r"$x$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=legend_markersize,
            markerfacecolor=COLORS["y_t"],
            markeredgecolor=COLORS["y_t"],
            label=r"$y$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=legend_markersize,
            markerfacecolor=COLORS["z_t"],
            markeredgecolor=COLORS["z_t"],
            label=r"$z$",
        ),
    ]

    # Render inside the figure on a single row
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=6,
        columnspacing=1.2,
        handlelength=2.0,
    )

    return fig


def save_case(
    data_dir: Path,
    output_dir: Path,
    case: CaseSpec,
    output_format: str,
    show: bool,
) -> Path:
    rollouts = load_case(data_dir, case)
    fig = plot_case(
        figsize_cm=(13.0, 6.5),
        rollouts=rollouts,
        custom_colors=COLORS_CASES,
        custom_linewidths=WIDTHS,
        legend_line_color="black",
        legend_lw=1.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{case.output_stem}.{output_format}"
    fig.savefig(output, dpi=300)

    if show:
        plt.show()
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sequential CPU benchmark rollout comparisons."
    )
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Rollout case to plot.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing SoRoSim, PyElastica, and SoRoMoX data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure files are written.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def main() -> int:
    configure_matplotlib()
    args = parse_args()
    selected_cases = CASES.values() if args.case == "all" else [CASES[args.case]]

    outputs = [
        save_case(args.data_dir, args.output_dir, case, args.format, args.show)
        for case in selected_cases
    ]

    for output in outputs:
        print(f"Saved {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
