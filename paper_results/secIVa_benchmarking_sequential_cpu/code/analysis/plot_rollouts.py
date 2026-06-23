#!/usr/bin/env python3
"""Plot sequential CPU rollout comparisons with matplotlib."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.lines import Line2D

Coordinate = Literal["x", "y", "z"]

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "outputs"

COORDINATES: tuple[Coordinate, ...] = ("x", "y", "z")
COORDINATE_COLORS: dict[Coordinate, str] = {
    "x": "#0072B2",
    "y": "#D55E00",
    "z": "#009E73",
}
COLORS = {
    "pre_opt_1": "#006BA6",
    "pre_opt_2": "#0496FF",
    "post_opt_1": "#D81159",
    "post_opt_2": "#8F2D56",
    "post_opt_3": "#FFBC42",
    "backbone_opt": "#7B2CBF",
    "backbone_init": "#0ead69",  # 0ead69, #3bceac
    "ground_truth": "#FFBC42",
    "x": "#2a9d8f",
    "y": "#e9c46a",
    "z": "#e76f51",
}
COLORS_CASES = {
    "SoRoSim_x": COLORS["x"],
    "SoRoSim_y": COLORS["y"],
    "SoRoSim_z": COLORS["z"],
    "SoRoMoX_x": "#037A6C",
    "SoRoMoX_y": "#c1921a",
    "SoRoMoX_z": "#ad2c0c",
    "PyElastica_x": "#025544",
    "PyElastica_y": "#7d5c07",
    "PyElastica_z": "#6c1a05",
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


# def configure_matplotlib() -> None:
#     """Configure a compact publication-style matplotlib theme."""

#     plt.rcParams.update(
#         {
#             "axes.grid": True,
#             "axes.labelsize": 11,
#             "axes.linewidth": 0.8,
#             "axes.spines.top": True,
#             "axes.spines.right": True,
#             "figure.dpi": 150,
#             "font.size": 10,
#             "grid.alpha": 0.35,
#             "grid.linestyle": ":",
#             "grid.linewidth": 0.7,
#             "legend.fontsize": 8.5,
#             "legend.frameon": True,
#             "lines.linewidth": 1.8,
#             "savefig.bbox": "tight",
#             "xtick.direction": "out",
#             "ytick.direction": "out",
#         }
#     )


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    # plt.rcParams.update(
    #     {
    # Adjust white padding around image
    #     "savefig.pad_inches": 0.1,
    #     "pdf.fonttype": 42,  # Force TrueType embedding inside your PDFs
    #     "ps.fonttype": 42,  # Force TrueType embedding inside EPS vector formats
    #     "svg.fonttype": "none",  # Stops converting characters to curves if exporting to SVG
    #     "text.usetex": False,  # Turn this off to prevent pdflatex path flattening
    #     "mathtext.fontset": "cm",
    #     # --- Grid Configuration ---
    #     "axes.grid": True,
    #     "grid.alpha": 0.5,  # Increased from 0.35
    #     "grid.linestyle": "-",  # Changed from ":" for better visibility
    #     "grid.linewidth": 0.8,  # Slightly thicker
    #     "grid.color": "#B0B0B0",  # Explicit darker gray
    #     # --- Font Configuration (Native LaTeX fallback) ---
    #     "font.family": "serif",
    #     "font.serif": ["CMU Serif", "Computer Modern Serif", "DejaVu Serif"],
    #     # "mathtext.fontset": "cm",
    #     "axes.formatter.use_mathtext": True,
    #     "font.size": 9,
    #     "axes.labelsize": 9,
    #     # --- Math Text Mapping ---
    #     # "mathtext.rm": "CMU Serif",
    #     # "mathtext.it": "CMU Serif:italic",
    #     # "mathtext.bf": "CMU Serif:bold",
    #     # "mathtext.sf": "CMU Serif",
    #     # --- Legend Configuration ---
    #     "legend.fontsize": 8,
    #     "legend.frameon": True,  # Toggle the box on
    #     "legend.facecolor": "white",  # Solid background to block grid/lines
    #     "legend.framealpha": 0.8,  # High opacity overlay
    #     "legend.edgecolor": "#A0A0A0",  # Distinct grey border
    #     "legend.fancybox": False,
    #     "patch.linewidth": 0.7,
    #     # --- Axes and Spines ---
    #     "axes.linewidth": 0.8,
    #     "xtick.direction": "out",
    #     "ytick.direction": "out",
    #     "savefig.bbox": "tight",
    #     "figure.dpi": 150,
    #     "lines.linewidth": 1.8,
    # }
    plt.rcParams.update(
        {
            # Adjust white padding around image
            "savefig.pad_inches": 0.1,
            # --- Grid Configuration ---
            "axes.grid": True,
            "grid.alpha": 0.5,  # Increased from 0.35
            "grid.linestyle": "-",  # Changed from ":" for better visibility
            "grid.linewidth": 0.8,  # Slightly thicker
            "grid.color": "#B0B0B0",  # Explicit darker gray
            # --- Font Configuration (Native LaTeX fallback) ---
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 9,
            "axes.labelsize": 9,
            # --- Legend Configuration ---
            "legend.fontsize": 8,
            "legend.frameon": True,  # Toggle the box on
            "legend.facecolor": "white",  # Solid background to block grid/lines
            "legend.framealpha": 0.8,  # High opacity overlay
            "legend.edgecolor": "#A0A0A0",  # Distinct grey border
            "legend.fancybox": False,
            "patch.linewidth": 0.7,
            # --- Axes and Spines ---
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "figure.dpi": 150,
            "lines.linewidth": 1.8,
        }
    )

    if shutil.which("latex") is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )


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
            markerfacecolor=COLORS["x"],
            markeredgecolor=COLORS["x"],
            label=r"$x$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=legend_markersize,
            markerfacecolor=COLORS["y"],
            markeredgecolor=COLORS["y"],
            label=r"$y$",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=legend_markersize,
            markerfacecolor=COLORS["z"],
            markeredgecolor=COLORS["z"],
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
    # fig = plot_case(case, rollouts, (15.0, 8.0))
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
    fig.savefig(output, dpi=150)

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
        default="svg",
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
