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
BACKEND_LINESTYLES = {
    "SoRoSim": "-",
    "PyElastica": "--",
    "SoRoMoX": "-.",
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

    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "font.size": 10,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "xtick.direction": "out",
            "ytick.direction": "out",
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


def plot_case(case: CaseSpec, rollouts: list[RolloutData]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)

    for rollout in rollouts:
        linestyle = BACKEND_LINESTYLES[rollout.backend]
        for coordinate in COORDINATES:
            ax.plot(
                rollout.t,
                getattr(rollout, coordinate),
                color=COORDINATE_COLORS[coordinate],
                linestyle=linestyle,
                label=rf"${coordinate}_t$ ({rollout.backend})",
            )

    ax.set_xlabel(r"Time $\mathrm{(s)}$")
    ax.set_ylabel(r"Tip coordinates $\mathrm{(m)}$")
    ax.margins(x=0.01)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        columnspacing=1.4,
        handlelength=2.4,
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
    fig = plot_case(case, rollouts)
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
