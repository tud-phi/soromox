"""Plot the saved full-vs-partial feedback-linearization comparison."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).parent.parent
PAPER_STYLE = Path(__file__).parents[3] / "paper.mplstyle"
DEFAULT_DATA_INPUT = (
    CASE_DIR / "data" / "impedance_feedback_linearization_comparison.npz"
)
DEFAULT_OUTPUT_DIR = CASE_DIR / "outputs"
POSITION_STEM = "operational_space_impedance_full_vs_partial_position_tracking"
ORIENTATION_STEM = "operational_space_impedance_full_vs_partial_orientation_tracking"
CONTROLLER_MODES = ("full", "partial")
RESULT_KEYS = (
    "t",
    "position_mm",
    "position_desired_mm",
    "position_error_mm",
    "orientation_deg",
    "orientation_desired_deg",
    "orientation_error_deg",
)
LINEARIZATION_COLORS = {"full": "#0072B2", "partial": "#D55E00"}
LINEARIZATION_STYLES = {"full": "-", "partial": "--"}
COORDINATE_COLORS = ("#2a9d8f", "#e9c46a", "#D81159")


def configure_plot_style() -> None:
    """Apply the shared publication style."""
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
    return tuple(value / 2.54 for value in size_cm)


def load_results(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load and validate the comparison NPZ schema."""
    with np.load(path, allow_pickle=False) as archive:
        modes = tuple(str(mode) for mode in archive["controller_modes"])
        if modes != CONTROLLER_MODES:
            raise ValueError(
                f"Expected controller modes {CONTROLLER_MODES}, found {modes}"
            )
        return {
            mode: {key: np.asarray(archive[f"{mode}_{key}"]) for key in RESULT_KEYS}
            for mode in CONTROLLER_MODES
        }


def output_paths(output_dir: Path, output_stem: str) -> tuple[Path, Path]:
    return tuple(output_dir / f"{output_stem}.{suffix}" for suffix in ("pdf", "png"))


def ensure_writable(paths: tuple[Path, ...], *, force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_feedback_error_axis(
    ax: plt.Axes,
    results: dict[str, dict[str, np.ndarray]],
    *,
    quantity: str,
    units: str,
    show_legend: bool = True,
) -> None:
    """Plot full/partial geometric error norms into one logarithmic axis."""
    t = results["full"]["t"]
    error_key = f"{quantity}_error_{units}"
    for mode in CONTROLLER_MODES:
        ax.semilogy(
            t,
            results[mode][error_key],
            color=LINEARIZATION_COLORS[mode],
            linestyle=LINEARIZATION_STYLES[mode],
            linewidth=1.3,
            label=f"{mode.capitalize()} linearization",
        )
    ax.set_yscale("log", nonpositive="clip")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylabel(rf"Error norm $\mathrm{{[{units}]}}$")
    ax.set_xlabel(r"Time $\mathrm{[s]}$")
    if show_legend:
        ax.legend(loc="upper right", ncol=2)


def build_tracking_figure(
    results: dict[str, dict[str, np.ndarray]],
    *,
    quantity: str,
    units: str,
    component_labels: tuple[str, str, str],
) -> plt.Figure:
    """Build the standalone coordinate and error-norm comparison."""
    configure_plot_style()
    reference_key = f"{quantity}_desired_{units}"
    value_key = f"{quantity}_{units}"
    t = results["full"]["t"]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=cm_to_inches((13.0, 14.0)),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.9]},
    )
    for component, (axis, label) in enumerate(zip(axes[:3], component_labels)):
        axis.plot(
            t,
            results["full"][reference_key][:, component],
            color="black",
            linestyle=":",
            linewidth=2.0,
            label="Reference",
            zorder=5,
        )
        for mode in CONTROLLER_MODES:
            axis.plot(
                t,
                results[mode][value_key][:, component],
                color=LINEARIZATION_COLORS[mode],
                linestyle=LINEARIZATION_STYLES[mode],
                linewidth=1.2,
                label=f"{mode.capitalize()} linearization",
            )
        axis.set_ylabel(rf"{label} $\mathrm{{[{units}]}}$")
        axis.set_xlim(float(t[0]), float(t[-1]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    plot_feedback_error_axis(
        axes[3], results, quantity=quantity, units=units, show_legend=False
    )
    return fig


def plot_tracking(
    results: dict[str, dict[str, np.ndarray]],
    *,
    quantity: str,
    units: str,
    component_labels: tuple[str, str, str],
    paths: tuple[Path, Path],
    show: bool = False,
) -> None:
    """Save a standalone tracking comparison in PDF and PNG formats."""
    fig = build_tracking_figure(
        results,
        quantity=quantity,
        units=units,
        component_labels=component_labels,
    )
    for path in paths:
        fig.savefig(path, bbox_inches=None)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    position_paths = output_paths(args.output_dir, POSITION_STEM)
    orientation_paths = output_paths(args.output_dir, ORIENTATION_STEM)
    ensure_writable(position_paths + orientation_paths, force=args.force)
    results = load_results(args.input)
    plot_tracking(
        results,
        quantity="position",
        units="mm",
        component_labels=(r"$\Delta p_x$", r"$\Delta p_y$", r"$\Delta p_z$"),
        paths=position_paths,
        show=args.show,
    )
    plot_tracking(
        results,
        quantity="orientation",
        units="deg",
        component_labels=(r"$\Delta r_x$", r"$\Delta r_y$", r"$\Delta r_z$"),
        paths=orientation_paths,
        show=args.show,
    )
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
