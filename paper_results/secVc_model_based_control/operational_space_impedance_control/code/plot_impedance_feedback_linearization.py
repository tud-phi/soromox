"""Plot the saved full-vs-partial feedback-linearization comparison."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

CASE_DIR = Path(__file__).resolve().parent.parent
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


def configure_plot_style() -> None:
    """Apply a compact publication-oriented plotting style."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 220,
        }
    )


def output_paths(output_dir: Path, output_stem: str) -> tuple[Path, Path]:
    return tuple(output_dir / f"{output_stem}.{suffix}" for suffix in ("pdf", "png"))


def ensure_writable(paths: tuple[Path, ...], *, force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def plot_tracking(
    results: dict[str, dict[str, np.ndarray]],
    *,
    quantity: str,
    units: str,
    component_labels: tuple[str, str, str],
    paths: tuple[Path, Path],
) -> None:
    """Plot three tracking coordinates and the geometric error norm."""
    reference_key = f"{quantity}_desired_{units}"
    value_key = f"{quantity}_{units}"
    error_key = f"{quantity}_error_{units}"
    t = results["full"]["t"]

    colors = {"full": "#0072B2", "partial": "#D55E00"}
    line_styles = {"full": "-", "partial": "--"}
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 8.0),
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
            zorder=3,
        )
        for mode in CONTROLLER_MODES:
            axis.plot(
                t,
                results[mode][value_key][:, component],
                color=colors[mode],
                linestyle=line_styles[mode],
                label=f"{mode.capitalize()} linearization",
            )
        axis.set_ylabel(f"{label} [{units}]")
        if component == 0:
            axis.legend(loc="upper left", ncol=3)

    error_axis = axes[3]
    for mode in CONTROLLER_MODES:
        error_axis.semilogy(
            t,
            np.maximum(results[mode][error_key], 1e-12),
            color=colors[mode],
            linestyle=line_styles[mode],
            label=f"{mode.capitalize()} linearization",
        )
    error_axis.set_ylabel(f"Error norm [{units}]")
    error_axis.set_xlabel("Time [s]")
    error_axis.legend(loc="upper left", ncol=2)
    fig.suptitle(
        f"{quantity.capitalize()} reference tracking: full vs partial feedback linearization"
    )

    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    position_paths = output_paths(args.output_dir, POSITION_STEM)
    orientation_paths = output_paths(args.output_dir, ORIENTATION_STEM)
    ensure_writable(position_paths + orientation_paths, force=args.force)

    results = load_results(args.input)
    configure_plot_style()
    plot_tracking(
        results,
        quantity="position",
        units="mm",
        component_labels=(r"$\Delta p_x$", r"$\Delta p_y$", r"$\Delta p_z$"),
        paths=position_paths,
    )
    plot_tracking(
        results,
        quantity="orientation",
        units="deg",
        component_labels=(r"$\Delta r_x$", r"$\Delta r_y$", r"$\Delta r_z$"),
        paths=orientation_paths,
    )
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
