"""Shared visual constants for paper-result plots and renderings."""

from pathlib import Path
from typing import Final

PAPER_STYLE_PATH: Final = Path(__file__).with_name("paper.mplstyle")

PAPER_COLORS: Final = {
    "pre_opt_1": "#006BA6",
    "pre_opt_2": "#0496FF",
    "post_opt_1": "#f1552e",
    "post_opt_2": "#D81159",
    "post_opt_3": "#8F2D56",
    "obstacle": "#7B2CBF",
    "target": "#0ead69",
    "ground_truth": "#FFBC42",
    "x_t": "#2a9d8f",
    "y_t": "#e9c46a",
    "z_t": "#D81159",
}
