"""Shared command-line handling for the two Section Vd optimizers."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from secvd_results import RESULTS_FILENAME


def parse_optimization_args(
    *,
    description: str,
    default_result_dir: Path,
    include_integral_error_saturation_scale: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--result-dir", type=Path, default=default_result_dir)
    parser.add_argument("--num-iters", type=int, default=100)
    if include_integral_error_saturation_scale:
        parser.add_argument(
            "--integral-error-saturation-scale", type=float, default=1e-2
        )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Mark this archive as development-only placeholder data.",
    )
    parser.add_argument("--debug-nans", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_result_dir(args: argparse.Namespace) -> Path:
    if args.num_iters < 1:
        raise ValueError("--num-iters must be at least 1")
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    result_dir = args.result_dir.resolve()
    path = result_dir / RESULTS_FILENAME
    if path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    return result_dir
