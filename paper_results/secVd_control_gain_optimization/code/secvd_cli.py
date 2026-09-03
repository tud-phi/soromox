"""Shared command-line handling for the Section Vd optimizer."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from secvd_init import (
    GAIN_ORDER,
    INIT_SCHEMES,
    SECVD_BATCH_SIZE,
    SECVD_INIT_SCHEME,
    SECVD_INIT_SEED,
    SECVD_INIT_SPREAD,
)
from secvd_results import METHODS, RESULTS_FILENAME

DEVICE_CHOICES = ("auto", "cpu", "gpu")

# ``gpu`` is the user-facing device name, but JAX names its CUDA platform
# ``cuda`` in JAX_PLATFORMS.
_PLATFORM_FOR_DEVICE = {"cpu": "cpu", "gpu": "cuda"}


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Backend for the optimization. 'auto' leaves the choice to JAX.",
    )


def _early_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the options needed before the full parser exists."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_device_argument(parser)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    args, _unknown = parser.parse_known_args(argv)
    return args


def requested_device_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read only the early device option, leaving all other arguments untouched."""
    return str(_early_args(argv).device)


def requested_method_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read only the early method option, leaving all other arguments untouched."""
    return str(_early_args(argv).method)


def jax_platforms_for(device: str) -> str:
    """Translate a user-facing device name into a ``JAX_PLATFORMS`` value."""
    if device not in _PLATFORM_FOR_DEVICE:
        raise ValueError(f"Unknown optimization device {device!r}")
    return _PLATFORM_FOR_DEVICE[device]


def configure_optimization_device(argv: Sequence[str] | None = None) -> str:
    """Pin the JAX platform, if asked, before an entrypoint imports JAX."""
    requested = requested_device_from_argv(argv)
    if requested != "auto":
        os.environ["JAX_PLATFORMS"] = jax_platforms_for(requested)
    return requested


def parse_optimization_args(
    *,
    description: str,
    default_result_dir_for: Callable[[str], Path],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the generator's arguments, with every default keyed on ``--method``.

    Args:
        description: Program description for ``--help``.
        default_result_dir_for: Maps a method to its archive directory.
        argv: Argument list, for tests. Defaults to ``sys.argv``.

    Returns:
        The parsed arguments, with ``ratio`` as a dict keyed by gain family.

    Raises:
        ValueError: If an argument is outside its supported range.
    """
    from secvd_case import COMMITTED_E_SAT, COMMITTED_LR_RATIO, TUNED_ALPHA

    method = requested_method_from_argv(argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    parser.add_argument(
        "--result-dir", type=Path, default=default_result_dir_for(method)
    )
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument(
        "--solver-dt", type=float, default=None, help="Integration step."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=TUNED_ALPHA[method],
        help="Learning-rate magnitude.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        metavar=tuple(GAIN_ORDER),
        default=[COMMITTED_LR_RATIO[name] for name in GAIN_ORDER],
        help="Per-family multipliers on the learning-rate magnitude.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SECVD_BATCH_SIZE,
        help="Number of independently optimized gain initializations.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=SECVD_INIT_SEED,
        help="PRNG seed behind the sampled gain initializations.",
    )
    parser.add_argument(
        "--init-scheme",
        choices=INIT_SCHEMES,
        default=SECVD_INIT_SCHEME,
        help="Initialization sampling scheme; recorded in the archive.",
    )
    parser.add_argument(
        "--init-spread",
        type=float,
        default=SECVD_INIT_SPREAD,
        help="Multiplicative half-width for the log-uniform scheme.",
    )
    if method == "collocated":
        parser.add_argument(
            "--e-sat",
            type=float,
            default=COMMITTED_E_SAT,
            help="Integral-error saturation scale, in metres.",
        )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Mark this archive as development-only placeholder data.",
    )
    parser.add_argument("--debug-nans", action="store_true")
    parser.add_argument("--force", action="store_true")
    _add_device_argument(parser)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not args.alpha > 0:
        raise ValueError("--alpha must be positive")
    if args.solver_dt is not None and not args.solver_dt > 0:
        raise ValueError("--solver-dt must be positive when given")
    if method == "collocated" and not (math.isfinite(args.e_sat) and args.e_sat > 0):
        raise ValueError("--e-sat must be finite and positive")
    args.ratio = dict(zip(GAIN_ORDER, args.ratio, strict=True))
    return args


def prepare_result_dir(args: argparse.Namespace) -> Path:
    if args.num_iters < 1:
        raise ValueError("--num-iters must be at least 1")
    result_dir = args.result_dir.resolve()
    path = result_dir / RESULTS_FILENAME
    if path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    import jax

    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    try:
        actual_device = jax.default_backend()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not initialize the {args.device!r} optimization backend. "
            "Select another backend with --device."
        ) from exc
    if args.device != "auto" and actual_device != args.device:
        raise RuntimeError(
            f"JAX initialized on {actual_device!r} but --device asked for "
            f"{args.device!r}. Device selection must run before JAX is imported."
        )
    args.resolved_device = actual_device
    print(
        f"Optimization device: {actual_device} "
        f"(batch size {args.batch_size}, requested {args.device})"
    )
    return result_dir
