"""Shared command-line handling for the two Section Vd optimizers."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from secvd_results import RESULTS_FILENAME

DEVICE_CHOICES = ("auto", "cpu", "gpu")

# ``gpu`` is the user-facing device name, but JAX names its CUDA platform
# ``cuda`` in JAX_PLATFORMS. Automatic GPU selection keeps CPU as a fallback;
# an explicit GPU request remains strict so it cannot silently run elsewhere.
_PLATFORM_FOR_DEVICE = {"cpu": "cpu", "gpu": "cuda"}


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Optimization device. 'auto' uses CPU for one optimization start "
            "and prefers GPU for batched starts, falling back to CPU."
        ),
    )


def requested_device_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read only the early device option, leaving all other arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_device_argument(parser)
    args, _unknown = parser.parse_known_args(argv)
    return str(args.device)


def resolve_optimization_device(*, requested: str, batch_size: int) -> str:
    """Resolve the backend from the number of independently optimized starts."""
    if requested not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown optimization device {requested!r}; expected {DEVICE_CHOICES}"
        )
    if batch_size < 1:
        raise ValueError("optimization batch_size must be at least 1")
    if requested != "auto":
        return requested
    return "cpu" if batch_size == 1 else "gpu"


def jax_platforms_for(device: str, *, allow_fallback: bool) -> str:
    """Translate a user-facing device into a valid ``JAX_PLATFORMS`` value."""
    if device not in _PLATFORM_FOR_DEVICE:
        raise ValueError(f"Unknown optimization device {device!r}")
    platform = _PLATFORM_FOR_DEVICE[device]
    if allow_fallback and platform != "cpu":
        return f"{platform},cpu"
    return platform


def configure_optimization_device(
    *, batch_size: int, argv: Sequence[str] | None = None
) -> str:
    """Select the JAX platform before JAX is imported by an optimizer entrypoint."""
    requested = requested_device_from_argv(argv)
    resolved = resolve_optimization_device(
        requested=requested,
        batch_size=batch_size,
    )
    os.environ["JAX_PLATFORMS"] = jax_platforms_for(
        resolved, allow_fallback=requested == "auto"
    )
    return resolved


def parse_optimization_args(
    *,
    description: str,
    default_result_dir: Path,
    include_integral_error_saturation_scale: bool = False,
    optimization_batch_size: int = 1,
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
    _add_device_argument(parser)
    args = parser.parse_args()
    args.optimization_batch_size = optimization_batch_size
    args.resolved_device = resolve_optimization_device(
        requested=args.device,
        batch_size=optimization_batch_size,
    )
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
            f"Could not initialize the requested {args.resolved_device!r} "
            "optimization backend. Select another backend with --device."
        ) from exc
    if actual_device != args.resolved_device:
        automatic_gpu_fallback = (
            args.device == "auto"
            and args.resolved_device == "gpu"
            and actual_device == "cpu"
        )
        if not automatic_gpu_fallback:
            raise RuntimeError(
                "JAX initialized on an unexpected backend: "
                f"expected {args.resolved_device!r}, got {actual_device!r}. Device "
                "selection must run before importing JAX."
            )
        print(
            f"[WARNING] Preferred {args.resolved_device!r} for "
            f"{args.optimization_batch_size} starts but JAX initialized on "
            f"{actual_device!r}; the run continues, more slowly."
        )
        args.resolved_device = actual_device
    print(
        f"Optimization device: {args.resolved_device} "
        f"(batch size {args.optimization_batch_size}, requested {args.device})"
    )
    return result_dir
