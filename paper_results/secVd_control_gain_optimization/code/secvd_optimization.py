"""Shared command-line handling for the two Section Vd optimizers."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from secvd_init import (
    INIT_SCHEMES,
    SECVD_BATCH_SIZE,
    SECVD_INIT_SCHEME,
    SECVD_INIT_SEED,
    SECVD_INIT_SPREAD,
)
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
            "Optimization device. 'auto' selects the CPU at any batch size: the "
            "rollout is sequential, so the GPU is launch-latency bound and "
            "measured over twenty times slower."
        ),
    )


def _add_batch_size_argument(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default,
        help=(
            "Number of independently optimized gain initializations. The "
            "published Section Vd results used six."
        ),
    )


def batch_size_from_argv(default: int, argv: Sequence[str] | None = None) -> int:
    """Read only the early batch-size option, leaving all other arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_batch_size_argument(parser, default)
    args, _unknown = parser.parse_known_args(argv)
    return int(args.batch_size)


def requested_device_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read only the early device option, leaving all other arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_device_argument(parser)
    args, _unknown = parser.parse_known_args(argv)
    return str(args.device)


def resolve_optimization_device(*, requested: str, batch_size: int) -> str:
    """Resolve the backend for a Section Vd optimization run.

    ``auto`` selects the **CPU regardless of batch size**. A Section Vd rollout is
    50000 strictly sequential solver steps, so the GPU is bound by kernel-launch
    latency rather than throughput, and batching adds parallel width without
    reducing that sequential depth. Measured on this case (AMD Threadripper PRO
    5975WX, RTX 4070 Ti SUPER, 100-iteration settings, three iterations timed):

    ==========  =================  ==================
    device      compile + iter 0   steady iteration
    ==========  =================  ==================
    CPU, B=1    51.5 s             40.5 s
    CPU, B=6    132.8 s            120.7 s
    GPU, B=1    did not finish a single iteration in 950 s
    ==========  =================  ==================

    The earlier rule sent ``B > 1`` to the GPU on the assumption that a batched
    optimization is throughput-bound. It is not, and that choice made the run
    more than twenty times slower. ``--device gpu`` remains available for
    settings with far fewer solver steps, where the trade-off reverses.

    Args:
        requested: ``"auto"``, ``"cpu"`` or ``"gpu"``.
        batch_size: Number of independently optimized starts.

    Returns:
        ``"cpu"`` or ``"gpu"``.

    Raises:
        ValueError: If ``requested`` is unknown or ``batch_size`` is below one.
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown optimization device {requested!r}; expected {DEVICE_CHOICES}"
        )
    if batch_size < 1:
        raise ValueError("optimization batch_size must be at least 1")
    if requested != "auto":
        return requested
    return "cpu"


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
        batch_size=batch_size_from_argv(batch_size, argv),
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
    optimization_batch_size: int = SECVD_BATCH_SIZE,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--result-dir", type=Path, default=default_result_dir)
    parser.add_argument("--num-iters", type=int, default=100)
    _add_batch_size_argument(parser, optimization_batch_size)
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    args.optimization_batch_size = args.batch_size
    args.resolved_device = resolve_optimization_device(
        requested=args.device,
        batch_size=args.batch_size,
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
