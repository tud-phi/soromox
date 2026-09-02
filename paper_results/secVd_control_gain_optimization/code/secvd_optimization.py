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
# ``cuda`` in JAX_PLATFORMS.
_PLATFORM_FOR_DEVICE = {"cpu": "cpu", "gpu": "cuda"}


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Backend for the optimization. 'auto' leaves the choice to JAX. "
            "Which backend is faster is a property of the machine, not of this "
            "case; see the README before overriding."
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


def requested_device_from_argv(argv: Sequence[str] | None = None) -> str:
    """Read only the early device option, leaving all other arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    _add_device_argument(parser)
    args, _unknown = parser.parse_known_args(argv)
    return str(args.device)


def jax_platforms_for(device: str) -> str:
    """Translate a user-facing device name into a ``JAX_PLATFORMS`` value."""
    if device not in _PLATFORM_FOR_DEVICE:
        raise ValueError(f"Unknown optimization device {device!r}")
    return _PLATFORM_FOR_DEVICE[device]


def configure_optimization_device(argv: Sequence[str] | None = None) -> str:
    """Pin the JAX platform, if asked, before an entrypoint imports JAX.

    ``auto`` sets nothing and lets JAX apply its own backend order. Which
    backend runs a Section Vd rollout faster depends on the machine -- on the
    relative FP64 throughput of its CPU and GPU, and on how the GPU handles a
    long chain of small sequential kernels -- so this case measures nothing and
    prescribes nothing. An explicit ``--device`` is strict: it pins that
    platform and fails rather than quietly running somewhere else.
    """
    requested = requested_device_from_argv(argv)
    if requested != "auto":
        os.environ["JAX_PLATFORMS"] = jax_platforms_for(requested)
    return requested


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
    # An explicit --device is a pin, not a preference: silently running on
    # another backend would make a timing or reproduction claim about the wrong
    # hardware. 'auto' asserts nothing, so whatever JAX picked is correct.
    if args.device != "auto" and actual_device != args.device:
        raise RuntimeError(
            f"JAX initialized on {actual_device!r} but --device asked for "
            f"{args.device!r}. Device selection must run before JAX is imported."
        )
    args.resolved_device = actual_device
    print(
        f"Optimization device: {actual_device} "
        f"(batch size {args.optimization_batch_size}, requested {args.device})"
    )
    return result_dir
