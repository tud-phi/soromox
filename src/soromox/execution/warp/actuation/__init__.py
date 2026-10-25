"""Shared Warp-native actuation contracts and kernels."""

from soromox.execution.warp.actuation.threadlike import (
    LinearThreadlikeActuationData,
    supports_linear_threadlike_force,
    supports_linear_threadlike_matrix,
)

__all__ = [
    "LinearThreadlikeActuationData",
    "StraightSpanPipeline",
    "supports_linear_threadlike_force",
    "supports_linear_threadlike_matrix",
]


def __getattr__(name):
    """Load native free-space actuation only when explicitly requested."""
    if name == "StraightSpanPipeline":
        from .straight_span_actuation import StraightSpanPipeline

        return StraightSpanPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
