"""Native threadlike actuation contracts and optional Warp execution."""

from .fixed_routing import (
    LinearThreadlikeActuationData,
    supports_linear_threadlike_force,
    supports_linear_threadlike_matrix,
)

__all__ = [
    "LinearThreadlikeActuationData",
    "StraightSpanActuation",
    "supports_linear_threadlike_force",
    "supports_linear_threadlike_matrix",
]


def __getattr__(name):
    """Import the native executor when requested, keeping Warp optional."""
    if name == "StraightSpanActuation":
        from .straight_span import StraightSpanActuation

        return StraightSpanActuation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
