"""Global autodiff configuration helpers.

The flag is read by public ``SoftRobot`` wrappers before dispatching to custom
JVP entry points. Jitted functions capture this Python value at trace time, so
change it before tracing or retrace after changing it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "custom_jvp_enabled",
    "custom_jvp_mode",
    "set_custom_jvp_enabled",
]

_CUSTOM_JVP_ENABLED = True


def custom_jvp_enabled() -> bool:
    """Return whether public APIs should use registered custom JVP rules."""
    return _CUSTOM_JVP_ENABLED


def set_custom_jvp_enabled(enabled: bool) -> None:
    """Set whether public APIs should use registered custom JVP rules."""
    global _CUSTOM_JVP_ENABLED
    _CUSTOM_JVP_ENABLED = bool(enabled)


@contextmanager
def custom_jvp_mode(enabled: bool) -> Iterator[None]:
    """Temporarily enable or disable custom JVP rules in public APIs."""
    previous = custom_jvp_enabled()
    set_custom_jvp_enabled(enabled)
    try:
        yield
    finally:
        set_custom_jvp_enabled(previous)
