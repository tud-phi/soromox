"""Shared fixtures for configuration-dependent routing backends."""

import pytest


@pytest.fixture(params=["jax", "warp"])
def straight_backend(request):
    """Select a straight routing backend, skipping an absent Warp dependency.

    Args:
        request: Pytest fixture request containing the backend name.

    Returns:
        The explicit routing backend name.
    """
    if request.param == "warp":
        pytest.importorskip("warp")
    return request.param
