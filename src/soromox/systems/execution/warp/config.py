"""Internal launch configuration shared by Warp dynamics executors."""

from __future__ import annotations

from soromox.systems.execution.config import default_gvs_block_dim


def gvs_block_dim(num_dofs: int, *, gpu: bool) -> int:
    """Return the retained shape-generic GVS launch configuration.

    Args:
        num_dofs: Number of active generalized coordinates.
        gpu: Whether execution targets a GPU. CPU execution uses one lane.

    Returns:
        One lane for CPU execution, 128 threads for at most 64 active
        coordinates, and 192 threads for larger systems.
    """

    if not gpu:
        return 1
    return default_gvs_block_dim(num_dofs)


__all__ = [
    "gvs_block_dim",
]
