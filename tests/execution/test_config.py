"""Tests for public execution-backend configuration."""

from __future__ import annotations

import pytest

from soromox.execution import (
    DEFAULT_GVS_BLOCK_DIM,
    DEFAULT_GVS_LARGE_BLOCK_DIM,
    DEFAULT_PCS_BLOCK_DIM,
    DEFAULT_PLANAR_PCS_BLOCK_DIM,
    GVSBackendParams,
    PCSBackendParams,
    default_gvs_block_dim,
)


@pytest.mark.parametrize("value", [32, 64, 128, 192, 1024])
def test_pcs_backend_params_accept_cuda_warp_multiples(value: int) -> None:
    """Accept every legal override from one warp to one CUDA block."""

    assert PCSBackendParams(warp_block_dim=value).warp_block_dim == value


@pytest.mark.parametrize("value", [True, 32.0, "128", None])
def test_pcs_backend_params_reject_non_integer_values(value: object) -> None:
    """Avoid silently converting ambiguous launch configuration values."""

    with pytest.raises(TypeError, match="integer multiple of 32"):
        PCSBackendParams(warp_block_dim=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-32, 0, 1, 31, 33, 1056])
def test_pcs_backend_params_reject_invalid_integer_values(value: int) -> None:
    """Enforce CUDA's block limit and full-warp granularity."""

    with pytest.raises(ValueError, match="multiple of 32 between 32 and 1024"):
        PCSBackendParams(warp_block_dim=value)


def test_pcs_defaults_do_not_depend_on_runtime_gpu_identity() -> None:
    """Keep defaults deterministic and leave tuning to explicit parameters."""

    assert DEFAULT_PLANAR_PCS_BLOCK_DIM == 128
    assert DEFAULT_PCS_BLOCK_DIM == 192


@pytest.mark.parametrize("value", [32, 64, 128, 192, 1024])
def test_gvs_backend_params_accept_cuda_warp_multiples(value: int) -> None:
    """Apply the same validated CUDA launch contract to GVS."""

    assert GVSBackendParams(warp_block_dim=value).warp_block_dim == value


@pytest.mark.parametrize("value", [True, 32.0, "128", None])
def test_gvs_backend_params_reject_non_integer_values(value: object) -> None:
    """Reject ambiguous GVS block-size conversions."""

    with pytest.raises(TypeError, match="integer multiple of 32"):
        GVSBackendParams(warp_block_dim=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-32, 0, 1, 31, 33, 1056])
def test_gvs_backend_params_reject_invalid_integer_values(value: int) -> None:
    """Keep GVS launches within CUDA's full-warp block limits."""

    with pytest.raises(ValueError, match="multiple of 32 between 32 and 1024"):
        GVSBackendParams(warp_block_dim=value)


def test_gvs_defaults_preserve_the_bounded_shape_rule() -> None:
    """Select deterministic defaults from the active-coordinate count."""

    assert DEFAULT_GVS_BLOCK_DIM == 128
    assert DEFAULT_GVS_LARGE_BLOCK_DIM == 192
    assert default_gvs_block_dim(1) == 128
    assert default_gvs_block_dim(64) == 128
    assert default_gvs_block_dim(65) == 192
