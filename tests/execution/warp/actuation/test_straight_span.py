"""Direct launches of free-space routing with caller-owned Warp buffers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("coordinates", [0, 2])
def test_straight_span_direct_launch(device, coordinates):
    """Overwrite all outputs, share intervals, and preserve environment ordering.

    Args:
        device: Warp execution device, with CUDA tested when available.
        coordinates: Configuration dimension, including a rigid host with no DOFs.
    """
    wp = pytest.importorskip("warp")
    if device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    from soromox.execution.warp.actuation.straight_span import launch_straight_spans

    links = np.tile(np.eye(4), (2, 2, 1, 1))
    links[:, :, 0, 3] = [0.1, 0.2]
    joints = np.tile(np.eye(4), (2, 2, 1, 1))
    bases = np.tile(np.eye(4), (2, 1, 1, 1))
    bases[1, 0, 1, 3] = 0.7
    body_jacobians = np.zeros((2, 2, 6, coordinates))
    if coordinates:
        body_jacobians[:, 0, 3, 0] = 1.0
        body_jacobians[:, 1, 3, 1] = 2.0
    offsets = np.array([[0, 0, 0], [0, 0.004, 0], [0, -0.003, 0]], dtype=float)
    with wp.ScopedDevice(device):
        inputs = [
            wp.array(values, dtype=wp.float64)
            for values in (links, joints, body_jacobians, bases)
        ]
        inputs += [
            wp.array([[0, 1], [1, 1]], dtype=wp.int32),
            wp.array([0, 0, 1], dtype=wp.int32),
            wp.array(offsets, dtype=wp.float64),
            wp.array(offsets, dtype=wp.float64),
        ]
        shapes = [
            (2, 2, 4, 4),
            (2, 2, 4, 4),
            (2, 2, 4, 4),
            (2, 2, 6, coordinates),
            (2, 3, 7),
            (2, 3, coordinates),
        ]
        outputs = [wp.full(shape, np.nan, dtype=wp.float64) for shape in shapes]
        outputs += [wp.zeros((2, 3), dtype=wp.bool)]
        malformed = inputs.copy()
        malformed[3] = wp.zeros((2, 1, 3, 3), dtype=wp.float64)
        with pytest.raises(ValueError, match="bases must have shape"):
            launch_straight_spans(*malformed, *outputs)
        for _ in range(2):
            launch_straight_spans(*inputs, *outputs)
        if device == "cuda":
            with wp.ScopedCapture(device=device) as capture:
                launch_straight_spans(*inputs, *outputs)
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        geometry, jacobians, valid = [value.numpy() for value in outputs[-3:]]
    expected_start = np.broadcast_to(offsets, (2, 3, 3)).copy()
    expected_start[:, 2, 0] += 0.1
    expected_start[1, :, 1] += 0.7
    expected_end = expected_start.copy()
    expected_end[:, :, 0] += [0.3, 0.3, 0.2]
    assert_allclose(geometry[:, :, :3], expected_start, atol=1e-15)
    assert_allclose(geometry[:, :, 3:6], expected_end, atol=1e-15)
    assert_allclose(geometry[:, :, 6], [[0.3, 0.3, 0.2]] * 2, atol=1e-15)
    assert valid.all()
    if coordinates:
        assert_allclose(jacobians, [[[1, 2], [1, 2], [0, 2]]] * 2, atol=1e-15)
    else:
        assert jacobians.shape == (2, 3, 0)
