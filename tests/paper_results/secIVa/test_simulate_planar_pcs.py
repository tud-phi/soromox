import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secIVa_benchmarking_sequential_cpu"
    / "code"
    / "soromox"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import simulate_planar_pcs  # noqa: E402


def test_planar_position_uses_both_translational_pose_columns():
    poses = np.array(
        [
            [0.1, 1.0, 2.0],
            [0.2, 3.0, 4.0],
        ]
    )

    positions = simulate_planar_pcs._planar_position_from_pose(poses)

    np.testing.assert_array_equal(positions, poses[:, 1:3])


def test_planar_position_rejects_non_planar_pose_shapes():
    with pytest.raises(ValueError, match="Expected planar poses"):
        simulate_planar_pcs._planar_position_from_pose(np.zeros((2, 4)))
