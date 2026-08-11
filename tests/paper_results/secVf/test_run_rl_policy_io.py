import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from numpy.testing import assert_allclose

MODULE_DIR = (
    Path(__file__).resolve().parents[3] / "paper_results" / "secVf_parallel_rl" / "code"
)
sys.path.insert(0, str(MODULE_DIR))

import run_rl_policy  # noqa: E402


def rollout_arrays(num_envs: int = 3):
    num_samples = 4
    return {
        "ts": np.arange(num_samples, dtype=np.float64) / 15.0,
        "q_ts": np.arange(num_samples * num_envs * 6, dtype=np.float64).reshape(
            num_samples, num_envs, 6
        ),
        "ball_ts": np.zeros((num_samples, num_envs, 3), dtype=np.float64),
        "actions": np.zeros((num_samples - 1, num_envs, 4), dtype=np.float64),
        "rewards": np.zeros((num_samples - 1, num_envs), dtype=np.float64),
    }


@pytest.mark.parametrize("num_envs", [1, 3])
def test_save_rollout_npz_preserves_all_environments_and_geometry(
    tmp_path,
    num_envs,
):
    arrays = rollout_arrays(num_envs)
    output = run_rl_policy.save_rollout_npz(
        tmp_path / f"rollout-{num_envs}.npz",
        **arrays,
        arm_length=0.3,
        arm_radius=0.02,
    )

    with np.load(output) as saved:
        assert saved["q_ts"].shape == (4, num_envs, 6)
        assert saved["ball_ts"].shape == (4, num_envs, 3)
        assert saved["actions"].shape == (3, num_envs, 4)
        assert saved["rewards"].shape == (3, num_envs)
        assert_allclose(saved["q_ts"], arrays["q_ts"])
        assert float(saved["arm_length"]) == pytest.approx(0.3)
        assert float(saved["arm_radius"]) == pytest.approx(0.02)
        assert str(saved["policy"]) == "trained"


def test_save_rollout_npz_refuses_overwrite_without_force(tmp_path):
    path = tmp_path / "rollout.npz"
    arrays = rollout_arrays()
    run_rl_policy.save_rollout_npz(
        path,
        **arrays,
        arm_length=0.25,
        arm_radius=0.025,
    )

    with pytest.raises(FileExistsError, match="--force"):
        run_rl_policy.save_rollout_npz(
            path,
            **arrays,
            arm_length=0.25,
            arm_radius=0.025,
        )

    run_rl_policy.save_rollout_npz(
        path,
        **arrays,
        arm_length=0.25,
        arm_radius=0.025,
        force=True,
    )


def test_save_rollout_npz_rejects_non_batched_q_shape(tmp_path):
    arrays = rollout_arrays()
    arrays["q_ts"] = arrays["q_ts"][:, 0, :]

    with pytest.raises(ValueError, match=r"q_ts must have shape \(T,N,D\)"):
        run_rl_policy.save_rollout_npz(
            tmp_path / "invalid.npz",
            **arrays,
            arm_length=0.25,
            arm_radius=0.025,
        )


def test_run_policy_cli_has_no_rendering_options():
    with pytest.raises(SystemExit):
        run_rl_policy.parse_args(["--render"])


def test_policy_specific_defaults_write_trajectories_to_data_directory():
    trained = run_rl_policy.parse_args([])
    random = run_rl_policy.parse_args(["--policy", "random"])

    assert trained.trajectory_output == (
        run_rl_policy.TRAJECTORY_DIR / "rl_rollout_trained_1_env.npz"
    )
    assert random.trajectory_output == (
        run_rl_policy.TRAJECTORY_DIR / "rl_rollout_initialized_1_env.npz"
    )


def test_random_action_batch_is_seeded_uniform_and_batched():
    class FakeEnv:
        def __init__(self):
            self.num_envs = 3
            self.action_space = spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
                dtype=np.float32,
            )

    first_env = FakeEnv()
    second_env = FakeEnv()
    first_env.action_space.seed(7)
    second_env.action_space.seed(7)

    first = run_rl_policy.sample_random_action_batch(first_env)
    second = run_rl_policy.sample_random_action_batch(second_env)

    assert first.shape == (3, 4)
    assert first.dtype == np.float32
    assert_allclose(first, second)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)
