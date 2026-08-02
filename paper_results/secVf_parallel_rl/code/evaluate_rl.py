"""Evaluate a trained PPO policy on the parallel SoRoMoX environment.

The evaluation entry point runs one fixed-length rollout and prints only the
last-step end-effector tracking error and success rate. It does not save NPZ
files or other intermediate values.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
DATA_DIR = CASE_DIR / "data"

import jax
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

if __package__:
    from .parallel_soromox_env import ParallelSoromoxEnv
else:
    env_path = CODE_DIR / "parallel_soromox_env.py"
    spec = importlib.util.spec_from_file_location("parallel_soromox_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ParallelSoromoxEnv from {env_path}")
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    ParallelSoromoxEnv = env_module.ParallelSoromoxEnv


DEFAULT_MODEL_PATH = str(DATA_DIR / "checkpoints" / "ppo_model.zip")
DEFAULT_VECNORMALIZE_PATH = str(DATA_DIR / "checkpoints" / "env_vecnormalize.pkl")


def to_numpy(value, dtype=None) -> np.ndarray:
    """Move a JAX or NumPy value to a NumPy array.

    Args:
        value: Array-like value.
        dtype: Optional NumPy dtype for the returned array.

    Returns:
        NumPy array on host memory.
    """

    array = np.asarray(jax.device_get(value))
    return array.astype(dtype) if dtype is not None else array


def build_eval_env(args: argparse.Namespace) -> tuple[ParallelSoromoxEnv, VecNormalize]:
    """Construct and normalize the parallel evaluation environment.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Raw parallel environment and loaded ``VecNormalize`` wrapper.
    """

    raw_env = ParallelSoromoxEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        game_time=args.game_time,
        control_fps=args.control_fps,
        length=args.arm_length,
        radius=args.arm_radius,
        ball_radius=args.ball_radius,
        ball_surface_vmax=args.ball_surface_vmax,
        success_threshold=args.success_threshold,
    )
    vec_env = VecNormalize.load(args.vecnormalize_path, raw_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return raw_env, vec_env


def run_fixed_episode(
    *,
    model: PPO,
    raw_env: ParallelSoromoxEnv,
    vec_env: VecNormalize,
    episode_length: int,
    deterministic: bool,
) -> np.ndarray:
    """Run one fixed-length parallel episode and return final-step diff.

    Args:
        model: Loaded PPO policy.
        raw_env: Raw parallel SoRoMoX environment.
        vec_env: Loaded normalization wrapper.
        episode_length: Number of action steps to execute.
        deterministic: Whether to use deterministic policy actions.

    Returns:
        Final-step diff array ``ee_pos - ball_pos`` with shape
        ``(num_envs, 3)``.
    """

    obs = raw_env.reset()
    obs = vec_env.normalize_obs(obs)
    final_diff = np.zeros((raw_env.num_envs, 3), dtype=np.float64)

    for _ in range(episode_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        (
            env_states,
            raw_obs,
            _rewards,
            _terminateds,
            _truncateds,
            ee_pos,
            _invalid,
        ) = raw_env._step_batch(raw_env._env_states, action)
        raw_env._env_states = env_states
        obs = vec_env.normalize_obs(to_numpy(raw_obs, dtype=np.float32))

        ee_pos_np = to_numpy(ee_pos, dtype=np.float64)
        ball_pos_np = to_numpy(env_states.ball_pos, dtype=np.float64)
        final_diff = ee_pos_np - ball_pos_np

    return final_diff


def print_final_metrics(
    final_diff: np.ndarray,
    success_threshold: float,
) -> None:
    """Print final-step diff and success metrics.

    Args:
        final_diff: Final-step ``ee_pos - ball_pos`` array.
        success_threshold: Distance threshold for success in meters.
    """

    final_distance = np.linalg.norm(final_diff, axis=1)
    final_success = final_distance < success_threshold

    print("Final-step evaluation metrics")
    print(f"mean_diff_xyz_m: {final_diff.mean(axis=0)}")
    print(f"mean_tracking_error_m: {final_distance.mean():.8f}")
    print(f"success_rate: {final_success.mean():.6f}")


def evaluate(args: argparse.Namespace) -> None:
    """Load a trained policy and print final-step evaluation metrics.

    Args:
        args: Parsed command-line arguments.
    """

    raw_env, vec_env = build_eval_env(args)
    model = PPO.load(args.model_path, env=vec_env)
    final_diff = run_fixed_episode(
        model=model,
        raw_env=raw_env,
        vec_env=vec_env,
        episode_length=args.episode_length,
        deterministic=not args.stochastic,
    )
    print_final_metrics(
        final_diff=final_diff,
        success_threshold=args.success_threshold,
    )
    vec_env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for policy evaluation.

    Args:
        argv: Optional explicit argument list for tests.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", default=DEFAULT_VECNORMALIZE_PATH)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=105)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--game-time", type=float, default=7.0)
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--arm-length", type=float, default=0.25)
    parser.add_argument("--arm-radius", type=float, default=0.025)
    parser.add_argument("--ball-radius", type=float, default=0.10)
    parser.add_argument("--ball-surface-vmax", type=float, default=0.015)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Run the command-line evaluation entry point."""

    evaluate(parse_args())


if __name__ == "__main__":
    main()
