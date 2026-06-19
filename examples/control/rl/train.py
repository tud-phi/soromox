"""Train PPO on the released parallel SoRoMoX environment.

The training entry point intentionally saves only the final PPO model and the
final SB3 VecNormalize state. It does not create intermediate checkpoints or
episode logs.
"""

import argparse
import importlib.util
import os
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent

if __package__:
    from .parallel_soromox_env import ParallelSoromoxEnv
else:
    env_path = RL_DIR / "parallel_soromox_env.py"
    spec = importlib.util.spec_from_file_location("parallel_soromox_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ParallelSoromoxEnv from {env_path}")
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    ParallelSoromoxEnv = env_module.ParallelSoromoxEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def positive_int(value: str) -> int:
    """Parse a strictly positive integer command-line argument.

    Args:
        value: Raw command-line string.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the parsed value is not positive.
    """

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_env(args: argparse.Namespace) -> VecNormalize:
    """Construct the normalized parallel training environment.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A ``VecNormalize`` wrapper around ``ParallelSoromoxEnv``.
    """

    env = ParallelSoromoxEnv(
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
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=args.clip_obs)


def build_model(env: VecNormalize, args: argparse.Namespace) -> PPO:
    """Create a PPO model for the parallel SoRoMoX environment.

    Args:
        env: Normalized vectorized environment.
        args: Parsed command-line arguments.

    Returns:
        Initialized PPO model.
    """

    return PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=args.verbose,
        seed=args.seed,
    )


def save_final_artifacts(model: PPO, env: VecNormalize, args: argparse.Namespace) -> None:
    """Save the final PPO model and VecNormalize state.

    Args:
        model: Trained PPO model.
        env: VecNormalize wrapper paired with the model.
        args: Parsed command-line arguments.
    """

    os.makedirs(args.save_dir, exist_ok=True)
    model.save(os.path.join(args.save_dir, args.model_name))
    env.save(os.path.join(args.save_dir, args.vecnormalize_name))


def train(args: argparse.Namespace) -> None:
    """Run PPO training and save only the final artifacts.

    Args:
        args: Parsed command-line arguments.
    """

    env = build_env(args)
    model = build_model(env, args)
    try:
        model.learn(total_timesteps=args.total_timesteps)
        save_final_artifacts(model, env, args)
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for parallel PPO training.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=positive_int, default=1_000_000)
    parser.add_argument("--n-steps", type=positive_int, default=200)
    parser.add_argument("--batch-size", type=positive_int, default=428)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--clip-obs", type=float, default=10.0)

    parser.add_argument("--game-time", type=float, default=7.0)
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--arm-length", type=float, default=0.25)
    parser.add_argument("--arm-radius", type=float, default=0.025)
    parser.add_argument("--ball-radius", type=float, default=0.10)
    parser.add_argument("--ball-surface-vmax", type=float, default=0.015)
    parser.add_argument("--success-threshold", type=float, default=0.01)

    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(RL_DIR / "checkpoints"),
    )
    parser.add_argument("--model-name", type=str, default="ppo_model")
    parser.add_argument(
        "--vecnormalize-name",
        type=str,
        default="env_vecnormalize.pkl",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line training entry point."""

    train(parse_args())


if __name__ == "__main__":
    main()
