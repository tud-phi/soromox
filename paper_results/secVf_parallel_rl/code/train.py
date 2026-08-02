"""Train PPO on the released parallel SoRoMoX environment.

The training entry point saves the final PPO model and SB3 VecNormalize state.
It can also periodically save checkpoints and reward CSV records under the
Section Vf data directory.
"""

import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
DATA_DIR = CASE_DIR / "data"

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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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


class RewardCheckpointCallback(BaseCallback):
    """Save periodic PPO checkpoints, VecNormalize states, and reward records."""

    def __init__(
        self,
        *,
        env: VecNormalize,
        checkpoint_dir: Path,
        checkpoint_freq: int,
        reward_csv_path: Path,
        reward_log_freq: int,
        reward_window: int,
    ) -> None:
        super().__init__()
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.reward_csv_path = reward_csv_path
        self.reward_log_freq = reward_log_freq
        self.reward_window = reward_window
        self.episode_returns: list[float] = []
        self.completed_returns: list[float] = []
        self.start_time = time.perf_counter()
        self.last_checkpoint_step = 0
        self.last_reward_step = 0

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reward_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_returns = [0.0 for _ in range(self.training_env.num_envs)]
        with self.reward_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wall_time", "episode_reward_mean"])
            writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self._last_original_rewards()
        dones = self.locals.get("dones")
        if rewards is not None and dones is not None:
            for idx, (reward, done) in enumerate(zip(rewards, dones)):
                self.episode_returns[idx] += float(reward)
                if bool(done):
                    self.completed_returns.append(self.episode_returns[idx])
                    self.episode_returns[idx] = 0.0

        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq
        ):
            self._save_checkpoint(self.num_timesteps)
            self.last_checkpoint_step = self.num_timesteps

        if (
            self.reward_log_freq > 0
            and self.num_timesteps - self.last_reward_step >= self.reward_log_freq
        ):
            self._write_reward_record()
            self.last_reward_step = self.num_timesteps

        return True

    def _last_original_rewards(self) -> list[float] | None:
        getter = getattr(self.env, "get_original_reward", None)
        rewards = getter() if getter is not None else self.locals.get("rewards")
        if rewards is None:
            return None
        return [float(value) for value in rewards]

    def _save_checkpoint(self, step: int) -> None:
        model_path = self.checkpoint_dir / f"ppo_model_step_{step}"
        vecnormalize_path = self.checkpoint_dir / f"env_vecnormalize_step_{step}.pkl"
        self.model.save(str(model_path))
        self.env.save(str(vecnormalize_path))
        print(f"Saved checkpoint at {step} timesteps: {model_path}.zip")

    def _write_reward_record(self) -> None:
        if not self.completed_returns:
            return
        window = self.completed_returns[-self.reward_window :]
        reward_mean = sum(window) / len(window)
        wall_time = time.perf_counter() - self.start_time
        with self.reward_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wall_time", "episode_reward_mean"])
            writer.writerow(
                {
                    "wall_time": f"{wall_time:.12g}",
                    "episode_reward_mean": f"{reward_mean:.12g}",
                }
            )
        print(
            f"Reward record at {self.num_timesteps} timesteps: "
            f"wall_time={wall_time:.3f}s, episode_reward_mean={reward_mean:.6g}"
        )


def save_final_artifacts(
    model: PPO, env: VecNormalize, args: argparse.Namespace
) -> None:
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

    model_filename = (
        args.model_name if args.model_name.endswith(".zip") else f"{args.model_name}.zip"
    )
    model_path = Path(args.save_dir) / model_filename
    vecnormalize_path = Path(args.save_dir) / args.vecnormalize_name
    reward_csv_path = Path(args.reward_csv)
    existing_outputs = [
        path for path in (model_path, vecnormalize_path, reward_csv_path) if path.exists()
    ]
    if existing_outputs and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing_outputs)
            + "; pass --force"
        )

    env = build_env(args)
    model = build_model(env, args)
    callback = RewardCheckpointCallback(
        env=env,
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_freq=args.checkpoint_freq,
        reward_csv_path=Path(args.reward_csv),
        reward_log_freq=args.reward_log_freq,
        reward_window=args.reward_window,
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
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
        default=str(DATA_DIR / "checkpoints"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DATA_DIR / "checkpoints"),
        help="Directory for periodic checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=positive_int,
        default=100_000,
        help="Save a model and VecNormalize checkpoint every N timesteps.",
    )
    parser.add_argument("--model-name", type=str, default="ppo_model")
    parser.add_argument(
        "--vecnormalize-name",
        type=str,
        default="env_vecnormalize.pkl",
    )
    parser.add_argument(
        "--reward-csv",
        type=str,
        default=str(DATA_DIR / "reward_logs" / "training_reward.csv"),
        help="CSV file created for wall_time and episode_reward_mean records.",
    )
    parser.add_argument(
        "--reward-log-freq",
        type=positive_int,
        default=10_000,
        help="Append one reward CSV record every N timesteps when episodes have completed.",
    )
    parser.add_argument(
        "--reward-window",
        type=positive_int,
        default=100,
        help="Number of completed episodes used for episode_reward_mean.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line training entry point."""

    train(parse_args())


if __name__ == "__main__":
    main()
