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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

# Temporary Weights & Biases logging for the revision runs.
try:
    import wandb
except ImportError:
    wandb = None


def load_env_class() -> type:
    """Import ``ParallelSoromoxEnv`` after the JAX platform has been chosen.

    Returns:
        The ``ParallelSoromoxEnv`` class.
    """

    if __package__:
        from .parallel_soromox_env import ParallelSoromoxEnv

        return ParallelSoromoxEnv
    env_path = CODE_DIR / "parallel_soromox_env.py"
    spec = importlib.util.spec_from_file_location("parallel_soromox_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ParallelSoromoxEnv from {env_path}")
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module.ParallelSoromoxEnv


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


def select_device_backend(device: str) -> str:
    """Pin the JAX platform and return the matching SoRoMoX backend.

    CPU runs use the pure JAX backend and GPU runs use the Warp backend.
    ``"auto"`` keeps JAX's default platform and picks the backend from it.

    Args:
        device: Requested device, ``"auto"``, ``"cpu"``, or ``"gpu"``.

    Returns:
        SoRoMoX execution backend name, ``"jax"`` or ``"warp"``.
    """

    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    import jax

    return "warp" if jax.default_backend() == "gpu" else "jax"


def build_env(args: argparse.Namespace) -> VecNormalize:
    """Construct the normalized parallel training environment.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A ``VecNormalize`` wrapper around ``ParallelSoromoxEnv``.
    """

    env = load_env_class()(
        num_envs=args.num_envs,
        seed=args.seed,
        game_time=args.game_time,
        control_fps=args.control_fps,
        length=args.arm_length,
        radius=args.arm_radius,
        ball_radius=args.ball_radius,
        ball_surface_vmax=args.ball_surface_vmax,
        success_threshold=args.success_threshold,
        backend=args.backend,
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
        reward_batch_episodes: int,
        start_time: float,
    ) -> None:
        super().__init__()
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.reward_csv_path = reward_csv_path
        self.reward_batch_episodes = reward_batch_episodes
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        # Completed (wall_time, return, length) tuples not yet averaged.
        self.pending_episodes: list[tuple[float, float, int]] = []
        self.start_time = start_time
        self.last_checkpoint_step = 0

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reward_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_returns = [0.0 for _ in range(self.training_env.num_envs)]
        self.episode_lengths = [0 for _ in range(self.training_env.num_envs)]
        with self.reward_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wall_time", "episode_reward_mean"])
            writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self._last_original_rewards()
        dones = self.locals.get("dones")
        if rewards is not None and dones is not None:
            wall_time = time.perf_counter() - self.start_time
            for idx, (reward, done) in enumerate(zip(rewards, dones)):
                self.episode_returns[idx] += float(reward)
                self.episode_lengths[idx] += 1
                if bool(done):
                    self.pending_episodes.append(
                        (
                            wall_time,
                            self.episode_returns[idx],
                            self.episode_lengths[idx],
                        )
                    )
                    self.episode_returns[idx] = 0.0
                    self.episode_lengths[idx] = 0

        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq
        ):
            self._save_checkpoint(self.num_timesteps)
            self.last_checkpoint_step = self.num_timesteps

        # At most one record per step, from the oldest N completed episodes;
        # extra episodes stay queued for later steps.
        if len(self.pending_episodes) >= self.reward_batch_episodes:
            batch = self.pending_episodes[: self.reward_batch_episodes]
            self.pending_episodes = self.pending_episodes[self.reward_batch_episodes :]
            self._write_reward_record(batch)

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

    def _write_reward_record(self, batch: list[tuple[float, float, int]]) -> None:
        n = len(batch)
        wall_time = sum(item[0] for item in batch) / n
        reward_mean = sum(item[1] for item in batch) / n
        length_mean = sum(item[2] for item in batch) / n
        with self.reward_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wall_time", "episode_reward_mean"])
            writer.writerow(
                {
                    "wall_time": f"{wall_time:.12g}",
                    "episode_reward_mean": f"{reward_mean:.12g}",
                }
            )
        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "episode_reward_mean": reward_mean,
                    "episode_length_mean": length_mean,
                    "wall_time": wall_time,
                    "total_timesteps": self.num_timesteps,
                    "episodes_in_batch": n,
                },
                step=self.num_timesteps,
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

    args.backend = select_device_backend(args.device)
    print(f"Device: {args.device}, SoRoMoX backend: {args.backend}")
    if args.wandb_project:
        if wandb is None:
            raise ImportError("wandb is not installed; drop --wandb-project")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    env = build_env(args)
    start_time = time.perf_counter()
    model = build_model(env, args)
    callback = RewardCheckpointCallback(
        env=env,
        checkpoint_dir=Path(args.checkpoint_dir),
        checkpoint_freq=args.checkpoint_freq,
        reward_csv_path=Path(args.reward_csv),
        reward_batch_episodes=args.reward_batch_episodes,
        start_time=start_time,
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        save_final_artifacts(model, env, args)
    finally:
        env.close()
        if wandb is not None and wandb.run is not None:
            wandb.finish()


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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Simulation device: cpu uses the JAX backend, gpu uses the Warp backend.",
    )

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
        "--reward-batch-episodes",
        type=positive_int,
        default=8,
        help="Write one reward record (CSV and wandb) per N completed episodes.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Temporary: log to this Weights & Biases project when set.",
    )
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line training entry point."""

    train(parse_args())


if __name__ == "__main__":
    main()
