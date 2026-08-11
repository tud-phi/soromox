"""Generate and save Section Vf PPO rollout trajectories."""

import argparse
import importlib.util
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
DATA_DIR = CASE_DIR / "data"
TRAJECTORY_DIR = DATA_DIR / "traj"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/soromox_matplotlib")

import jax
import jax.numpy as jnp
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


DEFAULT_MODEL_PATH = DATA_DIR / "checkpoints" / "ppo_model.zip"
DEFAULT_VECNORMALIZE_PATH = DATA_DIR / "checkpoints" / "env_vecnormalize.pkl"
DEFAULT_TRAINED_TRAJECTORY_OUTPUT = TRAJECTORY_DIR / "rl_rollout_trained_1_env.npz"
DEFAULT_RANDOM_TRAJECTORY_OUTPUT = TRAJECTORY_DIR / "rl_rollout_initialized_1_env.npz"
POLICY_CHOICES = ("trained", "random")


class Progress:
    """Small tqdm wrapper with a print-only fallback."""

    def __init__(self, total: int, desc: str):
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self._next_percent = 0
        try:
            from tqdm import tqdm
        except ImportError:
            self._bar = None
            print(f"{self.desc}: 0/{self.total}")
        else:
            self._bar = tqdm(total=self.total, desc=self.desc)

    def update(self, amount: int = 1) -> None:
        self.current += amount
        if self._bar is not None:
            self._bar.update(amount)
            return

        percent = int(self.current * 100 / self.total)
        if percent >= self._next_percent or self.current >= self.total:
            print(f"{self.desc}: {self.current}/{self.total} ({percent}%)")
            self._next_percent = percent + 10

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def to_numpy(value, dtype=None) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    return array.astype(dtype) if dtype is not None else array


def extract_q_batch(raw_env: ParallelSoromoxEnv) -> np.ndarray:
    if raw_env._env_states is None:
        raise RuntimeError("Environment must be reset before extracting q.")
    y = raw_env._env_states.arm_state.y
    q, _ = jnp.split(y, 2, axis=1)
    return to_numpy(q, dtype=np.float64)


def default_trajectory_output(policy: str) -> Path:
    """Return the case-local generated trajectory path for a policy type."""
    if policy == "trained":
        return DEFAULT_TRAINED_TRAJECTORY_OUTPUT
    if policy == "random":
        return DEFAULT_RANDOM_TRAJECTORY_OUTPUT
    raise ValueError(f"Unknown policy type: {policy!r}")


def sample_random_action_batch(
    raw_env: ParallelSoromoxEnv,
) -> np.ndarray:
    """Sample one uniformly random action for each parallel environment."""
    return np.stack(
        [raw_env.action_space.sample() for _ in range(raw_env.num_envs)],
        axis=0,
    ).astype(np.float32)


def run_policy_and_collect(
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    vec_env = None
    model = None
    if args.policy == "trained":
        vec_env = VecNormalize.load(str(args.vecnormalize_path), raw_env)
        vec_env.training = False
        vec_env.norm_reward = False
        model = PPO.load(str(args.model_path), env=vec_env)
    else:
        raw_env.action_space.seed(args.seed)

    obs = raw_env.reset()
    if vec_env is not None:
        obs = vec_env.normalize_obs(obs)

    q_list = [extract_q_batch(raw_env)]
    ball_list = [to_numpy(raw_env._env_states.ball_pos, dtype=np.float64)]
    action_list = []
    reward_list = []

    progress = Progress(args.n_steps, "Rollout")
    try:
        for _ in range(args.n_steps):
            if model is None:
                actions = sample_random_action_batch(raw_env)
            else:
                actions, _ = model.predict(
                    obs,
                    deterministic=not args.stochastic,
                )
                actions = np.asarray(actions, dtype=np.float32)
            (
                env_states,
                raw_obs,
                _rewards,
                _terminateds,
                _truncateds,
                _ee_pos,
                _invalid_obs,
            ) = raw_env._step_batch(raw_env._env_states, actions)
            raw_env._env_states = env_states
            if vec_env is not None:
                obs = vec_env.normalize_obs(to_numpy(raw_obs, dtype=np.float32))
            action_list.append(actions)
            reward_list.append(to_numpy(_rewards, dtype=np.float64))
            q_list.append(extract_q_batch(raw_env))
            ball_list.append(to_numpy(raw_env._env_states.ball_pos, dtype=np.float64))
            progress.update()
    finally:
        progress.close()

    q_ts = np.asarray(q_list, dtype=np.float64)
    ball_ts = np.asarray(ball_list, dtype=np.float64)
    actions = np.asarray(action_list, dtype=np.float64)
    rewards = np.asarray(reward_list, dtype=np.float64)
    ts = np.arange(q_ts.shape[0], dtype=np.float64) / float(args.control_fps)
    if vec_env is not None:
        vec_env.close()
    else:
        raw_env.close()
    return ts, q_ts, ball_ts, actions, rewards


def save_rollout_npz(
    path: Path,
    *,
    ts: np.ndarray,
    q_ts: np.ndarray,
    ball_ts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    arm_length: float,
    arm_radius: float,
    policy: str = "trained",
    force: bool = False,
) -> Path:
    """Save a complete parallel rollout using the time-major Section Vf schema."""
    ts = np.asarray(ts, dtype=np.float64)
    q_ts = np.asarray(q_ts, dtype=np.float64)
    ball_ts = np.asarray(ball_ts, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if ts.ndim != 1 or ts.size == 0:
        raise ValueError(f"ts must have shape (T,) with T > 0; got {ts.shape}")
    if ts.size > 1 and np.any(np.diff(ts) <= 0.0):
        raise ValueError("ts must be strictly increasing")
    if q_ts.ndim != 3 or q_ts.shape[0] != ts.size or q_ts.shape[1] == 0:
        raise ValueError(f"q_ts must have shape (T,N,D) matching ts; got {q_ts.shape}")
    num_steps, num_envs = q_ts.shape[:2]
    if ball_ts.shape != (num_steps, num_envs, 3):
        raise ValueError(
            f"ball_ts must have shape (T,N,3) matching q_ts; got {ball_ts.shape}"
        )
    if actions.ndim != 3 or actions.shape[:2] != (num_steps - 1, num_envs):
        raise ValueError(
            f"actions must have shape (T-1,N,A) matching q_ts; got {actions.shape}"
        )
    if rewards.shape != (num_steps - 1, num_envs):
        raise ValueError(
            f"rewards must have shape (T-1,N) matching q_ts; got {rewards.shape}"
        )
    if not np.isfinite(arm_length) or arm_length <= 0.0:
        raise ValueError("arm_length must be a positive finite value")
    if not np.isfinite(arm_radius) or arm_radius <= 0.0:
        raise ValueError("arm_radius must be a positive finite value")
    if policy not in POLICY_CHOICES:
        raise ValueError(f"policy must be one of {POLICY_CHOICES}; got {policy!r}")

    path = Path(path).resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ts=ts,
        q_ts=q_ts,
        ball_ts=ball_ts,
        actions=actions,
        rewards=rewards,
        arm_length=np.asarray(arm_length, dtype=np.float64),
        arm_radius=np.asarray(arm_radius, dtype=np.float64),
        policy=np.asarray(policy),
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default="trained",
        help="Use the released trained PPO or the uniform-random initialized baseline.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--vecnormalize-path", type=Path, default=DEFAULT_VECNORMALIZE_PATH
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="Output NPZ. Defaults to a policy-specific filename in data/traj/.",
    )

    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=105)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--game-time", type=float, default=7.0)
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--arm-length", type=float, default=0.25)
    parser.add_argument("--arm-radius", type=float, default=0.025)
    parser.add_argument("--ball-radius", type=float, default=0.10)
    parser.add_argument("--ball-surface-vmax", type=float, default=0.015)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.trajectory_output is None:
        args.trajectory_output = default_trajectory_output(args.policy)
    return args


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.vecnormalize_path = args.vecnormalize_path.resolve()
    args.trajectory_output = args.trajectory_output.resolve()
    if args.num_envs <= 0 or args.n_steps <= 0:
        raise ValueError("--num-envs and --n-steps must be positive.")
    if args.arm_length <= 0.0 or args.arm_radius <= 0.0:
        raise ValueError("--arm-length and --arm-radius must be positive.")
    missing_inputs = (
        [
            path
            for path in (args.model_path, args.vecnormalize_path)
            if not path.exists()
        ]
        if args.policy == "trained"
        else []
    )
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(map(str, missing_inputs))
        )
    if args.trajectory_output.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite {args.trajectory_output}; pass --force"
        )

    print(f"Policy: {args.policy}")
    if args.policy == "trained":
        print(f"Model: {args.model_path}")
        print(f"VecNormalize: {args.vecnormalize_path}")
    print(f"Trajectory data: {args.trajectory_output}")

    ts, q_ts, ball_ts, actions, rewards = run_policy_and_collect(args)
    output = save_rollout_npz(
        args.trajectory_output,
        ts=ts,
        q_ts=q_ts,
        ball_ts=ball_ts,
        actions=actions,
        rewards=rewards,
        arm_length=args.arm_length,
        arm_radius=args.arm_radius,
        policy=args.policy,
        force=args.force,
    )
    print(f"Saved {q_ts.shape[1]} environment(s): {output}")
    print(
        "Render with: uv run python "
        "paper_results/secVf_parallel_rl/code/render_rl_video.py "
        f"--data {output}"
    )


if __name__ == "__main__":
    main()
