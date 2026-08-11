# Section Vf: Parallel Reinforcement Learning

This case contains the complete parallel SoRoMoX PPO workflow: environment,
training, evaluation, rollout generation, reward plotting, and offline video
rendering. Released checkpoints and source data live in `data/`; canonical paper
outputs live in `outputs/`.

Install the paper and RL dependencies:

```bash
uv sync --extra paper_results --extra rl
```

## Training and evaluation

Train with the paper defaults. Checkpoints and reward records are written below
the case-local `data/` directory:

```bash
uv run python paper_results/secVf_parallel_rl/code/train.py \
  --num-envs 64 --total-timesteps 1000000 --n-steps 200 \
  --save-dir /tmp/soromox-secVf/checkpoints \
  --checkpoint-dir /tmp/soromox-secVf/checkpoints \
  --reward-csv /tmp/soromox-secVf/training_reward.csv
```

The training CLI refuses to replace the released checkpoint or an existing
reward CSV unless `--force` is supplied.

Evaluate the released checkpoint:

```bash
uv run python paper_results/secVf_parallel_rl/code/evaluate_rl.py
```

Evaluation reports the component-wise end-effector error, mean Euclidean
tracking error, and success rate for a fixed rollout.

## Paper artifacts

Recreate the reward curve from the committed training logs:

```bash
uv run python paper_results/secVf_parallel_rl/code/plot_rl.py
```

Rollout generation and rendering are separate steps. Generate the single-arm
trajectory data for both the uniformly random initialized-policy baseline and
the fully trained PPO policy with:

```bash
uv run python paper_results/secVf_parallel_rl/code/run_rl_policy.py \
  --policy random --num-envs 1 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVf_parallel_rl/data/traj/rl_rollout_initialized_1_env.npz \
  --force

uv run python paper_results/secVf_parallel_rl/code/run_rl_policy.py \
  --policy trained --num-envs 1 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVf_parallel_rl/data/traj/rl_rollout_trained_1_env.npz \
  --force
```

The random baseline samples reproducible uniform actions from the environment;
the trained mode loads the released PPO checkpoint and normalization state.
Render both saved trajectories without rerunning either policy:

```bash
uv run python paper_results/secVf_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVf_parallel_rl/data/traj/rl_rollout_initialized_1_env.npz \
  --force

uv run python paper_results/secVf_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVf_parallel_rl/data/traj/rl_rollout_trained_1_env.npz \
  --force
```

When `--output` is omitted, the renderer writes the MP4 and GIF to the case
study `outputs/` directory using the input NPZ stem. For example,
`rl_rollout_initialized_1_env.npz` produces
`rl_rollout_initialized_1_env.mp4` and `rl_rollout_initialized_1_env.gif`. Use
`--output` or `--gif-output` to override either path. A single selected
environment uses the fixed camera and visual style of the paper result.

To generate and render parallel rollouts for both policy states, save all
environments in one NPZ per policy and pass each file to the same renderer:

```bash
uv run python paper_results/secVf_parallel_rl/code/run_rl_policy.py \
  --policy random --num-envs 64 \
  --trajectory-output paper_results/secVf_parallel_rl/data/traj/rl_rollout_initialized_64_envs.npz \
  --force

uv run python paper_results/secVf_parallel_rl/code/run_rl_policy.py \
  --policy trained --num-envs 64 \
  --trajectory-output paper_results/secVf_parallel_rl/data/traj/rl_rollout_trained_64_envs.npz \
  --force

uv run python paper_results/secVf_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVf_parallel_rl/data/traj/rl_rollout_initialized_64_envs.npz \
  --force

uv run python paper_results/secVf_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVf_parallel_rl/data/traj/rl_rollout_trained_64_envs.npz \
  --force
```

The multi-arm renderer uses every stored environment by default, arranges the
arms in a centered near-square grid, and fits an oblique camera to the complete
scene. `--max-envs N` selects the first `N` environments; in particular,
`--max-envs 1` provides the first-class single-arm view from a batched rollout.
Use `--rows` and `--cols` together to override the automatic grid, and
`--grid-spacing` to change the distance between robot bases.

Generated trajectories use time-major arrays and retain every environment:
`q_ts` has shape `(T, N, D)` and `ball_ts` has shape `(T, N, 3)`. Existing
unbatched single-arm files remain supported. Trajectory NPZs belong under
`data/traj/`; generated plots, MP4s, and GIFs belong under `outputs/`. The names
follow `rl_rollout_<policy-state>_<environment-count>` so corresponding data and
media share a stem. Current case-study plots and rendered media in `outputs/`
are versioned; MP4 and GIF files use Git LFS. Both CLIs protect existing outputs;
pass `--force` when replacement is intentional. Rendering requires Open3D, a
working display or headless graphics setup, and ffmpeg. Pass `--visible` on
desktop graphics stacks that cannot create a hidden Open3D window. Training and
rollout results are stochastic and accelerator dependent; the committed
checkpoint, normalization state, reward CSVs, and rollout NPZ provide the paper
provenance.
