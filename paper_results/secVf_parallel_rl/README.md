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

Generate fresh trajectory data from the released policy. The default path is
protected because it contains the canonical rollout:

```bash
uv run python paper_results/secVf_parallel_rl/code/run_rl_policy.py \
  --trajectory-output /tmp/soromox-secVf-rollout.npz
```

Render saved trajectory data without rerunning the policy:

```bash
uv run python paper_results/secVf_parallel_rl/code/render_rl_video.py \
  --data /tmp/soromox-secVf-rollout.npz \
  --output paper_results/secVf_parallel_rl/outputs/rl_rollout.mp4
```

`run_rl_policy.py --render` additionally renders all selected environments in
one Open3D grid and derives a GIF with ffmpeg. Rendering requires Open3D, a
working display or headless graphics setup, and ffmpeg. Training and rollout
results are stochastic and accelerator dependent; the committed checkpoint,
normalization state, reward CSVs, and rollout NPZ provide the paper provenance.
