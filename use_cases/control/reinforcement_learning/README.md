# Parallel SoRoMoX Release

This folder contains the parallel SoRoMoX
reinforcement-learning workflow:

- `parallel_soromox_env.py`: JAX-vectorized Stable-Baselines3 `VecEnv`.
- `train.py`: PPO training file.
- `test.py`: model evaluation file.
- `plot_reward.py`: create reward-curve PDFs from local CSV exports.
- `visualize.py`: run the checkpointed PPO model, render the parallel rollout,
  and save one MP4 video.

Training saves the final PPO model and the final SB3 `VecNormalize` state.
Testing will print the final-step end-effector tracking error and success rate.

## Dependencies

The scripts assume `soromox` has already been installed in the active Python
environment. They do not modify `sys.path` to import from a source checkout.

The environment should provide:

- `soromox`
- `jax`
- `diffrax`
- `equinox`
- `gymnasium`
- `stable-baselines3`
- `numpy`
- `matplotlib`
- `open3d`
- `ffmpeg`

Please use `pip install soromox[examples]` to get all dependencies before
running `train.py`, `test.py`, or `visualize.py`.

## Environment

`ParallelSoromoxEnv` is an SB3-compatible `VecEnv`.

Observation layout:

```text
[ee_pos(3), ee_vel(3), tendon_force(4), ball_pos(3), ball_vel(3)]
```

Action layout:

```text
[delta_tendon_0, delta_tendon_1, delta_tendon_2, delta_tendon_3]
```

Default task parameters:

```text
num envs: 64
episode length: 105 action steps at default 7 s / 15 Hz
arm length: 0.25 m
arm radius: 0.025 m
target point hemisphere radius: 0.10 m
success threshold: 0.01 m
control fps: 15 Hz
dt: 1e-4 s
```

## Training

From the repository root:

```bash
python examples/control/rl/train.py \
  --num-envs 64 \
  --total-timesteps 1000000 \
  --n-steps 200
```

The training script saves:

```text
examples/control/rl/checkpoints/ppo_model.zip
examples/control/rl/checkpoints/env_vecnormalize.pkl
```

You can change the artifact names:

```bash
python examples/control/rl/train.py \
  --save-dir examples/control/rl/checkpoints \
  --model-name my_model \
  --vecnormalize-name my_vecnormalize.pkl
```

## Evaluation

```bash
python examples/control/rl/test.py
```

By default, the test script loads the packaged release model:

```text
examples/control/rl/checkpoints/ppo_model.zip
examples/control/rl/checkpoints/env_vecnormalize.pkl
```

To test a newly trained model, pass explicit paths:

```bash
python examples/control/rl/test.py \
  --model-path your_model \
  --vecnormalize-path your_vecnormalize.pkl
```

The default evaluation length is `105` action steps. The script prints:

```text
mean_diff_xyz_m
mean_tracking_error_m
success_rate
```

`mean_diff_xyz_m` is the component-wise average of:

```text
ee_pos - ball_pos
```

`mean_tracking_error_m` is the mean Euclidean distance:

```python
np.linalg.norm(ee_pos - ball_pos, axis=1).mean()
```

## Reward Plotting

`plot_reward.py` creates an averaged reward-curve PDF from local CSV exports.
By default it reads all CSV files from:

```text
examples/control/rl/reward_logs
```

Each CSV file must contain these columns:

```text
wall_time
episode_reward_mean
```

File names are used to group runs automatically. For example,
`SoRoMoX_64_envs_1bqkkr1d.csv` is plotted as `SoRoMoX 64 envs`, while
`PyElastica_*.csv` files are plotted as `PyElastica`.

From the repository root:

```bash
python examples/control/rl/plot_reward.py
```

The default output is:

```text
examples/control/rl/visualizations/reward_curve.pdf
```

To use another CSV folder or output path:

```bash
python examples/control/rl/plot_reward.py \
  --csv-dir path/to/reward_logs \
  --output path/to/reward_curve.pdf
```

The script interpolates averaged curves to 500 points and applies light Gaussian
smoothing by default. You can adjust those settings:

```bash
python examples/control/rl/plot_reward.py \
  --points 800 \
  --smooth-scale 0
```

Use `--smooth-scale 0` to disable smoothing.

## Visualization

The release includes checkpoint files in:

```text
examples/control/rl/checkpoints/ppo_model.zip
examples/control/rl/checkpoints/env_vecnormalize.pkl
```

Render a parallel rollout directly to MP4 and GIF:

```bash
python examples/control/rl/visualize.py
```

By default this runs 64 environments for 105 control steps, renders them in one
Open3D grid scene, and saves only:

```text
examples/control/rl/visualizations/parallel_track_video.mp4
examples/control/rl/visualizations/parallel_track_video.gif
```

No NPZ rollout files, PNG frame folders, or palette files are kept. The script
prints progress bars for the rollout stage and the render/encode stage.

Optional smaller preview:

```bash
python examples/control/rl/visualize.py \
  --num-envs 16 \
  --max-envs 16 \
  --n-steps 45 \
  --width 960 \
  --height 960 \
  --output examples/control/rl/visualizations/parallel_track_video.mp4 \
  --gif-width 480
```

![Parallel tracking demo](visualizations/parallel_track_video.gif)
