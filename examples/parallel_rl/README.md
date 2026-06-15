# Parallel SoRoMoX Release

This folder contains a cleaned release version of the parallel SoRoMoX
reinforcement-learning workflow:

- `parallel_soromox_env.py`: JAX-vectorized Stable-Baselines3 `VecEnv`.
- `train_parallel_soromox.py`: PPO training entry point.
- `test_parallel_soromox.py`: fixed-length parallel evaluation entry point.

The release code intentionally avoids experiment-specific side effects:

- no checkpoint saving
- no CSV logging
- no evaluation NPZ export

Training saves only the final PPO model and the final SB3 `VecNormalize` state.
Testing prints only the final-step end-effector tracking error and success rate.

## Dependencies

The scripts assume the project environment already provides:

- `soromox`
- `jax`
- `diffrax`
- `equinox`
- `gymnasium`
- `stable-baselines3`
- `numpy`

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
num_envs: 64
episode length: 105 action steps at default 7 s / 15 Hz
arm_length: 0.25 m
arm_radius: 0.025 m
ball_radius: 0.10 m
success_threshold: 0.01 m
control_fps: 15 Hz
dt: 1e-4 s
```

Each reset splits the JAX PRNG key into one key per parallel environment, so
target trajectories are different across vectorized environments.

## Training

From the repository root:

```bash
python release/train_parallel_soromox.py \
  --num-envs 64 \
  --total-timesteps 1000000 \
  --n-steps 200 \
  --save-dir soromox_models
```

The training script saves only:

```text
soromox_models/soromox_ppo_final.zip
soromox_models/soromox_ppo_vecnormalize_final.pkl
```

You can change the artifact names:

```bash
python release/train_parallel_soromox.py \
  --model-name my_model \
  --vecnormalize-name my_vecnormalize.pkl
```

## Evaluation

Run a fixed-length parallel rollout:

```bash
python release/test_parallel_soromox.py
```

By default, the test script loads the packaged release model:

```text
release/soromox_ppo_release.zip
release/soromox_ppo_vecnormalize_release.pkl
```

To test a newly trained model, pass explicit paths:

```bash
python release/test_parallel_soromox.py \
  --model-path soromox_models/soromox_ppo_final.zip \
  --vecnormalize-path soromox_models/soromox_ppo_vecnormalize_final.pkl
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

## Reproducibility Notes

- The top-level seed controls the JAX PRNG key used by the vectorized
  environment.
- `VecNormalize` must be saved and reloaded with the PPO model for evaluation.
- `test_parallel_soromox.py` sets `training=False` and `norm_reward=False` on
  `VecNormalize`, matching standard SB3 evaluation practice.

## File-Level API

`parallel_soromox_env.py` provides:

- `ParallelSoromoxEnv`
- `build_arm`
- `build_jax_env_fns`
- `make_default_y0`

Every function and class in the release Python files includes a docstring.
