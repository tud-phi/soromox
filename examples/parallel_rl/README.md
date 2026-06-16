# Parallel SoRoMoX Release

This folder contains the parallel SoRoMoX
reinforcement-learning workflow:

- `parallel_soromox_env.py`: JAX-vectorized Stable-Baselines3 `VecEnv`.
- `train.py`: PPO training file.
- `test.py`: model evaluation file.

Training saves the final PPO model and the final SB3 `VecNormalize` state.
Testing will print the final-step end-effector tracking error and success rate.

## Dependencies

The scripts assume the project environment already provides:

- `soromox`
- `jax`
- `diffrax`
- `equinox`
- `gymnasium`
- `stable-baselines3`
- `numpy`

Please use `pip install soromox[examples]` to get all dependencies.

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
python examples/parallel_rl/train.py \
  --num-envs 64 \
  --total-timesteps 1000000 \
  --n-steps 200
```

The training script saves:

```text
examples/parallel_rl/soromox_models/soromox_ppo_final.zip
examples/parallel_rl/soromox_models/soromox_ppo_vecnormalize_final.pkl
```

You can change the artifact names:

```bash
python examples/parallel_rl/train.py \
  --model-name my_model \
  --vecnormalize-name my_vecnormalize.pkl
```

## Evaluation

```bash
python examples/parallel_rl/test.py
```

By default, the test script loads the packaged release model:

```text
examples/parallel_rl/soromox_models/soromox_ppo_final.zip
examples/parallel_rl/soromox_models/soromox_ppo_vecnormalize_final.pkl
```

To test a newly trained model, pass explicit paths:

```bash
python examples/parallel_rl/test.py \
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
