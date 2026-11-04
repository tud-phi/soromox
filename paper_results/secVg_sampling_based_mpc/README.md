# Section Vg: Sampling-Based MPC

This case runs closed-loop sampling-based MPC, iCEM by default and MPPI with
`--mppi`, in which a tapered four-tendon soft arm on a kinematically driven
floating base pushes a box or cylinder around a 10 cm square. The arm is
defined in `code/four_tendon_soft_arm.py`, and the planner primitives and push
cost live in `code/smpc.py`. Each replan samples rollouts against the real
Newton dynamics, coupled through newton-cosserat's SAP solver.

## Environment

Unlike the other sections, this case has its own uv projects rather than a
repository-root extra, one for the experiments and one for the figures. Run
every command from this directory:

```bash
cd paper_results/secVg_sampling_based_mpc
uv sync --project env_newton17   # experiments
uv sync --project .              # renderers and summary figure
```

Both projects use this checkout's SoRoMoX through `override-dependencies`.

The experiment project, [`env_newton17/pyproject.toml`](env_newton17/pyproject.toml),
is the stack the paper data was generated on. It reproduces that stack's
trajectories bit for bit, while the Newton 1.6 stack below gives different
contact dynamics:

| Package | Version | Reason |
| --- | --- | --- |
| `newton-cosserat` | [`Johannap1/newton-cosserat`](https://github.com/Johannap1/newton-cosserat) `secVg-sampling-based-mpc` | upstream `main` plus only its Newton 1.7 port |
| `newton` | `1.7.0` | paper data stack |
| `warp-lang` | `1.18.0` | paper data stack |
| `jax` | `0.11.1` | paper data stack |

The figure project, [`pyproject.toml`](pyproject.toml), stays on Newton 1.6
because Newton 1.7's RTX viewer requires `ovrtx` 0.5, which renders the sample
clouds black:

| Package | Version | Reason |
| --- | --- | --- |
| `newton-cosserat` | [`tud-phi/newton-cosserat`](https://github.com/tud-phi/newton-cosserat) `main` | coupled SAP solver |
| `newton` | `1.6.0.dev0`, git rev `7c677b95` | newton-cosserat's own pin `e3c5fdf4` lacks `log_mesh` opacity |
| `ovrtx` | `0.4.1` | 0.5.x renders dense translucent sample clouds black |
| `warp-lang` | `1.16.0` | matches the newton-cosserat lock |

Experiments run on the CPU or a GPU (`--device cpu` or `--device cuda:0`).
Both renderers need an RTX GPU. On a machine without a GPU, set
`JAX_PLATFORMS=cpu`; otherwise the JAX CUDA 13 plugin fails at import.

## Experiments

Run one closed-loop push with `code/run_experiment.py`:

```bash
uv run --project env_newton17 python code/run_experiment.py --device cuda:0 \
  --stop-after-one-square --object-shape box --k-samples 200 100 50 --num-iters 3 --seed 0 \
  --save-path /tmp/soromox-secVg/square_icem_seed0.npz
```

`--k-samples` takes one value or a per-iteration schedule. `--save-samples`
also stores every candidate rollout, which the sample-cloud renderer needs; the
files are large.

Regenerate the complete paper data set:

```bash
DEVICE=cuda:0 code/run_paper.sh
```

The script runs three parts, each over seeds 0, 1 and 2:

- a 20-step planner comparison toward the single push goal (0.05, -0.10) m,
  written to
  `data/cmp_seeds/seed*/`: an MPPI sample-count sweep (`cmp_mppi_k*.npz`,
  K = 200 to 1600, white noise), an iCEM iteration sweep at K = 200
  (`cmp_icem_k200_it*.npz`, 1 to 4 iterations), and an iCEM sample-count sweep
  at 3 iterations (`cmp_icem_it3_k*.npz`, K = 50 to 400);
- full square pushes with iCEM (`data/square_icem_seed*.npz`) and MPPI with
  K = 800 (`data/square_mppi_k800_white_seed*.npz`); the seed-0 iCEM run is
  saved with `--save-samples`;
- two seed-0 iCEM square-push variants, a 100 g box
  (`data/square_icem_box100g.npz`) and a cylinder
  (`data/square_icem_cylinder.npz`).

The script replaces same-named files in `data/`. Set `RESULTS_DIR` to stage a
run elsewhere. The committed `data/` copies have the per-sample arrays stripped
for size, so rerun the seed-0 iCEM square push to render panel A.

## Paper artifacts

Recreate the summary figure from the committed data and frames:

```bash
uv run --project . python code/plot_smpc_summary.py
```

This writes `outputs/smpc_summary.pdf`. Panel A shows RTX renders of the sample
cloud across iCEM iterations, B the MPPI-versus-iCEM planning-time and
performance trade-off, C RTX renders of the Franka-mounted arm pushing along the
trajectory, and D the top-down square-push trajectories of iCEM and MPPI
K = 800 over three seeds each. Panel B averages the seeds under
`data/cmp_seeds/`. LaTeX is used for text when it is installed.

Render the panel A frames, one per iCEM iteration of MPC step 0, from a run
saved with `--save-samples`:

```bash
code/render_sample_cloud.sh data/square_icem_seed0.npz
```

An optional second argument selects the drawn horizon step (default 10);
further arguments pass through to `code/render_sample_cloud.py`. The frames are
written to `outputs/frames/square_icem_seed0_t0_iter{0,1,2}.png`; set `OUT_DIR`
to override the directory.

Render a saved run as an RTX path-traced video:

```bash
uv run --project . python code/rendering.py --run-path data/square_icem_cylinder.npz \
  --object-shape cylinder --output outputs/square_icem_cylinder.mp4 --gif
```

The renderer reads the saved trajectory without re-simulating and adds a
decorative Franka FR3 whose end effector follows the arm base; `--franka-base`
moves it. The numbered PNG frames are kept under `outputs/<stem>_frames/frames/`,
and panel C uses selected frames copied to `outputs/frames/`. Encoding the MP4
and GIF requires ffmpeg.
