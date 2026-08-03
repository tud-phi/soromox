# Configuration-Space Comparison

This case reproduces the Section Vc configuration-space controller comparisons.
The three generators share the robot, controller, trajectory, serialization,
plotting, and rendering implementation in `code/configuration_space_comparison_simulation.py`.

## Integral-error saturation

The PID integral state uses the unit-preserving smooth saturation

$$
\dot{\eta}_i = e_{\mathrm{sat},i}
\tanh\!\left(e_i/e_{\mathrm{sat},i}\right).
$$

The PCS coordinates combine rotational strains in $\mathrm{m^{-1}}$ and
dimensionless linear strains, so the benchmark uses separate physical scales:

- $e_{\kappa,\mathrm{sat}}=10\,\mathrm{m^{-1}}$, corresponding to a 1 rad
  angular error across the 0.1 m segment;
- $e_{\sigma,\mathrm{sat}}=0.1$, corresponding to a 10% linear strain error.

These scales keep the nominal PID integral gain unchanged for small errors.
They can be varied with `--rotational-integral-error-scale` and
`--linear-integral-error-scale` on each simulation command.

## Reproduce

From the repository root, generate all three datasets without touching the
checked-in artifacts:

```bash
mkdir -p /tmp/soromox-secVc/configuration/{data,outputs}
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/setpoint_regulation_comparison.py --output /tmp/soromox-secVc/configuration/data/setpoint_regulation_comparison.npz
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/trajectory_tracking_comparison.py --output /tmp/soromox-secVc/configuration/data/trajectory_tracking_comparison.npz
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/regulation_tracking_comparison.py --output /tmp/soromox-secVc/configuration/data/regulation_tracking_comparison.npz
```

Plot all datasets without opening interactive windows:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/plot_configuration_space_comparison.py /tmp/soromox-secVc/configuration/data/setpoint_regulation_comparison.npz --output-dir /tmp/soromox-secVc/configuration/outputs --no-show
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/plot_configuration_space_comparison.py /tmp/soromox-secVc/configuration/data/trajectory_tracking_comparison.npz --output-dir /tmp/soromox-secVc/configuration/outputs --no-show
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/plot_configuration_space_comparison.py /tmp/soromox-secVc/configuration/data/regulation_tracking_comparison.npz --output-dir /tmp/soromox-secVc/configuration/outputs --no-show
```

Generate the canonical regulation-to-tracking paper figure from the committed
data:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/plot_configuration_space_paper_figure.py --force
```

Generate the phase-separated RMSE diagnostic, with independent setpoint and
trajectory-tracking columns and one explicitly labeled row for each controlled
strain ($\kappa_y$, $\kappa_z$, and $\sigma_x$):

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/plot_configuration_space_comparison.py paper_results/secVc_model_based_control/configuration_space_comparison/data/regulation_tracking_comparison.npz --group phase-rmse --no-show --force
```

Record a saved rollout with the Viser renderer, selecting a target listed by the
plotter or renderer CLI:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/configuration_space_comparison/code/render_configuration_space_comparison.py /tmp/soromox-secVc/configuration/data/setpoint_regulation_comparison.npz --target setpoint-pid --video-output /tmp/soromox-secVc/configuration/outputs/setpoint-pid.mp4
```

Use `--force` to replace existing data or outputs. Custom `--output`,
`--output-dir`, and `--video-output` paths are supported. A first JAX run incurs
compilation overhead; full CPU runs can take several minutes. Video recording
requires Viser and a working video encoder, but no robot hardware.

## Artifacts

`data/` contains one NPZ per workflow. Each archive records metadata, scenario
and controller names, time vectors, states, inputs, references, and the
Section-Vc evaluation metrics. RMSE retains all six strain coordinates; the
combined regulation/tracking archive additionally stores the four terminal
setpoint-window MAEs and their equal-weight aggregate for regulation.
`outputs/configuration_space_comparison.pdf` contains the complete
regulation-to-tracking trajectory for all three controlled strains.
`outputs/regulation_tracking_rmse.pdf` places the full setpoint-regulation RMSE
next to the steady-state regulation MAE, followed by the trajectory-tracking
RMSE. The steady-state MAE is computed over the final 10% (0.3 s) of each
evaluated 3 s setpoint interval after the initial plateau and averaged equally
across the four intervals.
Scenario-specific tracking and error plots are reproducible diagnostics and
remain uncommitted. The checked-in NPZ files were regenerated from scratch with
the current Section Vc generators.
