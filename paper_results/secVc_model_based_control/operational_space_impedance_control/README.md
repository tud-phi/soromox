# Operational-Space Impedance Control

This case reproduces the Section Vc PCS operational-space impedance-control
study. Simulation and rendering are separated: generators write NPZ data,
plotters consume it, and renderers never repeat a simulation implicitly.

## Trajectory Study

Run the canonical pose-tracking simulation and produce its figures in an
isolated validation directory:

```bash
mkdir -p /tmp/soromox-secVc/operational/{data,outputs}
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/control_pcs_with_impedance.py --trajectory-output /tmp/soromox-secVc/operational/data/control_pcs_with_impedance_trajectory.npz
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/plot_control_pcs_with_impedance.py /tmp/soromox-secVc/operational/data/control_pcs_with_impedance_trajectory.npz --output-prefix /tmp/soromox-secVc/operational/outputs/control_pcs_with_impedance_trajectory --no-show
```

The generator accepts `--tracking`, `--position-primitive`,
`--orientation-primitive`, and `--surface` for the documented scenario variants.
Give each variant a distinct `--trajectory-output` path to prevent data clashes.
The default output is the canonical pose/figure-eight/surface-following dataset.

To record the saved canonical rollout:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/plot_control_pcs_with_impedance.py /tmp/soromox-secVc/operational/data/control_pcs_with_impedance_trajectory.npz --output-prefix /tmp/soromox-secVc/operational/outputs/control_pcs_with_impedance_trajectory --no-show --video-output /tmp/soromox-secVc/operational/outputs/control_pcs_with_impedance_trajectory-render.mp4
```

Use `--render --no-open-browser` to launch a non-recording Viser session without
opening a browser. Add `--non-blocking` for a headless renderer smoke test.
Plotting alone does not start a rendering server.

## Feedback-Linearization Comparison

Generate one NPZ containing the matched full and partial controller rollouts,
then plot it without rerunning either controller:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/compare_impedance_feedback_linearization.py --output /tmp/soromox-secVc/operational/data/impedance_feedback_linearization_comparison.npz
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/plot_impedance_feedback_linearization.py --input /tmp/soromox-secVc/operational/data/impedance_feedback_linearization_comparison.npz --output-dir /tmp/soromox-secVc/operational/outputs
```

Use `--force` before replacing existing artifacts. Full CPU simulations include
JAX compilation and may take several minutes. Viser video output additionally
requires the rendering dependencies and a working video encoder; no physical
robot hardware is required.

## Canonical Paper Figure

Generate the four-panel partial-feedback-linearization trajectory figure from
the committed trajectory NPZ:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/plot_operational_space_impedance_paper_figure.py --force
```

## Artifacts

- `data/control_pcs_with_impedance_trajectory.npz` stores the canonical trajectory
  study, controller metadata, states, references, and inputs.
- `data/impedance_feedback_linearization_comparison.npz` stores both matched
  feedback-linearization rollouts, geometric errors, settings, and runtimes for
  the standalone comparison diagnostic.
- `outputs/operational_space_impedance_control.pdf` is the canonical paper
  figure showing position and orientation tracking and component errors for the
  partial-feedback-linearization controller.
- `outputs/control_pcs_with_impedance_trajectory-render.mp4` is the prior
  canonical rendering of the saved operational-space rollout. Additional MP4
  files can be rendered from any saved NPZ data.

Standalone trajectory, configuration, and feedback-linearization plots are
reproducible diagnostics and remain uncommitted. The checked-in NPZ files were
regenerated from scratch with the current Section Vc generators.
