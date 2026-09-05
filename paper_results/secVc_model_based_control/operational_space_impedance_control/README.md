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

Render the saved canonical rollout, the four lossless paper snapshots at
4.5/5.5/6.5/7.5 s, and the publication-width PDF strip:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/render_control_pcs_with_impedance.py --force
```

The dedicated renderer consumes the committed NPZ and never repeats the
simulation. Viser recording requires a connected browser client; leave browser
auto-opening enabled, or use `--no-open-browser` only when a client is already
connected to `--viser-port`. The paper scene uses a 55-degree elevated view at
270-degree azimuth so both figure-eight lobes remain symmetrically visible, and
a restrained translucent mesh of the canonical spherical reference
surface. The plotting command is intentionally plot-only and no longer accepts
rendering or video arguments.

## Canonical Paper Figure

Generate the four-panel partial-feedback-linearization trajectory figure from
the committed trajectory NPZ:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/operational_space_impedance_control/code/plot_operational_space_impedance_paper_figure.py --force
```

## Artifacts

- `data/control_pcs_with_impedance_trajectory.npz` stores the canonical trajectory
  study, controller metadata, states, references, inputs, and the geometric
  position/orientation RMSE, reference-span, and NRMSE values used in the paper.
- `outputs/operational_space_impedance_control.pdf` is the canonical paper
  figure showing position and orientation tracking and component errors for the
  partial-feedback-linearization controller.
- `outputs/control_pcs_with_impedance_trajectory-render.mp4` is the canonical
  RL-inspired Viser rendering of the saved operational-space rollout.
- `outputs/operational_space_impedance_snapshots/` contains the four identically
  cropped, lossless PNG stills.
- `outputs/operational_space_impedance_snapshots.pdf` is the compact 15.2 cm-wide
  four-frame strip with per-frame timestamps and the paper-style time arrow.

Standalone trajectory and configuration plots are reproducible diagnostics and
remain uncommitted. The checked-in NPZ file was regenerated from scratch with
the current Section Vc generator.
