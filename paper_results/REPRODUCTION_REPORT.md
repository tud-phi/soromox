# Reproduction Validation Report

Validation date: 2026-08-01

Branch: `visualization/plots_standardization`

Host: macOS 26.5.2 arm64, Python 3.12.11, JAX 0.10.0 on CPU

Tools: uv 0.12.1, FFmpeg 8.1.2, Git LFS 3.7.1

Generated validation artifacts were written under `/private/tmp` or ignored
example artifact directories. The checked-in Section Vc PDFs, PNGs, and MP4 are
the prior pre-migration artifacts; the checked-in Section Vc NPZ trajectories
were regenerated from scratch as requested.

## Automated checks

| Check | Status | Result |
| --- | --- | --- |
| Full test suite | Full | `1170 passed in 999.52s` |
| Section Vc focused suite after final path tests | Full | `63 passed in 43.16s` |
| Ruff on changed examples, paper scripts, renderers, and tests | Full | No findings |
| Python compilation of changed entry points | Full | No failures |
| Unrelated-CWD Section Vc CLI startup | Full | All nine entry points start from a temporary directory |
| Shared plot style regression | Full | Representative Section IVa and Vf rasters are pixel-identical before/after extraction |
| LFS attributes | Full | Section Vc NPZ/PDF/PNG/MP4 files resolve to the `lfs` filter |

## Examples

All simulations below used their normal final time, solver step, and saved
sample settings. “Renderer smoke” means the numerical simulation completed but
a GUI renderer was replaced with a non-interactive test double. Wall timings
were not retained unless the entry point reported them.

| Entry point | Status | Validation and artifacts |
| --- | --- | --- |
| `control/actuation_space/setpoint_regulation_comparison.py` | Full simulation | Four PDFs written to sibling `figures/`; Open3D skipped with `--no-render` |
| `control/operational_space/control_tendon_actuated_pcs_with_synergistic.py` | Full simulation | Two PDFs in sibling `figures/`; mean RMSE 10.394 mm; renderer skipped |
| `simulation/articulated/simulate_articulated_soft_robot.py` | Full simulation, renderer smoke | Completed without rendering; custom recording paths are script-relative |
| `simulation/articulated/simulate_mckibben_umarm.py` | Full simulation, renderer smoke | Completed without rendering; default video is sibling `videos/` |
| `simulation/gvs/simulate_gvs.py` | Full simulation, renderer smoke | 201 saved steps completed |
| `simulation/gvs/simulate_tendon_actuated_gvs.py` | Full simulation, renderer smoke | 201 saved steps completed |
| `simulation/hsa/demo_planar_hsa_motor2ee_jacobian.py` | Full | All 10 Jacobian evaluations completed after updating stale force APIs |
| `simulation/hsa/simulate_planar_hsa.py` | Full | 5.01 s, 241,866-byte MP4 in sibling `videos/` |
| `simulation/pcs/simulate_batched_tendon_actuated_pcs.py` | Full simulation, renderer smoke | Batched rollout completed; both video destinations are configurable |
| `simulation/pcs/simulate_isupport.py` | Full simulation, renderer smoke | Completed with a non-interactive renderer |
| `simulation/pcs/simulate_pcs.py` | Full simulation, renderer smoke | Completed with non-interactive renderers |
| `simulation/pcs/simulate_planar_pcs.py` | Full simulation, renderer smoke | Completed; OpenCV destination is sibling `videos/` and configurable |
| `simulation/pcs/simulate_tendon_actuated_pcs.py` | Full simulation, renderer smoke | Completed; reported simulation time 6.942 s |
| `simulation/pcs/simulate_tendon_actuated_planar_pcs.py` | Full simulation, renderer smoke | Completed; removed creation of an unused root `videos/` directory |
| `simulation/pendulum/simulate_pendulum.py` | Full | 5.01 s, 188,710-byte MP4 in sibling `videos/` |
| `simulation/pendulum/simulate_tendon_actuated_pendulum.py` | Full | 5.01 s, 4,042,237-byte MP4 in sibling `videos/` |

## Paper workflows

| Section or case | Status | Command/stage and result |
| --- | --- | --- |
| IVa sequential CPU | Plotting full; generation unavailable | All four source-data plots rendered under `/private/tmp/soromox-paper-secIVa`; SoRoSim/MATLAB and benchmark generators are not in this repository |
| IVb parallel GPU | Plotting full; generation externally blocked | Three plots rendered under `/private/tmp/soromox-paper-secIVb`; no GPU was available and no benchmark generator is present |
| Va system identification | Base identification and plotting full; residual generation unavailable | The 100-step identifier completed in about 25 minutes and wrote 13 arrays under `/private/tmp/soromox-paper-secVa-generated`; four canonical-data plots rendered under `/private/tmp/soromox-paper-secVa` |
| Vc configuration-space comparison | Full | Setpoint, trajectory, and regulation-to-tracking NPZs generated from scratch; all plotters ran; Viser initialized headlessly with 1,501 frames |
| Vc operational impedance | Full | Canonical trajectory and full-vs-partial NPZs generated from scratch; plotters ran; Viser initialized headlessly with 1,001 frames |
| Vd control-gain optimization | Completed with scientific divergence | Full default three-iteration command wrote MAT, pickle, and five PDFs under `/private/tmp/soromox-paper-secVd`, but stopped after detecting NaNs at iteration 1 |
| Ve safety-constrained control | Full data/plot run; video not rerendered | Both controllers generated into `/private/tmp/soromox-paper-secVe`; PDF/PNG plotting succeeded; browser-backed Viser recording was not repeated |
| Vf parallel RL | Plot/video full; training unavailable | Reward plot rendered; full Open3D rollout produced a 301-frame, 20.0466 s, 2,423,925-byte H.264 MP4; training entry point/configuration is not present |

The Section Vf recorder originally encoded frames and then waited forever for
the viewer to close. `Open3DRenderer.render_sequence` now has an opt-in
`close_when_recording_done` mode. A five-frame smoke recording and the full
301-frame paper recording both terminate and decode successfully.

## Numerical comparison

| Artifact | Schema/inventory | Maximum difference | Interpretation |
| --- | --- | --- | --- |
| Vc setpoint scratch vs prior NPZ | Keys equal | abs 1020.2004 and rel 1.0357e7 in `s0_c2_qd` | Material divergence; relative maximum is amplified by near-zero prior values |
| Vc operational scratch vs prior NPZ | Keys and scenario metadata equal | abs 14.13955 and rel 1272.12 in `qd_traj` | Material divergence |
| Vc feedback-linearization repeat | NPZ byte-identical across two full runs | 0 | Deterministic on this host; regenerated PNGs were byte-identical to prior PNGs |
| Ve with CBF scratch vs canonical | Keys equal | abs 18.69498 in `u_ts`; max rel 6.20296 in near-zero `qd_ts` | Numerical rollout differs despite matching schema |
| Ve without CBF scratch vs canonical | Keys equal | abs 24.95466 in `u_ts`; max rel 193.838 in near-zero `qd_ts` | Numerical rollout differs despite matching schema |
| Va base identification scratch vs canonical | Matching parameter/error arrays have equal shapes | max abs 1.08e-7 for `E_hat`; other matching arrays at or below 1.79e-13 | Numerically reproducible on this CPU |

For the Vc setpoint case, prior PID strain RMSE values were
`[1.3972, 0.5422, 0.1664]`; scratch values are
`[3.4302, 1.3297, 0.0290]`. The scratch operational rollout has mean position
RMSE 1.609 mm and mean orientation RMSE 1.199 degrees, versus 0.379 mm and
approximately zero degrees in the prior trajectory. Visual inspection confirms
the changed setpoint transients and much larger operational orientation errors.

The full-vs-partial feedback comparison completed in 90.5 s and 80.7 s. Full
linearization position RMS/max errors were 3.36e-6/1.27e-5 mm and orientation
RMS/max errors were 5.46e-6/2.09e-5 degrees. Partial linearization position
RMS/max errors were 0.0152/0.0360 mm and orientation RMS/max errors were
0.1977/0.3944 degrees.

The Section Va base identifier selected step 41 with loss
`1.0514396e-03`, `E=315093.7702`, `nu=0.45`, and `rho=1317.6607`.
Matching committed parameter and error arrays reproduce to numerical precision;
the differently shaped curve/marker arrays belong to the separate residual
dataset described below.

## Unresolved issues

1. Section Vd's unmodified scientific settings do not produce a valid gain
   optimization on this host. Before optimization, the steady-state velocity
   norm is `4.48e-01`; after the 50.10 s compilation/first execution, all
   reported gains become NaN at iteration 1. The exact successful process
   command was the script with `--result-dir /private/tmp/soromox-paper-secVd/data
   --output-dir /private/tmp/soromox-paper-secVd/outputs --save-figures
   --no-show --no-render`; it exits zero despite the internal NaN warning.
2. The Section Va identification generator does not generate the residual-model
   arrays consumed by `plot_system_id_residual.py` (`curves_tau_star.npy`,
   `errnorm_*.npy`, `rmse_overall.npy`, and `violin_data.csv`). Those inputs are
   committed, but their producing workflow is still absent.
   Its base-identification `curves_orig.npy`, `markers_orig.npy`, and
   `measured_pts.npy` have 8 samples, while the residual dataset uses 22 samples
   with the same filenames. Generated base-identification files now default to
   `data/parameter_identification/` to prevent that collision.
3. Section Vc's scratch NPZs cannot reproduce the retained prior plots because
   the trajectories materially diverge. Both sets were intentionally kept in
   their requested roles: scratch data and prior visual artifacts.
4. Section Vc remains a visual-style exception. Its prior PDFs use DejaVu Sans;
   standardized sections use Computer Modern from `paper.mplstyle`. Updating Vc
   requires intentionally replacing its canonical figures under visual review.
5. GPU timing, MATLAB/SoRoSim generation, browser-backed Viser recording, and RL
   training could not be validated with the available hardware/code. No
   scientific parameters, gains, seeds, datasets, or benchmark definitions were
   changed to conceal these limitations.
