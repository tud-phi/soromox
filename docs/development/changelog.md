# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A top-level Paper & Results documentation page with CPU and GPU benchmarks,
  all six application case studies, reproduction pointers, and curated
  web-optimized figures, animations, and videos from the SoRoMoX paper.
- Configurable, base-aligned ground planes for the Matplotlib, Open3D, and Viser
  renderers, including inherited support in the I-SUPPORT and UMArm renderers.
- A responsive core-renderer gallery, system-specific rendering samples, and a
  concise top-level contributor guide.

### Changed

- Made the SoRoMoX arXiv preprint the primary academic citation, centralized
  citation guidance, and retained an exact-release software citation for
  reproducibility-sensitive work.
- Refocused the README and documentation home on the JAX-native model
  implementations for articulated and continuum soft robots, their
  control-oriented interface, installation, performance, and first use.
- Kept Citation at the top level of the documentation navigation immediately
  after Paper & Results, and grouped the supported systems following the paper.
- Reorganized the rendering documentation into an overview and gallery,
  backend API guide, and shared configuration reference; normalized Python
  examples throughout the generated package documentation as fenced code.
- Applied the shared camera field of view and tighter scene framing to
  Matplotlib's 3D renderer, bringing its gallery view closer to Open3D and
  Viser while retaining metric axes.
- Placed Actuation before Control in the API navigation and removed the
  outdated symbolic-derivation documentation page.
- Separated Section Vf reinforcement-learning rollout generation from rendering:
  trained and uniform-random baseline trajectories now retain the full parallel
  batch under `data/traj`, while the shared offline renderer derives single-arm
  or automatically framed grid MP4/GIF names from its input data under
  `outputs` and uses the paper's visual style. Renamed the canonical artifacts
  around policy state and environment count, versioned the current trajectory
  and render artifacts, and moved Section Vf MP4/GIF storage to Git LFS.
- Improved Open3D and Viser efficiency for multi-robot animations with
  renderer-owned vectorized actuator geometry, batched actuator and sphere
  scene objects, automatically merged multi-robot Open3D backbones, and
  instanced or merged Viser geometry with one atomic transaction per frame.
  Viser ground planes now follow the complete batched layout. This substantially
  reduces backend registrations and browser scene handles without adding
  trajectory-rendering concerns to soft-robot mechanics. In a 16-robot,
  80-point, eight-frame
  benchmark, merged Open3D improved initial/frame rendering by 21.0x/2.7x
  (6.51 s to 0.31 s and 41.2 ms to 15.1 ms), while batched Viser improved full
  swept-video/browser-capture time by 2.3x/3.6x (27.9 s to 12.0 s and 2.98 s
  to 0.83 s) with 38.5x fewer live handles.
- Separated backend-specific renderer coverage into dedicated Matplotlib,
  Open3D, and Viser test modules, leaving the shared base-renderer tests focused
  on backend-independent behavior.
- Moved detailed contributor and documentation tooling into the development
  guide and made `VERSION_BUMP_README.md` the authoritative maintainer release
  workflow, including the branch, pull-request, tag, and recovery procedures.
- Standardized the renderer gallery around consistent robot geometry, colors,
  ground planes, and framing, with tighter I-SUPPORT and UMArm close-ups.
- Removed the obsolete duplicate actuation overview page in favor of the
  structured actuation documentation section.

### Fixed

- Corrected the I-SUPPORT dynamic-model DOI and added platform references to
  the I-SUPPORT and McKibben UMArm system pages.

## [0.2.1] - 2026-08-03

### Added

- Section-scoped `paper_results/` workflows that colocate generation and
  plotting code, source or generated data, canonical figures and videos, and
  reproduction instructions for the results reported in the paper.
- Reproducible PCS configuration- and operational-space control case studies
  with shared simulation, plotting, serialization, Viser, phase-separated RMSE,
  and canonical paper-figure workflows.
- A shared Matplotlib publication style and case-specific figure composers for
  consistent typography, sizing, colors, labels, grids, and legends.
- Focused Section Vc tests for case-relative paths, NPZ schemas, overwrite
  protection, plotting, controller setup, trajectory primitives, and rendering.
- Complete Section IVa, IVb, Va, Vd, Ve, and Vf generation workflows alongside
  their paper data and outputs, including released RL checkpoints.
- Reproducible Section Vc evaluation metrics for all six configuration
  coordinates and geometric operational-space pose tracking, with canonical
  NPZ and JSON artifacts and a human- or machine-readable reporting CLI.
- A publication-sized Section Vc composite figure combining configuration-space
  comparisons, operational-space tracking and errors, and chronological
  experiment snapshots in committed PDF and SVG outputs.

### Changed

- Separated lightweight demonstrations under `examples/` from complete
  publication reproduction workflows and datasets under `paper_results/`.
- Made generated example figures and videos topic-local instead of writing them
  to the repository root, and organized paper sections into independent
  `code/`, `data/`, and `outputs/` ownership.
- Made affected example and paper entry points independent of the current
  working directory, added configurable artifact paths and overwrite guards,
  and migrated visualization workflows to the current renderer APIs.
- Standardized the Section Vc paper figures around full-width
  configuration-space tracking and partial-feedback-linearization
  operational-space tracking, with explicit strain labels and prominent
  reference trajectories.
- Store trained model checkpoints through Git LFS while leaving generated
  example artifacts and noncanonical paper diagnostics ignored.
- Made the built-in `tanh` PID integral-error saturation unit preserving as
  `Gamma^-1 tanh(Gamma e)`, retained unit slope at the origin, and validated
  scalar, vector, and matrix saturation parameters against their required
  positivity and shape constraints.
- Applied separate physical integral-error saturation scales to rotational and
  linear Section Vc strains, regenerated the canonical data and figures, and
  added steady-state regulation MAE over the final 10% of each evaluated
  setpoint interval.
- Refactored Viser sequence rendering around deterministic scene and frame
  hooks, synchronized video and lossless snapshot capture, explicit snapshot
  validation, and a render-only workflow for the operational-space paper scene.

### Fixed

- Rejected paused or looping playback modes for finite Viser video exports.
- Included `cbfpy` in the test dependency set so the Section Ve rendering tests
  collect across every supported Python version.
- Corrected the documented Section Vc steady-state evaluation window from the
  final quarter to the final 10% (0.3 s) of each 3 s setpoint interval.

## [0.2.0] - 2026-07-29

### Added

- First SoRoMoX package release published to PyPI.
- `ArticulatedSoftRobot` spatial serial-chain system with optional joint
  stiffness and damping, dense matrix dynamics, ABA forward dynamics, examples,
  documentation, and benchmark coverage.
- Geometric Variable-Strain (GVS), tendon-actuated GVS, I-SUPPORT, and
  McKibben-actuated UMArm models.
- Model-based control, system-identification, residual-learning, reinforcement
  learning, rendering, and CPU/GPU benchmarking examples.
- Open3D and Viser renderers with trajectory, actuator, and video support.

### Changed

- Generalized continuum actuation around composable threadlike actuation models.
- Standardized system, geometry, Lie algebra, operational-space, actuation-space,
  parameter, and base-pose APIs.
- Declared support for Python 3.11 through 3.14 and migrated the documentation
  build to Zensical.
- Set JAX 0.10.0 and Diffrax 0.7.2 as the minimum numerical dependency
  versions validated by CI.
- Expanded the software authorship and citation metadata.

### Fixed

- Corrected dynamics, Jacobian, rendering, control, rollout, and multi-segment
  I-SUPPORT behavior across the supported systems.

## [0.1.0] - 2025-09-01

### Added

- Initial SoRoMoX development milestone.

### Features

- Fast, parallelizable, and differentiable soft robot models implemented in JAX
- Implemented systems:
  - N-link pendulum
  - Planar Piecewise Constant Strain (PCS) continuum soft robot
  - Spatial Piecewise Constant Strain (PCS) continuum soft robot
  - Planar Handed Shearing Auxetics (HSA) robot

---

## Version History

- **v0.1.0**: Initial release with core functionality
<!-- - **Future**: Planned additions include 3D robot models, additional soft robot types, and enhanced visualization tools -->
