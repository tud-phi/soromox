# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Write user-facing notes under the section that best describes the change. Keep
entries concise, link to relevant pull requests or documentation when useful,
and include benchmark baseline and measurement context for performance claims.

## [Unreleased]

### Added

- Added fused Warp forward-kinematics and inertial-Jacobian execution for
  `GVS`, `PCS`, and `PlanarPCS`, including spatial, environment, and combined
  batches through the explicit `*_abscissa_batched` methods and `jax.vmap`.
- Added optional Warp dynamics backends for `GVS`, `PCS`, and `PlanarPCS`,
  including automatic GPU selection, JAX-routed differentiation, typed
  `GVSBackendParams` and `PCSBackendParams` launch configuration, and reusable
  Warp-native execution APIs; see
  [Execution Backends](../user-guide/execution-backends.md) and
  [PR #177](https://github.com/tud-phi/soromox/pull/177).
- Added public allocation-free, CUDA-graph-capturable Warp kernels, operand
  bundles, output-shape contracts, and fused direct-effort launchers for
  built-in linear threadlike actuation in `PlanarPCS`, `PCS`, and `GVS`.
  Eligible `forward_dynamics` calls combine dynamics and threadlike actuation
  in one Warp invocation for all three systems.

### Changed

- Made effort laws declare their transmission-state dependencies so
  `DirectEffort` skips unused actuator-coordinate and velocity evaluation, and
  generalized-force assembly evaluates each actuator moment matrix only once.
- Made the system-method and parallel-rollout benchmarks device-selectable and
  reproducible by recording source, software, precision, accelerator, and CPU
  affinity metadata. The system-method benchmark adds optional steady-state
  warmup and reports median synchronized execution samples; the parallel-rollout
  generator defaults explicitly to GPU, refuses silent CPU fallback when no
  compatible GPU is available, and supports opt-in CPU measurements.

### Performance

- Added fused Warp kinematics kernels and measured 324 matched warmed cases per
  device across two- and eight-segment `PlanarPCS`, `PCS`, and `GVS` models,
  1/16/256 environments, and 1/8/64 abscissae. On an RTX 5090, Warp improved
  the geometric mean over JAX by 2.79×; the eight-segment, 256-environment,
  64-abscissa PCS and GVS Jacobians improved 4.73× (1.596 to 0.337 ms) and
  12.67× (7.706 to 0.608 ms). On an Intel Core Ultra 9 285K, the overall
  geometric mean was 1.00× and large PCS cases ranged from 0.57–0.95×,
  supporting the existing JAX-on-CPU `auto` policy.
  Measurements used FP64, JAX 0.11.0, Warp 1.16.0, separate first-call timing,
  two warmups, and the median of nine GPU or seven CPU synchronized repeats.
  An Nsight Systems trace of the large fused PCS case confirmed two Warp
  launches per call: approximately 69 microseconds for the cooperative
  recurrence and 233 microseconds for sample evaluation.
- Accelerated batched FP64 continuum dynamics on an RTX 5090 with Warp. For
  four-segment models at batch 256, order-1 GVS dynamics terms improved 3.02×
  (2.224 to 0.735 ms) and forward dynamics 2.40× (2.482 to 1.034 ms), while
  five-point PlanarPCS and PCS forward dynamics improved 2.21× and 1.66×.
  Warp is not a CPU optimization: single-environment order-5/9-point GVS terms
  were 2.46× slower (0.587 to 1.445 ms), so `backend="auto"` selects JAX on CPU;
  see [PR #177](https://github.com/tud-phi/soromox/pull/177).

- Accelerated JAX GVS dynamics-term assembly with compact strain bases,
  velocity reuse, and four bounded active-DOF prefix buckets. Against the
  previous full-width scan for a 16-segment, strain-order-5/9-Gauss-point model,
  warmed runtime fell 36.4% on single-environment CPU (12.65 to 8.04 ms) and
  35.7% on 256-environment GPU (146.76 to 94.32 ms); see
  [PR #174](https://github.com/tud-phi/soromox/pull/174).
- Refreshed the RTX 5090 parallel-rollout results for 216 matched configurations
  spanning 1–256 environments, with a 1.49× geometric-mean speedup over the
  previous committed measurements, and synchronized the publication figures.

### Deprecated

### Breaking changes

- Renamed abscissa-only batch APIs to identify their mapped argument:
  `forward_kinematics_batched(...)` is now
  `forward_kinematics_abscissa_batched(...)`, `jacobian_batched(...)` is now
  `jacobian_abscissa_batched(...)`, and the body-frame, inertial-frame, and
  Jacobian-time-derivative variants follow the same `*_abscissa_batched`
  convention. The old names have been removed.
- Moved generic actuator-visualization adaptation out of
  `SoftRobot.actuator_visual_layers(...)` and into
  `soromox.rendering.actuator_visual_layers(...)`. Renderers continue to adapt
  standard `SoftRobot` systems automatically, while robot-defined hooks remain
  available for specialized visualization behavior.

### Fixed

- Fixed GVS abscissa-batched partial-cell poses and inertial Jacobians so
  rotational strain-basis values consistently honor
  `scale_rotational_basis_by_length` for every supported basis family.

### Documentation

- Added an [Execution Backends](../user-guide/execution-backends.md) guide, a
  backend-selection example in the Quick Start, and API reference pages for
  execution configuration and the reusable GVS and PCS Warp interfaces.

### Contributors

## [0.3.0] - 2026-08-25

### Added

- Shared continuum-component APIs under `soromox.systems.components` for link,
  joint, cross-section, and isotropic-material parameters and specifications.
- Common `from_links`, `params_from_links`, `update_link_params`,
  `link_matrices_from_material`, and `with_isotropic_material` workflows for
  PCS and PlanarPCS, with the same material/update APIs on GVS.
- GVS joint stiffness and damping parameters that participate in global matrix
  assembly, plus `update_joint_params` for immutable joint-local updates.
- Differentiable unit-response mappings from Young's modulus with either shear
  modulus or Poisson's ratio, plus optional material damping, to canonical
  generalized link matrices.
- Continuum-component, parameter-update, material-optimization, and
  [parameter-API migration](../user-guide/parameter-api-migration.md)
  documentation with complete PCS and GVS examples.

- Compiled parameter-update examples and regression coverage for every public
  system family, routed actuators, passive elements, gradients, eager cache
  refreshes, static-structure rejection, and no-retrace same-layout updates.

### Changed

- Accelerated PlanarPCS, PCS, and GVS forward dynamics through fused dynamics
  assembly, causal Jacobian propagation, and matrix-free Lie-algebra actions.
  Compared with v0.2.3 across one-environment CPU and 256-environment GPU
  benchmarks with 1, 4, 8, and 16 segments, warm runtime improved in every
  measured case: 1.06–2.21x for PlanarPCS, 1.32–8.28x for PCS, and 1.30–5.36x
  for GVS. At 16 segments on GPU, temporary memory decreased from 459 MB to
  95 MB, 1.73 GB to 270 MB, and 3.02 GB to 128 MB, respectively.
- Routed exact inverse-dynamics controllers through the contracted
  `dynamics_terms` interface and fused actuation- and operational-space term
  transformations to avoid repeated inertia, Coriolis, gravity, Jacobian, and
  matrix-factorization work. Across affected PCS and GVS controller paths with
  one and four segments, this reduced runtime by 38.3% on a geometric-mean
  basis (up to 55.9%).

### Fixed

- Made the Open3D viewer's Q/Esc shortcuts close windows cleanly so execution
  continues after the viewer exits.
- Made bound-aware finite-difference Jacobians compatible with JAX immutable
  arrays and documented the `numerical_jacobian` utility.

### Contributors

- Thanks to [@matteodv99tn](https://github.com/matteodv99tn) for his
  contributions to [PR #156](https://github.com/tud-phi/soromox/pull/156).
- Thanks to [@ivrolan](https://github.com/ivrolan) for his contributions to
  [PR #162](https://github.com/tud-phi/soromox/pull/162).
- Thanks to [@mohammedtarnini](https://github.com/mohammedtarnini) for his
  suggesting the improvement implemented in
  [PR #166](https://github.com/tud-phi/soromox/pull/166).

## [0.2.3] - 2026-08-19

### Added

- Reverse-mode-safe numerical primitives, strict singularity diagnostics, and
  pytree finiteness reporting for locating and handling degenerate model
  configurations.
- Public, dtype-aware ``jacobian_coefficients`` for forward and inverse
  Lie-group Jacobians, together with a reproducible Lie-algebra benchmark
  covering primal, Jacobian, Hessian, and analytic constant-strain derivative
  paths.
- Public ``constant_strain.se2.operators`` and
  ``constant_strain.se3.operators`` bundles with a fixed named result for
  callers that need adjoints, tangents, and an optional analytic tangent
  derivative at the same strain and arclength.
- Public ``se2`` and ``se3`` left Jacobians and analytic directional
  derivatives for accumulated Lie-algebra coordinates, independent of any
  constant-strain rod interpretation.
- Public ``constant_strain.se3.tangent_and_derivative`` for callers that need
  the selective tangent pair without constructing unused adjoint operators.
- An offline Section Vd post-processing renderer for saved optimization
  trajectories, with reproducible tracking plots, MP4 animations, optional GIF
  conversion, and intermediate metrics.

### Changed

- Harmonized PCS, PlanarPCS, and GVS around nested `params.link` ownership,
  descriptive public construction names, shared link/joint specifications, and
  canonical per-link generalized stiffness and damping matrices.
- Moved shared `LinkSpec` and `JointSpec` functionality out of the GVS package;
  GVS now retains only segment, strain-basis, quadrature, and runtime concepts
  specific to variable-strain mechanics.
- Simplified cross-section parameters to coefficient arrays populated by link
  factories; constant values and `LinearProfile` cover constant and linearly
  varying geometry without a profile-parameter class hierarchy.
- Made isotropic material parameters caller-owned construction/optimization
  PyTrees instead of duplicating material and generalized-matrix
  representations inside system parameters.
- Made geometry updates refresh material unit-response operators without
  silently replacing explicitly supplied canonical matrices.
- Reworked ``SO(3)``, ``SE(2)``, and ``SE(3)`` exponential/logarithmic maps,
  constant-strain adjoint and tangent operators, and PlanarPCS pose integration
  around stable closed forms and high-order near-zero series. Constant-strain
  tangent derivatives remain explicit analytic production paths; runtime
  autodiff is used only as a test oracle.
- Organized constant-strain operators into dedicated ``se2`` and ``se3``
  modules with unsuffixed names. General left-Jacobian machinery now belongs
  to the rigid-body Lie-algebra modules; the constant-strain package retains
  arclength scaling, prepared powers, rod-specific adjoints, and the shared
  ``ConstantStrainOperators`` result.
- Fused repeated constant-strain operator evaluation in PlanarPCS, PCS, and
  GVS. The planar path uses direct SE(2) blocks; the spatial operators share
  exact fourth-order matrix powers and tangent coefficients. PCS additionally
  prepares arclength-independent powers once per segment, assembles transported
  tangents directly, and propagates the contracted ``Jd @ qd`` needed by
  forward dynamics instead of materializing unused matrices. PlanarPCS and GVS
  now use the same contracted-derivative strategy in their dynamics-only
  recurrences while preserving full Jacobian derivatives in public kinematics.
- Replaced generated symbolic PlanarHSA expressions with a JAX-native
  PlanarPCS implementation, reusing shared planar kinematics, Jacobians,
  quadrature, and dynamics; migrated parameter files to the descriptive
  ``reference_strain`` schema and removed the obsolete symbolic and
  legacy-parameter modules.
- Compared with ``main`` on CPU/JAX 0.11.0 across five fresh processes, the
  40-case Lie benchmark reduced geometric-mean steady-state and compile time by
  24.3% and 5.3%. A fresh raw before/after table is reported in this PR for
  PlanarPCS, PCS, and GVS.

- Updated the dependency metadata and lockfile for a CUDA 13-enabled JAX and
  PyTorch stack, and declared `ipywidgets` with the optional Open3D extras.
- Simplified rendering, example, paper-result, and complete installations by
  removing an unused video-encoding package; video output continues to use the
  system `ffmpeg` executable.
- Consolidated the Section Vd control-gain optimization scripts around a shared,
  testable optimization loop and independently configured collocated and
  synergistic cases, with shared CLI/output helpers, explicit output locations,
  overwrite protection, paper-result iteration defaults, ``gamma`` derived from
  a configurable tendon-length error scale for collocated integral saturation
  (default 10 mm), finite saturation-parameter validation, stale-artifact
  documentation, and consistent NumPy archive, pickle, and diagnostic-figure
  handling. Diagnostic figures are always saved,
  the obsolete ``--save-figures`` switch is no longer needed, and the primary
  optimization data is persisted before plotting or interactive rendering
  begins; saved trajectory diagnostics can be rendered without rerunning
  optimization.
- Normalized explicit continuous-rollout `save_ts` sequences to JAX arrays and
  preserved traced initial times for JIT-compiled and vectorized open- and
  closed-loop rollouts.

### Fixed

- Made same-PyTree, same-shape, same-dtype system, environment, actuator,
  passive-element, and isotropic-material replacements usable inside
  JIT-compiled code. Concrete eager calls retain value validation, while traced
  calls enforce the statically observable type and shape contract.
- Split parameter validation into explicit `validate_structure()` and
  `validate_values()` subclass hooks, with framework-owned eager and traced
  dispatch for intrinsic and model-structure-dependent validation. Closed-over
  concrete values are checked at JAX trace time, while dynamic leaves retain
  structure-only validation. Component-to-robot checks use the same structural
  and concrete-value phases.
- Reused I-SUPPORT routing spans and McKibben joint-pair topology during
  compiled numeric updates, and kept GVS cross-section index dtypes stable when
  refreshing parameter-dependent caches.
- Included stored GVS joint stiffness and damping in global stiffness and
  damping assembly.
- Resolved omitted GVS padding values in the advanced `GVS(params, structure)`
  constructor and aligned rectangular geometry output with the shared
  `[height, width]` cross-section convention.
- Normalized caller-provided isotropic material sequences to JAX arrays and
  rejected non-finite or nonphysical link geometry, density, and material
  values during concrete construction and replacement.
- Migrated examples, benchmarks, and paper case studies to the harmonized PCS
  and GVS parameter APIs.
- Kept values and reverse-mode gradients finite at degenerate normalization,
  division, square-root, inverse-kinematics, and thread-routing sites across
  PCS, GVS, Planar HSA, articulated, McKibben, rotation, and reference-trajectory
  code paths.
- Sanitized inactive square-root and quotient inputs in SO(3)/SE(3) magnitude
  and quaternion-conversion branches, preventing batched selection from
  exposing NaNs under nested JVP/reverse-mode transforms; the SoftRobot
  forward-kinematics and spatial-Jacobian regressions from PR #148 now pass.
- Kept Lie exponential/logarithmic values, first derivatives, and Hessians
  finite and accurate at zero and small rotations, including batched SE(2) and
  arbitrary-axis SE(3) inputs, without adding autodiff to production paths.
- Made simulation benchmark timing treat Python solver/save parameters as static
  JAX arguments so the existing rollout benchmark can compile and run.
- Prevented automatically generated continuous-rollout save grids from
  overshooting the requested final time, while always including that endpoint
  exactly for both divisible and non-divisible save intervals.
- Set the Section Vd collocated and synergistic optimization defaults to the
  100 iterations used by the paper-result histories, and document that setting
  in the reproduction commands.
- Corrected Section Vd loss/parameter history pairing, rejected non-finite
  optimization candidates, and selected the best collocated and synergistic
  plotting batches independently.
- Guarded the collocated optimization reference against transient setpoints and
  collapsed tendon paths by using stable asymmetric tensions and failing fast
  when the generated steady state is unsuitable for reverse-mode optimization.
- Reported Open3D window-creation failures explicitly, with a clear error for
  environments that lack a usable display or OpenGL context.

### Breaking changes

- Removed flat PCS and PlanarPCS link/material fields. Access link data through
  `params.link`, and construct systems with `from_links` or
  `params_from_links`.
- Removed the global PCS damping matrix and cross-link damping coupling. Supply
  one generalized damping block per link.
- Removed `GVSLinkParams` and GVS-local exports of shared link, joint, and
  cross-section specifications. Import shared types from `soromox.systems` or
  `soromox.systems.components`.
- Replaced abbreviated GVS construction fields such as `E`, `nu`, `rho`,
  `eta`, `L`, `r_i`, and `r_f` with descriptive material, geometry, and profile
  arguments.
- Moved `reference_strain` from `StrainBasisSpec` to `LinkSpec`; basis specs now
  describe only the selected strains and basis order.
- No compatibility aliases are provided. See the compact
  [PCS/GVS parameter migration guide](../user-guide/parameter-api-migration.md)
  for direct replacements.

## [0.2.2] - 2026-08-11

### Added

- Configurable, base-aligned ground planes for the Matplotlib, Open3D, and Viser
  renderers, including inherited support in the I-SUPPORT and UMArm renderers.
- A top-level Paper & Results documentation page with CPU and GPU benchmarks,
  all six application case studies, reproduction pointers, and curated
  web-optimized figures, animations, and videos from the SoRoMoX paper.
- A responsive core-renderer gallery, system-specific rendering samples, and a
  concise top-level contributor guide.

### Changed

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
- Separated Section Vf reinforcement-learning rollout generation from rendering:
  trained and uniform-random baseline trajectories now retain the full parallel
  batch under `data/traj`, while the shared offline renderer derives single-arm
  or automatically framed grid MP4/GIF names from its input data under
  `outputs` and uses the paper's visual style. Renamed the canonical artifacts
  around policy state and environment count, versioned the current trajectory
  and render artifacts, and moved Section Vf MP4/GIF storage to Git LFS.
- Applied the shared camera field of view and tighter scene framing to
  Matplotlib's 3D renderer, bringing its gallery view closer to Open3D and
  Viser while retaining metric axes.
- Standardized the renderer gallery around consistent robot geometry, colors,
  ground planes, and framing, with tighter I-SUPPORT and UMArm close-ups.
- Refocused the README and documentation home on the JAX-native model
  implementations for articulated and continuum soft robots, their
  control-oriented interface, installation, performance, and first use.
- Reorganized the rendering documentation into an overview and gallery,
  backend API guide, and shared configuration reference; normalized Python
  examples throughout the generated package documentation as fenced code.
- Made the SoRoMoX arXiv preprint the primary academic citation, centralized
  citation guidance, and retained an exact-release software citation for
  reproducibility-sensitive work.
- Kept Citation at the top level of the documentation navigation immediately
  after Paper & Results, and grouped the supported systems following the paper.
- Placed Actuation before Control in the API navigation and removed the
  outdated symbolic-derivation documentation page.
- Moved detailed contributor and documentation tooling into the development
  guide and made `VERSION_BUMP_README.md` the authoritative maintainer release
  workflow, including the branch, pull-request, tag, and recovery procedures.
- Added Twine to the development dependency set so the documented distribution
  validation command is available in the project environment.
- Separated backend-specific renderer coverage into dedicated Matplotlib,
  Open3D, and Viser test modules, leaving the shared base-renderer tests focused
  on backend-independent behavior.
- Removed the obsolete duplicate actuation overview page in favor of the
  structured actuation documentation section.

### Fixed

- Renamed the operational-space Viser renderer test module and made Section Vf
  tests skip cleanly when optional paper dependencies are unavailable,
  preventing full-suite collection failures.
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
