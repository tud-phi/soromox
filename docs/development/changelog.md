# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Write user-facing notes under the section that best describes the change. Keep
entries concise, link to relevant pull requests or documentation when useful,
and include benchmark baseline and measurement context for performance claims.

## [Unreleased]

### Added

- Added renderer-neutral cross-section contours and loft construction, with a
  registration hook for extending both swept 3D renderers with new geometries.

### Changed

- Open3D and Viser swept backbones now preserve circular, elliptical, and
  rectangular cross-sections, including dimensions that vary with abscissa.

### Performance

### Deprecated

### Breaking changes

- Corrected the `PlanarPCS` threadlike routing convention so a positive
  material-`y` offset shortens under positive `kappa_z`. Existing planar
  threadlike models that compensated for the previous sign must update their
  routing offsets or actuator directions.
- Renamed the Viser `cylinder_sections` and Open3D `tube_resolution` options to
  the geometry-neutral `cross_section_resolution`.

### Fixed

- Made Viser discrete cross-section markers consistent with Open3D: circular,
  rectangular, and elliptical sections now use spheres, boxes, and ellipsoids,
  respectively.
- Fixed Open3D and Viser lofting across link boundaries, which produced a short
  shape-morphing funnel when adjacent links used different cross-sections.
- Fixed Open3D swept meshes reusing the first endpoint cross-section at both
  ends, which previously rendered tapered profiles as piecewise-constant.

### Documentation

### Contributors

## [0.4.0] - 2026-09-01

### Added

- Added explicit JAX and Warp execution backends for `PlanarPCS`, `PCS`, and
  `GVS`, with automatic backend selection, typed launch parameters, reusable
  APIs under `soromox.execution`, and arbitrary positive PCS quadrature counts.
  The Warp implementation provides fused forward-kinematics, full inertial
  Jacobian, dynamics-term, and forward-dynamics paths; allocation-free,
  CUDA-graph-capturable launchers; and direct-effort fusion for built-in linear
  threadlike actuation.
- Added native matrix-free Warp products for `J(q, s) @ qd` and
  `sum_s J(q, s).T @ wrench_s`. These paths reuse caller-owned workspace and
  floating-frame state instead of materializing a Jacobian at every spatial
  sample.
- Added fixed and floating-base configurations across all system families,
  including planar and quaternion base poses, distinct configuration and
  velocity dimensions, world-frame base velocities and wrenches, vectorized
  JAX composition, and specialized Warp execution; see
  [Floating-base systems](../user-guide/floating-base-systems.md).

### Changed

- Reworked the execution layer around shared quaternion, `SO(2)`, `SO(3)`,
  `SE(2)`, and `SE(3)` primitives and explicit operand/output contracts, so the
  same public model methods can dispatch to optimized JAX or Warp paths without
  changing the model definition.
- Reduced GVS dynamics assembly to compact strain bases and bounded active-DOF
  prefixes, and reused intermediate velocity and floating-root rotation terms.
- Made effort laws declare which transmission states they need. `DirectEffort`
  now skips unused actuator-coordinate and velocity evaluation, and each
  actuator moment matrix is assembled only once.
- Made the system-method and parallel-rollout benchmarks device-selectable and
  reproducible. Results now record source revision, software versions,
  precision, accelerator, CPU affinity, warmup, and synchronized sample data;
  GPU requests no longer fall back silently to CPU.

### Performance

- Fresh FP64 comparisons against v0.3.0 used Python 3.12.13, JAX 0.11.0, and
  Warp 1.16.0. CPU runs were restricted to an otherwise-idle performance core
  of an Intel Core Ultra 9 285K with all library thread counts set to one; GPU
  runs used an otherwise-idle NVIDIA RTX 5090. The main grid covered
  `PlanarPCS`, `PCS`, and order-1 `GVS`; 1, 2, 8, 16, and 32 segments; batches
  1, 16, 64, 256, and 1024; five-point quadrature; and forward kinematics,
  full inertial Jacobians, fused pose-plus-Jacobian evaluation, JVP, VJP,
  dynamics terms, and forward dynamics. Kinematics cases evaluated 60 spatial
  samples per system, whereas each dynamics case evaluated one batched state,
  so their absolute times are not directly comparable. Results are medians of
  seven synchronized samples after five warmups.
- Relative to v0.3.0 across the 25 segment/batch combinations, JAX GVS
  dynamics terms and forward dynamics improved by geometric means of 1.17× and
  1.14× on CPU and 1.10× and 1.08× on GPU. At 32 segments and batch 1024,
  terms/forward time fell 15.5%/14.1% on CPU and 21.2%/17.7% on GPU. GVS
  kinematics, JVP, and VJP remained near parity, as did the corresponding
  release-over-release `PlanarPCS` and `PCS` paths.
- An eight-segment discretization sweep used batches 1, 256, and 1024; three-
  and nine-point quadrature for `PlanarPCS` and `PCS`; and GVS strain-order/
  quadrature pairs 0/5, 3/7, and 5/9. At batch 1024, order-5 GVS JAX dynamics
  terms/forward dynamics improved 15.0%/14.1% on CPU and 29.8%/22.8% on GPU
  relative to v0.3.0. `PlanarPCS` and `PCS` quadrature sweeps showed no
  consistent release-over-release shift beyond run-to-run variation.
- For the main grid, current Warp versus current JAX improved full inertial
  GVS Jacobians by a 3.38× geometric mean on CPU and 10.88× across the 24
  matched GPU cases; fused GVS pose-plus-Jacobian evaluation improved 3.37×
  and 11.17×. GPU Warp also
  improved full Jacobians by 1.35× for `PlanarPCS` and 2.53× for `PCS`. Across
  all 25 matched GPU segment/batch cases, Warp/JAX geometric-mean speedups for
  dynamics terms/forward dynamics were 1.00×/1.62× for `PlanarPCS`,
  0.94×/1.16× for `PCS`, and 2.00×/1.77× for `GVS`; the value below one means
  that PCS dynamics terms alone were about 7% slower under Warp even though PCS
  forward dynamics was 16% faster. Individual terms/forward cases ranged from
  0.22–3.29×/0.89–2.80× for `PlanarPCS`, 0.33–2.47×/0.33–2.24× for `PCS`, and
  0.35–5.53×/0.41–4.44× for `GVS`. At batch 256 across the five segment counts,
  the corresponding geometric means were 1.11×/1.93×, 1.25×/1.43×, and
  2.70×/2.22×. The minima principally expose fixed launch cost for one-segment
  terms and under-occupied persistent chain kernels at 32 segments and batch 1;
  the maxima occurred at batch 1024. At 32 segments and batch 1024, the dense
  JAX GVS Jacobian case exhausted the RTX 5090's 32-GB memory while Warp
  completed it; Warp also improved terms/forward time by 1.27×/1.20×.
  In the quadrature sweep, GPU Warp terms/forward geometric-mean speedups were
  1.93×/2.66× at three points and 1.87×/1.80× at nine points for `PlanarPCS`,
  and 1.46×/2.05× and 1.42×/1.78× for `PCS`. Small GPU workloads can still
  favor JAX.
- CPU Warp is primarily a dense-kinematics optimization: its full Jacobian
  geometric means improved 1.46× for `PlanarPCS`, 1.50× for `PCS`, and 3.38×
  for `GVS`, but JAX was 1.82×/1.78× faster overall for GVS dynamics terms /
  forward dynamics. In the high-order sweep at batch 1024, Warp improved
  order-0 terms/forward dynamics by 1.38×/1.34×, but was 1.46×/1.41× slower at
  order 3 and 1.79×/1.70× slower at order 5. The CPU `auto` policy therefore
  continues to select JAX. On GPU at the same batch, Warp's terms/forward
  speedups tapered from 4.25×/3.68× at order 0 to 2.02×/1.64× at order 3 and
  1.21×/1.14× at order 5; the smallest order-5 cases still favored JAX.
- Native matrix-free Warp JVP/VJP was compared with dense Warp Jacobian
  materialization at eight segments, 60 samples, and batches 1, 16, 64, 256,
  and 1024. CPU JVP/VJP speedups ranged from 1.60–4.22×/1.31–5.47× for
  `PlanarPCS`, 4.05–6.90×/3.02–5.18× for `PCS`, and
  3.27–5.58×/1.59–1.98× for `GVS`; GPU ranges were
  3.43–4.62×/3.22–4.67×, 1.77–7.22×/1.91–5.15×, and
  2.08–7.89×/2.00–5.29×, respectively. Directional workspace was 21.5×,
  43.0×, and 83.6× smaller than dense materialization. These measurements used
  five warmups and the median of seven runs with 20 synchronized iterations.

### Breaking changes

- Changed all SoRoMoX quaternion coordinates to the scalar-last Hamilton
  convention `[qx, qy, qz, qw]`, including spatial base poses, operational-space
  poses, floating-base state, and JAX and Warp execution. Code that constructs
  or indexes quaternions directly must use the new coordinate order.
- Replaced the ambiguous `SoftRobot.num_dofs` attribute with explicit
  `num_internal_dofs`, `num_coordinates`, and `num_velocities` dimensions.
  Code that allocates configurations, velocities, or internal controller state
  must now select the corresponding dimension explicitly.
- Removed mounting poses from physical parameter PyTrees. Constructors retain
  the `base_pose` argument for fixed mounting, models expose the resolved value
  as `fixed_base_pose`, and `with_fixed_base_pose(...)` updates a fixed
  mounting independently of `with_params(...)` and `update_params(...)`.
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

- Fixed forward-kinematics, Jacobian, and energy custom JVPs to preserve
  derivatives with respect to dynamic model leaves, including fixed mounting
  poses and gravity updated inside compiled functions, across JAX and
  accelerated execution paths.
- Fixed GVS abscissa-batched partial-cell poses and inertial Jacobians so
  rotational strain-basis values consistently honor
  `scale_rotational_basis_by_length` for every supported basis family.

### Documentation

- Added an
  [Execution Devices and Backends](../user-guide/execution-devices-and-backends.md)
  guide, a backend-selection example in the Quick Start, and API reference
  pages for execution configuration and the reusable GVS and PCS Warp
  interfaces.

### Contributors

- Thanks to [@vdperfetta01](https://github.com/vdperfetta01) for sustained
  review of the GVS optimization, execution-backend, floating-base, and API
  changes in [PR #174](https://github.com/tud-phi/soromox/pull/174),
  [#177](https://github.com/tud-phi/soromox/pull/177),
  [#181](https://github.com/tud-phi/soromox/pull/181),
  [#183](https://github.com/tud-phi/soromox/pull/183),
  [#184](https://github.com/tud-phi/soromox/pull/184),
  [#185](https://github.com/tud-phi/soromox/pull/185), and
  [#186](https://github.com/tud-phi/soromox/pull/186), including identifying a
  test failure before merge.
- Thanks to [@mohammedtarnini](https://github.com/mohammedtarnini) and
  [@Johannap1](https://github.com/Johannap1) for reviewing the GVS assembly
  optimization in [PR #174](https://github.com/tud-phi/soromox/pull/174).

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
