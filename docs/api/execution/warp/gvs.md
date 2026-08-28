# GVS Warp Integration API

!!! warning "Integration API"
    SoRoMoX users should call the public `GVS` kinematics and dynamics methods.
    The symbols on this page are intended for packages that reuse the GVS
    kinematics, dynamics, or actuation pipelines with preallocated Warp
    buffers.

## Linear-threadlike actuation

`GVSThreadlikeOperands.from_model(robot)` exposes the ordered built-in linear
routing data together with interior body quadrature, link strain bases and
reference strains, physical segment spans, link-local/global coordinate maps,
and the rotational-basis scaling convention. `GVSThreadlikeShapes` specifies
caller-owned matrix output `(E, D, A)` and fused-force output `(E, D)`.

The exact launch sequences are:

1. `launch_gvs_threadlike_actuation_matrix(...)` clears `(E, D, A)`, integrates
   each link-local path contribution, maps it to active coordinates, and leaves
   joint-only coordinates zero.
2. `launch_gvs_threadlike_actuation_force(...)` clears `(E, D)`, applies
   `coordinate_scale[path] * controls[e, path]` before path reduction, then maps
   the fused link-local force. It never materializes `(E, D, A)`.

Floating-point inputs and outputs are FP64; coordinate maps, the one-entry
rotational-scaling option, and path spans are INT32. Callers own and retain all
Warp views and result buffers. Neither launcher allocates, calls JAX,
synchronizes, or transfers data, and both clear-then-integrate sequences may be
captured in a CUDA graph. Environment, segment, link-DOF, quadrature, and path
dimensions remain runtime-shaped. The normalization test is exactly
`norm > global_eps`; a collapsed tangent contributes zero, matching JAX.

::: soromox.systems.execution.warp.gvs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members_order: source
      docstring_section_style: table
