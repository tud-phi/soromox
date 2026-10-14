# Helicoidal

3D tendon-driven soft robots with helicoidal lattice structures.

## Overview

`Helicoidal` specializes GVS for effective helicoidal lattice mechanics. It uses
native segment specifications, strain bases, quadrature, and contiguous tendon
routing. Shared GVS mechanics support fixed and floating bases in JAX and Warp,
with distributed lattice inertia and attached bodies.

## Construction and Constitutive Mechanics

Use `Helicoidal.from_segments(...)` with `GVSSegment` specifications,
`mass_per_length`, and `rotational_inertia_per_length`. Supply one entry per link
for the line mass [kg/m], three principal rotational inertias [kg m], and optional
`tip_mass` [kg]. Zero inertia is supported for geometric connector links with
inactive strain bases. The line mass and rotational inertias specify lattice inertia. Tip masses have translational inertia only.

Supply a `HelicoidalSectionParams` object through `constitutive_law=` or
`with_constitutive_law(...)`. Its compliance coefficients describe the effective
lattice response in the local **backbone material frame**, with local x
longitudinal and y/z transverse. Individual helical strands' material properties
must first be reduced to this equivalent-rod law. The sectional order is
`[twist, bend_y, bend_z, extension, shear_y, shear_z]`:

```text
C(s) = compliance + (L - s) * compliance_gradient
K_section(s) = diag(1 / C(s))
K_link = integral B(s).T @ K_section(s) @ B(s) ds
D_link = tau * K_link
```

Here `s` is reference backbone arc length measured from the link's proximal end,
and `L` is its reference length. `compliance_gradient` differentiates the
backbone-frame compliance components with respect to `L-s`, so
`dC/ds = -compliance_gradient`. The damping time constant `tau` is a frame-independent
scalar per link: it scales the same backbone sectional stiffness before projection.

The sectional constitutive law is evaluated directly at GVS Gauss nodes, using the
configured basis and optional rotational length scaling. Both coefficient arrays
have shape `(num_links, 6)` in native strain order. Rotational compliance has
units 1/(N m²); translational compliance has units 1/N. Gradient coefficients
carry one additional inverse metre. All six compliances must be positive
throughout each link.

Distributed and attached-body matrices use the shared
[continuum-link inertia representation](../continuum-components.md#distributed-and-attached-body-inertia).

## Parameter Updates

Canonical generalized stiffness and damping live in `robot.params.link`, as in
GVS. Changes to other parameters preserve these matrices. Reproject explicitly
when changing lengths or constitutive coefficients:

```python
robot = robot.update_link_params(length=1.05 * robot.params.link.length)
robot = robot.with_constitutive_law(constitutive_law)
robot = robot.update_link_params(tip_inertia=1.1 * robot.params.link.tip_inertia)
```

These operations support JAX differentiation. Passing updated models into the
same compiled function, or updating parameters within a compiled function,
reuses compilation when topology, array shapes, and dtypes are fixed. All
links created by `Helicoidal.from_segments` have allocated tip inertia matrices,
so changing zero to a positive mass is a numeric update. Basis choices, link counts, quadrature counts, and
tendon spans are static. As with other system parameters, physical value
validation occurs for eager inputs; transformed code must preserve their domains.

## Specialized Viser Rendering

`HelicoidalViserRenderer` displays both winding families, end rings,
transitions, and tendon routes. Strand geometry follows the GVS material frames
through bending, twist, shear, and extension.

```python
from soromox.rendering import HelicoidalModuleVisual, HelicoidalViserRenderer

renderer = HelicoidalViserRenderer(
    robot,
    modules=tuple(HelicoidalModuleVisual(i, turns=3, strand_radius=0.002)
                  for i in (1, 4, 7)),
)
renderer.show(q)
```

Each module selects one link. Its circular cross-section radius gives the helix
centerline radius; `strand_radius`, `turns`, `strands_per_family`, `phase`, and
`family_phase` describe the strands. Links outside the module selection use
swept cross-section surfaces. `HelicoidalVisualConfig` controls strand colors
and mesh resolution. These visual dimensions describe the deformed lattice
geometry; the effective constitutive parameters determine its mechanics.

The renderer supports batched configurations, `render_sequence`, `render_frame`,
and `start_live_mode`. Trajectory and live updates reuse mesh handles. Image
and video capture use a connected Viser browser client.

## API Reference

::: soromox.systems.gvs.helicoidal.Helicoidal
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      inherited_members: false
      members_order: source

::: soromox.systems.gvs.params.HelicoidalSectionParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.rendering.helicoidal.HelicoidalViserRenderer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      inherited_members: false

::: soromox.rendering.helicoidal.HelicoidalModuleVisual
    options:
      heading_level: 3

::: soromox.rendering.helicoidal.HelicoidalVisualConfig
    options:
      heading_level: 3
