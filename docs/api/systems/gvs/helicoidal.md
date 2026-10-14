# Helicoidal

`Helicoidal` specializes GVS for effective helicoidal lattice mechanics. It uses
native segment specifications, strain bases, quadrature, and contiguous tendon
routing. Shared GVS mechanics support fixed and floating bases in JAX and Warp,
with distributed lattice inertia and attached bodies.

## Construction and mechanics

Use `Helicoidal.from_segments(...)` with `GVSSegment` specifications,
`mass_per_length`, and `rotational_inertia_per_length`. Supply one entry per link
for the line mass [kg/m], three principal rotational inertias [kg m], and optional
`tip_mass` [kg]. Zero inertia is supported for geometric connector links with
inactive strain bases. The line mass and rotational inertias specify lattice inertia. Tip masses have translational inertia only.

Supply a `HelicoidalSectionParams` object through `sectional_law=` or
`with_sectional_law(...)`. The local x-axis is tangent to the backbone, with
sectional order `[twist, bend_y, bend_z, extension, shear_y, shear_z]`:

```text
C(s) = compliance + (L - s) * compliance_gradient
K_section(s) = diag(1 / C(s))
K_link = integral B(s).T @ K_section(s) @ B(s) ds
D_link = tau * K_link
```

The analytic law is evaluated directly at GVS Gauss nodes, using the
configured basis and optional rotational length scaling. Both coefficient arrays
have shape `(num_links, 6)` in native strain order. Rotational compliance has
units 1/(N m²); translational compliance has units 1/N. Gradient coefficients
carry one additional inverse metre. All six compliances must be positive
throughout each link.

## Shared inertia representation

`robot.params.link.sectional_inertia` overrides geometric inertia with a constant
`(num_links, 6, 6)` distribution or values at interior Gauss nodes shaped
`(num_links, num_gauss_points, 6, 6)`. Values are per unit reference length.
`tip_inertia` stores `(num_links, 6, 6)` attached-body matrices about the link tip.
The helper `soromox.systems.spatial_inertia` prepares these matrices from mass,
center of mass, and rotational inertia about that center. They include gravity
and inertial effects of offset centers of mass. Planar PCS accepts the analogous
3×3 representation; PlanarHSA prepares its rod, platform, and payload reductions
from its physical parameters and uses the shared JAX inertia assembly.

Omitting both overrides uses the existing diagonal geometric inertia path.
Selecting full inertia adds dense matrix actions and a contribution from the
already computed link-tip kinematics. Allocating zero tip matrices allows later
attachment changes without changing the compiled structure. Switching between
`None` and arrays, or changing quadrature counts, changes that structure.

## Parameter updates

Canonical generalized stiffness and damping live in `robot.params.link`, as in
GVS. Changes to other parameters preserve these matrices. Reproject explicitly
when changing lengths or constitutive coefficients:

```python
robot = robot.update_link_params(length=1.05 * robot.params.link.length)
robot = robot.with_sectional_law(sectional_law)
robot = robot.update_link_params(tip_inertia=1.1 * robot.params.link.tip_inertia)
```

These operations support JAX differentiation. Passing updated models into the
same compiled function, or updating parameters within a compiled function,
reuses compilation when topology, array shapes, and dtypes are fixed. All
links created by `Helicoidal.from_segments` have allocated tip inertia matrices,
so changing zero to a positive mass is a numeric update. Basis choices, link counts, quadrature counts, and
tendon spans are static. As with other system parameters, physical value
validation occurs for eager inputs; transformed code must preserve their domains.

## Cylindrical example

Run from the repository checkout:

```bash
JAX_PLATFORMS=cpu python examples/simulation/gvs/simulate_helicoidal.py --duration 0.02
```

The example has three modules, three compliant transitions, two inactive
connector links, 63 strain coordinates, and nine tendons. Modules use shifted
Legendre orders `(2, 2, 2, 1, 1, 1)` in native strain order: quadratic twist and
bending, linear extension and shear. Transitions use constant strains. Each
module uses 16 Gauss points; transitions and connectors use eight. Use
`--duration 0` to inspect construction or `--output trajectory.npz` to save a
rollout. An implicit integrator handles the large effective shear stiffness.
The fixture preserves the source world frame: undeformed backbone along +z,
gravity along +z, and a base rotation mapping native local x to world z.

The analytic law is derived from the pinned toolkit's interface compliance.
The fixture README at `assets/robot_parameters/helicoidal/README.md` records its
provenance, coordinate conversion, and shear fallback. Tendons include every
transition and connector in their contiguous spans. Comparisons with upstream saved solutions should account for the discretization
and routing differences. The fixture describes the toolkit's illustrative
cylindrical arm.

## Viser visualization

Display the lattice interactively:

```bash
JAX_PLATFORMS=cpu python examples/simulation/gvs/simulate_helicoidal.py --duration 0 --viser
```

Use `--viser` with a positive duration to play a simulated trajectory. The
renderer uses the standard Studio preset and displays two winding families of
solid helices, module end rings, transition and connector surfaces, and tendon routes. Strand geometry follows
the GVS material frames through bending, twist, shear, and extension.

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

## API reference

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
