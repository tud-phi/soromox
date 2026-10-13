# Helicoidal

`Helicoidal` specializes GVS for effective helicoidal lattice mechanics. It uses
native segment specifications, strain bases, quadrature, and contiguous tendon
routing. The initial implementation supports fixed-base JAX dynamics with
independent distributed lattice inertia and platform point masses.

## Construction and mechanics

Use `Helicoidal.from_segments(...)` with ordinary `GVSSegment` specifications,
`mass_per_length`, and `rotational_inertia_per_length`. Supply one entry per link
for the line mass [kg/m], three principal rotational inertias [kg m], and optional
`tip_mass` [kg]. Zero inertia is supported for geometric connector links with
inactive strain bases. Rendering radii and `link.density` do not determine the
lattice inertia. Tip masses have translational inertia only.

Supply a `HelicoidalSectionParams` object through `sectional_law=` or
`with_sectional_law(...)`. The local x-axis is tangent to the backbone, with
sectional order `[twist, bend_y, bend_z, extension, shear_y, shear_z]`:

```text
EI(s) = 1 / (cA + cB * (L - s))
K_section(s) = diag(GJ, EI(s), EI(s), EA, GA, GA)
K_link = integral B(s).T @ K_section(s) @ B(s) ds
D_link = tau * K_link
```

The analytic law is evaluated directly at existing GVS Gauss nodes, using the
configured basis and optional rotational length scaling. No sampled stiffness
profile or interpolation is required. `cA` and `cB` have units 1/(N m²) and
1/(N m³), respectively. Bending compliance must be positive across the link.

## Parameter updates

Canonical generalized stiffness and damping live in `robot.params.link`, as in
GVS. Changes to other parameters preserve these matrices. Reproject explicitly
when changing lengths or constitutive coefficients:

```python
robot = robot.update_link_params(length=1.05 * robot.params.link.length)
robot = robot.with_sectional_law(sectional_law)
robot = robot.update_params(tip_mass=1.1 * robot.params.tip_mass)
```

These operations support JAX differentiation. Passing updated models into the
same compiled function, or updating parameters within a compiled function,
reuses compilation when topology, array shapes, and dtypes remain fixed. All
links have an allocated tip-mass entry, so changing zero to a positive mass is
an ordinary numeric update. Basis choices, link counts, quadrature counts, and
tendon spans remain static. As with other system parameters, physical value
validation occurs for eager inputs; transformed code must preserve their domains.

## Cylindrical example

Run from the repository checkout:

```bash
JAX_PLATFORMS=cpu python examples/simulation/gvs/simulate_helicoidal.py --duration 0.02
```

The example has three modules, three compliant transitions, two inactive
connector links, 36 zero-order strain coordinates, and nine tendons. Use
`--duration 0` to inspect construction or `--output trajectory.npz` to save a
rollout. An implicit integrator handles the large effective shear stiffness.
The fixture preserves the source world frame: undeformed backbone along +z,
gravity along +z, and a base rotation mapping native local x to world z.

The analytic law is derived from the pinned toolkit's interface compliance.
The fixture README at `assets/robot_parameters/helicoidal/README.md` records its
provenance, coordinate conversion, and shear fallback. Tendons include every
transition and connector in their contiguous spans. This discretization and
routing differ from the upstream example; its saved solutions are not exact
regression targets. The fixture is illustrative and does not reproduce the
paper's experimental arm.

Platform inertia, gravity, and velocity bias are included in both individual
mechanics methods and `dynamics_terms()` / `forward_dynamics()`. The initial bias
calculation uses a JAX directional derivative of the tip Jacobians. Warp and
floating-base execution, helix geometry generation, friction, length constraints,
and contact saturation are outside this implementation.

## API reference

::: soromox.systems.gvs.helicoidal.Helicoidal
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      inherited_members: false
      members_order: source

::: soromox.systems.gvs.params.HelicoidalParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: soromox.systems.gvs.params.HelicoidalSectionParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
