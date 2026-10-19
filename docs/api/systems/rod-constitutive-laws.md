# Rod Constitutive Laws

GVS links select their transverse-shear and rectangular-torsion assumptions
independently. The defaults preserve the existing solid-link behavior:
`TransverseShearModel.UNIFORM` and
`RectangularTorsionModel.SAINT_VENANT_SERIES`. Hollow circular and rectangular
links require `EFFECTIVE_AREA`; rectangular tubes also require `MARSHALL`.

```python
from soromox.systems import (
    LinkSpec, RectangularTorsionModel, RodConstitutiveLaw, TransverseShearModel,
)

law = RodConstitutiveLaw(
    transverse_shear=TransverseShearModel.EFFECTIVE_AREA,
    rectangular_torsion=RectangularTorsionModel.MARSHALL,
)
link = LinkSpec.rectangular(
    length=0.2, height=0.02, width=0.04, wall_thickness=0.003,
    density=1000.0, young_modulus=1e6, poisson_ratio=0.45,
    reference_strain=[0, 0, 0, 1, 0, 0], constitutive_law=law,
)
```

These are static choices retained in `GVSLinkStructure`. Geometry or material
updates preserve them. Reconstruct the model to change the choices. PCS and
PlanarPCS currently accept only the default law and solid circular geometry.
Helicoidal strand mechanics retain their separately documented assumptions.

## Stiffness and damping

In angular-first strain order
\((\kappa_x,\kappa_y,\kappa_z,\epsilon_x,\gamma_y,\gamma_z)\), the local
stiffness per reference length is

\[
C=\operatorname{diag}(GJ,EI_y,EI_z,EA,GA_{s,y},GA_{s,z}).
\]

Area and bending moments always subtract the cavity contribution exactly.
Rotational mass inertia uses the polar area moment \(I_y+I_z\), independently
of the torsion constant \(J\). Sections are centred and material is homogeneous
and isotropic; the formulas do not model restrained warping or local wall
buckling.

The existing scalar Kelvin–Voigt damping convention becomes
\(\eta\operatorname{diag}(J,3I_y,3I_z,3A,A_{s,y},A_{s,z})\).
The selected shear areas also enter damping, using the elastic Poisson ratio
for Cowper shear. This retains the existing factor-three axial/bending
convention; it does not introduce a separate viscous Poisson ratio.
Explicit generalized stiffness and damping matrices remain independent inputs.
With explicit stiffness and Cowper shear, supply explicit damping too or omit
it: a material damping coefficient alone does not determine Poisson's ratio.

## Transverse shear

`UNIFORM` uses \(A_{s,y}=A_{s,z}=A\) and requires solid geometry.
`EFFECTIVE_AREA` uses these geometry-specific approximations:

- **Circles and annuli:** Cowper's expression, with \(m=r/R\),
  \[
  A_s=kA,\qquad
  k=\frac{6(1+\nu)(1+m^2)^2}
          {(7+6\nu)(1+m^2)^2+(20+12\nu)m^2}.
  \]
  It has no thin-wall restriction and approaches
  \(6(1+\nu)/(7+6\nu)\) as the cavity closes. Poisson's ratio is supplied
  through the elastic material or derived as \(E/(2G)-1\), and must satisfy
  \(-1<\nu\leq 0.5\). The existing `IsotropicMaterialParams` explicit-Poisson
  input excludes the exactly incompressible endpoint. Material application
  recomputes the areas, including derivatives through Poisson's ratio.
  See [Cowper (1966)](https://doi.org/10.1115/1.3625046).
- **Rectangles and rectangular tubes:** match the energy of the Jourawski
  stress approximation \(\tau=VQ/(Ib)\):
  \[
  A_s=\frac{I^2}{\int Q(u)^2/b(u)\,du}.
  \]
  Here \(b(u)\) is the total material breadth of a strip and \(Q(u)\) its
  first area moment above the cut. The implementation integrates the quartic
  integrand exactly on each strip using three-point Gauss quadrature. Swapping
  height and width exchanges the two shear areas. This model has no thin-wall
  expansion, recovers \(5A/6\) for solids, and approaches that value continuously
  as the cavity closes. It approximates the shear stress as uniform across each
  strip's material breadth, neglecting transverse stress redistribution and
  corner effects; it is not a full two-dimensional elasticity solution.

Effective shear for elliptical sections is currently unsupported. Invalid
geometry/law combinations fail during construction and parameter validation.
As with other parameter value checks, validate physical domains before passing
traced values into JIT-compiled updates.

## Torsion and limits

Circles and annuli use exact \(J=\pi(R^4-r^4)/2\). Solid ellipses use exact
\(J=\pi a^3b^3/(a^2+b^2)\). The rectangular choice is:

- `SAINT_VENANT_SERIES`: the solid rectangular series with 64 odd terms.
- `MARSHALL`: \(J=4tA_m^2/P_m+P_mt^3/3\), for a centred, uniform-wall tube
  with sharp corners. \(A_m\) and \(P_m\) are the area enclosed by the wall
  midline and its perimeter. The second term corrects Bredt's thin-wall value;
  its relative contribution \(P_m^2t^2/(12A_m^2)\) vanishes quadratically with
  wall thickness. See Eq. (1) in
  [Ridley-Ellis et al. (2003)](https://doi.org/10.1016/S0143-974X(02)00060-3).

For solids, the Marshall option evaluates the same formula at
\(t=\min(h,w)/2\). This is an engineering extension providing continuity to
the hollow model. For a nonsquare rectangle the closing uniform-wall cavity
becomes a line; agreement with the exact elasticity solution is not implied.
The solid extension differs from the Saint-Venant series as follows:

| Longer/shorter side | Marshall relative error |
|---:|---:|
| 1 | +3.739% |
| 2 | −2.065% |
| 5 | −1.882% |
| 10 | −1.078% |

For square tubes with wall/outer-side ratios of 1%, 5%, and 10%, Marshall
exceeds Bredt by 0.0136%, 0.3693%, and 1.6461%, respectively. Regression tests
check these comparisons, cavity-collapse limits, material updates, and geometry
and material derivatives. Default selection can be revisited independently of
these explicit choices.
