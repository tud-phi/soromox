# Cylindrical helicoidal example

`demo_cyl.json` contains native SoRoMoX parameters derived from the illustrative
`demo_cyl` dataset in [CharlesQueen/helicoid-toolkit](https://github.com/CharlesQueen/helicoid-toolkit)
at commit `76c659d7900fb1a5ed01830a4fc1c6c6d473aa59`. The upstream data and the
conversion equations from `rod/equiv_rod.py` are distributed under Apache-2.0;
the accompanying `LICENSE` retains that license. This converted fixture changes
the coordinate convention, discretization, and routing as described below.

The modeling reference is Qin et al., *Cosserat Modeling of Trimmed Helicoid
Soft Arms with a Separated-Section Constitutive Law*,
[arXiv:2609.25264](https://arxiv.org/abs/2609.25264). The fixture describes the
toolkit's illustrative cylindrical arm.

The analytic `EI(s) = 1 / (cA + cB*(L-s))` law is recovered from the upstream
`module_compliance.npz`. Its stiffness agrees with the exported sectional values.
The fixture records the source compliance matrices and shear-fit residuals. All
three modules use the upstream `GA = 1e6 N` fallback for a nonpositive shear
compliance residual. This value is a modeling assumption.
Transitions have constant EI and GJ equal to 50 times the module's distal values;
EA and GA both equal 50 times the module EA. The damping time constant is 0.02 s.

Native local x is the backbone tangent. Local vectors are converted by
`P = [[0,0,1], [1,0,0], [0,1,0]]`; the base orientation is `P.T` to preserve the
source world frame. Consequently, the undeformed tip is `[0,0,0.6446] m` and world
gravity is `[0,0,9.81] m/s²`. Both angular and linear strain blocks use P.
Rotational inertia is reordered from source x/y/z to native z/x/y.

The example uses zero-order strains in all six channels of each of the six
compliant links (36 coordinates). Two 10.3 mm connectors have inactive strain
bases and zero distributed inertia. Platform point masses attach to the module
tips before their following gaps. Connector law coefficients are positive
placeholders that project to zero because their bases are inactive.

Each tendon runs continuously from the base to its anchor, including transitions
and connectors. There are nine independent tension inputs in newtons. Their
undeformed lengths are 0.208 m, 0.4263 m, and 0.6446 m (three of each). Upstream
uses higher-order module strains and omits transitions and gaps from tendon
lengths. Comparisons with its saved solutions should account for these modeling
differences.
