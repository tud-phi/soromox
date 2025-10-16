# GVS Systems (Geometric Variable Strain)

The Geometric Variable Strain (GVS) implementation extends the classic PCS formulation by allowing non-constant strain distributions along each continuum segment. It supports heterogeneous basis functions, mixed joint/link parametrizations, and high-order Gauss quadrature for precise Cosserat rod integration in 3D.

## Overview

GVS models combine:

- **Hybrid joint-link strain spaces** with individual basis selection (monomials, Fourier, Gaussian RBFs, Legendre/Chebyshev polynomials, and more)  
- **Magnus-based integration** at Gauss quadrature points for high accuracy even under large deformations  
- **Mixed joint types** (revolute, helical, planar, spherical, free, fixed, …) with reference strain offsets and segment-specific material data  
- **Differentiable computations** for forward kinematics, Jacobians, dynamics, energies, and batched evaluations compatible with JAX transformations

This makes the GVS system particularly suitable for soft robots that require spatially varying strain fields (e.g., tendon-guided or radially varying stiffness designs) while remaining compatible with gradient-based estimation, control, and optimization.

## API Reference

::: soromox.systems.gvs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## References

Key literature on the geometric variable-strain formulation and its applications:

- Renda, F., Armanini, C., Lebastard, V., Candelier, F., & Boyer, F. (2020). A geometric variable-strain approach for static modeling of soft manipulators with tendon and fluidic actuation. *IEEE Robotics and Automation Letters*, 5(3), 4006–4013.
- Mathew, A. T., Hmida, I. B., Armanini, C., Boyer, F., & Renda, F. (2022). Sorosim: A MATLAB toolbox for hybrid rigid–soft robots based on the geometric variable-strain approach. *IEEE Robotics & Automation Magazine*, 30(3), 106–122.
- Boyer, F., Lebastard, V., Candelier, F., & Renda, F. (2020). Dynamics of continuum and soft robots: A strain parameterization based approach. *IEEE Transactions on Robotics*, 37(3), 847–863.
