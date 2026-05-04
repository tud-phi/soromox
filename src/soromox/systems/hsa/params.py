__all__ = ["PlanarHSAParams"]

from jax import Array

from soromox.systems.params import BaseSystemParams


class PlanarHSAParams(BaseSystemParams):
    """Dynamic parameters for planar HSA systems.

    Field names denote one segment/platform quantity; leading axes store the
    batched values. Arrays store length, rod geometry, reference strain
    components, platform/cap dimensions, end-effector offset, and optional
    hysteresis coefficients used by the symbolic HSA expressions.
    """

    base_angle: Array
    length: Array
    proximal_cap_length: Array
    distal_cap_length: Array
    rod_height: Array
    rod_outer_radius: Array
    rod_inner_radius: Array
    rod_offset: Array
    bending_reference: Array
    shear_reference: Array
    axial_reference: Array
    strain_coupling: Array
    platform_dimension: Array
    rod_density: Array
    platform_density: Array
    end_cap_density: Array
    gravity: Array
    nominal_bending_stiffness: Array
    nominal_shear_stiffness: Array
    nominal_axial_stiffness: Array
    bending_shear_stiffness: Array
    bending_stiffness_correction: Array
    shear_stiffness_correction: Array
    axial_stiffness_correction: Array
    bending_damping: Array
    shear_damping: Array
    axial_damping: Array
    platform_mass: Array
    platform_center_of_gravity: Array
    end_effector_offset: Array
    hysteresis_basis: Array
    hysteresis_alpha: Array
    hysteresis_A: Array
    hysteresis_n: Array
    hysteresis_beta: Array
    hysteresis_gamma: Array
