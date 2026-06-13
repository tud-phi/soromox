__all__ = ["PlanarHSAParams"]

from dataclasses import fields
from pathlib import Path

import numpy as np
from jax import Array
from jax import numpy as jnp

from soromox.systems.params import BaseSoftRobotParams, validate_planar_base_pose


class PlanarHSAParams(BaseSoftRobotParams):
    """Dynamic parameters for planar HSA systems.

    Field names denote one segment/platform quantity; leading axes store the
    batched values. Arrays store length, rod geometry, reference strain
    components, platform/cap dimensions, end-effector offset, and optional
    hysteresis coefficients used by the symbolic HSA expressions. ``base_pose``
    stores the planar pose ``[theta, x, y]`` with shape ``(3,)``. ``theta`` is
    a right-handed angle in radians about the out-of-plane z-axis, and
    ``x``/``y`` are direct translations in the parent frame.
    """

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
    phi_max: Array
    platform_mass: Array
    platform_center_of_gravity: Array
    end_effector_offset: Array
    hysteresis_basis: Array
    hysteresis_alpha: Array
    hysteresis_A: Array
    hysteresis_n: Array
    hysteresis_beta: Array
    hysteresis_gamma: Array

    @classmethod
    def from_npz(cls, path: str | Path) -> "PlanarHSAParams":
        """Load planar HSA parameters from an explicit ``.npz`` file path."""
        with np.load(path) as data:
            field_names = [field.name for field in fields(cls)]
            missing = sorted(name for name in field_names if name not in data)
            if missing:
                missing_names = ", ".join(missing)
                raise KeyError(
                    f"Planar HSA parameter file is missing field(s): {missing_names}."
                )

            return cls(
                **{
                    field_name: jnp.asarray(data[field_name])
                    for field_name in field_names
                }
            )

    def validate(self) -> None:
        length = jnp.asarray(self.length)
        if len(length.shape) != 1:
            raise ValueError(
                "length must be one-dimensional with shape (num_segments,)."
            )
        n_segments = length.shape[0]
        if n_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {n_segments}.")
        rod_outer_radius = jnp.asarray(self.rod_outer_radius)
        if rod_outer_radius.ndim != 2 or rod_outer_radius.shape[0] != n_segments:
            raise ValueError(
                "rod_outer_radius must have shape "
                f"({n_segments}, num_rods_per_segment), got {rod_outer_radius.shape}."
            )
        rod_shape = rod_outer_radius.shape
        for name in (
            "rod_height",
            "rod_inner_radius",
            "rod_offset",
            "bending_reference",
            "shear_reference",
            "axial_reference",
            "strain_coupling",
            "rod_density",
            "nominal_bending_stiffness",
            "nominal_shear_stiffness",
            "nominal_axial_stiffness",
            "bending_shear_stiffness",
            "bending_stiffness_correction",
            "shear_stiffness_correction",
            "axial_stiffness_correction",
            "bending_damping",
            "shear_damping",
            "axial_damping",
            "phi_max",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != rod_shape:
                raise ValueError(
                    f"{name} must have shape {rod_shape}, got {value.shape}."
                )

        per_segment_shapes = {
            "proximal_cap_length": (n_segments,),
            "distal_cap_length": (n_segments,),
            "platform_dimension": (n_segments, 3),
            "platform_density": (n_segments,),
            "end_cap_density": (n_segments,),
        }
        for name, expected_shape in per_segment_shapes.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )

        scalar_shapes = {
            "platform_mass": (),
            "platform_center_of_gravity": (2,),
            "gravity": (2,),
            "end_effector_offset": (3,),
        }
        for name, expected_shape in scalar_shapes.items():
            value = jnp.asarray(getattr(self, name))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}."
                )
        validate_planar_base_pose("base_pose", self.base_pose)

        hysteresis_basis = jnp.asarray(self.hysteresis_basis)
        if hysteresis_basis.size == 0:
            return
        if hysteresis_basis.ndim != 2 or hysteresis_basis.shape[0] != 3 * n_segments:
            raise ValueError(
                "hysteresis_basis must have shape "
                f"({3 * n_segments}, num_hysteresis), got {hysteresis_basis.shape}."
            )
        num_hysteresis = hysteresis_basis.shape[1]
        for name in (
            "hysteresis_beta",
            "hysteresis_gamma",
            "hysteresis_A",
            "hysteresis_n",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != (num_hysteresis,):
                raise ValueError(
                    f"{name} must have shape ({num_hysteresis},), got {value.shape}."
                )
        hysteresis_alpha = jnp.asarray(self.hysteresis_alpha)
        if hysteresis_alpha.shape != (3 * n_segments,):
            raise ValueError(
                "hysteresis_alpha must have shape "
                f"({3 * n_segments},), got {hysteresis_alpha.shape}."
            )
