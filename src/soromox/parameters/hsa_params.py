__all__ = [
    "generate_base_params_for_fpu",
    "generate_base_params_for_epu",
    "PARAMS_FPU_CONTROL",
    "PARAMS_FPU_SYSTEM_ID",
    "PARAMS_EPU_CONTROL",
    "PARAMS_EPU_SYSTEM_ID",
]


from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def generate_common_base_params(
    num_segments: int = 1,
    num_rods_per_segment: int = 4,
    end_effector_attached: int = False,
    consider_hysteresis: bool = False,
) -> dict[str, Array]:
    assert num_rods_per_segment % 2 == 0, "num_rods_per_segment must be even"

    ones_rod = jnp.ones((num_segments, num_rods_per_segment))

    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "L": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
        # length of the rigid proximal caps of the rods connecting to the base [m]
        "lpc": 25e-3 * jnp.ones((num_segments,)),
        # length of the rig id distal caps of the rods connecting to the platform [m]
        "ldc": 14e-3 * jnp.ones((num_segments,)),
        # outside radius of each rod [m]. The rows correspond to the segments.
        "kappa_b_ref": 0.0 * ones_rod,  # bending reference curvatures of each rod
        "sigma_sh_ref": 0.0 * ones_rod,  # shear reference curvatures of each rod
        "sigma_a_ref": 1.0 * ones_rod,  # axial reference strains of each rod
        "rout": 25.4e-3 / 2 * ones_rod,
        # handedness of each rod. The rows correspond to the segments.
        "h": ones_rod,
        # offset [m] of each rod from the centerline. The rows correspond to the segments.
        "roff": jnp.repeat(
            jnp.repeat(jnp.array([[-24e-3, 24e-3]]), num_rods_per_segment // 2, axis=1),
            num_segments,
            axis=0,
        ),
        "pcudim": jnp.repeat(
            jnp.array([[80e-3, 12e-3, 80e-3]]), num_segments, axis=0
        ),  # width, height, depth of the platform [m]
        "g": jnp.array([0.0, 9.81]),
        "mpl": jnp.array(0.0),  # payload mass [kg]
        # center of origin of the payload relative to end-effector [m]
        "CoGpl": jnp.array([0.0, 0.0]),
        # rigid offset SE(2) transformation from the distal end of the platform to the end-effector
        "chiee_off": jnp.array([0.0, 0.0, 0.0]),
    }

    if end_effector_attached:
        # the end-effector is moved by 25mm in the y-dir relative to the top surface of the HSA platform
        params["chiee_off"] = jnp.array([0.0, 0.0, 0.025])
        params["mpl"] = jnp.array(
            0.018
        )  # the end-effector attachment has a mass of 18g
        # the end-effector attachment has a center of gravity of 3.63mm in y-dir from its base.
        # as it has a thickness of 25mm, this is -21.37mm from the top surface (i.e., end-effector position)
        params["CoGpl"] = jnp.array([0.0, -0.02137])

    params["hysteresis"] = {}
    if consider_hysteresis:
        """
        Parameters for modeling Bouc-Wen hysteresis
        https://en.wikipedia.org/wiki/Bouc%E2%80%93Wen_model_of_hysteresis

        Notation of Bouc-Wen hysteresis model is based on the paper:
        Song J. and Der Kiureghian A. (2006) 
        Generalized Bouc–Wen model for highly asymmetric hysteresis. 
        Journal of Engineering Mechanics. ASCE. Vol 132, No. 6 pp. 610–618

        Important: we assume that each strain experiences independent hysteresis
        """
        # mapping hysteresis displacements to the strains
        # per default, we only model the hysteresis on the axial strain
        B_z = jax.scipy.linalg.block_diag(
            *[
                jnp.array([[0.0], [0.0], [1.0]]) for _ in range(num_segments)
            ]  # assumes 3 strains per segment
        )
        params["hysteresis"]["basis"] = B_z
        # number of hysteresis states
        n_z = params["hysteresis"]["basis"].shape[1]
        # ratio of post-yield and pre-yield stiffness
        hys_alpha_val = 0.6
        hys_alpha = jnp.array(
            [
                (hys_alpha_val if B_z[xi_idx, :].sum() > 0 else 1.0)
                for xi_idx in range(3 * num_segments)
            ]
        )
        params["hysteresis"]["alpha"] = hys_alpha
        # params["hysteresis"]["alpha"] = 0.6 * jnp.ones((n_q, ))  # ratio of post-yield and pre-yield stiffness
        params["hysteresis"]["beta"] = 30.0 * jnp.ones(
            (n_z,)
        )  # dimensionless parameter in the Bouc-Wen model
        params["hysteresis"]["gamma"] = 1.0 * jnp.ones(
            (n_z,)
        )  # dimensionless parameter in the Bouc-Wen model
        params["hysteresis"]["n"] = 1.0 * jnp.ones(
            (n_z,)
        )  # dimensionless parameter in the Bouc-Wen model
        # to remove redundancy from Bouc-Wen model, choose A = 1
        params["hysteresis"]["A"] = jnp.ones((n_z,))

    return params


def generate_base_params_for_fpu(
    num_segments: int = 1,
    num_rods_per_segment: int = 4,
    rod_multiplier: int = 1,
    **kwargs,
) -> dict[str, Array]:
    common_params = generate_common_base_params(
        num_segments, num_rods_per_segment, **kwargs
    )

    ones_rod = jnp.ones((num_segments, num_rods_per_segment))
    # old params (1st ISER submission)
    """ params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "L": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
        # length of the rigid proximal caps of the rods connecting to the base [m]
        "lpc": 25e-3 * jnp.ones((num_segments,)),
        # length of the rig id distal caps of the rods connecting to the platform [m]
        "ldc": 14e-3 * jnp.ones((num_segments,)),
        "sigma_a_ref": 1.0 * ones_rod,  # axial reference strains of each rod
        # scale factor for the reference length as a function of the twist strain [1/(rad/m) = m / rad]
        # manually measured: Average: 0.009118994, Std: 0.000696435
        "C_varepsilon": 0.01032588 * ones_rod,
        # outside radius of each rod [m]. The rows correspond to the segments.
        "rout": 25.4e-3 / 2 * ones_rod,  # this is for FPU rods
        # inside radius of each rod [m]. The rows correspond to the segments.
        "rin": (25.4e-3 / 2 - 2.43e-3) * ones_rod,  # this is for FPU rods
        # handedness of each rod. The rows correspond to the segments.
        "h": ones_rod,
        # offset [m] of each rod from the centerline. The rows correspond to the segments.
        "roff": jnp.repeat(
            jnp.repeat(jnp.array([[-24e-3, 24e-3]]), num_rods_per_segment // 2, axis=1),
            num_segments,
            axis=0,
        ),
        "pcudim": jnp.repeat(
            jnp.array([[80e-3, 12e-3, 80e-3]]), num_segments, axis=0
        ),  # width, height, depth of the platform [m]
        # mass of FPU rod: 14 g, mass of EPU rod: 26 g
        # For FPU, this corresponds to a measure volume of 0000175355 m^3 --> rho = 798.38 kg/m^3
        "rhor": 798.38 * ones_rod,  # Volumetric density of rods [kg/m^3],
        # Volumetric density of platform [kg/m^3],
        # weight of platform + marker holder + cylinder top piece: 0.107 kg
        # subtracting 4 x 9g for distal cap: 0.071 kg
        # volume of platform (excluding proximal and distal caps): 0.0000768 m^3
        # --> rho = 925 kg/m^3
        "rhop": 925 * jnp.ones((num_segments,)),
        # volumetric density of the rigid end pieces [kg/m^3]
        # mass of 3D printed rod (between rin and rout): 8.5g
        # mass of the rigid end piece (up to rin): 9g
        # volume: pi*lpc*rout^2 = 0.0000126677 m^3
        # --> rho = 710.4 kg/m^3
        "rhoec": 710.4 * jnp.ones((num_segments,)),
        "g": jnp.array([0.0, 9.81]),
        # Nominal bending stiffness of each rod [Nm^2]
        "S_b_hat": 9.37790706e-05 * ones_rod,
        # Nominal shear stiffness of each rod [N]
        "S_sh_hat": 9.69609493e-01 * ones_rod,
        # Nominal axial stiffness of each rod [N]
        "S_a_hat": 0.79971525 * ones_rod,
        # Elastic coupling between bending and shear [Nm/rad]
        "S_b_sh": 7.61903729e-03 * ones_rod,
        # Scaling of bending stiffness with twist strain [Nm^3/rad]
        "C_S_b": -1.46893066e-07 * ones_rod,
        # Scaling of shear stiffness with twist strain [Nm/rad]
        "C_S_sh": -3.89972741e-03 * ones_rod,
        # Scaling of axial stiffness with twist strain [Nm/rad]
        "C_S_a": 0.01019387 * ones_rod,
        # damping coefficient for bending of shape (num_segments, rods_per_segment)
        "zetab": 8e-6 * ones_rod,
        # damping coefficient for shear of shape (num_segments, rods_per_segment)
        "zetash": 2e-4 * ones_rod,
        # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
        "zetaa": 2e-3 * ones_rod,
        # maximum twist angles (positive) [rad]
        "phi_max": 210 / 180 * jnp.pi * ones_rod,
    } """
    # new params (final ISER submission)
    params = common_params | {
        # scale factor for the reference length as a function of the twist strain [1/(rad/m) = m / rad]
        # manually measured: Average: 0.009118994, Std: 0.000696435
        "C_varepsilon": 0.00984819 * ones_rod,
        # inside radius of each rod [m]. The rows correspond to the segments.
        "rin": (25.4e-3 / 2 - 2.43e-3) * ones_rod,  # this is for FPU rods
        # mass of FPU rod: 14 g
        # For FPU, this corresponds to a measure volume of 0000175355 m^3 --> rho = 798.38 kg/m^3
        "rhor": 798.38
        * rod_multiplier
        * ones_rod,  # Volumetric density of rods [kg/m^3],
        # Volumetric density of platform [kg/m^3],
        # weight of platform + marker holder + cylinder top piece: 0.107 kg
        # subtracting 4 x 9g for distal cap: 0.071 kg
        # volume of platform (excluding proximal and distal caps): 0.0000768 m^3
        # --> rho = 925 kg/m^3
        "rhop": 925 * jnp.ones((num_segments,)),
        # volumetric density of the rigid end pieces [kg/m^3]
        # mass of 3D printed rod (between rin and rout): 8.5g
        # mass of the rigid end piece (up to rin): 9g
        # volume: pi*lpc*rout^2 = 0.0000126677 m^3
        # --> rho = 710.4 kg/m^3
        "rhoec": 710.4 * rod_multiplier * jnp.ones((num_segments,)),
        # Nominal bending stiffness of each rod [Nm^2]
        "S_b_hat": 5.71346377e-04 * rod_multiplier * ones_rod,
        # Nominal shear stiffness of each rod [N]
        "S_sh_hat": 5.91462074e-01 * rod_multiplier * ones_rod,
        # Nominal axial stiffness of each rod [N]
        "S_a_hat": 5.66472469 * rod_multiplier * ones_rod,
        # Elastic coupling between bending and shear [Nm/rad]
        "S_b_sh": 4.48419541e-03 * rod_multiplier * ones_rod,
        # Scaling of bending stiffness with twist strain [Nm^3/rad]
        "C_S_b": -9.67560524e-06 * rod_multiplier * ones_rod,
        # Scaling of shear stiffness with twist strain [Nm/rad]
        "C_S_sh": -4.75687961e-04 * rod_multiplier * ones_rod,
        # Scaling of axial stiffness with twist strain [Nm/rad]
        "C_S_a": 0.01508165 * rod_multiplier * ones_rod,
        # damping coefficient for bending of shape (num_segments, rods_per_segment)
        "zetab": 2e-6 * rod_multiplier * ones_rod,
        # damping coefficient for shear of shape (num_segments, rods_per_segment)
        "zetash": 1e-5 * rod_multiplier * ones_rod,
        # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
        "zetaa": 8e-4 * rod_multiplier * ones_rod,
        # maximum twist angles (positive) [rad]
        "phi_max": 200 / 180 * jnp.pi * ones_rod,
    }

    return params


def generate_base_params_for_epu(
    num_segments: int = 1,
    num_rods_per_segment: int = 4,
    rod_multiplier: int = 1,
    **kwargs: Any,
) -> dict[str, Array]:
    common_params = generate_common_base_params(
        num_segments, num_rods_per_segment, **kwargs
    )

    ones_rod = jnp.ones((num_segments, num_rods_per_segment))
    params = common_params | {
        # scale factor for the reference length as a function of the twist strain [1/(rad/m) = m / rad]
        "C_varepsilon": 0.0079049 * ones_rod,
        # inside radius of each rod [m]. The rows correspond to the segments.
        "rin": (25.4e-3 / 2 - 4.76e-3) * ones_rod,  # this is for EPU rods
        # mass of EPU rod: 26 g
        # For EPU, this corresponds to a measure volume of 0000314034 m^3 --> rho = 827.94 kg/m^3
        "rhor": 827.94
        * rod_multiplier
        * ones_rod,  # Volumetric density of rods [kg/m^3],
        # Volumetric density of platform [kg/m^3],
        # weight of platform + marker holder + cylinder top piece: 0.107 kg
        # subtracting 4 x 9g for distal cap: 0.071 kg
        # volume of platform (excluding proximal and distal caps): 0.0000768 m^3
        # --> rho = 925 kg/m^3
        "rhop": 925 * jnp.ones((num_segments,)),
        # volumetric density of the rigid end pieces [kg/m^3]
        # mass of 3D printed rod (between rin and rout): 8.5g
        # mass of the rigid end piece (up to rin): 9g
        # volume: pi*lpc*rout^2 = 0.0000126677 m^3
        # --> rho = 710.4 kg/m^3
        "rhoec": 710.4 * rod_multiplier * jnp.ones((num_segments,)),
        # Nominal bending stiffness of each rod [Nm^2]
        "S_b_hat": -2.53938133e-05 * rod_multiplier * ones_rod,
        # Nominal shear stiffness of each rod [N]
        "S_sh_hat": 4.28135773e-02 * rod_multiplier * ones_rod,
        # Nominal axial stiffness of each rod [N]
        "S_a_hat": 0.0 * rod_multiplier * ones_rod,
        # Elastic coupling between bending and shear [Nm/rad]
        "S_b_sh": 5.04204068e-04 * rod_multiplier * ones_rod,
        # Scaling of bending stiffness with twist strain [Nm^3/rad]
        "C_S_b": 3.90666351e-07 * rod_multiplier * ones_rod,
        # Scaling of shear stiffness with twist strain [Nm/rad]
        "C_S_sh": 2.93344701e-03 * rod_multiplier * ones_rod,
        # Scaling of axial stiffness with twist strain [Nm/rad]
        "C_S_a": 0.00981059 * rod_multiplier * ones_rod,
        # damping coefficient for bending of shape (num_segments, rods_per_segment)
        "zetab": 4e-6 * rod_multiplier * ones_rod,
        # damping coefficient for shear of shape (num_segments, rods_per_segment)
        "zetash": 8e-5 * rod_multiplier * ones_rod,
        # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
        "zetaa": 5e-3 * rod_multiplier * ones_rod,
        # maximum twist angles (positive) [rad]
        "phi_max": 270 / 180 * jnp.pi * ones_rod,
    }

    return params


PARAMS_FPU_SYSTEM_ID = generate_base_params_for_fpu(
    num_segments=1, num_rods_per_segment=4
)
PARAMS_FPU_SYSTEM_ID.update(
    {
        "h": jnp.array([[1.0, -1.0, 1.0, -1.0]]),
        "roff": 24e-3 * jnp.array([[1.0, 1.0, -1.0, -1.0]]),
    }
)
PARAMS_FPU_CONTROL = generate_base_params_for_fpu(
    num_segments=1, num_rods_per_segment=2, rod_multiplier=2
)
PARAMS_FPU_HYSTERESIS_CONTROL = generate_base_params_for_fpu(
    num_segments=1, num_rods_per_segment=2, rod_multiplier=2, consider_hysteresis=True
)

PARAMS_EPU_SYSTEM_ID = generate_base_params_for_epu(
    num_segments=1, num_rods_per_segment=4
)
PARAMS_EPU_SYSTEM_ID.update(
    {
        "h": jnp.array([[1.0, -1.0, 1.0, -1.0]]),
        "roff": 24e-3 * jnp.array([[1.0, 1.0, -1.0, -1.0]]),
    }
)

PARAMS_EPU_CONTROL = generate_base_params_for_epu(
    num_segments=1, num_rods_per_segment=2, rod_multiplier=2
)
PARAMS_EPU_HYSTERESIS_CONTROL = generate_base_params_for_epu(
    num_segments=1, num_rods_per_segment=2, rod_multiplier=2, consider_hysteresis=True
)
