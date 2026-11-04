import jax.numpy as jnp
from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    StrainBasisSpec,
)


class FourTendonSoftArm:
    """Tapered four-segment soft arm with four linear, non-crossing tendons at 45/135/225/315 deg."""

    def __init__(self, num_gauss_points=(8, 8, 8, 8), gravity=(0.0, 0.0, -9.81)):
        E_all = 1e6
        nu_all = 0.4
        rho_all = 1080.0
        eta_all = 10000.0

        L0 = 0.02
        segment_lengths = jnp.array([0.025, 0.255, 0.05, 0.03])
        L_total = L0 + jnp.sum(segment_lengths)
        cum_L = L0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(segment_lengths)[:-1]])
        Rb0 = 0.016
        Rt4 = 0.0048
        Rt0123 = Rb0 + (Rt4 - Rb0) * cum_L / L_total
        Rb = Rt0123
        Rt = jnp.concatenate([Rt0123[1:], jnp.array([Rt4])])

        link1, link2, link3, link4 = [
            LinkSpec.circular(
                length=segment_lengths[i],
                radius=LinearProfile(base=Rb[i], tip=Rt[i]),
                density=rho_all,
                young_modulus=E_all,
                poisson_ratio=nu_all,
                material_damping_coefficient=eta_all,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for i in range(4)
        ]

        joint = JointSpec.fixed()

        basis1 = StrainBasisSpec(
            type="legendre",
            strain_selector=[0, 1, 1, 1, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        )
        basis2 = StrainBasisSpec(
            type="legendre",
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[1, 1, 1, 1, 0, 0],
        )
        basis3 = StrainBasisSpec(
            type="legendre",
            strain_selector=[1, 1, 1, 1, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        )
        basis4 = StrainBasisSpec(
            type="legendre",
            strain_selector=[0, 1, 1, 0, 0, 0],
            basis_order=[0, 0, 0, 0, 0, 0],
        )

        clearances = jnp.array([0.004, 0.004, 0.003, 0.003])
        R_disc = Rb - clearances

        tendon_angles_rad = jnp.deg2rad(jnp.array([45.0, 135.0, 225.0, 315.0]))
        directions = jnp.stack(
            [
                jnp.zeros_like(tendon_angles_rad),
                jnp.cos(tendon_angles_rad),
                jnp.sin(tendon_angles_rad),
            ],
            axis=-1,
        )
        s0 = cum_L[0]
        s1 = cum_L[3]
        R0 = R_disc[0]
        R1 = R_disc[3]
        slope = (R1 - R0) / (s1 - s0) * directions
        intercept = R0 * directions - slope * s0

        active_tendon_routing = ThreadlikeRouting.linear(
            intercept=intercept,
            slope=slope,
            start_segment_index=0,
            end_segment_index=2,
        )

        segments = [
            GVSSegment(
                link=link1,
                joint=joint,
                basis=basis1,
                num_gauss_points=num_gauss_points[0],
            ),
            GVSSegment(
                link=link2,
                joint=joint,
                basis=basis2,
                num_gauss_points=num_gauss_points[1],
            ),
            GVSSegment(
                link=link3,
                joint=joint,
                basis=basis3,
                num_gauss_points=num_gauss_points[2],
            ),
            GVSSegment(
                link=link4,
                joint=joint,
                basis=basis4,
                num_gauss_points=num_gauss_points[3],
            ),
        ]

        self.model = GVS.from_segments(
            segments,
            gravity=jnp.asarray(gravity),
            floating_base=True,
            actuators=ThreadlikeActuator.tendons(active_tendon_routing),
            scale_rotational_basis_by_length=True,
            backend="auto",
        )
