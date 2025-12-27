from pathlib import Path

import dill
import sympy as sp

from .symbolic_utils import compute_coriolis_matrix, compute_dAdt


def symbolically_derive_planar_hsa_model(
    num_segments: int,
    filepath: str | Path = None,
    num_rods_per_segment: int = 2,
    simplify: bool = True,
) -> dict:
    """
    Symbolically derive the kinematics and dynamics of a planar hsa robot modelled with
    Piecewise Constant Strain.
    Args:
        num_segments: number of constant strain segments
        filepath: path to save the derived model
        num_rods_per_segment: number of HSA rods per segment
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # number of degrees of freedom
    num_dof = (
        3 * num_segments
    )  # we allow for 3 strains for each segment (bending, shear, elongation)

    th0 = sp.Symbol("th0", real=True)  # initial angle of the robot
    l_syms = list(
        sp.symbols(f"l1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of each segment [m]
    lpc_syms = list(
        sp.symbols(f"lpc1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of the rigid proximal caps of the rods connecting to the base [m]
    ldc_syms = list(
        sp.symbols(f"ldc1:{num_segments + 1}", nonnegative=True, nonzero=True)
    )  # length of the rigid distal caps of the rods connecting to the platform [m]
    h_syms = list(sp.symbols(f"h1:{num_segments * num_rods_per_segment + 1}"))
    rout_syms = list(
        sp.symbols(f"rout1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # outside radius of each segment [m]
    rin_syms = list(
        sp.symbols(f"rin1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # inner radius of each segment [m]
    roff_syms = list(
        sp.symbols(f"roff1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # radial offset of each rod from the centerline
    kappa_b_ref_syms = list(
        sp.symbols(f"kappa_b_ref1:{num_segments * num_rods_per_segment + 1}")
    )  # refuilibrium bending strain of each rod
    sigma_sh_ref_syms = list(
        sp.symbols(f"sigma_sh_ref1:{num_segments * num_rods_per_segment + 1}")
    )  # refuilibrium shear strain of each rod
    sigma_a_ref_syms = list(
        sp.symbols(f"sigma_a_ref1:{num_segments * num_rods_per_segment + 1}")
    )  # refuilibrium axial strain of each rod
    C_varepsilon_syms = list(
        sp.symbols(f"C_varepsilon1:{num_segments * num_rods_per_segment + 1}")
    )
    pcudim_syms = list(
        sp.symbols(f"pcudim1:{3 * num_segments + 1}", nonnegative=True)
    )  # dimensions of platform cuboid consisting of [width, height, depth] [m]
    rhor_syms = list(
        sp.symbols(f"rhor1:{num_segments * num_rods_per_segment + 1}", nonnegative=True)
    )  # volumetric mass density of the rods [kg/m^3]
    rhop_syms = list(
        sp.symbols(f"rhop1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density of the platform [kg/m^3]
    rhoec_syms = list(
        sp.symbols(f"rhoec1:{num_segments + 1}", nonnegative=True)
    )  # volumetric mass density of the rod end caps (both at the proximal and distal ends) [kg/m^3]
    g_syms = list(sp.symbols("g1:3"))  # gravity vector
    S_b_hat_syms = list(
        sp.symbols(
            f"S_b_hat1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # neutral bending stiffness of each rod [Nm^2]
    S_sh_hat_syms = list(
        sp.symbols(
            f"S_sh_hat1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # neutral shear stiffness of each rod [N]
    S_a_hat_syms = list(
        sp.symbols(
            f"S_a_hat1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # neutral axial stiffness of each rod [N]
    S_b_sh_syms = list(
        sp.symbols(
            f"S_b_sh1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # elastic coupling between bending and shearing of each rod [Nm/rad]
    C_S_b_syms = list(
        sp.symbols(
            f"C_S_b1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # change of bending stiffness of each rod [Nm^3/rad]
    C_S_sh_syms = list(
        sp.symbols(
            f"C_S_sh1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # change of shear stiffness of each rod [Nm/rad]
    C_S_a_syms = list(
        sp.symbols(
            f"C_S_a1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # change of axial stiffness of each rod [Nm/rad]
    zetab_syms = list(
        sp.symbols(
            f"zetab1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # damping coefficient for bending of each rod
    zetash_syms = list(
        sp.symbols(
            f"zetash1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # damping coefficient for shearing of each rod
    zetaa_syms = list(
        sp.symbols(
            f"zetaa1:{num_segments * num_rods_per_segment + 1}", nonnegative=True
        )
    )  # damping coefficient for elongation of each rod
    mpl = sp.Symbol("mpl", real=True, nonnegative=True)  # mass of payload [kg]
    CoGpl_syms = list(
        sp.symbols("CoGpl1:3", real=True)
    )  # CoG of payload relative to the end-effector position
    # rigid offset of the end-effector from the distal end of the last platform
    chiee_off_syms = list(sp.symbols("chiee_off1:4", real=True))

    # planar strains and their derivatives
    xi_syms = list(sp.symbols(f"xi1:{num_dof + 1}", nonzero=True))  # strains
    xid_syms = list(sp.symbols(f"xid1:{num_dof + 1}"))  # strain time derivatives
    xidd_syms = list(sp.symbols(f"xidd1:{num_dof + 1}"))  # strain accelerations
    phi_syms = list(
        sp.symbols(f"phi1:{num_segments * num_rods_per_segment + 1}")
    )  # twist angles

    # construct the symbolic matrices
    l = sp.Matrix(l_syms)  # length of each segment
    lpc = sp.Matrix(
        [lpc_syms]
    )  # length of the rigid proximal caps of the rods connecting to the base
    ldc = sp.Matrix(
        [ldc_syms]
    )  # length of the rigid distal end caps of the rods connecting to the platform
    h = sp.Matrix(h_syms).reshape(num_segments, num_rods_per_segment)  # handedness
    rout = sp.Matrix(rout_syms).reshape(
        num_segments, num_rods_per_segment
    )  # outside radius of each rod
    rin = sp.Matrix(rin_syms).reshape(
        num_segments, num_rods_per_segment
    )  # inside radius of each rod
    # radial offset of each rod from the centerline
    roff = sp.Matrix(roff_syms).reshape(num_segments, num_rods_per_segment)
    # bending reference strain of each rod
    kappa_b_ref = sp.Matrix(kappa_b_ref_syms).reshape(
        num_segments, num_rods_per_segment
    )
    # shear reference strain of each rod
    sigma_sh_ref = sp.Matrix(sigma_sh_ref_syms).reshape(
        num_segments, num_rods_per_segment
    )
    # axial reference strain of each rod
    sigma_a_ref = sp.Matrix(sigma_a_ref_syms).reshape(
        num_segments, num_rods_per_segment
    )
    C_varepsilon = sp.Matrix(C_varepsilon_syms).reshape(
        num_segments, num_rods_per_segment
    )  # elongation factor
    # dimensions of platform cuboid consisting of [width, height, depth] [m]
    pcudim = sp.Matrix(pcudim_syms).reshape(num_segments, 3)
    # volumetric mass density of the rods [kg/m^3]
    rhor = sp.Matrix(rhor_syms).reshape(num_segments, num_rods_per_segment)
    # volumetric mass density of the platform [kg/m^3]
    rhop = sp.Matrix(rhop_syms)
    # volumetric mass density of the rod end caps (both at the proximal and distal ends) [kg/m^3]
    rhoec = sp.Matrix(rhoec_syms)
    g = sp.Matrix(g_syms)  # gravity vector
    S_b_hat = sp.Matrix(S_b_hat_syms).reshape(num_segments, num_rods_per_segment)
    S_sh_hat = sp.Matrix(S_sh_hat_syms).reshape(num_segments, num_rods_per_segment)
    S_a_hat = sp.Matrix(S_a_hat_syms).reshape(num_segments, num_rods_per_segment)
    S_b_sh = sp.Matrix(S_b_sh_syms).reshape(
        num_segments, num_rods_per_segment
    )  # elastic coupling between bending and shearing of each rod
    C_S_b = sp.Matrix(C_S_b_syms).reshape(num_segments, num_rods_per_segment)
    C_S_sh = sp.Matrix(C_S_sh_syms).reshape(num_segments, num_rods_per_segment)
    C_S_a = sp.Matrix(C_S_a_syms).reshape(num_segments, num_rods_per_segment)

    # damping coefficient for bending of each rod
    zetab = sp.Matrix(zetab_syms).reshape(num_segments, num_rods_per_segment)
    # damping coefficient for shearing of each rod
    zetash = sp.Matrix(zetash_syms).reshape(num_segments, num_rods_per_segment)
    # damping coefficient for axial elongation of each rod
    zetaa = sp.Matrix(zetaa_syms).reshape(num_segments, num_rods_per_segment)
    # center of origin of the payload
    CoGpl = sp.Matrix(CoGpl_syms)

    # configuration variables and their derivatives
    xi = sp.Matrix(xi_syms)  # strains
    xid = sp.Matrix(xid_syms)  # strain time derivatives
    xidd = sp.Matrix(xidd_syms)  # strain accelerations
    # twist angle of rods
    phi = sp.Matrix(phi_syms)

    # matrix with symbolic expressions to derive the poses along the centerline of each segment
    chiv_sms = []
    # Jacobians (positional + orientation) in each segment as a function of the point coordinate s
    Jv_sms = []
    # poses and their Jacobian along each rod
    chir_sms, Jr_sms = [], []
    # poses and their Jacobian of each platform CoG
    chip_sms, Jp_sms = [], []

    # cross-sectional area of each rod
    Ar = sp.zeros(num_segments, num_rods_per_segment)
    # second area moment of inertia for bending of each rod
    Ir = sp.zeros(num_segments, num_rods_per_segment)
    # inertia matrix
    B = sp.zeros(num_dof, num_dof)
    # potential energy
    U_g = sp.Matrix([[0]])
    # stiffness matrix
    Shat = sp.zeros(3 * num_segments, 3 * num_segments)
    # elastic vector
    K = sp.zeros(3 * num_segments, 1)
    # damping matrix
    D = sp.zeros(3 * num_segments, 3 * num_segments)
    # actuation vector
    alpha = sp.zeros(3 * num_segments, 1)

    # symbol for the point coordinate
    s = sp.symbols("s", real=True, nonnegative=True)

    # initialize
    th_prev = th0
    p_prev = sp.Matrix([0, 0])
    for i in range(num_segments):
        # strains in current virtual backbone
        vxi = xi[i * 3 : (i + 1) * 3, :]

        # bending, shear and axial strains
        kappa_b, sigma_sh, sigma_a = vxi[0], vxi[1], vxi[2]

        # planar orientation of robot as a function of the point s
        th = th_prev + s * kappa_b

        # absolute rotation of link
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])

        # derivative of Cartesian position as function of the point s
        dp_ds = R @ sp.Matrix([sigma_sh, sigma_a])

        # position along the virtual center rod as a function of the point s
        p = p_prev + sp.Matrix([0, lpc[i]]) + sp.integrate(dp_ds, (s, 0.0, s))

        # symbolic expression for the pose of the virtual rod at the centerline as a function of the point s
        chiv = sp.zeros(3, 1)
        chiv[:2, 0] = p  # the x and y position
        chiv[2, 0] = th  # the orientation angle theta
        chiv_sms.append(chiv)

        # positional Jacobian as a function of the point s
        Jvp = sp.simplify(p.jacobian(xi))  # orientation Jacobian
        Jvo = sp.simplify(sp.Matrix([[th]]).jacobian(xi))

        # combine positional and orientation Jacobian
        # the first two rows correspond to the position and the last row to the orientation
        # the columns correspond to the strains xi
        Jv = Jvp.col_join(Jvo)
        Jv_sms.append(Jv)

        for j in range(num_rods_per_segment):
            # compute the cross-sectional area of the rod
            Ar[i, j] = sp.pi * (rout[i, j] ** 2 - rin[i, j] ** 2)
            # compute the second area moment of inertia of the rod
            Ir[i, j] = sp.pi / 4 * (rout[i, j] ** 4 - rin[i, j] ** 4)

            pr = p + R @ sp.Matrix([roff[i, j], 0.0])
            chir = sp.zeros(3, 1)
            chir[:2, 0] = pr  # the x and y position
            chir[2, 0] = th  # the orientation angle theta
            chir_sms.append(chir)

            # Jacobian of rod poses with respect to strains of virtual center rod
            Jr = chiv.jacobian(xi)
            Jrp = pr.jacobian(xi)  # positional Jacobian
            Jr_sms.append(Jr)

            # integrate mass matrix of each rod
            dBr_ds = rhor[i, j] * sp.simplify(
                Ar[i, j] * Jrp.T @ Jrp + Ir[i, j] * Jvo.T @ Jvo
            )
            # mass matrix of the current rod
            Br_ij = sp.integrate(dBr_ds, (s, 0, l[i]))
            if simplify:
                Br_ij = sp.simplify(Br_ij)
            # add the mass matrix
            B = B + Br_ij

            # derivative of the potential energy with respect to the point coordinate s
            dU_g_r_ds = sp.simplify(-rhor[i, j] * Ar[i, j] * g.T @ pr)
            # potential energy of the current segment
            U_g_ij = sp.simplify(sp.integrate(dU_g_r_ds, (s, 0, l[i])))
            # add potential energy of segment to previous segments
            U_g = U_g + U_g_ij

            # strains in physical HSA rod
            pxir = _sym_beta_fn(vxi, roff[i, j])
            pxi_refr = sp.Matrix(
                [[kappa_b_ref[i, j]], [sigma_sh_ref[i, j]], [sigma_a_ref[i, j]]]
            )
            # twist angle of the current rod
            phir = phi[i * num_rods_per_segment + j]

            # elongation of the reference length of the current rod
            varepsilonr = C_varepsilon[i, j] * h[i, j] / l[i] * phir

            # define the stiffness matrix of the current rod
            # difference between the nominal and current stiffness
            S_b_deltar = (
                C_S_b[i, j] * h[i, j] / l[i] * phi[i * num_rods_per_segment + j]
            )
            S_sh_deltar = (
                C_S_sh[i, j] * h[i, j] / l[i] * phi[i * num_rods_per_segment + j]
            )
            S_a_deltar = (
                C_S_a[i, j] * h[i, j] / l[i] * phi[i * num_rods_per_segment + j]
            )

            Shatr = _sym_compute_planar_stiffness_matrix(
                S_b_hat[i, j], S_sh_hat[i, j], S_a_hat[i, j], S_b_sh[i, j]
            )
            # compute the change in stiffness matrix due to the change in elastic and shear modulus
            Sdeltar = _sym_compute_planar_stiffness_matrix(
                S_b_deltar, S_sh_deltar, S_a_deltar
            )
            Sr = Shatr + Sdeltar

            # Jacobian of the strain of the physical HSA rods with respect to the configuration variables
            J_betar = sp.Matrix([[1, 0, 0], [0, 1, 0], [roff[i, j], 0, 1]])

            # nominal stiffness matrix of the virtual backbone
            Shat[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += (
                J_betar.T @ Shatr @ J_betar
            )

            # contribution of elastic forces of current rod to the virtual backbone
            vKr = J_betar.T @ Shatr @ (pxir - pxi_refr)
            # add contribution of elasticity vector
            K[3 * i : 3 * (i + 1), 0] += vKr

            # damping coefficient of the current rod
            vDr = J_betar.T @ sp.diag(zetab[i, j], zetash[i, j], zetaa[i, j]) @ J_betar
            # add contribution of damping matrix
            D[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] += vDr

            # actuation strain of the current rod
            pxiphir = sp.Matrix([[0.0], [0.0], [varepsilonr]])
            # actuation force of the current rod on the strain of the virtual backbone
            valphar = J_betar.T @ (-Sdeltar @ (pxir - pxi_refr) + Sr @ pxiphir)
            alpha[3 * i : 3 * (i + 1), 0] += valphar

        # mass of the platform itself
        mp_itself = rhop[i, 0] * pcudim[i, 0] * pcudim[i, 1] * pcudim[i, 2]
        # inertia of the platform itself about its own CoG
        Ip_itself = mp_itself / 12 * (pcudim[i, 0] ** 2 + pcudim[i, 1] ** 2)

        # contributions of the distal caps
        mdc_sum = 0
        Idc_sum = 0
        for j in range(num_rods_per_segment):
            # mass of the distal cap
            mdc = rhoec[i, 0] * ldc[i, 0] * rout[i, j] ** 2
            mdc_sum += mdc

            # moment of inertia of distal cap about its own CoG
            Idc = 1 / 12 * mdc * (3 * rout[i, j] ** 2 + ldc[i, 0] ** 2)
            Idc_sum += Idc

        # contributions of the proximal caps
        mpc_sum = 0
        Ipc_sum = 0
        if i < num_segments - 1:
            for j in range(num_rods_per_segment):
                # mass of the proximal cap
                mpc = rhoec[i + 1, 0] * lpc[i + 1, 0] * rout[i + 1, j] ** 2
                mpc_sum += mpc

                # moment of inertia of distal cap about its own CoG
                Ipc = 1 / 12 * mpc * (3 * rout[i + 1, j] ** 2 + lpc[i + 1, 0] ** 2)
                Ipc_sum += Ipc

        # total mass of the platform including the distal caps
        mp = mp_itself + mdc_sum + mpc_sum

        # position of the CoG with respect to the distal end of the rods
        _contrib_dc_pc = mdc_sum * ldc[i, 0] / 2 + mp_itself * (
            ldc[i, 0] + pcudim[i, 1] / 2
        )
        if i < num_segments - 1:
            relCoGp = (
                sp.Matrix(
                    [
                        0.0,  # TODO: fix in case the rods are not symmetric
                        _contrib_dc_pc
                        + mpc_sum * (ldc[i, 0] + pcudim[i, 1] + lpc[i + 1, 0] / 2),
                    ]
                )
                / mp
            )
        else:
            relCoGp = (
                sp.Matrix(
                    [
                        0.0,  # TODO: fix in case the rods are not symmetric
                        _contrib_dc_pc,
                    ]
                )
                / mp
            )

        # relative offsets
        doffCoGdc = sp.Matrix([0.0, ldc[i, 0] / 2]) - relCoGp
        doffCoGdc_norm = sp.sqrt(doffCoGdc[0] ** 2 + doffCoGdc[1] ** 2)
        doffCoGp = sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1] / 2]) - relCoGp
        doffCoGp_norm = sp.sqrt(doffCoGp[0] ** 2 + doffCoGp[1] ** 2)
        if i < num_segments - 1:
            doffCoGpc = sp.Matrix([0.0, lpc[i + 1, 0] / 2]) - relCoGp
        else:
            doffCoGpc = sp.Matrix([0.0, 0.0])
        doffCoGpc_norm = sp.sqrt(doffCoGpc[0] ** 2 + doffCoGpc[1] ** 2)

        # total inertia with respect to the CoG
        Ip = sp.simplify(
            Idc_sum
            + Ip_itself
            + Ipc_sum
            + mdc_sum * doffCoGdc_norm**2
            + mp_itself * doffCoGp_norm**2
            + mpc_sum * doffCoGpc_norm**2
        )

        # rotation of the platform
        Rp = R.subs(s, l[i])
        # position of the center of the platform
        pp = p.subs(s, l[i]) + Rp @ sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1] / 2])
        # pose of the platform
        chip = sp.zeros(3, 1)
        chip[:2, 0] = pp  # the x and y position
        chip[2, 0] = th.subs(s, l[i])  # the orientation angle theta
        chip_sms.append(chip)
        # position of the CoG of the platform
        pCoGp = p.subs(s, l[i]) + Rp @ relCoGp
        # pose of the CoG of the platform
        chiCoGp = sp.zeros(3, 1)
        chiCoGp[:2, 0] = pCoGp  # the x and y position
        chiCoGp[2, 0] = th.subs(s, l[i])  # the orientation angle theta
        # Jacobians of the CoG of the platform
        JCoGp = chiCoGp.jacobian(xi)
        JpCoGp = pCoGp.jacobian(xi)  # positional Jacobian of the platform
        JpCoGo = sp.simplify(sp.Matrix([[chiCoGp[2, 0]]]).jacobian(xi))
        # mass matrix of the platform
        Bp = mp * JpCoGp.T @ JpCoGp + Ip * JpCoGo.T @ JpCoGo
        if simplify:
            Bp = sp.simplify(Bp)
        B = B + Bp
        # gravitational potential energy of the platform
        U_g_p = sp.simplify(-mp * g.T @ pCoGp)
        U_g = U_g + U_g_p

        # update the orientation for the next segment
        th_prev = th.subs(s, l[i])

        # update the position for the next segment
        # add the length of the distal caps and the height of the platform
        p_prev = p.subs(s, l[i]) + Rp @ sp.Matrix([0.0, ldc[i, 0] + pcudim[i, 1]])

    # end-effector pose (the distal plane of the last platform)
    pee = p_prev  # end-effector position
    thee = th_prev  # end-effector orientation
    # rotation matrix of the last platform
    R_last_last_platform = sp.Matrix(
        [[sp.cos(thee), -sp.sin(thee)], [sp.sin(thee), sp.cos(thee)]]
    )
    # add the rigid offset of the end-effector from the distal end of the last platform
    pee += R_last_last_platform @ sp.Matrix([chiee_off_syms[0], chiee_off_syms[1]])
    thee += chiee_off_syms[2]
    # the rotation matrix of the end-effector
    Ree = sp.Matrix([[sp.cos(thee), -sp.sin(thee)], [sp.sin(thee), sp.cos(thee)]])
    # formulating the pose of the end-effector
    chiee = sp.Matrix([pee[0], pee[1], thee])
    print("chiee =\n", chiee)
    Jee = chiee.jacobian(xi)  # Jacobian of the end-effector
    print("Jee =\n", Jee)
    Jeed = compute_dAdt(Jee, xi, xid)  # time derivative of the end-effector Jacobian
    print("Jeed =\n", Jeed)

    # add contribution of the payload mass
    # the center of gravity of the payload mass should be specified relative to the end effector position
    pCoGpl = pee + Ree @ CoGpl
    # positional Jacobian of the payload CoG
    JpCoGpl = pCoGpl.jacobian(xi)
    # mass matrix of the payload while neglecting its rotational inertia around its CoG
    Bpl = mpl * JpCoGpl.T @ JpCoGpl
    if simplify:
        Bpl = sp.simplify(Bpl)
    B = B + Bpl
    # add the gravitational potential energy of the payload
    Upl = sp.simplify(-mpl * g.T @ pCoGpl)
    U_g = U_g + Upl

    # simplify mass matrix
    if simplify:
        B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, xi, xid, simplify=simplify)
    print("C =\n", C)

    # compute the gravity force vector
    G = U_g.jacobian(xi).transpose()
    if simplify:
        G = sp.simplify(G)
    print("G =\n", G)

    Shat = sp.simplify(Shat)
    print("Shat =\n", Shat)
    K = sp.simplify(K)
    print("K =\n", K)
    D = sp.simplify(D)
    print("D =\n", D)
    alpha = sp.simplify(alpha)
    print("alpha =\n", alpha)

    # dictionary with expressions
    sym_exps = {
        "params_syms": {
            "th0": th0,
            "L": l_syms,
            "lpc": lpc_syms,
            "ldc": ldc_syms,
            "h": h_syms,
            "rout": rout_syms,
            "rin": rin_syms,
            "roff": roff_syms,
            "kappa_b_ref": kappa_b_ref_syms,
            "sigma_sh_ref": sigma_sh_ref_syms,
            "sigma_a_ref": sigma_a_ref_syms,
            "C_varepsilon": C_varepsilon_syms,
            "pcudim": pcudim_syms,
            "rhor": rhor_syms,
            "rhop": rhop_syms,
            "rhoec": rhoec_syms,
            "g": g_syms,
            "S_b_hat": S_b_hat_syms,
            "S_sh_hat": S_sh_hat_syms,
            "S_a_hat": S_a_hat_syms,
            "S_b_sh": S_b_sh_syms,
            "C_S_b": C_S_b_syms,
            "C_S_sh": C_S_sh_syms,
            "C_S_a": C_S_a_syms,
            "zetab": zetab_syms,
            "zetash": zetash_syms,
            "zetaa": zetaa_syms,
            "mpl": mpl,
            "CoGpl": CoGpl_syms,
            "chiee_off": chiee_off_syms,
        },
        "state_syms": {
            "xi": xi_syms,
            "xid": xid_syms,
            "xidd": xidd_syms,
            "phi": phi_syms,
            "s": s,
        },
        "exps": {
            "Ar": Ar,  # cross-sectional area of the rods
            "Ir": Ir,  # second area moment of inertia for bending of each rod
            "chiv_sms": chiv_sms,  # list of pose expressions (for the virtual rod along the centerline of each segment)
            # list of pose expressions (for the centerline of each rod).
            # Total length is n_segments * num_rods_per_segment
            "chir_sms": chir_sms,
            "chip_sms": chip_sms,  # expression for the pose of the CoG of the platform of shape (3, )
            "chiee": chiee,  # expression for the pose of the end-effector of shape (3, )
            "Jv_sms": Jv_sms,  # list of the Jacobians of the virtual backbone of each segment
            "Jr_sms": Jr_sms,  # list of the Jacobians of the centerline of each rod
            "Jp_sms": Jp_sms,  # list of the platform Jacobians
            "Jee": Jee,  # Jacobian of the end-effector
            "Jeed": Jeed,  # time derivative of the Jacobian of the end-effector
            "B": B,
            "C": C,
            "G": G,
            "Shat": Shat,  # nominal stiffness matrix of the virtual backbone
            "K": K,  # nominal elastic forces on the virtual backbone
            "D": D,
            "alpha": alpha,
            "U_g": U_g,  # gravitational potential energy
        },
    }

    if filepath is not None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        with open(str(filepath), "wb") as f:
            dill.dump(sym_exps, f)

    return sym_exps


def _sym_beta_fn(vxi: sp.Matrix, roff: sp.Expr) -> sp.Matrix:
    """
    Symbolically map the generalized coordinates to the strains in the physical rods
    Args:
        vxi: strains of the virtual backbone of shape (3, )
    Returns:
        pxi: strains in the physical rods of shape (3, )
    """
    pxi = vxi.copy()
    pxi[2] = pxi[2] + roff * vxi[0]
    return pxi


def _sym_compute_planar_stiffness_matrix(
    S_b: sp.Expr, S_sh: sp.Expr, S_a: sp.Expr, S_b_sh: sp.Expr = 0.0
) -> sp.Matrix:
    """
    Symbolically compute the stiffness matrix of the system.
    Args:
        S_b: bending stiffness
        S_sh: shear stiffness
        S_a: axial stiffness
        S_b_sh: elastic coupling between bending and shear

    Returns:
        S: stiffness matrix of shape (3, 3)
    """
    S = sp.Matrix([[S_b, S_b_sh, 0.0], [S_b_sh, S_sh, 0.0], [0.0, 0.0, S_a]])

    return S
