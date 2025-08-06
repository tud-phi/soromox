import dill
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple, Union

from .symbolic_utils import compute_coriolis_matrix


def symbolically_derive_pendulum_model(
    num_links: int, filepath: Union[str, Path] = None
) -> Dict:
    """
    Symbolically derive the kinematics and dynamics of a n-link pendulum.
    We use the relative joint angles between links as the generalized coordinates.
    Args:
        num_links: number of pendulum links
        filepath: path to save the derived model
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    m_syms = list(sp.symbols(f"m1:{num_links + 1}"))  # mass of each link
    I_syms = list(sp.symbols(f"I1:{num_links + 1}"))  # moment of inertia of each link
    l_syms = list(sp.symbols(f"l1:{num_links + 1}"))  # length of each link
    lc_syms = list(
        sp.symbols(f"lc1:{num_links + 1}")
    )  # center of mass of each link (distance from joint)
    g_syms = list(sp.symbols(f"g1:3"))  # gravity vector

    # configuration variables and their derivatives
    q_syms = list(sp.symbols(f"q1:{num_links + 1}"))  # joint angle
    qd_syms = list(sp.symbols(f"qd1:{num_links + 1}"))  # joint velocity

    # construct the symbolic matrices
    m = sp.Matrix(m_syms)  # mass of each link
    I = sp.Matrix(I_syms)  # moment of inertia of each link
    l = sp.Matrix(l_syms)  # length of each link
    lc = sp.Matrix(lc_syms)  # center of mass of each link (distance from joint)
    g = sp.Matrix(g_syms)  # gravity vector

    # configuration variables and their derivatives
    q = sp.Matrix(q_syms)  # joint angle
    qd = sp.Matrix(qd_syms)  # joint velocity

    # matrix with tip of link and center of mass positions
    chi_ls, chic_ls = [], []
    # positional Jacobians of tip of link and center of mass respectively
    J_ls, Jc_ls = [], []
    # mass matrix
    B = sp.zeros(num_links, num_links)
    # potential energy
    U = sp.Matrix([[0]])

    # initialize
    th_prev = 0.0
    p_prev = sp.Matrix([0, 0])
    for i in range(num_links):
        # orientation of link
        th = th_prev + q[i]

        # absolute rotation of link
        R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])

        # absolute position of center of mass
        pc = sp.simplify(p_prev + R @ sp.Matrix([lc[i], 0]))

        # absolute position of end of link
        p = sp.simplify(p_prev + R @ sp.Matrix([l[i], 0]))

        # symbolic expression for the pose of the end of link
        chi = sp.zeros(3, 1)
        chi[:2, 0] = p  # the x and y position
        chi[2, 0] = th  # the orientation angle theta
        chi_ls.append(chi)

        print(f"chi of tip of link {i + 1}:\n", chi)

        # symbolic expression for the pose of the center of mass
        chic = sp.zeros(3, 1)
        chic[:2, 0] = pc  # the x and y position
        chic[2, 0] = th  # the orientation angle theta
        chic_ls.append(chic)

        # positional Jacobian of end of link
        Jp = sp.simplify(p.jacobian(q))
        # positional Jacobian of center of mass
        Jpc = sp.simplify(pc.jacobian(q))

        # orientation Jacobian
        Jo = sp.simplify(sp.Matrix([[th]]).jacobian(q))

        # combine positional and orientation Jacobian
        # the first two rows correspond to the position and the last row to the orientation
        # the columns correspond to the strains xi
        J = Jp.col_join(Jo)  # Jacobian of end of link
        Jc = Jpc.col_join(Jo)  # Jacobian of center of mass
        J_ls.append(J)
        Jc_ls.append(Jc)

        # add to mass matrix
        B = B + sp.simplify(m[i] * Jpc.T @ Jpc + I[i] * Jo.T @ Jo)

        # add to potential energy
        U = U + sp.simplify(m[i] * g.T @ pc)

        # update for next iteration
        th_prev = th
        p_prev = p

    # simplify mass matrix
    B = sp.simplify(B)
    print("B =\n", B)

    C = compute_coriolis_matrix(B, q, qd)
    print("C =\n", C)

    # compute the gravity force vector
    G = sp.simplify(-U.jacobian(q).transpose())
    print("G =\n", G)

    # dictionary with functions
    sym_exps = {
        "params_syms": {
            "m": m_syms,
            "I": I_syms,
            "l": l_syms,
            "lc": lc_syms,
            "g": g_syms,
        },
        "state_syms": {
            "q": q_syms,
            "qd": qd_syms,
        },
        "exps": {
            "chi_ls": chi_ls,  # matrix with tip poses of shape (3, n_q)
            "chic_ls": chic_ls,  # matrix with poses of center of masses of shape (3, n_q)
            "chiee": chi_ls[-1],  # matrix with end-effector poses of shape (3, 1)
            "J_ls": J_ls,  # list of end-of-link Jacobians
            "Jc_ls": Jc_ls,  # list of center-of-mass Jacobians
            "Jee": J_ls[-1],  # end-effector Jacobian of shape (3, n_q)
            "B": B,
            "C": C,
            "G": G,
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
