import sympy as sp


def compute_coriolis_matrix(
    B: sp.Matrix, q: sp.Matrix, qd: sp.Matrix, simplify: bool = True
) -> sp.Matrix:
    """
    Compute the matrix C(q, qd) containing the coriolis and centrifugal terms using Christoffel symbols.
    Args:
        B: mass / inertial matrix of shape (num_dof, num_dof)
        q: vector of generalized coordinates of shape (num_dof,)
        qd: vector of generalized velocities of shape (num_dof,)
        simplify: whether to simplify the result

    Returns:
        C: matrix of shape (num_dof, num_dof) containing the coriolis and centrifugal terms
    """
    # size of configuration space
    num_dof = q.shape[0]

    # compute the Christoffel symbols
    Ch_flat = []
    for i in range(num_dof):
        for j in range(num_dof):
            for k in range(num_dof):
                # Ch[i, j, k] = sp.simplify(0.5 * (B[i, j].diff(q[k]) + B[i, k].diff(q[j]) - B[j, k].diff(q[i])))
                Ch_ijk = 0.5 * (
                    B[i, j].diff(q[k]) + B[i, k].diff(q[j]) - B[j, k].diff(q[i])
                )
                if simplify:
                    Ch_ijk = sp.simplify(Ch_ijk)
                Ch_flat.append(Ch_ijk)
    Ch = sp.Array(Ch_flat, (num_dof, num_dof, num_dof))

    # compute the coriolis and centrifugal force matrix
    C = sp.zeros(num_dof, num_dof)
    for i in range(num_dof):
        for j in range(num_dof):
            for k in range(num_dof):
                C[i, j] = C[i, j] + Ch[i, j, k] * qd[k]
    if simplify:
        # simplify coriolis and centrifugal force matrix
        C = sp.simplify(C)

    return C


def compute_dAdt(A: sp.Matrix, x: sp.Matrix, xdot: sp.Matrix) -> sp.Matrix:
    dAdt = sp.zeros(A.shape[0], A.shape[1])
    for j in range(A.shape[1]):
        # iterate through columns
        dAdt[:, j] = A[:, j].jacobian(x) @ xdot

    return dAdt
