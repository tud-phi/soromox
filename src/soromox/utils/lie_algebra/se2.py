__all__ = [
    "hat_SE2",
    "exp_SE2",
    "adjoint_se2",
    "coadjoint_se2",
    "Adjoint_g_SE2",
    "Adjoint_g_inv_SE2",
    "Adjoint_gi_se2",
    "Adjoint_gi_se2_inv",
    "Tangent_gi_se2",
    "Tangent_dot_gi_se2",
]

import jax.numpy as jnp
from jax import Array, lax


J = jnp.array([[0, -1], [1, 0]])


def hat_SE2(vec3: Array) -> Array:
    """
    Computes the hat operator for a 3D vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw.
            The first element correspond to the angular component,
            and the last two elements correspond to the linear components.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the hat operator of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    ang = vec3[0]  # Angular part
    lin = vec3[1:].reshape((2, 1))  # Linear as a (2,1) vector

    angtilde = ang * J

    hat = jnp.block([[angtilde, lin], [jnp.zeros((1, 2)), jnp.zeros((1, 1))]])

    return hat


def exp_SE2(vec3: Array) -> Array:
    """
    Computes the exponential map for a 3D vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the position.
            [theta, x, y] where theta is the rotation angle and (x, y) is the translation vector.
    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the exponential map of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    theta = vec3[0]
    p = vec3[1:].reshape((2, 1))

    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    R = jnp.array([[cos, -sin], [sin, cos]])  # Rotation matrix

    g = jnp.block([[R, p], [jnp.zeros((1, 2)), jnp.ones((1, 1))]])

    return g


def adjoint_se2(vec3: Array) -> Array:
    """
    Computes the adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3, 1)
            A 3-dimensional vector representing the screw.
            The first element correspond to the angular component,
            and the last two elements correspond to the linear component.

    Returns:
        adj: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (3,1) vector

    adj = jnp.concatenate(
        [jnp.zeros((1, 3)), jnp.concatenate([-J @ lin, ang * J], axis=1)]
    )

    return adj


def coadjoint_se2(vec3: Array) -> Array:
    """
    Computes the co-adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): array-like, shape (3, 1)
            A 3-dimensional vector representing the screw.
            The first element correspond to the angular component,
            and the last two elements correspond to the linear component.

    Returns:
        coadj: shape (3, 3)
            A 3x3 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (3,1) vector

    coadj = jnp.concatenate(
        [jnp.zeros((3, 1)), jnp.concatenate([lin.T @ J, ang * J], axis=0)], axis=1
    )

    return coadj


def Adjoint_g_SE2(mat3: Array) -> Array:
    """
    Computes the adjoint representation of a 3x3 matrix.

    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.

    Returns:
        Adjoint: shape (3, 3)
            A 3x3 matrix representing the Adjoint transformation of the input matrix.
    """
    R = mat3[:2, :2]  # Extract the angular part (top-left 2x2 block)
    t = mat3[:2, 2].reshape((2, 1))  # Extract the linear part (top-right column)

    Adjoint = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-J @ t, R], axis=1),
        ]
    )

    return Adjoint


def Adjoint_g_inv_SE2(mat3: Array) -> Array:
    """
    Computes the adjoint representation of a 3x3 matrix.

    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.

    Returns:
        Adjoint: shape (3, 3)
            A 3x3 matrix representing the Adjoint transformation of the input matrix.
    """
    Adj = Adjoint_g_SE2(mat3)  # Adjoint representation of the input matrix

    # Extract R and -Jt from the Adjoint matrix
    R = Adj[1:, 1:]
    mJt = Adj[1:, 0].reshape(-1, 1)

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    # Construct the inverse Adjoint matrix
    inverse_Adjoint = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )

    return inverse_Adjoint


def Adjoint_gi_se2(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in the current segment.
        s_i (Array):
            The curvilinear coordinate along the rod, representing the position of a point in the current segment.
        eps (float): small value to avoid division by zero

    Returns:
        Adjoint: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    theta = jnp.linalg.norm(xi_i[0])  # Angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Adjoint = lax.cond(
        theta <= eps,
        lambda _: jnp.eye(3) + s_i * adjoint_xi_i,  # Avoid division by zero
        lambda _: (
            jnp.eye(3)
            + 1 / (2 * theta) * (3 * sin - s_i * theta * cos) * adjoint_xi_i
            + 1
            / (2 * jnp.power(theta, 2))
            * (4 - 4 * cos - s_i * theta * sin)
            * jnp.linalg.matrix_power(adjoint_xi_i, 2)
            + 1
            / (2 * jnp.power(theta, 3))
            * (sin - s_i * theta * cos)
            * jnp.linalg.matrix_power(adjoint_xi_i, 3)
            + 1
            / (2 * jnp.power(theta, 4))
            * (2 - 2 * cos - s_i * theta * sin)
            * jnp.linalg.matrix_power(adjoint_xi_i, 4)
        ),
        operand=None,
    )

    return Adjoint


def Adjoint_gi_se2_inv(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        s_i (Array):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Adjoint_inv: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    Adj = Adjoint_gi_se2(
        xi_i, s_i, eps=eps
    )  # Adjoint representation of the input vector

    # Extract R and -Jt from the Adjoint matrix
    R = Adj[1:, 1:]
    mJt = Adj[1:, 0].reshape(-1, 1)

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    # Construct the inverse Adjoint matrix
    Adjoint_inv = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )

    return Adjoint_inv


def Tangent_gi_se2(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed in the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        s_i (Array):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Tangent: shape (3, 3)
            A 3x3 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    theta = jnp.linalg.norm(xi_i[0])  # Angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Tangent = lax.cond(
        theta <= eps,
        lambda _: s_i * jnp.eye(3) + s_i**2 / 2 * adjoint_xi_i,
        lambda _: (
            s_i * jnp.eye(3)
            + 1
            / (2 * jnp.power(theta, 2))
            * (4 - 4 * cos - s_i * theta * sin)
            * adjoint_xi_i
            + 1
            / (2 * jnp.power(theta, 3))
            * (4 * s_i * theta - 5 * sin + s_i * theta * cos)
            * jnp.linalg.matrix_power(adjoint_xi_i, 2)
            + 1
            / (2 * jnp.power(theta, 4))
            * (2 - 2 * cos - s_i * theta * sin)
            * jnp.linalg.matrix_power(adjoint_xi_i, 3)
            + 1
            / (2 * jnp.power(theta, 5))
            * (2 * s_i * theta - 3 * sin + s_i * theta * cos)
            * jnp.linalg.matrix_power(adjoint_xi_i, 4)
        ),
        operand=None,
    )

    return Tangent

def Tangent_dot_gi_se2(
    xi_i: Array,
    xid_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the time derivative of the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed in the current segment according to a strain vector xi_i.
    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        xid_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the time derivative of the screw in SE(2).
        s_i (Array):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero
    Returns:
        Tangent_dot: Array of shape (3, 3)
            A 3x3 matrix representing the time derivative of the tangent transformation of the input screw vector at the specified position.
    """
    theta = jnp.linalg.norm(xi_i[0])  # Angular part
    thetad = xid_i[0]  # Time derivative of the angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector
    adjoint_xid_i = adjoint_se2(xid_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    def _adjoint_dot_power(r: int) -> Array:
        if r == 1:
            return adjoint_xid_i

        adjoint_dot_power_r = jnp.zeros_like(adjoint_xi_i)
        for u in range(1, r + 1):
            adjoint_dot_power_r = (
                adjoint_dot_power_r
                + jnp.linalg.matrix_power(adjoint_xi_i, r - u)
                @ adjoint_xid_i
                @ jnp.linalg.matrix_power(adjoint_xi_i, u - 1)
            )
        return adjoint_dot_power_r


    def _tangent_dot_nonzero() -> Array:
        adjoint_xi_i_square = jnp.linalg.matrix_power(adjoint_xi_i, 2)
        adjoint_xi_i_cube = jnp.linalg.matrix_power(adjoint_xi_i, 3)
        adjoint_xi_i_quad = jnp.linalg.matrix_power(adjoint_xi_i, 4)
        adjoint_dot_xi_i = _adjoint_dot_power(1)
        adjoint_dot_xi_i_square = _adjoint_dot_power(2)
        adjoint_dot_xi_i_cube = _adjoint_dot_power(3)
        adjoint_dot_xi_i_quad = _adjoint_dot_power(4)

        coeff_theta_common = -8 + (8 - s_i**2 * theta**2) * cos + 5 * s_i * theta * sin
        coeff_theta_alt = -8 * s_i * theta + (15 - s_i**2 * theta**2) * sin - 7 * s_i * theta * cos

        coeff1_theta = thetad / (2 * theta**3) * coeff_theta_common
        coeff1 = (4 - 4 * cos - s_i * theta * sin) / (2 * theta**2)

        coeff2_theta = thetad / (2 * theta**4) * coeff_theta_alt
        coeff2 = (4 * s_i * theta - 5 * sin + s_i * theta * cos) / (2 * theta**3)

        coeff3_theta = thetad / (2 * theta**5) * coeff_theta_common
        coeff3 = (2 - 2 * cos - s_i * theta * sin) / (2 * theta**4)

        coeff4_theta = thetad / (2 * theta**6) * coeff_theta_alt
        coeff4 = (2 * s_i * theta - 3 * sin + s_i * theta * cos) / (2 * theta**5)

        term1 = coeff1_theta * adjoint_xi_i + coeff1 * adjoint_dot_xi_i
        term2 = coeff2_theta * adjoint_xi_i_square + coeff2 * adjoint_dot_xi_i_square
        term3 = coeff3_theta * adjoint_xi_i_cube + coeff3 * adjoint_dot_xi_i_cube
        term4 = coeff4_theta * adjoint_xi_i_quad + coeff4 * adjoint_dot_xi_i_quad

        return term1 + term2 + term3 + term4

    Tangent_dot = lax.cond(
        jnp.abs(theta) <= eps,
        lambda _: s_i**2 / 2 * adjoint_xid_i,
        lambda _: _tangent_dot_nonzero(),
        operand=None,
    )
    return Tangent_dot
