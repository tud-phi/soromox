import jax.numpy as jnp
from jax import lax

# for documentation
from jax import Array

# ================================================================================================
# SE(2) operators
# ===================================
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
        Array: shape (3, 3)
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
        Array: shape (3, 3)
            A 3x3 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (3,1) vector

    adj_star = jnp.concatenate(
        [jnp.zeros((3, 1)), jnp.concatenate([lin.T @ J, ang * J], axis=0)], axis=1
    )

    return adj_star


def Adjoint_g_SE2(mat3: Array) -> Array:
    """
    Computes the adjoint representation of a 3x3 matrix.

    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.

    Returns:
        Array: shape (3, 3)
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
        Array: shape (3, 3)
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
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in the current segment.
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the current segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    theta = xi_i[0]  # Angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Adjoint = lax.cond(
        jnp.abs(theta) <= eps,
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
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (3, 3)
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
    inverse_Adjoint = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )

    return inverse_Adjoint


def Tangent_gi_se2(
    xi_i: Array,
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(2) deformed in the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (3,1)
            A 3-dimensional vector representing the screw in SE(2).
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    theta = xi_i[0]  # Angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Tangent = lax.cond(
        jnp.abs(theta) <= eps,
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


# ================================================================================================
# SE(3) operators
# ===================================
def tilde_SE3(vec3: Array) -> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.

    Args:
        vec3 (Array): array-like, shape (3,1)
            A 3-dimensional vector.

    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the tilde operator of the input vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    # Extract components of the vector
    x, y, z = vec3.flatten()

    # Construct the tilde operator
    tilde = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return tilde


def hat_SE3(vec6: Array) -> Array:
    """
    Computes the hat operator for a 6D vector of se(3).

    Args:
        vec6 (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw.
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear components.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the hat operator of the input screw vector.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part

    hat = jnp.block([[angtilde, lin], [jnp.zeros((1, 3)), jnp.zeros((1, 1))]])

    return hat


def exp_SE3(vec6: Array) -> Array:
    """
    Computes the exponential map for a 6D vector of se(3).

    Args:
        vec6 (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the position.
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.
    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the exponential map of the input screw vector.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    phi = vec6[0]
    theta = vec6[1]
    psi = vec6[2]
    cosphi, sinphi = jnp.cos(phi), jnp.sin(phi)
    costheta, sintheta = jnp.cos(theta), jnp.sin(theta)
    cospsi, sinpsi = jnp.cos(psi), jnp.sin(psi)

    p = vec6[3:].reshape((3, 1))

    Rphi = jnp.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]])
    Rtheta = jnp.array([[1, 0, 0], [0, costheta, -sintheta], [0, sintheta, costheta]])
    Rpsi = jnp.array([[cospsi, -sinpsi, 0], [sinpsi, cospsi, 0], [0, 0, 1]])
    # Combine the rotations
    R = Rpsi @ Rtheta @ Rphi  # Rotation matrix

    g = jnp.block([[R, p], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])

    return g


def log_SE3(g: Array, eps: float) -> Array:
    """
    Computes the logarithm map from SE(3) to se(3), i.e., extracts the twist from a transformation matrix.

    Args:
        g (Array): array-like, shape (4, 4)
            A transformation matrix in SE(3).
        eps (float): tolerance to avoid division by zero in small angle approximations.

    Returns:
        Array: shape (6,)
            A 6D vector (twist) representing the logarithm of the transformation.
    """
    R = g[:3, :3]
    p = g[:3, 3].reshape((3, 1))

    # Compute the rotation angle
    trace_R = jnp.trace(R)
    cos_theta = (trace_R - 1) / 2
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)  # For numerical stability
    theta = jnp.arccos(cos_theta)

    # Logarithm of R
    omega_hat = lax.cond(
        jnp.abs(theta) < eps,
        lambda _: jnp.zeros((3, 3)),
        lambda _: (theta / (2 * jnp.sin(theta))) * (R - R.T),
        operand=None,
    )

    omega = jnp.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]).reshape(
        (3, 1)
    )

    # Compute V inverse (Jacobian inverse)
    omega_tilde = omega_hat

    def compute_V_inv(theta):
        A = jnp.eye(3) - 0.5 * omega_tilde
        B = (1 / (theta**2)) * (
            1 - (theta * jnp.sin(theta)) / (2 * (1 - jnp.cos(theta)))
        )
        V_inv = A + B * (omega_tilde @ omega_tilde)
        return V_inv

    V_inv = lax.cond(
        jnp.abs(theta) < eps,
        lambda _: jnp.eye(3),
        lambda _: compute_V_inv(theta),
        operand=None,
    )

    v = V_inv @ p

    return jnp.vstack([omega, v]).reshape(-1)


def exp_gn_SE3(vec6: Array, eps: float) -> Array:
    """
    Function to compute the exponential map of the Magnus expansion.

    Args:
        vec6 (Array): shape (6,) JAX array
            The screw vector representing the Magnus expansion.

    Returns:
        g (Array): shape (4, 4) JAX array
            The exponential map of the Magnus expansion.
    """
    theta = jnp.linalg.norm(vec6[:3])  # Compute the norm of the angular part
    vec6_hat = hat_SE3(vec6)  # Compute the hat

    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    g = lax.cond(
        theta <= eps,
        lambda _: (
            jnp.eye(4)  # Avoid division by zero
            + vec6_hat
            + 1 / 2 * jnp.linalg.matrix_power(vec6_hat, 2)
            + 1 / 6 * jnp.linalg.matrix_power(vec6_hat, 3)
        ),
        lambda _: (
            jnp.eye(4)
            + vec6_hat
            + 1
            / jnp.power(theta, 2)
            * (1 - costheta)
            * jnp.linalg.matrix_power(vec6_hat, 2)
            + 1
            / jnp.power(theta, 3)
            * (theta - sintheta)
            * jnp.linalg.matrix_power(vec6_hat, 3)
        ),
        operand=None,
    )

    return g


def adjoint_se3(vec6: Array) -> Array:
    """
    Computes the adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): array-like, shape (3, 1)
            A 6-dimensional vector representing the screw.
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the adjoint transformation of the input screw vector.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    adj = jnp.block([[angtilde, jnp.zeros((3, 3))], [lintilde, angtilde]])

    return adj


def coadjoint_se3(vec6: Array) -> Array:
    """
    Computes the co-adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): array-like, shape (3, 1)
            A 6-dimensional vector representing the screw.
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    adj_star = jnp.block([[angtilde, lintilde], [jnp.zeros((3, 3)), angtilde]])

    return adj_star


def Adjoint_g_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the Adjoint transformation of the input matrix.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part

    Adjoint = jnp.block([[R, jnp.zeros((3, 3))], [ttilde @ R, R]])

    return Adjoint


def Adjoint_g_inv_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): array-like, shape (4,4)
            A 4x4 matrix representing the transformation.

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the Adjoint transformation of the input matrix.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T

    # Construct the inverse Adjoint matrix
    inverse_Adjoint = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return inverse_Adjoint


def Adjoint_gi_se3(
    xi_i: Array,
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in the current segment.
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the current segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    ang = xi_i[:3].reshape((3, 1))  # Angular as a (3,1) vector
    theta = jnp.linalg.norm(ang)  # Compute the norm of the angular part
    adjoint_xi_i = adjoint_se3(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Adjoint = lax.cond(
        jnp.abs(theta) <= eps,
        lambda _: jnp.eye(6) + s_i * adjoint_xi_i,  # Avoid division by zero
        lambda _: (
            jnp.eye(6)
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


def Adjoint_gi_se3_inv(
    xi_i: Array,
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in SE(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    Adj = Adjoint_gi_se3(
        xi_i, s_i, eps=eps
    )  # Adjoint representation of the input vector

    # Extract R and -Jt from the Adjoint matrix
    R = Adj[:3, :3]
    ttildeR = Adj[3:, :3]

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    ttilde = ttildeR @ R_inv  # Compute the tilde operator for the linear part
    # Construct the inverse Adjoint matrix
    inverse_Adjoint = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return inverse_Adjoint


def Tangent_gi_se3(
    xi_i: Array,
    s_i: float,
    eps: float,
) -> Array:
    """
    Computes the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in SE(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.
        s_i (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.
        eps (float): small value to avoid division by zero

    Returns:
        Array: shape (4, 4)
            A 4x4 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    # We suppose here that theta is not zero thanks to a previous use of apply_eps
    ang = xi_i[:3].reshape((3, 1))  # Angular as a (3,1) vector
    theta = jnp.linalg.norm(ang)  # Compute the norm of the angular part
    adjoint_xi_i = adjoint_se3(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Tangent = lax.cond(
        jnp.abs(theta) <= eps,
        lambda _: s_i * jnp.eye(6) + s_i**2 / 2 * adjoint_xi_i,
        lambda _: (
            s_i * jnp.eye(6)
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


# ================================================================================================
# Shared operators
# ============================
def compute_weighted_sums(M: Array, vecm: Array, idx: int) -> Array:
    """
    Compute the weighted sums of the matrix product of M and vecm,

    Args:
        M (Array): array of shape (N, m, m)
           Describes the matrix to be multiplied with vecm
        vecm (Array): array-like of shape (N, m)
           Describes the vector to be multiplied with M
        idx (int): index of the last row to be summed over

    Returns:
        Array: array of shape (N, m)
           The result of the weighted sums. For each i, the result is the sum of the products of M[i, j] and vecm[j] for j from 0 to idx.
    """
    N = M.shape[0]
    # Matrix product for each j: (N, m, m) @ (N, m, 1) -> (N, m)
    prod = jnp.einsum("nij,nj->ni", M, vecm)

    # Triangular mask for partial sum: (N, N)
    # mask[i, j] = 1 if j >= i and j <= idx
    mask = (jnp.arange(N)[:, None] <= jnp.arange(N)[None, :]) & (
        jnp.arange(N)[None, :] <= idx
    )
    mask = mask.astype(M.dtype)  # (N, N)

    # Extend 6-dimensional mask (N, N, 1) to apply to (N, m)
    masked_prod = mask[:, :, None] * prod[None, :, :]  # (N, N, m)

    # Sum over j for each i : (N, m)
    result = masked_prod.sum(axis=1)  # (N, m)
    return result
