__all__ = [
    "tilde_SE2",
    "hat_SE2",
    "exp_SE2",
    "log_SE2",
    "exp_gn_SE2",
    "adjoint_se2",
    "coadjoint_se2",
    "Adjoint_g_SE2",
    "Adjoint_g_inv_SE2",
    "Adjoint_gi_se2",
    "Adjoint_gi_se2_inv",
    "Tangent_gi_se2",
    "Tangent_derivative_gi_se2",
]

import jax.numpy as jnp
from jax import Array, lax


J = jnp.array([[0, -1], [1, 0]])


def tilde_SE2(angle: Array) -> Array:
    """Return the skew-symmetric matrix associated with a planar angle.

    Args:
        angle (Array): array-like broadcastable to a scalar representing the rotation angle.

    Returns:
        Array: shape (2, 2) skew-symmetric matrix encoding the cross-product in SE(2).
    """

    theta = jnp.asarray(angle).reshape(-1)[0]
    return theta * J


def hat_SE2(vec3: Array) -> Array:
    """
    Computes the hat operator for a 3D vector of se(2).

    Args:
        vec3 (Array): shape (3,) or (3, 1)
            Screw coordinates [theta, vx, vy] in se(2).
            The first element corresponds to the angular component,
            and the last two elements correspond to the linear components.

    Returns:
        hat: shape (3, 3)
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
        vec3 (Array): shape (3,) or (3, 1)
            Screw coordinates [theta, vx, vy] where theta is the rotation angle and (vx, vy) is the translation vector.
    Returns:
        g: shape (3, 3)
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


def log_SE2(g: Array, eps: float) -> Array:
    """Compute the logarithmic map from SE(2) to se(2).

    Args:
        g (Array): shape (3, 3) homogeneous transform in SE(2).
        eps (float): small positive tolerance for angle regularisation.

    Returns:
        Array: shape (3,) twist coordinates corresponding to ``g``.
    """

    R = g[:2, :2]
    p = g[:2, 2]

    theta = jnp.arctan2(R[1, 0], R[0, 0])
    theta = lax.cond(
        jnp.abs(theta) < eps,
        lambda _: jnp.zeros((), dtype=theta.dtype),
        lambda _: theta,
        operand=None,
    )

    return jnp.concatenate([jnp.array([theta]), p])


def exp_gn_SE2(vec3: Array, eps: float) -> Array:
    """Compute the exponential map using the Magnus expansion for SE(2).

    Args:
        vec3 (Array): shape (3,) screw coordinates.
        eps (float): threshold for switching to the series expansion.

    Returns:
        Array: shape (3, 3) matrix exponential of ``vec3`` in SE(2).
    """

    vec3 = vec3.reshape(-1)
    theta = jnp.abs(vec3[0])
    vec3_hat = hat_SE2(vec3)

    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    def _series(_: None) -> Array:
        return (
            jnp.eye(3)
            + vec3_hat
            + 0.5 * jnp.linalg.matrix_power(vec3_hat, 2)
            + (1.0 / 6.0) * jnp.linalg.matrix_power(vec3_hat, 3)
        )

    def _closed(_: None) -> Array:
        theta_sq = theta**2
        theta_cu = theta**3
        return (
            jnp.eye(3)
            + vec3_hat
            + ((1 - costheta) / theta_sq) * jnp.linalg.matrix_power(vec3_hat, 2)
            + ((theta - sintheta) / theta_cu) * jnp.linalg.matrix_power(vec3_hat, 3)
        )

    return lax.cond(theta <= eps, _series, _closed, operand=None)


def adjoint_se2(vec3: Array) -> Array:
    """
    Computes the adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): shape (3,) or (3, 1)
            Screw coordinates [theta, vx, vy] in se(2).
            The first element corresponds to the angular component,
            and the last two elements correspond to the linear component.

    Returns:
        ad: shape (3, 3)
            A 3x3 matrix representing the adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (2,1) vector

    ad = jnp.concatenate(
        [jnp.zeros((1, 3)), jnp.concatenate([-J @ lin, ang * J], axis=1)]
    )

    return ad


def coadjoint_se2(vec3: Array) -> Array:
    """
    Computes the co-adjoint representation of a vector of se(2).

    Args:
        vec3 (Array): shape (3,) or (3, 1)
            Screw coordinates [theta, vx, vy] in se(2).
            The first element corresponds to the angular component,
            and the last two elements correspond to the linear component.

    Returns:
        coad: shape (3, 3)
            A 3x3 matrix representing the co-adjoint transformation of the input screw vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    ang = vec3[0]
    lin = vec3[1:].reshape((2, 1))  # Linear as a (2,1) vector

    coad = jnp.concatenate(
        [jnp.zeros((3, 1)), jnp.concatenate([lin.T @ J, ang * J], axis=0)], axis=1
    )

    return coad


def Adjoint_g_SE2(mat3: Array) -> Array:
    """
    Computes the adjoint representation of an SE(2) homogeneous transform.

    Args:
        mat3 (Array): shape (3, 3)
            Homogeneous transform in SE(2).

    Returns:
        Ad (Array): shape (3, 3)
            Adjoint matrix associated with ``mat3``.
    """
    R = mat3[:2, :2]  # Extract the angular part (top-left 2x2 block)
    t = mat3[:2, 2].reshape((2, 1))  # Extract the linear part (top-right column)

    Ad = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-J @ t, R], axis=1),
        ]
    )

    return Ad


def Adjoint_g_inv_SE2(mat3: Array) -> Array:
    """
    Computes the inverse adjoint representation of an SE(2) homogeneous transform.

    Args:
        mat3 (Array): shape (3, 3)
            Homogeneous transform in SE(2).

    Returns:
        Ad_inv (Array): shape (3, 3)
            Inverse of ``Adjoint_g_SE2(mat3)``.
    """
    Ad = Adjoint_g_SE2(mat3)  # Adjoint representation of the input matrix

    # Extract R and -Jt from the Adjoint matrix
    R = Ad[1:, 1:]
    mJt = Ad[1:, 0].reshape(-1, 1)

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    # Construct the inverse Adjoint matrix
    Ad_inv = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )

    return Ad_inv


def Adjoint_gi_se2(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the adjoint at arclength ``s_i`` for a segment strained by ``xi_i``.

    Args:
        xi_i (Array): shape (3,) or (3, 1)
            Constant strain vector in the segment, [theta, vx, vy].
        s_i (Array): scalar arclength position along the segment.
        eps (float): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (3, 3)
            Adjoint matrix evaluated at ``s_i``.
    """
    theta = jnp.linalg.norm(xi_i[0])  # Angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    Ad = lax.cond(
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

    return Ad


def Adjoint_gi_se2_inv(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the inverse adjoint at arclength ``s_i`` for a segment strained by ``xi_i``.

    Args:
        xi_i (Array): shape (3,) or (3, 1)
            Constant strain vector in the segment, [theta, vx, vy].
        s_i (Array): scalar arclength position along the segment.
        eps (float): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (3, 3)
            Inverse adjoint matrix evaluated at ``s_i``.
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
    Ad_inv = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones(((1, 1))), jnp.zeros((1, 2))], axis=1),
            jnp.concatenate([-R_inv @ mJt, R_inv], axis=1),
        ]
    )

    return Ad_inv


def Tangent_gi_se2(
    xi_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the tangent operator accumulated over arclength ``s_i`` for strain ``xi_i``.

    Args:
        xi_i (Array): shape (3,) or (3, 1)
            Constant strain vector in the segment, [theta, vx, vy].
        s_i (Array): scalar arclength position along the segment.
        eps (float): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (3, 3)
            Tangent matrix evaluated at ``s_i``.
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

def Tangent_derivative_gi_se2(
    xi_i: Array,
    xid_i: Array,
    s_i: Array,
    eps: float,
) -> Array:
    """
    Computes the time derivative of the tangent operator at arclength ``s_i``.

    Args:
        xi_i (Array): shape (3,) or (3, 1)
            Constant strain vector in the segment, [theta, vx, vy].
        xid_i (Array): shape (3,) or (3, 1)
            Time derivative of the strain vector.
        s_i (Array): scalar arclength position along the segment.
        eps (float): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (3, 3)
            Time derivative of the tangent matrix evaluated at ``s_i``.
    """
    theta = jnp.linalg.norm(xi_i[0])  # Angular part
    thetad = xid_i[0]  # Time derivative of the angular part
    adjoint_xi_i = adjoint_se2(xi_i)  # Adjoint representation of the input vector
    adjoint_xid_i = adjoint_se2(xid_i)  # Adjoint representation of the input vector

    cos = jnp.cos(s_i * theta)
    sin = jnp.sin(s_i * theta)

    def _tangent_dot_nonzero() -> Array:
        def _adjoint_powers_with_derivatives(max_power: int):
            # Recursively build A^k and d/dt(A^k) using P_k = P_{k-1} @ A.
            current = jnp.eye(3)
            dot_current = jnp.zeros_like(adjoint_xi_i)
            powers = [current]
            dot_powers = [dot_current]

            for _ in range(max_power):
                dot_next = dot_current @ adjoint_xi_i + current @ adjoint_xid_i
                current = current @ adjoint_xi_i
                powers.append(current)
                dot_powers.append(dot_next)
                dot_current = dot_next

            return powers, dot_powers

        powers, dot_powers = _adjoint_powers_with_derivatives(4)

        adjoint_xi_i_square = powers[2]
        adjoint_xi_i_cube = powers[3]
        adjoint_xi_i_quad = powers[4]
        adjoint_dot_xi_i = dot_powers[1]
        adjoint_dot_xi_i_square = dot_powers[2]
        adjoint_dot_xi_i_cube = dot_powers[3]
        adjoint_dot_xi_i_quad = dot_powers[4]

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

    Td = lax.cond(
        jnp.abs(theta) <= eps,
        lambda _: s_i**2 / 2 * adjoint_xid_i,
        lambda _: _tangent_dot_nonzero(),
        operand=None,
    )
    return Td
