__all__ = [
    "tilde_SE3",
    "hat_SE3",
    "exp_SE3",
    "log_SE3",
    "exp_gn_SE3",
    "adjoint_se3",
    "coadjoint_se3",
    "Adjoint_g_SE3",
    "Adjoint_g_inv_SE3",
    "Adjoint_gi_se3",
    "Adjoint_gi_se3_inv",
    "Tangent_gi_se3",
    "Tangent_derivative_gi_se3",
]

import jax.numpy as jnp
from jax import Array, lax


def _rotational_strain_magnitude(xi: Array, eps: float | Array) -> Array:
    """
    Computes the magnitude of the rotational strain component of a 6D vector.

    Args:
        xi (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        theta: scalar
            Magnitude of the rotational strain component.
    """
    k = xi[:3]

    # not differentiable in reverse mode at k=0
    # theta = jnp.linalg.norm(k)

    # differentiable in reverse mode at k=0
    theta_sq = jnp.dot(k, k)
    theta = lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=xi.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )
    return theta


def tilde_SE3(vec3: Array) -> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.

    Args:
        vec3 (Array): shape (3,) or (3, 1)
            3D vector [x, y, z].

    Returns:
        tilde: shape (3, 3)
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
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        hat: shape (4, 4)
            Matrix representation of the Lie algebra element.
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
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [phi, theta, psi, vx, vy, vz] with ZYX Euler convention for the rotation part.
    
    Returns:
        g: shape (4, 4)
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


def log_SE3(g: Array, eps: float | Array) -> Array:
    """
    Computes the logarithm map from SE(3) to se(3), i.e., extracts the twist from a transformation matrix.

    Args:
        g (Array): shape (4, 4)
            Homogeneous transform in SE(3).
        eps (Union[float, Array]): tolerance to avoid division by zero in small angle approximations.

    Returns:
        log: shape (6,)
            Twist coordinates corresponding to ``g``.
    """
    R = g[:3, :3]
    p = g[:3, 3].reshape((3, 1))

    # Compute the rotation angle
    trace_R = jnp.trace(R)
    skew_part = R - R.T
    skew_norm_sq = jnp.sum(jnp.square(skew_part))
    cos_theta = (trace_R - 1.0) * 0.5
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    sin_theta = jnp.sqrt(jnp.maximum(0.0, skew_norm_sq) * 0.125)
    eps_scalar = 1e8 * jnp.asarray(eps, dtype=R.dtype)
    is_small_angle = sin_theta <= jnp.sin(eps_scalar)
    # jax.debug.print("sin(theta): {sinth}, sin(eps): {seps}", sinth=sin_theta, seps=jnp.sin(eps_scalar))
    # jax.debug.print("is_small_angle: {b}", b=is_small_angle)

    theta = lax.cond(
        is_small_angle,
        lambda _: jnp.zeros((), dtype=R.dtype),
        lambda _: jnp.arctan2(sin_theta, cos_theta),
        operand=None,
    )

    def _omega_hat_small(args):
        skew, _ = args
        return 0.5 * skew

    def _omega_hat_general(args):
        skew, angle = args
        sin_theta = jnp.sin(angle)
        factor = angle / (2.0 * sin_theta)
        return factor * skew

    omega_hat = lax.cond(
        is_small_angle,
        _omega_hat_small,
        _omega_hat_general,
        (skew_part, theta),
    )

    omega = jnp.array(
        [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=R.dtype
    ).reshape((3, 1))

    # Compute V inverse (Jacobian inverse)
    omega_tilde = omega_hat

    def _compute_V_inv_small(args):
        omega_local, _ = args
        omega_sq = omega_local @ omega_local
        return jnp.eye(3, dtype=R.dtype) - 0.5 * omega_local + (1.0 / 12.0) * omega_sq

    def _compute_V_inv_general(args):
        omega_local, angle = args
        sin_theta = jnp.sin(angle)
        cos_theta_local = jnp.cos(angle)
        omega_sq = omega_local @ omega_local
        A = jnp.eye(3, dtype=R.dtype) - 0.5 * omega_local
        B = (1.0 / (angle**2)) * (
            1.0 - (angle * sin_theta) / (2.0 * (1.0 - cos_theta_local))
        )
        return A + B * omega_sq

    V_inv = lax.cond(
        is_small_angle,
        _compute_V_inv_small,
        _compute_V_inv_general,
        (omega_tilde, theta),
    )

    v = V_inv @ p

    log = jnp.vstack([omega, v]).reshape(-1)
    return log


def exp_gn_SE3(vec6: Array, eps: float | Array) -> Array:
    """
    Function to compute the exponential map of the Magnus expansion.

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates used in the Magnus expansion.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.
    Returns:
        g (Array): shape (4, 4)
            Homogeneous transform obtained from the Magnus expansion.
    """
    theta = _rotational_strain_magnitude(vec6, eps)
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
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            Adjoint representation of ``vec6``.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    ad = jnp.block([[angtilde, jnp.zeros((3, 3))], [lintilde, angtilde]])

    return ad


def coadjoint_se3(vec6: Array) -> Array:
    """
    Computes the co-adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            Co-adjoint representation of ``vec6``.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    coad = jnp.block([[angtilde, lintilde], [jnp.zeros((3, 3)), angtilde]])

    return coad


def Adjoint_g_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): shape (4, 4)
            Homogeneous transform in SE(3).

    Returns:
        Array: shape (6, 6)
            Adjoint matrix associated with ``mat4``.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part

    Ad = jnp.block([[R, jnp.zeros((3, 3))], [ttilde @ R, R]])

    return Ad


def Adjoint_g_inv_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): shape (4, 4)
            Homogeneous transform in SE(3).

    Returns:
        Array: shape (6, 6)
            Inverse adjoint matrix associated with ``mat4``.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T

    # Construct the inverse Adjoint matrix
    Ad_inv = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return Ad_inv


def Adjoint_gi_se3(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (6, 6)
            Adjoint matrix evaluated at ``s_i``.
    """
    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)

    def _series_branch() -> Array:
        return jnp.eye(6) + s_i * adjoint_xi_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        adjoint_xi_i_square = adjoint_xi_i @ adjoint_xi_i
        adjoint_xi_i_cube = adjoint_xi_i_square @ adjoint_xi_i
        adjoint_xi_i_quad = adjoint_xi_i_cube @ adjoint_xi_i

        term1 = (
            (3 * sin_theta - s_i * theta_val * cos_theta)
            / (2 * theta_val)
            * adjoint_xi_i
        )
        term2 = (
            (4 - 4 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**2)
            * adjoint_xi_i_square
        )
        term3 = (
            (sin_theta - s_i * theta_val * cos_theta)
            / (2 * theta_val**3)
            * adjoint_xi_i_cube
        )
        term4 = (
            (2 - 2 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**4)
            * adjoint_xi_i_quad
        )

        return jnp.eye(6) + term1 + term2 + term3 + term4

    Ad = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )

    return Ad


def Adjoint_gi_se3_inv(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (6, 6)
            Inverse adjoint matrix evaluated at ``s_i``.
    """
    Ad = Adjoint_gi_se3(
        xi_i, s_i, eps=eps
    )  # Adjoint representation of the input vector

    # Extract R and -Jt from the Adjoint matrix
    R = Ad[:3, :3]
    ttildeR = Ad[3:, :3]

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    ttilde = ttildeR @ R_inv  # Compute the tilde operator for the linear part
    # Construct the inverse Adjoint matrix
    Ad_inv = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return Ad_inv


def Tangent_gi_se3(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        T (Array): shape (6, 6)
            A 6x6 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)

    def _series_branch() -> Array:
        return s_i * jnp.eye(6) + 0.5 * s_i**2 * adjoint_xi_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        adjoint_xi_i_square = adjoint_xi_i @ adjoint_xi_i
        adjoint_xi_i_cube = adjoint_xi_i_square @ adjoint_xi_i
        adjoint_xi_i_quad = adjoint_xi_i_cube @ adjoint_xi_i

        term1 = (
            (4 - 4 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**2)
            * adjoint_xi_i
        )
        term2 = (
            (4 * s_i * theta_val - 5 * sin_theta + s_i * theta_val * cos_theta)
            / (2 * theta_val**3)
            * adjoint_xi_i_square
        )
        term3 = (
            (2 - 2 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**4)
            * adjoint_xi_i_cube
        )
        term4 = (
            (2 * s_i * theta_val - 3 * sin_theta + s_i * theta_val * cos_theta)
            / (2 * theta_val**5)
            * adjoint_xi_i_quad
        )

        return s_i * jnp.eye(6) + term1 + term2 + term3 + term4

    T = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )

    return T


def Tangent_derivative_gi_se3(
    xi_i: Array,
    xid_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the tangent derivative representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_i and its derivative xid_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        xid_i (Array): shape (6,) or (6, 1)
            Time derivative of the strain vector.
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Td (Array): shape (6, 6)
            A 6x6 matrix representing the tangent derivative transformation of the input screw vector at the specified position.
    """
    k = xi_i[:3]
    kd = xid_i[:3]

    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)
    adjoint_xid_i = adjoint_se3(xid_i)

    def _adjoint_powers_with_derivatives(max_power: int):
        current = jnp.eye(6)
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

    def _series_branch() -> Array:
        return 0.5 * s_i**2 * adjoint_xid_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        powers, dot_powers = _adjoint_powers_with_derivatives(4)

        adjoint_xi_i_square = powers[2]
        adjoint_xi_i_cube = powers[3]
        adjoint_xi_i_quad = powers[4]
        adjoint_dot_xi_i = dot_powers[1]
        adjoint_dot_xi_i_square = dot_powers[2]
        adjoint_dot_xi_i_cube = dot_powers[3]
        adjoint_dot_xi_i_quad = dot_powers[4]

        thetad = jnp.dot(kd, k) / theta_val

        coeff_theta_common = (
            -8
            + (8 - s_i**2 * theta_val**2) * cos_theta
            + 5 * s_i * theta_val * sin_theta
        )
        coeff_theta_alt = (
            -8 * s_i * theta_val
            + (15 - s_i**2 * theta_val**2) * sin_theta
            - 7 * s_i * theta_val * cos_theta
        )

        coeff1_theta = thetad / (2 * theta_val**3) * coeff_theta_common
        coeff1 = (4 - 4 * cos_theta - s_i * theta_val * sin_theta) / (2 * theta_val**2)

        coeff2_theta = thetad / (2 * theta_val**4) * coeff_theta_alt
        coeff2 = (4 * s_i * theta_val - 5 * sin_theta + s_i * theta_val * cos_theta) / (
            2 * theta_val**3
        )

        coeff3_theta = thetad / (2 * theta_val**5) * coeff_theta_common
        coeff3 = (2 - 2 * cos_theta - s_i * theta_val * sin_theta) / (2 * theta_val**4)

        coeff4_theta = thetad / (2 * theta_val**6) * coeff_theta_alt
        coeff4 = (2 * s_i * theta_val - 3 * sin_theta + s_i * theta_val * cos_theta) / (
            2 * theta_val**5
        )

        term1 = coeff1_theta * adjoint_xi_i + coeff1 * adjoint_dot_xi_i
        term2 = coeff2_theta * adjoint_xi_i_square + coeff2 * adjoint_dot_xi_i_square
        term3 = coeff3_theta * adjoint_xi_i_cube + coeff3 * adjoint_dot_xi_i_cube
        term4 = coeff4_theta * adjoint_xi_i_quad + coeff4 * adjoint_dot_xi_i_quad

        return term1 + term2 + term3 + term4

    Td = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )
    return Td
