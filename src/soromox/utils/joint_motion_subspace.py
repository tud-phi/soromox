__all__ = [
    "joint_dSTF_dq",
    "joint_dSTF_dq_direction",
    "joint_dSdq_qd",
    "joint_motion_subspace_with_derivatives",
]

import jax.numpy as jnp
from jax import Array, lax, vmap

from soromox.utils.lie_algebra import se3
from soromox.utils.numerics import safe_norm

_THETA_SERIES_THRESHOLD = 1.0e-2


def joint_motion_subspace_with_derivatives(
    H,
    B_xi,
    xi_ref,
    q,
    qd,
    qdd,
    eps,
    q_directions=None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Compute the joint transform, motion subspace, and derivatives.

    This helper evaluates the constant-strain soft-joint motion subspace over
    an interval of arclength "H". The strain field is assumed constant over
    the interval and parameterized as "xi = B_xi @ q + xi_ref". The returned
    matrices are expressed in the local body-frame convention used by the PCS
    kinematic recursions.

    Args:
        H: Scalar interval length or local arclength over which the constant
            strain is integrated.
        B_xi: Strain basis matrix with shape "(6, num_dofs)".
        xi_ref: Reference strain vector with shape "(6,)".
        q: Generalized coordinates with shape "(num_dofs,)".
        qd: Generalized velocities with shape "(num_dofs,)".
        qdd: Generalized accelerations with shape "(num_dofs,)".
        eps: Small positive tolerance used to avoid singular divisions. The
            small-angle formulas switch branches at a fixed ``1e-2`` threshold.
        q_directions: Optional configuration directions with shape
            ``(num_dofs, num_directions)``. When provided, the three
            configuration derivatives are applied to these directions inside
            their analytical series instead of being materialized in full.

    Returns:
        A tuple "(g, S, Sd, dSdq_qd, dSdq_qdd, dSddq_qd)".

        - g: Transformation matrix over the interval, shape "(4, 4)".
        - S: Joint motion subspace over the interval, shape "(6, num_dofs)".
        - Sd: Time derivative of S induced by qd, shape "(6, num_dofs)".
        - dSdq_qd: Jacobian of ``S(q) @ qd`` with respect to ``q``.
        - dSdq_qdd: Jacobian of ``S(q) @ qdd`` with respect to ``q``.
        - dSddq_qd: Jacobian of ``Sd(q, qd) @ qd`` with respect to ``q``.

        The derivative shapes are ``(6, num_dofs)`` when ``q_directions`` is
        omitted and ``(6, num_directions)`` otherwise.
    """

    I_theta = jnp.diag(jnp.array([1, 1, 1, 0, 0, 0]))
    xi = B_xi @ q + xi_ref
    Omega = H * xi
    Z = H * B_xi
    Omegad = Z @ qd

    k = Omega[:3]
    kd = Omegad[:3]
    theta = jnp.maximum(safe_norm(Omega[:3]), eps)
    thetad = (kd @ k) / theta

    Omegahat = se3.hat(Omega)
    Omegahatp2 = Omegahat @ Omegahat
    Omegahatp3 = Omegahatp2 @ Omegahat

    adjOmega = se3.small_adjoint(Omega)
    adjOmegap2 = adjOmega @ adjOmega
    adjOmegap3 = adjOmegap2 @ adjOmega
    adjOmegap4 = adjOmegap3 @ adjOmega
    adjOmegap = jnp.stack((adjOmega, adjOmegap2, adjOmegap3, adjOmegap4))

    adjOmegad = se3.small_adjoint(Omegad)
    adjOmegap2d = adjOmegad @ adjOmega + adjOmega @ adjOmegad
    adjOmegap3d = adjOmegap2d @ adjOmega + adjOmegap2 @ adjOmegad
    adjOmegap4d = adjOmegap3d @ adjOmega + adjOmegap3 @ adjOmegad
    adjOmegapd = jnp.stack((adjOmegad, adjOmegap2d, adjOmegap3d, adjOmegap4d))

    def _g_series_branch() -> tuple[Array, Array, Array, Array, Array, Array]:
        # Keep the transform identical to the stable implementation used by
        # the main PCS kinematics while using the small-angle derivative
        # coefficients below.
        g = se3.exp(Omega, eps=eps)
        f = jnp.array([1 / 2, 1 / 6, 1 / 24, 1 / 120])
        fd = jnp.zeros((4,))
        fdd = jnp.zeros((4,))

        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0]
            + f[1] * adjOmegap[1]
            + f[2] * adjOmegap[2]
            + f[3] * adjOmegap[3]
        )

        Td = (
            f[0] * adjOmegapd[0]
            + f[1] * adjOmegapd[1]
            + f[2] * adjOmegapd[2]
            + f[3] * adjOmegapd[3]
        )

        return (f, fd, fdd, g, T, Td)

    def _g_general_branch(
        theta: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        tp2 = theta * theta
        tp3 = tp2 * theta
        tp4 = tp3 * theta
        tp5 = tp4 * theta
        tp6 = tp5 * theta
        tp7 = tp6 * theta
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        t1 = theta * sintheta
        t2 = theta * costheta
        t3 = -8 + (8 - tp2) * costheta + 5 * t1
        t4 = -8 * theta + (15 - tp2) * sintheta - 7 * t2
        t3d = 5 * sintheta + sintheta * (tp2 - 8) + 3 * theta * costheta
        t4d = -8 + 5 * theta * sintheta + (15 - tp2) * costheta - 7 * costheta

        f = jnp.stack(
            (
                (4 - 4 * costheta - t1) / (2 * tp2),
                (4 * theta - 5 * sintheta + t2) / (2 * tp3),
                (2 - 2 * costheta - t1) / (2 * tp4),
                (2 * theta - 3 * sintheta + t2) / (2 * tp5),
            )
        )

        fd = jnp.stack(
            (
                t3 / (2 * tp3),
                t4 / (2 * tp4),
                t3 / (2 * tp5),
                t4 / (2 * tp6),
            )
        )

        fdd = jnp.stack(
            (
                (theta * t3d - 3 * t3) / (2 * tp4),
                (theta * t4d - 4 * t4) / (2 * tp5),
                (theta * t3d - 5 * t3) / (2 * tp6),
                (theta * t4d - 6 * t4) / (2 * tp7),
            )
        )

        g = (
            jnp.eye(4)
            + Omegahat
            + ((1 - costheta) / tp2) * Omegahatp2
            + ((theta - sintheta) / tp3) * Omegahatp3
        )

        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0]
            + f[1] * adjOmegap[1]
            + f[2] * adjOmegap[2]
            + f[3] * adjOmegap[3]
        )

        Td = (
            fd[0] * thetad * adjOmegap[0]
            + f[0] * adjOmegapd[0]
            + fd[1] * thetad * adjOmegap[1]
            + f[1] * adjOmegapd[1]
            + fd[2] * thetad * adjOmegap[2]
            + f[2] * adjOmegapd[2]
            + fd[3] * thetad * adjOmegap[3]
            + f[3] * adjOmegapd[3]
        )

        return (f, fd, fdd, g, T, Td)

    f, fd, fdd, g, T, Td = lax.cond(
        theta <= _THETA_SERIES_THRESHOLD,
        lambda _: _g_series_branch(),
        lambda _: _g_general_branch(theta),
        operand=None,
    )

    S = T @ Z
    Sd = Td @ Z
    Zqdd = Z @ qdd
    Z_derivative = Z if q_directions is None else Z @ q_directions

    def _adjoint_power_or_eye(k: int) -> Array:
        return jnp.eye(6) if k < 0 else adjOmegap[k]

    def _sum_terms(terms) -> Array:
        return jnp.sum(jnp.stack(terms), axis=0)

    def _S_series_branch() -> tuple[Array, Array, Array]:
        dSdq_qd_terms = []
        dSdq_qdd_terms = []
        dSddq_qd_terms = []
        for r in range(4):
            for u in range(1, r + 2):
                adjOmegap1 = _adjoint_power_or_eye(u - 2)
                adjOmegap2 = _adjoint_power_or_eye(r - u)

                dSdq_qd_terms.append(
                    -f[r]
                    * (
                        adjOmegap1
                        @ se3.small_adjoint(adjOmegap2 @ Omegad)
                        @ Z_derivative
                    )
                )
                dSdq_qdd_terms.append(
                    -f[r]
                    * (adjOmegap1 @ se3.small_adjoint(adjOmegap2 @ Zqdd) @ Z_derivative)
                )

                for p in range(1, u):
                    adjOmegap3 = _adjoint_power_or_eye(p - 2)
                    adjOmegap4 = _adjoint_power_or_eye(u - p - 2)

                    dSddq_qd_terms.append(
                        -f[r]
                        * (
                            adjOmegap3
                            @ se3.small_adjoint(
                                adjOmegap4 @ adjOmegapd[0] @ adjOmegap2 @ Omegad
                            )
                            @ Z_derivative
                        )
                    )

                for p in range(1, r - u + 2):
                    adjOmegap3 = _adjoint_power_or_eye(p - 2)
                    adjOmegap4 = _adjoint_power_or_eye(r - u - p)

                    dSddq_qd_terms.append(
                        -f[r]
                        * (
                            adjOmegap1
                            @ adjOmegapd[0]
                            @ adjOmegap3
                            @ se3.small_adjoint(adjOmegap4 @ Omegad)
                            @ Z_derivative
                        )
                    )
        return (
            _sum_terms(dSdq_qd_terms),
            _sum_terms(dSdq_qdd_terms),
            _sum_terms(dSddq_qd_terms),
        )

    def _S_general_branch(theta: Array) -> tuple[Array, Array, Array]:
        dSdq_qd_terms = []
        dSdq_qdd_terms = []
        dSddq_qd_terms = []
        for r in range(4):
            adjr = adjOmegap[r]
            adjrd = adjOmegapd[r]

            dSdq_qd_terms.append(
                (1 / theta)
                * fd[r]
                * jnp.outer(
                    adjr @ Omegad,
                    Omega @ (I_theta @ Z_derivative),
                )
            )

            dSdq_qdd_terms.append(
                (1 / theta)
                * fd[r]
                * jnp.outer(
                    adjr @ Zqdd,
                    Omega @ (I_theta @ Z_derivative),
                )
            )

            term1 = (1 / theta) * jnp.outer(
                ((fdd[r] * thetad) * adjr + fd[r] * adjrd) @ Omegad,
                Omega @ (I_theta @ Z_derivative),
            )

            term2 = (
                (1 / theta)
                * fd[r]
                * jnp.outer(
                    adjr @ Omegad,
                    (Omegad - (thetad / theta) * Omega) @ (I_theta @ Z_derivative),
                )
            )

            dSddq_qd_terms.extend([term1, term2])
            for u in range(1, r + 2):
                adjOmegap1 = _adjoint_power_or_eye(u - 2)
                adjOmegap2 = _adjoint_power_or_eye(r - u)

                dSdq_qd_terms.append(
                    -f[r]
                    * (
                        adjOmegap1
                        @ se3.small_adjoint(adjOmegap2 @ Omegad)
                        @ Z_derivative
                    )
                )
                dSdq_qdd_terms.append(
                    -f[r]
                    * (adjOmegap1 @ se3.small_adjoint(adjOmegap2 @ Zqdd) @ Z_derivative)
                )
                dSddq_qd_terms.append(
                    -fd[r]
                    * thetad
                    * (
                        adjOmegap1
                        @ se3.small_adjoint(adjOmegap2 @ Omegad)
                        @ Z_derivative
                    )
                )

                for p in range(1, u):
                    adjOmegap3 = _adjoint_power_or_eye(p - 2)
                    adjOmegap4 = _adjoint_power_or_eye(u - p - 2)

                    dSddq_qd_terms.append(
                        -f[r]
                        * (
                            adjOmegap3
                            @ se3.small_adjoint(
                                adjOmegap4 @ adjOmegapd[0] @ adjOmegap2 @ Omegad
                            )
                            @ Z_derivative
                        )
                    )

                for p in range(1, r - u + 2):
                    adjOmegap3 = _adjoint_power_or_eye(p - 2)
                    adjOmegap4 = _adjoint_power_or_eye(r - u - p)

                    dSddq_qd_terms.append(
                        -f[r]
                        * (
                            adjOmegap1
                            @ adjOmegapd[0]
                            @ adjOmegap3
                            @ se3.small_adjoint(adjOmegap4 @ Omegad)
                            @ Z_derivative
                        )
                    )

        return (
            _sum_terms(dSdq_qd_terms),
            _sum_terms(dSdq_qdd_terms),
            _sum_terms(dSddq_qd_terms),
        )

    dSdq_qd, dSdq_qdd, dSddq_qd = lax.cond(
        theta <= _THETA_SERIES_THRESHOLD,
        lambda _: _S_series_branch(),
        lambda _: _S_general_branch(theta),
        operand=None,
    )

    return (g, S, Sd, dSdq_qd, dSdq_qdd, dSddq_qd)


def joint_dSdq_qd(H, B_xi, xi_ref, q, qd, eps) -> Array:
    """
    Compute only dS/dq contracted with qd.

    Args:
        H: Scalar interval length.
        B_xi: Strain basis matrix, shape (6, num_dofs).
        xi_ref: Reference strain vector, shape (6,).
        q: Generalized coordinates, shape (num_dofs,).
        qd: Generalized velocities, shape (num_dofs,).
        eps: Small positive tolerance used to avoid singular divisions. The
            small-angle formulas switch branches at a fixed ``1e-2`` threshold.

    Returns:
        dSdq_qd: Jacobian of ``S(q) @ qd`` with respect to ``q``, shape
            ``(6, num_dofs)``.
    """

    # Quantities required by both branches
    xi = B_xi @ q + xi_ref
    Omega = H * xi
    Z = H * B_xi
    Omegad = Z @ qd

    theta = jnp.maximum(safe_norm(Omega[:3]), eps)

    # Powers of ad_Omega
    adjOmega = se3.small_adjoint(Omega)
    adjOmegap2 = adjOmega @ adjOmega
    adjOmegap3 = adjOmegap2 @ adjOmega
    adjOmegap4 = adjOmegap3 @ adjOmega

    adjOmegap = jnp.stack(
        (
            adjOmega,
            adjOmegap2,
            adjOmegap3,
            adjOmegap4,
        )
    )

    I6 = jnp.eye(6, dtype=Omega.dtype)

    def _adjoint_power_or_eye(k: int) -> Array:
        return I6 if k < 0 else adjOmegap[k]

    def _series_branch(_: Array) -> Array:
        f = jnp.array(
            [1 / 2, 1 / 6, 1 / 24, 1 / 120],
            dtype=Omega.dtype,
        )

        result = jnp.zeros_like(Z)

        for r in range(4):
            for u in range(1, r + 2):
                adj1 = _adjoint_power_or_eye(u - 2)
                adj2 = _adjoint_power_or_eye(r - u)

                result = result - f[r] * (adj1 @ se3.small_adjoint(adj2 @ Omegad) @ Z)

        return result

    def _general_branch(theta: Array) -> Array:
        tp2 = theta * theta
        tp3 = tp2 * theta
        tp4 = tp3 * theta
        tp5 = tp4 * theta
        tp6 = tp5 * theta

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        t1 = theta * sintheta
        t2 = theta * costheta

        t3 = -8 + (8 - tp2) * costheta + 5 * t1
        t4 = -8 * theta + (15 - tp2) * sintheta - 7 * t2

        f = jnp.stack(
            (
                (4 - 4 * costheta - t1) / (2 * tp2),
                (4 * theta - 5 * sintheta + t2) / (2 * tp3),
                (2 - 2 * costheta - t1) / (2 * tp4),
                (2 * theta - 3 * sintheta + t2) / (2 * tp5),
            )
        )

        fd = jnp.stack(
            (
                t3 / (2 * tp3),
                t4 / (2 * tp4),
                t3 / (2 * tp5),
                t4 / (2 * tp6),
            )
        )

        rotational_projection = Omega[:3] @ Z[:3, :]

        result = jnp.zeros_like(Z)

        for r in range(4):
            adjr = adjOmegap[r]

            result = result + (fd[r] / theta) * jnp.outer(
                adjr @ Omegad,
                rotational_projection,
            )

            for u in range(1, r + 2):
                adj1 = _adjoint_power_or_eye(u - 2)
                adj2 = _adjoint_power_or_eye(r - u)

                result = result - f[r] * (adj1 @ se3.small_adjoint(adj2 @ Omegad) @ Z)

        return result

    return lax.cond(
        theta <= _THETA_SERIES_THRESHOLD,
        _series_branch,
        _general_branch,
        operand=theta,
    )


def joint_dSTF_dq(H, B_xi, xi_ref, q, wrench, eps) -> Array:
    """Compute ``d(S.T @ wrench) / dq`` without a rank-three ``dS/dq``.

    The result is equivalent to evaluating :func:`joint_dSdq_qd` along every
    generalized-coordinate basis direction and contracting each result with
    ``wrench``. Coadjoint actions perform the wrench contraction before the
    strain-basis products, avoiding the intermediate array with shape
    ``(num_dofs, 6, num_dofs)``.

    Args:
        H: Scalar interval length.
        B_xi: Strain basis matrix with shape ``(6, num_dofs)``.
        xi_ref: Reference strain vector with shape ``(6,)``.
        q: Generalized coordinates with shape ``(num_dofs,)``.
        wrench: Spatial dual vector with shape ``(6,)``.
        eps: Small positive tolerance used to avoid singular divisions.

    Returns:
        Jacobian of ``S(q).T @ wrench`` with respect to ``q``, with shape
        ``(num_dofs, num_dofs)``.
    """
    xi = B_xi @ q + xi_ref
    Omega = H * xi
    Z = H * B_xi
    theta = jnp.maximum(safe_norm(Omega[:3]), eps)

    adjOmega = se3.small_adjoint(Omega)
    adjOmegap2 = adjOmega @ adjOmega
    adjOmegap3 = adjOmegap2 @ adjOmega
    adjOmegap4 = adjOmegap3 @ adjOmega
    adjOmegap = jnp.stack((adjOmega, adjOmegap2, adjOmegap3, adjOmegap4))
    identity = jnp.eye(6, dtype=Omega.dtype)

    def adjoint_power_or_eye(power: int) -> Array:
        return identity if power < 0 else adjOmegap[power]

    def contracted_adjoint_term(adj1: Array, adj2: Array) -> Array:
        direction_twists = adj2 @ Z
        transported_wrench = adj1.T @ wrench
        coadjoint_actions = vmap(
            se3.coadjoint_action,
            in_axes=(1, None),
            out_axes=1,
        )(direction_twists, transported_wrench)
        return coadjoint_actions.T @ Z

    def series_branch(_: Array) -> Array:
        coefficients = jnp.array(
            [1 / 2, 1 / 6, 1 / 24, 1 / 120],
            dtype=Omega.dtype,
        )
        result = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        for r in range(4):
            for u in range(1, r + 2):
                adj1 = adjoint_power_or_eye(u - 2)
                adj2 = adjoint_power_or_eye(r - u)
                result = result + coefficients[r] * contracted_adjoint_term(
                    adj1,
                    adj2,
                )
        return result

    def general_branch(theta: Array) -> Array:
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta5 = theta4 * theta
        theta6 = theta5 * theta
        cosine = jnp.cos(theta)
        sine = jnp.sin(theta)
        theta_sine = theta * sine
        theta_cosine = theta * cosine
        t3 = -8 + (8 - theta2) * cosine + 5 * theta_sine
        t4 = -8 * theta + (15 - theta2) * sine - 7 * theta_cosine

        coefficients = jnp.stack(
            (
                (4 - 4 * cosine - theta_sine) / (2 * theta2),
                (4 * theta - 5 * sine + theta_cosine) / (2 * theta3),
                (2 - 2 * cosine - theta_sine) / (2 * theta4),
                (2 * theta - 3 * sine + theta_cosine) / (2 * theta5),
            )
        )
        coefficient_derivatives = jnp.stack(
            (
                t3 / (2 * theta3),
                t4 / (2 * theta4),
                t3 / (2 * theta5),
                t4 / (2 * theta6),
            )
        )
        rotational_projection = Omega[:3] @ Z[:3, :]

        result = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        for r in range(4):
            directional_coefficients = wrench @ adjOmegap[r] @ Z
            result = result + (
                coefficient_derivatives[r]
                / theta
                * jnp.outer(directional_coefficients, rotational_projection)
            )
            for u in range(1, r + 2):
                adj1 = adjoint_power_or_eye(u - 2)
                adj2 = adjoint_power_or_eye(r - u)
                result = result + coefficients[r] * contracted_adjoint_term(
                    adj1,
                    adj2,
                )
        return result

    return lax.cond(
        theta <= _THETA_SERIES_THRESHOLD,
        series_branch,
        general_branch,
        operand=theta,
    )


def joint_dSTF_dq_direction(
    H,
    B_xi,
    xi_ref,
    q,
    wrench,
    q_directions,
    eps,
) -> Array:
    """Apply ``d(S.T @ wrench) / dq`` to one or more directions.

    This is the directional counterpart of :func:`joint_dSTF_dq`. It
    contracts the configuration directions inside each analytical term and
    therefore never materializes the square generalized-coordinate Jacobian.

    Args:
        H: Scalar interval length.
        B_xi: Strain basis matrix with shape ``(6, num_dofs)``.
        xi_ref: Reference strain vector with shape ``(6,)``.
        q: Generalized coordinates with shape ``(num_dofs,)``.
        wrench: Spatial dual vector with shape ``(6,)``.
        q_directions: Configuration directions with shape
            ``(num_dofs, num_directions)``.
        eps: Small positive tolerance used to avoid singular divisions.

    Returns:
        Directional derivatives with shape
        ``(num_dofs, num_directions)``.
    """
    xi = B_xi @ q + xi_ref
    Omega = H * xi
    Z = H * B_xi
    theta = jnp.maximum(safe_norm(Omega[:3]), eps)

    adjOmega = se3.small_adjoint(Omega)
    adjOmegap2 = adjOmega @ adjOmega
    adjOmegap3 = adjOmegap2 @ adjOmega
    adjOmegap4 = adjOmegap3 @ adjOmega
    adjOmegap = jnp.stack((adjOmega, adjOmegap2, adjOmegap3, adjOmegap4))
    identity = jnp.eye(6, dtype=Omega.dtype)
    direction_strains = Z @ q_directions

    def adjoint_power_or_eye(power: int) -> Array:
        return identity if power < 0 else adjOmegap[power]

    def contracted_adjoint_term(adj1: Array, adj2: Array) -> Array:
        coordinate_twists = adj2 @ Z
        transported_wrench = adj1.T @ wrench
        coadjoint_actions = vmap(
            se3.coadjoint_action,
            in_axes=(1, None),
            out_axes=1,
        )(coordinate_twists, transported_wrench)
        return coadjoint_actions.T @ direction_strains

    def series_branch(_: Array) -> Array:
        coefficients = jnp.array(
            [1 / 2, 1 / 6, 1 / 24, 1 / 120],
            dtype=Omega.dtype,
        )
        result = jnp.zeros(
            (q.shape[0], q_directions.shape[1]),
            dtype=q.dtype,
        )
        for r in range(4):
            for u in range(1, r + 2):
                adj1 = adjoint_power_or_eye(u - 2)
                adj2 = adjoint_power_or_eye(r - u)
                result = result + coefficients[r] * contracted_adjoint_term(
                    adj1,
                    adj2,
                )
        return result

    def general_branch(theta: Array) -> Array:
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta5 = theta4 * theta
        theta6 = theta5 * theta
        cosine = jnp.cos(theta)
        sine = jnp.sin(theta)
        theta_sine = theta * sine
        theta_cosine = theta * cosine
        t3 = -8 + (8 - theta2) * cosine + 5 * theta_sine
        t4 = -8 * theta + (15 - theta2) * sine - 7 * theta_cosine

        coefficients = jnp.stack(
            (
                (4 - 4 * cosine - theta_sine) / (2 * theta2),
                (4 * theta - 5 * sine + theta_cosine) / (2 * theta3),
                (2 - 2 * cosine - theta_sine) / (2 * theta4),
                (2 * theta - 3 * sine + theta_cosine) / (2 * theta5),
            )
        )
        coefficient_derivatives = jnp.stack(
            (
                t3 / (2 * theta3),
                t4 / (2 * theta4),
                t3 / (2 * theta5),
                t4 / (2 * theta6),
            )
        )
        rotational_direction = Omega[:3] @ direction_strains[:3, :]

        result = jnp.zeros(
            (q.shape[0], q_directions.shape[1]),
            dtype=q.dtype,
        )
        for r in range(4):
            directional_coefficients = wrench @ adjOmegap[r] @ Z
            result = result + (
                coefficient_derivatives[r]
                / theta
                * jnp.outer(directional_coefficients, rotational_direction)
            )
            for u in range(1, r + 2):
                adj1 = adjoint_power_or_eye(u - 2)
                adj2 = adjoint_power_or_eye(r - u)
                result = result + coefficients[r] * contracted_adjoint_term(
                    adj1,
                    adj2,
                )
        return result

    return lax.cond(
        theta <= _THETA_SERIES_THRESHOLD,
        series_branch,
        general_branch,
        operand=theta,
    )
