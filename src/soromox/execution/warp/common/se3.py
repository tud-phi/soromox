# ruff: noqa: I001, UP018
"""Forward-only Warp Lie-algebra preparation kernels for GVS dynamics.

The implementation applies ``ad`` polynomials directly to spatial vectors
instead of materializing dense ``ad``, tangent, or tangent-derivative
matrices.
"""

from __future__ import annotations

import warp as wp


Vec6d = wp.types.vector(length=6, dtype=wp.float64)
SPATIAL_DIM = 6

wp.set_module_options({"enable_backward": False})


@wp.func
def _ad_action(left: Vec6d, right: Vec6d) -> Vec6d:
    """Apply the SE(3) Lie bracket without forming a dense matrix.

    Args:
        left: Left spatial twist in angular-linear ordering.
        right: Right spatial twist in angular-linear ordering.

    Returns:
        The six-dimensional bracket ``ad(left) @ right``.
    """
    left_omega = wp.vec3d(left[0], left[1], left[2])
    left_linear = wp.vec3d(left[3], left[4], left[5])
    right_omega = wp.vec3d(right[0], right[1], right[2])
    right_linear = wp.vec3d(right[3], right[4], right[5])
    angular = wp.cross(left_omega, right_omega)
    linear = wp.cross(left_linear, right_omega) + wp.cross(left_omega, right_linear)
    return Vec6d(
        angular[0],
        angular[1],
        angular[2],
        linear[0],
        linear[1],
        linear[2],
    )


@wp.func
def _left_coefficients(angle_sq: wp.float64) -> wp.vec4d:
    """Evaluate stable coefficients for the SE(3) left tangent polynomial.

    Args:
        angle_sq: Squared norm of the rotational exponential coordinate.

    Returns:
        Four coefficients multiplying the first through fourth powers of the
        spatial adjoint operator.
    """
    c1 = wp.float64(0.0)
    c2 = wp.float64(0.0)
    c3 = wp.float64(0.0)
    c4 = wp.float64(0.0)
    if angle_sq <= wp.float64(0.00580466633864119):
        x = angle_sq
        c1 = wp.float64(0.5) + x * x * (
            -wp.float64(1.0 / 720.0)
            + x * (wp.float64(1.0 / 20160.0) - x * wp.float64(1.0 / 1209600.0))
        )
        c2 = wp.float64(1.0 / 6.0) + x * x * (
            -wp.float64(1.0 / 5040.0)
            + x * (wp.float64(1.0 / 181440.0) - x * wp.float64(1.0 / 13305600.0))
        )
        c3 = wp.float64(1.0 / 24.0) + x * (
            -wp.float64(1.0 / 360.0)
            + x
            * (
                wp.float64(1.0 / 13440.0)
                + x * (-wp.float64(1.0 / 907200.0) + x * wp.float64(1.0 / 95800320.0))
            )
        )
        c4 = wp.float64(1.0 / 120.0) + x * (
            -wp.float64(1.0 / 2520.0)
            + x
            * (
                wp.float64(1.0 / 120960.0)
                + x
                * (-wp.float64(1.0 / 9979200.0) + x * wp.float64(1.0 / 1245404160.0))
            )
        )
    else:
        theta = wp.sqrt(angle_sq)
        sine = wp.sin(theta)
        cosine = wp.cos(theta)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        c1 = (wp.float64(4.0) - wp.float64(4.0) * cosine - theta * sine) / (
            wp.float64(2.0) * theta2
        )
        c2 = (wp.float64(4.0) * theta - wp.float64(5.0) * sine + theta * cosine) / (
            wp.float64(2.0) * theta3
        )
        c3 = (wp.float64(2.0) - wp.float64(2.0) * cosine - theta * sine) / (
            wp.float64(2.0) * theta4
        )
        c4 = (wp.float64(2.0) * theta - wp.float64(3.0) * sine + theta * cosine) / (
            wp.float64(2.0) * theta5
        )
    return wp.vec4d(c1, c2, c3, c4)


@wp.func
def _left_coefficient_x_derivatives(angle_sq: wp.float64) -> wp.vec4d:
    """Differentiate left-tangent coefficients with respect to angle squared.

    Args:
        angle_sq: Squared norm of the rotational exponential coordinate.

    Returns:
        Derivatives of the four left-tangent coefficients.
    """
    d1 = wp.float64(0.0)
    d2 = wp.float64(0.0)
    d3 = wp.float64(0.0)
    d4 = wp.float64(0.0)
    if angle_sq <= wp.float64(0.00580466633864119):
        x = angle_sq
        d1 = x * (
            -wp.float64(1.0 / 360.0)
            + x * (wp.float64(1.0 / 6720.0) - x * wp.float64(1.0 / 302400.0))
        )
        d2 = x * (
            -wp.float64(1.0 / 2520.0)
            + x * (wp.float64(1.0 / 60480.0) - x * wp.float64(1.0 / 3326400.0))
        )
        d3 = -wp.float64(1.0 / 360.0) + x * (
            wp.float64(1.0 / 6720.0)
            + x * (-wp.float64(1.0 / 302400.0) + x * wp.float64(1.0 / 23950080.0))
        )
        d4 = -wp.float64(1.0 / 2520.0) + x * (
            wp.float64(1.0 / 60480.0)
            + x * (-wp.float64(1.0 / 3326400.0) + x * wp.float64(1.0 / 311351040.0))
        )
    else:
        theta = wp.sqrt(angle_sq)
        sine = wp.sin(theta)
        cosine = wp.cos(theta)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        theta6 = theta3 * theta3
        theta7 = theta6 * theta
        common = (
            -wp.float64(8.0)
            + (wp.float64(8.0) - theta2) * cosine
            + wp.float64(5.0) * theta * sine
        )
        alternate = (
            -wp.float64(8.0) * theta
            + (wp.float64(15.0) - theta2) * sine
            - wp.float64(7.0) * theta * cosine
        )
        d1 = common / (wp.float64(4.0) * theta4)
        d2 = alternate / (wp.float64(4.0) * theta5)
        d3 = common / (wp.float64(4.0) * theta6)
        d4 = alternate / (wp.float64(4.0) * theta7)
    return wp.vec4d(d1, d2, d3, d4)


@wp.func
def _left_action(xi: Vec6d, value: Vec6d, coefficients: wp.vec4d) -> Vec6d:
    """Apply the SE(3) left tangent to a spatial vector.

    Args:
        xi: Spatial exponential coordinate.
        value: Spatial vector to transform.
        coefficients: Polynomial coefficients from :func:`_left_coefficients`.

    Returns:
        Left-tangent action on ``value``.
    """
    result = value
    power = value
    for order in range(4):
        power = _ad_action(xi, power)
        result += coefficients[order] * power
    return result


@wp.func
def _left_total_derivative_action(
    xi: Vec6d,
    xi_dot: Vec6d,
    value: Vec6d,
    value_dot: Vec6d,
    coefficients: wp.vec4d,
    coefficient_directions: wp.vec4d,
) -> Vec6d:
    """Evaluate the total derivative of a left-tangent action.

    Args:
        xi: Spatial exponential coordinate.
        xi_dot: Directional derivative of ``xi``.
        value: Spatial vector acted on by the tangent.
        value_dot: Directional derivative of ``value``.
        coefficients: Left-tangent polynomial coefficients.
        coefficient_directions: Directional derivatives of the coefficients.

    Returns:
        Directional derivative of ``J_l(xi) @ value``.
    """
    result = value_dot
    power = value
    power_dot = value_dot
    for order in range(4):
        power_previous = power
        power = _ad_action(xi, power_previous)
        power_dot = _ad_action(xi_dot, power_previous) + _ad_action(xi, power_dot)
        result += coefficient_directions[order] * power
        result += coefficients[order] * power_dot
    return result


@wp.func
def _forward_coefficients(angle_sq: wp.float64) -> wp.vec3d:
    """Evaluate stable coefficients for the SE(3) exponential map.

    Args:
        angle_sq: Squared norm of the rotational exponential coordinate.

    Returns:
        ``(sinc, cosc, tanc)`` coefficients used by rotation and translation.
    """
    sinc = wp.float64(0.0)
    cosc = wp.float64(0.0)
    tanc = wp.float64(0.0)
    if angle_sq <= wp.float64(0.0024607823917256556):
        x = angle_sq
        sinc = wp.float64(1.0) + x * (
            -wp.float64(1.0 / 6.0)
            + x
            * (
                wp.float64(1.0 / 120.0)
                + x * (-wp.float64(1.0 / 5040.0) + x * wp.float64(1.0 / 362880.0))
            )
        )
        cosc = wp.float64(0.5) + x * (
            -wp.float64(1.0 / 24.0)
            + x
            * (
                wp.float64(1.0 / 720.0)
                + x * (-wp.float64(1.0 / 40320.0) + x * wp.float64(1.0 / 3628800.0))
            )
        )
        tanc = wp.float64(1.0 / 6.0) + x * (
            -wp.float64(1.0 / 120.0)
            + x
            * (
                wp.float64(1.0 / 5040.0)
                + x * (-wp.float64(1.0 / 362880.0) + x * wp.float64(1.0 / 39916800.0))
            )
        )
    else:
        theta = wp.sqrt(angle_sq)
        sinc = wp.sin(theta) / theta
        cosc = (wp.float64(1.0) - wp.cos(theta)) / angle_sq
        tanc = (theta - wp.sin(theta)) / (angle_sq * theta)
    return wp.vec3d(sinc, cosc, tanc)


@wp.func
def _rotation_entry(
    omega: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    row: int,
    column: int,
) -> wp.float64:
    """Evaluate one entry of the SO(3) exponential rotation matrix.

    Args:
        omega: Rotational exponential coordinate.
        angle_sq: Squared norm of ``omega``.
        forward: Stable exponential-map coefficients.
        row: Matrix row index.
        column: Matrix column index.

    Returns:
        The selected rotation-matrix entry.
    """
    delta = wp.float64(0.0)
    if row == column:
        delta = wp.float64(1.0)
    skew = wp.float64(0.0)
    if row == 0 and column == 1:
        skew = -omega[2]
    elif row == 0 and column == 2:
        skew = omega[1]
    elif row == 1 and column == 0:
        skew = omega[2]
    elif row == 1 and column == 2:
        skew = -omega[0]
    elif row == 2 and column == 0:
        skew = -omega[1]
    elif row == 2 and column == 1:
        skew = omega[0]
    square = omega[row] * omega[column] - angle_sq * delta
    return delta + forward[0] * skew + forward[1] * square


@wp.func
def _translation(omega: wp.vec3d, linear: wp.vec3d, forward: wp.vec3d) -> wp.vec3d:
    """Evaluate the translational component of an SE(3) exponential.

    Args:
        omega: Rotational exponential coordinate.
        linear: Translational exponential coordinate.
        forward: Stable exponential-map coefficients.

    Returns:
        Translation of the resulting homogeneous transformation.
    """
    first = wp.cross(omega, linear)
    second = wp.cross(omega, first)
    return linear + forward[1] * first + forward[2] * second


@wp.func
def _adjoint_inverse_entry(
    omega: wp.vec3d,
    translation: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    row: int,
    column: int,
) -> wp.float64:
    """Evaluate one entry of an inverse SE(3) adjoint matrix.

    Args:
        omega: Rotational exponential coordinate.
        translation: Translation of the SE(3) transformation.
        angle_sq: Squared norm of ``omega``.
        forward: Stable exponential-map coefficients.
        row: Adjoint row index.
        column: Adjoint column index.

    Returns:
        The selected inverse-adjoint entry.
    """
    value = wp.float64(0.0)
    if row < 3 and column < 3:
        value = _rotation_entry(omega, angle_sq, forward, column, row)
    elif row >= 3 and column >= 3:
        value = _rotation_entry(omega, angle_sq, forward, column - 3, row - 3)
    elif row >= 3 and column < 3:
        local_row = row - 3
        for k in range(3):
            skew = wp.float64(0.0)
            if k == 0 and column == 1:
                skew = -translation[2]
            elif k == 0 and column == 2:
                skew = translation[1]
            elif k == 1 and column == 0:
                skew = translation[2]
            elif k == 1 and column == 2:
                skew = -translation[0]
            elif k == 2 and column == 0:
                skew = -translation[1]
            elif k == 2 and column == 1:
                skew = translation[0]
            value -= _rotation_entry(omega, angle_sq, forward, k, local_row) * skew
    return value


@wp.func
def _adjoint_inverse_action(
    omega: wp.vec3d,
    translation: wp.vec3d,
    angle_sq: wp.float64,
    forward: wp.vec3d,
    value: Vec6d,
) -> Vec6d:
    """Apply an inverse SE(3) adjoint without materializing the matrix.

    Args:
        omega: Rotational exponential coordinate.
        translation: Translation of the SE(3) transformation.
        angle_sq: Squared norm of ``omega``.
        forward: Stable exponential-map coefficients.
        value: Spatial vector to transform.

    Returns:
        Inverse-adjoint action on ``value``.
    """
    result = Vec6d()
    for row in range(SPATIAL_DIM):
        row_value = wp.float64(0.0)
        for column in range(SPATIAL_DIM):
            row_value += (
                _adjoint_inverse_entry(
                    omega,
                    translation,
                    angle_sq,
                    forward,
                    row,
                    column,
                )
                * value[column]
            )
        result[row] = row_value
    return result
