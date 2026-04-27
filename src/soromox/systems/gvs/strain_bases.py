__all__ = [
    "B_Monomial",
    "B_LegendrePolynomial",
    "B_Chebychev",
    "B_Fourier",
    "B_Gaussian",
    "B_IMQ",
    "dB_Monomial",
    "dB_LegendrePolynomial",
    "dB_Chebychev",
    "dB_Fourier",
    "dB_Gaussian",
    "dB_IMQ",
    "dof_Monomial",
    "dof_LegendrePolynomial",
    "dof_Chebychev",
    "dof_Fourier",
    "dof_Gaussian",
    "dof_IMQ",
]
import jax.numpy as jnp
from jax import Array, lax


def _prepared_basis_params(Bdof: Array, Bodr: Array) -> tuple[Array, Array]:
    """Return flattened integer basis activation/order arrays."""
    return Bdof.flatten().astype(jnp.int32), Bodr.flatten().astype(jnp.int32)


def _standard_widths(Bdof: Array, Bodr: Array) -> Array:
    """Column counts for non-Fourier basis families."""
    return Bdof * (Bodr + 1)


def _fourier_widths(Bdof: Array, Bodr: Array) -> Array:
    """Column counts for the Fourier basis family."""
    return Bdof * (2 * Bodr + 1)


def _row_offsets(widths: Array) -> Array:
    """Starting column of each strain row in the packed basis matrix."""
    return jnp.concatenate(
        [jnp.zeros((1,), dtype=widths.dtype), jnp.cumsum(widths[:-1])]
    )


def _monomial_slots(X: Array, max_dof: int) -> tuple[Array, Array]:
    slots = jnp.arange(max_dof, dtype=jnp.int32)
    slots_float = slots.astype(jnp.float64)
    values = X**slots
    derivative_power = jnp.maximum(slots - 1, 0)
    derivatives = slots_float * X**derivative_power
    derivatives = jnp.where(slots == 0, 0.0, derivatives)
    return values, derivatives


def _polynomial_slots(X: Array, max_dof: int, family: str) -> tuple[Array, Array]:
    z = 2 * X - 1
    slots = jnp.arange(max_dof, dtype=jnp.int32)

    def scan_body(
        carry: tuple[Array, Array, Array, Array], n: Array
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array]]:
        p_nm2, p_nm1, dp_nm2, dp_nm1 = carry
        m = jnp.maximum(n - 1, 1)
        m_float = m.astype(jnp.float64)

        if family == "legendre":
            p_rec = ((2 * m_float + 1) * z * p_nm1 - m_float * p_nm2) / (
                m_float + 1
            )
            dp_rec = (
                (2 * m_float + 1) * (2 * p_nm1 + z * dp_nm1)
                - m_float * dp_nm2
            ) / (m_float + 1)
        elif family == "chebychev":
            p_rec = 2 * z * p_nm1 - p_nm2
            dp_rec = 4 * p_nm1 + 2 * z * dp_nm1 - dp_nm2
        else:
            raise ValueError(f"Unsupported polynomial basis family: {family}")

        is_zero = n == 0
        is_one = n == 1
        p_n = jnp.where(is_zero, 1.0, jnp.where(is_one, z, p_rec))
        dp_n = jnp.where(is_zero, 0.0, jnp.where(is_one, 2.0, dp_rec))
        keep_carry = n <= 1
        next_carry = (
            jnp.where(keep_carry, p_nm2, p_nm1),
            jnp.where(keep_carry, p_nm1, p_n),
            jnp.where(keep_carry, dp_nm2, dp_nm1),
            jnp.where(keep_carry, dp_nm1, dp_n),
        )
        return next_carry, (p_n, dp_n)

    _, (values, derivatives) = lax.scan(
        scan_body,
        (jnp.array(1.0), z, jnp.array(0.0), jnp.array(2.0)),
        slots,
    )
    return values, derivatives


def _fourier_slots(X: Array, max_dof: int) -> tuple[Array, Array]:
    slots = jnp.arange(max_dof, dtype=jnp.int32)
    is_constant = slots == 0
    is_cosine = slots % 2 == 1
    modes = ((slots + 1) // 2).astype(jnp.float64)
    angle = 2 * jnp.pi * modes * X
    values = jnp.where(is_cosine, jnp.cos(angle), jnp.sin(angle))
    values = jnp.where(is_constant, 1.0, values)
    derivatives = jnp.where(
        is_cosine,
        -2 * jnp.pi * modes * jnp.sin(angle),
        2 * jnp.pi * modes * jnp.cos(angle),
    )
    derivatives = jnp.where(is_constant, 0.0, derivatives)
    return values, derivatives


def _gaussian_slots(X: Array, order: Array, max_dof: int) -> tuple[Array, Array]:
    slots = jnp.arange(max_dof, dtype=jnp.int32)
    order_safe = jnp.maximum(order, 1).astype(jnp.float64)
    centers = slots.astype(jnp.float64) / order_safe
    c = 2 * jnp.sqrt(jnp.log(2.0)) * order_safe
    values = jnp.exp(-((X - centers) ** 2) * c**2)
    derivatives = -2 * c**2 * (X - centers) * values
    values = jnp.where(order == 0, jnp.where(slots == 0, 1.0, 0.0), values)
    derivatives = jnp.where(order == 0, 0.0, derivatives)
    return values, derivatives


def _imq_slots(X: Array, order: Array, max_dof: int) -> tuple[Array, Array]:
    slots = jnp.arange(max_dof, dtype=jnp.int32)
    order_safe = jnp.maximum(order, 1).astype(jnp.float64)
    centers = slots.astype(jnp.float64) / order_safe
    c = 2 * jnp.sqrt(3.0) * order_safe
    scaled = 1.0 + ((X - centers) ** 2) * c**2
    values = 1.0 / jnp.sqrt(scaled)
    derivatives = -(c**2) * (X - centers) * scaled ** (-1.5)
    values = jnp.where(order == 0, jnp.where(slots == 0, 1.0, 0.0), values)
    derivatives = jnp.where(order == 0, 0.0, derivatives)
    return values, derivatives


def _basis_pair(
    X: Array,
    Bdof: Array,
    Bodr: Array,
    max_dof: int,
    slot_values_fn,
    widths_fn=_standard_widths,
) -> tuple[Array, Array]:
    """Assemble packed basis and derivative matrices using fixed-size scans."""
    if max_dof == 0:
        empty = jnp.zeros((6, 0), dtype=jnp.float64)
        return empty, empty

    Bdof, Bodr = _prepared_basis_params(Bdof, Bodr)
    widths = widths_fn(Bdof, Bodr)
    offsets = _row_offsets(widths)
    slots = jnp.arange(max_dof, dtype=jnp.int32)
    B0 = jnp.zeros((6, max_dof), dtype=jnp.float64)
    dB0 = jnp.zeros((6, max_dof), dtype=jnp.float64)

    def row_body(
        carry: tuple[Array, Array], row: Array
    ) -> tuple[tuple[Array, Array], None]:
        B, dB = carry
        row_values, row_derivatives = slot_values_fn(X, Bodr[row], max_dof)
        width = widths[row]
        offset = offsets[row]

        def slot_body(
            slot_carry: tuple[Array, Array], slot: Array
        ) -> tuple[tuple[Array, Array], None]:
            B_slot, dB_slot = slot_carry
            valid = slot < width
            col = offset + slot

            def write(update_carry: tuple[Array, Array]) -> tuple[Array, Array]:
                B_update, dB_update = update_carry
                B_update = B_update.at[row, col].set(row_values[slot])
                dB_update = dB_update.at[row, col].set(row_derivatives[slot])
                return B_update, dB_update

            return lax.cond(valid, write, lambda x: x, (B_slot, dB_slot)), None

        (B, dB), _ = lax.scan(slot_body, (B, dB), slots)
        return (B, dB), None

    (B, dB), _ = lax.scan(row_body, (B0, dB0), jnp.arange(6, dtype=jnp.int32))
    return B, dB


def _monomial_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(X, Bdof, Bodr, max_dof, lambda x, _order, n: _monomial_slots(x, n))


def _legendre_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(
        X,
        Bdof,
        Bodr,
        max_dof,
        lambda x, _order, n: _polynomial_slots(x, n, "legendre"),
    )


def _chebychev_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(
        X,
        Bdof,
        Bodr,
        max_dof,
        lambda x, _order, n: _polynomial_slots(x, n, "chebychev"),
    )


def _fourier_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(
        X,
        Bdof,
        Bodr,
        max_dof,
        lambda x, _order, n: _fourier_slots(x, n),
        widths_fn=_fourier_widths,
    )


def _gaussian_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(X, Bdof, Bodr, max_dof, _gaussian_slots)


def _imq_pair(X: Array, Bdof: Array, Bodr: Array, max_dof: int):
    return _basis_pair(X, Bdof, Bodr, max_dof, _imq_slots)


def B_Monomial(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the monomial strain basis evaluated at normalized position ``X``."""
    return _monomial_pair(X, Bdof, Bodr, max_dof)[0]


def B_LegendrePolynomial(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the Legendre strain basis evaluated at normalized position ``X``."""
    return _legendre_pair(X, Bdof, Bodr, max_dof)[0]


def B_Chebychev(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the Chebychev strain basis evaluated at normalized position ``X``."""
    return _chebychev_pair(X, Bdof, Bodr, max_dof)[0]


def B_Fourier(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the Fourier strain basis evaluated at normalized position ``X``."""
    return _fourier_pair(X, Bdof, Bodr, max_dof)[0]


def B_Gaussian(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the Gaussian strain basis evaluated at normalized position ``X``."""
    return _gaussian_pair(X, Bdof, Bodr, max_dof)[0]


def B_IMQ(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Compute the inverse-multiquadric strain basis at normalized position ``X``."""
    return _imq_pair(X, Bdof, Bodr, max_dof)[0]


def dB_Monomial(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Derivative of ``B_Monomial`` with respect to normalized coordinate ``X``."""
    return _monomial_pair(X, Bdof, Bodr, max_dof)[1]


def dB_LegendrePolynomial(
    X: Array, Bdof: Array, Bodr: Array, max_dof: int
) -> Array:
    """Derivative of ``B_LegendrePolynomial`` with respect to ``X``."""
    return _legendre_pair(X, Bdof, Bodr, max_dof)[1]


def dB_Chebychev(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Derivative of ``B_Chebychev`` with respect to normalized coordinate ``X``."""
    return _chebychev_pair(X, Bdof, Bodr, max_dof)[1]


def dB_Fourier(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Derivative of ``B_Fourier`` with respect to normalized coordinate ``X``."""
    return _fourier_pair(X, Bdof, Bodr, max_dof)[1]


def dB_Gaussian(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Derivative of ``B_Gaussian`` with respect to normalized coordinate ``X``."""
    return _gaussian_pair(X, Bdof, Bodr, max_dof)[1]


def dB_IMQ(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """Derivative of ``B_IMQ`` with respect to normalized coordinate ``X``."""
    return _imq_pair(X, Bdof, Bodr, max_dof)[1]


def dof_Monomial(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the monomial basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (Bodr + 1))


def dof_LegendrePolynomial(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the Legendre polynomial basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (Bodr + 1))


def dof_Chebychev(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the Chebychev basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (Bodr + 1))


def dof_Fourier(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the Fourier basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (2 * Bodr + 1))


def dof_Gaussian(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the Gaussian basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (Bodr + 1))


def dof_IMQ(Bdof: Array, Bodr: Array) -> Array:
    """
    Function to compute the number of degrees of freedom for the Inverse Multiquadric basis.

    Args:
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.

    Returns:
        dof (Array): number of degrees of freedom.
    """
    return jnp.sum(Bdof * (Bodr + 1))


# Example usage
if __name__ == "__main__":
    X = jnp.array(2.0)

    Bdof = jnp.array([1, 1, 1, 1, 0, 0])
    Bodr = jnp.array([2, 4, 1, 1, 1, 0])

    max_dof = (
        dof_Monomial(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_Monomial(X, Bdof, Bodr, max_dof)

    max_dof = (
        dof_LegendrePolynomial(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_LegendrePolynomial(X, Bdof, Bodr, max_dof)

    max_dof = (
        dof_Chebychev(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_Chebychev(X, Bdof, Bodr, max_dof)

    max_dof = (
        dof_Fourier(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_Fourier(X, Bdof, Bodr, max_dof)

    max_dof = (
        dof_Gaussian(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_Gaussian(X, Bdof, Bodr, max_dof)

    max_dof = (
        dof_IMQ(Bdof, Bodr).item() + 2
    )  # +2 to check that the 2 last colums are zero
    B = B_IMQ(X, Bdof, Bodr, max_dof)

    jnp.set_printoptions(precision=4, suppress=True)
    print("Resulting B matrix evaluated at X: \n", B)
