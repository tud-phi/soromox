__all__ = [
    "B_Monomial",
    "B_LegendrePolynomial",
    "B_Chebychev",
    "B_Fourier",
    "B_Gaussian",
    "B_IMQ",
    "dof_Monomial",
    "dof_LegendrePolynomial",
    "dof_Chebychev",
    "dof_Fourier",
    "dof_Gaussian",
    "dof_IMQ",
]
import jax.numpy as jnp
from jax import Array, lax


def B_Monomial(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the monomial basis for a given polynomial degree for each degree of freedom evaluated at X.
    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
        B (Array): the constructed monomial basis evaluated at X.
    """
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros((6, max_dof), dtype=jnp.float64)

    def fill_basis(i, carry):
        B, k = carry
        n_terms = Bdof[i] * (Bodr[i] + 1)

        def inner_body(j, inner_carry):
            B_inner, k_inner = inner_carry
            val = X**j
            B_inner = B_inner.at[i, k_inner].set(val)
            return B_inner, k_inner + 1

        B, k = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k))
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


def B_LegendrePolynomial(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the Legendre basis for a given polynomial degree for each degree of freedom evaluated at X.

    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
        B (Array): the constructed Legendre basis evaluated at X.
    """
    # Flatten Bdof and Bodr to ensure they are 1D arrays
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros((6, max_dof), dtype=jnp.float64)

    X = 2 * X - 1  # Transform X from [0, 1] to [-1, 1]

    def fill_basis(i, carry):
        B, k = carry
        n_terms = Bdof[i] * (Bodr[i] + 1)

        def inner_body(j, inner_carry):
            B_inner, k_inner = inner_carry

            def recurrence(n, state):
                P0, P1 = state
                P_next = ((2 * n + 1) * X * P1 - n * P0) / (n + 1)
                return P1, P_next

            def legendre_j(j):
                return lax.cond(
                    j == 0,
                    lambda j_1: 1.0,
                    lambda j_1: lax.cond(
                        j_1 == 1,
                        lambda j_2: X,
                        lambda j_2: lax.fori_loop(1, j_2, recurrence, (1.0, X))[1],
                        j_1,
                    ),
                    j,
                )

            val = legendre_j(j)
            B_inner = B_inner.at[i, k_inner].set(val)
            return B_inner, k_inner + 1

        B, k = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k))
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


def B_Chebychev(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the Chebychev basis for a given polynomial degree for each degree of freedom evaluated at X.

    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
         B (Array): the constructed Chebychev basis evaluated at X.
    """
    # Flatten Bdof and Bodr to ensure they are 1D arrays
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros(
        (6, max_dof), dtype=jnp.float64
    )  # dtype=object to support symbolic powers

    X = 2 * X - 1  # Transform X from [0, 1] to [-1, 1]

    def fill_basis(i, carry):
        B, k = carry
        n_terms = Bdof[i] * (Bodr[i] + 1)

        def inner_body(j, inner_carry):
            B_inner, k_inner = inner_carry
            T0 = 1.0
            T1 = X

            def recurrence(n, state):
                T0, T1 = state
                T_next = 2 * X * T1 - T0
                return T1, T_next

            def cheb_j(j):
                return lax.cond(
                    j == 0,
                    lambda j_1: 1.0,
                    lambda j_1: lax.cond(
                        j_1 == 1,
                        lambda j_2: X,
                        lambda j_2: lax.fori_loop(1, j_2, recurrence, (1.0, X))[1],
                        j_1,
                    ),
                    j,
                )

            val = cheb_j(j)
            B_inner = B_inner.at[i, k_inner].set(val)
            return B_inner, k_inner + 1

        B, k = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k))
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


def B_Fourier(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the Fourier basis for a given polynomial degree for each degree of freedom evaluated at X.

    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
        B (Array): the constructed Fourier basis evaluated at X.
    """
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros((6, max_dof), dtype=jnp.float64)

    def fill_basis(i, carry):
        B, k = carry
        n_terms = Bdof[i] * (Bodr[i] + 1)

        def inner_body(j, inner_carry):
            B_inner, k_inner = inner_carry
            j_val = j + 1
            val_cos = jnp.cos(2 * jnp.pi * (j_val - 1) * X)
            val_sin = jnp.sin(2 * jnp.pi * (j_val - 1) * X)

            B_inner = lax.cond(
                j_val == 1,
                lambda _: B_inner.at[i, k_inner].set(1.0),
                lambda _: (
                    B_inner.at[i, k_inner].set(val_cos).at[i, k_inner + 1].set(val_sin)
                ),
                operand=None,
            )

            k_inner = lax.cond(
                j_val == 1, lambda _: k_inner + 1, lambda _: k_inner + 2, operand=None
            )
            return B_inner, k_inner

        B, k = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k))
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


def B_Gaussian(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the Gaussian basis for a given polynomial degree for each degree of freedom evaluated at X.

    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
        B (Array): the constructed Gaussian basis evaluated at X.
    """
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros((6, max_dof), dtype=jnp.float64)

    def fill_basis(i, carry):
        B, k = carry

        def true_branch(args):
            B, k = args
            B = B.at[i, k].set(1.0)
            return B, k + 1

        def false_branch(args):
            B, k = args
            w = 1.0 / Bodr[i]
            c = 2 * jnp.sqrt(jnp.log(2.0)) / w
            a0 = 0.0

            def inner_body(j, inner_carry):
                B_inner, k_inner, a = inner_carry
                val = jnp.exp(-((X - a) ** 2) * c**2)
                B_inner = B_inner.at[i, k_inner].set(val)
                return B_inner, k_inner + 1, a + w

            n_terms = Bdof[i] * (Bodr[i] + 1)
            B, k, _ = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k, a0))
            return B, k

        B, k = lax.cond(
            (Bodr[i] == 0) & (Bdof[i] == 1), true_branch, false_branch, (B, k)
        )
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


def B_IMQ(X: Array, Bdof: Array, Bodr: Array, max_dof: int) -> Array:
    """
    Function to compute the Inverse Multiquadric basis for a given polynomial degree for each degree of freedom evaluated at X.

    Args:
        X (Array): shape () JAX float
            point at which the basis is evaluated.
        Bdof (Array): list of boolean values indicating the degree of freedom.
            0 means the degree of freedom is not used, 1 means it is used.
        Bodr (Array): contains the order of the polynomial for each degree of freedom.
        max_dof (int): maximum number of degrees of freedom.

    Returns:
        B (Array): the constructed Inverse Multiquadric basis evaluated at X.
    """
    Bdof = Bdof.flatten()
    Bodr = Bodr.flatten()

    B = jnp.zeros((6, max_dof), dtype=jnp.float64)

    def fill_basis(i, carry):
        B, k = carry

        def true_branch(args):
            B, k = args
            B = B.at[i, k].set(1.0)
            return B, k + 1

        def false_branch(args):
            B, k = args
            w = 1.0 / Bodr[i]
            c = 2 * jnp.sqrt(3.0) / w
            a0 = 0.0

            def inner_body(j, inner_carry):
                B_inner, k_inner, a = inner_carry
                val = 1.0 / jnp.sqrt(1.0 + ((X - a) ** 2) * c**2)
                B_inner = B_inner.at[i, k_inner].set(val)
                return B_inner, k_inner + 1, a + w

            n_terms = Bdof[i] * (Bodr[i] + 1)
            B, k, _ = lax.fori_loop(0, n_terms.astype(int), inner_body, (B, k, a0))
            return B, k

        B, k = lax.cond(
            (Bodr[i] == 0) & (Bdof[i] == 1), true_branch, false_branch, (B, k)
        )
        return B, k

    B, _ = lax.fori_loop(0, 6, fill_basis, (B, 0))
    return B


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
