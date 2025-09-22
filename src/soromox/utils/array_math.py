__all__ = ["blk_diag", "blk_concat", "compute_weighted_sums"]
from jax import numpy as jnp
from jax import Array, lax

def blk_diag(a: Array) -> Array:
    """
    Create a block diagonal matrix from a tensor of blocks.

    Args:
        a: matrices to be block diagonalized of shape (m, n, o)

    Returns:
        b: block diagonal matrix of shape (m * n, m * o)

    """

    def assign_block_diagonal(i: Array, _b: Array):
        """
        Save the ith block  into the block-diagonal matrix `_b`
        Args:
            i: Index of block which we save into the block-diagonal matrix.
            _b: Block diagonal matrix. Should still have zeros at the ith block.
        Returns
        """
        # Assign the block saved in ith entry of `a` to the ith block-diagonal of `_b`
        # Hint: use `jax.lax.dynamic_update_slice` to update the entries of `_b`
        _b = lax.dynamic_update_slice(
            operand=_b, update=a[i], start_indices=(i * a.shape[1], i * a.shape[2])
        )
        return _b

    # Implement for loop to assign each block in `a` to the block-diagonal of `b`
    # Hint: use `jax.lax.fori_loop` and pass `assign_block_diagonal` as an argument
    b = jnp.zeros((a.shape[0] * a.shape[1], a.shape[0] * a.shape[2]), dtype=a.dtype)
    b = lax.fori_loop(
        lower=0,
        upper=a.shape[0],
        body_fun=assign_block_diagonal,
        init_val=b,
    )

    return b


def blk_concat(a: Array) -> Array:
    """
    Concatenate horizontally (along the columns) a list of N matrices of size (m, n) to create a single matrix of size (m, n * N).

    Args:
        a (Array): matrices to be concatenated of shape (N, m, n)

    Returns:
        b (Array): concatenated matrix of shape (m, N * n)
    """
    b = a.transpose(1, 0, 2).reshape(a.shape[1], -1)
    return b


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
