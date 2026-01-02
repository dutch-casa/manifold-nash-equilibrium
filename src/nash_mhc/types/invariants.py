"""Runtime invariant assertions for mathematical properties."""

import jax.numpy as jnp
from jaxtyping import Float, Array


def assert_doubly_stochastic(
    m: Float[Array, "... n n"],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """
    Assert matrix is doubly stochastic (Birkhoff polytope member).

    Checks:
    - Rows sum to 1
    - Columns sum to 1
    - All entries non-negative

    Raises:
        AssertionError: If any invariant is violated.
    """
    row_sums = jnp.sum(m, axis=-1)
    col_sums = jnp.sum(m, axis=-2)
    ones = jnp.ones_like(row_sums)

    row_ok = jnp.allclose(row_sums, ones, rtol=rtol, atol=atol)
    col_ok = jnp.allclose(col_sums, ones, rtol=rtol, atol=atol)
    nonneg_ok = jnp.all(m >= -atol)

    if not row_ok:
        max_row_err = jnp.max(jnp.abs(row_sums - 1.0))
        raise AssertionError(f"Rows must sum to 1, max error: {max_row_err}")
    if not col_ok:
        max_col_err = jnp.max(jnp.abs(col_sums - 1.0))
        raise AssertionError(f"Columns must sum to 1, max error: {max_col_err}")
    if not nonneg_ok:
        min_val = jnp.min(m)
        raise AssertionError(f"All entries must be non-negative, min value: {min_val}")


def assert_simplex(
    w: Float[Array, "... L"],
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> None:
    """
    Assert weights are on the probability simplex.

    Checks:
    - Weights sum to 1
    - All weights non-negative

    Raises:
        AssertionError: If any invariant is violated.
    """
    sums = jnp.sum(w, axis=-1)
    ones = jnp.ones_like(sums)

    sum_ok = jnp.allclose(sums, ones, rtol=rtol, atol=atol)
    nonneg_ok = jnp.all(w >= -atol)

    if not sum_ok:
        max_err = jnp.max(jnp.abs(sums - 1.0))
        raise AssertionError(f"Weights must sum to 1, max error: {max_err}")
    if not nonneg_ok:
        min_val = jnp.min(w)
        raise AssertionError(f"Weights must be non-negative, min value: {min_val}")


def assert_spectral_norm_bounded(
    m: Float[Array, "... n d"],
    bound: float = 1.0,
    rtol: float = 1e-4,
) -> None:
    """
    Assert matrix has spectral norm <= bound (non-expansive).

    Uses SVD to compute the largest singular value.

    Note: This is expensive for large matrices. Use sparingly in production.

    Raises:
        AssertionError: If spectral norm exceeds bound.
    """
    # Flatten batch dimensions for SVD
    original_shape = m.shape
    m_2d = m.reshape(-1, original_shape[-2], original_shape[-1])

    # Compute singular values for each batch element
    max_sv = jnp.array(0.0)
    for i in range(m_2d.shape[0]):
        s = jnp.linalg.svd(m_2d[i], compute_uv=False, full_matrices=False)
        max_sv = jnp.maximum(max_sv, jnp.max(s))

    if max_sv > bound * (1 + rtol):
        raise AssertionError(f"Spectral norm {max_sv} exceeds bound {bound}")


def assert_row_stochastic(
    m: Float[Array, "... n n"],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """
    Assert matrix is row-stochastic (each row sums to 1).

    Used for attention weight matrices after softmax.

    Raises:
        AssertionError: If any invariant is violated.
    """
    row_sums = jnp.sum(m, axis=-1)
    ones = jnp.ones_like(row_sums)

    row_ok = jnp.allclose(row_sums, ones, rtol=rtol, atol=atol)
    nonneg_ok = jnp.all(m >= -atol)

    if not row_ok:
        max_err = jnp.max(jnp.abs(row_sums - 1.0))
        raise AssertionError(f"Rows must sum to 1, max error: {max_err}")
    if not nonneg_ok:
        min_val = jnp.min(m)
        raise AssertionError(f"All entries must be non-negative, min value: {min_val}")


def check_sequence_alignment(
    seq_len: int,
    num_scales: int,
    compression_ratio: int,
) -> bool:
    """
    Check if sequence length is valid for hierarchical decomposition.

    Returns True if seq_len is divisible by compression_ratio^(num_scales-1).
    """
    scale_factor = compression_ratio ** (num_scales - 1)
    return seq_len % scale_factor == 0
