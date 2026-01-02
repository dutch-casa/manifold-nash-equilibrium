"""
Sinkhorn-Knopp algorithm for Birkhoff polytope projection.

Projects arbitrary matrices onto the set of doubly stochastic matrices
using iterative row/column normalization in log-space.

Reference: Paper 2512.24880v1 (mHC)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array


@partial(jax.jit, static_argnames=["num_iterations"])
def sinkhorn_knopp(
    log_alpha: Float[Array, "... n n"],
    num_iterations: int = 10,
) -> Float[Array, "... n n"]:
    """
    Sinkhorn-Knopp algorithm for Birkhoff polytope projection.

    Computes a doubly stochastic matrix via iterative row/column normalization
    in log-space for numerical stability.

    The algorithm alternates between:
    1. Row normalization: u <- -logsumexp(log_alpha + v, axis=-1)
    2. Column normalization: v <- -logsumexp(log_alpha + u, axis=-2)

    After convergence, P = exp(log_alpha + u[:, None] + v[None, :]) is doubly stochastic.

    Args:
        log_alpha: Log-space input matrix [..., n, n]. Can be any real-valued matrix.
        num_iterations: Number of Sinkhorn iterations. 10-20 typically sufficient.

    Returns:
        Doubly stochastic matrix [..., n, n] on the Birkhoff polytope.

    Invariants (post-condition):
        - P @ 1 = 1 (rows sum to 1)
        - P.T @ 1 = 1 (columns sum to 1)
        - P >= 0 (non-negative)

    Note:
        - Uses float32 for logsumexp operations to prevent manifold drift
        - Returns in the same dtype as input
    """
    original_dtype = log_alpha.dtype
    # Promote to f32 for stability in logsumexp
    log_alpha_f32 = log_alpha.astype(jnp.float32)

    # Initialize dual variables
    n = log_alpha.shape[-1]
    u = jnp.zeros(log_alpha.shape[:-1], dtype=jnp.float32)
    v = jnp.zeros(log_alpha.shape[:-1], dtype=jnp.float32)

    def scan_body(
        carry: tuple[Float[Array, "..."], Float[Array, "..."]],
        _: None,
    ) -> tuple[tuple[Float[Array, "..."], Float[Array, "..."]], None]:
        u_prev, v_prev = carry

        # Row normalization: make rows sum to 1
        # u_new[i] = -log(sum_j exp(log_alpha[i,j] + v[j]))
        u_new = -jax.nn.logsumexp(log_alpha_f32 + v_prev[..., None, :], axis=-1)

        # Column normalization: make columns sum to 1
        # v_new[j] = -log(sum_i exp(log_alpha[i,j] + u[i]))
        v_new = -jax.nn.logsumexp(log_alpha_f32 + u_new[..., :, None], axis=-2)

        return (u_new, v_new), None

    (u_final, v_final), _ = lax.scan(
        scan_body,
        (u, v),
        xs=None,
        length=num_iterations,
    )

    # Compute final doubly stochastic matrix
    log_P = log_alpha_f32 + u_final[..., :, None] + v_final[..., None, :]
    P = jnp.exp(log_P)

    return P.astype(original_dtype)


@partial(jax.jit, static_argnames=["num_iterations", "epsilon"])
def sinkhorn_knopp_regularized(
    cost: Float[Array, "... n n"],
    num_iterations: int = 10,
    epsilon: float = 0.1,
) -> Float[Array, "... n n"]:
    """
    Entropy-regularized Sinkhorn for optimal transport.

    Solves the regularized OT problem:
        min_P <C, P> - epsilon * H(P)
        s.t. P @ 1 = a, P.T @ 1 = b

    With uniform marginals a = b = 1/n.

    Args:
        cost: Cost matrix [..., n, n].
        num_iterations: Number of Sinkhorn iterations.
        epsilon: Entropy regularization strength. Smaller = sharper transport.

    Returns:
        Transport plan [..., n, n] (doubly stochastic with uniform marginals).
    """
    log_alpha = -cost / epsilon
    return sinkhorn_knopp(log_alpha, num_iterations)


def sinkhorn_knopp_simple(
    M: Float[Array, "n n"],
    num_iterations: int = 10,
) -> Float[Array, "n n"]:
    """
    Simple Sinkhorn-Knopp without log-space (for small matrices).

    Directly alternates between row and column normalization.
    Less numerically stable but faster for small matrices.

    Args:
        M: Non-negative input matrix [n, n].
        num_iterations: Number of iterations.

    Returns:
        Doubly stochastic matrix [n, n].
    """
    P = jnp.abs(M) + 1e-10  # Ensure positive

    def body(_, P: Float[Array, "n n"]) -> Float[Array, "n n"]:
        # Row normalize
        P = P / jnp.sum(P, axis=-1, keepdims=True)
        # Column normalize
        P = P / jnp.sum(P, axis=-2, keepdims=True)
        return P

    return lax.fori_loop(0, num_iterations, body, P)
