"""
Nash equilibrium solver via best-response dynamics.

Implements game-theoretic aggregation where each scale is a player
competing to minimize reconstruction error to the consensus.

Reference: Paper 2512.14925v2 (MAHA), Section 4.3, Equation 10
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array


@partial(jax.jit, static_argnames=["num_iterations"])
def nash_best_response(
    scale_outputs: Float[Array, "B L N D"],
    num_iterations: int = 3,
) -> tuple[Float[Array, "B N D"], Float[Array, "B L"]]:
    """
    Nash equilibrium aggregation via iterated best-response dynamics.

    Models each scale as a "player" in a non-cooperative game where:
    - Each player's strategy is their weight w_l
    - Each player minimizes reconstruction error ||O_l - O*||²
    - O* is the consensus (weighted sum of all scale outputs)

    The algorithm iterates:
    1. Compute consensus: O* = sum(w_l * O_l)
    2. Compute per-scale error: error_l = ||O_l - O*||_2
    3. Update weights: w_l = softmax(-error_l)

    At equilibrium, scales with lower reconstruction error receive higher weights.

    Args:
        scale_outputs: Stacked outputs from all scales [B, L, N, D].
                      All scales must already be upsampled to the same length N.
        num_iterations: Number of best-response iterations (typically 3-5).

    Returns:
        Tuple of:
        - aggregated: Final consensus output [B, N, D]
        - weights: Equilibrium weights [B, L]

    Invariants (post-condition):
        - sum(weights, axis=-1) = 1 (simplex constraint)
        - weights >= 0 (non-negative)
    """
    B, L, N, D = scale_outputs.shape

    # Initialize uniform weights: each scale starts with equal weight
    init_weights = jnp.ones((B, L), dtype=scale_outputs.dtype) / L

    def iteration_body(
        i: int,
        weights: Float[Array, "B L"],
    ) -> Float[Array, "B L"]:
        w_expanded = weights[:, :, None, None]
        consensus = jnp.sum(scale_outputs * w_expanded, axis=1)
        diffs = scale_outputs - consensus[:, None, :, :]

        # L2 norm with epsilon to prevent sqrt(0) gradient explosion
        errors = jnp.sqrt(jnp.sum(diffs**2, axis=(-2, -1)) + 1e-8)

        new_weights = jax.nn.softmax(-errors, axis=-1)

        # Stop gradient through iterates - gradients flow only through final aggregation
        return lax.stop_gradient(new_weights)

    # Run fixed number of iterations using fori_loop (JIT-friendly)
    final_weights = lax.fori_loop(
        0,
        num_iterations,
        iteration_body,
        init_weights,
    )

    # Compute final aggregated output using equilibrium weights
    w_expanded = final_weights[:, :, None, None]
    aggregated = jnp.sum(scale_outputs * w_expanded, axis=1)

    return aggregated, final_weights


@partial(jax.jit, static_argnames=["num_iterations"])
def nash_best_response_with_temperature(
    scale_outputs: Float[Array, "B L N D"],
    num_iterations: int = 3,
    temperature: float = 1.0,
) -> tuple[Float[Array, "B N D"], Float[Array, "B L"]]:
    """
    Nash best-response with temperature-controlled weight sharpness.

    Lower temperature -> sharper weights (more confidence in best scale)
    Higher temperature -> softer weights (more uniform mixing)

    Args:
        scale_outputs: Stacked outputs from all scales [B, L, N, D].
        num_iterations: Number of best-response iterations.
        temperature: Softmax temperature (default 1.0).

    Returns:
        Tuple of (aggregated [B, N, D], weights [B, L]).
    """
    B, L, N, D = scale_outputs.shape
    init_weights = jnp.ones((B, L), dtype=scale_outputs.dtype) / L

    def iteration_body(i: int, weights: Float[Array, "B L"]) -> Float[Array, "B L"]:
        w_expanded = weights[:, :, None, None]
        consensus = jnp.sum(scale_outputs * w_expanded, axis=1)
        diffs = scale_outputs - consensus[:, None, :, :]
        errors = jnp.sqrt(jnp.sum(diffs**2, axis=(-2, -1)) + 1e-8)
        new_weights = jax.nn.softmax(-errors / temperature, axis=-1)
        return lax.stop_gradient(new_weights)

    final_weights = lax.fori_loop(0, num_iterations, iteration_body, init_weights)
    w_expanded = final_weights[:, :, None, None]
    aggregated = jnp.sum(scale_outputs * w_expanded, axis=1)

    return aggregated, final_weights


def convex_aggregation(
    scale_outputs: Float[Array, "B L N D"],
    logits: Float[Array, "L"],
) -> tuple[Float[Array, "B N D"], Float[Array, "L"]]:
    """
    Convex optimization aggregation with learned weights.

    Uses softmax to enforce simplex constraint: sum(w) = 1, w >= 0.

    Args:
        scale_outputs: Stacked outputs from all scales [B, L, N, D].
        logits: Learnable weight logits [L] (before softmax).

    Returns:
        Tuple of (aggregated [B, N, D], weights [L]).
    """
    # Enforce simplex constraint via softmax
    weights = jax.nn.softmax(logits)

    # Weighted sum: einsum is cleaner for this pattern
    aggregated = jnp.einsum("l,blnd->bnd", weights, scale_outputs)

    return aggregated, weights


def compute_sparsity_loss(
    weights: Float[Array, "... L"], lambda_sparsity: float
) -> Float[Array, ""]:
    """
    Compute L1 sparsity penalty for aggregation weights.

    Encourages some weights to go to zero, focusing on fewer scales.

    Args:
        weights: Simplex weights [..., L].
        lambda_sparsity: Regularization strength.

    Returns:
        Scalar sparsity loss.
    """
    return lambda_sparsity * jnp.sum(jnp.abs(weights))
