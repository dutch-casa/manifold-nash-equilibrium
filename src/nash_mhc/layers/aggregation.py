"""
Optimization-Driven Aggregation layer.

Aggregates outputs from multiple hierarchical scales using either
Nash equilibrium or convex optimization strategies.

Reference: Paper 2512.14925v2 (MAHA), Section 4.3
"""

from typing import Literal, cast

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from nash_mhc.primitives.nash_solver import nash_best_response, compute_sparsity_loss
from nash_mhc.primitives.upsample import nearest_upsample


class OptimizationAggregation(eqx.Module):
    """
    Optimization-driven aggregation of multiscale outputs.

    Strategies:
    - 'nash': Game-theoretic equilibrium via best-response dynamics
    - 'convex': Learned simplex weights with L1 sparsity regularization

    Nash equilibrium models each scale as a "player" competing to minimize
    reconstruction error to the consensus. Convex uses learned weights
    with sparsity penalty to focus on informative scales.
    """

    strategy: Literal["nash", "convex"] = eqx.field(static=True)
    num_scales: int = eqx.field(static=True)
    nash_iterations: int = eqx.field(static=True)
    lambda_sparsity: float = eqx.field(static=True)

    # Convex strategy parameters (only used when strategy="convex")
    convex_logits: Float[Array, "L"] | None

    def __init__(
        self,
        num_scales: int,
        strategy: Literal["nash", "convex"] = "nash",
        nash_iterations: int = 3,
        lambda_sparsity: float = 0.1,
        *,
        key: jax.Array,
    ):
        """
        Args:
            num_scales: Number of hierarchical scales (L).
            strategy: Aggregation strategy ('nash' or 'convex').
            nash_iterations: Number of best-response iterations for Nash.
            lambda_sparsity: L1 regularization strength for convex.
            key: PRNG key (unused for Nash, used for convex initialization).
        """
        self.strategy = strategy
        self.num_scales = num_scales
        self.nash_iterations = nash_iterations
        self.lambda_sparsity = lambda_sparsity

        if strategy == "convex":
            # Initialize uniform (zeros -> uniform after softmax)
            self.convex_logits = jnp.zeros(num_scales)
        else:
            self.convex_logits = None

    def __call__(
        self,
        scale_outputs: tuple[Float[Array, "B N_l D"], ...],
    ) -> tuple[Float[Array, "B N D"], Float[Array, ""]]:
        """
        Aggregate scale outputs to original sequence length.

        Args:
            scale_outputs: Tuple of tensors with decreasing sequence lengths.
                          scale_outputs[0] has the longest sequence.

        Returns:
            Tuple of:
            - aggregated: Combined output [B, N, D]
            - aux_loss: Auxiliary loss (sparsity for convex, 0 for Nash)
        """
        target_len = scale_outputs[0].shape[1]

        # Upsample all scales to target length
        upsampled = [scale_outputs[0]]  # First scale already at target length
        for out in scale_outputs[1:]:
            upsampled.append(nearest_upsample(out, target_len))

        # Stack for aggregation: list of [B, N, D] -> [B, L, N, D]
        stacked = jnp.stack(upsampled, axis=1)

        if self.strategy == "nash":
            result = cast(
                tuple[Float[Array, "B N D"], Float[Array, "B L"]],
                nash_best_response(stacked, self.nash_iterations),
            )
            aggregated = result[0]
            weights: Float[Array, "B L"] | Float[Array, "L"] = result[1]
            aux_loss = jnp.array(0.0)

        else:
            assert self.convex_logits is not None
            weights = jax.nn.softmax(self.convex_logits)
            aggregated = jnp.einsum("l,blnd->bnd", weights, stacked)
            aux_loss = compute_sparsity_loss(weights, self.lambda_sparsity)

        return aggregated, aux_loss

    def get_weights(
        self,
        scale_outputs: tuple[Float[Array, "B N_l D"], ...] | None = None,
    ) -> Float[Array, "... L"]:
        """
        Get current aggregation weights.

        For Nash, requires scale_outputs to compute equilibrium.
        For convex, returns learned weights.

        Args:
            scale_outputs: Required for Nash strategy.

        Returns:
            Aggregation weights.
        """
        if self.strategy == "convex":
            return jax.nn.softmax(self.convex_logits)

        if scale_outputs is None:
            raise ValueError("scale_outputs required for Nash strategy")

        target_len = scale_outputs[0].shape[1]
        upsampled = [scale_outputs[0]]
        for out in scale_outputs[1:]:
            upsampled.append(nearest_upsample(out, target_len))
        stacked = jnp.stack(upsampled, axis=1)

        _, weights = nash_best_response(stacked, self.nash_iterations)
        return weights


class AdaptiveAggregation(eqx.Module):
    """
    Content-adaptive aggregation with learned gating.

    Uses a learned projection to compute per-position, per-scale gates,
    allowing the model to dynamically weight scales based on content.
    """

    gate_proj: eqx.nn.Linear
    num_scales: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        num_scales: int,
        *,
        key: jax.Array,
    ):
        self.d_model = d_model
        self.num_scales = num_scales

        # Project from concatenated scales to gate logits
        # Input: [B, N, L * D] -> Output: [B, N, L]
        self.gate_proj = eqx.nn.Linear(d_model, num_scales, key=key)

    def __call__(
        self,
        scale_outputs: tuple[Float[Array, "B N D"], ...],
    ) -> tuple[Float[Array, "B N D"], Float[Array, ""]]:
        """
        Aggregate with content-adaptive gating.

        Uses scale 0 features to compute position-dependent gates.
        """
        target_len = scale_outputs[0].shape[1]

        # Upsample and stack
        upsampled = [scale_outputs[0]]
        for out in scale_outputs[1:]:
            upsampled.append(nearest_upsample(out, target_len))
        stacked = jnp.stack(upsampled, axis=1)  # [B, L, N, D]

        # Compute gates from scale 0 features
        gate_logits = jax.vmap(jax.vmap(self.gate_proj))(scale_outputs[0])  # [B, N, L]
        gates = jax.nn.softmax(gate_logits, axis=-1)

        # Apply gates: [B, L, N, D] * [B, N, L] (broadcast) -> [B, N, D]
        gates_expanded = gates[:, :, :, None]  # [B, N, L, 1]
        stacked_transposed = jnp.transpose(stacked, (0, 2, 1, 3))  # [B, N, L, D]
        aggregated = jnp.sum(stacked_transposed * gates_expanded, axis=2)  # [B, N, D]

        return aggregated, jnp.array(0.0)
