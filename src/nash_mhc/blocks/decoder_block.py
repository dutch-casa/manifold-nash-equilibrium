"""
MAHA Decoder Block with mHC residual connections.

Combines hierarchical multiscale attention with manifold-constrained
residual connections for stable deep training.

Reference: Papers 2512.14925v2 (MAHA) and 2512.24880v1 (mHC)
"""

from typing import Literal

import jax
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

from nash_mhc.layers.mhc import ManifoldHyperConnection
from nash_mhc.layers.decomposition import HierarchicalDecomposition
from nash_mhc.layers.attention import MultiscaleAttention
from nash_mhc.layers.aggregation import OptimizationAggregation
from nash_mhc.layers.ffn import SwiGLUFFN, RMSNorm


class MAHADecoderBlock(eqx.Module):
    """
    Full MAHA decoder block with mHC residual connections.

    Architecture:
    1. Pre-norm (RMSNorm)
    2. Hierarchical decomposition: X -> [X_0, X_1, ..., X_{L-1}]
    3. Multiscale attention with shared V projection
    4. Optimization-driven aggregation (Nash or Convex)
    5. mHC residual connection (attention branch)
    6. Pre-norm (RMSNorm)
    7. SwiGLU FFN
    8. mHC residual connection (FFN branch)

    Type Parameters:
        B: Batch size
        N: Sequence length
        D: Model dimension
    """

    # Normalization layers
    norm_attn: RMSNorm
    norm_ffn: RMSNorm

    # MAHA components
    decomposition: HierarchicalDecomposition
    attention: MultiscaleAttention
    aggregation: OptimizationAggregation

    # FFN
    ffn: SwiGLUFFN

    # mHC residual connections
    mhc_attn: ManifoldHyperConnection
    mhc_ffn: ManifoldHyperConnection

    # Static configuration
    d_model: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_scales: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_scales: int,
        max_seq_len: int,
        compression_ratio: int = 2,
        ffn_multiplier: float = 2.67,
        aggregation: Literal["nash", "convex"] = "nash",
        nash_iterations: int = 3,
        sinkhorn_iterations: int = 10,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            d_model: Model dimension (must be multiple of 128).
            num_heads: Number of attention heads.
            num_scales: Number of hierarchical scales (L).
            max_seq_len: Maximum sequence length.
            compression_ratio: Downsampling ratio between scales.
            ffn_multiplier: FFN hidden dimension multiplier.
            aggregation: 'nash' or 'convex' aggregation strategy.
            nash_iterations: Best-response iterations for Nash.
            sinkhorn_iterations: Sinkhorn-Knopp iterations for mHC.
            key: PRNG key for initialization.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_scales = num_scales

        # Split key for all components
        keys = jax.random.split(key, 6)

        # Normalization layers
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)

        # MAHA components
        self.decomposition = HierarchicalDecomposition(
            d_model=d_model,
            num_scales=num_scales,
            compression_ratio=compression_ratio,
            key=keys[0],
        )

        self.attention = MultiscaleAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_scales=num_scales,
            max_seq_len=max_seq_len,
            compression_ratio=compression_ratio,
            key=keys[1],
        )

        self.aggregation = OptimizationAggregation(
            num_scales=num_scales,
            strategy=aggregation,
            nash_iterations=nash_iterations,
            key=keys[2],
        )

        # FFN
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            ffn_multiplier=ffn_multiplier,
            key=keys[3],
        )

        # mHC residual connections (both branches)
        self.mhc_attn = ManifoldHyperConnection(
            d_model=d_model,
            sinkhorn_iters=sinkhorn_iterations,
            key=keys[4],
        )

        self.mhc_ffn = ManifoldHyperConnection(
            d_model=d_model,
            sinkhorn_iters=sinkhorn_iterations,
            key=keys[5],
        )

    def __call__(
        self,
        x: Float[Array, "B N D"],
        causal: bool = True,
    ) -> tuple[Float[Array, "B N D"], Float[Array, ""]]:
        """
        Forward pass through the MAHA decoder block.

        Args:
            x: Input tensor [B, N, D].
            causal: Whether to apply causal masking.

        Returns:
            Tuple of:
            - output: Block output [B, N, D]
            - aux_loss: Auxiliary loss from aggregation
        """
        # === Attention Branch ===
        residual = x

        # Pre-norm (vmap over batch and sequence)
        x_normed = jax.vmap(jax.vmap(self.norm_attn))(x)

        # Hierarchical decomposition
        scales = self.decomposition(x_normed)

        # Multiscale attention
        attn_scales = self.attention(scales, self.decomposition, causal)

        # Aggregation
        attn_out, aux_loss = self.aggregation(attn_scales)

        # mHC residual connection (manifold-constrained)
        x = self.mhc_attn(residual, attn_out)

        # === FFN Branch ===
        residual = x

        # Pre-norm
        x_normed = jax.vmap(jax.vmap(self.norm_ffn))(x)

        # SwiGLU FFN (vmap over batch and sequence)
        ffn_out = jax.vmap(jax.vmap(self.ffn))(x_normed)

        # mHC residual connection
        x = self.mhc_ffn(residual, ffn_out)

        return x, aux_loss


class MAHADecoderBlockLite(eqx.Module):
    """
    Lightweight MAHA block with standard residual connections.

    Uses standard residual addition instead of mHC for faster inference.
    For ablation studies and resource-constrained settings.
    """

    norm_attn: RMSNorm
    norm_ffn: RMSNorm
    decomposition: HierarchicalDecomposition
    attention: MultiscaleAttention
    aggregation: OptimizationAggregation
    ffn: SwiGLUFFN

    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_scales: int,
        max_seq_len: int,
        compression_ratio: int = 2,
        ffn_multiplier: float = 2.67,
        aggregation: Literal["nash", "convex"] = "nash",
        nash_iterations: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        self.d_model = d_model

        keys = jax.random.split(key, 4)

        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)

        self.decomposition = HierarchicalDecomposition(
            d_model=d_model,
            num_scales=num_scales,
            compression_ratio=compression_ratio,
            key=keys[0],
        )

        self.attention = MultiscaleAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_scales=num_scales,
            max_seq_len=max_seq_len,
            compression_ratio=compression_ratio,
            key=keys[1],
        )

        self.aggregation = OptimizationAggregation(
            num_scales=num_scales,
            strategy=aggregation,
            nash_iterations=nash_iterations,
            key=keys[2],
        )

        self.ffn = SwiGLUFFN(
            d_model=d_model,
            ffn_multiplier=ffn_multiplier,
            key=keys[3],
        )

    def __call__(
        self,
        x: Float[Array, "B N D"],
        causal: bool = True,
    ) -> tuple[Float[Array, "B N D"], Float[Array, ""]]:
        """Forward with standard residual connections."""
        # Attention
        residual = x
        x_normed = jax.vmap(jax.vmap(self.norm_attn))(x)
        scales = self.decomposition(x_normed)
        attn_scales = self.attention(scales, self.decomposition, causal)
        attn_out, aux_loss = self.aggregation(attn_scales)
        x = residual + attn_out  # Standard residual

        # FFN
        residual = x
        x_normed = jax.vmap(jax.vmap(self.norm_ffn))(x)
        ffn_out = jax.vmap(jax.vmap(self.ffn))(x_normed)
        x = residual + ffn_out  # Standard residual

        return x, aux_loss
