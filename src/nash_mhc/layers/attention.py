"""
Multiscale Attention layer with shared Value projection.

Implements the MAHA attention mechanism where each scale computes
attention independently with per-scale Q/K but shared V.

Reference: Paper 2512.14925v2 (MAHA), Section 4.2
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from nash_mhc.primitives.rope import apply_rope_per_scale, compute_rope_freqs
from nash_mhc.layers.decomposition import HierarchicalDecomposition


class MultiscaleAttention(eqx.Module):
    """
    MAHA multiscale attention with shared V projection.

    Key features:
    - Per-scale Q, K projections (scale-specific)
    - Shared V projection across all scales (parameter efficient)
    - Per-scale RoPE with adjusted position indices
    - Causal masking for autoregressive LM
    """

    q_projs: tuple[eqx.nn.Linear, ...]
    k_projs: tuple[eqx.nn.Linear, ...]
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_scales: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    compression_ratio: int = eqx.field(static=True)

    # RoPE cache
    rope_freqs: Float[Array, "M K2"]

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_scales: int,
        max_seq_len: int,
        compression_ratio: int = 2,
        *,
        key: jax.Array,
    ):
        """
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_scales: Number of hierarchical scales.
            max_seq_len: Maximum sequence length (for RoPE cache).
            compression_ratio: Downsampling ratio between scales.
            key: PRNG key for initialization.
        """
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_scales = num_scales
        self.compression_ratio = compression_ratio

        # Split keys for all components
        keys = jax.random.split(key, 2 * num_scales + 2)

        # Per-scale Q, K projections
        self.q_projs = tuple(
            eqx.nn.Linear(d_model, d_model, key=keys[i]) for i in range(num_scales)
        )
        self.k_projs = tuple(
            eqx.nn.Linear(d_model, d_model, key=keys[num_scales + i])
            for i in range(num_scales)
        )

        # Shared V and output projections
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=keys[-2])
        self.o_proj = eqx.nn.Linear(d_model, d_model, key=keys[-1])

        # Precompute RoPE frequencies
        self.rope_freqs = compute_rope_freqs(self.head_dim, max_seq_len)

    def _reshape_for_attention(
        self,
        x: Float[Array, "B N D"],
    ) -> Float[Array, "B N H K"]:
        """Reshape [B, N, D] -> [B, N, H, K] for attention."""
        B, N, D = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim)

    def _single_scale_attention(
        self,
        x: Float[Array, "B N D"],
        v: Float[Array, "B N D"],
        scale_idx: int,
        causal: bool = True,
    ) -> Float[Array, "B N D"]:
        """
        Compute attention for a single scale.

        Args:
            x: Input for this scale [B, N_l, D].
            v: Value tensor for this scale [B, N_l, D].
            scale_idx: Hierarchical scale index.
            causal: Whether to apply causal masking.

        Returns:
            Attention output [B, N_l, D].
        """
        B, N, D = x.shape

        q = jax.vmap(jax.vmap(self.q_projs[scale_idx]))(x)
        k = jax.vmap(jax.vmap(self.k_projs[scale_idx]))(x)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)  # [B, N, H, K]
        k = self._reshape_for_attention(k)
        v_heads = self._reshape_for_attention(v)

        # Apply per-scale RoPE
        q = apply_rope_per_scale(q, self.rope_freqs, scale_idx, self.compression_ratio)
        k = apply_rope_per_scale(k, self.rope_freqs, scale_idx, self.compression_ratio)

        # Transpose for attention: [B, N, H, K] -> [B, H, N, K]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v_heads = jnp.transpose(v_heads, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(jnp.array(self.head_dim, dtype=jnp.float32))
        scores = jnp.einsum("bhqk,bhmk->bhqm", q, k) * scale

        if causal:
            mask = jnp.triu(jnp.ones((N, N), dtype=jnp.bool_), k=1)
            scores = jnp.where(mask, jnp.finfo(scores.dtype).min, scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqm,bhmk->bhqk", attn_weights, v_heads)

        # Reshape back: [B, H, N, K] -> [B, N, D]
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(B, N, D)

        return context

    def __call__(
        self,
        scale_inputs: tuple[Float[Array, "B N D"], ...],
        decomposition: HierarchicalDecomposition,
        causal: bool = True,
    ) -> tuple[Float[Array, "B N D"], ...]:
        """
        Compute attention for all scales.

        V is computed once from scale 0, then decomposed to match each scale.

        Args:
            scale_inputs: Tuple of inputs for each scale.
            decomposition: Decomposition module for V projection.
            causal: Whether to apply causal masking.

        Returns:
            Tuple of attention outputs for each scale.
        """
        v_base = jax.vmap(jax.vmap(self.v_proj))(scale_inputs[0])
        v_scales = decomposition.decompose_values(v_base)

        outputs = []
        for i, (x_l, v_l) in enumerate(zip(scale_inputs, v_scales)):
            attn_out = self._single_scale_attention(x_l, v_l, i, causal)
            out_l = jax.vmap(jax.vmap(self.o_proj))(attn_out)
            outputs.append(out_l)

        return tuple(outputs)


class ScaledDotProductAttention(eqx.Module):
    """
    Standard scaled dot-product attention (single scale).

    For use in non-MAHA contexts or as a baseline comparison.
    """

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def __call__(
        self,
        q: Float[Array, "B N H K"],
        k: Float[Array, "B N H K"],
        v: Float[Array, "B N H K"],
        causal: bool = True,
    ) -> Float[Array, "B N H K"]:
        """Standard scaled dot-product attention."""
        B, N, H, K = q.shape

        # Transpose: [B, N, H, K] -> [B, H, N, K]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(K)
        scores = jnp.einsum("bhqk,bhmk->bhqm", q, k) * scale

        if causal:
            mask = jnp.triu(jnp.ones((N, N), dtype=jnp.bool_), k=1)
            scores = jnp.where(mask, -1e9, scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqm,bhmk->bhqk", attn_weights, v)

        # Transpose back: [B, H, N, K] -> [B, N, H, K]
        return jnp.transpose(context, (0, 2, 1, 3))
