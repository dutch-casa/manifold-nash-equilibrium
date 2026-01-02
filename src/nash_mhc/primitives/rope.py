"""
Rotary Position Embedding (RoPE) with per-scale adjustment.

Implements RoPE where position indices are scaled based on the hierarchical
level, maintaining consistent positional relationships across scales.

Reference: RoFormer (Su et al., 2021)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
from functools import partial


def compute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "N D2"]:
    """
    Precompute rotary position embedding frequencies.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for geometric progression (default 10000).
        dtype: Output dtype.

    Returns:
        Frequency matrix [max_seq_len, dim // 2].
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    # Inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))

    # Position indices
    positions = jnp.arange(max_seq_len, dtype=dtype)

    # Outer product: [max_seq_len, dim/2]
    freqs = jnp.outer(positions, inv_freq)

    return freqs


@jax.jit
def apply_rope(
    x: Float[Array, "B N H K"],
    freqs: Float[Array, "N K2"],
) -> Float[Array, "B N H K"]:
    """
    Apply rotary position embedding to input tensor.

    The rotation is applied in pairs: (x[0], x[1]), (x[2], x[3]), ...
    Each pair is rotated by the corresponding frequency.

    Args:
        x: Input tensor [B, N, H, K] where K = head dimension.
        freqs: Precomputed frequencies [N, K//2].

    Returns:
        Rotated tensor [B, N, H, K].
    """
    # Split into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]  # [B, N, H, K/2] each

    # Compute sin and cos
    # freqs is [N, K/2], need to broadcast to [1, N, 1, K/2]
    cos = jnp.cos(freqs)[None, :, None, :]
    sin = jnp.sin(freqs)[None, :, None, :]

    # Apply rotation
    # (x1, x2) -> (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # Interleave back
    # Stack and reshape: [B, N, H, K/2, 2] -> [B, N, H, K]
    return jnp.stack([y1, y2], axis=-1).reshape(x.shape)


@partial(jax.jit, static_argnames=["scale_idx", "compression_ratio"])
def apply_rope_per_scale(
    x: Float[Array, "B N H K"],
    freqs: Float[Array, "M K2"],
    scale_idx: int,
    compression_ratio: int,
) -> Float[Array, "B N H K"]:
    """
    Apply RoPE with position indices adjusted for hierarchical scale.

    At scale l, position i corresponds to original position i * r^l.
    This maintains consistent positional relationships across scales:
    - Scale 0: positions [0, 1, 2, 3, ...]
    - Scale 1 (r=2): positions [0, 2, 4, 6, ...] (every other position)
    - Scale 2 (r=2): positions [0, 4, 8, 12, ...] (every 4th position)

    Args:
        x: Input tensor [B, N, H, K].
        freqs: Precomputed frequencies [M, K//2] where M >= N * r^scale_idx.
        scale_idx: Hierarchical scale index (0 = finest, L-1 = coarsest).
        compression_ratio: Downsampling ratio between scales.

    Returns:
        Rotated tensor [B, N, H, K].
    """
    seq_len = x.shape[1]

    # Compute adjusted position indices
    # At scale l, position i maps to original position i * r^l
    scale_factor = compression_ratio ** scale_idx
    adjusted_positions = jnp.arange(seq_len) * scale_factor

    # Clip to valid frequency range
    max_pos = freqs.shape[0] - 1
    adjusted_positions = jnp.clip(adjusted_positions, 0, max_pos).astype(jnp.int32)

    # Gather frequencies for adjusted positions
    scaled_freqs = freqs[adjusted_positions]  # [N, K/2]

    # Apply standard RoPE with scaled frequencies
    return apply_rope(x, scaled_freqs)


def create_rope_cache(
    max_seq_len: int,
    head_dim: int,
    num_scales: int,
    compression_ratio: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "M K2"]:
    """
    Create RoPE frequency cache for all scales.

    The cache must be large enough to cover the maximum position
    at the finest scale: max_seq_len * r^(num_scales - 1).

    Args:
        max_seq_len: Maximum sequence length at scale 0.
        head_dim: Dimension per attention head.
        num_scales: Number of hierarchical scales.
        compression_ratio: Downsampling ratio between scales.
        base: Base for geometric progression.
        dtype: Output dtype.

    Returns:
        Frequency cache [M, head_dim // 2] where M accounts for all scales.
    """
    # Maximum position needed (at coarsest scale, positions are multiplied by r^(L-1))
    # But wait - coarsest scale has FEWER positions, not more
    # Actually we need freqs for scale 0 positions which go up to max_seq_len
    # At coarser scales, we use subset of those frequencies
    return compute_rope_freqs(head_dim, max_seq_len, base, dtype)


class RoPECache:
    """
    Cached RoPE frequencies for efficient inference.

    Precomputes and stores cos/sin values for faster application.
    """

    def __init__(
        self,
        max_seq_len: int,
        head_dim: int,
        base: float = 10000.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.freqs = compute_rope_freqs(head_dim, max_seq_len, base, dtype)
        self._cos_cached = jnp.cos(self.freqs)
        self._sin_cached = jnp.sin(self.freqs)

    @property
    def cos(self) -> Float[Array, "N K2"]:
        return self._cos_cached

    @property
    def sin(self) -> Float[Array, "N K2"]:
        return self._sin_cached

    def apply(
        self,
        x: Float[Array, "B N H K"],
        start_pos: int = 0,
    ) -> Float[Array, "B N H K"]:
        """Apply cached RoPE starting from given position."""
        seq_len = x.shape[1]
        cos = self._cos_cached[start_pos : start_pos + seq_len]
        sin = self._sin_cached[start_pos : start_pos + seq_len]

        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return jnp.stack([y1, y2], axis=-1).reshape(x.shape)
