"""
Nearest-neighbor upsampling primitive.

Implements manual upsampling via index gathering since JAX lacks F.interpolate.
Optimized for TPU by avoiding dynamic control flow.

Reference: Paper 2512.14925v2 (MAHA), Equation 13
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


from functools import partial


@partial(jax.jit, static_argnames=["target_len"])
def nearest_upsample(
    x: Float[Array, "B N D"],
    target_len: int,
) -> Float[Array, "B target_len D"]:
    """
    Nearest-neighbor upsampling without interpolation.

    Each target position is filled with the value from the nearest source position.
    Uses index gathering for TPU efficiency (no dynamic control flow).

    Args:
        x: Input tensor [B, N, D] where N < target_len.
        target_len: Desired output sequence length.

    Returns:
        Upsampled tensor [B, target_len, D].

    Example:
        x of length 4: [a, b, c, d]
        target_len 8:  [a, a, b, b, c, c, d, d]
    """
    B, N, D = x.shape

    # Compute source index for each target position
    # target_idx * (N / target_len) gives the fractional source position
    # floor() gives the nearest source index (nearest-neighbor behavior)
    scale = N / target_len
    target_indices = jnp.arange(target_len)
    source_indices = jnp.floor(target_indices * scale).astype(jnp.int32)

    # Clip to valid range (handles edge cases)
    source_indices = jnp.clip(source_indices, 0, N - 1)

    # Gather along sequence dimension
    # x[:, source_indices, :] broadcasts over batch
    return x[:, source_indices, :]


@partial(jax.jit, static_argnames=["target_len"])
def nearest_downsample(
    x: Float[Array, "B N D"],
    target_len: int,
) -> Float[Array, "B target_len D"]:
    """
    Nearest-neighbor downsampling (subsampling).

    Selects evenly-spaced positions from the source.

    Args:
        x: Input tensor [B, N, D] where N > target_len.
        target_len: Desired output sequence length.

    Returns:
        Downsampled tensor [B, target_len, D].
    """
    B, N, D = x.shape

    # Compute which source positions to sample
    scale = N / target_len
    target_indices = jnp.arange(target_len)
    source_indices = jnp.floor(target_indices * scale).astype(jnp.int32)
    source_indices = jnp.clip(source_indices, 0, N - 1)

    return x[:, source_indices, :]


def upsample_scale_outputs(
    scale_outputs: tuple[Float[Array, "B N_l D"], ...],
) -> Float[Array, "B L N D"]:
    """
    Upsample all scale outputs to match the first scale's length, then stack.

    Args:
        scale_outputs: Tuple of tensors with decreasing sequence lengths.
                      scale_outputs[0] has the longest sequence.

    Returns:
        Stacked tensor [B, L, N, D] where N = scale_outputs[0].shape[1].
    """
    target_len = scale_outputs[0].shape[1]
    L = len(scale_outputs)

    # Upsample each scale and collect
    upsampled = [scale_outputs[0]]  # First scale already at target length

    for i in range(1, L):
        upsampled.append(nearest_upsample(scale_outputs[i], target_len))

    # Stack along new axis: list of [B, N, D] -> [B, L, N, D]
    return jnp.stack(upsampled, axis=1)
