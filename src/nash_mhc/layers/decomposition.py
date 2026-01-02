"""
Hierarchical Multiscale Decomposition layer.

Decomposes input sequences into multiple hierarchical scales using
learnable strided convolutions.

Reference: Paper 2512.14925v2 (MAHA), Section 4.1
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from nash_mhc.primitives.strided_conv import StridedConv1d


class HierarchicalDecomposition(eqx.Module):
    """
    Hierarchical multiscale decomposition via strided convolutions.

    Transforms input X into L hierarchical scales:
        X -> [X_0, X_1, ..., X_{L-1}]

    where:
        X_0 = X (original input)
        X_l = D_l(X_{l-1}) = Conv1D(X_{l-1}, stride=r)
        len(X_l) = len(X_{l-1}) // r

    Key design decisions:
    - Returns tuple of arrays (pytree-friendly, not Python list)
    - All scales share the same feature dimension D
    - Uses learnable strided convolutions for downsampling
    """

    downsamplers: tuple[StridedConv1d, ...]
    num_scales: int = eqx.field(static=True)
    compression_ratio: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        num_scales: int,
        compression_ratio: int = 2,
        kernel_size: int = 3,
        *,
        key: jax.Array,
    ):
        """
        Args:
            d_model: Model dimension (preserved across scales).
            num_scales: Number of hierarchical scales (L).
            compression_ratio: Downsampling stride (r).
            kernel_size: Convolution kernel size.
            key: PRNG key for initialization.
        """
        self.d_model = d_model
        self.num_scales = num_scales
        self.compression_ratio = compression_ratio

        # Create (L-1) downsamplers (scale 0 is identity)
        keys = jax.random.split(key, num_scales - 1)

        self.downsamplers = tuple(
            StridedConv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=compression_ratio,
                key=keys[i],
            )
            for i in range(num_scales - 1)
        )

    def __call__(
        self,
        x: Float[Array, "B N D"],
    ) -> tuple[Float[Array, "B N D"], ...]:
        """
        Decompose input into hierarchical scales.

        Args:
            x: Input tensor [B, N, D].

        Returns:
            Tuple of scale representations:
                (
                    [B, N, D],          # Scale 0 (original)
                    [B, N/r, D],        # Scale 1
                    [B, N/r², D],       # Scale 2
                    ...
                )
        """
        outputs = [x]  # Scale 0 is the original input
        current = x

        for downsampler in self.downsamplers:
            current = downsampler(current)
            outputs.append(current)

        return tuple(outputs)

    def get_scale_lengths(self, input_len: int) -> tuple[int, ...]:
        """
        Compute expected sequence lengths per scale.

        Args:
            input_len: Input sequence length (N).

        Returns:
            Tuple of lengths for each scale.
        """
        lengths = [input_len]
        curr = input_len

        for _ in range(self.num_scales - 1):
            # Account for padding in strided conv
            # With kernel_size=3, padding=1, stride=r:
            # out_len = (in_len + 2*1 - 3) // r + 1 = (in_len - 1) // r + 1
            # For simplicity, we use floor division
            curr = curr // self.compression_ratio
            lengths.append(curr)

        return tuple(lengths)

    def decompose_values(
        self,
        v: Float[Array, "B N D"],
    ) -> tuple[Float[Array, "B N D"], ...]:
        """
        Decompose value tensor for shared V projection pattern.

        This applies the same decomposition to a pre-projected value tensor,
        enabling the shared V projection pattern in MAHA attention.

        Args:
            v: Value tensor [B, N, D] (already projected by shared W_V).

        Returns:
            Tuple of decomposed values for each scale.
        """
        return self(v)


class HierarchicalDecompositionPool(eqx.Module):
    """
    Non-learnable hierarchical decomposition via adaptive max pooling.

    Alternative to strided convolution when learnable downsampling
    is not desired.
    """

    num_scales: int = eqx.field(static=True)
    compression_ratio: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        num_scales: int,
        compression_ratio: int = 2,
    ):
        self.d_model = d_model
        self.num_scales = num_scales
        self.compression_ratio = compression_ratio

    def __call__(
        self,
        x: Float[Array, "B N D"],
    ) -> tuple[Float[Array, "B N D"], ...]:
        """Decompose using max pooling (non-learnable)."""
        outputs = [x]
        current_len = x.shape[1]

        for _ in range(self.num_scales - 1):
            target_len = max(1, current_len // self.compression_ratio)

            # Manual max pooling via reshape and reduce
            # Reshape to [B, target_len, pool_size, D] and take max
            pool_size = current_len // target_len
            padded_len = target_len * pool_size

            # Truncate to divisible length
            x_truncated = x[:, :padded_len, :]

            # Reshape and pool
            x_reshaped = x_truncated.reshape(
                x.shape[0], target_len, pool_size, self.d_model
            )
            x_pooled = jnp.max(x_reshaped, axis=2)

            outputs.append(x_pooled)
            x = x_pooled
            current_len = target_len

        return tuple(outputs)
