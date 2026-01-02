"""
TPU-aligned strided 1D convolution for hierarchical downsampling.

Implements learnable downsampling via strided convolution, optimized
for TPU's 128x128 systolic array.

Reference: Paper 2512.14925v2 (MAHA), Section 4.1, Equations 5, 109
"""

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
from jaxtyping import Float, Array


class StridedConv1d(eqx.Module):
    """
    1D strided convolution for hierarchical downsampling.

    Implements D_l(X) = Conv1D(X, W_l, stride=r) where r is the compression ratio.

    Key differences from PyTorch:
    - Explicit padding calculation for TPU alignment
    - Uses lax.conv_general_dilated for XLA optimization
    - bf16 weights with f32 accumulation on TPU
    """

    weight: Float[Array, "out_c in_c k"]
    bias: Float[Array, "out_c"] | None
    stride: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        *,
        key: jax.Array,
        use_bias: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (d_model).
            out_channels: Number of output channels (usually same as in_channels).
            kernel_size: Convolution kernel size (default 3).
            stride: Downsampling stride (compression_ratio, default 2).
            key: PRNG key for initialization.
            use_bias: Whether to include bias term.
        """
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Same padding for alignment

        # Initialize with truncated normal (similar to PyTorch default)
        wkey, bkey = jax.random.split(key)
        fan_in = in_channels * kernel_size
        scale = 1.0 / jnp.sqrt(fan_in)

        self.weight = jax.random.truncated_normal(
            wkey,
            lower=-2.0,
            upper=2.0,
            shape=(out_channels, in_channels, kernel_size),
        ) * scale

        self.bias = jnp.zeros(out_channels) if use_bias else None

    def __call__(self, x: Float[Array, "B N D"]) -> Float[Array, "B N2 D"]:
        """
        Apply strided convolution for downsampling.

        Args:
            x: Input tensor [B, N, D] where D = in_channels.

        Returns:
            Downsampled tensor [B, N // stride, D] where D = out_channels.
        """
        # Transpose for conv: [B, N, D] -> [B, D, N]
        x = jnp.swapaxes(x, -1, -2)

        # Apply conv1d via lax.conv_general_dilated
        # dimension_numbers: (batch, channel, spatial) for input, kernel, output
        y = lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            dimension_numbers=("NCH", "OIH", "NCH"),
            precision=lax.Precision.DEFAULT,  # TPU uses bf16->f32 automatically
        )

        if self.bias is not None:
            y = y + self.bias[:, None]

        # Transpose back: [B, D, N] -> [B, N, D]
        return jnp.swapaxes(y, -1, -2)


class StridedConv1dTPUAligned(eqx.Module):
    """
    TPU-optimized strided convolution with 128-aligned dimensions.

    Pads channel dimensions to multiples of 128 for optimal MXU utilization.
    """

    inner_conv: StridedConv1d
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    padded_in: int = eqx.field(static=True)
    padded_out: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        *,
        key: jax.Array,
        use_bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pad to 128 alignment
        self.padded_in = ((in_channels + 127) // 128) * 128
        self.padded_out = ((out_channels + 127) // 128) * 128

        self.inner_conv = StridedConv1d(
            in_channels=self.padded_in,
            out_channels=self.padded_out,
            kernel_size=kernel_size,
            stride=stride,
            key=key,
            use_bias=use_bias,
        )

    def __call__(self, x: Float[Array, "B N D"]) -> Float[Array, "B N2 D"]:
        """Apply with padding/unpadding for TPU alignment."""
        # Pad input channels
        if self.padded_in > self.in_channels:
            pad_width = self.padded_in - self.in_channels
            x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_width)))

        # Apply convolution
        y = self.inner_conv(x)

        # Unpad output channels
        if self.padded_out > self.out_channels:
            y = y[..., : self.out_channels]

        return y


def create_downsampler_stack(
    d_model: int,
    num_scales: int,
    compression_ratio: int = 2,
    kernel_size: int = 3,
    *,
    key: jax.Array,
) -> tuple[StridedConv1d, ...]:
    """
    Create a stack of strided convolutions for hierarchical decomposition.

    Returns (num_scales - 1) convolutions since scale 0 is the original input.

    Args:
        d_model: Model dimension (in and out channels).
        num_scales: Number of hierarchical scales.
        compression_ratio: Downsampling stride.
        kernel_size: Convolution kernel size.
        key: PRNG key.

    Returns:
        Tuple of StridedConv1d modules.
    """
    keys = jax.random.split(key, num_scales - 1)

    return tuple(
        StridedConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=compression_ratio,
            key=keys[i],
        )
        for i in range(num_scales - 1)
    )
