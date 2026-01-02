"""
Feed-Forward Network layers.

Implements SwiGLU FFN and RMSNorm for the transformer architecture.

Reference: GLU Variants Improve Transformer (Shazeer, 2020)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class RMSNorm(eqx.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes activations using RMS without centering (no mean subtraction).
    More efficient than LayerNorm and works well for transformers.

    Reference: Zhang & Sennrich (2019)
    """

    weight: Float[Array, "D"]
    eps: float = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Feature dimension.
            eps: Small constant for numerical stability.
        """
        self.d_model = d_model
        self.eps = eps
        self.weight = jnp.ones(d_model)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., D].

        Returns:
            Normalized tensor [..., D].
        """
        # Compute RMS
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale
        return (x / rms) * self.weight


class SwiGLUFFN(eqx.Module):
    """
    SwiGLU Feed-Forward Network.

    Uses the SwiGLU activation (Swish-Gated Linear Unit) which has shown
    improved performance over standard FFN with ReLU/GELU.

    Architecture:
        FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

    The hidden dimension is computed as:
        hidden = int(d_model * ffn_multiplier * 2/3)

    This accounts for the gating mechanism doubling the effective params.
    """

    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        ffn_multiplier: float = 2.67,
        *,
        key: jax.Array,
    ):
        """
        Args:
            d_model: Model dimension.
            ffn_multiplier: Hidden dimension multiplier.
                           With SwiGLU, use 2.67 for ~4x effective expansion.
            key: PRNG key for initialization.
        """
        self.d_model = d_model

        # Compute hidden dim with 2/3 factor for SwiGLU
        # This ensures parameter count matches standard 4x FFN
        self.hidden_dim = int(d_model * ffn_multiplier * 2 / 3)

        # Round to multiple of 128 for TPU alignment
        self.hidden_dim = ((self.hidden_dim + 127) // 128) * 128

        k1, k2, k3 = jax.random.split(key, 3)

        # Gate and up projections (both to hidden_dim)
        self.w_gate = eqx.nn.Linear(d_model, self.hidden_dim, key=k1)
        self.w_up = eqx.nn.Linear(d_model, self.hidden_dim, key=k2)

        # Down projection back to d_model
        self.w_down = eqx.nn.Linear(self.hidden_dim, d_model, key=k3)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        """
        Apply SwiGLU FFN.

        Args:
            x: Input vector [D].

        Returns:
            Output vector [D].
        """
        # Gate path with Swish activation
        gate = jax.nn.swish(self.w_gate(x))

        # Up path (linear)
        up = self.w_up(x)

        # Element-wise gating
        hidden = gate * up

        # Down projection
        return self.w_down(hidden)


class GEGLUFFN(eqx.Module):
    """
    GEGLU Feed-Forward Network.

    Alternative to SwiGLU using GELU activation instead of Swish.
    """

    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        ffn_multiplier: float = 2.67,
        *,
        key: jax.Array,
    ):
        self.d_model = d_model
        self.hidden_dim = int(d_model * ffn_multiplier * 2 / 3)
        self.hidden_dim = ((self.hidden_dim + 127) // 128) * 128

        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(d_model, self.hidden_dim, key=k1)
        self.w_up = eqx.nn.Linear(d_model, self.hidden_dim, key=k2)
        self.w_down = eqx.nn.Linear(self.hidden_dim, d_model, key=k3)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        gate = jax.nn.gelu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        return self.w_down(hidden)


class StandardFFN(eqx.Module):
    """
    Standard transformer FFN with GELU activation.

    For comparison with SwiGLU variants.
    """

    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear
    d_model: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        ffn_multiplier: float = 4.0,
        *,
        key: jax.Array,
    ):
        self.d_model = d_model
        self.hidden_dim = int(d_model * ffn_multiplier)
        self.hidden_dim = ((self.hidden_dim + 127) // 128) * 128

        k1, k2 = jax.random.split(key)
        self.w_up = eqx.nn.Linear(d_model, self.hidden_dim, key=k1)
        self.w_down = eqx.nn.Linear(self.hidden_dim, d_model, key=k2)

    def __call__(self, x: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.w_down(jax.nn.gelu(self.w_up(x)))
