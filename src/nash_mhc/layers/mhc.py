"""
Manifold-Constrained Hyper-Connection (mHC) layer.

Projects residual connection weight matrices onto the Birkhoff polytope
to ensure signal mean conservation and bounded spectral norm.

Reference: Paper 2512.24880v1
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from nash_mhc.primitives.sinkhorn import sinkhorn_knopp


class ManifoldHyperConnection(eqx.Module):
    """
    Manifold-constrained hyper-connection layer.

    Replaces standard residual connections with a manifold-constrained
    mixing operation that preserves signal mean across layers.

    The residual connection becomes:
        output = H @ residual + (I - H) @ block_output

    where H is projected onto the Birkhoff polytope (doubly stochastic matrices)
    via Sinkhorn-Knopp algorithm.

    Invariants:
    - H is doubly stochastic: H @ 1 = 1, H.T @ 1 = 1, H >= 0
    - ||H||_2 <= 1 (spectral norm bounded, non-expansive)
    - Signal mean is conserved across layers
    """

    log_alpha: Float[Array, "D D"]
    layer_scale: Float[Array, "D"]
    sinkhorn_iters: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        sinkhorn_iters: int = 10,
        *,
        key: jax.Array,
    ):
        """
        Args:
            d_model: Model dimension.
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations.
            key: PRNG key for initialization.
        """
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters

        k1, k2 = jax.random.split(key)

        # Initialize log_alpha near zero (identity-like after Sinkhorn)
        # Small noise breaks symmetry
        noise = jax.random.normal(k1, (d_model, d_model)) * 0.01
        self.log_alpha = noise

        # Layer scale initialized to 1 (identity behavior)
        self.layer_scale = jnp.ones(d_model)

    def __call__(
        self,
        residual: Float[Array, "B N D"],
        block_output: Float[Array, "B N D"],
    ) -> Float[Array, "B N D"]:
        """
        Apply manifold-constrained residual connection.

        The operation computes:
            H = sinkhorn(log_alpha)  # Doubly stochastic projection
            output = (H @ residual + (I - H) @ block_output) * layer_scale

        This can be rewritten as:
            output = H @ (residual - block_output) + block_output
            output = output * layer_scale

        Args:
            residual: Skip connection input [B, N, D].
            block_output: Output from attention/FFN block [B, N, D].

        Returns:
            Mixed output [B, N, D] on the manifold.
        """
        # Project log_alpha to doubly stochastic matrix
        H = sinkhorn_knopp(self.log_alpha, self.sinkhorn_iters)

        # Compute residual contribution
        # H @ residual: [D, D] @ [B, N, D] -> need einsum for batch handling
        # einsum "de,bne->bnd" applies H to the D dimension
        res_contrib = jnp.einsum("de,bne->bnd", H, residual)

        # Compute (I - H) @ block_output
        I_minus_H = jnp.eye(self.d_model, dtype=H.dtype) - H
        block_contrib = jnp.einsum("de,bne->bnd", I_minus_H, block_output)

        # Combine with learned layer scale
        output = (res_contrib + block_contrib) * self.layer_scale

        return output

    def get_mixing_matrix(self) -> Float[Array, "D D"]:
        """Return the current doubly stochastic mixing matrix."""
        return sinkhorn_knopp(self.log_alpha, self.sinkhorn_iters)


class ManifoldHyperConnectionLite(eqx.Module):
    """
    Lightweight mHC variant with per-channel mixing instead of full matrix.

    Uses diagonal mixing weights (element-wise) instead of full matrix
    for O(D) instead of O(D²) parameters and compute.

    The residual connection becomes:
        output = alpha * residual + (1 - alpha) * block_output

    where alpha is constrained to [0, 1] via sigmoid.
    """

    logit_alpha: Float[Array, "D"]
    layer_scale: Float[Array, "D"]
    d_model: int = eqx.field(static=True)

    def __init__(self, d_model: int, *, key: jax.Array):
        self.d_model = d_model

        k1, k2 = jax.random.split(key)

        # Initialize near 0.5 mixing (logit = 0)
        self.logit_alpha = jax.random.normal(k1, (d_model,)) * 0.01
        self.layer_scale = jnp.ones(d_model)

    def __call__(
        self,
        residual: Float[Array, "B N D"],
        block_output: Float[Array, "B N D"],
    ) -> Float[Array, "B N D"]:
        """Apply element-wise manifold-constrained mixing."""
        # Constrain alpha to [0, 1]
        alpha = jax.nn.sigmoid(self.logit_alpha)

        # Element-wise mixing
        output = alpha * residual + (1 - alpha) * block_output

        return output * self.layer_scale
