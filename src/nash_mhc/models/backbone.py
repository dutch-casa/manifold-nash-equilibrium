"""
MAHA Language Model backbone.

Full transformer architecture with hierarchical multiscale attention
and manifold-constrained residual connections.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array, PRNGKeyArray

from nash_mhc.types.configs import ModelConfig
from nash_mhc.blocks.decoder_block import MAHADecoderBlock
from nash_mhc.layers.ffn import RMSNorm


class MAHALanguageModel(eqx.Module):
    """
    Full MAHA language model for autoregressive generation.

    Architecture:
    - Token embedding
    - N × MAHA decoder blocks
    - Final RMSNorm
    - LM head (tied to embeddings optional)

    Type Parameters:
        B: Batch size
        N: Sequence length
        D: Model dimension
        V: Vocabulary size
    """

    # Embeddings
    token_embedding: eqx.nn.Embedding

    # Decoder blocks
    blocks: tuple[MAHADecoderBlock, ...]

    # Output
    final_norm: RMSNorm
    lm_head: eqx.nn.Linear

    # Configuration (static)
    config: ModelConfig = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            config: Model configuration.
            key: PRNG key for initialization.
        """
        self.config = config

        # Split keys
        keys = jax.random.split(key, config.num_layers + 3)

        # Token embedding
        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.d_model,
            key=keys[0],
        )

        # Decoder blocks
        self.blocks = tuple(
            MAHADecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_scales=config.num_scales,
                max_seq_len=config.max_seq_len,
                compression_ratio=config.compression_ratio,
                ffn_multiplier=config.ffn_multiplier,
                aggregation=config.aggregation,
                nash_iterations=config.nash_iterations,
                sinkhorn_iterations=config.sinkhorn_iterations,
                key=keys[i + 1],
            )
            for i in range(config.num_layers)
        )

        # Output layers
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = eqx.nn.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            key=keys[-1],
        )

    def __call__(
        self,
        input_ids: Int[Array, "B N"],
        causal: bool = True,
    ) -> tuple[Float[Array, "B N V"], Float[Array, ""]]:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token indices [B, N].
            causal: Whether to apply causal masking.

        Returns:
            Tuple of:
            - logits: Vocabulary logits [B, N, V]
            - total_aux_loss: Sum of aggregation auxiliary losses
        """
        # Embed tokens
        x = jax.vmap(jax.vmap(self.token_embedding))(input_ids)

        # Process through decoder blocks
        total_aux_loss = jnp.array(0.0)

        for block in self.blocks:
            x, aux_loss = block(x, causal=causal)
            total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        x = jax.vmap(jax.vmap(self.final_norm))(x)

        # LM head
        logits = jax.vmap(jax.vmap(self.lm_head))(x)

        return logits, total_aux_loss

    def generate(
        self,
        input_ids: Int[Array, "B N"],
        max_new_tokens: int,
        *,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Int[Array, "B M"]:
        """
        Autoregressive text generation.

        Args:
            input_ids: Initial token indices [B, N].
            max_new_tokens: Number of tokens to generate.
            key: PRNG key for sampling.
            temperature: Sampling temperature.
            top_k: If set, sample from top-k tokens only.

        Returns:
            Generated token indices [B, N + max_new_tokens].
        """
        B, N = input_ids.shape
        generated = input_ids

        for i in range(max_new_tokens):
            # Get logits for last position
            logits, _ = self(generated, causal=True)
            next_logits = logits[:, -1, :]  # [B, V]

            # Apply temperature
            next_logits = next_logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                # Get top-k values and indices
                top_values, top_indices = jax.lax.top_k(next_logits, top_k)
                # Create mask for non-top-k values
                mask = jnp.ones_like(next_logits) * (-1e9)
                # Scatter top-k values back
                next_logits = mask.at[jnp.arange(B)[:, None], top_indices].set(
                    top_values
                )

            # Sample
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits, axis=-1)

            # Append
            generated = jnp.concatenate(
                [generated, next_token[:, None]],
                axis=1,
            )

        return generated

    def count_params(self) -> int:
        """Count total parameters in the model."""
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


class MAHALanguageModelTied(eqx.Module):
    """
    MAHA Language Model with tied embedding weights.

    The LM head shares weights with the token embedding,
    reducing parameter count.
    """

    token_embedding: eqx.nn.Embedding
    blocks: tuple[MAHADecoderBlock, ...]
    final_norm: RMSNorm
    config: ModelConfig = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config

        keys = jax.random.split(key, config.num_layers + 2)

        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.d_model,
            key=keys[0],
        )

        self.blocks = tuple(
            MAHADecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_scales=config.num_scales,
                max_seq_len=config.max_seq_len,
                compression_ratio=config.compression_ratio,
                ffn_multiplier=config.ffn_multiplier,
                aggregation=config.aggregation,
                nash_iterations=config.nash_iterations,
                sinkhorn_iterations=config.sinkhorn_iterations,
                key=keys[i + 1],
            )
            for i in range(config.num_layers)
        )

        self.final_norm = RMSNorm(config.d_model)

    def __call__(
        self,
        input_ids: Int[Array, "B N"],
        causal: bool = True,
    ) -> tuple[Float[Array, "B N V"], Float[Array, ""]]:
        """Forward pass with tied output weights."""
        x = jax.vmap(jax.vmap(self.token_embedding))(input_ids)

        total_aux_loss = jnp.array(0.0)
        for block in self.blocks:
            x, aux_loss = block(x, causal=causal)
            total_aux_loss = total_aux_loss + aux_loss

        x = jax.vmap(jax.vmap(self.final_norm))(x)

        # Tied output: x @ embedding.T
        # embedding.weight: [V, D]
        # x: [B, N, D]
        # logits: [B, N, V]
        logits = jnp.einsum("bnd,vd->bnv", x, self.token_embedding.weight)

        return logits, total_aux_loss
