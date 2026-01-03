"""Loss functions for MAHA training."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class LossComponents:
    total: Float[Array, ""]
    cross_entropy: Float[Array, ""]
    aux_loss: Float[Array, ""]

    def tree_flatten(self):
        return ((self.total, self.cross_entropy, self.aux_loss), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        total, cross_entropy, aux_loss = children
        return cls(total=total, cross_entropy=cross_entropy, aux_loss=aux_loss)


def cross_entropy_loss(
    logits: Float[Array, "B N V"],
    labels: Int[Array, "B N"],
    *,
    padding_id: int,
    aux_loss: Float[Array, ""] | None = None,
    label_smoothing: float = 0.0,
) -> LossComponents:
    """Compute cross-entropy with padding mask."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    vocab = logits.shape[-1]
    targets = jax.nn.one_hot(labels, vocab)
    if label_smoothing > 0:
        smooth = label_smoothing / vocab
        targets = (1.0 - label_smoothing) * targets + smooth
    nll = -jnp.sum(targets * log_probs, axis=-1)
    mask = (labels != padding_id).astype(logits.dtype)
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    ce_loss = jnp.sum(nll * mask) / denom
    aux = aux_loss if aux_loss is not None else jnp.array(0.0, dtype=logits.dtype)
    total = ce_loss + aux
    return LossComponents(total=total, cross_entropy=ce_loss, aux_loss=aux)
