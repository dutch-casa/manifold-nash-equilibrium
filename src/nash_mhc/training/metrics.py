"""Training metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Float, Array


@dataclass(frozen=True, slots=True)
class MetricState:
    step: int
    loss: float
    tokens: int
    tokens_per_second: float
    perplexity: float


def compute_perplexity(loss: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.exp(loss)


def aggregate_metrics(
    total_loss: float,
    total_tokens: int,
    elapsed_time_s: float,
    step: int,
) -> MetricState:
    token_rate = total_tokens / max(elapsed_time_s, 1e-6)
    ppl = float(jnp.exp(jnp.array(total_loss)))
    return MetricState(
        step=step,
        loss=total_loss,
        tokens=total_tokens,
        tokens_per_second=token_rate,
        perplexity=ppl,
    )

