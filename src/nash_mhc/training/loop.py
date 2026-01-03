"""Training loop for MAHA language model."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable

import equinox as eqx
import jax
import optax

from nash_mhc.data.loader import SequenceBatch
from nash_mhc.models.backbone import MAHALanguageModel
from nash_mhc.training.loss import LossComponents, cross_entropy_loss
from nash_mhc.training.metrics import MetricState, aggregate_metrics
from nash_mhc.types.configs import TrainingConfig


@dataclass
class TrainState:
    model: MAHALanguageModel
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    step: int
    pad_token_id: int


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """AdamW schedule with linear warmup and cosine decay."""
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    decay = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(config.total_steps - config.warmup_steps, 1),
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[config.warmup_steps],
    )
    return optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )


def init_train_state(
    model: MAHALanguageModel,
    training_config: TrainingConfig,
    *,
    pad_token_id: int,
) -> TrainState:
    optimizer = create_optimizer(training_config)
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    return TrainState(
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        step=0,
        pad_token_id=pad_token_id,
    )


def _prepare_inputs(batch: SequenceBatch) -> tuple:
    inputs = batch.token_ids[:, :-1]
    labels = batch.token_ids[:, 1:]
    return inputs, labels


def train_step(state: TrainState, batch: SequenceBatch) -> tuple[TrainState, LossComponents]:
    """Single JIT-able training step."""
    inputs, labels = _prepare_inputs(batch)
    params, static = eqx.partition(state.model, eqx.is_array)

    def loss_fn(filtered_params):
        model = eqx.combine(filtered_params, static)
        logits, aux_loss = model(inputs, causal=True)
        logits = logits[:, :-1, :]
        components = cross_entropy_loss(
            logits,
            labels,
            padding_id=state.pad_token_id,
            aux_loss=aux_loss,
        )
        return components.total, components

    (loss_value, components), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = state.optimizer.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_model = eqx.combine(new_params, static)
    new_state = replace(state, model=new_model, opt_state=new_opt_state, step=state.step + 1)
    return new_state, components


def train_epoch(
    state: TrainState,
    batches: Iterable[SequenceBatch],
    *,
    max_steps: int | None = None,
) -> tuple[TrainState, MetricState]:
    """Iterate over an epoch of data."""
    total_loss = 0.0
    total_tokens = 0
    start = time.time()
    for idx, batch in enumerate(batches):
        if max_steps is not None and idx >= max_steps:
            break
        state, metrics = train_step(state, batch)
        total_loss += float(metrics.total)
        total_tokens += int(batch.token_ids.size)
    elapsed = time.time() - start
    average_loss = total_loss / max(idx + 1, 1)
    summary = aggregate_metrics(
        total_loss=average_loss,
        total_tokens=total_tokens,
        elapsed_time_s=elapsed,
        step=state.step,
    )
    return state, summary

