"""Grain-based batching for MAHA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import grain
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int, Array

from .datasets import (
    DatasetConfig,
    load_text_dataset,
    materialize_sequences,
)
from .tokenizer import TokenizerAdapter


class _SequenceMapDataset(grain.MapDataset[dict[str, np.ndarray]]):
    def __init__(self, sequences: Sequence[dict[str, np.ndarray]]):
        super().__init__()
        self._sequences = tuple(sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index):  # type: ignore[override]
        if 0 <= index < len(self._sequences):
            return self._sequences[index]
        return None


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    token_ids: Int[Array, "B N"]
    attention_mask: Int[Array, "B N"]


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    batch_size: int
    shuffle_seed: int = 0
    num_epochs: int = 1
    drop_remainder: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")


def _to_jax_batch(batch: dict[str, np.ndarray]) -> SequenceBatch:
    return SequenceBatch(
        token_ids=jnp.asarray(batch["input_ids"]),
        attention_mask=jnp.asarray(batch["attention_mask"]),
    )


def build_sequence_dataset(
    dataset_config: DatasetConfig,
    tokenizer: TokenizerAdapter,
    *,
    max_sequences: int | None = None,
) -> Sequence[dict[str, np.ndarray]]:
    """Materialize a bounded set of tokenized sequences for training."""
    ds = load_text_dataset(dataset_config)
    return materialize_sequences(
        ds,
        tokenizer,
        text_field=dataset_config.text_field,
        max_sequences=max_sequences,
    )


def create_grain_dataset(
    sequences: Sequence[dict[str, np.ndarray]],
    *,
    config: LoaderConfig,
) -> grain.MapDataset[SequenceBatch]:
    """Create a Grain pipeline with shuffle, repeat, and batching."""
    dataset = _SequenceMapDataset(sequences).seed(config.shuffle_seed)
    if len(sequences) == 0:
        raise ValueError("No sequences provided to Grain dataset")
    dataset = dataset.shuffle(seed=config.shuffle_seed)
    dataset = dataset.repeat(num_epochs=config.num_epochs, reseed_each_epoch=True)
    dataset = dataset.batch(config.batch_size, drop_remainder=config.drop_remainder)
    dataset = dataset.map(_to_jax_batch)
    return dataset


def iterate_batches(
    dataset: grain.MapDataset[SequenceBatch],
) -> Iterator[SequenceBatch]:
    """Yield batches from Grain dataset."""
    for batch in dataset:
        yield batch
