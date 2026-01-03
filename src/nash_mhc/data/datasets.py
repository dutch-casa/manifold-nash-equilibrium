"""HF dataset helpers for Grain-based loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from datasets import Dataset, IterableDataset, load_dataset

from .tokenizer import TokenizerAdapter, TokenizerOutput


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Declarative config for HF datasets."""

    path: str
    split: str
    name: str | None = None
    text_field: str = "text"
    streaming: bool = False


def load_text_dataset(config: DatasetConfig) -> Dataset | IterableDataset:
    """Load a HF dataset respecting streaming mode."""
    return load_dataset(
        path=config.path,
        name=config.name,
        split=config.split,
        streaming=config.streaming,
    )


def iter_tokenized_sequences(
    dataset: Dataset | IterableDataset,
    tokenizer: TokenizerAdapter,
    *,
    text_field: str,
) -> Iterator[TokenizerOutput]:
    """Yield fixed-length tokenized sequences."""
    for row in dataset:
        text = row[text_field]
        if not isinstance(text, str):
            text = str(text)
        yield tokenizer.encode(text)


def materialize_sequences(
    dataset: Dataset | IterableDataset,
    tokenizer: TokenizerAdapter,
    *,
    text_field: str,
    max_sequences: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Tokenize dataset rows eagerly to feed Grain MapDatasets."""
    sequences: list[dict[str, np.ndarray]] = []
    for idx, item in enumerate(iter_tokenized_sequences(dataset, tokenizer, text_field=text_field)):
        sequences.append(
            {
                "input_ids": np.asarray(item.input_ids, dtype=np.int32),
                "attention_mask": np.asarray(item.attention_mask, dtype=np.int32),
            }
        )
        if max_sequences is not None and idx + 1 >= max_sequences:
            break
    return sequences

