"""Streaming data loader for Nash-MHC project using Grain and HF datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import grain.python as grain
import jax
import numpy as np
from datasets import IterableDataset, load_dataset

from .loader import LoaderConfig, SequenceBatch
from .tokenizer import TokenizerAdapter


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Configuration for streaming data loading."""

    path: str = "HuggingFaceTB/smollm-corpus"
    split: str = "train"
    name: str | None = None
    text_field: str = "text"
    shuffle_buffer_size: int = 10_000
    num_workers: int = 8
    prefetch_buffer_size: int = 16


class HFStreamingIterator(grain.DatasetIterator[dict[str, np.ndarray]]):
    def __init__(self, dataset: "HFStreamingIterDataset"):
        super().__init__(parents=())
        self._dataset = dataset
        self._iterator = self._create_iterator()

    def _create_iterator(self):
        ds = self._dataset._get_sharded_dataset()

        def _generate():
            for row in ds:
                text = row[self._dataset._config.text_field]
                if not isinstance(text, str):
                    text = str(text)

                tokenized = self._dataset._tokenizer.encode(text)
                yield {
                    "input_ids": tokenized.input_ids,
                    "attention_mask": tokenized.attention_mask,
                }

        return _generate()

    def __next__(self) -> dict[str, np.ndarray]:
        return next(self._iterator)

    def __iter__(self) -> grain.DatasetIterator[dict[str, np.ndarray]]:
        return self

    def get_state(self) -> dict[str, Any]:
        return {"position": 0}

    def set_state(self, state: dict[str, Any]) -> None:
        pass


class HFStreamingIterDataset(grain.IterDataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        config: StreamingConfig,
        tokenizer: TokenizerAdapter,
        seed: int = 0,
    ):
        super().__init__()
        self._config = config
        self._tokenizer = tokenizer
        self._seed = seed
        self._process_index = jax.process_index()
        self._process_count = jax.process_count()
        self._worker_index = 0
        self._worker_count = 1

    def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
        assert sequential_slice, "Only sequential slicing is supported."
        self._worker_index = sl.start if sl.start is not None else 0
        self._worker_count = sl.step if sl.step is not None else 1

    def _get_sharded_dataset(self) -> IterableDataset:
        ds = load_dataset(
            path=self._config.path,
            name=self._config.name,
            split=self._config.split,
            streaming=True,
        )

        if not isinstance(ds, IterableDataset):
            raise TypeError(f"Expected IterableDataset, got {type(ds)}")

        if self._process_count > 1:
            ds = ds.shard(num_shards=self._process_count, index=self._process_index)

        if self._worker_count > 1:
            ds = ds.shard(num_shards=self._worker_count, index=self._worker_index)

        ds = ds.shuffle(seed=self._seed, buffer_size=self._config.shuffle_buffer_size)

        return ds

    def __iter__(self) -> grain.DatasetIterator[dict[str, np.ndarray]]:
        return HFStreamingIterator(self)


def _to_sequence_batch(batch: dict[str, Any]) -> SequenceBatch:
    """Convert dict batch to SequenceBatch dataclass."""
    return SequenceBatch(
        token_ids=jax.numpy.asarray(batch["input_ids"]),
        attention_mask=jax.numpy.asarray(batch["attention_mask"]),
    )


def create_streaming_dataloader(
    config: StreamingConfig,
    loader_config: LoaderConfig,
    tokenizer: TokenizerAdapter,
) -> grain.IterDataset[SequenceBatch]:
    ds = HFStreamingIterDataset(
        config=config,
        tokenizer=tokenizer,
        seed=loader_config.shuffle_seed,
    )

    ds = ds.batch(loader_config.batch_size, drop_remainder=loader_config.drop_remainder)
    ds = ds.map(_to_sequence_batch)

    if config.num_workers > 1:
        ds = grain.experimental.ThreadPrefetchIterDataset(
            parent=ds,
            prefetch_buffer_size=config.prefetch_buffer_size,
        )

    return ds


if __name__ == "__main__":
    # Example usage for 3B coding model training
    from transformers import AutoTokenizer
    from .tokenizer import TokenizerConfig

    # 1. Initialize Tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/smollm-360M")
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    tokenizer = TokenizerAdapter(hf_tokenizer, TokenizerConfig(max_length=2048))

    # 2. Configure Streaming (e.g. using cosmopedia subset)
    s_config = StreamingConfig(
        path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", num_workers=4
    )

    # 3. Configure Loader
    l_config = LoaderConfig(batch_size=32, shuffle_seed=42)

    # 4. Create DataLoader
    loader = create_streaming_dataloader(s_config, l_config, tokenizer)

    print("Streaming DataLoader initialized.")
    # for batch in loader:
    #     print(f"Batch shape: {batch.token_ids.shape}")
    #     break
