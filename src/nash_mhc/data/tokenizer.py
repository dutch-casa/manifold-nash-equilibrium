"""Tokenizer adapters with invariant enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import numpy.typing as npt


class TokenizerLike(Protocol):
    """Minimal HF tokenizer protocol."""

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_attention_mask: bool,
        return_tensors: str | None = None,
    ) -> Any: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Tokenizer hyperparameters aligned with `ModelConfig`."""

    max_length: int
    pad_id: int | None = None
    eos_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


@dataclass(frozen=True, slots=True)
class TokenizerOutput:
    """Canonical tokenized sequence representation using numpy arrays for data loading boundary."""

    input_ids: npt.NDArray[np.int32]
    attention_mask: npt.NDArray[np.int32]


class TokenizerAdapter:
    """Wraps a Hugging Face tokenizer-like object with strict outputs."""

    def __init__(self, tokenizer: TokenizerLike, config: TokenizerConfig):
        self._tokenizer = tokenizer
        pad = config.pad_id or getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            raise ValueError("Tokenizer must define pad_token_id")
        eos = config.eos_id or getattr(tokenizer, "eos_token_id", pad)
        self._pad_id = int(pad)
        self._eos_id = int(eos)
        self._config = config

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def max_length(self) -> int:
        return self._config.max_length

    def encode(self, text: str) -> TokenizerOutput:
        """Tokenize a single string with padding/truncation."""
        encoded = self._tokenizer(
            text,
            max_length=self._config.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        attn = np.asarray(encoded["attention_mask"], dtype=np.int32)
        if attn.ndim == 2:
            attn = attn[0]
        if input_ids.shape[0] != self._config.max_length:
            raise ValueError(
                f"Tokenized length {input_ids.shape[0]} must equal max_length {self._config.max_length}"
            )
        return TokenizerOutput(input_ids=input_ids, attention_mask=attn)

    def batch_encode(self, texts: Sequence[str]) -> list[TokenizerOutput]:
        """Tokenize multiple strings."""
        batch = self._tokenizer(
            list(texts),
            max_length=self._config.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        ids = np.asarray(batch["input_ids"], dtype=np.int32)
        attn = np.asarray(batch["attention_mask"], dtype=np.int32)
        outputs: list[TokenizerOutput] = []
        for i in range(ids.shape[0]):
            outputs.append(
                TokenizerOutput(
                    input_ids=ids[i],
                    attention_mask=attn[i],
                )
            )
        return outputs
