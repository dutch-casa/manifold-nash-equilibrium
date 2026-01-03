"""Configuration types with compile-time invariant enforcement."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """
    Immutable model hyperparameters.

    All dimensions are TPU-aligned (multiples of 128) for optimal MXU utilization.

    Invariants enforced at construction:
    - vocab_size % 128 == 0
    - max_seq_len % 128 == 0
    - d_model % 128 == 0
    - d_model % num_heads == 0
    - max_seq_len % (compression_ratio ^ (num_scales - 1)) == 0
    """

    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    num_scales: int
    compression_ratio: int
    ffn_multiplier: float
    sinkhorn_iterations: int
    nash_iterations: int
    aggregation: Literal["nash", "convex"]
    dtype: Literal["bfloat16", "float32"] = "bfloat16"

    def __post_init__(self) -> None:
        # TPU alignment invariants
        if self.vocab_size % 128 != 0:
            raise ValueError(
                f"vocab_size must be multiple of 128, got {self.vocab_size}"
            )
        if self.max_seq_len % 128 != 0:
            raise ValueError(
                f"max_seq_len must be multiple of 128, got {self.max_seq_len}"
            )
        if self.d_model % 128 != 0:
            raise ValueError(f"d_model must be multiple of 128, got {self.d_model}")

        # Attention head invariant
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        # Scale hierarchy invariant
        if self.num_scales < 2:
            raise ValueError(f"num_scales must be >= 2, got {self.num_scales}")
        if self.compression_ratio < 2:
            raise ValueError(
                f"compression_ratio must be >= 2, got {self.compression_ratio}"
            )

        # Sequence alignment invariant
        scale_factor = self.compression_ratio ** (self.num_scales - 1)
        if self.max_seq_len % scale_factor != 0:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) must be divisible by "
                f"compression_ratio^(num_scales-1) = {scale_factor}"
            )

        min_scale_len = self.max_seq_len // scale_factor
        if min_scale_len < 1:
            raise ValueError(
                f"max_seq_len too small: coarsest scale would have length {min_scale_len}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.num_heads

    @property
    def ffn_hidden(self) -> int:
        """Hidden dimension for FFN (SwiGLU uses 2/3 factor internally)."""
        return int(self.d_model * self.ffn_multiplier)

    def scale_lengths(self) -> tuple[int, ...]:
        """Compute sequence lengths at each scale."""
        lengths = [self.max_seq_len]
        for _ in range(self.num_scales - 1):
            lengths.append(lengths[-1] // self.compression_ratio)
        return tuple(lengths)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    max_grad_norm: float
    lambda_sparsity: float
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be < total_steps ({self.total_steps})"
            )

    @property
    def effective_batch_size(self) -> int:
        """Batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


# Default configurations for development
DEFAULT_MODEL_CONFIG = ModelConfig(
    vocab_size=32768,
    max_seq_len=2048,
    d_model=2048,
    num_heads=16,
    num_layers=24,
    num_scales=4,
    compression_ratio=2,
    ffn_multiplier=2.67,
    sinkhorn_iterations=10,
    nash_iterations=3,
    aggregation="nash",
    dtype="bfloat16",
)

SMALL_MODEL_CONFIG = ModelConfig(
    vocab_size=32768,
    max_seq_len=512,
    d_model=512,
    num_heads=8,
    num_layers=6,
    num_scales=3,
    compression_ratio=2,
    ffn_multiplier=2.67,
    sinkhorn_iterations=10,
    nash_iterations=3,
    aggregation="nash",
    dtype="float32",
)

MEDIUM_1_5B_CONFIG = ModelConfig(
    vocab_size=65536,
    max_seq_len=4096,
    d_model=1600,
    num_heads=16,
    num_layers=14,
    num_scales=4,
    compression_ratio=2,
    ffn_multiplier=2.67,
    sinkhorn_iterations=10,
    nash_iterations=3,
    aggregation="nash",
    dtype="bfloat16",
)

LARGE_3B_CONFIG = ModelConfig(
    vocab_size=65536,
    max_seq_len=4096,
    d_model=2048,
    num_heads=16,
    num_layers=28,
    num_scales=4,
    compression_ratio=2,
    ffn_multiplier=2.67,
    sinkhorn_iterations=10,
    nash_iterations=3,
    aggregation="nash",
    dtype="bfloat16",
)

LARGE_3B_TRAINING_CONFIG = TrainingConfig(
    batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=2000,
    total_steps=100000,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lambda_sparsity=0.01,
    seed=42,
)

MEDIUM_1_5B_TRAINING_CONFIG = TrainingConfig(
    batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=2000,
    total_steps=100000,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lambda_sparsity=0.01,
    seed=42,
)
