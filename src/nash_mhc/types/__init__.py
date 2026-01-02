"""Type definitions for nash-mhc."""

from nash_mhc.types.configs import ModelConfig, TrainingConfig
from nash_mhc.types.arrays import (
    DoublyStochastic,
    SimplexWeights,
    BoundedResidual,
    AttentionWeights,
)
from nash_mhc.types.invariants import (
    assert_doubly_stochastic,
    assert_simplex,
    assert_spectral_norm_bounded,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DoublyStochastic",
    "SimplexWeights",
    "BoundedResidual",
    "AttentionWeights",
    "assert_doubly_stochastic",
    "assert_simplex",
    "assert_spectral_norm_bounded",
]
