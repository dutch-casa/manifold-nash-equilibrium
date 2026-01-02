"""Semantic array types for compile-time documentation and runtime validation."""

from typing import NewType, TypeVar

from jaxtyping import Float, Int, Array


# === Dimension type variables (for documentation) ===
B = TypeVar("B")  # Batch
N = TypeVar("N")  # Sequence length
D = TypeVar("D")  # Model dimension
H = TypeVar("H")  # Number of heads
K = TypeVar("K")  # Head dimension (D // H)
L = TypeVar("L")  # Number of scales
V = TypeVar("V")  # Vocabulary size


# === Semantic newtypes ===
# These provide compile-time documentation about array invariants.
# Runtime enforcement is handled by the invariants module.

DoublyStochastic = NewType(
    "DoublyStochastic",
    Float[Array, "... n n"],
)
"""
Matrix on the Birkhoff Polytope.

Invariants:
- M @ 1 = 1 (rows sum to 1)
- M.T @ 1 = 1 (columns sum to 1)
- M >= 0 (non-negative entries)
"""

SimplexWeights = NewType(
    "SimplexWeights",
    Float[Array, "... L"],
)
"""
Weights on the probability simplex.

Invariants:
- sum(w) = 1
- w >= 0
"""

BoundedResidual = NewType(
    "BoundedResidual",
    Float[Array, "B N D"],
)
"""
Residual connection output with bounded spectral norm.

Invariant:
- ||H||_2 <= 1 (non-expansive)
"""

AttentionWeights = NewType(
    "AttentionWeights",
    Float[Array, "B H N N"],
)
"""
Attention probability matrix (row-stochastic).

Invariant:
- Each row sums to 1 (after softmax)
"""

RawLogits = NewType(
    "RawLogits",
    Float[Array, "B H N N"],
)
"""Pre-softmax attention scores (unbounded)."""

TokenIds = NewType(
    "TokenIds",
    Int[Array, "B N"],
)
"""Integer token indices."""

Embeddings = NewType(
    "Embeddings",
    Float[Array, "B N D"],
)
"""Token embeddings."""

ScaleOutputs = NewType(
    "ScaleOutputs",
    Float[Array, "B L N D"],
)
"""Stacked outputs from all hierarchical scales."""
