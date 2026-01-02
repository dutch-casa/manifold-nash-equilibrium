# Deep-Equilibrium Engine: JAX/TPU Implementation Plan

## Overview

Greenfield JAX/Equinox implementation of a "Deep-Equilibrium" language model combining:
- **mHC**: Manifold-Constrained Hyper-Connections (Birkhoff polytope via Sinkhorn-Knopp)
- **MAHA**: Nash Equilibrium Multiscale Hierarchical Attention

**Target**: TPU v5p, 1-3B parameters, autoregressive language modeling

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RoPE | Per-scale adjusted | Position `i` at scale `l` maps to `floor(i * r^l)` |
| mHC scope | Both branches | Full signal conservation on attention AND FFN |
| Sequence alignment | Assert failure | Enforce `max_seq_len % (r^(L-1)) == 0` at data load |
| Package manager | uv | User requirement |
| Type checker | ty | User requirement |

---

## Dependencies

```toml
[project]
name = "nash-mhc"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Core JAX ecosystem
    "jax[tpu]>=0.4.30",
    "equinox>=0.11.0",
    "flax>=0.8.0",
    "optax>=0.2.0",
    "lineax>=0.0.5",

    # Type safety
    "jaxtyping>=0.2.28",
    "beartype>=0.18.0",

    # Data pipeline
    "datasets>=2.16.0",
    "grain>=0.2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "hypothesis>=6.100",
]
```

---

## Project Structure

```
nash_mhc/
├── pyproject.toml
├── ty.toml
├── PLAN.md                     # This file
├── src/nash_mhc/
│   ├── __init__.py
│   ├── py.typed
│   │
│   ├── types/                  # Phase 1: Type system
│   │   ├── __init__.py
│   │   ├── arrays.py           # Semantic newtypes (DoublyStochastic, SimplexWeights)
│   │   ├── configs.py          # Frozen ModelConfig, TrainingConfig
│   │   ├── shapes.py           # Compile-time shape tracking
│   │   └── invariants.py       # Runtime assertion helpers
│   │
│   ├── primitives/             # Phase 2: Hardware-aligned ops
│   │   ├── __init__.py
│   │   ├── sinkhorn.py         # Sinkhorn-Knopp via lax.scan + Lineax
│   │   ├── nash_solver.py      # Best-response via lax.fori_loop
│   │   ├── strided_conv.py     # TPU-aligned Conv1d
│   │   ├── upsample.py         # Manual nearest-neighbor
│   │   └── rope.py             # Per-scale rotary position encoding
│   │
│   ├── layers/                 # Phase 3: Equinox modules
│   │   ├── __init__.py
│   │   ├── mhc.py              # ManifoldHyperConnection
│   │   ├── decomposition.py    # HierarchicalDecomposition
│   │   ├── attention.py        # MultiscaleAttention (shared V)
│   │   ├── aggregation.py      # OptimizationAggregation (Nash/Convex)
│   │   └── ffn.py              # SwiGLU feed-forward
│   │
│   ├── blocks/                 # Phase 4: Composed blocks
│   │   ├── __init__.py
│   │   └── decoder_block.py    # MAHADecoderBlock with mHC on both branches
│   │
│   ├── models/                 # Phase 5: Full architecture
│   │   ├── __init__.py
│   │   ├── backbone.py         # MAHALanguageModel
│   │   └── lm_head.py          # Tied embeddings option
│   │
│   ├── sharding/               # Phase 6: TPU distribution
│   │   ├── __init__.py
│   │   ├── mesh.py             # Device mesh (dp, fsdp, tp)
│   │   ├── specs.py            # PartitionSpec definitions
│   │   └── checkpoint.py       # Orbax distributed checkpointing
│   │
│   ├── data/                   # Phase 7: Data pipeline
│   │   ├── __init__.py
│   │   ├── loader.py           # Grain-based deterministic loading
│   │   ├── tokenizer.py        # HF tokenizer wrapper
│   │   └── datasets.py         # GSM8k, The Stack adapters
│   │
│   └── training/               # Phase 8: Training loop
│       ├── __init__.py
│       ├── loop.py             # Main training loop
│       ├── loss.py             # CE + aux loss
│       └── metrics.py          # Perplexity, throughput
│
└── tests/
    ├── conftest.py
    ├── invariants/             # Mathematical property tests
    │   ├── test_sinkhorn.py    # Doubly stochastic invariant
    │   ├── test_nash.py        # Equilibrium convergence
    │   └── test_mhc.py         # Spectral norm bound
    └── shapes/
        └── test_layers.py      # Shape preservation
```

---

## Theoretical Foundations

### 1. mHC (Manifold-Constrained Hyper-Connections)

**Source**: Paper 2512.24880v1

The mHC layer projects residual connection weight matrices onto the **Birkhoff Polytope** (the set of doubly stochastic matrices) to ensure signal mean conservation and bounded spectral norm.

**Birkhoff Polytope Constraint**:
```
P_Mres(H) = {H ∈ R^{n×n} | H·1 = 1, 1^T·H = 1^T, H >= 0}
```

**Sinkhorn-Knopp Algorithm** (iterative projection):
```
M^(t) = T_r(T_c(M^(t-1)))
```
where T_r normalizes rows and T_c normalizes columns. Converges in ~10-20 iterations.

**Key Invariants**:
- Rows and columns sum to 1 (doubly stochastic)
- `||H^res||_2 <= 1` (spectral norm bounded, non-expansive)
- Signal mean conserved across layers

**Mixed Precision**: bf16 compute for matrix multiplications, f32 for Sinkhorn normalization to prevent "manifold drift".

### 2. MAHA (Nash Equilibrium Multiscale Attention)

**Source**: Paper 2512.14925v2

MAHA reformulates attention as a game between hierarchical scales competing for bandwidth.

**Hierarchical Decomposition**:
```
X -> [X_0, X_1, ..., X_{L-1}]
where len(X_l) = len(X_{l-1}) // r
```

**Nash Equilibrium Aggregation**:
```
for t in range(nash_iterations):
    O* = Σ_l w_l * O_l          # Consensus
    Error_l = ||O_l - O*||_2     # Reconstruction error
    w_l ← softmax(-Error)        # Best response
```

**Key Properties**:
- Sub-quadratic complexity: O(N·1.5) vs O(N²)
- Shared V projection across scales (parameter efficient)
- Dynamic weight allocation per sample

### 3. Closed-Form Continuous-Time (CfC)

**Source**: Paper 2106.13898v2

CfC provides closed-form solutions to differential equations, eliminating ODE solver overhead.

**Closed-Form Approximation**:
```
x̃(t) = (x(0) - A)e^{-[w_τt + f(I(t))t]} · f(-I(t)) + A
```

**Benefit**: 150x faster than ODE-based counterparts.

---

## Core Invariants

### 1. Doubly Stochastic (mHC)
```
P @ 1 = 1  AND  P.T @ 1 = 1  AND  P >= 0
```
Enforced by Sinkhorn-Knopp projection.

### 2. Simplex (Nash weights)
```
sum(w) = 1  AND  w >= 0
```
Enforced by softmax over error terms.

### 3. Spectral Norm Bound (mHC)
```
||H^res||_2 <= 1
```
Non-expansive residual connection.

### 4. Sequence Alignment
```
max_seq_len % (compression_ratio ^ (num_scales - 1)) == 0
```
Asserted at config construction.

---

## Data-Oriented Design Rules

1. **Data Dominates Everything**: Enumerate representations and select the one with the smallest failure surface and highest TPU MXU locality.

2. **Structure of Arrays (SoA)**: Prioritize SoA over AoS to maximize memory bandwidth and ensure O(1) hardware-aligned data flow.

3. **Illegal States are Unrepresentable**: Use typestates and exhaustive constructors so that unstable manifolds or non-equilibrium weights cannot exist in memory.

4. **Invariant-First Logic**: Define explicit Pre-conditions {P} and Post-conditions {Q} for every state transition.

5. **Pure Functional Core**: Treat the model as a stateless PyTree transformation; isolate all side effects, I/O, and randomness in the Imperative Shell.

6. **Closed-Form Physics**: Favor closed-form equations over iterative numerical solvers to eliminate computational latency and non-determinism.

7. **Linear Control Flow**: Keep logic top-to-bottom with O(1) complexity per line; use guard clauses to fail-fast instead of deep nesting.

8. **Flat Memory Primitives**: Use contiguous buffers and bitwise primitives (AND, OR, LSHIFT) for O(1) state validation in bounded spaces.

9. **Hardware-First Representation**: Align data structures with the 128×128 systolic array; if the data doesn't fit the hardware physics, change the representation.

10. **Test Invariants, Not Examples**: Assert mathematical properties rather than brittle, example-based outcomes.

11. **Zero-Puffery Interface**: Strictly use blunt, directive phrasing in APIs; avoid boolean flags, sentinel values.

12. **Null-Zero Boundary**: Eliminate Null/None at foreign boundaries; distinguish absence and failure using explicit variants/enums.

---

## Implementation Phases

### Phase 1: Project Setup
- [ ] Initialize uv project with dependencies
- [ ] Configure ty.toml for strict type checking
- [ ] Create ModelConfig with alignment assertions
- [ ] Create TrainingConfig
- [ ] Create semantic array newtypes
- [ ] Create invariant assertion helpers

### Phase 2: Primitives
- [ ] `sinkhorn.py`: Sinkhorn-Knopp via `lax.scan`, mixed precision
- [ ] `nash_solver.py`: Best-response via `lax.fori_loop`
- [ ] `strided_conv.py`: JAX Conv1d with TPU-aligned padding
- [ ] `upsample.py`: Nearest-neighbor via index gathering
- [ ] `rope.py`: Per-scale RoPE with adjusted position indices

### Phase 3: Layers
- [ ] `mhc.py`: ManifoldHyperConnection (learnable log_alpha, Sinkhorn projection)
- [ ] `decomposition.py`: HierarchicalDecomposition (tuple of scales, not list)
- [ ] `attention.py`: MultiscaleAttention (per-scale Q/K, shared V, per-scale RoPE)
- [ ] `aggregation.py`: OptimizationAggregation (Nash or Convex strategy)
- [ ] `ffn.py`: SwiGLU FFN

### Phase 4: Blocks
- [ ] `decoder_block.py`: MAHADecoderBlock
  - RMSNorm -> Decompose -> Attention -> Aggregate -> mHC residual
  - RMSNorm -> FFN -> mHC residual

### Phase 5: Model
- [ ] `backbone.py`: MAHALanguageModel
  - Token embedding
  - N × MAHADecoderBlock
  - Final RMSNorm
  - LM head

### Phase 6: Sharding
- [ ] `mesh.py`: TPU mesh construction (dp, fsdp, tp axes)
- [ ] `specs.py`: PartitionSpec for weights and activations
- [ ] `checkpoint.py`: Orbax checkpointing with sharding

### Phase 7: Data
- [ ] `loader.py`: Grain DataLoader with deterministic shuffling
- [ ] `tokenizer.py`: HF tokenizer with padding to aligned length
- [ ] `datasets.py`: GSM8k, The Stack dataset adapters

### Phase 8: Training
- [ ] `loop.py`: Training loop with gradient accumulation
- [ ] `loss.py`: Cross-entropy + lambda × aux_loss
- [ ] `metrics.py`: Perplexity, tokens/sec

---

## Key Implementation Details

### Sinkhorn-Knopp (primitives/sinkhorn.py)

```python
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array

@partial(jax.jit, static_argnames=["num_iterations"])
def sinkhorn_knopp(
    log_alpha: Float[Array, "... n n"],
    num_iterations: int = 10,
) -> Float[Array, "... n n"]:
    """
    Sinkhorn-Knopp algorithm for Birkhoff polytope projection.

    Computes doubly stochastic matrix via iterative row/column normalization
    in log-space for numerical stability.

    Invariant: Output M satisfies M @ 1 = 1 and M.T @ 1 = 1
    """
    n = log_alpha.shape[-1]
    u = jnp.zeros(log_alpha.shape[:-1])
    v = jnp.zeros(log_alpha.shape[:-1])

    def scan_body(carry, _):
        u, v = carry
        # f32 for numerical stability
        u_new = -jax.nn.logsumexp(log_alpha + v[..., None, :], axis=-1)
        v_new = -jax.nn.logsumexp(log_alpha + u_new[..., :, None], axis=-2)
        return (u_new, v_new), None

    (u_final, v_final), _ = lax.scan(
        scan_body,
        (u, v),
        xs=None,
        length=num_iterations
    )

    log_P = log_alpha + u_final[..., :, None] + v_final[..., None, :]
    return jnp.exp(log_P).astype(log_alpha.dtype)
```

### Nash Best-Response (primitives/nash_solver.py)

```python
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array

@partial(jax.jit, static_argnames=["num_iterations"])
def nash_best_response(
    scale_outputs: Float[Array, "B L N D"],
    num_iterations: int = 3,
) -> tuple[Float[Array, "B N D"], Float[Array, "B L"]]:
    """
    Nash equilibrium aggregation via iterated best-response dynamics.

    Args:
        scale_outputs: Stacked outputs from all scales [B, L, N, D]
        num_iterations: Number of best-response iterations

    Returns:
        aggregated: Final consensus output [B, N, D]
        weights: Equilibrium weights [B, L]
    """
    B, L, N, D = scale_outputs.shape
    init_weights = jnp.ones((B, L)) / L

    def iteration_body(i, weights):
        w_expanded = weights[:, :, None, None]
        consensus = jnp.sum(scale_outputs * w_expanded, axis=1)
        diffs = scale_outputs - consensus[:, None, :, :]
        errors = jnp.sqrt(jnp.sum(diffs ** 2, axis=(-2, -1)))
        return jax.nn.softmax(-errors, axis=-1)

    final_weights = lax.fori_loop(0, num_iterations, iteration_body, init_weights)
    w_expanded = final_weights[:, :, None, None]
    aggregated = jnp.sum(scale_outputs * w_expanded, axis=1)

    return aggregated, final_weights
```

### Per-Scale RoPE (primitives/rope.py)

```python
import jax.numpy as jnp
from jaxtyping import Float, Array

def compute_rope_freqs(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Float[Array, "N D2"]:
    """Compute rotary position embedding frequencies."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
    positions = jnp.arange(max_seq_len)
    freqs = jnp.outer(positions, inv_freq)
    return freqs

def apply_rope_per_scale(
    x: Float[Array, "B N H K"],
    scale_idx: int,
    compression_ratio: int,
    freqs: Float[Array, "N K2"],
) -> Float[Array, "B N H K"]:
    """
    Apply RoPE with position indices adjusted for scale.

    At scale l, position i maps to floor(i * r^l) in the original sequence.
    """
    seq_len = x.shape[1]
    scale_factor = compression_ratio ** scale_idx

    # Adjusted position indices
    positions = jnp.arange(seq_len) * scale_factor
    positions = jnp.clip(positions, 0, freqs.shape[0] - 1).astype(jnp.int32)

    # Gather frequencies for these positions
    scaled_freqs = freqs[positions]

    # Split x into pairs and apply rotation
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = jnp.cos(scaled_freqs)[None, :, None, :]
    sin = jnp.sin(scaled_freqs)[None, :, None, :]

    x_rot = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1).reshape(x.shape)

    return x_rot
```

### mHC Layer (layers/mhc.py)

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from ..primitives.sinkhorn import sinkhorn_knopp

class ManifoldHyperConnection(eqx.Module):
    """
    Manifold-constrained hyper-connection layer.

    Projects residual connection weight matrix onto Birkhoff polytope
    to ensure signal mean conservation and bounded spectral norm.
    """
    log_alpha: Float[Array, "D D"]
    layer_scale: Float[Array, "D"]
    sinkhorn_iters: int = eqx.field(static=True)

    def __init__(self, d_model: int, sinkhorn_iters: int = 10, *, key: jax.Array):
        self.sinkhorn_iters = sinkhorn_iters
        k1, k2 = jax.random.split(key)
        noise = jax.random.normal(k1, (d_model, d_model)) * 0.01
        self.log_alpha = noise
        self.layer_scale = jnp.ones(d_model)

    def __call__(
        self,
        residual: Float[Array, "B N D"],
        block_output: Float[Array, "B N D"],
    ) -> Float[Array, "B N D"]:
        H = sinkhorn_knopp(self.log_alpha, self.sinkhorn_iters)
        D = H.shape[0]
        I_minus_H = jnp.eye(D) - H

        res_contrib = jnp.einsum("de,bne->bnd", H, residual)
        block_contrib = jnp.einsum("de,bne->bnd", I_minus_H, block_output)

        return (res_contrib + block_contrib) * self.layer_scale
```

---

## Model Configuration (1-3B)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 32768        # 256 × 128 (TPU aligned)
    max_seq_len: int = 2048        # Divisible by 2^4 = 16
    d_model: int = 2048            # 16 × 128 (TPU aligned)
    num_heads: int = 16
    num_layers: int = 24           # ~1.5B params
    num_scales: int = 4            # L=4
    compression_ratio: int = 2     # r=2
    ffn_multiplier: float = 2.67   # SwiGLU
    sinkhorn_iterations: int = 10
    nash_iterations: int = 3
    aggregation: Literal["nash", "convex"] = "nash"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"

    def __post_init__(self):
        assert self.vocab_size % 128 == 0, "vocab_size must align to 128"
        assert self.max_seq_len % 128 == 0, "max_seq_len must align to 128"
        assert self.d_model % 128 == 0, "d_model must align to 128"
        assert self.d_model % self.num_heads == 0

        # Sequence alignment invariant
        min_scale_len = self.max_seq_len // (self.compression_ratio ** (self.num_scales - 1))
        assert min_scale_len >= 1, f"max_seq_len too small for {self.num_scales} scales"
        assert self.max_seq_len % (self.compression_ratio ** (self.num_scales - 1)) == 0
```

---

## Testing Strategy

### Invariant Tests (Hypothesis property-based)

```python
from hypothesis import given, strategies as st
import jax
import jax.numpy as jnp

from nash_mhc.primitives.sinkhorn import sinkhorn_knopp
from nash_mhc.types.invariants import assert_doubly_stochastic

@given(
    n=st.integers(min_value=4, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_sinkhorn_doubly_stochastic(n: int, seed: int):
    """Sinkhorn output must be doubly stochastic."""
    key = jax.random.PRNGKey(seed)
    log_alpha = jax.random.normal(key, (n, n))
    P = sinkhorn_knopp(log_alpha, num_iterations=20)
    assert_doubly_stochastic(P, rtol=1e-4, atol=1e-5)

def test_nash_weights_simplex():
    """Nash weights must be valid probability distribution."""
    key = jax.random.PRNGKey(0)
    scale_outputs = jax.random.normal(key, (2, 4, 32, 64))
    _, weights = nash_best_response(scale_outputs, num_iterations=5)
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, rtol=1e-5)
    assert jnp.all(weights >= 0)
```

---

## Reference Files

| Component | Reference Location |
|-----------|-------------------|
| Nash best-response | `references/MAHA-Project/src/layers/aggregation.py` |
| Strided conv | `references/MAHA-Project/src/layers/decomposition.py` |
| Shared V attention | `references/MAHA-Project/src/layers/attention.py` |
| Block composition | `references/MAHA-Project/src/models/maha_block.py` |
| Training loop | `references/MAHA-Project/train.py` |

---

## Data-Oriented Design Checklist

- [x] SoA over AoS: Scales as tuple, not Python list
- [x] Illegal states unrepresentable: Frozen configs with validators
- [x] Invariant-first: Assertions for doubly stochastic, simplex, spectral norm
- [x] Pure functional core: Equinox PyTree modules, stateless forward
- [x] Linear control flow: Guard clauses, no deep nesting
- [x] Hardware-first: 128-aligned dimensions, TPU MXU locality
- [x] Zero-puffery: Blunt APIs, no boolean flags
