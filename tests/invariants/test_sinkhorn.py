"""Tests for Sinkhorn-Knopp invariants."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nash_mhc.primitives.sinkhorn import sinkhorn_knopp
from nash_mhc.types.invariants import assert_doubly_stochastic


class TestSinkhornInvariants:
    """Test that Sinkhorn output satisfies doubly stochastic invariants."""

    def test_doubly_stochastic_small(self, key):
        """Output is doubly stochastic for small matrices."""
        log_alpha = jax.random.normal(key, (8, 8))
        P = sinkhorn_knopp(log_alpha, num_iterations=20)

        # Rows sum to 1
        row_sums = jnp.sum(P, axis=-1)
        assert jnp.allclose(row_sums, 1.0, rtol=1e-4)

        # Columns sum to 1
        col_sums = jnp.sum(P, axis=-2)
        assert jnp.allclose(col_sums, 1.0, rtol=1e-4)

        # All entries non-negative
        assert jnp.all(P >= -1e-6)

    def test_doubly_stochastic_larger(self, key):
        """Output is doubly stochastic for larger matrices."""
        log_alpha = jax.random.normal(key, (64, 64))
        P = sinkhorn_knopp(log_alpha, num_iterations=20)

        assert_doubly_stochastic(P, rtol=1e-3, atol=1e-5)

    def test_doubly_stochastic_batched(self, key):
        """Output is doubly stochastic for batched input."""
        log_alpha = jax.random.normal(key, (4, 16, 16))
        P = sinkhorn_knopp(log_alpha, num_iterations=15)

        # Check each batch element
        for i in range(4):
            assert_doubly_stochastic(P[i], rtol=1e-3, atol=1e-5)

    def test_iteration_convergence(self, key):
        """More iterations improve convergence."""
        log_alpha = jax.random.normal(key, (32, 32))

        P_5 = sinkhorn_knopp(log_alpha, num_iterations=5)
        P_20 = sinkhorn_knopp(log_alpha, num_iterations=20)

        # Compute error from doubly stochastic
        err_5 = jnp.max(jnp.abs(jnp.sum(P_5, axis=-1) - 1.0))
        err_20 = jnp.max(jnp.abs(jnp.sum(P_20, axis=-1) - 1.0))

        # More iterations should give lower error
        assert err_20 <= err_5

    def test_gradient_exists(self, key):
        """Sinkhorn is differentiable."""
        log_alpha = jax.random.normal(key, (8, 8))

        def loss_fn(log_alpha):
            P = sinkhorn_knopp(log_alpha, num_iterations=10)
            return jnp.sum(P ** 2)

        grad = jax.grad(loss_fn)(log_alpha)

        # Gradient should exist and be finite
        assert not jnp.any(jnp.isnan(grad))
        assert not jnp.any(jnp.isinf(grad))

    def test_preserves_dtype(self, key):
        """Output dtype matches input dtype."""
        log_alpha_f32 = jax.random.normal(key, (8, 8), dtype=jnp.float32)
        log_alpha_bf16 = log_alpha_f32.astype(jnp.bfloat16)

        P_f32 = sinkhorn_knopp(log_alpha_f32, num_iterations=10)
        P_bf16 = sinkhorn_knopp(log_alpha_bf16, num_iterations=10)

        assert P_f32.dtype == jnp.float32
        assert P_bf16.dtype == jnp.bfloat16


@settings(max_examples=20, deadline=5000)
@given(
    n=st.integers(min_value=4, max_value=32),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_sinkhorn_property_doubly_stochastic(n: int, seed: int):
    """Property test: Sinkhorn output is always doubly stochastic."""
    key = jax.random.PRNGKey(seed)
    log_alpha = jax.random.normal(key, (n, n))

    P = sinkhorn_knopp(log_alpha, num_iterations=20)

    row_sums = jnp.sum(P, axis=-1)
    col_sums = jnp.sum(P, axis=-2)

    assert jnp.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-4)
    assert jnp.allclose(col_sums, 1.0, rtol=1e-3, atol=1e-4)
    assert jnp.all(P >= -1e-5)
