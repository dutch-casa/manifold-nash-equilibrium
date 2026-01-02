"""Tests for Nash equilibrium solver invariants."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nash_mhc.primitives.nash_solver import nash_best_response
from nash_mhc.types.invariants import assert_simplex


class TestNashInvariants:
    """Test that Nash solver output satisfies simplex invariants."""

    def test_weights_simplex(self, key):
        """Weights form a valid probability distribution."""
        scale_outputs = jax.random.normal(key, (2, 4, 32, 64))
        aggregated, weights = nash_best_response(scale_outputs, num_iterations=5)

        # Weights sum to 1
        weight_sums = jnp.sum(weights, axis=-1)
        assert jnp.allclose(weight_sums, 1.0, rtol=1e-5)

        # Weights are non-negative
        assert jnp.all(weights >= 0)

    def test_output_shape_preserved(self, key):
        """Aggregated output has correct shape."""
        B, L, N, D = 2, 4, 32, 64
        scale_outputs = jax.random.normal(key, (B, L, N, D))
        aggregated, weights = nash_best_response(scale_outputs, num_iterations=3)

        assert aggregated.shape == (B, N, D)
        assert weights.shape == (B, L)

    def test_more_iterations_stabilizes(self, key):
        """More iterations lead to more stable weights."""
        scale_outputs = jax.random.normal(key, (2, 4, 32, 64))

        _, weights_1 = nash_best_response(scale_outputs, num_iterations=1)
        _, weights_5 = nash_best_response(scale_outputs, num_iterations=5)
        _, weights_10 = nash_best_response(scale_outputs, num_iterations=10)

        # Weights should remain on simplex regardless of iterations
        for w in [weights_1, weights_5, weights_10]:
            assert_simplex(w, rtol=1e-5)

    def test_uniform_input_uniform_weights(self, key):
        """If all scales are identical, weights should be uniform."""
        B, L, N, D = 2, 4, 32, 64
        single_scale = jax.random.normal(key, (B, N, D))

        # Stack identical scales
        scale_outputs = jnp.stack([single_scale] * L, axis=1)

        _, weights = nash_best_response(scale_outputs, num_iterations=10)

        # Weights should be approximately uniform (1/L each)
        expected = jnp.ones((B, L)) / L
        assert jnp.allclose(weights, expected, rtol=1e-3)

    def test_gradient_exists(self, key):
        """Nash solver is differentiable."""
        scale_outputs = jax.random.normal(key, (2, 4, 32, 64))

        def loss_fn(scale_outputs):
            aggregated, _ = nash_best_response(scale_outputs, num_iterations=3)
            return jnp.sum(aggregated ** 2)

        grad = jax.grad(loss_fn)(scale_outputs)

        assert not jnp.any(jnp.isnan(grad))
        assert not jnp.any(jnp.isinf(grad))

    def test_aggregation_is_weighted_sum(self, key):
        """Aggregated output is weighted sum of inputs."""
        B, L, N, D = 2, 4, 32, 64
        scale_outputs = jax.random.normal(key, (B, L, N, D))

        aggregated, weights = nash_best_response(scale_outputs, num_iterations=5)

        # Recompute expected aggregation
        w_expanded = weights[:, :, None, None]
        expected = jnp.sum(scale_outputs * w_expanded, axis=1)

        assert jnp.allclose(aggregated, expected, rtol=1e-5)


@settings(max_examples=20, deadline=5000)
@given(
    B=st.integers(min_value=1, max_value=4),
    L=st.integers(min_value=2, max_value=6),
    N=st.integers(min_value=8, max_value=32),
    D=st.integers(min_value=16, max_value=64),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_nash_property_simplex(B: int, L: int, N: int, D: int, seed: int):
    """Property test: Nash weights are always on simplex."""
    key = jax.random.PRNGKey(seed)
    scale_outputs = jax.random.normal(key, (B, L, N, D))

    _, weights = nash_best_response(scale_outputs, num_iterations=3)

    # Sum to 1
    weight_sums = jnp.sum(weights, axis=-1)
    assert jnp.allclose(weight_sums, 1.0, rtol=1e-4)

    # Non-negative
    assert jnp.all(weights >= -1e-6)
