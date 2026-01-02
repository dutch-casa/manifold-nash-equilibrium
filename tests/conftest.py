"""Pytest configuration and fixtures."""

import pytest
import jax


@pytest.fixture
def key():
    """Provide a PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_config():
    """Provide a small model config for testing."""
    from nash_mhc.types.configs import SMALL_MODEL_CONFIG
    return SMALL_MODEL_CONFIG
