from __future__ import annotations

import jax
from jax.sharding import Mesh
import pytest

from nash_mhc.sharding.mesh import create_adaptive_mesh, MeshConfig


def test_mesh_config_validation():
    """Test MeshConfig validates axis lengths and names."""

    config = MeshConfig(axis_lengths=(2, 2, 2), axis_names=("data", "fsdp", "tp"))
    assert config.total_devices == 8

    with pytest.raises(ValueError, match="must have equal rank"):
        MeshConfig(axis_lengths=(2, 2), axis_names=("data", "fsdp", "tp"))

    with pytest.raises(ValueError, match="must be positive"):
        MeshConfig(axis_lengths=(0, 2, 2), axis_names=("data", "fsdp", "tp"))


def test_create_adaptive_mesh_single_device():
    """Test adaptive mesh on single device."""
    with jax.default_device(jax.devices("cpu")[0]):
        mesh = create_adaptive_mesh((1, 1, 1), ("data", "fsdp", "tp"))
        assert isinstance(mesh, Mesh)
        assert mesh.size == 1


def test_create_adaptive_mesh_validation():
    """Test adaptive mesh validates against available devices."""
    available = jax.device_count()

    mesh = create_adaptive_mesh((1, 1, 1), ("data", "fsdp", "tp"))
    assert isinstance(mesh, Mesh)

    requested = (1, 1, available + 1)
    with pytest.raises(ValueError, match="exceeds available devices"):
        create_adaptive_mesh(requested, ("data", "fsdp", "tp"))

    with pytest.raises(ValueError, match="must have 3 dimensions"):
        create_adaptive_mesh((1, 1), ("data", "fsdp", "tp"))
