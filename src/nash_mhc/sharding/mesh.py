"""TPU mesh helpers and sharding utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding


AxisNames = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Declarative TPU mesh description."""

    axis_lengths: tuple[int, ...]
    axis_names: AxisNames = ("data", "fsdp", "tp")

    def __post_init__(self) -> None:
        if len(self.axis_lengths) != len(self.axis_names):
            raise ValueError(
                f"axis_lengths ({self.axis_lengths}) and axis_names "
                f"({self.axis_names}) must have equal rank"
            )
        if any(length <= 0 for length in self.axis_lengths):
            raise ValueError(f"All axis lengths must be positive, got {self.axis_lengths}")

    @property
    def total_devices(self) -> int:
        """Total devices implied by the mesh."""
        prod = 1
        for length in self.axis_lengths:
            prod *= length
        return prod


def _select_devices(num_devices: int) -> Sequence[jax.Device]:
    """Select the first `num_devices` devices from the global pool."""
    available = jax.devices()
    if num_devices > len(available):
        raise ValueError(
            f"Requested {num_devices} devices for mesh, only {len(available)} available"
        )
    return available[:num_devices]


def create_mesh(config: MeshConfig) -> Mesh:
    """Instantiate a `Mesh` aligned with the provided configuration."""
    devices = _select_devices(config.total_devices)
    mesh_array = mesh_utils.create_device_mesh(config.axis_lengths, devices=devices)
    return Mesh(mesh_array, config.axis_names)


@contextmanager
def mesh_context(config: MeshConfig) -> Iterable[Mesh]:
    """Context manager that installs a mesh as the default sharding scope."""
    mesh = create_mesh(config)
    with mesh:
        yield mesh


def with_named_sharding(mesh: Mesh, partition_spec) -> NamedSharding:
    """Create `NamedSharding` for a given mesh and `PartitionSpec`."""
    return NamedSharding(mesh, partition_spec)

