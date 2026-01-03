"""TPU mesh helpers and sharding utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS


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
            raise ValueError(
                f"All axis lengths must be positive, got {self.axis_lengths}"
            )

    @property
    def total_devices(self) -> int:
        """Total devices implied by mesh."""
        prod = 1
        for length in self.axis_lengths:
            prod *= length
        return prod


def create_mesh(config: MeshConfig) -> Mesh:
    """Instantiate a Mesh aligned with provided configuration."""
    return jax.make_mesh(config.axis_lengths, config.axis_names)


@contextmanager
def mesh_context(config: MeshConfig) -> Iterator[Mesh]:
    """Context manager that installs a mesh as default sharding scope."""
    mesh = create_mesh(config)
    with mesh:
        yield mesh


def with_named_sharding(mesh: Mesh, partition_spec: PS) -> NamedSharding:
    """Create NamedSharding for a given mesh and PartitionSpec."""
    return NamedSharding(mesh, partition_spec)
