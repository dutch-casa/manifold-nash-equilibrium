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
    from jax.sharding import AxisType

    return jax.make_mesh(
        config.axis_lengths,
        config.axis_names,
        axis_types=(AxisType.Explicit,) * len(config.axis_names),
    )


def create_adaptive_mesh(
    requested: tuple[int, int, int] = (1, 1, 1),
    axis_names: AxisNames = ("data", "fsdp", "tp"),
) -> Mesh:
    """Create mesh that fits within available devices.

    Validates requested mesh against available devices and returns a mesh
    that can be created. Falls back to 1×1×1 for single device.

    Args:
        requested: Tuple of (data, fsdp, tp) axis lengths
        axis_names: Tuple of axis names

    Returns:
        JAX Mesh object

    Raises:
        ValueError: If requested mesh exceeds available devices
    """
    from jax.sharding import AxisType

    available = jax.device_count()

    if len(requested) != 3:
        raise ValueError(f"Requested mesh must have 3 dimensions, got {len(requested)}")

    if len(axis_names) != 3:
        raise ValueError(f"Axis names must have 3 elements, got {len(axis_names)}")

    requested_total = requested[0] * requested[1] * requested[2]

    if requested_total > available:
        raise ValueError(
            f"Requested mesh ({requested[0]}×{requested[1]}×{requested[2]} = "
            f"{requested_total} devices) exceeds available devices ({available})"
        )

    if available == 1:
        return jax.make_mesh((1, 1, 1), axis_names, axis_types=(AxisType.Explicit,) * 3)

    if requested_total == available:
        return jax.make_mesh(requested, axis_names, axis_types=(AxisType.Explicit,) * 3)

    return jax.make_mesh(requested, axis_names, axis_types=(AxisType.Explicit,) * 3)


@contextmanager
def mesh_context(config: MeshConfig) -> Iterator[Mesh]:
    """Context manager that installs a mesh as default sharding scope."""
    mesh = create_mesh(config)
    with mesh:
        yield mesh


def with_named_sharding(mesh: Mesh, partition_spec: PS) -> NamedSharding:
    """Create NamedSharding for a given mesh and PartitionSpec."""
    return NamedSharding(mesh, partition_spec)
