"""Equinox-aware sharding utilities for Nash-MHC models."""

from __future__ import annotations

from typing import Any, TypeVar

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

from .specs import SpecLayout, parameter_spec_from_name


T = TypeVar("T")


def make_shard_spec_tree(model: eqx.Module, layout: SpecLayout | None = None) -> Any:
    """Creates a tree of PartitionSpecs matching the model's array structure."""
    layout = layout or SpecLayout()
    params, _ = eqx.partition(model, eqx.is_array)

    def _get_spec(path, val):
        path_names = []
        for p in path:
            if hasattr(p, "name"):
                path_names.append(str(p.name))
            elif hasattr(p, "key"):
                path_names.append(str(p.key))
            else:
                # Fallback for other key types
                s = str(p)
                # Strip key type wrappers if possible
                if "(" in s and ")" in s:
                    s = s.split("(")[-1].split(")")[0].split("=")[-1].strip("'")
                path_names.append(s)
        
        name = ".".join(path_names)
        return parameter_spec_from_name(name, layout)

    return jax.tree_util.tree_map_with_path(_get_spec, params)


def shard_model(model: eqx.Module, mesh: Mesh, layout: SpecLayout | None = None) -> eqx.Module:
    """Shards model parameters across the provided mesh using NamedSharding."""
    params, static = eqx.partition(model, eqx.is_array)
    spec_tree = make_shard_spec_tree(params, layout)
    
    sharding_tree = jax.tree_util.tree_map(
        lambda s: NamedSharding(mesh, s), spec_tree
    )
    
    sharded_params = jax.device_put(params, sharding_tree)
    return eqx.combine(sharded_params, static)


def shard_input(batch: T, mesh: Mesh, axis_name: str = "data") -> T:
    """Shards input batch along the leading dimension using the specified mesh axis."""
    def _shard_array(x):
        if not isinstance(x, jax.Array):
            return x
        spec = PS(axis_name, *(None for _ in range(x.ndim - 1)))
        return jax.device_put(x, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map(_shard_array, batch)


def shard_train_state(state: Any, mesh: Mesh, layout: SpecLayout | None = None) -> Any:
    """Shards a TrainState across the mesh, including model and optimizer state."""
    from ..training.loop import TrainState
    
    if not isinstance(state, TrainState):
        raise TypeError(f"Expected TrainState, got {type(state)}")

    params, static = eqx.partition(state.model, eqx.is_array)
    param_specs = make_shard_spec_tree(params, layout)
    
    sharding_tree = jax.tree_util.tree_map(
        lambda s: NamedSharding(mesh, s), param_specs
    )
    sharded_params = jax.device_put(params, sharding_tree)
    new_model = eqx.combine(sharded_params, static)

    def _map_optimizer_state(opt_state, specs):
        if isinstance(opt_state, (list, tuple)):
            # Check if it's a namedtuple
            if hasattr(opt_state, "_asdict"):
                d = opt_state._asdict()
                new_d = {k: _map_optimizer_state(v, specs) if k in ("mu", "nu") else v for k, v in d.items()}
                if "count" in new_d:
                    new_d["count"] = jax.device_put(new_d["count"], NamedSharding(mesh, PS()))
                return type(opt_state)(**new_d)
            return type(opt_state)(_map_optimizer_state(s, specs) for s in opt_state)
        
        try:
            return jax.tree_util.tree_map(
                lambda s, x: jax.device_put(x, NamedSharding(mesh, s)) if isinstance(x, jax.Array) else x,
                specs, opt_state
            )
        except (ValueError, TypeError):
            return jax.tree_util.tree_map(
                lambda x: jax.device_put(x, NamedSharding(mesh, PS())) if isinstance(x, jax.Array) else x,
                opt_state
            )

    sharded_opt_state = _map_optimizer_state(state.opt_state, param_specs)
    sharded_step = jax.device_put(state.step, NamedSharding(mesh, PS()))

    return TrainState(
        model=new_model,
        optimizer=state.optimizer,
        opt_state=sharded_opt_state,
        step=sharded_step,
        pad_token_id=state.pad_token_id
    )
