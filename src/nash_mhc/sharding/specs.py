"""PartitionSpec helpers aligned with TPU mesh axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from jax.sharding import PartitionSpec as PS


Axis = str


@dataclass(frozen=True, slots=True)
class SpecLayout:
    """Canonical PartitionSpecs for MAHA parameters and activations."""

    data_axis: Axis = "data"
    fsdp_axis: Axis = "fsdp"
    tp_axis: Axis = "tp"

    def embeddings(self) -> PS:
        """Embedding tables replicated across data, sharded over fsdp×tp."""
        return PS((self.fsdp_axis, self.tp_axis), None)

    def qkv_projection(self) -> PS:
        """Attention projections shard along fsdp (rows) and tp (cols)."""
        return PS(self.fsdp_axis, self.tp_axis)

    def attn_output(self) -> PS:
        """Attention output projection shares tp axis on columns."""
        return PS(self.fsdp_axis, self.tp_axis)

    def ffn_up(self) -> PS:
        return PS(self.fsdp_axis, self.tp_axis)

    def ffn_down(self) -> PS:
        return PS(self.fsdp_axis, None)

    def layer_norm(self) -> PS:
        return PS(self.fsdp_axis, None)

    def activations(self) -> PS:
        """Runtime activations are sharded across data and tensor axes."""
        return PS(self.data_axis, None, self.tp_axis)

    def logits(self) -> PS:
        return PS(self.data_axis, None, self.tp_axis)


def parameter_spec_from_name(param_name: str, layout: SpecLayout | None = None) -> PS:
    """Heuristic PartitionSpec assignment based on parameter name."""
    layout = layout or SpecLayout()
    name = param_name.lower()

    if "embedding" in name:
        return layout.embeddings()
    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
        return layout.qkv_projection()
    if "o_proj" in name:
        return layout.attn_output()
    if "ffn" in name or "w_gate" in name or "w_up" in name:
        return layout.ffn_up()
    if "w_down" in name:
        return layout.ffn_down()
    if "norm" in name or "rms" in name:
        return layout.layer_norm()
    return PS()


def activation_specs(layout: SpecLayout | None = None) -> Mapping[str, PS]:
    """PartitionSpec mapping for common runtime tensors."""
    layout = layout or SpecLayout()
    return {
        "hidden": layout.activations(),
        "attention_context": layout.activations(),
        "logits": layout.logits(),
    }

