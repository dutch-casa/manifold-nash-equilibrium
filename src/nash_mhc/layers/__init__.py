"""Equinox layer modules for the Deep-Equilibrium engine."""

from nash_mhc.layers.mhc import ManifoldHyperConnection
from nash_mhc.layers.decomposition import HierarchicalDecomposition
from nash_mhc.layers.attention import MultiscaleAttention
from nash_mhc.layers.aggregation import OptimizationAggregation
from nash_mhc.layers.ffn import SwiGLUFFN, RMSNorm

__all__ = [
    "ManifoldHyperConnection",
    "HierarchicalDecomposition",
    "MultiscaleAttention",
    "OptimizationAggregation",
    "SwiGLUFFN",
    "RMSNorm",
]
