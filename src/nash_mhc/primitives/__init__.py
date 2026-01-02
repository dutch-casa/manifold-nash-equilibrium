"""Hardware-aligned primitive operations."""

from nash_mhc.primitives.sinkhorn import sinkhorn_knopp, sinkhorn_knopp_regularized
from nash_mhc.primitives.nash_solver import nash_best_response, convex_aggregation
from nash_mhc.primitives.upsample import nearest_upsample, upsample_scale_outputs
from nash_mhc.primitives.rope import compute_rope_freqs, apply_rope, apply_rope_per_scale
from nash_mhc.primitives.strided_conv import StridedConv1d, create_downsampler_stack

__all__ = [
    "sinkhorn_knopp",
    "sinkhorn_knopp_regularized",
    "nash_best_response",
    "convex_aggregation",
    "nearest_upsample",
    "upsample_scale_outputs",
    "compute_rope_freqs",
    "apply_rope",
    "apply_rope_per_scale",
    "StridedConv1d",
    "create_downsampler_stack",
]
