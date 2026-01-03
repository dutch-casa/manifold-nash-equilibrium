"""Training entry point for Nash-MHC language model."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Iterator

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from flax import struct
from clu import metrics as clu_metrics

from nash_mhc.data.loader import LoaderConfig
from nash_mhc.data.streaming import StreamingConfig, create_streaming_dataloader
from nash_mhc.data.tokenizer import TokenizerAdapter, TokenizerConfig
from nash_mhc.models.backbone import MAHALanguageModel
from nash_mhc.sharding.mesh import create_adaptive_mesh, mesh_context
from nash_mhc.sharding.shard import shard_train_state, shard_input
from nash_mhc.training.loop import init_train_state, train_step
from nash_mhc.training.checkpoint import OrbaxCheckpointManager
from nash_mhc.types.configs import (
    LARGE_3B_CONFIG,
    LARGE_3B_TRAINING_CONFIG,
    MEDIUM_1_5B_CONFIG,
    ModelConfig,
    TrainingConfig,
)


@struct.dataclass
class TrainMetrics(clu_metrics.Collection):
    loss: clu_metrics.Average.from_output("loss")  # type: ignore[misc]
    learning_rate: clu_metrics.LastValue.from_output("learning_rate")  # type: ignore[misc]
    throughput: clu_metrics.Average.from_output("throughput")  # type: ignore[misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nash-MHC language model")

    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceTB/smollm-corpus",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cosmopedia-v2",
        help="Dataset config name",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/smollm-360M",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None, help="Override total training steps"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=None, help="Override warmup steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mesh-data", type=int, default=1, help="Data parallelism axis size"
    )
    parser.add_argument("--mesh-fsdp", type=int, default=1, help="FSDP axis size")
    parser.add_argument(
        "--mesh-tp", type=int, default=1, help="Tensor parallelism axis size"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Steps between logs"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )

    parser.add_argument(
        "--use-small-model",
        action="store_true",
        help="Use small config for testing (0.09B)",
    )
    parser.add_argument(
        "--use-3b-model", action="store_true", help="Use 3B config (3.38B)"
    )
    parser.add_argument(
        "--no-shard",
        action="store_true",
        help="Disable parameter sharding (for single-device training)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Data loading workers"
    )
    parser.add_argument(
        "--worker-buffer-size", type=int, default=16, help="Prefetch buffer per worker"
    )

    return parser.parse_args()


def build_model_config(
    args: argparse.Namespace, vocab_size: int, max_seq_len: int
) -> ModelConfig:
    if args.use_3b_model:
        return replace(LARGE_3B_CONFIG, vocab_size=vocab_size, max_seq_len=max_seq_len)
    if args.use_small_model:
        return ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=512,
            num_heads=8,
            num_layers=6,
            num_scales=3,
            compression_ratio=2,
            ffn_multiplier=2.67,
            sinkhorn_iterations=10,
            nash_iterations=3,
            aggregation="nash",
            dtype="float32",
        )
    return replace(MEDIUM_1_5B_CONFIG, vocab_size=vocab_size, max_seq_len=max_seq_len)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    base = LARGE_3B_TRAINING_CONFIG
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    overrides["seed"] = args.seed
    return replace(base, **overrides) if overrides else base


def round_vocab_size(size: int, alignment: int = 128) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def round_seq_len(
    length: int, num_scales: int, compression_ratio: int, alignment: int = 128
) -> int:
    scale_factor = compression_ratio ** (num_scales - 1)
    lcm = alignment * scale_factor
    return ((length + lcm - 1) // lcm) * lcm


def main() -> None:
    args = parse_args()

    print(f"JAX process {jax.process_index()} / {jax.process_count()} started.")
    print(f"Devices: {jax.devices()}")

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    print(f"Loading tokenizer: {args.tokenizer}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

        num_scales = 3 if args.use_small_model else 4
        if args.use_3b_model:
            num_scales = 4
        compression_ratio = 2
    raw_vocab = len(hf_tokenizer)
    aligned_vocab = round_vocab_size(raw_vocab)
    aligned_seq_len = round_seq_len(
        hf_tokenizer.model_max_length, num_scales, compression_ratio
    )
    aligned_seq_len = min(aligned_seq_len, 4096)

    tokenizer_config = TokenizerConfig(
        max_length=aligned_seq_len, pad_id=hf_tokenizer.pad_token_id
    )
    tokenizer = TokenizerAdapter(hf_tokenizer, tokenizer_config)

    model_config = build_model_config(args, aligned_vocab, aligned_seq_len)
    training_config = build_training_config(args)

    print(f"Model: {model_config}")
    if not (args.use_small_model or args.use_3b_model):\n        model_type = "1.5B (default)" if not (args.use_small_model or args.use_3b_model) else ("0.09B" if args.use_small_model else "3B" if args.use_3b_model\n    print("Model: " + model_type)\n    print("Model config: " + str(model_config))\n    print("Training: " + str(training_config))\n    print("Creating data loader...")\n    stream_config = StreamingConfig(\n        path=args.dataset,\n        name=args.dataset_name,\n        split=args.split,\n        text_field="text",\n        num_workers=args.num_workers,\n        prefetch_buffer_size=args.worker_buffer_size,\n    )\n    loader = create_streaming_dataloader(\n        stream_config,\n        LoaderConfig(\n            batch_size=training_config.batch_size,\n            shuffle_seed=training_config.seed,\n        ),\n        tokenizer,\n    )\n