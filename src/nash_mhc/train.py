"""Training entry point for Nash-MHC language model."""

import argparse
import time
from dataclasses import replace
from typing import Iterator

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from nash_mhc.data.loader import LoaderConfig
from nash_mhc.data.streaming import StreamingConfig, create_streaming_dataloader
from nash_mhc.data.tokenizer import TokenizerAdapter, TokenizerConfig
from nash_mhc.models.backbone import MAHALanguageModel
from nash_mhc.training.loop import init_train_state, train_step
from nash_mhc.training.checkpoint import OrbaxCheckpointManager
from nash_mhc.types.configs import (
    DEFAULT_MODEL_CONFIG,
    SINGLE_TPU_TRAINING_CONFIG,
    ModelConfig,
    TrainingConfig,
)


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
    return replace(DEFAULT_MODEL_CONFIG, vocab_size=vocab_size, max_seq_len=max_seq_len)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    base = SINGLE_TPU_TRAINING_CONFIG
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

    device = jax.devices()[0]
    jax.config.update("jax_default_device", device)

    print(f"JAX process {jax.process_index()} / {jax.process_count()} started.")
    print(f"Device: {device}")

    memory_stats = device.memory_stats()
    gb_used = memory_stats.get("bytes_in_use", 0) / (1024**3)
    gb_peak = memory_stats.get("peak_bytes_in_use", 0) / (1024**3)
    print(f"TPU HBM: {gb_used:.2f} GB used / {gb_peak:.2f} GB peak")

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    print(f"Loading tokenizer: {args.tokenizer}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    num_scales = 3
    compression_ratio = 2
    raw_vocab = len(hf_tokenizer)
    aligned_vocab = round_vocab_size(raw_vocab)
    aligned_seq_len = round_seq_len(
        hf_tokenizer.model_max_length, num_scales, compression_ratio
    )
    aligned_seq_len = min(aligned_seq_len, 2048)

    tokenizer_config = TokenizerConfig(
        max_length=aligned_seq_len, pad_id=hf_tokenizer.pad_token_id
    )
    tokenizer = TokenizerAdapter(hf_tokenizer, tokenizer_config)

    model_config = build_model_config(args, aligned_vocab, max_seq_len=aligned_seq_len)
    training_config = build_training_config(args)

    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")
    print("Creating data loader...")

    stream_config = StreamingConfig(
        path=args.dataset,
        name=args.dataset_name,
        split=args.split,
        text_field="text",
        num_workers=args.num_workers,
        prefetch_buffer_size=args.worker_buffer_size,
    )
    loader = create_streaming_dataloader(
        stream_config,
        LoaderConfig(
            batch_size=training_config.batch_size,
            shuffle_seed=training_config.seed,
        ),
        tokenizer,
    )

    print("Initializing model...")
    model = MAHALanguageModel(model_config, key=model_key)
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,}")
    print(f"Model size (bfloat16): ~{param_count * 2 / (1024**3):.2f} GB")
    print(
        f"Batch size: {training_config.batch_size} | Sequence length: {model_config.max_seq_len}"
    )
    print(
        f"Expected activation memory: ~{training_config.batch_size * model_config.max_seq_len * model_config.d_model * 4 / (1024**3):.2f} GB"
    )

    print("Initializing train state...")
    state = init_train_state(
        model, training_config, pad_token_id=hf_tokenizer.pad_token_id
    )

    checkpoint_manager = OrbaxCheckpointManager(args.checkpoint_dir)

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        state = checkpoint_manager.restore(state)

    print("Compiling train step...")
    train_step_jit = jax.jit(train_step, static_argnames=("optimizer",))

    print("Starting training...")
    print(f"Total steps: {training_config.total_steps}")
    print(f"Effective batch size: {training_config.effective_batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")

    step_times = []
    start_time = time.time()

    for step_count, batch in enumerate(loader, start=1):
        step_start = time.time()

        state, loss_components = train_step_jit(state, batch)

        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 100:
            step_times.pop(0)

        avg_step_time = sum(step_times) / len(step_times)
        throughput = training_config.batch_size / avg_step_time

        current_lr = training_config.learning_rate
        if training_config.warmup_steps > 0:
            if step_count <= training_config.warmup_steps:
                current_lr = current_lr * (step_count / training_config.warmup_steps)
            else:
                decay_steps = training_config.total_steps - training_config.warmup_steps
                progress = (step_count - training_config.warmup_steps) / decay_steps
                current_lr = current_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))

        if step_count % args.log_interval == 0:
            print(
                f"Step {step_count}/{training_config.total_steps} | "
                f"loss: {float(loss_components.total):.4f} | "
                f"lr: {float(current_lr):.2e} | "
                f"throughput: {throughput:.1f} samples/s | "
                f"step_time: {avg_step_time:.3f}s"
            )

        if step_count % args.checkpoint_interval == 0:
            print(f"Saving checkpoint at step {step_count}...")
            checkpoint_manager.save(
                state,
                metrics={
                    "loss": float(loss_components.total),
                    "learning_rate": float(current_lr),
                    "throughput": float(throughput),
                },
            )

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    checkpoint_manager.close()


if __name__ == "__main__":
    main()
