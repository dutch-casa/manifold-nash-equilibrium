"""Inference script for Nash-MHC language model."""

import argparse
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from nash_mhc.data.loader import SequenceBatch
from nash_mhc.data.tokenizer import TokenizerConfig
from nash_mhc.models.backbone import MAHALanguageModel
from nash_mhc.training.checkpoint import OrbaxCheckpointManager
from nash_mhc.training.loop import init_train_state, TrainState
from nash_mhc.types.configs import TPU_V6E_MEDIUM_CONFIG, TPU_V6E_MEDIUM_TRAINING_CONFIG


def load_model(checkpoint_dir: str, model_config, training_config):
    """Load model from checkpoint."""
    checkpoint_manager = OrbaxCheckpointManager(checkpoint_dir)

    print(f"Initializing model from {checkpoint_dir}...")
    key = jax.random.PRNGKey(42)
    _, model_key = jax.random.split(key)

    model = MAHALanguageModel(model_config, key=model_key)
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,}")

    state = init_train_state(model, training_config, pad_token_id=0)

    restored_state = checkpoint_manager.restore(state)
    print(f"Loaded checkpoint from step {restored_state.step}")

    return restored_state, model


@jax.jit
def generate(state: TrainState, input_ids: jnp.ndarray, config, max_tokens: int = 100):
    """Generate text autoregressively."""
    batch_size, seq_len = input_ids.shape

    generated = input_ids

    for _ in range(max_tokens):
        model = state.model
        inputs = generated[:, -config.max_seq_len :]

        logits, _ = model(inputs, causal=False)

        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]

        generated = jnp.concatenate([generated, next_token], axis=1)

        if jnp.all(next_token == 0):
            break

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Nash-MHC inference")
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/smollm-360M",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    print(f"Loading checkpoint from: {args.checkpoint_dir}")
    state, model = load_model(
        args.checkpoint_dir, TPU_V6E_MEDIUM_CONFIG, TPU_V6E_MEDIUM_TRAINING_CONFIG
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens...")
    print("-" * 50)

    input_ids = jnp.array([[hf_tokenizer.encode(args.prompt)]])

    generated = generate(state, input_ids, TPU_V6E_MEDIUM_CONFIG, args.max_tokens)

    output_tokens = generated[0].tolist()
    output_text = hf_tokenizer.decode(output_tokens)

    print("\n" + "=" * 50)
    print("Generated:")
    print("=" * 50)
    print(output_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
