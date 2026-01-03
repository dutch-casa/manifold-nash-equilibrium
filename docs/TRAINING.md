# Training Nash-MHC

Guide for training the 3.38B parameter Nash-MHC model on JAX/TPU v6e-1.

## Quick Start

### 1. Requirements
Ensure JAX with TPU support and all dependencies are installed:
```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install equinox flax optax lineax jaxtyping beartype
pip install orbax-checkpoint clu wandb datasets grain-nightly transformers
```

### 2. Google Colab Setup
The optimized training notebook is available here:
[train_3b_tpu.ipynb](./notebooks/train_3b_tpu.ipynb)

**Note:** For TPU v6e-1, use the "TPU v6e" runtime in Google Colab.

## Training Configuration

### Model Config (`LARGE_3B_CONFIG`)
- **Parameters**: 3.38 Billion
- **Layers**: 28
- **D_Model**: 2048
- **Heads**: 16
- **Scales**: 4 (MHC hierarchy)
- **Vocab Size**: 65,536 (aligned to 128)
- **Max Seq Len**: 4,096

### Training Config (`LARGE_3B_TRAINING_CONFIG`)
- **Base Batch Size**: 32 (per step)
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 256
- **Learning Rate**: 3e-4
- **Warmup**: 2,000 steps
- **Total Steps**: 100,000
- **Weight Decay**: 0.1
- **Precision**: `bfloat16` (Mixed Precision)

## Data Setup
The pipeline uses streaming data from HuggingFace. No local pre-download required.

- **Corpus**: `HuggingFaceTB/smollm-corpus`
- **Config**: `cosmopedia-v2`
- **Tokenizer**: `HuggingFaceTB/smollm-360M` (or any compatible HF tokenizer)

## Execution

### Full 3B Training
Run the training script with TPU sharding (optimized for 8 cores):
```bash
python -m nash_mhc.train \
  --dataset HuggingFaceTB/smollm-corpus \
  --dataset-name cosmopedia-v2 \
  --tokenizer HuggingFaceTB/smollm-360M \
  --batch-size 32 \
  --mesh-data 2 --mesh-fsdp 2 --mesh-tp 2 \
  --checkpoint-dir ./checkpoints \
  --wandb-project nash-mhc-3b
```

### Small Model (Smoke Test)
To verify the pipeline on a single device or CPU:
```bash
python -m nash_mhc.train \
  --use-small-model \
  --total-steps 100 \
  --batch-size 8
```

## Important Flags
| Flag | Description | Default |
|------|-------------|---------|
| `--mesh-data` | Data parallelism axis size | 1 |
| `--mesh-fsdp` | Fully Sharded Data Parallel axis size | 1 |
| `--mesh-tp` | Tensor parallelism axis size | 1 |
| `--checkpoint-interval` | Steps between saves | 1000 |
| `--resume-from` | Path to restore checkpoint | None |
| `--num-workers` | Data loading workers | 8 |

## Monitoring
Metrics are logged via **W&B** and **CLU**.
- **Loss**: Training loss and components (total, sparsity).
- **Throughput**: Tokens per second.
- **Checkpoints**: Saved via Orbax with async support.

## TPU Setup Resources
- [Google Cloud TPU v6e Guide](https://cloud.google.com/tpu/docs/v6e-setup)
- [JAX TPU Documentation](https://jax.readthedocs.io/en/latest/tpu_quickstart.html)
