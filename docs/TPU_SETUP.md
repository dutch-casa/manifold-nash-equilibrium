# TPU v6e-1 Setup Guide

Guide for setting up and configuring Google Cloud TPU v6e-1 for training Nash-MHC.

## Overview

**TPU v6e-1** is Google's latest TPU generation optimized for large language model training:
- **8 cores** per v6e-1 pod slice
- **High-bandwidth memory** (HBM) for 3B+ parameter models
- **JAX integration** via `jax[tpu]` package

## Prerequisites

### 1. Google Cloud Account
- GCP account with billing enabled
- [GCP Project](https://console.cloud.google.com/projectcreate) created
- Quota request for TPU v6e-1 (may need approval)

### 2. Installation
Local development environment or Google Colab with TPU runtime:

```bash
# JAX with TPU support
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U jaxlib

# Nash-MHC dependencies
pip install equinox optax orbax-checkpoint clu
pip install jaxtyping beartype
pip install datasets grain-nightly transformers
```

## Colab Setup (Recommended)

### 1. Open Notebook
Use [notebooks/train_3b_tpu.ipynb](../notebooks/train_3b_tpu.ipynb)

### 2. Select TPU Runtime
- **Runtime** → **Change runtime type**
- Select **TPU v6e** (not just "TPU")

### 3. Verify TPU Detection
```python
import jax
print(f"Devices: {jax.devices()}")
# Should show: 8 TPU devices
```

### 4. Training Command
Use 3D mesh sharding for optimal throughput:

```bash
python -m nash_mhc.train \
  --dataset HuggingFaceTB/smollm-corpus \
  --dataset-name cosmopedia-v2 \
   --tokenizer HuggingFaceTB/smollm-360M \
   --batch-size 32 \
   --mesh-data 2 --mesh-fsdp 2 --mesh-tp 2 \
   --checkpoint-dir ./checkpoints
   ```

**Mesh Configuration**:
- `--mesh-data 2`: Data parallelism (batch dimension)
- `--mesh-fsdp 2`: Fully Sharded Data Parallelism (model sharding)
- `--mesh-tp 2`: Tensor parallelism (attention head sharding)
- **Total**: 2 × 2 × 2 = 8 devices (full v6e-1)

**Note**: The training script automatically validates mesh configuration against available devices. For single-device testing (e.g., GPU), use `--mesh-data 1 --mesh-fsdp 1 --mesh-tp 1` or omit the flags (defaults to 1×1×1).

## GCP Cloud TPU Setup

### 1. Create TPU VM

```bash
# Set project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Create v6e-1 TPU
gcloud compute tpus tpu-vm create nash-mhc-tpu \
  --zone us-central2-b \
  --accelerator-type v6e-1 \
  --version v2-alpha \
  --tpu-size 2x2  # 8 cores
```

### 2. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh nash-mhc-tpu \
  --zone us-central2-b \
  --worker 0
```

### 3. Install Dependencies on TPU

```bash
# Clone repository
git clone https://github.com/dutchcaz/nash-mhc.git
cd nash-mhc

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install JAX [TPU]
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ".[tpu,dev]"
```

### 4. Run Training

```bash
# Single-device (debugging) - mesh auto-configured to 1×1×1
python -m nash_mhc.train \
  --use-small-model \
  --total-steps 100 \
  --batch-size 8

# Full 8-device training (multislice)
python -m nash_mhc.train \
  --dataset HuggingFaceTB/smollm-corpus \
  --dataset-name cosmopedia-v2 \
  --tokenizer HuggingFaceTB/smollm-360M \
  --batch-size 32 \
  --mesh-data 2 --mesh-fsdp 2 --mesh-tp 2 \
  --total-steps 100000
  ```

## Performance Tuning

### Batch Size Tuning

**OOM Error**: Reduce batch size
```bash
--batch-size 16  # from 32
```

**Low Utilization**: Increase batch size (if memory allows)
```bash
--batch-size 64  # if memory permits
```

### Mesh Configuration

**Default**: `--mesh-data 2 --mesh-fsdp 2 --mesh-tp 2` (balanced)

**For larger models** (>10B parameters):
```bash
--mesh-data 1 --mesh-fsdp 4 --mesh-tp 2  # More FSDP sharding
```

**For smaller batches**:
```bash
--mesh-data 4 --mesh-fsdp 1 --mesh-tp 2  # More data parallelism
```

### Checkpointing

Reduce checkpoint frequency for faster training:
```bash
--checkpoint-interval 5000  # Save every 5K steps (vs default 1K)
```

## Monitoring
Metrics are tracked via **CLU**:
- **Loss**: Training loss and components (total, sparsity).
- **Throughput**: Tokens per second.
- **Checkpoints**: Saved via Orbax with async support.

### TensorBoard
```bash
# On local machine after SSH into TPU
tensorboard --logdir ./checkpoints
```

### TensorBoard
```bash
# On local machine after SSH into TPU
tensorboard --logdir ./checkpoints
```

### Cloud Monitoring
[GCP Console → TPU → Your TPU](https://console.cloud.google.com/tpus)
- Utilization graphs
- Memory usage
- Network I/O

## Troubleshooting

### Issue: TPU Not Detected
```python
# Check JAX backend
import jax
print(f"Backend: {jax.config.jax_backend_name}")
# Should print: "tpu"
```

**Fix**: Restart runtime, ensure "TPU v6e" is selected.

### Issue: Out of Memory
**Symptoms**: `ResourceExhaustedError` or allocation failures

**Solutions**:
1. Reduce batch size: `--batch-size 16`
2. Reduce sequence length (requires model config change)
3. Increase FSDP sharding: `--mesh-fsdp 4`

### Issue: Slow Training (Low Throughput)
**Symptoms**: < 20K tokens/sec on v6e-1

**Solutions**:
1. Increase `--log-interval 100` (less frequent host transfers)
2. Increase `--worker-buffer-size 4` (prefetch more data)
3. Use less frequent checkpoints: `--checkpoint-interval 5000`

### Issue: NaN Gradients
**Symptoms**: Training loss explodes to NaN

**Root Causes** (already fixed in Nash-MHC):
- Sinkhorn gradient overflow → Fixed: `+ 1e-8` epsilon
- Nash equilibrium `sqrt(0)` → Fixed: `stop_gradient` through iterates

**If still occurs**:
1. Reduce learning rate: `--learning-rate 1e-4`
2. Check data quality (NaNs in token IDs)
3. Verify TPU HBM health (GCP monitoring)

## Cost Estimates

**TPU v6e-1 pricing** (us-central2):
- **On-demand**: ~$5-6/hour for 2x2 slice
- **Preemptible**: ~$1-2/hour (interruptible)

**3B Model Training**:
- **Steps**: 100K
- **Throughput**: ~50K tokens/sec
- **Time**: ~6-14 days (8-core training)

**Total Cost**: ~$700-1600 (on-demand)

## Cleanup

### Delete TPU VM
```bash
gcloud compute tpus tpu-vm delete nash-mhc-tpu \
  --zone us-central2-b
```

### Stop TPU (keep VM)
```bash
gcloud compute tpus tpu stop nash-mhc-tpu \
  --zone us-central2-b
```

## References

- [Google Cloud TPU v6e Setup](https://cloud.google.com/tpu/docs/v6e-setup)
- [JAX TPU Quickstart](https://jax.readthedocs.io/en/latest/tpu_quickstart.html)
- [JAX Distributed Arrays](https://jax.readthedocs.io/en/latest/jax.distributed.html)
- [Weights & Biases Documentation](https://docs.wandb.ai)
