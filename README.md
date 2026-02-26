# TempoRAL

**Cross-Embodiment Temporal Abstraction Transfer from Human Demonstrations via Internal Reinforcement Learning in VLM Backbones**

Discovers and transfers temporal abstractions from human demonstrations to robot manipulation via VLM backbone internal representations, building on [Internal RL](https://arxiv.org/abs/2512.20605).

## Key Hypothesis

Human demonstration data induces **embodiment-invariant temporal abstractions** in VLM backbones that can be discovered unsupervised by a metacontroller and transferred cross-embodiment to robots.

## Three-Phase Framework

1. **Phase 1**: Fine-tune VLM backbone on human manipulation demos (LoRA), then freeze
2. **Phase 2**: Train self-supervised metacontroller on frozen VLM's residual stream to discover subtask boundaries (no annotations)
3. **Phase 3**: Internal RL in abstract controller-code space for novel task composition under sparse rewards

## Project Structure

```
src/internalrl/
  envs/           # Gridworld-Pinpad environment, expert policy
  models/         # Transformer, SSM, Metacontroller, RL policy
  training/       # Pretrain, metacontroller, internal RL, GRPO
  evaluation/     # Linear probing, beta analysis, RL evaluation
  vla/            # VLA extension (π0.5, Groot N1.6)
    models/       # BaseVLAWrapper (ABC), Pi05Wrapper, GrootWrapper
    training/     # VLA metacontroller training (model-agnostic)
    data/         # Dummy datasets (residual, π0.5 format, Groot format)

Isaac-GR00T/      # Git submodule: NVIDIA Groot N1.6
openpi/           # Git submodule: Physical Intelligence π0.5

scripts/
  run_all.py      # Gridworld pipeline (--quick)
  run_vla.py      # VLA pipeline (--model pi05|groot)
  setup_server.sh # One-command 4090 server setup

configs/
  pi05_metacontroller.yaml   # π0.5 config (RTX 4090 ~13GB)
  groot_metacontroller.yaml  # Groot config (RTX 4090 ~16GB)
```

## Target VLA Architectures

| Model | VLM Backbone | Layers | Hidden | Controlled Layer | Action Head |
|-------|-------------|--------|--------|-----------------|-------------|
| π0.5 | PaliGemma (Gemma 2B) | 18 | 2048 | 9 (L/2) | Gemma Expert 300M |
| Groot N1.6 | Eagle (Qwen3 1.7B) | 28 | 2048 | 12 (paper) | DiT 32L |

## Why VLM Backbone (not Action Head)

The metacontroller targets the **VLM backbone** because:
- VLM is **autoregressive + causal** → temporal abstractions emerge (per Internal RL prerequisites)
- Action Head uses **diffusion/flow matching** → no sequential prediction → no temporal abstraction
- Groot paper: middle-layer (12th) LLM embeddings outperform final layer
- VLM = planner ("what to do"), Action Head = executor ("how to move")

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12.x (for GPU training)

### Quick Setup (Development)

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd internalRL

# Install base dependencies
uv sync

# Install VLA dependencies (π0.5)
uv pip install -e ".[vla]"

# Install Groot dependencies (if using Groot)
uv pip install -e ".[groot]"
```

### Full Setup (RTX 4090 Training Server)

```bash
git clone --recursive <repo-url>
cd internalRL
bash scripts/setup_server.sh
```

This script:
1. Installs all Python dependencies (base + VLA + Groot)
2. Applies `transformers_replace` patches (required for π0.5 AdaRMS)
3. Verifies all imports (PI0Pytorch, Gr00tN1d6, wrappers)
4. Runs 53 tests
5. Shows GPU information

### transformers_replace Patches (π0.5)

π0.5 requires patched `transformers` with AdaRMS and custom attention. The setup script handles this automatically. For manual setup:

```bash
# Backup and apply patches
OPENPI_REPLACE="openpi/src/openpi/models_pytorch/transformers_replace"
TRANSFORMERS_DIR=".venv/lib/python3.11/site-packages/transformers"

for f in $OPENPI_REPLACE/*.py; do
    target=$(find $TRANSFORMERS_DIR -name "$(basename $f)" -type f | head -1)
    [ -n "$target" ] && cp "$target" "${target}.bak" && cp "$f" "$target"
done
```

---

## Gridworld PoC

Validates the 3-phase framework on a simple Gridworld-Pinpad task before scaling to VLAs.

```bash
# Full pipeline (quick mode: ~2 min)
uv run python scripts/run_all.py --quick

# Run tests (29 gridworld tests)
uv run pytest tests/test_gridworld.py tests/test_metacontroller.py tests/test_transformer.py -v
```

**Results**: 29/29 tests passing, all 3 stages run end-to-end.

---

## VLA Training Guide

### Overview

The VLA training pipeline has 3 stages:

```
Stage 1: VLM Fine-tuning (LoRA)     → done externally (OpenPI / Isaac-GR00T tools)
Stage 2: Metacontroller Training     → this project (vla_metacontroller_train.py)
Stage 3: Internal RL                 → stub (needs robot simulator)
```

**Stage 2 is the main training target.** The metacontroller learns temporal abstractions from the frozen VLM's residual stream.

### Stage 2: Metacontroller Training

#### Step 1: Prepare Model Weights

**π0.5**: Download or use your fine-tuned π0.5 weights (safetensors format):
```bash
# Set checkpoint path in config
# configs/pi05_metacontroller.yaml
model:
  checkpoint_path: "/path/to/pi05_weights.safetensors"
```

**Groot N1.6**: Use your fine-tuned Groot checkpoint:
```bash
# configs/groot_metacontroller.yaml
model:
  checkpoint_path: "/path/to/groot_checkpoint"
```

#### Step 2: Prepare Data

현재 dummy data로 구조 검증 완료. 실제 학습 시:

**Residual 데이터**: VLA에 에피소드를 통과시켜 중간 레이어 hidden states를 추출 → 저장
```python
# data format per sample:
{
    "residual": Tensor(seq_len, 2048),   # VLM hidden states at controlled_layer
    "actions": Tensor(action_horizon, action_dim),  # target actions
    "noise": Tensor(action_horizon, action_dim),    # flow matching noise
    "time": Tensor(),                                # diffusion timestep
}
```

**또는 Full forward 데이터**: 원본 에피소드를 그대로 전달 (wrapper가 residual 추출)
```python
# π0.5 format:
{
    "images": {"base_0_rgb": Tensor(3, 224, 224), ...},
    "image_masks": {"base_0_rgb": Tensor(bool), ...},
    "tokenized_prompt": Tensor(max_token_len),
    "tokenized_prompt_mask": Tensor(max_token_len, bool),
    "state": Tensor(state_dim),
    "actions": Tensor(action_horizon, action_dim),
}

# Groot format:
{
    "images": {"cam_0": Tensor(3, 224, 224)},
    "state": Tensor(state_dim),
    "actions": Tensor(action_horizon, action_dim),
    "text": "pick up the red block",
    "input_ids": Tensor(seq_len, long),
    "attention_mask": Tensor(seq_len, long),
}
```

#### Step 3: Run Training

**Dummy data (구조 검증)**:
```bash
# π0.5 dummy training
uv run python scripts/run_vla.py --model pi05 --dummy-data --quick

# Groot dummy training
uv run python scripts/run_vla.py --model groot --dummy-data --quick
```

**Full training with config**:
```bash
# π0.5
uv run python scripts/run_vla.py --model pi05 --config configs/pi05_metacontroller.yaml

# Groot
uv run python scripts/run_vla.py --model groot --config configs/groot_metacontroller.yaml
```

#### Step 4: Key Hyperparameters

| Parameter | π0.5 | Groot | Description |
|-----------|------|-------|-------------|
| `embed_dim` | 2048 | 2048 | VLM hidden dimension |
| `latent_dim` | 16 | 16 | Metacontroller latent code z_t |
| `controlled_layer` | 9 | 12 | VLM layer to intervene |
| `kl_alpha` | 0.17 | 0.17 | KL loss weight (temporal switching) |
| `controller_rank` | 32 | 32 | Low-rank controller U_t rank |
| `batch_size` | 4 | 2 | Per-GPU batch size |
| `lr` | 1e-4 | 1e-4 | AdamW learning rate |
| `train_steps` | 64000 | 64000 | Total training steps |

**VRAM estimates** (RTX 4090, 24GB):
- π0.5: ~13GB (VLM 4GB + Expert 0.6GB + activations 8GB + MC 0.01GB)
- Groot: ~16GB (VLM 3.4GB + DiT 2GB + activations 10GB + MC 0.01GB)

### Training Architecture

```
Frozen VLA (no grad)          Trainable MC (~5M params)
┌─────────────────────┐       ┌──────────────────────────┐
│ VLM Layers 0..L/2   │──→ e_t│ InternalSeqEmbedder(SSM) │
│  (extract residual)  │       │ ControllerEncoder(GRU)   │
│                      │       │ SwitchingUnit(β_t gate)  │
│ VLM Layers L/2..L   │←─────│ ControllerDecoder(z→U_t)  │
│  (with controlled e) │       │ e_ctrl = e + U_t·e       │
│                      │       └──────────────────────────┘
│ Action Head          │
│  (flow matching)     │──→ MSE loss + KL loss
└─────────────────────┘
```

- VLM is **completely frozen** → optimizer state only for MC (~40MB)
- MC outputs: `e_controlled`, `z_seq` (latent codes), `beta_seq` (switching gate), `kl_loss`
- Loss = action MSE + `kl_alpha` × KL divergence

### Monitoring

Training outputs to TensorBoard:
```bash
tensorboard --logdir logs/vla/
```

Key metrics to watch:
- `action_loss`: should decrease (model learns to predict actions)
- `kl_loss`: should stabilize around 0.1-0.3 (not collapse to 0)
- `beta_mean`: average switching gate (~0.3-0.6 is healthy)

Checkpoints saved to `checkpoints/vla/{model}/`.

---

## Tests

```bash
# All tests (53)
uv run pytest tests/ -v

# Gridworld only (29)
uv run pytest tests/test_gridworld.py tests/test_metacontroller.py tests/test_transformer.py -v

# VLA only (24)
uv run pytest tests/test_vla/ -v
```

---

## Ablation Design

| Condition | VLM State | Fine-tune Data | Expected |
|-----------|-----------|---------------|----------|
| A (Base) | Pretrained only | None | No temporal abstraction |
| B (Human) | + LoRA fine-tune | Human demonstrations | Temporal abstractions emerge |
| C (Robot) | + LoRA fine-tune | Robot demonstrations | Weaker abstractions |

Hypothesis: Condition B shows clearest temporal boundaries (β_t switching) because human demonstrations have natural subtask structure that transfers to the VLM backbone.
