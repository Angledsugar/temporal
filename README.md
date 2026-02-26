# TempoRAL

**Cross-Embodiment Temporal Abstraction Transfer from Human Demonstrations via Internal Reinforcement Learning in VLM Backbones**

Discovers and transfers temporal abstractions from human demonstrations to robot manipulation via VLM backbone internal representations, building on [Internal RL](https://arxiv.org/abs/2512.20605).

## Three-Phase Framework

1. **Phase 1**: Fine-tune VLM backbone on human manipulation demos (LoRA), then freeze
2. **Phase 2**: Train self-supervised metacontroller on frozen VLM's residual stream to discover subtask boundaries
3. **Phase 3**: Internal RL in abstract controller-code space for novel task composition

## Target VLA Architectures

| Model | VLM Backbone | Layers | Hidden | Controlled Layer | Action Head |
|-------|-------------|--------|--------|-----------------|-------------|
| π0.5 | PaliGemma (Gemma 2B) | 18 | 2048 | 9 (L/2) | Gemma Expert 300M |
| Groot N1.6 | Eagle (Qwen3 1.7B) | 28 | 2048 | 12 (paper) | DiT 32L |

---

## Quick Start (서버 설정 + 학습)

### 1. 환경 설치

```bash
# 클론
git clone --recursive <repo-url>
cd temporal

# 방법 A: 원커맨드 설치 (모든 의존성 + 패치 + 테스트)
bash scripts/setup_server.sh

# 방법 B: 수동 설치
uv sync                           # 기본 의존성
uv pip install -e ".[vla]"        # VLA 의존성 (π0.5)
uv pip install -e ".[groot]"      # Groot 의존성

# 테스트 실행으로 설치 확인
uv run pytest tests/ -v           # 67 tests (66 passed, 1 skipped)
```

**요구사항**: Python 3.11+, CUDA 12.x, [uv](https://docs.astral.sh/uv/), GPU (RTX 4090 24GB 권장)

### 2. Gridworld PoC 검증 (선택)

```bash
# 3-stage 파이프라인 전체 실행 (~2분)
uv run python scripts/run_all.py --quick
```

### 3. VLA Dummy 검증

실제 모델/데이터 없이 파이프라인 구조 검증:

```bash
# π0.5 dummy training
uv run python scripts/run_vla.py --model pi05 --dummy-data --quick

# Groot dummy training
uv run python scripts/run_vla.py --model groot --dummy-data --quick
```

---

## 실제 데이터로 학습하기

### Step 1: 모델 가중치 준비

**Groot N1.6** (finetuned checkpoint 필요):
```bash
# configs/groot_metacontroller.yaml 수정
model:
  checkpoint_path: "/path/to/groot_checkpoint"
```

**π0.5** (safetensors 가중치 필요):
```bash
# configs/pi05_metacontroller.yaml 수정
model:
  checkpoint_path: "/path/to/pi05_weights.safetensors"
```

### Step 2: 데이터셋 준비

3가지 방법 중 선택:

#### 방법 A: HuggingFace LeRobot 데이터셋 (권장)

YAML config에서 `data.type: "lerobot"` 설정:

```yaml
# configs/groot_metacontroller.yaml
data:
  type: "lerobot"
  repo_id: "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot"
  image_size: [224, 224]
```

```yaml
# configs/pi05_metacontroller.yaml
data:
  type: "lerobot"
  repo_id: "lerobot/libero_object_no_noops"
  image_size: [224, 224]
```

첫 실행 시 HuggingFace에서 자동 다운로드됩니다.

**사용 가능한 LeRobot 데이터셋 예시**:
| Dataset | repo_id | Robot | Action Dim |
|---------|---------|-------|------------|
| LIBERO-10 | `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` | Franka Panda | 7 |
| LIBERO Goal | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` | Franka Panda | 7 |
| LIBERO Object | `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` | Franka Panda | 7 |

#### 방법 B: 로컬 LeRobot 형식 데이터셋

GROOT demo 데이터 또는 자체 변환한 데이터:

```yaml
data:
  type: "lerobot"
  local_path: "./Isaac-GR00T/demo_data/cube_to_bowl_5"
  image_size: [224, 224]
```

**로컬 데이터 요구 구조**:
```
dataset_dir/
├── meta/
│   ├── info.json          # fps, features, 경로 패턴
│   ├── episodes.jsonl     # 에피소드별 길이/태스크
│   ├── tasks.jsonl        # 태스크 설명
│   ├── stats.json         # 정규화 통계 (min/max/mean/std)
│   └── modality.json      # (Groot용) joint group 매핑
├── data/chunk-000/
│   └── episode_000000.parquet  # state, action, timestamp 등
└── videos/chunk-000/
    └── observation.images.front/
        └── episode_000000.mp4
```

#### 방법 C: Dummy 데이터 (구조 검증용)

```yaml
data:
  type: "dummy"
  num_samples: 100
```

### Step 3: 학습 실행

```bash
# Groot 학습
uv run python scripts/run_vla.py --model groot \
    --config configs/groot_metacontroller.yaml

# π0.5 학습
uv run python scripts/run_vla.py --model pi05 \
    --config configs/pi05_metacontroller.yaml
```

### Step 4: 모니터링

```bash
# TensorBoard
tensorboard --logdir logs/vla/

# 체크포인트 위치
ls checkpoints/vla/groot/   # Groot 체크포인트
ls checkpoints/vla/pi05/    # π0.5 체크포인트
```

**핵심 모니터링 메트릭**:
- `action_loss`: 감소해야 함 (action 예측 학습)
- `kl_loss`: 0.1~0.3에서 안정화 (0으로 collapse하면 안 됨)
- `beta_mean`: 0.3~0.6이 건강한 범위 (subtask 전환 빈도)

---

## 핵심 하이퍼파라미터

| Parameter | π0.5 | Groot | Description |
|-----------|------|-------|-------------|
| `embed_dim` | 2048 | 2048 | VLM hidden dimension |
| `latent_dim` | 16 | 16 | Metacontroller latent code z_t |
| `controlled_layer` | 9 | 12 | VLM 개입 레이어 |
| `kl_alpha` | 0.17 | 0.17 | KL loss 가중치 |
| `controller_rank` | 32 | 32 | Low-rank controller U_t rank |
| `batch_size` | 4 | 2 | per-GPU batch size |
| `lr` | 1e-4 | 1e-4 | AdamW learning rate |
| `train_steps` | 64000 | 64000 | 학습 스텝 수 |

**VRAM 사용량 (RTX 4090, 24GB)**:
- π0.5: ~13GB (VLM 4GB + Expert 0.6GB + activations 8GB + MC 0.01GB)
- Groot: ~16GB (VLM 3.4GB + DiT 2GB + activations 10GB + MC 0.01GB)

---

## 학습 아키텍처

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

- VLM은 **완전히 동결** → optimizer state는 MC만 (~40MB)
- Loss = action MSE + kl_alpha × KL divergence
- MC가 발견하는 것: subtask 경계 (β_t), 추상 제어 코드 (z_t)

---

## Project Structure

```
src/temporal/
  envs/           # Gridworld-Pinpad environment, expert policy
  models/         # Transformer, SSM, Metacontroller, RL policy
  training/       # Pretrain, metacontroller, internal RL, GRPO
  evaluation/     # Linear probing, beta analysis, RL evaluation
  vla/            # VLA extension (π0.5, Groot N1.6)
    models/       # BaseVLAWrapper (ABC), Pi05Wrapper, GrootWrapper
    training/     # VLA metacontroller training (model-agnostic)
    data/         # Dummy + real LeRobot datasets

Isaac-GR00T/      # NVIDIA Groot N1.6
openpi/           # Physical Intelligence π0.5

scripts/
  run_all.py      # Gridworld pipeline (--quick)
  run_vla.py      # VLA pipeline (--model pi05|groot)
  setup_server.sh # One-command server setup

configs/
  pi05_metacontroller.yaml   # π0.5 config (~13GB VRAM)
  groot_metacontroller.yaml  # Groot config (~16GB VRAM)
```

## Tests

```bash
uv run pytest tests/ -v              # 전체 (67 tests)
uv run pytest tests/test_vla/ -v     # VLA만 (38 tests)
uv run pytest tests/test_gridworld.py tests/test_metacontroller.py -v  # Gridworld만
```

---

## Ablation Design

| Condition | VLM State | Fine-tune Data | Expected |
|-----------|-----------|---------------|----------|
| A (Base) | Pretrained only | None | No temporal abstraction |
| B (Human) | + LoRA fine-tune | Human demonstrations | Temporal abstractions emerge |
| C (Robot) | + LoRA fine-tune | Robot demonstrations | Weaker abstractions |

Hypothesis: Condition B shows clearest temporal boundaries (β_t switching) because human demonstrations have natural subtask structure that transfers to the VLM backbone.
