# VLA 논문 종합 분석 및 Metacontroller 적용 방안

## 1. 각 논문 요약

### 1.1 GR00T N1 (NVIDIA, 2025.03)
**"An Open Foundation Model for Generalist Humanoid Robots"**

- **아키텍처**: Dual-system VLA
  - **System 2 (VLM)**: Eagle-2 VLM (SmolLM2 1.34B LLM + SigLIP-2 vision encoder). 이미지+언어를 처리하여 의미적 이해 수행. 10Hz로 동작.
  - **System 1 (DiT)**: Diffusion Transformer. Flow matching으로 고주파(120Hz) motor action 생성. VLM 출력에 cross-attention하여 action chunk (H=16) 생성.
- **핵심 기여**:
  - Data Pyramid: 웹 데이터 → 합성 데이터 → 실제 로봇 데이터의 계층적 학습
  - Latent Actions (VQ-VAE): 동영상에서 action label 없이 latent action을 추출하여 학습 데이터로 활용
  - Neural Trajectories: 비디오 생성 모델로 합성 trajectory 생성 (실제 데이터 10배 확장)
  - Cross-embodiment: 단일 모델로 single-arm ~ humanoid까지 지원
- **학습 비용**: H100 GPU 1024장, ~50,000 H100 GPU hours (pretraining)
- **Post-training**: VL backbone 동결, 나머지 fine-tune. A6000 1장에서 가능 (adapter layers + DiT만 tune시 batch=200)
- **중요 발견**: **middle layer (12번째) LLM 임베딩**이 final layer보다 downstream 성능이 더 좋음 → residual stream의 중간 레이어에 풍부한 정보가 있다는 증거

### 1.2 π0 (Physical Intelligence, 2024.10)
**"A Vision-Language-Action Flow Model for General Robot Control"**

- **아키텍처**: Two-expert VLA
  - **PaliGemma VLM** (Gemma 2B + SigLIP 400M): vision + language backbone. 2048-dim, 18 layers.
  - **Action Expert** (Gemma 300M): action token 전용 별도 weight. 1024-dim, 18 layers.
  - 두 expert가 같은 transformer 안에서 다른 weight로 처리 (MoE 유사)
- **핵심 기여**:
  - Flow matching으로 continuous action chunk (H=50) 생성
  - VLM pre-training → cross-embodiment 학습 → task-specific fine-tuning
  - 7종 로봇, 68개 태스크에서 학습
- **Action Expert 역할**: action token만 별도 weight로 처리. Bidirectional attention (전체 action chunk 동시 처리).
- **학습 비용**: 10,000시간 로봇 데이터, 700K training steps
- **핵심 인사이트**: VLM initialization이 language following과 generalization에 결정적. π0-small (VLM 미사용)은 성능 대폭 하락.

### 1.3 π0.5 (Physical Intelligence, 2025.02)
**"A Vision-Language-Action Model with Open-World Generalization"**

- **아키텍처**: π0 기반 + hierarchical inference
  - Pre-training: 표준 autoregressive transformer로 discrete token 예측 (FAST tokenizer)
  - Post-training: action expert 추가하여 flow matching으로 continuous action 생성
  - **2단계 추론**: 먼저 high-level subtask 텍스트 예측 ("pick up the pillow") → 이에 조건화된 low-level action 생성
- **핵심 기여**:
  - Co-training: 로봇 데이터 + high-level subtask prediction + web data (captioning, VQA, object detection)
  - Knowledge Insulation (KI): next-token prediction과 flow matching을 분리하여 각각 최적화
  - Verbal instructions: 사람이 실시간으로 subtask 명령을 제공하여 학습
  - **Open-world generalization**: 학습에 없던 집에서도 주방/침실 청소 수행 (10-15분 장기 태스크)
- **Pre-training 데이터**: 97.6%가 non-target-robot 데이터 (다른 로봇, 웹 데이터 등)
- **핵심 인사이트**: Diverse pretraining이 generalization의 핵심. Cross-embodiment(ME, CE) 데이터 제거 시 성능 대폭 하락.

### 1.4 π*0.6 / RECAP (Physical Intelligence, 2025)
**"A VLA That Learns From Experience"**

- **아키텍처**: π0.5 기반 + advantage conditioning + value function
  - π0.6 VLA: Gemma 3 4B backbone + 860M action expert
  - Value function: 별도 670M Gemma 3 backbone으로 distributional value function 학습
  - Advantage indicator I_t를 VLA 입력에 추가하여 advantage conditioning
- **핵심 기여 (RECAP)**:
  1. **Data collection**: 자율 rollout + 전문가 개입(correction)
  2. **Value function training**: multi-task distributional value function (201 bins)
  3. **Advantage-conditioned training**: I_t = 1(A > ε_l)로 good/bad action 구분
  - Offline RL 방식으로 on-policy 없이 학습 가능
- **결과**: 세탁물 접기/에스프레소 만들기/박스 조립에서 throughput 2배+, failure rate 2배 감소
- **핵심 인사이트**: PPO/AWR 대비 advantage conditioning이 flow matching VLA에 더 적합. 경험 데이터로부터 iterative하게 개선 가능.

### 1.5 Human-to-Robot Transfer (Physical Intelligence, 2025)
**"Emergence of Human to Robot Transfer in Vision-Language-Action Models"**

- **핵심 발견**: Human-to-robot transfer는 **diverse VLA pretraining의 emergent property**
  - Pretraining diversity가 낮으면(0-25%): human data co-training 효과 없음
  - Pretraining diversity가 높으면(75-100%): human data로 성능 대폭 향상 (최대 2배)
- **메커니즘**: Diverse pretraining → **embodiment-agnostic representations** 형성
  - TSNE 분석: pretraining diversity 증가 시 human/robot latent representation이 수렴
  - 명시적 alignment 없이 co-training만으로 자연스럽게 정렬됨
- **실험**: π0.5 + ego (egocentric human data)
  - Scene transfer: Spice 32→71%, Dresser 25→50%
  - Object transfer: Bussing 53→63%
  - Task transfer: Sort Eggs (robot data에 없는 새 태스크) 57% 정확도
- **Human data 수집**: head camera + wrist cameras, 3D hand tracking, end-effector pose 추출
- **핵심 인사이트**: Human data ≈ 다른 로봇 embodiment data와 유사한 효과. High-level과 low-level 모두에서 transfer 발생.

---

## 2. VLM Backbone에 Metacontroller 적용이 올바른 이유

### 2.1 논문(arXiv:2512.20605)의 핵심 전제

Temporal abstraction의 출현 조건:
1. **Autoregressive next-token prediction** 학습
2. **시간 순서(temporal sequence)** 를 따라 처리
3. **장기 예측** 필요성 → 모델이 내부적으로 서브골을 표현

### 2.2 VLM Backbone이 올바른 이유

#### Groot N1의 Eagle-2 VLM:
```
시간축 →
[image₁ + text₁] → [image₂ + text₂] → ... → [imageₜ + textₜ]
      ↓                    ↓                         ↓
  layer 1..28          layer 1..28              layer 1..28
      ↓                    ↓                         ↓
  residual stream에 temporal context 축적
```
- **Autoregressive**: Causal attention으로 과거→미래 순서 처리 ✓
- **Temporal sequence**: 시간순 관측 시퀀스를 처리 ✓
- **장기 계획 필요**: "pick up the apple and place it on the shelf" → 여러 서브태스크 내부 표현 필요 ✓
- **Groot 논문 증거**: **12번째 (중간) layer 임베딩**이 final layer보다 downstream에 더 유용 → 중간 레이어에 추상적 계획 정보 존재

#### π0의 PaliGemma VLM:
- Gemma 2B (18 layers): autoregressive language model ✓
- 시간순 관측 처리 ✓
- VLM initialization이 task reasoning에 결정적 (π0-small vs π0 비교로 검증)

#### π0.5의 hierarchical inference:
```
High-level: "clean the bedroom" → "pick up the pillow" → "put clothes in basket"
Low-level:  각 subtask에 대해 action chunk 생성
```
- **π0.5는 이미 temporal abstraction을 명시적으로 사용**: subtask prediction이 곧 서브골
- Metacontroller가 이를 **비지도로** 발견할 수 있다면, 수동 annotation 불필요

#### Human-to-Robot Transfer 논문의 증거:
- **Embodiment-agnostic representations**: diverse pretraining으로 형성
- TSNE에서 human/robot representation 수렴 → VLM backbone이 추상적 표현을 학습
- 이 추상 표현이 곧 temporal abstraction의 기반

### 2.3 Action Head(DiT/Action Expert)에 적용하면 다른 이유

#### Groot N1의 DiT:
```
디노이징 축 →
noise → step 1 → step 2 → ... → step K=4 → clean actions [a_t, ..., a_{t+15}]
         ↑           ↑                           ↑
    cross-attention to VLM outputs (컨텍스트 제공)
```
| 속성 | VLM Backbone | DiT Action Head |
|------|-------------|-----------------|
| **예측 방식** | Autoregressive (next-token) | Diffusion (iterative denoising) |
| **시퀀스 의미** | 에피소드 시간축 (o₁→o₂→...→oₜ) | Action chunk 공간 ([a_t...a_{t+15}]) |
| **인과성** | Causal (과거→미래) | Bidirectional (전체 동시) |
| **처리 대상** | 전체 에피소드 컨텍스트 | 16~50 스텝 action chunk |
| **Temporal abstraction** | 자연 출현 (논문 검증) | 출현 근거 없음 |

**DiT/Action Expert가 temporal abstraction을 형성하지 않는 이유:**

1. **Autoregressive가 아님**: Flow matching은 noise→signal 변환. "다음에 뭐가 올까?"를 예측하지 않으므로 장기 계획을 내부 표현할 유인이 없음.

2. **시간 범위가 짧음**: Action chunk는 0.3~1초 분량(Groot H=16 @ 120Hz). 서브태스크 전환은 수초~수십초 단위이므로 chunk 내에서 발견 불가.

3. **컨텍스트 의존적**: DiT는 VLM output에 cross-attend하여 "무엇을 할지"를 VLM에서 받음. 자체적으로 계획하지 않음.

4. **π0 논문의 설계 의도**: Action Expert는 별도 weight로 분리된 이유가 "action 생성의 전문화"이지, "계획 수립"이 아님.

#### π0의 Action Expert:
```
[VLM이 결정한 것] → cross-attention → Action Expert → continuous actions
  "무엇을 할 것인가"                    "어떻게 할 것인가"
  (서브골/계획)                         (운동 제어)
```
- Action Expert는 **실행기(executor)**이지 **계획기(planner)**가 아님
- Metacontroller가 제어해야 할 것은 **"무엇을 할 것인가"** = VLM의 계획 표현

### 2.4 VLA 논문들에서의 추가 증거

#### Groot N1의 middle-layer 발견:
> "We found that using **middle-layer LLM embeddings** instead of final-layer LLM embeddings resulted in both faster inference speed and **higher downstream policy success rate**." (p.4)

→ 중간 레이어에 task-relevant abstract information이 집중. 이것이 metacontroller가 개입해야 할 지점.

#### π0.5의 subtask prediction:
> "At inference time, the model first produces a **high-level subtask** for the robot to perform and then, conditioned on this subtask, predicts the low-level actions via the action expert."

→ π0.5는 이미 subtask를 명시적으로 예측. Metacontroller는 이를 비지도로 발견하는 대안.

#### RECAP의 advantage conditioning:
> "The advantage indicator I_t appears in the training sequence after l̂ but **before the (discretized and continuous) actions**"

→ RECAP도 VLM 출력(subtask/advantage) 수준에서 개입하지, action expert 내부에서 개입하지 않음.

---

## 3. VLM 학습 비용 문제와 대안

### 3.1 현재 문제

| 모델 | Pretraining 비용 | Fine-tuning 비용 |
|------|---------------|---------------|
| Groot N1 | 50,000 H100 GPU hours | A6000 1장 (adapter only) |
| π0 | ~수만 GPU hours (미공개) | task별 수시간~수일 |
| π0.5 | 280K steps pre + 80K steps post | 상동 |
| π*0.6 | 위 + RECAP iterations | 상동 |

**VLM 자체를 학습하는 것은 비용이 매우 큼.** 하지만 metacontroller 접근의 핵심 장점:

> **Base model(VLM)을 완전히 동결**하고 metacontroller만 학습

### 3.2 세 가지 접근법

#### 접근법 A: Finetuned VLA를 동결하고 Metacontroller만 학습 (추천)
```
[Finetuned Groot N1.6 / π0] ← 완전 동결 (이미 학습 완료)
         ↓
   residual stream 추출
         ↓
   [Metacontroller] ← 이것만 학습 (작은 모델)
         ↓
   [Internal RL Policy] ← 이것만 학습
```
- **장점**: VLM 재학습 불필요. Metacontroller는 작은 모델 (수 MB)
- **비용**: 단일 GPU에서 가능
- **논문과 완전 일치**: base model 동결이 핵심 (R-D phase transition)
- **전제**: Finetuned VLA의 residual stream에 이미 temporal abstraction이 형성되어 있어야 함

#### 접근법 B: 일반 VLM (Qwen2.5-VL / Gemma3) + Human Data
```
[Qwen2.5-VL-7B or Gemma3-4B] ← 공개 pretrained VLM
         ↓
   Human demonstration video로 action prediction fine-tune
         ↓
   동결 → Metacontroller 학습 → Internal RL
```
- **장점**:
  - 공개 모델 사용으로 비용 대폭 절감
  - 더 큰 모델(7B+) 사용 가능 → 더 풍부한 representation
  - Human-to-Robot Transfer 논문의 발견: diverse pretraining → embodiment-agnostic representation
- **구현 방법**:
  1. Human demo에서 end-effector trajectory 추출 (3D hand tracking)
  2. VLM에 action prediction head 추가 (LoRA/linear probe)
  3. Flow matching 또는 discrete action tokenizer로 학습
  4. 동결 → metacontroller → internal RL
- **근거**: Human-to-Robot Transfer 논문에서 검증
  > "human-to-robot transfer emerges once the VLA is pre-trained on sufficient scenes, tasks, and embodiments"
  - Qwen2.5-VL / Gemma3은 이미 massive diverse pretraining 완료 → embodiment-agnostic representation 기반 존재

#### 접근법 C: Frozen VLM + Lightweight Adapter (가장 저비용)
```
[Qwen2.5-VL-7B] ← 완전 동결 (pretrained 그대로)
         ↓
   [LoRA Adapter] ← human data로 action prediction 학습 (수 MB)
         ↓
   동결 → Metacontroller 학습 → Internal RL
```
- **장점**: VLM 자체를 전혀 건드리지 않음. 가장 저비용.
- **이론적 근거**: Pretrained VLM도 이미 temporal/spatial reasoning 능력 보유. Human demo의 시각적 패턴에서 temporal structure를 이미 이해할 수 있음.
- **위험**: Action prediction 품질이 full fine-tuning 대비 낮을 수 있음

### 3.3 Qwen2.5-VL / Gemma3 적용 방안 구체안

#### Qwen2.5-VL-7B (추천)
```python
# 아키텍처 구조
Qwen2.5-VL-7B:
  - Vision Encoder: ViT (dynamic resolution)
  - LLM: Qwen2.5-7B (28 layers, 3584-dim hidden)
  - Total: ~7B parameters

# Metacontroller 적용
controlled_layer = 14  # 28 layers 중 중간
embed_dim = 3584       # Qwen2.5-7B hidden size
metacontroller = MetaController(
    embed_dim=3584,    # 큰 모델에 맞게 조정
    latent_dim=16,     # 8→16 (더 풍부한 표현)
    controller_rank=32 # 16→32
)
```

**장점**:
- 한국어/영어 모두 강력한 언어 이해
- Dynamic resolution으로 다양한 이미지 처리
- 7B라 24GB GPU 1장에서 동작 가능 (4-bit quantization)

#### Gemma3-4B
```python
# 아키텍처 구조
Gemma3-4B:
  - Vision Encoder: SigLIP2
  - LLM: Gemma3-4B (34 layers, 2560-dim hidden)
  - Total: ~4B parameters

# Metacontroller 적용
controlled_layer = 17  # 34 layers 중 중간
embed_dim = 2560
```

**장점**:
- π*0.6이 이미 Gemma3 기반 → 호환성 높음
- 상대적으로 작아서 학습 효율적
- SigLIP2 vision encoder는 Groot N1과 동일 계열

### 3.4 추천 전략

```
Phase 1 (검증): Gridworld PoC 완성 (현재 진행 중)
     ↓
Phase 2 (저비용 검증):
  Option 1: Finetuned Groot N1.6 동결 → Metacontroller 학습
  Option 2: Qwen2.5-VL-7B + LoRA + Human data → 동결 → Metacontroller 학습
     ↓
Phase 3 (논문):
  β_t 분석, 비지도 서브골 발견 검증, Internal RL 성능 비교
```

**핵심 논점 (논문 contribution)**:
1. VLA에서 human data로 학습한 모델의 residual stream에도 temporal abstraction이 출현하는가?
2. Metacontroller가 이를 비지도로 발견하여 β_t로 서브골 전환을 포착하는가?
3. Internal RL로 새로운 장기 태스크를 학습할 수 있는가?
4. 이것이 full VLM fine-tuning 없이도 가능한가? (Qwen/Gemma3 + adapter)

---

## 4. VLA별 Metacontroller 적용 대비 비교

| | Groot N1.6 | π0/π0.5 | Qwen2.5-VL | Gemma3-4B |
|---|---|---|---|---|
| VLM layers | 28 (Qwen3) | 18 (Gemma 2B) | 28 (Qwen2.5) | 34 (Gemma3) |
| VLM hidden | 2048 | 2048 | 3584 | 2560 |
| Action head | DiT 32L | Expert 18L | 추가 필요 | 추가 필요 |
| Controlled layer | ~14 | ~9 | ~14 | ~17 |
| 공개 모델 | ✓ | ✓ (PyTorch) | ✓ | ✓ |
| Human data 학습 | ✓ (latent action) | ✓ (flow matching) | LoRA fine-tune | LoRA fine-tune |
| GPU 요구량 (MC학습) | 1x A6000 | 1x A6000 | 1x 24GB | 1x 24GB |
| 학습 비용 (VLM) | 이미 완료 | 이미 완료 | LoRA만 | LoRA만 |
