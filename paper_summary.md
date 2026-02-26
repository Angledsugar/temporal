# Emergent Temporal Abstractions in Autoregressive Models Enable Hierarchical Reinforcement Learning

**논문**: arXiv:2512.20605v2 (2025년 12월)
**저자**: Seijin Kobayashi, Yanick Schimpf, Maximilian Schlegel 외 (Google, Paradigms of Intelligence Team)

---

## 1. 핵심 아이디어 (Core Idea)

대규모 자기회귀 시퀀스 모델(Transformer, SSM)이 **next-token prediction**으로 사전학습될 때, 내부 활성화(residual stream)에 **시간적으로 추상화된 행동(temporally-abstract actions)**의 표현이 자연스럽게 형성된다. 이를 활용하여 **계층적 강화학습(hierarchical RL)**을 수행하면, 기존 token-by-token RL이 실패하는 **희소 보상(sparse reward)** 환경에서도 학습이 가능하다.

### 핵심 문제
- 자기회귀 모델은 한 번에 하나의 토큰만 생성 → 탐색이 토큰 수준 변동에만 의존
- 보상이 희소한 환경에서 효율적 탐색 불가능
- 해결: **내부 활성화 공간에서 추상적 행동 수준으로 탐색**하면 탐색 공간 축소 + 신용 할당 효율화

---

## 2. 주요 구성 요소

### 2.1. Base Autoregressive Model (f_θ)
- Transformer 또는 SSM (Hawk) 기반
- 행동 데이터셋 D에서 next-action prediction + next-observation prediction으로 사전학습
- 손실 함수: L(θ) = Σ [-log p_θ(a_t|o_{1:t}) - λ log p_θ(o_{t+1}|o_{1:t})]
- 학습 후 파라미터 θ는 **동결(frozen)**

### 2.2. Metacontroller (c_φ)
자기회귀 모델의 residual stream을 읽고 쓰는 **상위 레벨 신경망**

**구조 (Encoder-Decoder Generative Model)**:
1. **Internal Sequence Embedder** (f_emb): 비인과적(non-causal) SSM. 전체 residual stream 시퀀스 e_{1:T,l}을 읽어 시퀀스 임베딩 s(e_{1:T,l}) 생성
2. **Controller Encoder** (f_enc): GRU 기반. 과거 활성화 히스토리 h_t와 시퀀스 임베딩을 조건으로 잠재 코드 후보 z̃_t를 가우시안 분포 N(μ_t, Σ_t)에서 샘플링
3. **Switching Unit** (f_switch): 시간적 통합률(temporal integration rate) β_t ∈ [0,1] 결정
4. **Temporal Integration**: z_t = β_t ⊙ z̃_t + (1 - β_t) ⊙ z_{t-1}
   - β_t ≈ 1: 새 추상 행동으로 전환
   - β_t ≈ 0: 기존 추상 행동 유지
5. **Controller Decoder** (f_hyp): 하이퍼네트워크. z_t → 선형 컨트롤러 U_t 생성
6. **Internal Controller**: e_{t,l} ← e_{t,l} + U_t · e_{t,l} (residual stream 조작)

**학습 목적 함수**:
```
L(φ) = Σ [-log p_{θ,φ}(a_t|o_{1:t}, z_{1:t}) + α · D_KL(N(μ_t, Σ_t) || N(0,I))]
```
- 자기지도(self-supervised) 학습: 보상이나 서브골 라벨 없이 행동 예측 손실만으로 학습
- KL 정규화(α): variational bottleneck으로 서브골 정렬 스위칭 유도

### 2.3. Internal RL
- Metacontroller 학습 후, 비인과적 인코더를 **인과적 정책 π(z_t | e_{1:t})**로 교체
- β_t를 이진화: H(β_t - β_threshold) (Heaviside step function)
- **전체 시스템(base model + metacontroller)을 환경으로 취급**, 정책 π만 RL로 학습
- 행동 공간: z (잠재 코드, 차원 n_z < n_e) → 크게 축소된 행동 공간
- 시간 척도: 추상 행동 수준 → 효과적 시간 지평 대폭 단축
- RL 알고리즘: GRPO 변형 (PPO의 critic을 상대적 이점 추정으로 대체)

---

## 3. 실험 환경

### 3.1. Gridworld-Pinpad
- 7×7 격자, 8개 색상 셀, 4개 벽
- 태스크: 특정 순서로 색상 셀 방문
- 이산 관찰/행동 (4방향)
- T = 100 타임스텝

### 3.2. Ant-Pinpad (MuJoCo)
- 4×4 격자, 4개 색상 셀, 1개 벽
- MuJoCo 개미 로봇: 8차원 연속 관절 토크
- 연속 관찰/행동
- T = 500 타임스텝

### 태스크 구성
- **사전학습 태스크**: 짧은 서브골 시퀀스 (길이 3~6)
- **사후학습 태스크**: 사전학습에서 보지 못한 **더 긴 조합적 서브골 시퀀스** → 길이 및 조합적 일반화 필요

---

## 4. 핵심 발견 (Key Results)

### 4.1. 내부 표현에서 추상 행동이 창발
- **Linear Probing**: residual stream 활성화에서 현재 서브골을 선형적으로 디코딩 가능
  - 중간 레이어(layer 4)부터 정확도 급증, 최종 레이어 직전까지 유지
  - 시간이 지날수록(에이전트가 서브골에 가까워질수록) 디코딩 정확도 증가

### 4.2. Metacontroller가 시간적 추상화 발견
- 스위칭 게이트 β_t가 ground truth 추상 행동 전환 시점과 정렬
- 비지도 학습인데도 **준이진(quasi-binary) 스위칭** 패턴 자연 발생
- 잠재 코드 z가 서브골 추구 행동(subgoal-seeking behavior)을 인코딩

### 4.3. Internal RL의 압도적 성능
- **Raw action RL (GRPO)**: 100만 에피소드 내 학습 실패
- **CompILE**: 학습 실패
- **Internal RL (co-train)**: 학습 실패
- **Internal RL (w/o temporal abstraction)**: 높은 초기 성공률이나 credit assignment 실패
- **Internal RL (full)**: 높은 초기 성공률 + 효율적 credit assignment → **성공**

### 4.4. Rate-Distortion 분석
- 동결된 base model에서의 metacontroller: 서브골 정렬 스위칭이 나타나는 명확한 위상 전이
- 공동 학습(co-train)된 metacontroller: 서브골 정렬 구조 없음
- → **사전학습 후 base model 동결이 추상 행동 발견에 필수적**

---

## 5. Metacontroller 검증 방법

### 5.1. Belief State Probing (선형 프로빙)
- 동결된 사전학습 모델의 각 레이어 residual stream에서 선형 분류기로 현재 서브골 디코딩
- 정확도 측정 → 중간~후반 레이어에서 추상 행동 표현 확인

### 5.2. Temporal Abstraction Discovery
- 스위칭 게이트 β_t와 ground truth 추상 행동 g_t를 시각적으로 비교
- β_t ≈ 1 시점이 실제 서브골 전환 시점과 일치하는지 확인

### 5.3. Quality of Abstract Actions
- Unconditioned rollout에서 잠재 코드 z를 수집
- 동일 잠재 코드를 다른 시나리오에 삽입하여 서브골 추구 행동 일반화 확인
- 새로운 그리드 구성 및 전환 타이밍에 대한 일반화 테스트
  - "go to blue" 잠재 코드를 timestep 30에 주입 → 성공률 23% → 36%
- OOD 시퀀스 일반화: 학습 데이터에 없는 전환 → 성공률 10% → 24%

### 5.4. Rate-Distortion Analysis
- KL 정규화 α를 변화시키며 rate-distortion 곡선 추적
- 동결 base model: 서브골 정렬 스위칭이 나타나는 위상 전이 (수평 갭)
- 공동 학습 base model: 구조 없음 → base model 동결의 중요성 입증

### 5.5. Ablation Studies
- 시퀀스 모델 학습 단계, weight decay, 관찰 보조 손실, expert suboptimality 변화에 따른 영향 분석

---

## 6. Internal RL 검증 방법

### 6.1. Post-training Task 성공률 비교
- 사전학습에서 보지 못한 더 긴, 조합적 서브골 시퀀스에 대해 성공률 측정
- 에피소드 수에 따른 학습 곡선 비교

### 6.2. 베이스라인 비교
| 방법 | 결과 |
|------|------|
| Raw action RL (GRPO) | 실패 (탐색 공간 너무 큼) |
| CompILE | 실패 (사전학습 모델 내부 표현 미활용) |
| Internal RL (co-train) | 실패 (추상 행동 구조 미형성) |
| Internal RL (w/o temporal abstraction, β_t=1 강제) | 높은 초기 성공률이나 credit assignment 실패 |
| **Internal RL (full)** | **성공** - 높은 초기 성공률 + 효율적 credit assignment |

### 6.3. Credit Assignment 효율성 분석 (Appendix E.2)
- 정책 그래디언트 분산 비교:
  - Raw action space PG: 분산이 타임스텝 수 T와 행동 공간 차원에 비례
  - Internal RL PG: 분산이 O(1) — 추상 행동 수 M에만 비례, 타임스텝 무관
- → 장기 수평(long-horizon) 태스크에서 극적 이점

### 6.4. Log-scale 분석 (Appendix Fig. A5)
- Internal RL: 학습 초기에 가장 높은 성공률 (노이즈 주입이 탐색에 유용)
- 다른 베이스라인: 초기 성공이 있어도 보상 최대화 정책으로 전환 실패

---

## 7. VLA (Groot, π) 적용 가능성 분석

### 적용 가능 조건
1. ✅ 자기회귀 모델 기반이어야 함 → Groot-N1, π0 모두 자기회귀 구조
2. ✅ Residual stream 구조가 있어야 함 → Transformer/SSM 기반으로 존재
3. ✅ 행동 데이터로 사전학습되어야 함 → VLA는 로봇 행동 데이터로 학습
4. ⚠️ 계층적 태스크 구조가 있어야 함 → 로봇 조작 태스크는 자연스럽게 계층적

### 잠재적 이점
- 로봇 조작의 long-horizon 태스크에서 효율적 탐색
- 희소 보상 환경 (태스크 완료 시에만 보상)에서 학습 가능
- 사전학습된 VLA의 내부 표현을 활용한 fine-tuning

### 주요 도전과제
- 논문은 소규모 환경(7×7 gridworld, 4×4 ant)에서만 검증
- VLA의 규모(수십억 파라미터)에서의 확장성 미검증
- 실제 로봇의 연속 관찰/행동 공간의 복잡성
- VLA의 멀티모달 입력(비전+언어)에 대한 metacontroller 적응 필요

---

## 8. 주요 수식 요약

| 수식 | 설명 |
|------|------|
| e_{t,l} ← e_{t,l} + U_t · e_{t,l} | Internal controller의 residual stream 개입 |
| z_t = β_t ⊙ z̃_t + (1-β_t) ⊙ z_{t-1} | Temporal integration (추상 행동 유지/전환) |
| L(φ) = Σ[-log p + α·D_KL] | Metacontroller 학습 목적함수 (ELBO) |
| U_t = f_hyp(z_t) | Controller decoder (하이퍼네트워크) |
| β_t = f_switch(e_{t,l}, h_{t-1}, z_{t-1}) | Switching unit |

---

## 9. 논문의 한계 및 향후 과제
- 소규모 제어 환경에서만 실험
- LLM 등 대규모 모델로의 확장 미검증
- 실제 로봇 환경에서의 검증 부재
- Metacontroller의 학습 안정성 및 수렴성에 대한 이론적 보장 부족
- 더 복잡한 태스크 구조(비선형 의존성, 병렬 서브태스크 등)에 대한 확장 필요
