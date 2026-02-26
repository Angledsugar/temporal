# TempoRAL: 사람 시연으로부터의 Cross-Embodiment 시간적 추상화 전이 — VLM Backbone에서의 Internal Reinforcement Learning

**저자**: Anonymous Author (Anonymous Institution)

---

## 초록

$\pi_{0.5}$, GR00T 등 Vision-Language-Action(VLA) 시스템은 VLM(Vision-Language Model) backbone이 지시를 서브태스크로 분해하고, action expert가 이를 실행하는 계층적 아키텍처를 채택하고 있다.
본 연구는 VLM backbone — 자기회귀(autoregressive), 인과적(causal) transformer — 이 residual stream에 조작 서브태스크의 **시간적으로 추상화된 표현(temporally-abstract representations)**을 발달시키며, 이 표현이 **embodiment-invariant(신체 구조에 무관)**하다는 가설을 제시한다: 사람이 컵을 잡는 것과 로봇이 컵을 잡는 것은 운동학적으로 크게 다르지만, reach→approach→close→lift라는 동일한 경계 패턴을 공유한다.

이 통찰을 바탕으로 **TempoRAL**이라는 3단계 프레임워크를 제안한다:
1. VLM backbone을 사람 조작 시연 데이터로 파인튜닝하여 조작 특화 시간 구조를 내부 표현에 인코딩한 후 동결
2. 동결된 VLM backbone의 residual stream에서 자기지도 meta-controller를 학습하여 어떠한 annotation 없이 서브태스크 경계를 발견
3. 발견된 추상 행동 공간에서 Internal Reinforcement Learning을 적용하여 인과적 정책이 희소 보상 하에서 새로운 서브태스크 시퀀스를 구성할 수 있도록 함

핵심 ablation 실험에서는 base VLM(사전학습만)과 사람 데이터로 파인튜닝한 VLM을 비교하여, 사람 시연이 시간적 추상화 형성에 기여하는 정도를 분리한다.
이는 기존 연구에서 보고된 표현 수준의 cross-embodiment 정렬을 넘어, 조작의 **시간적 구조** 자체를 사람에서 로봇으로 전이하는 새로운 형태의 *시간적 추상화 전이(temporal abstraction transfer)*를 구성한다.

---

## 1. 서론

### 배경 및 동기

$\pi_0$, $\pi_{0.5}$, GR00T N1 등 최근 VLA 모델들은 자연어 지시에 조건화될 때 다양한 조작 태스크에 걸쳐 일반화할 수 있는 단일 정책을 시연하였다.
이 모델들은 공통된 2단 구조를 공유한다: 시각 관측과 언어 지시를 처리하여 고수준 계획을 형성하는 **VLM backbone**(자기회귀, 인과적 transformer)과, 이 계획을 flow matching 또는 diffusion을 통해 저수준 모터 명령으로 변환하는 **action expert**.
$\pi_{0.5}$가 이를 잘 보여준다: VLM이 먼저 지시를 서브태스크("컵 잡기", "컵을 기계 아래에 놓기")로 분해하고, action expert가 50Hz로 실행한다.

아직 충분히 탐구되지 않은 핵심 질문은 VLM backbone이 **내부 시간적 추상화(internal temporal abstractions)** — 하나의 조작 서브태스크가 끝나고 다른 것이 시작되는 시점의 구조화된 표현 — 를 발달시키며, 이를 명시적 annotation 없이 발견하고 활용할 수 있는가 하는 것이다.

Kobayashi et al. [Internal RL]은 최근 next-token prediction으로 학습된 자기회귀 모델이 residual stream 활성화에서 이러한 시간적 추상화를 자발적으로 발달시킨다는 것을 보여주었다.
Meta-controller — residual stream을 읽고 쓰는 작은 하이퍼네트워크 — 는 순수하게 자기지도 방식으로 이 추상화를 발견할 수 있다: switching gate $\beta_t$가 의미론적으로 의미 있는 경계(예: 서브골 전환)에서 발화하는 것을 annotation 없이 학습한다.
더 나아가 이 추상 행동 공간에서 직접 RL을 수행하면("Internal RL") 희소 보상 하에서 토큰 수준 RL을 크게 능가한다.

### 핵심 관찰: 왜 Action Expert가 아닌 VLM Backbone인가

이 접근법의 핵심 전제조건은 base model이 **자기회귀적이고 인과적(autoregressive and causal)**이어야 한다는 것이다 — 시간적 추상화는 장기 수평에 걸친 순차적 next-token prediction에서 출현한다.

VLA 시스템에서 이 조건을 충족하는 것은 **VLM backbone**이다:
- GR00T의 Eagle-2: 28개 인과적 transformer 레이어
- $\pi_0$의 PaliGemma: 18개 인과적 transformer 레이어
- 에피소드 길이의 관측 시퀀스를 자기회귀적으로 처리

반면 **action expert**는 이 조건을 충족하지 않는다:
- **Diffusion 또는 flow matching** 사용 (자기회귀가 아님)
- **양방향 어텐션(bidirectional attention)** 사용 (인과적이 아님)
- 짧은 action chunk(0.1~1초)만 처리 → 수초~수십초 단위의 서브태스크 전환을 포착할 수 없음
- 실행기("어떻게 움직일 것인가")이지, 계획기("다음에 무엇을 할 것인가")가 아님

이 구분은 실증적 증거로도 뒷받침된다:
- GR00T N1은 **중간 레이어** VLM 임베딩이 최종 레이어 임베딩보다 downstream 정책 성공률이 높다는 것을 발견
- $\pi_{0.5}$는 이미 VLM 내에서 서브태스크 텍스트를 명시적으로 예측

### 사람 데이터 가설

**사람 시연 데이터**가 VLM backbone 내에서 조작 관련 시간적 추상화를 형성하는 데 결정적 역할을 한다고 가설을 세운다.
조작의 시간적 경계 구조 — 하나의 원시 행동이 끝나고 다음이 시작되는 시점 — 는 대체로 **embodiment-invariant**하다: 사람이 컵을 잡는 것과 로봇이 컵을 잡는 것은 reach→approach→close→lift라는 동일한 패턴을 공유하며, 저수준 운동학에서만 차이가 난다.

Physical Intelligence의 $\pi_{0.5}$ 분석은 충분한 규모에서 사람과 로봇의 잠재 표현이 VLM의 임베딩 공간에서 자발적으로 합쳐진다는 것을 시연했다.
그러나 이 정렬이 **시간적 추상화** — 정적 표현이 아닌 서브태스크 간 **동적 스위칭 구조** — 로까지 확장되는지는 **미검증** 상태이다.

### 기여

**TempoRAL** 프레임워크의 기여:
1. VLM backbone을 사람 조작 시연으로 **파인튜닝**하여 조작 특화 시간 구조를 인코딩한 후 **동결** (§3.1)
2. 동결된 VLM backbone의 residual stream에서 자기지도 meta-controller로 서브태스크 경계를 **비지도 발견** — annotation 불필요 (§3.2)
3. 저차원 controller-code 공간에서 Internal RL을 통해 서브태스크 구성을 **최적화** — 희소 보상 하 새로운 서브태스크 시퀀스 가능 (§3.3)
4. Base VLM(사전학습만) vs 사람 데이터 파인튜닝 VLM의 비교 **ablation** — cross-embodiment *시간적 추상화 전이*에 대한 최초의 정량적 증거 제공 (§4)

---

## 2. 관련 연구

### 2.1 Vision-Language-Action 모델

VLA 패러다임은 시각 인식, 언어 이해, 모터 제어를 단일 모델로 통합한다.
RT-2는 VLM을 파인튜닝하여 로봇 행동을 텍스트 토큰으로 출력할 수 있음을 최초로 보여주었다.
$\pi_0$는 flow-matching action head를 도입하여 동결된 PaliGemma backbone과 함께 50Hz 연속 제어를 가능하게 했다.
$\pi_{0.5}$는 계층적 서브태스크-행동 아키텍처로 이를 확장하여 open-world 일반화를 시연했다.
$\pi_0$-FAST는 flow matching을 FAST 토큰화로 대체하여 5배 빠른 학습을 달성했다.
OpenVLA와 Octo는 오픈소스 대안을 제공한다.
GR00T N1은 공유 Diffusion Transformer 주위에 per-embodiment MLP adapter를 도입하여 단일 모델로 다양한 인간형 형태를 제어할 수 있게 했다.

**이 모든 시스템은 서브태스크-행동 인터페이스를 고정된 설계 선택으로 취급하며, VLM backbone의 내부 시간적 표현을 활용하여 데이터로부터 서브태스크 경계를 발견하지 않는다.**

### 2.2 계층적 강화학습과 시간적 추상화

Options 프레임워크는 RL에서 시간적 추상화를 시작 집합, 내부 정책, 종료 조건을 가진 정책으로 공식화했다.
Option-Critic은 options를 end-to-end로 학습한다.
HIRO와 HAM은 목표 조건 하위 정책을 사용한다.
CompILE은 variational inference를 통해 세그먼트를 발견한다.

가장 최근에 **Internal RL**이 제안되어, 동결된 자기회귀 모델의 residual stream에서 학습된 meta-controller가 ground-truth 서브골과 정렬되는 시간적 추상화를 발견하고, 결과적 추상 행동 공간에서의 RL이 희소 보상 하 토큰 수준 RL을 크게 능가함을 보여주었다. Internal RL은 gridworld와 MuJoCo 이동에서만 시연되었으며, **VLA 시스템의 VLM backbone과 실세계 로봇 조작에서의 잠재력은 미탐구 상태**이다.

### 2.3 사람 데이터로부터의 로봇 정책 학습

로봇 시연 데이터는 희소하다 (Open X-Embodiment에서 ~$10^5$ 궤적). 이에 비해 사람 상호작용 데이터는 훨씬 풍부하다 (Ego4D ~$10^8$ 샘플, UniHand, Something-Something V2 등).
LAPA는 사람 비디오에서 VQ-VAE를 통해 잠재 행동을 학습하여 로봇 제어에 전이하며, 30배 적은 연산으로 OpenVLA를 능가한다.
MT-$\pi$는 motion track을 cross-embodiment 행동 표현으로 사용하여 86.5% 실세계 성공률을 달성한다.
Being-H0는 1.65억 사람 손 샘플로 dexterous VLA를 학습한다.
Physical Intelligence의 $\pi_{0.5}$ 분석은 모델 용량이 증가할 때 사람 손과 로봇 그리퍼의 잠재 클러스터가 자발적으로 합쳐지는 **창발적** 정렬을 밝혔다.

**기존 연구는 *저수준 행동*이나 *시각적 특징*을 전이하지만, 본 연구는 VLM backbone의 residual stream에서 비지도로 발견된 *시간적 추상화 구조*를 전이한다 — 이는 더 추상적이므로 더 embodiment-invariant하다.**

핵심적으로, 기존 연구가 사람과 로봇의 *표현*이 규모에서 수렴함을 보여주었지만, 이 정렬이 *시간적 추상화* (서브태스크 간 동적 스위칭 구조)로까지 확장되는지는 미검증이다.

### 2.4 서브태스크 분해와 태스크 계획

SayCan, Inner Monologue, Code-as-Policies 등 LLM 기반 플래너는 지시를 실행 가능한 단계로 분해한다.
이들은 일반적으로 *고정된* affordance 모델이나 성공 탐지기에 의존하여 세분화 수준을 결정한다.
Voyager는 스킬 라이브러리를 학습하지만 분해를 실행기의 능력에 연결하지 않는다.

**기존 방법 중 *계획기 자체의 내부 시간적 추상화*로부터 분해 세분화 수준을 학습하는 것은 없다.**

---

## 3. TempoRAL 프레임워크

TempoRAL은 3단계 순차 학습으로 구성된다 (그림 1).
VLM backbone의 파라미터를 $\theta$, meta-controller의 파라미터를 $\phi$, Internal RL 정책의 파라미터를 $\psi$로 표기한다.

```
Phase 1 (VLM θ 파인튜닝 후 동결) → Phase 2 (Meta-controller φ 학습; θ 동결) → Phase 3 (RL 정책 ψ 학습; θ,φ 동결)
```

### 3.1 Phase 1: VLM Backbone의 사람 시연 데이터 파인튜닝

Phase 1의 목표는 VLM backbone에 사람 시연 데이터를 통해 조작 특화 시간 구조를 부여하고, 이후 모든 단계에서 **동결**하는 것이다.

#### 왜 Action Expert가 아닌 VLM Backbone인가

시간적 추상화는 장기 시간 시퀀스를 처리하는 *자기회귀적, 인과적* 모델에서만 출현할 수 있다 [Internal RL].
- **VLM backbone**: Eagle-2 (28 인과 transformer 레이어), PaliGemma (18 인과 레이어) → 에피소드 길이 관측-행동 시퀀스를 자기회귀적으로 처리 → **조건 충족**
- **Action expert**: Diffusion/flow matching + 양방향 어텐션 + 짧은 action chunk (0.1~1초) → 수초 단위 서브태스크 전환의 시간적 추상화 형성 불가능

#### 왜 사람 데이터인가

**규모**: 로봇 시연 데이터셋은 ~$10^5$ 궤적. 사람 조작 데이터셋은 수 자릿수 더 큼 (Ego4D: 3,670시간; YouTube: 사실상 무한).

**Embodiment-invariant 시간 구조**: 저수준 운동학은 다르지만, *시간적 경계 구조*는 공유된다. "컵 잡기": 사람과 로봇 모두 reach → pre-grasp → close → lift를 실행하며, 접촉 개시, 힘 폐합, 수직 가속 시점에서 전환이 발생.

**창발적 cross-embodiment 정렬**: $\pi_{0.5}$ 분석에서 충분한 규모에서 사람 손과 로봇 그리퍼의 VLA 잠재 표현이 VLM 임베딩 공간에서 자발적으로 합쳐짐.

#### 아키텍처 및 학습

VLM backbone은 사전학습된 인과 transformer (Eagle-2: 폭 $n_e = 2048$, 깊이 28; 또는 PaliGemma: 폭 $n_e = 2048$, 깊이 18).
사람 조작 시연에 대해 관측-행동 시퀀스의 next-token prediction으로 파인튜닝:

$$\mathcal{L}(\theta) = \sum_{t=1}^{T} \Big[ -\ln p_\theta(a_t \mid o_{1:t}) - \lambda \ln p_\theta(o_{t+1} \mid o_{1:t}) \Big]$$

여기서 $o_t$는 시각 관측, $a_t$는 행동, $\lambda \geq 0$은 월드 모델 형성을 촉진하는 보조 관측 예측 손실의 가중치이다.
파인튜닝은 **LoRA**로 수행하여 학습 파라미터를 전체의 1% 미만으로 유지하면서 조작 특화 시간적 지식을 주입한다.

이 단계 후 **$\theta$는 동결**된다. base model과 meta-controller의 공동 학습은 시간적 추상화를 붕괴시킨다 (원 논문의 rate-distortion 분석으로 입증).

#### Ablation 설계

사람 데이터의 기여를 분리하기 위해 복수 조건을 고려한다:

| 조건 | VLM 상태 | 파인튜닝 데이터 |
|------|---------|-------------|
| **A (Base)** | 사전학습만 | 없음 |
| **B (Human)** | + LoRA 파인튜닝 | 사람 시연 |
| **C (Robot)** | + LoRA 파인튜닝 | 로봇 시연 |

- A vs B: 사람 데이터가 VLM backbone에서 시간적 추상화 형성을 강화하는지 분리
- B vs C: 사람 시연이 로봇 데이터만큼 (혹은 그 이상으로) 효과적인 시간적 추상화를 생성하는지 검증 → cross-embodiment 전이 가설의 핵심

---

### 3.2 Phase 2: Meta-Controller 학습 (자기지도)

이 단계의 목표는 동결된 VLM backbone의 내부 표현으로부터, 어떠한 경계 annotation 없이 **서브태스크 경계가 자연스럽게 발생하는 위치**를 발견하는 것이다.

#### Residual Stream 추출

시연 궤적 $(o_{1:T}, a_{1:T})$가 주어지면, 동결된 VLM backbone을 통해 forward pass를 수행하고 선택된 레이어 $l$에서 residual-stream 활성화를 추출한다:

$$e_{t,l} = \text{ResidualStream}_\theta^{(l)}(o_{1:t}) \in \mathbb{R}^{n_e}$$

여기서 $n_e = 2048$ (VLM hidden dimension). GR00T N1의 중간 레이어 관찰에 기반하여, 중간 깊이 근처의 $l$을 선택한다 (예: Eagle은 28개 중 14번째, PaliGemma는 18개 중 9번째).
VLM backbone이 인과적 자기회귀 transformer이므로, 전체 에피소드 길이 관측 시퀀스를 처리한다 — 시간적 추상화가 출현하는 것으로 입증된 바로 그 설정이다.

#### Meta-controller 아키텍처

Meta-controller $\phi$는 [Internal RL]의 아키텍처를 따르며 3가지 구성 요소로 이루어진다:

**인코더 (양방향 GRU)**: VLM residual-stream 활성화의 *전체* 시퀀스를 처리(비인과적 접근, 미래 정보 사용)하여 타임스텝별 잠재 통계를 생성:

$$h_t = \text{BiGRU}_\phi(e_{1:T,l})$$
$$\mu_t = W_\mu h_t + b_\mu \in \mathbb{R}^{n_z}$$
$$\sigma_t^2 = \text{softplus}(W_\sigma h_t + b_\sigma) \in \mathbb{R}^{n_z}$$
$$\tilde{z}_t \sim \mathcal{N}(\mu_t, \text{diag}(\sigma_t^2))$$

여기서 $n_z \ll n_e$는 controller-code 차원 (일반적으로 $n_z = 32$).

**Switching Unit**: 연속 게이트 $\beta_t \in [0,1]$이 새 controller code를 채택할지 이전 것을 유지할지 결정:

$$\beta_t = \sigma(W_\beta [e_{t,l}; h_t; z_{t-1}] + b_\beta)$$
$$z_t = \beta_t \odot \tilde{z}_t + (1 - \beta_t) \odot z_{t-1}$$

- $\beta_t \approx 0$: 이전 controller 유지 (동일 서브태스크)
- $\beta_t \approx 1$: 새 controller code 샘플링 (서브태스크 경계)
- $\beta_t$는 명시적 정규화 없이도 ground-truth 서브골 변경에 정렬된 준이진(quasi-binary), 희소 스위칭 패턴을 학습

**디코더 (하이퍼네트워크)**: Controller code $z_t$를 저순위 선형 컨트롤러 $U_t$로 디코딩하여 additive intervention으로 residual stream을 수정:

$$U_t = B_t A_t, \quad e'_{t,l} = e_{t,l} + U_t e_{t,l}$$

여기서 순위 $r = 32$, 학습 파라미터 ~2M (동결된 VLM backbone 1.7~2B 대비 매우 작음).

#### 학습 목적함수

제어된 VLM residual stream 하에서 행동 예측 로그 우도의 variational 하한을 최소화:

$$\mathcal{L}(\phi) = \sum_{t=1}^{T} \Big[ \underbrace{-\ln p_{\theta,\phi}(a_t \mid o_{1:t}, z_{1:t})}_{\text{행동 예측}} + \alpha \underbrace{D_{\text{KL}}(\mathcal{N}(\mu_t, \sigma_t^2) \| \mathcal{N}(0, I))}_{\text{사전 정규화}} \Big]$$

$\alpha$가 rate-distortion 트레이드오프를 제어: 큰 $\alpha$는 더 거친(추상적) 경계, 작은 $\alpha$는 더 세밀한 분할.

**VLM backbone 동결이 필수적**: $\theta$를 $\phi$와 공동 학습하면 VLM이 controller의 영향을 "흡수"하여 $\beta_t$가 균일 스위칭으로 퇴화한다. $\theta$ 동결이 $\beta_t$가 진정한 시간 구조를 포착하도록 강제하는 정보 병목을 생성한다.

#### 창발적 서브태스크 경계

학습 후, $\beta_t$는 조작 단계 전환에 대응하는 희소, 준이진 발화 패턴을 보인다 — 경계 supervision 없이. 이 경계는 VLM backbone의 "하나의 일관된 행동 계획이 끝나고 다른 것이 시작되는 곳"에 대한 *내부적* 개념을 반영하며, 이것이 바로 cross-embodiment로 발견하고 전이하고자 하는 시간적 추상화 구조이다.

---

### 3.3 Phase 3: 새로운 태스크 구성을 위한 Internal RL

Phase 2는 자기지도 방식으로 시간 경계를 발견하지만, *태스크 성공*에 대해 최적화하지는 않는다.
Phase 3은 전체 VLA 시스템(VLM backbone + meta-controller + action expert + 환경)을 환경으로 취급하고, 추상 controller-code 공간에서 RL을 적용하여 이 루프를 닫는다.

#### Internal 환경

VLM backbone의 표현 내부에 상태와 행동 공간이 존재하는 "internal" MDP를 구성:

| 구성 요소 | 정의 |
|---------|------|
| **상태** | $o_t^{\text{int}} = e_{t,l}$ (VLM residual stream, 레이어 $l$) |
| **행동** | $z_t \in \mathbb{R}^{n_z}$ (controller code) |
| **보상** | 태스크 성공 시 1, 그 외 0 |

"환경 역학"의 구성:
1. 디코더가 $z_t$로부터 $U_t$ 생성
2. 동결된 VLM backbone이 제어된 residual stream을 action expert에 전파
3. Action expert가 모터 명령 생성
4. 물리적(또는 시뮬레이션) 세계가 명령 실행

RL 에이전트 관점에서 이 모든 것은 $z_t$를 $(o_{t+1}^{\text{int}}, r_t)$로 매핑하는 블랙 박스이다.

#### 이진화된 스위칭

Phase 3에서 연속 $\beta_t$는 Heaviside 계단 함수로 이산화:

$$\beta_t^{\text{bin}} = H(\beta_t - \beta_{\text{thresh}})$$

- $\beta_t^{\text{bin}} = 0$: 이전 $z_{t-1}$ 재사용, RL 정책 *쿼리하지 않음* → VLM backbone이 동일 추상 행동으로 계속
- $\beta_t^{\text{bin}} = 1$: RL 정책이 새 $z_t$ 발출
- **시간 축소 (temporal contraction)**: $T$ 원시 스텝의 궤적에 $M$개 스위치 포인트가 있으면 ($M \ll T$), RL 정책은 $M$번만 결정 → 탐색 공간 대폭 축소 + 신용 할당 개선

#### 인과적 정책

Phase 2의 비인과적 BiGRU 인코더를 과거와 현재만 관측 가능한 *인과적* 정책 $\pi_\psi$로 대체:

$$z_t \sim \pi_\psi(z_t \mid e_{1:t,l})$$

인과적 GRU + 가우시안 정책 헤드로 구현. Phase 2의 디코더(하이퍼네트워크) 가중치는 재사용 및 동결; $\psi$만 학습.

#### 정책 그래디언트

상대적 이점 추정을 통한 정책 그래디언트로 $\psi$ 최적화:

$$\nabla_\psi J = \mathbb{E} \Big[ \sum_{m=1}^{M} A_m \nabla_\psi \ln \pi_\psi(z_{t_m} \mid e_{1:t_m,l}) \Big]$$

$M$이 작으므로 (에피소드당 일반적으로 3~8), 그래디언트 분산이 토큰 수준 정책 그래디언트보다 극적으로 낮음 → 이것이 Internal RL의 핵심 이점.

#### Phase 3이 학습하는 것

RL 정책은 조작 서브태스크를 새로운 시퀀스로 구성하는 controller code $z_t$를 발출하는 것을 학습: 각 $z_t$가 VLM backbone의 계획 표현을 일관된 조작 단계를 통해 조종하고, 새 서브태스크가 시작되어야 할 때 정확히 스위치가 트리거된다.
$z_t$가 원시 행동 공간이 아닌 VLM의 추상 계획 공간에서 작동하므로, 사전학습 중 보지 못한 것을 포함한 새로운 서브태스크 조합이 controller-code 공간에서의 조합적 일반화를 통해 도달 가능해진다.

---

## 4. 제안 실험

사람 시연 데이터가 VLM backbone에 전이 가능한 시간적 추상화를 유도한다는 가설을 검증하기 위한 실험 프로토콜을 제시한다.

### 4.1 실험 설정

**VLM backbone**: (1) GR00T N1.6의 Eagle-2 VLM (Qwen3-1.7B, 28 레이어, $n_e = 2048$), (2) $\pi_0$의 PaliGemma VLM (Gemma 2B, 18 레이어, $n_e = 2048$). 비용 효율적 검증을 위해 Qwen2.5-VL-7B (28 레이어, $n_e = 3584$) + LoRA도 고려.

**사람 시연 데이터**: 3D 손 추적으로 end-effector 궤적이 추출된 자아중심(egocentric) 조작 비디오. 각 궤적은 관측-행동 시퀀스 $(o_{1:T}, a_{1:T})$.

**Meta-controller 설정**: Controller-code 차원 $n_z = 16$, 하이퍼네트워크 순위 $r = 32$, 중간 깊이 개입 ($l = L/2$), KL 가중치 $\alpha$를 $[0.001, 0.1]$ 범위에서 sweep.

### 4.2 Ablation: Base VLM vs 사람 파인튜닝 VLM

| 조건 | VLM 상태 | 파인튜닝 데이터 |
|------|---------|-------------|
| A (Base) | 사전학습만 | 없음 |
| B (Human) | + LoRA 파인튜닝 | 사람 시연 |
| C (Robot) | + LoRA 파인튜닝 | 로봇 시연 |

**지표 1: Linear probing 정확도** — 각 조건에서 동결된 VLM의 residual stream 각 레이어에서 선형 분류기를 학습하여 현재 ground-truth 서브태스크 라벨을 디코딩. 중간 레이어 probing 정확도가 높을수록 더 풍부한 시간적 추상화 표현.

**지표 2: $\beta_t$의 서브태스크 경계 정렬** — Meta-controller의 switching gate $\beta_t$와 ground-truth 서브태스크 전환 시점 간의 정렬을 정규화 상호 정보(NMI)로 측정. 진정한 전환에 정렬된 희소, 준이진 $\beta_t$ 패턴은 성공적 시간적 추상화 발견을 나타냄.

**지표 3: Internal RL 성공률** — 사전학습이나 meta-controller 학습 시 보지 못한 새로운 서브태스크 시퀀스에 대해 조건 A, B, C 간 Internal RL 성공률 비교. Raw-action RL (GRPO)과 시간적 추상화 없는 Internal RL ($\beta_t = 1$ 강제)도 baseline으로 비교.

### 4.3 예상 결과

- **A < B**: 사람 파인튜닝이 시간적 추상화 품질을 유의미하게 향상 → 사람 데이터가 VLM backbone에 조작 관련 시간 구조를 인코딩함을 검증
- **B ≈ C**: 사람과 로봇 시연이 유사한 시간적 추상화를 생성 → *cross-embodiment 시간적 추상화 전이*의 강한 증거
- **B > C**: 사람 데이터가 *우월한* 시간적 추상화를 생성 (더 큰 데이터 다양성 또는 더 자연스러운 조작 구조로 인해) → 특히 강한 기여

어떤 결과든 기존 연구에서 보고된 표현 수준 cross-embodiment 정렬을 넘어, *시간적 추상화* 수준으로 확장하는 새로운 실증적 증거를 제공한다.

---

## 5. 결론

**TempoRAL** — VLM backbone의 내부 표현을 통해 사람 시연에서 로봇 조작으로 시간적 추상화를 발견하고 전이하는 프레임워크를 제시하였다.

3가지 핵심 통찰:
1. VLM backbone(자기회귀, 인과적 transformer)이 VLA 시스템에서 시간적 추상화 발견의 올바른 위치이다 (diffusion 기반 action expert와 달리 [Internal RL]의 전제조건을 충족)
2. 동결된 VLM backbone의 residual stream에서 학습된 meta-controller가 supervision 없이 서브태스크 경계를 발견한다
3. 결과적 추상 행동 공간에서의 Internal RL이 희소 보상 하 새로운 서브태스크 시퀀스로의 조합적 일반화를 가능하게 한다

본 연구의 핵심 가설은 사람 시연 데이터가 VLM backbone에 **embodiment-invariant**한 조작 특화 시간적 추상화를 유도한다는 것이다 — 기존 연구에서 보고된 표현 수준 cross-embodiment 정렬을 넘어 조작의 *시간적 구조*로 확장.
제안된 ablation (base VLM vs 사람 파인튜닝 VLM vs 로봇 파인튜닝 VLM)은 이 주장에 대한 최초의 정량적 증거를 제공하도록 설계되었다.

### 향후 연구

1. Meta-controller의 $\beta_t$ 스위칭 신호를 *런타임 재계획 트리거*로 활용: 배포 시 $\beta_t$가 예기치 않게 발화하면 현재 서브태스크 계획이 무효화되었음을 신호하여 VLM이 수정된 분해를 생성
2. 학습된 시간적 추상화 구조를 capability-aware prompting prior로 증류하여 수동 프롬프트 엔지니어링 없이 VLM이 적절한 세분화 수준의 서브태스크를 생성하도록 안내
3. 추가 embodiment(사족보행, dexterous 손 등)로 확장하고 다양한 하드웨어 플랫폼에서 검증하여 cross-embodiment 시간적 추상화 전이의 일반성을 확인
