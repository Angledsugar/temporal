# TempoRAL

**Discovering Action-Expert-Aware Subtask Boundaries via Internal Reinforcement Learning on Human Temporal Abstractions**

<p align="center">
  <img src="assets/temporal_overview.png" alt="TempoRAL Overview" width="800"/>
</p>

---

## Overview

Hierarchical Vision-Language-Action (VLA) systems decompose instructions into subtasks via a VLM planner, then execute each subtask with a lightweight action expert. A critical yet under-explored question remains: **at what granularity should subtasks be specified so the action expert can reliably execute them?**

**TempoRAL** addresses this by observing that temporal abstraction structures — where one subtask ends and another begins — are largely **embodiment-invariant**: a human grasping a cup and a robot grasping a cup share the same *reach → approach → close → lift* boundary pattern, differing only in low-level kinematics.

We propose a three-phase framework that:
1. **Pretrains** an action expert on large-scale human manipulation data to learn embodiment-invariant temporal priors
2. **Discovers** subtask boundaries from the frozen expert's residual stream via a self-supervised MetaController
3. **Optimizes** subtask granularity for task success through Internal RL in the abstract controller-code space

At deployment, the learned boundary predictor serves a dual purpose: providing a **capability-aware prior** for VLM prompting and enabling **runtime re-planning** when the action expert detects a subtask exceeds its capability.

## Key Insight

<table>
<tr>
<td width="50%">

**Subtask Granularity Mismatch**

| Type | Example | Problem |
|------|---------|---------|
| Too abstract | "Make coffee" | Expert fails — too many contact phases |
| Too detailed | "Rotate wrist 15°" | Wastes VLM budget on trivial segments |
| Misaligned | "Place cup under machine" | Doesn't respect expert's actual capability |

</td>
<td width="50%">

**TempoRAL's Solution**

The frozen action expert's internal representations already encode temporal structure. A small MetaController (~2M params) learns to discover where the expert naturally transitions between subtasks — **without any boundary supervision**.

The switching gate β_t learns sparse, quasi-binary patterns aligned with semantic action transitions.

</td>
</tr>
</table>

## Architecture

```
Phase 1 (θ pretrain) → Phase 2 (φ learn; θ frozen) → Phase 3 (ψ learn; θ,φ frozen) → Deploy
```

### Phase 1: Action Expert Pretraining

Pretrain a Gemma-300M transformer on human manipulation data using conditional flow matching:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \tau, \epsilon} \left\| v_\theta(a_\tau^{(t)}, t \mid s, q_\tau) - u_t(a_\tau^{(t)} \mid a_\tau) \right\|^2$$

**Data sources**: Something-Something V2 (40%), Ego4D (40%), UniHand (20%)

| Parameter | Value |
|-----------|-------|
| Width | 1024 |
| Depth | 18 layers |
| MLP dim | 4096 |
| Total params | ~311M |

**After this phase, θ is frozen permanently.**

### Phase 2: MetaController Discovery (Self-Supervised)

Extract the residual stream at layer 9 from the frozen expert, then train a MetaController with three components:

- **Encoder** (BiGRU): Processes full sequence → per-timestep latent statistics (μ_t, σ_t)
- **Switching Unit**: Predicts β_t ∈ [0,1] — switch to new controller or maintain current one
- **Decoder** (Hypernetwork): z_t → low-rank controller U_t = B_t A_t

```
Emergent β_t pattern (no boundary supervision):

Time:  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
β_t:   1  0  0  0  1  0  0  1  0   0   0   0   1   0   0
       ↑           ↑        ↑                  ↑
    subtask1    subtask2  subtask3          subtask4
    (reach)     (grasp)   (lift)            (place)
```

### Phase 3: Internal RL

Optimize subtask granularity via RL in the abstract controller-code space:

| Component | Definition |
|-----------|-----------|
| **State** | Residual stream e_{t,l} ∈ ℝ^1024 |
| **Action** | Controller code z_t ∈ ℝ^32 |
| **Reward** | Binary task success |

**Temporal contraction**: Instead of T decisions per episode, the RL policy acts only at M switching points (M << T), reducing search space by T/M (typically 12-50×).

### Deployment

```
User: "Make coffee"
       ↓
VLM + Capability-Aware Prompt
       ↓
["Pick up cup", "Place under machine", "Press button"]
       ↓
For each subtask:
  Action Expert + MetaController → Execute
  Monitor β_t:
    β_t < 0.5 → continue
    β_t ≥ 0.5 → subtask complete ✓
    β_t fires early → re-plan via VLM
```

## Project Structure

```
temporal/
├── models/
│   ├── action_expert.py          # Wraps OpenPi Gemma-300M, exposes residual stream
│   ├── metacontroller.py         # Encoder + Switching Unit + Decoder (~2M params)
│   └── internal_rl_policy.py     # Causal GRU policy for Phase 3
│
├── data/
│   ├── human_motion_dataset.py   # Human manipulation data loader
│   ├── retargeting.py            # Human hand → canonical EE representation
│   └── robot_dataset.py          # Robot demo loader (for Phase 3 evaluation)
│
├── training/
│   ├── phase1_trainer.py         # Flow-matching pretraining loop
│   ├── phase2_trainer.py         # Self-supervised MetaController training
│   └── phase3_trainer.py         # Internal RL training loop (REINFORCE)
│
├── envs/
│   └── internal_env.py           # Internal MDP (state=residual, action=controller code)
│
├── deploy/
│   ├── vlm_interface.py          # External VLM API (Gemma 3 / Gemini / GPT)
│   ├── capability_prompt.py      # Capability-aware system prompt generation
│   ├── replanner.py              # Runtime re-planning on β_t trigger
│   └── pipeline.py               # Full deployment pipeline
│
├── utils/
│   ├── visualization.py          # β_t plots, z_t t-SNE
│   └── metrics.py                # NMI, switching accuracy, TCR, success rate
│
├── configs/
│   ├── phase1_pretrain.yaml
│   ├── phase2_metacontroller.yaml
│   ├── phase3_internal_rl.yaml
│   └── deploy.yaml
│
└── scripts/
    ├── train_phase1.py
    ├── train_phase2.py
    ├── train_phase3.py
    ├── deploy.py
    └── evaluate.py
```

## Installation

```bash
pip install -r temporal/requirements.txt
```

**Dependencies**: PyTorch ≥ 2.0, NumPy ≥ 1.24, Gymnasium ≥ 0.29, PyYAML, scikit-learn, matplotlib

## Usage

### Training

```bash
# Phase 1: Pretrain action expert on human data
python scripts/train_phase1.py \
    --config temporal/configs/phase1_pretrain.yaml \
    --data /path/to/human_motion_data \
    --output checkpoints/phase1/

# Phase 2: Train MetaController (self-supervised boundary discovery)
python scripts/train_phase2.py \
    --config temporal/configs/phase2_metacontroller.yaml \
    --expert checkpoints/phase1/action_expert.pt \
    --data /path/to/demonstration_data \
    --output checkpoints/phase2/

# Phase 3: Internal RL (subtask granularity optimization)
python scripts/train_phase3.py \
    --config temporal/configs/phase3_internal_rl.yaml \
    --expert checkpoints/phase1/action_expert.pt \
    --meta checkpoints/phase2/metacontroller.pt \
    --env sim_manipulation \
    --output checkpoints/phase3/
```

### Deployment

```bash
python scripts/deploy.py \
    --config temporal/configs/deploy.yaml \
    --vlm gemini-pro \
    --expert checkpoints/phase1/action_expert.pt \
    --meta checkpoints/phase2/metacontroller.pt \
    --policy checkpoints/phase3/rl_policy.pt \
    --task "make a cup of coffee"
```

### Evaluation

```bash
python scripts/evaluate.py \
    --expert checkpoints/phase1/action_expert.pt \
    --meta checkpoints/phase2/metacontroller.pt \
    --policy checkpoints/phase3/rl_policy.pt \
    --env sim_manipulation \
    --metrics success_rate switching_nmi tcr
```

## Key Design Decisions

1. **Frozen expert is critical**: Co-training θ with φ causes temporal abstractions to collapse. Freezing θ creates an information bottleneck that forces β_t to capture genuine temporal structure.

2. **Embodiment-invariant transfer**: Temporal boundary structures (reach → grasp → lift → place) are shared across embodiments, enabling pretraining on abundant human data.

3. **Residual stream intervention**: Rather than modifying model weights, the MetaController applies additive control on residual streams: e' = e + U·e. This adds only ~2M parameters on top of the frozen 311M expert.

4. **Non-causal → Causal transition**: Phase 2 uses BiGRU (sees future) for optimal boundary discovery. Phase 3 replaces this with a causal GRU for online execution.

## References

- Physical Intelligence. *π₀: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
- Physical Intelligence. *π₀.₅: A Vision-Language-Action Model with Open-World Generalization.* 2025.
- Kobayashi et al. *Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning.* arXiv:2512.20605, 2025.

## License

This project is for research purposes.
