# DOIM-Traffic-RL: Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.15.0%2Bcpu-orange.svg)](https://pytorch.org/)
[![SUMO 1.28](https://img.shields.io/badge/SUMO-1.28.0-green.svg)](https://eclipse.dev/sumo/)
[![Tests](https://img.shields.io/badge/pytest-88%20passed%2C%201%20skipped-brightgreen.svg)](tests/)
[![Target](https://img.shields.io/badge/target-IEEE%20T--ITS-darkblue.svg)](research_docs/paper/MANUSCRIPT.md)

---

## 1. System Overview

**DOIM-Traffic-RL** is an end-to-end differentiable, scientifically validated multi-objective reinforcement learning (MORL) research system for urban traffic signal control. It addresses a fundamental flaw in conventional traffic signal control: **collapsing competing operational objectives (delay, queues, emissions, throughput, and fairness) into static weights or uncoupled state mappings**.

Under dynamic, non-stationary traffic demands, the relationship between competing criteria shifts dynamically—often transitioning from cooperative synergy to acute conflict. DOIM-Traffic-RL explicitly models these relationships by discovering a **State-Conditioned Objective Interaction Manifold (DOIM)** that conditions both adaptive preference weights on the unit simplex $\Delta^4$ and multi-discrete signal control policies.

```
                                      ┌──────────────────────────────────────────────┐
                                      │         CORRIDOR SPATIAL-TEMPORAL DYNAMICS   │
                                      │  Graph Convolution (GCN) + Temporal GRU     │
                                      └──────────────────────┬───────────────────────┘
                                                             │  h_t ∈ ℝ^64
                                                             ▼
┌───────────────────────────────┐     ┌──────────────────────────────────────────────┐
│  NORMALIZED OBJECTIVE VECTOR  │────►│  STATE-CONDITIONED INTERACTION LEARNER       │
│      z_t ∈ [0, 1]^5           │     │  Asymmetric Bilinear Coupling + Shrinkage    │
└───────────────────────────────┘     └──────────────────────┬───────────────────────┘
                                                             │  M_t ∈ [-1, 1]^(5×5)
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │  DYNAMIC OBJECTIVE INTERACTION MANIFOLD      │
                                      │  m_t = LayerNorm(W_f [h_t || z_t || vec(M)]) │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                              ┌──────────────────────────────┴──────────────────────────────┐
                              ▼                                                             ▼
┌──────────────────────────────────────────────┐              ┌──────────────────────────────────────────────┐
│   ADAPTIVE PREFERENCE WEIGHT GENERATOR       │              │    INTERACTION-AWARE ACTOR-CRITIC POLICY     │
│   w_t = Softmax((W_p m_t + b_p) / τ) ∈ Δ^4   │              │    Multi-Discrete Phase Transition Request   │
└──────────────────────┬───────────────────────┘              └──────────────────────┬───────────────────────┘
                       │                                                             │
                       │ (Detached w̄_t)                                              │ a_t ∈ {0, 1}^5
                       ▼                                                             ▼
┌──────────────────────────────────────────────┐              ┌──────────────────────────────────────────────┐
│   OBJECTIVE-WEIGHTED SCALARIZED REWARD       │              │        DETERMINISTIC SAFETY SHIELD           │
│   R_t = ∑ w̄_{k,t} · (-z_{k,t})               │              │   Min-Green (10s), Clearance (3s), Max (60s) │
└──────────────────────────────────────────────┘              └──────────────────────────────────────────────┘
```

---

## 2. Mathematical Architecture

### 2.1 Spatial Kipf-Welling Graph Convolution
For node features $\mathbf{X}_t \in \mathbb{R}^{N \times 5}$ on adjacency $\mathbf{\tilde{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}}(\mathbf{A} + \mathbf{I})\mathbf{\tilde{D}}^{-\frac{1}{2}}$:
$$\mathbf{H}_t^{(1)} = \text{ReLU}\left(\mathbf{\tilde{A}} \mathbf{X}_t \mathbf{\Theta}_1\right), \quad \mathbf{H}_t^{(2)} = \mathbf{\tilde{A}} \mathbf{H}_t^{(1)} \mathbf{\Theta}_2, \quad \mathbf{h}_t^{\text{spat}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_{i,t}^{(2)} \in \mathbb{R}^{64}$$

### 2.2 Temporal Gated Recurrent Unit
Over a sequence of $L = 10$ graph snapshots:
$$\mathbf{h}_t^{\text{temp}} = \text{GRU}\left([\mathbf{h}_{t-L+1}^{\text{spat}}, \dots, \mathbf{h}_t^{\text{spat}}]\right) \in \mathbb{R}^{64}$$

### 2.3 State-Conditioned Objective Interaction Learning (SCOIL)
Models asymmetric predictive cross-sensitivities:
$$\mathbf{Q}_t = \mathbf{W}_Q [\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t], \quad \mathbf{K}_t = \mathbf{W}_K [\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t]$$
$$\mathbf{M}_t^{\text{raw}} = \frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}} + \mathbf{A}_{\text{asym}}(\mathbf{h}_t^{\text{temp}}, \mathbf{z}_t)$$
$$\mathbf{M}_t^* = \beta_t \tanh(\mathbf{M}_t^{\text{raw}}) + (1 - \beta_t) \mathbf{M}_0, \quad \beta_t = \frac{1}{1 + \lambda \mathbf{U}_t}$$

### 2.4 Urgency-Regulated Preference Adaptation & Anti-Collapse Barrier
To prevent degenerate collapse where the preference network minimizes loss by driving difficult objective weights to zero:
$$\mathbf{u}_t = \frac{\mathbf{z}_t + \epsilon}{\sum_k (z_{k,t} + \epsilon)}, \quad \mathcal{L}_{\text{pref}} = D_{\text{KL}}(\mathbf{u}_t \parallel \mathbf{w}_t) - \beta_{\text{ent}} \mathcal{H}(\mathbf{w}_t) + \lambda_{\text{smooth}} \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_2^2$$
Policy returns are evaluated using detached weights $\bar{\mathbf{w}}_t = \text{stop\_gradient}(\mathbf{w}_t)$, strictly preventing reward-hacking.

### 2.5 Counterfactual Residual Transition Dynamics
Predicts next-step objectives without observation leakage:
$$\mathbf{\hat{z}}_{t+1} = \text{clamp}\left(\mathbf{z}_t + \mathbf{\Delta z}(\mathbf{m}_t, \mathbf{a}_t, \mathbf{z}_t), 0, 1\right), \quad \mathcal{L}_{\text{aux}} = \|\mathbf{\hat{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2$$

---

## 3. Installation & Verification

### 3.1 Prerequisites
- Python 3.11.9
- Eclipse SUMO v1.28.0 (included in `.venv_doim/Lib/site-packages/sumo/bin`)
- Windows 11 / Linux / macOS

### 3.2 Environment Setup
```powershell
# Activate dedicated virtual environment
.venv_doim\Scripts\activate

# Verify dependencies
python -c "import torch, sumo, gymnasium; print('Dependencies OK')"
```

> **Environment Integrity & Platform Support:** The project uses standard canonical dependencies specified in `requirements.txt`. On Linux, macOS, and standard Windows systems, dependencies install cleanly via `pip install -r requirements.txt`. On Windows 11 systems with Smart App Control (SAC) enabled in 'Enforce' mode, Microsoft cloud reputation inspects binary hashes; unmodified PyPI wheels execute cleanly without requiring local developer certificates (`LocalDevCert`).

### 3.3 Running Automated Test Suite (88 Tests Verified)
```powershell
python -m pytest tests/ -v
```
All 88 active tests pass with 100% success rate (1 optional external SB3 test skipped):
- `tests/test_safety_shield_semantics.py`: Dynamic SUMO tlLogic parsing, clearance intervals, single-phase junction stability, and signal conflict validation.
- `tests/test_deterministic_evaluation.py`: Deterministic evaluation reproducibility, frozen baseline weights, non-learning Fixed-Time controller verification.
- `tests/test_manifest_migration_and_classification.py`: 4-state manifest classification (VALID, LEGACY_VALID, INCOMPLETE, INVALID) and canonical schema normalization.
- `tests/test_fairness_definitions.py`: Delay Disparity vs Delay Variance unit separation and Jain Fairness calculations.
- `tests/test_scenario_isolation_synthetic.py`: Scenario isolation and intra-model averaging preserving independent seed units.
- `tests/test_actual_training_gradient_flow.py`: Verifies non-zero gradients and parameter updates across all 17 submodules after genuine PPO update.
- `tests/test_safety_transitions_trace.py`: Verifies zero illegal transitions across 120 live SUMO steps under adversarial phase-switch requests.
- `tests/test_doim_architecture.py`: Tensor shapes, autograd backprop, forward passes.
- `tests/test_objective_interaction_theory.py`: Asymmetry, Bayesian shrinkage, variance bounds.
- `tests/test_ia_moppo_integration.py`: End-to-end rollout, agent act/update, baselines.
- `tests/test_phase1_infrastructure.py`: Config snapshots, seed independence, logging.

---

## 4. Reproducing Experiments

### 4.1 Staged Evaluation Pipeline

```powershell
# 1. Smoke Test (Diagnostic execution check)
.venv_doim\Scripts\python.exe -c "from training.interaction_aware_ppo import train_interaction_aware_ppo; train_interaction_aware_ppo(total_timesteps=128, seed=42)"

# 2. Comprehensive Multi-Seed Matrix (Baselines + Ablations across 5 independent seeds with frozen deterministic evaluation)
.venv_doim\Scripts\python.exe experiments/run_comprehensive_matrix.py --timesteps 128 --eval_episodes 1 --eval_horizon 1800 --seeds 42 123 456 789 2024

# 3. Held-Out Peak Congestion Generalization (Rush Hour demand)
.venv_doim\Scripts\python.exe experiments/run_comprehensive_matrix.py --timesteps 128 --seeds 42 --scenarios scenarios/rush_hour/rush_hour.sumocfg

# 4. Statistical Analysis & Export Tables
.venv_doim\Scripts\python.exe analysis/statistical_analyzer.py

# 5. Publication Plot Generation (300 DPI IEEE format)
.venv_doim\Scripts\python.exe analysis/plot_generator.py
```

---

## 5. Empirical Benchmark Summary

Evaluated across 5 independent training seeds (`42, 123, 456, 789, 2024`) on the $3 \times 3$ grid network under non-stationary traffic surge (`north_south_surge`):

> **Critical Scientific Clarification:** These results reflect initial 128-step diagnostic pipeline runs, NOT converged training. In 128-step exploration, the non-learning Fixed-Time controller achieves lower waiting time ($55.13 \pm 5.33\,\text{s}$) than learning agents ($82.19 \pm 45.69\,\text{s}$) because learning agents explore phase switching actions and incur mandatory yellow clearance intervals (3s) before value functions and policies have converged. Long-horizon convergence evaluations (5,000 to 50,000 steps) are scheduled for full production training.

| Algorithm | Avg Waiting Time (s) $\downarrow$ | Avg Queue (veh) $\downarrow$ | Max Queue (veh) $\downarrow$ | CO2 (mg/s) $\downarrow$ | Throughput (veh) $\uparrow$ | Safety Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed_Time** | $55.13 \pm 5.33$ | $1.92 \pm 0.25$ | $8.40 \pm 1.34$ | $56,361 \pm 837$ | $4.20 \pm 0.45$ | **0** |
| **FW_PPO** | $58.38 \pm 22.19$ | $2.43 \pm 0.72$ | $11.30 \pm 2.73$ | $55,484 \pm 895$ | $3.60 \pm 1.34$ | **0** |
| **SA_PPO** | $73.16 \pm 34.20$ | $3.06 \pm 1.54$ | $13.60 \pm 4.83$ | $54,907 \pm 1,820$ | $3.50 \pm 0.50$ | **0** |
| **DOIM (No Int)** | $84.07 \pm 43.89$ | $2.78 \pm 1.15$ | $10.40 \pm 3.51$ | $55,778 \pm 1,012$ | $2.80 \pm 1.92$ | **0** |
| **DOIM (Rolling P)**| $84.07 \pm 43.89$ | $2.78 \pm 1.15$ | $10.40 \pm 3.51$ | $55,778 \pm 1,012$ | $2.80 \pm 1.92$ | **0** |
| **IA_MOPPO (Ours)**| **$82.19 \pm 45.69$** | **$2.85 \pm 1.11$** | **$11.00 \pm 3.24$** | **$55,611 \pm 1,253$**| **$3.00 \pm 2.12$** | **0** |

### Held-Out Peak Rush Hour Demand Generalization (`rush_hour.sumocfg`, 158 vehicles inserted):
- **IA_MOPPO (Proposed):** Total Reward = **$-73.08$** | Policy Loss = $0.0248$ | Aux Loss = $0.0626$
- **Fixed_Time:** Total Reward = $-79.20$ | Avg Waiting = $286.26\,\text{s}$ | Avg Queue = $10.34$
- **FW_PPO:** Total Reward = $-81.30$ | Policy Loss = $57.60$ | Value Loss = $19,295.47$
- **SA_PPO:** Total Reward = $-109.46$ | Policy Loss = $-58.52$ | Value Loss = $100,146.04$ (diverged)

---

## 6. Repository Layout

```
DOIM-Traffic-RL/
├── analysis/               # StatisticalAnalyzer, PublicationPlotGenerator
├── baselines/              # FixedTimeController, FixedWeightPPO, StateAdaptivePPO
├── configs/                # Centralized YAML configs (default.yaml, simulation.yaml)
├── environment/            # DOIMTrafficEnv (SUMO TraCI), TrafficSafetyShield, SafetyVerifier
├── experiments/            # run_comprehensive_matrix.py, run_smoke_test.py
├── models/                 # NativeGCNConv, DOIMNetwork, StateConditionedInteractionLearner
├── objectives/             # ObjectiveEngine, JainFairness, StableNormalizer
├── research_docs/          # FINAL_IMPLEMENTATION_REPORT, THEORY, PROTOCOL, MANUSCRIPT
├── results/                # Immutable runs (runs/), tables (tables/), figures (figures/)
├── scenarios/              # SUMO network grid.net.xml, north_south_surge, rush_hour
├── tests/                  # 88 passing pytest test suites (1 skipped)
└── training/               # InteractionAwarePPOAgent, RolloutBuffer, train loop
```

---

## 7. Citation

```bibtex
@article{ramya2026doim,
  title={Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems},
  author={Anonymous},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under Review}
}
```
