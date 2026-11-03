# DOIM-Traffic-RL: Baseline Validation Report (Phase 19)

**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard Compliance:** Transportation Engineering Standards (NEMA TS 2) & Multi-Agent RL Parity  
**Document Version:** 2.0.0  
**Generated UTC:** 2026-11-03 18:00:00 UTC  

---

## 1. Overview of Benchmarked Controllers

To establish genuine scientific progress, DOIM and IA-MOPPO are benchmarked against six established classical and reinforcement learning controllers operating in identical microscopic simulation environments:

| Controller | Class | Decision Frequency | Action Mechanism | Theoretical Foundation |
| :--- | :---: | :---: | :--- | :--- |
| **Fixed-Time (FT)** | Classical | Deterministic Cycle | Fixed phase sequence & durations (27s green, 3s yellow) | Webster Cycle Theory (1958) |
| **Actuated Signal Control (ASC)** | Classical | Dynamic Loop Call | NEMA passage timers, min-green 10s, max-green 60s, gap-out 3s | NEMA TS 2 Transportation Standard |
| **Max-Pressure (MP)** | Classical Feedback | Step-wise Queue State | Serves phase maximizing downstream-upstream queue differential | Varaiya (2013) Network Stabilization |
| **MLP-PPO** | Learning Baseline | Step-wise State | Flattened state MLP without spatial GCN or temporal GRU | PPO without Inductive Priors |
| **Fixed-Weight PPO (FW-PPO)** | Learning Baseline | Step-wise State | Static linear scalarization with uniform objective weights $w_k = 0.2$ | Conventional MORL Scalarization |
| **State-Adaptive PPO (SA-PPO)** | Learning Baseline | Step-wise State | MLP mapping raw features to preference weights without interaction core | Direct Black-Box Preference Learning |
| **IA-MOPPO (Proposed)** | Proposed | Step-wise State | Bilinear interaction manifold with GCN-GRU corridor context & safety shield | Dynamic Objective Interaction Manifolds |

---

## 2. NEMA Actuated Controller Implementation & Audit

The forensic audit (Phase 14) identified that earlier actuated controllers had unused gap parameters. In Phase 19, the controller was redesigned to strictly conform to NEMA TS 2 standards:
- **Minimum Green:** $g_{\min} = 10\,\text{s}$ strictly enforced before passage timer evaluation.
- **Passage Timer / Gap Threshold:** $t_{\text{gap}} = 3.0\,\text{s}$. If no vehicle is detected on the active approach for $3.0\,\text{s}$ following minimum green, the phase gaps out ($a=1$).
- **Unit Extension:** Arriving vehicles reset the passage timer to $0.0$ and grant an additional $2.0\,\text{s}$ extension.
- **Maximum Green:** $g_{\max} = 60\,\text{s}$ max-out cap forces phase transition to prevent starvation of competing movements.

---

## 3. Varaiya Max-Pressure Controller Implementation

The network-wide Max-Pressure controller implements the classical queue differential:
$$P(p) = \sum_{(l, m) \in \text{Movements}(p)} \left( Q_l - Q_m \right)$$
where $Q_l$ is the queue length on incoming lane $l$, and $Q_m$ is the queue length on downstream receiving lane $m$.
At each junction, the controller selects the candidate signal phase maximizing pressure $P^*(p) = \arg\max_p P(p)$. If the highest-pressure phase differs from the active phase and minimum green is satisfied, a transition request ($a=1$) is issued.

---

## 4. Parameter-Matched MLP-PPO Architecture

To isolate the specific contribution of the spatial Kipf-Welling GCN and temporal GRU encoders, the parameter-matched MLP-PPO baseline matches the parameter capacity ($\sim 112\,\text{K}$ parameters):
- Input: Flattened state vector $\mathbf{x} \in \mathbb{R}^{70}$.
- Backbone: 3-layer MLP $[70 \to 128 \to 128 \to 128]$ with LayerNorm and ReLU activations.
- Policy Heads: Multi-discrete categorical heads identical to DOIM-Network.
- Critic Head: Linear value projection $V(s)$.
- Training Objective: Standard PPO clipped surrogate loss with GAE advantage estimation.
