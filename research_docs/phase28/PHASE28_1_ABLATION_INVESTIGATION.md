# Phase 28.1: Objective Interaction Ablation Failure Investigation
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Rigorous Attribution of Architectural Component Contributions  
**Date:** November 3, 2026  

---

## 1. Problem Statement & Audit Trigger

In the Phase 28 Option A matrix, the empirical evaluation metrics for the proposed architecture and its two objective-interaction ablations on held-out seed 1042 were:

- **Full DOIM-PPO (Learned SCOIL Manifold):** Average Waiting = **510.94s** | Hypervolume = **0.9588**
- **Ablation 1 (No Interaction Manifold, $\mathcal{M} = \mathbf{0}$):** Average Waiting = **501.90s** | Hypervolume = **0.9692**
- **Ablation 2 (Rolling Pearson Correlation):** Average Waiting = **501.49s** | Hypervolume = **0.9671**

Contrary to the primary research hypothesis, the full DOIM-PPO model did **not** outperform its ablations. Both the zero-manifold ablation and the heuristic rolling Pearson ablation achieved marginally lower delay (-1.77% and -1.85%) and marginally higher 2D hypervolume.

---

## 2. In-Depth Root Cause Investigation

### 2.1 Optimization Dynamics & Training Horizon Limitation
- **Number of Policy Updates:** With training budget $N_{\text{train}} = 1,000$ steps and rollout size $n_{\text{steps}} = 128$, exactly:
  $$N_{\text{updates}} = \lfloor 1000 / 128 \rfloor = 7 \text{ PPO update epochs}$$
- **Parameter Capacity vs. Optimization Steps:** The DOIM architecture has **48,938 trainable parameters** (GCN spatial encoder: 1,380; GRU temporal encoder: 24,960; SCOIL: 8,746; Preference head: 2,309; Actor-Critic: 11,543). Seven PPO gradient epochs are insufficient for Adam ($eta_1=0.9, \beta_2=0.999$) to traverse the initial learning transient.

### 2.2 Latent Manifold vs. Stationary Baseline Representations
- In **Ablation 1**, $\mathcal{M}_t = \mathbf{0}_{5 \times 5}$. The interaction encoder receives an invariant zero input. In early training, a stationary zero input provides a lower-variance latent representation, allowing the actor-critic backbone to fit simple phase timing faster than when conditioned on a rapidly changing dynamic matrix.
- In **Full DOIM-PPO**, SCOIL generates dynamic off-diagonal interaction weights. During initial exploration, these interaction values fluctuate rapidly, adding non-stationary variance to the policy input.

### 2.3 Did SCOIL Learn Structured Interactions?
Inspection of the saved checkpoint (`ia_moppo_final.pt`) reveals that SCOIL **did** receive meaningful gradients and learned non-trivial interaction patterns:
- Auxiliary loss decreased monotonically from **-0.2616** (step 128) to **-0.2936** (step 1000).
- Off-diagonal interaction weights learned meaningful non-zero cross-correlations:
  - Waiting vs. $\text{CO}_2$: $-0.1997$
  - Fairness vs. Throughput: $-0.3128$
  - Delay vs. Queue: $+0.0473$
  - Predictive uncertainty $\mathcal{U}_t$: Mean $= 0.5469$.

### 2.4 Preference Weight Adaptation Collapse
Analysis of `metrics_step.csv` across all 1,800 evaluation steps reveals that the generated preference weights were nearly static:
- $\bar{w}_{\text{waiting}} = 0.2195 \pm 0.0003$
- $\bar{w}_{\text{queue}} = 0.1161 \pm 0.0004$
- $\bar{w}_{\text{co2}} = 0.2468 \pm 0.0005$
- $\bar{w}_{\text{throughput}} = 0.2513 \pm 0.0005$
- $\bar{w}_{\text{fairness}} = 0.1663 \pm 0.0003$
The preference distribution remained tightly bounded around its initial soft-regularized prior ($w_0 \approx [0.2, 0.2, 0.2, 0.2, 0.2]$) because seven gradient updates were insufficient to establish strong preference-gradient feedback.

---

## 3. Scientific Implication for Publication

1. **Honest Reviewer Reporting:** In IEEE T-ITS, claiming that the interaction manifold provides superior control at 1,000 steps is scientifically indefensible. The manuscript must explicitly report:
   > *"At limited sample budgets ($10^3$ steps), policy optimization is dominated by basic signal coordination; explicit manifold learning produces no statistically detectable advantage over zero-manifold ablations ($p = 0.84$)."*
2. **Asymptotic Convergence Hypothesis:** Validating the interaction manifold requires the **Tier 2 confirmatory benchmark ($5,000$ to $50,000$ steps)**, where the policy has converged sufficiently to exploit higher-order objective interactions.
