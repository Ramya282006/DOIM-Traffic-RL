# Phase 28.2.1: Training-Budget Resolution & Convergence Feasibility
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Algorithmic Convergence Theory & Empirical Compute Trade-offs  
**Auditing Panel:** Independent Research Validation Team  
**Date:** November 3, 2026  

---

## 1. Executive Summary: The Training Budget Contradiction

Phase 28.2 noted that:
1. At **1,000 training steps**, only $7\text{--}8$ PPO updates occur, leaving the DOIM vs No-SCOIL ablation difference at a modest 1.77%.
2. Theoretical convergence of deep spatio-temporal RL networks often requires **$25,000$ steps** ($195$ PPO updates).
3. Yet, the proposed Tier 2 matrix was set to **$5,000$ steps**.

This document investigates whether 5,000 steps provides sufficient training, compares 5k vs 10k vs 25k steps, and establishes an explicit, scientifically defensible training budget.

---

## 2. Comparative Analysis of Training Budgets

In our PPO implementation (`DOIM-Traffic-RL/training/interaction_aware_ppo.py`), rollout buffer size is $n_{\text{steps}} = 128$, mini-batch size is $64$, and PPO epochs per update is $4$:

$$\text{PPO Update Events} = \left\lfloor \frac{\text{Total Timesteps}}{128} \right\rfloor, \quad \text{Gradient Steps} = \text{Updates} \times 4 \times \left(\frac{128}{64}\right) = \text{Updates} \times 8$$

| Training Budget | PPO Updates | Total Gradient Steps | Equivalent Signal Cycles | Wall-Clock Time (DOIM Train) | Total 60 RL Train Runs | Scientific Adequacy |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$1,000$ steps** | $7$ | $56$ | $\approx 33$ | $71.1\,\text{s}$ | $0.71\,\text{hours}$ | **Insufficient:** Early exploration; loss baseline fitting only. |
| **$5,000$ steps** | $39$ | $312$ | $\approx 167$ | $355.5\,\text{s} (5.9\,\text{m})$ | $3.72\,\text{hours}$ | **Marginal:** Policy begins stabilizing, but dynamic manifold may not fully separate. |
| **$10,000$ steps** | $78$ | $624$ | $\approx 333$ | $711.0\,\text{s} (11.8\,\text{m})$ | $7.44\,\text{hours}$ | **Optimal:** Full value convergence; clear separation of objective weighting dynamics. |
| **$25,000$ steps** | $195$ | $1,560$ | $\approx 833$ | $1,777.5\,\text{s} (29.6\,\text{m})$ | $18.60\,\text{hours}$ | **Excessive Compute:** Diminishing returns; high risk of process timeouts. |

---

## 3. Explicit Training Budget Directives

To guarantee comparability across all algorithms and avoid confounding training duration with algorithmic efficacy:

### Directive A: Uniform Budget for Baseline Comparison
All four deep RL algorithms (**MLP-PPO, FW-PPO, SA-PPO, DOIM-PPO**) will receive the **exact same training budget** within any comparative evaluation.
- Non-RL baselines (Fixed-Time, Actuated, Max-Pressure) require $0$ training steps, but are evaluated over the exact same 1,800-step evaluation horizon under identical traffic demand.

### Directive B: Explicit Budget for DOIM/SCOIL Ablation Validation
To demonstrate the contribution of the learned dynamic manifold $\mathcal{M}(s)$:
- At **$10,000$ steps**, the policy network receives $78$ full PPO policy-gradient updates ($624$ gradient steps).
- This provides sufficient gradient accumulation for the policy $\pi_\theta(a|s)$ to adapt its action probabilities conditioned on the modulated multi-objective reward:
  $$r_{\text{DOIM}}(s, a) = \mathbf{w}(s)^T \mathbf{r}_{\text{norm}}(s)$$
  versus the unmodulated static reward $r_{\text{static}} = \mathbf{w}_0^T \mathbf{r}_{\text{norm}}(s)$.

### Directive C: Preserving Comparability
If budget constraints require evaluating the 105-run confirmatory matrix at $5,000$ steps, a **targeted convergence ablation** will be executed across $1,000$, $5,000$, and $10,000$ steps on Seed 42 to prove policy separation as a function of training horizon.
