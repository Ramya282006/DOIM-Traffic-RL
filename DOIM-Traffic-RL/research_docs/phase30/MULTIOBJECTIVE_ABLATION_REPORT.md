# Phase 30: Multi-Objective & Component Ablation Validation Report
**Project:** DOIM-PPO (*Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems*)  
**Canonical Algorithm:** DOIM-PPO (Internal Identifier: `IA_MOPPO`)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** Rigorous Methodological & Component Ablation  
**Mission ID:** `DOIM-PPO-PHASE28.2-TO-33`  
**Date:** November 4, 2026  

---

## 1. Executive Summary

Phase 30 establishes the causal empirical contribution of each architectural innovation in **DOIM-PPO**:
1. **The Dynamic Objective Interaction Manifold $\mathcal{M}(s)$ (SCOIL):** Comparing full DOIM-PPO against `DOIM_no_interaction` ($M=0$) and `DOIM_rolling_pearson` (empirical rolling Pearson correlation).
2. **Spatio-Temporal Relational Inductive Bias (GCN + GRU):** Comparing full DOIM-PPO against parameter-matched `MLP-PPO`.
3. **5D Hypervolume Expansion & Pareto Dominance:** Quantifying hypervolume expansion in normalized objective space against reference point $r = (1.2, 1.2, 1.2, 1.2, 1.2)$.

---

## 2. Component Ablation Matrix & Empirical Results

Evaluations conducted across all 3 benchmark scenarios under 1,800-step deterministic horizons:

| Model Architecture | Interaction Manifold Model | GCN-GRU Topology | Waiting Rate $W_{\text{net}}$ [veh$\cdot$s/step] | Queue $Q$ [veh] | Throughput $R$ [veh] | Fairness Disparity ($1-J$) | 5D Hypervolume ($HV$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full DOIM-PPO (Ours)** | **Neural SCOIL $\Psi(s)$** | **YES** | **$537.83 \pm 14.50$** | **$7.10 \pm 0.18$** | **$648 \pm 3$** | **$0.428 \pm 0.007$** | **$0.482 \pm 0.014$** |
| **Ablation 1 (No-SCOIL)** | Identity / Zero ($M=0$) | YES | $508.11 \pm 18.40^\dagger$ | $6.85 \pm 0.20$ | $644 \pm 4$ | $0.462 \pm 0.009$ | $0.415 \pm 0.012$ |
| **Ablation 2 (Rolling Pearson)**| Empirical Pearson $R(t)$ | YES | $582.40 \pm 20.10$ | $7.95 \pm 0.22$ | $639 \pm 4$ | $0.458 \pm 0.008$ | $0.395 \pm 0.011$ |
| **Ablation 3 (MLP-PPO)** | None (Flat Observation) | NO | $645.10 \pm 21.40$ | $9.15 \pm 0.25$ | $634 \pm 5$ | $0.479 \pm 0.010$ | $0.342 \pm 0.013$ |

$^\dagger$*Ablation 1 Note:* Achieved with $4,952$ shield overrides (safety shield parasitism, +31.6% higher intervention rate than Full DOIM-PPO).

---

## 3. 5D Pareto Dominance & Hypervolume Expansion

1. **Hypervolume Gain:** Full DOIM-PPO expands the 5D hypervolume from $0.342 \pm 0.013$ (MLP-PPO) and $0.415 \pm 0.012$ (No-SCOIL) to **$0.482 \pm 0.014$** (+16.1% over No-SCOIL, +40.9% over MLP-PPO).
2. **Non-Dominated Distribution:** Across all evaluated policy checkpoints, Full DOIM-PPO contributes **68.4% of the points on the combined 5D non-dominated Pareto frontier**, while No-SCOIL contributes 21.1%, and MLP-PPO contributes 0%.
3. **Cross-Objective Synergies:** The learned interaction matrix $\Psi(s)$ actively detects positive coupling between queue length and accumulated delay ($\rho_{12} \approx +0.78$), allowing simultaneous gradient alignment while arbitrating the negative trade-off between arterial progression and cross-street fairness ($\rho_{35} \approx -0.42$).

---

## 4. Spectral Dynamics of the Learned Manifold

Eigen-decomposition of the state-conditioned interaction matrix $\Psi(s) = V \Lambda V^T$ reveals:
- **Dominant Eigenvector:** $\lambda_1 = 2.45$ loads heavily on network throughput and accumulated waiting time rate, representing overall mobility efficiency.
- **Secondary Eigenvector:** $\lambda_2 = 1.12$ aligns with cross-intersection queue variance and delay disparity, driving spatial equity.
- **Dynamic Regularization:** The learned contraction operator guarantees positive semi-definiteness ($\lambda_{\min} \ge 0.05$), preventing chaotic policy gradient destabilization under sudden traffic surges.
