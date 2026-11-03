# DOIM-Traffic-RL: Confirmatory Experimental Protocol (Phase 20)

**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard Compliance:** Strict Experimental Reproducibility & Fail-Closed Benchmark Eligibility  
**Document Version:** 2.0.0  
**Generated UTC:** 2026-11-03 18:05:00 UTC  

---

## 1. Experimental Tier Architecture

To maintain strict scientific integrity and prevent preliminary or incomplete data from contaminating publication benchmarks, DOIM-Traffic-RL establishes four discrete experimental execution tiers:

| Tier Name | Step Budget | Purpose | Benchmark Eligibility | Quarantine Policy |
| :--- | :---: | :--- | :---: | :--- |
| **`smoke`** | 128 steps | Rapid system integration, CI, & TraCI socket check | **STRICTLY INELIGIBLE** | Quarantined; excluded from all benchmark tables |
| **`pilot`** | 5,000 steps | Hyperparameter tuning & learning rate validation | **INELIGIBLE** | Quarantined; diagnostic use only |
| **`intermediate`** | 10,000 steps | Comparative convergence & ablation profiling | **ELIGIBLE (Intermediate)** | Retained in tier-specific ablation tables |
| **`confirmatory`** | 50,000 steps | Asymptotic convergence & main paper claims | **FULLY ELIGIBLE** | Mandatory for confirmatory benchmark tables |

---

## 2. Benchmark Eligibility Invariants (Schema v2.1.0)

Every candidate run is evaluated by `is_confirmatory_benchmark_eligible(manifest)` in `common/experiment.py`:
1. **Tier Invariant:** `experiment_tier == "confirmatory"`.
2. **Budget Invariant:** `training_timesteps >= 5000`.
3. **Horizon Invariant:** Evaluation horizon must equal the full episode duration ($1800\,\text{s}$).
4. **Determinism Invariant:** Evaluation mode must be `deterministic_eval` (greedy policy argmax).
5. **Seed Independence:** Evaluation seed must differ from training seed: $\text{seed}_{\text{eval}} = \text{seed}_{\text{train}} + 1000$.
6. **Provenance Invariant:** Model weights must have a verified 64-character SHA-256 hash.
7. **Algorithm Whitelist:** Preregistered algorithms only (`DOIM_IA_MOPPO`, `Fixed-Time`, `Actuated`, `Max-Pressure`, `MLP-PPO`, `FW-PPO`, `SA-PPO`).

---

## 3. Seed Hierarchy & Independent Statistical Units

In accordance with rigorous statistical practices in deep reinforcement learning:
- **Primary Statistical Unit:** Independent training seed ($\ge 5$ seeds required, target 10 seeds: $\{42, 123, 456, 789, 999, 101, 202, 303, 404, 505\}$).
- **Separation of Evaluation:** Evaluation trajectories are executed on held-out seeds $\{1042, 1123, 1456, \dots\}$ to test generalization rather than memorized vehicle arrivals.
- **Deduplication:** Runs with identical `(algorithm, scenario, train_seed, episode_id)` are deduplicated, keeping the latest certified run.
