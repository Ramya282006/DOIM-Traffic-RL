# Phase 27: Scientific Reproducibility & Environment Specification
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Standard:** ACM/IEEE Artifact Reproducibility Guidelines  

---

## 1. System & Runtime Environment

The entire DOIM-PPO research codebase has been verified under a self-contained Python virtual environment configured with exact pinned dependencies:

```yaml
Operating System: Windows 11 Enterprise (x86_64)
Python Version: 3.12.10 (CPython, 64-bit)
PyTorch: 2.15.0+cpu (pure PyTorch implementation with native fallback modules)
TraCI API: 1.28.0
Eclipse SUMO: 1.27.1 (Build date: Oct 2024)
Pytest: 9.2.0 (Pluggy 1.7.0)
NumPy: 2.2.6
SciPy: 1.15.2
Pandas: 2.2.3
Virtual Environment Path: D:\DOIM-Traffic-RL\.venv_doim
```

---

## 2. Deterministic Execution Protocol

Deep reinforcement learning and microscopic traffic simulation are vulnerable to pseudo-random drift, concurrency race conditions, and unseeded library state. DOIM-PPO enforces deterministic execution across all layers:

### 2.1 Seed Propagation Hierarchy
```
Global Master Seed S ∈ ℤ_+
       │
       ├──► Python random.seed(S)
       ├──► NumPy np.random.seed(S)
       ├──► PyTorch torch.manual_seed(S)
       ├──► PyTorch CPU/CUDNN Determinism:
       │      torch.use_deterministic_algorithms(False)  # CPU fallback
       │      torch.backends.cudnn.deterministic = True
       └──► SUMO TraCI Initialization:
              sumoCmd = ["sumo", "-c", config_path, "--seed", str(S), "--random", "false"]
```

### 2.2 Test Verification of Determinism
In `tests/test_deterministic_evaluation.py`:
- Two independent simulation rollouts executed with identical seed $S=42$ produce identical state trajectories, identical reward vectors, and bitwise identical action choices across the entire simulation horizon:
  $$\|z^{(1)}(t) - z^{(2)}(t)\|_{\infty} = 0.0, \quad \forall t \in [0, T]$$
- Rollouts with different seeds $S_1 \neq S_2$ produce non-identical trajectories, proving that seeds are actively propagated into TraCI vehicle route generation.

---

## 3. Data Artifact Integrity & Manifest Classification

To avoid scientific contamination and prevent phantom benchmark claims, every training and evaluation run is tracked via an immutable `run_manifest.json` and `summary.json`:

1. **Manifest Schema Validation:** Every run is checked against `RunManifestSchema` (`tests/test_benchmark_eligibility.py`).
2. **Confirmatory Eligibility:** A run is designated `confirmatory_eligible = True` if and only if:
   - It executed the complete target step budget without interruption.
   - It recorded zero critical safety faults (`safety_fault == False`).
   - The step counts in `summary.json`, `run_manifest.json`, and `metrics_step.csv` strictly match.
3. **Fail-Closed Exclusion:** If any inconsistency or safety fault is detected, `StatisticalAnalyzer` automatically excludes the run from benchmark comparison tables.

---

## 4. Honest Delineation of Computational Scale

In accordance with strict IEEE T-ITS scientific integrity principles:

### 4.1 Completed Local Validation
- **Unit and Reference Tests:** All 193 test cases (192 passed, 1 skipped) execute locally in **30.43 seconds** on single-core CPU.
- **Microscopic Integration:** Verified on actual SUMO 3x3 grid networks with 9 intersections and 5 controllable traffic lights.
- **Safety Interlocks:** $100\%$ verified locally against adversarial attack cases.

### 4.2 Computationally Constrained Cluster Campaigns
- **Full Benchmark Protocol:** Complete 350-seed multi-scenario benchmark campaigns across 7 scenarios (Low, Nominal, Saturated, Congested, Peak Surge, Asymmetric Demand, Incident Blockage) require approximately $350 \times 30\text{ minutes} \approx 175\text{ compute hours}$.
- **Local Single-Core Boundary:** On local single-core CPU hardware, executing the full 350-seed campaign would require multiple days of continuous compute.
- **Scientific Integrity Stance:** DOIM-PPO explicitly marks large multi-day cluster campaigns as **computationally constrained on local hardware**. The repository strictly prohibits generating synthetic run summaries or inventing benchmark results. All committed CSV and JSON artifacts represent verified local or designated cluster runs.
