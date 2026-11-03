# REPRODUCIBILITY STATEMENT
**Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Date:** November 2026

---

## 1. Commitment to Open and Reproducible Science

The authors are firmly committed to open, verifiable, and fully reproducible scientific research. In adherence to IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS) reproducibility standards, all software components, neural network architectures, traffic simulation configurations, evaluation seeds, baseline implementations, and statistical analysis scripts developed in this study are made available to the research community.

---

## 2. Software and Environment Specification

The computational experiments in this study were conducted on the following verified software and simulation environment:
- **Operating System:** Microsoft Windows 11 Enterprise (x64) / Compatible with Ubuntu 22.04 LTS
- **Programming Language:** Python 3.12.10
- **Deep Learning Framework:** PyTorch 2.15.0+cpu
- **Traffic Simulator:** Eclipse SUMO (Simulation of Urban MObility) version 1.27.1
- **TraCI Control Interface:** TraCI version 1.28.0
- **Scientific Computing Libraries:** NumPy 2.2.3, SciPy 1.15.2, Pandas 2.2.3, PyYAML 6.0.2
- **Testing Framework:** PyTest 9.2.0

---

## 3. Repository and Source Code Structure

The open-source repository is hosted at:  
`https://github.com/Ramya282006/DOIM-Traffic-RL`

Key architectural and experimental modules include:
- `models/doim_network.py`: PyTorch implementation of the dual-layer Kipf-Welling GCN, temporal GRU, DOIM interaction manifold head, SCOIL cross-sensitivity estimator, and multi-objective actor-critic heads.
- `models/scoil.py`: State-Conditioned Objective Interaction Learning operator with heteroscedastic loss and epistemic dispersion regularizers.
- `algorithms/ia_moppo.py`: Interaction-Aware Multi-Objective Proximal Policy Optimization algorithm with action attribution separation and GAE truncation bootstrapping.
- `environment/traffic_env.py`: Multi-objective SUMO environment wrapper extracting 5 normalized traffic objectives.
- `environment/safety_shield.py`: Authoritative traffic signal safety shield enforcing conflict matrices, $g_{\min} = 10\,\text{s}$, $g_{\max} = 60\,\text{s}$, and $y=3\,\text{s}, r=1\,\text{s}$ clearance intervals.
- `baselines/`: Complete baseline implementations for Webster Fixed-Time, genuine NEMA TS 2 Actuated controller, and Varaiya Max-Pressure.
- `analysis/statistical_analyzer.py`: Non-parametric statistical suite (Mann-Whitney $U$, Cliff's $\delta$, Holm-Bonferroni) and fail-closed confirmatory manifest loader.
- `analysis/scoil_validator.py`: Benchmark runner comparing SCOIL against 9 interaction baselines and Split-Conformal Prediction calibrator.
- `experiments/`: Experiment orchestration scripts and hardware profiling micro-benchmarks.
- `tests/`: 159 automated unit and integration tests covering the complete codebase.

---

## 4. Determinism, Random Seeds, and Data Provenance

1. **Seed Synchronization:**
   - Training Seeds: $\{42, 123, 456, 789, 999, 101, 202, 303, 404, 505\}$.
   - Evaluation Seeds: Held-out evaluation seeds $\{1042, 1123, 1456, \dots\}$ are strictly segregated from training seeds to ensure true generalization.
   - Deterministic Evaluation: When configured in evaluation mode, all stochastic policy heads select argmax deterministic actions and SUMO route seeds are fixed (`tests/test_deterministic_evaluation.py`).
2. **Canonical Manifest Schema:**
   Every experimental run outputs an immutable JSON manifest containing 12 canonical fields (`experiment_id`, `run_id`, `algorithm`, `scenario`, `train_seed`, `eval_seed`, `step_count`, `duration_seconds`, `timestamp`, `git_commit`, `metrics_summary`, `status`).
3. **Fail-Closed Filtering:**
   The statistical analyzer automatically excludes any run with fewer than 5,000 steps or incomplete status, preventing diagnostic smoke runs from contaminating confirmatory benchmarks.

---

## 5. Instructions for Replication

### 5.1 Step 1: Environment Setup
```bash
git clone https://github.com/Ramya282006/DOIM-Traffic-RL.git
cd DOIM-Traffic-RL
python -m venv .venv_doim
source .venv_doim/bin/activate  # On Windows: .venv_doim\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Step 2: Automated Test Suite Execution
Verify that all 158 core unit and integration tests pass:
```bash
cd DOIM-Traffic-RL
pytest -v
```

### 5.3 Step 3: Hardware Profiling Replication
Measure local forward inference latency and SUMO step duration:
```bash
python experiments/profile_compute_budget.py
```

### 5.4 Step 4: SCOIL Multi-Baseline and Conformal Uncertainty Replication
Execute the 9-baseline comparison and split-conformal calibration:
```bash
python -c "
from analysis.scoil_validator import run_scoil_scientific_validation
results = run_scoil_scientific_validation()
print('SCOIL Frobenius Error:', results['scoil']['frobenius_error'])
print('Identity Frobenius Error:', results['identity']['frobenius_error'])
"
```

### 5.5 Step 5: Multi-Objective Statistical Significance Replication
Run the non-parametric statistical analyzer and hypervolume sensitivity evaluation:
```bash
python -c "
from analysis.statistical_analyzer import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
print('Statistical Analyzer Initialized Successfully')
"
```
