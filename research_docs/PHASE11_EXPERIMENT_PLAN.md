# PHASE 11 — EXPERIMENTAL EXECUTION PLAN & CONVERGENCE STAGING
**Research Project:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems  
**Target Publication:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Author:** Independent Senior Research Engineer & Strict IEEE T-ITS Reviewer  
**Output Destination:** `results/phase11_audit/`  
**Execution Status:** PROPOSED (Awaiting User Approval Before Launch)  

---

## 1. Staged Experimental Roadmap

To prevent wasted computational resources and guarantee scientifically defensible evidence, experiments are structured into five strictly gated stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  STAGE 1: SMOKE │ ──► │  STAGE 2: PILOT │ ──► │   STAGE 3: INT  │ ──► │  STAGE 4: MAIN  │ ──► │ STAGE 5: EXTEND │
│    128 Steps    │     │   5,000 Steps   │     │  10,000 Steps   │     │  50,000 Steps   │     │  100,000 Steps  │
│  (Passed 59/59) │     │ (PROPOSED HERE) │     │   (If Justified)│     │   (Full Bench)  │     │ (If Divergent)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

| Stage | Budget (Steps) | Scope | Purpose | Gating Criterion |
| :--- | :---: | :--- | :--- | :--- |
| **Stage 1: Smoke** | 128 | 1 seed, 1 scenario | Verify gradient flow, absence of NaNs, TraCI socket communication. | All unit tests pass; zero NaNs. **[PASSED]** |
| **Stage 2: Pilot** | **5,000** | **3 seeds, 1 scenario** | **First convergence check; verify whether IA-MOPPO learns to outperform Fixed-Time after clearance exploration penalty.** | **Reward monotonically improves; delay < Fixed-Time.** |
| **Stage 3: Intermediate** | 10,000 | 3 seeds, 2 scenarios | Verify multi-scenario stability and preference weight specialization. | Generalization without divergence. |
| **Stage 4: Main Benchmark**| 50,000 | 5 seeds, all baselines & ablations | Asymptotic convergence, statistical significance (p < 0.05 with Holm-Bonferroni). | Required for IEEE T-ITS submission. |
| **Stage 5: Extended** | 100,000 | 5 seeds | Contingency run only if 50,000 steps shows positive gradient but incomplete plateau. | Slope > 0 at step 50k. |

---

## 2. Stage 2 Pilot Specifications (5,000 Steps)

### 2.1 Experimental Matrix
- **Algorithms to Benchmark:**
  1. `Fixed_Time`: 60-second cycle (27s green N-S, 3s yellow, 27s green E-W, 3s yellow).
  2. `FW_PPO`: Fixed-Weight PPO ($\mathbf{w} = [0.2, 0.2, 0.2, 0.2, 0.2]^\top$).
  3. `SA_PPO`: State-Adaptive PPO (direct state-to-preference MLP).
  4. `IA_MOPPO`: Proposed Dynamic Objective Interaction Manifold + SCOIL.
- **Traffic Scenario:** `scenarios/directional_shift/north_south_surge.sumocfg` (9 intersections, 5 controllable multi-phase traffic lights).
- **Independent Training Seeds ($N=3$):** `42`, `123`, `456`.
- **Decoupled Evaluation Seeds ($N=3$):** `1042`, `1123`, `1456`.
- **Evaluation Protocol:**
  - Checkpoints saved every 2,500 steps (`model_step_2500.pt`, `model_step_5000.pt`).
  - At step 5,000, model parameters are frozen (`eval()`, `torch.no_grad()`).
  - Evaluated over 1 full simulation episode ($1,800\,\text{s}$) with deterministic argmax action selection.
  - Tracking 8 primary metrics: Cumulative Reward, Average Waiting Time ($\text{s}$), Average Queue ($\text{veh}$), $\text{CO}_2$ ($\text{mg/s}$), Throughput ($\text{veh}$), Jain Delay Disparity ($[0, 1]$), Shield Interventions (count), and Preference Weight simplex trajectories ($\mathbf{w}_t \in \Delta^4$).

### 2.2 Baseline Fairness Guarantees
- **Interaction Budget:** Every RL agent interacts with SUMO for exactly 5,000 simulation steps.
- **Observation Space:** All algorithms receive the identical spatial node feature matrix $\mathbf{X}_t \in \mathbb{R}^{9 \times 5}$ and local intersection states $\mathbf{s}_t \in \mathbb{R}^{25}$.
- **Action Space:** Identical multi-discrete action space $\prod_{j=1}^5 \{0, 1\}$ subjected to the identical `TrafficSafetyShield`.
- **Traffic Demand:** Identical SUMO route files and insertion schedules across all algorithms for each corresponding seed.

---

## 3. Computational Requirements and Runtime Estimates

### 3.1 Hardware Environment
- CPU: AMD64 Multi-Core Processor (Windows 11)
- RAM: 16 GB Available
- GPU: CPU-fallback (PyTorch CPU execution)

### 3.2 Benchmarked Execution Speed
- Eclipse SUMO execution on `grid.net.xml`: $\approx 35\text{--}40\,\text{simulation steps per second}$ in headless TraCI mode.
- Neural network forward/backward pass (GCN + GRU + PPO update): $\approx 10\text{--}15\,\text{ms}$ per batch update.
- Per 5,000-step training run: $\approx 2.5\text{--}3.0\,\text{minutes}$ per seed per algorithm.

### 3.3 Resource Budget for Stage 2 Pilot
- **Total Runs:** 4 algorithms $\times$ 3 seeds = 12 training runs + 12 evaluation runs.
- **Estimated Total Runtime:**
  - Training: $12 \times 2.8\,\text{min} \approx 33.6\,\text{minutes}$.
  - Frozen Evaluation (1,800 steps per model): $12 \times 0.9\,\text{min} \approx 10.8\,\text{minutes}$.
  - Statistical aggregation and plot generation: $\approx 1.0\,\text{minute}$.
  - **Total Estimated Wall-Clock Time:** $\mathbf{\approx 45\text{--}50\,\text{minutes}}$.
- **Disk Footprint:**
  - Checkpoints: $12 \times 2 \times 460\,\text{KB} \approx 11\,\text{MB}$.
  - Step logs: $12 \times 1.2\,\text{MB} \approx 14.4\,\text{MB}$.
  - Total Storage: $< 30\,\text{MB}$ in `results/phase11_audit/runs/`.

---

## 4. Proposed Pilot Execution Commands

### Prerequisites (Automated via Remediation Script)
Before running the pilot, the remediation patch resolves DEF-P11-03, DEF-P11-04, and DEF-P11-05.

### Command Line Invocation
```powershell
# Execute the 5,000-step Stage 2 Pilot into isolated audit directory
.venv_doim\Scripts\python.exe experiments/run_comprehensive_matrix.py `
    --timesteps 5000 `
    --seeds 42 123 456 `
    --scenarios scenarios/directional_shift/north_south_surge.sumocfg `
    --eval_episodes 1 `
    --checkpoint_interval 2500 `
    --output_root results/phase11_audit/runs `
    --no_ablations
```

### Analysis & Plot Generation Commands
```powershell
# Generate isolated Phase 11 scenario-specific benchmark table
.venv_doim\Scripts\python.exe -c "from analysis.statistical_analyzer import StatisticalAnalyzer; a = StatisticalAnalyzer(runs_root='results/phase11_audit/runs'); a.export_summary_tables(output_dir='results/phase11_audit/tables')"

# Generate isolated publication plots
.venv_doim\Scripts\python.exe -c "from analysis.plot_generator import PublicationPlotGenerator; p = PublicationPlotGenerator(runs_root='results/phase11_audit/runs', output_dir='results/phase11_audit/figures'); p.generate_all_plots()"
```

---

## 5. Strict Gate Review Before Launch

> [!IMPORTANT]
> **GO / NO-GO CHECKLIST:**
> 1. [x] Git commit and branch confirmed (`phase10-experimental-validation` at `6882d3c`).
> 2. [x] 59/59 pytest regression suite passed.
> 3. [x] Checkpoint save and restore verified bitwise identical.
> 4. [x] Phase 11 Defect Register authored (`DEF-P11-01` through `DEF-P11-09`).
> 5. [x] Isolated directory created (`results/phase11_audit/`).
> 6. [ ] User approval received to apply remediations and launch 5,000-step pilot.

**Under the directive instructions, training will NOT be launched until explicit user approval is granted.**
