# Comprehensive Repository Audit & Baseline Verification Report

**Project Title:** Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Systems  
**Repository:** [https://github.com/Ramya282006/DOIM-Traffic-RL](https://github.com/Ramya282006/DOIM-Traffic-RL)  
**Branch Reviewed:** `ramya` / `main`  
**Audit Date:** October 31, 2026  
**Auditor Roles:** Senior AI/ML Scientist, Multi-Objective RL Specialist, Traffic & SUMO Researcher, GNN/Temporal Learning Specialist, Scientific Software Architect, Senior Reviewer (IEEE T-ITS / T-PAMI standard)

---

## 1. Executive Summary

This audit performs an exhaustive, line-by-line inspection of the `DOIM-Traffic-RL` codebase, its simulation pipelines, neural architectures, mathematical formulations, training routines, evaluation protocols, and committed empirical artifacts.

The repository presents an end-to-end prototype integrating Eclipse SUMO with TraCI, Gymnasium, PyTorch, and Stable-Baselines3 (PPO) for multi-objective traffic signal control on a 3×3 grid network. It implements spatial graph neural networks, temporal GRU sequence models, a Dynamic Objective Interaction Manifold (DOIM) neural block, five traffic objectives, and comparative evaluation routines.

However, a rigorous scientific and code-level audit reveals a fundamental **method-versus-implementation gap** alongside critical reproducibility, statistical, safety, and architectural defects. The central research claim—that a dynamic objective interaction manifold is learned and adaptively guides multi-objective policy optimization—is **not supported by the current code execution**. Specifically:
1. The spatial GNN, temporal GRU, and DOIM modules are placed in `.eval()` mode and wrapped in `torch.no_grad()`. They receive zero gradients during training.
2. The PPO algorithm optimizes only an off-the-shelf Stable-Baselines3 MLP policy over a frozen, randomly initialized latent space.
3. The "interaction manifold" is derived from a 30-step rolling Pearson correlation matrix, which represents raw sample correlation confounded by nonstationary demand rather than learned interaction dynamics.
4. The reported multi-seed evaluation results for the fixed-weight baseline are completely duplicated across all seeds because `--seed` was omitted from the SUMO launch command in `fixed_weight_env.py`.
5. The committed evaluation summary JSON files and multi-seed CSV results contradict each other.
6. The traffic signal controller allows arbitrary phase switches every 1 second without minimum green, maximum green, or yellow clearance enforcement, resulting in an ~80% phase change ratio (rapid flickering).

This document establishes the verified baseline state of the repository, details every code and scientific limitation with line-level evidence, and presents a prioritized defect register that guides subsequent implementation phases.

---

## 2. Current Architecture and Data Flow

### 2.1 Existing End-to-End Pipeline

The operational pipeline implemented in `environment/sumo_env.py` and `environment/doim_processor.py` proceeds as follows:

```
[SUMO Simulation Step (1s)]
       │
       ▼
[TrafficStateExtractor]
  ├── Per-TLS incoming lane statistics (halting count, mean speed, density)
  └── Vehicle-level accumulated waiting time and instantaneous CO2
       │
       ├──► Feature Matrix X ∈ ℝ^(9×5) [queue, waiting, speed, density, CO2]
       └──► Network States Dictionary (9 junctions)
       │
       ▼
[MultiObjectiveEngine]
  ├── total_waiting:  ∑ waiting (vehicle-seconds)
  ├── total_queue:    ∑ halting vehicles
  ├── total_co2:      ∑ vehicle instantaneous CO2
  ├── throughput:     max(0, completed - prev_completed)
  └── fairness proxy: var(waiting across intersections)
       │
       ▼
[ObjectiveNormalizer]
  ├── Static lower bounds [0, 0, 0, 0, 0]
  ├── Static upper bounds [600, 20, 120000, 20, 3000]
  └── Invert throughput: urgency[3] = 1.0 - scaled[3]
       │
       ▼ Raw Normalized Objective Vector z_t ∈ [0, 1]^5
       │
       ▼
[ObjectiveInteractionAnalyzer]
  └── Rolling FIFO deque (window_size=30)
  └── Pearson Correlation: C = corrcoef(history^T) ∈ ℝ^(5×5)
       │
       ▼
[DOIMProcessor (torch.no_grad(), eval mode)]
  ├── SpatialTrafficEncoder: 2-layer GCNConv over 9-node grid graph -> (9, 64) -> mean pool -> (64,)
  ├── TemporalTrafficEncoder: 1-layer GRU over 10-step history of pooled spatial embeddings -> (64,)
  ├── DOIM:
  │     ├── Objective MLP: ℝ^5 -> ℝ^32
  │     ├── Interaction MLP: ℝ^(5×5) -> ℝ^32
  │     ├── Concat [traffic(64), obj(32), int(32)] = 128 -> MLP -> LayerNorm -> Tanh -> Manifold m_t ∈ ℝ^64
  │     └── Preference Head: m_t -> Linear -> Softmax -> Weights w_t ∈ Δ^4 (sum to 1)
       │
       ├──► Observation for PPO: [m_t (64), w_t (5)] ∈ ℝ^69
       └──► DOIMReward: r_t = - ∑ (w_t,i * z_t,i)
       │
       ▼
[Stable-Baselines3 PPO Policy]
  └── MlpPolicy: ℝ^69 -> Hidden Layers -> Categorical Distribution for MultiDiscrete([2, 2, 2, 2, 2])
       │
       ▼
[TrafficSignalController]
  └── For each of 5 controllable TLS: if action==1: phase = (phase + 1) % num_phases
```

---

## 3. Module Inventory and Responsibilities

| Module Path | Primary Class / Function | Responsibility | Implementation Status / Defect |
|---|---|---|---|
| `environment/sumo_env.py` | `DOIMTrafficEnv` | Gymnasium environment wrapping SUMO; coordinates extraction, DOIM processing, reward calculation, and stepping. | Working execution; frozen representation; observation lacks raw traffic state; no truncation handling. |
| `environment/fixed_weight_env.py` | `FixedWeightTrafficEnv` | Baseline environment with uniform fixed weights `[0.2, 0.2, 0.2, 0.2, 0.2]`. | Defect: `--seed` flag omitted in `reset()`; identical trajectories across all evaluation seeds. |
| `environment/doim_processor.py` | `DOIMProcessor` | Orchestrates feature extraction, GNN, GRU, DOIM inference, and temporal deque history. | Critical Defect: Runs in `.eval()` mode with `torch.no_grad()`; zero gradient propagation. |
| `environment/traffic_signal_controller.py` | `TrafficSignalController` | Discovers controllable TLS (`len(phases) > 1`) and applies binary phase advance. | Safety Defect: No minimum green, no yellow enforcement, allows rapid phase fluttering. |
| `environment/state_extractor.py` | `TrafficStateExtractor` | Queries TraCI for lane halting, vehicles, speed, waiting, and CO2 emissions. | Discrepancy: Queries all 9 junctions (including 4 static single-phase corners). |
| `models/spatial_encoder.py` | `SpatialTrafficEncoder`, `create_grid_edge_index` | 2-layer GCNConv mapping `[9, 5]` node features to `[9, 64]`; hardcoded 3×3 grid graph. | Untrained; hardcoded topology; graph size (9) differs from controllable lights (5). |
| `models/temporal_encoder.py` | `TemporalTrafficEncoder` | 1-layer GRU processing sequence of aggregated spatial embeddings `[1, L, 64]`. | Untrained; no gradient connection to downstream losses. |
| `models/doim.py` | `DOIM` | Encodes objective vector, flattened interaction matrix, and traffic embedding into manifold and softmax weights. | Untrained; arbitrary output weights used for reward scalarization. |
| `objectives/objective_engine.py` | `MultiObjectiveEngine` | Calculates 5 raw scalar metrics from intersection dictionary and completed vehicle count. | Simplified fairness proxy (variance of waiting); throughput reset vulnerability. |
| `objectives/objective_normalizer.py` | `ObjectiveNormalizer` | Min-max normalization with hardcoded upper/lower bounds; inverts throughput to urgency. | Hardcoded bounds; throughput clipped at 20; sensitive to scenario scale. |
| `objectives/objective_interaction.py` | `ObjectiveInteractionAnalyzer` | Rolling 30-step window; computes Pearson correlation via `np.corrcoef(data.T)`. | Duplicate class in `analysis/`; pure statistical sample correlation without causal or state-conditioned structure. |
| `objectives/doim_reward.py` | `DOIMReward` | Computes scalar reward $r_t = -\mathbf{w}_t^\top \mathbf{z}_t$. | Moving target for RL optimization; weights shift arbitrarily during policy training. |
| `objectives/emissions.py`, `fairness.py`, `queue.py`, `throughput.py`, `waiting_time.py` | None | Empty placeholder files (0 bytes). | Dead code / unwritten modules. |
| `training/train_ppo.py` | `main` | Trains SB3 PPO on `DOIMTrafficEnv` for 1,000 timesteps using `MlpPolicy`. | Budget far too small (1,000 steps); only trains SB3 policy; DOIM completely disconnected. |
| `training/train_fixed_ppo.py` | `main` | Trains SB3 PPO on `FixedWeightTrafficEnv` for 1,000 timesteps. | Budget far too small (1,000 steps); baseline comparator. |
| `training/evaluate_multiseed.py` | `evaluate_model` | Evaluates single saved checkpoint across 5 seeds. | Flawed evaluation: does not evaluate independently trained seeds; fixed PPO duplicate bug. |
| `training/evaluate_comparison.py` | `main` | Compares DOIM vs Fixed PPO from saved summary JSON files. | Compares flawed artifacts rather than running live validated evaluation. |
| `analysis/analyze_objective_interactions.py` | `analyze_interactions` | Plots interaction correlation matrix, correlation distributions, and trajectories. | Post-hoc analysis of non-trainable rolling Pearson matrix. |

---

## 4. In-Depth Inspection of Previously Identified Concerns

### 4.1 Verification 1: Are Spatial GNN, Temporal GRU, and DOIM Trainable?

**Code Evidence:**  
In `environment/doim_processor.py`:
- Lines 149–151:
  ```python
  self.spatial_encoder.eval()
  self.temporal_encoder.eval()
  self.doim.eval()
  ```
- Lines 235–242:
  ```python
  with torch.no_grad():
      spatial_embeddings = self.spatial_encoder(x, self.edge_index)
  ```
- Lines 270–276:
  ```python
  with torch.no_grad():
      temporal_embedding = self.temporal_encoder(sequence)
  ```
- Lines 296–304:
  ```python
  with torch.no_grad():
      manifold, weights = self.doim(
          temporal_embedding,
          objective_tensor,
          interaction_tensor,
      )
  ```
- Lines 325–337:
  All outputs are converted to NumPy arrays via `.cpu().numpy()` and returned inside a dictionary.

**Scientific Assessment:**  
The neural modules (GNN, GRU, DOIM) never participate in any backpropagation pass. They exist entirely outside the PyTorch autograd graph during both simulation and training. Their parameters remain at their random PyTorch initialization throughout all training experiments.

---

### 4.2 Verification 2: Does PPO Only Optimize the Stable-Baselines3 Policy?

**Code Evidence:**  
In `training/train_ppo.py`:
- Line 142:
  ```python
  model = PPO(
      policy="MlpPolicy",
      env=env,
      learning_rate=3e-4,
      ...
  )
  ```
- Line 201:
  ```python
  model.learn(total_timesteps=TOTAL_TIMESTEPS, ...)
  ```

**Scientific Assessment:**  
Stable-Baselines3 `PPO` creates an internal `ActorCriticPolicy` (a standard 2-layer MLP) taking the 69-dimensional observation vector as input. The loss function:
$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] + c_1 \mathcal{L}_t^{VF}(\theta) - c_2 S[\pi_\theta](s_t)$$
computes gradients exclusively with respect to $\theta$ (the weights of SB3's MLP). Gradients terminate at the boundary of `env.step()`. The parameters $\phi_{\text{GNN}}, \phi_{\text{GRU}}, \phi_{\text{DOIM}}$ are never registered in any optimizer and receive zero gradient updates.

---

### 4.3 Verification 3: Is Objective Interaction Based Only on Rolling Pearson Correlation?

**Code Evidence:**  
In `objectives/objective_interaction.py` lines 40–74 and `analysis/objective_interaction.py` lines 63–106:
```python
data = np.asarray(self.history, dtype=np.float32)
correlation = np.corrcoef(data, rowvar=False)
correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
correlation = (correlation + correlation.T) / 2.0
np.fill_diagonal(correlation, 1.0)
return correlation.astype(np.float32)
```

**Scientific Assessment:**  
The interaction matrix $M_t$ is computed via the standard sample Pearson correlation coefficient over a sliding window of 30 discrete simulation seconds:
$$\rho_{ij} = \frac{\sum_{\tau=t-W+1}^t (z_{\tau,i} - \bar{z}_i)(z_{\tau,j} - \bar{z}_j)}{\sqrt{\sum_{\tau} (z_{\tau,i} - \bar{z}_i)^2 \sum_{\tau} (z_{\tau,j} - \bar{z}_j)^2}}$$
This calculation exhibits several severe flaws:
1. **Confounding by Shared Nonstationarity:** In traffic networks, when an inflow surge occurs, waiting time, queue length, and CO2 emissions all simultaneously increase due to overall vehicle volume. A high positive Pearson correlation merely captures the exogenous demand surge, not a structural trade-off between control objectives.
2. **Lag Ignorance:** Signal decisions taken at time $t$ affect queue formation immediately, but impact waiting time and emissions with dynamic transport delay (10–60 seconds). Instantaneous static correlation misses this lagged cross-dependency.
3. **Absence of State-Conditioned Learning:** The correlation matrix is descriptive and purely observational. It contains no state-conditioned predictive model, no representation of policy intervention, and no learned manifold structure.

---

### 4.4 Verification 4: Training Timestep Budget

**Code Evidence:**  
In `training/train_ppo.py` line 27 and `training/train_fixed_ppo.py` line 27:
```python
TOTAL_TIMESTEPS = 1000
```
With `n_steps=128`, PPO performs only $\lfloor 1000 / 128 \rfloor = 7$ gradient updates across the entire training run!

**Scientific Assessment:**  
In a 3×3 grid with 5 controllable intersections and continuous traffic flow, 1,000 simulation steps equals approximately 16.6 minutes of simulated traffic (less than one complete 1,800-second scenario episode). A policy cannot learn coordinated signal progression, network clearing, or multi-objective trade-offs in 7 rollout iterations. Legitimate RL convergence in urban traffic networks requires at least $50,000$ to $200,000$ environment steps.

---

### 4.5 Verification 5: Seeding and Fixed Baseline Replication Defect

**Code Evidence:**  
In `environment/fixed_weight_env.py` lines 114–122:
```python
command = [
    self.sumo_binary,
    "-c",
    self.sumo_config,
    "--start",
    "--quit-on-end",
]
traci.start(command)
```
Compare this with `environment/sumo_env.py` lines 121–129:
```python
command = [
    self.sumo_binary,
    "-c",
    self.sumo_config,
    "--start",
    "--quit-on-end",
    "--seed",
    str(seed if seed is not None else 42),
]
```
`fixed_weight_env.py` completely omitted `--seed` from the SUMO launch arguments!

**Empirical Evidence:**  
In `results/evaluation/multiseed_results.csv`:
- Seed 42:  `fixed_ppo, 42, 1000, -459.1029588878155, -0.4591029588878155, 108.045, 313.0, 2.007, 9.0, 100387.91831054687, 255.0, 611.2077043885589...`
- Seed 123: `fixed_ppo, 123, 1000, -459.1029588878155, -0.4591029588878155, 108.045, 313.0, 2.007, 9.0, 100387.91831054687, 255.0, 611.2077043885589...`
- Seed 456: `fixed_ppo, 456, 1000, -459.1029588878155, -0.4591029588878155, 108.045, 313.0, 2.007, 9.0, 100387.91831054687, 255.0, 611.2077043885589...`
- Seed 789: `fixed_ppo, 789, 1000, -459.1029588878155, -0.4591029588878155, 108.045, 313.0, 2.007, 9.0, 100387.91831054687, 255.0, 611.2077043885589...`
- Seed 2024:`fixed_ppo, 2024, 1000, -459.1029588878155, -0.4591029588878155, 108.045, 313.0, 2.007, 9.0, 100387.91831054687, 255.0, 611.2077043885589...`

All metrics for `fixed_ppo` across seeds 42, 123, 456, 789, and 2024 are identical to 16 decimal places. The evaluation script was running the default SUMO seed every time for `fixed_ppo` while varying the seed for `doim_ppo`.

Furthermore, `training/evaluate_multiseed.py` did not train 5 independent models per method; it loaded a single pre-trained checkpoint (`doim_ppo_final.zip` and `fixed_ppo_final.zip`) and only changed the evaluation seed. Treating multiple evaluations of one model checkpoint as independent training seeds violates standard experimental methodology in RL.

---

### 4.6 Verification 6: Inconsistency Between Summary JSON and CSV Results

**Empirical Evidence:**  
Comparing `results/evaluation/doim_ppo_summary.json` with the seed 42 row in `results/evaluation/multiseed_results.csv`:

| Metric | `doim_ppo_summary.json` | `multiseed_results.csv` (Seed 42) | Absolute Discrepancy |
|---|---|---|---|
| Total Reward | -447.8831 | -430.7931 | 17.0900 |
| Average Reward | -0.4479 | -0.4308 | 0.0171 |
| Average Waiting Time | 93.3390 s | 81.9260 s | 11.4130 s |
| Maximum Waiting Time | 266.0 s | 186.0 s | 80.0 s |
| Average Queue Length | 1.4960 | 1.3270 | 0.1690 |
| Total Throughput | 249.0 veh | 248.0 veh | 1.0 veh |
| Average Fairness (Variance) | 710.0752 | 515.2287 | 194.8465 |

**Scientific Assessment:**  
The summary JSON reports results from a different, unversioned run. It does not match the multi-seed evaluation log for the same default seed (42). In scientific publications, reporting mismatched metrics across figures and tables is an immediate fatal flaw.

---

### 4.7 Verification 7: Network Graph vs. Controlled Traffic Light Mismatch

**Code Evidence:**  
In `models/spatial_encoder.py` lines 72–137:
`create_grid_edge_index()` hardcodes an undirected graph with 9 nodes:
```
0 (A0) ── 1 (A1) ── 2 (A2)
│         │         │
3 (B0) ── 4 (B1) ── 5 (B2)
│         │         │
6 (C0) ── 7 (C1) ── 8 (C2)
```
In `environment/network/research_3x3/grid.net.xml`:
- Junctions A0, A2, C0, C2 have 1 phase (`['GGGG']`) and are uncontrolled corners.
- Junctions A1, B0, B1, B2, C1 have 4 phases and are controllable traffic lights.

In `environment/state_extractor.py`, `TrafficStateExtractor` extracts states for all 9 junctions. In `models/spatial_encoder.py`, the GCN operates on 9 nodes, outputting `[9, 64]`. Then `doim_processor.py` line 249 averages all 9 node embeddings into a single vector `[64]`. However:
1. The 4 uncontrolled corners contribute fixed green statistics that dilute congestion signals from the central controllable corridors.
2. The graph ordering relies on `sorted(traci.trafficlight.getIDList())` matching the arbitrary numeric indices $(0..8)$ in `create_grid_edge_index()`. If node IDs differ or sort differently, edge connections are silently transposed.
3. The action space controls only the 5 internal junctions, but the observation gives the agent no direct representation of which junction has what phase or queue length.

---

### 4.8 Verification 8: Traffic Signal Safety and Phase Duration Violations

**Code Evidence:**  
In `environment/traffic_signal_controller.py` lines 106–133:
```python
if decision == 1:
    current_phase = traci.trafficlight.getPhase(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    num_phases = len(logics[0].phases)
    next_phase = (current_phase + 1) % num_phases
    traci.trafficlight.setPhase(tls_id, next_phase)
```

**Scientific Assessment:**  
1. **Zero Minimum Green Duration:** An action of `1` changes the phase on the very next simulation second. If the agent outputs `1` consecutively, the controller advances through green, yellow, all-red/green in 2 seconds.
2. **Phase Fluttering:** In `results/evaluation/multiseed_results.csv`, the reported `change_ratio` is $0.798$ ($3,990$ changes out of $5,000$ decisions). Traffic signals are switching phases nearly every second, which is hazardous and physically infeasible in real traffic systems.
3. **No Intermediate Clearance Guarantee:** Transitions do not enforce standard 3–4 second yellow clearance before switching right-of-way, risking simulated vehicle collisions or emergency stops.

---

### 4.9 Verification 9: Traffic Objectives and Units

**Code Evidence:**  
In `objectives/objective_engine.py`:
- `total_waiting`: Sum of `traci.vehicle.getAccumulatedWaitingTime(veh_id)`. Unit: vehicle-seconds. (Correct SUMO accumulator, but sums across all vehicles present on incoming lanes of all 9 intersections).
- `total_queue`: Sum of `traci.lane.getLastStepHaltingNumber(lane_id)`. Unit: count of vehicles with speed $< 0.1$ m/s.
- `total_co2`: Sum of `traci.vehicle.getCO2Emission(veh_id)`. SUMO returns instantaneous emission in milligrams per second (mg/s). Over a 1-second simulation step, $\sum \text{mg/s} \times 1\text{s} = \text{mg}$. Upper bound in normalizer is 120,000 mg (120 g/s).
- `throughput`: `max(0, completed_vehicles - self.previous_completed)` where `completed_vehicles = traci.simulation.getArrivedNumber()`. Notice that `getArrivedNumber()` in SUMO returns the number of vehicles that arrived during the *current step*, NOT cumulative arrivals!
  Therefore, subtracting `self.previous_completed` causes throughput to alternate between 0 and negative (clipped to 0), destroying the throughput measurement!
- `fairness`: `np.var(waiting)` across the 9 intersection totals.
  Variance of intersection aggregate waiting times is NOT vehicle-level fairness. If Intersection A has 100 vehicles waiting 10s (total 1000s) and Intersection B has 10 vehicles waiting 100s (total 1000s), variance is 0, yet vehicle-level delay disparity is extreme.

---

## 5. Prioritized Defect Register

| Defect ID | Priority | Subsystem | File & Lines | Root Cause | Scientific Consequence | Proposed Correction | Verification Method |
|---|---|---|---|---|---|---|---|
| **DEF-01** | **P0** | Architecture / Autograd | `environment/doim_processor.py`: 149-151, 235, 270, 296 | Neural modules invoked inside `torch.no_grad()` and `.eval()` mode. | Claim of a "learned manifold" is invalid; representations are static random projections. | Implement end-to-end differentiable PyTorch feature extractor or multi-task policy architecture. | PyTorch gradient flow assertion: $\forall p \in \Theta_{\text{DOIM}}, \|\nabla_p \mathcal{L}\| > 0$. |
| **DEF-02** | **P0** | RL Formulation | `training/train_ppo.py`: 142-206 | PPO optimizes external SB3 MLP; DOIM is not in the computational graph. | Policy updates cannot adapt representation or weights via policy loss. | Integrate DOIM into a unified Actor-Critic network or differentiable preference-conditioned policy. | Parameter checksum tracking before and after training. |
| **DEF-03** | **P0** | Interaction Modeling | `objectives/objective_interaction.py`: 40-74 | Interaction matrix is sample Pearson correlation over 30 steps. | Confounded by exogenous demand surges; lacks state conditioning and directional/causal structure. | Implement learned state-conditioned interaction operator with temporal context and regularization. | Comparison against Pearson correlation and state-only baselines on perturbation tasks. |
| **DEF-04** | **P0** | Reproducibility | `environment/fixed_weight_env.py`: 114-122 | `--seed` argument omitted from `traci.start()` call. | Baseline results identical across all evaluation seeds; multi-seed statistical claims fabricated. | Pass `--seed str(seed)` to `traci.start()` command in `fixed_weight_env.py`. | Distinct output metrics across independent evaluation seeds. |
| **DEF-05** | **P0** | Experimental Protocol | `training/evaluate_multiseed.py`: 20-98 | Single checkpoint evaluated across 5 seeds; models not independently trained per seed. | Conflates evaluation seed variation with training seed replication. | Implement separate multi-seed training pipeline ($N \ge 5$ distinct training seeds). | Distinct model checkpoint hashes for each training seed. |
| **DEF-06** | **P0** | Data Integrity | `results/evaluation/` (JSON vs CSV) | Summary JSON values differ from corresponding CSV rows for seed 42. | Results cannot be verified or traced to a single experiment run. | Single experiment runner generating immutable runs with hash manifests and automated consistency check. | Automated verification script asserting exact equality between raw step logs, episode CSV, and summary JSON. |
| **DEF-07** | **P0** | Traffic Safety | `environment/traffic_signal_controller.py`: 106-133 | Direct phase increment without minimum green or yellow clearance. | Unsafe phase flickering (79.8% change ratio); unacceptable for real-world publication. | Implement stateful traffic controller enforcing minimum green ($10\text{s}$), maximum green ($60\text{s}$), and yellow clearance ($3\text{s}$). | Safety unit test verifying zero illegal phase transitions and change ratio bounded by cycle constraints. |
| **DEF-08** | **P1** | Metric Validity | `objectives/objective_engine.py`: 56-61 | `completed_vehicles` treats step arrival as cumulative arrival. | Throughput metric corrupted, measuring 0 in most steps. | Correctly accumulate `traci.simulation.getArrivedNumber()` per step. | Unit test comparing accumulated arrivals against SUMO summary XML output. |
| **DEF-09** | **P1** | Fairness Formulation | `objectives/objective_engine.py`: 70-72 | Fairness proxy is variance of intersection aggregates. | Fails to measure vehicle-level delay inequity or approach disparity. | Formulate Jain's Fairness Index or Gini coefficient over approach/vehicle delay distributions. | Fairness test with known unequal vehicle distributions. |
| **DEF-10** | **P1** | Network Topology | `models/spatial_encoder.py`: 72-137 | Hardcoded 9-node grid with arbitrary indexing; controls 5 lights. | GNN representation pooled with uncontrolled static junctions; potential node mismatch. | Dynamic graph generation mapped directly from SUMO junction topology and TLS IDs. | Graph assertion verifying 1-to-1 node ID mapping and edge connectivity. |
| **DEF-11** | **P1** | Training Budget | `training/train_ppo.py`: 27 | Training timesteps set to 1,000 (only 7 PPO updates). | Models under-trained; random performance noise misinterpreted as algorithm differences. | Scale training to at least 50,000 timesteps with convergence diagnostics. | Learning curves showing asymptotic policy reward improvement. |
| **DEF-12** | **P2** | Code Hygiene | `objectives/` | Five objective files (`emissions.py`, `fairness.py`, etc.) are empty 0-byte placeholders. | Misleading repository structure; dead files. | Consolidate validated objective computation into clean, tested modules. | Clean directory audit with zero 0-byte placeholder files. |

---

## 6. Mathematical Baseline Formulations

### 6.1 Observation Space
The current observation vector $s_t \in \mathbb{R}^{69}$ is:
$$s_t = [m_t, w_t]^\top$$
where $m_t \in \mathbb{R}^{64}$ is the manifold representation and $w_t \in \Delta^4$ is the 5-dimensional preference distribution ($w_t \ge 0, \sum_{i=1}^5 w_{t,i} = 1$).
Notably, raw traffic states (queue lengths, approach speeds, detector occupancy, active signal phase indices) are omitted from $s_t$, forcing the downstream policy to rely entirely on the latent manifold $m_t$.

### 6.2 Action Space
The action space is a Gymnasium `MultiDiscrete([2, 2, 2, 2, 2])`:
$$a_t = (a_{t,1}, a_{t,2}, a_{t,3}, a_{t,4}, a_{t,5}) \in \{0, 1\}^5$$
where $a_{t,k} = 0$ maintains the current phase and $a_{t,k} = 1$ advances the phase index of controllable traffic light $k$.

### 6.3 Reward Formulation
The current scalarized reward is:
$$r_t = - \sum_{i=1}^5 w_{t,i} z_{t,i}$$
where $z_{t,i} \in [0, 1]$ is the normalized urgency of objective $i$. Because $w_t$ varies at every step according to an untrained network, the reward function is non-stationary:
$$R(s_t, a_t; w_t) = - \mathbf{w}_t^\top \mathbf{z}_t$$
In standard reinforcement learning, an agent optimizing $\mathbb{E}[\sum_t \gamma^t R(s_t, a_t; w_t)]$ under arbitrary time-varying weights without preference conditioning experiences non-stationary value estimation and policy instability.

---

## 7. Action Plan for Research Upgrade

Based on this audit, the implementation follows these rigorous phases:

1. **Phase 1: Reproducibility & Experimental Infrastructure**
   - Centralized YAML configuration system.
   - Deterministic seeding across Python, NumPy, PyTorch, Gymnasium, and SUMO (`--seed`).
   - Immutable experiment run runner with cryptographic manifests and raw per-step metric logging.
   - Clean Python 3.11 virtual environment setup with all verified dependencies.

2. **Phase 2: Scientifically Sound DOIM Architecture**
   - Correct topology mapping between SUMO network junctions and GNN graph structure.
   - Removal of `torch.no_grad()` / `.eval()` barriers; complete end-to-end differentiability.
   - Unified Actor-Critic architecture where GNN, GRU, and DOIM parameters are directly optimized by policy and auxiliary objective gradients.

3. **Phase 3: Genuine Objective Interaction Learning Mechanism**
   - Replace rolling Pearson correlation with a state-conditioned, learned interaction operator.
   - Formulate temporal interaction modeling with predictive objective transition dynamics.
   - Mathematical justification distinguishing statistical correlation from decision-relevant interaction.

4. **Phase 4: Interaction-Aware Multi-Objective Policy Optimization**
   - Preference-conditioned policy architecture or coherent joint scalarization objective.
   - Preference smoothness and entropy regularization to prevent reward fluttering.
   - Gradient-flow verification tests for all representation layers.

5. **Phase 5: Safety and Fairness Constraints**
   - Safety shield enforcing legal minimum green ($10\text{s}$), maximum green ($60\text{s}$), and yellow transitions ($3\text{s}$).
   - Rigorous vehicle-level delay fairness metrics (Jain's index, 95th percentile delay).
   - Multi-scenario framework (normal, directional surge, rush hour, gridlock).

6. **Phase 6–10: Validation, Ablations, Benchmarks, and IEEE Documentation**
   - Multi-seed independent training runs.
   - Comprehensive ablations (no GNN, no GRU, no DOIM, rolling correlation baseline, fixed weight baseline).
   - Full IEEE Transactions-style manuscript draft with complete theoretical and experimental exposition.
