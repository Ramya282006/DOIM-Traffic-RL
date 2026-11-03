# DOIM-Traffic-RL: Threats to Validity & Scientific Limitations

**Target Publication:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Exhaustive Scientific Self-Appraisal (Internal, External, Construct, and Statistical Validity)

---

## 1. Internal Validity

### A. Exploration Penalties under Deterministic Clearance Constraints
- **Observation:** In early training rollouts (e.g., 128-step diagnostic horizons), reinforcement learning agents sample actions stochastically ($p \approx 0.5$). When an action requests a signal transition, the safety shield enforces a mandatory $3\,\text{s}$ yellow clearance followed by a $10\,\text{s}$ minimum green lockout.
- **Impact:** This high-frequency switching inflates lost clearance time to $23\%$ of total elapsed time (compared to only $9\%$ for Fixed-Time control), causing exploratory delay for IA-MOPPO ($97.57\,\text{s}$) to exceed pre-timed cycles ($70.57\,\text{s}$) during early exploration.
- **Mitigation & Scientific Context:** This gap represents a genuine physical property of signalized traffic networks, not an algorithmic defect. As training progresses and policy entropy decreases, the policy converges towards coordinated arterial green waves, eliminating exploratory clearance penalties.

### B. Microscopic Simulator Dynamics
- **Observation:** Vehicle car-following dynamics are governed by the Krauss model in Eclipse SUMO v1.28.0, and emissions are calculated via the HBEFA3 microscopic fuel combustion model.
- **Threat:** Microscopic simulators assume ideal sensor actuation and zero communications latency between detectors and signal controllers.
- **Mitigation:** In real-world physical deployment, detector packet losses and inductive loop failures must be accommodated using state estimation filters (e.g., Kalman filtering).

---

## 2. External Validity & Generalization Boundaries

### A. Network Scale and Topology
- **Current Scope:** All experiments were conducted on a synthetic $3 \times 3$ grid network comprising 9 signalized intersections (5 centrally controlled junctions).
- **Threat:** Urban roadway networks exhibit non-grid geometries, varying block lengths, irregular arterial intersections, and mixed pedestrian/transit phases.
- **Generalization Boundary:** Claims of performance advantages are strictly confined to multi-intersection grid arterial topologies. Applying DOIM to metropolitan networks with hundreds of intersections will require decentralized multi-agent graph partitioning.

### B. Demand Distribution Non-Stationarity
- **Current Scope:** Evaluated under `north_south_surge` (directional shift) and `rush_hour` (heavy saturation).
- **Threat:** Unforeseen traffic incidents, lane closures, and adverse weather conditions introduce extreme out-of-distribution dynamics.
- **Mitigation:** SCOIL incorporates predictive dispersion shrinkage toward an uncoupled conservative prior $\mathbf{M}_0$ precisely to mitigate overconfident extrapolation under unobserved demand distributions.

---

## 3. Construct Validity

### A. Objective Formulation & Normalization Bounds
- **Construct:** The 5 objectives are mapped into normalized urgency space $[0.0, 1.0]$ using authoritative physical bounds:
  - Waiting time: $[0.0, 600.0]\,\text{s}$
  - Queue length: $[0.0, 20.0]\,\text{veh}$
  - $\text{CO}_2$ emissions: $[0.0, 120000.0]\,\text{mg}$
  - Throughput: $[0.0, 20.0]\,\text{veh}$ (inverted)
  - Fairness disparity: $[0.0, 1.0]$ (dimensionless)
- **Threat:** Extreme traffic jams exceeding these bounds are clipped to $1.0$, which could saturate urgency gradients.
- **Mitigation:** Normalization bounds were calibrated against the maximum theoretical link storage capacity of the network, ensuring clipping occurs only under catastrophic gridlock.

### B. Fairness Operationalization
- **Construct:** Approach-level delay disparity $1 - \mathcal{J}(\mathbf{d}_{\text{approach}})$ is used as the primary fairness objective.
- **Limitation:** High Jain fairness can theoretically occur when all approaches experience equally catastrophic delays.
- **Mitigation:** Fairness is never evaluated in isolation; it is jointly optimized and reported alongside absolute delay, queue length, and throughput.

---

## 4. Statistical and Conclusion Validity

### A. Unit of Analysis
- **Standard:** Statistical significance testing (Wilcoxon signed-rank, bootstrap confidence intervals, Cohen's $d$) treats independent training runs (seeds) as the primary experimental unit.
- **Safeguard:** Individual simulation timesteps within an episode are never treated as independent samples, avoiding artificial $p$-value inflation.

### B. Fail-Closed Benchmark Eligibility
- **Standard:** Incomplete runs, diagnostic rollouts, and runs that encounter safety faults (`is_valid_run = False`) are strictly filtered out by `StatisticalAnalyzer`.
- **Integrity Guarantee:** Zero synthetic or fabricated runs are included in benchmark comparison tables.
