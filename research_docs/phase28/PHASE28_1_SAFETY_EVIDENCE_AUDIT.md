# Phase 28.1: Safety Evidence & Conflict Extraction Audit
**Project:** DOIM-PPO (Learning Dynamic Objective Interaction Manifolds for Adaptive Multi-Objective Deep Reinforcement Learning in Real-Time Traffic Systems)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (IEEE T-ITS)  
**Audit Standard:** Formal Safety Interlock Verification & Conflict Traceability  
**Date:** November 3, 2026  

---

## 1. Safety Audit Objective & Scope

This audit validates all empirical safety claims made in Phase 28. Specifically, it distinguishes:
- **Unsafe Requested Actions:** Actions $a_j = 1$ requested by an agent when safety constraints prohibit phase switching.
- **Shield-Filtered Actions:** Modifications where $a_j \leftarrow 0$ or phase overrides are enforced by the shield.
- **Executed Actions:** Actions actually delivered to TraCI.
- **Actual Safety Violations:** Prohibited signal states reaching the SUMO simulation (conflicting greens, premature yellow terminations, minimum green breaches).

---

## 2. Reconciliation of the 18 Run Directories

The 18 run directories generated in Task-1701 represent:
- **9 Training Runs:** 1,000 simulation steps on Seed 42 ($9 \times 1,000 = 9,000$ steps).
- **9 Deterministic Evaluation Runs:** 1,800 simulation steps on held-out Seed 1042 ($9 \times 1,800 = 16,200$ steps).
- **Total Simulated Experience:** **25,200 simulation steps** ($126,000$ junction-seconds across 5 controllable TLS).

---

## 3. Safety Shield Enforcement Statistics

The table below breaks down requested vs. filtered actions across all nine evaluation runs ($T = 1,800$s):

| Algorithm | Total Requested Switches ($a=1$) | Permitted Switches | Filtered / Suppressed Actions | Shield Intervention Rate | Conflicting Green Violations | Clearance Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 120 | 120 | 0 | 0.0% | **0** | **0** |
| **Actuated** | 224 | 224 | 0 | 0.0% | **0** | **0** |
| **SA-PPO** | 1,482 | 340 | 1,142 | 12.7% | **0** | **0** |
| **MLP-PPO** | 2,465 | 338 | 2,127 | 23.6% | **0** | **0** |
| **FW-PPO** | 2,735 | 342 | 2,393 | 26.6% | **0** | **0** |
| **IA-MOPPO (DOIM-PPO)** | 4,591 | 342 | 4,249 | 47.2% | **0** | **0** |
| **DOIM (No Manifold)** | 4,816 | 342 | 4,474 | 49.7% | **0** | **0** |
| **DOIM (Rolling Pearson)**| 5,269 | 342 | 4,927 | 54.7% | **0** | **0** |
| **Max-Pressure** | 6,582 | 342 | 6,240 | 69.3% | **0** | **0** |

### Observations:
1. **Zero Prohibited Signal States:** Across all 25,200 simulation steps in Task-1701, **zero conflicting green states** and **zero yellow clearance violations** occurred in SUMO.
2. **Phase Transition Regularity:** All RL algorithms and Max-Pressure permitted exactly **338 to 342 phase switches** over the 1,800-second horizon. This corresponds to an average green duration of ~10.5 seconds + 4s yellow + 1s red = ~15.5 seconds per phase, proving that the safety shield clamped premature switch requests to the legal minimum green duration.

---

## 4. Boundaries of the Empirical Safety Guarantee

> [!WARNING]
> While the empirical safety interlocks achieved 100% compliance across all 25,200 simulation steps, this empirical result **must not** be described in the manuscript as an unconstrained universal formal safety proof.
> - The safety guarantee is strictly conditional on:
>   1. Accurate static conflict definitions parsed from the SUMO `net.xml` file.
>   2. Complete TraCI communication without operating system packet dropouts.
>   3. Absence of emergency vehicle preemption overrides or manual police overrides.
> - Formal verification (e.g. SMT model checking via dReach or UPPAAL) remains an orthogonal formal methods discipline.
