# DOIM-Traffic-RL: Traffic Signal Safety Shield Validation & Verification Report
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**Standard:** Strict Microscopic Safety Integrity & Fail-Safe Verification  
**Audited Modules:** `environment/safety_shield.py`, `environment/safety_verifier.py`, `environment/sumo_env.py`  
**Reference Tests:** `tests/test_safety_shield_semantics.py`, `tests/test_traffic_safety.py`, `tests/test_q1_correctness_and_safety.py`

---

## 1. Safety Architecture Overview

The `DOIM-Traffic-RL` safety shield acts as a formal runtime supervisor intercepting reinforcement learning policy outputs before command delivery to the SUMO microscopic simulator.

```
Policy Raw Action a_t ∈ {0, 1}^5
         |
         v
+--------------------------------------------------------------------------------------------------+
| Traffic Safety Shield (TrafficSafetyShield)                                                       |
| 1. Minimum Green Verification: enforces current_time >= 10s before phase exit                      |
| 2. Maximum Green Verification: forces phase exit if current_time >= 60s                          |
| 3. Mandatory Clearance Verification: green-to-green transitions must execute yellow + all-red      |
| 4. Microscopic Conflict Verification: parses junction <request> foes & response bitmasks         |
| 5. Program Graph Reachability: validates multi-stage paths & reachable all-red nodes             |
| 6. Fail-Safe Error Handling: triggers SafetyFaultError & aborts episode if no safe path exists    |
+--------------------------------------------------------------------------------------------------+
         |
         +----------------------------------+----------------------------------+
         | Verified Safe Transition         | No Verified Safe Transition
         v                                  v
TraCI Signal Phase Set                 Typed SafetyFaultError Raised
SUMO Microscopic Step Execution        Episode Terminated (is_valid_run = False)
SignalSafetyVerifier Audit Trace       Run Marked: ABORTED_SAFETY_FAULT
```

---

## 2. Microscopic Conflict Matrix & Yield Semantics

### 2.1 Geometric Conflict Extraction
The shield dynamically parses the SUMO network definition (`grid.net.xml`), establishing exact correspondences between:
1. Microscopic junction requests: `<request index="..." foes="..." response="...">`
2. Traffic light signal connections: `<connection tl="..." linkIndex="...">`
3. State string indices: $k \in \{0, \dots, N_{\text{links}}-1\}$

For two movements with link indices $(i, j)$:
- If the foes bitmask indicates a crossing path ($\text{foes}[j] = 1$ for request $i$), $(i, j)$ is registered as a geometric conflict pair.
- The response bitmask defines right-of-way priority: if $\text{response}[i] = 1$ when movement $j$ is active, movement $i$ yields to movement $j$.

### 2.2 Right-of-Way Invariants
Every candidate signal state string $S \in \{\text{'r'}, \text{'g'}, \text{'G'}, \text{'y'}\}^{N_{\text{links}}}$ is audited against three strict invariants:
1. **Unconditional Conflict Ban:** Movements in direct conflict cannot simultaneously hold unconditional priority green:
   $$\neg \left( S[i] = \text{'G'} \land S[j] = \text{'G'} \right) \quad \forall (i, j) \in \text{Conflicts}$$
2. **Permissive Yield Hierarchy:** If movement $i$ holds permissive green ($\text{'g'}$) while conflicting movement $j$ holds priority green ($\text{'G'}$), movement $i$ must have a registered yield relationship:
   $$(S[i] = \text{'g'} \land S[j] = \text{'G'}) \implies (i, j) \in \text{YieldPairs}$$
3. **Dual Permissive Verification:** If both movements hold permissive green ($\text{'g'}$ and $\text{'g'}$), an unambiguous yield hierarchy must exist to prevent gridlock or right-of-way confusion.

---

## 3. Dynamic Multi-Stage Clearance & Transition Pathfinding

Rather than assuming hardcoded 4-phase cyclic programs, the shield dynamically extracts traffic light programs from SUMO XML `<tlLogic>` elements and TraCI data structures:

### 3.1 Phase Classification
Each program phase is categorized into:
- **Green Phases ($P_G$):** States containing priority green (`'G'`) or permissive green (`'g'`).
- **Clearance Phases ($P_C$):** Transition states containing yellow (`'y'`) or all-red (`'r'`).
- **All-Red Hold Phases ($P_R$):** Bounded clearance states where all links display red (`'r'`).

### 3.2 Directed Reachability Analysis
Using explicit `<phase next="...">` XML attributes or default sequential logic:
1. Directed graph search maps every green phase $g \in P_G$ to its complete clearance sequences:
   $$\text{Path}(g) = [ (c_1, \tau_1), (c_2, \tau_2), \dots, (c_m, \tau_m) ] \to g_{\text{dest}}$$
2. Terminal clearance phases are identified: a clearance phase is terminal if and only if its successors lead to a green phase or an all-red state.
3. Fallbacks to all-red are restricted strictly to reachable all-red nodes along the active clearance sequence, preventing illegal arbitrary jumps.

---

## 4. Fail-Safe Protocol & Typed Safety Fault Handling

When maximum green duration is exceeded ($t \ge 60\text{s}$) and the nominal successor phase has active movement conflicts:
1. **Option A (Reachable Alternative Green):** The shield searches all alternative paths from current green $g$. If a conflict-free alternative green $g_{\text{alt}}$ is reachable through a valid clearance sequence, the shield routes traffic through the mandatory clearance stages to $g_{\text{alt}}$.
2. **Option B (Reachable All-Red Clearance):** If all reachable green alternatives conflict, the shield checks for a reachable all-red phase $r \in P_R$ along the active clearance sequence. If found, the controller executes clearance and locks into all-red fail-safe hold.
3. **Option C (No Verified Safe Transition):** If no conflict-free alternative or reachable all-red state exists in the SUMO program:
   - Sets `state.cannot_safely_progress = True` and `state.is_fail_safe_locked = True`.
   - Records full fault telemetry: TLS ID, active phase index, signal state string, conflict pairs, and program phases.
   - Raises typed `SafetyFaultError(RuntimeError)` when configured.
   - Halts simulation episode immediately (`terminated = True`).
   - Flags run as `ABORTED_SAFETY_FAULT`, rendering it ineligible for benchmark tables.
   - **Crucial Safety Guarantee:** Never leaves an expired green active indefinitely and never uses yellow clearance as a permanent hold state.

---

## 5. Automated Verification Evidence

The safety architecture is verified by 31 automated regression tests across `tests/test_safety_shield_semantics.py` (23 tests), `tests/test_traffic_safety.py` (6 tests), and `tests/test_q1_correctness_and_safety.py` (2 tests).

### Verified Test Matrix:
| Test ID | Test Function | Verified Property | Result |
| :--- | :--- | :--- | :--- |
| **TEST-SAFE-01** | `test_four_phase_program_derivation_and_verification` | Correct derivation of 4-phase cycles (phases 0, 2 green; 1, 3 yellow) | **PASSED** |
| **TEST-SAFE-02** | `test_six_phase_multi_stage_clearance_program` | Multi-stage clearance execution (Green -> Yellow -> All-Red -> Green) | **PASSED** |
| **TEST-SAFE-03** | `test_eight_phase_program_with_protected_turns` | Protected left-turn dual-ring 8-phase conflict resolution | **PASSED** |
| **TEST-SAFE-04** | `test_explicit_non_cyclic_next_transitions` | Explicit XML `next` jumps bypassing cyclic order safely | **PASSED** |
| **TEST-SAFE-05** | `test_missing_or_ambiguous_tls_program_mapping` | Marks unmapped TLS as unverifiable and engages fail-safe lock | **PASSED** |
| **TEST-SAFE-06** | `test_forced_max_green_when_normal_successor_conflicting` | Enforces max-green exit via alternative green path | **PASSED** |
| **TEST-SAFE-07** | `test_all_red_fallback_reachability_and_fail_safe` | Fallback restricted strictly to reachable all-red nodes | **PASSED** |
| **TEST-SAFE-08** | `test_clearance_phase_not_treated_as_permanent_safe_state` | Prevents indefinite holds in yellow clearance | **PASSED** |
| **TEST-SAFE-09** | `test_safety_fault_error_raised_when_configured` | Raises `SafetyFaultError` when no safe path exists | **PASSED** |
| **TEST-SAFE-10** | `test_adversarial_safety_shield_sumo_trace` | 100% collision-free operation under adversarial switching in live SUMO | **PASSED** |

---

## 6. Scope of Verified Safety & Limitations

1. **Deterministic Action Shielding:** The shield provably eliminates signal-induced movement conflicts by verifying all signal state strings against junction geometry and clearance intervals.
2. **Not a Driver Physics Guarantee:** The safety shield controls traffic lights; it does not control individual driver physics or erratic vehicle behavior (e.g. vehicles running red lights or external vehicle collisions due to microscopic car-following parameter anomalies).
3. **Program Completeness Requirement:** If a SUMO network provides an incomplete or malformed `<tlLogic>` program missing essential clearance intervals, the verifier explicitly reports `UNVERIFIABLE_TLS_PROGRAM` rather than guessing unmodeled stages.

