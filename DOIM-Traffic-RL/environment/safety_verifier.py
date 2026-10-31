"""
DOIM-Traffic-RL: Automated Traffic Signal Safety Verifier
Validates microscopic simulation phase traces and signal state strings against
traffic engineering safety invariants:
  1. Conflict-free signal states: link-level conflict matrix derived from junction geometry
     guarantees that crossing movements never receive simultaneous priority green ('G').
  2. Minimum green time enforcement: prevents rapid phase fluttering (duration >= min_green).
  3. Maximum green time enforcement: prevents traffic approach starvation (duration <= max_green).
  4. Clearance interval legality: every green-to-green phase transition must pass through yellow clearance.
  5. Trace-based verification: audits complete simulation logs post-hoc or online.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np


def extract_junction_conflicts_from_net(
    net_xml_path: Union[str, Path],
    controllable_tls_ids: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses SUMO network XML and extracts the exact microscopic conflict matrix
    for each controllable traffic light from junction <request foes="..."> definitions.

    In SUMO net XML, for junction link i with length N foes bitmask:
      foes[N - 1 - k] == '1' indicates link k conflicts geometrically with link i.
    """
    p = Path(net_xml_path)
    if not p.exists():
        return {}

    tree = ET.parse(p)
    root = tree.getroot()

    target_tls = set(controllable_tls_ids or ["A1", "B0", "B1", "B2", "C1"])
    conflicts_by_tls: Dict[str, List[Tuple[int, int]]] = {}

    for junc in root.findall("junction"):
        jid = junc.get("id")
        if jid not in target_tls:
            continue

        requests = junc.findall("request")
        n_links = len(requests)
        foe_pairs: List[Tuple[int, int]] = []

        req_dict = {int(r.get("index")): r.get("foes", "") for r in requests}

        for i in range(n_links):
            foes_str = req_dict.get(i, "")
            n = len(foes_str)
            for k in range(n_links):
                if i != k and n > k:
                    # In SUMO net format, foes bit is read from right: foes[n - 1 - k]
                    if foes_str[n - 1 - k] == "1":
                        if (k, i) not in foe_pairs:
                            foe_pairs.append((i, k))

        conflicts_by_tls[jid] = foe_pairs

    return conflicts_by_tls


class SignalSafetyVerifier:
    """
    Automated runtime and trace verifier for traffic signal phase transitions and conflict states.
    Derives conflict matrices directly from network topology and verifies SUMO signal strings.
    """

    def __init__(
        self,
        controllable_tls_ids: List[str] = ["A1", "B0", "B1", "B2", "C1"],
        min_green: int = 10,
        max_green: int = 60,
        yellow_duration: int = 3,
        net_xml_path: Optional[Union[str, Path]] = "environment/network/research_3x3/grid.net.xml",
    ):
        self.controllable_tls = list(controllable_tls_ids)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.yellow_duration = int(yellow_duration)

        # Parse microscopic network conflict matrices
        self.conflicts_by_tls: Dict[str, List[Tuple[int, int]]] = {}
        if net_xml_path is not None:
            self.conflicts_by_tls = extract_junction_conflicts_from_net(
                net_xml_path=net_xml_path,
                controllable_tls_ids=self.controllable_tls,
            )

        self.reset()

    def reset(self):
        self.transition_log: List[Dict] = []
        self.violations: List[Dict] = []
        self.last_phase: Dict[str, int] = {}
        self.time_in_phase: Dict[str, int] = {tid: 0 for tid in self.controllable_tls}

    def verify_signal_string(self, tls_id: str, state_str: str, sim_time: float = 0.0) -> List[Dict]:
        """
        Verifies that no conflicting movements both display priority green ('G') simultaneously.
        """
        violations = []
        conflicts = self.conflicts_by_tls.get(tls_id, [])

        for i, k in conflicts:
            if i < len(state_str) and k < len(state_str):
                # Major Green collision check
                if state_str[i] == "G" and state_str[k] == "G":
                    viol = {
                        "type": "CONFLICTING_MAJOR_GREENS",
                        "time": sim_time,
                        "tls": tls_id,
                        "link_i": i,
                        "link_k": k,
                        "state_str": state_str,
                        "description": f"Conflicting links {i} and {k} both display priority Green ('G')",
                    }
                    violations.append(viol)

        return violations

    def record_step(
        self,
        sim_time: float,
        current_phases: Dict[str, int],
        phase_strings: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Records phase states at a simulation step and verifies all safety invariants.

        Args:
            sim_time: current simulation second
            current_phases: mapping from tls_id -> current phase index (e.g. 0, 1, 2, 3)
            phase_strings: optional mapping from tls_id -> signal state string (e.g. 'GGggrrrrGGGg')

        Returns:
            List of violations detected at this step.
        """
        step_violations = []

        for tid in self.controllable_tls:
            if tid not in current_phases:
                continue

            curr_p = current_phases[tid]
            prev_p = self.last_phase.get(tid)

            # 1. Microscopic Conflict Matrix check on live signal string
            if phase_strings and tid in phase_strings:
                state_str = phase_strings[tid]
                str_viols = self.verify_signal_string(tid, state_str, sim_time=sim_time)
                for v in str_viols:
                    self.violations.append(v)
                    step_violations.append(v)

            if prev_p is None:
                # Initial step registration
                self.last_phase[tid] = curr_p
                self.time_in_phase[tid] = 1
                continue

            if curr_p == prev_p:
                self.time_in_phase[tid] += 1
                # Check maximum green overflow (even phases 0, 2 are green)
                if prev_p in [0, 2] and self.time_in_phase[tid] > self.max_green:
                    viol = {
                        "type": "MAX_GREEN_EXCEEDED",
                        "time": sim_time,
                        "tls": tid,
                        "phase": curr_p,
                        "duration": self.time_in_phase[tid],
                        "limit": self.max_green,
                        "description": f"Phase {curr_p} on TLS {tid} exceeded max green limit ({self.time_in_phase[tid]}s > {self.max_green}s)",
                    }
                    self.violations.append(viol)
                    step_violations.append(viol)
            else:
                # Phase transition occurred!
                duration_before_switch = self.time_in_phase[tid]

                # 2. Verify Minimum Green on green phase exit
                if prev_p in [0, 2] and duration_before_switch < self.min_green:
                    viol = {
                        "type": "MIN_GREEN_VIOLATION",
                        "time": sim_time,
                        "tls": tid,
                        "from_phase": prev_p,
                        "to_phase": curr_p,
                        "duration": duration_before_switch,
                        "required_min": self.min_green,
                        "description": f"Phase {prev_p} on TLS {tid} switched after only {duration_before_switch}s (< {self.min_green}s min green)",
                    }
                    self.violations.append(viol)
                    step_violations.append(viol)

                # 3. Verify Yellow Clearance (no direct green -> green transition)
                if prev_p in [0, 2] and curr_p in [0, 2]:
                    viol = {
                        "type": "DIRECT_GREEN_JUMP_NO_YELLOW",
                        "time": sim_time,
                        "tls": tid,
                        "from_phase": prev_p,
                        "to_phase": curr_p,
                        "description": f"Direct green-to-green transition on TLS {tid} from phase {prev_p} to {curr_p} without yellow clearance",
                    }
                    self.violations.append(viol)
                    step_violations.append(viol)

                # 4. Verify Legal Phase Sequence (0 -> 1 -> 2 -> 3 -> 0)
                expected_next = (prev_p + 1) % 4
                if curr_p != expected_next:
                    viol = {
                        "type": "ILLEGAL_PHASE_SEQUENCE",
                        "time": sim_time,
                        "tls": tid,
                        "from_phase": prev_p,
                        "to_phase": curr_p,
                        "expected": expected_next,
                        "description": f"Illegal phase transition on TLS {tid}: jumped from {prev_p} to {curr_p} (expected {expected_next})",
                    }
                    self.violations.append(viol)
                    step_violations.append(viol)

                self.transition_log.append({
                    "time": sim_time,
                    "tls": tid,
                    "from_phase": prev_p,
                    "to_phase": curr_p,
                    "duration": duration_before_switch,
                })

                self.last_phase[tid] = curr_p
                self.time_in_phase[tid] = 1

        return step_violations

    def verify_trace(self, trace_records: List[Dict]) -> Dict:
        """
        Audits a full simulation step trace post-hoc.
        """
        self.reset()
        for row in trace_records:
            sim_time = float(row.get("sim_time", row.get("time", row.get("step", 0))))
            phases = row.get("phases", row.get("current_phases", {}))
            if isinstance(phases, list):
                phases = {tid: phases[i] for i, tid in enumerate(self.controllable_tls) if i < len(phases)}
            strings = row.get("phase_strings", None)
            self.record_step(sim_time=sim_time, current_phases=phases, phase_strings=strings)

        return self.get_summary()

    def get_summary(self) -> Dict:
        """
        Produces an aggregated safety verification report.
        """
        min_green_viols = [v for v in self.violations if v["type"] == "MIN_GREEN_VIOLATION"]
        max_green_viols = [v for v in self.violations if v["type"] == "MAX_GREEN_EXCEEDED"]
        yellow_viols = [v for v in self.violations if v["type"] == "DIRECT_GREEN_JUMP_NO_YELLOW"]
        seq_viols = [v for v in self.violations if v["type"] == "ILLEGAL_PHASE_SEQUENCE"]
        conflict_viols = [v for v in self.violations if v["type"] == "CONFLICTING_MAJOR_GREENS"]

        return {
            "total_transitions": len(self.transition_log),
            "total_violations": len(self.violations),
            "min_green_violations": len(min_green_viols),
            "max_green_violations": len(max_green_viols),
            "yellow_clearance_violations": len(yellow_viols),
            "illegal_sequence_violations": len(seq_viols),
            "conflicting_major_green_violations": len(conflict_viols),
            "conflict_matrices_loaded": {tid: len(pairs) for tid, pairs in self.conflicts_by_tls.items()},
            "verified_safe": len(self.violations) == 0,
        }
