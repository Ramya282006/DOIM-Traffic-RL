"""
DOIM-Traffic-RL: Traffic Signal Safety Shield & Transition State Machine
Enforces legal phase transitions, minimum green time, maximum green time,
and yellow/all-red clearance intervals to prevent phase fluttering and accidents.
Dynamically derives phase transition graphs and clearance semantics from actual
SUMO network XML and tlLogic programs rather than assuming hardcoded 4-phase cycles.
Includes formal movement conflict verification and clearance interval checking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np


VALID_SUMO_SIGNAL_CHARS = {"r", "R", "y", "Y", "g", "G", "u", "o", "O", "s"}


def validate_signal_state_string(state: str) -> bool:
    """
    Validates that a signal state string contains only legitimate SUMO signal characters.
    """
    if not isinstance(state, str) or len(state) == 0:
        return False
    return set(state).issubset(VALID_SUMO_SIGNAL_CHARS)


def verify_phase_conflicts(
    state_str: str,
    conflict_pairs: Optional[List[Tuple[int, int]]] = None,
    yield_pairs: Optional[Union[Set[Tuple[int, int]], List[Tuple[int, int]]]] = None,
) -> Tuple[bool, List[str]]:
    """
    Verifies that a signal state string does not have simultaneous priority green ('G')
    on known conflicting movement indices, and respects SUMO right-of-way yield semantics
    for permissive green ('g').

    Args:
        state_str: SUMO signal state string (e.g., 'GGggrrrrGGggrrrr')
        conflict_pairs: list of 0-indexed (idx1, idx2) linkIndex pairs that represent conflicting movements
        yield_pairs: optional set/list of (idx_yield, idx_priority) where idx_yield yields to idx_priority

    Returns:
        (is_valid, list_of_violation_descriptions)
    """
    violations = []
    if not validate_signal_state_string(state_str):
        violations.append(f"Invalid signal characters in state string: '{state_str}'")
        return False, violations

    if not conflict_pairs:
        return True, []

    yields_set = set(yield_pairs) if yield_pairs is not None else None
    n = len(state_str)

    for idx1, idx2 in conflict_pairs:
        if idx1 < n and idx2 < n:
            s1 = state_str[idx1]
            s2 = state_str[idx2]

            # 1. Conflicting movements cannot both hold unconditional priority green ('G')
            if s1 == "G" and s2 == "G":
                violations.append(
                    f"Conflict violation: movements {idx1} and {idx2} both have priority green 'G' in state '{state_str}'"
                )

            # 2. Permissive green ('g') vs Priority green ('G'):
            # s1 has 'g' and s2 has 'G'. s1 must yield to s2.
            elif s1 == "g" and s2 == "G":
                if yields_set is not None and (idx1, idx2) not in yields_set:
                    violations.append(
                        f"Permissive green violation: movement {idx1} has 'g' and conflicting movement {idx2} has 'G', "
                        f"but movement {idx1} does not have registered right-of-way yield to {idx2}"
                    )
            elif s1 == "G" and s2 == "g":
                if yields_set is not None and (idx2, idx1) not in yields_set:
                    violations.append(
                        f"Permissive green violation: movement {idx2} has 'g' and conflicting movement {idx1} has 'G', "
                        f"but movement {idx2} does not have registered right-of-way yield to {idx1}"
                    )

            # 3. Dual permissive green ('g' and 'g'):
            elif s1 == "g" and s2 == "g":
                if yields_set is not None:
                    y1_to_2 = (idx1, idx2) in yields_set
                    y2_to_1 = (idx2, idx1) in yields_set
                    if not y1_to_2 and not y2_to_1:
                        violations.append(
                            f"Dual permissive conflict violation: movements {idx1} and {idx2} both have 'g' "
                            f"without defined yield hierarchy in state '{state_str}'"
                        )

    return len(violations) == 0, violations


def verify_transition_clearance(
    from_state: str,
    to_state: str,
    clearance_states: List[str],
) -> Tuple[bool, List[str]]:
    """
    Verifies that any movement losing right-of-way (transitioning from green to red)
    is granted an adequate clearance interval (yellow 'y' or 'Y') before turning red.

    Args:
        from_state: Signal state of departure green phase
        to_state: Signal state of arrival green phase
        clearance_states: List of intermediate clearance state strings executed between from and to

    Returns:
        (is_valid, list_of_clearance_violations)
    """
    violations = []
    n = min(len(from_state), len(to_state))
    for k in range(n):
        was_green = from_state[k] in ("G", "g")
        will_be_red = to_state[k] in ("r", "R")

        if was_green and will_be_red:
            # Movement k lost green and turned red: MUST have had yellow in clearance_states
            had_yellow = any(
                k < len(cs) and cs[k] in ("y", "Y") for cs in clearance_states
            )
            if not had_yellow:
                violations.append(
                    f"Clearance violation on movement {k}: changed from '{from_state[k]}' to '{to_state[k]}' "
                    f"without intermediate yellow clearance in {[cs for cs in clearance_states]}"
                )

    return len(violations) == 0, violations


def find_net_file_from_sumocfg(sumo_config_path: Union[str, Path]) -> Optional[Path]:
    """
    Parses a SUMO .sumocfg file and resolves the absolute path to the referenced .net.xml file.
    """
    p = Path(sumo_config_path)
    if not p.is_file():
        return None
    try:
        tree = ET.parse(p)
        root = tree.getroot()
        net_elem = root.find(".//net-file")
        if net_elem is not None and net_elem.get("value"):
            net_val = net_elem.get("value")
            resolved = (p.parent / net_val).resolve()
            if resolved.is_file():
                return resolved
    except Exception:
        pass
    return None


def extract_tls_programs_from_net(
    net_xml_path: Union[str, Path],
    preferred_program_id: Optional[str] = "0",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parses <tlLogic> tags from SUMO network XML and returns phase programs per TLS.
    Properly handles multiple programs per traffic light by selecting matching programID,
    and extracts declared explicit 'next' attributes if specified.
    """
    p = Path(net_xml_path)
    if not p.is_file():
        return {}
    try:
        tree = ET.parse(p)
        root = tree.getroot()
        programs_by_tl: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for tl in root.findall("tlLogic"):
            tid = tl.get("id")
            pid = str(tl.get("programID", "0"))
            phases = []
            for ph in tl.findall("phase"):
                p_entry: Dict[str, Any] = {
                    "duration": float(ph.get("duration", 30)),
                    "state": ph.get("state", ""),
                }
                next_attr = ph.get("next")
                if next_attr is not None:
                    p_entry["next"] = next_attr
                phases.append(p_entry)
            if phases and tid:
                programs_by_tl.setdefault(tid, {})[pid] = phases

        selected_programs: Dict[str, List[Dict[str, Any]]] = {}
        for tid, progs in programs_by_tl.items():
            if preferred_program_id and preferred_program_id in progs:
                selected_programs[tid] = progs[preferred_program_id]
            elif "0" in progs:
                selected_programs[tid] = progs["0"]
            else:
                first_pid = next(iter(progs))
                selected_programs[tid] = progs[first_pid]

        return selected_programs
    except Exception:
        return {}


def extract_tls_index_mapping(
    net_xml_path: Union[str, Path]
) -> Dict[str, Dict[str, Any]]:
    """
    Parses SUMO network XML to establish the explicit relationship between:
      - Junction <request index="..." foes="..." response="..."> entries
      - Traffic Light <connection tl="..." linkIndex="..."> entries
      - Signal state character positions (linkIndex)

    Accounts for:
      - Multiple connections controlled by the same signal index (linkIndex)
      - Complex internal junction topology (:internal_*)
      - Exact foes bitmasks for microscopic geometric conflict detection
      - Right-of-way yield semantics from response bitmasks
      - Reliability verification: validates that all connections are mapped

    Returns:
        Dict[tls_id, {
            "is_reliable": bool,
            "req_to_link": Dict[int, int],
            "link_to_reqs": Dict[int, List[int]],
            "link_to_conns": Dict[int, List[Dict[str, Any]]],
            "conflict_pairs": List[Tuple[int, int]],
            "yield_pairs": List[Tuple[int, int]],
            "unmapped_connections": List[Dict[str, Any]],
            "unmapped_count": int,
        }]
    """
    p = Path(net_xml_path)
    if not p.is_file():
        return {}

    try:
        tree = ET.parse(p)
        root = tree.getroot()

        junction_data: Dict[str, Dict[str, Any]] = {}
        internal_lane_to_junc: Dict[str, str] = {}

        for j in root.findall("junction"):
            jid = j.get("id")
            jtype = j.get("type")
            if jtype == "traffic_light":
                int_lanes = j.get("intLanes", "").split()
                reqs = {}
                for r in j.findall("request"):
                    try:
                        idx = int(r.get("index"))
                        reqs[idx] = {
                            "foes": str(r.get("foes", "")),
                            "response": str(r.get("response", "")),
                            "cont": int(r.get("cont", "0")),
                        }
                    except (ValueError, TypeError):
                        continue
                junction_data[jid] = {"intLanes": int_lanes, "requests": reqs}
            elif jtype == "internal":
                inc = j.get("incLanes", "").split()
                for in_l in inc:
                    internal_lane_to_junc[in_l] = jid

        tls_connections: Dict[str, List[Dict[str, Any]]] = {}
        for c in root.findall("connection"):
            tl = c.get("tl")
            if tl and c.get("linkIndex") is not None:
                try:
                    tls_connections.setdefault(tl, []).append({
                        "from": c.get("from"),
                        "to": c.get("to"),
                        "fromLane": int(c.get("fromLane", 0)),
                        "toLane": int(c.get("toLane", 0)),
                        "linkIndex": int(c.get("linkIndex")),
                        "via": c.get("via"),
                        "dir": c.get("dir"),
                        "state": c.get("state"),
                    })
                except (ValueError, TypeError):
                    continue

        results: Dict[str, Dict[str, Any]] = {}
        for tl, conns in tls_connections.items():
            junc = junction_data.get(tl)
            if not junc:
                results[tl] = {
                    "is_reliable": False,
                    "reason": f"No traffic_light junction found matching TLS ID '{tl}'",
                    "req_to_link": {},
                    "link_to_reqs": {},
                    "link_to_conns": {},
                    "conflict_pairs": [],
                    "yield_pairs": [],
                    "unmapped_connections": conns,
                    "unmapped_count": len(conns),
                }
                continue

            int_lanes = junc["intLanes"]
            reqs = junc["requests"]
            req_to_link: Dict[int, int] = {}
            link_to_reqs: Dict[int, List[int]] = {}
            link_to_conns: Dict[int, List[Dict[str, Any]]] = {}
            unmapped: List[Dict[str, Any]] = []

            for conn in conns:
                l_idx = conn["linkIndex"]
                via = conn.get("via")
                link_to_conns.setdefault(l_idx, []).append(conn)

                req_idx = None
                if via and via in int_lanes:
                    req_idx = int_lanes.index(via)
                elif via and via in internal_lane_to_junc:
                    int_junc_id = internal_lane_to_junc[via]
                    for idx, il in enumerate(int_lanes):
                        if il.startswith(int_junc_id):
                            req_idx = idx
                            break

                if req_idx is not None:
                    req_to_link[req_idx] = l_idx
                    link_to_reqs.setdefault(l_idx, []).append(req_idx)
                else:
                    unmapped.append(conn)

            is_reliable = (len(unmapped) == 0 and len(req_to_link) > 0)

            # Derive linkIndex conflict pairs and yield pairs
            signal_conflicts = set()
            yield_pairs = set()

            for r_idx, r_data in reqs.items():
                foes = r_data["foes"]
                resp = r_data["response"]
                l1 = req_to_link.get(r_idx)
                if l1 is None:
                    continue

                n_bits = len(foes)
                for m in range(n_bits):
                    if foes[-(m + 1)] == "1" and r_idx != m:
                        l2 = req_to_link.get(m)
                        if l2 is not None and l1 != l2:
                            signal_conflicts.add((min(l1, l2), max(l1, l2)))

                n_resp = len(resp)
                for m in range(n_resp):
                    if resp[-(m + 1)] == "1" and r_idx != m:
                        l_prio = req_to_link.get(m)
                        if l_prio is not None and l1 != l_prio:
                            yield_pairs.add((l1, l_prio))

            results[tl] = {
                "is_reliable": is_reliable,
                "req_to_link": req_to_link,
                "link_to_reqs": link_to_reqs,
                "link_to_conns": link_to_conns,
                "conflict_pairs": sorted(list(signal_conflicts)),
                "yield_pairs": sorted(list(yield_pairs)),
                "unmapped_connections": unmapped,
                "unmapped_count": len(unmapped),
            }

        return results
    except Exception:
        return {}


def extract_tls_conflicts_from_net(
    net_xml_path: Union[str, Path]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses SUMO network XML and returns verified linkIndex conflict pairs per traffic light.
    Uses extract_tls_index_mapping to guarantee correct linkIndex resolution rather than
    assuming junction request indices match signal state character indices.
    """
    mappings = extract_tls_index_mapping(net_xml_path)
    return {tl: data.get("conflict_pairs", []) for tl, data in mappings.items()}


class SignalPhaseState:
    """
    Maintains the temporal state and legal phase timing for a single controllable traffic light.
    Dynamically maps phase transitions based on microscopic SUMO signal definitions.
    Supports single-phase junctions, multi-phase systems, mixed green/yellow signals,
    multi-stage clearance intervals (e.g., yellow followed by all-red), and explicit
    non-cyclic transitions.
    """

    def __init__(
        self,
        tls_id: str,
        num_phases: int,
        min_green_seconds: int = 10,
        max_green_seconds: int = 60,
        yellow_seconds: int = 3,
        all_red_seconds: int = 1,
        program_phases: Optional[List[Dict[str, Any]]] = None,
    ):
        self.tls_id = str(tls_id)
        self.num_phases = int(num_phases)
        self.min_green = int(min_green_seconds)
        self.max_green = int(max_green_seconds)
        self.default_yellow_duration = int(yellow_seconds)
        self.all_red_duration = int(all_red_seconds)

        # Runtime state
        self.current_phase_index = 0
        self.time_in_current_phase = 0
        self.is_yellow = False
        self.clearance_timer = 0
        self.clearance_stage_idx = 0
        self.current_clearance_stages: List[Tuple[int, int]] = []
        self.pending_next_green_phase: Optional[int] = None
        self.total_interventions = 0
        self.total_phase_switches = 0

        # Phase metadata & transition graph
        self.program_phases = list(program_phases) if program_phases else []
        self.phase_types: Dict[int, str] = {}
        self.green_phases: List[int] = []
        self.clearance_phases: List[int] = []
        self.all_red_phases: List[int] = []
        self.successors: Dict[int, List[int]] = {}

        # Transition paths:
        # transition_map: green_phase_idx -> (List[(clearance_phase_idx, duration)], nominal_next_green_idx)
        self.transition_map: Dict[int, Tuple[List[Tuple[int, int]], int]] = {}
        # all_clearance_paths: green_phase_idx -> List[(List[(clearance_phase_idx, duration)], destination_green_idx)]
        self.all_clearance_paths: Dict[int, List[Tuple[List[Tuple[int, int]], int]]] = {}
        # reachable_all_red_phases: green_phase_idx -> List[(List[(clearance_phase_idx, duration)], all_red_phase_idx)]
        self.reachable_all_red_phases: Dict[int, List[Tuple[List[Tuple[int, int]], int]]] = {}

        # Fail-safe state tracking
        self.is_fail_safe_locked: bool = False
        self.fail_safe_hold_phase: Optional[int] = None
        self.fail_safe_reason: Optional[str] = None
        self.cannot_safely_progress: bool = False
        self.pending_fail_safe_lock: bool = False
        self.last_intervention_reason: Optional[str] = None
        self.intervention_history: List[Dict[str, Any]] = []

        self._init_program_transitions(self.program_phases)

    def _init_program_transitions(self, program_phases: List[Dict[str, Any]]) -> None:
        """
        Derives valid clearance and green phase transition graphs from program phases.
        Supports explicit non-cyclic 'next' attributes, multi-stage clearance sequences,
        and derives all reachable green and all-red fallback transitions.
        """
        if program_phases and len(program_phases) >= 1:
            n = len(program_phases)
            self.num_phases = n

            for idx, p in enumerate(program_phases):
                st = str(p.get("state", "")).lower()
                has_yellow = "y" in st
                has_green = "g" in st
                is_all_red = set(st).issubset({"r"}) and len(st) > 0

                # Explicit classification of phase semantics
                if has_yellow and has_green:
                    self.phase_types[idx] = "mixed_clearance"
                    self.clearance_phases.append(idx)
                elif has_yellow:
                    self.phase_types[idx] = "yellow_clearance"
                    self.clearance_phases.append(idx)
                elif is_all_red:
                    self.phase_types[idx] = "all_red_clearance"
                    self.clearance_phases.append(idx)
                    self.all_red_phases.append(idx)
                elif has_green:
                    self.phase_types[idx] = "green"
                    self.green_phases.append(idx)
                else:
                    self.phase_types[idx] = "clearance"
                    self.clearance_phases.append(idx)

                # Parse declared successors (explicit 'next' attribute or default cyclic (idx + 1) % n)
                if "next" in p and p["next"] is not None:
                    next_val = p["next"]
                    if isinstance(next_val, str):
                        self.successors[idx] = [int(x) for x in next_val.split()]
                    elif isinstance(next_val, (list, tuple, set)):
                        self.successors[idx] = [int(x) for x in next_val]
                    else:
                        self.successors[idx] = [int(next_val)]
                else:
                    self.successors[idx] = [(idx + 1) % n]

            # Case A: Exactly 1 phase (e.g., perimeter un-signalized green node A0, A2, C0, C2)
            if len(self.green_phases) <= 1 and len(self.clearance_phases) == 0:
                self.transition_map = {}
                self.all_clearance_paths = {}
                self.reachable_all_red_phases = {}
                return

            # Case B: Multi-phase program with green and clearance phases
            if self.green_phases and self.clearance_phases:
                for g_idx in self.green_phases:
                    self.all_clearance_paths[g_idx] = []
                    self.reachable_all_red_phases[g_idx] = []

                    # Explore directed paths through clearance phases starting from g_idx
                    start_succs = self.successors.get(g_idx, [])
                    for s in start_succs:
                        if s in self.clearance_phases:
                            queue = [[s]]
                            visited_paths = set()
                            while queue:
                                curr_path = queue.pop(0)
                                path_tuple = tuple(curr_path)
                                if path_tuple in visited_paths:
                                    continue
                                visited_paths.add(path_tuple)

                                last_c = curr_path[-1]
                                next_cands = self.successors.get(last_c, [])

                                for cand in next_cands:
                                    if cand in self.green_phases and cand != g_idx:
                                        durations = [
                                            max(1, int(round(float(program_phases[cp].get("duration", self.default_yellow_duration)))))
                                            for cp in curr_path
                                        ]
                                        stages = list(zip(curr_path, durations))
                                        self.all_clearance_paths[g_idx].append((stages, cand))
                                    elif cand in self.clearance_phases and cand not in curr_path:
                                        queue.append(curr_path + [cand])

                    # Check for reachable all-red phases along clearance paths starting from g_idx
                    for stages, dest_g in self.all_clearance_paths[g_idx]:
                        for s_idx, (cp, dur) in enumerate(stages):
                            if cp in self.all_red_phases:
                                stages_before_red = stages[:s_idx]
                                entry = (stages_before_red, cp)
                                if entry not in self.reachable_all_red_phases[g_idx]:
                                    self.reachable_all_red_phases[g_idx].append(entry)

                    # Also check if g_idx directly has an all-red successor in clearance phases
                    for s in start_succs:
                        if s in self.all_red_phases:
                            entry = ([], s)
                            if entry not in self.reachable_all_red_phases[g_idx]:
                                self.reachable_all_red_phases[g_idx].append(entry)

                    # Primary transition map: select the first discovered path or cyclic path
                    if self.all_clearance_paths[g_idx]:
                        self.transition_map[g_idx] = self.all_clearance_paths[g_idx][0]
                    else:
                        fallback_c = (g_idx + 1) % n
                        fallback_g = (g_idx + 2) % n if n >= 3 else g_idx
                        self.transition_map[g_idx] = ([(fallback_c, self.default_yellow_duration)], fallback_g)
                return

            # Case C: Multiple green phases without defined clearance phases
            if len(self.green_phases) >= 2 and not self.clearance_phases:
                for idx, g_idx in enumerate(self.green_phases):
                    next_g = self.green_phases[(idx + 1) % len(self.green_phases)]
                    self.transition_map[g_idx] = ([(g_idx, self.default_yellow_duration)], next_g)
                    self.all_clearance_paths[g_idx] = [self.transition_map[g_idx]]
                    self.reachable_all_red_phases[g_idx] = []
                return

        # Fallback to standard cyclic transition model if no program provided
        n = self.num_phases
        if n <= 1:
            self.green_phases = [0]
            self.transition_map = {}
            self.all_clearance_paths = {}
            self.reachable_all_red_phases = {}
            return

        if n % 2 == 0 and n >= 4:
            # Alternating Green / Clearance: 0(G), 1(Y), 2(G), 3(Y)...
            for k in range(0, n, 2):
                self.green_phases.append(k)
                clearance = (k + 1) % n
                next_green = (k + 2) % n
                self.clearance_phases.append(clearance)
                self.successors[k] = [clearance]
                self.successors[clearance] = [next_green]
                stages = [(clearance, self.default_yellow_duration)]
                self.transition_map[k] = (stages, next_green)
                self.all_clearance_paths[k] = [(stages, next_green)]
                self.reachable_all_red_phases[k] = []
        else:
            for k in range(n):
                self.green_phases.append(k)
                clearance = (k + 1) % n
                next_green = (k + 1) % n
                self.clearance_phases.append(clearance)
                self.successors[k] = [clearance]
                self.successors[clearance] = [next_green]
                stages = [(clearance, self.default_yellow_duration)]
                self.transition_map[k] = (stages, next_green)
                self.all_clearance_paths[k] = [(stages, next_green)]
                self.reachable_all_red_phases[k] = []

    def get_clearance_stages(self, current_phase: int) -> Tuple[List[Tuple[int, int]], int]:
        """
        Returns (list_of_clearance_stages, next_green_phase) for current green phase.
        Each stage is (phase_index, duration_seconds).
        """
        if current_phase in self.transition_map:
            return self.transition_map[current_phase]
        if self.num_phases <= 1:
            return [], current_phase
        # Safety fallback
        clearance = (current_phase + 1) % self.num_phases
        next_green = (current_phase + 2) % self.num_phases if self.num_phases >= 4 else clearance
        return [(clearance, self.default_yellow_duration)], next_green

    def step_clock(self) -> None:
        """Advances internal phase timer by 1 second."""
        self.time_in_current_phase += 1
        if self.is_yellow:
            self.clearance_timer += 1

    def can_switch_phase(self) -> bool:
        """Phase can only be switched if not currently in clearance or fail-safe lock, min green has elapsed, and switches are possible."""
        if self.is_fail_safe_locked:
            return False
        if self.is_yellow:
            return False
        if len(self.green_phases) <= 1:
            return False
        return self.time_in_current_phase >= self.min_green

    def is_max_green_exceeded(self) -> bool:
        """True if current green phase has exceeded maximum duration."""
        if self.is_fail_safe_locked:
            return False
        if self.is_yellow:
            return False
        if len(self.green_phases) <= 1:
            return False
        return self.time_in_current_phase >= self.max_green


class TrafficSafetyShield:
    """
    Supervises multi-intersection action execution, enforcing legal signal safety constraints.
    Supports dynamic phase program extraction from SUMO network XML or TraCI logics.
    Enforces conflict-free clearance intervals and protects against unsafe phase jumps using
    movement conflicts derived from microscopic SUMO junction connection topologies.
    """

    def __init__(
        self,
        controllable_tls_ids: List[str],
        tls_phase_counts: Optional[Dict[str, int]] = None,
        tls_programs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        net_xml_path: Optional[Union[str, Path]] = None,
        tls_conflicts: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        min_green: int = 10,
        max_green: int = 60,
        yellow_duration: int = 3,
        all_red_duration: int = 1,
    ):
        self.controllable_tls = list(controllable_tls_ids)
        self.states: Dict[str, SignalPhaseState] = {}
        self.net_xml_path = Path(net_xml_path) if net_xml_path else None

        # Parse program phases from network XML if provided
        programs_from_xml = extract_tls_programs_from_net(net_xml_path) if net_xml_path else {}
        programs = tls_programs or programs_from_xml
        counts = tls_phase_counts or {}

        # Parse movement conflicts and right-of-way yield semantics from network XML
        self.index_mappings = extract_tls_index_mapping(net_xml_path) if net_xml_path else {}
        conflicts_from_xml = {tid: data.get("conflict_pairs", []) for tid, data in self.index_mappings.items()}
        self.conflicts: Dict[str, List[Tuple[int, int]]] = tls_conflicts or conflicts_from_xml
        self.yield_pairs: Dict[str, Set[Tuple[int, int]]] = {
            tid: set(data.get("yield_pairs", [])) for tid, data in self.index_mappings.items()
        }

        for tid in self.controllable_tls:
            prog = programs.get(tid)
            n_phases = len(prog) if prog else counts.get(tid, 4)
            self.states[tid] = SignalPhaseState(
                tls_id=tid,
                num_phases=n_phases,
                min_green_seconds=min_green,
                max_green_seconds=max_green,
                yellow_seconds=yellow_duration,
                all_red_seconds=all_red_duration,
                program_phases=prog,
            )

    def filter_action(
        self,
        raw_actions: np.ndarray,
        current_phases: List[int],
    ) -> Tuple[np.ndarray, List[Optional[int]], int]:
        """
        Filters raw agent actions against safety constraints.
        
        Args:
            raw_actions: [num_controllable] binary actions (0=keep, 1=request next)
            current_phases: current active phase index per controllable TLS
            
        Returns:
            Tuple of:
              - safe_actions: filtered binary actions
              - target_phases: explicit target phase to set in TraCI (or None if no change)
              - total_interventions: count of actions modified by safety shield this step
        """
        raw_actions = np.asarray(raw_actions, dtype=np.int64)
        safe_actions = raw_actions.copy()
        target_phases: List[Optional[int]] = [None] * len(self.controllable_tls)
        step_interventions = 0

        for i, tid in enumerate(self.controllable_tls):
            state = self.states[tid]
            state.step_clock()
            sim_phase = current_phases[i]

            # Fail-safe check 1: if state is locked into fail-safe, maintain hold and reject switch
            if state.is_fail_safe_locked:
                safe_actions[i] = 0
                state.total_interventions += 1
                step_interventions += 1
                if state.fail_safe_hold_phase is not None:
                    target_phases[i] = state.fail_safe_hold_phase
                continue

            # Fail-safe check 2: if mapping for this TLS could not be reliably established, enforce fail-safe
            mapping = self.index_mappings.get(tid)
            is_unreliable_mapping = bool(mapping and not mapping.get("is_reliable", True))

            # Synchronize state with simulator phase if unexpected jump occurred
            if not state.is_yellow and sim_phase != state.current_phase_index:
                state.current_phase_index = sim_phase
                state.time_in_current_phase = 0

            # 1. Handle in-progress clearance (yellow/all-red) transition stages
            if state.is_yellow:
                safe_actions[i] = 0
                if state.current_clearance_stages:
                    current_stage_dur = state.current_clearance_stages[state.clearance_stage_idx][1]
                    if state.clearance_timer >= current_stage_dur:
                        # Current clearance stage elapsed: check if another clearance stage remains
                        if state.clearance_stage_idx + 1 < len(state.current_clearance_stages):
                            state.clearance_stage_idx += 1
                            state.clearance_timer = 0
                            next_stage_phase = state.current_clearance_stages[state.clearance_stage_idx][0]
                            target_phases[i] = next_stage_phase
                        else:
                            # All clearance stages complete -> enter next destination phase
                            state.is_yellow = False
                            state.current_phase_index = state.pending_next_green_phase
                            state.time_in_current_phase = 0
                            state.clearance_timer = 0
                            state.clearance_stage_idx = 0
                            state.current_clearance_stages = []
                            target_phases[i] = state.current_phase_index
                            if state.pending_fail_safe_lock:
                                state.is_fail_safe_locked = True
                                state.fail_safe_hold_phase = state.current_phase_index
                                state.pending_fail_safe_lock = False
                            state.pending_next_green_phase = None
                else:
                    state.is_yellow = False
                    state.current_phase_index = state.pending_next_green_phase
                    state.time_in_current_phase = 0
                    target_phases[i] = state.current_phase_index
                    if state.pending_fail_safe_lock:
                        state.is_fail_safe_locked = True
                        state.fail_safe_hold_phase = state.current_phase_index
                        state.pending_fail_safe_lock = False
                    state.pending_next_green_phase = None
                continue

            # 2. Check if agent requests phase change or max green is exceeded
            agent_requested_switch = (raw_actions[i] == 1)
            force_max_green = state.is_max_green_exceeded()

            if agent_requested_switch or force_max_green:
                if is_unreliable_mapping and not force_max_green:
                    # Fail-safe: Reject unverified agent switch request on unreliable or unmapped connection topology
                    safe_actions[i] = 0
                    state.total_interventions += 1
                    step_interventions += 1
                    reason = f"REJECTED_UNRELIABLE_MAPPING: TLS {tid} topology is unmapped or ambiguous"
                    state.last_intervention_reason = reason
                    continue
                elif state.can_switch_phase() or force_max_green:
                    # Legal transition: derive clearance stages and destination green
                    curr = state.current_phase_index
                    stages, nominal_next = state.get_clearance_stages(curr)
                    if not stages:
                        # No clearance defined (e.g. 1-phase or immediate): safe action is 0
                        safe_actions[i] = 0
                        continue

                    # Verify nominal next green phase does not violate network-derived movement conflicts
                    is_valid_phase = True
                    violations = []
                    if nominal_next < len(state.program_phases):
                        next_state_str = state.program_phases[nominal_next].get("state", "")
                        tls_conf = self.conflicts.get(tid, [])
                        tls_yield = self.yield_pairs.get(tid, set())
                        is_valid_phase, violations = verify_phase_conflicts(next_state_str, tls_conf, tls_yield)

                    if not is_valid_phase:
                        if not force_max_green:
                            # Voluntary switch requested by agent: safely reject and stay in current green
                            safe_actions[i] = 0
                            state.total_interventions += 1
                            step_interventions += 1
                            reason = f"REJECTED_UNSAFE_SWITCH: next_green {nominal_next} has conflicts on TLS {tid}"
                            state.last_intervention_reason = reason
                            continue
                        else:
                            # FORCED MAX GREEN: Cannot stay in current green indefinitely!
                            # Verify valid reachable fallbacks through configured SUMO transition sequences
                            destination_phase = None
                            destination_stages = None
                            fallback_reason = None

                            # Option A: Check alternative green phases reachable via valid configured clearance paths
                            all_paths = state.all_clearance_paths.get(curr, [])
                            for alt_stages, alt_green in all_paths:
                                if alt_green != nominal_next and alt_green < len(state.program_phases):
                                    cand_str = state.program_phases[alt_green].get("state", "")
                                    cand_valid, _ = verify_phase_conflicts(
                                        cand_str, self.conflicts.get(tid, []), self.yield_pairs.get(tid, set())
                                    )
                                    if cand_valid:
                                        destination_phase = alt_green
                                        destination_stages = alt_stages
                                        fallback_reason = (
                                            f"MAX_GREEN_FALLBACK_ALT_GREEN: nominal next_green {nominal_next} conflicting; "
                                            f"routed to valid reachable green phase {alt_green} via configured clearance on TLS {tid}"
                                        )
                                        break

                            # Option B: Check if an all-red phase is reachable along configured clearance sequence
                            if destination_phase is None:
                                reachable_reds = state.reachable_all_red_phases.get(curr, [])
                                if reachable_reds:
                                    red_stages, red_phase = reachable_reds[0]
                                    destination_phase = red_phase
                                    destination_stages = red_stages
                                    fallback_reason = (
                                        f"MAX_GREEN_FALLBACK_ALL_RED: nominal next_green {nominal_next} conflicting; "
                                        f"routed to reachable all-red phase {red_phase} in fail-safe hold on TLS {tid}"
                                    )
                                    state.pending_fail_safe_lock = True
                                    state.fail_safe_reason = fallback_reason

                            # Option C: No verified safe transition exists in configured SUMO program
                            if destination_phase is None:
                                # Enter explicitly documented fail-safe error state
                                # Report that the controller cannot safely progress
                                # Do NOT silently leave conflicting green active indefinitely
                                # Do NOT use yellow clearance as a permanent hold state
                                state.is_fail_safe_locked = True
                                state.cannot_safely_progress = True
                                fail_reason = (
                                    f"NO_VERIFIED_SAFE_TRANSITION: TLS {tid} max green exceeded on phase {curr}; "
                                    f"successor {nominal_next} is conflicting, and no reachable alternative green or "
                                    f"all-red clearance transition exists in SUMO program. Controller cannot safely progress."
                                )
                                state.fail_safe_reason = fail_reason
                                state.last_intervention_reason = fail_reason
                                state.intervention_history.append({
                                    "step_time": state.time_in_current_phase,
                                    "reason": fail_reason,
                                })
                                state.total_interventions += 1
                                step_interventions += 1
                                safe_actions[i] = 0
                                continue

                            # Reachable fallback found (Option A or Option B):
                            # Record intervention and transition through mandatory clearance stages
                            state.last_intervention_reason = fallback_reason
                            state.intervention_history.append({
                                "step_time": state.time_in_current_phase,
                                "reason": fallback_reason,
                            })
                            state.total_interventions += 1
                            step_interventions += 1
                            stages = destination_stages
                    else:
                        destination_phase = nominal_next

                    # Mandatory clearance intervals (yellow / all-red) MUST be executed
                    # Fallback cannot bypass yellow or other required clearance intervals
                    if stages:
                        state.is_yellow = True
                        state.clearance_timer = 0
                        state.clearance_stage_idx = 0
                        state.current_clearance_stages = stages
                        state.pending_next_green_phase = destination_phase
                        state.total_phase_switches += 1
                        target_phases[i] = stages[0][0]
                        safe_actions[i] = 1
                    else:
                        state.current_phase_index = destination_phase
                        state.time_in_current_phase = 0
                        state.total_phase_switches += 1
                        target_phases[i] = destination_phase
                        safe_actions[i] = 1
                        if state.pending_fail_safe_lock:
                            state.is_fail_safe_locked = True
                            state.fail_safe_hold_phase = destination_phase
                            state.pending_fail_safe_lock = False
                else:
                    # VIOLATION PREVENTED: Rejected switch because min green has not elapsed or TLS cannot switch!
                    safe_actions[i] = 0
                    state.total_interventions += 1
                    step_interventions += 1
                    state.last_intervention_reason = f"REJECTED_MIN_GREEN: min green duration {state.min_green}s not elapsed"

        return safe_actions, target_phases, step_interventions

    def get_total_interventions(self) -> int:
        """Returns cumulative count of unsafe actions rejected across all lights."""
        return sum(s.total_interventions for s in self.states.values())
