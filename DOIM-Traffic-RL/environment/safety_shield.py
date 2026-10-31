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
) -> Tuple[bool, List[str]]:
    """
    Verifies that a signal state string does not have simultaneous priority green ('G')
    on known conflicting movement indices.

    Args:
        state_str: SUMO signal state string (e.g., 'GGggrrrrGGggrrrr')
        conflict_pairs: list of 0-indexed (idx1, idx2) pairs that represent conflicting movements

    Returns:
        (is_valid, list_of_violation_descriptions)
    """
    violations = []
    if not validate_signal_state_string(state_str):
        violations.append(f"Invalid signal characters in state string: '{state_str}'")
        return False, violations

    if not conflict_pairs:
        return True, []

    n = len(state_str)
    for idx1, idx2 in conflict_pairs:
        if idx1 < n and idx2 < n:
            s1 = state_str[idx1]
            s2 = state_str[idx2]
            # Conflicting movements cannot both hold priority green ('G')
            if s1 == "G" and s2 == "G":
                violations.append(
                    f"Conflict violation: movements {idx1} and {idx2} both have priority green 'G' in state '{state_str}'"
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
    Properly handles multiple programs per traffic light by selecting matching programID.
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
                phases.append({
                    "duration": float(ph.get("duration", 30)),
                    "state": ph.get("state", ""),
                })
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


def extract_tls_conflicts_from_net(
    net_xml_path: Union[str, Path]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses <junction type="traffic_light"> and their <request> foes bitmasks
    from SUMO network XML to derive exact microscopic conflict pairs.
    Each pair (idx1, idx2) represents intersecting/conflicting movements that
    cannot simultaneously hold priority green 'G'.
    """
    p = Path(net_xml_path)
    if not p.is_file():
        return {}
    try:
        tree = ET.parse(p)
        root = tree.getroot()
        conflicts: Dict[str, List[Tuple[int, int]]] = {}
        for j in root.findall("junction"):
            if j.get("type") == "traffic_light":
                jid = j.get("id")
                reqs = j.findall("request")
                j_conflicts = set()
                for r in reqs:
                    try:
                        k = int(r.get("index"))
                        foes = str(r.get("foes", ""))
                        n_bits = len(foes)
                        for m in range(n_bits):
                            if foes[-(m + 1)] == "1" and k != m:
                                pair = (min(k, m), max(k, m))
                                j_conflicts.add(pair)
                    except (ValueError, TypeError):
                        continue
                if j_conflicts and jid:
                    conflicts[jid] = sorted(list(j_conflicts))
        return conflicts
    except Exception:
        return {}


class SignalPhaseState:
    """
    Maintains the temporal state and legal phase timing for a single controllable traffic light.
    Dynamically maps phase transitions based on microscopic SUMO signal definitions.
    Supports single-phase junctions, multi-phase systems, mixed green/yellow signals,
    and multi-stage clearance intervals (e.g., yellow followed by all-red).
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

        # Phase metadata
        self.program_phases = list(program_phases) if program_phases else []
        self.phase_types: Dict[int, str] = {}
        self.green_phases: List[int] = []
        self.clearance_phases: List[int] = []
        # Dynamic transition map: green_phase_idx -> (List[(clearance_phase_idx, duration)], next_green_phase_idx)
        self.transition_map: Dict[int, Tuple[List[Tuple[int, int]], int]] = {}

        self._init_program_transitions(self.program_phases)

    def _init_program_transitions(self, program_phases: List[Dict[str, Any]]) -> None:
        """
        Derives valid clearance and green phase transition graphs from program phases.
        Correctly classifies mixed green/yellow states, yellow clearance, and all-red clearance.
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
                else:
                    self.phase_types[idx] = "green"
                    self.green_phases.append(idx)

            # Case A: Exactly 1 phase (e.g., perimeter un-signalized green node A0, A2, C0, C2)
            if len(self.green_phases) <= 1 and len(self.clearance_phases) == 0:
                self.transition_map = {}
                return

            # Case B: Multi-phase program with green and clearance phases
            if self.green_phases and self.clearance_phases:
                for g_idx in self.green_phases:
                    # Collect intermediate clearance stages between g_idx and the next green phase
                    stages: List[Tuple[int, int]] = []
                    next_green_cand = None
                    for offset in range(1, n):
                        cand = (g_idx + offset) % n
                        if cand in self.clearance_phases and next_green_cand is None:
                            dur = int(round(float(program_phases[cand].get("duration", self.default_yellow_duration))))
                            stages.append((cand, max(1, dur)))
                        elif cand in self.green_phases:
                            next_green_cand = cand
                            break

                    if not stages:
                        # If no explicit clearance in between, synthesize default yellow clearance
                        fallback_c = (g_idx + 1) % n
                        stages = [(fallback_c, self.default_yellow_duration)]
                    
                    next_green = next_green_cand if next_green_cand is not None else (stages[-1][0] + 1) % n
                    self.transition_map[g_idx] = (stages, next_green)
                return

            # Case C: Multiple green phases without defined clearance phases
            if len(self.green_phases) >= 2 and not self.clearance_phases:
                for idx, g_idx in enumerate(self.green_phases):
                    next_g = self.green_phases[(idx + 1) % len(self.green_phases)]
                    # Insert virtual clearance
                    self.transition_map[g_idx] = ([(g_idx, self.default_yellow_duration)], next_g)
                return

        # Fallback to standard cyclic transition model if no program provided
        n = self.num_phases
        if n <= 1:
            self.green_phases = [0]
            self.transition_map = {}
            return

        if n % 2 == 0 and n >= 4:
            # Alternating Green / Clearance: 0(G), 1(Y), 2(G), 3(Y)...
            for k in range(0, n, 2):
                self.green_phases.append(k)
                clearance = (k + 1) % n
                next_green = (k + 2) % n
                self.clearance_phases.append(clearance)
                self.transition_map[k] = ([(clearance, self.default_yellow_duration)], next_green)
        else:
            for k in range(n):
                self.green_phases.append(k)
                clearance = (k + 1) % n
                next_green = (k + 1) % n
                self.transition_map[k] = ([(clearance, self.default_yellow_duration)], next_green)

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
        """Phase can only be switched if not currently in clearance, min green has elapsed, and switches are possible."""
        if self.is_yellow:
            return False
        if len(self.green_phases) <= 1:
            return False
        return self.time_in_current_phase >= self.min_green

    def is_max_green_exceeded(self) -> bool:
        """True if current green phase has exceeded maximum duration."""
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

        # Parse movement conflicts from network XML if provided
        conflicts_from_xml = extract_tls_conflicts_from_net(net_xml_path) if net_xml_path else {}
        self.conflicts: Dict[str, List[Tuple[int, int]]] = tls_conflicts or conflicts_from_xml

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

            # Synchronize state with simulator phase if unexpected jump occurred
            if not state.is_yellow and sim_phase != state.current_phase_index:
                state.current_phase_index = sim_phase
                state.time_in_current_phase = 0

            # 1. Handle in-progress clearance (yellow/all-red) transition stages
            if state.is_yellow:
                safe_actions[i] = 0
                current_stage_dur = state.current_clearance_stages[state.clearance_stage_idx][1]
                if state.clearance_timer >= current_stage_dur:
                    # Current clearance stage elapsed: check if another clearance stage remains
                    if state.clearance_stage_idx + 1 < len(state.current_clearance_stages):
                        state.clearance_stage_idx += 1
                        state.clearance_timer = 0
                        next_stage_phase = state.current_clearance_stages[state.clearance_stage_idx][0]
                        target_phases[i] = next_stage_phase
                    else:
                        # All clearance stages complete -> enter next green phase
                        state.is_yellow = False
                        state.current_phase_index = state.pending_next_green_phase
                        state.time_in_current_phase = 0
                        state.clearance_timer = 0
                        state.clearance_stage_idx = 0
                        state.current_clearance_stages = []
                        state.pending_next_green_phase = None
                        target_phases[i] = state.current_phase_index
                continue

            # 2. Check if agent requests phase change or max green is exceeded
            agent_requested_switch = (raw_actions[i] == 1)
            force_max_green = state.is_max_green_exceeded()

            if agent_requested_switch or force_max_green:
                if state.can_switch_phase() or force_max_green:
                    # Legal transition: derive clearance stages and destination green
                    curr = state.current_phase_index
                    stages, next_green = state.get_clearance_stages(curr)
                    if not stages:
                        # No clearance defined (e.g. 1-phase or immediate): safe action is 0
                        safe_actions[i] = 0
                        continue

                    # Verify next green phase does not violate network-derived movement conflicts
                    if next_green < len(state.program_phases):
                        next_state_str = state.program_phases[next_green].get("state", "")
                        tls_conf = self.conflicts.get(tid, [])
                        is_valid_phase, violations = verify_phase_conflicts(next_state_str, tls_conf)
                        if not is_valid_phase:
                            # Conflicting green detected: reject transition
                            safe_actions[i] = 0
                            state.total_interventions += 1
                            step_interventions += 1
                            continue

                    state.is_yellow = True
                    state.clearance_timer = 0
                    state.clearance_stage_idx = 0
                    state.current_clearance_stages = stages
                    state.pending_next_green_phase = next_green
                    state.total_phase_switches += 1
                    target_phases[i] = stages[0][0]
                    safe_actions[i] = 1
                else:
                    # VIOLATION PREVENTED: Rejected switch because min green has not elapsed or TLS cannot switch!
                    safe_actions[i] = 0
                    state.total_interventions += 1
                    step_interventions += 1

        return safe_actions, target_phases, step_interventions

    def get_total_interventions(self) -> int:
        """Returns cumulative count of unsafe actions rejected across all lights."""
        return sum(s.total_interventions for s in self.states.values())
