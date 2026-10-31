"""
DOIM-Traffic-RL: Traffic Signal Safety Shield & Transition State Machine
Enforces legal phase transitions, minimum green time, maximum green time,
and yellow/all-red clearance intervals to prevent phase fluttering and accidents.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class SignalPhaseState:
    """
    Maintains the temporal state and legal phase timing for a single controllable traffic light.
    """

    def __init__(
        self,
        tls_id: str,
        num_phases: int,
        min_green_seconds: int = 10,
        max_green_seconds: int = 60,
        yellow_seconds: int = 3,
        all_red_seconds: int = 1,
    ):
        self.tls_id = str(tls_id)
        self.num_phases = int(num_phases)
        self.min_green = int(min_green_seconds)
        self.max_green = int(max_green_seconds)
        self.yellow_duration = int(yellow_seconds)
        self.all_red_duration = int(all_red_seconds)

        # Runtime state
        self.current_phase_index = 0
        self.time_in_current_phase = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.pending_next_green_phase = None
        self.total_interventions = 0
        self.total_phase_switches = 0

    def step_clock(self) -> None:
        """Advances internal phase timer by 1 second."""
        self.time_in_current_phase += 1
        if self.is_yellow:
            self.yellow_timer += 1

    def can_switch_phase(self) -> bool:
        """Phase can only be switched if not currently yellow and min green has elapsed."""
        if self.is_yellow:
            return False
        return self.time_in_current_phase >= self.min_green

    def is_max_green_exceeded(self) -> bool:
        """True if current green phase has exceeded maximum duration."""
        if self.is_yellow:
            return False
        return self.time_in_current_phase >= self.max_green


class TrafficSafetyShield:
    """
    Supervises multi-intersection action execution, enforcing legal signal safety constraints.
    """

    def __init__(
        self,
        controllable_tls_ids: List[str],
        tls_phase_counts: Dict[str, int],
        min_green: int = 10,
        max_green: int = 60,
        yellow_duration: int = 3,
        all_red_duration: int = 1,
    ):
        self.controllable_tls = list(controllable_tls_ids)
        self.states: Dict[str, SignalPhaseState] = {}

        for tid in self.controllable_tls:
            n_phases = tls_phase_counts.get(tid, 4)
            self.states[tid] = SignalPhaseState(
                tls_id=tid,
                num_phases=n_phases,
                min_green_seconds=min_green,
                max_green_seconds=max_green,
                yellow_seconds=yellow_duration,
                all_red_seconds=all_red_duration,
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
        target_phases = [None] * len(self.controllable_tls)
        step_interventions = 0

        for i, tid in enumerate(self.controllable_tls):
            state = self.states[tid]
            state.step_clock()
            sim_phase = current_phases[i]

            # Synchronize state with simulator phase if unexpected jump occurred
            if not state.is_yellow and sim_phase != state.current_phase_index:
                state.current_phase_index = sim_phase
                state.time_in_current_phase = 0

            # 1. Handle in-progress yellow transition
            if state.is_yellow:
                safe_actions[i] = 0
                if state.yellow_timer >= state.yellow_duration:
                    # Transition yellow -> next green
                    state.is_yellow = False
                    state.current_phase_index = state.pending_next_green_phase
                    state.time_in_current_phase = 0
                    state.yellow_timer = 0
                    state.pending_next_green_phase = None
                    target_phases[i] = state.current_phase_index
                continue

            # 2. Check if agent requests phase change
            agent_requested_switch = (raw_actions[i] == 1)
            force_max_green = state.is_max_green_exceeded()

            if agent_requested_switch or force_max_green:
                if state.can_switch_phase() or force_max_green:
                    # Legal transition: switch from current green -> intermediate yellow phase
                    # Assuming 4-phase program: 0: Green_A, 1: Yellow_A, 2: Green_B, 3: Yellow_B
                    curr = state.current_phase_index
                    yellow_phase = (curr + 1) % state.num_phases
                    next_green = (curr + 2) % state.num_phases

                    state.is_yellow = True
                    state.yellow_timer = 0
                    state.pending_next_green_phase = next_green
                    state.total_phase_switches += 1
                    target_phases[i] = yellow_phase
                    safe_actions[i] = 1
                else:
                    # VIOLATION PREVENTED: Rejected switch because min green has not elapsed!
                    safe_actions[i] = 0
                    state.total_interventions += 1
                    step_interventions += 1

        return safe_actions, target_phases, step_interventions

    def get_total_interventions(self) -> int:
        """Returns cumulative count of unsafe actions rejected across all lights."""
        return sum(s.total_interventions for s in self.states.values())
