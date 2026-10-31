import traci


class TrafficSignalController:

    """
    Controls the signalized intersections that have
    multiple phases.

    Action:

        0 = keep current phase
        1 = request transition to next phase
    """

    def __init__(self, tls_ids):

        self.tls_ids = list(tls_ids)

        # Only intersections with meaningful
        # multi-phase signal programs.
        self.controllable_tls = []

        for tls_id in self.tls_ids:

            logics = (
                traci.trafficlight
                .getAllProgramLogics(tls_id)
            )

            if not logics:
                continue

            logic = logics[0]

            if len(logic.phases) > 1:

                self.controllable_tls.append(
                    tls_id
                )

        # Keep deterministic ordering
        self.controllable_tls.sort()

    # ======================================================
    # ACTION SPACE INFORMATION
    # ======================================================

    @property
    def num_controllable(self):

        return len(
            self.controllable_tls
        )

    # ======================================================
    # CURRENT PHASE
    # ======================================================

    def get_current_phases(self):

        phases = []

        for tls_id in self.controllable_tls:

            phase = traci.trafficlight.getPhase(
                tls_id
            )

            phases.append(phase)

        return phases

    # ======================================================
    # APPLY ACTION
    # ======================================================

    def apply_action(self, action):

        if len(action) != self.num_controllable:

            raise ValueError(
                f"Expected {self.num_controllable} "
                f"actions, got {len(action)}"
            )

        for tls_id, decision in zip(
            self.controllable_tls,
            action,
        ):

            decision = int(decision)

            # ----------------------------------------------
            # 0 = keep current phase
            # ----------------------------------------------

            if decision == 0:

                continue

            # ----------------------------------------------
            # 1 = request next phase
            # ----------------------------------------------

            if decision == 1:

                current_phase = (
                    traci.trafficlight
                    .getPhase(tls_id)
                )

                logics = (
                    traci.trafficlight
                    .getAllProgramLogics(tls_id)
                )

                if not logics:
                    continue

                num_phases = len(
                    logics[0].phases
                )

                next_phase = (
                    current_phase + 1
                ) % num_phases

                traci.trafficlight.setPhase(
                    tls_id,
                    next_phase,
                )

            else:

                raise ValueError(
                    f"Invalid action {decision} "
                    f"for TLS {tls_id}"
                )