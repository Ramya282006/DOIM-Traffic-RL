import numpy as np


class MultiObjectiveEngine:

    def __init__(self):

        self.previous_completed = 0

    def compute_objectives(
        self,
        network_states,
        completed_vehicles,
    ):

        queues = []
        waiting = []
        emissions = []

        for tls_id, state in network_states.items():

            queues.append(state["queue"])
            waiting.append(state["waiting"])
            emissions.append(state["co2"])

        # --------------------------------------------------
        # Objective 1: Total waiting
        # --------------------------------------------------

        total_waiting = float(
            np.sum(waiting)
        )

        # --------------------------------------------------
        # Objective 2: Total queue
        # --------------------------------------------------

        total_queue = float(
            np.sum(queues)
        )

        # --------------------------------------------------
        # Objective 3: Total instantaneous CO2
        # --------------------------------------------------

        total_co2 = float(
            np.sum(emissions)
        )

        # --------------------------------------------------
        # Objective 4: Throughput
        #
        # Incremental number of completed vehicles
        # --------------------------------------------------

        throughput = max(
            0,
            completed_vehicles - self.previous_completed
        )

        self.previous_completed = completed_vehicles

        # --------------------------------------------------
        # Objective 5: Fairness
        #
        # Initial proxy:
        # variance of intersection waiting times
        # --------------------------------------------------

        fairness = float(
            np.var(waiting)
        )

        return np.array(
            [
                total_waiting,
                total_queue,
                total_co2,
                float(throughput),
                fairness,
            ],
            dtype=np.float32,
        )