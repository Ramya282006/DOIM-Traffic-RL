import traci
import numpy as np


class TrafficStateExtractor:

    def __init__(self, tls_ids):

        self.tls_ids = list(tls_ids)

        # Mapping:
        # traffic light -> incoming lanes
        self.incoming_lanes = {}

        for tls_id in self.tls_ids:

            links = traci.trafficlight.getControlledLinks(tls_id)

            lanes = set()

            for link_group in links:

                if not link_group:
                    continue

                # SUMO returns groups containing connection tuples.
                for connection in link_group:

                    if connection is None:
                        continue

                    if len(connection) >= 1:

                        incoming_lane = connection[0]

                        lanes.add(incoming_lane)

            self.incoming_lanes[tls_id] = sorted(lanes)

    # ======================================================
    # Extract state for one intersection
    # ======================================================

    def get_intersection_state(self, tls_id):

        lanes = self.incoming_lanes[tls_id]

        total_queue = 0
        total_waiting = 0.0
        total_speed = 0.0
        total_density = 0.0
        total_co2 = 0.0

        vehicle_ids = set()

        lane_count = 0

        for lane_id in lanes:

            lane_count += 1

            # Vehicles currently on this incoming lane
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)

            vehicle_ids.update(vehicles)

            # Queue
            halting = traci.lane.getLastStepHaltingNumber(lane_id)

            total_queue += halting

            # Density
            total_density += traci.lane.getLastStepVehicleNumber(lane_id)

            # Speed
            lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)

            total_speed += lane_speed

        # --------------------------------------------------
        # Vehicle-level waiting + emissions
        # --------------------------------------------------

        for vehicle_id in vehicle_ids:

            total_waiting += traci.vehicle.getAccumulatedWaitingTime(
                vehicle_id
            )

            total_co2 += traci.vehicle.getCO2Emission(
                vehicle_id
            )

        # Mean speed across lanes
        if lane_count > 0:

            mean_speed = total_speed / lane_count

        else:

            mean_speed = 0.0

        return {
            "queue": float(total_queue),
            "waiting": float(total_waiting),
            "speed": float(mean_speed),
            "density": float(total_density),
            "co2": float(total_co2),
        }

    # ======================================================
    # Extract complete network state
    # ======================================================

    def get_network_state(self):

        states = {}

        for tls_id in self.tls_ids:

            states[tls_id] = self.get_intersection_state(
                tls_id
            )

        return states

    # ======================================================
    # Convert to matrix for GNN
    # ======================================================

    def get_feature_matrix(self):

        states = self.get_network_state()

        matrix = []

        for tls_id in self.tls_ids:

            state = states[tls_id]

            matrix.append(
                [
                    state["queue"],
                    state["waiting"],
                    state["speed"],
                    state["density"],
                    state["co2"],
                ]
            )

        return np.array(
            matrix,
            dtype=np.float32,
        )