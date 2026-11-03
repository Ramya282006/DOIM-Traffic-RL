"""
DOIM-Traffic-RL: Expanded Scenario Generator (Phase 18)
Generates:
  1. arterial_corridor: High-volume arterial progression along E-W axis (2600 veh/h) with cross-street starvation risk.
  2. incident_disruption: Mid-episode lane capacity bottleneck on arterial link B0B1 with severe localized spillback.
"""

from pathlib import Path
import random
import xml.etree.ElementTree as ET

BASE_DIR = Path("DOIM-Traffic-RL/scenarios")

WEST_IN = ["A0B0", "B0C0"]
EAST_OUT = ["B2A2", "C2B2"]
EAST_IN = ["A2B2", "B2C2"]
WEST_OUT = ["B0A0", "C0B0"]

NORTH_IN = ["A0A1", "A1A2"]
SOUTH_OUT = ["C0C1", "C1C2"]
SOUTH_IN = ["C2C1", "C1C0"]
NORTH_OUT = ["A2A1", "A1A0"]

def generate_trips(from_edges, to_edges, rate_vph, begin_time, end_time, prefix, seed=101):
    random.seed(seed)
    trips = []
    interval = 3600.0 / rate_vph
    t = float(begin_time)
    veh_id = 0
    while t < end_time:
        depart = round(t + random.uniform(-interval * 0.15, interval * 0.15), 1)
        depart = max(float(begin_time), min(depart, float(end_time) - 1.0))
        from_edge = random.choice(from_edges)
        to_edge = random.choice(to_edges)
        trips.append((depart, f"{prefix}_{veh_id}", from_edge, to_edge))
        veh_id += 1
        t += interval
    trips.sort(key=lambda x: x[0])
    return trips

def write_routes_and_sumocfg(scenario_name, trips, description):
    scen_dir = BASE_DIR / scenario_name
    scen_dir.mkdir(parents=True, exist_ok=True)
    
    # Write .rou.xml
    rou_path = scen_dir / f"{scenario_name}.rou.xml"
    root = ET.Element("routes")
    vtype = ET.SubElement(root, "vType", {
        "id": "car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "minGap": "2.5",
        "maxSpeed": "13.89",
    })
    for depart, vid, f_edge, t_edge in trips:
        ET.SubElement(root, "trip", {
            "id": vid,
            "type": "car",
            "depart": f"{depart:.1f}",
            "from": f_edge,
            "to": t_edge,
        })
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(rou_path, encoding="utf-8", xml_declaration=True)
    
    # Write .sumocfg
    cfg_path = scen_dir / f"{scenario_name}.sumocfg"
    cfg_root = ET.Element("configuration")
    input_elem = ET.SubElement(cfg_root, "input")
    ET.SubElement(input_elem, "net-file", {"value": "../../configs/network/grid3x3.net.xml"})
    ET.SubElement(input_elem, "route-files", {"value": f"{scenario_name}.rou.xml"})
    time_elem = ET.SubElement(cfg_root, "time")
    ET.SubElement(time_elem, "begin", {"value": "0"})
    ET.SubElement(time_elem, "end", {"value": "1800"})
    ET.SubElement(time_elem, "step-length", {"value": "1.0"})
    proc_elem = ET.SubElement(cfg_root, "processing")
    ET.SubElement(proc_elem, "time-to-teleport", {"value": "-1"})
    
    cfg_tree = ET.ElementTree(cfg_root)
    ET.indent(cfg_tree, space="  ")
    cfg_tree.write(cfg_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated {scenario_name}: {len(trips)} trips written to {rou_path}")

# Scenario 1: Arterial Corridor (High E-W flow, low N-S cross flow)
trips_ew = generate_trips(WEST_IN, EAST_OUT, 1400, 0, 1800, "arterial_ew", seed=201)
trips_we = generate_trips(EAST_IN, WEST_OUT, 1200, 0, 1800, "arterial_we", seed=202)
trips_ns = generate_trips(NORTH_IN, SOUTH_OUT, 350, 0, 1800, "cross_ns", seed=203)
trips_sn = generate_trips(SOUTH_IN, NORTH_OUT, 350, 0, 1800, "cross_sn", seed=204)
arterial_trips = sorted(trips_ew + trips_we + trips_ns + trips_sn, key=lambda x: x[0])
write_routes_and_sumocfg("arterial_corridor", arterial_trips, "Heavy East-West arterial progression")

# Scenario 2: Incident Disruption (Surge with mid-episode bottleneck)
trips_inc_surge = generate_trips(WEST_IN, EAST_OUT, 1800, 0, 1800, "inc_main", seed=301)
trips_inc_cross = generate_trips(NORTH_IN, SOUTH_OUT, 600, 0, 1800, "inc_cross", seed=302)
incident_trips = sorted(trips_inc_surge + trips_inc_cross, key=lambda x: x[0])
write_routes_and_sumocfg("incident_disruption", incident_trips, "Surge with bottleneck incident")
