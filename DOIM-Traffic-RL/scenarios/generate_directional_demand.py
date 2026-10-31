import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================================
# DOIM Traffic RL
# Controlled Directional Demand Generator
# ============================================================

SEED = 100
BEGIN = 0
END = 1800

random.seed(SEED)

OUTPUT_DIR = Path("scenarios/directional_shift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Boundary edges of our 3x3 network
# ------------------------------------------------------------

NORTH_IN = ["A0A1", "A1A2"]
NORTH_OUT = ["A2A1", "A1A0"]

SOUTH_IN = ["C2C1", "C1C0"]
SOUTH_OUT = ["C0C1", "C1C2"]

WEST_IN = ["A0B0", "B0C0"]
WEST_OUT = ["B0A0", "C0B0"]

EAST_IN = ["A2B2", "B2C2"]
EAST_OUT = ["B2A2", "C2B2"]


def create_vehicles(flow_id, from_edges, to_edges, rate):
    """
    Generate individual vehicle trips.

    rate = vehicles/hour
    """

    interval = 3600.0 / rate
    departure = BEGIN
    vehicle_index = 0

    vehicles = []

    while departure < END:

        from_edge = random.choice(from_edges)
        to_edge = random.choice(to_edges)

        if from_edge != to_edge:

            vehicles.append(
                {
                    "id": f"{flow_id}_{vehicle_index}",
                    "depart": departure,
                    "from": from_edge,
                    "to": to_edge,
                }
            )

            vehicle_index += 1

        departure += interval

    return vehicles


def create_scenario(name, ns_rate, ew_rate):

    vehicles = []

    # North → South
    vehicles.extend(
        create_vehicles(
            "N2S",
            NORTH_IN,
            SOUTH_OUT,
            ns_rate,
        )
    )

    # South → North
    vehicles.extend(
        create_vehicles(
            "S2N",
            SOUTH_IN,
            NORTH_OUT,
            ns_rate,
        )
    )

    # East → West
    vehicles.extend(
        create_vehicles(
            "E2W",
            EAST_IN,
            WEST_OUT,
            ew_rate,
        )
    )

    # West → East
    vehicles.extend(
        create_vehicles(
            "W2E",
            WEST_IN,
            EAST_OUT,
            ew_rate,
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Sort ALL vehicles globally by departure time.
    # --------------------------------------------------------

    vehicles.sort(key=lambda v: v["depart"])

    root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation":
                "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    for vehicle in vehicles:

        ET.SubElement(
            root,
            "trip",
            {
                "id": vehicle["id"],
                "depart": f"{vehicle['depart']:.2f}",
                "from": vehicle["from"],
                "to": vehicle["to"],
                "departLane": "best",
                "departSpeed": "max",
            },
        )

    tree = ET.ElementTree(root)

    ET.indent(tree, space="    ")

    output = OUTPUT_DIR / f"{name}.trips.xml"

    tree.write(
        output,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print()
    print("=" * 55)
    print("DOIM Directional Demand Generated")
    print("=" * 55)
    print(f"Scenario : {name}")
    print(f"N/S rate : {ns_rate} veh/hour")
    print(f"E/W rate : {ew_rate} veh/hour")
    print(f"Duration : {END - BEGIN} seconds")
    print(f"Vehicles : {len(vehicles)}")
    print(f"Seed     : {SEED}")
    print(f"Output   : {output}")
    print("=" * 55)


# ============================================================
# North-South Surge
# ============================================================

create_scenario(
    name="north_south_surge",
    ns_rate=600,
    ew_rate=100,
)