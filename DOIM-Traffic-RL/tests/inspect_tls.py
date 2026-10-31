import traci
import shutil


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


sumo = shutil.which("sumo")

if sumo is None:
    raise RuntimeError("SUMO was not found in PATH.")


traci.start([
    sumo,
    "-c",
    CONFIG,
    "--start",
])


print("\n" + "=" * 70)
print("TRAFFIC LIGHT CONTROLLERS")
print("=" * 70)

tls_ids = traci.trafficlight.getIDList()

print(f"\nNumber of traffic lights: {len(tls_ids)}\n")


for tls_id in tls_ids:

    print("-" * 70)
    print(f"Traffic Light ID: {tls_id}")

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    print(f"Controlled links: {len(controlled_links)}")

    for index, link in enumerate(controlled_links):

        # SUMO may return None for an uncontrolled connection
        if link is None:
            print(f"  Link {index}: NONE")
            continue

        # Print the raw structure first.
        # This avoids making assumptions about tuple length.
        print(f"  Link {index}: {link}")


print("\n" + "=" * 70)
print("TRAFFIC LIGHT PHASE INFORMATION")
print("=" * 70)

for tls_id in tls_ids:

    print("-" * 70)
    print(f"Traffic Light: {tls_id}")

    program = traci.trafficlight.getProgram(tls_id)
    phase = traci.trafficlight.getPhase(tls_id)
    phase_duration = traci.trafficlight.getPhaseDuration(tls_id)

    print(f"  Program : {program}")
    print(f"  Phase   : {phase}")
    print(f"  Duration: {phase_duration:.1f} s")


traci.close()

print("\nInspection completed successfully.")