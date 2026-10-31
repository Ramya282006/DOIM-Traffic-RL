import traci
import shutil

# Find SUMO automatically from PATH
sumo_binary = shutil.which("sumo")

if sumo_binary is None:
    raise RuntimeError("SUMO was not found in PATH.")

print("SUMO found at:", sumo_binary)

# Start SUMO
traci.start([
    sumo_binary,
    "--version"
])

print("TraCI successfully connected to SUMO!")

traci.close()

print("Connection closed successfully.")