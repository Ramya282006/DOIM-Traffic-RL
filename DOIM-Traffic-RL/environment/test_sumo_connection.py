from environment.sumo_locator import find_sumo_binary
import traci


def test_connection():
    sumo_binary = find_sumo_binary()
    if sumo_binary is None:
        raise RuntimeError("SUMO was not found in PATH or environment.")

    print("SUMO found at:", sumo_binary)
    traci.start([sumo_binary, "--version"])
    print("TraCI successfully connected to SUMO!")
    traci.close()
    print("Connection closed successfully.")


if __name__ == "__main__":
    test_connection()