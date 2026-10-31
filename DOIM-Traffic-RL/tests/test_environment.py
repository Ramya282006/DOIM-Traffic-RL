from environment.sumo_env import DOIMTrafficEnv


CONFIG = (
    "scenarios/directional_shift/"
    "north_south_surge.sumocfg"
)


env = DOIMTrafficEnv(CONFIG)

observation, info = env.reset()

print()
print("Initial observation:", observation)
print("Initial info:", info)

for step in range(10):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"obs={observation}, "
        f"reward={reward:.2f}, "
        f"done={terminated}"
    )

    if terminated or truncated:
        break

env.close()

print()
print("Environment test completed successfully.")