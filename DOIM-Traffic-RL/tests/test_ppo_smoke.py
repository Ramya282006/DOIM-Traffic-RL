"""
DOIM-Traffic-RL: Optional Stable-Baselines3 PPO Integration Smoke Test.
Safely skips if stable_baselines3 or SUMO runtime is unavailable.
"""

from pathlib import Path
import pytest


def test_ppo_smoke():
    sb3 = pytest.importorskip("stable_baselines3")
    PPO = sb3.PPO

    config = "scenarios/directional_shift/north_south_surge.sumocfg"
    if not Path(config).exists():
        pytest.skip(f"Config not found: {config}")

    try:
        from environment.sumo_env import DOIMTrafficEnv
        env = DOIMTrafficEnv(config, max_steps=10)
    except Exception as e:
        pytest.skip(f"SUMO runtime unavailable: {e}")

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            n_steps=8,
            batch_size=8,
            seed=42,
        )
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_ppo_smoke()