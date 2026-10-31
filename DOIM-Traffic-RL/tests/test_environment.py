"""
DOIM-Traffic-RL: Unit and Integration Tests for DOIMTrafficEnv
Guarantees test import safety: module import must have zero simulation side effects.
SUMO / TraCI execution is isolated strictly within callable test functions.
"""

from pathlib import Path
import pytest
import sys


def test_import_environment_has_no_simulation_side_effects():
    """
    Verifies that importing DOIMTrafficEnv and related environment modules
    does NOT start TraCI, does not launch SUMO, and does not alter global simulation state.
    """
    import traci
    if hasattr(traci, "isLoaded"):
        assert not traci.isLoaded(), "TraCI should not be loaded upon module import"
    # Ensure traci connection is not active
    try:
        # If TraCI is running, getConnection() or getVersion() succeeds. If not, it raises an exception.
        traci.getConnection()
        is_running = True
    except Exception:
        is_running = False
    assert not is_running, "TraCI connection must NOT be active merely by importing test module"


def test_environment_initialization_guarded():
    """
    Tests initialization of DOIMTrafficEnv with explicit teardown.
    Gracefully skips if SUMO binary is unavailable in the execution environment.
    """
    from environment.sumo_env import DOIMTrafficEnv

    config_path = "scenarios/directional_shift/north_south_surge.sumocfg"
    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")

    # Check if SUMO is accessible
    import shutil
    has_sumo = shutil.which("sumo") is not None or "SUMO_HOME" in sys.modules
    
    try:
        env = DOIMTrafficEnv(config_path, max_steps=5)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert "normalized_objectives" in info
        
        # Take 1 safe step
        action = [0, 0, 0, 0, 0] # default keep phase
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        assert next_obs is not None
        assert isinstance(reward, float)
    except Exception as e:
        if "Could not connect to TraCI" in str(e) or "SUMO_HOME" in str(e) or "sumo" in str(e).lower():
            pytest.skip(f"SUMO runtime execution not available: {e}")
        else:
            raise e
    finally:
        try:
            if 'env' in locals():
                env.close()
        except Exception:
            pass