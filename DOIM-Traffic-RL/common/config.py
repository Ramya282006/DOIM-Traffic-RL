"""
DOIM-Traffic-RL: Centralized Configuration Management
Handles loading, merging, validation, and snapshotting of experiment configurations.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigDict(dict):
    """
    Dictionary supporting both attribute-style and item-style access.
    """
    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[name] = val
            return val
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)


def to_plain_dict(obj: Any) -> Any:
    """Recursively converts ConfigDict and nested dictionaries to plain python dicts."""
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj


# Register ConfigDict with yaml SafeDumper
yaml.SafeDumper.add_representer(
    ConfigDict,
    lambda dumper, data: dumper.represent_dict(to_plain_dict(data)),
)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges override dictionary into base dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Computes a canonical SHA-256 fingerprint for a configuration dictionary.
    """
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def load_experiment_config(
    default_config_path: Union[str, Path] = "configs/default.yaml",
    scenario_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigDict:
    """
    Loads and merges the default configuration with scenario and explicit overrides.
    
    Args:
        default_config_path: Path to baseline default.yaml
        scenario_config_path: Optional path to scenario-specific YAML
        overrides: Optional runtime dictionary overrides
        
    Returns:
        ConfigDict wrapping the final merged configuration.
    """
    config = load_yaml(default_config_path)
    
    if scenario_config_path:
        scen_str = str(scenario_config_path)
        if scen_str.endswith(".yaml") or scen_str.endswith(".yml"):
            scenario_cfg = load_yaml(scenario_config_path)
            if isinstance(scenario_cfg, dict):
                config = deep_merge(config, scenario_cfg)
        elif scen_str.endswith(".sumocfg"):
            if "simulation" not in config:
                config["simulation"] = {}
            config["simulation"]["sumo_config"] = scen_str
            config["simulation"]["scenario_name"] = Path(scen_str).stem
        
    if overrides:
        config = deep_merge(config, overrides)
        
    # Inject canonical SHA fingerprint
    config["config_hash"] = compute_config_hash(config)
    
    return ConfigDict(config)


def snapshot_config(config: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
    """
    Saves an immutable snapshot of the active configuration in the experiment folder.
    """
    out_path = Path(output_dir) / "config_snapshot.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain_dict(config), f, default_flow_style=False, sort_keys=False)
    return out_path
