"""
DOIM-Traffic-RL: Authoritative Typed Multi-Objective Configuration
Defines exact physical units, optimization directions, aggregation methods,
calibrated ranges, clipping semantics, and fairness definitions for all 5 traffic objectives.
Enforces fail-fast validation across the entire training and evaluation pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class ObjectiveDimensionSpec:
    """
    Authoritative specification for an individual objective dimension.
    """
    name: str
    unit: str
    direction: str  # "minimize" or "maximize"
    aggregation_method: str
    lower_bound: float
    upper_bound: float
    valid_range: Tuple[float, float]
    clipping_behavior: str = "clip_to_bounds"
    missing_value_behavior: str = "zero_fill"
    fairness_interpretation: Optional[str] = None

    def validate_value(self, val: float) -> None:
        """Fails fast if value is non-finite or outside valid physical range."""
        if not np.isfinite(val):
            raise ValueError(f"Objective '{self.name}' received non-finite value: {val}")
        if val < self.valid_range[0] or val > self.valid_range[1]:
            raise ValueError(
                f"Objective '{self.name}' received value {val} outside valid physical range {self.valid_range}"
            )


class AuthoritativeObjectiveConfig:
    """
    Authoritative container and validator for the 5 core traffic objectives:
      0. Waiting Time (vehicle-seconds, minimize)
      1. Queue Length (halting vehicles, minimize)
      2. CO2 Emissions (integrated milligrams, minimize)
      3. Throughput (completed vehicles, maximize)
      4. Fairness Disparity (1 - Jain Index in [0, 1], minimize)
    """

    NUM_OBJECTIVES: int = 5
    SCHEMA_VERSION: str = "2.0-IEEE-TITS"

    def __init__(
        self,
        specs: Optional[List[ObjectiveDimensionSpec]] = None,
        fairness_mode: str = "jain_disparity",
    ):
        self.fairness_mode = str(fairness_mode)
        if specs is not None:
            if len(specs) != self.NUM_OBJECTIVES:
                raise ValueError(f"Expected {self.NUM_OBJECTIVES} objective specs, got {len(specs)}")
            self.specs = tuple(specs)
        else:
            self.specs = self._default_specs()

        self.lower_bounds = np.array([s.lower_bound for s in self.specs], dtype=np.float32)
        self.upper_bounds = np.array([s.upper_bound for s in self.specs], dtype=np.float32)

        # Fail fast on invalid bounds
        if np.any(self.upper_bounds <= self.lower_bounds):
            raise ValueError("All upper bounds must be strictly greater than lower bounds.")

    def _default_specs(self) -> Tuple[ObjectiveDimensionSpec, ...]:
        return (
            ObjectiveDimensionSpec(
                name="waiting",
                unit="vehicle_seconds",
                direction="minimize",
                aggregation_method="sum_over_intersections",
                lower_bound=0.0,
                upper_bound=600.0,
                valid_range=(0.0, 3600.0 * 20),
            ),
            ObjectiveDimensionSpec(
                name="queue",
                unit="halting_vehicles",
                direction="minimize",
                aggregation_method="sum_over_intersections",
                lower_bound=0.0,
                upper_bound=20.0,
                valid_range=(0.0, 1000.0),
            ),
            ObjectiveDimensionSpec(
                name="co2",
                unit="milligrams",
                direction="minimize",
                aggregation_method="integrated_over_step_time",
                lower_bound=0.0,
                upper_bound=120000.0,
                valid_range=(0.0, 1e8),
            ),
            ObjectiveDimensionSpec(
                name="throughput",
                unit="completed_vehicles",
                direction="maximize",
                aggregation_method="step_arrival_count",
                lower_bound=0.0,
                upper_bound=20.0,
                valid_range=(0.0, 500.0),
            ),
            ObjectiveDimensionSpec(
                name="fairness",
                unit="dimensionless_disparity",
                direction="minimize",
                aggregation_method="approach_delay_disparity",
                lower_bound=0.0,
                upper_bound=1.0,
                valid_range=(0.0, 1.0),
                fairness_interpretation="1.0 - JainIndex(approach_waiting_times); 0=perfect equity, 1=starvation",
            ),
        )

    def validate_raw_vector(self, raw_vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Validates raw objective vector. Fails fast on invalid shapes, non-finite values,
        or physical bounds violations.
        """
        arr = np.asarray(raw_vector, dtype=np.float32)
        if arr.shape != (self.NUM_OBJECTIVES,):
            raise ValueError(f"Expected raw objective vector of shape ({self.NUM_OBJECTIVES},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Raw objective vector contains non-finite entries: {arr}")
        for i, spec in enumerate(self.specs):
            spec.validate_value(float(arr[i]))
        return arr

    def normalize(self, raw_vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Normalizes raw objective vector into urgency values in [0.0, 1.0]^5:
          - Minimization objectives: 0 = minimum urgency (good), 1 = maximum urgency (critical)
          - Maximization objectives (throughput): inverted such that 0 = high throughput (good), 1 = zero throughput (critical)
        """
        arr = self.validate_raw_vector(raw_vector)
        clipped = np.clip(arr, self.lower_bounds, self.upper_bounds)
        scaled = (clipped - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

        urgency = scaled.copy()
        for i, spec in enumerate(self.specs):
            if spec.direction == "maximize":
                urgency[i] = 1.0 - scaled[i]

        return urgency.astype(np.float32)

    @classmethod
    def from_config_dict(cls, config: Any) -> "AuthoritativeObjectiveConfig":
        """
        Creates instance from loaded experiment config dictionary, strictly verifying bounds.
        """
        obj_cfg = getattr(config, "objectives", config.get("objectives", {})) if hasattr(config, "objectives") or isinstance(config, dict) else {}
        fairness_mode = obj_cfg.get("fairness_metric", "jain_disparity") if isinstance(obj_cfg, dict) else getattr(obj_cfg, "fairness_metric", "jain_disparity")
        
        # Check normalization bounds if explicitly declared
        bounds = obj_cfg.get("normalization_bounds", {}) if isinstance(obj_cfg, dict) else getattr(obj_cfg, "normalization_bounds", {})
        if bounds:
            lower = bounds.get("lower", [0.0]*5) if isinstance(bounds, dict) else getattr(bounds, "lower", [0.0]*5)
            upper = bounds.get("upper", [600.0, 20.0, 120000.0, 20.0, 1.0]) if isinstance(bounds, dict) else getattr(bounds, "upper", [600.0, 20.0, 120000.0, 20.0, 1.0])
            specs = [
                ObjectiveDimensionSpec("waiting", "vehicle_seconds", "minimize", "sum_over_intersections", float(lower[0]), float(upper[0]), (0.0, 3600.0*20)),
                ObjectiveDimensionSpec("queue", "halting_vehicles", "minimize", "sum_over_intersections", float(lower[1]), float(upper[1]), (0.0, 1000.0)),
                ObjectiveDimensionSpec("co2", "milligrams", "minimize", "integrated_over_step_time", float(lower[2]), float(upper[2]), (0.0, 1e8)),
                ObjectiveDimensionSpec("throughput", "completed_vehicles", "maximize", "step_arrival_count", float(lower[3]), float(upper[3]), (0.0, 500.0)),
                ObjectiveDimensionSpec("fairness", "dimensionless_disparity", "minimize", "approach_delay_disparity", float(lower[4]), float(upper[4]), (0.0, 1.0), fairness_interpretation="1 - Jain(delays)"),
            ]
            return cls(specs=specs, fairness_mode=str(fairness_mode))
        return cls(fairness_mode=str(fairness_mode))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes objective metadata for manifest and provenance storage."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "fairness_mode": self.fairness_mode,
            "num_objectives": self.NUM_OBJECTIVES,
            "dimensions": [
                {
                    "name": s.name,
                    "unit": s.unit,
                    "direction": s.direction,
                    "aggregation_method": s.aggregation_method,
                    "lower_bound": s.lower_bound,
                    "upper_bound": s.upper_bound,
                    "valid_range": list(s.valid_range),
                    "fairness_interpretation": s.fairness_interpretation,
                }
                for s in self.specs
            ],
            "lower_bounds": [float(b) for b in self.lower_bounds],
            "upper_bounds": [float(b) for b in self.upper_bounds],
        }

