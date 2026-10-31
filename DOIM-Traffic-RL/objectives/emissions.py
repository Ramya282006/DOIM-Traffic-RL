"""
DOIM-Traffic-RL: Carbon Dioxide Emissions Metric Module
Integrates instantaneous vehicle emissions from SUMO's HBEFA3 pollutant model.
"""

from typing import List
import numpy as np


def compute_step_co2(vehicle_emissions_mg_per_sec: List[float], step_length_sec: float = 1.0) -> float:
    """
    Computes total integrated CO2 emission in milligrams for the current simulation step.
    
    Args:
        vehicle_emissions_mg_per_sec: List of instantaneous emission rates (mg/s) per active vehicle.
        step_length_sec: Simulation step duration in seconds.
        
    Returns:
        Integrated milligrams of CO2 emitted.
    """
    if not vehicle_emissions_mg_per_sec:
        return 0.0
    arr = np.asarray(vehicle_emissions_mg_per_sec, dtype=np.float64)
    total_rate = np.sum(np.clip(arr, 0.0, None))
    return float(total_rate * step_length_sec)
