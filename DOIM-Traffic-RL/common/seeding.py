"""
DOIM-Traffic-RL: Reproducibility & Seeding Infrastructure
Provides centralized, deterministic pseudo-random number generator initialization
across Python, NumPy, PyTorch, Gymnasium, and Eclipse SUMO.
"""

import os
import random
from typing import Dict, List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def seed_everything(seed: int) -> Dict[str, int]:
    """
    Sets random seeds across all supported execution frameworks.
    
    Args:
        seed: Base integer seed.
        
    Returns:
        Dictionary recording the seed applied to each framework.
    """
    seed = int(seed)
    
    # 1. Python standard library
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch (if installed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    return {
        "base_seed": seed,
        "python_hash_seed": seed,
        "numpy_seed": seed,
        "torch_seeded": TORCH_AVAILABLE,
    }


def generate_independent_seeds(master_seed: int, count: int) -> List[int]:
    """
    Derives a deterministic list of independent seeds from a master seed
    using NumPy's SeedSequence to prevent correlations between seeds.
    
    Args:
        master_seed: Primary experiment seed.
        count: Number of independent child seeds to generate.
        
    Returns:
        List of distinct integer seeds.
    """
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.generate_state(count)
    return [int(s % (2**31 - 1)) for s in child_seeds]


def format_sumo_seed(seed: int) -> str:
    """
    Formats an integer seed safely for SUMO's --seed command-line parameter.
    SUMO expects a positive 32-bit signed integer.
    """
    return str(abs(int(seed)) % (2**31 - 1))
