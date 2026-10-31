"""
DOIM-Traffic-RL: Robust Eclipse SUMO Binary Locator
Discovers the SUMO binary across system PATH, SUMO_HOME, python eclipse-sumo package,
and active virtual environment directories.
"""

import os
from pathlib import Path
import shutil
import sys
from typing import Optional


def find_sumo_binary(prefer_gui: bool = False) -> str:
    """
    Locates the executable path for 'sumo' or 'sumo-gui'.
    
    Args:
        prefer_gui: If True, searches for sumo-gui, falling back to sumo.
        
    Returns:
        Absolute string path to the SUMO executable.
        
    Raises:
        RuntimeError: If no valid SUMO executable is found.
    """
    target_names = ["sumo-gui", "sumo"] if prefer_gui else ["sumo"]
    
    # 1. System PATH
    for name in target_names:
        binary = shutil.which(name)
        if binary:
            return binary
            
    # 2. Check python 'sumo' package (eclipse-sumo)
    try:
        import sumo
        if hasattr(sumo, "SUMO_HOME"):
            os.environ.setdefault("SUMO_HOME", str(sumo.SUMO_HOME))
        for name in target_names:
            if hasattr(sumo, f"{name.replace('-', '_')}_binary"):
                bin_path = getattr(sumo, f"{name.replace('-', '_')}_binary")
                if bin_path and Path(bin_path).exists():
                    return str(bin_path)
            candidate = Path(sumo.SUMO_HOME) / "bin" / (name + (".exe" if sys.platform == "win32" else ""))
            if candidate.exists():
                return str(candidate)
    except ImportError:
        pass

    # 3. Check SUMO_HOME environment variable
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for name in target_names:
            exe_name = name + (".exe" if sys.platform == "win32" else "")
            candidate = Path(sumo_home) / "bin" / exe_name
            if candidate.exists():
                return str(candidate)

    # 4. Check active Python virtual environment (sys.prefix)
    prefix = Path(sys.prefix)
    search_dirs = [
        prefix / "Scripts",
        prefix / "bin",
        prefix / "Lib" / "site-packages" / "sumo" / "bin",
        prefix / "Lib" / "site-packages" / "sumo",
    ]
    for d in search_dirs:
        for name in target_names:
            exe_name = name + (".exe" if sys.platform == "win32" else "")
            candidate = d / exe_name
            if candidate.exists():
                return str(candidate)

    # 5. Standard Windows installation locations
    if sys.platform == "win32":
        standard_windows_paths = [
            Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin"),
            Path(r"C:\Program Files\Eclipse\Sumo\bin"),
            Path(r"C:\Sumo\bin"),
        ]
        for p in standard_windows_paths:
            for name in target_names:
                candidate = p / (name + ".exe")
                if candidate.exists():
                    return str(candidate)

    raise RuntimeError(
        "SUMO executable was not found. Please ensure 'eclipse-sumo' is installed via pip "
        "or that SUMO is in PATH or SUMO_HOME is set."
    )
