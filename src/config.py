"""
Configuration manager and system-wide constants for Factory Layout Optimizer.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Canonical equipment classes
# Note: In TorchVision Faster R-CNN, 0 is reserved for background.
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "__background__",
    1: "Cutting machine",
    2: "Drilling machine",
    3: "Assembly workstation",
    4: "Inspection station",
    5: "Packaging station",
    6: "Storage area"
}

CLASS_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items() if k > 0}

CLASS_COLORS: Dict[str, str] = {
    "Cutting machine": "#E74C3C",     # Crimson Red
    "Drilling machine": "#E67E22",    # Industrial Amber
    "Assembly workstation": "#2ECC71",# Emerald Green
    "Inspection station": "#9B59B6",  # Amethyst Purple
    "Packaging station": "#F39C12",   # Orange
    "Storage area": "#3498DB",        # Steel Blue
    "Obstacle": "#7F8C8D",            # Slate Gray
    "Receiving Dock": "#4A90E2",      # Ocean Blue
    "Dispatch Dock": "#50E3C2"        # Mint Cyan
}

DEFAULT_DIMENSIONS: Dict[str, tuple[float, float]] = {
    "Cutting machine": (3.0, 2.0),
    "Drilling machine": (2.5, 2.0),
    "Assembly workstation": (3.5, 2.5),
    "Inspection station": (2.0, 2.0),
    "Packaging station": (3.0, 2.5),
    "Storage area": (4.0, 3.0),
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON with fallback defaults."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Sensible fallback dictionary if config file is not found
    return {
        "factory": {
            "width": 30.0,
            "height": 20.0,
            "grid_resolution": 0.5,
            "min_aisle_width": 1.5,
            "default_clearance": 0.5,
            "receiving_dock": {"name": "Receiving Dock", "x": 0.0, "y": 8.0, "width": 2.5, "height": 4.0, "color": "#4A90E2"},
            "dispatch_dock": {"name": "Dispatch Dock", "x": 27.5, "y": 8.0, "width": 2.5, "height": 4.0, "color": "#50E3C2"}
        },
        "equipment_classes": {str(k): {"id": k, "name": v} for k, v in CLASS_ID_TO_NAME.items() if k > 0},
        "obstacles": [],
        "sample_machines": [],
        "sample_flows": [],
        "optimizer": {
            "default_algorithm": "simulated_annealing",
            "cooling_rate": 0.985,
            "initial_temperature": 500.0,
            "min_temperature": 0.1,
            "max_iterations": 1500,
            "step_size": 1.0,
            "random_seed": 42
        }
    }
