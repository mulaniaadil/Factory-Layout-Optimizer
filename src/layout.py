"""
Domain models for factory floor, machines, obstacles, and material flows.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import copy
import pandas as pd
from src.config import CLASS_NAME_TO_ID, CLASS_ID_TO_NAME, CLASS_COLORS, load_config


@dataclass
class Machine:
    """Represents a factory equipment unit with 2D footprint and coordinates."""
    id: str
    name: str
    category: str
    x: float  # Bottom-left X coordinate in metres
    y: float  # Bottom-left Y coordinate in metres
    width: float  # Machine footprint width in metres
    height: float  # Machine footprint height in metres
    class_id: int = 1
    is_fixed: bool = False
    clearance: float = 0.5  # Required safety clearance margin around perimeter (metres)

    def __post_init__(self):
        if self.class_id not in CLASS_ID_TO_NAME and self.category in CLASS_NAME_TO_ID:
            self.class_id = CLASS_NAME_TO_ID[self.category]

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the geometric center (cx, cy) of the machine."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (xmin, ymin, xmax, ymax) of the physical machine."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def bounds_with_clearance(self) -> Tuple[float, float, float, float]:
        """Returns (xmin, ymin, xmax, ymax) expanded by safety clearance."""
        return (
            self.x - self.clearance,
            self.y - self.clearance,
            self.x + self.width + self.clearance,
            self.y + self.height + self.clearance,
        )

    @property
    def area(self) -> float:
        """Physical footprint area in square metres."""
        return self.width * self.height

    @property
    def color(self) -> str:
        """Visual color hex code for rendering."""
        return CLASS_COLORS.get(self.category, "#7F8C8D")


@dataclass
class Obstacle:
    """Structural pillar, stairwell, electrical room, or prohibited zone."""
    id: str
    name: str
    x: float
    y: float
    width: float
    height: float
    color: str = "#7F8C8D"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Dock:
    """Receiving or dispatch loading bay dock."""
    name: str
    x: float
    y: float
    width: float
    height: float
    color: str = "#4A90E2"

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass
class MaterialFlow:
    """Directed material transfer frequency between two entities per shift."""
    from_id: str
    to_id: str
    frequency: float  # Number of pallet/part moves per shift


@dataclass
class FactoryLayout:
    """Complete representation of factory space, equipment, and material flows."""
    factory_width: float = 30.0
    factory_height: float = 20.0
    grid_resolution: float = 0.5  # Discretization grid step in metres
    min_aisle_width: float = 1.5  # Metres
    machines: List[Machine] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    docks: Dict[str, Dock] = field(default_factory=dict)
    flows: List[MaterialFlow] = field(default_factory=list)

    def clone(self) -> FactoryLayout:
        """Create a deep copy of the layout."""
        return copy.deepcopy(self)

    def get_entity_center(self, entity_id: str) -> Optional[Tuple[float, float]]:
        """Resolve center coordinates for machine, dock, or obstacle."""
        # Check machines
        for m in self.machines:
            if m.id == entity_id or m.name == entity_id:
                return m.center
        # Check docks
        for name, dock in self.docks.items():
            if name == entity_id or dock.name == entity_id:
                return dock.center
        # Check obstacles
        for obs in self.obstacles:
            if obs.id == entity_id or obs.name == entity_id:
                return obs.center
        return None

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        for m in self.machines:
            if m.id == machine_id:
                return m
        return None

    @property
    def total_floor_area(self) -> float:
        return self.factory_width * self.factory_height

    @property
    def occupied_equipment_area(self) -> float:
        return sum(m.area for m in self.machines)

    @property
    def occupied_obstacle_area(self) -> float:
        return sum(obs.area for obs in self.obstacles)

    @property
    def space_utilization_percent(self) -> float:
        if self.total_floor_area <= 0:
            return 0.0
        return (self.occupied_equipment_area / self.total_floor_area) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert layout to JSON-serializable dictionary."""
        return {
            "factory_width": self.factory_width,
            "factory_height": self.factory_height,
            "grid_resolution": self.grid_resolution,
            "min_aisle_width": self.min_aisle_width,
            "machines": [asdict(m) for m in self.machines],
            "obstacles": [asdict(obs) for obs in self.obstacles],
            "docks": {k: asdict(v) for k, v in self.docks.items()},
            "flows": [{"from": f.from_id, "to": f.to_id, "frequency": f.frequency} for f in self.flows],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FactoryLayout:
        """Instantiate layout from dictionary."""
        layout = cls(
            factory_width=float(data.get("factory_width", 30.0)),
            factory_height=float(data.get("factory_height", 20.0)),
            grid_resolution=float(data.get("grid_resolution", 0.5)),
            min_aisle_width=float(data.get("min_aisle_width", 1.5)),
        )
        for m_data in data.get("machines", []):
            layout.machines.append(
                Machine(
                    id=str(m_data["id"]),
                    name=str(m_data["name"]),
                    category=str(m_data["category"]),
                    x=float(m_data["x"]),
                    y=float(m_data["y"]),
                    width=float(m_data["width"]),
                    height=float(m_data["height"]),
                    class_id=int(m_data.get("class_id", CLASS_NAME_TO_ID.get(m_data["category"], 1))),
                    is_fixed=bool(m_data.get("is_fixed", False)),
                    clearance=float(m_data.get("clearance", 0.5)),
                )
            )
        for obs_data in data.get("obstacles", []):
            layout.obstacles.append(
                Obstacle(
                    id=str(obs_data["id"]),
                    name=str(obs_data["name"]),
                    x=float(obs_data["x"]),
                    y=float(obs_data["y"]),
                    width=float(obs_data["width"]),
                    height=float(obs_data["height"]),
                    color=str(obs_data.get("color", "#7F8C8D")),
                )
            )
        for dock_key, dock_data in data.get("docks", {}).items():
            layout.docks[dock_key] = Dock(
                name=str(dock_data.get("name", dock_key)),
                x=float(dock_data["x"]),
                y=float(dock_data["y"]),
                width=float(dock_data["width"]),
                height=float(dock_data["height"]),
                color=str(dock_data.get("color", "#4A90E2")),
            )
        for flow_data in data.get("flows", []):
            layout.flows.append(
                MaterialFlow(
                    from_id=str(flow_data.get("from", flow_data.get("from_id", ""))),
                    to_id=str(flow_data.get("to", flow_data.get("to_id", ""))),
                    frequency=float(flow_data.get("frequency", 0.0)),
                )
            )
        return layout

    def to_dataframe(self) -> pd.DataFrame:
        """Export machines table as pandas DataFrame for UI and analysis."""
        records = []
        for m in self.machines:
            records.append({
                "ID": m.id,
                "Name": m.name,
                "Category": m.category,
                "X (m)": round(m.x, 2),
                "Y (m)": round(m.y, 2),
                "Width (m)": round(m.width, 2),
                "Height (m)": round(m.height, 2),
                "Fixed": m.is_fixed,
                "Clearance (m)": round(m.clearance, 2)
            })
        return pd.DataFrame(records)

    @classmethod
    def load_default(cls) -> FactoryLayout:
        """Create the benchmark 30m x 20m synthetic factory layout from default config."""
        cfg = load_config()
        factory_info = cfg.get("factory", {})
        layout = cls(
            factory_width=factory_info.get("width", 30.0),
            factory_height=factory_info.get("height", 20.0),
            grid_resolution=factory_info.get("grid_resolution", 0.5),
            min_aisle_width=factory_info.get("min_aisle_width", 1.5),
        )
        # Load docks
        if "receiving_dock" in factory_info:
            rd = factory_info["receiving_dock"]
            layout.docks["Receiving Dock"] = Dock(
                name=rd["name"], x=rd["x"], y=rd["y"], width=rd["width"], height=rd["height"], color=rd["color"]
            )
        if "dispatch_dock" in factory_info:
            dd = factory_info["dispatch_dock"]
            layout.docks["Dispatch Dock"] = Dock(
                name=dd["name"], x=dd["x"], y=dd["y"], width=dd["width"], height=dd["height"], color=dd["color"]
            )
        # Load obstacles
        for obs in cfg.get("obstacles", []):
            layout.obstacles.append(
                Obstacle(id=obs["id"], name=obs["name"], x=obs["x"], y=obs["y"], width=obs["width"], height=obs["height"], color=obs.get("color", "#7F8C8D"))
            )
        # Load machines
        for m in cfg.get("sample_machines", []):
            layout.machines.append(
                Machine(
                    id=m["id"],
                    name=m["name"],
                    category=m["category"],
                    class_id=m["class_id"],
                    x=m["x"],
                    y=m["y"],
                    width=m["width"],
                    height=m["height"],
                    is_fixed=m.get("is_fixed", False),
                    clearance=m.get("clearance", 0.5)
                )
            )
        # Load flows
        for flow in cfg.get("sample_flows", []):
            layout.flows.append(
                MaterialFlow(from_id=flow["from"], to_id=flow["to"], frequency=flow["frequency"])
            )
        return layout
