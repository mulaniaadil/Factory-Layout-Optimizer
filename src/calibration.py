"""
Spatial reconstruction and coordinate transformation engine.
Converts between top-left pixel coordinates (computer vision) and
bottom-left metric coordinates (industrial layout engineering).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
from src.layout import FactoryLayout, Machine
from src.config import DEFAULT_DIMENSIONS, CLASS_NAME_TO_ID, CLASS_ID_TO_NAME


@dataclass
class CalibrationScale:
    """Mapping factors between pixel space and real-world metric space."""
    image_width_px: int
    image_height_px: int
    factory_width_m: float
    factory_height_m: float

    @property
    def scale_x(self) -> float:
        """Metres per pixel along X-axis."""
        return self.factory_width_m / max(1, self.image_width_px)

    @property
    def scale_y(self) -> float:
        """Metres per pixel along Y-axis."""
        return self.factory_height_m / max(1, self.image_height_px)

    @property
    def pixels_per_metre_x(self) -> float:
        return self.image_width_px / max(1e-3, self.factory_width_m)

    @property
    def pixels_per_metre_y(self) -> float:
        return self.image_height_px / max(1e-3, self.factory_height_m)


def pixel_box_to_metric(
    bbox_px: Tuple[float, float, float, float],
    scale: CalibrationScale,
    use_default_footprint: bool = False,
    category: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Transforms [xmin_px, ymin_px, xmax_px, ymax_px] (top-left origin)
    to [x_m, y_m, width_m, height_m] (bottom-left origin).
    """
    xmin_px, ymin_px, xmax_px, ymax_px = bbox_px

    raw_width_m = (xmax_px - xmin_px) * scale.scale_x
    raw_height_m = (ymax_px - ymin_px) * scale.scale_y

    # Inverted Y-axis: ymin in pixels is highest point from bottom
    raw_x_m = xmin_px * scale.scale_x
    raw_y_m = (scale.image_height_px - ymax_px) * scale.scale_y

    if use_default_footprint and category in DEFAULT_DIMENSIONS:
        dw, dh = DEFAULT_DIMENSIONS[category]
        cx = raw_x_m + raw_width_m / 2.0
        cy = raw_y_m + raw_height_m / 2.0
        return (max(0.0, cx - dw / 2.0), max(0.0, cy - dh / 2.0), dw, dh)

    return (max(0.0, raw_x_m), max(0.0, raw_y_m), raw_width_m, raw_height_m)


def metric_to_pixel_box(
    x_m: float,
    y_m: float,
    width_m: float,
    height_m: float,
    scale: CalibrationScale
) -> Tuple[int, int, int, int]:
    """
    Transforms metric [x_m, y_m, width_m, height_m] (bottom-left origin)
    to [xmin_px, ymin_px, xmax_px, ymax_px] (top-left origin).
    """
    xmin_px = int(round(x_m * scale.pixels_per_metre_x))
    xmax_px = int(round((x_m + width_m) * scale.pixels_per_metre_x))

    # Top-left Y in pixels corresponds to top of machine (y_m + height_m)
    ymin_px = int(round((scale.factory_height_m - (y_m + height_m)) * scale.pixels_per_metre_y))
    ymax_px = int(round((scale.factory_height_m - y_m) * scale.pixels_per_metre_y))

    return (xmin_px, ymin_px, xmax_px, ymax_px)


def reconstruct_layout_from_detections(
    detections: List[Dict[str, Any]],
    scale: CalibrationScale,
    use_standard_dimensions: bool = True,
    base_layout: Optional[FactoryLayout] = None
) -> FactoryLayout:
    """
    Builds a populated FactoryLayout from computer-vision detection outputs.
    Preserves obstacles and docks from base_layout if provided.
    """
    if base_layout is not None:
        layout = base_layout.clone()
        layout.machines = []
    else:
        layout = FactoryLayout(
            factory_width=scale.factory_width_m,
            factory_height=scale.factory_height_m
        )

    for idx, det in enumerate(detections, start=1):
        cat = det.get("class_name", "Cutting machine")
        class_id = det.get("class_id", CLASS_NAME_TO_ID.get(cat, 1))
        bbox_px = det["bbox_px"]  # [xmin, ymin, xmax, ymax]

        x_m, y_m, w_m, h_m = pixel_box_to_metric(
            bbox_px,
            scale,
            use_default_footprint=use_standard_dimensions,
            category=cat
        )

        m_id = f"M{idx}"
        m_name = f"{cat} {idx:02d}"

        layout.machines.append(
            Machine(
                id=m_id,
                name=m_name,
                category=cat,
                class_id=class_id,
                x=round(x_m, 2),
                y=round(y_m, 2),
                width=round(w_m, 2),
                height=round(h_m, 2),
                is_fixed=False,
                clearance=0.5
            )
        )

    return layout
