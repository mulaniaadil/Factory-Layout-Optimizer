"""
Automated unit tests for spatial calibration and coordinate transformations.
"""

import pytest
from src.calibration import (
    CalibrationScale,
    pixel_box_to_metric,
    metric_to_pixel_box,
    reconstruct_layout_from_detections
)


def test_calibration_scale_factors():
    """Verify scale factor calculations between pixels and metres."""
    scale = CalibrationScale(
        image_width_px=800,
        image_height_px=600,
        factory_width_m=40.0,
        factory_height_m=30.0
    )
    # 40 / 800 = 0.05 metres per pixel
    assert scale.scale_x == 0.05
    # 30 / 600 = 0.05 metres per pixel
    assert scale.scale_y == 0.05
    assert scale.pixels_per_metre_x == 20.0
    assert scale.pixels_per_metre_y == 20.0


def test_round_trip_coordinate_conversion():
    """
    Verify that transforming metric to pixel and back to metric
    preserves original metric coordinates within numerical tolerance.
    """
    scale = CalibrationScale(
        image_width_px=1000,
        image_height_px=500,
        factory_width_m=50.0,
        factory_height_m=25.0
    )

    orig_x_m = 10.0
    orig_y_m = 5.0
    orig_w_m = 4.0
    orig_h_m = 3.0

    # Metric -> Pixel
    px_box = metric_to_pixel_box(orig_x_m, orig_y_m, orig_w_m, orig_h_m, scale)

    # Pixel -> Metric
    rec_x_m, rec_y_m, rec_w_m, rec_h_m = pixel_box_to_metric(px_box, scale)

    assert pytest.approx(rec_x_m, abs=0.1) == orig_x_m
    assert pytest.approx(rec_y_m, abs=0.1) == orig_y_m
    assert pytest.approx(rec_w_m, abs=0.1) == orig_w_m
    assert pytest.approx(rec_h_m, abs=0.1) == orig_h_m


def test_reconstruct_layout_from_detections():
    """Verify layout reconstruction from detection dictionaries."""
    scale = CalibrationScale(
        image_width_px=800,
        image_height_px=600,
        factory_width_m=32.0,
        factory_height_m=24.0
    )

    detections = [
        {"class_id": 1, "class_name": "Cutting machine", "confidence": 0.95, "bbox_px": [100, 100, 200, 180]},
        {"class_id": 2, "class_name": "Drilling machine", "confidence": 0.88, "bbox_px": [400, 300, 500, 380]},
    ]

    layout = reconstruct_layout_from_detections(detections, scale, use_standard_dimensions=True)
    assert len(layout.machines) == 2
    assert layout.machines[0].category == "Cutting machine"
    assert layout.machines[1].category == "Drilling machine"
    assert layout.machines[0].width > 0
    assert layout.machines[0].height > 0
