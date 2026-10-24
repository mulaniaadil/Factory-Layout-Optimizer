"""
Automated unit tests for synthetic factory floor generation and annotation validation.
"""

import pytest
import tempfile
from pathlib import Path
from src.data_generation import SyntheticFactoryGenerator, EQUIPMENT_STYLE_CONFIG
from src.config import CLASS_ID_TO_NAME


def test_synthetic_image_and_boxes_validity():
    """Verify generated image dimensions and bounding box validity."""
    gen = SyntheticFactoryGenerator(image_size=(640, 480), min_machines=5, max_machines=8, seed=42)
    img, annotations = gen.generate_single_layout()

    assert img.width == 640
    assert img.height == 480
    assert len(annotations) >= 5

    for ann in annotations:
        xmin, ymin, xmax, ymax = ann["bbox_px"]
        # Valid coordinates within image
        assert 0 <= xmin < xmax <= 640
        assert 0 <= ymin < ymax <= 480
        # Valid class ID and mapping
        assert ann["class_id"] in CLASS_ID_TO_NAME
        assert ann["class_name"] == CLASS_ID_TO_NAME[ann["class_id"]]
        # Valid COCO bbox [x, y, w, h]
        cx, cy, cw, ch = ann["coco_bbox"]
        assert cw > 0 and ch > 0
        assert ann["area"] == pytest.approx(cw * ch)


def test_generated_machines_do_not_overlap():
    """Verify that synthetic layout generator enforces collision-free machine placements."""
    gen = SyntheticFactoryGenerator(image_size=(800, 600), min_machines=6, max_machines=10, seed=123)
    _, annotations = gen.generate_single_layout()

    boxes = [ann["bbox_px"] for ann in annotations]
    num_boxes = len(boxes)

    for i in range(num_boxes):
        b1 = boxes[i]
        for j in range(i + 1, num_boxes):
            b2 = boxes[j]
            # Check bounding box intersection
            x_left = max(b1[0], b2[0])
            y_top = max(b1[1], b2[1])
            x_right = min(b1[2], b2[2])
            y_bottom = min(b1[3], b2[3])

            overlap = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            assert overlap == 0, f"Synthetic generator produced overlapping boxes: {b1} vs {b2}"


def test_coco_dataset_export():
    """Verify COCO dataset directory structure and JSON schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gen = SyntheticFactoryGenerator(image_size=(400, 300), min_machines=3, max_machines=4, seed=7)
        paths = gen.generate_coco_dataset(
            output_dir=tmp_path,
            split_counts={"train": 2, "val": 1, "test": 1}
        )

        assert "train" in paths
        assert (paths["train"] / "_annotations.coco.json").exists()
        assert len(list(paths["train"].glob("*.png"))) == 2
